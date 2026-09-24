"""Canonical dense MLP placement of the production prefill (four 32-row placements in B128).

TPU results of the dense MLP depend on the physical row placement; this runs the dense suffix as
four 32-live-row placements inside B128 (B114 padded and cropped) so production reproduces the
retained narrow-suffix bytes. The body of ``greenfield/kernels/ws32_prefill_dense_canonical.py``
calling the production MLP suffix explicitly (S2d fold of the former function rebinding); the
frozen module stays untouched as the numerical oracle of the tests.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from .reference.moe import GlmMoeNumericalContract
from .bf16_resident import Bf16DenseWeights
from .prefill_layer import ws32_prefill_mlp_mapped


def ws32_prefill_dense_canonical_mapped(
    normalized: Any,
    live: Any,
    dense: Bf16DenseWeights,
    *,
    moe_contract: GlmMoeNumericalContract,
    linear_interpret: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Four uniform device calls, each32 live slots in physical rows0:32 of128.

    Only the dense suffix changes. No cache, normalization, host stride, MoE,
    precision or checkpoint change. B114 is padded to the same four placements
    and cropped back after assembly; malformed input geometry fails at trace.
    Padded input rows are zeroed before entering the original dense arithmetic.
    All owners execute all four iterations, including empty tiles.
    """
    if (
        type(linear_interpret) is not bool
        or normalized.ndim != 2
        or normalized.shape[0] not in (114, 128)
        or normalized.dtype != jnp.bfloat16
        or normalized.shape[1] * 4 != moe_contract.hidden_size
        or live.shape != (normalized.shape[0],)
        or live.dtype != jnp.bool_
        or not isinstance(dense, Bf16DenseWeights)
    ):
        raise ValueError(
            "canonical dense requires B114/B128 and original dense weights"
        )
    rows, width = normalized.shape
    safe = jnp.where(live[:, None], normalized, jnp.bfloat16(0))
    tiles = jnp.pad(safe, ((0, 128 - rows), (0, 0))).reshape(4, 32, width)
    masks = jnp.pad(live, ((0, 128 - rows),), constant_values=False).reshape(4, 32)

    def body(unused: None, inputs: tuple[Any, Any]) -> tuple[None, tuple]:
        values, mask = inputs
        padded = jnp.pad(values, ((0, 96), (0, 0)))
        padded_live = jnp.pad(mask, ((0, 96),), constant_values=False)
        output, ids, weights, valid = ws32_prefill_mlp_mapped(
            padded,
            padded_live,
            dense,
            None,
            moe_contract=moe_contract,
            linear_interpret=linear_interpret,
        )
        # Preserve the existing suffix's live output/health contract; never hide
        # a nonfinite live result by masking it after deriving validity.
        valid = valid & (~padded_live | jnp.all(jnp.isfinite(output), axis=1))
        output = jnp.where(padded_live[:, None], output, jnp.bfloat16(0))
        return None, tuple(v[:32] for v in (output, ids, weights, valid))

    with jax.named_scope("greenfield_ws32_prefill_dense_canonical"):
        _, stacked = lax.scan(body, None, (tiles, masks), unroll=1)
    return tuple(v.reshape((128, *v.shape[2:]))[:rows] for v in stacked)
