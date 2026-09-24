"""Opt-in multirow WS32 projections for the layer-major prefill candidate.

No decoder imports this module by default. These primitives retain the raw-FP8
checkpoint layout and subgroup reduction boundaries of the existing one-row
Pallas path; they do not implement attention, routing or complete prefill.
Callers must use the validated WS32 expert8/feature4 mesh.
"""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax import lax

from .pallas.fp8_matmul import Fp8BlockMatmulConfig, fp8_block_matmul_f32
from glm_tpu.optimized.prefill_linear import _require_rows  # S2f: moved to production


def _config(block_shape: tuple[int, int]) -> Fp8BlockMatmulConfig:
    # The existing primitive validates raw bits, scales and tile geometry.
    return Fp8BlockMatmulConfig(
        block_shape=block_shape,
        output_tile=block_shape[0],
        contraction_tile=block_shape[1],
    )


def ws32_prefill_linear_mapped(
    lhs_local: Any,
    weight_bits_local: Any,
    weight_scale_local: Any,
    *,
    reduction_axis: Literal["feature", "expert"],
    block_shape: tuple[int, int] = (128, 128),
    interpret: bool = False,
) -> Any:
    """Project live prompt rows, reduce local FP32 partials, then round BF16.

    ``feature`` contracts hidden shards; ``expert`` contracts reciprocal
    intermediate shards. Neither operation reconstructs full-pod hidden state.
    The weight tile is shared across rows inside the existing Pallas call,
    not reloaded by a scan of individual token projections.
    """

    _require_rows(lhs_local)
    if reduction_axis not in ("feature", "expert"):
        raise ValueError("WS32 prefill reduction must be feature or expert")
    partial = fp8_block_matmul_f32(
        lhs_local,
        weight_bits_local,
        weight_scale_local,
        config=_config(block_shape),
        interpret=interpret,
    )
    with jax.named_scope(f"greenfield_ws32_prefill_linear/{reduction_axis}_reduce"):
        return lax.psum(partial, axis_name=reduction_axis).astype(jnp.bfloat16)


def ws32_prefill_dense_mapped(
    hidden_local: Any,
    gate_bits_local: Any,
    gate_scale_local: Any,
    up_bits_local: Any,
    up_scale_local: Any,
    down_bits_local: Any,
    down_scale_local: Any,
    *,
    block_shape: tuple[int, int] = (128, 128),
    interpret: bool = False,
) -> Any:
    """Batched reciprocal-sharded dense MLP, not the StrategyND overlay.

    This is a candidate building block with the same numerical boundaries as
    ``ws32_dense_pallas_mapped``. It is not automatically interchangeable with
    the promoted decoder's StrategyND dense association. A future layer path
    must earn its own bounded comparisons and §21 decoder evidence.
    """

    _require_rows(hidden_local)
    if gate_bits_local.ndim != 2 or (
        gate_bits_local.shape != up_bits_local.shape
        or gate_bits_local.shape[1] != hidden_local.shape[1]
    ):
        raise ValueError("WS32 prefill gate/up geometry drifted")
    local_intermediate, local_hidden = gate_bits_local.shape
    if down_bits_local.shape != (local_hidden, local_intermediate):
        raise ValueError("WS32 prefill down geometry drifted")
    config = _config(block_shape)
    gate_partial = fp8_block_matmul_f32(
        hidden_local,
        gate_bits_local,
        gate_scale_local,
        config=config,
        interpret=interpret,
    )
    up_partial = fp8_block_matmul_f32(
        hidden_local,
        up_bits_local,
        up_scale_local,
        config=config,
        interpret=interpret,
    )
    with jax.named_scope("greenfield_ws32_prefill_dense/feature_gate_up_reduce"):
        gate_up = lax.psum(
            jnp.stack((gate_partial, up_partial)), axis_name="feature"
        ).astype(jnp.bfloat16)
    activated = (gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]).astype(
        jnp.bfloat16
    )
    down_partial = fp8_block_matmul_f32(
        activated,
        down_bits_local,
        down_scale_local,
        config=config,
        interpret=interpret,
    )
    with jax.named_scope("greenfield_ws32_prefill_dense/expert_down_reduce"):
        return lax.psum(down_partial, axis_name="expert").astype(jnp.bfloat16)
