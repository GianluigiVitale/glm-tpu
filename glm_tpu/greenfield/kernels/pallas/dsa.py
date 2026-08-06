"""One-row TPU-v4 Pallas scorer for the GLM lightning indexer.

The historical key cache remains BF16.  Each Pallas program reads one
context tile, converts it to FP32 in VMEM, and evaluates all 32 query heads
inside that program. ReLU, signed head weighting, and the FP32 head reduction
happen before the score tile leaves the kernel, so the implementation never
creates the ``[heads, context]`` intermediate in HBM.

This module is deliberately isolated and default-off.  ``dsa_scores_kernel``
retains the readable JAX scorer as its fallback until the Pallas path has
passed protected TPU correctness, HLO, and microbenchmark gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..reference.dsa import dsa_scores


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True, slots=True)
class DsaScoreConfig:
    """Compile-time TPU-v4 tile and numerical contract."""

    context_tile: int = 128
    head_dim: int = 128
    score_dtype: Any = jnp.float32

    def __post_init__(self) -> None:
        if self.context_tile != 128:
            raise ValueError("TPU-v4 DSA context tiles must contain 128 keys")
        if self.head_dim != 128:
            raise ValueError("the GLM DSA head dimension must be 128")
        if jnp.dtype(self.score_dtype) != jnp.dtype(jnp.float32):
            raise ValueError("the exact DSA score dtype must be FP32")


def _validate_inputs(
    query: Any,
    index_keys: Any,
    head_weights: Any,
    config: DsaScoreConfig,
) -> tuple[int, int]:
    if query.ndim != 3 or index_keys.ndim != 2 or head_weights.ndim != 2:
        raise ValueError("DSA query/key/head-weight ranks must be 3/2/2")
    rows, heads, head_dim = query.shape
    if rows != 1:
        raise ValueError("decode_batch1 DSA scorer must contain exactly one row")
    if heads <= 0 or index_keys.shape[0] <= 0:
        raise ValueError("DSA heads and local context must be nonempty")
    if head_dim != config.head_dim or index_keys.shape[1] != config.head_dim:
        raise ValueError("DSA query/key head dimensions disagree with the contract")
    if head_weights.shape != (rows, heads):
        raise ValueError("DSA head weights disagree with query rows/heads")
    if query.dtype != jnp.float32 or head_weights.dtype != jnp.float32:
        raise ValueError("DSA query and head weights must remain FP32")
    if index_keys.dtype != jnp.bfloat16:
        raise ValueError("the cache-resident DSA index keys must remain BF16")
    return heads, index_keys.shape[0]


def dsa_scores_pallas(
    query: Any,
    index_keys: Any,
    head_weights: Any,
    *,
    config: DsaScoreConfig = DsaScoreConfig(),
    interpret: bool = False,
) -> Any:
    """Compute exact one-row FP32 DSA scores without a per-head HBM tensor."""

    heads, context = _validate_inputs(query, index_keys, head_weights, config)
    padded_context = _ceil_div(context, config.context_tile) * config.context_tile
    if padded_context != context:
        index_keys = jnp.pad(index_keys, ((0, padded_context - context), (0, 0)))

    context_tiles = padded_context // config.context_tile

    def kernel(
        query_ref: Any,
        key_ref: Any,
        head_weight_ref: Any,
        output_ref: Any,
    ) -> None:
        per_head = lax.dot_general(
            query_ref[...],
            key_ref[...].astype(jnp.float32),
            dimension_numbers=(((2,), (1,)), ((), ())),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        ) * jnp.float32(config.head_dim**-0.5)
        per_head = jnp.maximum(per_head, jnp.float32(0.0))
        output_ref[...] = lax.dot_general(
            head_weight_ref[...],
            per_head.reshape((heads, config.context_tile)),
            dimension_numbers=(((1,), (0,)), ((), ())),
            precision=lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )

    def query_index(context_index: Any) -> tuple[int, int, int]:
        del context_index
        return 0, 0, 0

    def key_index(context_index: Any) -> tuple[Any, int]:
        return context_index, 0

    def head_weight_index(context_index: Any) -> tuple[int, int]:
        del context_index
        return 0, 0

    def output_index(context_index: Any) -> tuple[int, Any]:
        return 0, context_index

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, padded_context), jnp.float32),
        grid=(context_tiles,),
        in_specs=(
            pl.BlockSpec((1, heads, config.head_dim), query_index),
            pl.BlockSpec((config.context_tile, config.head_dim), key_index),
            # The full [1,heads] array is legal even though it is narrower
            # than a TPU tile; each block dimension equals the array extent.
            pl.BlockSpec((1, heads), head_weight_index),
        ),
        out_specs=pl.BlockSpec((1, config.context_tile), output_index),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            "greenfield_dsa_score_"
            f"r1_h{heads}_d{config.head_dim}_s{padded_context}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=(
                heads
                * padded_context
                * (2 * config.head_dim + 4)
            ),
            bytes_accessed=(
                heads * config.head_dim * 4
                + padded_context * config.head_dim * 2
                + heads * 4
                + padded_context * 4
            ),
            transcendentals=0,
        ),
    )
    return call(query, index_keys, head_weights)[:, :context]


def dsa_scores_kernel(
    query: Any,
    index_keys: Any,
    head_weights: Any,
    *,
    backend: Literal["reference", "pallas"] = "reference",
    config: DsaScoreConfig = DsaScoreConfig(),
    interpret: bool = False,
) -> Any:
    """Dispatch the default-off Pallas scorer or the exact readable fallback."""

    if backend == "reference":
        return dsa_scores(query, index_keys, head_weights)
    if backend == "pallas":
        return dsa_scores_pallas(
            query,
            index_keys,
            head_weights,
            config=config,
            interpret=interpret,
        )
    raise ValueError(f"unsupported DSA score backend {backend!r}")
