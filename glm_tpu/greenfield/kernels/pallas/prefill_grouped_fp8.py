"""Opt-in ragged-row FP8 projection; no production decoder imports this yet.

Adapts JAX0.10.1 Megablox group scheduling to the existing raw-U8 [G,N,K]
checkpoint and FP8 arithmetic. Stock Megablox gmm execution is not called.
Returns (values, valid); the caller MUST refuse serving when valid is false.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas.ops.tpu.megablox.gmm import make_group_metadata

from .fp8_matmul import _scale_value


def prefill_grouped_fp8_matmul(
    lhs: Any,
    weight_bits: Any,
    scale: Any,
    group_sizes: Any,
    group_offset: Any,
    *,
    row_tile: int = 8,
    block_shape: tuple[int, int] = (128, 128),
    result_dtype: Any = jnp.float32,
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Project sorted route rows for one contiguous shard of expert weights.

    Group counts cover ALL sorted rows, not just this owner's rows. Empty and
    unowned rows are defined zeros. Invalid counts/offset skip device work and
    return false health. Only active row/group tiles execute, including masked
    revisits where groups share one row tile. No per-expert full-B matmul.
    """
    if lhs.ndim != 2 or weight_bits.ndim != 3 or scale.ndim != 3:
        raise ValueError("grouped FP8 ranks must be 2/3/3")
    m, k = lhs.shape
    local_groups, n, wk = weight_bits.shape
    if min(m, k, local_groups, n) <= 0 or k != wk:
        raise ValueError("grouped FP8 weight/input geometry disagrees")
    if (
        lhs.dtype != jnp.bfloat16
        or weight_bits.dtype != jnp.uint8
        or scale.dtype != jnp.float32
    ):
        raise ValueError("grouped FP8 requires BF16/U8/F32 operands")
    if (
        group_sizes.ndim != 1
        or group_sizes.size < local_groups
        or group_sizes.dtype != jnp.int32
    ):
        raise ValueError("group sizes must be a complete int32 expert vector")
    if group_offset.shape != () or group_offset.dtype != jnp.int32:
        raise ValueError("group offset must be an int32 scalar")
    if row_tile not in (8, 16, 32):
        raise ValueError("grouped FP8 row tile must be 8/16/32")
    if len(block_shape) != 2 or any(type(v) is not int or v <= 0 for v in block_shape):
        raise ValueError("block shape must contain positive integers")
    bn, bk = block_shape
    # Current runtime projections are exactly divisible. Tail matrices need
    # their own padding admission; never copy a full expert table to pad it.
    if n % bn or k % bk or n // bn > 128:
        raise ValueError("grouped FP8 requires divisible block geometry")
    if scale.shape != (local_groups, n // bn, k // bk):
        raise ValueError("grouped FP8 scale geometry disagrees")
    dtype = jnp.dtype(result_dtype)
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16)):
        raise ValueError("grouped FP8 output must be F32/BF16")
    valid = (
        jnp.all((group_sizes >= 0) & (group_sizes <= m))
        & (jnp.sum(group_sizes) == m)
        & (group_offset >= 0)
        & (group_offset <= group_sizes.size - local_groups)
    )
    padded_m = ((m + row_tile - 1) // row_tile) * row_tile
    tiles_k, tiles_n = k // bk, n // bn

    def compute(_: None) -> Any:
        metadata, active_tiles = make_group_metadata(
            group_sizes=group_sizes,
            m=padded_m,
            tm=row_tile,
            start_group=group_offset,
            num_nonzero_groups=local_groups,
            visit_empty_groups=False,
        )
        # Only compact FP32 scale metadata is transposed/padded, never weights.
        scale_table = jnp.pad(
            jnp.swapaxes(scale, 1, 2),
            ((0, 0), (0, ((tiles_k + 7) // 8) * 8 - tiles_k), (0, 128 - tiles_n)),
        )
        padded_lhs = jnp.pad(lhs, ((0, padded_m - m), (0, 0)))
        zeros = jnp.zeros((padded_m, n), dtype)

        def kernel(meta, offset, x_ref, w_ref, s_ref, initial_ref, out_ref, acc_ref):
            del offset
            offsets, ids, row_tiles = meta
            ni, gi, ki = pl.program_id(0), pl.program_id(1), pl.program_id(2)

            @pl.when(ki == 0)
            def initialize():
                acc_ref[...] = jnp.zeros_like(acc_ref)
                first_visit = (gi == 0) | (
                    row_tiles[gi] != row_tiles[jnp.maximum(gi - 1, 0)]
                )

                @pl.when(first_visit)
                def initialize_output():
                    out_ref[...] = initial_ref[...]

            decoded = (
                lax.bitcast_convert_type(w_ref[...], jnp.float8_e4m3fn).astype(
                    jnp.float32
                )
                * _scale_value(s_ref[...], ki, ni)
            ).astype(jnp.bfloat16)
            acc_ref[...] += lax.dot_general(
                x_ref[...],
                decoded,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )

            @pl.when(ki == tiles_k - 1)
            def store():
                positions = row_tiles[gi] * row_tile + lax.broadcasted_iota(
                    jnp.int32, (row_tile, bn), 0
                )
                group = ids[gi]
                mask = (positions >= offsets[group]) & (positions < offsets[group + 1])
                out_ref[...] = jnp.where(mask, acc_ref[...].astype(dtype), out_ref[...])

        def x_index(ni, gi, ki, meta, offset):
            del ni, offset
            return meta[2][gi], ki

        def w_index(ni, gi, ki, meta, offset):
            return meta[1][gi] - offset[0], ni, ki

        def s_index(ni, gi, ki, meta, offset):
            del ni
            return meta[1][gi] - offset[0], ki // 8, 0

        def out_index(ni, gi, ki, meta, offset):
            del ki, offset
            return meta[2][gi], ni

        out_spec = pl.BlockSpec((row_tile, bn), out_index)
        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((padded_m, n), dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=2,
                in_specs=(
                    pl.BlockSpec((row_tile, bk), x_index),
                    pl.BlockSpec((None, bn, bk), w_index),
                    pl.BlockSpec((None, 8, 128), s_index),
                    out_spec,
                ),
                out_specs=out_spec,
                grid=(tiles_n, active_tiles, tiles_k),
                scratch_shapes=(pltpu.VMEM((row_tile, bn), jnp.float32),),
            ),
            # Three metadata leaves, offset, lhs, weights, scales, initial out.
            input_output_aliases={7: 0},
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary", "arbitrary")
            ),
            interpret=interpret,
            name="greenfield_prefill_grouped_raw_fp8",
        )
        return lax.cond(
            active_tiles > 0,
            lambda _: call(
                metadata,
                group_offset[None],
                padded_lhs,
                weight_bits,
                scale_table,
                zeros,
            ),
            lambda _: zeros,
            operand=None,
        )[:m]

    return (
        lax.cond(valid, compute, lambda _: jnp.zeros((m, n), dtype), operand=None),
        valid,
    )
