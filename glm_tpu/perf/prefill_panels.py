"""P4: wider output panels with the frozen M32/K128 arithmetic boundaries.

Mirror of the frozen panel kernel, supporting N256 and N512. Independent
N128 accumulators keep increasing K128 order and separate FP8 scale rows.
Only the outer N grid / raw-byte DMA width changes; route packing is unchanged.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..greenfield.kernels.prefill_expert_panels import (
    ExpertPanels,
    pack_expert_panel_rows,
    unpack_expert_panel_rows,
)


def _full_panel_scale(table: Any, ki: Any, ni: Any) -> Any:
    # Unlike _scale_value, this is the FULL K table, not a ki//8 slab.
    mask = (lax.broadcasted_iota(jnp.int32, table.shape, 0) == ki) & (
        lax.broadcasted_iota(jnp.int32, table.shape, 1) == ni
    )
    return jnp.sum(jnp.where(mask, table, jnp.float32(0)), dtype=jnp.float32)


def wide_prefill_panel_fp8_matmul(
    lhs: Any,
    weight_bits: Any,
    scale: Any,
    panels: ExpertPanels,
    *,
    result_dtype: Any = jnp.float32,
    output_tile: int = 512,
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Use a plan built for this contiguous local expert shard and sorted rows.

    Plan construction belongs outside repeated gate/up/down calls. As with
    grouped FP8, this is an internal primitive: caller aggregates health across
    chips and validates live model operands. A false plan performs no work.
    """
    if type(output_tile) is not int or output_tile not in (256, 512):
        raise ValueError("prefill panel output_tile must be 256 or 512")
    stripes = output_tile // 128
    if lhs.ndim != 2 or weight_bits.ndim != 3 or scale.ndim != 3:
        raise ValueError("panel FP8 ranks must be 2/3/3")
    m, k = lhs.shape
    groups, n, wk = weight_bits.shape
    dtype = jnp.dtype(result_dtype)
    if (
        min(m, k, groups, n) <= 0
        or wk != k
        or k % 128
        or n % output_tile
        or n // 128 > 128
        or lhs.dtype != jnp.bfloat16
        or weight_bits.dtype != jnp.uint8
        or scale.dtype != jnp.float32
        or scale.shape != (groups, n // 128, k // 128)
        or panels.restore_indices.shape != (m,)
        or panels.expert_ids.size != (m + 31) // 32 + groups - 1
        or dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16))
    ):
        raise ValueError("panel FP8 geometry/dtypes disagree with M32/output_tile/K128")
    tiles_k = k // 128
    scale_rows = (tiles_k + 7) // 8 * 8

    def compute(_: None) -> Any:
        packed = pack_expert_panel_rows(lhs, panels)
        scales = jnp.pad(
            jnp.swapaxes(scale, 1, 2),
            ((0, 0), (0, scale_rows - tiles_k), (0, 128 - n // 128)),
        )
        zeros = jnp.zeros((panels.expert_ids.size, 32, n), dtype)

        def kernel(ids, x_ref, w_ref, s_ref, initial_ref, out_ref, *accumulators):
            del ids, initial_ref
            ni = pl.program_id(0)
            for acc in accumulators:
                acc[...] = jnp.zeros((32, 128), jnp.float32)

            def contract(ki, _):
                x = x_ref[:, pl.ds(ki * 128, 128)]
                for stripe, acc in enumerate(accumulators):
                    bits = w_ref[pl.ds(stripe * 128, 128), pl.ds(ki * 128, 128)]
                    decoded = (
                        lax.bitcast_convert_type(bits, jnp.float8_e4m3fn).astype(
                            jnp.float32
                        )
                        * _full_panel_scale(s_ref[...], ki, ni * stripes + stripe)
                    ).astype(jnp.bfloat16)
                    acc[...] += lax.dot_general(
                        x,
                        decoded,
                        dimension_numbers=(((1,), (1,)), ((), ())),
                        preferred_element_type=jnp.float32,
                    )
                return None

            lax.fori_loop(0, tiles_k, contract, None)
            for stripe, acc in enumerate(accumulators):
                out_ref[:, pl.ds(stripe * 128, 128)] = acc[...].astype(dtype)

        def x_index(ni, pi, ids):
            del ni, ids
            return pi, 0, 0

        def w_index(ni, pi, ids):
            return ids[pi], ni, 0

        def s_index(ni, pi, ids):
            del ni
            return ids[pi], 0, 0

        def out_index(ni, pi, ids):
            del ids
            return pi, 0, ni

        out_spec = pl.BlockSpec((None, 32, output_tile), out_index)
        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(zeros.shape, dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                in_specs=(
                    pl.BlockSpec((None, 32, k), x_index),
                    pl.BlockSpec((None, output_tile, k), w_index),
                    pl.BlockSpec((None, scale_rows, 128), s_index),
                    out_spec,
                ),
                out_specs=out_spec,
                grid=(n // output_tile, panels.active_panels),
                scratch_shapes=tuple(pltpu.VMEM((32, 128), jnp.float32) for _ in range(stripes)),
            ),
            input_output_aliases={4: 0},
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary")
            ),
            interpret=interpret,
            name="glm_perf_prefill_expert_panel_raw_fp8_n" + str(output_tile),
        )
        result = call(panels.expert_ids, packed, weight_bits, scales, zeros)
        return unpack_expert_panel_rows(result, panels)

    return (
        lax.cond(
            panels.valid & (panels.active_panels > 0),
            compute,
            lambda _: jnp.zeros((m, n), dtype),
            operand=None,
        ),
        panels.valid,
    )
