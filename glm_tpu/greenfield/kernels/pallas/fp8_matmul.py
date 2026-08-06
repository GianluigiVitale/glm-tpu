"""TPU-v4 raw-FP8 block matmul without a persistent BF16 weight overlay.

The checkpoint stores E4M3FN payloads as their exact ``uint8`` encodings in
``[out, in]`` order and one FP32 inverse scale per 128x128 block. Selected
gate/up tables use final ``[expert, in, gate_then_up]`` order so one weight
pipeline and MXU stream can produce both projections. TPU v4 has
no native FP8 MXU matmul, so this kernel DMAs one checkpoint block at a time,
bitcasts the bytes to E4M3FN, dequantizes that tile in VMEM, and executes the
BF16 matmul with an FP32 accumulator.  Complete matrices are never decoded or
materialized in HBM.

This is an independent greenfield kernel.  It imports no legacy execution
code and intentionally exposes a narrow numerical contract suitable for
reference comparison before MoE fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True, slots=True)
class Fp8BlockMatmulConfig:
    """Static tile and arithmetic contract for ``fp8_block_matmul``."""

    block_shape: tuple[int, int] = (128, 128)
    row_tile: int = 8
    output_tile: int = 128
    contraction_tile: int = 128
    output_dtype: Any = jnp.bfloat16
    accumulator_dtype: Any = jnp.float32

    def __post_init__(self) -> None:
        if len(self.block_shape) != 2 or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in self.block_shape
        ):
            raise ValueError("FP8 block shape must contain two positive integers")
        for name in ("row_tile", "output_tile", "contraction_tile"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.output_tile != self.block_shape[0] or (
            self.contraction_tile % self.block_shape[1] != 0
        ):
            raise ValueError(
                "TPU-v4 tiles require one output scale block and an integral "
                "number of contraction scale blocks"
            )
        if jnp.dtype(self.output_dtype) != jnp.dtype(jnp.bfloat16):
            raise ValueError("the TPU-v4 FP8 matmul output must be BF16")
        if jnp.dtype(self.accumulator_dtype) != jnp.dtype(jnp.float32):
            raise ValueError("the TPU-v4 FP8 matmul accumulator must be FP32")


def _validate_inputs(
    lhs: Any,
    weight_bits: Any,
    scale: Any,
    config: Fp8BlockMatmulConfig,
) -> tuple[int, int, int]:
    if lhs.ndim != 2 or weight_bits.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP8 matmul inputs must have ranks two, two, and two")
    rows, contraction = lhs.shape
    output, weight_contraction = weight_bits.shape
    if rows <= 0 or contraction <= 0 or output <= 0:
        raise ValueError("FP8 matmul dimensions must be positive")
    if weight_contraction != contraction:
        raise ValueError("FP8 weight/input contraction dimensions disagree")
    if lhs.dtype != jnp.bfloat16:
        raise ValueError(f"FP8 matmul lhs must be BF16, got {lhs.dtype}")
    if weight_bits.dtype != jnp.uint8:
        raise ValueError(
            f"FP8 checkpoint payload must be uint8 bits, got {weight_bits.dtype}"
        )
    if scale.dtype != jnp.float32:
        raise ValueError(f"FP8 block scales must be FP32, got {scale.dtype}")
    expected_scale = (
        _ceil_div(output, config.block_shape[0]),
        _ceil_div(contraction, config.block_shape[1]),
    )
    if scale.shape != expected_scale:
        raise ValueError(
            f"FP8 scale shape must be {expected_scale}, got {scale.shape}"
        )
    return rows, contraction, output


def _scale_value(
    scale_tile: Any, contraction_index: Any, output_index: Any
) -> Any:
    """Select one scalar from an aligned VMEM scale tile in registers."""

    scale_rows = lax.broadcasted_iota(jnp.int32, scale_tile.shape, 0)
    scale_columns = lax.broadcasted_iota(jnp.int32, scale_tile.shape, 1)
    scale_mask = (
        (scale_rows == contraction_index % jnp.int32(8))
        & (scale_columns == output_index)
    )
    # Dynamic slicing a VMEM vector is not implemented by the v4 Mosaic
    # lowering. The other 1,023 entries are masked to zero before reduction.
    return jnp.sum(
        jnp.where(scale_mask, scale_tile, jnp.float32(0.0)),
        dtype=jnp.float32,
    )


def _aligned_scale_table(
    scale: Any, *, output_tiles: int, contraction_tiles: int
) -> Any:
    if output_tiles > 128:
        raise ValueError("FP8 matmul supports at most 128 output-scale blocks")
    aligned_contraction_tiles = _ceil_div(contraction_tiles, 8) * 8
    # Mosaic cannot issue a dynamically indexed scalar VMEM load from the
    # compact [N-block,K-block] table. This bounded metadata staging is 24 KiB
    # for GLM up, compared with 12 MiB of FP8 weight bytes.
    return jnp.pad(
        jnp.transpose(scale),
        (
            (0, aligned_contraction_tiles - contraction_tiles),
            (0, 128 - output_tiles),
        ),
    )


def _selected_scale_table(
    scale: Any,
    local_ids: Any,
    *,
    padded_output: int,
    padded_contraction: int,
    block_shape: tuple[int, int],
) -> Any:
    """Expand only top-k selected scale metadata to direct vector loads."""

    selected = jnp.take(scale, local_ids, axis=0)
    output_blocks = padded_output // block_shape[0]
    contraction_blocks = padded_contraction // block_shape[1]
    selected = jnp.pad(
        selected,
        (
            (0, 0),
            (0, output_blocks - selected.shape[1]),
            (0, contraction_blocks - selected.shape[2]),
        ),
    )
    # [route,N-block,K-block] -> [route,K-block,1,N]. This is bounded by
    # top-k=8 rather than local_experts=64 and matches the MXU output vector.
    return jnp.transpose(
        jnp.repeat(selected, block_shape[0], axis=1),
        (0, 2, 1),
    )[:, :, None, :]


def _validate_selected_inputs(
    hidden: Any,
    route_indices: Any,
    expert_start: Any,
    weight_bits: Any,
    scale: Any,
    config: Fp8BlockMatmulConfig,
) -> tuple[int, int, int, int]:
    if hidden.ndim != 2 or hidden.shape[0] != 1:
        raise ValueError("selected-expert hidden input must contain one row")
    if route_indices.ndim != 1 or route_indices.shape[0] <= 0:
        raise ValueError("selected-expert routes must be a nonempty vector")
    if route_indices.dtype != jnp.int32:
        raise ValueError("selected-expert routes must be int32")
    if expert_start.shape != () or expert_start.dtype != jnp.int32:
        raise ValueError("selected-expert start must be an int32 scalar")
    if weight_bits.ndim != 3 or scale.ndim != 3:
        raise ValueError("selected expert weights/scales must have rank three")
    # Selected-expert tables use final MXU access order [G,K,N]. This is
    # intentionally different from the standalone [N,K] checkpoint probe and
    # prevents a complete-table transpose in the decode JIT.
    local_experts, contraction, output = weight_bits.shape
    if local_experts <= 0:
        raise ValueError("selected-expert local weight table must be nonempty")
    if hidden.shape[1] != contraction:
        raise ValueError("selected-expert hidden/weight contraction disagrees")
    if hidden.dtype != jnp.bfloat16:
        raise ValueError("selected-expert hidden input must be BF16")
    if weight_bits.dtype != jnp.uint8:
        raise ValueError("selected-expert checkpoint payload must be uint8 bits")
    if scale.dtype != jnp.float32:
        raise ValueError("selected-expert block scales must be FP32")
    expected_scale = (
        local_experts,
        _ceil_div(output, config.block_shape[0]),
        _ceil_div(contraction, config.block_shape[1]),
    )
    if scale.shape != expected_scale:
        raise ValueError(
            f"selected-expert scale shape must be {expected_scale}, got {scale.shape}"
        )
    return route_indices.shape[0], local_experts, contraction, output


def fp8_block_matmul(
    lhs: Any,
    weight_bits: Any,
    scale: Any,
    *,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(),
    interpret: bool = False,
) -> Any:
    """Return ``lhs @ dequant(weight_bits).T`` using tile-local dequantization.

    ``interpret=True`` runs through the Pallas interpreter for semantic tests.
    TPU promotion still requires a compiled kernel correctness proof, HLO/custom
    call assertion, microbenchmark, and real-layer fallback comparison.
    """

    rows, contraction, output = _validate_inputs(
        lhs, weight_bits, scale, config
    )
    padded_rows = _ceil_div(rows, config.row_tile) * config.row_tile
    padded_contraction = (
        _ceil_div(contraction, config.contraction_tile)
        * config.contraction_tile
    )
    padded_output = (
        _ceil_div(output, config.output_tile) * config.output_tile
    )
    if padded_rows != rows or padded_contraction != contraction:
        lhs = jnp.pad(
            lhs,
            ((0, padded_rows - rows), (0, padded_contraction - contraction)),
        )
    if padded_output != output or padded_contraction != contraction:
        weight_bits = jnp.pad(
            weight_bits,
            (
                (0, padded_output - output),
                (0, padded_contraction - contraction),
            ),
        )

    # Same-width bitcast: no checkpoint bytes are decoded on the host and no
    # BF16/F32 matrix is created outside the Pallas call.
    weight_fp8 = lax.bitcast_convert_type(weight_bits, jnp.float8_e4m3fn)
    output_tiles = padded_output // config.output_tile
    contraction_tiles = padded_contraction // config.contraction_tile
    scale_table = _aligned_scale_table(
        scale,
        output_tiles=output_tiles,
        contraction_tiles=contraction_tiles,
    )

    def kernel(
        lhs_ref: Any,
        weight_ref: Any,
        scale_ref: Any,
        output_ref: Any,
        accumulator_ref: Any,
    ) -> None:
        contraction_index = pl.program_id(1)

        @pl.when(contraction_index == 0)
        def initialize_accumulator() -> None:
            accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

        # TPU v4 cannot consume FP8 directly in its MXU.  Decode only this
        # 128x128 VMEM tile, apply its scalar inverse scale, then feed BF16 to
        # the MXU while accumulating in FP32.
        output_index = pl.program_id(0)
        scale_value = _scale_value(
            scale_ref[...], contraction_index, output_index
        )
        decoded_weight = (
            weight_ref[...].astype(config.accumulator_dtype)
            * scale_value.astype(config.accumulator_dtype)
        ).astype(jnp.bfloat16)
        update = lax.dot_general(
            lhs_ref[...],
            decoded_weight,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=config.accumulator_dtype,
        )
        accumulator_ref[...] += update

        @pl.when(contraction_index == contraction_tiles - 1)
        def store_output() -> None:
            output_ref[...] = accumulator_ref[...].astype(config.output_dtype)

    def lhs_index(output_index: Any, contraction_index: Any) -> tuple[Any, Any]:
        del output_index
        return 0, contraction_index

    def weight_index(
        output_index: Any, contraction_index: Any
    ) -> tuple[Any, Any]:
        return output_index, contraction_index

    def output_index(
        output_index_value: Any, contraction_index: Any
    ) -> tuple[Any, Any]:
        del contraction_index
        return 0, output_index_value

    def scale_index(
        output_index_value: Any, contraction_index: Any
    ) -> tuple[Any, int]:
        del output_index_value
        return contraction_index // jnp.int32(8), 0

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_output), config.output_dtype
        ),
        grid=(output_tiles, contraction_tiles),
        in_specs=(
            pl.BlockSpec(
                (padded_rows, config.contraction_tile), lhs_index
            ),
            pl.BlockSpec(
                (config.output_tile, config.contraction_tile), weight_index
            ),
            pl.BlockSpec((8, 128), scale_index),
        ),
        out_specs=pl.BlockSpec(
            (padded_rows, config.output_tile), output_index
        ),
        scratch_shapes=(
            pltpu.VMEM(
                (padded_rows, config.output_tile),
                config.accumulator_dtype,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            "greenfield_fp8_block_matmul_"
            f"m{padded_rows}_k{padded_contraction}_n{padded_output}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=2 * padded_rows * padded_contraction * padded_output,
            bytes_accessed=(
                padded_rows * padded_contraction * 2
                + padded_output * padded_contraction
                + output_tiles * contraction_tiles * 4
                + padded_rows * padded_output * 2
            ),
            transcendentals=0,
        ),
    )
    return call(lhs, weight_fp8, scale_table)[:rows, :output]


def fp8_block_up_gate(
    lhs: Any,
    gate_bits: Any,
    gate_scale: Any,
    up_bits: Any,
    up_scale: Any,
    *,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(),
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Compute gate and up projections in one TPU custom call.

    The two projections keep distinct FP32 accumulators and BF16 outputs.  The
    following SwiGLU activation remains outside this first kernel so its exact
    dtype boundary can be protected independently before a later fusion.
    """

    rows, contraction, output = _validate_inputs(
        lhs, gate_bits, gate_scale, config
    )
    up_geometry = _validate_inputs(lhs, up_bits, up_scale, config)
    if up_geometry != (rows, contraction, output):
        raise ValueError("FP8 gate/up projection geometries disagree")
    padded_rows = _ceil_div(rows, config.row_tile) * config.row_tile
    padded_contraction = (
        _ceil_div(contraction, config.contraction_tile)
        * config.contraction_tile
    )
    padded_output = _ceil_div(output, config.output_tile) * config.output_tile
    if padded_rows != rows or padded_contraction != contraction:
        lhs = jnp.pad(
            lhs,
            ((0, padded_rows - rows), (0, padded_contraction - contraction)),
        )

    def pad_weight(value: Any) -> Any:
        if padded_output == output and padded_contraction == contraction:
            return value
        return jnp.pad(
            value,
            (
                (0, padded_output - output),
                (0, padded_contraction - contraction),
            ),
        )

    gate_fp8 = lax.bitcast_convert_type(
        pad_weight(gate_bits), jnp.float8_e4m3fn
    )
    up_fp8 = lax.bitcast_convert_type(
        pad_weight(up_bits), jnp.float8_e4m3fn
    )
    output_tiles = padded_output // config.output_tile
    contraction_tiles = padded_contraction // config.contraction_tile
    gate_scale_table = _aligned_scale_table(
        gate_scale,
        output_tiles=output_tiles,
        contraction_tiles=contraction_tiles,
    )
    up_scale_table = _aligned_scale_table(
        up_scale,
        output_tiles=output_tiles,
        contraction_tiles=contraction_tiles,
    )

    def kernel(
        lhs_ref: Any,
        gate_ref: Any,
        gate_scale_ref: Any,
        up_ref: Any,
        up_scale_ref: Any,
        gate_output_ref: Any,
        up_output_ref: Any,
        gate_accumulator_ref: Any,
        up_accumulator_ref: Any,
    ) -> None:
        contraction_index = pl.program_id(1)

        @pl.when(contraction_index == 0)
        def initialize_accumulators() -> None:
            gate_accumulator_ref[...] = jnp.zeros_like(gate_accumulator_ref)
            up_accumulator_ref[...] = jnp.zeros_like(up_accumulator_ref)

        output_index = pl.program_id(0)
        gate_value = _scale_value(
            gate_scale_ref[...], contraction_index, output_index
        )
        up_value = _scale_value(
            up_scale_ref[...], contraction_index, output_index
        )
        decoded_gate = (
            gate_ref[...].astype(config.accumulator_dtype) * gate_value
        ).astype(jnp.bfloat16)
        decoded_up = (
            up_ref[...].astype(config.accumulator_dtype) * up_value
        ).astype(jnp.bfloat16)
        dimensions = (((1,), (1,)), ((), ()))
        gate_accumulator_ref[...] += lax.dot_general(
            lhs_ref[...],
            decoded_gate,
            dimension_numbers=dimensions,
            preferred_element_type=config.accumulator_dtype,
        )
        up_accumulator_ref[...] += lax.dot_general(
            lhs_ref[...],
            decoded_up,
            dimension_numbers=dimensions,
            preferred_element_type=config.accumulator_dtype,
        )

        @pl.when(contraction_index == contraction_tiles - 1)
        def store_outputs() -> None:
            gate_output_ref[...] = gate_accumulator_ref[...].astype(
                config.output_dtype
            )
            up_output_ref[...] = up_accumulator_ref[...].astype(
                config.output_dtype
            )

    def lhs_index(output_index: Any, contraction_index: Any) -> tuple[Any, Any]:
        del output_index
        return 0, contraction_index

    def weight_index(
        output_index: Any, contraction_index: Any
    ) -> tuple[Any, Any]:
        return output_index, contraction_index

    def output_index(
        output_index_value: Any, contraction_index: Any
    ) -> tuple[Any, Any]:
        del contraction_index
        return 0, output_index_value

    def scale_index(
        output_index_value: Any, contraction_index: Any
    ) -> tuple[Any, int]:
        del output_index_value
        return contraction_index // jnp.int32(8), 0

    output_shape = jax.ShapeDtypeStruct(
        (padded_rows, padded_output), config.output_dtype
    )
    output_spec = pl.BlockSpec(
        (padded_rows, config.output_tile), output_index
    )
    call = pl.pallas_call(
        kernel,
        out_shape=(output_shape, output_shape),
        grid=(output_tiles, contraction_tiles),
        in_specs=(
            pl.BlockSpec(
                (padded_rows, config.contraction_tile), lhs_index
            ),
            pl.BlockSpec(
                (config.output_tile, config.contraction_tile), weight_index
            ),
            pl.BlockSpec((8, 128), scale_index),
            pl.BlockSpec(
                (config.output_tile, config.contraction_tile), weight_index
            ),
            pl.BlockSpec((8, 128), scale_index),
        ),
        out_specs=(output_spec, output_spec),
        scratch_shapes=(
            pltpu.VMEM(
                (padded_rows, config.output_tile), config.accumulator_dtype
            ),
            pltpu.VMEM(
                (padded_rows, config.output_tile), config.accumulator_dtype
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            "greenfield_fp8_block_up_gate_"
            f"m{padded_rows}_k{padded_contraction}_n{padded_output}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=4 * padded_rows * padded_contraction * padded_output,
            bytes_accessed=(
                padded_rows * padded_contraction * 2
                + 2 * padded_output * padded_contraction
                + 2 * output_tiles * contraction_tiles * 4
                + 2 * padded_rows * padded_output * 2
            ),
            transcendentals=0,
        ),
    )
    gate, up = call(
        lhs, gate_fp8, gate_scale_table, up_fp8, up_scale_table
    )
    return gate[:rows, :output], up[:rows, :output]


def fp8_selected_up_gate(
    hidden: Any,
    route_indices: Any,
    expert_start: Any,
    gate_up_bits: Any,
    gate_up_scale: Any,
    *,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(
        contraction_tile=512
    ),
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Project one batch-one row through its locally owned selected experts.

    Outputs retain top-k route order. A route outside this chip's contiguous
    expert interval produces an exact zero row. Owned routes are compacted on
    device and their dynamic count bounds the Pallas pipeline, so a chip does
    not DMA or compute the other chips' route slots. No selected BF16 matrix,
    host routing, or eight-way conditional graph is materialized.
    """

    (
        route_count,
        local_experts,
        contraction,
        combined_output,
    ) = _validate_selected_inputs(
        hidden,
        route_indices,
        expert_start,
        gate_up_bits,
        gate_up_scale,
        config,
    )
    if combined_output % (2 * config.block_shape[0]) != 0:
        raise ValueError(
            "selected gate/up packed output must contain two complete "
            "128-column block sequences"
        )
    projection_output = combined_output // 2

    padded_contraction = (
        _ceil_div(contraction, config.contraction_tile)
        * config.contraction_tile
    )
    padded_output = (
        _ceil_div(combined_output, config.output_tile) * config.output_tile
    )
    hidden = jnp.pad(
        hidden,
        (
            (0, config.row_tile - 1),
            (0, padded_contraction - contraction),
        ),
    )

    def pad_weight(value: Any) -> Any:
        if (
            padded_output == combined_output
            and padded_contraction == contraction
        ):
            return value
        return jnp.pad(
            value,
            (
                (0, 0),
                (0, padded_contraction - contraction),
                (0, padded_output - combined_output),
            ),
        )

    # Keep complete expert tables as checkpoint-native U8 Pallas operands.
    # Bitcasting only each DMA'd VMEM tile prevents a complete F8 table view
    # from entering the optimized HLO contract.
    gate_up_raw = pad_weight(gate_up_bits)
    output_tiles = padded_output // config.output_tile
    contraction_tiles = padded_contraction // config.contraction_tile
    contraction_blocks_per_tile = (
        config.contraction_tile // config.block_shape[1]
    )
    local_ids = jnp.clip(
        route_indices - expert_start,
        jnp.int32(0),
        jnp.int32(local_experts - 1),
    )
    active = (route_indices >= expert_start) & (
        route_indices < expert_start + jnp.int32(local_experts)
    )
    active_i32 = active.astype(jnp.int32)
    inactive_i32 = jnp.logical_not(active).astype(jnp.int32)
    active_count = jnp.sum(active_i32, dtype=jnp.int32)
    # Stable device compaction without a sort. compact_index maps each
    # original route slot to its unique compacted slot; active slots precede
    # inactive slots while preserving order within each partition.
    compact_index = jnp.where(
        active,
        jnp.cumsum(active_i32) - jnp.int32(1),
        active_count + jnp.cumsum(inactive_i32) - jnp.int32(1),
    )
    compact_local_ids = jnp.zeros_like(local_ids).at[compact_index].set(
        local_ids
    )
    gate_up_scale_table = _selected_scale_table(
        gate_up_scale,
        compact_local_ids,
        padded_output=padded_output,
        padded_contraction=padded_contraction,
        block_shape=config.block_shape,
    )

    def kernel(
        local_ids_value: Any,
        active_count_value: Any,
        hidden_hbm_ref: Any,
        gate_up_hbm_ref: Any,
        gate_up_scale_hbm_ref: Any,
        gate_up_output_hbm_ref: Any,
        gate_up_accumulator_ref: Any,
    ) -> None:
        def inner_kernel(
            hidden_ref: Any,
            gate_up_ref: Any,
            gate_up_scale_ref: Any,
            gate_up_output_ref: Any,
            gate_up_accumulator: Any,
        ) -> None:
            output_index = pl.program_id(0)
            route_index = pl.program_id(1)
            contraction_index = pl.program_id(2)

            @pl.when(contraction_index == 0)
            def initialize_accumulators() -> None:
                gate_up_accumulator[...] = jnp.zeros_like(
                    gate_up_accumulator
                )

            decoded_gate_up = (
                lax.bitcast_convert_type(
                    gate_up_ref[...], jnp.float8_e4m3fn
                )
                .astype(config.accumulator_dtype)
                .reshape(
                    contraction_blocks_per_tile,
                    config.block_shape[1],
                    config.output_tile,
                )
                * gate_up_scale_ref[...].astype(config.accumulator_dtype)
            ).reshape(config.contraction_tile, config.output_tile).astype(
                jnp.bfloat16
            )
            dimensions = (((1,), (0,)), ((), ()))
            gate_up_accumulator[...] += lax.dot_general(
                hidden_ref[...],
                decoded_gate_up,
                dimension_numbers=dimensions,
                preferred_element_type=config.accumulator_dtype,
            )

            @pl.when(contraction_index == contraction_tiles - 1)
            def store_outputs() -> None:
                gate_up_output_ref[...] = gate_up_accumulator[...].astype(
                    config.output_dtype
                )

        def hidden_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[int, Any]:
            del route_index, output_index
            return 0, contraction_index

        def weight_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, Any, Any]:
            return (
                local_ids_value[route_index],
                contraction_index,
                output_index,
            )

        def scale_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, Any, int, Any]:
            return (
                route_index,
                contraction_index,
                0,
                output_index,
            )

        def output_index(
            output_index_value: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, int, Any]:
            del contraction_index
            return route_index, 0, output_index_value

        hidden_spec = pl.BlockSpec(
            (config.row_tile, config.contraction_tile), hidden_index
        )
        weight_spec = pl.BlockSpec(
            (None, config.contraction_tile, config.output_tile),
            weight_index,
            pipeline_mode=pl.Buffered(buffer_count=3),
        )
        scale_spec = pl.BlockSpec(
            (
                None,
                contraction_blocks_per_tile,
                1,
                config.output_tile,
            ),
            scale_index,
        )
        output_spec = pl.BlockSpec(
            (None, config.row_tile, config.output_tile), output_index
        )
        pipeline = pltpu.emit_pipeline(
            inner_kernel,
            grid=(
                output_tiles,
                active_count_value[...],
                contraction_tiles,
            ),
            in_specs=(
                hidden_spec,
                weight_spec,
                scale_spec,
            ),
            out_specs=output_spec,
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            no_pipelining=interpret,
        )
        pipeline(
            hidden_hbm_ref,
            gate_up_hbm_ref,
            gate_up_scale_hbm_ref,
            gate_up_output_hbm_ref,
            scratches=(gate_up_accumulator_ref,),
        )

    output_shape = jax.ShapeDtypeStruct(
        (route_count, config.row_tile, padded_output), config.output_dtype
    )
    call = pl.pallas_call(
        kernel,
        out_shape=output_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM(
                    (config.row_tile, config.output_tile),
                    config.accumulator_dtype,
                ),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            "greenfield_fp8_selected_up_gate_"
            f"r{route_count}_g{local_experts}_k{padded_contraction}_n{padded_output}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=(
                2
                * route_count
                * config.row_tile
                * padded_contraction
                * padded_output
            ),
            bytes_accessed=(
                config.row_tile * padded_contraction * 2
                + route_count
                * padded_output
                * padded_contraction
                + route_count * output_tiles * contraction_tiles * 4
                + route_count * padded_output * config.row_tile * 2
            ),
            transcendentals=0,
        ),
    )
    gate_up = call(
        compact_local_ids,
        active_count,
        hidden,
        gate_up_raw,
        gate_up_scale_table,
    )
    compact_slot_active = (
        jnp.arange(route_count, dtype=jnp.int32) < active_count
    )[:, None, None]
    gate_up = jnp.where(
        compact_slot_active, gate_up, jnp.zeros_like(gate_up)
    )
    # compact_index is original->compact, restoring the binding top-k order.
    gate_up = gate_up[compact_index, 0, :combined_output]
    return (
        gate_up[:, :projection_output],
        gate_up[:, projection_output:],
    )
