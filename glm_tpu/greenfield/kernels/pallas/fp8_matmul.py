"""TPU-v4 raw-FP8 block matmul without a persistent BF16 weight overlay.

The checkpoint stores E4M3FN payloads as their exact ``uint8`` encodings in
``[out, in]`` order and one FP32 inverse scale per 128x128 block.  TPU v4 has
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
import jax.numpy as jnp
from jax import lax
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
        if self.output_tile % self.block_shape[0] != 0 or (
            self.contraction_tile % self.block_shape[1] != 0
        ):
            raise ValueError(
                "TPU-v4 tiles require integral output and contraction scale "
                "blocks"
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
    if config.output_tile not in (
        config.block_shape[0],
        2 * config.block_shape[0],
    ) or config.contraction_tile != config.block_shape[1]:
        raise ValueError(
            "standalone FP8 matmul output tiles must span one or two scale "
            "blocks and contraction tiles must equal one scale block"
        )
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


def _scale_vector(
    scale_tile: Any,
    contraction_index: Any,
    output_index: Any,
    *,
    blocks_per_tile: int,
    block_output: int,
) -> Any:
    """Load one scalar per output block and expand it over output rows."""

    scale_rows = lax.broadcasted_iota(jnp.int32, scale_tile.shape, 0)
    scale_columns = lax.broadcasted_iota(jnp.int32, scale_tile.shape, 1)
    values = []
    for block_offset in range(blocks_per_tile):
        target_column = (
            output_index * jnp.int32(blocks_per_tile)
            + jnp.int32(block_offset)
        )
        scale_mask = (
            (scale_rows == contraction_index % jnp.int32(8))
            & (scale_columns == target_column)
        )
        value = jnp.sum(
            jnp.where(scale_mask, scale_tile, jnp.float32(0.0)),
            dtype=jnp.float32,
        )
        values.append(jnp.broadcast_to(value, (block_output,)))
    return jnp.concatenate(values, axis=0)[:, None]


def _aligned_scale_table(
    scale: Any, *, output_tiles: int, contraction_tiles: int
) -> Any:
    del output_tiles
    output_scale_blocks = scale.shape[0]
    if output_scale_blocks > 128:
        raise ValueError("FP8 matmul supports at most 128 output-scale blocks")
    aligned_contraction_tiles = _ceil_div(contraction_tiles, 8) * 8
    # Mosaic cannot issue a dynamically indexed scalar VMEM load from the
    # compact [N-block,K-block] table. This bounded metadata staging is 24 KiB
    # for GLM up, compared with 12 MiB of FP8 weight bytes.
    return jnp.pad(
        jnp.transpose(scale),
        (
            (0, aligned_contraction_tiles - contraction_tiles),
            (0, 128 - output_scale_blocks),
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
    # Selected-expert tables use their final MXU access order [G,K,N].  This
    # is intentionally different from the standalone [N,K] checkpoint probe:
    # transposing a complete 64-expert table inside the decode JIT is both an
    # extra full-HBM pass and an extra TPU custom call.
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


def _fp8_block_matmul_impl(
    lhs: Any,
    weight_bits: Any,
    scale: Any,
    *,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(),
    result_dtype: Any,
    kernel_prefix: str,
    interpret: bool = False,
) -> Any:
    """Shared raw-FP8 implementation with an explicit result boundary."""

    rows, contraction, output = _validate_inputs(
        lhs, weight_bits, scale, config
    )
    result_dtype = jnp.dtype(result_dtype)
    if result_dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32)):
        raise ValueError("FP8 block matmul result must be BF16 or FP32")
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

    output_tiles = padded_output // config.output_tile
    contraction_tiles = padded_contraction // config.contraction_tile
    blocks_per_output_tile = config.output_tile // config.block_shape[0]
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
        # one- or two-block VMEM tile, apply its inverse scales, then feed BF16
        # to the MXU while accumulating in FP32.
        output_index = pl.program_id(0)
        # Keep the complete checkpoint operand raw U8 through the custom-call
        # boundary. Reinterpret only the resident VMEM tile: an outer bitcast
        # makes TPU XLA execute a whole-table formatting fusion.
        weight_tile = lax.bitcast_convert_type(
            weight_ref[...], jnp.float8_e4m3fn
        )
        if blocks_per_output_tile == 1:
            scale_value = _scale_value(
                scale_ref[...], contraction_index, output_index
            )
            decoded_weight = (
                weight_tile.astype(config.accumulator_dtype)
                * scale_value.astype(config.accumulator_dtype)
            ).astype(jnp.bfloat16)
        else:
            scale_vector = _scale_vector(
                scale_ref[...],
                contraction_index,
                output_index,
                blocks_per_tile=blocks_per_output_tile,
                block_output=config.block_shape[0],
            )
            decoded_weight = (
                weight_tile.astype(config.accumulator_dtype)
                * scale_vector.astype(config.accumulator_dtype)
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
            output_ref[...] = accumulator_ref[...].astype(result_dtype)

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

    kernel_name = (
        f"{kernel_prefix}m{padded_rows}_k{padded_contraction}_n{padded_output}"
    )
    if config.output_tile != config.block_shape[0]:
        kernel_name += f"_ot{config.output_tile}"
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_output), result_dtype
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
        name=kernel_name,
        cost_estimate=pl.CostEstimate(
            flops=2 * padded_rows * padded_contraction * padded_output,
            bytes_accessed=(
                padded_rows * padded_contraction * 2
                + padded_output * padded_contraction
                + output_tiles
                * contraction_tiles
                * blocks_per_output_tile
                * 4
                + padded_rows * padded_output * result_dtype.itemsize
            ),
            transcendentals=0,
        ),
    )
    return call(lhs, weight_bits, scale_table)[:rows, :output]


def fp8_block_matmul(
    lhs: Any,
    weight_bits: Any,
    scale: Any,
    *,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(),
    interpret: bool = False,
) -> Any:
    """Return a BF16 raw-FP8 projection using tile-local dequantization.

    ``interpret=True`` runs through the Pallas interpreter for semantic tests.
    TPU promotion still requires a compiled kernel correctness proof, HLO/custom
    call assertion, microbenchmark, and real-layer fallback comparison.
    """

    return _fp8_block_matmul_impl(
        lhs,
        weight_bits,
        scale,
        config=config,
        result_dtype=config.output_dtype,
        kernel_prefix="greenfield_fp8_block_matmul_",
        interpret=interpret,
    )


def fp8_block_matmul_f32(
    lhs: Any,
    weight_bits: Any,
    scale: Any,
    *,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(),
    interpret: bool = False,
) -> Any:
    """Return an FP32 raw-FP8 projection for DSA query/key arithmetic.

    The checkpoint tile is decoded to BF16 exactly as in the reference path,
    the MXU accumulates in FP32, and the accumulator is retained as FP32 for
    the DSA LayerNorm/RoPE/scoring path.  No complete decoded matrix is ever
    materialized in HBM.
    """

    return _fp8_block_matmul_impl(
        lhs,
        weight_bits,
        scale,
        config=config,
        result_dtype=jnp.float32,
        kernel_prefix="greenfield_fp8_block_matmul_f32_",
        interpret=interpret,
    )


def fp8_block_vector_matmul_f32(
    lhs: Any,
    weight_bits: Any,
    scale: Any,
    *,
    block_shape: tuple[int, int] = (128, 128),
    interpret: bool = False,
) -> Any:
    """Return a one-row FP32 reduction candidate without a decoded overlay.

    TPU XLA lowers the accepted GLM DSA ``M=1`` query projection to a fused
    FP32 elementwise multiply/reduce rather than an MXU matmul.  This kernel
    preserves that numerical shape while streaming one 128-output raw-FP8
    tile through VMEM/HBM.  Protected TPU comparison decides exactness.  It is
    intentionally restricted to the aligned DSA query geometry; the general
    MXU kernel remains the high-throughput path.
    """

    config = Fp8BlockMatmulConfig(block_shape=block_shape)
    rows, contraction, output = _validate_inputs(
        lhs, weight_bits, scale, config
    )
    block_output, block_contraction = block_shape
    if rows != 1:
        raise ValueError("FP8 vector matmul requires one exact decode row")
    if contraction % block_contraction or output % block_output:
        raise ValueError("FP8 vector matmul requires block-aligned dimensions")
    contraction_blocks = contraction // block_contraction
    output_tiles = output // block_output
    if contraction_blocks > 128:
        raise ValueError("FP8 vector matmul supports at most 128 scale blocks")

    def dequantize_tile(weight_tile: Any, scale_row: Any) -> Any:
        padded_scale = jnp.pad(
            scale_row,
            ((0, 7), (0, 128 - contraction_blocks)),
        )

        def kernel(
            weight_ref: Any,
            scale_ref: Any,
            output_ref: Any,
        ) -> None:
            weight = lax.bitcast_convert_type(
                weight_ref[...], jnp.float8_e4m3fn
            ).astype(jnp.float32)
            scale_rows = lax.broadcasted_iota(
                jnp.int32, (8, 128), 0
            )
            scale_columns = lax.broadcasted_iota(
                jnp.int32, (8, 128), 1
            )
            vectors = []
            for index in range(contraction_blocks):
                value = jnp.sum(
                    jnp.where(
                        (scale_rows == 0) & (scale_columns == index),
                        scale_ref[...],
                        jnp.float32(0.0),
                    ),
                    dtype=jnp.float32,
                )
                vectors.append(
                    jnp.broadcast_to(value, (block_contraction,))
                )
            scale_vector = jnp.concatenate(vectors, axis=0)
            output_ref[...] = weight * scale_vector[None, :]

        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(
                (block_output, contraction), jnp.float32
            ),
            in_specs=(
                pl.BlockSpec((block_output, contraction), lambda: (0, 0)),
                pl.BlockSpec((8, 128), lambda: (0, 0)),
            ),
            out_specs=pl.BlockSpec(
                (block_output, contraction), lambda: (0, 0)
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=(),
                disable_bounds_checks=True,
            ),
            interpret=interpret,
            name=(
                "greenfield_fp8_dequantize_f32_"
                f"n{block_output}_k{contraction}"
            ),
            cost_estimate=pl.CostEstimate(
                flops=contraction * block_output,
                bytes_accessed=(
                    contraction * block_output
                    + 8 * 128 * 4
                    + contraction * block_output * 4
                ),
                transcendentals=0,
            ),
        )
        return call(weight_tile, padded_scale)

    lhs_f32 = lhs.astype(jnp.float32)

    def project_tile(index: Any) -> Any:
        weight_tile = lax.dynamic_slice(
            weight_bits,
            (index * jnp.int32(block_output), jnp.int32(0)),
            (block_output, contraction),
        )
        scale_row = lax.dynamic_slice(
            scale,
            (index, jnp.int32(0)),
            (1, contraction_blocks),
        )
        decoded = dequantize_tile(weight_tile, scale_row)
        return lax.dot_general(
            lhs_f32,
            decoded,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

    projected = lax.map(
        project_tile,
        jnp.arange(output_tiles, dtype=jnp.int32),
    )
    return projected.reshape(1, output)


def fp32_to_bf16_pallas_boundary(
    value: Any,
    *,
    row_tile: int = 8,
    output_tile: int = 128,
    interpret: bool = False,
) -> Any:
    """Cast one rank-two FP32 tensor behind an opaque device boundary.

    This diagnostic kernel prevents TPU XLA from commuting a downstream BF16
    cast into an upstream FP32 collective.  It is intentionally separate from
    the arithmetic kernels so the optimized HLO must expose an FP32 operand at
    the custom-call boundary.
    """

    if value.ndim != 2 or value.dtype != jnp.float32:
        raise ValueError("Pallas precision boundary requires rank-two FP32")
    if (
        not isinstance(row_tile, int)
        or isinstance(row_tile, bool)
        or row_tile <= 0
        or not isinstance(output_tile, int)
        or isinstance(output_tile, bool)
        or output_tile <= 0
    ):
        raise ValueError("Pallas precision-boundary tiles must be positive")
    rows, width = value.shape
    if rows <= 0 or width <= 0:
        raise ValueError("Pallas precision-boundary dimensions must be positive")
    padded_rows = _ceil_div(rows, row_tile) * row_tile
    padded_width = _ceil_div(width, output_tile) * output_tile
    if (padded_rows, padded_width) != value.shape:
        value = jnp.pad(
            value,
            ((0, padded_rows - rows), (0, padded_width - width)),
        )

    def kernel(input_ref: Any, output_ref: Any) -> None:
        output_ref[...] = input_ref[...].astype(jnp.bfloat16)

    def tile_index(
        row_index: Any, output_index: Any
    ) -> tuple[Any, Any]:
        return row_index, output_index

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_width), jnp.bfloat16
        ),
        grid=(padded_rows // row_tile, padded_width // output_tile),
        in_specs=(pl.BlockSpec((row_tile, output_tile), tile_index),),
        out_specs=pl.BlockSpec((row_tile, output_tile), tile_index),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            "greenfield_fp32_to_bf16_"
            f"r{padded_rows}_h{padded_width}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=0,
            bytes_accessed=padded_rows * padded_width * 6,
            transcendentals=0,
        ),
    )(value)
    return call[:rows, :width]


def _validate_structured_kv_b(
    activation: Any,
    weight_bits: Any,
    scale: Any,
    *,
    qk_nope_head_dim: int,
    config: Fp8BlockMatmulConfig,
) -> tuple[int, int, int, int]:
    """Validate the production per-head MLA ``kv_b`` checkpoint layout."""

    if config.block_shape != (128, 128) or config.row_tile != 8:
        raise ValueError("structured kv_b requires TPU-v4 128x128 blocks and row tile 8")
    if config.output_tile != 128 or config.contraction_tile != 128:
        raise ValueError("structured kv_b requires exact 128-wide MXU tiles")
    if activation.ndim != 3 or activation.shape[0] != 1:
        raise ValueError("structured kv_b activation must contain one decode row")
    if activation.dtype != jnp.bfloat16:
        raise ValueError("structured kv_b activation must be BF16")
    if weight_bits.ndim != 2 or weight_bits.dtype != jnp.uint8:
        raise ValueError("structured kv_b weight must be a rank-two uint8 payload")
    if scale.ndim != 2 or scale.dtype != jnp.float32:
        raise ValueError("structured kv_b scale must be a rank-two FP32 table")
    if (
        not isinstance(qk_nope_head_dim, int)
        or isinstance(qk_nope_head_dim, bool)
        or qk_nope_head_dim <= 0
    ):
        raise ValueError("structured kv_b qk_nope width must be positive")
    heads = activation.shape[1]
    if heads <= 0 or weight_bits.shape[0] % heads:
        raise ValueError("structured kv_b rows must divide exactly over heads")
    combined_width = weight_bits.shape[0] // heads
    latent = weight_bits.shape[1]
    value_width = combined_width - qk_nope_head_dim
    if latent != 512 or value_width <= 0:
        raise ValueError("structured kv_b latent/value widths are invalid")
    if combined_width != 448 or qk_nope_head_dim != 192 or value_width != 256:
        raise ValueError("structured kv_b requires the exact GLM 192/256 head contract")
    expected_scale = (
        _ceil_div(weight_bits.shape[0], 128),
        _ceil_div(latent, 128),
    )
    if scale.shape != expected_scale:
        raise ValueError(
            f"structured kv_b scale shape must be {expected_scale}, got {scale.shape}"
        )
    return heads, combined_width, latent, value_width


def fp8_structured_kv_b_q_absorb(
    q_nope: Any,
    weight_bits: Any,
    scale: Any,
    *,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(),
    interpret: bool = False,
) -> Any:
    """Apply the transposed per-head ``kv_b`` key slice from raw FP8.

    The checkpoint stores each head as ``[192 key rows, 256 value rows, 512
    latent columns]`` inside one flattened ``[head * 448, 512]`` matrix.  Head
    starts alternate between 128-row alignment and a 64-row offset.  This call
    pads only the compact query activation into the two aligned source blocks,
    reads the original raw checkpoint blocks without a runtime weight
    transpose, and returns ``[1, heads, 512]`` BF16 absorbed queries.
    """

    heads, combined_width, latent, _ = _validate_structured_kv_b(
        q_nope,
        weight_bits,
        scale,
        qk_nope_head_dim=q_nope.shape[-1],
        config=config,
    )
    if q_nope.shape != (1, heads, 192):
        raise ValueError("structured kv_b q_nope must have shape [1, heads, 192]")
    output_tiles = latent // 128
    contraction_tiles = 2

    head_rows = jnp.transpose(q_nope, (1, 0, 2))
    head_rows = jnp.pad(head_rows, ((0, 0), (0, 7), (0, 0)))
    aligned = jnp.where(
        ((jnp.arange(heads, dtype=jnp.int32) * combined_width) % 128)[
            :, None, None
        ]
        == 0,
        jnp.pad(head_rows, ((0, 0), (0, 0), (0, 64))),
        jnp.pad(head_rows, ((0, 0), (0, 0), (64, 0))),
    )
    aligned = jnp.transpose(
        aligned.reshape(heads, 8, contraction_tiles, 128), (0, 2, 1, 3)
    ).reshape(heads * contraction_tiles, 8, 128)

    head_ids = jnp.arange(heads, dtype=jnp.int32)[:, None]
    source_blocks = (
        head_ids * jnp.int32(combined_width) // jnp.int32(128)
        + jnp.arange(contraction_tiles, dtype=jnp.int32)[None, :]
    )
    selected_scale = scale[source_blocks]
    selected_scale = jnp.repeat(selected_scale[..., None], 128, axis=-1)
    selected_scale = selected_scale.reshape(
        heads * contraction_tiles * output_tiles, 1, 128
    )
    selected_scale = jnp.repeat(selected_scale, 8, axis=1)
    def kernel(
        query_ref: Any,
        weight_ref: Any,
        scale_ref: Any,
        output_ref: Any,
        accumulator_ref: Any,
    ) -> None:
        contraction_index = pl.program_id(1)

        @pl.when(contraction_index == 0)
        def initialize_accumulator() -> None:
            accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

        weight_tile = lax.bitcast_convert_type(
            weight_ref[...], jnp.float8_e4m3fn
        )
        decoded_weight = (
            weight_tile.astype(jnp.float32)
            * scale_ref[0, 0, 0].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        accumulator_ref[...] += lax.dot_general(
            query_ref[0, ...],
            decoded_weight,
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

        @pl.when(contraction_index == contraction_tiles - 1)
        def store_output() -> None:
            output_ref[0, ...] = accumulator_ref[...].astype(jnp.bfloat16)

    def query_index(program_index: Any, contraction_index: Any) -> tuple[Any, int, int]:
        head = program_index // output_tiles
        return head * contraction_tiles + contraction_index, 0, 0

    def weight_index(program_index: Any, contraction_index: Any) -> tuple[Any, Any]:
        head = program_index // output_tiles
        output_index = program_index % output_tiles
        return (
            (head * combined_width) // jnp.int32(128) + contraction_index,
            output_index,
        )

    def scale_index(program_index: Any, contraction_index: Any) -> tuple[Any, int, int]:
        head = program_index // output_tiles
        output_index = program_index % output_tiles
        return (
            (head * contraction_tiles + contraction_index) * output_tiles
            + output_index,
            0,
            0,
        )

    def output_index(program_index: Any, contraction_index: Any) -> tuple[Any, int, int]:
        del contraction_index
        return program_index, 0, 0

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (heads * output_tiles, 8, 128), jnp.bfloat16
        ),
        grid=(heads * output_tiles, contraction_tiles),
        in_specs=(
            pl.BlockSpec((1, 8, 128), query_index),
            pl.BlockSpec((128, 128), weight_index),
            pl.BlockSpec((1, 8, 128), scale_index),
        ),
        out_specs=pl.BlockSpec((1, 8, 128), output_index),
        scratch_shapes=(pltpu.VMEM((8, 128), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            "greenfield_fp8_structured_kv_b_q_absorb_"
            f"h{heads}_p192_l{latent}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=2 * heads * 192 * latent,
            bytes_accessed=(
                heads * contraction_tiles * 8 * 128 * 2
                + heads * contraction_tiles * latent * 128
                + selected_scale.size * 4
                + heads * 8 * latent * 2
            ),
            transcendentals=0,
        ),
    )
    result = call(aligned, weight_bits, selected_scale)
    result = result.reshape(heads, output_tiles, 8, 128)
    result = jnp.transpose(result, (2, 0, 1, 3)).reshape(8, heads, latent)
    return result[:1]


def fp8_structured_kv_b_value(
    attended_latent: Any,
    weight_bits: Any,
    scale: Any,
    *,
    qk_nope_head_dim: int = 192,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(),
    interpret: bool = False,
) -> Any:
    """Apply the per-head ``kv_b`` value slice from raw FP8 tiles."""

    heads, combined_width, latent, value_width = _validate_structured_kv_b(
        attended_latent,
        weight_bits,
        scale,
        qk_nope_head_dim=qk_nope_head_dim,
        config=config,
    )
    if attended_latent.shape != (1, heads, latent):
        raise ValueError(
            "structured kv_b attended latent must have shape [1, heads, 512]"
        )
    contraction_tiles = latent // 128
    output_blocks = 3
    head_rows = jnp.transpose(attended_latent, (1, 0, 2))
    head_rows = jnp.pad(head_rows, ((0, 0), (0, 7), (0, 0)))

    head_ids = jnp.arange(heads, dtype=jnp.int32)[:, None]
    value_starts = head_ids * jnp.int32(combined_width) + jnp.int32(
        qk_nope_head_dim
    )
    source_blocks = value_starts // jnp.int32(128) + jnp.arange(
        output_blocks, dtype=jnp.int32
    )[None, :]
    active_blocks = jnp.arange(output_blocks, dtype=jnp.int32)[None, :] < (
        (value_starts % jnp.int32(128) + value_width + 127)
        // jnp.int32(128)
    )
    source_blocks = jnp.clip(source_blocks, 0, scale.shape[0] - 1)
    selected_scale = scale[source_blocks]
    selected_scale = jnp.where(active_blocks[..., None], selected_scale, 0.0)
    selected_scale = jnp.repeat(selected_scale[..., None], 128, axis=-1)
    selected_scale = selected_scale.reshape(
        heads * output_blocks * contraction_tiles, 1, 128
    )
    selected_scale = jnp.repeat(selected_scale, 8, axis=1)
    def kernel(
        latent_ref: Any,
        weight_ref: Any,
        scale_ref: Any,
        output_ref: Any,
        accumulator_ref: Any,
    ) -> None:
        contraction_index = pl.program_id(1)

        @pl.when(contraction_index == 0)
        def initialize_accumulator() -> None:
            accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

        weight_tile = lax.bitcast_convert_type(
            weight_ref[...], jnp.float8_e4m3fn
        )
        decoded_weight = (
            weight_tile.astype(jnp.float32)
            * scale_ref[0, 0, 0].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        accumulator_ref[...] += lax.dot_general(
            latent_ref[0, ...],
            decoded_weight,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )

        @pl.when(contraction_index == contraction_tiles - 1)
        def store_output() -> None:
            output_ref[0, ...] = accumulator_ref[...].astype(jnp.bfloat16)

    def latent_index(program_index: Any, contraction_index: Any) -> tuple[Any, int, Any]:
        head = program_index // output_blocks
        return head, 0, contraction_index

    def source_block(program_index: Any) -> Any:
        head = program_index // output_blocks
        output_block = program_index % output_blocks
        block = (
            (head * combined_width + qk_nope_head_dim) // jnp.int32(128)
            + output_block
        )
        return jnp.minimum(block, jnp.int32(scale.shape[0] - 1))

    def weight_index(program_index: Any, contraction_index: Any) -> tuple[Any, Any]:
        return source_block(program_index), contraction_index

    def scale_index(program_index: Any, contraction_index: Any) -> tuple[Any, int, int]:
        return (
            program_index * contraction_tiles + contraction_index,
            0,
            0,
        )

    def output_index(program_index: Any, contraction_index: Any) -> tuple[Any, int, int]:
        del contraction_index
        return program_index, 0, 0

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (heads * output_blocks, 8, 128), jnp.bfloat16
        ),
        grid=(heads * output_blocks, contraction_tiles),
        in_specs=(
            pl.BlockSpec((1, 8, 128), latent_index),
            pl.BlockSpec((128, 128), weight_index),
            pl.BlockSpec((1, 8, 128), scale_index),
        ),
        out_specs=pl.BlockSpec((1, 8, 128), output_index),
        scratch_shapes=(pltpu.VMEM((8, 128), jnp.float32),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(
            "greenfield_fp8_structured_kv_b_value_"
            f"h{heads}_l{latent}_v{value_width}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=2 * heads * latent * value_width,
            bytes_accessed=(
                heads * output_blocks * 8 * latent * 2
                + heads * output_blocks * latent * 128
                + selected_scale.size * 4
                + heads * output_blocks * 8 * 128 * 2
            ),
            transcendentals=0,
        ),
    )
    result = call(head_rows, weight_bits, selected_scale)
    result = result.reshape(heads, output_blocks, 8, 128)[:, :, 0, :]
    aligned = jnp.concatenate((result[:, 0], result[:, 1]), axis=-1)
    unaligned = jnp.concatenate(
        (result[:, 0, 64:], result[:, 1], result[:, 2, :64]), axis=-1
    )
    value_offsets = (
        jnp.arange(heads, dtype=jnp.int32) * combined_width
        + qk_nope_head_dim
    ) % 128
    output = jnp.where(value_offsets[:, None] == 0, aligned, unaligned)
    return output[None, ...].astype(jnp.bfloat16)


def fp8_fused_structured_kv_b_value_output(
    attended_latent: Any,
    kv_b_bits: Any,
    kv_b_scale: Any,
    output_bits: Any,
    output_scale: Any,
    *,
    qk_nope_head_dim: int = 192,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(),
    interpret: bool = False,
) -> Any:
    """Fuse structured value and attention-output projections in one call.

    The structured per-head value blocks are accumulated and repacked in
    VMEM. The second raw-FP8 projection consumes that BF16 scratch directly,
    retaining the reference arithmetic while removing the intermediate HBM
    result and its separate custom-call boundary.
    """

    heads, combined_width, latent, value_width = _validate_structured_kv_b(
        attended_latent,
        kv_b_bits,
        kv_b_scale,
        qk_nope_head_dim=qk_nope_head_dim,
        config=config,
    )
    if attended_latent.shape != (1, heads, latent):
        raise ValueError(
            "fused structured value input must have shape [1, heads, 512]"
        )
    value_contraction = heads * value_width
    if output_bits.ndim != 2 or output_bits.dtype != jnp.uint8:
        raise ValueError("fused attention-output weight must be rank-two U8")
    output_width, output_contraction = output_bits.shape
    if output_width <= 0 or output_contraction != value_contraction:
        raise ValueError(
            "fused attention-output weight must contract all value heads"
        )
    expected_output_scale = (
        _ceil_div(output_width, config.block_shape[0]),
        _ceil_div(value_contraction, config.block_shape[1]),
    )
    if output_scale.shape != expected_output_scale or (
        output_scale.dtype != jnp.float32
    ):
        raise ValueError(
            "fused attention-output FP32 scale shape must be "
            f"{expected_output_scale}, got "
            f"{output_scale.shape}/{output_scale.dtype}"
        )
    if value_contraction % config.contraction_tile != 0 or (
        output_width % config.output_tile != 0
    ):
        raise ValueError(
            "fused attention-output production geometry must be tile aligned"
        )

    value_contraction_tiles = latent // config.contraction_tile
    value_output_blocks = 3
    head_rows = jnp.transpose(attended_latent, (1, 0, 2))
    head_rows = jnp.pad(
        head_rows,
        ((0, 0), (0, config.row_tile - 1), (0, 0)),
    )
    head_ids = jnp.arange(heads, dtype=jnp.int32)[:, None]
    value_starts = head_ids * jnp.int32(combined_width) + jnp.int32(
        qk_nope_head_dim
    )
    source_blocks = value_starts // jnp.int32(128) + jnp.arange(
        value_output_blocks, dtype=jnp.int32
    )[None, :]
    active_blocks = jnp.arange(
        value_output_blocks, dtype=jnp.int32
    )[None, :] < (
        (value_starts % jnp.int32(128) + value_width + 127)
        // jnp.int32(128)
    )
    source_blocks = jnp.clip(source_blocks, 0, kv_b_scale.shape[0] - 1)
    selected_value_scale = kv_b_scale[source_blocks]
    selected_value_scale = jnp.where(
        active_blocks[..., None], selected_value_scale, 0.0
    )
    selected_value_scale = jnp.repeat(
        selected_value_scale[..., None], 128, axis=-1
    ).reshape(
        heads * value_output_blocks * value_contraction_tiles,
        1,
        128,
    )
    selected_value_scale = jnp.repeat(
        selected_value_scale, config.row_tile, axis=1
    )
    output_tiles = output_width // config.output_tile
    output_contraction_tiles = (
        value_contraction // config.contraction_tile
    )
    output_scale_table = _aligned_scale_table(
        output_scale,
        output_tiles=output_tiles,
        contraction_tiles=output_contraction_tiles,
    )

    def kernel(
        latent_hbm_ref: Any,
        kv_b_hbm_ref: Any,
        kv_b_scale_hbm_ref: Any,
        output_weight_hbm_ref: Any,
        output_scale_hbm_ref: Any,
        output_hbm_ref: Any,
        raw_value_vmem_ref: Any,
        value_vmem_ref: Any,
        value_accumulator_ref: Any,
        output_accumulator_ref: Any,
    ) -> None:
        def value_kernel(
            latent_ref: Any,
            weight_ref: Any,
            scale_ref: Any,
            value_ref: Any,
            accumulator_ref: Any,
        ) -> None:
            contraction_index = pl.program_id(1)

            @pl.when(contraction_index == 0)
            def initialize_accumulator() -> None:
                accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

            decoded_weight = (
                lax.bitcast_convert_type(
                    weight_ref[...], jnp.float8_e4m3fn
                ).astype(config.accumulator_dtype)
                * scale_ref[0, 0, 0].astype(config.accumulator_dtype)
            ).astype(config.output_dtype)
            accumulator_ref[...] += lax.dot_general(
                latent_ref[0, ...],
                decoded_weight,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=config.accumulator_dtype,
            )

            @pl.when(
                contraction_index == value_contraction_tiles - 1
            )
            def store_value() -> None:
                value_ref[0, ...] = accumulator_ref[...].astype(
                    config.output_dtype
                )

        def value_latent_index(
            program_index: Any, contraction_index: Any
        ) -> tuple[Any, int, Any]:
            head = program_index // value_output_blocks
            return head, 0, contraction_index

        def value_source_block(program_index: Any) -> Any:
            head = program_index // value_output_blocks
            output_block = program_index % value_output_blocks
            block = (
                (head * combined_width + qk_nope_head_dim)
                // jnp.int32(128)
                + output_block
            )
            return jnp.minimum(
                block, jnp.int32(kv_b_scale.shape[0] - 1)
            )

        def value_weight_index(
            program_index: Any, contraction_index: Any
        ) -> tuple[Any, Any]:
            return value_source_block(program_index), contraction_index

        def value_scale_index(
            program_index: Any, contraction_index: Any
        ) -> tuple[Any, int, int]:
            return (
                program_index * value_contraction_tiles
                + contraction_index,
                0,
                0,
            )

        def raw_value_index(
            program_index: Any, contraction_index: Any
        ) -> tuple[Any, int, int]:
            del contraction_index
            return program_index, 0, 0

        value_pipeline = pltpu.emit_pipeline(
            value_kernel,
            grid=(
                heads * value_output_blocks,
                value_contraction_tiles,
            ),
            in_specs=(
                pl.BlockSpec((1, config.row_tile, 128), value_latent_index),
                pl.BlockSpec((128, 128), value_weight_index),
                pl.BlockSpec((1, config.row_tile, 128), value_scale_index),
            ),
            out_specs=pl.BlockSpec(
                (1, config.row_tile, 128), raw_value_index
            ),
            dimension_semantics=("parallel", "arbitrary"),
            no_pipelining=interpret,
        )
        value_pipeline(
            latent_hbm_ref,
            kv_b_hbm_ref,
            kv_b_scale_hbm_ref,
            raw_value_vmem_ref,
            scratches=(value_accumulator_ref,),
        )

        def repack_kernel(
            raw_value_ref: Any,
            packed_value_ref: Any,
        ) -> None:
            head = pl.program_id(0)
            raw_value = raw_value_ref[...]
            aligned = jnp.concatenate(
                (raw_value[0], raw_value[1]), axis=-1
            )
            unaligned = jnp.concatenate(
                (
                    raw_value[0, :, 64:],
                    raw_value[1],
                    raw_value[2, :, :64],
                ),
                axis=-1,
            )
            value_offset = (
                head * combined_width + qk_nope_head_dim
            ) % jnp.int32(128)
            packed_value_ref[...] = jnp.where(
                value_offset == 0, aligned, unaligned
            )

        def raw_repack_index(head: Any) -> tuple[Any, int, int]:
            return head, 0, 0

        def packed_value_index(head: Any) -> tuple[int, Any]:
            return 0, head

        repack_pipeline = pltpu.emit_pipeline(
            repack_kernel,
            grid=(heads,),
            in_specs=pl.BlockSpec(
                (value_output_blocks, config.row_tile, 128),
                raw_repack_index,
            ),
            out_specs=pl.BlockSpec(
                (config.row_tile, value_width), packed_value_index
            ),
            dimension_semantics=("parallel",),
            no_pipelining=interpret,
        )
        repack_pipeline(raw_value_vmem_ref, value_vmem_ref)

        def output_kernel(
            value_ref: Any,
            weight_ref: Any,
            scale_ref: Any,
            result_ref: Any,
            accumulator_ref: Any,
        ) -> None:
            output_index = pl.program_id(0)
            contraction_index = pl.program_id(1)

            @pl.when(contraction_index == 0)
            def initialize_accumulator() -> None:
                accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

            scale_value = _scale_value(
                scale_ref[...], contraction_index, output_index
            )
            decoded_weight = (
                lax.bitcast_convert_type(
                    weight_ref[...], jnp.float8_e4m3fn
                ).astype(config.accumulator_dtype)
                * scale_value.astype(config.accumulator_dtype)
            ).astype(config.output_dtype)
            accumulator_ref[...] += lax.dot_general(
                value_ref[...],
                decoded_weight,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=config.accumulator_dtype,
            )

            @pl.when(
                contraction_index == output_contraction_tiles - 1
            )
            def store_output() -> None:
                result_ref[...] = accumulator_ref[...].astype(
                    config.output_dtype
                )

        def packed_input_index(
            output_index: Any, contraction_index: Any
        ) -> tuple[int, Any]:
            del output_index
            return 0, contraction_index

        def output_weight_index(
            output_index: Any, contraction_index: Any
        ) -> tuple[Any, Any]:
            return output_index, contraction_index

        def output_scale_index(
            output_index: Any, contraction_index: Any
        ) -> tuple[Any, int]:
            del output_index
            return contraction_index // jnp.int32(8), 0

        def output_result_index(
            output_index: Any, contraction_index: Any
        ) -> tuple[int, Any]:
            del contraction_index
            return 0, output_index

        output_pipeline = pltpu.emit_pipeline(
            output_kernel,
            grid=(output_tiles, output_contraction_tiles),
            in_specs=(
                pl.BlockSpec(
                    (config.row_tile, config.contraction_tile),
                    packed_input_index,
                ),
                pl.BlockSpec(
                    (config.output_tile, config.contraction_tile),
                    output_weight_index,
                    pipeline_mode=pl.Buffered(buffer_count=3),
                ),
                pl.BlockSpec((8, 128), output_scale_index),
            ),
            out_specs=pl.BlockSpec(
                (config.row_tile, config.output_tile),
                output_result_index,
            ),
            dimension_semantics=("parallel", "arbitrary"),
            no_pipelining=interpret,
        )
        output_pipeline(
            value_vmem_ref,
            output_weight_hbm_ref,
            output_scale_hbm_ref,
            output_hbm_ref,
            scratches=(output_accumulator_ref,),
        )

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (config.row_tile, output_width), config.output_dtype
        ),
        grid=(),
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 5,
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=(
            pltpu.VMEM(
                (
                    heads * value_output_blocks,
                    config.row_tile,
                    config.block_shape[0],
                ),
                config.output_dtype,
            ),
            pltpu.VMEM(
                (config.row_tile, value_contraction),
                config.output_dtype,
            ),
            pltpu.VMEM(
                (config.row_tile, config.block_shape[0]),
                config.accumulator_dtype,
            ),
            pltpu.VMEM(
                (config.row_tile, config.output_tile),
                config.accumulator_dtype,
            ),
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=interpret,
        name=(
            "greenfield_fp8_fused_structured_value_output_"
            f"h{heads}_l{latent}_v{value_width}_o{output_width}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=(
                2 * heads * latent * value_width
                + 2
                * config.row_tile
                * value_contraction
                * output_width
            ),
            bytes_accessed=(
                heads * config.row_tile * latent * 2
                + heads
                * value_output_blocks
                * latent
                * config.block_shape[0]
                + selected_value_scale.size * 4
                + output_width * value_contraction
                + output_tiles * output_contraction_tiles * 4
                + config.row_tile * output_width * 2
            ),
            transcendentals=0,
        ),
    )
    return call(
        head_rows,
        kv_b_bits,
        selected_value_scale,
        output_bits,
        output_scale_table,
    )[:1, :output_width]


def fp8_rmsnorm_block_matmul(
    hidden: Any,
    norm_weight: Any,
    weight_bits: Any,
    scale: Any,
    *,
    epsilon: float,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(),
    interpret: bool = False,
) -> Any:
    """Fuse exact batch-one RMSNorm pointwise work into a raw-FP8 linear.

    The FP32 square/mean/rsqrt reduction remains an ordinary JAX reduction.
    The Pallas call consumes the original BF16 row plus that one FP32 inverse
    RMS value, performs the contract's BF16 rounding and norm-weight multiply
    in VMEM, and immediately feeds the result to the raw-FP8 matmul.  No
    normalized activation or decoded weight matrix is materialized in HBM.
    """

    rows, contraction, output = _validate_inputs(
        hidden, weight_bits, scale, config
    )
    if rows != 1:
        raise ValueError("fused stage-local RMSNorm requires one decode row")
    if norm_weight.shape != (contraction,):
        raise ValueError(
            "RMSNorm weight must match the linear contraction width"
        )
    if norm_weight.dtype != jnp.bfloat16:
        raise ValueError("fused stage-local RMSNorm weight must be BF16")
    if (
        not isinstance(epsilon, (int, float))
        or isinstance(epsilon, bool)
        or epsilon <= 0
    ):
        raise ValueError("fused stage-local RMSNorm epsilon must be positive")

    padded_rows = _ceil_div(rows, config.row_tile) * config.row_tile
    padded_contraction = (
        _ceil_div(contraction, config.contraction_tile)
        * config.contraction_tile
    )
    padded_output = _ceil_div(output, config.output_tile) * config.output_tile
    hidden_f32 = hidden.astype(jnp.float32)
    inverse_rms = lax.rsqrt(
        jnp.mean(lax.square(hidden_f32), axis=-1, keepdims=True)
        + jnp.float32(epsilon)
    )
    hidden = jnp.pad(
        hidden,
        ((0, padded_rows - rows), (0, padded_contraction - contraction)),
    )
    inverse_rms = jnp.pad(inverse_rms, ((0, padded_rows - rows), (0, 0)))
    norm_weight = jnp.pad(
        norm_weight, (0, padded_contraction - contraction)
    )[None, :]
    weight_bits = jnp.pad(
        weight_bits,
        (
            (0, padded_output - output),
            (0, padded_contraction - contraction),
        ),
    )
    output_tiles = padded_output // config.output_tile
    contraction_tiles = padded_contraction // config.contraction_tile
    scale_table = _aligned_scale_table(
        scale,
        output_tiles=output_tiles,
        contraction_tiles=contraction_tiles,
    )

    def kernel(
        hidden_ref: Any,
        norm_ref: Any,
        inverse_rms_ref: Any,
        weight_ref: Any,
        scale_ref: Any,
        output_ref: Any,
        accumulator_ref: Any,
    ) -> None:
        contraction_index = pl.program_id(1)

        @pl.when(contraction_index == 0)
        def initialize_accumulator() -> None:
            accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

        output_index = pl.program_id(0)
        normalized = (
            hidden_ref[...].astype(jnp.float32)
            * inverse_rms_ref[...].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        normalized = (
            normalized * norm_ref[...].astype(jnp.bfloat16)
        ).astype(jnp.bfloat16)
        scale_value = _scale_value(
            scale_ref[...], contraction_index, output_index
        )
        weight_tile = lax.bitcast_convert_type(
            weight_ref[...], jnp.float8_e4m3fn
        )
        decoded_weight = (
            weight_tile.astype(config.accumulator_dtype)
            * scale_value.astype(config.accumulator_dtype)
        ).astype(jnp.bfloat16)
        accumulator_ref[...] += lax.dot_general(
            normalized,
            decoded_weight,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=config.accumulator_dtype,
        )

        @pl.when(contraction_index == contraction_tiles - 1)
        def store_output() -> None:
            output_ref[...] = accumulator_ref[...].astype(config.output_dtype)

    def contraction_index(
        output_index: Any, contraction_index_value: Any
    ) -> tuple[int, Any]:
        del output_index
        return 0, contraction_index_value

    def weight_index(
        output_index: Any, contraction_index_value: Any
    ) -> tuple[Any, Any]:
        return output_index, contraction_index_value

    def output_index(
        output_index_value: Any, contraction_index_value: Any
    ) -> tuple[int, Any]:
        del contraction_index_value
        return 0, output_index_value

    def scale_index(
        output_index_value: Any, contraction_index_value: Any
    ) -> tuple[Any, int]:
        del output_index_value
        return contraction_index_value // jnp.int32(8), 0

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_output), config.output_dtype
        ),
        grid=(output_tiles, contraction_tiles),
        in_specs=(
            pl.BlockSpec(
                (padded_rows, config.contraction_tile), contraction_index
            ),
            pl.BlockSpec((1, config.contraction_tile), contraction_index),
            pl.BlockSpec((padded_rows, 1), lambda *_: (0, 0)),
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
            "greenfield_fp8_rmsnorm_block_matmul_"
            f"m{padded_rows}_k{padded_contraction}_n{padded_output}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=(
                2 * padded_rows * padded_contraction * padded_output
                + 4 * rows * contraction
            ),
            bytes_accessed=(
                padded_rows * padded_contraction * 2
                + padded_contraction * 2
                + rows * 4
                + padded_output * padded_contraction
                + output_tiles * contraction_tiles * 4
                + padded_rows * padded_output * 2
            ),
            transcendentals=rows,
        ),
    )
    return call(
        hidden,
        norm_weight,
        inverse_rms,
        weight_bits,
        scale_table,
    )[:rows, :output]


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

    gate_bits = pad_weight(gate_bits)
    up_bits = pad_weight(up_bits)
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
        gate_tile = lax.bitcast_convert_type(
            gate_ref[...], jnp.float8_e4m3fn
        )
        up_tile = lax.bitcast_convert_type(
            up_ref[...], jnp.float8_e4m3fn
        )
        decoded_gate = (
            gate_tile.astype(config.accumulator_dtype) * gate_value
        ).astype(jnp.bfloat16)
        decoded_up = (
            up_tile.astype(config.accumulator_dtype) * up_value
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
        lhs, gate_bits, gate_scale_table, up_bits, up_scale_table
    )
    return gate[:rows, :output], up[:rows, :output]


def fp8_fused_block_swiglu(
    lhs: Any,
    gate_bits: Any,
    gate_scale: Any,
    up_bits: Any,
    up_scale: Any,
    down_bits: Any,
    down_scale: Any,
    *,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(),
    interpret: bool = False,
) -> Any:
    """Execute raw-FP8 gate/up, exact BF16 SwiGLU, and down in one call.

    Complete checkpoint matrices remain raw U8 HBM operands. Gate and up
    outputs are VMEM scratch shared by two nested Mosaic pipelines, so only
    the final BF16 projection crosses the Pallas/HBM boundary.
    """

    rows, contraction, intermediate = _validate_inputs(
        lhs, gate_bits, gate_scale, config
    )
    up_geometry = _validate_inputs(lhs, up_bits, up_scale, config)
    if up_geometry != (rows, contraction, intermediate):
        raise ValueError("fused block gate/up projection geometries disagree")
    if down_bits.ndim != 2 or down_bits.dtype != jnp.uint8:
        raise ValueError("fused block down weights must be rank-two U8")
    output, down_contraction = down_bits.shape
    if output <= 0 or down_contraction != intermediate:
        raise ValueError(
            "fused block down weight must contract the gate/up output"
        )
    expected_down_scale = (
        _ceil_div(output, config.block_shape[0]),
        _ceil_div(intermediate, config.block_shape[1]),
    )
    if down_scale.shape != expected_down_scale or (
        down_scale.dtype != jnp.float32
    ):
        raise ValueError(
            "fused block down FP32 scale shape must be "
            f"{expected_down_scale}, got {down_scale.shape}/{down_scale.dtype}"
        )

    padded_rows = _ceil_div(rows, config.row_tile) * config.row_tile
    padded_contraction = (
        _ceil_div(contraction, config.contraction_tile)
        * config.contraction_tile
    )
    padded_intermediate = (
        _ceil_div(intermediate, config.output_tile) * config.output_tile
    )
    padded_output = _ceil_div(output, config.output_tile) * config.output_tile
    lhs = jnp.pad(
        lhs,
        (
            (0, padded_rows - rows),
            (0, padded_contraction - contraction),
        ),
    )

    def pad_projection(value: Any) -> Any:
        return jnp.pad(
            value,
            (
                (0, padded_intermediate - intermediate),
                (0, padded_contraction - contraction),
            ),
        )

    down_raw = jnp.pad(
        down_bits,
        (
            (0, padded_output - output),
            (0, padded_intermediate - intermediate),
        ),
    )
    gate_raw = pad_projection(gate_bits)
    up_raw = pad_projection(up_bits)
    up_output_tiles = padded_intermediate // config.output_tile
    up_contraction_tiles = padded_contraction // config.contraction_tile
    down_output_tiles = padded_output // config.output_tile
    down_contraction_tiles = padded_intermediate // config.contraction_tile
    gate_scale_table = _aligned_scale_table(
        gate_scale,
        output_tiles=up_output_tiles,
        contraction_tiles=up_contraction_tiles,
    )
    up_scale_table = _aligned_scale_table(
        up_scale,
        output_tiles=up_output_tiles,
        contraction_tiles=up_contraction_tiles,
    )
    down_scale_table = _aligned_scale_table(
        down_scale,
        output_tiles=down_output_tiles,
        contraction_tiles=down_contraction_tiles,
    )

    def kernel(
        lhs_hbm_ref: Any,
        gate_hbm_ref: Any,
        gate_scale_hbm_ref: Any,
        up_hbm_ref: Any,
        up_scale_hbm_ref: Any,
        down_hbm_ref: Any,
        down_scale_hbm_ref: Any,
        output_hbm_ref: Any,
        gate_vmem_ref: Any,
        up_vmem_ref: Any,
        gate_accumulator_ref: Any,
        up_accumulator_ref: Any,
        down_accumulator_ref: Any,
    ) -> None:
        def up_gate_kernel(
            lhs_ref: Any,
            gate_ref: Any,
            gate_scale_ref: Any,
            up_ref: Any,
            up_scale_ref: Any,
            gate_output_ref: Any,
            up_output_ref: Any,
            gate_accumulator: Any,
            up_accumulator: Any,
        ) -> None:
            output_index = pl.program_id(0)
            contraction_index = pl.program_id(1)

            @pl.when(contraction_index == 0)
            def initialize_accumulators() -> None:
                gate_accumulator[...] = jnp.zeros_like(gate_accumulator)
                up_accumulator[...] = jnp.zeros_like(up_accumulator)

            gate_value = _scale_value(
                gate_scale_ref[...], contraction_index, output_index
            )
            up_value = _scale_value(
                up_scale_ref[...], contraction_index, output_index
            )
            decoded_gate = (
                lax.bitcast_convert_type(
                    gate_ref[...], jnp.float8_e4m3fn
                ).astype(config.accumulator_dtype)
                * gate_value
            ).astype(jnp.bfloat16)
            decoded_up = (
                lax.bitcast_convert_type(
                    up_ref[...], jnp.float8_e4m3fn
                ).astype(config.accumulator_dtype)
                * up_value
            ).astype(jnp.bfloat16)
            dimensions = (((1,), (1,)), ((), ()))
            gate_accumulator[...] += lax.dot_general(
                lhs_ref[...],
                decoded_gate,
                dimension_numbers=dimensions,
                preferred_element_type=config.accumulator_dtype,
            )
            up_accumulator[...] += lax.dot_general(
                lhs_ref[...],
                decoded_up,
                dimension_numbers=dimensions,
                preferred_element_type=config.accumulator_dtype,
            )

            @pl.when(contraction_index == up_contraction_tiles - 1)
            def store_outputs() -> None:
                gate_output_ref[...] = gate_accumulator[...].astype(
                    config.output_dtype
                )
                up_output_ref[...] = up_accumulator[...].astype(
                    config.output_dtype
                )

        def lhs_index(
            output_index: Any, contraction_index: Any
        ) -> tuple[int, Any]:
            del output_index
            return 0, contraction_index

        def projection_index(
            output_index: Any, contraction_index: Any
        ) -> tuple[Any, Any]:
            return output_index, contraction_index

        def scale_index(
            output_index: Any, contraction_index: Any
        ) -> tuple[Any, int]:
            del output_index
            return contraction_index // jnp.int32(8), 0

        def projection_output_index(
            output_index: Any, contraction_index: Any
        ) -> tuple[int, Any]:
            del contraction_index
            return 0, output_index

        lhs_spec = pl.BlockSpec(
            (padded_rows, config.contraction_tile), lhs_index
        )
        projection_spec = pl.BlockSpec(
            (config.output_tile, config.contraction_tile),
            projection_index,
            pipeline_mode=pl.Buffered(buffer_count=3),
        )
        scale_spec = pl.BlockSpec((8, 128), scale_index)
        projection_output_spec = pl.BlockSpec(
            (padded_rows, config.output_tile), projection_output_index
        )
        up_gate_pipeline = pltpu.emit_pipeline(
            up_gate_kernel,
            grid=(up_output_tiles, up_contraction_tiles),
            in_specs=(
                lhs_spec,
                projection_spec,
                scale_spec,
                projection_spec,
                scale_spec,
            ),
            out_specs=(projection_output_spec, projection_output_spec),
            dimension_semantics=("parallel", "arbitrary"),
            no_pipelining=interpret,
        )
        up_gate_pipeline(
            lhs_hbm_ref,
            gate_hbm_ref,
            gate_scale_hbm_ref,
            up_hbm_ref,
            up_scale_hbm_ref,
            gate_vmem_ref,
            up_vmem_ref,
            scratches=(gate_accumulator_ref, up_accumulator_ref),
        )

        def down_kernel(
            gate_ref: Any,
            up_ref: Any,
            down_ref: Any,
            scale_ref: Any,
            output_ref: Any,
            accumulator: Any,
        ) -> None:
            output_index = pl.program_id(0)
            contraction_index = pl.program_id(1)

            @pl.when(contraction_index == 0)
            def initialize_accumulator() -> None:
                accumulator[...] = jnp.zeros_like(accumulator)

            activated = (
                gate_ref[...]
                * jax.nn.sigmoid(gate_ref[...])
                * up_ref[...]
            ).astype(jnp.bfloat16)
            scale_value = _scale_value(
                scale_ref[...], contraction_index, output_index
            )
            decoded_down = (
                lax.bitcast_convert_type(
                    down_ref[...], jnp.float8_e4m3fn
                ).astype(config.accumulator_dtype)
                * scale_value
            ).astype(jnp.bfloat16)
            accumulator[...] += lax.dot_general(
                activated,
                decoded_down,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=config.accumulator_dtype,
            )

            @pl.when(contraction_index == down_contraction_tiles - 1)
            def store_output() -> None:
                output_ref[...] = accumulator[...].astype(
                    config.output_dtype
                )

        def activation_index(
            output_index: Any, contraction_index: Any
        ) -> tuple[int, Any]:
            del output_index
            return 0, contraction_index

        def down_output_index(
            output_index: Any, contraction_index: Any
        ) -> tuple[int, Any]:
            del contraction_index
            return 0, output_index

        activation_spec = pl.BlockSpec(
            (padded_rows, config.contraction_tile), activation_index
        )
        down_output_spec = pl.BlockSpec(
            (padded_rows, config.output_tile), down_output_index
        )
        down_pipeline = pltpu.emit_pipeline(
            down_kernel,
            grid=(down_output_tiles, down_contraction_tiles),
            in_specs=(
                activation_spec,
                activation_spec,
                projection_spec,
                scale_spec,
            ),
            out_specs=down_output_spec,
            dimension_semantics=("parallel", "arbitrary"),
            no_pipelining=interpret,
        )
        down_pipeline(
            gate_vmem_ref,
            up_vmem_ref,
            down_hbm_ref,
            down_scale_hbm_ref,
            output_hbm_ref,
            scratches=(down_accumulator_ref,),
        )

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(
            (padded_rows, padded_output), config.output_dtype
        ),
        grid=(),
        in_specs=(pl.BlockSpec(memory_space=pltpu.HBM),) * 7,
        out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
        scratch_shapes=(
            pltpu.VMEM(
                (padded_rows, padded_intermediate), config.output_dtype
            ),
            pltpu.VMEM(
                (padded_rows, padded_intermediate), config.output_dtype
            ),
            pltpu.VMEM(
                (padded_rows, config.output_tile),
                config.accumulator_dtype,
            ),
            pltpu.VMEM(
                (padded_rows, config.output_tile),
                config.accumulator_dtype,
            ),
            pltpu.VMEM(
                (padded_rows, config.output_tile),
                config.accumulator_dtype,
            ),
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=interpret,
        name=(
            "greenfield_fp8_fused_block_swiglu_"
            f"m{padded_rows}_h{padded_contraction}_i{padded_intermediate}"
            f"_o{padded_output}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=(
                4 * padded_rows * padded_contraction * padded_intermediate
                + 2 * padded_rows * padded_intermediate * padded_output
            ),
            bytes_accessed=(
                padded_rows * padded_contraction * 2
                + 2 * padded_intermediate * padded_contraction
                + padded_output * padded_intermediate
                + 2 * up_output_tiles * up_contraction_tiles * 4
                + down_output_tiles * down_contraction_tiles * 4
                + padded_rows * padded_output * 2
            ),
            transcendentals=padded_rows * padded_intermediate,
        ),
    )
    return call(
        lhs,
        gate_raw,
        gate_scale_table,
        up_raw,
        up_scale_table,
        down_raw,
        down_scale_table,
    )[:rows, :output]


def fp8_selected_up_gate(
    hidden: Any,
    route_indices: Any,
    expert_start: Any,
    gate_bits: Any,
    gate_scale: Any,
    up_bits: Any,
    up_scale: Any,
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

    route_count, local_experts, contraction, output = _validate_selected_inputs(
        hidden,
        route_indices,
        expert_start,
        gate_bits,
        gate_scale,
        config,
    )
    up_geometry = _validate_selected_inputs(
        hidden,
        route_indices,
        expert_start,
        up_bits,
        up_scale,
        config,
    )
    if up_geometry != (route_count, local_experts, contraction, output):
        raise ValueError("selected expert gate/up geometries disagree")

    padded_contraction = (
        _ceil_div(contraction, config.contraction_tile)
        * config.contraction_tile
    )
    padded_output = _ceil_div(output, config.output_tile) * config.output_tile
    hidden = jnp.pad(
        hidden,
        (
            (0, config.row_tile - 1),
            (0, padded_contraction - contraction),
        ),
    )

    def pad_weight(value: Any) -> Any:
        if padded_output == output and padded_contraction == contraction:
            return value
        return jnp.pad(
            value,
            (
                (0, 0),
                (0, padded_contraction - contraction),
                (0, padded_output - output),
            ),
        )

    # Keep complete expert tables as checkpoint-native U8 Pallas operands.
    # Bitcasting only each DMA'd VMEM tile prevents a complete F8 table view
    # from entering the optimized HLO contract.
    gate_raw = pad_weight(gate_bits)
    up_raw = pad_weight(up_bits)
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
    gate_scale_table = _selected_scale_table(
        gate_scale,
        compact_local_ids,
        padded_output=padded_output,
        padded_contraction=padded_contraction,
        block_shape=config.block_shape,
    )
    up_scale_table = _selected_scale_table(
        up_scale,
        compact_local_ids,
        padded_output=padded_output,
        padded_contraction=padded_contraction,
        block_shape=config.block_shape,
    )

    def kernel(
        local_ids_value: Any,
        active_count_value: Any,
        hidden_hbm_ref: Any,
        gate_hbm_ref: Any,
        gate_scale_hbm_ref: Any,
        up_hbm_ref: Any,
        up_scale_hbm_ref: Any,
        gate_output_hbm_ref: Any,
        up_output_hbm_ref: Any,
        gate_accumulator_ref: Any,
        up_accumulator_ref: Any,
    ) -> None:
        def inner_kernel(
            hidden_ref: Any,
            gate_ref: Any,
            gate_scale_ref: Any,
            up_ref: Any,
            up_scale_ref: Any,
            gate_output_ref: Any,
            up_output_ref: Any,
            gate_accumulator: Any,
            up_accumulator: Any,
        ) -> None:
            output_index = pl.program_id(0)
            route_index = pl.program_id(1)
            contraction_index = pl.program_id(2)

            @pl.when(contraction_index == 0)
            def initialize_accumulators() -> None:
                gate_accumulator[...] = jnp.zeros_like(gate_accumulator)
                up_accumulator[...] = jnp.zeros_like(up_accumulator)

            decoded_gate = (
                lax.bitcast_convert_type(
                    gate_ref[...], jnp.float8_e4m3fn
                )
                .astype(config.accumulator_dtype)
                .reshape(
                    contraction_blocks_per_tile,
                    config.block_shape[1],
                    config.output_tile,
                )
                * gate_scale_ref[...].astype(config.accumulator_dtype)
            ).reshape(config.contraction_tile, config.output_tile).astype(
                jnp.bfloat16
            )
            decoded_up = (
                lax.bitcast_convert_type(
                    up_ref[...], jnp.float8_e4m3fn
                )
                .astype(config.accumulator_dtype)
                .reshape(
                    contraction_blocks_per_tile,
                    config.block_shape[1],
                    config.output_tile,
                )
                * up_scale_ref[...].astype(config.accumulator_dtype)
            ).reshape(config.contraction_tile, config.output_tile).astype(
                jnp.bfloat16
            )
            dimensions = (((1,), (0,)), ((), ()))
            gate_accumulator[...] += lax.dot_general(
                hidden_ref[...],
                decoded_gate,
                dimension_numbers=dimensions,
                preferred_element_type=config.accumulator_dtype,
            )
            up_accumulator[...] += lax.dot_general(
                hidden_ref[...],
                decoded_up,
                dimension_numbers=dimensions,
                preferred_element_type=config.accumulator_dtype,
            )

            @pl.when(contraction_index == contraction_tiles - 1)
            def store_outputs() -> None:
                gate_output_ref[...] = gate_accumulator[...].astype(
                    config.output_dtype
                )
                up_output_ref[...] = up_accumulator[...].astype(
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
                weight_spec,
                scale_spec,
            ),
            out_specs=(output_spec, output_spec),
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            no_pipelining=interpret,
        )
        pipeline(
            hidden_hbm_ref,
            gate_hbm_ref,
            gate_scale_hbm_ref,
            up_hbm_ref,
            up_scale_hbm_ref,
            gate_output_hbm_ref,
            up_output_hbm_ref,
            scratches=(gate_accumulator_ref, up_accumulator_ref),
        )

    output_shape = jax.ShapeDtypeStruct(
        (route_count, config.row_tile, padded_output), config.output_dtype
    )
    call = pl.pallas_call(
        kernel,
        out_shape=(output_shape, output_shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            scratch_shapes=(
                pltpu.VMEM(
                    (config.row_tile, config.output_tile),
                    config.accumulator_dtype,
                ),
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
                4
                * route_count
                * config.row_tile
                * padded_contraction
                * padded_output
            ),
            bytes_accessed=(
                config.row_tile * padded_contraction * 2
                + 2
                * route_count
                * padded_output
                * padded_contraction
                + 2 * route_count * output_tiles * contraction_tiles * 4
                + 2 * route_count * padded_output * config.row_tile * 2
            ),
            transcendentals=0,
        ),
    )
    gate, up = call(
        compact_local_ids,
        active_count,
        hidden,
        gate_raw,
        gate_scale_table,
        up_raw,
        up_scale_table,
    )
    compact_slot_active = (
        jnp.arange(route_count, dtype=jnp.int32) < active_count
    )[:, None, None]
    gate = jnp.where(compact_slot_active, gate, jnp.zeros_like(gate))
    up = jnp.where(compact_slot_active, up, jnp.zeros_like(up))
    # compact_index is original->compact, restoring the binding top-k order.
    gate = gate[compact_index]
    up = up[compact_index]
    return gate[:, 0, :output], up[:, 0, :output]


def fp8_selected_swiglu_down(
    gate: Any,
    up: Any,
    route_indices: Any,
    expert_start: Any,
    down_bits: Any,
    down_scale: Any,
    *,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(
        contraction_tile=512
    ),
    interpret: bool = False,
) -> Any:
    """Apply exact BF16 SwiGLU and selected raw-FP8 down projections.

    ``down_bits`` uses final ``[expert,intermediate,hidden]`` MXU order.
    Outputs retain top-k route order and non-owned routes are exact zeros.
    SwiGLU is formed inside the Pallas call, so no activated intermediate is
    written to HBM between activation and the down projection.
    """

    if gate.ndim != 2 or gate.shape[0] <= 0:
        raise ValueError("selected down gate must have shape [routes,intermediate]")
    if up.shape != gate.shape:
        raise ValueError("selected down gate/up shapes must agree")
    if gate.dtype != jnp.bfloat16 or up.dtype != jnp.bfloat16:
        raise ValueError("selected down gate/up inputs must be BF16")
    if route_indices.shape != (gate.shape[0],) or route_indices.dtype != jnp.int32:
        raise ValueError("selected down routes must be one int32 id per row")
    if expert_start.shape != () or expert_start.dtype != jnp.int32:
        raise ValueError("selected down expert start must be an int32 scalar")
    if down_bits.ndim != 3 or down_bits.dtype != jnp.uint8:
        raise ValueError("selected down weights must be rank-three uint8 bits")
    local_experts, contraction, output = down_bits.shape
    if local_experts <= 0 or contraction <= 0 or output <= 0:
        raise ValueError("selected down weight dimensions must be positive")
    if contraction != gate.shape[1]:
        raise ValueError("selected down activation/weight contraction disagrees")
    expected_scale = (
        local_experts,
        _ceil_div(output, config.block_shape[0]),
        _ceil_div(contraction, config.block_shape[1]),
    )
    if down_scale.shape != expected_scale or down_scale.dtype != jnp.float32:
        raise ValueError(
            f"selected down FP32 scale shape must be {expected_scale}, "
            f"got {down_scale.shape}/{down_scale.dtype}"
        )

    route_count = gate.shape[0]
    padded_contraction = (
        _ceil_div(contraction, config.contraction_tile)
        * config.contraction_tile
    )
    padded_output = _ceil_div(output, config.output_tile) * config.output_tile
    if padded_contraction != contraction:
        gate = jnp.pad(gate, ((0, 0), (0, padded_contraction - contraction)))
        up = jnp.pad(up, ((0, 0), (0, padded_contraction - contraction)))
    if padded_contraction != contraction or padded_output != output:
        down_bits = jnp.pad(
            down_bits,
            (
                (0, 0),
                (0, padded_contraction - contraction),
                (0, padded_output - output),
            ),
        )

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
    compact_index = jnp.where(
        active,
        jnp.cumsum(active_i32) - jnp.int32(1),
        active_count + jnp.cumsum(inactive_i32) - jnp.int32(1),
    )
    compact_local_ids = jnp.zeros_like(local_ids).at[compact_index].set(
        local_ids
    )
    compact_gate = jnp.zeros(
        (route_count, config.row_tile, padded_contraction), dtype=jnp.bfloat16
    ).at[compact_index, 0, :].set(gate)
    compact_up = jnp.zeros_like(compact_gate).at[compact_index, 0, :].set(up)
    down_scale_table = _selected_scale_table(
        down_scale,
        compact_local_ids,
        padded_output=padded_output,
        padded_contraction=padded_contraction,
        block_shape=config.block_shape,
    )

    def kernel(
        local_ids_value: Any,
        active_count_value: Any,
        gate_hbm_ref: Any,
        up_hbm_ref: Any,
        down_hbm_ref: Any,
        down_scale_hbm_ref: Any,
        output_hbm_ref: Any,
        accumulator_ref: Any,
    ) -> None:
        def inner_kernel(
            gate_ref: Any,
            up_ref: Any,
            down_ref: Any,
            scale_ref: Any,
            output_ref: Any,
            accumulator: Any,
        ) -> None:
            output_index = pl.program_id(0)
            route_index = pl.program_id(1)
            contraction_index = pl.program_id(2)

            @pl.when(contraction_index == 0)
            def initialize_accumulator() -> None:
                accumulator[...] = jnp.zeros_like(accumulator)

            activated = (
                gate_ref[...]
                * jax.nn.sigmoid(gate_ref[...])
                * up_ref[...]
            ).astype(jnp.bfloat16)
            decoded_down = (
                lax.bitcast_convert_type(
                    down_ref[...], jnp.float8_e4m3fn
                )
                .astype(config.accumulator_dtype)
                .reshape(
                    contraction_blocks_per_tile,
                    config.block_shape[1],
                    config.output_tile,
                )
                * scale_ref[...].astype(config.accumulator_dtype)
            ).reshape(config.contraction_tile, config.output_tile).astype(
                jnp.bfloat16
            )
            accumulator[...] += lax.dot_general(
                activated,
                decoded_down,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=config.accumulator_dtype,
            )

            @pl.when(contraction_index == contraction_tiles - 1)
            def store_output() -> None:
                output_ref[...] = accumulator[...].astype(config.output_dtype)

        def activation_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, int, Any]:
            del output_index
            return route_index, 0, contraction_index

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
            return route_index, contraction_index, 0, output_index

        def output_index(
            output_index_value: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, int, Any]:
            del contraction_index
            return route_index, 0, output_index_value

        activation_spec = pl.BlockSpec(
            (None, config.row_tile, config.contraction_tile), activation_index
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
            grid=(output_tiles, active_count_value[...], contraction_tiles),
            in_specs=(activation_spec, activation_spec, weight_spec, scale_spec),
            out_specs=output_spec,
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            no_pipelining=interpret,
        )
        pipeline(
            gate_hbm_ref,
            up_hbm_ref,
            down_hbm_ref,
            down_scale_hbm_ref,
            output_hbm_ref,
            scratches=(accumulator_ref,),
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
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=interpret,
        name=(
            "greenfield_fp8_selected_swiglu_down_"
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
                2 * route_count * config.row_tile * padded_contraction * 2
                + route_count * padded_output * padded_contraction
                + route_count * output_tiles * contraction_tiles * 4
                + route_count * config.row_tile * padded_output * 2
            ),
            transcendentals=route_count * config.row_tile * padded_contraction,
        ),
    )
    output_value = call(
        compact_local_ids,
        active_count,
        compact_gate,
        compact_up,
        down_bits,
        down_scale_table,
    )
    compact_slot_active = (
        jnp.arange(route_count, dtype=jnp.int32) < active_count
    )[:, None, None]
    output_value = jnp.where(
        compact_slot_active, output_value, jnp.zeros_like(output_value)
    )
    output_value = output_value[compact_index]
    return output_value[:, 0, :output]


def fp8_fused_selected_moe(
    hidden: Any,
    route_indices: Any,
    expert_start: Any,
    gate_bits: Any,
    gate_scale: Any,
    up_bits: Any,
    up_scale: Any,
    down_bits: Any,
    down_scale: Any,
    *,
    route_weights: Any | None = None,
    down_result_dtype: Any = jnp.bfloat16,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(
        contraction_tile=512
    ),
    interpret: bool = False,
) -> Any:
    """Execute selected gate/up, exact BF16 SwiGLU, and down in one call.

    The gate and up rows live only in VMEM scratch between two nested Mosaic
    pipelines. Complete routed weights remain raw U8 HBM operands and route
    work is compacted once. The default rounds each down result to BF16. The
    diagnostic FP32 result retains the final down accumulator so a caller can
    reconstruct a feature-sharded complete expert before that rounding point.
    When ``route_weights`` is supplied, the exact BF16 weighting and route-axis
    sum also stay inside the call so only one local hidden-width partial leaves
    it; this fused form is intentionally incompatible with FP32 down results.
    """

    route_count, local_experts, hidden_size, intermediate = (
        _validate_selected_inputs(
            hidden,
            route_indices,
            expert_start,
            gate_bits,
            gate_scale,
            config,
        )
    )
    up_geometry = _validate_selected_inputs(
        hidden,
        route_indices,
        expert_start,
        up_bits,
        up_scale,
        config,
    )
    if up_geometry != (
        route_count,
        local_experts,
        hidden_size,
        intermediate,
    ):
        raise ValueError("fused selected gate/up geometries disagree")
    if down_bits.ndim != 3 or down_bits.dtype != jnp.uint8:
        raise ValueError("fused selected down weights must be rank-three U8")
    if down_bits.shape != (local_experts, intermediate, hidden_size):
        raise ValueError(
            "fused selected down weights must use [G,intermediate,hidden]"
        )
    expected_down_scale = (
        local_experts,
        _ceil_div(hidden_size, config.block_shape[0]),
        _ceil_div(intermediate, config.block_shape[1]),
    )
    if down_scale.shape != expected_down_scale or (
        down_scale.dtype != jnp.float32
    ):
        raise ValueError(
            "fused selected down FP32 scale shape must be "
            f"{expected_down_scale}, got {down_scale.shape}/{down_scale.dtype}"
        )
    fuse_route_weighting = route_weights is not None
    if fuse_route_weighting and (
        route_weights.shape != (route_count,)
        or route_weights.dtype != jnp.float32
    ):
        raise ValueError(
            "fused selected route weights must be one FP32 top-k row"
        )
    down_result_dtype = jnp.dtype(down_result_dtype)
    if down_result_dtype not in (
        jnp.dtype(jnp.bfloat16),
        jnp.dtype(jnp.float32),
    ):
        raise ValueError("fused selected down result must be BF16 or FP32")
    if fuse_route_weighting and down_result_dtype != jnp.dtype(jnp.bfloat16):
        raise ValueError(
            "fused route weighting requires BF16 down results"
        )

    padded_hidden = (
        _ceil_div(hidden_size, config.contraction_tile)
        * config.contraction_tile
    )
    padded_intermediate = (
        _ceil_div(intermediate, config.contraction_tile)
        * config.contraction_tile
    )
    hidden = jnp.pad(
        hidden,
        (
            (0, config.row_tile - 1),
            (0, padded_hidden - hidden_size),
        ),
    )

    def pad_projection(value: Any) -> Any:
        if (
            padded_hidden == hidden_size
            and padded_intermediate == intermediate
        ):
            return value
        return jnp.pad(
            value,
            (
                (0, 0),
                (0, padded_hidden - hidden_size),
                (0, padded_intermediate - intermediate),
            ),
        )

    def pad_down(value: Any) -> Any:
        if (
            padded_intermediate == intermediate
            and padded_hidden == hidden_size
        ):
            return value
        return jnp.pad(
            value,
            (
                (0, 0),
                (0, padded_intermediate - intermediate),
                (0, padded_hidden - hidden_size),
            ),
        )

    gate_raw = pad_projection(gate_bits)
    up_raw = pad_projection(up_bits)
    down_raw = pad_down(down_bits)
    blocks_per_tile = config.contraction_tile // config.block_shape[1]
    up_output_tiles = padded_intermediate // config.output_tile
    up_contraction_tiles = padded_hidden // config.contraction_tile
    down_output_tiles = padded_hidden // config.output_tile
    down_contraction_tiles = (
        padded_intermediate // config.contraction_tile
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
    compact_index = jnp.where(
        active,
        jnp.cumsum(active_i32) - jnp.int32(1),
        active_count + jnp.cumsum(inactive_i32) - jnp.int32(1),
    )
    compact_local_ids = jnp.zeros_like(local_ids).at[compact_index].set(
        local_ids
    )
    compact_route_weights = None
    if fuse_route_weighting:
        active_weights = jnp.where(
            active,
            route_weights,
            jnp.zeros_like(route_weights),
        ).astype(jnp.bfloat16)
        compact_route_weights = jnp.zeros_like(active_weights).at[
            compact_index
        ].set(active_weights)
    gate_scale_table = _selected_scale_table(
        gate_scale,
        compact_local_ids,
        padded_output=padded_intermediate,
        padded_contraction=padded_hidden,
        block_shape=config.block_shape,
    )
    up_scale_table = _selected_scale_table(
        up_scale,
        compact_local_ids,
        padded_output=padded_intermediate,
        padded_contraction=padded_hidden,
        block_shape=config.block_shape,
    )
    down_scale_table = _selected_scale_table(
        down_scale,
        compact_local_ids,
        padded_output=padded_hidden,
        padded_contraction=padded_intermediate,
        block_shape=config.block_shape,
    )

    def kernel_body(
        local_ids_value: Any,
        active_count_value: Any,
        route_weights_value: Any,
        hidden_hbm_ref: Any,
        gate_hbm_ref: Any,
        gate_scale_hbm_ref: Any,
        up_hbm_ref: Any,
        up_scale_hbm_ref: Any,
        down_hbm_ref: Any,
        down_scale_hbm_ref: Any,
        output_hbm_ref: Any,
        gate_vmem_ref: Any,
        up_vmem_ref: Any,
        gate_accumulator_ref: Any,
        up_accumulator_ref: Any,
        down_accumulator_ref: Any,
        route_output_vmem_ref: Any,
    ) -> None:
        def up_gate_kernel(
            hidden_ref: Any,
            gate_ref: Any,
            gate_scale_ref: Any,
            up_ref: Any,
            up_scale_ref: Any,
            gate_output_ref: Any,
            up_output_ref: Any,
            gate_accumulator: Any,
            up_accumulator: Any,
        ) -> None:
            output_index = pl.program_id(0)
            route_index = pl.program_id(1)
            contraction_index = pl.program_id(2)

            @pl.when(contraction_index == 0)
            def initialize_accumulators() -> None:
                gate_accumulator[...] = jnp.zeros_like(gate_accumulator)
                up_accumulator[...] = jnp.zeros_like(up_accumulator)

            decoded_gate = (
                lax.bitcast_convert_type(gate_ref[...], jnp.float8_e4m3fn)
                .astype(config.accumulator_dtype)
                .reshape(
                    blocks_per_tile,
                    config.block_shape[1],
                    config.output_tile,
                )
                * gate_scale_ref[...].astype(config.accumulator_dtype)
            ).reshape(config.contraction_tile, config.output_tile).astype(
                jnp.bfloat16
            )
            decoded_up = (
                lax.bitcast_convert_type(up_ref[...], jnp.float8_e4m3fn)
                .astype(config.accumulator_dtype)
                .reshape(
                    blocks_per_tile,
                    config.block_shape[1],
                    config.output_tile,
                )
                * up_scale_ref[...].astype(config.accumulator_dtype)
            ).reshape(config.contraction_tile, config.output_tile).astype(
                jnp.bfloat16
            )
            dimensions = (((1,), (0,)), ((), ()))
            gate_accumulator[...] += lax.dot_general(
                hidden_ref[...],
                decoded_gate,
                dimension_numbers=dimensions,
                preferred_element_type=config.accumulator_dtype,
            )
            up_accumulator[...] += lax.dot_general(
                hidden_ref[...],
                decoded_up,
                dimension_numbers=dimensions,
                preferred_element_type=config.accumulator_dtype,
            )

            @pl.when(contraction_index == up_contraction_tiles - 1)
            def store_outputs() -> None:
                gate_output_ref[...] = gate_accumulator[...].astype(
                    config.output_dtype
                )
                up_output_ref[...] = up_accumulator[...].astype(
                    config.output_dtype
                )

        def hidden_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[int, Any]:
            del output_index, route_index
            return 0, contraction_index

        def up_weight_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, Any, Any]:
            return (
                local_ids_value[route_index],
                contraction_index,
                output_index,
            )

        def up_scale_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, Any, int, Any]:
            return route_index, contraction_index, 0, output_index

        def up_output_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, int, Any]:
            del contraction_index
            return route_index, 0, output_index

        hidden_spec = pl.BlockSpec(
            (config.row_tile, config.contraction_tile), hidden_index
        )
        up_weight_spec = pl.BlockSpec(
            (None, config.contraction_tile, config.output_tile),
            up_weight_index,
            pipeline_mode=pl.Buffered(buffer_count=3),
        )
        up_scale_spec = pl.BlockSpec(
            (None, blocks_per_tile, 1, config.output_tile),
            up_scale_index,
        )
        up_output_spec = pl.BlockSpec(
            (None, config.row_tile, config.output_tile), up_output_index
        )
        up_gate_pipeline = pltpu.emit_pipeline(
            up_gate_kernel,
            grid=(
                up_output_tiles,
                active_count_value[...],
                up_contraction_tiles,
            ),
            in_specs=(
                hidden_spec,
                up_weight_spec,
                up_scale_spec,
                up_weight_spec,
                up_scale_spec,
            ),
            out_specs=(up_output_spec, up_output_spec),
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            no_pipelining=interpret,
        )
        up_gate_pipeline(
            hidden_hbm_ref,
            gate_hbm_ref,
            gate_scale_hbm_ref,
            up_hbm_ref,
            up_scale_hbm_ref,
            gate_vmem_ref,
            up_vmem_ref,
            scratches=(gate_accumulator_ref, up_accumulator_ref),
        )

        def down_kernel(
            gate_ref: Any,
            up_ref: Any,
            down_ref: Any,
            scale_ref: Any,
            output_ref: Any,
            accumulator: Any,
        ) -> None:
            output_index = pl.program_id(0)
            route_index = pl.program_id(1)
            contraction_index = pl.program_id(2)

            @pl.when(contraction_index == 0)
            def initialize_accumulator() -> None:
                accumulator[...] = jnp.zeros_like(accumulator)

            activated = (
                gate_ref[...]
                * jax.nn.sigmoid(gate_ref[...])
                * up_ref[...]
            ).astype(jnp.bfloat16)
            decoded_down = (
                lax.bitcast_convert_type(down_ref[...], jnp.float8_e4m3fn)
                .astype(config.accumulator_dtype)
                .reshape(
                    blocks_per_tile,
                    config.block_shape[1],
                    config.output_tile,
                )
                * scale_ref[...].astype(config.accumulator_dtype)
            ).reshape(config.contraction_tile, config.output_tile).astype(
                jnp.bfloat16
            )
            accumulator[...] += lax.dot_general(
                activated,
                decoded_down,
                dimension_numbers=(((1,), (0,)), ((), ())),
                preferred_element_type=config.accumulator_dtype,
            )

            @pl.when(contraction_index == down_contraction_tiles - 1)
            def store_output() -> None:
                output_ref[...] = accumulator[...].astype(
                    down_result_dtype
                )

        def activation_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, int, Any]:
            del output_index
            return route_index, 0, contraction_index

        def down_weight_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, Any, Any]:
            return (
                local_ids_value[route_index],
                contraction_index,
                output_index,
            )

        def down_scale_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, Any, int, Any]:
            return route_index, contraction_index, 0, output_index

        def down_output_index(
            output_index: Any,
            route_index: Any,
            contraction_index: Any,
        ) -> tuple[Any, int, Any]:
            del contraction_index
            return route_index, 0, output_index

        activation_spec = pl.BlockSpec(
            (None, config.row_tile, config.contraction_tile),
            activation_index,
        )
        down_weight_spec = pl.BlockSpec(
            (None, config.contraction_tile, config.output_tile),
            down_weight_index,
            pipeline_mode=pl.Buffered(buffer_count=3),
        )
        down_scale_spec = pl.BlockSpec(
            (None, blocks_per_tile, 1, config.output_tile),
            down_scale_index,
        )
        down_output_spec = pl.BlockSpec(
            (None, config.row_tile, config.output_tile), down_output_index
        )
        down_pipeline = pltpu.emit_pipeline(
            down_kernel,
            grid=(
                down_output_tiles,
                active_count_value[...],
                down_contraction_tiles,
            ),
            in_specs=(
                activation_spec,
                activation_spec,
                down_weight_spec,
                down_scale_spec,
            ),
            out_specs=down_output_spec,
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            no_pipelining=interpret,
        )
        down_output_ref = (
            route_output_vmem_ref
            if fuse_route_weighting
            else output_hbm_ref
        )
        down_pipeline(
            gate_vmem_ref,
            up_vmem_ref,
            down_hbm_ref,
            down_scale_hbm_ref,
            down_output_ref,
            scratches=(down_accumulator_ref,),
        )

        if fuse_route_weighting:
            def combine_kernel(
                route_output_ref: Any,
                combined_output_ref: Any,
            ) -> None:
                combined = jnp.zeros(
                    (config.row_tile, config.output_tile),
                    dtype=config.output_dtype,
                )
                for route_slot in range(route_count):
                    active_slot = (
                        jnp.asarray(route_slot, dtype=jnp.int32)
                        < active_count_value[...]
                    )
                    route_output = jnp.where(
                        active_slot,
                        route_output_ref[route_slot, ...],
                        jnp.zeros_like(
                            route_output_ref[route_slot, ...]
                        ),
                    )
                    weight = route_weights_value[route_slot].astype(
                        config.output_dtype
                    )
                    weighted = (route_output * weight).astype(
                        config.output_dtype
                    )
                    combined = (combined + weighted).astype(
                        config.output_dtype
                    )
                combined_output_ref[...] = combined

            def combine_input_index(
                output_index: Any,
            ) -> tuple[int, int, Any]:
                return 0, 0, output_index

            def combine_output_index(
                output_index: Any,
            ) -> tuple[int, Any]:
                return 0, output_index

            combine_pipeline = pltpu.emit_pipeline(
                combine_kernel,
                grid=(down_output_tiles,),
                in_specs=pl.BlockSpec(
                    (route_count, config.row_tile, config.output_tile),
                    combine_input_index,
                ),
                out_specs=pl.BlockSpec(
                    (config.row_tile, config.output_tile),
                    combine_output_index,
                ),
                dimension_semantics=("parallel",),
                no_pipelining=interpret,
            )
            combine_pipeline(
                route_output_vmem_ref,
                output_hbm_ref,
            )

    if fuse_route_weighting:
        output_shape = jax.ShapeDtypeStruct(
            (config.row_tile, padded_hidden), config.output_dtype
        )
    else:
        output_shape = jax.ShapeDtypeStruct(
            (route_count, config.row_tile, padded_hidden),
            down_result_dtype,
        )
    kernel_name = (
        "greenfield_fp8_fused_selected_moe_"
        f"r{route_count}_g{local_experts}_h{padded_hidden}_i{padded_intermediate}"
    )
    if config.output_tile != config.block_shape[0]:
        kernel_name += f"_ot{config.output_tile}"
    if down_result_dtype == jnp.dtype(jnp.float32):
        kernel_name += "_downf32"
    if fuse_route_weighting:
        kernel_name += "_wsum"
    common_scratch_shapes = (
        pltpu.VMEM(
            (route_count, config.row_tile, padded_intermediate),
            config.output_dtype,
        ),
        pltpu.VMEM(
            (route_count, config.row_tile, padded_intermediate),
            config.output_dtype,
        ),
        pltpu.VMEM(
            (config.row_tile, config.output_tile),
            config.accumulator_dtype,
        ),
        pltpu.VMEM(
            (config.row_tile, config.output_tile),
            config.accumulator_dtype,
        ),
        pltpu.VMEM(
            (config.row_tile, config.output_tile),
            config.accumulator_dtype,
        ),
    )
    if fuse_route_weighting:
        def kernel(
            local_ids_value: Any,
            active_count_value: Any,
            route_weights_value: Any,
            hidden_hbm_ref: Any,
            gate_hbm_ref: Any,
            gate_scale_hbm_ref: Any,
            up_hbm_ref: Any,
            up_scale_hbm_ref: Any,
            down_hbm_ref: Any,
            down_scale_hbm_ref: Any,
            output_hbm_ref: Any,
            gate_vmem_ref: Any,
            up_vmem_ref: Any,
            gate_accumulator_ref: Any,
            up_accumulator_ref: Any,
            down_accumulator_ref: Any,
            route_output_vmem_ref: Any,
            _body: Any = kernel_body,
        ) -> None:
            _body(
                local_ids_value,
                active_count_value,
                route_weights_value,
                hidden_hbm_ref,
                gate_hbm_ref,
                gate_scale_hbm_ref,
                up_hbm_ref,
                up_scale_hbm_ref,
                down_hbm_ref,
                down_scale_hbm_ref,
                output_hbm_ref,
                gate_vmem_ref,
                up_vmem_ref,
                gate_accumulator_ref,
                up_accumulator_ref,
                down_accumulator_ref,
                route_output_vmem_ref,
            )

        scalar_prefetch = 3
        scratch_shapes = common_scratch_shapes + (
            pltpu.VMEM(
                (route_count, config.row_tile, padded_hidden),
                config.output_dtype,
            ),
        )
    else:
        def kernel(
            local_ids_value: Any,
            active_count_value: Any,
            hidden_hbm_ref: Any,
            gate_hbm_ref: Any,
            gate_scale_hbm_ref: Any,
            up_hbm_ref: Any,
            up_scale_hbm_ref: Any,
            down_hbm_ref: Any,
            down_scale_hbm_ref: Any,
            output_hbm_ref: Any,
            gate_vmem_ref: Any,
            up_vmem_ref: Any,
            gate_accumulator_ref: Any,
            up_accumulator_ref: Any,
            down_accumulator_ref: Any,
            _body: Any = kernel_body,
        ) -> None:
            _body(
                local_ids_value,
                active_count_value,
                jnp.zeros((route_count,), dtype=jnp.bfloat16),
                hidden_hbm_ref,
                gate_hbm_ref,
                gate_scale_hbm_ref,
                up_hbm_ref,
                up_scale_hbm_ref,
                down_hbm_ref,
                down_scale_hbm_ref,
                output_hbm_ref,
                gate_vmem_ref,
                up_vmem_ref,
                gate_accumulator_ref,
                up_accumulator_ref,
                down_accumulator_ref,
                None,
            )

        scalar_prefetch = 2
        scratch_shapes = common_scratch_shapes
    call = pl.pallas_call(
        kernel,
        out_shape=output_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=scalar_prefetch,
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(disable_bounds_checks=True),
        interpret=interpret,
        name=kernel_name,
        cost_estimate=pl.CostEstimate(
            flops=(
                (6
                * route_count
                * config.row_tile
                * padded_hidden
                * padded_intermediate)
                + (
                    2 * route_count * config.row_tile * padded_hidden
                    if fuse_route_weighting
                    else 0
                )
            ),
            bytes_accessed=(
                config.row_tile * padded_hidden * 2
                + 3
                * route_count
                * padded_hidden
                * padded_intermediate
                + 3
                * route_count
                * (
                    up_output_tiles * up_contraction_tiles
                    + down_output_tiles * down_contraction_tiles
                )
                * 4
                + (
                    config.row_tile
                    * padded_hidden
                    * jnp.dtype(config.output_dtype).itemsize
                    if fuse_route_weighting
                    else route_count
                    * config.row_tile
                    * padded_hidden
                    * down_result_dtype.itemsize
                )
            ),
            transcendentals=(
                route_count * config.row_tile * padded_intermediate
            ),
        ),
    )
    call_inputs = (
        compact_local_ids,
        active_count,
        *((compact_route_weights,) if fuse_route_weighting else ()),
        hidden,
        gate_raw,
        gate_scale_table,
        up_raw,
        up_scale_table,
        down_raw,
        down_scale_table,
    )
    output_value = call(*call_inputs)
    if fuse_route_weighting:
        return output_value[:1, :hidden_size]
    compact_slot_active = (
        jnp.arange(route_count, dtype=jnp.int32) < active_count
    )[:, None, None]
    output_value = jnp.where(
        compact_slot_active, output_value, jnp.zeros_like(output_value)
    )
    output_value = output_value[compact_index]
    return output_value[:, 0, :hidden_size]
