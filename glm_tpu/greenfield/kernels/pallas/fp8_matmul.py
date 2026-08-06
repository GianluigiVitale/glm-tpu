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
        if self.block_shape != (self.output_tile, self.contraction_tile):
            raise ValueError(
                "the first TPU-v4 kernel requires one scale per Pallas weight tile"
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
    if output_tiles > 128:
        raise ValueError("FP8 matmul supports at most 128 output-scale blocks")
    aligned_contraction_tiles = _ceil_div(contraction_tiles, 8) * 8
    # Mosaic cannot issue a dynamically indexed scalar VMEM load from the
    # compact [N-block,K-block] scale table: its two tiled dimensions require
    # 8x128-aligned vector access.  Transpose and pad the few-KiB table so each
    # program loads one aligned 8x128 scale tile, then selects its scalar in
    # registers.  This is bounded metadata staging, not a decoded-weight
    # overlay (GLM up: 24 KiB versus 12 MiB of FP8 weight bytes).
    scale_table = jnp.pad(
        jnp.transpose(scale),
        (
            (0, aligned_contraction_tiles - contraction_tiles),
            (0, 128 - output_tiles),
        ),
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
        scale_tile = scale_ref[...]
        scale_value = scale_tile[
            contraction_index % jnp.int32(8), output_index
        ]
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
