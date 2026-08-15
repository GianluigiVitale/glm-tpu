"""True-row TPU residual/RMS weighted-output boundaries.

The protected Gate-D oracle showed that XLA's ordinary one-row elementwise
fusion does not preserve the accepted BF16 weighted-output rounding.  This
kernel keeps the public and compiled tensor shape at one row while making the
BF16 round and weight multiply explicit inside one Pallas program.  It is
default-off at every caller and imports no legacy execution code.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def weighted_output_m1_m8_scratch(
    hidden_states: Any,
    residual: Any,
    inverse_rms: Any,
    weight: Any,
    *,
    interpret: bool = False,
) -> Any:
    """Apply the final M1 add/round/weight step through an M8 VMEM tile.

    The accepted M32 oracle processes row zero in a physical ``8 x 128``
    output tile whose seven sibling rows are NaN sentinels.  This bounded
    kernel recreates exactly that *internal* tile for each 128-wide feature
    block while keeping every public input and the result at one logical row.
    The inverse RMS is supplied by the caller so the probe changes only the
    final output boundary, not reduction or reciprocal arithmetic.
    """

    if hidden_states.shape != residual.shape or hidden_states.ndim != 2:
        raise ValueError("Pallas weighted-output inputs must share rank-two shape")
    if hidden_states.shape[0] != 1:
        raise ValueError("Pallas weighted-output requires exactly one live row")
    width = int(hidden_states.shape[1])
    if width <= 0 or width % 128:
        raise ValueError(
            "Pallas weighted-output width must be a positive multiple of 128"
        )
    if inverse_rms.shape != (1,) or inverse_rms.dtype != jnp.float32:
        raise ValueError("Pallas weighted-output inverse must be FP32[1]")
    if weight.shape != (width,):
        raise ValueError("Pallas weighted-output weight must match hidden width")
    if hidden_states.dtype != jnp.bfloat16 or residual.dtype != jnp.bfloat16:
        raise ValueError("Pallas weighted-output activations must be BF16")
    if weight.dtype != jnp.bfloat16:
        raise ValueError("Pallas weighted-output weight must be BF16")
    if not isinstance(interpret, bool):
        raise ValueError("Pallas weighted-output interpret flag must be boolean")

    def kernel(
        hidden_ref: Any,
        residual_ref: Any,
        inverse_ref: Any,
        weight_ref: Any,
        output_ref: Any,
        weighted_scratch_ref: Any,
    ) -> None:
        row_ids = jnp.arange(8, dtype=jnp.int32)[:, None]
        live = row_ids == jnp.int32(0)
        nan_lanes = jnp.full((8, 128), jnp.bfloat16(jnp.nan))
        hidden_lanes = jnp.where(
            live,
            jnp.broadcast_to(hidden_ref[...], (8, 128)),
            nan_lanes,
        )
        residual_lanes = jnp.where(
            live,
            jnp.broadcast_to(residual_ref[...], (8, 128)),
            nan_lanes,
        )
        summed = hidden_lanes.astype(jnp.float32) + residual_lanes.astype(
            jnp.float32
        )
        rounded = (
            summed * inverse_ref[0].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        weighted_scratch_ref[...] = (
            rounded * jnp.broadcast_to(weight_ref[...], (8, 128))
        ).astype(jnp.bfloat16)
        output_ref[...] = weighted_scratch_ref[...][:1, :]

    def row_index(feature_tile: Any) -> tuple[int, Any]:
        return 0, feature_tile

    def scalar_index(feature_tile: Any) -> tuple[int]:
        del feature_tile
        return (0,)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, width), jnp.bfloat16),
        grid=(width // 128,),
        in_specs=(
            pl.BlockSpec((1, 128), row_index),
            pl.BlockSpec((1, 128), row_index),
            pl.BlockSpec((1,), scalar_index),
            pl.BlockSpec((1, 128), row_index),
        ),
        out_specs=pl.BlockSpec((1, 128), row_index),
        scratch_shapes=(pltpu.VMEM((8, 128), jnp.bfloat16),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=f"greenfield_weighted_output_m1_m8_scratch_h{width}",
        cost_estimate=pl.CostEstimate(
            flops=width * 3,
            bytes_accessed=width * 10,
            transcendentals=0,
        ),
    )(hidden_states, residual, inverse_rms, weight[None, :])


def source_fused_output_m1_m8_scratch(
    dense: Any,
    attention: Any,
    embedding: Any,
    validity: Any,
    inverse_rms: Any,
    weight: Any,
    *,
    interpret: bool = False,
) -> Any:
    """Fuse the accepted native output sources inside one true-M1 kernel."""

    rows = (dense, attention, embedding)
    if any(value.ndim != 2 or value.shape != dense.shape for value in rows):
        raise ValueError("source-fused Pallas rows must share rank-two shape")
    if dense.shape[0] != 1:
        raise ValueError("source-fused Pallas output requires one live row")
    width = int(dense.shape[1])
    if width <= 0 or width % 128:
        raise ValueError(
            "source-fused Pallas width must be a positive multiple of 128"
        )
    if any(value.dtype != jnp.bfloat16 for value in rows):
        raise ValueError("source-fused Pallas rows must be BF16")
    if validity.shape != (1,) or validity.dtype != jnp.bool_:
        raise ValueError("source-fused Pallas validity must be BOOL[1]")
    if inverse_rms.shape != (1,) or inverse_rms.dtype != jnp.float32:
        raise ValueError("source-fused Pallas inverse must be FP32[1]")
    if weight.shape != (width,) or weight.dtype != jnp.bfloat16:
        raise ValueError("source-fused Pallas weight must be BF16[width]")
    if not isinstance(interpret, bool):
        raise ValueError("source-fused Pallas interpret flag must be boolean")

    def kernel(
        dense_ref: Any,
        attention_ref: Any,
        embedding_ref: Any,
        validity_ref: Any,
        inverse_ref: Any,
        weight_ref: Any,
        output_ref: Any,
        weighted_scratch_ref: Any,
    ) -> None:
        row_ids = jnp.arange(8, dtype=jnp.int32)[:, None]
        live = row_ids == jnp.int32(0)
        nan_lanes = jnp.full((8, 128), jnp.bfloat16(jnp.nan))
        dense_lanes = jnp.where(
            live, jnp.broadcast_to(dense_ref[...], (8, 128)), nan_lanes
        )
        attention_lanes = jnp.where(
            live,
            jnp.broadcast_to(attention_ref[...], (8, 128)),
            nan_lanes,
        )
        embedding_lanes = jnp.where(
            live,
            jnp.broadcast_to(embedding_ref[...], (8, 128)),
            nan_lanes,
        )
        selected_embedding = jnp.where(
            validity_ref[0], embedding_lanes, nan_lanes
        )
        carried = (
            attention_lanes.astype(jnp.float32)
            + selected_embedding.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        summed = dense_lanes.astype(jnp.float32) + carried.astype(jnp.float32)
        rounded = (
            summed * inverse_ref[0].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        weighted_scratch_ref[...] = (
            rounded * jnp.broadcast_to(weight_ref[...], (8, 128))
        ).astype(jnp.bfloat16)
        output_ref[...] = weighted_scratch_ref[...][:1, :]

    def row_index(feature_tile: Any) -> tuple[int, Any]:
        return 0, feature_tile

    def scalar_index(feature_tile: Any) -> tuple[int]:
        del feature_tile
        return (0,)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, width), jnp.bfloat16),
        grid=(width // 128,),
        in_specs=(
            pl.BlockSpec((1, 128), row_index),
            pl.BlockSpec((1, 128), row_index),
            pl.BlockSpec((1, 128), row_index),
            pl.BlockSpec((1,), scalar_index),
            pl.BlockSpec((1,), scalar_index),
            pl.BlockSpec((1, 128), row_index),
        ),
        out_specs=pl.BlockSpec((1, 128), row_index),
        scratch_shapes=(pltpu.VMEM((8, 128), jnp.bfloat16),),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=f"greenfield_source_fused_output_m1_m8_scratch_h{width}",
        cost_estimate=pl.CostEstimate(
            flops=width * 5,
            bytes_accessed=width * 14,
            transcendentals=0,
        ),
    )(
        dense,
        attention,
        embedding,
        validity,
        inverse_rms,
        weight[None, :],
    )


def source_fused_output_m1_feature_tiled_m8(
    dense: Any,
    attention: Any,
    embedding: Any,
    validity: Any,
    inverse_rms: Any,
    weight: Any,
    *,
    interpret: bool = False,
) -> Any:
    """Use an all-live ``8 x 128`` feature tile and return one semantic row.

    Unlike the rejected scratch discriminator, the eight physical rows here
    are feature blocking, not seven dead batch rows.  Every lane owns one of
    the 6,144 live features, and the final reshape restores exact row-major
    feature order without arithmetic.
    """

    rows = (dense, attention, embedding)
    if any(value.ndim != 2 or value.shape != dense.shape for value in rows):
        raise ValueError("feature-tiled Pallas rows must share rank-two shape")
    if dense.shape[0] != 1:
        raise ValueError("feature-tiled Pallas output requires one semantic row")
    width = int(dense.shape[1])
    if width <= 0 or width % (8 * 128):
        raise ValueError(
            "feature-tiled Pallas width must be a positive multiple of 1024"
        )
    if any(value.dtype != jnp.bfloat16 for value in rows):
        raise ValueError("feature-tiled Pallas rows must be BF16")
    if validity.shape != (1,) or validity.dtype != jnp.bool_:
        raise ValueError("feature-tiled Pallas validity must be BOOL[1]")
    if inverse_rms.shape != (1,) or inverse_rms.dtype != jnp.float32:
        raise ValueError("feature-tiled Pallas inverse must be FP32[1]")
    if weight.shape != (width,) or weight.dtype != jnp.bfloat16:
        raise ValueError("feature-tiled Pallas weight must be BF16[width]")
    if not isinstance(interpret, bool):
        raise ValueError("feature-tiled Pallas interpret flag must be boolean")

    tiled_shape = (8, width // 8)

    def kernel(
        dense_ref: Any,
        attention_ref: Any,
        embedding_ref: Any,
        validity_ref: Any,
        inverse_ref: Any,
        weight_ref: Any,
        output_ref: Any,
    ) -> None:
        selected_embedding = jnp.where(
            validity_ref[0],
            embedding_ref[...],
            jnp.full((8, 128), jnp.bfloat16(jnp.nan)),
        )
        carried = (
            attention_ref[...].astype(jnp.float32)
            + selected_embedding.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        summed = dense_ref[...].astype(jnp.float32) + carried.astype(
            jnp.float32
        )
        rounded = (
            summed * inverse_ref[0].astype(jnp.float32)
        ).astype(jnp.bfloat16)
        output_ref[...] = (rounded * weight_ref[...]).astype(jnp.bfloat16)

    def tile_index(feature_tile: Any) -> tuple[int, Any]:
        return 0, feature_tile

    def scalar_index(feature_tile: Any) -> tuple[int]:
        del feature_tile
        return (0,)

    tiled_output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(tiled_shape, jnp.bfloat16),
        grid=(width // (8 * 128),),
        in_specs=(
            pl.BlockSpec((8, 128), tile_index),
            pl.BlockSpec((8, 128), tile_index),
            pl.BlockSpec((8, 128), tile_index),
            pl.BlockSpec((1,), scalar_index),
            pl.BlockSpec((1,), scalar_index),
            pl.BlockSpec((8, 128), tile_index),
        ),
        out_specs=pl.BlockSpec((8, 128), tile_index),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=f"greenfield_source_fused_output_m1_feature_tiled_m8_h{width}",
        cost_estimate=pl.CostEstimate(
            flops=width * 5,
            bytes_accessed=width * 14,
            transcendentals=0,
        ),
    )(
        dense.reshape(tiled_shape),
        attention.reshape(tiled_shape),
        embedding.reshape(tiled_shape),
        validity,
        inverse_rms,
        weight.reshape(tiled_shape),
    )
    return tiled_output.reshape((1, width))


def fused_add_rms_norm_m1(
    hidden_states: Any,
    residual: Any,
    weight: Any,
    *,
    epsilon: float,
    interpret: bool = False,
) -> tuple[Any, Any]:
    """Return exact one-row normalized output and rounded carried residual.

    The arithmetic contract matches the reference boundary: add and RMS
    reduction in FP32, round the normalized value to BF16, multiply by the
    BF16 checkpoint weight, and return BF16.  The separate residual is the
    unnormalized FP32 sum rounded once to BF16.
    """

    if hidden_states.shape != residual.shape or hidden_states.ndim != 2:
        raise ValueError("Pallas fused RMS inputs must share rank-two shape")
    if hidden_states.shape[0] != 1:
        raise ValueError("Pallas fused RMS requires exactly one live row")
    width = int(hidden_states.shape[1])
    if width <= 0 or width % 128:
        raise ValueError("Pallas fused RMS width must be a positive multiple of 128")
    if weight.shape != (width,):
        raise ValueError("Pallas fused RMS weight must match hidden width")
    if hidden_states.dtype != jnp.bfloat16 or residual.dtype != jnp.bfloat16:
        raise ValueError("Pallas fused RMS activations must be BF16")
    if weight.dtype != jnp.bfloat16:
        raise ValueError("Pallas fused RMS weight must be BF16")
    if (
        not isinstance(epsilon, (int, float))
        or isinstance(epsilon, bool)
        or epsilon <= 0
    ):
        raise ValueError("Pallas fused RMS epsilon must be positive")
    if not isinstance(interpret, bool):
        raise ValueError("Pallas fused RMS interpret flag must be boolean")

    def kernel(
        hidden_ref: Any,
        residual_ref: Any,
        weight_ref: Any,
        output_ref: Any,
        carried_ref: Any,
    ) -> None:
        summed = hidden_ref[...].astype(jnp.float32) + residual_ref[
            ...
        ].astype(jnp.float32)
        carried_ref[...] = summed.astype(jnp.bfloat16)
        square_sum = jnp.sum(lax.square(summed), axis=1, keepdims=True)
        inverse = lax.rsqrt(
            square_sum * jnp.float32(1.0 / width) + jnp.float32(epsilon)
        )
        rounded = (summed * inverse).astype(jnp.bfloat16)
        output_ref[...] = (
            rounded * weight_ref[...][None, :]
        ).astype(jnp.bfloat16)

    def row_index(_: Any) -> tuple[int, int]:
        return 0, 0

    def weight_index(_: Any) -> tuple[int]:
        return (0,)

    call = pl.pallas_call(
        kernel,
        out_shape=(
            jax.ShapeDtypeStruct((1, width), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, width), jnp.bfloat16),
        ),
        grid=(1,),
        in_specs=(
            pl.BlockSpec((1, width), row_index),
            pl.BlockSpec((1, width), row_index),
            pl.BlockSpec((width,), weight_index),
        ),
        out_specs=(
            pl.BlockSpec((1, width), row_index),
            pl.BlockSpec((1, width), row_index),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=f"greenfield_fused_add_rms_norm_m1_h{width}",
        cost_estimate=pl.CostEstimate(
            flops=width * 7 + 2,
            bytes_accessed=width * 10,
            transcendentals=1,
        ),
    )
    output, carried = call(hidden_states, residual, weight)
    return output, carried
