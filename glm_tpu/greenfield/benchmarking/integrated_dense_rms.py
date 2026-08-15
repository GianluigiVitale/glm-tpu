"""One-rank-per-chip layer-0 dense-to-StrategyND-to-RMS discriminator.

This default-off diagnostic keeps both fused residual/RMS boundaries, the
real final-layout dense contractions, and the physical 32-chip reduction in
one compiled program.  It exists to decide whether externalizing the already
proven BF16 dense partials is the final source of Gate-D numerical drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import ml_dtypes
import numpy as np

from ..errors import BenchmarkValidationError
from ..kernels.reference.rmsnorm import fused_add_rms_norm
from ..kernels.pallas import (
    Fp8BlockMatmulConfig,
    fp8_structured_kv_b_value,
    fused_add_rms_norm_m1,
    source_fused_output_m1_feature_tiled_m8,
    source_fused_output_m1_m8_scratch,
)
from ..kernels.stage_local import (
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
    _decode_dense_fp8_in_out,
    _dense_bf16_convolution,
    _virtual_dense_final_layout_convolution_down_partials,
)
from .association_fingerprint import array_sha256
from .dense_rms_replay import (
    DENSE_RMS_ARRAY_RECORDS,
    DENSE_RMS_SOURCE_KEYS,
    DENSE_RMS_SOURCE_NPZ_SHA256,
)


POST_ATTENTION_NORM_RAW_SHA256 = (
    "b09022104ef7d5cfa9fba198ec605105faec10ad1352d0a21be49455421bd260"
)
CHECKPOINT_SUCCESS_SHA256 = (
    "368ef308c7937258c181cdc42fecddf8a7f7ca7bab04470d39cb622aaa3c5b24"
)
ACCEPTED_SOURCE_VALIDITY = np.asarray(
    (True,) + (False,) * 31,
    dtype=np.bool_,
)
NATIVE_SOURCE_TOKEN_IDS = np.asarray(
    (220,) + (154880,) * 31,
    dtype=np.int32,
)
NATIVE_SOURCE_ATTENDED_LATENT_SHA256 = (
    "923e9bfeb65864868cef359cf98ebad67f2756794ad142ac87e66b71877a2d2a"
)
NATIVE_SOURCE_ATTENDED_LATENT_KEY = (
    "attended_latent_bfloat16_bits__pregathered_h16_b512__greenfield_cache"
)
NATIVE_SOURCE_TAG = (
    "greenfield_layer0_attention_arithmetic_20260812T114701365714147Z"
)
NATIVE_SOURCE_NPZ_SHA256 = (
    "7d5ebe15dd006a70d77f17d41eb47f25f58c6d6784ee332fbc638f2916581f61"
)
NATIVE_SOURCE_RUNNER_SHA256 = (
    "7961622c297567a021edf2ab335f0d046db6665e8130e210bbd78919a2d7a4ec"
)
NATIVE_SOURCE_SUMMARY_SHA256 = (
    "9793f89aa0f0bbe2532106707e0a38600058d9ad5f30f36813f36c5fd7298540"
)
NATIVE_SOURCE_SUCCESS_SHA256 = (
    "ecc2b873c29ffd9bc6551136e90352b33523f50873827b00ae91279a4db3153c"
)
NATIVE_SOURCE_REMOTE_OBJECTS_SHA256 = (
    "c3e3f5b9d5452ce6c2a7a699b1f781efc4b3d741d072022f2a98991faf9b621c"
)


@dataclass(frozen=True, slots=True)
class IntegratedDenseRmsInputs:
    """Sealed activation inputs and exact accepted layer-1 target."""

    attention_update_bits: np.ndarray
    combined_residual_bits: np.ndarray
    post_attention_norm_bits: np.ndarray
    layer1_norm_bits: np.ndarray
    accepted_layer1_bits: np.ndarray


@dataclass(frozen=True, slots=True)
class IntegratedDenseWeights:
    """Final-layout layer-0 dense weights in physical-device order."""

    merged_bits: np.ndarray
    merged_scale: np.ndarray
    down_bits: np.ndarray
    down_scale: np.ndarray


@dataclass(frozen=True, slots=True)
class NativeSourceWeights:
    """Physical final-layout shards for the bounded native source graph."""

    embedding: np.ndarray
    kv_b_bits: np.ndarray
    kv_b_scale: np.ndarray
    o_bits_in_out: np.ndarray
    o_scale_in_out: np.ndarray
    merged_bits: np.ndarray
    merged_scale: np.ndarray
    down_bits: np.ndarray
    down_scale: np.ndarray


@dataclass(frozen=True, slots=True)
class NativeSourceInputs:
    """Coherent sealed activations for native embedding through layer 1."""

    attended_latent_bits: np.ndarray
    token_ids: np.ndarray
    post_attention_norm_bits: np.ndarray
    layer1_norm_bits: np.ndarray
    accepted_layer1_bits: np.ndarray


@dataclass(frozen=True, slots=True)
class CompiledIntegratedDenseRms:
    compiled: Any
    member_sharding: Any
    replicated_sharding: Any
    member_device_ids: tuple[int, ...]
    native_source_context: bool
    native_m32_output: bool
    native_m1_pallas_output: bool
    native_m1_pallas_sources_output: bool
    native_m1_pallas_feature_tiled_output: bool
    native_m1_xla_feature_tiled_output: bool
    stablehlo: str
    optimized_hlo: str
    stablehlo_contract: Mapping[str, Any]
    optimized_hlo_contract: Mapping[str, Any]
    accepted_source_context: bool
    preceding_attention_collective: bool
    split_predense_rms: bool
    split_layer1_rms: bool


def _validate_inputs(inputs: IntegratedDenseRmsInputs) -> None:
    records = {
        "attention_update_bits": (inputs.attention_update_bits, (1, 6144)),
        "combined_residual_bits": (inputs.combined_residual_bits, (1, 6144)),
        "post_attention_norm_bits": (inputs.post_attention_norm_bits, (6144,)),
        "layer1_norm_bits": (inputs.layer1_norm_bits, (6144,)),
        "accepted_layer1_bits": (inputs.accepted_layer1_bits, (6144,)),
    }
    for name, (value, shape) in records.items():
        if value.shape != shape or value.dtype != np.uint16:
            raise BenchmarkValidationError(
                f"integrated dense RMS {name} must be uint16{shape}"
            )


def native_source_inputs(
    attended_latent_bits: np.ndarray,
    base: IntegratedDenseRmsInputs,
) -> NativeSourceInputs:
    """Bind the exact DB537 latent to the coherent DB548 norm/target bundle."""

    latent = np.ascontiguousarray(attended_latent_bits)
    if (
        latent.shape != (64, 512)
        or latent.dtype != np.uint16
        or _raw_sha256(latent) != NATIVE_SOURCE_ATTENDED_LATENT_SHA256
    ):
        raise BenchmarkValidationError(
            "native source attended latent drifted"
        )
    _validate_inputs(base)
    result = NativeSourceInputs(
        attended_latent_bits=latent,
        token_ids=NATIVE_SOURCE_TOKEN_IDS.copy(),
        post_attention_norm_bits=base.post_attention_norm_bits,
        layer1_norm_bits=base.layer1_norm_bits,
        accepted_layer1_bits=base.accepted_layer1_bits,
    )
    _validate_native_source_inputs(result)
    return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_integrated_checkpoint_success(checkpoint_root: Path) -> str:
    """Rehash the exact terminal marker that authorizes the packed checkpoint."""

    marker = checkpoint_root / "SUCCESS"
    if (
        not marker.is_file()
        or _file_sha256(marker) != CHECKPOINT_SUCCESS_SHA256
    ):
        raise BenchmarkValidationError(
            "integrated dense checkpoint SUCCESS drifted"
        )
    return CHECKPOINT_SUCCESS_SHA256


def _raw_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_integrated_dense_rms_inputs(
    source_path: Path,
    post_attention_norm: np.ndarray,
) -> IntegratedDenseRmsInputs:
    """Load every activation from one SHA-pinned DB548 structural bundle."""

    if (
        not source_path.is_file()
        or _file_sha256(source_path) != DENSE_RMS_SOURCE_NPZ_SHA256
    ):
        raise BenchmarkValidationError(
            "integrated dense RMS source NPZ SHA-256 drifted"
        )
    arrays: dict[str, np.ndarray] = {}
    with np.load(source_path, allow_pickle=False) as payload:
        if tuple(payload.files) != DENSE_RMS_SOURCE_KEYS:
            raise BenchmarkValidationError(
                "integrated dense RMS source key order/set drifted"
            )
        for name in payload.files:
            value = np.ascontiguousarray(payload[name])
            shape, dtype, digest = DENSE_RMS_ARRAY_RECORDS[name]
            if (
                value.shape != shape
                or value.dtype != dtype
                or _raw_sha256(value) != digest
            ):
                raise BenchmarkValidationError(
                    f"integrated dense RMS source tensor drifted: {name}"
                )
            arrays[name] = value
    post_norm = np.ascontiguousarray(post_attention_norm).view(np.uint16)
    if (
        post_norm.shape != (6144,)
        or _raw_sha256(post_norm) != POST_ATTENTION_NORM_RAW_SHA256
    ):
        raise BenchmarkValidationError(
            "integrated dense RMS post-attention norm drifted"
        )
    result = IntegratedDenseRmsInputs(
        attention_update_bits=arrays["attention_update_bfloat16_bits"],
        combined_residual_bits=arrays["combined_residual_bfloat16_bits"],
        post_attention_norm_bits=post_norm,
        layer1_norm_bits=arrays["layer1_input_norm_bfloat16_bits"],
        accepted_layer1_bits=arrays[
            "accepted_layer1_normalized_bfloat16_bits"
        ],
    )
    _validate_inputs(result)
    return result


def _validate_weights(weights: IntegratedDenseWeights) -> None:
    records = {
        "merged_bits": (weights.merged_bits, (32, 1, 6144, 768)),
        "merged_scale": (weights.merged_scale, (32, 1, 48, 768)),
        "down_bits": (weights.down_bits, (32, 1, 384, 6144)),
        "down_scale": (weights.down_scale, (32, 1, 3, 6144)),
    }
    for name, (value, shape) in records.items():
        if value.shape != shape:
            raise BenchmarkValidationError(
                f"integrated dense RMS {name} shape drifted: {value.shape}"
            )
    if weights.merged_bits.dtype not in (
        np.uint8,
        np.dtype(ml_dtypes.float8_e4m3fn),
    ) or weights.down_bits.dtype not in (
        np.uint8,
        np.dtype(ml_dtypes.float8_e4m3fn),
    ):
        raise BenchmarkValidationError("integrated dense FP8 storage drifted")
    if weights.merged_scale.dtype != np.float32 or weights.down_scale.dtype != np.float32:
        raise BenchmarkValidationError("integrated dense scale dtype drifted")


def _validate_native_source_inputs(inputs: NativeSourceInputs) -> None:
    records = {
        "attended_latent_bits": (
            inputs.attended_latent_bits,
            (64, 512),
            np.dtype(np.uint16),
        ),
        "token_ids": (inputs.token_ids, (32,), np.dtype(np.int32)),
        "post_attention_norm_bits": (
            inputs.post_attention_norm_bits,
            (6144,),
            np.dtype(np.uint16),
        ),
        "layer1_norm_bits": (
            inputs.layer1_norm_bits,
            (6144,),
            np.dtype(np.uint16),
        ),
        "accepted_layer1_bits": (
            inputs.accepted_layer1_bits,
            (6144,),
            np.dtype(np.uint16),
        ),
    }
    for name, (value, shape, dtype) in records.items():
        if value.shape != shape or value.dtype != dtype:
            raise BenchmarkValidationError(
                f"native source {name} shape/dtype drifted"
            )
    if not np.array_equal(inputs.token_ids, NATIVE_SOURCE_TOKEN_IDS):
        raise BenchmarkValidationError("native source token ids drifted")
    if _raw_sha256(inputs.attended_latent_bits) != (
        NATIVE_SOURCE_ATTENDED_LATENT_SHA256
    ):
        raise BenchmarkValidationError("native source latent SHA drifted")


def _validate_native_source_weights(weights: NativeSourceWeights) -> None:
    records = {
        "embedding": (weights.embedding, (32, 4840, 6144)),
        "kv_b_bits": (weights.kv_b_bits, (32, 7168, 512)),
        "kv_b_scale": (weights.kv_b_scale, (32, 56, 4)),
        "o_bits_in_out": (weights.o_bits_in_out, (32, 512, 6144)),
        "o_scale_in_out": (weights.o_scale_in_out, (32, 4, 48)),
        "merged_bits": (weights.merged_bits, (32, 1, 6144, 768)),
        "merged_scale": (weights.merged_scale, (32, 1, 48, 768)),
        "down_bits": (weights.down_bits, (32, 1, 384, 6144)),
        "down_scale": (weights.down_scale, (32, 1, 3, 6144)),
    }
    for name, (value, shape) in records.items():
        if value.shape != shape:
            raise BenchmarkValidationError(
                f"native source {name} shape drifted: {value.shape}"
            )
    if weights.embedding.dtype != np.dtype(ml_dtypes.bfloat16):
        raise BenchmarkValidationError("native embedding dtype drifted")
    for name in ("kv_b_bits", "o_bits_in_out"):
        if getattr(weights, name).dtype != np.uint8:
            raise BenchmarkValidationError(
                f"native source {name} dtype drifted"
            )
    for name in ("kv_b_scale", "o_scale_in_out"):
        if getattr(weights, name).dtype != np.float32:
            raise BenchmarkValidationError(
                f"native source {name} dtype drifted"
            )
    _validate_weights(
        IntegratedDenseWeights(
            merged_bits=weights.merged_bits,
            merged_scale=weights.merged_scale,
            down_bits=weights.down_bits,
            down_scale=weights.down_scale,
        )
    )


def model_axis_weights_to_physical(
    value: np.ndarray,
    model_axis_device_ids: Sequence[int],
) -> np.ndarray:
    """Move a leading 32-way model axis into sorted physical-device order."""

    source = np.ascontiguousarray(value)
    mapping = tuple(int(device_id) for device_id in model_axis_device_ids)
    if source.shape[0] != 32 or len(mapping) != 32 or sorted(mapping) != list(range(32)):
        raise BenchmarkValidationError(
            "integrated dense model-axis mapping must bijectively cover 0..31"
        )
    physical = np.empty_like(source)
    physical[np.asarray(mapping, dtype=np.int32)] = source
    return np.ascontiguousarray(physical)


def assemble_native_source_weights(
    checkpoint_weights: Mapping[str, np.ndarray],
    dense_weights: IntegratedDenseWeights,
    model_axis_device_ids: Sequence[int],
) -> NativeSourceWeights:
    """Pack native layer-0 source weights into exact physical model order."""

    required = {
        "global.embedding",
        "attention.slot_00.kv_b.weight_bits",
        "attention.slot_00.kv_b.scale_inv",
        "attention.slot_00.o.weight_bits",
        "attention.slot_00.o.scale_inv",
    }
    if not required.issubset(checkpoint_weights):
        raise BenchmarkValidationError(
            "native source checkpoint weights are incomplete"
        )
    _validate_weights(dense_weights)
    model_embedding = np.ascontiguousarray(
        checkpoint_weights["global.embedding"].reshape(
            4, 8, 4840, 6144
        ).reshape(32, 4840, 6144)
    )
    model_kv_b_bits = np.ascontiguousarray(
        np.repeat(
            checkpoint_weights[
                "attention.slot_00.kv_b.weight_bits"
            ][:, None],
            8,
            axis=1,
        ).reshape(32, 7168, 512)
    )
    model_kv_b_scale = np.ascontiguousarray(
        np.repeat(
            checkpoint_weights[
                "attention.slot_00.kv_b.scale_inv"
            ][:, None],
            8,
            axis=1,
        ).reshape(32, 56, 4)
    )
    model_o_bits = np.ascontiguousarray(
        checkpoint_weights["attention.slot_00.o.weight_bits"]
        .reshape(4, 6144, 8, 512)
        .transpose(0, 2, 3, 1)
        .reshape(32, 512, 6144)
    )
    model_o_scale = np.ascontiguousarray(
        checkpoint_weights["attention.slot_00.o.scale_inv"]
        .reshape(4, 48, 8, 4)
        .transpose(0, 2, 3, 1)
        .reshape(32, 4, 48)
    )
    result = NativeSourceWeights(
        embedding=model_axis_weights_to_physical(
            model_embedding, model_axis_device_ids
        ),
        kv_b_bits=model_axis_weights_to_physical(
            model_kv_b_bits, model_axis_device_ids
        ),
        kv_b_scale=model_axis_weights_to_physical(
            model_kv_b_scale, model_axis_device_ids
        ),
        o_bits_in_out=model_axis_weights_to_physical(
            model_o_bits, model_axis_device_ids
        ),
        o_scale_in_out=model_axis_weights_to_physical(
            model_o_scale, model_axis_device_ids
        ),
        merged_bits=dense_weights.merged_bits,
        merged_scale=dense_weights.merged_scale,
        down_bits=dense_weights.down_bits,
        down_scale=dense_weights.down_scale,
    )
    _validate_native_source_weights(result)
    return result


def _integrated_function(
    *,
    split_layer1_rms: bool = False,
    preceding_attention_collective: bool = False,
    split_predense_rms: bool = False,
) -> Any:
    import jax
    from jax import lax
    import jax.numpy as jnp

    def integrated(
        attention_update: Any,
        combined_residual: Any,
        post_attention_norm: Any,
        merged_bits: Any,
        merged_scale: Any,
        down_bits: Any,
        down_scale: Any,
        layer1_norm: Any,
    ) -> Any:
        with jax.named_scope("integrated_dense_rms_predense_sources"):
            attention_m32 = jnp.pad(
                attention_update,
                ((0, 31), (0, 0)),
                constant_values=jnp.bfloat16(0),
            )
            residual_m32 = jnp.pad(
                combined_residual,
                ((0, 31), (0, 0)),
                constant_values=jnp.bfloat16(0),
            )
        if preceding_attention_collective:
            # The accepted full decoder executes its layer-0 attention psum
            # immediately before the dense psum.  Recreate only that physical
            # collective context from the already-sealed final BF16 attention
            # row: rank zero contributes it and every other rank contributes
            # exact zeros, so the value is unchanged under any add tree.
            with jax.named_scope(
                "integrated_dense_rms_preceding_attention_collective"
            ):
                is_owner = lax.axis_index("member") == jnp.int32(0)
                local_attention = jnp.where(
                    is_owner,
                    attention_m32,
                    jnp.zeros_like(attention_m32),
                )
                attention_m32 = lax.psum(local_attention, "member")
        with jax.named_scope("integrated_dense_rms_predense_norm"):
            if split_predense_rms:
                # The accepted layer-0 schedule keeps only the scalar RMS
                # reduction live here.  It recomputes the rounded residual
                # and normalized BF16 value in the gate contraction fusion.
                # This is the pre-dense analogue of the already protected
                # layer-1 split schedule below.
                with jax.named_scope("accepted_predense_split_reduction"):
                    reduction_attention = lax.optimization_barrier(
                        attention_m32
                    )
                    reduction_residual = lax.optimization_barrier(residual_m32)
                    reduction_sum = (
                        reduction_attention.astype(jnp.float32)
                        + reduction_residual.astype(jnp.float32)
                    )
                    inverse = lax.rsqrt(
                        jnp.mean(
                            lax.square(reduction_sum), axis=-1, keepdims=True
                        )
                        + jnp.float32(1e-5)
                    )
                with jax.named_scope("accepted_predense_split_recompute"):
                    output_sum = (
                        attention_m32.astype(jnp.float32)
                        + residual_m32.astype(jnp.float32)
                    )
                    carried = output_sum.astype(attention_m32.dtype)
                    normalized = (
                        (output_sum * inverse).astype(post_attention_norm.dtype)
                        * post_attention_norm
                    ).astype(attention_m32.dtype)
            else:
                normalized, carried = fused_add_rms_norm(
                    attention_m32,
                    residual_m32,
                    post_attention_norm,
                    epsilon=1e-5,
                )
        with jax.named_scope("integrated_dense_rms_contraction"):
            local_partials = _virtual_dense_final_layout_convolution_down_partials(
                normalized,
                merged_bits[0],
                merged_scale[0],
                down_bits[0],
                down_scale[0],
                block_shape=(128, 128),
                compile_rows=32,
                virtual_shards=1,
                accepted_gate_singleton=True,
                accepted_gate_dequant_fusion=True,
            )
            local_partial = local_partials[0]
        with jax.named_scope("integrated_dense_rms_strategy_nd_collective"):
            dense_m32 = lax.psum(local_partial, "member")
        with jax.named_scope("integrated_dense_rms_layer1_norm"):
            if split_layer1_rms:
                # The accepted full-model M32 HLO schedules only the scalar
                # reduction here, then recomputes the residual sum in the
                # live weighted-output fusion.  Keep that exact boundary in
                # the same graph as the real contraction and physical psum.
                with jax.named_scope("accepted_split_reduction"):
                    reduction_dense = lax.optimization_barrier(dense_m32)
                    reduction_residual = lax.optimization_barrier(carried)
                    reduction_sum = (
                        reduction_dense.astype(jnp.float32)
                        + reduction_residual.astype(jnp.float32)
                    )
                    inverse = lax.rsqrt(
                        jnp.mean(
                            lax.square(reduction_sum), axis=-1, keepdims=True
                        )
                        + jnp.float32(1e-5)
                    )
                with jax.named_scope("accepted_split_recompute"):
                    output_sum = (
                        dense_m32.astype(jnp.float32)
                        + carried.astype(jnp.float32)
                    )
                    layer1 = (
                        (output_sum * inverse).astype(layer1_norm.dtype)
                        * layer1_norm
                    ).astype(dense_m32.dtype)
            else:
                layer1, _ = fused_add_rms_norm(
                    dense_m32,
                    carried,
                    layer1_norm,
                    epsilon=1e-5,
                )
        with jax.named_scope("integrated_dense_rms_live_row"):
            return lax.bitcast_convert_type(layer1[:1, :], jnp.uint16)

    return integrated


def _accepted_source_context_function() -> Any:
    """Recreate the accepted embedding/attention/dense source context once."""

    import jax
    from jax import lax
    import jax.numpy as jnp

    def integrated(
        attention_update: Any,
        embedding_row: Any,
        valid_rows: Any,
        post_attention_norm: Any,
        merged_bits: Any,
        merged_scale: Any,
        down_bits: Any,
        down_scale: Any,
        layer1_norm: Any,
    ) -> Any:
        with jax.named_scope("accepted_source_context_embedding_input"):
            embedding_input_m32 = jnp.pad(
                embedding_row,
                ((0, 31), (0, 0)),
                constant_values=jnp.bfloat16(0),
            )
            is_embedding_owner = lax.axis_index("member") == jnp.int32(0)
            local_embedding = jnp.where(
                is_embedding_owner,
                embedding_input_m32,
                jnp.zeros_like(embedding_input_m32),
            )
        with jax.named_scope("accepted_source_context_embedding_collective"):
            embedding_m32 = lax.psum(local_embedding, "member")

        with jax.named_scope("accepted_source_context_attention_input"):
            attention_input_m32 = jnp.pad(
                attention_update,
                ((0, 31), (0, 0)),
                constant_values=jnp.bfloat16(0),
            )
            is_attention_owner = lax.axis_index("member") == jnp.int32(0)
            local_attention = jnp.where(
                is_attention_owner,
                attention_input_m32,
                jnp.zeros_like(attention_input_m32),
            )
            # The accepted attention path is downstream of the embedding
            # lookup.  The bounded graph starts from its sealed final BF16
            # attention row, so retain that ordering through a non-arithmetic
            # finite-source guard rather than letting XLA coalesce the two
            # independent psums into one tuple collective.
            embedding_is_finite = jnp.isfinite(embedding_m32[0, 0])
            local_attention = lax.select(
                embedding_is_finite,
                local_attention,
                jnp.zeros_like(local_attention),
            )
        with jax.named_scope("accepted_source_context_attention_collective"):
            attention_m32 = lax.psum(local_attention, "member")

        def source_sum(scope: str, barrier_order: int) -> Any:
            with jax.named_scope(scope):
                if barrier_order == 0:
                    attention_source, embedding_source, validity_source = (
                        lax.optimization_barrier(
                            (attention_m32, embedding_m32, valid_rows)
                        )
                    )
                elif barrier_order == 1:
                    embedding_source, validity_source, attention_source = (
                        lax.optimization_barrier(
                            (embedding_m32, valid_rows, attention_m32)
                        )
                    )
                elif barrier_order == 2:
                    validity_source, attention_source, embedding_source = (
                        lax.optimization_barrier(
                            (valid_rows, attention_m32, embedding_m32)
                        )
                    )
                else:
                    validity_source, embedding_source, attention_source = (
                        lax.optimization_barrier(
                            (valid_rows, embedding_m32, attention_m32)
                        )
                    )
                selected_embedding = jnp.where(
                    validity_source[:, None],
                    embedding_source,
                    jnp.full_like(
                        embedding_source, jnp.bfloat16(jnp.nan)
                    ),
                )
                return (
                    attention_source.astype(jnp.float32)
                    + selected_embedding.astype(jnp.float32)
                )

        with jax.named_scope("accepted_source_context_predense_norm"):
            with jax.named_scope("accepted_source_context_predense_reduction"):
                reduction_sum = source_sum(
                    "accepted_source_context_predense_reduction_sources",
                    0,
                )
                predense_inverse = lax.rsqrt(
                    jnp.mean(
                        lax.square(reduction_sum), axis=-1, keepdims=True
                    )
                    + jnp.float32(1e-5)
                )
            with jax.named_scope("accepted_source_context_predense_recompute"):
                predense_sum = source_sum(
                    "accepted_source_context_predense_gate_sources",
                    1,
                )
                normalized = (
                    (predense_sum * predense_inverse).astype(
                        post_attention_norm.dtype
                    )
                    * post_attention_norm
                ).astype(attention_m32.dtype)

        with jax.named_scope("integrated_dense_rms_contraction"):
            local_partials = _virtual_dense_final_layout_convolution_down_partials(
                normalized,
                merged_bits[0],
                merged_scale[0],
                down_bits[0],
                down_scale[0],
                block_shape=(128, 128),
                compile_rows=32,
                virtual_shards=1,
                accepted_gate_singleton=True,
                accepted_gate_dequant_fusion=True,
            )
            local_partial = local_partials[0]
        with jax.named_scope("integrated_dense_rms_strategy_nd_collective"):
            dense_m32 = lax.psum(local_partial, "member")

        with jax.named_scope("accepted_source_context_layer1_norm"):
            with jax.named_scope("accepted_source_context_layer1_reduction"):
                reduction_dense = lax.optimization_barrier(dense_m32)
                reduction_carried = source_sum(
                    "accepted_source_context_layer1_reduction_sources",
                    2,
                ).astype(jnp.bfloat16)
                reduction_sum = (
                    reduction_dense.astype(jnp.float32)
                    + reduction_carried.astype(jnp.float32)
                )
                layer1_inverse = lax.rsqrt(
                    jnp.mean(
                        lax.square(reduction_sum), axis=-1, keepdims=True
                    )
                    + jnp.float32(1e-5)
                )
            with jax.named_scope("accepted_source_context_layer1_recompute"):
                output_carried = source_sum(
                    "accepted_source_context_layer1_output_sources",
                    3,
                ).astype(jnp.bfloat16)
                output_sum = (
                    dense_m32.astype(jnp.float32)
                    + output_carried.astype(jnp.float32)
                )
                layer1 = (
                    (output_sum * layer1_inverse).astype(layer1_norm.dtype)
                    * layer1_norm
                ).astype(dense_m32.dtype)
        with jax.named_scope("integrated_dense_rms_live_row"):
            return lax.bitcast_convert_type(layer1[:1, :], jnp.uint16)

    return integrated


def _native_source_context_function(
    *,
    full_m32_output: bool = False,
    pallas_m1_output: bool = False,
    pallas_m1_sources_output: bool = False,
    pallas_m1_feature_tiled_output: bool = False,
    xla_m1_feature_tiled_output: bool = False,
) -> Any:
    """Run the exact native embedding and attention producers in one graph."""

    if not isinstance(full_m32_output, bool):
        raise BenchmarkValidationError(
            "native source M32-output flag must be boolean"
        )
    if not isinstance(pallas_m1_output, bool):
        raise BenchmarkValidationError(
            "native source Pallas-M1-output flag must be boolean"
        )
    if not isinstance(pallas_m1_sources_output, bool):
        raise BenchmarkValidationError(
            "native source-fused Pallas-M1 flag must be boolean"
        )
    if not isinstance(pallas_m1_feature_tiled_output, bool):
        raise BenchmarkValidationError(
            "native all-live feature-tiled Pallas-M1 flag must be boolean"
        )
    if not isinstance(xla_m1_feature_tiled_output, bool):
        raise BenchmarkValidationError(
            "native all-live feature-tiled XLA-M1 flag must be boolean"
        )
    if sum(
        (
            full_m32_output,
            pallas_m1_output,
            pallas_m1_sources_output,
            pallas_m1_feature_tiled_output,
            xla_m1_feature_tiled_output,
        )
    ) > 1:
        raise BenchmarkValidationError(
            "native M32 and Pallas-M1 output modes are disjoint"
        )

    import jax
    from jax import lax
    import jax.numpy as jnp

    model_position_by_physical = jnp.asarray(
        STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
        dtype=jnp.int32,
    )
    fp8_config = Fp8BlockMatmulConfig(
        block_shape=(128, 128),
        output_tile=128,
        contraction_tile=128,
    )

    def integrated(
        attended_latent: Any,
        token_ids: Any,
        post_attention_norm: Any,
        embedding_shard: Any,
        kv_b_bits: Any,
        kv_b_scale: Any,
        o_bits_in_out: Any,
        o_scale_in_out: Any,
        merged_bits: Any,
        merged_scale: Any,
        down_bits: Any,
        down_scale: Any,
        layer1_norm: Any,
    ) -> Any:
        physical_index = lax.axis_index("member")
        model_position = model_position_by_physical[physical_index]
        owner = model_position // jnp.int32(8)
        dcp_rank = model_position % jnp.int32(8)

        with jax.named_scope("native_source_context_embedding_lookup"):
            adjusted_ids = jnp.where(
                token_ids < jnp.int32(0),
                token_ids + jnp.int32(154880),
                token_ids,
            )
            valid_rows = (adjusted_ids >= jnp.int32(0)) & (
                adjusted_ids <= jnp.int32(154879)
            )
            local_start = model_position * jnp.int32(4840)
            local_end = local_start + jnp.int32(4839)
            outside_owner = (adjusted_ids < local_start) | (
                adjusted_ids > local_end
            )
            local_ids = jnp.clip(
                adjusted_ids - local_start,
                jnp.int32(0),
                jnp.int32(4839),
            )
            gathered = jnp.take(
                embedding_shard[0],
                local_ids,
                axis=0,
                mode="clip",
            )
            local_embedding = jnp.where(
                outside_owner[:, None],
                jnp.zeros_like(gathered),
                gathered,
            )
        with jax.named_scope("native_source_context_embedding_collective"):
            embedding_m32 = lax.psum(local_embedding, "member")

        with jax.named_scope("native_source_context_attention_projection"):
            owner_latent = lax.dynamic_slice_in_dim(
                attended_latent,
                owner * jnp.int32(16),
                16,
                axis=0,
            )[None, ...]
            value_states = fp8_structured_kv_b_value(
                owner_latent,
                kv_b_bits[0],
                kv_b_scale[0],
                qk_nope_head_dim=192,
                config=fp8_config,
            )
            owner_value = value_states.reshape(1, 4096)
            local_value = lax.dynamic_slice_in_dim(
                owner_value,
                dcp_rank * jnp.int32(512),
                512,
                axis=1,
            )
            local_value_m32 = jnp.pad(
                local_value,
                ((0, 31), (0, 0)),
                constant_values=jnp.bfloat16(0),
            )
            decoded_o = _decode_dense_fp8_in_out(
                o_bits_in_out[0],
                o_scale_in_out[0],
                block_shape=(128, 128),
            )
            local_attention = _dense_bf16_convolution(
                local_value_m32,
                decoded_o,
            )
            # The accepted attention projection is downstream of the
            # embedding lookup.  Preserve that dependency so XLA cannot
            # coalesce or reorder these two otherwise independent psums.
            embedding_is_finite = jnp.isfinite(embedding_m32[0, 0])
            local_attention = lax.select(
                embedding_is_finite,
                local_attention,
                jnp.zeros_like(local_attention),
            )
        with jax.named_scope("native_source_context_attention_collective"):
            attention_m32 = lax.psum(local_attention, "member")

        def source_sum(scope: str, barrier_order: int) -> Any:
            with jax.named_scope(scope):
                if barrier_order == 0:
                    attention_source, embedding_source, validity_source = (
                        lax.optimization_barrier(
                            (attention_m32, embedding_m32, valid_rows)
                        )
                    )
                elif barrier_order == 1:
                    embedding_source, validity_source, attention_source = (
                        lax.optimization_barrier(
                            (embedding_m32, valid_rows, attention_m32)
                        )
                    )
                elif barrier_order == 2:
                    validity_source, attention_source, embedding_source = (
                        lax.optimization_barrier(
                            (valid_rows, attention_m32, embedding_m32)
                        )
                    )
                else:
                    validity_source, embedding_source, attention_source = (
                        lax.optimization_barrier(
                            (valid_rows, embedding_m32, attention_m32)
                        )
                    )
                selected_embedding = jnp.where(
                    validity_source[:, None],
                    embedding_source,
                    jnp.full_like(
                        embedding_source, jnp.bfloat16(jnp.nan)
                    ),
                )
                return (
                    attention_source.astype(jnp.float32)
                    + selected_embedding.astype(jnp.float32)
                )

        with jax.named_scope("native_source_context_predense_norm"):
            with jax.named_scope(
                "native_source_context_predense_reduction"
            ):
                reduction_sum = source_sum(
                    "native_source_context_predense_reduction_sources",
                    0,
                )
                predense_inverse = lax.rsqrt(
                    jnp.mean(
                        lax.square(reduction_sum), axis=-1, keepdims=True
                    )
                    + jnp.float32(1e-5)
                )
            with jax.named_scope("native_source_context_predense_recompute"):
                predense_sum = source_sum(
                    "native_source_context_predense_gate_sources",
                    1,
                )
                normalized = (
                    (predense_sum * predense_inverse).astype(
                        post_attention_norm.dtype
                    )
                    * post_attention_norm
                ).astype(attention_m32.dtype)

        with jax.named_scope("integrated_dense_rms_contraction"):
            local_partials = (
                _virtual_dense_final_layout_convolution_down_partials(
                    normalized,
                    merged_bits[0],
                    merged_scale[0],
                    down_bits[0],
                    down_scale[0],
                    block_shape=(128, 128),
                    compile_rows=32,
                    virtual_shards=1,
                    accepted_gate_singleton=True,
                    accepted_gate_dequant_fusion=True,
                )
            )
            local_partial = local_partials[0]
        with jax.named_scope("integrated_dense_rms_strategy_nd_collective"):
            dense_m32 = lax.psum(local_partial, "member")

        with jax.named_scope("native_source_context_layer1_norm"):
            if pallas_m1_output:
                with jax.named_scope(
                    "native_source_context_layer1_pallas_m1_sources"
                ):
                    output_carried = source_sum(
                        "native_source_context_layer1_output_sources",
                        3,
                    ).astype(jnp.bfloat16)
                    dense_row = dense_m32[:1, :]
                    carried_row = output_carried[:1, :]
                with jax.named_scope(
                    "native_source_context_layer1_pallas_m1"
                ):
                    layer1, _ = fused_add_rms_norm_m1(
                        dense_row,
                        carried_row,
                        layer1_norm,
                        epsilon=1e-5,
                    )
                with jax.named_scope("integrated_dense_rms_live_row"):
                    return lax.bitcast_convert_type(layer1, jnp.uint16)
            with jax.named_scope("native_source_context_layer1_reduction"):
                reduction_dense = lax.optimization_barrier(dense_m32)
                reduction_carried = source_sum(
                    "native_source_context_layer1_reduction_sources",
                    2,
                ).astype(jnp.bfloat16)
                reduction_sum = (
                    reduction_dense.astype(jnp.float32)
                    + reduction_carried.astype(jnp.float32)
                )
                layer1_inverse = lax.rsqrt(
                    jnp.mean(
                        lax.square(reduction_sum), axis=-1, keepdims=True
                    )
                    + jnp.float32(1e-5)
                )
            with jax.named_scope("native_source_context_layer1_output"):
                if xla_m1_feature_tiled_output:
                    with jax.named_scope(
                        "native_source_context_layer1_xla_feature_tiled"
                    ):
                        (
                            validity_source,
                            embedding_source,
                            attention_source,
                        ) = lax.optimization_barrier(
                            (valid_rows, embedding_m32, attention_m32)
                        )
                        tiled_shape = (8, 768)
                        dense_tiled = dense_m32[:1, :].reshape(tiled_shape)
                        attention_tiled = attention_source[:1, :].reshape(
                            tiled_shape
                        )
                        embedding_tiled = embedding_source[:1, :].reshape(
                            tiled_shape
                        )
                        selected_embedding = jnp.where(
                            validity_source[:1, None],
                            embedding_tiled,
                            jnp.full_like(
                                embedding_tiled, jnp.bfloat16(jnp.nan)
                            ),
                        )
                        carried_tiled = (
                            attention_tiled.astype(jnp.float32)
                            + selected_embedding.astype(jnp.float32)
                        ).astype(jnp.bfloat16)
                        summed_tiled = dense_tiled.astype(jnp.float32) + (
                            carried_tiled.astype(jnp.float32)
                        )
                        rounded_tiled = (
                            summed_tiled * layer1_inverse[:1, :]
                        ).astype(jnp.bfloat16)
                        layer1 = (
                            rounded_tiled * layer1_norm.reshape(tiled_shape)
                        ).astype(jnp.bfloat16).reshape((1, 6144))
                elif (
                    pallas_m1_sources_output
                    or pallas_m1_feature_tiled_output
                ):
                    with jax.named_scope(
                        "native_source_context_layer1_pallas_feature_tiled"
                        if pallas_m1_feature_tiled_output
                        else "native_source_context_layer1_pallas_sources"
                    ):
                        (
                            validity_source,
                            embedding_source,
                            attention_source,
                        ) = lax.optimization_barrier(
                            (valid_rows, embedding_m32, attention_m32)
                        )
                        if pallas_m1_feature_tiled_output:
                            layer1 = (
                                source_fused_output_m1_feature_tiled_m8(
                                    dense_m32[:1, :],
                                    attention_source[:1, :],
                                    embedding_source[:1, :],
                                    validity_source[:1],
                                    layer1_inverse[:1, 0],
                                    layer1_norm,
                                )
                            )
                        else:
                            layer1 = source_fused_output_m1_m8_scratch(
                                dense_m32[:1, :],
                                attention_source[:1, :],
                                embedding_source[:1, :],
                                validity_source[:1],
                                layer1_inverse[:1, 0],
                                layer1_norm,
                            )
                else:
                    output_carried = source_sum(
                        "native_source_context_layer1_output_sources",
                        3,
                    ).astype(jnp.bfloat16)
                    output_sum = dense_m32.astype(
                        jnp.float32
                    ) + output_carried.astype(jnp.float32)
                    layer1 = (
                        (output_sum * layer1_inverse).astype(layer1_norm.dtype)
                        * layer1_norm
                    ).astype(dense_m32.dtype)
        with jax.named_scope("integrated_dense_rms_live_row"):
            captured = layer1 if full_m32_output else layer1[:1, :]
            return lax.bitcast_convert_type(captured, jnp.uint16)

    return integrated


def build_integrated_dense_rms(
    member_device_ids: Sequence[int],
    *,
    devices: Sequence[Any] | None = None,
    validate_hlo: bool = True,
    split_layer1_rms: bool = False,
    preceding_attention_collective: bool = False,
    split_predense_rms: bool = False,
    accepted_source_context: bool = False,
    native_source_context: bool = False,
    native_m32_output: bool = False,
    native_m1_pallas_output: bool = False,
    native_m1_pallas_sources_output: bool = False,
    native_m1_pallas_feature_tiled_output: bool = False,
    native_m1_xla_feature_tiled_output: bool = False,
) -> CompiledIntegratedDenseRms:
    """Compile the exact one-rank-per-chip diagnostic on 32 devices."""

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    runtime_devices = tuple(jax.devices() if devices is None else devices)
    members = tuple(int(value) for value in member_device_ids)
    by_id = {int(device.id): device for device in runtime_devices}
    if members != tuple(range(32)) or set(by_id) != set(members):
        raise BenchmarkValidationError(
            "integrated dense RMS requires global physical ids 0..31"
        )
    mesh = Mesh(
        np.asarray([by_id[device_id] for device_id in members], dtype=object),
        ("member",),
    )
    member = NamedSharding(mesh, P("member"))
    replicated = NamedSharding(mesh, P())
    if not isinstance(split_layer1_rms, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS split flag must be boolean"
        )
    if not isinstance(preceding_attention_collective, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS preceding-attention flag must be boolean"
        )
    if not isinstance(split_predense_rms, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS pre-dense split flag must be boolean"
        )
    if not isinstance(accepted_source_context, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS accepted-source flag must be boolean"
        )
    if not isinstance(native_source_context, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS native-source flag must be boolean"
        )
    if not isinstance(native_m32_output, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS native M32-output flag must be boolean"
        )
    if not isinstance(native_m1_pallas_output, bool):
        raise BenchmarkValidationError(
            "native Pallas-M1-output flag must be boolean"
        )
    if not isinstance(native_m1_pallas_sources_output, bool):
        raise BenchmarkValidationError(
            "native source-fused Pallas-M1-output flag must be boolean"
        )
    if not isinstance(native_m1_pallas_feature_tiled_output, bool):
        raise BenchmarkValidationError(
            "native all-live feature-tiled Pallas-M1-output flag must be boolean"
        )
    if not isinstance(native_m1_xla_feature_tiled_output, bool):
        raise BenchmarkValidationError(
            "native all-live feature-tiled XLA-M1-output flag must be boolean"
        )
    if native_m32_output and not native_source_context:
        raise BenchmarkValidationError(
            "native M32 output requires native source context"
        )
    if native_m1_pallas_output and not native_source_context:
        raise BenchmarkValidationError(
            "native Pallas-M1 output requires native source context"
        )
    if native_m1_pallas_sources_output and not native_source_context:
        raise BenchmarkValidationError(
            "native source-fused Pallas-M1 output requires native source context"
        )
    if native_m1_pallas_feature_tiled_output and not native_source_context:
        raise BenchmarkValidationError(
            "native all-live feature-tiled Pallas-M1 output requires native source context"
        )
    if native_m1_xla_feature_tiled_output and not native_source_context:
        raise BenchmarkValidationError(
            "native all-live feature-tiled XLA-M1 output requires native source context"
        )
    if sum(
        (
            native_m32_output,
            native_m1_pallas_output,
            native_m1_pallas_sources_output,
            native_m1_pallas_feature_tiled_output,
            native_m1_xla_feature_tiled_output,
        )
    ) > 1:
        raise BenchmarkValidationError(
            "native M32 and Pallas-M1 output modes are disjoint"
        )
    if accepted_source_context and native_source_context:
        raise BenchmarkValidationError(
            "accepted and native source contexts are disjoint"
        )
    if accepted_source_context and any(
        (split_layer1_rms, preceding_attention_collective, split_predense_rms)
    ):
        raise BenchmarkValidationError(
            "accepted source context is a distinct integrated discriminator"
        )
    if native_source_context and any(
        (split_layer1_rms, preceding_attention_collective, split_predense_rms)
    ):
        raise BenchmarkValidationError(
            "native source context is a distinct integrated discriminator"
        )
    if preceding_attention_collective and not split_layer1_rms:
        raise BenchmarkValidationError(
            "preceding attention collective requires the frozen split RMS arm"
        )
    if split_predense_rms and not split_layer1_rms:
        raise BenchmarkValidationError(
            "pre-dense split RMS requires the frozen layer-1 split RMS arm"
        )
    if split_predense_rms and preceding_attention_collective:
        raise BenchmarkValidationError(
            "pre-dense split RMS and rejected ordinal arms are disjoint"
        )
    mapped_function = (
        _native_source_context_function(
            full_m32_output=native_m32_output,
            pallas_m1_output=native_m1_pallas_output,
            pallas_m1_sources_output=native_m1_pallas_sources_output,
            pallas_m1_feature_tiled_output=(
                native_m1_pallas_feature_tiled_output
            ),
            xla_m1_feature_tiled_output=(
                native_m1_xla_feature_tiled_output
            ),
        )
        if native_source_context
        else _accepted_source_context_function()
        if accepted_source_context
        else _integrated_function(
            split_layer1_rms=split_layer1_rms,
            preceding_attention_collective=preceding_attention_collective,
            split_predense_rms=split_predense_rms,
        )
    )
    in_specs = (
        (
            P(),
            P(),
            P(),
            P("member", None, None),
            P("member", None, None),
            P("member", None, None),
            P("member", None, None),
            P("member", None, None),
            P("member", None, None, None),
            P("member", None, None, None),
            P("member", None, None, None),
            P("member", None, None, None),
            P(),
        )
        if native_source_context
        else
        (
            P(),
            P(),
            P(),
            P(),
            P("member", None, None, None),
            P("member", None, None, None),
            P("member", None, None, None),
            P("member", None, None, None),
            P(),
        )
        if accepted_source_context
        else (
            P(),
            P(),
            P(),
            P("member", None, None, None),
            P("member", None, None, None),
            P("member", None, None, None),
            P("member", None, None, None),
            P(),
        )
    )
    mapped = jax.shard_map(
        mapped_function,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=P(),
        check_vma=False,
    )
    common_examples = (
        jax.ShapeDtypeStruct((1, 6144), jnp.bfloat16, sharding=replicated),
        jax.ShapeDtypeStruct((1, 6144), jnp.bfloat16, sharding=replicated),
    )
    tail_examples = (
        jax.ShapeDtypeStruct((6144,), jnp.bfloat16, sharding=replicated),
        jax.ShapeDtypeStruct((32, 1, 6144, 768), jnp.float8_e4m3fn, sharding=member),
        jax.ShapeDtypeStruct((32, 1, 48, 768), jnp.float32, sharding=member),
        jax.ShapeDtypeStruct((32, 1, 384, 6144), jnp.float8_e4m3fn, sharding=member),
        jax.ShapeDtypeStruct((32, 1, 3, 6144), jnp.float32, sharding=member),
        jax.ShapeDtypeStruct((6144,), jnp.bfloat16, sharding=replicated),
    )
    examples = (
        (
            jax.ShapeDtypeStruct(
                (64, 512), jnp.bfloat16, sharding=replicated
            ),
            jax.ShapeDtypeStruct((32,), jnp.int32, sharding=replicated),
            jax.ShapeDtypeStruct(
                (6144,), jnp.bfloat16, sharding=replicated
            ),
            jax.ShapeDtypeStruct(
                (32, 4840, 6144), jnp.bfloat16, sharding=member
            ),
            jax.ShapeDtypeStruct(
                (32, 7168, 512), jnp.uint8, sharding=member
            ),
            jax.ShapeDtypeStruct(
                (32, 56, 4), jnp.float32, sharding=member
            ),
            jax.ShapeDtypeStruct(
                (32, 512, 6144), jnp.uint8, sharding=member
            ),
            jax.ShapeDtypeStruct(
                (32, 4, 48), jnp.float32, sharding=member
            ),
        )
        + tail_examples[1:]
        if native_source_context
        else
        common_examples
        + (
            jax.ShapeDtypeStruct((32,), jnp.bool_, sharding=replicated),
        )
        + tail_examples
        if accepted_source_context
        else common_examples + tail_examples
    )
    lowered = jax.jit(mapped).lower(*examples)
    stablehlo = lowered.as_text()
    compiled = lowered.compile()
    optimized_hlo = compiled.as_text()
    if validate_hlo:
        from .integrated_dense_rms_hlo import (
            validate_integrated_dense_rms_hlo,
            validate_integrated_dense_rms_stablehlo,
        )

        stablehlo_contract = validate_integrated_dense_rms_stablehlo(
            stablehlo,
            split_layer1_rms=split_layer1_rms,
            preceding_attention_collective=preceding_attention_collective,
            split_predense_rms=split_predense_rms,
            accepted_source_context=accepted_source_context,
            native_source_context=native_source_context,
            native_m32_output=native_m32_output,
            native_m1_pallas_output=native_m1_pallas_output,
            native_m1_pallas_sources_output=(
                native_m1_pallas_sources_output
            ),
            native_m1_pallas_feature_tiled_output=(
                native_m1_pallas_feature_tiled_output
            ),
            native_m1_xla_feature_tiled_output=(
                native_m1_xla_feature_tiled_output
            ),
        )
        optimized_hlo_contract = validate_integrated_dense_rms_hlo(
            optimized_hlo,
            members,
            split_layer1_rms=split_layer1_rms,
            preceding_attention_collective=preceding_attention_collective,
            split_predense_rms=split_predense_rms,
            accepted_source_context=accepted_source_context,
            native_source_context=native_source_context,
            native_m32_output=native_m32_output,
            native_m1_pallas_output=native_m1_pallas_output,
            native_m1_pallas_sources_output=(
                native_m1_pallas_sources_output
            ),
            native_m1_pallas_feature_tiled_output=(
                native_m1_pallas_feature_tiled_output
            ),
            native_m1_xla_feature_tiled_output=(
                native_m1_xla_feature_tiled_output
            ),
        )
    else:
        stablehlo_contract = {
            "passed": False,
            "violations": ["HLO validation explicitly disabled"],
        }
        optimized_hlo_contract = {
            "passed": False,
            "violations": ["HLO validation explicitly disabled"],
        }
    return CompiledIntegratedDenseRms(
        compiled=compiled,
        member_sharding=member,
        replicated_sharding=replicated,
        member_device_ids=members,
        native_source_context=native_source_context,
        native_m32_output=native_m32_output,
        native_m1_pallas_output=native_m1_pallas_output,
        native_m1_pallas_sources_output=native_m1_pallas_sources_output,
        native_m1_pallas_feature_tiled_output=(
            native_m1_pallas_feature_tiled_output
        ),
        native_m1_xla_feature_tiled_output=(
            native_m1_xla_feature_tiled_output
        ),
        stablehlo=stablehlo,
        optimized_hlo=optimized_hlo,
        stablehlo_contract=stablehlo_contract,
        optimized_hlo_contract=optimized_hlo_contract,
        accepted_source_context=accepted_source_context,
        preceding_attention_collective=preceding_attention_collective,
        split_predense_rms=split_predense_rms,
        split_layer1_rms=split_layer1_rms,
    )


def execute_integrated_dense_rms(
    compiled: CompiledIntegratedDenseRms,
    weights: IntegratedDenseWeights,
    inputs: IntegratedDenseRmsInputs,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Execute twice and return the deterministic replicated U16 row."""

    import jax

    if compiled.native_source_context:
        raise BenchmarkValidationError(
            "native source context requires its exact executor"
        )
    _validate_weights(weights)
    _validate_inputs(inputs)
    common_arguments = (
        jax.device_put(
            inputs.attention_update_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
        jax.device_put(
            inputs.combined_residual_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
    )
    tail_arguments = (
        jax.device_put(
            inputs.post_attention_norm_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
        jax.device_put(weights.merged_bits, compiled.member_sharding),
        jax.device_put(weights.merged_scale, compiled.member_sharding),
        jax.device_put(weights.down_bits, compiled.member_sharding),
        jax.device_put(weights.down_scale, compiled.member_sharding),
        jax.device_put(
            inputs.layer1_norm_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
    )
    arguments = (
        common_arguments
        + (
            jax.device_put(
                ACCEPTED_SOURCE_VALIDITY,
                compiled.replicated_sharding,
            ),
        )
        + tail_arguments
        if compiled.accepted_source_context
        else common_arguments + tail_arguments
    )

    def execute_once() -> tuple[np.ndarray, tuple[str, ...]]:
        result = compiled.compiled(*arguments)
        jax.block_until_ready(result)
        local = tuple(
            np.asarray(jax.device_get(shard.data), dtype=np.uint16).reshape(6144)
            for shard in sorted(
                result.addressable_shards,
                key=lambda shard: int(shard.device.id),
            )
        )
        hashes = tuple(array_sha256(value) for value in local)
        if not local or len(set(hashes)) != 1:
            raise BenchmarkValidationError(
                "integrated dense RMS output differs across local replicas"
            )
        return np.ascontiguousarray(local[0]), hashes

    output, local_hashes = execute_once()
    repeated, repeated_hashes = execute_once()
    if not np.array_equal(output, repeated):
        raise BenchmarkValidationError("integrated dense RMS is nondeterministic")
    return output, {
        "invocation_count": 2,
        "local_replica_output_sha256": list(local_hashes),
        "output_bits_sha256": array_sha256(output),
        "repeated_local_replica_output_sha256": list(repeated_hashes),
        "repeated_output_bits_sha256": array_sha256(repeated),
    }


def execute_native_source_context(
    compiled: CompiledIntegratedDenseRms,
    weights: NativeSourceWeights,
    inputs: NativeSourceInputs,
) -> tuple[np.ndarray, Mapping[str, Any], np.ndarray]:
    """Execute twice and return row zero, capture evidence and the full result."""

    import jax

    if not compiled.native_source_context:
        raise BenchmarkValidationError(
            "native source executor received another integrated mode"
        )
    _validate_native_source_weights(weights)
    _validate_native_source_inputs(inputs)
    arguments = (
        jax.device_put(
            inputs.attended_latent_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
        jax.device_put(inputs.token_ids, compiled.replicated_sharding),
        jax.device_put(
            inputs.post_attention_norm_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
        jax.device_put(weights.embedding, compiled.member_sharding),
        jax.device_put(weights.kv_b_bits, compiled.member_sharding),
        jax.device_put(weights.kv_b_scale, compiled.member_sharding),
        jax.device_put(weights.o_bits_in_out, compiled.member_sharding),
        jax.device_put(weights.o_scale_in_out, compiled.member_sharding),
        jax.device_put(weights.merged_bits, compiled.member_sharding),
        jax.device_put(weights.merged_scale, compiled.member_sharding),
        jax.device_put(weights.down_bits, compiled.member_sharding),
        jax.device_put(weights.down_scale, compiled.member_sharding),
        jax.device_put(
            inputs.layer1_norm_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
    )

    def execute_once() -> tuple[
        np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]
    ]:
        result = compiled.compiled(*arguments)
        jax.block_until_ready(result)
        local_full = tuple(
            np.asarray(jax.device_get(shard.data), dtype=np.uint16).reshape(
                (32, 6144) if compiled.native_m32_output else (1, 6144)
            )
            for shard in sorted(
                result.addressable_shards,
                key=lambda shard: int(shard.device.id),
            )
        )
        full_hashes = tuple(array_sha256(value) for value in local_full)
        if not local_full or len(set(full_hashes)) != 1:
            raise BenchmarkValidationError(
                "native source output differs across local replicas"
            )
        rows = tuple(np.ascontiguousarray(value[0]) for value in local_full)
        row_hashes = tuple(array_sha256(value) for value in rows)
        return (
            rows[0],
            np.ascontiguousarray(local_full[0]),
            row_hashes,
            full_hashes,
        )

    output, full_output, local_hashes, full_hashes = execute_once()
    repeated, repeated_full, repeated_hashes, repeated_full_hashes = (
        execute_once()
    )
    if (
        not np.array_equal(output, repeated)
        or not np.array_equal(full_output, repeated_full)
        or full_hashes != repeated_full_hashes
    ):
        raise BenchmarkValidationError("native source graph is nondeterministic")
    capture: dict[str, Any] = {
        "invocation_count": 2,
        "local_replica_output_sha256": list(local_hashes),
        "output_bits_sha256": array_sha256(output),
        "repeated_local_replica_output_sha256": list(repeated_hashes),
        "repeated_output_bits_sha256": array_sha256(repeated),
    }
    if compiled.native_m32_output:
        capture.update(
            {
                "full_m32_output": True,
                "full_output_sha256": full_hashes[0],
                "local_replica_full_output_sha256": list(full_hashes),
                "repeated_full_output_sha256": repeated_full_hashes[0],
                "repeated_local_replica_full_output_sha256": list(
                    repeated_full_hashes
                ),
            }
        )
    return output, capture, full_output
