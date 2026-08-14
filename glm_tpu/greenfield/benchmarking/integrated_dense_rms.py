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
from ..kernels.stage_local import (
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
class CompiledIntegratedDenseRms:
    compiled: Any
    member_sharding: Any
    replicated_sharding: Any
    member_device_ids: tuple[int, ...]
    stablehlo: str
    optimized_hlo: str
    stablehlo_contract: Mapping[str, Any]
    optimized_hlo_contract: Mapping[str, Any]
    preceding_attention_collective: bool
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


def _integrated_function(
    *,
    split_layer1_rms: bool = False,
    preceding_attention_collective: bool = False,
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


def build_integrated_dense_rms(
    member_device_ids: Sequence[int],
    *,
    devices: Sequence[Any] | None = None,
    validate_hlo: bool = True,
    split_layer1_rms: bool = False,
    preceding_attention_collective: bool = False,
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
    member = NamedSharding(mesh, P("member", None, None, None))
    replicated = NamedSharding(mesh, P())
    if not isinstance(split_layer1_rms, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS split flag must be boolean"
        )
    if not isinstance(preceding_attention_collective, bool):
        raise BenchmarkValidationError(
            "integrated dense RMS preceding-attention flag must be boolean"
        )
    if preceding_attention_collective and not split_layer1_rms:
        raise BenchmarkValidationError(
            "preceding attention collective requires the frozen split RMS arm"
        )
    mapped = jax.shard_map(
        _integrated_function(
            split_layer1_rms=split_layer1_rms,
            preceding_attention_collective=preceding_attention_collective,
        ),
        mesh=mesh,
        in_specs=(
            P(),
            P(),
            P(),
            P("member", None, None, None),
            P("member", None, None, None),
            P("member", None, None, None),
            P("member", None, None, None),
            P(),
        ),
        out_specs=P(),
        check_vma=False,
    )
    examples = (
        jax.ShapeDtypeStruct((1, 6144), jnp.bfloat16, sharding=replicated),
        jax.ShapeDtypeStruct((1, 6144), jnp.bfloat16, sharding=replicated),
        jax.ShapeDtypeStruct((6144,), jnp.bfloat16, sharding=replicated),
        jax.ShapeDtypeStruct((32, 1, 6144, 768), jnp.float8_e4m3fn, sharding=member),
        jax.ShapeDtypeStruct((32, 1, 48, 768), jnp.float32, sharding=member),
        jax.ShapeDtypeStruct((32, 1, 384, 6144), jnp.float8_e4m3fn, sharding=member),
        jax.ShapeDtypeStruct((32, 1, 3, 6144), jnp.float32, sharding=member),
        jax.ShapeDtypeStruct((6144,), jnp.bfloat16, sharding=replicated),
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
        )
        optimized_hlo_contract = validate_integrated_dense_rms_hlo(
            optimized_hlo,
            members,
            split_layer1_rms=split_layer1_rms,
            preceding_attention_collective=preceding_attention_collective,
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
        stablehlo=stablehlo,
        optimized_hlo=optimized_hlo,
        stablehlo_contract=stablehlo_contract,
        optimized_hlo_contract=optimized_hlo_contract,
        preceding_attention_collective=preceding_attention_collective,
        split_layer1_rms=split_layer1_rms,
    )


def execute_integrated_dense_rms(
    compiled: CompiledIntegratedDenseRms,
    weights: IntegratedDenseWeights,
    inputs: IntegratedDenseRmsInputs,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    """Execute twice and return the deterministic replicated U16 row."""

    import jax

    _validate_weights(weights)
    _validate_inputs(inputs)
    arguments = (
        jax.device_put(
            inputs.attention_update_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
        jax.device_put(
            inputs.combined_residual_bits.view(ml_dtypes.bfloat16),
            compiled.replicated_sharding,
        ),
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
