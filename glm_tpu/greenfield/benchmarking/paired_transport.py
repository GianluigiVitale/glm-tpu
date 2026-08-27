"""Production-shaped residual plus metadata stage-transfer benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import prod
import time
from typing import Any, Sequence

from ..errors import BenchmarkValidationError, HloContractViolationError
from ..kernels.pallas import StageRemoteCopyConfig, stage_remote_copy_kernel
from ..sharding.hlo_contract import parse_hlo_module
from ..types import PlanName
from .collective_chain import addressable_checksum, latency_distribution
from .transport_chain import validate_transport_pairs


_STAGE_COUNTS = {PlanName.PP8_LP4: 8, PlanName.PP16_LP2: 16}


class PairedTransportKind(StrEnum):
    """Mechanisms for transporting both live stage-boundary payloads."""

    CONTROL = "control"
    DEVICE_RESIDENT = "device_resident"
    PACKED_DEVICE_RESIDENT = "packed_device_resident"
    PALLAS_REMOTE_COPY = "pallas_remote_copy"


@dataclass(frozen=True, slots=True)
class PairedTransportConfig:
    plan: PlanName
    kind: PairedTransportKind
    residual_shape: tuple[int, ...] = (1, 6144)
    metadata_shape: tuple[int, ...] = (1, 2052)
    warmup_iterations: int = 200
    measured_iterations: int = 2000

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "plan", PlanName(self.plan))
        except ValueError as error:
            raise BenchmarkValidationError(f"unknown plan {self.plan!r}") from error
        try:
            value = getattr(self.kind, "value", self.kind)
            object.__setattr__(self, "kind", PairedTransportKind(value))
        except ValueError as error:
            raise BenchmarkValidationError(f"unknown kind {self.kind!r}") from error
        if self.plan not in _STAGE_COUNTS:
            raise BenchmarkValidationError("paired transport supports PP8/PP16 only")
        for name in ("warmup_iterations", "measured_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise BenchmarkValidationError(f"{name} must be a positive integer")
        for name in ("residual_shape", "metadata_shape"):
            shape = getattr(self, name)
            if (
                not isinstance(shape, tuple)
                or not shape
                or any(
                    not isinstance(dimension, int)
                    or isinstance(dimension, bool)
                    or dimension <= 0
                    for dimension in shape
                )
            ):
                raise BenchmarkValidationError(
                    f"{name} must be a nonempty tuple of positive integers"
                )
        if self.residual_shape[-1] != 6144:
            raise BenchmarkValidationError(
                "paired transport residual width must remain 6144"
            )
        if self.metadata_shape[0] != 1:
            raise BenchmarkValidationError(
                "paired transport metadata must retain one live row"
            )
        if self.kind is PairedTransportKind.PALLAS_REMOTE_COPY and (
            self.residual_shape != (1, 6144)
            or self.metadata_shape != (1, 2052)
        ):
            raise BenchmarkValidationError(
                "paired Pallas remote copy supports only its sealed legacy payload"
            )

    @property
    def stage_count(self) -> int:
        return _STAGE_COUNTS[self.plan]

    @property
    def hidden_width(self) -> int:
        return self.residual_shape[-1]

    @property
    def metadata_width(self) -> int:
        return self.metadata_shape[-1]

    @property
    def packed_width(self) -> int:
        """Number of uint16 lanes in the exact bit-preserving payload."""

        return prod(self.residual_shape) + 2 * prod(self.metadata_shape)

    def require_protected_contract(self) -> None:
        if self.warmup_iterations < 200 or self.measured_iterations < 1000:
            raise BenchmarkValidationError(
                "protected paired transport requires 200 warmups and 1000 samples"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_width": self.hidden_width,
            "kind": self.kind.value,
            "measured_iterations": self.measured_iterations,
            "metadata_shape": list(self.metadata_shape),
            "metadata_width": self.metadata_width,
            "packed_dtype": "uint16",
            "packed_width": self.packed_width,
            "plan": self.plan.value,
            "residual_shape": list(self.residual_shape),
            "stage_count": self.stage_count,
            "warmup_iterations": self.warmup_iterations,
        }


@dataclass(frozen=True, slots=True)
class CompiledPairedTransport:
    config: PairedTransportConfig
    compiled: Any
    residual: Any
    metadata: Any
    physical_pairs: tuple[tuple[int, int], ...]
    optimized_hlo: str
    hlo_contract: dict[str, Any]
    compile_seconds: float
    memory_analysis: dict[str, int]


def validate_paired_transport_hlo(
    optimized_hlo: str,
    *,
    config: PairedTransportConfig,
    physical_pairs: Sequence[Sequence[int]],
    total_devices: int,
) -> dict[str, Any]:
    """Require the exact physical mechanism and payload shapes for each hop."""

    pairs = validate_transport_pairs(
        physical_pairs,
        total_devices=total_devices,
        stage_count=config.stage_count,
    )
    module = parse_hlo_module(optimized_hlo)
    permutes = [
        item for item in module.collectives if item.opcode == "collective-permute"
    ]
    other_collectives = [
        item.name
        for item in module.collectives
        if item.opcode != "collective-permute"
    ]
    expected_permute_count = {
        PairedTransportKind.CONTROL: 0,
        PairedTransportKind.DEVICE_RESIDENT: 2 * config.stage_count,
        PairedTransportKind.PACKED_DEVICE_RESIDENT: config.stage_count,
        PairedTransportKind.PALLAS_REMOTE_COPY: 0,
    }[config.kind]
    expected_pair_multiset = sorted(pairs * expected_permute_count)
    observed_pair_multiset = sorted(
        pair for item in permutes for pair in item.source_target_pairs
    )
    kernel_name = (
        "greenfield_stage_remote_copy_"
        f"bf16_{config.hidden_width}_s32_{config.metadata_width}"
    )
    custom_calls = [
        line.strip()
        for line in optimized_hlo.splitlines()
        if " custom-call(" in line
    ]
    kernel_calls = [
        line
        for line in custom_calls
        if kernel_name in line and 'custom_call_target="tpu_custom_call"' in line
    ]
    expected_kernel_count = (
        config.stage_count
        if config.kind is PairedTransportKind.PALLAS_REMOTE_COPY
        else 0
    )
    unexpected_tpu_calls = [
        line
        for line in custom_calls
        if 'custom_call_target="tpu_custom_call"' in line and line not in kernel_calls
    ]
    violations = []
    if len(permutes) != expected_permute_count:
        violations.append(
            f"expected {expected_permute_count} permutes, found {len(permutes)}"
        )
    if observed_pair_multiset != expected_pair_multiset:
        violations.append("paired transport lane pairs or counts drifted")
    if other_collectives:
        violations.append(
            f"paired transport has forbidden collectives {other_collectives}"
        )
    if len(kernel_calls) != expected_kernel_count:
        violations.append(
            f"expected {expected_kernel_count} paired Pallas calls, "
            f"found {len(kernel_calls)}"
        )
    residual_hlo_shape = "bf16[" + ",".join(
        str(item) for item in config.residual_shape
    ) + "]"
    metadata_hlo_shape = "s32[" + ",".join(
        str(item) for item in config.metadata_shape
    ) + "]"
    packed_hlo_shape = f"u16[{config.packed_width}]"
    required_shapes = (residual_hlo_shape, metadata_hlo_shape)
    malformed_calls = [
        line
        for line in kernel_calls
        if any(shape not in line for shape in required_shapes)
        or '"has_communication":true' not in line
    ]
    if malformed_calls:
        violations.append("paired Pallas calls lack payload/communication contract")
    if unexpected_tpu_calls:
        violations.append(
            f"paired transport has unexpected TPU calls {unexpected_tpu_calls}"
        )
    def expanded_live_row_shapes(
        dtype: str,
        shape: tuple[int, ...],
    ) -> tuple[str, ...]:
        expanded = set()
        for axis, dimension in enumerate(shape):
            if dimension != 1:
                continue
            dimensions = list(shape)
            dimensions[axis] = 32
            expanded.add(
                f"{dtype}[" + ",".join(str(item) for item in dimensions) + "]"
            )
        return tuple(sorted(expanded))

    forbidden_dead_rows = [
        shape
        for shape in (
            *expanded_live_row_shapes("bf16", config.residual_shape),
            *expanded_live_row_shapes("s32", config.metadata_shape),
        )
        if shape in optimized_hlo
    ]
    if forbidden_dead_rows:
        violations.append(f"paired transport has dead rows {forbidden_dead_rows}")
    residual_permutes = []
    metadata_permutes = []
    packed_permutes = []
    malformed_permutes = []
    for item in permutes:
        shapes = item.result_shapes + item.operand_shapes
        if any(
            shape.dtype == "bf16"
            and shape.dimensions == config.residual_shape
            for shape in shapes
        ):
            residual_permutes.append(item.name)
        elif any(
            shape.dtype == "s32"
            and shape.dimensions == config.metadata_shape
            for shape in shapes
        ):
            metadata_permutes.append(item.name)
        elif any(
            shape.dtype == "u16"
            and shape.dimensions == (config.packed_width,)
            for shape in shapes
        ):
            packed_permutes.append(item.name)
        else:
            malformed_permutes.append(item.name)
    if config.kind is PairedTransportKind.DEVICE_RESIDENT and (
        len(residual_permutes) != config.stage_count
        or len(metadata_permutes) != config.stage_count
        or packed_permutes
    ):
        violations.append("separate paired permute payload shapes or counts drifted")
    if config.kind is PairedTransportKind.PACKED_DEVICE_RESIDENT and (
        residual_permutes
        or metadata_permutes
        or len(packed_permutes) != config.stage_count
    ):
        violations.append(
            "packed transport did not use one exact uint16 payload per hop"
        )
    if malformed_permutes:
        violations.append(
            f"paired transport has malformed permutes {malformed_permutes}"
        )
    return {
        "collective_count": len(module.collectives),
        "collectives": [item.to_dict() for item in module.collectives],
        "expected_kernel_name": kernel_name,
        "forbidden_dead_rows": forbidden_dead_rows,
        "kernel_custom_call_count": len(kernel_calls),
        "kernel_custom_calls": kernel_calls,
        "metadata_hlo_shape": metadata_hlo_shape,
        "module_name": module.name,
        "packed_hlo_shape": packed_hlo_shape,
        "packed_permute_count": len(packed_permutes),
        "passed": not violations,
        "residual_hlo_shape": residual_hlo_shape,
        "separate_metadata_permute_count": len(metadata_permutes),
        "separate_residual_permute_count": len(residual_permutes),
        "unexpected_tpu_custom_calls": unexpected_tpu_calls,
        "violations": violations,
    }


def pack_paired_transport_payload(
    residual: Any,
    metadata: Any,
    *,
    config: PairedTransportConfig,
) -> Any:
    """Bit-pack BF16 residual and int32 metadata into one uint16 vector."""

    import jax.numpy as jnp
    from jax import lax

    if residual.shape != config.residual_shape or residual.dtype != jnp.bfloat16:
        raise BenchmarkValidationError(
            "packed transport residual shape/dtype differs from its static contract"
        )
    if metadata.shape != config.metadata_shape or metadata.dtype != jnp.int32:
        raise BenchmarkValidationError(
            "packed transport metadata shape/dtype differs from its static contract"
        )
    residual_bits = lax.bitcast_convert_type(residual, jnp.uint16).reshape(-1)
    metadata_bits = lax.bitcast_convert_type(metadata, jnp.uint16).reshape(-1)
    packed = jnp.concatenate((residual_bits, metadata_bits), axis=0)
    if packed.shape != (config.packed_width,) or packed.dtype != jnp.uint16:
        raise BenchmarkValidationError("packed transport bit width drifted")
    return packed


def unpack_paired_transport_payload(
    packed: Any,
    *,
    config: PairedTransportConfig,
) -> tuple[Any, Any]:
    """Recover the exact BF16 and int32 arrays from one uint16 vector."""

    import jax.numpy as jnp
    from jax import lax

    if packed.shape != (config.packed_width,) or packed.dtype != jnp.uint16:
        raise BenchmarkValidationError(
            "packed transport vector shape/dtype differs from its static contract"
        )
    residual_size = prod(config.residual_shape)
    residual_bits = packed[:residual_size].reshape(config.residual_shape)
    metadata_bits = packed[residual_size:].reshape(config.metadata_shape + (2,))
    residual = lax.bitcast_convert_type(residual_bits, jnp.bfloat16)
    metadata = lax.bitcast_convert_type(metadata_bits, jnp.int32)
    return residual, metadata


def build_paired_transport(
    config: PairedTransportConfig,
    physical_pairs: Sequence[Sequence[int]],
    *,
    devices: Sequence[Any] | None = None,
    enforce_hlo_contract: bool = True,
) -> CompiledPairedTransport:
    """Compile one production-shaped full stage ring."""

    import jax
    from jax import lax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    runtime_devices = tuple(jax.devices() if devices is None else devices)
    total_devices = len(runtime_devices)
    pairs = validate_transport_pairs(
        physical_pairs,
        total_devices=total_devices,
        stage_count=config.stage_count,
    )
    by_id = {int(device.id): device for device in runtime_devices}
    if set(by_id) != set(range(total_devices)):
        raise BenchmarkValidationError("paired transport requires contiguous ids")
    mesh = Mesh(
        np.asarray([by_id[index] for index in range(total_devices)], dtype=object),
        ("device",),
    )
    residual_spec = P("device", *(None for _ in config.residual_shape))
    metadata_spec = P("device", *(None for _ in config.metadata_shape))
    residual_sharding = NamedSharding(mesh, residual_spec)
    metadata_sharding = NamedSharding(mesh, metadata_spec)
    remote_config = StageRemoteCopyConfig(
        total_devices=total_devices,
        hidden_width=config.hidden_width,
        metadata_width=config.metadata_width,
    )

    def mapped(local_residual: Any, local_metadata: Any) -> tuple[Any, Any]:
        member = lax.axis_index("device")
        residual = local_residual[0]
        metadata = local_metadata[0]
        for hop in range(config.stage_count):
            if config.kind is PairedTransportKind.DEVICE_RESIDENT:
                residual, metadata = (
                    lax.ppermute(residual, "device", pairs),
                    lax.ppermute(metadata, "device", pairs),
                )
            elif config.kind is PairedTransportKind.PACKED_DEVICE_RESIDENT:
                packed = pack_paired_transport_payload(
                    residual,
                    metadata,
                    config=config,
                )
                packed = lax.ppermute(packed, "device", pairs)
                residual, metadata = unpack_paired_transport_payload(
                    packed,
                    config=config,
                )
            elif config.kind is PairedTransportKind.PALLAS_REMOTE_COPY:
                residual, metadata = stage_remote_copy_kernel(
                    residual,
                    metadata,
                    axis_name="device",
                    pairs=pairs,
                    backend="pallas",
                    config=remote_config,
                )
            metadata_anchor = metadata.reshape(-1)[hop % metadata.size]
            metadata_feedback = jnp.bitwise_xor(
                metadata_anchor, jnp.right_shift(metadata_anchor, 3)
            )
            metadata = jnp.mod(
                metadata + metadata_feedback + member.astype(jnp.int32) + hop + 1,
                jnp.asarray(1048573, jnp.int32),
            )
            residual_anchor = residual.reshape(-1)[hop % residual.size].astype(
                jnp.float32
            )
            residual_feedback = residual_anchor / (
                jnp.asarray(1.0) + jnp.abs(residual_anchor)
            )
            residual_feedback += member.astype(jnp.float32) / jnp.asarray(256.0)
            residual = (
                residual * jnp.asarray(0.75, residual.dtype)
                + residual_feedback.astype(residual.dtype)
                / jnp.asarray(32.0, residual.dtype)
            )
            residual = lax.optimization_barrier(residual)
            metadata = lax.optimization_barrier(metadata)
        return residual[None, ...], metadata[None, ...]

    execute = jax.jit(
        jax.shard_map(
            mapped,
            mesh=mesh,
            in_specs=(residual_spec, metadata_spec),
            out_specs=(residual_spec, metadata_spec),
            check_vma=False,
        )
    )
    residual_host = np.linspace(
        -0.25,
        0.25,
        num=total_devices * prod(config.residual_shape),
        dtype=np.float32,
    ).reshape((total_devices,) + config.residual_shape)
    metadata_host = (
        np.arange(total_devices * prod(config.metadata_shape), dtype=np.int32)
        .reshape((total_devices,) + config.metadata_shape)
        % 251
    )
    residual = jax.device_put(
        residual_host.astype(jnp.bfloat16), residual_sharding
    )
    metadata = jax.device_put(metadata_host, metadata_sharding)
    started = time.perf_counter()
    compiled = execute.lower(residual, metadata).compile()
    compile_seconds = time.perf_counter() - started
    optimized_hlo = compiled.as_text()
    contract = validate_paired_transport_hlo(
        optimized_hlo,
        config=config,
        physical_pairs=pairs,
        total_devices=total_devices,
    )
    if enforce_hlo_contract and not contract["passed"]:
        raise HloContractViolationError(
            f"paired transport HLO rejected: {contract['violations']}"
        )
    analysis = compiled.memory_analysis()
    memory_analysis = {
        name: int(getattr(analysis, name))
        for name in (
            "argument_size_in_bytes",
            "output_size_in_bytes",
            "temp_size_in_bytes",
        )
    }
    return CompiledPairedTransport(
        config=config,
        compiled=compiled,
        residual=residual,
        metadata=metadata,
        physical_pairs=pairs,
        optimized_hlo=optimized_hlo,
        hlo_contract=contract,
        compile_seconds=compile_seconds,
        memory_analysis=memory_analysis,
    )


def benchmark_paired_transport(compiled: CompiledPairedTransport) -> dict[str, Any]:
    """Measure synchronized profiler-free latency for both live payloads."""

    import jax

    first = compiled.compiled(compiled.residual, compiled.metadata)
    jax.block_until_ready(first)
    first_checksum = addressable_checksum(first)
    for _ in range(compiled.config.warmup_iterations - 1):
        warmed = compiled.compiled(compiled.residual, compiled.metadata)
        jax.block_until_ready(warmed)
    samples_ms = []
    last = first
    for _ in range(compiled.config.measured_iterations):
        started_ns = time.perf_counter_ns()
        last = compiled.compiled(compiled.residual, compiled.metadata)
        jax.block_until_ready(last)
        samples_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
    last_checksum = addressable_checksum(last)
    if first_checksum != last_checksum:
        raise BenchmarkValidationError("paired transport output is nondeterministic")
    distribution = latency_distribution(samples_ms)
    return {
        "compile_seconds": compiled.compile_seconds,
        "config": compiled.config.to_dict(),
        "first_addressable_checksum": first_checksum,
        "hlo": compiled.hlo_contract,
        "last_addressable_checksum": last_checksum,
        "latency": distribution.to_dict(),
        "mechanism_only": True,
        "memory_analysis": compiled.memory_analysis,
        "per_stage_hop_p50_ms": (
            None
            if compiled.config.kind is PairedTransportKind.CONTROL
            else distribution.p50_ms / compiled.config.stage_count
        ),
        "physical_pairs": [list(pair) for pair in compiled.physical_pairs],
    }
