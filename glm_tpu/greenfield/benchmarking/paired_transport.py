"""Production-shaped residual plus metadata stage-transfer benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

from ..errors import BenchmarkValidationError, HloContractViolationError
from ..kernels.pallas import StageRemoteCopyConfig, stage_remote_copy_kernel
from ..sharding.hlo_contract import parse_hlo_module
from ..types import PlanName
from .collective_chain import addressable_checksum, latency_distribution
from .transport_chain import TransportKind, validate_transport_pairs


_STAGE_COUNTS = {PlanName.PP8_LP4: 8, PlanName.PP16_LP2: 16}


@dataclass(frozen=True, slots=True)
class PairedTransportConfig:
    plan: PlanName
    kind: TransportKind
    hidden_width: int = 6144
    metadata_width: int = 2052
    warmup_iterations: int = 200
    measured_iterations: int = 2000

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "plan", PlanName(self.plan))
        except ValueError as error:
            raise BenchmarkValidationError(f"unknown plan {self.plan!r}") from error
        try:
            object.__setattr__(self, "kind", TransportKind(self.kind))
        except ValueError as error:
            raise BenchmarkValidationError(f"unknown kind {self.kind!r}") from error
        if self.plan not in _STAGE_COUNTS:
            raise BenchmarkValidationError("paired transport supports PP8/PP16 only")
        for name in (
            "hidden_width",
            "metadata_width",
            "warmup_iterations",
            "measured_iterations",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise BenchmarkValidationError(f"{name} must be a positive integer")

    @property
    def stage_count(self) -> int:
        return _STAGE_COUNTS[self.plan]

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
            "metadata_width": self.metadata_width,
            "plan": self.plan.value,
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
    """Require either two exact permutes or one paired DMA call per stage."""

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
    expected_permute_count = (
        2 * config.stage_count
        if config.kind is TransportKind.DEVICE_RESIDENT
        else 0
    )
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
        if config.kind is TransportKind.PALLAS_REMOTE_COPY
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
        violations.append(f"paired transport has forbidden collectives {other_collectives}")
    if len(kernel_calls) != expected_kernel_count:
        violations.append(
            f"expected {expected_kernel_count} paired Pallas calls, "
            f"found {len(kernel_calls)}"
        )
    required_shapes = (
        f"bf16[1,{config.hidden_width}]",
        f"s32[1,{config.metadata_width}]",
    )
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
    forbidden_dead_rows = [
        shape
        for shape in (
            f"bf16[32,{config.hidden_width}]",
            f"s32[32,{config.metadata_width}]",
        )
        if shape in optimized_hlo
    ]
    if forbidden_dead_rows:
        violations.append(f"paired transport has dead rows {forbidden_dead_rows}")
    return {
        "collective_count": len(module.collectives),
        "collectives": [item.to_dict() for item in module.collectives],
        "expected_kernel_name": kernel_name,
        "forbidden_dead_rows": forbidden_dead_rows,
        "kernel_custom_call_count": len(kernel_calls),
        "kernel_custom_calls": kernel_calls,
        "module_name": module.name,
        "passed": not violations,
        "unexpected_tpu_custom_calls": unexpected_tpu_calls,
        "violations": violations,
    }


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
    sharding = NamedSharding(mesh, P("device", None, None))
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
            if config.kind is not TransportKind.CONTROL:
                residual, metadata = stage_remote_copy_kernel(
                    residual,
                    metadata,
                    axis_name="device",
                    pairs=pairs,
                    backend=(
                        "pallas"
                        if config.kind is TransportKind.PALLAS_REMOTE_COPY
                        else "reference"
                    ),
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
            in_specs=(P("device", None, None), P("device", None, None)),
            out_specs=(P("device", None, None), P("device", None, None)),
            check_vma=False,
        )
    )
    residual_host = np.linspace(
        -0.25,
        0.25,
        num=total_devices * config.hidden_width,
        dtype=np.float32,
    ).reshape(total_devices, 1, config.hidden_width)
    metadata_host = (
        np.arange(total_devices * config.metadata_width, dtype=np.int32)
        .reshape(total_devices, 1, config.metadata_width)
        % 251
    )
    residual = jax.device_put(residual_host.astype(jnp.bfloat16), sharding)
    metadata = jax.device_put(metadata_host, sharding)
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
            if compiled.config.kind is TransportKind.CONTROL
            else distribution.p50_ms / compiled.config.stage_count
        ),
        "physical_pairs": [list(pair) for pair in compiled.physical_pairs],
    }
