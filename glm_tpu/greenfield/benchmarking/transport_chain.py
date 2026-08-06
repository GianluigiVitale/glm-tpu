"""Device-resident PP8/PP16 stage-transfer skeleton.

One compiled global JAX program advances every lane to its topology-adjacent
next stage with ``lax.ppermute``.  Repeating the operation once per stage
returns each rank-dependent payload to its exact origin.  There is no host or
Python stage dispatch inside a measured invocation.

This is a synthetic Gate-A mechanism benchmark, never model throughput.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import time
from typing import Any, Mapping, Sequence

from ..errors import BenchmarkValidationError, HloContractViolationError
from ..kernels.pallas import stage_value_remote_copy_pallas
from ..sharding.hlo_contract import (
    COLLECTIVE_OPCODES,
    CollectiveExpectation,
    HloContractPolicy,
    HloLintReport,
    lint_hlo,
    parse_hlo_module,
)
from ..types import PlanName
from .collective_chain import (
    addressable_checksum,
    jax_dtype,
    latency_distribution,
)


class TransportKind(StrEnum):
    CONTROL = "control"
    DEVICE_RESIDENT = "device_resident"
    PALLAS_REMOTE_COPY = "pallas_remote_copy"


_STAGE_COUNTS = {
    PlanName.PP8_LP4: 8,
    PlanName.PP16_LP2: 16,
}
_DTYPES = frozenset(("bfloat16", "float32", "int32"))
_HLO_DTYPES = {"bfloat16": "bf16", "float32": "f32", "int32": "s32"}


@dataclass(frozen=True, slots=True)
class TransportChainConfig:
    plan: PlanName
    kind: TransportKind
    rows: int
    width: int
    dtype: str
    warmup_iterations: int = 200
    measured_iterations: int = 2000

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "plan", PlanName(self.plan))
        except ValueError as error:
            raise BenchmarkValidationError(f"unknown transport plan {self.plan!r}") from error
        try:
            object.__setattr__(self, "kind", TransportKind(self.kind))
        except ValueError as error:
            raise BenchmarkValidationError(
                f"unknown transport kind {self.kind!r}"
            ) from error
        if self.plan not in _STAGE_COUNTS:
            raise BenchmarkValidationError(
                "transport skeleton supports only PP8_LP4 and PP16_LP2"
            )
        for field in ("rows", "width", "warmup_iterations", "measured_iterations"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise BenchmarkValidationError(
                    f"{field} must be a positive integer, got {value!r}"
                )
        if self.dtype not in _DTYPES:
            raise BenchmarkValidationError(
                f"dtype must be one of {sorted(_DTYPES)}, got {self.dtype!r}"
            )

    @property
    def stage_count(self) -> int:
        return _STAGE_COUNTS[self.plan]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.width)

    def require_protected_contract(self) -> None:
        problems = []
        if self.warmup_iterations < 200:
            problems.append("warmup_iterations must be at least 200")
        if self.measured_iterations < 1000:
            problems.append("measured_iterations must be at least 1000")
        if problems:
            raise BenchmarkValidationError(
                "protected transport benchmark: " + "; ".join(problems)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "kind": self.kind.value,
            "measured_iterations": self.measured_iterations,
            "plan": self.plan.value,
            "rows": self.rows,
            "stage_count": self.stage_count,
            "warmup_iterations": self.warmup_iterations,
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class CompiledTransportChain:
    config: TransportChainConfig
    compiled: Any
    input_value: Any
    physical_pairs: tuple[tuple[int, int], ...]
    optimized_hlo: str
    hlo_report: HloLintReport
    compile_seconds: float


def validate_transport_pairs(
    pairs: Sequence[Sequence[int]],
    *,
    total_devices: int,
    stage_count: int,
) -> tuple[tuple[int, int], ...]:
    """Require disjoint closed cycles of exactly one device per stage."""

    canonical = tuple((int(pair[0]), int(pair[1])) for pair in pairs if len(pair) == 2)
    if len(canonical) != len(pairs) or len(canonical) != total_devices:
        raise BenchmarkValidationError(
            "transport pairs must contain exactly one pair per physical device"
        )
    sources = [source for source, _ in canonical]
    targets = [target for _, target in canonical]
    expected = list(range(total_devices))
    if sorted(sources) != expected or sorted(targets) != expected:
        raise BenchmarkValidationError(
            "transport pairs must be a permutation of every physical device"
        )
    if any(source == target for source, target in canonical):
        raise BenchmarkValidationError("transport pairs cannot contain self edges")
    successor = dict(canonical)
    for start in expected:
        visited = []
        current = start
        while current not in visited:
            visited.append(current)
            current = successor[current]
        if current != start or len(visited) != stage_count:
            raise BenchmarkValidationError(
                f"device {start} belongs to a {len(visited)}-hop transport cycle; "
                f"expected {stage_count}"
            )
    return canonical


def transport_chain_hlo_policy(
    config: TransportChainConfig,
    physical_pairs: Sequence[Sequence[int]],
    *,
    total_devices: int,
    partition_id_to_device_id: Sequence[int],
) -> HloContractPolicy:
    pairs = validate_transport_pairs(
        physical_pairs,
        total_devices=total_devices,
        stage_count=config.stage_count,
    )
    expected_permute = (
        config.stage_count
        if config.kind is TransportKind.DEVICE_RESIDENT
        else 0
    )
    expectations = tuple(
        CollectiveExpectation(
            opcode,
            expected_permute if opcode == "collective-permute" else 0,
            repeated_only=False,
        )
        for opcode in sorted(COLLECTIVE_OPCODES)
    )
    return HloContractPolicy(
        name=(
            f"{config.plan.value.lower()}-{config.kind.value}-transport-"
            f"{config.dtype}-{config.rows}x{config.width}"
        ),
        total_devices=total_devices,
        repeated_region_patterns=(
            r"device_resident_stage_transport",
            r"manual_computation_body",
        ),
        maximum_repeated_collective_group_size=2,
        expected_collectives=expectations,
        expected_collective_permute_pairs=(
            pairs * config.stage_count
            if config.kind is TransportKind.DEVICE_RESIDENT
            else ()
        ),
        partition_id_to_device_id=tuple(partition_id_to_device_id),
        allow_full_pod_repeated_collectives=False,
    )


def _transport_function(
    config: TransportChainConfig,
    logical_pairs: tuple[tuple[int, int], ...],
) -> Any:
    import jax
    from jax import lax
    import jax.numpy as jnp

    targets = tuple(dict(logical_pairs)[rank] for rank in range(len(logical_pairs)))

    def transport(initial: Any) -> Any:
        with jax.named_scope("device_resident_stage_transport"):
            member = lax.axis_index("device")
            if config.dtype == "int32":
                state = initial + (member.astype(jnp.int32) + 1) * 17
            else:
                state = initial + (member.astype(jnp.float32) + 1).astype(
                    initial.dtype
                ) / jnp.asarray(256.0, initial.dtype)
            for hop in range(config.stage_count):
                if config.kind is TransportKind.DEVICE_RESIDENT:
                    state = lax.ppermute(state, "device", logical_pairs)
                elif config.kind is TransportKind.PALLAS_REMOTE_COPY:
                    destination = jnp.asarray(targets, dtype=jnp.int32)[member]
                    state = stage_value_remote_copy_pallas(
                        state,
                        destination,
                    )
                anchor = state.reshape(-1)[hop % state.size]
                if config.dtype == "int32":
                    feedback = jnp.bitwise_xor(anchor, jnp.right_shift(anchor, 3))
                    state = jnp.mod(
                        state + feedback + member.astype(jnp.int32) + hop + 1,
                        jnp.asarray(1048573, jnp.int32),
                    )
                else:
                    anchor_f32 = anchor.astype(jnp.float32)
                    feedback = anchor_f32 / (jnp.asarray(1.0) + jnp.abs(anchor_f32))
                    feedback += member.astype(jnp.float32) / jnp.asarray(256.0)
                    state = (
                        state * jnp.asarray(0.75, state.dtype)
                        + feedback.astype(state.dtype)
                        / jnp.asarray(32.0, state.dtype)
                    )
                state = lax.optimization_barrier(state)
            return state

    return transport


def _require_exact_payload_shapes(
    report: HloLintReport,
    config: TransportChainConfig,
) -> None:
    if config.kind is not TransportKind.DEVICE_RESIDENT:
        return
    expected = (config.rows, config.width)
    expected_dtype = _HLO_DTYPES[config.dtype]
    collectives = tuple(
        instruction
        for instruction in report.module.collectives
        if instruction.opcode == "collective-permute"
    )
    malformed = []
    for instruction in collectives:
        shapes = instruction.result_shapes + instruction.operand_shapes
        if not any(
            shape.dimensions == expected and shape.dtype == expected_dtype
            for shape in shapes
        ):
            malformed.append(instruction.name)
    if len(collectives) != config.stage_count or malformed:
        raise HloContractViolationError(
            f"transport payload shape contract expected {config.stage_count} "
            f"{expected_dtype}{expected} permutes; malformed={malformed}"
        )


def _require_exact_pallas_remote_copy(
    optimized_hlo: str,
    config: TransportChainConfig,
) -> dict[str, Any]:
    dtype = _HLO_DTYPES[config.dtype]
    kernel_name = (
        f"greenfield_stage_remote_copy_{dtype}_{config.rows}x{config.width}"
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
    violations = []
    expected_count = (
        config.stage_count
        if config.kind is TransportKind.PALLAS_REMOTE_COPY
        else 0
    )
    if len(kernel_calls) != expected_count:
        violations.append(
            f"expected {expected_count} {kernel_name} calls, found {len(kernel_calls)}"
        )
    unexpected_tpu_calls = [
        line
        for line in custom_calls
        if 'custom_call_target="tpu_custom_call"' in line and line not in kernel_calls
    ]
    if unexpected_tpu_calls:
        violations.append(
            f"unexpected transport TPU custom calls: {unexpected_tpu_calls}"
        )
    payload_shape = f"{dtype}[{config.rows},{config.width}]"
    malformed = [line for line in kernel_calls if payload_shape not in line]
    if malformed:
        violations.append(
            f"Pallas remote-copy calls lack exact payload shape {payload_shape}"
        )
    return {
        "expected_kernel_name": kernel_name,
        "kernel_custom_call_count": len(kernel_calls),
        "kernel_custom_calls": kernel_calls,
        "passed": not violations,
        "unexpected_tpu_custom_calls": unexpected_tpu_calls,
        "violations": violations,
    }


def build_transport_chain(
    config: TransportChainConfig,
    physical_pairs: Sequence[Sequence[int]],
    *,
    devices: Sequence[Any] | None = None,
    enforce_hlo_contract: bool = True,
) -> CompiledTransportChain:
    """Compile one full stage ring and fail unless its physical HLO is exact."""

    import jax
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
        raise BenchmarkValidationError(
            "transport benchmark requires contiguous global JAX device ids"
        )
    partition_id_to_device_id = tuple(range(total_devices))
    mesh = Mesh(
        np.asarray([by_id[item] for item in partition_id_to_device_id], dtype=object),
        ("device",),
    )
    sharding = NamedSharding(mesh, P("device", None))
    mapped = jax.shard_map(
        _transport_function(config, pairs),
        mesh=mesh,
        in_specs=P("device", None),
        out_specs=P("device", None),
        check_vma=False,
    )
    executable = jax.jit(mapped)
    if config.dtype == "int32":
        host = np.arange(
            total_devices * config.rows * config.width,
            dtype=np.int32,
        ).reshape(total_devices * config.rows, config.width) % 251
    else:
        host = np.linspace(
            -0.25,
            0.25,
            num=total_devices * config.rows * config.width,
            dtype=np.float32,
        ).reshape(total_devices * config.rows, config.width)
    input_value = jax.device_put(host.astype(jax_dtype(config.dtype)), sharding)
    started = time.perf_counter()
    compiled = executable.lower(input_value).compile()
    compile_seconds = time.perf_counter() - started
    optimized_hlo = compiled.as_text()
    policy = transport_chain_hlo_policy(
        config,
        pairs,
        total_devices=total_devices,
        partition_id_to_device_id=partition_id_to_device_id,
    )
    report = lint_hlo(parse_hlo_module(optimized_hlo), policy)
    if enforce_hlo_contract:
        report.raise_for_violations()
        _require_exact_payload_shapes(report, config)
        pallas_contract = _require_exact_pallas_remote_copy(
            optimized_hlo, config
        )
        if not pallas_contract["passed"]:
            raise HloContractViolationError(
                f"Pallas transport HLO rejected: {pallas_contract['violations']}"
            )
    return CompiledTransportChain(
        config=config,
        compiled=compiled,
        input_value=input_value,
        physical_pairs=pairs,
        optimized_hlo=optimized_hlo,
        hlo_report=report,
        compile_seconds=compile_seconds,
    )


def validate_compiled_transport(compiled: CompiledTransportChain) -> None:
    """Apply count/pair/group and exact payload-shape gates to saved HLO."""

    compiled.hlo_report.raise_for_violations()
    _require_exact_payload_shapes(compiled.hlo_report, compiled.config)
    pallas_contract = _require_exact_pallas_remote_copy(
        compiled.optimized_hlo, compiled.config
    )
    if not pallas_contract["passed"]:
        raise HloContractViolationError(
            f"Pallas transport HLO rejected: {pallas_contract['violations']}"
        )


def benchmark_transport_chain(compiled: CompiledTransportChain) -> dict[str, Any]:
    """Measure synchronized profiler-free wall latency for one full ring."""

    import jax

    first = compiled.compiled(compiled.input_value)
    jax.block_until_ready(first)
    first_checksum = addressable_checksum(first)
    for _ in range(compiled.config.warmup_iterations - 1):
        warmed = compiled.compiled(compiled.input_value)
        jax.block_until_ready(warmed)
    samples_ms = []
    last = first
    for _ in range(compiled.config.measured_iterations):
        started_ns = time.perf_counter_ns()
        last = compiled.compiled(compiled.input_value)
        jax.block_until_ready(last)
        samples_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000.0)
    last_checksum = addressable_checksum(last)
    if first_checksum != last_checksum:
        raise BenchmarkValidationError(
            "identical transport invocations produced different output bytes"
        )
    distribution = latency_distribution(samples_ms)
    pallas_contract = _require_exact_pallas_remote_copy(
        compiled.optimized_hlo, compiled.config
    )
    return {
        "compile_seconds": compiled.compile_seconds,
        "config": compiled.config.to_dict(),
        "first_addressable_checksum": first_checksum,
        "hlo": compiled.hlo_report.to_dict(),
        "last_addressable_checksum": last_checksum,
        "latency": distribution.to_dict(),
        "mechanism_only": True,
        "pallas_remote_copy": pallas_contract,
        "per_stage_hop_p50_ms": (
            None
            if compiled.config.kind is TransportKind.CONTROL
            else distribution.p50_ms / compiled.config.stage_count
        ),
        "physical_pairs": [list(pair) for pair in compiled.physical_pairs],
    }
