"""Genuinely dependent collective-chain mechanism benchmark.

The benchmark is deliberately model-free.  Each compiled invocation contains
an unrolled chain whose next collective depends nonlinearly on the previous
result.  Rank-dependent state, optimization barriers, output checksums, and an
exact optimized-HLO contract make compiler elision or reassociation fail loud.

Synthetic latency is a communication-floor discriminator, never a model
throughput claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import math
import statistics
import time
from typing import Any, Iterable, Mapping, Sequence

from ..errors import BenchmarkValidationError
from ..sharding.hlo_contract import (
    CollectiveExpectation,
    HloContractPolicy,
    HloLintReport,
    lint_hlo,
    parse_hlo_module,
)


class CollectiveKind(StrEnum):
    CONTROL = "control"
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    COLLECTIVE_PERMUTE = "collective_permute"
    ALL_TO_ALL = "all_to_all"
    FUSED_TUPLE_ALL_REDUCE = "fused_tuple_all_reduce"


_DTYPE_NAMES = frozenset(
    {
        "bfloat16",
        "float32",
        "int32",
    }
)


@dataclass(frozen=True, slots=True)
class CollectiveChainConfig:
    kind: CollectiveKind
    group_size: int
    rows: int
    width: int
    dtype: str
    chain_length: int = 75
    warmup_iterations: int = 200
    measured_iterations: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CollectiveKind):
            object.__setattr__(self, "kind", CollectiveKind(self.kind))
        for field in (
            "group_size",
            "rows",
            "width",
            "chain_length",
            "warmup_iterations",
            "measured_iterations",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise BenchmarkValidationError(
                    f"{field} must be a positive integer, got {value!r}"
                )
        if self.group_size not in (2, 4, 8, 32):
            raise BenchmarkValidationError(
                "group_size must be one of 2, 4, 8, or 32"
            )
        if self.dtype not in _DTYPE_NAMES:
            raise BenchmarkValidationError(
                f"dtype must be one of {sorted(_DTYPE_NAMES)}, got {self.dtype!r}"
            )
        if self.kind is CollectiveKind.ALL_TO_ALL and self.width % self.group_size:
            raise BenchmarkValidationError(
                "all-to-all width must be divisible by group_size"
            )
        if (
            self.kind is CollectiveKind.FUSED_TUPLE_ALL_REDUCE
            and self.dtype == "int32"
        ):
            raise BenchmarkValidationError(
                "fused tuple reduction requires a floating primary payload"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.width)

    @property
    def expected_opcode(self) -> str | None:
        return {
            CollectiveKind.CONTROL: None,
            CollectiveKind.ALL_REDUCE: "all-reduce",
            CollectiveKind.REDUCE_SCATTER: "reduce-scatter",
            CollectiveKind.ALL_GATHER: "all-gather",
            CollectiveKind.COLLECTIVE_PERMUTE: "collective-permute",
            CollectiveKind.ALL_TO_ALL: "all-to-all",
            CollectiveKind.FUSED_TUPLE_ALL_REDUCE: "all-reduce",
        }[self.kind]

    def require_protected_contract(self) -> None:
        problems = []
        if self.chain_length != 75:
            problems.append("chain_length must equal 75")
        if self.warmup_iterations < 200:
            problems.append("warmup_iterations must be at least 200")
        if self.measured_iterations < 1000:
            problems.append("measured_iterations must be at least 1000")
        if problems:
            raise BenchmarkValidationError(
                "protected collective benchmark: " + "; ".join(problems)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_length": self.chain_length,
            "dtype": self.dtype,
            "group_size": self.group_size,
            "kind": self.kind.value,
            "measured_iterations": self.measured_iterations,
            "rows": self.rows,
            "warmup_iterations": self.warmup_iterations,
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class LatencyDistribution:
    samples_ms: tuple[float, ...]
    mean_ms: float
    standard_deviation_ms: float
    minimum_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    maximum_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_ms": self.maximum_ms,
            "mean_ms": self.mean_ms,
            "minimum_ms": self.minimum_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "samples_ms": list(self.samples_ms),
            "standard_deviation_ms": self.standard_deviation_ms,
        }


@dataclass(frozen=True, slots=True)
class CompiledCollectiveChain:
    config: CollectiveChainConfig
    compiled: Any
    input_value: Any
    optimized_hlo: str
    hlo_report: HloLintReport
    compile_seconds: float
    compiler_options: Mapping[str, Any]


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise BenchmarkValidationError("latency distribution cannot be empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _distribution(samples_ms: Iterable[float]) -> LatencyDistribution:
    samples = tuple(float(value) for value in samples_ms)
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise BenchmarkValidationError(
            "latency samples must be finite positive values"
        )
    return LatencyDistribution(
        samples_ms=samples,
        mean_ms=statistics.fmean(samples),
        standard_deviation_ms=(statistics.pstdev(samples) if len(samples) > 1 else 0.0),
        minimum_ms=min(samples),
        p50_ms=_percentile(samples, 0.50),
        p90_ms=_percentile(samples, 0.90),
        p95_ms=_percentile(samples, 0.95),
        p99_ms=_percentile(samples, 0.99),
        maximum_ms=max(samples),
    )


def _validate_groups(
    groups: Sequence[Sequence[int]],
    group_size: int,
    total_devices: int,
) -> tuple[tuple[int, ...], ...]:
    canonical = tuple(tuple(int(device) for device in group) for group in groups)
    members = [device for group in canonical for device in group]
    if not canonical or any(len(group) != group_size for group in canonical):
        raise BenchmarkValidationError(
            f"every collective group must contain exactly {group_size} devices"
        )
    if (
        len(members) != total_devices
        or len(members) != len(set(members))
        or sorted(members) != list(range(total_devices))
    ):
        raise BenchmarkValidationError(
            "collective groups must partition global device ids 0..N-1 exactly once"
        )
    return canonical


def collective_chain_hlo_policy(
    config: CollectiveChainConfig,
    groups: Sequence[Sequence[int]],
    *,
    total_devices: int,
) -> HloContractPolicy:
    canonical_groups = _validate_groups(groups, config.group_size, total_devices)
    expectations = ()
    if config.expected_opcode is not None:
        expectations = (
            CollectiveExpectation(
                config.expected_opcode,
                config.chain_length,
                repeated_only=True,
            ),
        )
    pairs: tuple[tuple[int, int], ...] = ()
    if config.kind is CollectiveKind.COLLECTIVE_PERMUTE:
        one_iteration = tuple(
            (group[index], group[(index + 1) % len(group)])
            for group in canonical_groups
            for index in range(len(group))
        )
        pairs = one_iteration * config.chain_length
    return HloContractPolicy(
        name=(
            f"dependent-{config.kind.value}-g{config.group_size}-"
            f"{config.dtype}-{config.rows}x{config.width}"
        ),
        total_devices=total_devices,
        repeated_region_patterns=(r"dependent_collective_chain", r"manual_computation_body"),
        maximum_repeated_collective_group_size=config.group_size,
        expected_repeated_replica_groups=(
            ()
            if config.kind in (CollectiveKind.CONTROL, CollectiveKind.COLLECTIVE_PERMUTE)
            else canonical_groups
        ),
        expected_collectives=expectations,
        expected_collective_permute_pairs=pairs,
        # XLA collective ids index the executable device assignment.  The
        # mesh is deliberately physical-ring ordered, so preserve the exact
        # assignment needed to interpret replica groups and permute pairs.
        partition_id_to_device_id=tuple(
            device for group in canonical_groups for device in group
        ),
        # Current XLA HLO has no use_global_device_ids attribute for all-to-all;
        # its explicit replica_groups remain mandatory and exact.
        global_device_id_exempt_opcodes=(
            ("all-to-all",)
            if config.kind is CollectiveKind.ALL_TO_ALL
            else ()
        ),
        allow_full_pod_repeated_collectives=config.group_size == total_devices,
    )


def _jax_dtype(name: str) -> Any:
    import jax.numpy as jnp

    return {
        "bfloat16": jnp.bfloat16,
        "float32": jnp.float32,
        "int32": jnp.int32,
    }[name]


def _float_feedback(
    state: Any,
    member: Any,
    iteration: int,
    group_size: int,
) -> Any:
    from jax import lax
    import jax.numpy as jnp

    anchor = state.reshape(-1)[iteration % state.size].astype(jnp.float32)
    feedback = anchor / (jnp.asarray(1.0, jnp.float32) + jnp.abs(anchor))
    feedback += member.astype(jnp.float32) / jnp.asarray(256.0, jnp.float32)
    # A reduction may multiply the state by group_size.  Normalize enough to
    # keep all 75 steps finite even for the full 32-chip diagnostic.
    scale = jnp.asarray(0.5 / group_size, state.dtype)
    updated = state * scale + feedback.astype(state.dtype)
    return lax.optimization_barrier(updated)


def _integer_feedback(state: Any, member: Any, iteration: int) -> Any:
    from jax import lax
    import jax.numpy as jnp

    anchor = state.reshape(-1)[iteration % state.size]
    nonlinear = jnp.bitwise_xor(anchor, jnp.right_shift(anchor, 3))
    feedback = jnp.mod(nonlinear, jnp.asarray(65521, jnp.int32))
    feedback += member.astype(jnp.int32) * jnp.asarray(17, jnp.int32)
    updated = jnp.mod(
        state + feedback + jnp.asarray(iteration + 1, jnp.int32),
        jnp.asarray(1048573, jnp.int32),
    )
    return lax.optimization_barrier(updated)


def _feedback(
    state: Any,
    member: Any,
    iteration: int,
    dtype: str,
    group_size: int,
) -> Any:
    if dtype == "int32":
        return _integer_feedback(state, member, iteration)
    return _float_feedback(state, member, iteration, group_size)


def _chain_function(config: CollectiveChainConfig) -> Any:
    import jax
    from jax import lax
    import jax.numpy as jnp

    ring = tuple((index, (index + 1) % config.group_size) for index in range(config.group_size))

    def chain(initial: Any) -> Any:
        with jax.named_scope("dependent_collective_chain"):
            member = lax.axis_index("member")
            if config.dtype == "int32":
                state = initial + (member.astype(jnp.int32) + 1) * 17
            else:
                state = initial + (member.astype(jnp.float32) + 1).astype(
                    initial.dtype
                ) / jnp.asarray(256.0, initial.dtype)

            if config.kind is CollectiveKind.FUSED_TUPLE_ALL_REDUCE:
                auxiliary = state.astype(jnp.float32) * jnp.asarray(0.5, jnp.float32)
                for iteration in range(config.chain_length):
                    state, auxiliary = lax.psum((state, auxiliary), "member")
                    primary_anchor = state.reshape(-1)[
                        iteration % state.size
                    ].astype(jnp.float32)
                    auxiliary_anchor = auxiliary.reshape(-1)[
                        (iteration + 1) % auxiliary.size
                    ]
                    cross = primary_anchor + auxiliary_anchor
                    feedback = cross / (jnp.asarray(1.0) + jnp.abs(cross))
                    feedback += member.astype(jnp.float32) / jnp.asarray(256.0)
                    state = lax.optimization_barrier(
                        state
                        * jnp.asarray(0.5 / config.group_size, state.dtype)
                        + feedback.astype(state.dtype)
                    )
                    auxiliary = lax.optimization_barrier(
                        auxiliary
                        * jnp.asarray(0.25 / config.group_size, auxiliary.dtype)
                        + feedback
                    )
                return state, auxiliary

            for iteration in range(config.chain_length):
                if config.kind is CollectiveKind.CONTROL:
                    transformed = state
                elif config.kind is CollectiveKind.ALL_REDUCE:
                    transformed = lax.psum(state, "member")
                elif config.kind is CollectiveKind.REDUCE_SCATTER:
                    # Every destination segment is distinct and every source
                    # rank already has distinct state.  A duplicated tile lets
                    # TPU XLA replace reduce-scatter with all-reduce + slice,
                    # which is a different operation and must fail this arm.
                    if config.dtype == "int32":
                        segment_bias = (
                            jnp.arange(config.group_size, dtype=jnp.int32)
                            .reshape(config.group_size, 1, 1)
                            * jnp.asarray(257, jnp.int32)
                        )
                    else:
                        segment_bias = (
                            jnp.arange(config.group_size, dtype=jnp.float32)
                            .reshape(config.group_size, 1, 1)
                            / jnp.asarray(128.0, jnp.float32)
                        ).astype(state.dtype)
                    expanded = (state[jnp.newaxis, :, :] + segment_bias).reshape(
                        config.group_size * config.rows,
                        config.width,
                    )
                    expanded = lax.optimization_barrier(expanded)
                    transformed = lax.psum_scatter(
                        expanded,
                        "member",
                        scatter_dimension=0,
                        tiled=True,
                    )
                elif config.kind is CollectiveKind.ALL_GATHER:
                    gathered = lax.all_gather(
                        state,
                        "member",
                        axis=0,
                        tiled=False,
                    )
                    transformed = gathered.sum(axis=0, dtype=state.dtype)
                elif config.kind is CollectiveKind.COLLECTIVE_PERMUTE:
                    transformed = lax.ppermute(state, "member", ring)
                elif config.kind is CollectiveKind.ALL_TO_ALL:
                    transformed = lax.all_to_all(
                        state,
                        "member",
                        split_axis=1,
                        concat_axis=1,
                        tiled=True,
                    )
                else:  # pragma: no cover - exhaustive enum guard
                    raise AssertionError(config.kind)
                state = _feedback(
                    transformed,
                    member,
                    iteration,
                    config.dtype,
                    config.group_size,
                )
            return state

    return chain


def build_collective_chain(
    config: CollectiveChainConfig,
    groups: Sequence[Sequence[int]],
    *,
    devices: Sequence[Any] | None = None,
    enforce_hlo_contract: bool = True,
) -> CompiledCollectiveChain:
    """Compile one chain and reject it unless optimized HLO is exact."""

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    runtime_devices = tuple(jax.devices() if devices is None else devices)
    by_id = {int(device.id): device for device in runtime_devices}
    canonical_groups = _validate_groups(
        groups, config.group_size, len(runtime_devices)
    )
    if set(by_id) != set(range(len(runtime_devices))):
        raise BenchmarkValidationError(
            "benchmark currently requires contiguous global JAX device ids"
        )
    mesh_devices = np.asarray(
        [[by_id[device_id] for device_id in group] for group in canonical_groups],
        dtype=object,
    )
    mesh = Mesh(mesh_devices, ("group", "member"))
    input_sharding = NamedSharding(mesh, P("group", None))
    output_spec = P(("group", "member"), None)
    out_specs: Any = output_spec
    if config.kind is CollectiveKind.FUSED_TUPLE_ALL_REDUCE:
        out_specs = (output_spec, output_spec)
    mapped = jax.shard_map(
        _chain_function(config),
        mesh=mesh,
        in_specs=P("group", None),
        out_specs=out_specs,
        check_vma=False,
    )
    compiler_options: dict[str, Any] = {}
    if (
        config.kind is CollectiveKind.REDUCE_SCATTER
        and runtime_devices
        and {device.platform for device in runtime_devices} == {"tpu"}
    ):
        # TPU's default standalone-RS legalizer rewrites this small primitive
        # to all-reduce + slice.  The RS arm is meaningful only when optimized
        # HLO retains the requested physical operation.
        compiler_options["xla_tpu_decompose_every_reduce_scatters_hlos"] = "false"
    executable = jax.jit(mapped, compiler_options=compiler_options)
    host = np.linspace(
        -0.25,
        0.25,
        num=len(canonical_groups) * config.rows * config.width,
        dtype=np.float32,
    ).reshape(len(canonical_groups) * config.rows, config.width)
    if config.dtype == "int32":
        host = np.arange(host.size, dtype=np.int32).reshape(host.shape) % 251
    input_value = jax.device_put(host.astype(_jax_dtype(config.dtype)), input_sharding)
    started = time.perf_counter()
    compiled = executable.lower(input_value).compile()
    compile_seconds = time.perf_counter() - started
    optimized_hlo = compiled.as_text()
    policy = collective_chain_hlo_policy(
        config, canonical_groups, total_devices=len(runtime_devices)
    )
    report = lint_hlo(parse_hlo_module(optimized_hlo), policy)
    if enforce_hlo_contract:
        report.raise_for_violations()
    return CompiledCollectiveChain(
        config=config,
        compiled=compiled,
        input_value=input_value,
        optimized_hlo=optimized_hlo,
        hlo_report=report,
        compile_seconds=compile_seconds,
        compiler_options=compiler_options,
    )


def _addressable_checksum(value: Any) -> str:
    import jax
    import numpy as np

    arrays = value if isinstance(value, tuple) else (value,)
    digest = sha256()
    for array_index, array in enumerate(arrays):
        digest.update(f"array={array_index};dtype={array.dtype};shape={array.shape};".encode())
        shards = sorted(array.addressable_shards, key=lambda shard: int(shard.device.id))
        for shard in shards:
            local = np.asarray(jax.device_get(shard.data))
            if local.dtype.kind in "fc" and not np.isfinite(local).all():
                raise BenchmarkValidationError("collective chain produced non-finite output")
            digest.update(f"device={int(shard.device.id)};".encode())
            digest.update(local.tobytes(order="C"))
    return digest.hexdigest()


def benchmark_collective_chain(compiled: CompiledCollectiveChain) -> dict[str, Any]:
    """Run warmups and a profiler-free synchronized wall distribution."""

    import jax

    first = compiled.compiled(compiled.input_value)
    jax.block_until_ready(first)
    first_checksum = _addressable_checksum(first)
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
    last_checksum = _addressable_checksum(last)
    if first_checksum != last_checksum:
        raise BenchmarkValidationError(
            "identical collective-chain invocations produced different output bytes"
        )
    distribution = _distribution(samples_ms)
    return {
        "compile_seconds": compiled.compile_seconds,
        "compiler_options": dict(compiled.compiler_options),
        "config": compiled.config.to_dict(),
        "first_addressable_checksum": first_checksum,
        "hlo": compiled.hlo_report.to_dict(),
        "last_addressable_checksum": last_checksum,
        "latency": distribution.to_dict(),
        "mechanism_only": True,
        "per_collective_p50_ms": (
            None
            if compiled.config.kind is CollectiveKind.CONTROL
            else distribution.p50_ms / compiled.config.chain_length
        ),
    }
