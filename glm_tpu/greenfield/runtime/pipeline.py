"""One-live-row device pipeline used as the complete-decoder control shell.

Unlike the closed synthetic transport benchmark, this shell models decoder
control flow: exactly one stage group is active at a time, each active group
executes one topology-local reduction, and only the live residual plus compact
IndexShare metadata advances to the next stage.  Real layer kernels replace
the marked local transform without changing the transport/state boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

from ...optimized.errors import PlanValidationError
from ...optimized.hlo_contract import parse_hlo_module


@dataclass(frozen=True, slots=True)
class PipelineSkeletonConfig:
    stage_count: int
    local_parallel_size: int
    hidden_width: int
    selected_width: int
    residual_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for field in (
            "stage_count",
            "local_parallel_size",
            "hidden_width",
            "selected_width",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PlanValidationError(f"{field} must be a positive integer")
        if self.residual_dtype not in ("bfloat16", "float32"):
            raise PlanValidationError("pipeline residual dtype must be bf16 or f32")

    @property
    def total_devices(self) -> int:
        return self.stage_count * self.local_parallel_size

    @property
    def metadata_width(self) -> int:
        # selected positions, valid count, producer layer, visited mask, active.
        return self.selected_width + 4


@dataclass(frozen=True, slots=True)
class CompiledPipelineSkeleton:
    config: PipelineSkeletonConfig
    compiled: Any
    residual: Any
    metadata: Any
    optimized_hlo: str
    hlo_contract: dict[str, Any]
    compile_seconds: float


def _canonical_groups(
    groups: Sequence[Sequence[int]], config: PipelineSkeletonConfig
) -> tuple[tuple[int, ...], ...]:
    value = tuple(tuple(int(rank) for rank in group) for group in groups)
    expected = set(range(config.total_devices))
    flat = [rank for group in value for rank in group]
    if (
        len(value) != config.stage_count
        or any(len(group) != config.local_parallel_size for group in value)
        or len(flat) != len(set(flat))
        or set(flat) != expected
    ):
        raise PlanValidationError(
            "pipeline stage groups must partition every logical device exactly once"
        )
    return value


def _canonical_pairs(
    pairs: Sequence[Sequence[int]],
    groups: tuple[tuple[int, ...], ...],
    config: PipelineSkeletonConfig,
) -> tuple[tuple[int, int], ...]:
    value = tuple((int(pair[0]), int(pair[1])) for pair in pairs if len(pair) == 2)
    if len(value) != config.total_devices or len(value) != len(pairs):
        raise PlanValidationError("pipeline requires one transport pair per device")
    expected = tuple(
        (groups[stage][slot], groups[(stage + 1) % config.stage_count][slot])
        for stage in range(config.stage_count)
        for slot in range(config.local_parallel_size)
    )
    if set(value) != set(expected):
        raise PlanValidationError(
            "pipeline transport pairs do not preserve stage-local lanes"
        )
    return value


def _partition_maps(
    groups: tuple[tuple[int, ...], ...], config: PipelineSkeletonConfig
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    stage = [-1] * config.total_devices
    slot = [-1] * config.total_devices
    for stage_id, group in enumerate(groups):
        for slot_id, rank in enumerate(group):
            stage[rank] = stage_id
            slot[rank] = slot_id
    return tuple(stage), tuple(slot)


def validate_pipeline_skeleton_hlo(
    optimized_hlo: str,
    *,
    config: PipelineSkeletonConfig,
    groups: Sequence[Sequence[int]],
    pairs: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Require only local reductions and two exact lane transfers per stage."""

    canonical_groups = _canonical_groups(groups, config)
    canonical_pairs = _canonical_pairs(pairs, canonical_groups, config)
    module = parse_hlo_module(optimized_hlo)
    collectives = module.collectives
    reductions = [item for item in collectives if item.opcode == "all-reduce"]
    permutes = [item for item in collectives if item.opcode == "collective-permute"]
    violations = []
    if len(reductions) != config.stage_count:
        violations.append(
            f"expected {config.stage_count} local reductions, found {len(reductions)}"
        )
    if len(permutes) != 2 * config.stage_count:
        violations.append(
            f"expected {2 * config.stage_count} live-payload permutes, found {len(permutes)}"
        )
    expected_groups = tuple(sorted(tuple(sorted(group)) for group in canonical_groups))
    for reduction in reductions:
        observed_groups = tuple(
            sorted(tuple(sorted(group)) for group in reduction.replica_groups)
        )
        if observed_groups != expected_groups:
            violations.append(
                f"reduction {reduction.name} escaped exact local groups"
            )
        if reduction.maximum_group_size != config.local_parallel_size:
            violations.append(
                f"reduction {reduction.name} group size is not local"
            )
    expected_pair_multiset = sorted(canonical_pairs * (2 * config.stage_count))
    observed_pair_multiset = sorted(
        pair for item in permutes for pair in item.source_target_pairs
    )
    if observed_pair_multiset != expected_pair_multiset:
        violations.append("collective-permute lane pairs/counts drifted")
    forbidden = [
        item.name
        for item in collectives
        if item.opcode not in ("all-reduce", "collective-permute")
    ]
    if forbidden:
        violations.append(f"pipeline has forbidden collectives: {forbidden}")
    for instruction in module.instructions:
        for shape in instruction.operand_shapes + instruction.result_shapes:
            if shape.dimensions in (
                (32, config.hidden_width),
                (32, 1, config.hidden_width),
            ):
                violations.append(
                    f"pipeline contains forbidden batch/full-pod hidden shape at {instruction.name}"
                )
    return {
        "all_reduce_count": len(reductions),
        "collective_count": len(collectives),
        "collective_permute_count": len(permutes),
        "collectives": [item.to_dict() for item in collectives],
        "module_name": module.name,
        "num_partitions": module.num_partitions,
        "passed": not violations,
        "violations": violations,
    }


def build_pipeline_skeleton(
    config: PipelineSkeletonConfig,
    groups: Sequence[Sequence[int]],
    pairs: Sequence[Sequence[int]],
    last_full_indexer_layer_by_stage: Sequence[int],
    *,
    devices: Sequence[Any] | None = None,
    enforce_hlo_contract: bool = True,
) -> CompiledPipelineSkeleton:
    """Compile the one-row control shell over a global logical device mesh."""

    import jax
    from jax import lax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    canonical_groups = _canonical_groups(groups, config)
    canonical_pairs = _canonical_pairs(pairs, canonical_groups, config)
    if (
        len(last_full_indexer_layer_by_stage) != config.stage_count
        or any(
            not isinstance(layer, int) or isinstance(layer, bool) or layer < 0
            for layer in last_full_indexer_layer_by_stage
        )
    ):
        raise PlanValidationError(
            "every pipeline stage requires its last full-indexer layer"
        )
    runtime_devices = tuple(jax.devices() if devices is None else devices)
    if len(runtime_devices) != config.total_devices:
        raise PlanValidationError(
            f"pipeline expected {config.total_devices} devices, found {len(runtime_devices)}"
        )
    stage_by_rank, _ = _partition_maps(canonical_groups, config)
    mesh = Mesh(np.asarray(runtime_devices, dtype=object), ("device",))
    sharding = NamedSharding(mesh, P("device", None, None))
    stage_map = jnp.asarray(stage_by_rank, dtype=jnp.int32)
    producer_map = jnp.asarray(last_full_indexer_layer_by_stage, dtype=jnp.int32)
    axis_groups = tuple(tuple(group) for group in canonical_groups)

    def mapped(local_residual: Any, local_metadata: Any) -> tuple[Any, Any]:
        rank = lax.axis_index("device")
        stage_id = stage_map[rank]
        residual = local_residual[0]
        metadata = local_metadata[0]
        active_index = config.metadata_width - 1
        visited_index = config.metadata_width - 2
        producer_index = config.metadata_width - 3
        for hop in range(config.stage_count):
            active = metadata[0, active_index] == jnp.int32(1)
            should_execute = active & (stage_id == jnp.int32(hop))

            def execute_stage(values: tuple[Any, Any]) -> tuple[Any, Any]:
                value, state = values
                partial = (
                    value + jnp.asarray(hop + 1, value.dtype)
                ) / jnp.asarray(config.local_parallel_size, value.dtype)
                value = lax.psum(
                    partial,
                    axis_name="device",
                    axis_index_groups=axis_groups,
                )
                state = state.at[0, producer_index].set(producer_map[hop])
                state = state.at[0, visited_index].set(
                    jnp.bitwise_or(
                        state[0, visited_index], jnp.int32(1 << hop)
                    )
                )
                return value, state

            residual, metadata = lax.cond(
                should_execute,
                execute_stage,
                lambda values: values,
                (residual, metadata),
            )
            residual = lax.ppermute(residual, "device", canonical_pairs)
            metadata = lax.ppermute(metadata, "device", canonical_pairs)
        return residual[None, ...], metadata[None, ...]

    execute = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=(P("device", None, None), P("device", None, None)),
        out_specs=(P("device", None, None), P("device", None, None)),
        check_vma=False,
    )
    residual_dtype = (
        jnp.bfloat16 if config.residual_dtype == "bfloat16" else jnp.float32
    )
    residual_host = np.zeros(
        (config.total_devices, 1, config.hidden_width), dtype=np.float32
    )
    metadata_host = np.full(
        (config.total_devices, 1, config.metadata_width), -1, dtype=np.int32
    )
    metadata_host[..., config.selected_width] = 0
    metadata_host[..., config.selected_width + 1] = -1
    metadata_host[..., config.selected_width + 2] = 0
    metadata_host[..., config.selected_width + 3] = 0
    selected = np.arange(config.selected_width - 1, -1, -1, dtype=np.int32)
    for rank in canonical_groups[0]:
        residual_host[rank, 0] = np.linspace(
            -0.5, 0.5, config.hidden_width, dtype=np.float32
        )
        metadata_host[rank, 0, : config.selected_width] = selected
        metadata_host[rank, 0, config.selected_width] = config.selected_width
        metadata_host[rank, 0, config.selected_width + 1] = 0
        metadata_host[rank, 0, config.selected_width + 2] = 0
        metadata_host[rank, 0, config.selected_width + 3] = 1
    residual = jax.device_put(residual_host.astype(residual_dtype), sharding)
    metadata = jax.device_put(metadata_host, sharding)
    started = time.perf_counter()
    compiled = jax.jit(execute).lower(residual, metadata).compile()
    compile_seconds = time.perf_counter() - started
    optimized_hlo = compiled.as_text()
    contract = validate_pipeline_skeleton_hlo(
        optimized_hlo,
        config=config,
        groups=canonical_groups,
        pairs=canonical_pairs,
    )
    if enforce_hlo_contract and not contract["passed"]:
        raise PlanValidationError(
            f"pipeline skeleton HLO rejected: {contract['violations']}"
        )
    return CompiledPipelineSkeleton(
        config=config,
        compiled=compiled,
        residual=residual,
        metadata=metadata,
        optimized_hlo=optimized_hlo,
        hlo_contract=contract,
        compile_seconds=compile_seconds,
    )
