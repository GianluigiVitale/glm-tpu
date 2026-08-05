"""Exact contiguous byte-balanced PP8/PP16 stage assignment.

The optimizer first minimizes the maximum per-chip accounted HBM.  Under that
exact optimum it minimizes IndexShare boundary crossings, then total squared
stage HBM for deterministic balance.  It never partitions by equal layer
count and never treats aggregate pod HBM as the capacity gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from ..errors import PartitioningValidationError
from ..topology.groups import (
    LocalReplicaGroup,
    build_pp16_lp2_groups,
    build_pp8_lp4_groups,
)
from ..types import (
    ExecutionPlan,
    ModelGeometry,
    PhysicalTopology,
    PlanName,
    StageAssignment,
)
from .memory_model import MemoryPolicy, RuntimeMemory, stage_runtime_memory
from .ownership import (
    BASE_LOAD_SET,
    MTP_LOAD_SET,
    PlacementLedger,
    PlacementRecipe,
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class LayerFootprint:
    """Exact final-layout bytes for one base layer on every local slot."""

    layer_id: int
    parameter_slot_bytes: tuple[int, ...]
    scale_slot_bytes: tuple[int, ...]
    source_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameter_slot_bytes", tuple(self.parameter_slot_bytes)
        )
        object.__setattr__(self, "scale_slot_bytes", tuple(self.scale_slot_bytes))
        if self.layer_id < 0 or self.source_bytes < 0:
            raise PartitioningValidationError("invalid layer footprint identity")
        if not self.parameter_slot_bytes or len(self.parameter_slot_bytes) != len(
            self.scale_slot_bytes
        ):
            raise PartitioningValidationError(
                "layer footprint slot vectors must align"
            )
        if any(
            value < 0
            for value in self.parameter_slot_bytes + self.scale_slot_bytes
        ):
            raise PartitioningValidationError(
                "layer footprint bytes must be non-negative"
            )

    @property
    def packed_bytes(self) -> int:
        return sum(self.parameter_slot_bytes) + sum(self.scale_slot_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "packed_bytes": self.packed_bytes,
            "parameter_slot_bytes": list(self.parameter_slot_bytes),
            "scale_slot_bytes": list(self.scale_slot_bytes),
            "source_bytes": self.source_bytes,
        }


@dataclass(frozen=True, slots=True)
class StageMemoryEstimate:
    """Every memory class and free byte count for each physical stage slot."""

    stage_id: int
    process_index: int
    device_ids: tuple[int, ...]
    layer_start: int
    layer_end_exclusive: int
    parameter_slot_bytes: tuple[int, ...]
    scale_slot_bytes: tuple[int, ...]
    runtime: RuntimeMemory
    accounted_slot_bytes: tuple[int, ...]
    free_slot_bytes: tuple[int, ...]
    incoming_index_share_crossing: bool
    optional_mtp_slot_bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        for field in (
            "device_ids",
            "parameter_slot_bytes",
            "scale_slot_bytes",
            "accounted_slot_bytes",
            "free_slot_bytes",
            "optional_mtp_slot_bytes",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        size = len(self.device_ids)
        if size == 0 or any(
            len(getattr(self, field)) != size
            for field in (
                "parameter_slot_bytes",
                "scale_slot_bytes",
                "accounted_slot_bytes",
                "free_slot_bytes",
                "optional_mtp_slot_bytes",
            )
        ):
            raise PartitioningValidationError(
                "stage memory slot vectors must align with physical devices"
            )
        for slot in range(size):
            expected = (
                self.parameter_slot_bytes[slot]
                + self.scale_slot_bytes[slot]
                + self.runtime.accounted_bytes
            )
            if self.accounted_slot_bytes[slot] != expected:
                raise PartitioningValidationError(
                    "stage accounted bytes do not reconcile memory classes"
                )

    @property
    def maximum_accounted_bytes(self) -> int:
        return max(self.accounted_slot_bytes)

    @property
    def minimum_free_bytes(self) -> int:
        return min(self.free_slot_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accounted_slot_bytes": list(self.accounted_slot_bytes),
            "device_ids": list(self.device_ids),
            "free_slot_bytes": list(self.free_slot_bytes),
            "incoming_index_share_crossing": self.incoming_index_share_crossing,
            "layer_end_exclusive": self.layer_end_exclusive,
            "layer_start": self.layer_start,
            "maximum_accounted_bytes": self.maximum_accounted_bytes,
            "minimum_free_bytes": self.minimum_free_bytes,
            "optional_mtp_slot_bytes": list(self.optional_mtp_slot_bytes),
            "parameter_slot_bytes": list(self.parameter_slot_bytes),
            "process_index": self.process_index,
            "runtime": self.runtime.to_dict(),
            "scale_slot_bytes": list(self.scale_slot_bytes),
            "stage_id": self.stage_id,
        }


@dataclass(frozen=True, slots=True)
class PartitionedPlan:
    """Execution plan plus the exact Gate-B planning ledger."""

    execution_plan: ExecutionPlan
    memory_policy: MemoryPolicy
    source_inventory_sha256: str
    layer_footprints: tuple[LayerFootprint, ...]
    stages: tuple[StageMemoryEstimate, ...]
    index_share_crossings: tuple[int, ...]
    mtp_stage_id: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer_footprints",
            tuple(sorted(self.layer_footprints, key=lambda item: item.layer_id)),
        )
        object.__setattr__(
            self, "stages", tuple(sorted(self.stages, key=lambda item: item.stage_id))
        )
        object.__setattr__(
            self, "index_share_crossings", tuple(self.index_share_crossings)
        )
        if len(self.stages) != self.execution_plan.pipeline_stages:
            raise PartitioningValidationError(
                "partition memory stage count disagrees with execution plan"
            )
        if tuple(stage.stage_id for stage in self.stages) != tuple(
            range(len(self.stages))
        ):
            raise PartitioningValidationError("partition stages are not contiguous")
        if not 0 <= self.mtp_stage_id < len(self.stages):
            raise PartitioningValidationError("MTP stage id is invalid")
        for estimate, assignment in zip(
            self.stages, self.execution_plan.stage_assignments
        ):
            if (
                estimate.layer_start != assignment.layer_start
                or estimate.layer_end_exclusive != assignment.layer_end_exclusive
                or estimate.device_ids != assignment.device_ids
                or max(estimate.parameter_slot_bytes)
                != assignment.persistent_weight_bytes
                or max(estimate.scale_slot_bytes) != assignment.fp8_scale_bytes
                or estimate.runtime.kv_cache_bytes
                != assignment.kv_bytes_at_target_context
                or estimate.runtime.dsa_state_bytes != assignment.dsa_state_bytes
                or estimate.runtime.temporary_floor_bytes
                != assignment.temporary_bytes
                or estimate.runtime.reserved_overlay_bytes
                != assignment.reserved_overlay_bytes
            ):
                raise PartitioningValidationError(
                    f"stage {estimate.stage_id} estimate disagrees with execution plan"
                )

    @property
    def capacity_feasible(self) -> bool:
        return all(stage.minimum_free_bytes >= 0 for stage in self.stages)

    @property
    def promotion_memory_proven(self) -> bool:
        return (
            self.capacity_feasible
            and self.memory_policy.full_decoder_overlay_measured
        )

    @property
    def plan_manifest_sha256(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity_feasible": self.capacity_feasible,
            "execution_plan": self.execution_plan.to_dict(),
            "execution_plan_sha256": self.execution_plan.plan_hash,
            "index_share_crossings": list(self.index_share_crossings),
            "layer_footprints": [item.to_dict() for item in self.layer_footprints],
            "memory_policy": self.memory_policy.to_dict(),
            "memory_policy_sha256": self.memory_policy.policy_sha256,
            "mtp_stage_id": self.mtp_stage_id,
            "promotion_memory_proven": self.promotion_memory_proven,
            "source_inventory_sha256": self.source_inventory_sha256,
            "stages": [stage.to_dict() for stage in self.stages],
        }


def build_layer_footprints(
    ledger: PlacementLedger, geometry: ModelGeometry
) -> tuple[LayerFootprint, ...]:
    """Reduce every base placement recipe to exact per-slot layer bytes."""

    local_size = ledger.local_parallel_size
    parameters = [[0] * local_size for _ in range(geometry.num_layers)]
    scales = [[0] * local_size for _ in range(geometry.num_layers)]
    source = [0] * geometry.num_layers
    for recipe in ledger.recipes:
        if recipe.load_set != BASE_LOAD_SET or recipe.layer_id is None:
            continue
        if not 0 <= recipe.layer_id < geometry.num_layers:
            raise PartitioningValidationError(
                f"base load set contains invalid layer {recipe.layer_id}"
            )
        source[recipe.layer_id] += recipe.source_byte_count
        target = scales if recipe.value_class == "fp8_scale" else parameters
        for shard in recipe.shards:
            target[recipe.layer_id][shard.device_slot] += shard.byte_count
    if any(value == 0 for value in source):
        raise PartitioningValidationError(
            "every base layer must have a non-empty placement footprint"
        )
    return tuple(
        LayerFootprint(
            layer_id=layer,
            parameter_slot_bytes=tuple(parameters[layer]),
            scale_slot_bytes=tuple(scales[layer]),
            source_bytes=source[layer],
        )
        for layer in range(geometry.num_layers)
    )


def _global_slot_bytes(
    ledger: PlacementLedger, *, stage_id: int, stage_count: int, value_class: str
) -> tuple[int, ...]:
    totals = [0] * ledger.local_parallel_size
    for recipe in ledger.recipes:
        if (
            recipe.load_set != BASE_LOAD_SET
            or recipe.layer_id is not None
            or recipe.value_class != value_class
        ):
            continue
        target_stage = 0 if recipe.source_name == "model.embed_tokens.weight" else stage_count - 1
        if recipe.source_name not in (
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
        ):
            raise PartitioningValidationError(
                f"global placement rule missing for {recipe.source_name!r}"
            )
        if target_stage == stage_id:
            for shard in recipe.shards:
                totals[shard.device_slot] += shard.byte_count
    return tuple(totals)


def _mtp_slot_bytes(ledger: PlacementLedger) -> tuple[int, ...]:
    totals = [0] * ledger.local_parallel_size
    for recipe in ledger.recipes:
        if recipe.load_set != MTP_LOAD_SET:
            continue
        for shard in recipe.shards:
            totals[shard.device_slot] += shard.byte_count
    return tuple(totals)


def _prefix(values: tuple[LayerFootprint, ...], field: str) -> list[list[int]]:
    local_size = len(values[0].parameter_slot_bytes)
    result = [[0] * (len(values) + 1) for _ in range(local_size)]
    for layer, footprint in enumerate(values):
        slots = getattr(footprint, field)
        for slot in range(local_size):
            result[slot][layer + 1] = result[slot][layer] + slots[slot]
    return result


def _segment_slots(prefix: list[list[int]], start: int, end: int) -> tuple[int, ...]:
    return tuple(values[end] - values[start] for values in prefix)


def _groups_for_plan(
    name: PlanName, topology: PhysicalTopology
) -> tuple[LocalReplicaGroup, ...]:
    if name is PlanName.PP8_LP4:
        return build_pp8_lp4_groups(topology)
    if name is PlanName.PP16_LP2:
        return build_pp16_lp2_groups(topology)
    raise PartitioningValidationError(
        "byte-balanced pipeline assignment supports PP8/PP16 only"
    )


def build_pipeline_plan(
    *,
    name: PlanName,
    geometry: ModelGeometry,
    topology: PhysicalTopology,
    ledger: PlacementLedger,
    memory_policy: MemoryPolicy,
    mtp_stage_id: int = 0,
) -> PartitionedPlan:
    """Choose exact contiguous ranges and produce a complete per-chip plan."""

    groups = _groups_for_plan(name, topology)
    stage_count = len(groups)
    local_size = len(groups[0].device_ids)
    if ledger.local_parallel_size != local_size:
        raise PartitioningValidationError(
            "placement ledger local size disagrees with execution plan"
        )
    if memory_policy.target_context_length > geometry.max_position_embeddings:
        raise PartitioningValidationError(
            "memory target exceeds geometry context limit"
        )
    if geometry.num_layers < stage_count:
        raise PartitioningValidationError(
            "pipeline requires at least one base layer per stage"
        )
    footprints = build_layer_footprints(ledger, geometry)
    parameter_prefix = _prefix(footprints, "parameter_slot_bytes")
    scale_prefix = _prefix(footprints, "scale_slot_bytes")
    global_parameter = tuple(
        _global_slot_bytes(
            ledger,
            stage_id=stage,
            stage_count=stage_count,
            value_class="parameter",
        )
        for stage in range(stage_count)
    )
    global_scale = tuple(
        _global_slot_bytes(
            ledger,
            stage_id=stage,
            stage_count=stage_count,
            value_class="fp8_scale",
        )
        for stage in range(stage_count)
    )

    def segment_cost(stage: int, start: int, end: int) -> int:
        parameter = _segment_slots(parameter_prefix, start, end)
        scales = _segment_slots(scale_prefix, start, end)
        runtime = stage_runtime_memory(
            geometry,
            layer_start=start,
            layer_end_exclusive=end,
            local_parallel_size=local_size,
            policy=memory_policy,
        )
        return max(
            parameter[slot]
            + global_parameter[stage][slot]
            + scales[slot]
            + global_scale[stage][slot]
            + runtime.accounted_bytes
            for slot in range(local_size)
        )

    # Phase one: exact minimax HBM.
    infinity = 1 << 120
    minimax = [[infinity] * (geometry.num_layers + 1) for _ in range(stage_count + 1)]
    minimax[0][0] = 0
    for used in range(1, stage_count + 1):
        minimum_end = used
        maximum_end = geometry.num_layers - (stage_count - used)
        for end in range(minimum_end, maximum_end + 1):
            for start in range(used - 1, end):
                previous = minimax[used - 1][start]
                if previous == infinity:
                    continue
                candidate = max(previous, segment_cost(used - 1, start, end))
                if candidate < minimax[used][end]:
                    minimax[used][end] = candidate
    optimum = minimax[stage_count][geometry.num_layers]
    if optimum == infinity:
        raise PartitioningValidationError("no contiguous pipeline partition exists")

    # Phase two: under the exact minimax ceiling, prefer intact IndexShare
    # groups, then lower total squared HBM, then stable earlier boundaries.
    # value = (crossing_count, squared_sum, boundaries)
    best: list[list[tuple[int, int, tuple[int, ...]] | None]] = [
        [None] * (geometry.num_layers + 1) for _ in range(stage_count + 1)
    ]
    best[0][0] = (0, 0, ())
    for used in range(1, stage_count + 1):
        minimum_end = used
        maximum_end = geometry.num_layers - (stage_count - used)
        for end in range(minimum_end, maximum_end + 1):
            candidates = []
            for start in range(used - 1, end):
                previous = best[used - 1][start]
                if previous is None:
                    continue
                cost = segment_cost(used - 1, start, end)
                if cost > optimum:
                    continue
                crossing = int(
                    used > 1 and geometry.indexer_types[start] == "shared"
                )
                candidates.append(
                    (
                        previous[0] + crossing,
                        previous[1] + cost * cost,
                        previous[2] + (end,),
                    )
                )
            if candidates:
                best[used][end] = min(candidates)
    selected = best[stage_count][geometry.num_layers]
    if selected is None:
        raise PartitioningValidationError(
            "failed to reconstruct minimax pipeline partition"
        )
    ends = selected[2]
    starts = (0,) + ends[:-1]
    optional_mtp = _mtp_slot_bytes(ledger)
    estimates = []
    assignments = []
    crossings = []
    for stage, (start, end, group) in enumerate(zip(starts, ends, groups)):
        layer_parameter = _segment_slots(parameter_prefix, start, end)
        layer_scale = _segment_slots(scale_prefix, start, end)
        parameter = tuple(
            layer_parameter[slot] + global_parameter[stage][slot]
            for slot in range(local_size)
        )
        scales = tuple(
            layer_scale[slot] + global_scale[stage][slot]
            for slot in range(local_size)
        )
        runtime = stage_runtime_memory(
            geometry,
            layer_start=start,
            layer_end_exclusive=end,
            local_parallel_size=local_size,
            policy=memory_policy,
        )
        accounted = tuple(
            parameter[slot] + scales[slot] + runtime.accounted_bytes
            for slot in range(local_size)
        )
        free = tuple(
            memory_policy.hbm_limit_bytes - value for value in accounted
        )
        incoming_crossing = bool(
            stage > 0 and geometry.indexer_types[start] == "shared"
        )
        if incoming_crossing:
            crossings.append(start)
        estimate = StageMemoryEstimate(
            stage_id=stage,
            process_index=group.process_index,
            device_ids=group.device_ids,
            layer_start=start,
            layer_end_exclusive=end,
            parameter_slot_bytes=parameter,
            scale_slot_bytes=scales,
            runtime=runtime,
            accounted_slot_bytes=accounted,
            free_slot_bytes=free,
            incoming_index_share_crossing=incoming_crossing,
            optional_mtp_slot_bytes=(
                optional_mtp if stage == mtp_stage_id else (0,) * local_size
            ),
        )
        estimates.append(estimate)
        assignments.append(
            StageAssignment(
                stage_id=stage,
                process_index=group.process_index,
                device_ids=group.device_ids,
                layer_start=start,
                layer_end_exclusive=end,
                persistent_weight_bytes=max(parameter),
                fp8_scale_bytes=max(scales),
                kv_bytes_at_target_context=runtime.kv_cache_bytes,
                dsa_state_bytes=runtime.dsa_state_bytes,
                temporary_bytes=runtime.temporary_floor_bytes,
                reserved_overlay_bytes=runtime.reserved_overlay_bytes,
            )
        )
    if any(stage.minimum_free_bytes < 0 for stage in estimates):
        worst = min(estimates, key=lambda item: item.minimum_free_bytes)
        raise PartitioningValidationError(
            f"{name.value} exceeds per-chip HBM at stage {worst.stage_id} by "
            f"{-worst.minimum_free_bytes} bytes"
        )
    execution_plan = ExecutionPlan(
        name=name,
        geometry=geometry,
        topology=topology,
        target_context_length=memory_policy.target_context_length,
        pipeline_stages=stage_count,
        local_parallel_size=local_size,
        stage_assignments=tuple(assignments),
        local_mesh_shape=(local_size,),
        residual_layout="stage_local_replicated",
        expert_layout=f"complete_expert_identity_lp{local_size}",
        kv_layout=f"stage_layer_context_sharded_lp{local_size}",
        transport="collective_permute_stage_ring",
    )
    return PartitionedPlan(
        execution_plan=execution_plan,
        memory_policy=memory_policy,
        source_inventory_sha256=ledger.source_inventory_sha256,
        layer_footprints=footprints,
        stages=tuple(estimates),
        index_share_crossings=tuple(crossings),
        mtp_stage_id=mtp_stage_id,
    )
