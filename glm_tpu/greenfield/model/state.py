"""Paged decoder-state shapes and exact per-chip byte accounting."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from ...optimized.errors import PlanValidationError
from ...optimized.geometry import ExecutionPlan
from .schedule import PipelineSchedule


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


@dataclass(frozen=True, slots=True)
class StageStateLayout:
    """Actual and globally padded state owned by one chip in one stage."""

    stage_id: int
    layer_count: int
    full_indexer_count: int
    physical_page_count: int
    local_rows_per_page: int
    packed_kv_width: int
    index_key_width: int
    selected_width: int
    padded_layer_count: int
    padded_full_indexer_count: int
    cache_element_bytes: int = 2
    selected_buffers: int = 2

    def __post_init__(self) -> None:
        for field in (
            "layer_count",
            "physical_page_count",
            "local_rows_per_page",
            "packed_kv_width",
            "index_key_width",
            "selected_width",
            "padded_layer_count",
            "padded_full_indexer_count",
            "cache_element_bytes",
            "selected_buffers",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PlanValidationError(f"{field} must be a positive integer")
        for field in ("stage_id", "full_indexer_count"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PlanValidationError(f"{field} must be non-negative")
        if self.padded_layer_count < self.layer_count:
            raise PlanValidationError("padded KV layer count is too small")
        if self.padded_full_indexer_count < self.full_indexer_count:
            raise PlanValidationError("padded indexer count is too small")

    @property
    def kv_cache_shape(self) -> tuple[int, int, int, int]:
        return (
            self.layer_count,
            self.physical_page_count,
            self.local_rows_per_page,
            self.packed_kv_width,
        )

    @property
    def padded_kv_cache_shape(self) -> tuple[int, int, int, int]:
        return (
            self.padded_layer_count,
            self.physical_page_count,
            self.local_rows_per_page,
            self.packed_kv_width,
        )

    @property
    def indexer_cache_shape(self) -> tuple[int, int, int, int]:
        return (
            self.full_indexer_count,
            self.physical_page_count,
            self.local_rows_per_page,
            self.index_key_width,
        )

    @property
    def padded_indexer_cache_shape(self) -> tuple[int, int, int, int]:
        return (
            self.padded_full_indexer_count,
            self.physical_page_count,
            self.local_rows_per_page,
            self.index_key_width,
        )

    @property
    def kv_cache_bytes(self) -> int:
        return (
            self.layer_count
            * self.physical_page_count
            * self.local_rows_per_page
            * self.packed_kv_width
            * self.cache_element_bytes
        )

    @property
    def indexer_cache_bytes(self) -> int:
        return (
            self.full_indexer_count
            * self.physical_page_count
            * self.local_rows_per_page
            * self.index_key_width
            * self.cache_element_bytes
        )

    @property
    def selected_index_bytes(self) -> int:
        return self.selected_width * 4 * self.selected_buffers

    @property
    def transport_bytes(self) -> int:
        return self.selected_width * 4 + 6144 * self.cache_element_bytes

    @property
    def actual_bytes(self) -> int:
        return (
            self.kv_cache_bytes
            + self.indexer_cache_bytes
            + self.selected_index_bytes
            + self.transport_bytes
        )

    @property
    def padded_bytes(self) -> int:
        kv = (
            self.padded_layer_count
            * self.physical_page_count
            * self.local_rows_per_page
            * self.packed_kv_width
            * self.cache_element_bytes
        )
        indexer = (
            self.padded_full_indexer_count
            * self.physical_page_count
            * self.local_rows_per_page
            * self.index_key_width
            * self.cache_element_bytes
        )
        return kv + indexer + self.selected_index_bytes + self.transport_bytes

    @property
    def padding_bytes(self) -> int:
        return self.padded_bytes - self.actual_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual_bytes": self.actual_bytes,
            "cache_element_bytes": self.cache_element_bytes,
            "full_indexer_count": self.full_indexer_count,
            "index_key_width": self.index_key_width,
            "indexer_cache_bytes": self.indexer_cache_bytes,
            "indexer_cache_shape": list(self.indexer_cache_shape),
            "kv_cache_bytes": self.kv_cache_bytes,
            "kv_cache_shape": list(self.kv_cache_shape),
            "layer_count": self.layer_count,
            "local_rows_per_page": self.local_rows_per_page,
            "packed_kv_width": self.packed_kv_width,
            "padded_bytes": self.padded_bytes,
            "padded_full_indexer_count": self.padded_full_indexer_count,
            "padded_indexer_cache_shape": list(
                self.padded_indexer_cache_shape
            ),
            "padded_kv_cache_shape": list(self.padded_kv_cache_shape),
            "padded_layer_count": self.padded_layer_count,
            "padding_bytes": self.padding_bytes,
            "physical_page_count": self.physical_page_count,
            "selected_buffers": self.selected_buffers,
            "selected_index_bytes": self.selected_index_bytes,
            "selected_width": self.selected_width,
            "stage_id": self.stage_id,
            "transport_bytes": self.transport_bytes,
        }


@dataclass(frozen=True, slots=True)
class DecoderStateLayout:
    """Complete short/long-context state geometry for one execution plan."""

    plan_hash: str
    schedule_hash: str
    context_capacity: int
    logical_page_size: int
    local_parallel_size: int
    stages: tuple[StageStateLayout, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        if len(self.plan_hash) != 64 or len(self.schedule_hash) != 64:
            raise PlanValidationError("state layout hashes must be SHA-256")
        if self.context_capacity <= 0 or self.logical_page_size <= 0:
            raise PlanValidationError("state context/page sizes must be positive")
        if self.logical_page_size % self.local_parallel_size:
            raise PlanValidationError("state page does not divide over local stage")
        if tuple(stage.stage_id for stage in self.stages) != tuple(
            range(len(self.stages))
        ):
            raise PlanValidationError("state stages are not contiguous")
        padded_shapes = {
            (stage.padded_kv_cache_shape, stage.padded_indexer_cache_shape)
            for stage in self.stages
        }
        if len(padded_shapes) != 1:
            raise PlanValidationError("global padded state shapes disagree by stage")

    @property
    def state_layout_hash(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @property
    def maximum_padded_bytes_per_chip(self) -> int:
        return max(stage.padded_bytes for stage in self.stages)

    @property
    def padded_state_shape_per_device(self) -> dict[str, tuple[int, ...]]:
        stage = self.stages[0]
        return {
            "index_keys": stage.padded_indexer_cache_shape,
            "kv": stage.padded_kv_cache_shape,
            "selected_positions": (1, stage.selected_width),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_capacity": self.context_capacity,
            "local_parallel_size": self.local_parallel_size,
            "logical_page_size": self.logical_page_size,
            "maximum_padded_bytes_per_chip": self.maximum_padded_bytes_per_chip,
            "plan_hash": self.plan_hash,
            "schedule_hash": self.schedule_hash,
            "stages": [stage.to_dict() for stage in self.stages],
        }


def build_decoder_state_layout(
    plan: ExecutionPlan,
    schedule: PipelineSchedule,
    *,
    context_capacity: int,
    logical_page_size: int = 512,
    packed_kv_width: int = 640,
    cache_element_bytes: int = 2,
    selected_buffers: int = 2,
) -> DecoderStateLayout:
    """Build exact actual/padded state and reconcile the target-context plan."""

    if schedule.plan_hash != plan.plan_hash:
        raise PlanValidationError("decoder state schedule belongs to another plan")
    if not isinstance(context_capacity, int) or isinstance(context_capacity, bool) or not (
        0 < context_capacity <= plan.geometry.max_position_embeddings
    ):
        raise PlanValidationError("decoder context capacity is invalid")
    if logical_page_size % plan.local_parallel_size:
        raise PlanValidationError("logical page must divide over local stage")
    local_rows = logical_page_size // plan.local_parallel_size
    page_count = _ceil_div(context_capacity, logical_page_size)
    maximum_layers = max(stage.layer_count for stage in schedule.stages)
    full_counts = tuple(
        sum(layer.indexer_kind == "full" for layer in stage.layers)
        for stage in schedule.stages
    )
    maximum_full = max(full_counts)
    layouts = tuple(
        StageStateLayout(
            stage_id=stage.assignment.stage_id,
            layer_count=stage.layer_count,
            full_indexer_count=full_count,
            physical_page_count=page_count,
            local_rows_per_page=local_rows,
            packed_kv_width=packed_kv_width,
            index_key_width=plan.geometry.dsa_indexer_head_dim,
            selected_width=plan.geometry.dsa_top_k,
            padded_layer_count=maximum_layers,
            padded_full_indexer_count=maximum_full,
            cache_element_bytes=cache_element_bytes,
            selected_buffers=selected_buffers,
        )
        for stage, full_count in zip(schedule.stages, full_counts, strict=True)
    )
    result = DecoderStateLayout(
        plan_hash=plan.plan_hash,
        schedule_hash=schedule.schedule_hash,
        context_capacity=context_capacity,
        logical_page_size=logical_page_size,
        local_parallel_size=plan.local_parallel_size,
        stages=layouts,
    )
    if context_capacity == plan.target_context_length:
        for assignment, layout in zip(
            plan.stage_assignments, result.stages, strict=True
        ):
            if assignment.kv_bytes_at_target_context != layout.kv_cache_bytes:
                raise PlanValidationError(
                    f"stage {assignment.stage_id} target KV bytes drifted"
                )
            if assignment.dsa_state_bytes != (
                layout.indexer_cache_bytes
                + layout.selected_index_bytes
                + layout.transport_bytes
            ):
                raise PlanValidationError(
                    f"stage {assignment.stage_id} target DSA bytes drifted"
                )
    return result
