"""Exact layer order and compact-state flow for a pipeline decoder.

The checkpoint plan owns bytes; this module turns its contiguous stage ranges
into the immutable execution schedule consumed by the decoder.  In particular,
it makes every full-DSA producer and IndexShare consumer explicit.  A shared
layer may never infer its producer from a local slot number, and a stage
boundary may carry only the compact score-ordered position vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from ...optimized.errors import PlanValidationError
from ...optimized.geometry import ExecutionPlan, StageAssignment


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class LayerExecution:
    """One transformer layer's exact global, stage, and padded-slot identity."""

    layer_id: int
    stage_id: int
    stage_slot: int
    mlp_kind: str
    indexer_kind: str
    dense_slot: int | None
    sparse_slot: int | None
    index_state_producer_layer: int

    def __post_init__(self) -> None:
        for field in ("layer_id", "stage_id", "stage_slot"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PlanValidationError(f"{field} must be a non-negative integer")
        if self.mlp_kind not in ("dense", "sparse"):
            raise PlanValidationError(f"invalid MLP kind {self.mlp_kind!r}")
        if self.indexer_kind not in ("full", "shared"):
            raise PlanValidationError(
                f"invalid indexer kind {self.indexer_kind!r}"
            )
        expected_slots = (
            self.dense_slot is not None,
            self.sparse_slot is not None,
        )
        if expected_slots != (self.mlp_kind == "dense", self.mlp_kind == "sparse"):
            raise PlanValidationError(
                "a layer must own exactly the padded slot matching its MLP kind"
            )
        for field in ("dense_slot", "sparse_slot"):
            value = getattr(self, field)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise PlanValidationError(f"{field} must be non-negative or null")
        if self.index_state_producer_layer < 0:
            raise PlanValidationError("IndexShare producer layer must be non-negative")
        if self.indexer_kind == "full":
            if self.index_state_producer_layer != self.layer_id:
                raise PlanValidationError("a full DSA layer must produce its own state")
        elif self.index_state_producer_layer >= self.layer_id:
            raise PlanValidationError(
                "an IndexShare layer must consume an earlier full-DSA state"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dense_slot": self.dense_slot,
            "index_state_producer_layer": self.index_state_producer_layer,
            "indexer_kind": self.indexer_kind,
            "layer_id": self.layer_id,
            "mlp_kind": self.mlp_kind,
            "sparse_slot": self.sparse_slot,
            "stage_id": self.stage_id,
            "stage_slot": self.stage_slot,
        }


@dataclass(frozen=True, slots=True)
class StageExecution:
    """The ordered local work and bounded padding contract for one stage."""

    assignment: StageAssignment
    layers: tuple[LayerExecution, ...]
    dense_slot_count: int
    sparse_slot_count: int
    padded_dense_slot_count: int
    padded_sparse_slot_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        if not self.layers:
            raise PlanValidationError("a decoder stage must contain layers")
        expected = tuple(
            range(
                self.assignment.layer_start,
                self.assignment.layer_end_exclusive,
            )
        )
        if tuple(layer.layer_id for layer in self.layers) != expected:
            raise PlanValidationError("stage schedule does not match its layer range")
        if any(layer.stage_id != self.assignment.stage_id for layer in self.layers):
            raise PlanValidationError("stage schedule contains a foreign layer")
        if tuple(layer.stage_slot for layer in self.layers) != tuple(
            range(len(self.layers))
        ):
            raise PlanValidationError("stage-local layer slots are not contiguous")
        observed_dense = sum(layer.mlp_kind == "dense" for layer in self.layers)
        observed_sparse = sum(layer.mlp_kind == "sparse" for layer in self.layers)
        if (self.dense_slot_count, self.sparse_slot_count) != (
            observed_dense,
            observed_sparse,
        ):
            raise PlanValidationError("stage MLP slot counts do not reconcile")
        if (
            self.padded_dense_slot_count < self.dense_slot_count
            or self.padded_sparse_slot_count < self.sparse_slot_count
        ):
            raise PlanValidationError("padded stage slot count is too small")

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def incoming_index_state(self) -> bool:
        first = self.layers[0]
        return (
            first.indexer_kind == "shared"
            and first.index_state_producer_layer < self.assignment.layer_start
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment": self.assignment.to_dict(),
            "dense_slot_count": self.dense_slot_count,
            "incoming_index_state": self.incoming_index_state,
            "layers": [layer.to_dict() for layer in self.layers],
            "padded_dense_slot_count": self.padded_dense_slot_count,
            "padded_sparse_slot_count": self.padded_sparse_slot_count,
            "sparse_slot_count": self.sparse_slot_count,
        }


@dataclass(frozen=True, slots=True)
class IndexShareTransfer:
    """One required compact-state crossing at a stage boundary."""

    boundary_layer: int
    producer_layer: int
    source_stage: int
    target_stage: int
    positions_shape: tuple[int, int]
    positions_dtype: str = "int32"

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions_shape", tuple(self.positions_shape))
        if self.target_stage != self.source_stage + 1:
            raise PlanValidationError("IndexShare transfer must cross one stage edge")
        if self.producer_layer >= self.boundary_layer:
            raise PlanValidationError("IndexShare transfer producer is not earlier")
        if self.positions_shape[0] != 1 or self.positions_shape[1] <= 0:
            raise PlanValidationError("IndexShare transfer must have one live row")
        if self.positions_dtype != "int32":
            raise PlanValidationError("IndexShare positions must remain int32")

    @property
    def byte_count(self) -> int:
        return self.positions_shape[0] * self.positions_shape[1] * 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_layer": self.boundary_layer,
            "byte_count": self.byte_count,
            "positions_dtype": self.positions_dtype,
            "positions_shape": list(self.positions_shape),
            "producer_layer": self.producer_layer,
            "source_stage": self.source_stage,
            "target_stage": self.target_stage,
        }


@dataclass(frozen=True, slots=True)
class PipelineSchedule:
    """Complete ordered decoder schedule derived from an immutable plan."""

    plan_hash: str
    stages: tuple[StageExecution, ...]
    index_share_transfers: tuple[IndexShareTransfer, ...]
    maximum_dense_slots: int
    maximum_sparse_slots: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(
            self, "index_share_transfers", tuple(self.index_share_transfers)
        )
        if len(self.plan_hash) != 64:
            raise PlanValidationError("schedule plan hash must be SHA-256")
        if tuple(stage.assignment.stage_id for stage in self.stages) != tuple(
            range(len(self.stages))
        ):
            raise PlanValidationError("schedule stages are not contiguous")
        layers = tuple(layer for stage in self.stages for layer in stage.layers)
        if tuple(layer.layer_id for layer in layers) != tuple(range(len(layers))):
            raise PlanValidationError("schedule does not cover layers exactly once")
        if any(
            stage.padded_dense_slot_count != self.maximum_dense_slots
            or stage.padded_sparse_slot_count != self.maximum_sparse_slots
            for stage in self.stages
        ):
            raise PlanValidationError("stage padding disagrees with schedule maxima")

    @property
    def schedule_hash(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    @property
    def layer_count(self) -> int:
        return sum(stage.layer_count for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_share_transfers": [
                transfer.to_dict() for transfer in self.index_share_transfers
            ],
            "maximum_dense_slots": self.maximum_dense_slots,
            "maximum_sparse_slots": self.maximum_sparse_slots,
            "plan_hash": self.plan_hash,
            "stages": [stage.to_dict() for stage in self.stages],
        }


def build_pipeline_schedule(plan: ExecutionPlan) -> PipelineSchedule:
    """Derive and validate every local slot and IndexShare stage crossing."""

    geometry = plan.geometry
    producer_by_layer: list[int] = []
    last_full: int | None = None
    for layer, indexer_kind in enumerate(geometry.indexer_types):
        if indexer_kind == "full":
            last_full = layer
        elif last_full is None:
            raise PlanValidationError(
                "the decoder starts with IndexShare but has no full-DSA producer"
            )
        if last_full is None:  # Kept explicit for static analyzers.
            raise PlanValidationError("missing full-DSA producer")
        if layer - last_full >= geometry.index_share_group_size:
            raise PlanValidationError(
                f"layer {layer} reuses DSA state beyond index_share_group_size"
            )
        producer_by_layer.append(last_full)

    layer_to_stage: dict[int, int] = {}
    counts: list[tuple[int, int]] = []
    for assignment in plan.stage_assignments:
        dense = sum(
            geometry.mlp_layer_types[layer] == "dense"
            for layer in range(
                assignment.layer_start, assignment.layer_end_exclusive
            )
        )
        sparse = (
            assignment.layer_end_exclusive - assignment.layer_start - dense
        )
        counts.append((dense, sparse))
        for layer in range(
            assignment.layer_start, assignment.layer_end_exclusive
        ):
            layer_to_stage[layer] = assignment.stage_id
    maximum_dense = max(dense for dense, _ in counts)
    maximum_sparse = max(sparse for _, sparse in counts)

    stages = []
    transfers = []
    for assignment, (dense_count, sparse_count) in zip(
        plan.stage_assignments, counts, strict=True
    ):
        dense_slot = 0
        sparse_slot = 0
        layers = []
        for stage_slot, layer in enumerate(
            range(assignment.layer_start, assignment.layer_end_exclusive)
        ):
            mlp_kind = geometry.mlp_layer_types[layer]
            layer_dense_slot = dense_slot if mlp_kind == "dense" else None
            layer_sparse_slot = sparse_slot if mlp_kind == "sparse" else None
            dense_slot += int(mlp_kind == "dense")
            sparse_slot += int(mlp_kind == "sparse")
            layers.append(
                LayerExecution(
                    layer_id=layer,
                    stage_id=assignment.stage_id,
                    stage_slot=stage_slot,
                    mlp_kind=mlp_kind,
                    indexer_kind=geometry.indexer_types[layer],
                    dense_slot=layer_dense_slot,
                    sparse_slot=layer_sparse_slot,
                    index_state_producer_layer=producer_by_layer[layer],
                )
            )
        stage = StageExecution(
            assignment=assignment,
            layers=tuple(layers),
            dense_slot_count=dense_count,
            sparse_slot_count=sparse_count,
            padded_dense_slot_count=maximum_dense,
            padded_sparse_slot_count=maximum_sparse,
        )
        stages.append(stage)
        if stage.incoming_index_state:
            first = stage.layers[0]
            producer_stage = layer_to_stage[first.index_state_producer_layer]
            if producer_stage != assignment.stage_id - 1:
                raise PlanValidationError(
                    "IndexShare state would skip a pipeline stage boundary"
                )
            transfers.append(
                IndexShareTransfer(
                    boundary_layer=first.layer_id,
                    producer_layer=first.index_state_producer_layer,
                    source_stage=producer_stage,
                    target_stage=assignment.stage_id,
                    positions_shape=(1, geometry.dsa_top_k),
                )
            )

    return PipelineSchedule(
        plan_hash=plan.plan_hash,
        stages=tuple(stages),
        index_share_transfers=tuple(transfers),
        maximum_dense_slots=maximum_dense,
        maximum_sparse_slots=maximum_sparse,
    )
