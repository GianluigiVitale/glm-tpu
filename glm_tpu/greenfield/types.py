"""Immutable contracts shared by every greenfield execution plan.

These types are deliberately independent of JAX.  Discovery code will turn
runtime JAX devices into :class:`PhysicalTopology`; compilation code will
consume an :class:`ExecutionPlan`.  Keeping the boundary pure makes the
configuration testable without initializing a TPU and gives every run a
stable, content-addressed plan fingerprint.

Python's process-randomized ``hash()`` is never provenance.  ``plan_hash`` and
``topology_hash`` are SHA-256 digests of canonical JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from hashlib import sha256
import json
from operator import mul
from typing import Any, Mapping, Sequence

from .errors import (
    GeometryValidationError,
    PlanValidationError,
    TopologyValidationError,
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object, field: str, error: type[ValueError]) -> int:
    if not _is_int(value) or value <= 0:
        raise error(f"{field} must be a positive integer, got {value!r}")
    return value


def _nonnegative_int(
    value: object, field: str, error: type[ValueError]
) -> int:
    if not _is_int(value) or value < 0:
        raise error(f"{field} must be a non-negative integer, got {value!r}")
    return value


def _nonempty(value: object, field: str, error: type[ValueError]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error(f"{field} must be a non-empty string")
    return value


def _product(values: Sequence[int]) -> int:
    return reduce(mul, values, 1)


class PlanName(StrEnum):
    """Architectures required by the greenfield comparison contract."""

    PP8_LP4 = "PP8_LP4"
    PP16_LP2 = "PP16_LP2"
    WS32_2D = "WS32_2D"
    LEGACY_TP32_DCP8 = "LEGACY_TP32_DCP8"


@dataclass(frozen=True, slots=True)
class ModelGeometry:
    """Exact compile-relevant GLM-5.2 model geometry."""

    model_id: str
    num_layers: int
    first_dense_layers: int
    hidden_size: int
    dense_intermediate_size: int
    num_routed_experts: int
    num_shared_experts: int
    routed_top_k: int
    moe_intermediate_size: int
    dsa_top_k: int
    dsa_indexer_heads: int
    dsa_indexer_head_dim: int
    index_share_group_size: int
    attention_heads: int
    kv_heads: int
    kv_lora_rank: int
    q_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    num_nextn_predict_layers: int
    max_position_embeddings: int
    vocab_size: int
    activation_dtype: str
    weight_storage_dtype: str
    fp8_block_shape: tuple[int, int]
    mlp_layer_types: tuple[str, ...]
    indexer_types: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fp8_block_shape", tuple(self.fp8_block_shape))
        object.__setattr__(self, "mlp_layer_types", tuple(self.mlp_layer_types))
        object.__setattr__(self, "indexer_types", tuple(self.indexer_types))

        for field in (
            "model_id",
            "activation_dtype",
            "weight_storage_dtype",
        ):
            _nonempty(getattr(self, field), field, GeometryValidationError)
        for field in (
            "num_layers",
            "hidden_size",
            "dense_intermediate_size",
            "num_routed_experts",
            "num_shared_experts",
            "routed_top_k",
            "moe_intermediate_size",
            "dsa_top_k",
            "dsa_indexer_heads",
            "dsa_indexer_head_dim",
            "index_share_group_size",
            "attention_heads",
            "kv_heads",
            "kv_lora_rank",
            "q_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "num_nextn_predict_layers",
            "max_position_embeddings",
            "vocab_size",
        ):
            _positive_int(getattr(self, field), field, GeometryValidationError)
        _nonnegative_int(
            self.first_dense_layers,
            "first_dense_layers",
            GeometryValidationError,
        )

        if self.first_dense_layers > self.num_layers:
            raise GeometryValidationError(
                "first_dense_layers cannot exceed num_layers"
            )
        if self.routed_top_k > self.num_routed_experts:
            raise GeometryValidationError(
                "routed_top_k cannot exceed num_routed_experts"
            )
        if self.dsa_top_k > self.max_position_embeddings:
            raise GeometryValidationError(
                "dsa_top_k cannot exceed max_position_embeddings"
            )
        if self.num_nextn_predict_layers != 1:
            raise GeometryValidationError(
                "the exact GLM-5.2 target requires one MTP layer"
            )
        if self.hidden_size % self.attention_heads:
            raise GeometryValidationError(
                "hidden_size must be divisible by attention_heads"
            )
        if self.attention_heads % self.kv_heads:
            raise GeometryValidationError(
                "attention_heads must be divisible by kv_heads"
            )
        if len(self.fp8_block_shape) != 2 or any(
            not _is_int(v) or v <= 0 for v in self.fp8_block_shape
        ):
            raise GeometryValidationError(
                "fp8_block_shape must contain exactly two positive integers"
            )
        if len(self.mlp_layer_types) != self.num_layers:
            raise GeometryValidationError(
                "mlp_layer_types must contain one entry per transformer layer"
            )
        if len(self.indexer_types) != self.num_layers:
            raise GeometryValidationError(
                "indexer_types must contain one entry per transformer layer"
            )
        if set(self.mlp_layer_types) - {"dense", "sparse"}:
            raise GeometryValidationError(
                "mlp_layer_types entries must be 'dense' or 'sparse'"
            )
        if set(self.indexer_types) - {"full", "shared"}:
            raise GeometryValidationError(
                "indexer_types entries must be 'full' or 'shared'"
            )
        expected_mlp = (
            ("dense",) * self.first_dense_layers
            + ("sparse",) * (self.num_layers - self.first_dense_layers)
        )
        if self.mlp_layer_types != expected_mlp:
            raise GeometryValidationError(
                "mlp_layer_types does not match first_dense_layers"
            )

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any]) -> "ModelGeometry":
        """Build the exact geometry from the checked-in HF configuration."""

        if config.get("model_type") != "glm_moe_dsa":
            raise GeometryValidationError(
                "expected model_type='glm_moe_dsa', got "
                f"{config.get('model_type')!r}"
            )
        quant = config.get("quantization_config")
        if not isinstance(quant, Mapping) or quant.get("quant_method") != "fp8":
            raise GeometryValidationError("the greenfield target requires FP8 weights")
        fmt = _nonempty(quant.get("fmt"), "quantization_config.fmt", GeometryValidationError)
        block_shape = quant.get("weight_block_size")
        if not isinstance(block_shape, Sequence) or isinstance(block_shape, str):
            raise GeometryValidationError(
                "quantization_config.weight_block_size must be a sequence"
            )
        try:
            return cls(
                model_id="zai-org/GLM-5.2-FP8",
                num_layers=config["num_hidden_layers"],
                first_dense_layers=config["first_k_dense_replace"],
                hidden_size=config["hidden_size"],
                dense_intermediate_size=config["intermediate_size"],
                num_routed_experts=config["n_routed_experts"],
                num_shared_experts=config["n_shared_experts"],
                routed_top_k=config["num_experts_per_tok"],
                moe_intermediate_size=config["moe_intermediate_size"],
                dsa_top_k=config["index_topk"],
                dsa_indexer_heads=config["index_n_heads"],
                dsa_indexer_head_dim=config["index_head_dim"],
                index_share_group_size=config["index_topk_freq"],
                attention_heads=config["num_attention_heads"],
                kv_heads=config["num_key_value_heads"],
                kv_lora_rank=config["kv_lora_rank"],
                q_lora_rank=config["q_lora_rank"],
                qk_nope_head_dim=config["qk_nope_head_dim"],
                qk_rope_head_dim=config["qk_rope_head_dim"],
                v_head_dim=config["v_head_dim"],
                num_nextn_predict_layers=config["num_nextn_predict_layers"],
                max_position_embeddings=config["max_position_embeddings"],
                vocab_size=config["vocab_size"],
                activation_dtype=config["dtype"],
                weight_storage_dtype=f"fp8:{fmt}",
                fp8_block_shape=tuple(block_shape),
                mlp_layer_types=tuple(config["mlp_layer_types"]),
                indexer_types=tuple(config["indexer_types"]),
            )
        except KeyError as exc:
            raise GeometryValidationError(
                f"missing required HF config field {exc.args[0]!r}"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_dtype": self.activation_dtype,
            "attention_heads": self.attention_heads,
            "dense_intermediate_size": self.dense_intermediate_size,
            "dsa_indexer_head_dim": self.dsa_indexer_head_dim,
            "dsa_indexer_heads": self.dsa_indexer_heads,
            "dsa_top_k": self.dsa_top_k,
            "first_dense_layers": self.first_dense_layers,
            "fp8_block_shape": list(self.fp8_block_shape),
            "hidden_size": self.hidden_size,
            "index_share_group_size": self.index_share_group_size,
            "indexer_types": list(self.indexer_types),
            "kv_heads": self.kv_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "max_position_embeddings": self.max_position_embeddings,
            "mlp_layer_types": list(self.mlp_layer_types),
            "model_id": self.model_id,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_layers": self.num_layers,
            "num_routed_experts": self.num_routed_experts,
            "num_shared_experts": self.num_shared_experts,
            "num_nextn_predict_layers": self.num_nextn_predict_layers,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "q_lora_rank": self.q_lora_rank,
            "routed_top_k": self.routed_top_k,
            "vocab_size": self.vocab_size,
            "v_head_dim": self.v_head_dim,
            "weight_storage_dtype": self.weight_storage_dtype,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelGeometry":
        return cls(**dict(value))

    @property
    def geometry_hash(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class PhysicalDevice:
    """Runtime-observed identity of one JAX-visible TPU chip."""

    device_id: int
    process_index: int
    local_device_id: int
    coordinates: tuple[int, ...]
    core_on_chip: int
    platform: str
    device_kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        for field in (
            "device_id",
            "process_index",
            "local_device_id",
            "core_on_chip",
        ):
            _nonnegative_int(getattr(self, field), field, TopologyValidationError)
        if not self.coordinates or any(
            not _is_int(v) or v < 0 for v in self.coordinates
        ):
            raise TopologyValidationError(
                "coordinates must be a non-empty tuple of non-negative integers"
            )
        _nonempty(self.platform, "platform", TopologyValidationError)
        _nonempty(self.device_kind, "device_kind", TopologyValidationError)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": list(self.coordinates),
            "core_on_chip": self.core_on_chip,
            "device_id": self.device_id,
            "device_kind": self.device_kind,
            "local_device_id": self.local_device_id,
            "platform": self.platform,
            "process_index": self.process_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalDevice":
        return cls(**dict(value))


@dataclass(frozen=True, slots=True)
class PhysicalTopology:
    """Canonical runtime device inventory; never inferred from JAX ordering."""

    slice_name: str
    topology_shape: tuple[int, ...]
    devices: tuple[PhysicalDevice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology_shape", tuple(self.topology_shape))
        object.__setattr__(
            self,
            "devices",
            tuple(sorted(self.devices, key=lambda device: device.device_id)),
        )
        _nonempty(self.slice_name, "slice_name", TopologyValidationError)
        if not self.topology_shape or any(
            not _is_int(v) or v <= 0 for v in self.topology_shape
        ):
            raise TopologyValidationError(
                "topology_shape must contain positive integer dimensions"
            )
        if not self.devices:
            raise TopologyValidationError("devices must not be empty")
        if _product(self.topology_shape) != len(self.devices):
            raise TopologyValidationError(
                "topology_shape product must equal the number of devices"
            )
        dimensions = len(self.topology_shape)
        if any(len(device.coordinates) != dimensions for device in self.devices):
            raise TopologyValidationError(
                "every device coordinate must match topology dimensionality"
            )
        if any(
            coordinate >= self.topology_shape[axis]
            for device in self.devices
            for axis, coordinate in enumerate(device.coordinates)
        ):
            raise TopologyValidationError(
                "device coordinate lies outside topology_shape"
            )
        self._require_unique("device_id", [d.device_id for d in self.devices])
        self._require_unique("coordinates", [d.coordinates for d in self.devices])
        self._require_unique(
            "(process_index, local_device_id)",
            [(d.process_index, d.local_device_id) for d in self.devices],
        )
        platforms = {device.platform for device in self.devices}
        kinds = {device.device_kind for device in self.devices}
        if len(platforms) != 1 or len(kinds) != 1:
            raise TopologyValidationError(
                "all devices must report one platform and one device_kind"
            )
        local_ids: dict[int, list[int]] = {}
        for device in self.devices:
            local_ids.setdefault(device.process_index, []).append(
                device.local_device_id
            )
        for process, ids in local_ids.items():
            if sorted(ids) != list(range(len(ids))):
                raise TopologyValidationError(
                    f"process {process} local_device_id values must be contiguous from zero"
                )

    @staticmethod
    def _require_unique(field: str, values: Sequence[object]) -> None:
        if len(set(values)) != len(values):
            raise TopologyValidationError(f"duplicate physical device {field}")

    @property
    def process_indices(self) -> tuple[int, ...]:
        return tuple(sorted({device.process_index for device in self.devices}))

    @property
    def topology_hash(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": [device.to_dict() for device in self.devices],
            "slice_name": self.slice_name,
            "topology_shape": list(self.topology_shape),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PhysicalTopology":
        return cls(
            slice_name=value["slice_name"],
            topology_shape=tuple(value["topology_shape"]),
            devices=tuple(
                PhysicalDevice.from_dict(device) for device in value["devices"]
            ),
        )


@dataclass(frozen=True, slots=True)
class StageAssignment:
    """Contiguous layer ownership and complete per-stage memory accounting."""

    stage_id: int
    process_index: int | None
    device_ids: tuple[int, ...]
    layer_start: int
    layer_end_exclusive: int
    persistent_weight_bytes: int
    fp8_scale_bytes: int
    kv_bytes_at_target_context: int
    dsa_state_bytes: int
    temporary_bytes: int
    reserved_overlay_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_ids", tuple(self.device_ids))
        _nonnegative_int(self.stage_id, "stage_id", PlanValidationError)
        if self.process_index is not None:
            _nonnegative_int(
                self.process_index, "process_index", PlanValidationError
            )
        if not self.device_ids:
            raise PlanValidationError("stage device_ids must not be empty")
        if any(not _is_int(device_id) or device_id < 0 for device_id in self.device_ids):
            raise PlanValidationError(
                "stage device_ids must be non-negative integers"
            )
        if len(set(self.device_ids)) != len(self.device_ids):
            raise PlanValidationError("stage device_ids must be unique")
        _nonnegative_int(self.layer_start, "layer_start", PlanValidationError)
        _positive_int(
            self.layer_end_exclusive,
            "layer_end_exclusive",
            PlanValidationError,
        )
        if self.layer_end_exclusive <= self.layer_start:
            raise PlanValidationError(
                "each stage must own at least one contiguous layer"
            )
        for field in (
            "persistent_weight_bytes",
            "fp8_scale_bytes",
            "kv_bytes_at_target_context",
            "dsa_state_bytes",
            "temporary_bytes",
            "reserved_overlay_bytes",
        ):
            _nonnegative_int(getattr(self, field), field, PlanValidationError)

    @property
    def accounted_bytes(self) -> int:
        return sum(
            (
                self.persistent_weight_bytes,
                self.fp8_scale_bytes,
                self.kv_bytes_at_target_context,
                self.dsa_state_bytes,
                self.temporary_bytes,
                self.reserved_overlay_bytes,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_ids": list(self.device_ids),
            "dsa_state_bytes": self.dsa_state_bytes,
            "fp8_scale_bytes": self.fp8_scale_bytes,
            "kv_bytes_at_target_context": self.kv_bytes_at_target_context,
            "layer_end_exclusive": self.layer_end_exclusive,
            "layer_start": self.layer_start,
            "persistent_weight_bytes": self.persistent_weight_bytes,
            "process_index": self.process_index,
            "reserved_overlay_bytes": self.reserved_overlay_bytes,
            "stage_id": self.stage_id,
            "temporary_bytes": self.temporary_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageAssignment":
        return cls(**dict(value))


_PLAN_GEOMETRIES: dict[PlanName, tuple[int, int]] = {
    PlanName.PP8_LP4: (8, 4),
    PlanName.PP16_LP2: (16, 2),
    PlanName.WS32_2D: (1, 32),
    PlanName.LEGACY_TP32_DCP8: (1, 32),
}


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Fully explicit, serializable execution and ownership contract."""

    name: PlanName
    geometry: ModelGeometry
    topology: PhysicalTopology
    target_context_length: int
    pipeline_stages: int
    local_parallel_size: int
    stage_assignments: tuple[StageAssignment, ...]
    local_mesh_shape: tuple[int, ...]
    residual_layout: str
    expert_layout: str
    kv_layout: str
    transport: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "name", PlanName(self.name))
        except ValueError as exc:
            raise PlanValidationError(f"unknown execution plan {self.name!r}") from exc
        object.__setattr__(
            self,
            "stage_assignments",
            tuple(sorted(self.stage_assignments, key=lambda stage: stage.stage_id)),
        )
        object.__setattr__(self, "local_mesh_shape", tuple(self.local_mesh_shape))

        _positive_int(self.schema_version, "schema_version", PlanValidationError)
        _positive_int(
            self.target_context_length,
            "target_context_length",
            PlanValidationError,
        )
        _positive_int(
            self.pipeline_stages, "pipeline_stages", PlanValidationError
        )
        _positive_int(
            self.local_parallel_size,
            "local_parallel_size",
            PlanValidationError,
        )
        for field in (
            "residual_layout",
            "expert_layout",
            "kv_layout",
            "transport",
        ):
            _nonempty(getattr(self, field), field, PlanValidationError)
        if self.target_context_length > self.geometry.max_position_embeddings:
            raise PlanValidationError(
                "target_context_length exceeds model max_position_embeddings"
            )
        if not self.local_mesh_shape or any(
            not _is_int(value) or value <= 0 for value in self.local_mesh_shape
        ):
            raise PlanValidationError(
                "local_mesh_shape must contain positive integer dimensions"
            )
        if _product(self.local_mesh_shape) != self.local_parallel_size:
            raise PlanValidationError(
                "local_mesh_shape product must equal local_parallel_size"
            )
        expected = _PLAN_GEOMETRIES[self.name]
        actual = (self.pipeline_stages, self.local_parallel_size)
        if actual != expected:
            raise PlanValidationError(
                f"{self.name.value} requires stages/local size {expected}, got {actual}"
            )
        if len(self.stage_assignments) != self.pipeline_stages:
            raise PlanValidationError(
                "stage_assignments count must equal pipeline_stages"
            )
        stage_ids = tuple(stage.stage_id for stage in self.stage_assignments)
        if stage_ids != tuple(range(self.pipeline_stages)):
            raise PlanValidationError("stage_id values must be contiguous from zero")

        expected_layer_start = 0
        all_device_ids: list[int] = []
        devices = {device.device_id: device for device in self.topology.devices}
        process_stage_counts: dict[int, int] = {}
        for stage in self.stage_assignments:
            if stage.layer_start != expected_layer_start:
                raise PlanValidationError(
                    "stage layer ranges must be contiguous and gap-free"
                )
            expected_layer_start = stage.layer_end_exclusive
            if len(stage.device_ids) != self.local_parallel_size:
                raise PlanValidationError(
                    f"stage {stage.stage_id} must own exactly "
                    f"{self.local_parallel_size} devices"
                )
            unknown = set(stage.device_ids) - devices.keys()
            if unknown:
                raise PlanValidationError(
                    f"stage {stage.stage_id} references unknown devices {sorted(unknown)}"
                )
            all_device_ids.extend(stage.device_ids)
            stage_processes = {devices[device_id].process_index for device_id in stage.device_ids}
            if self.name in (PlanName.PP8_LP4, PlanName.PP16_LP2):
                if len(stage_processes) != 1:
                    raise PlanValidationError(
                        f"{self.name.value} stage {stage.stage_id} must be host-local"
                    )
                process = next(iter(stage_processes))
                if stage.process_index != process:
                    raise PlanValidationError(
                        f"stage {stage.stage_id} process_index does not match its devices"
                    )
                process_stage_counts[process] = process_stage_counts.get(process, 0) + 1
            elif stage.process_index is not None:
                raise PlanValidationError(
                    f"global plan {self.name.value} stage process_index must be null"
                )
        if expected_layer_start != self.geometry.num_layers:
            raise PlanValidationError(
                "stage layer ranges must cover every transformer layer exactly once"
            )
        topology_ids = [device.device_id for device in self.topology.devices]
        if len(all_device_ids) != len(set(all_device_ids)):
            raise PlanValidationError("a physical device is assigned to multiple stages")
        if set(all_device_ids) != set(topology_ids):
            raise PlanValidationError(
                "stage assignments must use every topology device exactly once"
            )
        expected_per_process = 1 if self.name is PlanName.PP8_LP4 else 2
        if self.name in (PlanName.PP8_LP4, PlanName.PP16_LP2) and any(
            count != expected_per_process for count in process_stage_counts.values()
        ):
            raise PlanValidationError(
                f"{self.name.value} requires {expected_per_process} stage(s) per host"
            )
        if self.name in (PlanName.PP8_LP4, PlanName.PP16_LP2) and set(
            process_stage_counts
        ) != set(self.topology.process_indices):
            raise PlanValidationError("every topology host must own pipeline stages")

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert_layout": self.expert_layout,
            "geometry": self.geometry.to_dict(),
            "kv_layout": self.kv_layout,
            "local_mesh_shape": list(self.local_mesh_shape),
            "local_parallel_size": self.local_parallel_size,
            "name": self.name.value,
            "pipeline_stages": self.pipeline_stages,
            "residual_layout": self.residual_layout,
            "schema_version": self.schema_version,
            "stage_assignments": [
                assignment.to_dict() for assignment in self.stage_assignments
            ],
            "target_context_length": self.target_context_length,
            "topology": self.topology.to_dict(),
            "transport": self.transport,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionPlan":
        return cls(
            name=PlanName(value["name"]),
            geometry=ModelGeometry.from_dict(value["geometry"]),
            topology=PhysicalTopology.from_dict(value["topology"]),
            target_context_length=value["target_context_length"],
            pipeline_stages=value["pipeline_stages"],
            local_parallel_size=value["local_parallel_size"],
            stage_assignments=tuple(
                StageAssignment.from_dict(stage)
                for stage in value["stage_assignments"]
            ),
            local_mesh_shape=tuple(value["local_mesh_shape"]),
            residual_layout=value["residual_layout"],
            expert_layout=value["expert_layout"],
            kv_layout=value["kv_layout"],
            transport=value["transport"],
            schema_version=value["schema_version"],
        )

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is None:
            return _canonical_json(self.to_dict())
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, value: str) -> "ExecutionPlan":
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise PlanValidationError("execution plan JSON must contain an object")
        return cls.from_dict(decoded)

    @property
    def plan_hash(self) -> str:
        return _fingerprint(self.to_dict())
