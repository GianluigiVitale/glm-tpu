"""Exact logical layout contract for the WS32_2D architecture challenger.

WS32 uses the complete 32-chip slice as an ``expert=8 x feature=4`` mesh.
The live batch-one residual is sharded only over ``feature`` and replicated
over ``expert``.  Gate/up weights shard their contracting hidden dimension
over ``feature`` and their output/expert identity over ``expert``; reciprocal
down weights reverse those roles.  No layout in this module represents a
physical ``[32, hidden]`` activation.

This is an independent greenfield contract.  The legacy 2D branches are
design evidence only and are never imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any

from glm_tpu.exceptions import PlanValidationError
from glm_tpu.config.model import ModelGeometry
from glm_tpu.distributed.topology import PhysicalTopology
from glm_tpu.config.parallel import EXPERT_AXIS, FEATURE_AXIS


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PlanValidationError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class MeshContract:
    """Compile-relevant WS32 logical mesh and ownership geometry."""

    expert_axis_size: int = 8
    feature_axis_size: int = 4
    shared_expert_layout: str = "feature_sharded_expert_axis_replicated"

    def __post_init__(self) -> None:
        _positive_int(self.expert_axis_size, "expert_axis_size")
        _positive_int(self.feature_axis_size, "feature_axis_size")
        if self.device_count != 32:
            raise PlanValidationError("WS32_2D requires exactly 32 devices")
        if self.shared_expert_layout != ("feature_sharded_expert_axis_replicated"):
            raise PlanValidationError("unknown WS32 shared-expert layout")

    @property
    def device_count(self) -> int:
        return self.expert_axis_size * self.feature_axis_size

    @property
    def axis_names(self) -> tuple[str, str]:
        return (EXPERT_AXIS, FEATURE_AXIS)

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return (self.expert_axis_size, self.feature_axis_size)

    def validate_topology(self, topology: PhysicalTopology) -> None:
        if len(topology.devices) != self.device_count:
            raise PlanValidationError(f"WS32 expected {self.device_count} devices, found {len(topology.devices)}")
        if topology.topology_shape != (2, 4, 4):
            raise PlanValidationError("WS32 requires the protected physical 2x4x4 topology")

    def validate_geometry(self, geometry: ModelGeometry) -> None:
        divisibility = {
            "hidden_size/feature": (
                geometry.hidden_size,
                self.feature_axis_size,
            ),
            "dense_intermediate/expert": (
                geometry.dense_intermediate_size,
                self.expert_axis_size,
            ),
            "routed_experts/expert": (
                geometry.num_routed_experts,
                self.expert_axis_size,
            ),
            "attention_heads/feature": (
                geometry.attention_heads,
                self.feature_axis_size,
            ),
            "indexer_heads/feature": (
                geometry.dsa_indexer_heads,
                self.feature_axis_size,
            ),
            "vocabulary/expert": (
                geometry.vocab_size,
                self.expert_axis_size,
            ),
        }
        for name, (value, divisor) in divisibility.items():
            if value % divisor:
                raise PlanValidationError(f"WS32 requires exact {name} division")
        block_out, block_in = geometry.fp8_block_shape
        block_dimensions = {
            "local hidden": geometry.hidden_size // self.feature_axis_size,
            "local dense intermediate": (geometry.dense_intermediate_size // self.expert_axis_size),
            "MoE intermediate": geometry.moe_intermediate_size,
        }
        for name, value in block_dimensions.items():
            if value % block_out or value % block_in:
                raise PlanValidationError(f"WS32 {name} must preserve complete FP8 blocks")

    def layout_summary(self, geometry: ModelGeometry) -> dict[str, Any]:
        """Return the exact global/local tensor ownership prototype."""

        self.validate_geometry(geometry)
        expert = self.expert_axis_size
        feature = self.feature_axis_size
        hidden = geometry.hidden_size
        dense = geometry.dense_intermediate_size
        routed = geometry.num_routed_experts
        moe = geometry.moe_intermediate_size
        return {
            "axis_names": list(self.axis_names),
            "mesh_shape": list(self.mesh_shape),
            "one_live_decode_row": True,
            "residual": {
                "global_shape": [1, hidden],
                "local_shape": [1, hidden // feature],
                "partition_spec": [None, FEATURE_AXIS],
                "replicated_axis": EXPERT_AXIS,
            },
            "dense_gate_up": {
                "global_shape": [dense, hidden],
                "local_shape": [dense // expert, hidden // feature],
                "partition_spec": [EXPERT_AXIS, FEATURE_AXIS],
                "partial_reduction_axis": FEATURE_AXIS,
                "partial_reduction_dtype": "float32",
            },
            "dense_down": {
                "global_shape": [hidden, dense],
                "local_shape": [hidden // feature, dense // expert],
                "partition_spec": [FEATURE_AXIS, EXPERT_AXIS],
                "partial_reduction_axis": EXPERT_AXIS,
                "partial_reduction_dtype": "float32",
            },
            "routed_gate_up": {
                "global_shape": [routed, moe, hidden],
                "local_shape": [routed // expert, moe, hidden // feature],
                "partition_spec": [
                    EXPERT_AXIS,
                    None,
                    FEATURE_AXIS,
                ],
                "partial_reduction_axis": FEATURE_AXIS,
                "partial_reduction_dtype": "float32",
            },
            "routed_down": {
                "global_shape": [routed, hidden, moe],
                "local_shape": [routed // expert, hidden // feature, moe],
                "partition_spec": [
                    EXPERT_AXIS,
                    FEATURE_AXIS,
                    None,
                ],
                "combine_axis": EXPERT_AXIS,
                "combine_dtype": "float32_then_bfloat16_boundary",
            },
            "shared_expert": {
                "layout": self.shared_expert_layout,
                "gate_up_partition_spec": [None, FEATURE_AXIS],
                "down_partition_spec": [FEATURE_AXIS, None],
                "declared_replication_factor": expert,
            },
            "attention": {
                # Q/KV low-rank projections contract the persistent hidden
                # shard over feature-4 and remain compact/replicated.  Head
                # projections shard complete heads over expert-8, and the
                # reciprocal output projection returns the persistent hidden
                # feature shard with one expert-8 reduction.
                "q_a_kv_a": {
                    "weight_partition_spec": [None, FEATURE_AXIS],
                    "result_partition_spec": [None, None],
                    "reduction_axis": FEATURE_AXIS,
                },
                "q_b_kv_b": {
                    "weight_partition_spec": [EXPERT_AXIS, None],
                    "head_partition_axis": EXPERT_AXIS,
                    "replicated_axis": FEATURE_AXIS,
                },
                "o_projection": {
                    "weight_partition_spec": [
                        FEATURE_AXIS,
                        EXPERT_AXIS,
                    ],
                    "result_partition_spec": [None, FEATURE_AXIS],
                    "reduction_axis": EXPERT_AXIS,
                },
                "selected_cache": {
                    "context_partition_axis": EXPERT_AXIS,
                    "replicated_axis": FEATURE_AXIS,
                    "exchange_axis": EXPERT_AXIS,
                    "physical_group_size": expert,
                },
            },
            "dsa": {
                "query_head_partition_axis": EXPERT_AXIS,
                "hidden_contraction_axis": FEATURE_AXIS,
                "context_partition_axis": EXPERT_AXIS,
                "score_head_reduction_axis": EXPERT_AXIS,
                "candidate_merge_axis": EXPERT_AXIS,
            },
            "embedding_logits": {
                "weight_partition_spec": [
                    EXPERT_AXIS,
                    FEATURE_AXIS,
                ],
                "embedding_owner_reduce_axis": EXPERT_AXIS,
                "logit_contraction_axis": FEATURE_AXIS,
            },
            "forbidden": {
                "batch_32_decode_rows": True,
                "full_pod_hidden_reconstruction": True,
                "repeated_collective_group_size_32": True,
            },
        }


@dataclass(frozen=True, slots=True)
class PhysicalMesh:
    """Topology-derived logical mesh and its two repeated subgroup families."""

    device_ids: tuple[tuple[int, ...], ...]
    expert_groups: tuple[tuple[int, ...], ...]
    feature_groups: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "device_ids",
            tuple(tuple(row) for row in self.device_ids),
        )
        object.__setattr__(
            self,
            "expert_groups",
            tuple(tuple(group) for group in self.expert_groups),
        )
        object.__setattr__(
            self,
            "feature_groups",
            tuple(tuple(group) for group in self.feature_groups),
        )
        if len(self.device_ids) != 8 or any(len(row) != 4 for row in self.device_ids):
            raise PlanValidationError("WS32 physical mesh must be exactly 8x4")
        flattened = tuple(item for row in self.device_ids for item in row)
        if len(set(flattened)) != 32:
            raise PlanValidationError("WS32 physical mesh must use 32 unique devices")
        if self.feature_groups != self.device_ids:
            raise PlanValidationError("WS32 feature groups must be the logical mesh rows")
        expected_expert = tuple(tuple(row[column] for row in self.device_ids) for column in range(4))
        if self.expert_groups != expected_expert:
            raise PlanValidationError("WS32 expert groups must be the logical mesh columns")

    @property
    def flattened_device_ids(self) -> tuple[int, ...]:
        return tuple(item for row in self.device_ids for item in row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_names": [EXPERT_AXIS, FEATURE_AXIS],
            "device_ids": [list(row) for row in self.device_ids],
            "expert_groups": [list(group) for group in self.expert_groups],
            "feature_groups": [list(group) for group in self.feature_groups],
            "mesh_shape": [8, 4],
        }

    @property
    def mesh_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


def build_physical_mesh(topology: PhysicalTopology) -> PhysicalMesh:
    """Map physical ``(x,y,z)`` to logical ``expert=(x,y), feature=z``."""

    contract = MeshContract()
    contract.validate_topology(topology)
    by_coordinates = {device.coordinates: device.device_id for device in topology.devices}
    rows = tuple(tuple(by_coordinates[(x, y, z)] for z in range(4)) for x in range(2) for y in range(4))
    mesh = PhysicalMesh(
        device_ids=rows,
        feature_groups=rows,
        expert_groups=tuple(tuple(row[column] for row in rows) for column in range(4)),
    )
    if set(mesh.flattened_device_ids) != {device.device_id for device in topology.devices}:
        raise PlanValidationError("WS32 physical mesh does not cover the observed topology")
    return mesh
