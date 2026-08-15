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
import re
from typing import Any

from ..errors import PlanValidationError
from .hlo_contract import HloInstruction, parse_hlo_module
from ..partitioning.source_inventory import SourceInventory, SourceTensor
from ..types import (
    ExecutionPlan,
    ModelGeometry,
    PhysicalTopology,
    PlanName,
    StageAssignment,
)


WS32_EXPERT_AXIS = "expert"
WS32_FEATURE_AXIS = "feature"

_ROUTED_MLP = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
_SHARED_MLP = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
_DENSE_MLP = re.compile(
    r"^model\.layers\.(\d+)\.mlp\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
_ROUTER_MLP = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.gate\."
    r"(weight|e_score_correction_bias)$"
)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise PlanValidationError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class Ws32MeshContract:
    """Compile-relevant WS32 logical mesh and ownership geometry."""

    expert_axis_size: int = 8
    feature_axis_size: int = 4
    shared_expert_layout: str = "feature_sharded_expert_axis_replicated"

    def __post_init__(self) -> None:
        _positive_int(self.expert_axis_size, "expert_axis_size")
        _positive_int(self.feature_axis_size, "feature_axis_size")
        if self.device_count != 32:
            raise PlanValidationError("WS32_2D requires exactly 32 devices")
        if self.shared_expert_layout != (
            "feature_sharded_expert_axis_replicated"
        ):
            raise PlanValidationError("unknown WS32 shared-expert layout")

    @property
    def device_count(self) -> int:
        return self.expert_axis_size * self.feature_axis_size

    @property
    def axis_names(self) -> tuple[str, str]:
        return (WS32_EXPERT_AXIS, WS32_FEATURE_AXIS)

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return (self.expert_axis_size, self.feature_axis_size)

    def validate_topology(self, topology: PhysicalTopology) -> None:
        if len(topology.devices) != self.device_count:
            raise PlanValidationError(
                f"WS32 expected {self.device_count} devices, "
                f"found {len(topology.devices)}"
            )
        if topology.topology_shape != (2, 4, 4):
            raise PlanValidationError(
                "WS32 requires the protected physical 2x4x4 topology"
            )

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
            "local dense intermediate": (
                geometry.dense_intermediate_size // self.expert_axis_size
            ),
            "MoE intermediate": geometry.moe_intermediate_size,
        }
        for name, value in block_dimensions.items():
            if value % block_out or value % block_in:
                raise PlanValidationError(
                    f"WS32 {name} must preserve complete FP8 blocks"
                )

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
                "partition_spec": [None, WS32_FEATURE_AXIS],
                "replicated_axis": WS32_EXPERT_AXIS,
            },
            "dense_gate_up": {
                "global_shape": [dense, hidden],
                "local_shape": [dense // expert, hidden // feature],
                "partition_spec": [WS32_EXPERT_AXIS, WS32_FEATURE_AXIS],
                "partial_reduction_axis": WS32_FEATURE_AXIS,
                "partial_reduction_dtype": "float32",
            },
            "dense_down": {
                "global_shape": [hidden, dense],
                "local_shape": [hidden // feature, dense // expert],
                "partition_spec": [WS32_FEATURE_AXIS, WS32_EXPERT_AXIS],
                "partial_reduction_axis": WS32_EXPERT_AXIS,
                "partial_reduction_dtype": "float32",
            },
            "routed_gate_up": {
                "global_shape": [routed, moe, hidden],
                "local_shape": [routed // expert, moe, hidden // feature],
                "partition_spec": [
                    WS32_EXPERT_AXIS,
                    None,
                    WS32_FEATURE_AXIS,
                ],
                "partial_reduction_axis": WS32_FEATURE_AXIS,
                "partial_reduction_dtype": "float32",
            },
            "routed_down": {
                "global_shape": [routed, hidden, moe],
                "local_shape": [routed // expert, hidden // feature, moe],
                "partition_spec": [
                    WS32_EXPERT_AXIS,
                    WS32_FEATURE_AXIS,
                    None,
                ],
                "combine_axis": WS32_EXPERT_AXIS,
                "combine_dtype": "float32_then_bfloat16_boundary",
            },
            "shared_expert": {
                "layout": self.shared_expert_layout,
                "gate_up_partition_spec": [None, WS32_FEATURE_AXIS],
                "down_partition_spec": [WS32_FEATURE_AXIS, None],
                "declared_replication_factor": expert,
            },
            "attention": {
                # Q/KV low-rank projections contract the persistent hidden
                # shard over feature-4 and remain compact/replicated.  Head
                # projections shard complete heads over expert-8, and the
                # reciprocal output projection returns the persistent hidden
                # feature shard with one expert-8 reduction.
                "q_a_kv_a": {
                    "weight_partition_spec": [None, WS32_FEATURE_AXIS],
                    "result_partition_spec": [None, None],
                    "reduction_axis": WS32_FEATURE_AXIS,
                },
                "q_b_kv_b": {
                    "weight_partition_spec": [WS32_EXPERT_AXIS, None],
                    "head_partition_axis": WS32_EXPERT_AXIS,
                    "replicated_axis": WS32_FEATURE_AXIS,
                },
                "o_projection": {
                    "weight_partition_spec": [
                        WS32_FEATURE_AXIS,
                        WS32_EXPERT_AXIS,
                    ],
                    "result_partition_spec": [None, WS32_FEATURE_AXIS],
                    "reduction_axis": WS32_EXPERT_AXIS,
                },
                "selected_cache": {
                    "context_partition_axis": WS32_EXPERT_AXIS,
                    "replicated_axis": WS32_FEATURE_AXIS,
                    "exchange_axis": WS32_EXPERT_AXIS,
                    "physical_group_size": expert,
                },
            },
            "dsa": {
                "query_head_partition_axis": WS32_EXPERT_AXIS,
                "hidden_contraction_axis": WS32_FEATURE_AXIS,
                "context_partition_axis": WS32_EXPERT_AXIS,
                "score_head_reduction_axis": WS32_EXPERT_AXIS,
                "candidate_merge_axis": WS32_EXPERT_AXIS,
            },
            "embedding_logits": {
                "weight_partition_spec": [
                    WS32_EXPERT_AXIS,
                    WS32_FEATURE_AXIS,
                ],
                "embedding_owner_reduce_axis": WS32_EXPERT_AXIS,
                "logit_contraction_axis": WS32_FEATURE_AXIS,
            },
            "forbidden": {
                "batch_32_decode_rows": True,
                "full_pod_hidden_reconstruction": True,
                "repeated_collective_group_size_32": True,
            },
        }


@dataclass(frozen=True, slots=True)
class Ws32PhysicalMesh:
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
        if len(self.device_ids) != 8 or any(
            len(row) != 4 for row in self.device_ids
        ):
            raise PlanValidationError("WS32 physical mesh must be exactly 8x4")
        flattened = tuple(item for row in self.device_ids for item in row)
        if len(set(flattened)) != 32:
            raise PlanValidationError(
                "WS32 physical mesh must use 32 unique devices"
            )
        if self.feature_groups != self.device_ids:
            raise PlanValidationError(
                "WS32 feature groups must be the logical mesh rows"
            )
        expected_expert = tuple(
            tuple(row[column] for row in self.device_ids)
            for column in range(4)
        )
        if self.expert_groups != expected_expert:
            raise PlanValidationError(
                "WS32 expert groups must be the logical mesh columns"
            )

    @property
    def flattened_device_ids(self) -> tuple[int, ...]:
        return tuple(item for row in self.device_ids for item in row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_names": [WS32_EXPERT_AXIS, WS32_FEATURE_AXIS],
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


def build_ws32_physical_mesh(topology: PhysicalTopology) -> Ws32PhysicalMesh:
    """Map physical ``(x,y,z)`` to logical ``expert=(x,y), feature=z``."""

    contract = Ws32MeshContract()
    contract.validate_topology(topology)
    by_coordinates = {
        device.coordinates: device.device_id for device in topology.devices
    }
    rows = tuple(
        tuple(by_coordinates[(x, y, z)] for z in range(4))
        for x in range(2)
        for y in range(4)
    )
    mesh = Ws32PhysicalMesh(
        device_ids=rows,
        feature_groups=rows,
        expert_groups=tuple(
            tuple(row[column] for row in rows) for column in range(4)
        ),
    )
    if set(mesh.flattened_device_ids) != {
        device.device_id for device in topology.devices
    }:
        raise PlanValidationError(
            "WS32 physical mesh does not cover the observed topology"
        )
    return mesh


@dataclass(frozen=True, slots=True)
class Ws32PerChipMemory:
    """Exact per-chip values supplied by the future final-layout pack plan."""

    persistent_weight_bytes: int
    fp8_scale_bytes: int
    kv_bytes_at_target_context: int
    dsa_state_bytes: int
    temporary_bytes: int
    reserved_overlay_bytes: int

    def __post_init__(self) -> None:
        for field in (
            "persistent_weight_bytes",
            "fp8_scale_bytes",
            "kv_bytes_at_target_context",
            "dsa_state_bytes",
            "temporary_bytes",
            "reserved_overlay_bytes",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PlanValidationError(
                    f"WS32 {field} must be a non-negative integer"
                )


@dataclass(frozen=True, slots=True)
class Ws32MlpCapacityReport:
    """Exact base-decoder MLP source and final-owner byte accounting."""

    source_inventory_sha256: str
    source_tensor_count: int
    source_bytes: int
    packed_bytes: int
    parameter_bytes_by_chip: tuple[int, ...]
    fp8_scale_bytes_by_chip: tuple[int, ...]
    shared_replication_extra_bytes: int
    compact_replication_extra_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameter_bytes_by_chip", tuple(self.parameter_bytes_by_chip)
        )
        object.__setattr__(
            self, "fp8_scale_bytes_by_chip", tuple(self.fp8_scale_bytes_by_chip)
        )
        if len(self.parameter_bytes_by_chip) != 32 or (
            len(self.fp8_scale_bytes_by_chip) != 32
        ):
            raise PlanValidationError("WS32 MLP capacity requires 32 chip totals")
        if self.packed_bytes != sum(self.parameter_bytes_by_chip) + sum(
            self.fp8_scale_bytes_by_chip
        ):
            raise PlanValidationError("WS32 MLP packed bytes do not reconcile")

    @property
    def maximum_persistent_bytes(self) -> int:
        return max(
            parameter + scale
            for parameter, scale in zip(
                self.parameter_bytes_by_chip,
                self.fp8_scale_bytes_by_chip,
                strict=True,
            )
        )

    @property
    def minimum_persistent_bytes(self) -> int:
        return min(
            parameter + scale
            for parameter, scale in zip(
                self.parameter_bytes_by_chip,
                self.fp8_scale_bytes_by_chip,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fp8_scale_bytes_by_chip": list(self.fp8_scale_bytes_by_chip),
            "compact_replication_extra_bytes": (
                self.compact_replication_extra_bytes
            ),
            "maximum_persistent_bytes": self.maximum_persistent_bytes,
            "minimum_persistent_bytes": self.minimum_persistent_bytes,
            "packed_bytes": self.packed_bytes,
            "parameter_bytes_by_chip": list(self.parameter_bytes_by_chip),
            "shared_replication_extra_bytes": self.shared_replication_extra_bytes,
            "source_bytes": self.source_bytes,
            "source_inventory_sha256": self.source_inventory_sha256,
            "source_tensor_count": self.source_tensor_count,
        }


@dataclass(frozen=True, slots=True)
class Ws32BaseCapacityReport:
    """Complete base-decoder final-layout persistent byte accounting."""

    source_inventory_sha256: str
    source_tensor_count: int
    source_bytes: int
    packed_bytes: int
    parameter_bytes_by_chip: tuple[int, ...]
    fp8_scale_bytes_by_chip: tuple[int, ...]
    mlp: Ws32MlpCapacityReport
    non_mlp_replication_extra_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "parameter_bytes_by_chip", tuple(self.parameter_bytes_by_chip)
        )
        object.__setattr__(
            self, "fp8_scale_bytes_by_chip", tuple(self.fp8_scale_bytes_by_chip)
        )
        if len(self.parameter_bytes_by_chip) != 32 or (
            len(self.fp8_scale_bytes_by_chip) != 32
        ):
            raise PlanValidationError("WS32 base capacity requires 32 chip totals")
        if self.packed_bytes != sum(self.parameter_bytes_by_chip) + sum(
            self.fp8_scale_bytes_by_chip
        ):
            raise PlanValidationError("WS32 base packed bytes do not reconcile")

    @property
    def maximum_persistent_bytes(self) -> int:
        return max(
            parameter + scale
            for parameter, scale in zip(
                self.parameter_bytes_by_chip,
                self.fp8_scale_bytes_by_chip,
                strict=True,
            )
        )

    @property
    def minimum_persistent_bytes(self) -> int:
        return min(
            parameter + scale
            for parameter, scale in zip(
                self.parameter_bytes_by_chip,
                self.fp8_scale_bytes_by_chip,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fp8_scale_bytes_by_chip": list(self.fp8_scale_bytes_by_chip),
            "maximum_persistent_bytes": self.maximum_persistent_bytes,
            "minimum_persistent_bytes": self.minimum_persistent_bytes,
            "mlp": self.mlp.to_dict(),
            "non_mlp_replication_extra_bytes": (
                self.non_mlp_replication_extra_bytes
            ),
            "packed_bytes": self.packed_bytes,
            "parameter_bytes_by_chip": list(self.parameter_bytes_by_chip),
            "source_bytes": self.source_bytes,
            "source_inventory_sha256": self.source_inventory_sha256,
            "source_tensor_count": self.source_tensor_count,
        }


def _add_shard(
    totals: list[int],
    tensor: SourceTensor,
    slots: tuple[int, ...],
    divisor: int,
) -> None:
    if tensor.byte_count % divisor:
        raise PlanValidationError(
            f"WS32 tensor {tensor.name!r} bytes do not divide over {divisor}"
        )
    amount = tensor.byte_count // divisor
    for slot in slots:
        totals[slot] += amount


def build_ws32_mlp_capacity_report(
    inventory: SourceInventory,
    geometry: ModelGeometry,
) -> Ws32MlpCapacityReport:
    """Account every base MLP leaf under the exact 8x4 ownership prototype."""

    if inventory.model_id != geometry.model_id:
        raise PlanValidationError("WS32 inventory and geometry model ids differ")
    parameter = [0] * 32
    scales = [0] * 32
    source_bytes = 0
    source_count = 0
    shared_source_bytes = 0
    compact_replicated_source_bytes = 0
    for tensor in inventory.tensors:
        if ".mlp." not in tensor.name or tensor.layer_id is None:
            continue
        if tensor.layer_id >= geometry.num_layers:
            continue
        routed = _ROUTED_MLP.fullmatch(tensor.name)
        shared = _SHARED_MLP.fullmatch(tensor.name)
        dense = _DENSE_MLP.fullmatch(tensor.name)
        router = _ROUTER_MLP.fullmatch(tensor.name)
        matches = sum(item is not None for item in (routed, shared, dense, router))
        if matches != 1:
            raise PlanValidationError(
                f"WS32 has no unique MLP ownership rule for {tensor.name!r}"
            )
        target = scales if tensor.is_fp8_scale else parameter
        source_bytes += tensor.byte_count
        source_count += 1
        if routed is not None:
            expert_id = int(routed.group(2))
            if not 0 <= expert_id < geometry.num_routed_experts:
                raise PlanValidationError("WS32 routed expert id is out of range")
            expert_coordinate = expert_id // (
                geometry.num_routed_experts // 8
            )
            slots = tuple(expert_coordinate * 4 + feature for feature in range(4))
            _add_shard(target, tensor, slots, 4)
        elif shared is not None:
            shared_source_bytes += tensor.byte_count
            _add_shard(target, tensor, tuple(range(32)), 4)
        elif dense is not None:
            _add_shard(target, tensor, tuple(range(32)), 32)
        else:
            assert router is not None
            if router.group(2) == "weight":
                _add_shard(target, tensor, tuple(range(32)), 32)
            else:
                compact_replicated_source_bytes += tensor.byte_count
                _add_shard(target, tensor, tuple(range(32)), 8)
    packed_bytes = sum(parameter) + sum(scales)
    expected_packed = (
        source_bytes
        + 7 * shared_source_bytes
        + 3 * compact_replicated_source_bytes
    )
    if packed_bytes != expected_packed:
        raise PlanValidationError(
            "WS32 MLP replication/sharding bytes do not reconcile"
        )
    return Ws32MlpCapacityReport(
        source_inventory_sha256=inventory.inventory_sha256,
        source_tensor_count=source_count,
        source_bytes=source_bytes,
        packed_bytes=packed_bytes,
        parameter_bytes_by_chip=tuple(parameter),
        fp8_scale_bytes_by_chip=tuple(scales),
        shared_replication_extra_bytes=7 * shared_source_bytes,
        compact_replication_extra_bytes=3 * compact_replicated_source_bytes,
    )


def _non_mlp_divisor(tensor: SourceTensor) -> int:
    """Return the exact full-decoder WS32 divisor for a non-MLP tensor.

    The divisor is derived from the executable layout, not merely from an
    even-capacity split.  Compact q/kv-a projections are hidden-feature
    sharded and expert-replicated; q/kv-b head projections are expert-sharded
    and feature-replicated; output projections and vocabulary tables are
    genuinely two-dimensional.  This costs about 0.58 GB/chip more than the
    earlier capacity-only 32-way split, but it avoids a full-mesh reshard and
    leaves the residual persistently feature-sharded.
    """

    name = tensor.name
    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return 32
    if name == "model.norm.weight" or name.endswith(
        (".input_layernorm.weight", ".post_attention_layernorm.weight")
    ):
        return 4
    if name.endswith(
        (
            ".self_attn.q_a_layernorm.weight",
            ".self_attn.kv_a_layernorm.weight",
        )
    ):
        return 1
    if name.endswith(
        (
            ".self_attn.indexer.k_norm.bias",
            ".self_attn.indexer.k_norm.weight",
        )
    ):
        return 1
    if name.endswith(".self_attn.indexer.wk.weight") or name.endswith(
        ".self_attn.indexer.wk.weight_scale_inv"
    ):
        return 4
    if name.endswith(".self_attn.kv_a_proj_with_mqa.weight") or name.endswith(
        ".self_attn.kv_a_proj_with_mqa.weight_scale_inv"
    ):
        return 4
    if name.endswith(".self_attn.indexer.weights_proj.weight"):
        return 32
    if name.endswith(
        (
            ".self_attn.indexer.wq_b.weight",
            ".self_attn.indexer.wq_b.weight_scale_inv",
            ".self_attn.kv_b_proj.weight",
            ".self_attn.kv_b_proj.weight_scale_inv",
            ".self_attn.q_b_proj.weight",
            ".self_attn.q_b_proj.weight_scale_inv",
        )
    ):
        return 8
    if name.endswith(
        (
            ".self_attn.o_proj.weight",
            ".self_attn.o_proj.weight_scale_inv",
        )
    ):
        return 32
    if name.endswith(
        (
            ".self_attn.q_a_proj.weight",
            ".self_attn.q_a_proj.weight_scale_inv",
        )
    ):
        return 4
    raise PlanValidationError(
        f"WS32 has no non-MLP ownership rule for {tensor.name!r}"
    )


def build_ws32_base_capacity_report(
    inventory: SourceInventory,
    geometry: ModelGeometry,
) -> Ws32BaseCapacityReport:
    """Account every base tensor under explicit WS32 final ownership."""

    mlp = build_ws32_mlp_capacity_report(inventory, geometry)
    parameter = list(mlp.parameter_bytes_by_chip)
    scales = list(mlp.fp8_scale_bytes_by_chip)
    non_mlp_source = 0
    non_mlp_packed = 0
    non_mlp_count = 0
    for tensor in inventory.tensors:
        if tensor.layer_id is not None and tensor.layer_id >= geometry.num_layers:
            continue
        if ".mlp." in tensor.name:
            continue
        divisor = _non_mlp_divisor(tensor)
        target = scales if tensor.is_fp8_scale else parameter
        _add_shard(target, tensor, tuple(range(32)), divisor)
        non_mlp_source += tensor.byte_count
        non_mlp_packed += tensor.byte_count * (32 // divisor)
        non_mlp_count += 1
    source_bytes = mlp.source_bytes + non_mlp_source
    packed_bytes = sum(parameter) + sum(scales)
    if packed_bytes != mlp.packed_bytes + non_mlp_packed:
        raise PlanValidationError("WS32 complete base bytes do not reconcile")
    expected_base = sum(
        tensor.byte_count
        for tensor in inventory.tensors
        if tensor.layer_id is None or tensor.layer_id < geometry.num_layers
    )
    if source_bytes != expected_base:
        raise PlanValidationError("WS32 base source coverage is incomplete")
    return Ws32BaseCapacityReport(
        source_inventory_sha256=inventory.inventory_sha256,
        source_tensor_count=mlp.source_tensor_count + non_mlp_count,
        source_bytes=source_bytes,
        packed_bytes=packed_bytes,
        parameter_bytes_by_chip=tuple(parameter),
        fp8_scale_bytes_by_chip=tuple(scales),
        mlp=mlp,
        non_mlp_replication_extra_bytes=non_mlp_packed - non_mlp_source,
    )


def build_ws32_execution_plan(
    *,
    geometry: ModelGeometry,
    topology: PhysicalTopology,
    target_context_length: int,
    memory: Ws32PerChipMemory,
) -> tuple[ExecutionPlan, Ws32PhysicalMesh]:
    """Build the one-stage WS32 plan from explicit audited memory inputs."""

    contract = Ws32MeshContract()
    contract.validate_geometry(geometry)
    physical_mesh = build_ws32_physical_mesh(topology)
    assignment = StageAssignment(
        stage_id=0,
        process_index=None,
        device_ids=physical_mesh.flattened_device_ids,
        layer_start=0,
        layer_end_exclusive=geometry.num_layers,
        persistent_weight_bytes=memory.persistent_weight_bytes,
        fp8_scale_bytes=memory.fp8_scale_bytes,
        kv_bytes_at_target_context=memory.kv_bytes_at_target_context,
        dsa_state_bytes=memory.dsa_state_bytes,
        temporary_bytes=memory.temporary_bytes,
        reserved_overlay_bytes=memory.reserved_overlay_bytes,
    )
    return (
        ExecutionPlan(
            name=PlanName.WS32_2D,
            geometry=geometry,
            topology=topology,
            target_context_length=target_context_length,
            pipeline_stages=1,
            local_parallel_size=32,
            stage_assignments=(assignment,),
            local_mesh_shape=contract.mesh_shape,
            residual_layout=(
                "persistent_hidden_feature_shard_h4_replicated_e8"
            ),
            expert_layout="expert8_identity_hidden4_reciprocal_2d",
            kv_layout="context8_head4_weight_stationary",
            transport="none_single_weight_stationary_stage",
        ),
        physical_mesh,
    )


@dataclass(frozen=True, slots=True)
class Ws32HloReport:
    """Structured fail-closed result for one WS32 repeated layer body."""

    kind: str
    all_reduce_count: int
    feature_reduce_count: int
    expert_reduce_count: int
    f32_operand_reduce_count: int
    bf16_result_reduce_count: int
    f32_result_reduce_count: int
    maximum_group_size: int
    violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.violations

    def raise_for_violations(self) -> None:
        if self.violations:
            raise PlanValidationError(
                "WS32 HLO contract failed: " + "; ".join(self.violations)
            )


def _logical_groups() -> tuple[
    frozenset[tuple[int, ...]], frozenset[tuple[int, ...]]
]:
    feature = frozenset(
        tuple(range(row * 4, row * 4 + 4)) for row in range(8)
    )
    expert = frozenset(
        tuple(row * 4 + column for row in range(8))
        for column in range(4)
    )
    return feature, expert


def _group_family(
    instruction: HloInstruction,
) -> str | None:
    groups = frozenset(tuple(group) for group in instruction.replica_groups)
    feature, expert = _logical_groups()
    if groups == feature:
        return WS32_FEATURE_AXIS
    if groups == expert:
        return WS32_EXPERT_AXIS
    return None


def _exact_scalar_add_reducer(
    item: HloInstruction,
    *,
    module_instructions: tuple[HloInstruction, ...],
) -> bool:
    match = re.search(r"\bto_apply=([A-Za-z0-9_.%:-]+)", item.raw_line)
    if (
        match is None
        or len(item.result_shapes) != 1
        or len(item.operand_shapes) != 1
    ):
        return False
    reducer_name = match.group(1).lstrip("%")
    reducer = tuple(
        instruction
        for instruction in module_instructions
        if instruction.computation.split(" ", 1)[0].lstrip("%")
        == reducer_name
    )
    if len(reducer) != 3:
        return False
    # TPU may fuse the post-reduction BF16 conversion into the scheduled
    # all-reduce result while retaining an F32 operand and F32 reducer.  The
    # reducer contract therefore follows the operand/accumulator dtype rather
    # than the externally scheduled result dtype.
    dtype = item.operand_shapes[0].dtype
    parameters = tuple(
        instruction for instruction in reducer if instruction.raw_opcode == "parameter"
    )
    adds = tuple(
        instruction for instruction in reducer if instruction.raw_opcode == "add"
    )
    if len(parameters) != 2 or len(adds) != 1:
        return False
    if {
        re.search(r"\bparameter\(([01])\)", parameter.raw_line).group(1)
        if re.search(r"\bparameter\(([01])\)", parameter.raw_line)
        else None
        for parameter in parameters
    } != {"0", "1"}:
        return False
    scalar = lambda instruction: (
        len(instruction.result_shapes) == 1
        and instruction.result_shapes[0].dtype == dtype
        and instruction.result_shapes[0].dimensions == ()
    )
    add = adds[0]
    return (
        all(scalar(parameter) for parameter in parameters)
        and scalar(add)
        and add.raw_line.startswith("ROOT ")
        and set(add.operand_names) == {parameter.name for parameter in parameters}
        and len(add.operand_names) == 2
    )


def validate_ws32_repeated_hlo(
    hlo_text: str,
    *,
    kind: str,
    hidden_size: int,
    dense_intermediate_size: int | None = None,
    moe_intermediate_size: int | None = None,
    top_k: int | None = None,
    expected_result_dtype: str | None = "f32",
) -> Ws32HloReport:
    """Validate exact WS32 subgroup geometry for a dense or routed body."""

    if kind not in {"dense", "moe"}:
        raise ValueError("WS32 HLO kind must be 'dense' or 'moe'")
    if expected_result_dtype not in {None, "bf16", "f32"}:
        raise ValueError("WS32 result dtype must be BF16, F32, or unspecified")
    contract = Ws32MeshContract()
    if hidden_size <= 0 or hidden_size % contract.feature_axis_size:
        raise ValueError("WS32 hidden size must divide over feature axis")
    if kind == "dense":
        if (
            dense_intermediate_size is None
            or dense_intermediate_size <= 0
            or dense_intermediate_size % contract.expert_axis_size
        ):
            raise ValueError("WS32 dense intermediate size is invalid")
        expected_feature = 1
        expected_expert = 1
        expected_feature_shape = (
            2,
            1,
            dense_intermediate_size // contract.expert_axis_size,
        )
        scope = "greenfield_ws32_dense"
    else:
        if (
            moe_intermediate_size is None
            or moe_intermediate_size <= 0
            or top_k is None
            or top_k <= 0
        ):
            raise ValueError("WS32 MoE geometry is incomplete")
        expected_feature = top_k + 1
        expected_expert = 1
        expected_feature_shape = (2, 1, moe_intermediate_size)
        scope = "greenfield_ws32_moe"
    expected_expert_shape = (
        1,
        hidden_size // contract.feature_axis_size,
    )

    module = parse_hlo_module(hlo_text)
    violations: list[str] = []
    collectives = module.collectives
    if any(item.raw_opcode != "all-reduce" for item in collectives):
        violations.append("only synchronous all-reduce is permitted")
    all_reduces = tuple(
        item for item in collectives if item.raw_opcode == "all-reduce"
    )
    feature_reduces = []
    expert_reduces = []
    for item in all_reduces:
        if not item.use_global_device_ids:
            violations.append(f"{item.name} omits global device ids")
        op_name = item.op_name or ""
        if scope not in op_name:
            violations.append(f"{item.name} is outside exact {scope} scope")
        family = _group_family(item)
        if family == WS32_FEATURE_AXIS:
            feature_reduces.append(item)
            expected_shape = expected_feature_shape
        elif family == WS32_EXPERT_AXIS:
            expert_reduces.append(item)
            expected_shape = expected_expert_shape
        else:
            violations.append(f"{item.name} uses an unknown replica group")
            continue
        if len(item.result_shapes) != 1 or (
            item.result_shapes[0].dimensions != expected_shape
        ):
            violations.append(
                f"{item.name} result shape drifted from {expected_shape}"
            )
        if not item.operand_shapes or (
            item.operand_shapes[0].dimensions != expected_shape
        ):
            violations.append(
                f"{item.name} operand shape drifted from {expected_shape}"
            )
        if len(item.operand_shapes) != 1 or (
            item.operand_shapes[0].dtype != "f32"
        ):
            violations.append(f"{item.name} operand dtype drifted from f32")
        if expected_result_dtype is not None and (
            len(item.result_shapes) != 1
            or item.result_shapes[0].dtype != expected_result_dtype
        ):
            violations.append(
                f"{item.name} result dtype drifted from {expected_result_dtype}"
            )
        if not _exact_scalar_add_reducer(
            item, module_instructions=module.instructions
        ):
            violations.append(f"{item.name} reducer is not exact scalar add")
    if len(feature_reduces) != expected_feature:
        violations.append(
            f"expected {expected_feature} feature reductions, "
            f"found {len(feature_reduces)}"
        )
    if len(expert_reduces) != expected_expert:
        violations.append(
            f"expected {expected_expert} expert reductions, "
            f"found {len(expert_reduces)}"
        )
    if any(item.maximum_group_size >= 32 for item in collectives):
        violations.append("full-pod repeated collective is forbidden")
    return Ws32HloReport(
        kind=kind,
        all_reduce_count=len(all_reduces),
        feature_reduce_count=len(feature_reduces),
        expert_reduce_count=len(expert_reduces),
        f32_operand_reduce_count=sum(
            len(item.operand_shapes) == 1
            and item.operand_shapes[0].dtype == "f32"
            for item in all_reduces
        ),
        bf16_result_reduce_count=sum(
            len(item.result_shapes) == 1
            and item.result_shapes[0].dtype == "bf16"
            for item in all_reduces
        ),
        f32_result_reduce_count=sum(
            len(item.result_shapes) == 1
            and item.result_shapes[0].dtype == "f32"
            for item in all_reduces
        ),
        maximum_group_size=max(
            (item.maximum_group_size for item in collectives), default=0
        ),
        violations=tuple(violations),
    )
