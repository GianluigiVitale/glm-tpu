"""Declarative full-checkpoint placement rules for the WS32_2D decoder.

The bounded WS32 packer proves one sparse layer.  This module extends only
the ownership calculation to every base-model source tensor.  It is payload
free: callers may stream each returned source slice into the named final-owner
tensor without constructing the model or a global checkpoint array.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
import re
from typing import Any, Iterable, Iterator

from glm_tpu.exceptions import PlanValidationError
from glm_tpu.model_loader.source_inventory import SourceInventory, SourceTensor
from glm_tpu.config.model import ModelGeometry
from glm_tpu.config.parallel import EXPERT_AXIS, FEATURE_AXIS


_ROUTED = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
_SHARED = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
_DENSE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\."
    r"(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
)
_ROUTER = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.gate\."
    r"(weight|e_score_correction_bias)$"
)

_DTYPE_BYTES = {"F8_E4M3": 1, "U8": 1, "BF16": 2, "F32": 4}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _slot(expert: int, feature: int) -> int:
    if not 0 <= expert < 8 or not 0 <= feature < 4:
        raise PlanValidationError("WS32 logical coordinates are out of range")
    return expert * 4 + feature


def _fp8_contract(
    shape: tuple[int, int], geometry: ModelGeometry
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    block_rows, block_columns = geometry.fp8_block_shape
    rows, columns = shape
    return (
        ("F8_E4M3", shape),
        (
            "F32",
            (
                (rows + block_rows - 1) // block_rows,
                (columns + block_columns - 1) // block_columns,
            ),
        ),
    )


def _require_source_contract(
    source: SourceTensor,
    geometry: ModelGeometry,
) -> None:
    """Bind every source role to the exact GLM-5.2 base-model geometry."""

    layer = source.layer_id
    if layer is not None and layer >= geometry.num_layers:
        return
    expected: tuple[str, tuple[int, ...]] | None = None
    routed = _ROUTED.fullmatch(source.name)
    shared = _SHARED.fullmatch(source.name)
    dense = _DENSE.fullmatch(source.name)
    router = _ROUTER.fullmatch(source.name)
    if routed is not None:
        layer_text, expert_text, projection, role = routed.groups()
        layer = int(layer_text)
        expert = int(expert_text)
        if geometry.mlp_layer_types[layer] != "sparse":
            raise PlanValidationError("WS32 routed expert appears in a dense layer")
        if not 0 <= expert < geometry.num_routed_experts:
            raise PlanValidationError("WS32 routed expert id is out of range")
        weight_shape = (
            (geometry.hidden_size, geometry.moe_intermediate_size)
            if projection == "down_proj"
            else (geometry.moe_intermediate_size, geometry.hidden_size)
        )
        expected = _fp8_contract(weight_shape, geometry)[
            role == "weight_scale_inv"
        ]
    elif shared is not None:
        layer_text, projection, role = shared.groups()
        layer = int(layer_text)
        if geometry.mlp_layer_types[layer] != "sparse":
            raise PlanValidationError("WS32 shared expert appears in a dense layer")
        weight_shape = (
            (geometry.hidden_size, geometry.moe_intermediate_size)
            if projection == "down_proj"
            else (geometry.moe_intermediate_size, geometry.hidden_size)
        )
        expected = _fp8_contract(weight_shape, geometry)[
            role == "weight_scale_inv"
        ]
    elif dense is not None:
        layer_text, projection, role = dense.groups()
        layer = int(layer_text)
        if geometry.mlp_layer_types[layer] != "dense":
            raise PlanValidationError("WS32 dense MLP appears in a sparse layer")
        weight_shape = (
            (geometry.hidden_size, geometry.dense_intermediate_size)
            if projection == "down_proj"
            else (geometry.dense_intermediate_size, geometry.hidden_size)
        )
        expected = _fp8_contract(weight_shape, geometry)[
            role == "weight_scale_inv"
        ]
    elif router is not None:
        layer_text, role = router.groups()
        layer = int(layer_text)
        if geometry.mlp_layer_types[layer] != "sparse":
            raise PlanValidationError("WS32 router appears in a dense layer")
        expected = (
            ("BF16", (geometry.num_routed_experts, geometry.hidden_size))
            if role == "weight"
            else ("F32", (geometry.num_routed_experts,))
        )
    elif source.name in {"model.embed_tokens.weight", "lm_head.weight"}:
        expected = ("BF16", (geometry.vocab_size, geometry.hidden_size))
    elif source.name == "model.norm.weight" or source.name.endswith(
        (".input_layernorm.weight", ".post_attention_layernorm.weight")
    ):
        expected = ("BF16", (geometry.hidden_size,))
    elif source.name.endswith(".self_attn.q_a_layernorm.weight"):
        expected = ("BF16", (geometry.q_lora_rank,))
    elif source.name.endswith(".self_attn.kv_a_layernorm.weight"):
        expected = ("BF16", (geometry.kv_lora_rank,))
    elif source.name.endswith(
        (".self_attn.indexer.k_norm.bias", ".self_attn.indexer.k_norm.weight")
    ):
        expected = ("BF16", (geometry.dsa_indexer_head_dim,))
    elif source.name.endswith(".self_attn.indexer.weights_proj.weight"):
        expected = ("BF16", (geometry.dsa_indexer_heads, geometry.hidden_size))
    else:
        attention_shapes = {
            ".self_attn.indexer.wk": (
                geometry.dsa_indexer_head_dim,
                geometry.hidden_size,
            ),
            ".self_attn.indexer.wq_b": (
                geometry.dsa_indexer_heads * geometry.dsa_indexer_head_dim,
                geometry.q_lora_rank,
            ),
            ".self_attn.q_a_proj": (geometry.q_lora_rank, geometry.hidden_size),
            ".self_attn.q_b_proj": (
                geometry.attention_heads
                * (geometry.qk_nope_head_dim + geometry.qk_rope_head_dim),
                geometry.q_lora_rank,
            ),
            ".self_attn.kv_a_proj_with_mqa": (
                geometry.kv_lora_rank + geometry.qk_rope_head_dim,
                geometry.hidden_size,
            ),
            ".self_attn.kv_b_proj": (
                geometry.kv_heads
                * (geometry.qk_nope_head_dim + geometry.v_head_dim),
                geometry.kv_lora_rank,
            ),
            ".self_attn.o_proj": (
                geometry.hidden_size,
                geometry.attention_heads * geometry.v_head_dim,
            ),
        }
        for suffix, weight_shape in attention_shapes.items():
            if source.name.endswith(f"{suffix}.weight"):
                expected = _fp8_contract(weight_shape, geometry)[0]
                break
            if source.name.endswith(f"{suffix}.weight_scale_inv"):
                expected = _fp8_contract(weight_shape, geometry)[1]
                break
    if expected is None:
        raise PlanValidationError(
            f"WS32 has no source geometry contract for {source.name!r}"
        )
    if source.layer_id is not None and ".self_attn.indexer." in source.name:
        if geometry.indexer_types[source.layer_id] != "full":
            raise PlanValidationError(
                "WS32 full indexer tensor appears in a shared layer"
            )
    if (source.dtype, source.shape) != expected:
        raise PlanValidationError(
            f"WS32 source geometry drifted for {source.name!r}: "
            f"observed={(source.dtype, source.shape)!r}, expected={expected!r}"
        )


def _slice_bounds(
    shape: tuple[int, ...],
    partitions: tuple[int | None, ...],
    coordinates: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(shape) != len(partitions):
        raise PlanValidationError("WS32 slice rank and partition rank disagree")
    starts: list[int] = []
    stops: list[int] = []
    for dimension, partition_axis in zip(shape, partitions, strict=True):
        if partition_axis is None:
            starts.append(0)
            stops.append(dimension)
            continue
        divisor = (8, 4)[partition_axis]
        coordinate = coordinates[partition_axis]
        if dimension % divisor:
            raise PlanValidationError(
                f"WS32 dimension {dimension} does not divide over {divisor}"
            )
        width = dimension // divisor
        starts.append(coordinate * width)
        stops.append((coordinate + 1) * width)
    return tuple(starts), tuple(stops)


def _runtime_name(source: SourceTensor, *, routed: re.Match[str] | None) -> str:
    name = source.name
    if routed is not None:
        layer, _, projection, role = routed.groups()
        name = f"model.layers.{layer}.mlp.experts.{projection}.{role}"
    if source.dtype == "F8_E4M3" and name.endswith(".weight"):
        return name[: -len(".weight")] + ".weight_bits"
    if name.endswith(".weight_scale_inv"):
        return name[: -len(".weight_scale_inv")] + ".scale_inv"
    return name


@dataclass(frozen=True, slots=True)
class SourcePlacement:
    """One exact source interval placed into one final-owner tensor interval."""

    source_name: str
    source_dtype: str
    source_shape: tuple[int, ...]
    source_starts: tuple[int, ...]
    source_stops: tuple[int, ...]
    slot: int
    expert_coordinate: int
    feature_coordinate: int
    destination_name: str
    destination_dtype: str
    destination_shape: tuple[int, ...]
    destination_starts: tuple[int, ...]
    destination_stops: tuple[int, ...]
    global_shape: tuple[int, ...]
    partition_spec: tuple[str | None, ...]
    transform: str

    def __post_init__(self) -> None:
        tuple_fields = (
            "source_shape",
            "source_starts",
            "source_stops",
            "destination_shape",
            "destination_starts",
            "destination_stops",
            "global_shape",
            "partition_spec",
        )
        for field in tuple_fields:
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if self.source_dtype not in _DTYPE_BYTES or (
            self.destination_dtype not in _DTYPE_BYTES
        ):
            raise PlanValidationError("WS32 placement dtype is unsupported")
        if self.source_dtype == "F8_E4M3":
            if self.destination_dtype != "U8" or self.transform != "fp8_bits":
                raise PlanValidationError("WS32 FP8 storage must remain exact U8 bits")
        elif self.source_dtype != self.destination_dtype or self.transform != "identity":
            raise PlanValidationError("WS32 non-FP8 placement must be identity")
        if self.slot != _slot(self.expert_coordinate, self.feature_coordinate):
            raise PlanValidationError("WS32 placement slot/coordinate mismatch")
        for shape, starts, stops, label in (
            (
                self.source_shape,
                self.source_starts,
                self.source_stops,
                "source",
            ),
            (
                self.destination_shape,
                self.destination_starts,
                self.destination_stops,
                "destination",
            ),
        ):
            if len(shape) != len(starts) or len(shape) != len(stops):
                raise PlanValidationError(f"WS32 {label} slice rank drifted")
            if any(
                not 0 <= start < stop <= dimension
                for dimension, start, stop in zip(
                    shape, starts, stops, strict=True
                )
            ):
                raise PlanValidationError(f"WS32 {label} slice is out of range")
        if self.source_element_count != self.destination_element_count:
            raise PlanValidationError("WS32 source/destination slice bytes drifted")
        if len(self.global_shape) != len(self.partition_spec):
            raise PlanValidationError("WS32 global shape/spec rank drifted")

    @property
    def source_element_count(self) -> int:
        return prod(
            stop - start
            for start, stop in zip(
                self.source_starts, self.source_stops, strict=True
            )
        )

    @property
    def destination_element_count(self) -> int:
        return prod(
            stop - start
            for start, stop in zip(
                self.destination_starts, self.destination_stops, strict=True
            )
        )

    @property
    def byte_count(self) -> int:
        return self.destination_element_count * _DTYPE_BYTES[self.destination_dtype]

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "destination_dtype": self.destination_dtype,
            "destination_name": self.destination_name,
            "destination_shape": list(self.destination_shape),
            "destination_slice": [
                list(self.destination_starts),
                list(self.destination_stops),
            ],
            "expert_coordinate": self.expert_coordinate,
            "feature_coordinate": self.feature_coordinate,
            "global_shape": list(self.global_shape),
            "partition_spec": list(self.partition_spec),
            "slot": self.slot,
            "source_dtype": self.source_dtype,
            "source_name": self.source_name,
            "source_shape": list(self.source_shape),
            "source_slice": [list(self.source_starts), list(self.source_stops)],
            "transform": self.transform,
        }


@dataclass(frozen=True, slots=True)
class RuntimePlacementReport:
    source_inventory_sha256: str
    geometry_sha256: str
    source_tensor_count: int
    source_bytes: int
    placement_count: int
    destination_tensor_count: int
    packed_bytes: int
    bytes_by_slot: tuple[int, ...]
    placement_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bytes_by_slot", tuple(self.bytes_by_slot))
        if len(self.bytes_by_slot) != 32 or len(set(self.bytes_by_slot)) != 1:
            raise PlanValidationError("WS32 runtime placements must balance 32 slots")
        if sum(self.bytes_by_slot) != self.packed_bytes:
            raise PlanValidationError("WS32 runtime placement bytes do not reconcile")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes_by_slot": list(self.bytes_by_slot),
            "destination_tensor_count": self.destination_tensor_count,
            "geometry_sha256": self.geometry_sha256,
            "packed_bytes": self.packed_bytes,
            "placement_count": self.placement_count,
            "placement_sha256": self.placement_sha256,
            "source_bytes": self.source_bytes,
            "source_inventory_sha256": self.source_inventory_sha256,
            "source_tensor_count": self.source_tensor_count,
        }


def _placement(
    source: SourceTensor,
    *,
    expert: int,
    feature: int,
    source_partitions: tuple[int | None, ...],
    global_shape: tuple[int, ...] | None = None,
    partition_spec: tuple[str | None, ...] | None = None,
    destination_name: str | None = None,
    destination_shape: tuple[int, ...] | None = None,
    destination_starts: tuple[int, ...] | None = None,
    destination_stops: tuple[int, ...] | None = None,
) -> SourcePlacement:
    coordinates = (expert, feature)
    source_starts, source_stops = _slice_bounds(
        source.shape, source_partitions, coordinates
    )
    selected_shape = tuple(
        stop - start
        for start, stop in zip(source_starts, source_stops, strict=True)
    )
    destination_shape = selected_shape if destination_shape is None else destination_shape
    destination_starts = (
        (0,) * len(destination_shape)
        if destination_starts is None
        else destination_starts
    )
    destination_stops = (
        destination_shape if destination_stops is None else destination_stops
    )
    if global_shape is None:
        global_shape = source.shape
    if partition_spec is None:
        partition_spec = tuple(
            None
            if axis is None
            else (EXPERT_AXIS, FEATURE_AXIS)[axis]
            for axis in source_partitions
        )
    return SourcePlacement(
        source_name=source.name,
        source_dtype=source.dtype,
        source_shape=source.shape,
        source_starts=source_starts,
        source_stops=source_stops,
        slot=_slot(expert, feature),
        expert_coordinate=expert,
        feature_coordinate=feature,
        destination_name=(
            _runtime_name(source, routed=None)
            if destination_name is None
            else destination_name
        ),
        destination_dtype="U8" if source.dtype == "F8_E4M3" else source.dtype,
        destination_shape=destination_shape,
        destination_starts=destination_starts,
        destination_stops=destination_stops,
        global_shape=global_shape,
        partition_spec=partition_spec,
        transform="fp8_bits" if source.dtype == "F8_E4M3" else "identity",
    )


def placements_for_source_tensor(
    source: SourceTensor,
    geometry: ModelGeometry,
) -> tuple[SourcePlacement, ...]:
    """Return all exact final-owner intervals for one base-model source leaf."""

    if source.layer_id is not None and source.layer_id >= geometry.num_layers:
        return ()
    _require_source_contract(source, geometry)
    routed = _ROUTED.fullmatch(source.name)
    shared = _SHARED.fullmatch(source.name)
    dense = _DENSE.fullmatch(source.name)
    router = _ROUTER.fullmatch(source.name)
    if ".mlp." in source.name:
        if sum(value is not None for value in (routed, shared, dense, router)) != 1:
            raise PlanValidationError(
                f"WS32 has no unique MLP placement for {source.name!r}"
            )
        if routed is not None:
            _, expert_text, projection, _ = routed.groups()
            expert_id = int(expert_text)
            expert = expert_id // (geometry.num_routed_experts // 8)
            local_expert = expert_id % (geometry.num_routed_experts // 8)
            feature_axis = 0 if projection == "down_proj" else 1
            global_shape = (geometry.num_routed_experts, *source.shape)
            partition_spec = (
                EXPERT_AXIS,
                *(FEATURE_AXIS if axis == feature_axis else None for axis in range(2)),
            )
            destination_name = _runtime_name(source, routed=routed)
            placements = []
            for feature in range(4):
                source_partitions = tuple(
                    1 if axis == feature_axis else None for axis in range(2)
                )
                starts, stops = _slice_bounds(
                    source.shape, source_partitions, (expert, feature)
                )
                selected_shape = tuple(
                    stop - start
                    for start, stop in zip(starts, stops, strict=True)
                )
                destination_shape = (
                    geometry.num_routed_experts // 8,
                    *selected_shape,
                )
                destination_starts = (local_expert, *([0] * len(selected_shape)))
                destination_stops = (
                    local_expert + 1,
                    *selected_shape,
                )
                placements.append(
                    _placement(
                        source,
                        expert=expert,
                        feature=feature,
                        source_partitions=source_partitions,
                        global_shape=global_shape,
                        partition_spec=partition_spec,
                        destination_name=destination_name,
                        destination_shape=destination_shape,
                        destination_starts=destination_starts,
                        destination_stops=destination_stops,
                    )
                )
            return tuple(placements)
        if shared is not None:
            _, projection, _ = shared.groups()
            feature_axis = 0 if projection == "down_proj" else 1
            source_partitions = tuple(
                1 if axis == feature_axis else None for axis in range(2)
            )
            return tuple(
                _placement(
                    source,
                    expert=expert,
                    feature=feature,
                    source_partitions=source_partitions,
                )
                for expert in range(8)
                for feature in range(4)
            )
        if dense is not None:
            _, projection, _ = dense.groups()
            partitions = (1, 0) if projection == "down_proj" else (0, 1)
            return tuple(
                _placement(
                    source,
                    expert=expert,
                    feature=feature,
                    source_partitions=partitions,
                )
                for expert in range(8)
                for feature in range(4)
            )
        assert router is not None
        role = router.group(2)
        partitions = (0, 1) if role == "weight" else (0,)
        return tuple(
            _placement(
                source,
                expert=expert,
                feature=feature,
                source_partitions=partitions,
            )
            for expert in range(8)
            for feature in range(4)
        )

    name = source.name
    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        partitions = (0, 1)
    elif name == "model.norm.weight" or name.endswith(
        (".input_layernorm.weight", ".post_attention_layernorm.weight")
    ):
        partitions = (1,)
    elif name.endswith(
        (
            ".self_attn.q_a_layernorm.weight",
            ".self_attn.kv_a_layernorm.weight",
            ".self_attn.indexer.k_norm.bias",
            ".self_attn.indexer.k_norm.weight",
        )
    ):
        partitions = tuple(None for _ in source.shape)
    elif name.endswith(
        (
            ".self_attn.indexer.wk.weight",
            ".self_attn.indexer.wk.weight_scale_inv",
            ".self_attn.kv_a_proj_with_mqa.weight",
            ".self_attn.kv_a_proj_with_mqa.weight_scale_inv",
            ".self_attn.q_a_proj.weight",
            ".self_attn.q_a_proj.weight_scale_inv",
        )
    ):
        partitions = (None, 1)
    elif name.endswith(".self_attn.indexer.weights_proj.weight"):
        partitions = (0, 1)
    elif name.endswith(
        (
            ".self_attn.indexer.wq_b.weight",
            ".self_attn.indexer.wq_b.weight_scale_inv",
            ".self_attn.kv_b_proj.weight",
            ".self_attn.kv_b_proj.weight_scale_inv",
            ".self_attn.q_b_proj.weight",
            ".self_attn.q_b_proj.weight_scale_inv",
        )
    ):
        partitions = (0, None)
    elif name.endswith(
        (
            ".self_attn.o_proj.weight",
            ".self_attn.o_proj.weight_scale_inv",
        )
    ):
        partitions = (1, 0)
    else:
        raise PlanValidationError(
            f"WS32 has no non-MLP placement rule for {source.name!r}"
        )
    return tuple(
        _placement(
            source,
            expert=expert,
            feature=feature,
            source_partitions=partitions,
        )
        for expert in range(8)
        for feature in range(4)
    )


def iter_source_placements(
    tensors: Iterable[SourceTensor],
    geometry: ModelGeometry,
) -> Iterator[SourcePlacement]:
    for source in tensors:
        yield from placements_for_source_tensor(source, geometry)


def build_runtime_placement_report(
    inventory: SourceInventory,
    geometry: ModelGeometry,
) -> RuntimePlacementReport:
    """Hash and reconcile every declarative base-checkpoint placement."""

    if inventory.model_id != geometry.model_id:
        raise PlanValidationError("WS32 inventory/model ids disagree")
    base_tensors = tuple(
        tensor
        for tensor in inventory.tensors
        if tensor.layer_id is None or tensor.layer_id < geometry.num_layers
    )
    source_bytes = sum(tensor.byte_count for tensor in base_tensors)
    bytes_by_slot = [0] * 32
    placement_count = 0
    digest = sha256()
    destinations: dict[tuple[int, str], list[SourcePlacement]] = {}
    for placement in iter_source_placements(base_tensors, geometry):
        bytes_by_slot[placement.slot] += placement.byte_count
        placement_count += 1
        destinations.setdefault(
            (placement.slot, placement.destination_name), []
        ).append(placement)
        digest.update(_canonical_json(placement.to_dict()).encode("utf-8"))
        digest.update(b"\n")
    for (slot, destination_name), placements in destinations.items():
        first = placements[0]
        invariant = (
            first.destination_dtype,
            first.destination_shape,
            first.global_shape,
            first.partition_spec,
        )
        if any(
            (
                item.destination_dtype,
                item.destination_shape,
                item.global_shape,
                item.partition_spec,
            )
            != invariant
            for item in placements[1:]
        ):
            raise PlanValidationError(
                f"WS32 destination schema disagrees for slot {slot} "
                f"tensor {destination_name!r}"
            )
        if len(placements) == 1:
            item = placements[0]
            if item.destination_starts != (0,) * len(item.destination_shape) or (
                item.destination_stops != item.destination_shape
            ):
                raise PlanValidationError(
                    f"WS32 singleton destination {destination_name!r} is incomplete"
                )
            continue
        if len(placements) != geometry.num_routed_experts // 8:
            raise PlanValidationError(
                f"WS32 aggregate destination {destination_name!r} has wrong arity"
            )
        ordered = sorted(placements, key=lambda item: item.destination_starts)
        for local_expert, item in enumerate(ordered):
            if _ROUTED.fullmatch(item.source_name) is None:
                raise PlanValidationError(
                    f"WS32 non-routed destination {destination_name!r} overlaps"
                )
            if item.destination_starts != (
                local_expert,
                *(0 for _ in item.destination_shape[1:]),
            ) or item.destination_stops != (
                local_expert + 1,
                *item.destination_shape[1:],
            ):
                raise PlanValidationError(
                    f"WS32 routed destination {destination_name!r} has a gap or overlap"
                )
    return RuntimePlacementReport(
        source_inventory_sha256=inventory.inventory_sha256,
        geometry_sha256=geometry.geometry_hash,
        source_tensor_count=len(base_tensors),
        source_bytes=source_bytes,
        placement_count=placement_count,
        destination_tensor_count=len(destinations),
        packed_bytes=sum(bytes_by_slot),
        bytes_by_slot=tuple(bytes_by_slot),
        placement_sha256=digest.hexdigest(),
    )
