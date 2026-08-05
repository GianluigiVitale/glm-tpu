"""Executable-ready, uniformly padded decoder-weight layout.

The protected Gate-B checkpoint is already in final physical ownership, but
it retains source-leaf granularity.  A single global SPMD decoder needs the
same input tree on every partition.  This module defines the offline runtime
derivative: one tensor per padded stage slot and weight role, with zeros only
for explicitly recorded absent slots.  It never moves a real weight to a new
owner and makes every padding byte part of the content-addressed contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from hashlib import sha256
import json
from operator import mul
from typing import Any, Mapping

from ..errors import PlanValidationError
from ..types import ExecutionPlan
from .schedule import LayerExecution, PipelineSchedule, StageExecution


_DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1}
_SLOT_KINDS = frozenset(("layer", "full_indexer", "dense", "sparse", "global"))
_VALUE_CLASSES = frozenset(("parameter", "fp8_weight", "fp8_scale"))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _product(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


def _scale_shape(
    weight_shape: tuple[int, int], block_shape: tuple[int, int]
) -> tuple[int, int]:
    return tuple(
        (dimension + block - 1) // block
        for dimension, block in zip(weight_shape, block_shape, strict=True)
    )


@dataclass(frozen=True, slots=True)
class RuntimeTensorSpec:
    """One uniform per-chip tensor in the executable input tree."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    slot_kind: str
    slot_index: int
    value_class: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if not self.name or self.dtype not in _DTYPE_BYTES:
            raise PlanValidationError("runtime tensor name/dtype is invalid")
        if not self.shape or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.shape
        ):
            raise PlanValidationError("runtime tensor shape must be positive")
        if self.slot_kind not in _SLOT_KINDS:
            raise PlanValidationError("runtime tensor slot kind is invalid")
        if (
            not isinstance(self.slot_index, int)
            or isinstance(self.slot_index, bool)
            or self.slot_index < 0
        ):
            raise PlanValidationError("runtime tensor slot must be non-negative")
        if self.value_class not in _VALUE_CLASSES:
            raise PlanValidationError("runtime tensor value class is invalid")
        expected_class = {
            "F8_E4M3": "fp8_weight",
        }.get(self.dtype)
        if expected_class is not None and self.value_class != expected_class:
            raise PlanValidationError("FP8 runtime tensor is not marked as a weight")
        if self.value_class == "fp8_scale" and self.dtype != "F32":
            raise PlanValidationError("FP8 scale runtime tensor must be F32")

    @property
    def byte_count(self) -> int:
        return _product(self.shape) * _DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "dtype": self.dtype,
            "name": self.name,
            "shape": list(self.shape),
            "slot_index": self.slot_index,
            "slot_kind": self.slot_kind,
            "value_class": self.value_class,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSourceLeaf:
    """Expected source leaf already present in one final-owner file."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if not self.name or self.dtype not in _DTYPE_BYTES:
            raise PlanValidationError("runtime source name/dtype is invalid")
        if not self.shape or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.shape
        ):
            raise PlanValidationError("runtime source shape must be positive")

    @property
    def byte_count(self) -> int:
        return _product(self.shape) * _DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "dtype": self.dtype,
            "name": self.name,
            "shape": list(self.shape),
        }


@dataclass(frozen=True, slots=True)
class DeviceRuntimeTensor:
    """One actual source binding or one complete zero-padding tensor."""

    spec: RuntimeTensorSpec
    sources: tuple[RuntimeSourceLeaf, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        if not self.sources:
            return
        if any(source.dtype != self.spec.dtype for source in self.sources):
            raise PlanValidationError("runtime source dtype disagrees with tensor")
        if len(self.sources) == 1:
            expected_shape = self.sources[0].shape
        else:
            leaf_shape = self.sources[0].shape
            if any(source.shape != leaf_shape for source in self.sources):
                raise PlanValidationError("runtime concatenation source shapes drifted")
            expected_shape = (len(self.sources), *leaf_shape)
        if expected_shape != self.spec.shape:
            raise PlanValidationError(
                f"runtime sources do not fill {self.spec.name}: "
                f"expected={self.spec.shape} observed={expected_shape}"
            )
        if len({source.name for source in self.sources}) != len(self.sources):
            raise PlanValidationError("runtime tensor repeats a source leaf")
        if self.source_byte_count != self.spec.byte_count:
            raise PlanValidationError("runtime tensor source bytes do not reconcile")

    @property
    def is_padding(self) -> bool:
        return not self.sources

    @property
    def source_byte_count(self) -> int:
        return sum(source.byte_count for source in self.sources)

    @property
    def padding_bytes(self) -> int:
        return self.spec.byte_count if self.is_padding else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "padding": self.is_padding,
            "padding_bytes": self.padding_bytes,
            "runtime": self.spec.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass(frozen=True, slots=True)
class DeviceRuntimeWeightLayout:
    """Complete executable-ready file for one stage-local physical slot."""

    stage_id: int
    device_slot: int
    device_id: int
    tensors: tuple[DeviceRuntimeTensor, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tensors", tuple(self.tensors))
        for field in ("stage_id", "device_slot", "device_id"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PlanValidationError(f"runtime {field} must be non-negative")
        names = tuple(tensor.spec.name for tensor in self.tensors)
        if len(set(names)) != len(names):
            raise PlanValidationError("runtime device tensor names are duplicate")
        source_names = [
            source.name for tensor in self.tensors for source in tensor.sources
        ]
        if len(set(source_names)) != len(source_names):
            raise PlanValidationError("runtime device consumes a source leaf twice")

    @property
    def runtime_bytes(self) -> int:
        return sum(tensor.spec.byte_count for tensor in self.tensors)

    @property
    def source_bytes(self) -> int:
        return sum(tensor.source_byte_count for tensor in self.tensors)

    @property
    def padding_bytes(self) -> int:
        return sum(tensor.padding_bytes for tensor in self.tensors)

    @property
    def source_leaf_count(self) -> int:
        return sum(len(tensor.sources) for tensor in self.tensors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_slot": self.device_slot,
            "padding_bytes": self.padding_bytes,
            "runtime_bytes": self.runtime_bytes,
            "source_bytes": self.source_bytes,
            "source_leaf_count": self.source_leaf_count,
            "stage_id": self.stage_id,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }


@dataclass(frozen=True, slots=True)
class DecoderRuntimeWeightLayout:
    """Content-addressed uniform input layout for the global decoder."""

    plan_hash: str
    schedule_hash: str
    specs: tuple[RuntimeTensorSpec, ...]
    devices: tuple[DeviceRuntimeWeightLayout, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", tuple(self.specs))
        object.__setattr__(self, "devices", tuple(self.devices))
        if len(self.plan_hash) != 64 or len(self.schedule_hash) != 64:
            raise PlanValidationError("runtime weight hashes must be SHA-256")
        spec_names = tuple(spec.name for spec in self.specs)
        if len(set(spec_names)) != len(spec_names):
            raise PlanValidationError("runtime weight specs are duplicate")
        expected = tuple((spec.name, spec.to_dict()) for spec in self.specs)
        for device in self.devices:
            observed = tuple(
                (tensor.spec.name, tensor.spec.to_dict())
                for tensor in device.tensors
            )
            if observed != expected:
                raise PlanValidationError("runtime input tree differs by device")
            if device.source_bytes + device.padding_bytes != device.runtime_bytes:
                raise PlanValidationError("runtime device bytes do not reconcile")
        if len({device.device_id for device in self.devices}) != len(self.devices):
            raise PlanValidationError("runtime device ids are duplicate")
        if len({device.runtime_bytes for device in self.devices}) != 1:
            raise PlanValidationError("runtime bytes are not uniform by device")

    @property
    def runtime_bytes_per_chip(self) -> int:
        return self.devices[0].runtime_bytes

    @property
    def maximum_padding_bytes_per_chip(self) -> int:
        return max(device.padding_bytes for device in self.devices)

    @property
    def minimum_padding_bytes_per_chip(self) -> int:
        return min(device.padding_bytes for device in self.devices)

    @property
    def source_leaf_count(self) -> int:
        return sum(device.source_leaf_count for device in self.devices)

    @property
    def layout_hash(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "devices": [device.to_dict() for device in self.devices],
            "maximum_padding_bytes_per_chip": self.maximum_padding_bytes_per_chip,
            "minimum_padding_bytes_per_chip": self.minimum_padding_bytes_per_chip,
            "plan_hash": self.plan_hash,
            "runtime_bytes_per_chip": self.runtime_bytes_per_chip,
            "schedule_hash": self.schedule_hash,
            "source_leaf_count": self.source_leaf_count,
            "specs": [spec.to_dict() for spec in self.specs],
        }


@dataclass(frozen=True, slots=True)
class _Definition:
    spec: RuntimeTensorSpec
    owner_kind: str
    source_suffix: str
    expert_projection: str | None = None
    source_leaf_shape: tuple[int, ...] | None = None
    global_stage: int | None = None


def _layers_for_kind(stage: StageExecution, kind: str) -> tuple[LayerExecution, ...]:
    if kind == "layer":
        return stage.layers
    if kind == "full_indexer":
        return tuple(layer for layer in stage.layers if layer.indexer_kind == "full")
    if kind == "dense":
        return tuple(layer for layer in stage.layers if layer.mlp_kind == "dense")
    if kind == "sparse":
        return tuple(layer for layer in stage.layers if layer.mlp_kind == "sparse")
    raise PlanValidationError(f"unknown runtime owner kind {kind!r}")


def build_decoder_runtime_weight_layout(
    plan: ExecutionPlan,
    schedule: PipelineSchedule,
) -> DecoderRuntimeWeightLayout:
    """Derive every executable tensor, source binding, and padding byte."""

    if schedule.plan_hash != plan.plan_hash:
        raise PlanValidationError("runtime weights schedule belongs to another plan")
    geometry = plan.geometry
    local_size = plan.local_parallel_size
    divisibility = {
        "attention heads": geometry.attention_heads,
        "dense intermediate": geometry.dense_intermediate_size,
        "indexer heads": geometry.dsa_indexer_heads,
        "routed experts": geometry.num_routed_experts,
        "shared intermediate": (
            geometry.num_shared_experts * geometry.moe_intermediate_size
        ),
        "vocabulary": geometry.vocab_size,
    }
    for name, value in divisibility.items():
        if value % local_size:
            raise PlanValidationError(f"{name} does not divide over local stage")

    hidden = geometry.hidden_size
    q_rank = geometry.q_lora_rank
    kv_rank = geometry.kv_lora_rank
    rope = geometry.qk_rope_head_dim
    qk_head = geometry.qk_nope_head_dim + rope
    local_heads = geometry.attention_heads // local_size
    local_index_heads = geometry.dsa_indexer_heads // local_size
    local_experts = geometry.num_routed_experts // local_size
    local_dense = geometry.dense_intermediate_size // local_size
    shared_width = (
        geometry.num_shared_experts * geometry.moe_intermediate_size
    )
    local_shared = shared_width // local_size
    block_shape = geometry.fp8_block_shape
    definitions: list[_Definition] = []

    def add(
        name: str,
        dtype: str,
        shape: tuple[int, ...],
        slot_kind: str,
        slot: int,
        suffix: str,
        *,
        value_class: str = "parameter",
        expert_projection: str | None = None,
        source_leaf_shape: tuple[int, ...] | None = None,
        global_stage: int | None = None,
    ) -> None:
        definitions.append(
            _Definition(
                RuntimeTensorSpec(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    slot_kind=slot_kind,
                    slot_index=slot,
                    value_class=value_class,
                ),
                owner_kind=slot_kind,
                source_suffix=suffix,
                expert_projection=expert_projection,
                source_leaf_shape=source_leaf_shape,
                global_stage=global_stage,
            )
        )

    def add_fp8_pair(
        name: str,
        shape: tuple[int, ...],
        slot_kind: str,
        slot: int,
        suffix: str,
        *,
        expert_projection: str | None = None,
        source_leaf_shape: tuple[int, int] | None = None,
    ) -> None:
        leaf_shape = shape if source_leaf_shape is None else source_leaf_shape
        if len(leaf_shape) != 2:
            raise PlanValidationError("FP8 source leaf must be a matrix")
        scale_leaf_shape = _scale_shape(leaf_shape, block_shape)
        scale_shape = (
            scale_leaf_shape
            if expert_projection is None
            else (local_experts, *scale_leaf_shape)
        )
        add(
            f"{name}.weight_bits",
            "F8_E4M3",
            shape,
            slot_kind,
            slot,
            f"{suffix}.weight",
            value_class="fp8_weight",
            expert_projection=expert_projection,
            source_leaf_shape=leaf_shape,
        )
        add(
            f"{name}.scale_inv",
            "F32",
            scale_shape,
            slot_kind,
            slot,
            f"{suffix}.weight_scale_inv",
            value_class="fp8_scale",
            expert_projection=expert_projection,
            source_leaf_shape=scale_leaf_shape,
        )

    for slot in range(max(stage.layer_count for stage in schedule.stages)):
        base = f"attention.slot_{slot:02d}"
        add(f"{base}.input_norm", "BF16", (hidden,), "layer", slot, "input_layernorm.weight")
        add(f"{base}.post_norm", "BF16", (hidden,), "layer", slot, "post_attention_layernorm.weight")
        add_fp8_pair(f"{base}.q_a", (q_rank, hidden), "layer", slot, "self_attn.q_a_proj")
        add(f"{base}.q_a_norm", "BF16", (q_rank,), "layer", slot, "self_attn.q_a_layernorm.weight")
        add_fp8_pair(
            f"{base}.q_b",
            (local_heads * qk_head, q_rank),
            "layer",
            slot,
            "self_attn.q_b_proj",
        )
        add_fp8_pair(
            f"{base}.kv_a",
            (kv_rank + rope, hidden),
            "layer",
            slot,
            "self_attn.kv_a_proj_with_mqa",
        )
        add(f"{base}.kv_a_norm", "BF16", (kv_rank,), "layer", slot, "self_attn.kv_a_layernorm.weight")
        add_fp8_pair(
            f"{base}.kv_b",
            (
                local_heads
                * (geometry.qk_nope_head_dim + geometry.v_head_dim),
                kv_rank,
            ),
            "layer",
            slot,
            "self_attn.kv_b_proj",
        )
        add_fp8_pair(
            f"{base}.o",
            (hidden, local_heads * geometry.v_head_dim),
            "layer",
            slot,
            "self_attn.o_proj",
        )

    maximum_full = max(
        sum(layer.indexer_kind == "full" for layer in stage.layers)
        for stage in schedule.stages
    )
    for slot in range(maximum_full):
        base = f"indexer.slot_{slot:02d}"
        add_fp8_pair(
            f"{base}.wq_b",
            (local_index_heads * geometry.dsa_indexer_head_dim, q_rank),
            "full_indexer",
            slot,
            "self_attn.indexer.wq_b",
        )
        add_fp8_pair(
            f"{base}.wk",
            (geometry.dsa_indexer_head_dim, hidden),
            "full_indexer",
            slot,
            "self_attn.indexer.wk",
        )
        add(
            f"{base}.key_norm_weight",
            "BF16",
            (geometry.dsa_indexer_head_dim,),
            "full_indexer",
            slot,
            "self_attn.indexer.k_norm.weight",
        )
        add(
            f"{base}.key_norm_bias",
            "BF16",
            (geometry.dsa_indexer_head_dim,),
            "full_indexer",
            slot,
            "self_attn.indexer.k_norm.bias",
        )
        add(
            f"{base}.head_weight",
            "BF16",
            (local_index_heads, hidden),
            "full_indexer",
            slot,
            "self_attn.indexer.weights_proj.weight",
        )

    for slot in range(schedule.maximum_dense_slots):
        base = f"dense.slot_{slot:02d}"
        add_fp8_pair(
            f"{base}.gate", (local_dense, hidden), "dense", slot, "mlp.gate_proj"
        )
        add_fp8_pair(
            f"{base}.up", (local_dense, hidden), "dense", slot, "mlp.up_proj"
        )
        add_fp8_pair(
            f"{base}.down", (hidden, local_dense), "dense", slot, "mlp.down_proj"
        )

    expert_shapes = {
        "gate_proj": (geometry.moe_intermediate_size, hidden),
        "up_proj": (geometry.moe_intermediate_size, hidden),
        "down_proj": (hidden, geometry.moe_intermediate_size),
    }
    shared_shapes = {
        "gate_proj": (local_shared, hidden),
        "up_proj": (local_shared, hidden),
        "down_proj": (hidden, local_shared),
    }
    for slot in range(schedule.maximum_sparse_slots):
        base = f"sparse.slot_{slot:02d}"
        add(
            f"{base}.router_weight",
            "BF16",
            (geometry.num_routed_experts, hidden),
            "sparse",
            slot,
            "mlp.gate.weight",
        )
        add(
            f"{base}.correction_bias",
            "F32",
            (geometry.num_routed_experts,),
            "sparse",
            slot,
            "mlp.gate.e_score_correction_bias",
        )
        for projection, leaf_shape in expert_shapes.items():
            add_fp8_pair(
                f"{base}.experts.{projection}",
                (local_experts, *leaf_shape),
                "sparse",
                slot,
                "",
                expert_projection=projection,
                source_leaf_shape=leaf_shape,
            )
        for projection, shape in shared_shapes.items():
            add_fp8_pair(
                f"{base}.shared.{projection}",
                shape,
                "sparse",
                slot,
                f"mlp.shared_experts.{projection}",
            )

    local_vocab = geometry.vocab_size // local_size
    final_stage = len(schedule.stages) - 1
    add(
        "global.embedding",
        "BF16",
        (local_vocab, hidden),
        "global",
        0,
        "model.embed_tokens.weight",
        global_stage=0,
    )
    add(
        "global.final_norm",
        "BF16",
        (hidden,),
        "global",
        1,
        "model.norm.weight",
        global_stage=final_stage,
    )
    add(
        "global.lm_head",
        "BF16",
        (local_vocab, hidden),
        "global",
        2,
        "lm_head.weight",
        global_stage=final_stage,
    )

    specs = tuple(definition.spec for definition in definitions)
    if len({spec.name for spec in specs}) != len(specs):
        raise PlanValidationError("runtime definition names are duplicate")
    devices = []
    for stage in schedule.stages:
        by_kind = {
            kind: _layers_for_kind(stage, kind)
            for kind in ("layer", "full_indexer", "dense", "sparse")
        }
        for device_slot, device_id in enumerate(stage.assignment.device_ids):
            tensors = []
            for definition in definitions:
                if definition.owner_kind == "global":
                    layer = None
                    live = stage.assignment.stage_id == definition.global_stage
                else:
                    candidates = by_kind[definition.owner_kind]
                    live = definition.spec.slot_index < len(candidates)
                    layer = candidates[definition.spec.slot_index] if live else None
                sources: tuple[RuntimeSourceLeaf, ...]
                if not live:
                    sources = ()
                elif definition.owner_kind == "global":
                    sources = (
                        RuntimeSourceLeaf(
                            definition.source_suffix,
                            definition.spec.dtype,
                            definition.spec.shape,
                        ),
                    )
                elif definition.expert_projection is not None:
                    assert layer is not None
                    assert definition.source_leaf_shape is not None
                    expert_start = device_slot * local_experts
                    suffix = (
                        ".weight_scale_inv"
                        if definition.spec.value_class == "fp8_scale"
                        else ".weight"
                    )
                    sources = tuple(
                        RuntimeSourceLeaf(
                            f"model.layers.{layer.layer_id}.mlp.experts."
                            f"{expert}.{definition.expert_projection}{suffix}",
                            definition.spec.dtype,
                            definition.source_leaf_shape,
                        )
                        for expert in range(
                            expert_start, expert_start + local_experts
                        )
                    )
                else:
                    assert layer is not None
                    sources = (
                        RuntimeSourceLeaf(
                            f"model.layers.{layer.layer_id}."
                            f"{definition.source_suffix}",
                            definition.spec.dtype,
                            definition.spec.shape,
                        ),
                    )
                tensors.append(DeviceRuntimeTensor(definition.spec, sources))
            devices.append(
                DeviceRuntimeWeightLayout(
                    stage_id=stage.assignment.stage_id,
                    device_slot=device_slot,
                    device_id=device_id,
                    tensors=tuple(tensors),
                )
            )
    return DecoderRuntimeWeightLayout(
        plan_hash=plan.plan_hash,
        schedule_hash=schedule.schedule_hash,
        specs=specs,
        devices=tuple(devices),
    )
