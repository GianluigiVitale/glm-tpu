"""Executable-ready, uniformly padded decoder-weight layout.

The protected Gate-B checkpoint is already in final physical ownership, but
it retains source-leaf granularity.  A single global SPMD decoder needs the
same input tree on every partition.  This module defines the offline runtime
derivative: one tensor per padded stage slot and weight role, with zeros only
for explicitly recorded absent slots.  It never moves a real weight to a new
owner and makes every padding byte part of the content-addressed contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import reduce
from hashlib import sha256
from operator import mul
from typing import Any

from ..errors import PlanValidationError
from ..types import ExecutionPlan
from .schedule import (
    LayerExecution,
    PipelineSchedule,
    StageExecution,
    build_pipeline_schedule,
)

_DTYPE_BYTES = {"BF16": 2, "F32": 4, "F8_E4M3": 1}
_SLOT_KINDS = frozenset(("layer", "full_indexer", "dense", "sparse", "global"))
_VALUE_CLASSES = frozenset(("parameter", "fp8_weight", "fp8_scale"))
_RUNTIME_SOURCE_TRANSFORMS = frozenset(
    (
        "identity_concat",
        "identity_runtime_tensor",
        "concat_experts_slice_output_transpose",
        "concat_experts_slice_contraction_transpose",
        "concat_experts_slice_scale_output",
        "concat_experts_slice_scale_contraction",
        "fuse_qkv_a_output_shards",
        "fuse_qkv_a_expanded_scales",
        "pack_dense_gate_up_bits_in_out",
        "pack_dense_gate_up_scales_in_out",
        "pack_dense_down_bits_in_out",
        "pack_dense_down_scales_in_out",
    )
)
COMPLETE_EXPERT_RUNTIME_LAYOUT = "complete_expert_identity"
FEATURE_EXPERT_RUNTIME_LAYOUT = "expert_intermediate_feature_lp4_pallas_kn_v1"
SEPARATE_QKV_A_RUNTIME_LAYOUT = "separate_q_a_kv_a_v1"
FUSED_QKV_A_N82_RUNTIME_LAYOUT = "fused_qkv_a_virtual_tp32_n82_v1"
LEGACY_DENSE_RUNTIME_LAYOUT = "legacy_dense_output_major_v1"
FINAL_DENSE_CONVOLUTION_RUNTIME_LAYOUT = (
    "virtual_tp32_dense_convolution_in_out_v1"
)


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
    source_device_slot: int | None = None
    selected_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if self.selected_shape is not None:
            object.__setattr__(self, "selected_shape", tuple(self.selected_shape))
        if not self.name or self.dtype not in _DTYPE_BYTES:
            raise PlanValidationError("runtime source name/dtype is invalid")
        if not self.shape or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.shape
        ):
            raise PlanValidationError("runtime source shape must be positive")
        if self.source_device_slot is not None and (
            not isinstance(self.source_device_slot, int)
            or isinstance(self.source_device_slot, bool)
            or self.source_device_slot < 0
        ):
            raise PlanValidationError(
                "runtime source device slot must be non-negative"
            )
        if self.selected_shape is not None and (
            not self.selected_shape
            or len(self.selected_shape) != len(self.shape)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in self.selected_shape
            )
        ):
            raise PlanValidationError(
                "runtime selected source shape must match source rank and be positive"
            )

    @property
    def byte_count(self) -> int:
        return _product(self.shape) * _DTYPE_BYTES[self.dtype]

    @property
    def selected_byte_count(self) -> int:
        shape = self.shape if self.selected_shape is None else self.selected_shape
        return _product(shape) * _DTYPE_BYTES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        value = {
            "byte_count": self.byte_count,
            "dtype": self.dtype,
            "name": self.name,
            "shape": list(self.shape),
        }
        if self.source_device_slot is not None:
            value["source_device_slot"] = self.source_device_slot
        if self.selected_shape is not None:
            value["selected_byte_count"] = self.selected_byte_count
            value["selected_shape"] = list(self.selected_shape)
        return value


@dataclass(frozen=True, slots=True)
class DeviceRuntimeTensor:
    """One actual source binding or one complete zero-padding tensor."""

    spec: RuntimeTensorSpec
    sources: tuple[RuntimeSourceLeaf, ...]
    transform: str = "identity_concat"

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        if self.transform not in _RUNTIME_SOURCE_TRANSFORMS:
            raise PlanValidationError("runtime source transform is invalid")
        if not self.sources:
            if self.transform != "identity_concat":
                raise PlanValidationError(
                    "runtime padding cannot declare a source transform"
                )
            return
        if any(source.dtype != self.spec.dtype for source in self.sources):
            raise PlanValidationError("runtime source dtype disagrees with tensor")
        selected_shapes = tuple(
            source.shape
            if source.selected_shape is None
            else source.selected_shape
            for source in self.sources
        )
        if self.transform in ("identity_concat", "identity_runtime_tensor"):
            if any(source.selected_shape is not None for source in self.sources):
                raise PlanValidationError(
                    "identity runtime source cannot select a partial tensor"
                )
            if len(self.sources) == 1:
                expected_shape = self.sources[0].shape
            else:
                leaf_shape = self.sources[0].shape
                if any(source.shape != leaf_shape for source in self.sources):
                    raise PlanValidationError(
                        "runtime concatenation source shapes drifted"
                    )
                expected_shape = (len(self.sources), *leaf_shape)
            if (
                self.transform == "identity_runtime_tensor"
                and len(self.sources) != 1
            ):
                raise PlanValidationError(
                    "runtime tensor identity requires exactly one source"
                )
        elif self.transform in (
            "fuse_qkv_a_output_shards",
            "fuse_qkv_a_expanded_scales",
        ):
            if len(self.spec.shape) != 3:
                raise PlanValidationError(
                    "fused qkv-a destination must have rank three"
                )
            if len(self.sources) != 2 or any(
                source.source_device_slot is None for source in self.sources
            ):
                raise PlanValidationError(
                    "fused qkv-a tensor requires two explicit source owners"
                )
            q_source, kv_source = self.sources
            if q_source.selected_shape is not None or (
                kv_source.selected_shape is not None
            ):
                raise PlanValidationError(
                    "fused qkv-a sources cannot select partial tensors"
                )
            if self.transform == "fuse_qkv_a_output_shards":
                if self.spec.dtype != "F8_E4M3" or any(
                    len(source.shape) != 2 for source in self.sources
                ):
                    raise PlanValidationError(
                        "fused qkv-a weights must be FP8 matrices"
                    )
                if q_source.shape[1] != kv_source.shape[1]:
                    raise PlanValidationError(
                        "fused qkv-a weight contractions disagree"
                    )
                shards, hidden, local_width = self.spec.shape
                if hidden != q_source.shape[1] or (
                    shards * local_width
                    != q_source.shape[0] + kv_source.shape[0]
                ):
                    raise PlanValidationError(
                        "fused qkv-a weight shape does not reconcile"
                    )
            else:
                if self.spec.dtype != "F32" or any(
                    len(source.shape) != 2 for source in self.sources
                ):
                    raise PlanValidationError(
                        "fused qkv-a scales must be FP32 matrices"
                    )
                if q_source.shape[1] != kv_source.shape[1]:
                    raise PlanValidationError(
                        "fused qkv-a scale contractions disagree"
                    )
                shards, scale_rows, local_width = self.spec.shape
                q_width = q_source.shape[0] * 128
                total_width = shards * local_width
                kv_width = total_width - q_width
                if (
                    scale_rows != q_source.shape[1]
                    or kv_width <= 0
                    or (kv_width + 127) // 128 != kv_source.shape[0]
                ):
                    raise PlanValidationError(
                        "fused qkv-a expanded scale shape does not reconcile"
                    )
            expected_shape = self.spec.shape
        elif self.transform in (
            "pack_dense_gate_up_bits_in_out",
            "pack_dense_gate_up_scales_in_out",
            "pack_dense_down_bits_in_out",
            "pack_dense_down_scales_in_out",
        ):
            virtual_shards = 8
            if any(
                source.source_device_slot is None
                or source.selected_shape is not None
                for source in self.sources
            ):
                raise PlanValidationError(
                    "packed dense sources require explicit complete owners"
                )
            if self.transform.startswith("pack_dense_gate_up_"):
                if len(self.sources) != 2:
                    raise PlanValidationError(
                        "packed dense gate/up requires two source tensors"
                    )
                gate, up = self.sources
                if gate.shape != up.shape:
                    raise PlanValidationError(
                        "packed dense gate/up source shapes drifted"
                    )
                if self.transform.endswith("bits_in_out"):
                    if self.spec.dtype != "F8_E4M3" or len(gate.shape) != 2:
                        raise PlanValidationError(
                            "packed dense gate/up bits must be FP8 matrices"
                        )
                    output, hidden = gate.shape
                    if output % virtual_shards:
                        raise PlanValidationError(
                            "packed dense gate/up output is not virtual-shard divisible"
                        )
                    expected_shape = (
                        virtual_shards,
                        hidden,
                        2 * output // virtual_shards,
                    )
                else:
                    if self.spec.dtype != "F32" or len(gate.shape) != 2:
                        raise PlanValidationError(
                            "packed dense gate/up scales must be FP32 matrices"
                        )
                    output_blocks, input_blocks = gate.shape
                    if output_blocks % virtual_shards:
                        raise PlanValidationError(
                            "packed dense gate/up scales are not virtual-shard divisible"
                        )
                    expected_shape = (
                        virtual_shards,
                        input_blocks,
                        2 * output_blocks * 128 // virtual_shards,
                    )
            else:
                if len(self.sources) != 1:
                    raise PlanValidationError(
                        "packed dense down requires one source tensor"
                    )
                (down,) = self.sources
                if self.transform.endswith("bits_in_out"):
                    if self.spec.dtype != "F8_E4M3" or len(down.shape) != 2:
                        raise PlanValidationError(
                            "packed dense down bits must be an FP8 matrix"
                        )
                    hidden, contraction = down.shape
                    if contraction % virtual_shards:
                        raise PlanValidationError(
                            "packed dense down contraction is not virtual-shard divisible"
                        )
                    expected_shape = (
                        virtual_shards,
                        contraction // virtual_shards,
                        hidden,
                    )
                else:
                    if self.spec.dtype != "F32" or len(down.shape) != 2:
                        raise PlanValidationError(
                            "packed dense down scales must be an FP32 matrix"
                        )
                    hidden_blocks, contraction_blocks = down.shape
                    if contraction_blocks % virtual_shards:
                        raise PlanValidationError(
                            "packed dense down scales are not virtual-shard divisible"
                        )
                    expected_shape = (
                        virtual_shards,
                        contraction_blocks // virtual_shards,
                        hidden_blocks * 128,
                    )
        else:
            if len(self.sources) < 2 or any(
                source.source_device_slot is None for source in self.sources
            ):
                raise PlanValidationError(
                    "transformed runtime tensor requires multiple explicit source owners"
                )
            leaf_shape = selected_shapes[0]
            if any(shape != leaf_shape for shape in selected_shapes):
                raise PlanValidationError(
                    "runtime transformed source selections drifted"
                )
            expected_shape = (
                sum(shape[0] for shape in selected_shapes),
                *leaf_shape[1:],
            )
        if expected_shape != self.spec.shape:
            raise PlanValidationError(
                f"runtime sources do not fill {self.spec.name}: "
                f"expected={self.spec.shape} observed={expected_shape}"
            )
        source_keys = tuple(
            (source.source_device_slot, source.name) for source in self.sources
        )
        if len(set(source_keys)) != len(self.sources):
            raise PlanValidationError("runtime tensor repeats a source leaf")
        if self.source_byte_count + self.derived_bytes != self.spec.byte_count:
            raise PlanValidationError("runtime tensor source bytes do not reconcile")

    @property
    def is_padding(self) -> bool:
        return not self.sources

    @property
    def source_byte_count(self) -> int:
        return sum(source.selected_byte_count for source in self.sources)

    @property
    def derived_bytes(self) -> int:
        if self.is_padding:
            return 0
        if self.transform in (
            "fuse_qkv_a_expanded_scales",
            "pack_dense_gate_up_scales_in_out",
            "pack_dense_down_scales_in_out",
        ):
            return self.spec.byte_count - self.source_byte_count
        return 0

    @property
    def padding_bytes(self) -> int:
        return self.spec.byte_count if self.is_padding else 0

    def to_dict(self) -> dict[str, Any]:
        value = {
            "padding": self.is_padding,
            "padding_bytes": self.padding_bytes,
            "runtime": self.spec.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.transform != "identity_concat":
            value["transform"] = self.transform
        if self.derived_bytes:
            value["derived_bytes"] = self.derived_bytes
        return value


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
        source_keys = [
            (source.source_device_slot, source.name)
            for tensor in self.tensors
            for source in tensor.sources
        ]
        if len(set(source_keys)) != len(source_keys):
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

    @property
    def derived_bytes(self) -> int:
        return sum(tensor.derived_bytes for tensor in self.tensors)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "device_id": self.device_id,
            "device_slot": self.device_slot,
            "padding_bytes": self.padding_bytes,
            "runtime_bytes": self.runtime_bytes,
            "source_bytes": self.source_bytes,
            "source_leaf_count": self.source_leaf_count,
            "stage_id": self.stage_id,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }
        if self.derived_bytes:
            value["derived_bytes"] = self.derived_bytes
        return value


@dataclass(frozen=True, slots=True)
class DecoderRuntimeWeightLayout:
    """Content-addressed uniform input layout for the global decoder."""

    plan_hash: str
    schedule_hash: str
    specs: tuple[RuntimeTensorSpec, ...]
    devices: tuple[DeviceRuntimeWeightLayout, ...]
    routed_expert_layout: str = COMPLETE_EXPERT_RUNTIME_LAYOUT
    attention_projection_layout: str = SEPARATE_QKV_A_RUNTIME_LAYOUT
    dense_projection_layout: str = LEGACY_DENSE_RUNTIME_LAYOUT

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", tuple(self.specs))
        object.__setattr__(self, "devices", tuple(self.devices))
        if len(self.plan_hash) != 64 or len(self.schedule_hash) != 64:
            raise PlanValidationError("runtime weight hashes must be SHA-256")
        if self.routed_expert_layout not in (
            COMPLETE_EXPERT_RUNTIME_LAYOUT,
            FEATURE_EXPERT_RUNTIME_LAYOUT,
        ):
            raise PlanValidationError("runtime routed expert layout is invalid")
        if self.attention_projection_layout not in (
            SEPARATE_QKV_A_RUNTIME_LAYOUT,
            FUSED_QKV_A_N82_RUNTIME_LAYOUT,
        ):
            raise PlanValidationError(
                "runtime attention projection layout is invalid"
            )
        if self.dense_projection_layout not in (
            LEGACY_DENSE_RUNTIME_LAYOUT,
            FINAL_DENSE_CONVOLUTION_RUNTIME_LAYOUT,
        ):
            raise PlanValidationError("runtime dense projection layout is invalid")
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
            if (
                device.source_bytes
                + device.padding_bytes
                + device.derived_bytes
                != device.runtime_bytes
            ):
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
        value = {
            "devices": [device.to_dict() for device in self.devices],
            "maximum_padding_bytes_per_chip": self.maximum_padding_bytes_per_chip,
            "minimum_padding_bytes_per_chip": self.minimum_padding_bytes_per_chip,
            "plan_hash": self.plan_hash,
            "runtime_bytes_per_chip": self.runtime_bytes_per_chip,
            "schedule_hash": self.schedule_hash,
            "source_leaf_count": self.source_leaf_count,
            "specs": [spec.to_dict() for spec in self.specs],
        }
        if self.routed_expert_layout != COMPLETE_EXPERT_RUNTIME_LAYOUT:
            value["routed_expert_layout"] = self.routed_expert_layout
        if self.attention_projection_layout != SEPARATE_QKV_A_RUNTIME_LAYOUT:
            value["attention_projection_layout"] = (
                self.attention_projection_layout
            )
        if self.dense_projection_layout != LEGACY_DENSE_RUNTIME_LAYOUT:
            value["dense_projection_layout"] = self.dense_projection_layout
        return value


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


def _feature_routed_spec(
    spec: RuntimeTensorSpec,
    *,
    num_experts: int,
    hidden_size: int,
    local_intermediate: int,
    output_scale_blocks: int,
    contraction_scale_blocks: int,
) -> tuple[RuntimeTensorSpec, str, tuple[int, ...]] | None:
    """Return the final Pallas feature-owner shape and source transform."""

    marker = ".experts."
    if marker not in spec.name:
        return None
    projection_and_role = spec.name.split(marker, maxsplit=1)[1]
    if projection_and_role in (
        "gate_proj.weight_bits",
        "up_proj.weight_bits",
    ):
        shape = (num_experts, hidden_size, local_intermediate)
        selected_shape = (
            spec.shape[0],
            hidden_size,
            local_intermediate,
        )
        transform = "concat_experts_slice_output_transpose"
    elif projection_and_role == "down_proj.weight_bits":
        shape = (num_experts, local_intermediate, hidden_size)
        selected_shape = (
            spec.shape[0],
            local_intermediate,
            hidden_size,
        )
        transform = "concat_experts_slice_contraction_transpose"
    elif projection_and_role in (
        "gate_proj.scale_inv",
        "up_proj.scale_inv",
    ):
        shape = (num_experts, output_scale_blocks, spec.shape[2])
        selected_shape = (spec.shape[0], output_scale_blocks, spec.shape[2])
        transform = "concat_experts_slice_scale_output"
    elif projection_and_role == "down_proj.scale_inv":
        shape = (num_experts, spec.shape[1], contraction_scale_blocks)
        selected_shape = (
            spec.shape[0],
            spec.shape[1],
            contraction_scale_blocks,
        )
        transform = "concat_experts_slice_scale_contraction"
    else:
        raise PlanValidationError(
            f"unknown routed runtime tensor role {projection_and_role!r}"
        )
    return replace(spec, shape=shape), transform, selected_shape


def build_decoder_feature_runtime_weight_layout(
    plan: ExecutionPlan,
    schedule: PipelineSchedule,
    source_layout: DecoderRuntimeWeightLayout,
) -> DecoderRuntimeWeightLayout:
    """Derive DB441's exact final expert-feature ownership from a runtime pack.

    The source is the verified complete-expert executable artifact.  Every
    non-routed tensor remains on the same physical owner.  Routed tensors are
    redistributed offline across the four files of one host-local stage: all
    expert identities become local while each destination owns one contiguous
    intermediate-feature slice in the exact ``[expert,K,N]`` Pallas order.
    """

    if schedule.plan_hash != plan.plan_hash:
        raise PlanValidationError(
            "feature runtime weights schedule belongs to another plan"
        )
    if plan.name.value != "PP8_LP4" or plan.local_parallel_size != 4:
        raise PlanValidationError(
            "feature runtime layout currently requires PP8_LP4"
        )
    if plan.expert_layout != FEATURE_EXPERT_RUNTIME_LAYOUT:
        raise PlanValidationError(
            "feature runtime plan does not declare the exact routed layout"
        )
    geometry = plan.geometry
    intermediate = geometry.moe_intermediate_size
    local_size = plan.local_parallel_size
    if intermediate % local_size:
        raise PlanValidationError(
            "expert intermediate width does not divide over the local stage"
        )
    local_intermediate = intermediate // local_size
    block_output, block_contraction = geometry.fp8_block_shape
    if (
        local_intermediate % block_output
        or local_intermediate % block_contraction
    ):
        raise PlanValidationError(
            "expert feature shard must preserve complete FP8 scale blocks"
        )

    source_plan = replace(
        plan,
        expert_layout=f"complete_expert_identity_lp{local_size}",
    )
    source_schedule = build_pipeline_schedule(source_plan)
    expected_source = build_decoder_runtime_weight_layout(
        source_plan,
        source_schedule,
    )
    if source_layout.to_dict() != expected_source.to_dict():
        raise PlanValidationError(
            "feature runtime source is not the exact complete-expert layout"
        )

    output_scale_blocks = local_intermediate // block_output
    contraction_scale_blocks = local_intermediate // block_contraction
    transformed_by_name: dict[
        str, tuple[RuntimeTensorSpec, str, tuple[int, ...]] | None
    ] = {}
    target_specs = []
    for spec in source_layout.specs:
        transformed = _feature_routed_spec(
            spec,
            num_experts=geometry.num_routed_experts,
            hidden_size=geometry.hidden_size,
            local_intermediate=local_intermediate,
            output_scale_blocks=output_scale_blocks,
            contraction_scale_blocks=contraction_scale_blocks,
        )
        transformed_by_name[spec.name] = transformed
        target_specs.append(spec if transformed is None else transformed[0])

    source_by_owner = {
        (device.stage_id, device.device_slot): device
        for device in source_layout.devices
    }
    target_spec_by_name = {spec.name: spec for spec in target_specs}
    devices = []
    for stage in schedule.stages:
        stage_sources = tuple(
            source_by_owner[(stage.assignment.stage_id, source_slot)]
            for source_slot in range(local_size)
        )
        source_tensor_by_slot = tuple(
            {tensor.spec.name: tensor for tensor in device.tensors}
            for device in stage_sources
        )
        for device_slot, device_id in enumerate(stage.assignment.device_ids):
            source_device = source_by_owner[
                (stage.assignment.stage_id, device_slot)
            ]
            tensors = []
            for source_binding in source_device.tensors:
                name = source_binding.spec.name
                target_spec = target_spec_by_name[name]
                transformed = transformed_by_name[name]
                if source_binding.is_padding:
                    if any(
                        not by_name[name].is_padding
                        for by_name in source_tensor_by_slot
                    ):
                        raise PlanValidationError(
                            "feature runtime routed liveness differs within a stage"
                        )
                    tensors.append(DeviceRuntimeTensor(target_spec, ()))
                    continue
                if transformed is None:
                    tensors.append(
                        DeviceRuntimeTensor(
                            target_spec,
                            (
                                RuntimeSourceLeaf(
                                    name=name,
                                    dtype=source_binding.spec.dtype,
                                    shape=source_binding.spec.shape,
                                    source_device_slot=device_slot,
                                ),
                            ),
                            transform="identity_runtime_tensor",
                        )
                    )
                    continue
                _, transform, selected_shape = transformed
                if any(
                    by_name[name].is_padding
                    for by_name in source_tensor_by_slot
                ):
                    raise PlanValidationError(
                        "feature runtime routed liveness differs within a stage"
                    )
                sources = tuple(
                    RuntimeSourceLeaf(
                        name=name,
                        dtype=source_tensor_by_slot[source_slot][name].spec.dtype,
                        shape=source_tensor_by_slot[source_slot][name].spec.shape,
                        source_device_slot=source_slot,
                        selected_shape=selected_shape,
                    )
                    for source_slot in range(local_size)
                )
                tensors.append(
                    DeviceRuntimeTensor(
                        target_spec,
                        sources,
                        transform=transform,
                    )
                )
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
        specs=tuple(target_specs),
        devices=tuple(devices),
        routed_expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )


def build_decoder_fused_qkv_runtime_weight_layout(
    plan: ExecutionPlan,
    schedule: PipelineSchedule,
    source_layout: DecoderRuntimeWeightLayout,
) -> DecoderRuntimeWeightLayout:
    """Replace separate q-a/kv-a tensors with DB502's final N82 layout."""

    if schedule.plan_hash != plan.plan_hash or (
        source_layout.plan_hash != plan.plan_hash
    ):
        raise PlanValidationError(
            "fused qkv-a runtime weights belong to another plan"
        )
    if source_layout.schedule_hash != schedule.schedule_hash:
        raise PlanValidationError(
            "fused qkv-a runtime source has another schedule"
        )
    if (
        source_layout.attention_projection_layout
        != SEPARATE_QKV_A_RUNTIME_LAYOUT
    ):
        raise PlanValidationError(
            "fused qkv-a source must retain separate q-a/kv-a tensors"
        )
    geometry = plan.geometry
    virtual_shards = 32
    q_width = geometry.q_lora_rank
    kv_width = geometry.kv_lora_rank + geometry.qk_rope_head_dim
    if q_width % virtual_shards or kv_width % virtual_shards:
        raise PlanValidationError(
            "fused qkv-a widths must divide over 32 virtual shards"
        )
    hidden = geometry.hidden_size
    if geometry.fp8_block_shape != (128, 128):
        raise PlanValidationError(
            "fused qkv-a v1 requires 128x128 FP8 scale blocks"
        )
    contraction_block = geometry.fp8_block_shape[1]
    if hidden % contraction_block:
        raise PlanValidationError(
            "fused qkv-a contraction must contain complete scale blocks"
        )
    packed_width = (q_width + kv_width) // virtual_shards
    source_specs = {spec.name: spec for spec in source_layout.specs}
    target_specs: list[RuntimeTensorSpec] = []
    for spec in source_layout.specs:
        if spec.name.endswith(".q_a.weight_bits"):
            kv_name = spec.name.replace(".q_a.weight_bits", ".kv_a.weight_bits")
            kv_spec = source_specs.get(kv_name)
            if kv_spec is None or spec.shape != (q_width, hidden) or (
                kv_spec.shape != (kv_width, hidden)
            ):
                raise PlanValidationError(
                    "fused qkv-a source weight geometry drifted"
                )
            target_specs.append(
                replace(
                    spec,
                    name=spec.name.replace(".q_a.", ".qkv_a."),
                    shape=(virtual_shards, hidden, packed_width),
                )
            )
        elif spec.name.endswith(".q_a.scale_inv"):
            kv_name = spec.name.replace(".q_a.scale_inv", ".kv_a.scale_inv")
            kv_spec = source_specs.get(kv_name)
            expected_q_scale = _scale_shape(
                (q_width, hidden), geometry.fp8_block_shape
            )
            expected_kv_scale = _scale_shape(
                (kv_width, hidden), geometry.fp8_block_shape
            )
            if kv_spec is None or spec.shape != expected_q_scale or (
                kv_spec.shape != expected_kv_scale
            ):
                raise PlanValidationError(
                    "fused qkv-a source scale geometry drifted"
                )
            target_specs.append(
                replace(
                    spec,
                    name=spec.name.replace(".q_a.", ".qkv_a."),
                    shape=(
                        virtual_shards,
                        hidden // contraction_block,
                        packed_width,
                    ),
                )
            )
        elif spec.name.endswith((".kv_a.weight_bits", ".kv_a.scale_inv")):
            continue
        else:
            target_specs.append(spec)

    target_names = tuple(spec.name for spec in target_specs)
    if len(set(target_names)) != len(target_names):
        raise PlanValidationError("fused qkv-a target specs are duplicate")
    devices = []
    for source_device in source_layout.devices:
        source_by_name = {
            tensor.spec.name: tensor for tensor in source_device.tensors
        }
        tensors = []
        for target_spec in target_specs:
            if ".qkv_a." not in target_spec.name:
                source = source_by_name[target_spec.name]
                if source.is_padding:
                    tensors.append(DeviceRuntimeTensor(target_spec, ()))
                else:
                    tensors.append(
                        DeviceRuntimeTensor(
                            target_spec,
                            (
                                RuntimeSourceLeaf(
                                    name=target_spec.name,
                                    dtype=target_spec.dtype,
                                    shape=target_spec.shape,
                                    source_device_slot=(
                                        source_device.device_slot
                                    ),
                                ),
                            ),
                            transform="identity_runtime_tensor",
                        )
                    )
                continue
            q_name = target_spec.name.replace(".qkv_a.", ".q_a.")
            kv_name = target_spec.name.replace(".qkv_a.", ".kv_a.")
            q_source = source_by_name[q_name]
            kv_source = source_by_name[kv_name]
            if q_source.is_padding != kv_source.is_padding:
                raise PlanValidationError(
                    "fused qkv-a source liveness differs within one layer"
                )
            if q_source.is_padding:
                tensors.append(DeviceRuntimeTensor(target_spec, ()))
                continue
            tensors.append(
                DeviceRuntimeTensor(
                    target_spec,
                    (
                        RuntimeSourceLeaf(
                            name=q_name,
                            dtype=q_source.spec.dtype,
                            shape=q_source.spec.shape,
                            source_device_slot=source_device.device_slot,
                        ),
                        RuntimeSourceLeaf(
                            name=kv_name,
                            dtype=kv_source.spec.dtype,
                            shape=kv_source.spec.shape,
                            source_device_slot=source_device.device_slot,
                        ),
                    ),
                    transform=(
                        "fuse_qkv_a_expanded_scales"
                        if target_spec.value_class == "fp8_scale"
                        else "fuse_qkv_a_output_shards"
                    ),
                )
            )
        devices.append(
            DeviceRuntimeWeightLayout(
                stage_id=source_device.stage_id,
                device_slot=source_device.device_slot,
                device_id=source_device.device_id,
                tensors=tuple(tensors),
            )
        )
    return DecoderRuntimeWeightLayout(
        plan_hash=plan.plan_hash,
        schedule_hash=schedule.schedule_hash,
        specs=tuple(target_specs),
        devices=tuple(devices),
        routed_expert_layout=source_layout.routed_expert_layout,
        attention_projection_layout=FUSED_QKV_A_N82_RUNTIME_LAYOUT,
    )


def build_decoder_feature_fused_qkv_runtime_weight_layout(
    plan: ExecutionPlan,
    schedule: PipelineSchedule,
    source_layout: DecoderRuntimeWeightLayout,
) -> DecoderRuntimeWeightLayout:
    """Derive feature experts and fused N82 attention in one offline pass."""

    feature_layout = build_decoder_feature_runtime_weight_layout(
        plan,
        schedule,
        source_layout,
    )
    source_plan = replace(
        plan,
        expert_layout=f"complete_expert_identity_lp{plan.local_parallel_size}",
    )
    source_schedule = build_pipeline_schedule(source_plan)
    fused_attention_layout = build_decoder_fused_qkv_runtime_weight_layout(
        source_plan,
        source_schedule,
        source_layout,
    )
    feature_specs = {spec.name: spec for spec in feature_layout.specs}
    specs = tuple(
        spec if ".qkv_a." in spec.name else feature_specs[spec.name]
        for spec in fused_attention_layout.specs
    )
    devices = []
    for feature_device, qkv_device in zip(
        feature_layout.devices,
        fused_attention_layout.devices,
        strict=True,
    ):
        if (
            feature_device.stage_id,
            feature_device.device_slot,
            feature_device.device_id,
        ) != (
            qkv_device.stage_id,
            qkv_device.device_slot,
            qkv_device.device_id,
        ):
            raise PlanValidationError(
                "feature and fused qkv-a device ownership disagree"
            )
        feature_tensors = {
            tensor.spec.name: tensor for tensor in feature_device.tensors
        }
        qkv_tensors = {
            tensor.spec.name: tensor for tensor in qkv_device.tensors
        }
        tensors = tuple(
            qkv_tensors[spec.name]
            if ".qkv_a." in spec.name
            else feature_tensors[spec.name]
            for spec in specs
        )
        devices.append(
            DeviceRuntimeWeightLayout(
                stage_id=feature_device.stage_id,
                device_slot=feature_device.device_slot,
                device_id=feature_device.device_id,
                tensors=tensors,
            )
        )
    return DecoderRuntimeWeightLayout(
        plan_hash=plan.plan_hash,
        schedule_hash=schedule.schedule_hash,
        specs=specs,
        devices=tuple(devices),
        routed_expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
        attention_projection_layout=FUSED_QKV_A_N82_RUNTIME_LAYOUT,
    )


def build_decoder_dense_convolution_runtime_weight_layout(
    plan: ExecutionPlan,
    schedule: PipelineSchedule,
    source_layout: DecoderRuntimeWeightLayout,
) -> DecoderRuntimeWeightLayout:
    """Derive exact virtual-TP32 dense convolution inputs offline.

    The source is the complete-owner runtime layout.  Dense gate/up shards are
    merged and transposed into eight ``[in, out]`` virtual shards; their small
    FP32 block scales are expanded only along the output dimension.  Down
    shards receive the reciprocal layout.  All transforms preserve the
    existing physical owner and checkpoint bytes.
    """

    if schedule.plan_hash != plan.plan_hash:
        raise PlanValidationError(
            "dense convolution schedule belongs to another plan"
        )
    if plan.local_parallel_size != 4:
        raise PlanValidationError(
            "virtual-TP32 dense convolution currently requires PP8 LP4"
        )
    source_specs = {spec.name: spec for spec in source_layout.specs}
    if len(source_specs) != len(source_layout.specs):
        raise PlanValidationError("dense convolution source specs are duplicate")
    target_specs: list[RuntimeTensorSpec] = []
    for spec in source_layout.specs:
        if spec.slot_kind != "dense":
            target_specs.append(spec)
            continue
        if spec.name.endswith(".gate.weight_bits"):
            up = source_specs.get(spec.name.replace(".gate.", ".up."))
            if up is None or up.shape != spec.shape or spec.dtype != "F8_E4M3":
                raise PlanValidationError(
                    "dense gate/up weight source geometry drifted"
                )
            output, hidden = spec.shape
            if output % 8:
                raise PlanValidationError(
                    "dense gate/up output is not virtual-shard divisible"
                )
            target_specs.append(
                replace(
                    spec,
                    name=spec.name.replace(
                        ".gate.weight_bits",
                        ".merged_gate_up.weight_bits_in_out",
                    ),
                    shape=(8, hidden, 2 * output // 8),
                )
            )
        elif spec.name.endswith(".gate.scale_inv"):
            up = source_specs.get(spec.name.replace(".gate.", ".up."))
            if up is None or up.shape != spec.shape or spec.dtype != "F32":
                raise PlanValidationError(
                    "dense gate/up scale source geometry drifted"
                )
            output_blocks, input_blocks = spec.shape
            if output_blocks % 8:
                raise PlanValidationError(
                    "dense gate/up scale output is not virtual-shard divisible"
                )
            target_specs.append(
                replace(
                    spec,
                    name=spec.name.replace(
                        ".gate.scale_inv",
                        ".merged_gate_up.scale_inv_in_out",
                    ),
                    shape=(
                        8,
                        input_blocks,
                        2 * output_blocks * 128 // 8,
                    ),
                )
            )
        elif spec.name.endswith((".up.weight_bits", ".up.scale_inv")):
            continue
        elif spec.name.endswith(".down.weight_bits"):
            hidden, contraction = spec.shape
            if spec.dtype != "F8_E4M3" or contraction % 8:
                raise PlanValidationError("dense down weight geometry drifted")
            target_specs.append(
                replace(
                    spec,
                    name=spec.name.replace(
                        ".down.weight_bits", ".down.weight_bits_in_out"
                    ),
                    shape=(8, contraction // 8, hidden),
                )
            )
        elif spec.name.endswith(".down.scale_inv"):
            hidden_blocks, contraction_blocks = spec.shape
            if spec.dtype != "F32" or contraction_blocks % 8:
                raise PlanValidationError("dense down scale geometry drifted")
            target_specs.append(
                replace(
                    spec,
                    name=spec.name.replace(
                        ".down.scale_inv", ".down.scale_inv_in_out"
                    ),
                    shape=(
                        8,
                        contraction_blocks // 8,
                        hidden_blocks * 128,
                    ),
                )
            )
        else:
            raise PlanValidationError(
                f"unknown dense runtime tensor {spec.name!r}"
            )

    devices = []
    for source_device in source_layout.devices:
        source_by_name = {
            tensor.spec.name: tensor for tensor in source_device.tensors
        }
        tensors = []
        for target_spec in target_specs:
            if target_spec.slot_kind != "dense":
                source = source_by_name[target_spec.name]
                tensors.append(
                    DeviceRuntimeTensor(target_spec, ())
                    if source.is_padding
                    else DeviceRuntimeTensor(
                        target_spec,
                        (
                            RuntimeSourceLeaf(
                                name=target_spec.name,
                                dtype=target_spec.dtype,
                                shape=target_spec.shape,
                                source_device_slot=source_device.device_slot,
                            ),
                        ),
                        transform="identity_runtime_tensor",
                    )
                )
                continue
            base = target_spec.name.rsplit(".", 2)[0]
            if ".merged_gate_up." in target_spec.name:
                suffix = (
                    ".weight_bits"
                    if target_spec.value_class == "fp8_weight"
                    else ".scale_inv"
                )
                gate_name = base + ".gate" + suffix
                up_name = gate_name.replace(".gate.", ".up.")
                gate = source_by_name[gate_name]
                up = source_by_name[up_name]
                if gate.is_padding != up.is_padding:
                    raise PlanValidationError(
                        "dense gate/up source liveness differs"
                    )
                if gate.is_padding:
                    tensors.append(DeviceRuntimeTensor(target_spec, ()))
                    continue
                sources = tuple(
                    RuntimeSourceLeaf(
                        name=name,
                        dtype=source_by_name[name].spec.dtype,
                        shape=source_by_name[name].spec.shape,
                        source_device_slot=source_device.device_slot,
                    )
                    for name in (gate_name, up_name)
                )
                transform = (
                    "pack_dense_gate_up_bits_in_out"
                    if target_spec.value_class == "fp8_weight"
                    else "pack_dense_gate_up_scales_in_out"
                )
            else:
                source_name = target_spec.name.replace("_in_out", "")
                source = source_by_name[source_name]
                if source.is_padding:
                    tensors.append(DeviceRuntimeTensor(target_spec, ()))
                    continue
                sources = (
                    RuntimeSourceLeaf(
                        name=source_name,
                        dtype=source.spec.dtype,
                        shape=source.spec.shape,
                        source_device_slot=source_device.device_slot,
                    ),
                )
                transform = (
                    "pack_dense_down_bits_in_out"
                    if target_spec.value_class == "fp8_weight"
                    else "pack_dense_down_scales_in_out"
                )
            tensors.append(
                DeviceRuntimeTensor(target_spec, sources, transform=transform)
            )
        devices.append(
            DeviceRuntimeWeightLayout(
                stage_id=source_device.stage_id,
                device_slot=source_device.device_slot,
                device_id=source_device.device_id,
                tensors=tuple(tensors),
            )
        )
    return DecoderRuntimeWeightLayout(
        plan_hash=plan.plan_hash,
        schedule_hash=schedule.schedule_hash,
        specs=tuple(target_specs),
        devices=tuple(devices),
        routed_expert_layout=source_layout.routed_expert_layout,
        attention_projection_layout=source_layout.attention_projection_layout,
        dense_projection_layout=FINAL_DENSE_CONVOLUTION_RUNTIME_LAYOUT,
    )


def build_decoder_feature_fused_qkv_dense_runtime_weight_layout(
    plan: ExecutionPlan,
    schedule: PipelineSchedule,
    source_layout: DecoderRuntimeWeightLayout,
) -> DecoderRuntimeWeightLayout:
    """Compose feature experts, fused qkv-a, and dense final layout."""

    feature_qkv = build_decoder_feature_fused_qkv_runtime_weight_layout(
        plan, schedule, source_layout
    )
    dense = build_decoder_dense_convolution_runtime_weight_layout(
        plan, schedule, source_layout
    )
    dense_specs = {
        spec.name: spec for spec in dense.specs if spec.slot_kind == "dense"
    }
    specs_list: list[RuntimeTensorSpec] = []
    for source_spec in feature_qkv.specs:
        if source_spec.slot_kind != "dense":
            specs_list.append(source_spec)
        elif source_spec.name.endswith(".gate.weight_bits"):
            prefix = source_spec.name.rsplit(".", 2)[0].replace(".gate", "")
            specs_list.extend(
                spec
                for spec in dense.specs
                if spec.slot_kind == "dense"
                and spec.name.startswith(f"{prefix}.")
            )
    specs = tuple(specs_list)
    expected_dense_count = schedule.maximum_dense_slots * 4
    if len(dense_specs) != expected_dense_count:
        raise PlanValidationError("dense final-layout spec count drifted")
    devices = []
    for feature_device, dense_device in zip(
        feature_qkv.devices, dense.devices, strict=True
    ):
        if (
            feature_device.stage_id,
            feature_device.device_slot,
            feature_device.device_id,
        ) != (
            dense_device.stage_id,
            dense_device.device_slot,
            dense_device.device_id,
        ):
            raise PlanValidationError(
                "feature/qkv and dense final-layout ownership disagree"
            )
        feature_by_name = {
            tensor.spec.name: tensor for tensor in feature_device.tensors
        }
        dense_by_name = {
            tensor.spec.name: tensor for tensor in dense_device.tensors
        }
        devices.append(
            DeviceRuntimeWeightLayout(
                stage_id=feature_device.stage_id,
                device_slot=feature_device.device_slot,
                device_id=feature_device.device_id,
                tensors=tuple(
                    dense_by_name[spec.name]
                    if spec.slot_kind == "dense"
                    else feature_by_name[spec.name]
                    for spec in specs
                ),
            )
        )
    return DecoderRuntimeWeightLayout(
        plan_hash=plan.plan_hash,
        schedule_hash=schedule.schedule_hash,
        specs=specs,
        devices=tuple(devices),
        routed_expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
        attention_projection_layout=FUSED_QKV_A_N82_RUNTIME_LAYOUT,
        dense_projection_layout=FINAL_DENSE_CONVOLUTION_RUNTIME_LAYOUT,
    )
