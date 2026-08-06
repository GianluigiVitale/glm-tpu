"""Offline complete-runtime derivative for PP8 expert-feature ownership.

The input is the already verified executable-ready complete-expert artifact.
Four source files belonging to one host-local stage are transformed together
so each routed byte is read once and written directly to its final owner.  No
full-model destination file or runtime repartition is staged in host storage.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from typing import Any, BinaryIO

from ..errors import CheckpointValidationError
from ..model.weights import (
    FEATURE_EXPERT_RUNTIME_LAYOUT,
    DecoderRuntimeWeightLayout,
    DeviceRuntimeWeightLayout,
    RuntimeSourceLeaf,
)
from .runtime_pack import (
    RuntimeDestinationFilePlan,
    RuntimeDestinationTensorPlan,
)

FEATURE_RUNTIME_LAYOUT_ARTIFACT_KIND = (
    "greenfield_decoder_feature_runtime_weight_layout"
)
FEATURE_RUNTIME_PACK_CONTROL_KIND = "greenfield_feature_runtime_checkpoint_pack_control"
FEATURE_RUNTIME_PACKED_ARTIFACT_KIND = "greenfield_feature_runtime_packed_checkpoint"
FEATURE_RUNTIME_FORMAT_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_hash(value: dict[str, Any], *, hash_field: str) -> str:
    unhashed = dict(value)
    unhashed.pop(hash_field, None)
    return sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()


def _digest(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class FeatureRuntimeDestinationFilePlan:
    """One feature-owner output and the four runtime files feeding it."""

    filename: str
    source_filenames: tuple[str, ...]
    stage_id: int
    device_slot: int
    device_id: int
    header: bytes
    payload_bytes: int
    tensors: tuple[RuntimeDestinationTensorPlan, ...]
    device_layout: DeviceRuntimeWeightLayout
    runtime_layout_hash: str
    source_runtime_manifest_sha256: str

    @property
    def file_bytes(self) -> int:
        return len(self.header) + self.payload_bytes


@dataclass(frozen=True, slots=True)
class FeatureRuntimeSourceEvidence:
    source_device_slot: int
    source_filename: str
    source_tensor_name: str
    source_tensor_sha256: str
    source_shape: tuple[int, ...]
    selected_shape: tuple[int, ...]
    feature_axis: int | None
    feature_start: int | None
    feature_stop: int | None
    transpose_axes: tuple[int, ...] | None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "selected_shape": list(self.selected_shape),
            "source_device_slot": self.source_device_slot,
            "source_filename": self.source_filename,
            "source_shape": list(self.source_shape),
            "source_tensor_name": self.source_tensor_name,
            "source_tensor_sha256": self.source_tensor_sha256,
        }
        if self.feature_axis is not None:
            value["feature_slice"] = {
                "axis": self.feature_axis,
                "start": self.feature_start,
                "stop": self.feature_stop,
            }
        if self.transpose_axes is not None:
            value["transpose_axes"] = list(self.transpose_axes)
        return value


@dataclass(frozen=True, slots=True)
class FeatureRuntimeTensorEvidence:
    name: str
    byte_count: int
    padding: bool
    sha256: str
    transform: str
    sources: tuple[FeatureRuntimeSourceEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "name": self.name,
            "padding": self.padding,
            "sha256": self.sha256,
            "sources": [source.to_dict() for source in self.sources],
            "transform": self.transform,
        }


@dataclass(frozen=True, slots=True)
class StreamedFeatureRuntimeFileEvidence:
    filename: str
    file_bytes: int
    payload_bytes: int
    sha256: str
    source_files: tuple[tuple[str, str], ...]
    source_payload_bytes: int
    source_tensor_count: int
    padding_bytes: int
    tensors: tuple[FeatureRuntimeTensorEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_bytes": self.file_bytes,
            "filename": self.filename,
            "padding_bytes": self.padding_bytes,
            "payload_bytes": self.payload_bytes,
            "sha256": self.sha256,
            "source_files": [
                {"filename": filename, "sha256": digest}
                for filename, digest in self.source_files
            ],
            "source_payload_bytes": self.source_payload_bytes,
            "source_tensor_count": self.source_tensor_count,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }


def build_feature_runtime_layout_document(
    layout: DecoderRuntimeWeightLayout,
) -> dict[str, Any]:
    """Return the self-authenticating feature-runtime semantic layout."""

    if layout.routed_expert_layout != FEATURE_EXPERT_RUNTIME_LAYOUT:
        raise CheckpointValidationError(
            "feature runtime document received the wrong routed layout"
        )
    value: dict[str, Any] = {
        "artifact_kind": FEATURE_RUNTIME_LAYOUT_ARTIFACT_KIND,
        "format_version": FEATURE_RUNTIME_FORMAT_VERSION,
        "layout": layout.to_dict(),
        "runtime_layout_hash": layout.layout_hash,
    }
    value["manifest_sha256"] = _mapping_hash(
        value,
        hash_field="manifest_sha256",
    )
    return value


def _feature_runtime_header(
    *,
    filename: str,
    tensors: Sequence[RuntimeDestinationTensorPlan],
    runtime_layout_hash: str,
    source_runtime_manifest_sha256: str,
) -> bytes:
    value: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "greenfield_artifact_kind": "greenfield_feature_runtime_weight_file",
            "greenfield_destination_filename": filename,
            "greenfield_runtime_layout_hash": runtime_layout_hash,
            "greenfield_source_runtime_manifest_sha256": (
                source_runtime_manifest_sha256
            ),
        }
    }
    for tensor in tensors:
        spec = tensor.spec
        storage_dtype = "U8" if spec.dtype == "F8_E4M3" else spec.dtype
        value[spec.name] = {
            "data_offsets": [tensor.data_offset_start, tensor.data_offset_end],
            "dtype": storage_dtype,
            "shape": list(spec.shape),
        }
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def build_feature_runtime_destination_file_plans(
    layout: DecoderRuntimeWeightLayout,
    source_plans: Sequence[RuntimeDestinationFilePlan],
    *,
    source_runtime_manifest_sha256: str,
) -> tuple[FeatureRuntimeDestinationFilePlan, ...]:
    """Build one final feature-owner plan for every PP8 physical device."""

    source_hash = _digest(
        source_runtime_manifest_sha256,
        field="source_runtime_manifest_sha256",
    )
    if layout.routed_expert_layout != FEATURE_EXPERT_RUNTIME_LAYOUT:
        raise CheckpointValidationError(
            "feature runtime plans require the feature expert layout"
        )
    source_by_owner = {(plan.stage_id, plan.device_slot): plan for plan in source_plans}
    if len(source_by_owner) != len(layout.devices):
        raise CheckpointValidationError(
            "feature runtime source files do not cover every base owner"
        )
    plans = []
    for device in layout.devices:
        stage_sources = tuple(
            source_by_owner[(device.stage_id, source_slot)] for source_slot in range(4)
        )
        if tuple(plan.device_slot for plan in stage_sources) != tuple(range(4)):
            raise CheckpointValidationError(
                "feature runtime stage source slots are incomplete"
            )
        offset = 0
        tensors = []
        for spec in layout.specs:
            tensors.append(
                RuntimeDestinationTensorPlan(
                    spec=spec,
                    data_offset_start=offset,
                    data_offset_end=offset + spec.byte_count,
                )
            )
            offset += spec.byte_count
        if offset != layout.runtime_bytes_per_chip:
            raise CheckpointValidationError(
                "feature runtime tensor bytes do not reconcile"
            )
        filename = (
            f"base_decoder_runtime_feature/stage_{device.stage_id:02d}/"
            f"device_slot_{device.device_slot:02d}.safetensors"
        )
        tensor_tuple = tuple(tensors)
        plans.append(
            FeatureRuntimeDestinationFilePlan(
                filename=filename,
                source_filenames=tuple(source.filename for source in stage_sources),
                stage_id=device.stage_id,
                device_slot=device.device_slot,
                device_id=device.device_id,
                header=_feature_runtime_header(
                    filename=filename,
                    tensors=tensor_tuple,
                    runtime_layout_hash=layout.layout_hash,
                    source_runtime_manifest_sha256=source_hash,
                ),
                payload_bytes=offset,
                tensors=tensor_tuple,
                device_layout=device,
                runtime_layout_hash=layout.layout_hash,
                source_runtime_manifest_sha256=source_hash,
            )
        )
    return tuple(plans)


def _write(
    output: BinaryIO,
    value: bytes | bytearray | memoryview,
    *,
    file_digest: Any,
    tensor_digest: Any,
) -> int:
    count = output.write(value)
    if count is not None and count != len(value):
        raise CheckpointValidationError("feature runtime output write was incomplete")
    file_digest.update(value)
    tensor_digest.update(value)
    return len(value)


def _write_zeros(
    output: BinaryIO,
    *,
    byte_count: int,
    file_digest: Any,
    tensor_digest: Any,
    chunk_bytes: int,
) -> int:
    chunk = bytes(min(chunk_bytes, byte_count))
    written = 0
    while written < byte_count:
        value = chunk[: min(len(chunk), byte_count - written)]
        written += _write(
            output,
            value,
            file_digest=file_digest,
            tensor_digest=tensor_digest,
        )
    return written


def _copy_exact(
    source: BinaryIO,
    output: BinaryIO,
    *,
    byte_count: int,
    file_digest: Any,
    tensor_digest: Any,
    chunk_bytes: int,
) -> int:
    copied = 0
    while copied < byte_count:
        value = source.read(min(chunk_bytes, byte_count - copied))
        if not value:
            raise CheckpointValidationError(
                "feature runtime source tensor is truncated"
            )
        copied += _write(
            output,
            value,
            file_digest=file_digest,
            tensor_digest=tensor_digest,
        )
    return copied


def _read_exact_at(source: BinaryIO, *, offset: int, byte_count: int) -> bytearray:
    source.seek(offset)
    value = bytearray(byte_count)
    view = memoryview(value)
    received = 0
    while received < byte_count:
        count = source.readinto(view[received:])
        if not count:
            raise CheckpointValidationError(
                "feature runtime source tensor is truncated"
            )
        received += count
    return value


def _transform_contract(
    source: RuntimeSourceLeaf,
    *,
    transform: str,
    destination_slot: int,
    stage_size: int,
) -> tuple[int | None, int | None, int | None, tuple[int, ...] | None]:
    if transform == "identity_runtime_tensor":
        return None, None, None, None
    axis_and_transpose = {
        "concat_experts_slice_output_transpose": (1, (0, 2, 1)),
        "concat_experts_slice_contraction_transpose": (2, (0, 2, 1)),
        "concat_experts_slice_scale_output": (1, None),
        "concat_experts_slice_scale_contraction": (2, None),
    }
    try:
        axis, transpose_axes = axis_and_transpose[transform]
    except KeyError as error:
        raise CheckpointValidationError(
            f"unsupported feature runtime transform {transform!r}"
        ) from error
    width = source.shape[axis]
    if width % stage_size:
        raise CheckpointValidationError(
            "feature runtime source axis does not divide over its stage"
        )
    local_width = width // stage_size
    start = destination_slot * local_width
    return axis, start, start + local_width, transpose_axes


def _source_evidence(
    *,
    source: RuntimeSourceLeaf,
    transform: str,
    destination_slot: int,
    source_filenames: tuple[str, ...],
    source_tensor_sha256: Mapping[tuple[str, str], str],
) -> FeatureRuntimeSourceEvidence:
    assert source.source_device_slot is not None
    filename = source_filenames[source.source_device_slot]
    digest = _digest(
        source_tensor_sha256[(filename, source.name)],
        field=f"{filename}:{source.name}.sha256",
    )
    axis, start, stop, transpose = _transform_contract(
        source,
        transform=transform,
        destination_slot=destination_slot,
        stage_size=len(source_filenames),
    )
    selected_shape = (
        source.shape if source.selected_shape is None else source.selected_shape
    )
    return FeatureRuntimeSourceEvidence(
        source_device_slot=source.source_device_slot,
        source_filename=filename,
        source_tensor_name=source.name,
        source_tensor_sha256=digest,
        source_shape=source.shape,
        selected_shape=selected_shape,
        feature_axis=axis,
        feature_start=start,
        feature_stop=stop,
        transpose_axes=transpose,
    )


def _transform_expert(
    raw: bytearray,
    *,
    source_shape: tuple[int, ...],
    dtype: str,
    transform: str,
    destination_slot: int,
    stage_size: int,
) -> bytes:
    import numpy as np

    if dtype == "F8_E4M3":
        storage_dtype = np.dtype("u1")
    elif dtype == "F32":
        storage_dtype = np.dtype("<f4")
    else:
        raise CheckpointValidationError(
            f"feature transform received unsupported dtype {dtype!r}"
        )
    expert_shape = source_shape[1:]
    if (
        len(expert_shape) != 2
        or len(raw) != prod(expert_shape) * storage_dtype.itemsize
    ):
        raise CheckpointValidationError(
            "feature runtime expert source bytes do not reconcile"
        )
    value = np.frombuffer(raw, dtype=storage_dtype).reshape(expert_shape)
    if transform in (
        "concat_experts_slice_output_transpose",
        "concat_experts_slice_scale_output",
    ):
        width = expert_shape[0]
        if width % stage_size:
            raise CheckpointValidationError(
                "feature runtime output axis is not stage divisible"
            )
        local = width // stage_size
        selected = value[
            destination_slot * local : (destination_slot + 1) * local,
            :,
        ]
    elif transform in (
        "concat_experts_slice_contraction_transpose",
        "concat_experts_slice_scale_contraction",
    ):
        width = expert_shape[1]
        if width % stage_size:
            raise CheckpointValidationError(
                "feature runtime contraction axis is not stage divisible"
            )
        local = width // stage_size
        selected = value[
            :,
            destination_slot * local : (destination_slot + 1) * local,
        ]
    else:
        raise CheckpointValidationError(
            f"unsupported feature runtime transform {transform!r}"
        )
    if transform.endswith("_transpose"):
        selected = selected.T
    return selected.tobytes(order="C")


def stream_feature_runtime_stage(
    *,
    source_plans: Sequence[RuntimeDestinationFilePlan],
    destination_plans: Sequence[FeatureRuntimeDestinationFilePlan],
    sources: Mapping[int, BinaryIO],
    outputs: Mapping[int, BinaryIO],
    verified_source_file_sha256: Mapping[str, str],
    source_tensor_sha256: Mapping[tuple[str, str], str],
    chunk_bytes: int = 64 * 1024 * 1024,
) -> tuple[StreamedFeatureRuntimeFileEvidence, ...]:
    """Transform all four files of one PP8 stage with bounded host memory."""

    if chunk_bytes <= 0:
        raise ValueError("feature runtime pack chunk_bytes must be positive")
    source_by_slot = {plan.device_slot: plan for plan in source_plans}
    destination_by_slot = {plan.device_slot: plan for plan in destination_plans}
    expected_slots = set(range(4))
    if (
        set(source_by_slot) != expected_slots
        or set(destination_by_slot) != expected_slots
        or set(sources) != expected_slots
        or set(outputs) != expected_slots
    ):
        raise CheckpointValidationError(
            "feature runtime stage requires exact slots 0,1,2,3"
        )
    stage_ids = {plan.stage_id for plan in (*source_plans, *destination_plans)}
    if len(stage_ids) != 1:
        raise CheckpointValidationError(
            "feature runtime source/destination stages disagree"
        )
    source_filenames = tuple(source_by_slot[slot].filename for slot in range(4))
    for plan in destination_plans:
        if plan.source_filenames != source_filenames:
            raise CheckpointValidationError(
                "feature runtime destination source identities disagree"
            )
    source_tensor_by_slot = tuple(
        {tensor.spec.name: tensor for tensor in source_by_slot[slot].tensors}
        for slot in range(4)
    )
    if any(
        len(by_name) != len(source_by_slot[slot].tensors)
        for slot, by_name in enumerate(source_tensor_by_slot)
    ):
        raise CheckpointValidationError(
            "feature runtime source tensor names are duplicate"
        )
    for slot in range(4):
        plan = source_by_slot[slot]
        digest = _digest(
            verified_source_file_sha256[plan.filename],
            field=f"{plan.filename}.sha256",
        )
        if not digest:
            raise AssertionError("validated digest is unexpectedly empty")
        source = sources[slot]
        source.seek(0)
        if source.read(len(plan.header)) != plan.header:
            raise CheckpointValidationError(
                f"feature runtime source header drifted for {plan.filename!r}"
            )

    file_digests = {}
    output_bytes = {}
    tensor_evidence: dict[int, list[FeatureRuntimeTensorEvidence]] = {
        slot: [] for slot in range(4)
    }
    for slot in range(4):
        plan = destination_by_slot[slot]
        count = outputs[slot].write(plan.header)
        if count is not None and count != len(plan.header):
            raise CheckpointValidationError(
                "feature runtime output header write was incomplete"
            )
        file_digests[slot] = sha256(plan.header)
        output_bytes[slot] = len(plan.header)

    destination_tensor_by_slot = tuple(
        {tensor.spec.name: tensor for tensor in destination_by_slot[slot].tensors}
        for slot in range(4)
    )
    binding_by_slot = tuple(
        {
            tensor.spec.name: tensor
            for tensor in destination_by_slot[slot].device_layout.tensors
        }
        for slot in range(4)
    )
    spec_names = tuple(tensor.spec.name for tensor in destination_by_slot[0].tensors)
    if any(
        tuple(tensor.spec.name for tensor in destination_by_slot[slot].tensors)
        != spec_names
        for slot in range(1, 4)
    ):
        raise CheckpointValidationError(
            "feature runtime destination tensor order differs by owner"
        )

    for name in spec_names:
        bindings = tuple(binding_by_slot[slot][name] for slot in range(4))
        destinations = tuple(
            destination_tensor_by_slot[slot][name] for slot in range(4)
        )
        tensor_digests = tuple(sha256() for _ in range(4))
        tensor_bytes = [0, 0, 0, 0]
        if all(binding.is_padding for binding in bindings):
            for slot in range(4):
                tensor_bytes[slot] = _write_zeros(
                    outputs[slot],
                    byte_count=destinations[slot].byte_count,
                    file_digest=file_digests[slot],
                    tensor_digest=tensor_digests[slot],
                    chunk_bytes=chunk_bytes,
                )
        elif any(binding.is_padding for binding in bindings):
            raise CheckpointValidationError(
                "feature runtime tensor liveness differs within a stage"
            )
        elif all(
            binding.transform == "identity_runtime_tensor" for binding in bindings
        ):
            for slot, binding in enumerate(bindings):
                if len(binding.sources) != 1:
                    raise CheckpointValidationError(
                        "feature runtime identity source count drifted"
                    )
                source_leaf = binding.sources[0]
                if source_leaf.source_device_slot != slot:
                    raise CheckpointValidationError(
                        "feature runtime identity moved physical ownership"
                    )
                source_tensor = source_tensor_by_slot[slot][name]
                if (
                    source_tensor.spec.dtype != source_leaf.dtype
                    or source_tensor.spec.shape != source_leaf.shape
                    or source_tensor.byte_count != destinations[slot].byte_count
                ):
                    raise CheckpointValidationError(
                        f"feature runtime identity contract drifted for {name!r}"
                    )
                sources[slot].seek(
                    len(source_by_slot[slot].header) + source_tensor.data_offset_start
                )
                tensor_bytes[slot] = _copy_exact(
                    sources[slot],
                    outputs[slot],
                    byte_count=source_tensor.byte_count,
                    file_digest=file_digests[slot],
                    tensor_digest=tensor_digests[slot],
                    chunk_bytes=chunk_bytes,
                )
        else:
            transforms = {binding.transform for binding in bindings}
            if len(transforms) != 1:
                raise CheckpointValidationError(
                    "feature runtime transforms differ by destination"
                )
            transform = next(iter(transforms))
            for source_slot in range(4):
                source_tensor = source_tensor_by_slot[source_slot][name]
                source_shape = source_tensor.spec.shape
                if len(source_shape) != 3 or source_shape[0] <= 0:
                    raise CheckpointValidationError(
                        f"feature runtime routed source shape drifted for {name!r}"
                    )
                expert_bytes = source_tensor.byte_count // source_shape[0]
                if expert_bytes * source_shape[0] != source_tensor.byte_count:
                    raise CheckpointValidationError(
                        "feature runtime expert bytes do not reconcile"
                    )
                tensor_start = (
                    len(source_by_slot[source_slot].header)
                    + source_tensor.data_offset_start
                )
                for expert in range(source_shape[0]):
                    raw = _read_exact_at(
                        sources[source_slot],
                        offset=tensor_start + expert * expert_bytes,
                        byte_count=expert_bytes,
                    )
                    for destination_slot in range(4):
                        value = _transform_expert(
                            raw,
                            source_shape=source_shape,
                            dtype=source_tensor.spec.dtype,
                            transform=transform,
                            destination_slot=destination_slot,
                            stage_size=4,
                        )
                        tensor_bytes[destination_slot] += _write(
                            outputs[destination_slot],
                            value,
                            file_digest=file_digests[destination_slot],
                            tensor_digest=tensor_digests[destination_slot],
                        )
                    del raw

        for slot in range(4):
            destination = destinations[slot]
            binding = bindings[slot]
            if tensor_bytes[slot] != destination.byte_count:
                raise CheckpointValidationError(
                    f"feature runtime tensor {name!r} byte count drifted"
                )
            output_bytes[slot] += tensor_bytes[slot]
            sources_evidence = tuple(
                _source_evidence(
                    source=source,
                    transform=binding.transform,
                    destination_slot=slot,
                    source_filenames=source_filenames,
                    source_tensor_sha256=source_tensor_sha256,
                )
                for source in binding.sources
            )
            tensor_evidence[slot].append(
                FeatureRuntimeTensorEvidence(
                    name=name,
                    byte_count=tensor_bytes[slot],
                    padding=binding.is_padding,
                    sha256=tensor_digests[slot].hexdigest(),
                    transform=binding.transform,
                    sources=sources_evidence,
                )
            )

    evidence = []
    source_file_evidence = tuple(
        (
            filename,
            _digest(
                verified_source_file_sha256[filename],
                field=f"{filename}.sha256",
            ),
        )
        for filename in source_filenames
    )
    for slot in range(4):
        plan = destination_by_slot[slot]
        if output_bytes[slot] != plan.file_bytes:
            raise CheckpointValidationError(
                f"feature runtime output size drifted for {plan.filename!r}"
            )
        evidence.append(
            StreamedFeatureRuntimeFileEvidence(
                filename=plan.filename,
                file_bytes=output_bytes[slot],
                payload_bytes=plan.payload_bytes,
                sha256=file_digests[slot].hexdigest(),
                source_files=source_file_evidence,
                source_payload_bytes=plan.device_layout.source_bytes,
                source_tensor_count=plan.device_layout.source_leaf_count,
                padding_bytes=plan.device_layout.padding_bytes,
                tensors=tuple(tensor_evidence[slot]),
            )
        )
    return tuple(evidence)
