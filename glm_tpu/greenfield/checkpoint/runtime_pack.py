"""Streaming derivative from final-owner leaves to executable slot tensors."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import struct
from typing import Any, BinaryIO, Sequence

from ..errors import CheckpointValidationError
from ..model.weights import (
    DecoderRuntimeWeightLayout,
    DeviceRuntimeWeightLayout,
    RuntimeTensorSpec,
)
from ..partitioning import BASE_LOAD_SET
from .stream_pack import DestinationFilePlan, DestinationTensorPlan


RUNTIME_LAYOUT_ARTIFACT_KIND = "greenfield_decoder_runtime_weight_layout"
RUNTIME_PACK_CONTROL_KIND = "greenfield_runtime_checkpoint_pack_control"
RUNTIME_PACKED_ARTIFACT_KIND = "greenfield_runtime_packed_checkpoint"
RUNTIME_FORMAT_VERSION = 1


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
class RuntimeDestinationTensorPlan:
    spec: RuntimeTensorSpec
    data_offset_start: int
    data_offset_end: int

    @property
    def byte_count(self) -> int:
        return self.data_offset_end - self.data_offset_start


@dataclass(frozen=True, slots=True)
class RuntimeDestinationFilePlan:
    filename: str
    source_filename: str
    stage_id: int
    device_slot: int
    device_id: int
    header: bytes
    payload_bytes: int
    tensors: tuple[RuntimeDestinationTensorPlan, ...]
    device_layout: DeviceRuntimeWeightLayout
    runtime_layout_hash: str
    source_packed_manifest_sha256: str

    @property
    def file_bytes(self) -> int:
        return len(self.header) + self.payload_bytes


@dataclass(frozen=True, slots=True)
class RuntimeTensorEvidence:
    name: str
    byte_count: int
    padding: bool
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_count": self.byte_count,
            "name": self.name,
            "padding": self.padding,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class StreamedRuntimeFileEvidence:
    filename: str
    file_bytes: int
    payload_bytes: int
    sha256: str
    source_filename: str
    source_file_sha256: str
    source_payload_bytes: int
    source_leaf_count: int
    padding_bytes: int
    tensors: tuple[RuntimeTensorEvidence, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_bytes": self.file_bytes,
            "filename": self.filename,
            "padding_bytes": self.padding_bytes,
            "payload_bytes": self.payload_bytes,
            "sha256": self.sha256,
            "source_filename": self.source_filename,
            "source_file_sha256": self.source_file_sha256,
            "source_leaf_count": self.source_leaf_count,
            "source_payload_bytes": self.source_payload_bytes,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }


def build_runtime_layout_document(
    layout: DecoderRuntimeWeightLayout,
) -> dict[str, Any]:
    """Return the self-authenticating semantic layout stored with an artifact."""

    value: dict[str, Any] = {
        "artifact_kind": RUNTIME_LAYOUT_ARTIFACT_KIND,
        "format_version": RUNTIME_FORMAT_VERSION,
        "layout": layout.to_dict(),
        "runtime_layout_hash": layout.layout_hash,
    }
    value["manifest_sha256"] = _mapping_hash(
        value,
        hash_field="manifest_sha256",
    )
    return value


def _runtime_header(
    *,
    filename: str,
    tensors: Sequence[RuntimeDestinationTensorPlan],
    runtime_layout_hash: str,
    source_packed_manifest_sha256: str,
) -> bytes:
    value: dict[str, Any] = {
        "__metadata__": {
            "format": "pt",
            "greenfield_artifact_kind": "greenfield_runtime_weight_file",
            "greenfield_destination_filename": filename,
            "greenfield_runtime_layout_hash": runtime_layout_hash,
            "greenfield_source_packed_manifest_sha256": (
                source_packed_manifest_sha256
            ),
        }
    }
    for tensor in tensors:
        spec = tensor.spec
        storage_dtype = "U8" if spec.dtype == "F8_E4M3" else spec.dtype
        value[spec.name] = {
            "data_offsets": [
                tensor.data_offset_start,
                tensor.data_offset_end,
            ],
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


def build_runtime_destination_file_plans(
    layout: DecoderRuntimeWeightLayout,
    source_plans: Sequence[DestinationFilePlan],
    *,
    source_packed_manifest_sha256: str,
) -> tuple[RuntimeDestinationFilePlan, ...]:
    """Build one uniform runtime file plan for every base physical owner."""

    source_hash = _digest(
        source_packed_manifest_sha256,
        field="source_packed_manifest_sha256",
    )
    source_by_owner = {
        (plan.stage_id, plan.device_slot): plan
        for plan in source_plans
        if plan.load_set == BASE_LOAD_SET
    }
    if len(source_by_owner) != len(layout.devices):
        raise CheckpointValidationError(
            "runtime derivative source files do not cover every base owner"
        )
    layout_hash = layout.layout_hash
    plans = []
    for device in layout.devices:
        key = (device.stage_id, device.device_slot)
        try:
            source = source_by_owner[key]
        except KeyError as error:
            raise CheckpointValidationError(
                f"runtime derivative lacks source owner {key}"
            ) from error
        if source.device_id != device.device_id:
            raise CheckpointValidationError(
                f"runtime/source physical device disagrees for owner {key}"
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
                "runtime derivative tensor bytes do not reconcile"
            )
        filename = (
            f"base_decoder_runtime/stage_{device.stage_id:02d}/"
            f"device_slot_{device.device_slot:02d}.safetensors"
        )
        tensor_tuple = tuple(tensors)
        plans.append(
            RuntimeDestinationFilePlan(
                filename=filename,
                source_filename=source.filename,
                stage_id=device.stage_id,
                device_slot=device.device_slot,
                device_id=device.device_id,
                header=_runtime_header(
                    filename=filename,
                    tensors=tensor_tuple,
                    runtime_layout_hash=layout_hash,
                    source_packed_manifest_sha256=source_hash,
                ),
                payload_bytes=offset,
                tensors=tensor_tuple,
                device_layout=device,
                runtime_layout_hash=layout_hash,
                source_packed_manifest_sha256=source_hash,
            )
        )
    return tuple(plans)


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
                f"runtime source truncated with {byte_count - copied} bytes remaining"
            )
        written = output.write(value)
        if written is not None and written != len(value):
            raise CheckpointValidationError("runtime output write was incomplete")
        file_digest.update(value)
        tensor_digest.update(value)
        copied += len(value)
    return copied


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
        output_count = output.write(value)
        if output_count is not None and output_count != len(value):
            raise CheckpointValidationError("runtime padding write was incomplete")
        file_digest.update(value)
        tensor_digest.update(value)
        written += len(value)
    return written


def stream_runtime_weight_file(
    *,
    source_plan: DestinationFilePlan,
    runtime_plan: RuntimeDestinationFilePlan,
    source: BinaryIO,
    output: BinaryIO,
    verified_source_file_sha256: str,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> StreamedRuntimeFileEvidence:
    """Stream one verified final owner into its padded executable-ready file.

    Callers must verify the source file SHA-256 against the protected packed
    manifest before entering this transformation.  This function then proves
    the exact source leaf set/shapes/dtypes and hashes every output tensor and
    complete runtime file while using bounded host memory.
    """

    if chunk_bytes <= 0:
        raise ValueError("runtime pack chunk_bytes must be positive")
    source_file_sha256 = _digest(
        verified_source_file_sha256,
        field="verified_source_file_sha256",
    )
    identity = (
        source_plan.filename == runtime_plan.source_filename
        and source_plan.stage_id == runtime_plan.stage_id
        and source_plan.device_slot == runtime_plan.device_slot
        and source_plan.device_id == runtime_plan.device_id
        and source_plan.load_set == BASE_LOAD_SET
    )
    if not identity:
        raise CheckpointValidationError("runtime/source file identities disagree")
    source_by_name = {tensor.name: tensor for tensor in source_plan.tensors}
    if len(source_by_name) != len(source_plan.tensors):
        raise CheckpointValidationError("runtime source tensor names are duplicate")
    expected_sources = {
        leaf.name: leaf
        for tensor in runtime_plan.device_layout.tensors
        for leaf in tensor.sources
    }
    if set(source_by_name) != set(expected_sources):
        raise CheckpointValidationError(
            "runtime derivative does not consume the exact source leaf set"
        )
    for name, leaf in expected_sources.items():
        tensor = source_by_name[name]
        if (
            tensor.dtype != leaf.dtype
            or tensor.shape != leaf.shape
            or tensor.byte_count != leaf.byte_count
        ):
            raise CheckpointValidationError(
                f"runtime source contract drifted for {name!r}"
            )
    source.seek(0)
    if source.read(len(source_plan.header)) != source_plan.header:
        raise CheckpointValidationError("runtime source safetensors header drifted")

    header_count = output.write(runtime_plan.header)
    if header_count is not None and header_count != len(runtime_plan.header):
        raise CheckpointValidationError("runtime header write was incomplete")
    file_digest = sha256(runtime_plan.header)
    output_bytes = len(runtime_plan.header)
    tensor_evidence = []
    binding_by_name = {
        tensor.spec.name: tensor
        for tensor in runtime_plan.device_layout.tensors
    }
    for destination in runtime_plan.tensors:
        binding = binding_by_name[destination.spec.name]
        digest = sha256()
        tensor_bytes = 0
        if binding.is_padding:
            tensor_bytes += _write_zeros(
                output,
                byte_count=destination.byte_count,
                file_digest=file_digest,
                tensor_digest=digest,
                chunk_bytes=chunk_bytes,
            )
        else:
            for leaf in binding.sources:
                source_tensor: DestinationTensorPlan = source_by_name[leaf.name]
                source.seek(
                    len(source_plan.header) + source_tensor.data_offset_start
                )
                tensor_bytes += _copy_exact(
                    source,
                    output,
                    byte_count=source_tensor.byte_count,
                    file_digest=file_digest,
                    tensor_digest=digest,
                    chunk_bytes=chunk_bytes,
                )
        if tensor_bytes != destination.byte_count:
            raise CheckpointValidationError(
                f"runtime tensor {destination.spec.name!r} byte count drifted"
            )
        output_bytes += tensor_bytes
        tensor_evidence.append(
            RuntimeTensorEvidence(
                name=destination.spec.name,
                byte_count=tensor_bytes,
                padding=binding.is_padding,
                sha256=digest.hexdigest(),
            )
        )
    if output_bytes != runtime_plan.file_bytes:
        raise CheckpointValidationError("runtime output file size drifted")
    return StreamedRuntimeFileEvidence(
        filename=runtime_plan.filename,
        file_bytes=output_bytes,
        payload_bytes=runtime_plan.payload_bytes,
        sha256=file_digest.hexdigest(),
        source_filename=source_plan.filename,
        source_file_sha256=source_file_sha256,
        source_payload_bytes=runtime_plan.device_layout.source_bytes,
        source_leaf_count=runtime_plan.device_layout.source_leaf_count,
        padding_bytes=runtime_plan.device_layout.padding_bytes,
        tensors=tuple(tensor_evidence),
    )


def verify_source_file_sha256(path: Path, expected_sha256: str) -> str:
    """Sequentially authenticate a source owner before runtime derivation."""

    expected = _digest(expected_sha256, field="expected_sha256")
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected:
        raise CheckpointValidationError(
            f"runtime source file SHA-256 mismatch: expected={expected} observed={observed}"
        )
    return observed
