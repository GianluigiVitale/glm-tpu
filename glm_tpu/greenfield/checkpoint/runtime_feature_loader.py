"""Fail-closed verifier for complete PP8/PP16 feature-runtime derivatives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..errors import CheckpointValidationError
from ..model.weights import (
    LEGACY_DENSE_RUNTIME_LAYOUT,
    SEPARATE_QKV_A_RUNTIME_LAYOUT,
    DecoderRuntimeWeightLayout,
)
from .full_loader import _mapping_hash, _read_json, _sha256_file
from .runtime_feature import (
    FEATURE_RUNTIME_FORMAT_VERSION,
    FEATURE_RUNTIME_LAYOUT_ARTIFACT_KIND,
    FEATURE_RUNTIME_PACK_CONTROL_KIND,
    FEATURE_RUNTIME_PACKED_ARTIFACT_KIND,
    FeatureRuntimeDestinationFilePlan,
    _source_evidence,
    build_feature_runtime_destination_file_plans,
)
from .runtime_loader import VerifiedRuntimeCheckpoint


def _digest(value: str, *, field: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase digest")
    return value


@dataclass(frozen=True, slots=True)
class FeatureRuntimeCheckpointLoadExpectation:
    """Externally pinned identities required before feature bytes are read."""

    runtime_manifest_sha256: str
    runtime_layout_manifest_sha256: str
    runtime_layout_hash: str
    source_runtime_manifest_sha256: str
    source_runtime_layout_manifest_sha256: str
    source_runtime_layout_hash: str
    plan_hash: str
    schedule_hash: str
    pack_code_hash: str
    destination: str
    source_destination: str
    plan_id: str = "PP8_LP4"
    model_id: str = "zai-org/GLM-5.2-FP8"

    def __post_init__(self) -> None:
        for field in (
            "runtime_manifest_sha256",
            "runtime_layout_manifest_sha256",
            "runtime_layout_hash",
            "source_runtime_manifest_sha256",
            "source_runtime_layout_manifest_sha256",
            "source_runtime_layout_hash",
            "plan_hash",
            "schedule_hash",
        ):
            _digest(getattr(self, field), field=field)
        _digest(self.pack_code_hash, field="pack_code_hash", lengths=(40, 64))
        for field in ("destination", "source_destination"):
            value = getattr(self, field)
            if not value.startswith("gs://driftbench-dsv4-uc/checkpoints/greenfield/"):
                raise ValueError(f"{field} must be a greenfield approved-bucket prefix")
        if self.plan_id not in ("PP8_LP4", "PP16_LP2"):
            raise ValueError(
                "feature runtime loader supports only PP8_LP4 and PP16_LP2"
            )
        if self.model_id != "zai-org/GLM-5.2-FP8":
            raise ValueError("feature runtime loader supports only GLM-5.2-FP8")


def _feature_runtime_totals(
    layout: DecoderRuntimeWeightLayout,
    plans: Sequence[FeatureRuntimeDestinationFilePlan],
) -> dict[str, int]:
    return {
        "file_count": len(plans),
        "padding_bytes": sum(device.padding_bytes for device in layout.devices),
        "runtime_file_bytes": sum(plan.file_bytes for plan in plans),
        "runtime_payload_bytes": sum(plan.payload_bytes for plan in plans),
        "source_payload_bytes": sum(device.source_bytes for device in layout.devices),
        "source_tensor_count": layout.source_leaf_count,
        "tensor_count": len(layout.specs) * len(layout.devices),
    }


def _source_hashes(
    source_checkpoint: VerifiedRuntimeCheckpoint,
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    files = {}
    tensors = {}
    for filename, record in source_checkpoint.evidence_by_filename.items():
        files[filename] = _digest(
            record.get("sha256"),
            field=f"{filename}.sha256",
        )
        tensor_records = record.get("tensors")
        if not isinstance(tensor_records, list):
            raise CheckpointValidationError(
                f"feature runtime source lacks tensor evidence for {filename!r}"
            )
        for tensor in tensor_records:
            if not isinstance(tensor, Mapping) or not isinstance(
                tensor.get("name"), str
            ):
                raise CheckpointValidationError(
                    "feature runtime source tensor evidence is invalid"
                )
            key = (filename, tensor["name"])
            if key in tensors:
                raise CheckpointValidationError(
                    "feature runtime source tensor evidence is duplicate"
                )
            tensors[key] = _digest(
                tensor.get("sha256"),
                field=f"{filename}:{tensor['name']}.sha256",
            )
    return files, tensors


def verify_feature_runtime_packed_checkpoint(
    root: Path,
    expectation: FeatureRuntimeCheckpointLoadExpectation,
    layout: DecoderRuntimeWeightLayout,
    source_checkpoint: VerifiedRuntimeCheckpoint,
) -> VerifiedRuntimeCheckpoint:
    """Verify the semantic layout, transform ledger, sizes, and completion."""

    root = Path(root)
    required = {
        "control": root / "control.json",
        "layout": root / "runtime_layout.json",
        "manifest": root / "runtime_manifest.json",
        "success": root / "SUCCESS",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise CheckpointValidationError(
            f"feature runtime checkpoint is incomplete; missing {sorted(missing)}"
        )
    control = _read_json(required["control"])
    layout_document = _read_json(required["layout"])
    manifest = _read_json(required["manifest"])
    control_hash = _mapping_hash(control, hash_field="control_sha256")
    layout_manifest_hash = _mapping_hash(
        layout_document,
        hash_field="manifest_sha256",
    )
    manifest_hash = _mapping_hash(manifest, hash_field="manifest_sha256")
    if manifest_hash != expectation.runtime_manifest_sha256:
        raise CheckpointValidationError("feature runtime manifest identity drifted")
    if layout_manifest_hash != expectation.runtime_layout_manifest_sha256:
        raise CheckpointValidationError(
            "feature runtime layout manifest identity drifted"
        )
    if control.get("artifact_kind") != FEATURE_RUNTIME_PACK_CONTROL_KIND:
        raise CheckpointValidationError("wrong feature runtime checkpoint control kind")
    if layout_document.get("artifact_kind") != FEATURE_RUNTIME_LAYOUT_ARTIFACT_KIND:
        raise CheckpointValidationError("wrong feature runtime layout kind")
    if manifest.get("artifact_kind") != FEATURE_RUNTIME_PACKED_ARTIFACT_KIND:
        raise CheckpointValidationError("wrong feature runtime checkpoint kind")
    if any(
        value.get("format_version") != FEATURE_RUNTIME_FORMAT_VERSION
        for value in (control, layout_document, manifest)
    ):
        raise CheckpointValidationError(
            "feature runtime checkpoint format version drifted"
        )
    if (
        layout_document.get("runtime_layout_hash") != layout.layout_hash
        or layout_document.get("layout") != layout.to_dict()
        or layout.layout_hash != expectation.runtime_layout_hash
    ):
        raise CheckpointValidationError(
            "feature runtime semantic weight layout drifted"
        )
    source_manifest_hash = source_checkpoint.runtime_manifest.get("manifest_sha256")
    source_layout_manifest_hash = source_checkpoint.runtime_layout_document.get(
        "manifest_sha256"
    )
    source_layout_hash = source_checkpoint.runtime_layout_document.get(
        "runtime_layout_hash"
    )
    if (
        source_manifest_hash != expectation.source_runtime_manifest_sha256
        or source_layout_manifest_hash
        != expectation.source_runtime_layout_manifest_sha256
        or source_layout_hash != expectation.source_runtime_layout_hash
        or source_checkpoint.runtime_manifest.get("plan_id")
        != expectation.plan_id
        or source_checkpoint.runtime_manifest.get("destination")
        != expectation.source_destination
    ):
        raise CheckpointValidationError(
            "feature runtime source checkpoint identity drifted"
        )
    plans = build_feature_runtime_destination_file_plans(
        layout,
        source_checkpoint.plans,
        source_runtime_manifest_sha256=(expectation.source_runtime_manifest_sha256),
    )
    totals = _feature_runtime_totals(layout, plans)
    common = {
        "destination": expectation.destination,
        "format_version": FEATURE_RUNTIME_FORMAT_VERSION,
        "model_id": expectation.model_id,
        "pack_code_hash": expectation.pack_code_hash,
        "plan_hash": expectation.plan_hash,
        "plan_id": expectation.plan_id,
        "routed_expert_layout": layout.routed_expert_layout,
        "runtime_layout_hash": expectation.runtime_layout_hash,
        "runtime_layout_manifest_sha256": (expectation.runtime_layout_manifest_sha256),
        "schedule_hash": expectation.schedule_hash,
        "source_checkpoint_destination": expectation.source_destination,
        "source_runtime_layout_hash": expectation.source_runtime_layout_hash,
        "source_runtime_layout_manifest_sha256": (
            expectation.source_runtime_layout_manifest_sha256
        ),
        "source_runtime_manifest_sha256": (expectation.source_runtime_manifest_sha256),
        **totals,
    }
    if layout.attention_projection_layout != SEPARATE_QKV_A_RUNTIME_LAYOUT:
        common["attention_projection_layout"] = (
            layout.attention_projection_layout
        )
    if layout.dense_projection_layout != LEGACY_DENSE_RUNTIME_LAYOUT:
        common["dense_projection_layout"] = layout.dense_projection_layout
    for name, value in (("control", control), ("manifest", manifest)):
        for field, expected in common.items():
            if value.get(field) != expected:
                raise CheckpointValidationError(
                    f"feature runtime {name} drifted field {field!r}"
                )
    if control.get("runtime_layout_file_sha256") != _sha256_file(required["layout"]):
        raise CheckpointValidationError("feature runtime layout file SHA-256 drifted")
    if manifest.get("control_sha256") != control_hash:
        raise CheckpointValidationError(
            "feature runtime manifest does not bind control"
        )

    records = manifest.get("files")
    if not isinstance(records, list):
        raise CheckpointValidationError(
            "feature runtime manifest lacks its file ledger"
        )
    evidence_by_filename: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise CheckpointValidationError(
                "feature runtime file record is not an object"
            )
        filename = record.get("destination_filename")
        if not isinstance(filename, str) or filename in evidence_by_filename:
            raise CheckpointValidationError(
                "feature runtime filenames are invalid or duplicate"
            )
        evidence_by_filename[filename] = record
    if set(evidence_by_filename) != {plan.filename for plan in plans}:
        raise CheckpointValidationError("feature runtime file ledger is incomplete")

    source_file_hashes, source_tensor_hashes = _source_hashes(source_checkpoint)
    for plan in plans:
        record = evidence_by_filename[plan.filename]
        expected_source_files = [
            {
                "filename": filename,
                "sha256": source_file_hashes[filename],
            }
            for filename in plan.source_filenames
        ]
        expected_record = {
            "destination_filename": plan.filename,
            "device_id": plan.device_id,
            "device_slot": plan.device_slot,
            "file_bytes": plan.file_bytes,
            "header_bytes": len(plan.header),
            "header_sha256": sha256(plan.header).hexdigest(),
            "padding_bytes": plan.device_layout.padding_bytes,
            "payload_bytes": plan.payload_bytes,
            "runtime_layout_hash": plan.runtime_layout_hash,
            "source_files": expected_source_files,
            "source_payload_bytes": plan.device_layout.source_bytes,
            "source_runtime_manifest_sha256": (
                expectation.source_runtime_manifest_sha256
            ),
            "source_tensor_count": plan.device_layout.source_leaf_count,
            "stage_id": plan.stage_id,
            "tensor_count": len(plan.tensors),
        }
        for field, expected in expected_record.items():
            if record.get(field) != expected:
                raise CheckpointValidationError(
                    f"feature runtime file {plan.filename!r} drifted field {field!r}"
                )
        _digest(record.get("sha256"), field=f"{plan.filename}.sha256")
        if (
            not isinstance(record.get("generation"), int)
            or record["generation"] <= 0
            or not isinstance(record.get("crc32c"), str)
            or not record["crc32c"]
        ):
            raise CheckpointValidationError(
                f"feature runtime file {plan.filename!r} lacks GCS identity"
            )
        tensor_records = record.get("tensors")
        if not isinstance(tensor_records, list):
            raise CheckpointValidationError(
                f"feature runtime file {plan.filename!r} lacks tensor evidence"
            )
        by_name = {}
        for tensor_record in tensor_records:
            if not isinstance(tensor_record, Mapping):
                raise CheckpointValidationError(
                    "feature runtime tensor record is invalid"
                )
            name = tensor_record.get("name")
            if not isinstance(name, str) or name in by_name:
                raise CheckpointValidationError(
                    "feature runtime tensor names are invalid or duplicate"
                )
            by_name[name] = tensor_record
        bindings = {tensor.spec.name: tensor for tensor in plan.device_layout.tensors}
        if set(by_name) != {tensor.spec.name for tensor in plan.tensors}:
            raise CheckpointValidationError(
                f"feature runtime tensor ledger is incomplete for {plan.filename!r}"
            )
        for tensor in plan.tensors:
            binding = bindings[tensor.spec.name]
            expected_sources = [
                _source_evidence(
                    source=source,
                    transform=binding.transform,
                    destination_slot=plan.device_slot,
                    source_filenames=plan.source_filenames,
                    source_tensor_sha256=source_tensor_hashes,
                ).to_dict()
                for source in binding.sources
            ]
            tensor_record = by_name[tensor.spec.name]
            expected_tensor = {
                "byte_count": tensor.byte_count,
                "name": tensor.spec.name,
                "padding": binding.is_padding,
                "sources": expected_sources,
                "transform": binding.transform,
            }
            if any(
                tensor_record.get(field) != expected
                for field, expected in expected_tensor.items()
            ):
                raise CheckpointValidationError(
                    f"feature runtime tensor evidence drifted for {tensor.spec.name!r}"
                )
            _digest(
                tensor_record.get("sha256"),
                field=f"{plan.filename}:{tensor.spec.name}.sha256",
            )
        sidecar = _read_json(root / "evidence" / f"{plan.filename}.json")
        if sidecar != dict(record):
            raise CheckpointValidationError(
                f"feature runtime file {plan.filename!r} sidecar disagrees"
            )
        try:
            observed_size = (root / plan.filename).stat().st_size
        except OSError as error:
            raise CheckpointValidationError(
                f"feature runtime payload is missing: {plan.filename}"
            ) from error
        if observed_size != plan.file_bytes:
            raise CheckpointValidationError(
                f"feature runtime payload size drift for {plan.filename!r}"
            )
    expected_success = f"{manifest_hash}  runtime_manifest.json\n"
    if required["success"].read_text() != expected_success:
        raise CheckpointValidationError(
            "feature runtime checkpoint SUCCESS marker drifted"
        )
    return VerifiedRuntimeCheckpoint(
        root=root,
        control=control,
        runtime_layout_document=layout_document,
        runtime_manifest=manifest,
        plans=plans,  # type: ignore[arg-type]
        evidence_by_filename=evidence_by_filename,
    )
