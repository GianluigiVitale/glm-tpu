from __future__ import annotations

import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.checkpoint import (
    RUNTIME_FORMAT_VERSION,
    RUNTIME_PACK_CONTROL_KIND,
    RUNTIME_PACKED_ARTIFACT_KIND,
    RuntimeCheckpointLoadExpectation,
    VerifiedPackedCheckpoint,
    build_runtime_destination_file_plans,
    build_runtime_layout_document,
    load_runtime_checkpoint,
    stream_runtime_weight_file,
    verify_runtime_packed_checkpoint,
)
from glm_tpu.greenfield.errors import CheckpointValidationError
from glm_tpu.greenfield.model import (
    build_decoder_runtime_weight_layout,
    build_pipeline_schedule,
)
from glm_tpu.greenfield.types import ExecutionPlan
from tests.greenfield.checkpoint.test_runtime_pack import (
    _small_plan,
    _source_plans,
)

SOURCE_MANIFEST_SHA = "2" * 64
SOURCE_LAYOUT_SHA = "3" * 64
PACK_CODE_HASH = "4" * 40
SOURCE_DESTINATION = (
    "gs://driftbench-dsv4-uc/checkpoints/greenfield/test/source"
)
RUNTIME_DESTINATION = (
    "gs://driftbench-dsv4-uc/checkpoints/greenfield/test/runtime"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_hash(value: dict[str, object], field: str) -> str:
    unhashed = dict(value)
    unhashed.pop(field, None)
    return sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_file(plan: object) -> bytes:
    return plan.header + bytes([1]) * plan.payload_bytes


def _build_artifact(
    tmp_path: Path,
    plan: ExecutionPlan | None = None,
) -> tuple[object, object, object, object]:
    if plan is None:
        plan = _small_plan()
    schedule = build_pipeline_schedule(plan)
    layout = build_decoder_runtime_weight_layout(plan, schedule)
    source_plans = _source_plans(layout)
    source_evidence = {}
    for source_plan in source_plans:
        value = _source_file(source_plan)
        source_evidence[source_plan.filename] = {
            "sha256": sha256(value).hexdigest()
        }
    source_checkpoint = VerifiedPackedCheckpoint(
        root=tmp_path / "source",
        layout={"manifest_sha256": SOURCE_LAYOUT_SHA},
        packed_manifest={
            "destination": SOURCE_DESTINATION,
            "manifest_sha256": SOURCE_MANIFEST_SHA,
        },
        control={},
        plans=source_plans,
        evidence_by_filename=source_evidence,
    )
    runtime_plans = build_runtime_destination_file_plans(
        layout,
        source_plans,
        source_packed_manifest_sha256=SOURCE_MANIFEST_SHA,
    )
    layout_document = build_runtime_layout_document(layout)
    layout_path = tmp_path / "runtime_layout.json"
    _write_json(layout_path, layout_document)
    common: dict[str, object] = {
        "destination": RUNTIME_DESTINATION,
        "file_count": len(runtime_plans),
        "format_version": RUNTIME_FORMAT_VERSION,
        "model_id": "zai-org/GLM-5.2-FP8",
        "pack_code_hash": PACK_CODE_HASH,
        "padding_bytes": sum(
            device.padding_bytes for device in layout.devices
        ),
        "plan_hash": layout.plan_hash,
        "plan_id": "PP8_LP4",
        "runtime_file_bytes": sum(item.file_bytes for item in runtime_plans),
        "runtime_layout_hash": layout.layout_hash,
        "runtime_layout_manifest_sha256": layout_document["manifest_sha256"],
        "runtime_payload_bytes": sum(
            item.payload_bytes for item in runtime_plans
        ),
        "schedule_hash": layout.schedule_hash,
        "source_checkpoint_destination": SOURCE_DESTINATION,
        "source_layout_manifest_sha256": SOURCE_LAYOUT_SHA,
        "source_leaf_count": layout.source_leaf_count,
        "source_packed_manifest_sha256": SOURCE_MANIFEST_SHA,
        "source_payload_bytes": sum(
            device.source_bytes for device in layout.devices
        ),
        "tensor_count": len(layout.specs) * len(layout.devices),
    }
    control: dict[str, object] = {
        "artifact_kind": RUNTIME_PACK_CONTROL_KIND,
        **common,
        "runtime_layout_file_sha256": sha256(
            layout_path.read_bytes()
        ).hexdigest(),
    }
    control["control_sha256"] = _mapping_hash(control, "control_sha256")
    _write_json(tmp_path / "control.json", control)

    records = []
    source_by_filename = {item.filename: item for item in source_plans}
    for runtime_plan in runtime_plans:
        source_plan = source_by_filename[runtime_plan.source_filename]
        source_value = _source_file(source_plan)
        target = tmp_path / runtime_plan.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as output:
            evidence = stream_runtime_weight_file(
                source_plan=source_plan,
                runtime_plan=runtime_plan,
                source=BytesIO(source_value),
                output=output,
                verified_source_file_sha256=sha256(source_value).hexdigest(),
                chunk_bytes=31,
            )
        record = evidence.to_dict()
        record["destination_filename"] = record.pop("filename")
        record.update(
            {
                "crc32c": "AAAAAA==",
                "device_id": runtime_plan.device_id,
                "device_slot": runtime_plan.device_slot,
                "generation": runtime_plan.device_id + 1,
                "header_bytes": len(runtime_plan.header),
                "header_sha256": sha256(runtime_plan.header).hexdigest(),
                "runtime_layout_hash": layout.layout_hash,
                "source_packed_manifest_sha256": SOURCE_MANIFEST_SHA,
                "stage_id": runtime_plan.stage_id,
                "tensor_count": len(runtime_plan.tensors),
            }
        )
        records.append(record)
        _write_json(
            tmp_path / "evidence" / f"{runtime_plan.filename}.json",
            record,
        )
    manifest: dict[str, object] = {
        "artifact_kind": RUNTIME_PACKED_ARTIFACT_KIND,
        **common,
        "control_sha256": control["control_sha256"],
        "files": records,
    }
    manifest["manifest_sha256"] = _mapping_hash(manifest, "manifest_sha256")
    _write_json(tmp_path / "runtime_manifest.json", manifest)
    (tmp_path / "SUCCESS").write_text(
        f"{manifest['manifest_sha256']}  runtime_manifest.json\n"
    )
    expectation = RuntimeCheckpointLoadExpectation(
        runtime_manifest_sha256=manifest["manifest_sha256"],
        runtime_layout_manifest_sha256=layout_document["manifest_sha256"],
        runtime_layout_hash=layout.layout_hash,
        source_packed_manifest_sha256=SOURCE_MANIFEST_SHA,
        source_layout_manifest_sha256=SOURCE_LAYOUT_SHA,
        plan_hash=layout.plan_hash,
        schedule_hash=layout.schedule_hash,
        pack_code_hash=PACK_CODE_HASH,
        destination=RUNTIME_DESTINATION,
        source_destination=SOURCE_DESTINATION,
    )
    return layout, source_checkpoint, expectation, runtime_plans


def test_runtime_artifact_verification_is_complete_and_fail_closed(
    tmp_path: Path,
) -> None:
    layout, source_checkpoint, expectation, runtime_plans = _build_artifact(
        tmp_path
    )
    verified = verify_runtime_packed_checkpoint(
        tmp_path,
        expectation,
        layout,
        source_checkpoint,
    )
    assert len(verified.plans) == len(runtime_plans) == 32
    assert len(verified.evidence_by_filename) == 32

    sidecar = tmp_path / "evidence" / f"{runtime_plans[0].filename}.json"
    value = json.loads(sidecar.read_text())
    value["generation"] += 1
    _write_json(sidecar, value)
    with pytest.raises(CheckpointValidationError, match="sidecar disagrees"):
        verify_runtime_packed_checkpoint(
            tmp_path,
            expectation,
            layout,
            source_checkpoint,
        )


def test_runtime_loader_reuses_final_owner_buffers_without_reshard(
    tmp_path: Path,
) -> None:
    import jax
    from jax.sharding import Mesh

    if len(jax.devices()) != 32:
        pytest.skip("requires 32 forced CPU devices")
    layout, source_checkpoint, expectation, _ = _build_artifact(tmp_path)
    verified = verify_runtime_packed_checkpoint(
        tmp_path,
        expectation,
        layout,
        source_checkpoint,
    )
    mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("device",))
    loaded = load_runtime_checkpoint(
        verified,
        expectation,
        layout,
        mesh,
        verify_device_roundtrip=True,
        chunk_bytes=37,
    )
    try:
        assert len(loaded.weights) == len(layout.specs)
        assert loaded.addressable_device_ids == tuple(range(32))
        assert loaded.load_record["runtime_checkpoint_reshards"] == 0
        assert loaded.load_record["host_global_concatenations"] == 0
        assert loaded.load_record["fp8_device_dequantizations"] == 0
        assert loaded.load_record["fp8_host_dequantizations"] == 0
        assert loaded.load_record["device_roundtrip_bytes"] == (
            loaded.load_record["loaded_payload_bytes"]
        )
        final_norm = np.asarray(jax.device_get(loaded.weights["global.final_norm"]))
        assert final_norm.shape == (32, 8)
        assert np.count_nonzero(final_norm) == 4 * 8
        attention = np.asarray(
            jax.device_get(
                loaded.weights["attention.slot_00.q_a.weight_bits"]
            )
        )
        assert attention.shape == (32, 4, 8)
        assert np.all(attention == 1)
    finally:
        loaded.close()
    assert loaded.closed
    assert not loaded.weights
