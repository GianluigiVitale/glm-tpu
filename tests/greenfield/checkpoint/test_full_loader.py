from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from glm_tpu.greenfield.checkpoint import (
    FullCheckpointLoadExpectation,
    StageDeviceResolution,
    load_final_layout_stage,
    verify_full_packed_checkpoint,
)
from glm_tpu.greenfield.checkpoint.full_loader import _validate_finite
from glm_tpu.greenfield.errors import CheckpointValidationError
from tests.greenfield.checkpoint.test_stream_pack import pack_all


PACK_CODE_HASH = "f" * 40
DESTINATION = (
    "gs://driftbench-dsv4-uc/checkpoints/greenfield/tests/"
    "full-loader-fixture"
)


def _canonical_json(value: dict[str, object]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_hash(value: dict[str, object]) -> str:
    return sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _fixture_artifact(
    tmp_path: Path,
) -> tuple[Path, FullCheckpointLoadExpectation]:
    _, layout, plans, root, streamed = pack_all(tmp_path)
    by_filename = {item.filename: item for item in streamed}
    layout_path = root / "layout_manifest.json"
    _write_json(layout_path, layout)
    control: dict[str, object] = {
        "artifact_kind": "greenfield_streaming_full_checkpoint_pack",
        "code_hash": PACK_CODE_HASH,
        "destination": DESTINATION,
        "destination_file_count": len(plans),
        "expected_file_bytes": sum(plan.file_bytes for plan in plans),
        "layout_file_sha256": sha256(layout_path.read_bytes()).hexdigest(),
        "layout_manifest_sha256": layout["manifest_sha256"],
        "packed_payload_bytes": layout["packed_payload_bytes"],
        "plan_id": layout["plan_id"],
        "source_inventory_sha256": layout["source"]["inventory_sha256"],
        "source_revision": layout["source"]["revision"],
    }
    control["control_sha256"] = _mapping_hash(control)
    _write_json(root / "control.json", control)
    files = []
    for generation, plan in enumerate(plans, start=1):
        streamed_file = by_filename[plan.filename]
        record: dict[str, object] = {
            "crc32c": f"fixture-{generation}",
            "destination_filename": plan.filename,
            "device_id": plan.device_id,
            "device_slot": plan.device_slot,
            "file_bytes": plan.file_bytes,
            "generation": generation,
            "header_bytes": len(plan.header),
            "header_sha256": sha256(plan.header).hexdigest(),
            "layout_manifest_sha256": layout["manifest_sha256"],
            "load_set": plan.load_set,
            "pack_code_hash": PACK_CODE_HASH,
            "payload_bytes": plan.payload_bytes,
            "sha256": streamed_file.sha256,
            "stage_id": plan.stage_id,
        }
        files.append(record)
        _write_json(root / "evidence" / f"{plan.filename}.json", record)
    packed: dict[str, object] = {
        "artifact_kind": "greenfield_full_packed_checkpoint",
        "code_hash": PACK_CODE_HASH,
        "control_sha256": control["control_sha256"],
        "created_utc": "2026-08-05T00:00:00+00:00",
        "destination": DESTINATION,
        "elapsed_seconds": 1.0,
        "file_count": len(plans),
        "files": files,
        "layout_manifest_sha256": layout["manifest_sha256"],
        "packed_file_bytes": sum(plan.file_bytes for plan in plans),
        "packed_payload_bytes": layout["packed_payload_bytes"],
        "plan_id": layout["plan_id"],
        "source_inventory_sha256": layout["source"]["inventory_sha256"],
        "source_payload_bytes": layout["source"]["payload_bytes"],
        "source_revision": layout["source"]["revision"],
    }
    packed["manifest_sha256"] = _mapping_hash(packed)
    _write_json(root / "packed_manifest.json", packed)
    (root / "SUCCESS").write_text(
        f"{packed['manifest_sha256']}  packed_manifest.json\n"
    )
    expectation = FullCheckpointLoadExpectation(
        packed_manifest_sha256=str(packed["manifest_sha256"]),
        layout_manifest_sha256=str(layout["manifest_sha256"]),
        source_inventory_sha256=str(layout["source"]["inventory_sha256"]),
        source_revision=str(layout["source"]["revision"]),
        topology_hash=str(layout["topology_hash"]),
        plan_group_hash=str(layout["plan_group_hash"]),
        plan_manifest_sha256=str(layout["plan_manifest_sha256"]),
        execution_plan_sha256=str(
            layout["plan_manifest"]["execution_plan_sha256"]
        ),
        layout_code_hash=str(layout["code_hash"]),
        pack_code_hash=PACK_CODE_HASH,
        destination=DESTINATION,
        plan_id=str(layout["plan_id"]),
        model_id=str(layout["source"]["model_id"]),
    )
    _write_json(root / "test_expectation.json", asdict(expectation))
    return root, expectation


def _expectation(root: Path) -> FullCheckpointLoadExpectation:
    return FullCheckpointLoadExpectation(
        **json.loads((root / "test_expectation.json").read_text())
    )


def _resolution(checkpoint: object) -> StageDeviceResolution:
    import jax

    assignment = checkpoint.layout["plan_manifest"]["execution_plan"][
        "stage_assignments"
    ][0]
    devices = tuple(jax.devices())
    return StageDeviceResolution(
        devices=devices,
        coordinates=tuple((index,) for index in range(len(devices))),
        captured_device_ids=tuple(assignment["device_ids"]),
        captured_process_index=int(assignment["process_index"]),
        stage_id=int(assignment["stage_id"]),
    )


def _run_forced_cpu_loader(root: Path) -> None:
    import jax

    if len(jax.devices()) != 4:
        raise AssertionError(f"expected four forced CPU devices, got {jax.devices()}")
    expected = _expectation(root)
    checkpoint = verify_full_packed_checkpoint(root, expected)
    resolution = _resolution(checkpoint)
    loaded = load_final_layout_stage(
        checkpoint,
        expected,
        resolution,
        verify_device_roundtrip=True,
        chunk_bytes=17,
    )
    selected = [
        plan
        for plan in checkpoint.plans
        if plan.load_set == "base_decoder" and plan.stage_id == 0
    ]
    assert loaded.device_ids == resolution.captured_device_ids
    assert loaded.load_record["loaded_payload_bytes"] == sum(
        plan.payload_bytes for plan in selected
    )
    assert loaded.load_record["device_roundtrip_bytes"] == loaded.load_record[
        "loaded_payload_bytes"
    ]
    assert loaded.load_record["fp8_host_dequantizations"] == 0
    assert loaded.load_record["runtime_checkpoint_reshards"] == 0
    assert loaded.state_manifest["device_roundtrip_verified"] is True
    assert len(loaded.leaves_by_slot) == 4
    assert all(
        leaf.storage_dtype == "F32"
        for slot in loaded.leaves_by_slot
        for leaf in slot.values()
    )
    loaded.close()
    assert loaded.closed and all(not slot for slot in loaded.leaves_by_slot)


def _run_forced_cpu_corruption_refusal(root: Path) -> None:
    expected = _expectation(root)
    checkpoint = verify_full_packed_checkpoint(root, expected)
    with pytest.raises(CheckpointValidationError, match="SHA-256 mismatch"):
        load_final_layout_stage(
            checkpoint,
            expected,
            _resolution(checkpoint),
            verify_device_roundtrip=True,
            chunk_bytes=17,
        )


def _forced_cpu(root: Path, function: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=4".strip()
    )
    code = (
        "from pathlib import Path; "
        "from tests.greenfield.checkpoint.test_full_loader import "
        f"{function}; {function}(Path({str(root)!r}))"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def test_complete_artifact_verifier_reconciles_every_file(tmp_path: Path) -> None:
    root, expectation = _fixture_artifact(tmp_path)
    checkpoint = verify_full_packed_checkpoint(root, expectation)
    assert len(checkpoint.plans) == 33
    assert len(checkpoint.evidence_by_filename) == 33
    assert checkpoint.packed_manifest["packed_payload_bytes"] == 512


@pytest.mark.parametrize(
    ("dtype", "finite", "nonfinite"),
    (
        ("F8_E4M3", bytes((0x7E, 0xFE)), bytes((0x7F,))),
        ("BF16", bytes.fromhex("7f7f7fff"), bytes.fromhex("807f")),
        ("F32", bytes.fromhex("ffff7f7f"), bytes.fromhex("0000807f")),
    ),
)
def test_nonfinite_scan_rejects_fp8_bf16_and_f32_encodings(
    dtype: str,
    finite: bytes,
    nonfinite: bytes,
) -> None:
    _validate_finite(bytearray(finite), dtype, chunk_bytes=17)
    with pytest.raises(CheckpointValidationError, match="non-finite"):
        _validate_finite(bytearray(nonfinite), dtype, chunk_bytes=17)


def test_complete_artifact_verifier_refuses_missing_payload(tmp_path: Path) -> None:
    root, expectation = _fixture_artifact(tmp_path)
    target = next((root / "base_decoder").rglob("*.safetensors"))
    target.unlink()
    with pytest.raises(CheckpointValidationError, match="missing"):
        verify_full_packed_checkpoint(root, expectation)


def test_complete_artifact_metadata_lineage_survives_payload_reclamation(
    tmp_path: Path,
) -> None:
    root, expectation = _fixture_artifact(tmp_path)
    for plan in verify_full_packed_checkpoint(root, expectation).plans:
        (root / plan.filename).unlink()
        (root / "evidence" / f"{plan.filename}.json").unlink()
    verified = verify_full_packed_checkpoint(
        root,
        expectation,
        require_payloads=False,
    )
    assert len(verified.plans) == 33
    with pytest.raises(CheckpointValidationError, match="cannot parse|missing"):
        verify_full_packed_checkpoint(root, expectation)


def test_complete_artifact_verifier_refuses_stale_identity(tmp_path: Path) -> None:
    root, expectation = _fixture_artifact(tmp_path)
    stale = FullCheckpointLoadExpectation(
        **{
            **asdict(expectation),
            "packed_manifest_sha256": "0" * 64,
        }
    )
    with pytest.raises(CheckpointValidationError, match="identity mismatch"):
        verify_full_packed_checkpoint(root, stale)


def test_direct_loader_keeps_raw_final_owner_state_and_roundtrips(
    tmp_path: Path,
) -> None:
    root, _ = _fixture_artifact(tmp_path)
    completed = _forced_cpu(root, "_run_forced_cpu_loader")
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_direct_loader_deletes_and_refuses_corrupt_payload(tmp_path: Path) -> None:
    root, _ = _fixture_artifact(tmp_path)
    target = root / "base_decoder/stage_00/device_slot_00.safetensors"
    with target.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        original = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes((original[0] ^ 1,)))
    completed = _forced_cpu(root, "_run_forced_cpu_corruption_refusal")
    assert completed.returncode == 0, completed.stdout + completed.stderr
