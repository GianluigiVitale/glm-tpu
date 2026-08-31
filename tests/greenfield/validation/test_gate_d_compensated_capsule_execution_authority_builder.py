from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.errors import BenchmarkValidationError


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO_ROOT
    / "scripts/greenfield/build_gate_d_compensated_capsule_execution_authority.py"
)
SPEC = importlib.util.spec_from_file_location("gate_d_compensated_authority_builder", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def _canonical(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def _write(path: Path, raw: bytes) -> dict[str, str]:
    path.write_bytes(raw)
    return {"path": path.name, "sha256": sha256(raw).hexdigest()}


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    state = tmp_path / "candidate-state.npz"
    np.savez(
        state,
        event1_positions=np.arange(2048, dtype=np.int32)[None, :],
        event1_scores=np.arange(2048, 0, -1, dtype=np.float32)[None, :],
        event1_valid_count=np.array([2048], dtype=np.int32),
        rms_hidden_update=np.zeros((6144,), dtype=np.uint16),
        rms_residual=np.ones((6144,), dtype=np.uint16),
    )
    inputs = tmp_path / "candidate-inputs.npz"
    device = tmp_path / "candidate-device-evidence.npz"
    np.savez(inputs, value=np.zeros((1,), dtype=np.uint8))
    np.savez(device, value=np.ones((1,), dtype=np.uint8))
    state_raw = state.read_bytes()
    input_raw = inputs.read_bytes()
    device_raw = device.read_bytes()
    receipt = {
        "artifact": {"bytes": len(state_raw), "sha256": sha256(state_raw).hexdigest()},
        "backend": {"device_count": 2, "device_ids": [0, 1], "platform": "cpu"},
        "candidate": {"id": "compensated_auxiliary_dependency"},
        "claim_scope": "fixture",
        "coherence_id": "fixture.gate.d.capsule",
        "device_evidence": {
            "bytes": len(device_raw),
            "sha256": sha256(device_raw).hexdigest(),
        },
        "environment": {"fixture": True},
        "execution": {"tpus_used": 0},
        "input_arrays": {"fixture": {}},
        "installed_producer": {"fixture": True},
        "loaded_dependencies": {"native_mappings": [], "python_modules": []},
        "producer": {"fixture": True},
        "schema_version": 1,
        "source_snapshot": {"fixture": True},
        "tensor_receipts": [],
        "upstream_inputs": {"fixture": {}},
        "watchpoint_manifest_sha256": "0" * 64,
    }
    receipt_binding = _write(tmp_path / "producer_receipt.json", _canonical(receipt))
    success = {
        "artifact_sha256": sha256(state_raw).hexdigest(),
        "device_evidence_sha256": sha256(device_raw).hexdigest(),
        "input_artifact_sha256": sha256(input_raw).hexdigest(),
        "producer_receipt_sha256": receipt_binding["sha256"],
        "schema_version": 1,
    }
    success_binding = _write(tmp_path / "SUCCESS", _canonical(success))
    capsule = {
        "artifact": {"path": state.name, "sha256": success["artifact_sha256"]},
        "candidate_id": "compensated_auxiliary_dependency",
        "claim_scope": "fixture",
        "code_pin": "0" * 40,
        "coherence_id": "fixture.gate.d.capsule",
        "plan_sha256": "0" * 64,
        "producer_device_evidence": {
            "path": device.name,
            "sha256": success["device_evidence_sha256"],
        },
        "producer_input_artifact": {
            "path": inputs.name,
            "sha256": success["input_artifact_sha256"],
        },
        "producer_receipt": receipt_binding,
        "producer_success": success_binding,
        "schema_version": 2,
        "source_authority_sha256": "0" * 64,
        "stablehlo_authority_sha256": "0" * 64,
        "watchpoints": [{"fixture": True}],
    }
    capsule_path = tmp_path / "capsule.json"
    capsule_raw = _canonical(capsule)
    capsule_path.write_bytes(capsule_raw)
    return capsule_path, sha256(capsule_raw).hexdigest()


def _rebind_state(capsule_path: Path) -> str:
    root = capsule_path.parent
    state = root / "candidate-state.npz"
    state_raw = state.read_bytes()
    receipt_path = root / "producer_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact"] = {
        "bytes": len(state_raw),
        "sha256": sha256(state_raw).hexdigest(),
    }
    receipt_raw = _canonical(receipt)
    receipt_path.write_bytes(receipt_raw)
    success_path = root / "SUCCESS"
    success = json.loads(success_path.read_text())
    success["artifact_sha256"] = receipt["artifact"]["sha256"]
    success["producer_receipt_sha256"] = sha256(receipt_raw).hexdigest()
    success_raw = _canonical(success)
    success_path.write_bytes(success_raw)
    capsule = json.loads(capsule_path.read_text())
    capsule["artifact"]["sha256"] = receipt["artifact"]["sha256"]
    capsule["producer_receipt"]["sha256"] = sha256(receipt_raw).hexdigest()
    capsule["producer_success"]["sha256"] = sha256(success_raw).hexdigest()
    capsule_raw = _canonical(capsule)
    capsule_path.write_bytes(capsule_raw)
    return sha256(capsule_raw).hexdigest()


def _rebind_metadata(capsule_path: Path) -> str:
    root = capsule_path.parent
    receipt_path = root / "producer_receipt.json"
    receipt_raw = _canonical(json.loads(receipt_path.read_text()))
    receipt_path.write_bytes(receipt_raw)
    success_path = root / "SUCCESS"
    success = json.loads(success_path.read_text())
    success["producer_receipt_sha256"] = sha256(receipt_raw).hexdigest()
    success_raw = _canonical(success)
    success_path.write_bytes(success_raw)
    capsule = json.loads(capsule_path.read_text())
    capsule["producer_receipt"]["sha256"] = sha256(receipt_raw).hexdigest()
    capsule["producer_success"]["sha256"] = sha256(success_raw).hexdigest()
    capsule_raw = _canonical(capsule)
    capsule_path.write_bytes(capsule_raw)
    return sha256(capsule_raw).hexdigest()


def test_builder_derives_only_expected_output_bits(tmp_path: Path) -> None:
    capsule, capsule_sha = _fixture(tmp_path)
    authority = BUILDER.build_execution_authority(capsule, capsule_sha)
    assert authority["authority_kind"] == "gate.d.capsule.execution.v1"
    assert authority["candidate_id"] == "compensated_auxiliary_dependency"
    assert authority["expected_outputs"]["event1_valid_count"] == 2048
    assert "execution" not in authority
    assert "jax" not in BUILDER.sys.modules


def test_builder_rejects_receipt_drift(tmp_path: Path) -> None:
    capsule, capsule_sha = _fixture(tmp_path)
    (tmp_path / "producer_receipt.json").write_bytes(b"{}\n")
    with pytest.raises(BenchmarkValidationError, match="SHA-256 drifted"):
        BUILDER.build_execution_authority(capsule, capsule_sha)


def test_builder_rejects_authority_array_shape(tmp_path: Path) -> None:
    capsule, _ = _fixture(tmp_path)
    state = tmp_path / "candidate-state.npz"
    np.savez(
        state,
        event1_positions=np.arange(2048, dtype=np.int32),
        event1_scores=np.arange(2048, 0, -1, dtype=np.float32)[None, :],
        event1_valid_count=np.array([2048], dtype=np.int32),
        rms_hidden_update=np.zeros((6144,), dtype=np.uint16),
        rms_residual=np.ones((6144,), dtype=np.uint16),
    )
    capsule_sha = _rebind_state(capsule)
    with pytest.raises(BenchmarkValidationError, match="authority array drifted"):
        BUILDER.build_execution_authority(capsule, capsule_sha)


def test_builder_rejects_noncanonical_capsule(tmp_path: Path) -> None:
    capsule, capsule_sha = _fixture(tmp_path)
    document = json.loads(capsule.read_text())
    capsule.write_text(json.dumps(document, indent=2) + "\n")
    reformatted_sha = sha256(capsule.read_bytes()).hexdigest()
    with pytest.raises(BenchmarkValidationError, match="not canonical JSON"):
        BUILDER.build_execution_authority(capsule, reformatted_sha)


@pytest.mark.parametrize(
    ("document", "schema_version"),
    [
        ("capsule", 2.0),
        ("capsule", True),
        ("receipt", 1.0),
        ("receipt", True),
        ("success", 1.0),
        ("success", True),
    ],
)
def test_builder_rejects_non_integer_schema_versions(
    tmp_path: Path, document: str, schema_version: object
) -> None:
    capsule_path, _ = _fixture(tmp_path)
    target = {
        "capsule": capsule_path,
        "receipt": tmp_path / "producer_receipt.json",
        "success": tmp_path / "SUCCESS",
    }[document]
    payload = json.loads(target.read_text())
    payload["schema_version"] = schema_version
    target.write_bytes(_canonical(payload))
    capsule_sha = (
        sha256(capsule_path.read_bytes()).hexdigest()
        if document == "capsule"
        else _rebind_metadata(capsule_path)
    )
    with pytest.raises(BenchmarkValidationError, match="positive integer"):
        BUILDER.build_execution_authority(capsule_path, capsule_sha)
