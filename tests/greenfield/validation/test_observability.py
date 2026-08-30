from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import zipfile

import numpy as np
import pytest

from glm_tpu.greenfield.errors import BenchmarkValidationError
from glm_tpu.greenfield import observability as subject


REPO_ROOT = Path(__file__).resolve().parents[3]


def _plan(role: str) -> dict[str, object]:
    name = role.upper()
    return {
        "authority_file_sha256": None,
        "authority_json_path": [],
        "authority_kind": "canonical_name_sha256",
        "authority_path": None,
        "id": f"{role}.plan",
        "name": name,
        "plan_sha256": sha256(name.encode()).hexdigest(),
    }


def _plan_sha256(role: str) -> str:
    value = _plan(role)["plan_sha256"]
    assert isinstance(value, str)
    return value


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _artifact(
    tmp_path: Path, name: str, **arrays: np.ndarray
) -> tuple[Path, dict[str, dict[str, object]]]:
    path = tmp_path / name
    np.savez(path, **arrays)
    report = subject.inspect_npz_artifact(path, _file_sha256(path))
    return path, {item["key"]: item for item in report["members"]}


def _watchpoint(
    identifier: str,
    order: int,
    semantic_dtype: str,
    shape: list[int],
) -> dict[str, object]:
    return {
        "id": identifier,
        "layer": 1,
        "order": order,
        "position": 8155,
        "semantic_dtype": semantic_dtype,
        "shape": shape,
    }


def _binding(
    watchpoint: dict[str, object],
    key: str,
    member: dict[str, object],
    *,
    index_prefix: list[int] | None = None,
) -> dict[str, object]:
    return {
        "array_sha256": member["raw_sha256"],
        "index_prefix": [] if index_prefix is None else index_prefix,
        "key": key,
        "shape": watchpoint["shape"],
        "storage_dtype": member["storage_dtype"],
        "watchpoint_id": watchpoint["id"],
    }


def _source(
    role: str,
    path: Path,
    observations: list[dict[str, object]],
    *,
    coherence_id: str | None = None,
) -> dict[str, object]:
    properties = {
        "accepted": ("accepted_protected", "protected_oracle", "a" * 40),
        "candidate": ("candidate_protected", "device_output", "b" * 40),
    }
    trust, method, code_pin = properties[role]
    coherence = coherence_id or f"{role}.coherent"
    executable_sha256 = None if role == "accepted" else "d" * 64
    artifact_sha256 = _file_sha256(path)
    identity = {
        "artifact_sha256": artifact_sha256,
        "code_pin": code_pin,
        "coherence_id": coherence,
        "executable_identity_sha256": executable_sha256,
    }
    identity_path = path.parent / f"{coherence.replace('.', '-')}-identity.json"
    identity_path.write_text(json.dumps(identity, sort_keys=True))
    bindings = [
        {"json_path": ["artifact_sha256"], "source_field": "artifact_sha256"},
        {"json_path": ["code_pin"], "source_field": "code_pin"},
        {"json_path": ["coherence_id"], "source_field": "coherence_id"},
    ]
    if executable_sha256 is not None:
        bindings.append(
            {
                "json_path": ["executable_identity_sha256"],
                "source_field": "executable_identity_sha256",
            }
        )
    return {
        "artifact_path": str(path),
        "artifact_sha256": artifact_sha256,
        "code_pin": code_pin,
        "coherence_id": coherence,
        "executable_identity_kind": (
            "unavailable" if role == "accepted" else "optimized_hlo"
        ),
        "executable_identity_sha256": executable_sha256,
        "id": f"{role}.source",
        "identity_evidence": [
            {
                "artifact_path": str(identity_path),
                "artifact_sha256": _file_sha256(identity_path),
                "bindings": bindings,
            }
        ],
        "observation_method": method,
        "observations": observations,
        "plan_id": f"{role}.plan",
        "plan_sha256": _plan_sha256(role),
        "role": role,
        "trust": trust,
    }


def _write_contract(
    tmp_path: Path,
    watchpoints: list[dict[str, object]],
    sources: list[dict[str, object]],
    *,
    name: str = "contract.json",
) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "contract_id": "gate.d.watchpoints",
                "plans": [_plan("accepted"), _plan("candidate")],
                "schema_version": subject.OBSERVABILITY_SCHEMA_VERSION,
                "sources": sources,
                "watchpoints": watchpoints,
            },
            sort_keys=True,
        )
    )
    return path


def _audit(path: Path) -> dict[str, object]:
    return subject.audit_observability_contract(path, _file_sha256(path))


def _paired_contract(
    tmp_path: Path,
    accepted_values: np.ndarray,
    candidate_values: np.ndarray,
    *,
    semantic_dtype: str,
) -> Path:
    watchpoint = _watchpoint(
        "layer1.value", 0, semantic_dtype, list(accepted_values.shape)
    )
    accepted_path, accepted_members = _artifact(
        tmp_path, "accepted.npz", value=accepted_values
    )
    candidate_path, candidate_members = _artifact(
        tmp_path, "candidate.npz", value=candidate_values
    )
    return _write_contract(
        tmp_path,
        [watchpoint],
        [
            _source(
                "accepted",
                accepted_path,
                [_binding(watchpoint, "value", accepted_members["value"])],
            ),
            _source(
                "candidate",
                candidate_path,
                [_binding(watchpoint, "value", candidate_members["value"])],
            ),
        ],
    )


def test_localizes_first_bf16_bit_mismatch(tmp_path: Path) -> None:
    contract = _paired_contract(
        tmp_path,
        np.asarray([0x3F80, 0x4000], dtype=np.uint16),
        np.asarray([0x3F80, 0x4040], dtype=np.uint16),
        semantic_dtype="bf16",
    )
    result = _audit(contract)
    assert result["classification"] == "DIVERGENCE_LOCALIZED"
    assert result["causal_frontier"] == {
        "id": "layer1.value",
        "kind": "divergence",
        "order": 0,
    }
    divergence = result["first_divergence"]
    assert divergence["mismatch_count"] == 1
    assert divergence["first_mismatch"] == {
        "accepted": {"raw_hex": "0040", "value": 2.0},
        "candidate": {"raw_hex": "4040", "value": 3.0},
        "index": [1],
    }
    assert divergence["max_abs_error"] == 1.0
    assert result["contract_sha256"] == _file_sha256(contract)


def test_upstream_gap_precedes_downstream_divergence(tmp_path: Path) -> None:
    rms_input = _watchpoint("layer1.rms.input", 0, "float32", [2])
    normalized = _watchpoint("layer1.normalized", 1, "bf16", [2])
    accepted_path, accepted_members = _artifact(
        tmp_path,
        "accepted.npz",
        normalized=np.asarray([0x3F80, 0x4000], dtype=np.uint16),
    )
    candidate_path, candidate_members = _artifact(
        tmp_path,
        "candidate.npz",
        normalized=np.asarray([0x3F80, 0x4040], dtype=np.uint16),
    )
    contract = _write_contract(
        tmp_path,
        [rms_input, normalized],
        [
            _source(
                "accepted",
                accepted_path,
                [_binding(normalized, "normalized", accepted_members["normalized"])],
            ),
            _source(
                "candidate",
                candidate_path,
                [_binding(normalized, "normalized", candidate_members["normalized"])],
            ),
        ],
    )
    result = _audit(contract)
    assert result["classification"] == "OBSERVABILITY_GAP"
    assert result["causal_frontier"] == {
        "id": "layer1.rms.input",
        "kind": "unobservable",
        "missing_accepted": True,
        "missing_candidate": True,
        "order": 0,
    }
    assert result["first_divergence"]["watchpoint_id"] == "layer1.normalized"


def test_divergence_precedes_later_gap(tmp_path: Path) -> None:
    normalized = _watchpoint("layer1.normalized", 0, "bf16", [2])
    query = _watchpoint("layer1.query", 1, "float32", [2])
    accepted_path, accepted_members = _artifact(
        tmp_path,
        "accepted.npz",
        normalized=np.asarray([0x3F80, 0x4000], dtype=np.uint16),
    )
    candidate_path, candidate_members = _artifact(
        tmp_path,
        "candidate.npz",
        normalized=np.asarray([0x3F80, 0x4040], dtype=np.uint16),
    )
    contract = _write_contract(
        tmp_path,
        [normalized, query],
        [
            _source(
                "accepted",
                accepted_path,
                [_binding(normalized, "normalized", accepted_members["normalized"])],
            ),
            _source(
                "candidate",
                candidate_path,
                [_binding(normalized, "normalized", candidate_members["normalized"])],
            ),
        ],
    )
    result = _audit(contract)
    assert result["classification"] == "DIVERGENCE_LOCALIZED"
    assert result["first_unobservable"]["id"] == "layer1.query"


def test_rejects_mixed_candidate_coherence(tmp_path: Path) -> None:
    first = _watchpoint("layer1.first", 0, "float32", [1])
    second = _watchpoint("layer1.second", 1, "float32", [1])
    path, members = _artifact(
        tmp_path,
        "values.npz",
        first=np.asarray([1], dtype=np.float32),
        second=np.asarray([2], dtype=np.float32),
    )
    accepted = _source(
        "accepted",
        path,
        [
            _binding(first, "first", members["first"]),
            _binding(second, "second", members["second"]),
        ],
    )
    candidate_a = _source(
        "candidate",
        path,
        [_binding(first, "first", members["first"])],
        coherence_id="candidate.a",
    )
    candidate_b = _source(
        "candidate",
        path,
        [_binding(second, "second", members["second"])],
        coherence_id="candidate.b",
    )
    candidate_a["id"] = "candidate.a.source"
    candidate_b["id"] = "candidate.b.source"
    contract = _write_contract(
        tmp_path, [first, second], [accepted, candidate_a, candidate_b]
    )
    with pytest.raises(BenchmarkValidationError, match="coherent authority"):
        _audit(contract)


def test_rejects_role_method_and_hash_drift(tmp_path: Path) -> None:
    contract = _paired_contract(
        tmp_path,
        np.asarray([1], dtype=np.float32),
        np.asarray([1], dtype=np.float32),
        semantic_dtype="float32",
    )
    value = json.loads(contract.read_text())
    value["sources"][1]["observation_method"] = "rejected_callback"
    contract.write_text(json.dumps(value))
    with pytest.raises(BenchmarkValidationError, match="observation method"):
        _audit(contract)

    value["sources"][1]["observation_method"] = "device_output"
    value["sources"][1]["artifact_sha256"] = "0" * 64
    contract.write_text(json.dumps(value))
    with pytest.raises(BenchmarkValidationError, match="disagrees with source"):
        _audit(contract)

    value["sources"][1]["artifact_sha256"] = _file_sha256(
        Path(value["sources"][1]["artifact_path"])
    )
    value["sources"][1]["observations"][0]["array_sha256"] = "0" * 64
    contract.write_text(json.dumps(value))
    with pytest.raises(BenchmarkValidationError, match="array SHA-256 drifted"):
        _audit(contract)


def test_rejects_semantic_storage_and_vacuous_shape(tmp_path: Path) -> None:
    contract = _paired_contract(
        tmp_path,
        np.asarray([1], dtype=np.float32),
        np.asarray([1], dtype=np.float32),
        semantic_dtype="float32",
    )
    value = json.loads(contract.read_text())
    value["sources"][1]["observations"][0]["storage_dtype"] = "<u2"
    contract.write_text(json.dumps(value))
    with pytest.raises(BenchmarkValidationError, match="semantic/storage"):
        _audit(contract)

    value["watchpoints"][0]["shape"] = [0]
    contract.write_text(json.dumps(value))
    with pytest.raises(BenchmarkValidationError, match="watchpoint shape"):
        _audit(contract)


def test_rejects_duplicate_json_keys_and_symlink_inputs(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"contract_id":"first","contract_id":"second"}')
    with pytest.raises(BenchmarkValidationError, match="duplicate JSON key"):
        _audit(duplicate)

    contract = _paired_contract(
        tmp_path,
        np.asarray([1], dtype=np.float32),
        np.asarray([1], dtype=np.float32),
        semantic_dtype="float32",
    )
    contract_link = tmp_path / "contract-link.json"
    contract_link.symlink_to(contract)
    with pytest.raises(BenchmarkValidationError, match="cannot open"):
        _audit(contract_link)

    value = json.loads(contract.read_text())
    artifact = Path(value["sources"][1]["artifact_path"])
    artifact_link = tmp_path / "artifact-link.npz"
    artifact_link.symlink_to(artifact)
    value["sources"][1]["artifact_path"] = str(artifact_link)
    contract.write_text(json.dumps(value))
    with pytest.raises(BenchmarkValidationError, match="cannot open"):
        _audit(contract)


def test_rejects_nonfinite_json_and_plan_authority_drift(tmp_path: Path) -> None:
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}')
    with pytest.raises(BenchmarkValidationError, match="non-finite JSON"):
        _audit(nonfinite)

    contract = _paired_contract(
        tmp_path,
        np.asarray([1], dtype=np.float32),
        np.asarray([1], dtype=np.float32),
        semantic_dtype="float32",
    )
    value = json.loads(contract.read_text())
    value["plans"][1]["plan_sha256"] = "5" * 64
    value["sources"][1]["plan_sha256"] = "5" * 64
    contract.write_text(json.dumps(value))
    with pytest.raises(BenchmarkValidationError, match="plan authority"):
        _audit(contract)


def test_external_contract_sha_rejects_coordinated_contract_edit(
    tmp_path: Path,
) -> None:
    contract = _paired_contract(
        tmp_path,
        np.asarray([1], dtype=np.float32),
        np.asarray([1], dtype=np.float32),
        semantic_dtype="float32",
    )
    expected_sha256 = _file_sha256(contract)
    value = json.loads(contract.read_text())
    value["contract_id"] = "coordinated.edit"
    contract.write_text(json.dumps(value))
    with pytest.raises(BenchmarkValidationError, match="contract SHA-256 drifted"):
        subject.audit_observability_contract(contract, expected_sha256)


def test_inspection_rejects_duplicate_npz_members(tmp_path: Path) -> None:
    payload = io.BytesIO()
    np.save(payload, np.asarray([1], dtype=np.float32), allow_pickle=False)
    path = tmp_path / "duplicate.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("value.npy", payload.getvalue())
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("value.npy", payload.getvalue())
    with pytest.raises(BenchmarkValidationError, match="duplicate members"):
        subject.inspect_npz_artifact(path, _file_sha256(path))


def test_report_write_is_canonical_append_only_and_symlink_safe(
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    subject.write_observability_report(output, {"z": 1, "a": [2]})
    assert output.read_bytes() == b'{"a":[2],"z":1}\n'
    with pytest.raises(BenchmarkValidationError, match="occupied"):
        subject.write_observability_report(output, {"new": True})

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing.json")
    with pytest.raises(BenchmarkValidationError, match="occupied"):
        subject.write_observability_report(dangling, {"new": True})

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(BenchmarkValidationError, match="safely open output parent"):
        subject.write_observability_report(linked_parent / "report.json", {})
    assert not (real_parent / "report.json").exists()


def test_cli_inspects_npz_without_jax(tmp_path: Path) -> None:
    path, members = _artifact(
        tmp_path, "values.npz", value=np.asarray([1, 2], dtype=np.int32)
    )
    output = tmp_path / "inventory.json"
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            str(REPO_ROOT / "scripts/greenfield/audit_observability.py"),
            "--inspect-npz",
            str(path),
            "--artifact-sha256",
            _file_sha256(path),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["mode"] == "inspect_npz"
    report = json.loads(output.read_text())
    assert report["member_count"] == 1
    assert report["members"][0] == members["value"]

    contract = _paired_contract(
        tmp_path,
        np.asarray([1], dtype=np.float32),
        np.asarray([2], dtype=np.float32),
        semantic_dtype="float32",
    )
    audit_output = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(REPO_ROOT / "scripts/greenfield/audit_observability.py"),
            "--contract",
            str(contract),
            "--contract-sha256",
            _file_sha256(contract),
            "--output",
            str(audit_output),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(audit_output.read_text())["classification"] == (
        "DIVERGENCE_LOCALIZED"
    )


def test_public_compare_refuses_malformed_buffers_and_hashes() -> None:
    raw = np.asarray([1, 2], dtype=np.float32).tobytes()
    valid = subject.ArrayObservation(
        source_id="accepted",
        watchpoint_id="value",
        semantic_dtype="float32",
        dtype="<f4",
        shape=(2,),
        raw=raw,
        raw_sha256=sha256(raw).hexdigest(),
    )
    empty = subject.ArrayObservation(
        source_id="candidate",
        watchpoint_id="value",
        semantic_dtype="float32",
        dtype="<f4",
        shape=(2,),
        raw=b"",
        raw_sha256=sha256(b"").hexdigest(),
    )
    with pytest.raises(BenchmarkValidationError, match="raw byte count"):
        subject.compare_array_observations(valid, empty)
    trailing = subject.ArrayObservation(
        source_id="candidate",
        watchpoint_id="value",
        semantic_dtype="float32",
        dtype="<f4",
        shape=(2,),
        raw=raw + b"extra",
        raw_sha256=sha256(raw + b"extra").hexdigest(),
    )
    with pytest.raises(BenchmarkValidationError, match="raw byte count"):
        subject.compare_array_observations(valid, trailing)
    bad_hash = subject.ArrayObservation(
        source_id="candidate",
        watchpoint_id="value",
        semantic_dtype="float32",
        dtype="<f4",
        shape=(2,),
        raw=raw,
        raw_sha256="0" * 64,
    )
    with pytest.raises(BenchmarkValidationError, match="raw SHA-256"):
        subject.compare_array_observations(valid, bad_hash)

    malformed_values = (
        ("dtype", ["<f4"], "dtype is invalid"),
        ("semantic_dtype", ["float32"], "semantic/storage dtype"),
        ("raw", bytearray(raw), "raw value must be bytes"),
        ("raw_sha256", 7, "raw SHA-256"),
    )
    for field, malformed, message in malformed_values:
        values = {
            "source_id": "candidate",
            "watchpoint_id": "value",
            "semantic_dtype": "float32",
            "dtype": "<f4",
            "shape": (2,),
            "raw": raw,
            "raw_sha256": sha256(raw).hexdigest(),
        }
        values[field] = malformed
        observation = subject.ArrayObservation(**values)
        with pytest.raises(BenchmarkValidationError, match=message):
            subject.compare_array_observations(valid, observation)


def test_nonfinite_mismatch_has_no_finite_max_error() -> None:
    accepted_raw = struct.pack("<f", float("nan"))
    candidate_raw = struct.pack("<f", float("inf"))
    observations = []
    for source_id, raw in (
        ("accepted", accepted_raw),
        ("candidate", candidate_raw),
    ):
        observations.append(
            subject.ArrayObservation(
                source_id=source_id,
                watchpoint_id="nonfinite",
                semantic_dtype="float32",
                dtype="<f4",
                shape=(1,),
                raw=raw,
                raw_sha256=sha256(raw).hexdigest(),
            )
        )
    result = subject.compare_array_observations(*observations)
    assert result["exact"] is False
    assert result["max_abs_error"] is None
    assert result["nonfinite_mismatch_count"] == 1


def test_mixed_finite_and_nonfinite_mismatches_have_no_global_max() -> None:
    accepted_raw = struct.pack("<ff", 0.0, float("nan"))
    candidate_raw = struct.pack("<ff", 1.0, float("inf"))
    observations = []
    for source_id, raw in (
        ("accepted", accepted_raw),
        ("candidate", candidate_raw),
    ):
        observations.append(
            subject.ArrayObservation(
                source_id=source_id,
                watchpoint_id="mixed-nonfinite",
                semantic_dtype="float32",
                dtype="<f4",
                shape=(2,),
                raw=raw,
                raw_sha256=sha256(raw).hexdigest(),
            )
        )
    result = subject.compare_array_observations(*observations)
    assert result["mismatch_count"] == 2
    assert result["max_abs_error"] is None
    assert result["nonfinite_mismatch_count"] == 1


def test_prefix_selection_is_exact_and_out_of_bounds_refuses(
    tmp_path: Path,
) -> None:
    watchpoint = _watchpoint("row", 0, "int32", [2])
    values = np.asarray([[1, 2], [3, 4]], dtype=np.int32)
    accepted_path, accepted_members = _artifact(
        tmp_path, "accepted.npz", values=values
    )
    candidate_path, candidate_members = _artifact(
        tmp_path, "candidate.npz", values=values.copy()
    )
    selected_sha = sha256(np.ascontiguousarray(values[1]).tobytes()).hexdigest()
    accepted_member = dict(accepted_members["values"])
    candidate_member = dict(candidate_members["values"])
    accepted_member["raw_sha256"] = selected_sha
    candidate_member["raw_sha256"] = selected_sha
    contract = _write_contract(
        tmp_path,
        [watchpoint],
        [
            _source(
                "accepted",
                accepted_path,
                [
                    _binding(
                        watchpoint,
                        "values",
                        accepted_member,
                        index_prefix=[1],
                    )
                ],
            ),
            _source(
                "candidate",
                candidate_path,
                [
                    _binding(
                        watchpoint,
                        "values",
                        candidate_member,
                        index_prefix=[1],
                    )
                ],
            ),
        ],
    )
    assert _audit(contract)["classification"] == "EXACT"
    value = json.loads(contract.read_text())
    value["sources"][1]["observations"][0]["index_prefix"] = [2]
    contract.write_text(json.dumps(value))
    with pytest.raises(BenchmarkValidationError, match="out of bounds"):
        _audit(contract)


def test_hostile_npy_headers_and_trailing_bytes_refuse(tmp_path: Path) -> None:
    fortran_path = tmp_path / "fortran.npz"
    np.savez(fortran_path, value=np.asfortranarray(np.ones((2, 2))))
    with pytest.raises(BenchmarkValidationError, match="C-contiguous"):
        subject.inspect_npz_artifact(fortran_path, _file_sha256(fortran_path))

    payload = io.BytesIO()
    np.save(payload, np.asarray([1], dtype=np.float32), allow_pickle=False)
    unsupported = bytearray(payload.getvalue())
    unsupported[7] = 1
    unsupported_path = tmp_path / "unsupported-version.npz"
    with zipfile.ZipFile(unsupported_path, "w") as archive:
        archive.writestr("value.npy", unsupported)
    with pytest.raises(BenchmarkValidationError, match="unsupported NPY version"):
        subject.inspect_npz_artifact(
            unsupported_path, _file_sha256(unsupported_path)
        )

    unhashable_header = (
        b"{'descr': [], 'fortran_order': False, 'shape': (1,), }"
    )
    unhashable_payload = (
        b"\x93NUMPY"
        + b"\x01\x00"
        + struct.pack("<H", len(unhashable_header))
        + unhashable_header
    )
    unhashable_path = tmp_path / "unhashable-descr.npz"
    with zipfile.ZipFile(unhashable_path, "w") as archive:
        archive.writestr("value.npy", unhashable_payload)
    with pytest.raises(BenchmarkValidationError, match="dtype is unsupported"):
        subject.inspect_npz_artifact(
            unhashable_path, _file_sha256(unhashable_path)
        )

    trailing_path = tmp_path / "trailing.npz"
    with zipfile.ZipFile(trailing_path, "w") as archive:
        archive.writestr("value.npy", payload.getvalue() + b"x")
    with pytest.raises(BenchmarkValidationError, match="byte count drifted"):
        subject.inspect_npz_artifact(trailing_path, _file_sha256(trailing_path))

    corrupt_path = tmp_path / "corrupt.npz"
    corrupt_path.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises(BenchmarkValidationError, match="cannot open NPZ"):
        subject.inspect_npz_artifact(corrupt_path, _file_sha256(corrupt_path))


def test_artifact_is_parsed_from_the_hashed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _paired_contract(
        tmp_path,
        np.asarray([1], dtype=np.float32),
        np.asarray([2], dtype=np.float32),
        semantic_dtype="float32",
    )
    value = json.loads(contract.read_text())
    candidate_path = Path(value["sources"][1]["artifact_path"])
    original_snapshot = subject._snapshot_regular_file

    @contextmanager
    def mutate_after_snapshot(
        path: Path, label: str, expected_sha256: str
    ):
        with original_snapshot(path, label, expected_sha256) as snapshot:
            if path == candidate_path:
                np.savez(candidate_path, value=np.asarray([99], dtype=np.float32))
            yield snapshot

    monkeypatch.setattr(subject, "_snapshot_regular_file", mutate_after_snapshot)
    report = _audit(contract)
    assert report["classification"] == "DIVERGENCE_LOCALIZED"
    assert report["first_divergence"]["first_mismatch"]["candidate"]["value"] == 2.0


def test_sealed_db518_contract_reproduces_the_known_causal_frontier() -> None:
    contract = REPO_ROOT / "configs/greenfield-gate-d-observability.json"
    value = json.loads(contract.read_text())
    source_paths = [Path(source["artifact_path"]) for source in value["sources"]]
    if not all(path.is_file() for path in source_paths):
        pytest.skip("sealed accepted/DB518 artifacts are not locally mounted")
    assert _file_sha256(contract) == (
        "6fcf46066e7e500202baeaffab5a731ff17a40b67f49db711f0ebc51e37914fc"
    )
    report = _audit(contract)
    assert report["classification"] == "OBSERVABILITY_GAP"
    assert report["causal_frontier"] == {
        "id": "layer1.rms_input_fp32",
        "kind": "unobservable",
        "missing_accepted": True,
        "missing_candidate": True,
        "order": 0,
    }
    divergence = report["first_divergence"]
    assert divergence["watchpoint_id"] == "layer1.normalized.owner0"
    assert divergence["mismatch_count"] == 1
    assert divergence["first_mismatch"] == {
        "accepted": {"raw_hex": "27bd", "value": -0.040771484375},
        "candidate": {"raw_hex": "26bd", "value": -0.04052734375},
        "index": [2795],
    }
    mismatch_counts = {
        item["id"]: (
            None
            if item["comparison"] is None
            else item["comparison"]["mismatch_count"]
        )
        for item in report["coverage"]
    }
    assert mismatch_counts["layer1.normalized.owner1"] == 1
    assert mismatch_counts["layer1.q_a.owner0"] == 47
    assert mismatch_counts["layer1.q_a.owner1"] == 47
    assert mismatch_counts["layer1.query.owner0"] == 4096
    assert mismatch_counts["layer1.query.owner1"] == 4096
    assert mismatch_counts["layer1.head.owner0"] == 32
    assert mismatch_counts["layer1.head.owner1"] == 32
    canonical_report = subject._canonical_json(report).encode() + b"\n"
    assert sha256(canonical_report).hexdigest() == (
        "366898b88ffdd2205ecf94838e5af3d0042c4ce3f735cec7f8ee86bd4487150b"
    )
