from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

import pytest

from glm_tpu.greenfield import capsule_constructability as subject
from glm_tpu.greenfield.errors import BenchmarkValidationError


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_CONTRACT = REPO_ROOT / "configs/greenfield-gate-d-capsule-constructability.json"
CLI = REPO_ROOT / "scripts/greenfield/audit_capsule_constructability.py"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True))


def _contract(tmp_path: Path) -> Path:
    value = json.loads(REAL_CONTRACT.read_text())
    for item in value["evidence"]:
        item["path"] = str((REAL_CONTRACT.parent / item["path"]).resolve())
    value["candidate_artifact"]["path"] = str(
        Path(value["candidate_artifact"]["path"]).resolve()
    )
    path = tmp_path / "contract.json"
    _write(path, value)
    return path


def _audit(path: Path) -> dict[str, object]:
    return subject.audit_capsule_constructability(path, _sha(path))


def _rewrite_npz(source: Path, destination: Path, *, omit: set[str]) -> None:
    with zipfile.ZipFile(source) as input_archive, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_STORED
    ) as output_archive:
        for name in input_archive.namelist():
            if name not in omit:
                output_archive.writestr(name, input_archive.read(name))


def _mutate_npz_member(source: Path, destination: Path, member: str) -> None:
    with zipfile.ZipFile(source) as input_archive, zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_STORED
    ) as output_archive:
        for name in input_archive.namelist():
            payload = bytearray(input_archive.read(name))
            if name == member:
                payload[-1] ^= 1
            output_archive.writestr(name, payload)


def _bool_npy_header(shape: tuple[int, ...]) -> bytes:
    header = repr(
        {"descr": "|b1", "fortran_order": False, "shape": shape}
    ).encode("latin1")
    padding = (-((6 + 2 + 2) + len(header) + 1)) % 16
    header += b" " * padding + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header


def test_real_contract_fails_closed_without_jax() -> None:
    report = _audit(REAL_CONTRACT)
    assert report["classification"] == (
        "SEALED_STATE_INCOMPLETE;PRECOMPILE_SOURCE_STABLEHLO_AUTHORITY_UNREPRESENTED;"
        "GATE_D_OPEN;NO_JAX_OR_TPU_SUCCESSOR"
    )
    assert report["missing_old_artifact_watchpoints"] == [
        "layer1.rms_input_fp32",
        "layer1.current_key",
    ]
    assert report["new_variant_capsule_constructable"] is False
    assert report["variant_source_bound"] is False
    assert report["gate_d_closed"] is False
    assert report["tpu_successor_authorized"] is False
    authority = report["precompile_authority_audit"]
    assert authority["capsule_executable_sha256_required"] is True
    schema = authority["observability_schema_findings"]
    assert schema["stablehlo_identity_kind_supported"] is False
    assert schema["candidate_requires_nonnull_executable_identity"] is True
    assert schema["explicit_source_ast_authority"] is False
    assert report["candidate_authority"]["coherence_id"].startswith(
        "greenfield_pp16_feature2_layer0_db518"
    )
    assert "jax" not in sys.modules


def test_cli_is_offline_and_canonical(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command = [
        sys.executable,
        "-S",
        str(CLI),
        "--contract",
        str(REAL_CONTRACT),
        "--contract-sha256",
        _sha(REAL_CONTRACT),
        "--output",
    ]
    for output in (first, second):
        result = subprocess.run(
            [*command, str(output)],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("schema_version", True), "schema drifted"),
        (
            lambda value: value["required_watchpoints"].pop(),
            "seven-watchpoint contract drifted",
        ),
        (
            lambda value: value["variant_ids"].reverse(),
            "surviving variant set drifted",
        ),
        (lambda value: value.__setitem__("claim_scope", ""), "claim scope"),
    ],
)
def test_contract_cannot_be_weakened(
    tmp_path: Path, mutation: object, message: str
) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    mutation(value)
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match=message):
        _audit(contract)


def test_evidence_hash_and_catalogue_are_closed(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["evidence"][0]["sha256"] = "0" * 64
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="SHA-256 drifted"):
        _audit(contract)

    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["evidence"].pop()
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="evidence catalogue drifted"):
        _audit(contract)


def test_present_and_absent_candidate_keys_are_authenticated(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    source = Path(value["candidate_artifact"]["path"])
    missing = tmp_path / "missing.npz"
    _rewrite_npz(source, missing, omit={"current_dsa_query_owners.npy"})
    value["candidate_artifact"] = {"path": str(missing), "sha256": _sha(missing)}
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="authority disagree"):
        _audit(contract)

    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    source = Path(value["candidate_artifact"]["path"])
    added = tmp_path / "added.npz"
    _rewrite_npz(source, added, omit=set())
    with zipfile.ZipFile(added, "a", compression=zipfile.ZIP_STORED) as archive:
        with zipfile.ZipFile(source) as original:
            archive.writestr(
                "current_key.npy", original.read("position113_current_key_owners.npy")
            )
    value["candidate_artifact"] = {"path": str(added), "sha256": _sha(added)}
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="authority disagree"):
        _audit(contract)


def test_candidate_bytes_cannot_diverge_from_observability_authority(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    source = Path(value["candidate_artifact"]["path"])
    mutated = tmp_path / "mutated.npz"
    _mutate_npz_member(source, mutated, "current_dsa_query_owners.npy")
    value["candidate_artifact"] = {
        "path": str(mutated),
        "sha256": _sha(mutated),
    }
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="authority disagree"):
        _audit(contract)


def test_npz_aggregate_budget_and_compression_fail_before_array_reads(
    tmp_path: Path,
) -> None:
    bomb = tmp_path / "bomb.npz"
    per_member = 33 * 1024 * 1024
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(2):
            with archive.open(f"bomb{index}.npy", "w") as stream:
                stream.write(_bool_npy_header((per_member,)))
                block = b"\x00" * (1024 * 1024)
                for _ in range(33):
                    stream.write(block)
    raw = bomb.read_bytes()
    assert len(raw) < subject._MAX_CANDIDATE_BYTES
    with pytest.raises(BenchmarkValidationError, match="aggregate uncompressed"):
        subject._inspect_npz_bytes(raw, sha256(raw).hexdigest())

    unsupported = tmp_path / "unsupported.npz"
    with zipfile.ZipFile(unsupported, "w", compression=zipfile.ZIP_BZIP2) as archive:
        archive.writestr("value.npy", _bool_npy_header((1,)) + b"\x00")
    unsupported_raw = unsupported.read_bytes()
    with pytest.raises(BenchmarkValidationError, match="unsupported compression"):
        subject._inspect_npz_bytes(
            unsupported_raw, sha256(unsupported_raw).hexdigest()
        )


def test_inventory_labels_cannot_overstate_availability(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["watchpoint_inventory"][0]["status"] = "present_direct_old_authority"
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="inventory drifted"):
        _audit(contract)


def test_duplicate_nonfinite_and_symlinked_contracts_are_refused(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(BenchmarkValidationError, match="duplicate JSON key"):
        subject.audit_capsule_constructability(duplicate, _sha(duplicate))

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}')
    with pytest.raises(BenchmarkValidationError, match="non-finite JSON value"):
        subject.audit_capsule_constructability(nonfinite, _sha(nonfinite))

    contract = _contract(tmp_path)
    alias = tmp_path / "contract-link.json"
    alias.symlink_to(contract)
    with pytest.raises(BenchmarkValidationError, match="cannot open"):
        subject.audit_capsule_constructability(alias, _sha(contract))


def test_intermediate_artifact_symlink_and_occupied_output_are_refused(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    contract = _contract(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(BenchmarkValidationError, match="cannot safely open"):
        subject.audit_capsule_constructability(alias / contract.name, _sha(contract))

    report = _audit(contract)
    occupied = tmp_path / "occupied.json"
    occupied.write_text("occupied")
    with pytest.raises(BenchmarkValidationError, match="occupied"):
        subject.write_capsule_constructability_report(occupied, report)


def test_writer_failure_never_unlinks_a_concurrent_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _audit(_contract(tmp_path))
    output = tmp_path / "report.json"
    replacement = tmp_path / "replacement.json"
    replacement.write_text("attacker-owned")
    invoked = False
    original_fsync = subject.os.fsync

    def hostile_fsync(descriptor: int) -> None:
        nonlocal invoked
        if not invoked:
            invoked = True
            subject.os.replace(replacement, output)
            raise OSError("injected fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(subject.os, "fsync", hostile_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        subject.write_capsule_constructability_report(output, report)
    assert output.read_text() == "attacker-owned"
