from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import pytest

from glm_tpu.greenfield.errors import BenchmarkValidationError
from glm_tpu.greenfield import shadow_variant_adjudication as subject


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_CONTRACT = REPO_ROOT / "configs/greenfield-gate-d-shadow-variant-adjudication.json"
CLI = REPO_ROOT / "scripts/greenfield/adjudicate_shadow_variants.py"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True))


def _contract(tmp_path: Path) -> Path:
    value = json.loads(REAL_CONTRACT.read_text())
    for item in value["evidence"]:
        item["path"] = str((REAL_CONTRACT.parent / item["path"]).resolve())
    path = tmp_path / "contract.json"
    _write(path, value)
    return path


def _adjudicate(path: Path) -> dict[str, object]:
    return subject.adjudicate_shadow_variants(path, _sha(path))


def test_real_contract_is_fail_closed_and_canonical() -> None:
    report = _adjudicate(REAL_CONTRACT)
    assert report["classification"] == (
        "NO_OFFLINE_COMPLETE_SHADOW_VARIANT;"
        "AUXILIARY_DEVICE_VARIANTS_REMAIN_UNADJUDICATED;"
        "GATE_D_OPEN;NO_TPU_SUCCESSOR"
    )
    assert report["gate_d_closed"] is False
    assert report["tpu_successor_authorized"] is False
    assert report["unresolved_variant_ids"] == [
        "auxiliary_device_tuple_dependency",
        "compensated_auxiliary_dependency",
    ]
    results = {item["id"]: item for item in report["variant_results"]}
    assert results["rounded_widened_primary"]["classification"] == "DUPLICATE_CLOSED"
    assert results["compensated_restore_unrounded"]["classification"] == (
        "REJECTED_SEMANTIC_DRIFT"
    )
    assert results["compensated_cancelled_no_dependency"]["classification"] == (
        "REJECTED_NONCAUSAL"
    )
    assert results["host_shadow_observer"]["classification"] == "ILLEGAL_LOCALITY"
    assert set(results["host_shadow_observer"]["reasons"]) >= {
        "DEAD_ROWS_FORBIDDEN",
        "FULL_POD_OR_OVERSIZED_GROUP_FORBIDDEN",
        "HOST_EFFECT_FORBIDDEN",
        "FULL_POD_HIDDEN_RECONSTRUCTION_FORBIDDEN",
    }


def test_renaming_cannot_bypass_versioned_catalogue(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    original = value["variants"][0]
    original["id"] = "brand_new_name"
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="catalogue drifted"):
        _adjudicate(contract)


def test_restoring_unrounded_cannot_be_mislabeled_auxiliary(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    variant = value["variants"][3]
    variant["normal_form"]["compensation"] = "restore.unrounded"
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="normal form drifted"):
        _adjudicate(contract)


def test_catalogue_deletion_addition_and_class_drift_are_refused(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["variants"].pop()
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="catalogue drifted"):
        _adjudicate(contract)

    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    extra = dict(value["variants"][0])
    extra["id"] = "extra_variant"
    value["variants"].append(extra)
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="catalogue drifted"):
        _adjudicate(contract)

    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["variants"][4]["normal_form"]["compensation"] = "cancelled"
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="normal form drifted"):
        _adjudicate(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logical_rows", 32),
        ("max_collective_group_size", 32),
        ("host_effect", True),
        ("reconstructs_full_pod_hidden", True),
    ],
)
def test_illegal_execution_shapes_never_survive(
    tmp_path: Path, field: str, value: object
) -> None:
    contract = _contract(tmp_path)
    document = json.loads(contract.read_text())
    document["variants"][4][field] = value
    _write(contract, document)
    with pytest.raises(
        BenchmarkValidationError, match="unresolved shadow variant set drifted"
    ):
        _adjudicate(contract)


def test_accepted_recurrence_and_watchpoints_cannot_be_weakened(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["accepted_semantics"]["returned_recurrence"] = "fp32.unrounded"
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="BF16 recurrence"):
        _adjudicate(contract)

    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["required_coherent_watchpoints"].pop()
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="seven-watchpoint"):
        _adjudicate(contract)


def test_claimed_hlo_or_capsule_fields_are_refused_not_trusted(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["variants"][4]["causal_hlo_certificate"] = {
        "path": "made-up.json",
        "sha256": "0" * 64,
    }
    value["variants"][4]["coherent_state_capsule"] = {
        "path": "made-up.json",
        "sha256": "0" * 64,
    }
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="keys drifted"):
        _adjudicate(contract)


def test_wrong_evidence_hash_refuses_entire_contract(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["evidence"][0]["sha256"] = "0" * 64
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="SHA-256 drifted"):
        _adjudicate(contract)


def test_duplicate_and_nonfinite_json_are_refused(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(BenchmarkValidationError, match="duplicate JSON key"):
        subject.adjudicate_shadow_variants(duplicate, _sha(duplicate))

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}')
    with pytest.raises(BenchmarkValidationError, match="non-finite JSON value"):
        subject.adjudicate_shadow_variants(nonfinite, _sha(nonfinite))


def test_boolean_schema_and_unhashable_list_members_are_refused(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["schema_version"] = True
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="schema drifted"):
        _adjudicate(contract)

    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["required_coherent_watchpoints"][0] = ["not", "hashable"]
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="canonical identifier"):
        _adjudicate(contract)

    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    value["variants"][0]["evidence_ids"][0] = ["not", "hashable"]
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="canonical identifier"):
        _adjudicate(contract)


def test_symlinked_input_output_and_occupied_output_are_refused(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    alias = tmp_path / "contract-link.json"
    alias.symlink_to(contract)
    with pytest.raises(BenchmarkValidationError, match="cannot open"):
        subject.adjudicate_shadow_variants(alias, _sha(contract))

    report = _adjudicate(contract)
    occupied = tmp_path / "report.json"
    occupied.write_text("occupied")
    with pytest.raises(BenchmarkValidationError, match="occupied"):
        subject.write_shadow_variant_report(occupied, report)
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "absent.json")
    with pytest.raises(BenchmarkValidationError, match="occupied"):
        subject.write_shadow_variant_report(dangling, report)


def test_intermediate_and_evidence_symlinks_are_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    contract = _contract(real)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(BenchmarkValidationError, match="cannot safely open"):
        subject.adjudicate_shadow_variants(alias / contract.name, _sha(contract))
    with pytest.raises(BenchmarkValidationError, match="cannot safely open"):
        subject.write_shadow_variant_report(alias / "report.json", {"ok": False})

    contract = _contract(tmp_path)
    value = json.loads(contract.read_text())
    evidence = Path(value["evidence"][0]["path"])
    evidence_alias = tmp_path / "evidence-link.json"
    evidence_alias.symlink_to(evidence)
    value["evidence"][0]["path"] = str(evidence_alias)
    _write(contract, value)
    with pytest.raises(BenchmarkValidationError, match="cannot open"):
        _adjudicate(contract)


def test_canonical_output_is_byte_stable(tmp_path: Path) -> None:
    report = _adjudicate(REAL_CONTRACT)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    subject.write_shadow_variant_report(first, report)
    subject.write_shadow_variant_report(second, report)
    assert first.read_bytes() == second.read_bytes()


def test_cli_runs_under_python_s_without_importing_jax(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            "/home/gianl/vllm-env/bin/python",
            "-S",
            str(CLI),
            "--contract",
            str(REAL_CONTRACT),
            "--contract-sha256",
            _sha(REAL_CONTRACT),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert "jax" not in completed.stdout.lower()
    report = json.loads(output.read_text())
    assert report["tpu_successor_authorized"] is False
    assert report["gate_d_closed"] is False
