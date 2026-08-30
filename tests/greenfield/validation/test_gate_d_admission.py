from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from glm_tpu.greenfield.errors import BenchmarkValidationError
from glm_tpu.greenfield import gate_d_admission as subject
from glm_tpu.greenfield import observability


REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_CONTRACT = REPO_ROOT / "configs/greenfield-gate-d-mechanism-admission.json"
FRONTIER_CAPSULE = (
    REPO_ROOT / "docs/artifacts/gate-d-mechanism-admission-frontier.json"
)
ADMISSION_CORE = REPO_ROOT / "glm_tpu/greenfield/gate_d_admission.py"
ADMISSION_CLI = REPO_ROOT / "scripts/greenfield/admit_gate_d_mechanisms.py"
REQUIRED_WATCHPOINTS = [
    "layer1.rms_input_fp32",
    "layer1.normalized",
    "layer1.cache_history",
    "layer1.query",
    "layer1.head_weights",
    "layer1.current_key",
    "layer1.scorer_event1",
]
NOVEL_FINGERPRINT = {
    "association": "new.association",
    "consumer_boundary": "rms.weighted",
    "reduction": "local.fp32",
    "representation": "new.representation",
    "transport": "stage.local",
}
CLOSED_FINGERPRINT = {
    "association": "closed.association",
    "consumer_boundary": "rms.weighted",
    "reduction": "closed.reduction",
    "representation": "closed.representation",
    "transport": "stage.local",
}


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False, sort_keys=True))


def _fingerprint_sha256(value: dict[str, str]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return sha256(payload).hexdigest()


def _plan(role: str) -> dict[str, object]:
    name = f"{role.upper()}_TEST"
    return {
        "authority_file_sha256": None,
        "authority_json_path": [],
        "authority_kind": "canonical_name_sha256",
        "authority_path": None,
        "id": f"{role}.plan",
        "name": name,
        "plan_sha256": sha256(name.encode()).hexdigest(),
    }


def _identity_source(
    tmp_path: Path,
    *,
    role: str,
    artifact: Path,
    observations: list[dict[str, object]],
) -> dict[str, object]:
    accepted = role == "accepted"
    code_pin = ("a" if accepted else "b") * 40
    coherence_id = f"{role}.event8155"
    executable_sha256 = None if accepted else "d" * 64
    identity = {
        "artifact_sha256": _sha(artifact),
        "code_pin": code_pin,
        "coherence_id": coherence_id,
        "executable_identity_sha256": executable_sha256,
    }
    identity_path = tmp_path / f"{role}-identity.json"
    _write_json(identity_path, identity)
    bindings = [
        {"json_path": ["artifact_sha256"], "source_field": "artifact_sha256"},
        {"json_path": ["code_pin"], "source_field": "code_pin"},
        {"json_path": ["coherence_id"], "source_field": "coherence_id"},
    ]
    if not accepted:
        bindings.append(
            {
                "json_path": ["executable_identity_sha256"],
                "source_field": "executable_identity_sha256",
            }
        )
    return {
        "artifact_path": str(artifact),
        "artifact_sha256": _sha(artifact),
        "code_pin": code_pin,
        "coherence_id": coherence_id,
        "executable_identity_kind": "unavailable" if accepted else "optimized_hlo",
        "executable_identity_sha256": executable_sha256,
        "id": f"{role}.source",
        "identity_evidence": [
            {
                "artifact_path": str(identity_path),
                "artifact_sha256": _sha(identity_path),
                "bindings": bindings,
            }
        ],
        "observation_method": "protected_oracle" if accepted else "device_output",
        "observations": observations,
        "plan_id": f"{role}.plan",
        "plan_sha256": _plan(role)["plan_sha256"],
        "role": role,
        "trust": "accepted_protected" if accepted else "candidate_protected",
    }


def _typed_capsule(tmp_path: Path, mechanism_id: str = "new.mechanism") -> Path:
    definitions = [
        ("layer1.rms_input_fp32", "float32", np.asarray([[1.0, 2.0]], np.float32)),
        ("layer1.normalized", "bf16", np.asarray([[0x3F80, 0x4000]], np.uint16)),
        ("layer1.cache_history", "float32", np.asarray([[1.0, 2.0]], np.float32)),
        ("layer1.query", "float32", np.asarray([[3.0, 4.0]], np.float32)),
        ("layer1.head_weights", "float32", np.asarray([0.5], np.float32)),
        ("layer1.current_key", "float32", np.asarray([5.0, 6.0], np.float32)),
        ("layer1.scorer_event1", "float32", np.asarray([7.0, 8.0], np.float32)),
    ]
    arrays = {f"value_{index}": item[2] for index, item in enumerate(definitions)}
    candidate_artifact = tmp_path / "candidate-state.npz"
    np.savez(candidate_artifact, **arrays)
    candidate_inventory = {
        item["key"]: item
        for item in observability.inspect_npz_artifact(
            candidate_artifact, _sha(candidate_artifact)
        )["members"]
    }
    accepted_artifact = tmp_path / "accepted-state.npz"
    np.savez(accepted_artifact, normalized=definitions[1][2])
    accepted_inventory = {
        item["key"]: item
        for item in observability.inspect_npz_artifact(
            accepted_artifact, _sha(accepted_artifact)
        )["members"]
    }
    watchpoints: list[dict[str, object]] = []
    candidate_bindings: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    layout_document: dict[str, object] = {"watchpoints": {}}
    for order, (identifier, semantic_dtype, array) in enumerate(definitions):
        key = f"value_{order}"
        member = candidate_inventory[key]
        watchpoints.append(
            {
                "id": identifier,
                "layer": 1,
                "order": order,
                "position": 8155,
                "semantic_dtype": semantic_dtype,
                "shape": list(array.shape),
            }
        )
        candidate_bindings.append(
            {
                "array_sha256": member["raw_sha256"],
                "index_prefix": [],
                "key": key,
                "shape": list(array.shape),
                "storage_dtype": member["storage_dtype"],
                "watchpoint_id": identifier,
            }
        )
        layout_document["watchpoints"][identifier] = {
            "layout": "lp4.replicated",
            "owner_ids": [0, 1, 2, 3],
        }
    accepted_binding = {
        "array_sha256": accepted_inventory["normalized"]["raw_sha256"],
        "index_prefix": [],
        "key": "normalized",
        "shape": list(definitions[1][2].shape),
        "storage_dtype": accepted_inventory["normalized"]["storage_dtype"],
        "watchpoint_id": "layer1.normalized",
    }
    observability_contract = tmp_path / "candidate-observability.json"
    _write_json(
        observability_contract,
        {
            "contract_id": "gate.d.candidate.coherence",
            "plans": [_plan("accepted"), _plan("candidate")],
            "schema_version": observability.OBSERVABILITY_SCHEMA_VERSION,
            "sources": [
                _identity_source(
                    tmp_path,
                    role="accepted",
                    artifact=accepted_artifact,
                    observations=[accepted_binding],
                ),
                _identity_source(
                    tmp_path,
                    role="candidate",
                    artifact=candidate_artifact,
                    observations=candidate_bindings,
                ),
            ],
            "watchpoints": watchpoints,
        },
    )
    layout_path = tmp_path / "layout-evidence.json"
    _write_json(layout_path, layout_document)
    for order, (identifier, semantic_dtype, array) in enumerate(definitions):
        key = f"value_{order}"
        member = candidate_inventory[key]
        metadata.append(
            {
                "array_key": key,
                "array_sha256": member["raw_sha256"],
                "id": identifier,
                "index_prefix": [],
                "layer": 1,
                "layout": "lp4.replicated",
                "layout_evidence": {
                    "json_path": ["watchpoints", identifier],
                    "path": layout_path.name,
                    "sha256": _sha(layout_path),
                },
                "owner_ids": [0, 1, 2, 3],
                "position": 8155,
                "semantic_dtype": semantic_dtype,
                "shape": list(array.shape),
                "source_id": "candidate.source",
            }
        )
    capsule = tmp_path / "capsule.json"
    _write_json(
        capsule,
        {
            "causal_frontier": {
                "action": "resolve",
                "id": "layer1.rms_input_fp32",
            },
            "code_pin": "b" * 40,
            "coherence_id": "candidate.event8155",
            "executable_identity_sha256": "d" * 64,
            "mechanism_id": mechanism_id,
            "observability_contract": {
                "path": observability_contract.name,
                "sha256": _sha(observability_contract),
            },
            "plan_sha256": _plan("candidate")["plan_sha256"],
            "schema_version": subject.GATE_D_ADMISSION_SCHEMA_VERSION,
            "watchpoint_metadata": metadata,
        },
    )
    return capsule


def _opaque_capsule(tmp_path: Path) -> Path:
    member = tmp_path / "opaque-state.bin"
    member.write_bytes(b"complete candidate-coherent state")
    capsule = tmp_path / "opaque-capsule.json"
    _write_json(
        capsule,
        {
            "causal_frontier": {
                "action": "resolve",
                "id": "layer1.rms_input_fp32",
                "member_id": "state",
            },
            "code_pin": "b" * 40,
            "coherence_id": "candidate.event8155",
            "executable_identity_sha256": "d" * 64,
            "mechanism_id": "new.mechanism",
            "members": [{"id": "state", "path": member.name, "sha256": _sha(member)}],
            "plan_sha256": "c" * 64,
            "schema_version": subject.GATE_D_ADMISSION_SCHEMA_VERSION,
            "watchpoints": [
                {"id": item, "member_id": "state"} for item in REQUIRED_WATCHPOINTS
            ],
        },
    )
    return capsule


def _contract(
    tmp_path: Path,
    *,
    candidate: dict[str, object] | None = None,
    capsule: Path | None = None,
) -> Path:
    evidence = tmp_path / "evidence.json"
    _write_json(evidence, {"classification": "SEALED"})
    if candidate is None:
        candidate = {
            "causal_frontier_action": "resolve",
            "coherent_state_capsule": (
                None
                if capsule is None
                else {"path": capsule.name, "sha256": _sha(capsule)}
            ),
            "evidence_ids": ["frontier"],
            "family_id": None,
            "host_effect": False,
            "id": "new.mechanism",
            "logical_rows": 1,
            "max_collective_group_size": 4,
            "mechanism_fingerprint": NOVEL_FINGERPRINT,
            "reconstructs_full_pod_hidden": False,
            "summary": "A genuinely new plan-local mechanism.",
        }
    path = tmp_path / "contract.json"
    _write_json(
        path,
        {
            "candidates": [candidate],
            "causal_frontier": {
                "evidence_id": "frontier",
                "id": "layer1.rms_input_fp32",
                "order": 0,
            },
            "closed_families": [
                {
                    "evidence_ids": ["frontier"],
                    "id": "closed.family",
                    "mechanism_fingerprint_sha256s": [
                        _fingerprint_sha256(CLOSED_FINGERPRINT)
                    ],
                    "reason": "Previously rejected.",
                }
            ],
            "contract_id": "gate.d.test.admission",
            "evidence": [
                {
                    "claim_scope": "Test evidence only.",
                    "id": "frontier",
                    "path": evidence.name,
                    "sha256": _sha(evidence),
                }
            ],
            "locality_contract": {
                "logical_rows": 1,
                "max_collective_group_size": 4,
                "no_full_pod_hidden_reconstruction": True,
                "no_host_effects": True,
            },
            "required_coherent_watchpoints": REQUIRED_WATCHPOINTS,
            "schema_version": subject.GATE_D_ADMISSION_SCHEMA_VERSION,
            "upstream_snapshots": [
                {
                    "commit": "d" * 40,
                    "id": "upstream.test",
                    "scope": "Test snapshot only.",
                }
            ],
        },
    )
    return path


def _admit(path: Path) -> dict[str, object]:
    return subject.admit_gate_d_mechanisms(path, _sha(path))


def test_real_contract_rejects_every_current_candidate_without_jax() -> None:
    jax_module = sys.modules.get("jax")
    result = _admit(REAL_CONTRACT)
    assert result["classification"] == "NO_ADMISSIBLE_MECHANISM"
    assert result["admitted_candidate_ids"] == []
    assert result["tpu_successor_authorized"] is False
    assert result["jax_or_tpu_work_performed"] is False
    assert len(result["candidate_results"]) == 15
    assert sys.modules.get("jax") is jax_module


def test_complete_new_candidate_passes_offline_admission_only(tmp_path: Path) -> None:
    capsule = _typed_capsule(tmp_path)
    contract = _contract(tmp_path, capsule=capsule)
    result = _admit(contract)
    assert result["classification"] == "OFFLINE_CANDIDATE_ADMITTED"
    assert result["admitted_candidate_ids"] == ["new.mechanism"]
    assert result["candidate_results"][0]["reasons"] == []
    assert result["tpu_successor_authorized"] is False


def test_closed_family_is_rejected_even_with_complete_capsule(tmp_path: Path) -> None:
    capsule = _typed_capsule(tmp_path)
    candidate = {
        "causal_frontier_action": "resolve",
        "coherent_state_capsule": {"path": capsule.name, "sha256": _sha(capsule)},
        "evidence_ids": ["frontier"],
        "family_id": "closed.family",
        "host_effect": False,
        "id": "new.mechanism",
        "logical_rows": 1,
        "max_collective_group_size": 4,
        "mechanism_fingerprint": CLOSED_FINGERPRINT,
        "reconstructs_full_pod_hidden": False,
        "summary": "A renamed closed mechanism.",
    }
    result = _admit(_contract(tmp_path, candidate=candidate))
    item = result["candidate_results"][0]
    assert item["admitted_offline"] is False
    assert item["reasons"] == ["DUPLICATES_CLOSED_FAMILY"]


def test_renamed_closed_fingerprint_is_rejected_with_null_family(
    tmp_path: Path,
) -> None:
    capsule = _typed_capsule(tmp_path)
    candidate = {
        "causal_frontier_action": "resolve",
        "coherent_state_capsule": {"path": capsule.name, "sha256": _sha(capsule)},
        "evidence_ids": ["frontier"],
        "family_id": None,
        "host_effect": False,
        "id": "renamed.closed.mechanism",
        "logical_rows": 1,
        "max_collective_group_size": 4,
        "mechanism_fingerprint": CLOSED_FINGERPRINT,
        "reconstructs_full_pod_hidden": False,
        "summary": "The same closed mechanism under a new name.",
    }
    capsule_value = json.loads(capsule.read_text())
    capsule_value["mechanism_id"] = candidate["id"]
    _write_json(capsule, capsule_value)
    candidate["coherent_state_capsule"]["sha256"] = _sha(capsule)
    item = _admit(_contract(tmp_path, candidate=candidate))["candidate_results"][0]
    assert item["reasons"] == ["DUPLICATES_CLOSED_FAMILY"]


def test_illegal_geometry_host_effect_and_downstream_action_are_all_named(
    tmp_path: Path,
) -> None:
    candidate = {
        "causal_frontier_action": "downstream",
        "coherent_state_capsule": None,
        "evidence_ids": ["frontier"],
        "family_id": None,
        "host_effect": True,
        "id": "illegal.mechanism",
        "logical_rows": 32,
        "max_collective_group_size": 32,
        "mechanism_fingerprint": NOVEL_FINGERPRINT,
        "reconstructs_full_pod_hidden": True,
        "summary": "An illegal mechanism.",
    }
    result = _admit(_contract(tmp_path, candidate=candidate))
    assert result["candidate_results"][0]["reasons"] == [
        "NOT_TRUE_ONE_ROW",
        "NONLOCAL_COLLECTIVE_GROUP",
        "FULL_POD_HIDDEN_RECONSTRUCTION",
        "HOST_EFFECT_OR_DISPATCH",
        "DOES_NOT_MOVE_CAUSAL_FRONTIER",
        "MISSING_CANDIDATE_COHERENT_CAPSULE",
    ]


def test_missing_capsule_rejects_unclosed_candidate(tmp_path: Path) -> None:
    result = _admit(_contract(tmp_path))
    assert result["candidate_results"][0]["reasons"] == [
        "MISSING_CANDIDATE_COHERENT_CAPSULE"
    ]


def test_incomplete_coherent_watchpoints_reject_capsule(tmp_path: Path) -> None:
    capsule = _typed_capsule(tmp_path)
    value = json.loads(capsule.read_text())
    value["watchpoint_metadata"] = value["watchpoint_metadata"][:-1]
    _write_json(capsule, value)
    result = _admit(_contract(tmp_path, capsule=capsule))
    item = result["candidate_results"][0]
    assert item["reasons"] == ["INVALID_CANDIDATE_COHERENT_CAPSULE"]
    assert "coherent watchpoints missing" in item["coherent_state_capsule"]["refusal"]


def test_capsule_mechanism_identity_is_bound(tmp_path: Path) -> None:
    capsule = _typed_capsule(tmp_path, mechanism_id="wrong.mechanism")
    result = _admit(_contract(tmp_path, capsule=capsule))
    item = result["candidate_results"][0]
    assert item["reasons"] == ["INVALID_CANDIDATE_COHERENT_CAPSULE"]
    assert item["coherent_state_capsule"]["refusal"] == (
        "candidate capsule mechanism drifted"
    )


def test_capsule_observability_artifact_replacement_fails_closed(
    tmp_path: Path,
) -> None:
    capsule = _typed_capsule(tmp_path)
    contract = _contract(tmp_path, capsule=capsule)
    (tmp_path / "candidate-state.npz").write_bytes(b"replaced")
    result = _admit(contract)
    item = result["candidate_results"][0]
    assert item["reasons"] == ["INVALID_CANDIDATE_COHERENT_CAPSULE"]
    assert "SHA-256 drifted" in item["coherent_state_capsule"]["refusal"]


def test_opaque_named_watchpoint_blob_is_rejected(tmp_path: Path) -> None:
    capsule = _opaque_capsule(tmp_path)
    item = _admit(_contract(tmp_path, capsule=capsule))["candidate_results"][0]
    assert item["reasons"] == ["INVALID_CANDIDATE_COHERENT_CAPSULE"]
    assert "keys drifted" in item["coherent_state_capsule"]["refusal"]


def test_evidence_replacement_refuses_entire_contract(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    (tmp_path / "evidence.json").write_text("replaced")
    with pytest.raises(BenchmarkValidationError, match="evidence frontier SHA-256"):
        _admit(contract)


def test_duplicate_json_keys_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(BenchmarkValidationError, match="duplicate JSON key"):
        subject.admit_gate_d_mechanisms(path, _sha(path))


def test_nonfinite_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text('{"value":NaN}')
    with pytest.raises(BenchmarkValidationError, match="non-finite JSON value"):
        subject.admit_gate_d_mechanisms(path, _sha(path))


def test_symlink_contract_and_evidence_are_refused(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    contract_link = tmp_path / "contract-link.json"
    contract_link.symlink_to(contract)
    with pytest.raises(BenchmarkValidationError, match="cannot open"):
        subject.admit_gate_d_mechanisms(contract_link, _sha(contract))

    evidence = tmp_path / "evidence.json"
    target = tmp_path / "evidence-target.json"
    evidence.rename(target)
    evidence.symlink_to(target)
    with pytest.raises(BenchmarkValidationError, match="cannot open"):
        _admit(contract)


def test_intermediate_directory_symlinks_are_refused_for_input_and_output(
    tmp_path: Path,
) -> None:
    target = tmp_path / "real"
    target.mkdir()
    contract = _contract(target)
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(BenchmarkValidationError, match="cannot safely open"):
        subject.admit_gate_d_mechanisms(alias / contract.name, _sha(contract))

    redirected = alias / "redirected-report.json"
    with pytest.raises(BenchmarkValidationError, match="cannot safely open"):
        subject.write_gate_d_admission_report(
            redirected, {"classification": "NO_ADMISSIBLE_MECHANISM"}
        )
    assert not (target / redirected.name).exists()


def test_unknown_closed_family_is_refused(tmp_path: Path) -> None:
    candidate = {
        "causal_frontier_action": "resolve",
        "coherent_state_capsule": None,
        "evidence_ids": ["frontier"],
        "family_id": "not.registered",
        "host_effect": False,
        "id": "new.mechanism",
        "logical_rows": 1,
        "max_collective_group_size": 4,
        "mechanism_fingerprint": NOVEL_FINGERPRINT,
        "reconstructs_full_pod_hidden": False,
        "summary": "Unknown family.",
    }
    with pytest.raises(BenchmarkValidationError, match="unknown closed family"):
        _admit(_contract(tmp_path, candidate=candidate))


def test_append_only_writer_refuses_regular_and_dangling_paths(
    tmp_path: Path,
) -> None:
    report = {"classification": "NO_ADMISSIBLE_MECHANISM"}
    occupied = tmp_path / "report.json"
    occupied.write_text("occupied")
    with pytest.raises(BenchmarkValidationError, match="occupied"):
        subject.write_gate_d_admission_report(occupied, report)

    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "absent.json")
    with pytest.raises(BenchmarkValidationError, match="occupied"):
        subject.write_gate_d_admission_report(dangling, report)


def test_cli_runs_with_python_s_without_importing_jax(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            "/home/gianl/vllm-env/bin/python",
            "-S",
            str(REPO_ROOT / "scripts/greenfield/admit_gate_d_mechanisms.py"),
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
    summary = json.loads(completed.stdout)
    capsule = json.loads(FRONTIER_CAPSULE.read_text())
    assert summary["classification"] == "NO_ADMISSIBLE_MECHANISM"
    assert json.loads(output.read_text())["tpu_successor_authorized"] is False
    assert _sha(output) == capsule["reproduction"]["report_sha256"]
    assert _sha(REAL_CONTRACT) == capsule["reproduction"]["contract_sha256"]
    assert _sha(ADMISSION_CORE) == capsule["reproduction"]["core_sha256"]
    assert _sha(ADMISSION_CLI) == capsule["reproduction"]["cli_sha256"]


def test_cli_wrong_contract_hash_refuses_before_output(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            "/home/gianl/vllm-env/bin/python",
            "-S",
            str(REPO_ROOT / "scripts/greenfield/admit_gate_d_mechanisms.py"),
            "--contract",
            str(REAL_CONTRACT),
            "--contract-sha256",
            "0" * 64,
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "SHA-256 drifted" in completed.stderr
    assert not output.exists()
