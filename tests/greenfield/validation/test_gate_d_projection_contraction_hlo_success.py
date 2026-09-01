from __future__ import annotations

import json
import os
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from glm_tpu.greenfield.validation.gate_d_projection_contraction_hlo import (
    ProjectionContractionHloError,
    audit_optimized_hlo_structure,
)
from glm_tpu.greenfield.validation.gate_d_projection_contraction_hlo_success import (
    EXPECTED_OPTIMIZED_HLO_SHA256,
    EXPECTED_REMOTE_REPLAY,
    EXPECTED_REMOTE_REPLAY_SHA256,
    EXPECTED_RUN_DIR,
    EXPECTED_TERMINAL_GENERATION,
    EXPECTED_TERMINAL_SHA256,
    adjudicate_success_run,
    audit_success_evidence,
    audit_success_marker,
    audit_success_remote_replay,
    audit_success_runner,
    audit_success_summary,
)

OPTIMIZED = (
    EXPECTED_RUN_DIR / "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt"
)
MARKER = EXPECTED_RUN_DIR / "HLO_ACQUIRED"
RUNNER = EXPECTED_RUN_DIR / "runner.json"
SUMMARY = EXPECTED_RUN_DIR / "summary.json"
EVIDENCE = EXPECTED_RUN_DIR / "evidence.json"
MODULE_SOURCE = Path(__file__).resolve().parents[3] / (
    "glm_tpu/greenfield/validation/gate_d_projection_contraction_hlo_success.py"
)
ARTIFACT = Path(__file__).resolve().parents[3] / (
    "docs/artifacts/gate-d-projection-contraction-pp16-success-hlo-adjudication.json"
)
SCRIPT = Path(__file__).resolve().parents[3] / (
    "scripts/greenfield/adjudicate_gate_d_projection_contraction_pp16_hlo_success.py"
)
pytestmark = pytest.mark.skipif(
    not (OPTIMIZED.is_file() and MARKER.is_file()),
    reason="successful projection-contraction HLO is not present",
)


def _replace_once(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def test_exact_success_run_accepts_persistence_but_not_numerical_authority() -> None:
    report = adjudicate_success_run()
    assert report["classification"] == (
        "HLO_CAUSAL_STRUCTURE_ACCEPTED;PP16_OWNER_LOCALITY_ACCEPTED;"
        "TPU_NUMERICAL_UNPROVEN;HLO_ACQUIRED_TERMINAL_VERIFIED;GATE_D_OPEN"
    )
    assert report["authorization"] == {
        "full_8k": False,
        "numerical_execution": False,
        "performance_claim": False,
        "persistence_only": True,
    }
    assert report["gate_d_closed"] is False
    assert report["optimized_hlo"]["sha256"] == EXPECTED_OPTIMIZED_HLO_SHA256
    assert report["optimized_hlo"]["structure"]["collective_count"] == 0
    assert report["optimized_hlo"]["structure"]["host_effect_count"] == 0
    assert report["optimized_hlo"]["structure"]["live_rows_per_owner"] == 1
    assert report["terminal"] == {
        "generation": EXPECTED_TERMINAL_GENERATION,
        "sha256": EXPECTED_TERMINAL_SHA256,
    }


def test_committed_success_adjudication_is_exact_isolated_cli_output() -> None:
    result = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", str(SCRIPT)],
        cwd=Path("/"),
        env={
            "HOME": "/nonexistent",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("ascii")
    assert result.stderr == b""
    artifact = ARTIFACT.read_bytes()
    assert artifact == result.stdout
    assert len(artifact) == 2301
    assert sha256(artifact).hexdigest() == (
        "54eb6105b81d9ffdbdd3b90735059fb324701509fe8efd003033bb3cff7e0ad7"
    )


def test_remote_replay_is_generation_qualified_terminal_last_and_fail_closed() -> None:
    raw = EXPECTED_REMOTE_REPLAY.read_bytes()
    assert sha256(raw).hexdigest() == EXPECTED_REMOTE_REPLAY_SHA256
    objects = audit_success_remote_replay(raw)
    assert len(objects) == 17
    assert objects["HLO_ACQUIRED"] == {
        "crc32c": "6B0KjQ==",
        "generation": EXPECTED_TERMINAL_GENERATION,
        "path": "HLO_ACQUIRED",
        "sha256": EXPECTED_TERMINAL_SHA256,
        "size": 764,
    }
    mutation = raw.replace(
        b'"success_terminal_last": true',
        b'"success_terminal_last": false',
        1,
    )
    with pytest.raises(ProjectionContractionHloError):
        audit_success_remote_replay(mutation)


def test_terminal_marker_is_self_bound_to_the_replayed_ledger() -> None:
    remote = audit_success_remote_replay(EXPECTED_REMOTE_REPLAY.read_bytes())
    marker = audit_success_marker(MARKER.read_bytes(), remote)
    assert marker["marker_payload_sha256"] == (
        "a3655718fe41efef9185681878f66bb0ad4aa76b734ed8bc642ad52788708f25"
    )
    hostile_remote = {path: dict(record) for path, record in remote.items()}
    hostile_remote["evidence.json"]["sha256"] = "0" * 64
    with pytest.raises(ProjectionContractionHloError):
        audit_success_marker(MARKER.read_bytes(), hostile_remote)


@pytest.mark.parametrize(
    ("path", "field", "auditor"),
    (
        (RUNNER, "gate_d_closed", audit_success_runner),
        (RUNNER, "numerical_claim", audit_success_runner),
        (RUNNER, "performance_claim", audit_success_runner),
        (SUMMARY, "gate_d_closed", audit_success_summary),
        (SUMMARY, "numerical_claim", audit_success_summary),
        (SUMMARY, "performance_claim", audit_success_summary),
    ),
)
def test_runner_and_summary_contradictory_claim_attacks_fail(
    path: Path, field: str, auditor: object
) -> None:
    hostile = json.loads(path.read_bytes())
    hostile[field] = True
    with pytest.raises(ProjectionContractionHloError):
        auditor(hostile)  # type: ignore[operator]


@pytest.mark.parametrize(
    "field", ("gate_d_closed", "numerical_claim", "performance_claim")
)
def test_evidence_contradictory_claim_attacks_fail(field: str) -> None:
    remote = audit_success_remote_replay(EXPECTED_REMOTE_REPLAY.read_bytes())
    evidence_paths = set(remote) - {
        "HLO_ACQUIRED",
        "remote_objects.json",
        "evidence.json",
    }
    hostile = json.loads(EVIDENCE.read_bytes())
    hostile[field] = True
    with pytest.raises(ProjectionContractionHloError):
        audit_success_evidence(hostile, remote, evidence_paths)


def test_exact_claim_schemas_and_duplicate_evidence_records_fail() -> None:
    runner = json.loads(RUNNER.read_bytes())
    runner["unexpected_claim"] = False
    with pytest.raises(ProjectionContractionHloError):
        audit_success_runner(runner)

    summary = json.loads(SUMMARY.read_bytes())
    summary["claim_scope"] = "numerical success"
    with pytest.raises(ProjectionContractionHloError):
        audit_success_summary(summary)

    remote = audit_success_remote_replay(EXPECTED_REMOTE_REPLAY.read_bytes())
    evidence_paths = set(remote) - {
        "HLO_ACQUIRED",
        "remote_objects.json",
        "evidence.json",
    }
    evidence = json.loads(EVIDENCE.read_bytes())
    evidence["files"][-1] = dict(evidence["files"][0])
    with pytest.raises(ProjectionContractionHloError):
        audit_success_evidence(evidence, remote, evidence_paths)


def test_success_hlo_parameter_to_root_bypass_attack_fails() -> None:
    source = OPTIMIZED.read_text(encoding="ascii")
    hostile = _replace_once(
        source,
        "add(%mul.44, %convert_element_type.15)",
        "add(%convert_element_type.15, %convert_element_type.15)",
    )
    with pytest.raises(ProjectionContractionHloError):
        audit_optimized_hlo_structure(hostile)


def test_success_adjudicator_source_is_offline_only() -> None:
    source = MODULE_SOURCE.read_text(encoding="ascii")
    assert "import jax" not in source
    assert "jax.jit" not in source
    assert ".compile(" not in source
    assert "block_until_ready" not in source
    assert "google.cloud" not in source
    assert "subprocess" not in source


def test_isolated_success_cli_ignores_hostile_package_initializers(
    tmp_path: Path,
) -> None:
    hostile = tmp_path / "hostile"
    package = hostile / "glm_tpu"
    package.mkdir(parents=True)
    marker = tmp_path / "package-initializer-ran"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
        encoding="ascii",
    )
    startup = tmp_path / "startup.py"
    startup.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('startup')\n",
        encoding="ascii",
    )
    result = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", str(SCRIPT)],
        cwd=hostile,
        env={
            **os.environ,
            "PYTHONPATH": str(hostile),
            "PYTHONSTARTUP": str(startup),
        },
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode("ascii")
    assert not marker.exists()
    assert json.loads(result.stdout)["process_contract"] == {
        "accelerator_fds_after": [],
        "accelerator_fds_before": [],
        "forbidden_modules_after": [],
        "forbidden_modules_before": [],
        "package_initializers_executed": [],
        "python_flags": ["-I", "-S", "-B"],
    }
