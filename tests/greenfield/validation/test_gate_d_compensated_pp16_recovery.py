from __future__ import annotations

import json
import importlib.util
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking import gate_d_compensated_pp16_recovery as recovery
from glm_tpu.greenfield.errors import BenchmarkValidationError

ROOT = Path(__file__).parents[3]
SOURCE_RUN = Path("/home/gianl/gate-d-runs") / recovery.SOURCE_TAG
CAPSULE_ROOT = Path("/home/gianl/gate-d-runs") / (
    "greenfield_gate_d_compensated_capsule_20260831T124838Z"
)
CAPSULE_AUTHORITY = Path("/home/gianl/gate-d-runs") / (
    "greenfield_gate_d_compensated_capsule_20260831T124838Z-execution-authority.json"
)
ORCHESTRATOR = ROOT / "scripts/greenfield/recover_gate_d_compensated_pp16_numerical.py"


def _load_orchestrator():
    spec = importlib.util.spec_from_file_location(
        "gate_d_recovery_orchestrator", ORCHESTRATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _remote_manifest() -> dict[str, object]:
    ledger = json.loads((SOURCE_RUN / "diagnostic_objects.json").read_text())
    receipt = json.loads((SOURCE_RUN / "diagnostic_upload_receipt.json").read_text())
    records = [
        {**record, "uri": f"{recovery.SOURCE_REMOTE}/{record['path']}"}
        for record in ledger["objects"]
    ]
    terminal = receipt["terminal"]
    records.append(
        {
            **terminal,
            "uri": f"{recovery.SOURCE_REMOTE}/diagnostic_objects.json",
        }
    )
    return {
        "artifact_kind": "gate_d_compensated_pp16_source_remote_manifest",
        "exact_object_count": len(records),
        "objects": sorted(records, key=lambda item: item["path"]),
        "source_remote": recovery.SOURCE_REMOTE,
        "status": "SOURCE_DIAGNOSTIC_AUTHENTICATED",
    }


def _helper_escape() -> dict[str, object]:
    return {
        "bytes": recovery.SOURCE_HELPER_BYTES,
        "device": 2049,
        "inode": 1_565_814,
        "path": recovery.SOURCE_HELPER_PATH,
        "sha256": recovery.SOURCE_HELPER_SHA256,
    }


def test_recovery_policy_allows_only_the_exact_git_authenticated_helper() -> None:
    recovery._require_exact_helper_escape([], [_helper_escape()])
    second = {**_helper_escape(), "path": "/tmp/second.py"}
    with pytest.raises(BenchmarkValidationError, match="exactly one"):
        recovery._require_exact_helper_escape([], [_helper_escape(), second])
    hostile = {**_helper_escape(), "sha256": "0" * 64}
    with pytest.raises(BenchmarkValidationError, match="exactly one"):
        recovery._require_exact_helper_escape([], [hostile])
    with pytest.raises(BenchmarkValidationError, match="exactly one"):
        recovery._require_exact_helper_escape([_helper_escape()], [_helper_escape()])


@pytest.mark.parametrize(
    ("live", "versions", "deleted"),
    [(["x"], [], []), ([], ["x"], []), ([], [], ["x"])],
)
def test_recovery_prefix_requires_live_version_and_soft_delete_vacancy(
    live: list[str], versions: list[str], deleted: list[str]
) -> None:
    with pytest.raises(BenchmarkValidationError, match="historically vacant"):
        recovery.validate_recovery_remote_vacancy(live, versions, deleted)
    recovery.validate_recovery_remote_vacancy([], [], [])


def test_source_remote_manifest_requires_terminal_last_and_exact_generation() -> None:
    records, ledger_raw = recovery._validate_ledger(SOURCE_RUN)
    receipt = recovery._validate_diagnostic_receipt(SOURCE_RUN, ledger_raw)
    report = _remote_manifest()
    recovery._validate_remote_manifest(report, records, receipt)

    hostile = deepcopy(report)
    hostile["objects"][0]["generation"] = "1"  # type: ignore[index]
    with pytest.raises(BenchmarkValidationError, match="identity drifted"):
        recovery._validate_remote_manifest(hostile, records, receipt)

    hostile = deepcopy(report)
    terminal = next(
        item
        for item in hostile["objects"]  # type: ignore[union-attr]
        if item["path"] == "diagnostic_objects.json"
    )
    terminal["generation"] = "1"
    hostile_receipt = {**receipt, "generation": "1"}
    with pytest.raises(BenchmarkValidationError, match="terminal-last"):
        recovery._validate_remote_manifest(hostile, records, hostile_receipt)


@pytest.mark.skipif(not SOURCE_RUN.is_dir(), reason="protected source evidence absent")
def test_real_rejection_recomputes_offline_from_authenticated_outputs(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "source_remote_objects.json"
    manifest.write_bytes(recovery.canonical_json(_remote_manifest()))
    result = recovery.authenticate_compensated_pp16_rejection(
        SOURCE_RUN,
        worktree=ROOT,
        capsule=CAPSULE_ROOT / "capsule.json",
        capsule_inputs=CAPSULE_ROOT / "candidate-inputs.npz",
        capsule_state=CAPSULE_ROOT / "candidate-state.npz",
        capsule_execution_authority=CAPSULE_AUTHORITY,
        source_remote_manifest=manifest,
        verify_live_dependencies=False,
    )
    assert result["status"] == "NUMERICAL_REJECTED_RECOVERED"
    assert result["compiled_executable_invocation_count"] == 1
    assert result["historical_publisher_accepted"] is False
    assert result["tpu_rerun_performed"] is False
    assert result["gate_d_closed"] is False
    assert result["hlo"]["bridge_byte_replay_exact"] is True
    assert result["first_divergence"]["first_mismatching_watchpoint"] == (
        "layer1.normalized"
    )
    assert (
        result["first_divergence"]["derived_rms_input_from_bf16_operands_bit_exact"]
        is True
    )
    assert result["numerical_policy"]["tolerance"] is None
    numerical = result["numerical"]
    assert numerical["accepted_tpu_event_match"] is False
    assert numerical["contract_valid"] == [1, 1]
    assert numerical["tie_order_exact"] is True
    assert all(numerical["owner_agreement"].values())


def test_orchestrator_is_cpu_only_default_off_and_terminal_last() -> None:
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    assert "import jax" not in source
    assert 'os.environ.get("GLM_GATE_D_COMPENSATED_PP16_RECOVERY") != "1"' in source
    assert 'os.environ.get("JAX_PLATFORMS") != "cpu"' in source
    assert "os.O_EXCL" in source
    assert "--if-generation-match=0" in source
    assert "--all-versions" in source
    assert "--soft-deleted" in source
    assert source.index("_verify_remote_catalogue(remote, preterminal)") < source.index(
        "terminal_receipt = _upload_one(terminal_raw, terminal_name, terminal_remote)"
    )
    assert source.index(
        "store.write(remote_objects_name, remote_objects)"
    ) < source.index("store.write(terminal_name, terminal_raw)")
    assert '"preterminal_remote_ledger": dict(remote_objects_receipt)' in source
    assert '"storage",\n                        "rm"' not in source
    assert "os.pread" in source
    assert "dir_fd=self.run_fd" in source
    assert "class BoundLock" in source
    assert "lock.revalidate()" in source
    assert "remote prefix listing contains an unknown record" in source
    assert "recovery terminal was not the final mutation" in source
    assert "NUMERICAL_REJECTED_RECOVERED" in source
    assert "historical_publisher_accepted" in source


def test_remote_listing_rejects_unknown_or_extra_records(monkeypatch) -> None:
    orchestrator = _load_orchestrator()

    def response(payload: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess([], 0, json.dumps(payload).encode(), b"")

    monkeypatch.setattr(
        orchestrator, "_run", lambda *args, **kwargs: response([{"type": "future"}])
    )
    with pytest.raises(orchestrator.RecoveryError, match="unknown record"):
        orchestrator._list_prefix("gs://bucket/tag", soft_deleted=False)

    hostile = [
        {"type": "prefix", "url": "gs://bucket/tag/"},
        {
            "type": "cloud_object",
            "url": "gs://bucket/tag/x#1",
            "metadata": {},
            "ignored": True,
        },
    ]
    monkeypatch.setattr(orchestrator, "_run", lambda *args, **kwargs: response(hostile))
    with pytest.raises(orchestrator.RecoveryError, match="schema drifted"):
        orchestrator._list_prefix("gs://bucket/tag", soft_deleted=False)


def test_evidence_member_is_mode_and_directory_durable_before_use(monkeypatch) -> None:
    orchestrator = _load_orchestrator()
    store = orchestrator.EvidenceStore.__new__(orchestrator.EvidenceStore)
    store.members = {}
    store.run_fd = 7
    store.revalidate = lambda: None
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(orchestrator.os, "open", lambda *a, **k: 8)
    monkeypatch.setattr(
        orchestrator.os,
        "write",
        lambda descriptor, raw: calls.append(("write", descriptor, bytes(raw)))
        or len(raw),
    )
    monkeypatch.setattr(
        orchestrator.os,
        "fchmod",
        lambda descriptor, mode: calls.append(("fchmod", descriptor, mode)),
    )
    monkeypatch.setattr(
        orchestrator.os,
        "fsync",
        lambda descriptor: calls.append(("fsync", descriptor)),
    )

    store.write("proof.json", b"proof")
    assert calls == [
        ("write", 8, b"proof"),
        ("fchmod", 8, 0o400),
        ("fsync", 8),
        ("fsync", 7),
    ]
    source = ORCHESTRATOR.read_text(encoding="utf-8")
    mkdir_at = source.index("os.mkdir(tag, mode=0o700, dir_fd=self.root_fd)")
    root_fsync = source.index("os.fsync(self.root_fd)", mkdir_at)
    run_open = source.index("self.run_fd = os.open(", root_fsync)
    assert mkdir_at < root_fsync < run_open
