from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / (
    "docs/artifacts/gate-d-forced-round-pp16-hlo-pre-census-busy-failure.json"
)


def test_pre_census_busy_failure_evidence_is_fail_closed() -> None:
    evidence = json.loads(ARTIFACT.read_text())

    assert evidence["artifact_kind"] == (
        "gate_d_forced_round_pp16_hlo_pre_census_busy_failure"
    )
    assert evidence["classification"] == (
        "PRE_DRIVER_CENSUS_CONTENTION;PROTECTED_WORKFLOW_NO_JAX_OR_HLO;"
        "PROTECTED_WORKFLOW_NO_TPU_COMPILE_OR_EXECUTION;GATE_D_OPEN"
    )
    assert evidence["authorization"] == {
        "cloud_write": False,
        "full_8k": False,
        "hlo_acquisition": False,
        "numerical_execution": False,
        "persistence_only": True,
        "tpu_compile": False,
        "tpu_execution": False,
    }
    assert evidence["failure"] == {
        "driver_started": False,
        "failure_stage": "strict_census.pre",
        "hlo_acquired_terminal_present": False,
        "hlo_directory_empty": True,
        "jax_imported_by_protected_workflow": False,
        "launcher_started_at_utc": "2026-09-01T13:46:50Z",
        "reason": (
            "worker 0 reported CENSUS_BUSY while workers 1-7 reported CENSUS_OK"
        ),
        "success_terminal_present": False,
    }
    assert evidence["no_retry_same_tag"] is True


def test_diagnostic_archive_is_generation_bound_and_terminal_last() -> None:
    archive = json.loads(ARTIFACT.read_text())["diagnostic_archive"]
    objects = archive["objects"]
    expected_paths = {
        "census_failure_exit.txt",
        "census_pre.txt",
        "diagnostic_objects.json",
        "failure_status.json",
        "mirror.sha256",
        "orchestrator.failure.log",
        "publisher_runtime.json",
        "remote_vacancy.raw.txt",
        "remote_vacancy.txt",
        "sync.txt",
    }

    assert archive["bucket_location"] == "US-CENTRAL2"
    assert archive["terminal_last"] is True
    assert len(objects) == 10
    assert {item["path"] for item in objects} == expected_paths
    assert len({item["generation"] for item in objects}) == len(objects)
    for item in objects:
        assert item["generation"].isdigit()
        assert item["size"] > 0
        assert len(item["sha256"]) == 64
        int(item["sha256"], 16)
        assert item["crc32c"]

    terminal = next(
        item for item in objects if item["path"] == "diagnostic_objects.json"
    )
    assert terminal["generation"] == archive["terminal_generation"]
    assert int(terminal["generation"]) > max(
        int(item["generation"]) for item in objects if item is not terminal
    )


def test_holder_attribution_and_post_failure_audit_are_bounded() -> None:
    evidence = json.loads(ARTIFACT.read_text())
    holder = evidence["pre_existing_holder"]
    audit = evidence["post_failure_audit"]

    assert holder["origin"] == "unknown"
    assert holder["may_have_initialized_tpu_backend"] is True
    assert holder["predated_protected_launcher"] is True
    assert holder["directly_satisfied_libtpu_holder_branch"] is True
    assert holder["exited_without_intervention"] is True
    assert holder["pid"] == 304691
    assert holder["worker"] == "t1v-n-ae271d05-w-0"
    assert evidence["scope_note"] == (
        "No-JAX, no-HLO and no-TPU-work claims apply only to the protected "
        "workflow. The unrelated holder imported JAX, called jax.devices(), may "
        "have initialized the TPU backend and has unknown provenance."
    )
    assert audit["pid_304691_absent"] is True
    assert audit["libtpu_holders_worker_0"] == []
    assert audit["pod_state"] == "READY"
    assert audit["pod_health"] == "HEALTHY"
    assert audit["bucket_location"] == "US-CENTRAL2"
    assert all(
        audit[key]
        for key in (
            "root_pod_lease_free",
            "root_rsync_lease_free",
            "user_pod_lease_free",
            "user_rsync_lease_free",
        )
    )
    assert audit["unique_clean_hosts"] == [
        f"t1v-n-ae271d05-w-{worker}" for worker in range(8)
    ]
