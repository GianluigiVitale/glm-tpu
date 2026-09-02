"""Offline adjudication of the successful Gate-D PP16 projection HLO run.

This module imports no JAX and never lowers, compiles, or invokes an executable.
It binds the successful V3 compile-only run to exact local bytes, generation-
qualified same-region replay evidence, its terminal marker, and causal HLO.
"""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

from .gate_d_projection_contraction_hlo import (
    ProjectionContractionHloError,
    _read_regular,
    audit_optimized_hlo_structure,
    audit_stablehlo_structure,
)

EXPECTED_RUN_TAG = "gate_d_projection_contraction_pp16_hlo_20260901T213605719107105Z"
EXPECTED_RUN_DIR = Path("/home/gianl/gate-d-runs") / EXPECTED_RUN_TAG
EXPECTED_CODE_HASH = "d5da766f26b882ca47ba686745631ffc67e122b7"
EXPECTED_OPTIMIZED_HLO_SHA256 = (
    "817ba2ed87c33ec928f834fcf3a003ce63d3dedf4061a7c64fe354a26ac498ea"
)
EXPECTED_STABLEHLO_SHA256 = (
    "4b3fa252e837d381208453b06d7e369947fa4c6c58c795edc8ff5a8ea8c9e2e3"
)
EXPECTED_REMOTE_LEDGER_SHA256 = (
    "5924d3f248b20bb5bf6042e9145bb31feb96c314c49b98e7d6aae2c8f68edb25"
)
EXPECTED_REMOTE_LEDGER_GENERATION = "1788298701112785"
EXPECTED_TERMINAL_SHA256 = (
    "bbeca36401beaf7e3df585b6701b9c35d8477ef2b2ce6f60c1d524059f2dcffe"
)
EXPECTED_TERMINAL_GENERATION = "1788298702037876"
EXPECTED_REMOTE_PREFIX = (
    "gs://driftbench-dsv4-uc/results/greenfield/glm52/"
    "gate_d_projection_contraction_pp16_hlo/" + EXPECTED_RUN_TAG
)
EXPECTED_CLAIM_SCOPE = (
    "One abstract-input optimized-HLO acquisition on exact adjacent PP16 stage-zero TPU-v4 "
    "devices. The compiled executable was never invoked; HLO is unadjudicated and proves no "
    "numerical, performance or Gate-D claim."
)
EXPECTED_REMOTE_REPLAY = (
    Path(__file__).resolve().parents[3]
    / "docs/artifacts/gate-d-projection-contraction-pp16-hlo-success-remote-replay.json"
)
EXPECTED_REMOTE_REPLAY_SHA256 = (
    "9c809a939c93ab6b5081665d6c20598a1ed80c47dfd868c4e8f0e6be08767085"
)
EXPECTED_REMOTE_PATHS = {
    "HLO_ACQUIRED",
    "census_post.txt",
    "census_pre.txt",
    "dependencies.json",
    "evidence.json",
    "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt",
    "hlo/projection_contraction_pp16_stage0.stablehlo.mlir",
    "mirror.sha256",
    "orchestrator.sealed.log",
    "publisher_runtime.json",
    "remote_objects.json",
    "remote_vacancy.raw.txt",
    "remote_vacancy.txt",
    "runner.json",
    "runner.log",
    "summary.json",
    "sync.txt",
}
_RUNNER_KEYS = {
    "artifact_kind",
    "claim_scope",
    "code_hash",
    "compile_only",
    "compile_seconds",
    "compiled_executable_invocation_count",
    "compiler_dependency_manifest",
    "gate_d_closed",
    "hlo",
    "input_spec",
    "lowering_seconds",
    "memory_after_compile",
    "memory_analysis",
    "memory_before_compile",
    "numerical_claim",
    "output_spec",
    "performance_claim",
    "persistent_compilation_cache_enabled",
    "physical_group",
    "projection_contraction_source_authority",
    "projection_contraction_source_sha256",
    "runtime",
    "sealed_project_source",
    "status",
    "topology_authority_sha256",
    "tpu_numerical_execution_performed",
}
_SUMMARY_KEYS = {
    "artifact_kind",
    "claim_scope",
    "code_hash",
    "elapsed_seconds",
    "gate_d_closed",
    "hlo_acquisition_source_authority",
    "numerical_claim",
    "optimized_hlo_sha256",
    "performance_claim",
    "remote_prefix",
    "run_tag",
    "stablehlo_sha256",
    "status",
    "tpu_numerical_execution_performed",
}
_EVIDENCE_KEYS = {
    "artifact_kind",
    "code_hash",
    "files",
    "gate_d_closed",
    "numerical_claim",
    "performance_claim",
    "run_tag",
    "status",
}


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _records(items: Any, expected_paths: set[str]) -> dict[str, dict[str, Any]]:
    if not isinstance(items, list) or len(items) != len(expected_paths):
        raise ProjectionContractionHloError("success object inventory drifted")
    mapped = {item.get("path"): item for item in items if isinstance(item, dict)}
    if set(mapped) != expected_paths or any(
        set(item) != {"crc32c", "generation", "path", "sha256", "size"}
        or not isinstance(item["crc32c"], str)
        or not isinstance(item["generation"], str)
        or re.fullmatch(r"[0-9]+", item["generation"]) is None
        or not isinstance(item["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        or type(item["size"]) is not int
        or item["size"] < 0
        for item in mapped.values()
    ):
        raise ProjectionContractionHloError("success object record drifted")
    return mapped


def audit_success_remote_replay(raw: bytes) -> dict[str, dict[str, Any]]:
    """Validate the independently downloaded, generation-qualified object set."""

    if sha256(raw).hexdigest() != EXPECTED_REMOTE_REPLAY_SHA256:
        raise ProjectionContractionHloError("success remote replay hash drifted")
    replay = json.loads(raw)
    if (
        replay.get("artifact_kind")
        != "gate_d_projection_contraction_pp16_hlo_independent_success_remote_replay"
        or replay.get("bucket")
        != {"location": "US-CENTRAL2", "name": "driftbench-dsv4-uc"}
        or replay.get("prefix") != EXPECTED_REMOTE_PREFIX
        or replay.get("run_tag") != EXPECTED_RUN_TAG
        or replay.get("claims")
        != {
            "cloud_mutation_performed": False,
            "generation_downloads_verified": True,
            "read_only": True,
            "soft_deleted_query_exhaustive": True,
            "soft_deleted_query_matched_no_objects": True,
            "success_terminal_last": True,
        }
        or replay.get("all_versions_catalogue")
        != {
            "live_generation_count": 17,
            "soft_deleted_generation_count": 0,
            "unique_path_count": 17,
        }
        or replay.get("soft_deleted_result")
        != {
            "exit_code": 1,
            "stderr": "ERROR: (gcloud.storage.ls) One or more URLs matched no objects.\n",
            "stdout_bytes": 0,
        }
    ):
        raise ProjectionContractionHloError("success remote replay boundary drifted")
    mapped = _records(replay.get("objects"), EXPECTED_REMOTE_PATHS)
    terminal = mapped["HLO_ACQUIRED"]
    if (
        terminal["generation"] != EXPECTED_TERMINAL_GENERATION
        or terminal["sha256"] != EXPECTED_TERMINAL_SHA256
        or int(terminal["generation"])
        != max(int(item["generation"]) for item in mapped.values())
        or mapped["remote_objects.json"]["generation"]
        != EXPECTED_REMOTE_LEDGER_GENERATION
        or mapped["remote_objects.json"]["sha256"] != EXPECTED_REMOTE_LEDGER_SHA256
    ):
        raise ProjectionContractionHloError("success terminal ordering drifted")
    return mapped


def audit_success_marker(
    raw: bytes, remote_objects: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Validate the terminal's self-bound payload and remote-ledger authority."""

    if sha256(raw).hexdigest() != EXPECTED_TERMINAL_SHA256:
        raise ProjectionContractionHloError("success terminal hash drifted")
    marker = json.loads(raw)
    marker_payload_sha256 = marker.pop("marker_payload_sha256", None)
    if marker_payload_sha256 != sha256(_canonical(marker)).hexdigest() or marker != {
        "adjudicated": False,
        "artifact_kind": "gate_d_projection_contraction_pp16_hlo_acquired",
        "evidence_sha256": remote_objects["evidence.json"]["sha256"],
        "gate_d_closed": False,
        "numerical_claim": False,
        "performance_claim": False,
        "remote_ledger": remote_objects["remote_objects.json"],
        "run_tag": EXPECTED_RUN_TAG,
        "status": "HLO_ACQUIRED_UNADJUDICATED",
        "summary_sha256": remote_objects["summary.json"]["sha256"],
        "tpu_numerical_execution_performed": False,
    }:
        raise ProjectionContractionHloError("success terminal payload drifted")
    marker["marker_payload_sha256"] = marker_payload_sha256
    return marker


def audit_success_runner(runner: Any) -> None:
    """Reject any runner schema or claim outside the compile-only boundary."""

    expected_input_spec = [
        {"dtype": "bfloat16", "name": "normalized_hidden_bf16", "shape": [2, 1, 6144]},
        {"dtype": "float32", "name": "wk_weight_fp32", "shape": [2, 128, 6144]},
        {"dtype": "bfloat16", "name": "key_norm_weight_bf16", "shape": [2, 128]},
        {"dtype": "bfloat16", "name": "key_norm_bias_bf16", "shape": [2, 128]},
    ]
    expected_output_spec = [
        {
            "dtype": "bfloat16",
            "name": "normalized_hidden_owners",
            "shape": [2, 1, 6144],
        },
        {"dtype": "float32", "name": "projected_key_owners", "shape": [2, 1, 128]},
        {"dtype": "float32", "name": "current_key_owners", "shape": [2, 1, 128]},
    ]
    if (
        not isinstance(runner, dict)
        or set(runner) != _RUNNER_KEYS
        or runner.get("artifact_kind")
        != "gate_d_projection_contraction_pp16_optimized_hlo_acquisition"
        or runner.get("claim_scope") != EXPECTED_CLAIM_SCOPE
        or runner.get("code_hash") != EXPECTED_CODE_HASH
        or runner.get("status") != "HLO_ACQUIRED_UNADJUDICATED"
        or runner.get("compile_only") is not True
        or runner.get("compiled_executable_invocation_count") != 0
        or runner.get("tpu_numerical_execution_performed") is not False
        or runner.get("gate_d_closed") is not False
        or runner.get("numerical_claim") is not False
        or runner.get("performance_claim") is not False
        or runner.get("input_spec") != expected_input_spec
        or runner.get("output_spec") != expected_output_spec
        or runner.get("physical_group")
        != {
            "coordinates": [[0, 0, 0], [1, 0, 0]],
            "device_ids": [0, 1],
            "local_device_count_visible": 4,
            "mesh_device_count": 2,
            "process_index": 0,
            "stage_id": 0,
        }
        or runner.get("hlo", {}).get("optimized", {}).get("sha256")
        != EXPECTED_OPTIMIZED_HLO_SHA256
        or runner.get("hlo", {}).get("stablehlo", {}).get("sha256")
        != EXPECTED_STABLEHLO_SHA256
        or runner.get("hlo", {}).get("surface", {}).get("collective_count") != 0
        or runner.get("projection_contraction_source_sha256")
        != "5744eee0ef2cf35a4566cc0de165be1338160daaae3f4dd133554b2aa8280e9f"
        or runner.get("topology_authority_sha256")
        != "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
    ):
        raise ProjectionContractionHloError("successful runner boundary drifted")


def audit_success_summary(summary: Any) -> None:
    """Reject any summary schema or claim outside the compile-only boundary."""

    if (
        not isinstance(summary, dict)
        or set(summary) != _SUMMARY_KEYS
        or summary.get("artifact_kind")
        != "gate_d_projection_contraction_pp16_hlo_acquisition_summary"
        or summary.get("claim_scope") != EXPECTED_CLAIM_SCOPE
        or summary.get("code_hash") != EXPECTED_CODE_HASH
        or summary.get("run_tag") != EXPECTED_RUN_TAG
        or summary.get("remote_prefix") != EXPECTED_REMOTE_PREFIX
        or summary.get("status") != "HLO_ACQUIRED_UNADJUDICATED"
        or summary.get("optimized_hlo_sha256") != EXPECTED_OPTIMIZED_HLO_SHA256
        or summary.get("stablehlo_sha256") != EXPECTED_STABLEHLO_SHA256
        or summary.get("tpu_numerical_execution_performed") is not False
        or summary.get("gate_d_closed") is not False
        or summary.get("numerical_claim") is not False
        or summary.get("performance_claim") is not False
    ):
        raise ProjectionContractionHloError("successful summary boundary drifted")


def audit_success_evidence(
    evidence: Any,
    remote: dict[str, dict[str, Any]],
    evidence_paths: set[str],
) -> None:
    """Reject contradictory claims, duplicate paths, or incomplete evidence records."""

    if (
        not isinstance(evidence, dict)
        or set(evidence) != _EVIDENCE_KEYS
        or evidence.get("artifact_kind")
        != "gate_d_projection_contraction_pp16_hlo_local_evidence"
        or evidence.get("code_hash") != EXPECTED_CODE_HASH
        or evidence.get("run_tag") != EXPECTED_RUN_TAG
        or evidence.get("status") != "HLO_ACQUIRED_UNADJUDICATED"
        or evidence.get("gate_d_closed") is not False
        or evidence.get("numerical_claim") is not False
        or evidence.get("performance_claim") is not False
    ):
        raise ProjectionContractionHloError("successful evidence boundary drifted")
    items = evidence.get("files")
    if not isinstance(items, list) or len(items) != len(evidence_paths):
        raise ProjectionContractionHloError("successful evidence inventory drifted")
    mapped: dict[str, dict[str, Any]] = {}
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"byte_count", "path", "sha256"}
            or not isinstance(item.get("path"), str)
            or item["path"] in mapped
        ):
            raise ProjectionContractionHloError("successful evidence record drifted")
        mapped[item["path"]] = item
    if set(mapped) != evidence_paths or any(
        mapped[path]
        != {
            "byte_count": remote[path]["size"],
            "path": path,
            "sha256": remote[path]["sha256"],
        }
        for path in evidence_paths
    ):
        raise ProjectionContractionHloError("successful evidence files drifted")


def adjudicate_success_run(run_dir: Path = EXPECTED_RUN_DIR) -> dict[str, Any]:
    """Adjudicate V3 persistence and HLO structure without numerical authority."""

    if run_dir != EXPECTED_RUN_DIR:
        raise ProjectionContractionHloError("unexpected successful run directory")
    replay_raw = _read_regular(EXPECTED_REMOTE_REPLAY, 1 << 20)
    remote = audit_success_remote_replay(replay_raw)
    local = {
        path: _read_regular(run_dir / path, 1 << 20) for path in EXPECTED_REMOTE_PATHS
    }
    if any(
        sha256(local[path]).hexdigest() != remote[path]["sha256"]
        or len(local[path]) != remote[path]["size"]
        for path in EXPECTED_REMOTE_PATHS
    ):
        raise ProjectionContractionHloError("local/remote successful bytes disagree")

    ledger = json.loads(local["remote_objects.json"])
    ledger_paths = EXPECTED_REMOTE_PATHS - {"HLO_ACQUIRED", "remote_objects.json"}
    ledger_objects = _records(ledger.get("objects"), ledger_paths)
    if (
        ledger.get("artifact_kind")
        != "gate_d_projection_contraction_pp16_hlo_remote_ledger"
        or ledger.get("run_tag") != EXPECTED_RUN_TAG
        or any(ledger_objects[path] != remote[path] for path in ledger_paths)
    ):
        raise ProjectionContractionHloError("successful remote ledger drifted")
    audit_success_marker(local["HLO_ACQUIRED"], remote)

    receipt = json.loads(
        _read_regular(run_dir / "terminal_upload_receipt.json", 1 << 20)
    )
    if receipt != {
        "artifact_kind": "gate_d_projection_contraction_pp16_hlo_terminal_receipt",
        "remote": EXPECTED_REMOTE_PREFIX + "/HLO_ACQUIRED",
        "terminal": remote["HLO_ACQUIRED"],
    }:
        raise ProjectionContractionHloError("success terminal receipt drifted")

    audit_success_runner(json.loads(local["runner.json"]))
    summary = json.loads(local["summary.json"])
    evidence = json.loads(local["evidence.json"])
    evidence_paths = ledger_paths - {"evidence.json"}
    audit_success_summary(summary)
    audit_success_evidence(evidence, remote, evidence_paths)

    optimized = audit_optimized_hlo_structure(
        local["hlo/projection_contraction_pp16_stage0.optimized_hlo.txt"].decode(
            "ascii"
        )
    )
    stable = audit_stablehlo_structure(
        local["hlo/projection_contraction_pp16_stage0.stablehlo.mlir"].decode("ascii")
    )
    if optimized["live_rows_per_owner"] != stable["live_rows_per_owner"]:
        raise ProjectionContractionHloError(
            "successful HLO live-row contracts disagree"
        )
    return {
        "artifact_kind": "gate_d_projection_contraction_pp16_success_hlo_adjudication",
        "authorization": {
            "full_8k": False,
            "numerical_execution": False,
            "performance_claim": False,
            "persistence_only": True,
        },
        "classification": (
            "HLO_CAUSAL_STRUCTURE_ACCEPTED;PP16_OWNER_LOCALITY_ACCEPTED;"
            "TPU_NUMERICAL_UNPROVEN;HLO_ACQUIRED_TERMINAL_VERIFIED;GATE_D_OPEN"
        ),
        "code_hash": EXPECTED_CODE_HASH,
        "gate_d_closed": False,
        "optimized_hlo": {
            "sha256": EXPECTED_OPTIMIZED_HLO_SHA256,
            "structure": optimized,
        },
        "remote_replay": {
            "all_versions_live_generation_count": len(remote),
            "generation_download_count": len(remote),
            "sha256": EXPECTED_REMOTE_REPLAY_SHA256,
            "soft_deleted_generation_count": 0,
            "terminal_last": True,
        },
        "run_tag": EXPECTED_RUN_TAG,
        "schema_version": 1,
        "stablehlo": {"sha256": EXPECTED_STABLEHLO_SHA256, "structure": stable},
        "terminal": {
            "generation": EXPECTED_TERMINAL_GENERATION,
            "sha256": EXPECTED_TERMINAL_SHA256,
        },
    }
