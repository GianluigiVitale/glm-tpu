"""Reconstruct the sealer's `validate` argv for a sealed run from its summary.

Used by the end-to-end sealer tests so that `_validate` is exercised against
real protected evidence rather than asserted about by reading its source.
"""
from __future__ import annotations

import json
from pathlib import Path

TOPOLOGY_ROOT = (
    "/home/gianl/gcs-models/results/greenfield_topology_20260826T194116460015528Z/host_records"
)
TOKEN_ORACLE = (
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/"
    "greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle"
)
DSA_ORACLE = (
    "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
    "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle"
)
SEALED_C512_RUN = Path(
    "/home/gianl/glm-run/greenfield_ws32_short_decoder_8k_numerical_c512_"
    "20260905T182725949766820Z"
)


def available(run_dir: Path = SEALED_C512_RUN) -> bool:
    return (
        (run_dir / "summary.json").is_file()
        and Path(TOPOLOGY_ROOT).is_dir()
        and (Path(TOKEN_ORACLE) / "manifest.json").is_file()
        and (Path(DSA_ORACLE) / "dsa_events.safetensors").is_file()
    )


def build(run_dir: Path, output: Path, repository_root: Path, **overrides: str) -> list[str]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    graphs = summary["graph_sha256"]
    argv = [
        "validate",
        "--run-dir", str(run_dir),
        "--topology-capture-root", TOPOLOGY_ROOT,
        "--token-oracle-dir", TOKEN_ORACLE,
        "--dsa-oracle-dir", DSA_ORACLE,
        "--mode", summary["mode"],
        "--context-label", summary["context_label"],
        "--tag", summary["run_tag"],
        "--code-hash", summary["code_hash"],
        "--checkpoint-manifest-sha256", summary["checkpoint_manifest_sha256"],
        "--checkpoint-success-sha256", summary["checkpoint_success_sha256"],
        "--token-oracle-manifest-sha256", summary["token_oracle_manifest_sha256"],
        "--dsa-oracle-manifest-sha256", summary["dsa_oracle_manifest_sha256"],
        "--token-oracle-success-sha256", summary["token_oracle_success_sha256"],
        "--dsa-oracle-success-sha256", summary["dsa_oracle_success_sha256"],
        "--dsa-association-summary-sha256", summary["dsa_association_summary_sha256"],
        "--dsa-association-success-sha256", summary["dsa_association_success_sha256"],
        "--topology-sha256", summary["topology_sha256"],
        "--topology-fleet-sha256", summary["topology_fleet_sha256"],
        "--mesh-sha256", summary["mesh_sha256"],
        "--source-inventory-sha256", summary["source_inventory_sha256"],
        "--context-capacity", str(summary["context_capacity"]),
        "--observer-steps", "14",
        "--warmup", "3",
        "--iterations", "20",
        "--trace-steps", "3",
        "--exact-dsa", "1" if summary["exact_dsa"] else "0",
        "--checkpoint-transport", summary["checkpoint_transport"],
        "--evidence-layout", "hlo_per_rank_v1",
        "--strategy-nd-dense", "1" if summary["strategy_nd_dense"] else "0",
        "--strategy-nd-dense-overlay-manifest-sha256",
        summary["strategy_nd_dense_overlay_manifest_sha256"],
        "--strategy-nd-dense-overlay-manifest-file-sha256",
        summary["strategy_nd_dense_overlay_manifest_file_sha256"],
        "--strategy-nd-dense-overlay-success-file-sha256",
        summary["strategy_nd_dense_overlay_success_file_sha256"],
        "--recovery-code-hash", summary.get("recovery_code_hash") or "",
        "--later-event-alarm-acknowledged",
        "1" if summary["later_event_alarm"]["acknowledged"] else "0",
        "--later-event-alarm-lessons-pin",
        summary["later_event_alarm"].get("lessons_pin", ""),
        "--prefill-chunk", "512",
        "--output", str(output),
    ]
    adjudication = summary.get("dsa_adjudication")
    if adjudication:
        argv += [
            "--dsa-adjudication-record", str(repository_root / adjudication["record_path"]),
            "--dsa-adjudication-sha256", adjudication["record_sha256"],
        ]
    for flag_name, key in (
        ("exact-materialize", "exact_materialize"),
        ("exact-promote", "exact_promote"),
        ("prefill-chunk", "prefill_chunk"),
        ("prefill-tail", "prefill_tail"),
        ("observer", "observer"),
        ("decode", "decode"),
        ("cache-probe", "cache_probe"),
    ):
        if key in graphs:
            argv += [
                f"--expected-{flag_name}-stablehlo-sha256", graphs[key]["stablehlo_sha256"],
                f"--expected-{flag_name}-optimized-hlo-sha256", graphs[key]["optimized_hlo_sha256"],
            ]
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if flag in argv:
            argv[argv.index(flag) + 1] = value
        else:
            argv += [flag, value]
    return argv
