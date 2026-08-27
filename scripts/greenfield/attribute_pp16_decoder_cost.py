#!/usr/bin/env python3
"""Attribute a sealed PP16 diagnostic step from protected bounded results.

This is deliberately offline.  It reads one immutable decoder summary/HLO
contract and exact rows from the protected results DB; it never imports JAX or
initializes TPU.  The projection is a prioritization aid, not a performance
claim, because independently synchronized microbenchmarks are not additive.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _one_item(
    db: sqlite3.Connection, run_id: int, item_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    run = db.execute(
        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
    ).fetchall()
    item = db.execute(
        "SELECT * FROM items WHERE run_id = ? AND item_id = ?",
        (run_id, item_id),
    ).fetchall()
    if len(run) != 1 or len(item) != 1:
        raise ValueError(
            f"expected one protected DB row for run={run_id} item={item_id}"
        )
    run_record = dict(run[0])
    item_record = dict(item[0])
    if item_record.get("correct") != 1 or item_record.get("latency_ms") is None:
        raise ValueError(f"protected DB result is not accepted: run={run_id}")
    return run_record, item_record


def attribute(
    *,
    summary_path: Path,
    hlo_contract_path: Path,
    results_db: Path,
    pp16_moe_run_id: int,
    pp8_moe_run_id: int,
    attention_run_id: int,
    exact_query_run_id: int,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    hlo = json.loads(hlo_contract_path.read_text())
    if (
        summary.get("status") != "SUCCESS"
        or summary.get("plan_id") != "PP16_LP2"
        or summary.get("diagnostic_only") is not True
        or summary.get("performance_claim") is not False
        or summary.get("numerical_claim") is not False
        or summary.get("gate_d_passed") is not False
        or summary.get("results_db_run_id") is not None
        or hlo.get("passed") is not True
        or hlo.get("backend_contract")
        != "tpu_v4_pp16_pallas_feature_linear"
    ):
        raise ValueError("sealed PP16 diagnostic identity failed")

    feature_counts = hlo["pallas_feature_contract"]["kernel_counts"]
    if len(feature_counts) != 3 or len(set(feature_counts.values())) != 1:
        raise ValueError("PP16 feature-MoE kernel cardinality drifted")
    moe_layers = next(iter(feature_counts.values()))
    attention_layers = int(hlo["layer_count"])
    query_contract = hlo["dsa_query_association_contract"]
    query_layers = int(query_contract["tuple4_reduction_fusion_count"])
    if (
        query_layers
        != int(query_contract["expected_tuple4_reduction_fusion_count"])
        or int(query_contract["expected_runtime_tuple4_reduction_count"])
        != 2 * query_layers
    ):
        raise ValueError("exact-query runtime cardinality drifted")

    uri = f"file:{results_db}?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    try:
        pp16_run, pp16_item = _one_item(db, pp16_moe_run_id, "normal")
        pp8_run, pp8_item = _one_item(db, pp8_moe_run_id, "normal")
        attention_run, attention_item = _one_item(
            db, attention_run_id, "pp16_lp2_2k_balanced_owner512"
        )
        query_run, query_item = _one_item(
            db, exact_query_run_id, "layer0_position8155_lp2_tuple4_chunk2"
        )
    finally:
        db.close()

    attention_raw = json.loads(attention_item["raw_output"])
    reference_attention_ms = float(
        attention_raw["reference_latency"]["balanced_owner0"]["p50_ms"]
    )
    pp16_moe_ms = float(pp16_item["latency_ms"])
    pp8_moe_ms = float(pp8_item["latency_ms"])
    query_ms = float(query_item["latency_ms"])
    step_ms = float(summary["maximum_diagnostic_step_ms"])

    moe_projection = moe_layers * pp16_moe_ms
    attention_projection = attention_layers * reference_attention_ms
    query_projection = query_layers * query_ms
    accounted = moe_projection + attention_projection + query_projection
    pp8_moe_projection = moe_layers * pp8_moe_ms
    pp8_delta = moe_projection - pp8_moe_projection

    def source(run: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        return {
            "harness_git": run["harness_git"],
            "item_id": item["item_id"],
            "item_latency_ms": float(item["latency_ms"]),
            "model": run["model"],
            "run_id": int(run["run_id"]),
        }

    report: dict[str, Any] = {
        "artifact_kind": "greenfield_pp16_decoder_offline_cost_attribution",
        "diagnostic_only": True,
        "gate_d_passed": False,
        "hlo_contract_sha256": _sha256(hlo_contract_path),
        "model_workload_rerun": False,
        "numerical_claim": False,
        "performance_claim": False,
        "results_db_sha256": _sha256(results_db),
        "source_summary_sha256": _sha256(summary_path),
        "sources": {
            "attention": source(attention_run, attention_item),
            "exact_query": source(query_run, query_item),
            "pp16_moe": source(pp16_run, pp16_item),
            "pp8_moe_comparator": source(pp8_run, pp8_item),
        },
        "structural_counts": {
            "attention_layers": attention_layers,
            "exact_query_layers": query_layers,
            "exact_query_runtime_tuple4_reductions": int(
                query_contract["expected_runtime_tuple4_reduction_count"]
            ),
            "moe_layers": moe_layers,
            "named_pallas_custom_calls": sum(feature_counts.values())
            + sum(hlo["pallas_stage_linear_contract"]["kernel_counts"].values()),
        },
        "projection": {
            "accounted_ms": accounted,
            "accounted_share_of_step": accounted / step_ms,
            "attention_ms": attention_projection,
            "arithmetic_step_if_pp16_moe_equaled_pp8_comparator_ms": (
                step_ms - pp8_delta
            ),
            "exact_query_ms": query_projection,
            "moe_ms": moe_projection,
            "moe_share_of_step": moe_projection / step_ms,
            "pp16_step_ms": step_ms,
            "projection_inputs_ms": {
                "exact_query_item_p50": query_ms,
                "pp16_moe_item_p50": pp16_moe_ms,
                "pp8_moe_comparator_item_p50": pp8_moe_ms,
                "reference_attention_p50": reference_attention_ms,
            },
            "unaccounted_ms": step_ms - accounted,
        },
        "pp8_comparator_caveat": {
            "causal_tile_prediction": False,
            "differences": [
                "PP8_LP4 versus PP16_LP2 plan",
                "output tile 256 versus 128",
                "feature_reconstruct_down_fp32 true versus false",
            ],
            "purpose": "priority bound only; not an isolated output-tile estimate",
        },
        "next_discriminator": {
            "change": "PP16 feature-MoE output tile 128 -> 256 only",
            "reason": "MoE is the largest measured term; this PP16 run holds the plan and reconstruction mode fixed and isolates only the output tile.",
            "scope": "one real layer, two adjacent chips, exact routes/output, HLO, XPlane, HBM, warmed profiler-free distribution",
        },
        "warning": "Independent synchronized component timings are non-additive projections and are not decoder performance evidence.",
    }
    report["report_sha256"] = sha256(
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--hlo-contract", type=Path, required=True)
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pp16-moe-run-id", type=int, default=558)
    parser.add_argument("--pp8-moe-run-id", type=int, default=482)
    parser.add_argument("--attention-run-id", type=int, default=560)
    parser.add_argument("--exact-query-run-id", type=int, default=561)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"attribution output is append-only: {args.output}")
    report = attribute(
        summary_path=args.summary,
        hlo_contract_path=args.hlo_contract,
        results_db=args.results_db,
        pp16_moe_run_id=args.pp16_moe_run_id,
        pp8_moe_run_id=args.pp8_moe_run_id,
        attention_run_id=args.attention_run_id,
        exact_query_run_id=args.exact_query_run_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "PP16_OFFLINE_ATTRIBUTION "
        f"moe_share={report['projection']['moe_share_of_step']:.6f} "
        f"report={report['report_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
