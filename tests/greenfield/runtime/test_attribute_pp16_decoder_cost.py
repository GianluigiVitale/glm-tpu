from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import pytest

from scripts.greenfield.attribute_pp16_decoder_cost import attribute


def _write_db(path: Path) -> None:
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE runs (
          run_id INTEGER PRIMARY KEY, model TEXT, harness_git TEXT
        );
        CREATE TABLE items (
          run_id INTEGER, item_id TEXT, correct INTEGER,
          latency_ms REAL, raw_output TEXT
        );
        """
    )
    rows = (
        (558, "pp16-moe", "moe16", "normal", 3.0, "{}"),
        (482, "pp8-moe", "moe8", "normal", 2.0, "{}"),
        (
            560,
            "attention",
            "attention",
            "pp16_lp2_2k_balanced_owner512",
            0.4,
            json.dumps(
                {"reference_latency": {"balanced_owner0": {"p50_ms": 0.25}}}
            ),
        ),
        (
            561,
            "query",
            "query",
            "layer0_position8155_lp2_tuple4_chunk2",
            0.5,
            "{}",
        ),
    )
    for run_id, model, git, item, latency, raw in rows:
        db.execute("INSERT INTO runs VALUES (?, ?, ?)", (run_id, model, git))
        db.execute(
            "INSERT INTO items VALUES (?, ?, 1, ?, ?)",
            (run_id, item, latency, raw),
        )
    db.commit()
    db.close()


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "plan_id": "PP16_LP2",
                "diagnostic_only": True,
                "performance_claim": False,
                "numerical_claim": False,
                "gate_d_passed": False,
                "results_db_run_id": None,
                "maximum_diagnostic_step_ms": 400.0,
            }
        )
    )
    hlo = tmp_path / "contract.json"
    hlo.write_text(
        json.dumps(
            {
                "passed": True,
                "backend_contract": "tpu_v4_pp16_pallas_feature_linear",
                "layer_count": 78,
                "pallas_feature_contract": {
                    "kernel_counts": {"a": 75, "b": 75, "c": 75}
                },
                "pallas_stage_linear_contract": {
                    "kernel_counts": {"d": 10, "e": 20}
                },
                "dsa_query_association_contract": {
                    "tuple4_reduction_fusion_count": 21,
                    "expected_tuple4_reduction_fusion_count": 21,
                    "expected_runtime_tuple4_reduction_count": 42,
                },
            }
        )
    )
    db = tmp_path / "results.db"
    _write_db(db)
    return summary, hlo, db


def test_offline_attribution_selects_moe_discriminator(tmp_path: Path) -> None:
    summary, hlo, db = _inputs(tmp_path)
    report = attribute(
        summary_path=summary,
        hlo_contract_path=hlo,
        results_db=db,
        pp16_moe_run_id=558,
        pp8_moe_run_id=482,
        attention_run_id=560,
        exact_query_run_id=561,
    )
    assert report["structural_counts"] == {
        "attention_layers": 78,
        "exact_query_layers": 21,
        "exact_query_runtime_tuple4_reductions": 42,
        "moe_layers": 75,
        "named_pallas_custom_calls": 255,
    }
    assert report["projection"]["moe_ms"] == 225.0
    assert report["projection"]["attention_ms"] == 19.5
    assert report["projection"]["exact_query_ms"] == 10.5
    assert report["projection"]["accounted_ms"] == 255.0
    assert report["projection"]["unaccounted_ms"] == 145.0
    assert report["projection"]["projection_inputs_ms"] == {
        "exact_query_item_p50": 0.5,
        "pp16_moe_item_p50": 3.0,
        "pp8_moe_comparator_item_p50": 2.0,
        "reference_attention_p50": 0.25,
    }
    assert report["sources"]["attention"]["item_latency_ms"] == 0.4
    assert "latency_ms" not in report["sources"]["attention"]
    assert (
        report["projection"][
            "arithmetic_step_if_pp16_moe_equaled_pp8_comparator_ms"
        ]
        == 325.0
    )
    assert report["pp8_comparator_caveat"]["causal_tile_prediction"] is False
    assert report["next_discriminator"]["change"].endswith("128 -> 256 only")
    assert "isolates only the output tile" in report["next_discriminator"]["reason"]
    assert report["performance_claim"] is False
    assert report["model_workload_rerun"] is False


def test_offline_attribution_rejects_nonterminal_summary(tmp_path: Path) -> None:
    summary, hlo, db = _inputs(tmp_path)
    value = json.loads(summary.read_text())
    value["status"] = "FAILED"
    summary.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="diagnostic identity"):
        attribute(
            summary_path=summary,
            hlo_contract_path=hlo,
            results_db=db,
            pp16_moe_run_id=558,
            pp8_moe_run_id=482,
            attention_run_id=560,
            exact_query_run_id=561,
        )
