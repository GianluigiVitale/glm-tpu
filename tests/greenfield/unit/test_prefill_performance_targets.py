"""Registration arithmetic/provenance only, never proof targets are feasible."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def test_registered_targets_preserve_owner_objective_and_actual_baseline_pins():
    targets = json.loads(
        (ROOT / "configs/prefill-performance-targets-v1.json").read_text()
    )
    assert targets["requested_final_prompt_tokens_per_second"] == 10000
    assert targets["intermediate_prompt_tokens_per_second"] == 500
    assert targets["intermediate_is_completion"] is False
    assert targets["final_target_feasibility_proven"] is False
    for name, expected in (("l7_each_depth", 127363), ("l8_e0", 262144)):
        workload = targets["workloads"][name]
        assert workload["prompt_tokens"] == expected
        for prefix, rate in (("final", 10000), ("intermediate", 500)):
            total = workload[f"{prefix}_request_prefill_seconds"]
            assert total == pytest.approx(expected / rate)
            assert sum(
                workload[f"{prefix}_component_seconds"].values()
            ) == pytest.approx(total)
            assert workload[f"{prefix}_warm_local_ttft_seconds"] == pytest.approx(
                total + 1
            )
    for run, receipt, field in (
        ("597", "prefill-paired-short-sealed-20260909.json", "run_code_hash"),
        ("598", "prefill-missing-budget-sealed-20260909.json", "code_hash"),
    ):
        original = json.loads((ROOT / "docs/artifacts" / receipt).read_text())
        assert targets["baseline_runtime_pins"][run] == original[field]
