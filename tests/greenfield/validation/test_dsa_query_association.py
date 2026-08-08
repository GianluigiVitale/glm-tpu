from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / (
    "scripts/greenfield/probe_layer0_dsa_query_association.py"
)
WRAPPER = REPO_ROOT / (
    "scripts/greenfield/run_layer0_dsa_query_association_probe.sh"
)
SPEC = importlib.util.spec_from_file_location("dsa_query_association", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_query_comparison_records_exact_and_divergent_values() -> None:
    actual = np.arange(12, dtype=np.float32).reshape(3, 4)
    exact = subject._compare(actual, actual.copy())
    assert exact["elementwise_exact"]
    assert exact["mismatch_count"] == 0
    candidate = actual.copy()
    candidate[1, 2] += np.float32(0.25)
    divergent = subject._compare(actual, candidate)
    assert not divergent["elementwise_exact"]
    assert divergent["mismatch_count"] == 1
    assert divergent["max_abs"] == 0.25


def test_query_candidate_matrix_covers_legacy_and_production_shapes() -> None:
    assert set(subject._candidate_functions()) == {
        "global_m32_n4096",
        "global_m1_n4096",
        "head_lax_map_m32_n128",
        "head_vmap_m32_n128",
        "head_unrolled_m32_n128",
        "head_fori_m32_n128",
        "head_lax_map_highest_m32_n128",
        "head_lax_map_m1_n128",
        "head_unrolled_m1_n128",
        "lp4_unrolled_m32_n1024",
        "lp4_unrolled_m1_n1024",
    }


def test_protected_query_wrapper_is_bounded_and_fail_closed() -> None:
    source = WRAPPER.read_text()
    for required in (
        "CAPTURE_SUCCESS_SHA=20deb19c",
        "CAPTURE_COMPARISON_SHA=623818dc",
        "CAPTURE_TENSORS_SHA=0a724bad",
        "TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1",
        "TPU_PROCESS_BOUNDS=1,1,1",
        "TPU_VISIBLE_DEVICES=0,1,2,3",
        "timeout --signal=TERM --kill-after=60 1800",
        "strict_census pre",
        "strict_census post",
        "results_ckpt.db",
        "remote_objects.json",
        "performance_claim",
    ):
        assert required in source
    for forbidden in ("launch_glm_32chip.sh", "glm_longctx.py"):
        assert forbidden not in source
