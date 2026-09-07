"""Diagnostic captures never turn arithmetic mismatches into admission passes."""

from hashlib import sha256
import copy

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield import prefill_moe_boundaries as boundaries
from scripts.greenfield import ws32_prefill_moe_campaign as campaign
from scripts.greenfield.probe_ws32_prefill_moe import BOUNDARY_PROTOCOL
from tests.greenfield.hlo.test_prefill_real_moe_admission import (
    valid_workers,
    valid_hlo,
)


def test_boundary_comparison_records_bit_drift_without_passing_arithmetic():
    a = np.ones((17, 3), dtype=ml_dtypes.bfloat16)
    b = a.copy()
    b[0, 0] = 1.125
    record = boundaries.compare_arrays(a, b)
    assert record["bit_mismatches"] == 1 and record["max_abs"] == 0.125
    assert "passed" not in record
    b[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        boundaries.compare_arrays(a, b)


def test_boundary_npz_replay_detects_mutation(tmp_path):
    arrays, comparisons = {}, {}
    for name in boundaries.BOUNDARIES:
        dtype = np.float32 if name in boundaries.F32 else ml_dtypes.bfloat16
        a = np.ones((17, 2), dtype=dtype)
        b = a.copy()
        b[0, 0] = 1.125
        comparisons[name + "_0"] = boundaries.compare_arrays(a, b)
        arrays["candidate_" + name + "_0"] = (
            a if name in boundaries.F32 else a.view(np.uint16)
        )
        arrays["reference_" + name + "_0"] = (
            b if name in boundaries.F32 else b.view(np.uint16)
        )
    path = tmp_path / "boundaries.npz"
    np.savez_compressed(path, **arrays)
    record = dict(
        local_device_slots=[dict(device_id=0)],
        boundaries=dict(
            comparisons=comparisons, npz_sha256=sha256(path.read_bytes()).hexdigest()
        ),
    )
    boundaries.validate_boundaries(path, record)
    record["boundaries"]["comparisons"]["routed_0"]["bit_mismatches"] = 0
    with pytest.raises(ValueError, match="comparison"):
        boundaries.validate_boundaries(path, record)


def test_boundary_workflow_is_not_an_arithmetic_admission():
    records = valid_workers()
    for r in records:
        r.update(
            protocol=BOUNDARY_PROTOCOL, boundary_diagnostic=True, admission_only=False
        )
        del r["cases"]["concentrated"]
        for s in r["cases"]["normal"]["shards"]:
            s.update(finite_and_healthy=True, passed=False, bit_mismatches=3)
    campaign.validate_workers(records, "b" * 40, boundary=True)
    with pytest.raises(ValueError):
        campaign.validate_workers(records, "b" * 40)
    record = dict(
        status="SUCCESS",
        code_hash="b" * 40,
        kernel=campaign.BOUNDARY_KERNEL,
        protocol=BOUNDARY_PROTOCOL,
        boundary_diagnostic=True,
        admission_only=False,
        performance_claim=False,
        baseline_only=False,
        diagnostic_only=True,
        latency=None,
        warmup=0,
        iterations=0,
        profiler_free_timing=False,
        workers=records,
    )
    campaign.validate_record(record, "b" * 40, boundary=True)
    changed = copy.deepcopy(record)
    changed["admission_only"] = True
    with pytest.raises(ValueError):
        campaign.validate_record(changed, "b" * 40, boundary=True)
    files = campaign.evidence_files(
        "greenfield_fp8_ws32_prefill_moe_boundary_diagnostic_test"
    )
    assert (
        "boundaries.npz" in files
        and "normal.npz" in files
        and "concentrated.npz" not in files
    )


def test_lowering_manifest_is_review_evidence_not_equivalence_proof():
    hlo = valid_hlo()
    record = boundaries.lowering_manifest(hlo)
    assert record["hlo_sha256"] == sha256(hlo.encode()).hexdigest()
    assert len(record["collectives"]) == 2
    assert record["original_lowering_equivalent"] == "REQUIRES_REVIEW"
