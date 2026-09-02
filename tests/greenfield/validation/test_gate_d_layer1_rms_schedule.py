from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.validation.gate_d_layer1_rms_schedule import (
    ACCEPTED_LAYER1_NORMALIZED_SHA256,
    DB548_LAYER1_NORMALIZED_SHA256,
    _array_sha256,
    audit_layer1_rms_schedule_optimized_hlo,
    classify_layer1_rms_schedule_outputs,
)

CAPTURE = Path(
    "/home/gianl/glm-run/greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
    "dense_partial_capture.npz"
)
ENVELOPE = Path(
    "/home/gianl/glm-run/greenfield_layer0_dense_envelope_cross_layer_20260813T120703034434907Z/"
    "dense_envelope_cross_layer.npz"
)
SPLIT_HLO = Path(
    os.environ.get(
        "GLM_GREENFIELD_CAPTURED_RMS_ACCEPTED_SPLIT_HLO",
        "/home/gianl/gcs-models/results/greenfield_layer0_captured_rms_replay_20260813T214428668951467Z/"
        "hlo/accepted_split.optimized_hlo.txt",
    )
)


def _rows() -> tuple[np.ndarray, np.ndarray]:
    with np.load(CAPTURE, allow_pickle=False) as payload:
        accepted = np.ascontiguousarray(payload["accepted_layer1_normalized_bfloat16_bits"])
    with np.load(ENVELOPE, allow_pickle=False) as payload:
        db548 = np.ascontiguousarray(payload["layer1_normalized_bfloat16_bits"])
    return db548, accepted


@pytest.mark.skipif(not (CAPTURE.exists() and ENVELOPE.exists()), reason="sealed DB548 rows unavailable")
def test_classification_binds_references_and_decides_both_arms() -> None:
    db548, accepted = _rows()
    assert _array_sha256(db548) == DB548_LAYER1_NORMALIZED_SHA256
    assert _array_sha256(accepted) == ACCEPTED_LAYER1_NORMALIZED_SHA256
    exact = classify_layer1_rms_schedule_outputs(
        control_bits=db548, schedule_bits=accepted, db548_bits=db548, accepted_bits=accepted
    )
    assert exact["harness_admissible"] and exact["schedule_arm_exact"]
    assert exact["schedule_vs_db548"]["mismatch_count"] == 1
    assert exact["schedule_vs_db548"]["first_mismatch_index"] == 2795
    nonexact = classify_layer1_rms_schedule_outputs(
        control_bits=db548, schedule_bits=db548, db548_bits=db548, accepted_bits=accepted
    )
    assert nonexact["harness_admissible"] and not nonexact["schedule_arm_exact"]
    refused = classify_layer1_rms_schedule_outputs(
        control_bits=accepted, schedule_bits=accepted, db548_bits=db548, accepted_bits=accepted
    )
    assert not refused["harness_admissible"] and not refused["schedule_arm_exact"]
    with pytest.raises(ValueError, match="DB548 layer-1 reference row drifted"):
        classify_layer1_rms_schedule_outputs(
            control_bits=db548, schedule_bits=db548, db548_bits=accepted, accepted_bits=accepted
        )


@pytest.mark.skipif(not SPLIT_HLO.exists(), reason="archived accepted-schedule HLO unavailable")
def test_optimized_contract_accepts_archived_accepted_schedule_and_rejects_control_flag() -> None:
    hlo = SPLIT_HLO.read_text()
    result = audit_layer1_rms_schedule_optimized_hlo(hlo, fp32_carry_schedule=True)
    assert result["passed"], result["violations"]
    assert result["fp32_carry_schedule"] is True
    assert len(result["accepted_scheduled_reduction_values"]) == 1
    stripped = hlo.replace('"megacore_allreduce_bytes":"4096"', '"megacore_allreduce_bytes":"8192"')
    assert stripped != hlo
    rejected = audit_layer1_rms_schedule_optimized_hlo(stripped, fp32_carry_schedule=True)
    assert not rejected["passed"]


def test_driver_is_default_off_bounded_and_binds_sealed_inputs() -> None:
    source = Path("scripts/greenfield/run_gate_d_layer1_rms_schedule.py").read_text()
    for marker in (
        'DB548_CAPTURE_SHA256 = (\n    "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298"',
        'DB548_ENVELOPE_SHA256 = (\n    "6cb76623bd79e1712b6c323fa786e516abd05f6f0ca51a367bc871c4a920480f"',
        '"gate_d_layer1_rms_schedule_[0-9]{8}T[0-9]{15}Z"',
        '"GLM_GATE_D_LAYER1_RMS_SCHEDULE": "1"',
        "gate-d-layer1-rms-schedule-v3",
        '"gate_d_closed": False',
        '"performance_claim": False',
        "if invocation_count != 2:",
        'raise RuntimeError(\n            "Gate-D schedule control arm did not reproduce the protected DB548 row"',
    ):
        assert marker in source, marker
    assert "HARNESS_REFUSED" in source
