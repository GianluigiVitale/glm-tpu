from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.gate_d_scalar_frontier import (
    ScalarFrontierError,
    _strict_double_round,
    _widen_bf16_bits,
    analyze_scalar_frontier,
    strict_scalar_preimage,
)

ROOT = Path(__file__).parents[3]
RUN_ROOT = Path("/home/gianl/gate-d-runs")
CAPSULE = RUN_ROOT / "greenfield_gate_d_compensated_capsule_20260831T124838Z"
SOURCE = RUN_ROOT / "gate_d_compensated_pp16_numerical_20260901T094622505067868Z"
SCRIPT = ROOT / "scripts/greenfield/analyze_gate_d_scalar_frontier.py"


def _real_arrays() -> dict[str, np.ndarray]:
    with np.load(CAPSULE / "candidate-inputs.npz", allow_pickle=False) as archive:
        inputs = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(CAPSULE / "candidate-state.npz", allow_pickle=False) as archive:
        state = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    with np.load(SOURCE / "outputs.npz", allow_pickle=False) as archive:
        outputs = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    return {**inputs, **state, **outputs}


@pytest.mark.skipif(not SOURCE.is_dir(), reason="protected source evidence absent")
def test_real_frontier_proves_conversion_placement_exactly() -> None:
    arrays = _real_arrays()
    report = analyze_scalar_frontier(
        hidden_update_bits=arrays["rms_hidden_update_bf16_bits"],
        residual_bits=arrays["rms_residual_bf16_bits"],
        weight_bits=arrays["rms_weight_bf16_bits"],
        accepted_rms_input=arrays["rms_input"],
        observed_rms_input_owners=arrays["rms_input_fp32_owners"][:, 0],
        accepted_output_owners=arrays["normalized"][:, 0],
        observed_output_owners=arrays["normalized_hidden_owners"][:, 0],
    )
    assert report["status"] == "EXACT_EXPLANATION_PROVED_PHYSICAL_CAUSE_UNPROVEN"
    assert report["first_divergence"]["accepted_vs_observed_mismatch_count"] == 1622
    assert report["first_divergence"]["strict_double_round_matches_accepted"] is True
    assert (
        report["first_divergence"]["retained_fp32_round_matches_observed_tpu"] is True
    )
    assert report["reference_scale"]["bits"] == "0x4332984c"
    assert (
        report["reference_scale"]["accepted_strict_preimage"]["width_f32_values"] == 44
    )
    assert report["observed_tpu_strict_scalar_preimage"]["reason"] == (
        "target_skipped_by_double_round"
    )


def test_first_tpu_value_is_impossible_under_strict_double_round() -> None:
    rms = np.ones(6144, dtype=np.float32)
    weight = np.ones(6144, dtype=np.float32)
    target = np.ones(6144, dtype=np.uint16)
    rms[0] = np.asarray(0x3A3D0000, dtype=np.uint32).view(np.float32)
    weight[0] = np.asarray(0x3D940000, dtype=np.uint32).view(np.float32)
    target[0] = 0x3C18
    result = strict_scalar_preimage(rms, weight, target)
    assert result == {
        "constrained_elements": 0,
        "empty_at_index": 0,
        "reason": "target_skipped_by_double_round",
        "status": "EMPTY",
        "target_bf16_bits": "0x3c18",
    }


def test_owner_or_input_drift_fails_closed() -> None:
    arrays = _real_arrays()
    hostile = arrays["rms_input_fp32_owners"][:, 0].copy()
    hostile[1, 0] = np.nextafter(hostile[1, 0], np.float32(np.inf))
    with pytest.raises(ScalarFrontierError, match="authority drifted"):
        analyze_scalar_frontier(
            hidden_update_bits=arrays["rms_hidden_update_bf16_bits"],
            residual_bits=arrays["rms_residual_bf16_bits"],
            weight_bits=arrays["rms_weight_bf16_bits"],
            accepted_rms_input=arrays["rms_input"],
            observed_rms_input_owners=hostile,
            accepted_output_owners=arrays["normalized"][:, 0],
            observed_output_owners=arrays["normalized_hidden_owners"][:, 0],
        )


@pytest.mark.parametrize(
    ("hostile_factor", "hostile_target"),
    [
        (np.float32(np.inf), np.uint16(0)),
        (np.float32(np.nan), np.uint16(0)),
        (np.float32(1), np.uint16(0x7F80)),
        (np.float32(1), np.uint16(0x7FC1)),
    ],
)
def test_scalar_preimage_rejects_nonfinite_domain(
    hostile_factor: np.float32, hostile_target: np.uint16
) -> None:
    rms = np.zeros(6144, dtype=np.float32)
    weight = np.ones(6144, dtype=np.float32)
    target = np.zeros(6144, dtype=np.uint16)
    rms[0] = hostile_factor
    target[0] = hostile_target
    with pytest.raises(ScalarFrontierError, match="requires finite"):
        strict_scalar_preimage(rms, weight, target)


def test_scalar_preimage_handles_zero_subnormal_and_finite_endpoint() -> None:
    rms = np.zeros(6144, dtype=np.float32)
    weight = np.ones(6144, dtype=np.float32)
    target = np.zeros(6144, dtype=np.uint16)
    all_zero = strict_scalar_preimage(rms, weight, target)
    assert all_zero["lower_bits"] == "0x00000001"
    assert all_zero["upper_inclusive_bits"] == "0x7f7fffff"

    rms[0] = np.asarray(1, dtype=np.uint32).view(np.float32)
    subnormal = strict_scalar_preimage(rms, weight, target)
    assert subnormal["status"] == "NONEMPTY"

    rms[0] = np.float32(1)
    target[0] = np.uint16(0x7F7F)
    endpoint = strict_scalar_preimage(rms, weight, target)
    assert endpoint["status"] == "NONEMPTY"
    assert int(endpoint["upper_inclusive_bits"], 16) <= 0x7F7FFFFF


def test_reference_double_round_is_not_single_round() -> None:
    arrays = _real_arrays()
    rms = np.ascontiguousarray(arrays["rms_input"], dtype=np.float32)
    weight = _widen_bf16_bits(arrays["rms_weight_bf16_bits"])
    scale = np.asarray(0x4332984C, dtype=np.uint32).view(np.float32)
    strict = _strict_double_round(rms, weight, scale)
    assert np.count_nonzero(strict != arrays["normalized_hidden_owners"][0, 0]) == 1622


def test_cli_is_default_off_cpu_only_and_imports_no_jax() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "import jax" not in source
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "default-off" in result.stderr
