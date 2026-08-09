from __future__ import annotations

import importlib.util
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[3]


def _contract(hlo: str, source: str) -> dict[str, Any]:
    path = REPO / "scripts/greenfield/compare_accepted_prompt_key_internals.py"
    spec = importlib.util.spec_from_file_location("prompt_key_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._projection_weight_source_contract(hlo, source=source)


def _comparison(expected: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    path = REPO / "scripts/greenfield/compare_accepted_prompt_key_internals.py"
    spec = importlib.util.spec_from_file_location("prompt_key_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._bitwise_array_comparison(expected, observed)


def test_bitwise_array_comparison_records_exact_and_float_drift() -> None:
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    exact = _comparison(expected, expected.copy())
    assert exact["elementwise_exact"] is True
    assert exact["mismatch_count"] == 0
    assert exact["observed_sha256"] == sha256(expected.tobytes()).hexdigest()

    observed = expected.copy()
    observed[1, 2] += np.float32(0.25)
    drift = _comparison(expected, observed)
    assert drift["elementwise_exact"] is False
    assert drift["first_mismatch_index"] == [1, 2]
    assert drift["mismatch_count"] == 1
    assert drift["mismatch_max_abs"] == 0.25


def test_bitwise_array_comparison_refuses_shape_or_dtype_equivalence() -> None:
    expected = np.arange(8, dtype=np.uint8)
    shape_drift = _comparison(expected, expected.reshape(1, 8))
    assert shape_drift["elementwise_exact"] is False
    assert shape_drift["mismatch_count"] is None

    dtype_drift = _comparison(expected, expected.astype(np.int16))
    assert dtype_drift["elementwise_exact"] is False
    assert dtype_drift["mismatch_count"] is None


def test_accepts_materialized_fp32_wk_parameter() -> None:
    result = _contract(
        "ENTRY %main (wk: f32[128,6144]) -> bf16[1] {\n"
        "  ROOT %value = bf16[1] parameter(0)\n"
        "}\n",
        "materialized_parameter",
    )

    assert result["passed"] is True
    assert result["entry_f32_wk_parameter_count"] == 1
    assert result["entry_raw_fp8_wk_parameter_count"] == 0


def test_accepts_lp4_stage_local_materialized_fp32_wk_parameter() -> None:
    result = _contract(
        "ENTRY %main (wk: f32[128,6144]) -> bf16[1] {\n"
        "  ROOT %value = bf16[1] parameter(0)\n"
        "}\n",
        "materialized_lp4_stage_local",
    )

    assert result["passed"] is True
    assert result["source"] == "materialized_lp4_stage_local"
    assert result["entry_f32_wk_parameter_count"] == 1
    assert result["entry_raw_fp8_wk_parameter_count"] == 0


def test_accepts_internal_raw_fp8_with_explicit_bf16_round() -> None:
    result = _contract(
        "ENTRY %main (wk: u8[128,6144]) -> bf16[1] {\n"
        "  %round = bf16[786432]{0:T(1024)(128)(2,1)} convert(%mul), "
        "metadata={op_name=\"convert_element_type\"}\n"
        "  %cache_cast = bf16[24,128,128]{2,1,0} convert(%keys), "
        "metadata={op_name=\"convert_element_type\"}\n"
        "}\n",
        "raw_fp8_inside_executable",
    )

    assert result["passed"] is True
    assert result["bf16_round_count"] == 1
    assert result["entry_f32_wk_parameter_count"] == 0
    assert result["entry_raw_fp8_wk_parameter_count"] == 1


def test_rejects_weight_source_boundary_drift() -> None:
    internal_without_round = _contract(
        "ENTRY %main (wk: u8[128,6144]) -> f32[1] {\n}\n",
        "raw_fp8_inside_executable",
    )
    materialized_with_raw = _contract(
        "ENTRY %main (wk: u8[128,6144]) -> f32[1] {\n}\n",
        "materialized_parameter",
    )

    assert internal_without_round["passed"] is False
    assert materialized_with_raw["passed"] is False
