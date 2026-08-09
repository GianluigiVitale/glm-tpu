from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]


def _contract(hlo: str, source: str) -> dict[str, Any]:
    path = REPO / "scripts/greenfield/compare_accepted_prompt_key_internals.py"
    spec = importlib.util.spec_from_file_location("prompt_key_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._projection_weight_source_contract(hlo, source=source)


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


def test_accepts_internal_raw_fp8_with_explicit_bf16_round() -> None:
    result = _contract(
        "ENTRY %main (wk: u8[128,6144]) -> bf16[1] {\n"
        "  %round = bf16[128,6144]{1,0:T(8,128)(2,1)} convert(%mul), "
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
