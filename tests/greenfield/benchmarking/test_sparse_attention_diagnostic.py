from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.greenfield.microbench_sparse_attention import (
    _reference_hlo_contract,
    _selection_cases_lp2_2k,
    _validate_sampling_contract,
)


def test_lp2_2k_selection_cases_have_exact_owner_counts() -> None:
    cases = _selection_cases_lp2_2k(2035)
    for case in cases.values():
        positions = case["positions"][0]
        count = int(case["valid_count"][0])
        live = positions[:count]
        assert np.all(positions[count:] == -1)
        assert len(np.unique(live)) == count
        owner = int(case["owner"])
        assert int(np.count_nonzero(((live % 512) // 256) == owner)) == int(
            case["expected_owner_rows"]
        )


def test_lp2_diagnostic_sampling_is_fail_closed() -> None:
    _validate_sampling_contract(
        geometry="pp16_lp2_2k",
        diagnostic_reference_timing=True,
        warmup=1,
        iterations=3,
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        _validate_sampling_contract(
            geometry="pp16_lp2_2k",
            diagnostic_reference_timing=False,
            warmup=200,
            iterations=1000,
        )
    with pytest.raises(ValueError, match="only for the PP16"):
        _validate_sampling_contract(
            geometry="pp8_lp4_256k",
            diagnostic_reference_timing=True,
            warmup=1,
            iterations=3,
        )


def test_reference_hlo_contract_requires_three_exact_convolutions() -> None:
    hlo = "\n".join(
        (
            "x = bf16[2048,640] parameter(0)",
            'a = f32[64,2048] convolution(x), metadata={op_name="rhd,rkd->rhk/dot_general"}',
            'b = f32[64,2048] convolution(x), metadata={op_name="rhd,rkd->rhk/dot_general"}',
            'c = f32[64,512] convolution(x), metadata={op_name="rhk,rkd->rhd/dot_general"}',
        )
    )
    assert _reference_hlo_contract(hlo)["passed"]
    assert not _reference_hlo_contract(hlo.replace(" convolution(", " add(", 1))[
        "passed"
    ]


def test_discriminator_persists_hlo_before_fail_closed_validation() -> None:
    source = Path("scripts/greenfield/microbench_sparse_attention.py").read_text()
    write_index = source.index("args.hlo_output.write_text(hlo)")
    validate_index = source.index("hlo_contract = validate_sparse_attention_hlo(")
    assert write_index < validate_index
