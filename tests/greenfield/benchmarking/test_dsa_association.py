from __future__ import annotations

import pytest

from glm_tpu.greenfield.benchmarking.dsa_association import (
    validate_dsa_association_hlo,
)


def test_legacy_score_hlo_requires_exact_diagnostic_geometry() -> None:
    hlo = "\n".join(
        (
            "f32[32,32,128]",
            "bf16[8156,128]",
            "f32[32,32]",
            "f32[32,32,512]",
            "f32[8156]",
        )
    )
    result = validate_dsa_association_hlo(hlo, phase="legacy_score")
    assert result["passed"] is True
    assert result["diagnostic_batch32_allowed"] is True


def test_one_row_hlo_rejects_legacy_dead_rows() -> None:
    hlo = "\n".join(
        (
            "f32[1,32,128]",
            "bf16[8156,128]",
            "f32[1,32]",
            "f32[32,512]",
            "f32[8156]",
            "f32[32,32,128]",
        )
    )
    result = validate_dsa_association_hlo(hlo, phase="one_row_score")
    assert result["passed"] is False
    assert result["forbidden_dead_rows"] == ["f32[32,32,128]"]


def test_association_hlo_rejects_collective() -> None:
    hlo = "\n".join(
        (
            "f32[1,32,128]",
            "bf16[8156,128]",
            "f32[1,32]",
            "f32[32,512]",
            "f32[8156]",
            " all-reduce(",
        )
    )
    result = validate_dsa_association_hlo(hlo, phase="one_row_score")
    assert result["passed"] is False
    assert result["forbidden_operations"] == [" all-reduce("]


def test_association_hlo_rejects_unknown_phase_and_dimensions() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        validate_dsa_association_hlo("", phase="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        validate_dsa_association_hlo("", phase="legacy_state", context=0)
