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
            "f32[32,512,32]",
            "f32[8156]",
            "thd,tpd->thp/dot_general",
            "th,thp->tp/dot_general",
        )
    )
    result = validate_dsa_association_hlo(hlo, phase="legacy_score")
    assert result["passed"] is True
    assert result["diagnostic_batch32_allowed"] is True


def test_fused_qkv_state_hlo_requires_live_exact_output_width() -> None:
    hlo = "\n".join(
        (
            "f32[32,32,128]",
            "bf16[8156,128]",
            "f32[32,32]",
            "bf16[2624,6144]",
            "bf16[32,576]",
            "bf16[2624,32]",
            "legacy_fused_qkv_a_m32_n2624",
        )
    )
    result = validate_dsa_association_hlo(
        hlo,
        phase="legacy_fused_qkv_state",
    )
    assert result["passed"] is True
    assert result["fused_qkv_intermediate_shapes"] == ["bf16[2624,32]"]


def test_one_row_hlo_rejects_legacy_dead_rows() -> None:
    hlo = "\n".join(
        (
            "f32[1,32,128]",
            "bf16[8156,128]",
            "f32[1,32]",
            "f32[32,512]",
            "f32[8156]",
            "f32[32,32,128]",
            "hd,pd->hp/dot_general",
            "h,hp->p/dot_general",
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
            "hd,pd->hp/dot_general",
            "h,hp->p/dot_general",
        )
    )
    result = validate_dsa_association_hlo(hlo, phase="one_row_score")
    assert result["passed"] is False
    assert result["forbidden_operations"] == [" all-reduce("]


def test_score_hlo_rejects_missing_exact_tile_or_source_marker() -> None:
    valid = "\n".join(
        (
            "f32[32,32,128]",
            "bf16[8156,128]",
            "f32[32,32]",
            "f32[32,512,32]",
            "f32[8156]",
            "thd,tpd->thp/dot_general",
            "th,thp->tp/dot_general",
        )
    )
    no_tile = validate_dsa_association_hlo(
        valid.replace("f32[32,512,32]", "f32[32,256,64]"),
        phase="legacy_score",
    )
    no_source_marker = validate_dsa_association_hlo(
        valid.replace("th,thp->tp/dot_general", "unrelated/dot_general"),
        phase="legacy_score",
    )
    assert no_tile["passed"] is False
    assert "exact logical/physical score tile" in no_tile["violations"][0]
    assert no_source_marker["passed"] is False
    assert no_source_marker["missing_score_markers"] == [
        "th,thp->tp/dot_general"
    ]


def test_association_hlo_rejects_unknown_phase_and_dimensions() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        validate_dsa_association_hlo("", phase="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive"):
        validate_dsa_association_hlo("", phase="legacy_state", context=0)
