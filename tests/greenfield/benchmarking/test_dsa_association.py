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


def test_legacy_local_dcp_xla_hlo_requires_sealed_8k_shapes() -> None:
    hlo = "\n".join(
        (
            "f32[32,32,128]",
            "bf16[24,512,128]",
            "f32[32,32]",
            "s32[32,3]",
            "s32[32]",
            "f32[32,1536]",
            "f32[32,512,32]",
            "thd,tpd->thp/dot_general",
            "th,thp->tp/dot_general",
        )
    )
    result = validate_dsa_association_hlo(
        hlo, phase="legacy_local_dcp_xla_score"
    )
    assert result["passed"] is True
    assert result["diagnostic_batch32_allowed"] is True

    wrong_width = validate_dsa_association_hlo(
        hlo.replace("s32[32,3]", "s32[32,84]"),
        phase="legacy_local_dcp_xla_score",
    )
    assert wrong_width["passed"] is False
    assert "s32[32,3]" in wrong_width["missing_shapes"]


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


def test_runtime_fused_qkv_pack_hlo_pins_raw_fp8_tp32_layout() -> None:
    hlo = "\n".join(
        (
            "u8[2048,6144]",
            "f32[16,48]",
            "u8[576,6144]",
            "f32[5,48]",
            "f8e4m3fn[6144,2624]",
            "f32[48,2624]",
            "f8e4m3fn[32,6144,82]",
            "f32[32,48,82]",
            "legacy_fused_qkv_runtime_pack_tp32_n82",
        )
    )
    result = validate_dsa_association_hlo(
        hlo,
        phase="legacy_fused_qkv_runtime_pack",
    )
    assert result["passed"] is True


@pytest.mark.parametrize(
    ("phase", "weight_shape", "scale_shape", "tile_shape", "marker"),
    (
        (
            "legacy_runtime_fused_qkv_global_state",
            "f8e4m3fn[6144,2624]",
            "f32[48,2624]",
            "bf16[32,2624]",
            "legacy_runtime_fused_qkv_a_m32_global_n2624",
        ),
        (
            "legacy_runtime_fused_qkv_sharded_state",
            "f8e4m3fn[32,6144,82]",
            "f32[32,48,82]",
            "bf16[32,82,32]",
            "legacy_runtime_fused_qkv_a_m32_tp32_n82",
        ),
    ),
)
def test_runtime_fused_qkv_state_hlo_requires_exact_live_layout(
    phase: str,
    weight_shape: str,
    scale_shape: str,
    tile_shape: str,
    marker: str,
) -> None:
    hlo = "\n".join(
        (
            "f32[32,32,128]",
            "bf16[8156,128]",
            "f32[32,32]",
            weight_shape,
            scale_shape,
            "bf16[32,576]",
            tile_shape,
            marker,
        )
    )
    result = validate_dsa_association_hlo(hlo, phase=phase)  # type: ignore[arg-type]
    assert result["passed"] is True
    assert result["fused_qkv_intermediate_shapes"] == [tile_shape]


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


def test_local_wide_and_pagewise_hlo_pin_distinct_score_geometry() -> None:
    wide = "\n".join(
        (
            "f32[1,32,128]",
            "bf16[2048,128]",
            "f32[1,32]",
            "f32[1,2048]",
            "f32[32,2048]",
            "rhd,sd->rhs/dot_general",
            "rh,rhs->rs/dot_general",
        )
    )
    wide_result = validate_dsa_association_hlo(
        wide, phase="local_wide_score", context=2048
    )
    assert wide_result["passed"] is True
    assert wide_result["map_trip_count"] is None
    assert wide_result["diagnostic_batch32_allowed"] is False

    pagewise = "\n".join(
        (
            "f32[1,32,128]",
            "bf16[2048,128]",
            "f32[1,32]",
            "f32[2048]",
            "bf16[4,512,128]",
            "f32[4,512]",
            "f32[32,512]",
            "hd,pd->hp/dot_general",
            "h,hp->p/dot_general",
            " while(",
            'backend_config={"known_trip_count":{"n":"4"}}',
        )
    )
    pagewise_result = validate_dsa_association_hlo(
        pagewise, phase="local_pagewise_score", context=2048
    )
    assert pagewise_result["passed"] is True
    assert pagewise_result["map_trip_count"] == 4
    assert pagewise_result["diagnostic_batch32_allowed"] is False


def test_local_wide_hlo_rejects_pagewise_body() -> None:
    wide_with_pagewise_body = "\n".join(
        (
            "f32[1,32,128]",
            "bf16[2048,128]",
            "f32[1,32]",
            "f32[1,2048]",
            "f32[32,2048]",
            "f32[32,512]",
            "rhd,sd->rhs/dot_general",
            "rh,rhs->rs/dot_general",
        )
    )
    result = validate_dsa_association_hlo(
        wide_with_pagewise_body, phase="local_wide_score", context=2048
    )
    assert result["passed"] is False
    assert result["forbidden_score_shapes"] == ["f32[32,512]"]


def test_local_pagewise_hlo_rejects_wide_body_or_wrong_map_geometry() -> None:
    pagewise = "\n".join(
        (
            "f32[1,32,128]",
            "bf16[2048,128]",
            "f32[1,32]",
            "f32[2048]",
            "bf16[4,512,128]",
            "f32[8,512]",
            "f32[32,512]",
            "f32[32,2048]",
            "hd,pd->hp/dot_general",
            "h,hp->p/dot_general",
            " while(",
            'backend_config={"known_trip_count":{"n":"8"}}',
        )
    )
    result = validate_dsa_association_hlo(
        pagewise, phase="local_pagewise_score", context=2048
    )
    assert result["passed"] is False
    assert result["forbidden_score_shapes"] == ["f32[32,2048]"]
    assert "f32[4,512]" in result["missing_shapes"]


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


def _distributed_norm_hlo(*, gather_dtype: str = "bf16") -> str:
    group = ",".join(str(rank) for rank in range(32))
    return "\n".join(
        (
            "HloModule distributed_norm, num_partitions=32, replica_count=1",
            "ENTRY main {",
            "  weight = f8e4m3fn[6144,82] parameter(0), "
            'metadata={op_name="legacy_runtime_fused_qkv_a_m32_tp32_distributed_norm"}',
            "  scale = f32[48,82] parameter(1)",
            "  q = bf16[32,2048] parameter(2)",
            "  companion = bf16[32,18] parameter(3)",
            "  local_sum = f32[32] parameter(4), "
            'metadata={op_name="legacy_tp32_q_a_rms_norm_variance_psum"}',
            "  reduced = f32[32] all-reduce(local_sum), "
            f"replica_groups={{{{{group}}}}}, "
            "use_global_device_ids=true, to_apply=add, "
            'metadata={op_name="legacy_tp32_q_a_rms_norm_variance_psum/psum"}',
            f"  local_norm = {gather_dtype}[32,64] parameter(5)",
            f"  gathered = {gather_dtype}[32,64,32] all-gather(local_norm), "
            f"dimensions={{2}}, replica_groups={{{{{group}}}}}, "
            "use_global_device_ids=true, "
            'metadata={op_name="legacy_tp32_q_a_rms_norm_bf16_all_gather/all_gather"}',
            "  ROOT result = (bf16[32,2048], bf16[32,18]) tuple(q, companion)",
            "}",
        )
    )


def test_distributed_norm_hlo_requires_exact_two_tp32_collectives() -> None:
    result = validate_dsa_association_hlo(
        _distributed_norm_hlo(),
        phase="legacy_tp32_distributed_q_a_norm",
    )
    assert result["passed"] is True
    assert result["distributed_collective_contract"][
        "collective_counts"
    ] == {"all-gather": 1, "all-reduce": 1}
    assert result["distributed_collective_violations"] == []


def test_distributed_norm_hlo_rejects_tpu_dtype_or_group_drift() -> None:
    promoted = validate_dsa_association_hlo(
        _distributed_norm_hlo(gather_dtype="f32"),
        phase="legacy_tp32_distributed_q_a_norm",
    )
    assert promoted["passed"] is False
    assert "all-gather payload drifted" in promoted["violations"][0]
    cpu_only = validate_dsa_association_hlo(
        _distributed_norm_hlo(gather_dtype="f32"),
        phase="legacy_tp32_distributed_q_a_norm",
        allow_cpu_bf16_collective_promotion=True,
    )
    assert cpu_only["passed"] is True

    exact_group = "replica_groups={{" + ",".join(
        str(rank) for rank in range(32)
    ) + "}}"
    wrong_group = _distributed_norm_hlo().replace(
        exact_group,
        "replica_groups={{0,1,2,3}}",
    )
    drifted = validate_dsa_association_hlo(
        wrong_group,
        phase="legacy_tp32_distributed_q_a_norm",
    )
    assert drifted["passed"] is False
    assert any(
        "replica groups drifted" in violation
        for violation in drifted["distributed_collective_violations"]
    )


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
