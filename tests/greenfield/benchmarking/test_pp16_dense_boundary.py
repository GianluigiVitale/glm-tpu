from __future__ import annotations

from pathlib import Path
import subprocess

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.pp16_dense_boundary import (
    derive_expected_dense_boundary_bits,
    exact_bfloat16_bits,
    fused_add_rms_norm_output_owned,
    pack_pp16_dense_final_layout,
    replay_pp16_strategy_nd_y_x_z_bits,
    validate_pp16_dense_final_layout_records,
    validate_pp16_dense_boundary_hlo,
)
from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm


REPO = Path(__file__).resolve().parents[3]


def _optimized_hlo(*, group: str = "{{0,1}}", dead_row: bool = False) -> str:
    dead = (
        "  %dead = bf16[32,6144]{1,0} broadcast(%zero), dimensions={}\n"
        if dead_row
        else ""
    )
    return f"""HloModule dense, replica_count=1, num_partitions=2

%add (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}}

ENTRY %main (p: bf16[1,6144]) -> (bf16[1,6144], bf16[1,6144], bf16[1,6144]) {{
  %p = bf16[1,6144] parameter(0)
  %zero = bf16[] constant(0)
{dead}  %kernel = bf16[1,6144] custom-call(%p), custom_call_target="tpu_custom_call", metadata={{op_name="greenfield_fp8_fused_block_swiglu_m8_h6144_i6144_o6144/pallas_call"}}
  %reduce = bf16[1,6144] all-reduce(%kernel), replica_groups={group}, to_apply=%add
  %norm = bf16[1,6144] copy(%reduce)
  %residual = bf16[1,6144] copy(%reduce)
  ROOT %result = (bf16[1,6144], bf16[1,6144], bf16[1,6144]) tuple(%reduce, %norm, %residual)
}}
"""


def _stablehlo(*, extra: str = "") -> str:
    return f"""module {{
  func.func @main(%arg0: tensor<1x6144xbf16>) -> tensor<1x6144xbf16> {{
    %0 = \"stablehlo.all_reduce\"(%arg0) <{{replica_groups=dense<[[0,1]]>:tensor<1x2xi64>}}> : (tensor<1x6144xbf16>) -> tensor<1x6144xbf16>
    {extra}
    return %0 : tensor<1x6144xbf16>
  }}
}}
"""


def _final_layout_hlo() -> tuple[str, str]:
    optimized_instructions = []
    stable_instructions = []
    partials = []
    for rank in range(16):
        scope = f"greenfield_dense_convolution_virtual_rank_{rank:02d}"
        optimized_instructions.extend(
            (
                f"  %gate{rank} = f32[1,768] convolution(%p, %wg), "
                f'dim_labels=bf_io->bf, metadata={{op_name="{scope}/gate"}}',
                f"  %activated{rank} = bf16[1,384] copy(%gate{rank})",
                f"  %down{rank} = f32[1,6144] convolution(%activated{rank}, %wd), "
                f'dim_labels=bf_io->bf, metadata={{op_name="{scope}/down"}}',
                f"  %partial{rank} = bf16[1,6144] copy(%down{rank})",
            )
        )
        partials.append(f"%partial{rank}")
        stable_instructions.extend(
            (
                f"    %g{rank} = stablehlo.convolution %arg0, %arg1 : "
                "dim_numbers = [b, f] x [i, o] -> [b, f], "
                "window = {stride = [], pad = [], lhs_dilate = [], "
                "rhs_dilate = [], reverse = []}, batch_group_count = 1 : i64, "
                "feature_group_count = 1 : i64, precision_config = "
                "[#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>] : "
                "(tensor<1x6144xbf16>, tensor<6144x768xbf16>) "
                "-> tensor<1x768xf32>",
                f"    %d{rank} = stablehlo.convolution %arg2, %arg3 : "
                "dim_numbers = [b, f] x [i, o] -> [b, f], "
                "window = {stride = [], pad = [], lhs_dilate = [], "
                "rhs_dilate = [], reverse = []}, batch_group_count = 1 : i64, "
                "feature_group_count = 1 : i64, precision_config = "
                "[#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>] : "
                "(tensor<1x384xbf16>, tensor<384x6144xbf16>) "
                "-> tensor<1x6144xf32>",
            )
        )
    optimized = f"""HloModule dense, replica_count=1, num_partitions=2

%add (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}}

ENTRY %main (p: bf16[1,6144], wg: bf16[6144,768], a: bf16[1,384], wd: bf16[384,6144]) -> (bf16[1,6144], bf16[1,6144], bf16[1,6144]) {{
  %p = bf16[1,6144] parameter(0)
  %wg = bf16[6144,768]{{1,0}} parameter(1)
  %a = bf16[1,384] parameter(2)
  %wd = bf16[384,6144]{{1,0}} parameter(3)
{chr(10).join(optimized_instructions)}
  %local_y = bf16[4,1,6144] concatenate({', '.join(partials)}), dimensions={0}
  %reduce = bf16[4,1,6144] all-reduce(%local_y), replica_groups={{{{0,1}}}}, to_apply=%add
  %dense = bf16[1,6144] slice(%reduce)
  %norm = bf16[1,6144] copy(%dense)
  %residual = bf16[1,6144] copy(%dense)
  ROOT %result = (bf16[1,6144], bf16[1,6144], bf16[1,6144]) tuple(%dense, %norm, %residual)
}}
"""
    stable = f"""module {{
  func.func @main(%arg0: tensor<1x6144xbf16>, %arg1: tensor<6144x768xbf16>, %arg2: tensor<1x384xbf16>, %arg3: tensor<384x6144xbf16>, %arg4: tensor<4x1x6144xbf16>) -> tensor<4x1x6144xbf16> {{
{chr(10).join(stable_instructions)}
    %reduce = "stablehlo.all_reduce"(%arg4) <{{replica_groups=dense<[[0,1]]>:tensor<1x2xi64>}}> : (tensor<4x1x6144xbf16>) -> tensor<4x1x6144xbf16>
    return %reduce : tensor<4x1x6144xbf16>
  }}
}}
"""
    return stable, optimized


def _final_layout_nested_fusion_hlo() -> tuple[str, str]:
    stable, _ = _final_layout_hlo()
    computations = []
    entry_instructions = []
    partials = []
    for rank in range(16):
        scope = f"greenfield_dense_convolution_virtual_rank_{rank:02d}"
        computations.append(
            f"""
%gate_comp_{rank} (p: bf16[1,6144], wg: bf16[6144,768]) -> bf16[1,768] {{
  %p = bf16[1,6144] parameter(0)
  %wg = bf16[6144,768]{{1,0}} parameter(1)
  %gate = f32[1,768] convolution(%p, %wg), dim_labels=bf_io->bf, metadata={{op_name="{scope}/gate"}}
  ROOT %gate_bits = bf16[1,768] convert(%gate)
}}

%down_comp_{rank} (activated: bf16[1,384], wd: bf16[384,6144]) -> bf16[1,6144] {{
  %activated = bf16[1,384] parameter(0)
  %wd = bf16[384,6144]{{1,0}} parameter(1)
  %down = f32[1,6144] convolution(%activated, %wd), dim_labels=bf_io->bf, metadata={{op_name="{scope}/down"}}
  ROOT %partial = bf16[1,6144] convert(%down)
}}
"""
        )
        entry_instructions.extend(
            (
                f"  %gate_value{rank} = bf16[1,768] fusion(%p, %wg), "
                f"kind=kLoop, calls=%gate_comp_{rank}",
                f"  %activated{rank} = bf16[1,384] slice(%gate_value{rank}), "
                "slice={[0:1], [0:384]}",
                f"  %partial{rank} = bf16[1,6144] fusion(%activated{rank}, %wd), "
                f"kind=kLoop, calls=%down_comp_{rank}",
            )
        )
        partials.append(f"%partial{rank}")
    optimized = f"""HloModule dense, replica_count=1, num_partitions=2

%add (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}}
{''.join(computations)}
ENTRY %main (p: bf16[1,6144], wg: bf16[6144,768], wd: bf16[384,6144]) -> (bf16[1,6144], bf16[1,6144], bf16[1,6144]) {{
  %p = bf16[1,6144] parameter(0)
  %wg = bf16[6144,768]{{1,0}} parameter(1)
  %wd = bf16[384,6144]{{1,0}} parameter(2)
{chr(10).join(entry_instructions)}
  %local_y = bf16[4,1,6144] concatenate({', '.join(partials)}), dimensions={0}
  %reduce = bf16[4,1,6144] all-reduce(%local_y), replica_groups={{{{0,1}}}}, to_apply=%add
  %dense = bf16[1,6144] slice(%reduce)
  %norm = bf16[1,6144] copy(%dense)
  %residual = bf16[1,6144] copy(%dense)
  ROOT %result = (bf16[1,6144], bf16[1,6144], bf16[1,6144]) tuple(%dense, %norm, %residual)
}}
"""
    return stable, optimized


def test_exact_bfloat16_bits_reports_first_mismatch() -> None:
    expected = np.array([1, 2, 3], dtype=np.uint16)
    exact = exact_bfloat16_bits(expected, expected.copy())
    assert exact["elementwise_exact"]
    assert exact["mismatch_count"] == 0
    assert exact["first_mismatch_flat_index"] is None

    observed = np.array([1, 9, 3], dtype=np.uint16)
    mismatch = exact_bfloat16_bits(expected, observed)
    assert not mismatch["elementwise_exact"]
    assert mismatch["mismatch_count"] == 1
    assert mismatch["first_mismatch_flat_index"] == 1


def test_output_owned_rms_is_one_row_and_reference_exact() -> None:
    import jax.numpy as jnp

    hidden = jnp.asarray(
        np.linspace(-2.0, 2.0, 6144, dtype=np.float32)[None, :],
        dtype=jnp.bfloat16,
    )
    residual = jnp.asarray(
        np.linspace(1.0, -1.0, 6144, dtype=np.float32)[None, :],
        dtype=jnp.bfloat16,
    )
    weight = jnp.asarray(
        np.linspace(0.5, 1.5, 6144, dtype=np.float32),
        dtype=jnp.bfloat16,
    )
    expected = fused_add_rms_norm(
        hidden, residual, weight, epsilon=1e-5
    )
    observed = fused_add_rms_norm_output_owned(
        hidden, residual, weight, epsilon=1e-5
    )
    for expected_value, observed_value in zip(
        expected, observed, strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(observed_value).view(np.uint16),
            np.asarray(expected_value).view(np.uint16),
        )
        assert observed_value.shape == (1, 6144)

    with pytest.raises(ValueError, match="exactly one logical row"):
        fused_add_rms_norm_output_owned(
            jnp.broadcast_to(hidden, (2, 6144)),
            jnp.broadcast_to(residual, (2, 6144)),
            weight,
            epsilon=1e-5,
        )


def test_derive_expected_dense_boundary_bits_preserves_zero_update() -> None:
    partials = np.zeros((4, 8, 1, 6144), dtype=np.uint16)
    residual = np.full(
        (1, 6144), ml_dtypes.bfloat16(1.5), dtype=ml_dtypes.bfloat16
    ).view(np.uint16)
    dense, carried = derive_expected_dense_boundary_bits(
        partials, residual, tuple(range(32))
    )
    assert np.array_equal(dense, np.zeros((1, 6144), dtype=np.uint16))
    assert np.array_equal(carried, residual)


def test_pp16_y_x_z_replay_matches_db533_32_leaf_association() -> None:
    rng = np.random.default_rng(53316)
    partials = np.asarray(
        rng.standard_normal((4, 8, 1, 6144), dtype=np.float32)
        * np.exp2(rng.integers(-4, 9, size=(4, 8, 1, 6144))),
        dtype=ml_dtypes.bfloat16,
    ).view(np.uint16)
    model_axis_device_ids = tuple(
        int(value)
        for value in np.argsort(
            np.asarray(
                (
                    0, 16, 4, 20, 8, 24, 12, 28,
                    1, 17, 5, 21, 9, 25, 13, 29,
                    2, 18, 6, 22, 10, 26, 14, 30,
                    3, 19, 7, 23, 11, 27, 15, 31,
                )
            )
        )
    )
    expected, _ = derive_expected_dense_boundary_bits(
        partials,
        np.zeros((1, 6144), dtype=ml_dtypes.bfloat16).view(np.uint16),
        model_axis_device_ids,
    )
    observed = replay_pp16_strategy_nd_y_x_z_bits(partials)
    np.testing.assert_array_equal(observed, expected)


def test_pp16_final_layout_pack_preserves_two_by_sixteen_rank_ownership() -> None:
    names = (
        "dense.slot_00.gate.weight_bits",
        "dense.slot_00.gate.scale_inv",
        "dense.slot_00.up.weight_bits",
        "dense.slot_00.up.scale_inv",
        "dense.slot_00.down.weight_bits",
        "dense.slot_00.down.scale_inv",
    )
    owners = []
    for owner in range(2):
        owners.append(
            {
                names[0]: np.broadcast_to(
                    np.array(owner + 1, dtype=np.uint8), (6144, 6144)
                ),
                names[1]: np.broadcast_to(
                    np.array(owner + 1.25, dtype=np.float32), (48, 48)
                ),
                names[2]: np.broadcast_to(
                    np.array(owner + 11, dtype=np.uint8), (6144, 6144)
                ),
                names[3]: np.broadcast_to(
                    np.array(owner + 11.25, dtype=np.float32), (48, 48)
                ),
                names[4]: np.broadcast_to(
                    np.array(owner + 21, dtype=np.uint8), (6144, 6144)
                ),
                names[5]: np.broadcast_to(
                    np.array(owner + 21.25, dtype=np.float32), (48, 48)
                ),
            }
        )
    packed, records = pack_pp16_dense_final_layout(tuple(owners))
    assert [value.shape for value in packed] == [
        (2, 16, 6144, 768),
        (2, 16, 48, 768),
        (2, 16, 384, 6144),
        (2, 16, 3, 6144),
    ]
    assert packed[0][0, 15, 0, 0] == 1
    assert packed[0][1, 0, 0, 384] == 12
    assert packed[1][0, 0, 0, 0] == np.float32(1.25)
    assert packed[1][1, 15, 47, 767] == np.float32(12.25)
    assert packed[2][0, 15, 383, 6143] == 21
    assert packed[2][1, 0, 0, 0] == 22
    assert packed[3][1, 15, 2, 6143] == np.float32(22.25)
    assert sum(record["byte_count"] for record in records.values()) == sum(
        value.nbytes for value in packed
    )
    with pytest.raises(ValueError, match="DB550-proven payload"):
        validate_pp16_dense_final_layout_records(records)


def test_pp16_dense_boundary_hlo_requires_exact_lp2_one_row_contract() -> None:
    report = validate_pp16_dense_boundary_hlo(_stablehlo(), _optimized_hlo())
    assert report["passed"], report
    assert report["kernel_count"] == 1
    assert report["num_partitions"] == 2
    assert report["stable_group_exact"]

    rejected = validate_pp16_dense_boundary_hlo(
        _stablehlo(extra='"stablehlo.all_gather"() : () -> ()'),
        _optimized_hlo(group="{{0,1,2,3}}", dead_row=True),
    )
    assert not rejected["passed"]
    assert rejected["forbidden_shapes"]
    assert rejected["stablehlo_forbidden_collectives"] == [
        "stablehlo.all_gather"
    ]


def test_pp16_dense_boundary_hlo_accepts_only_y_x_z_contract() -> None:
    kernel = "greenfield_fp8_fused_block_swiglu_m8_h6144_i384_o6144"
    calls = "\n".join(
        f'  %kernel{index} = bf16[1,6144] custom-call(%p), custom_call_target="tpu_custom_call", metadata={{op_name="{kernel}/pallas_call"}}'
        for index in range(16)
    )
    optimized = f"""HloModule dense, replica_count=1, num_partitions=2

%add (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}}

ENTRY %main (p: bf16[1,6144], q: bf16[4,1,6144]) -> (bf16[1,6144], bf16[1,6144], bf16[1,6144]) {{
  %p = bf16[1,6144] parameter(0)
  %q = bf16[4,1,6144] parameter(1)
{calls}
  %local_y = bf16[4,1,6144] concatenate(%kernel0, %kernel1, %kernel2, %kernel3, %kernel4, %kernel5, %kernel6, %kernel7, %kernel8, %kernel9, %kernel10, %kernel11, %kernel12, %kernel13, %kernel14, %kernel15), dimensions={0}
  %reduce = bf16[4,1,6144] all-reduce(%local_y), replica_groups={{{{0,1}}}}, to_apply=%add
  %dense = bf16[1,6144] slice(%reduce)
  %norm = bf16[1,6144] copy(%dense)
  %residual = bf16[1,6144] copy(%dense)
  ROOT %result = (bf16[1,6144], bf16[1,6144], bf16[1,6144]) tuple(%dense, %norm, %residual)
}}
"""
    report = validate_pp16_dense_boundary_hlo(
        _stablehlo(), optimized, association="strategy_nd_y_x_z"
    )
    assert report["passed"], report
    assert report["kernel_count"] == 16
    assert report["collective_input_custom_call_count"] == 16
    assert report["collective_reaches_root"]
    assert report["association"] == "strategy_nd_y_x_z"

    orphaned = validate_pp16_dense_boundary_hlo(
        _stablehlo(),
        optimized.replace("%kernel14, %kernel15)", "%kernel14, %kernel14)"),
        association="strategy_nd_y_x_z",
    )
    assert not orphaned["passed"]
    assert orphaned["collective_input_custom_call_count"] == 15
    assert any("do not all feed" in item for item in orphaned["violations"])

    detached = validate_pp16_dense_boundary_hlo(
        _stablehlo(),
        optimized.replace("%dense = bf16[1,6144] slice(%reduce)",
                          "%dense = bf16[1,6144] slice(%q)"),
        association="strategy_nd_y_x_z",
    )
    assert not detached["passed"]
    assert not detached["collective_reaches_root"]
    assert any("does not feed" in item for item in detached["violations"])

    wrong = validate_pp16_dense_boundary_hlo(
        _stablehlo(), _optimized_hlo(), association="strategy_nd_y_x_z"
    )
    assert not wrong["passed"]


def test_pp16_dense_boundary_hlo_accepts_final_layout_convolutions_only() -> None:
    stable, optimized = _final_layout_hlo()
    report = validate_pp16_dense_boundary_hlo(
        stable,
        optimized,
        association="strategy_nd_y_x_z_final_layout",
    )
    assert report["passed"], report
    assert report["custom_call_count"] == 0
    assert report["dense_convolution_count"] == 32
    assert report["collective_input_convolution_count"] == 32
    assert report["stablehlo_dense_gate_convolution_count"] == 16
    assert report["stablehlo_dense_down_convolution_count"] == 16
    assert report["dense_gate_convolution_ranks"] == {
        rank: 1 for rank in range(16)
    }
    assert report["dense_down_convolution_ranks"] == {
        rank: 1 for rank in range(16)
    }

    output_owned = validate_pp16_dense_boundary_hlo(
        stable,
        optimized,
        association="strategy_nd_y_x_z_final_layout_output_owned",
    )
    assert output_owned["passed"], output_owned

    orphaned = validate_pp16_dense_boundary_hlo(
        stable,
        optimized.replace("%partial14, %partial15)", "%partial14, %partial14)"),
        association="strategy_nd_y_x_z_final_layout",
    )
    assert not orphaned["passed"]
    assert orphaned["collective_input_convolution_count"] == 30
    assert any("do not all feed" in item for item in orphaned["violations"])


def test_pp16_dense_boundary_hlo_follows_nested_tpu_fusion_lineage() -> None:
    stable, optimized = _final_layout_nested_fusion_hlo()
    report = validate_pp16_dense_boundary_hlo(
        stable,
        optimized,
        association="strategy_nd_y_x_z_final_layout",
    )
    assert report["passed"], report
    assert report["collective_input_convolution_count"] == 32
    assert report["dense_gate_down_bijection"]

    orphaned = validate_pp16_dense_boundary_hlo(
        stable,
        optimized.replace("%partial14, %partial15)", "%partial14, %partial14)"),
        association="strategy_nd_y_x_z_final_layout",
    )
    assert not orphaned["passed"]
    assert orphaned["collective_input_convolution_count"] == 30

    wrong_rhs_layout = validate_pp16_dense_boundary_hlo(
        stable,
        optimized.replace(
            "bf16[6144,768]{1,0} parameter(1)",
            "bf16[6144,768]{0,1} parameter(1)",
            1,
        ),
        association="strategy_nd_y_x_z_final_layout",
    )
    assert not wrong_rhs_layout["passed"]
    assert any(
        "convolution set drifted" in item
        for item in wrong_rhs_layout["violations"]
    )

    wrong_stable_precision = validate_pp16_dense_boundary_hlo(
        stable.replace(
            "#stablehlo<precision DEFAULT>",
            "#stablehlo<precision HIGHEST>",
            1,
        ),
        optimized,
        association="strategy_nd_y_x_z_final_layout",
    )
    assert not wrong_stable_precision["passed"]
    assert any(
        "StableHLO final-layout convolution set drifted" in item
        for item in wrong_stable_precision["violations"]
    )


def test_pp16_dense_boundary_wrapper_is_bounded_and_default_off() -> None:
    wrapper = REPO / "scripts/greenfield/run_pp16_lp2_dense_boundary.sh"
    text = wrapper.read_text()
    completed = subprocess.run(
        ["bash", str(wrapper)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "default-off" in completed.stderr
    assert "source_db=548" in text
    assert "strategy_nd_y_x_z_final_layout" in text
    assert '--warmup "$WARMUP" --iterations "$ITERATIONS"' in text
    assert "acquisition:strategy_nd_y_x_z_final_layout_output_owned" in text
    assert "compile acquisition requires deliberately empty HLO pins" in text
    assert "TPU_VISIBLE_DEVICES=0,1,2,3" in text
    assert "probe_pp16_lp2_dense_boundary.py" in text
    assert "strict_census pre" in text
    assert "strict_census post" in text
    assert "results_ckpt.db" in text
    assert "gate_d_passed':False" in text
    assert "next_residual_comparison" in text
    assert "dense_update_comparison" in text
    assert "DIAGNOSTIC_UPLOAD_FAILED" in text
    assert 'gsutil help rsync' in text
    assert "gsutil -m rsync -r" in text
    assert "orchestrator.log$" in text
    assert "gcloud storage rsync" not in text
    assert "REJECTED_SEALED" in text
    assert "verify_success_evidence" in text
    assert "SUCCESS_EVIDENCE_OK exact preterminal remote archive verified" in text
    assert "rollback_provisional_db" in text
    assert "PROVISIONAL_DB_ROLLBACK_FAILED" in text
    assert "greenfield_run_tag" in text
    assert "results_db_run_id.txt" in text
    assert "performance_claim" in text
    assert "gate_d_passed" in text
    assert "terminal_success_done=1\ntrap - EXIT\n" in text

    gsutil = subprocess.run(
        ["gsutil", "help", "rsync"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert gsutil.returncode == 0, gsutil.stderr


def test_pp16_dense_boundary_failure_recovery_is_bounded_and_default_off() -> None:
    recovery = (
        REPO / "scripts/greenfield/recover_pp16_lp2_dense_boundary_failure.sh"
    )
    text = recovery.read_text()
    completed = subprocess.run(
        ["bash", str(recovery)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "default-off" in completed.stderr
    assert "RUN_PIN=1f5c88ebd1c1dc5206075c22da26ede8cff90f2f" in text
    assert "mode=ro" in text
    assert "EXPECTED_DB_MAX_RUN=564" in text
    assert "pp16-lp2-dense" in text
    assert "gsutil -m rsync -r -x" in text
    assert "REJECTED$" in text
    assert '"$REMOTE_PREFIX/REJECTED"' in text
    assert '"$REMOTE_PREFIX/SUCCESS"' in text
    assert "performance_claim" in text
    assert "gate_d_passed" in text
    assert "PP16_LP2_DENSE_BOUNDARY_REJECTION_ALREADY_SEALED" in text
    assert "jax" not in text.lower()
    assert "TPU_VISIBLE_DEVICES" not in text


def test_pp16_dense_boundary_runner_is_selective_and_one_row() -> None:
    runner = REPO / "scripts/greenfield/probe_pp16_lp2_dense_boundary.py"
    text = runner.read_text()
    assert "jax.local_devices()[:2]" in text
    assert "safe_open(path, framework=\"pt\", device=\"cpu\")" in text
    assert "stage_local_dense_fp8_mapped" in text
    assert "pack_pp16_dense_final_layout" in text
    assert "final_layout_convolution=True" in text
    assert "precomputed_normalized=normalized_value" in text
    assert "fused_add_rms_norm_output_owned" in text
    assert 'choices=("acquisition", "bounded")' in text
    assert "COMPILE_ACQUIRED_UNPINNED" in text
    assert "deliberately refusing before arithmetic" in text
    assert "axis_index_groups=((0, 1),)" in text
    assert "accepted_layer1_normalized_bfloat16_bits" in text
    assert "accepted_next_residual_bfloat16_bits" in text
    assert "sample_output_bits" in text
    assert "replica_agreement" in text
    assert "dense_comparison[\"elementwise_exact\"]" in text
    assert "compile_short_decoder" not in text
