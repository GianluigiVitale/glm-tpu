from __future__ import annotations

from pathlib import Path
import subprocess

import ml_dtypes
import numpy as np

from glm_tpu.greenfield.benchmarking.pp16_dense_boundary import (
    derive_expected_dense_boundary_bits,
    exact_bfloat16_bits,
    validate_pp16_dense_boundary_hlo,
)


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
{dead}  %reduce = bf16[1,6144] all-reduce(%p), replica_groups={group}, to_apply=%add
  %kernel = bf16[1,6144] custom-call(%reduce), custom_call_target="tpu_custom_call", metadata={{op_name="greenfield_fp8_fused_block_swiglu_m8_h6144_i6144_o6144/pallas_call"}}
  %norm = bf16[1,6144] copy(%kernel)
  %residual = bf16[1,6144] copy(%kernel)
  ROOT %result = (bf16[1,6144], bf16[1,6144], bf16[1,6144]) tuple(%kernel, %norm, %residual)
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
    assert "--warmup 1 --iterations 3" in text
    assert "TPU_VISIBLE_DEVICES=0,1,2,3" in text
    assert "probe_pp16_lp2_dense_boundary.py" in text
    assert "strict_census pre" in text
    assert "strict_census post" in text
    assert "results_ckpt.db" in text
    assert "gate_d_passed':False" in text
    assert "next_residual_comparison" in text
    assert "DIAGNOSTIC_UPLOAD_FAILED" in text
    assert "gcloud storage rsync --recursive" in text


def test_pp16_dense_boundary_runner_is_selective_and_one_row() -> None:
    runner = REPO / "scripts/greenfield/probe_pp16_lp2_dense_boundary.py"
    text = runner.read_text()
    assert "jax.local_devices()[:2]" in text
    assert "safe_open(path, framework=\"pt\", device=\"cpu\")" in text
    assert "stage_local_dense_fp8_mapped" in text
    assert "precomputed_normalized=normalized_value" in text
    assert "axis_index_groups=((0, 1),)" in text
    assert "accepted_layer1_normalized_bfloat16_bits" in text
    assert "accepted_next_residual_bfloat16_bits" in text
    assert "sample_output_bits" in text
    assert "replica_agreement" in text
    assert "compile_short_decoder" not in text
