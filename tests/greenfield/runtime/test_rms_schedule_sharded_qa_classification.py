"""The virtual-TP32 sharded q-a norm is bound as its own lineage kind, exactly.

The fused qkv-a q-a norm (per-shard sums over 64 lanes, then the 32-shard sum,
1/2048, 1e-05, scalar rsqrt) is legacy-faithful and must stay off the 32-row
accepted schedule; the contract admits it only in that exact emitted form.
"""

from __future__ import annotations

import pytest

from glm_tpu.greenfield.runtime.decoder import (
    _validate_rms_accepted_schedule_hlo,
    _validate_rms_accepted_schedule_stablehlo,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.runtime.test_rms_schedule_layernorm_classification import HLO as RMS_HLO, STABLEHLO as RMS_STABLEHLO

# Optimized-HLO form (XLA merges both reduces): one fusion reducing f32[32,1,64] over all axes.
QA_FUSION = """
%qa_reduce (qp: bf16[32,1,82]) -> f32[] {
  %qp = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} parameter(0)
  %qs = bf16[32,1,64]{2,1,0:T(2,128)(2,1)} slice(%qp), slice={[0:32], [0:1], [0:64]}
  %qc = f32[32,1,64]{2,1,0:T(1,128)} convert(%qs)
  %qsq = f32[32,1,64]{2,1,0:T(1,128)} multiply(%qc, %qc)
  %qz = f32[] constant(0)
  ROOT %qsum = f32[] reduce(%qsq, %qz), dimensions={0,1,2}, to_apply=%add_combiner
}

%qa_scale (qq: f32[]) -> f32[] {
  %qq = f32[] parameter(0)
  %qinv = f32[] constant(0.00048828125)
  %qmean = f32[] multiply(%qq, %qinv)
  %qeps = f32[] constant(1e-05)
  %qvar = f32[] add(%qmean, %qeps)
  ROOT %qrs = f32[] rsqrt(%qvar)
}
"""

HLO = RMS_HLO.replace("ENTRY %main (carry: f32[32,128], key: f32[1,128]) -> (f32[32], f32[1]) {", QA_FUSION + """
ENTRY %main (carry: f32[32,128], key: f32[1,128], qproj: bf16[32,1,82]) -> (f32[32], f32[1]) {
  %qproj = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} parameter(2)
  %qa_sum = f32[] fusion(%qproj), kind=kLoop, calls=%qa_reduce
  %qa_scale.1 = f32[] fusion(%qa_sum), kind=kLoop, calls=%qa_scale""")

QA_STABLE = """    %201 = stablehlo.slice %arg2 [0:32, 0:1, 0:64] : (tensor<32x1x82xbf16>) -> tensor<32x1x64xbf16>
    %202 = stablehlo.convert %201 : (tensor<32x1x64xbf16>) -> tensor<32x1x64xf32>
    %203 = chlo.square %202 : tensor<32x1x64xf32> -> tensor<32x1x64xf32>
    %cst_46 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %204 = stablehlo.reduce(%203 init: %cst_46) applies stablehlo.add across dimensions = [2] : (tensor<32x1x64xf32>, tensor<f32>) -> tensor<32x1xf32>
    %205 = stablehlo.reduce(%204 init: %cst_46) applies stablehlo.add across dimensions = [0] : (tensor<32x1xf32>, tensor<f32>) -> tensor<1xf32>
    %cst_48 = stablehlo.constant dense<2.048000e+03> : tensor<f32>
    %206 = stablehlo.broadcast_in_dim %cst_48, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %207 = stablehlo.divide %205, %206 : tensor<1xf32>
    %cst_49 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %208 = stablehlo.broadcast_in_dim %cst_49, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %209 = stablehlo.add %207, %208 : tensor<1xf32>
    %210 = stablehlo.rsqrt %209 : tensor<1xf32>
"""
STABLEHLO = RMS_STABLEHLO.replace(
    "  func.func public @main(%arg0: tensor<32x128xf32>, %arg1: tensor<1x128xf32>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {\n",
    "  func.func public @main(%arg0: tensor<32x128xf32>, %arg1: tensor<1x128xf32>, %arg2: tensor<32x1x82xbf16>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {\n" + QA_STABLE,
)


def _hlo(text: str) -> dict:
    return _validate_rms_accepted_schedule_hlo(parse_hlo_module(text), enabled=True, layernorm_width=128)


def _stable(text: str) -> dict:
    return _validate_rms_accepted_schedule_stablehlo(text, enabled=True, layernorm_width=128)


def test_exact_sharded_qa_norm_is_classified_apart() -> None:
    assert HLO != RMS_HLO and STABLEHLO != RMS_STABLEHLO
    hlo = _hlo(HLO)
    assert hlo["passed"], hlo["violations"]
    assert (hlo["rsqrt_count"], hlo["conforming_rsqrt_count"], hlo["layernorm_rsqrt_count"], hlo["sharded_qa_rsqrt_count"]) == (3, 1, 1, 1)
    stable = _stable(STABLEHLO)
    assert stable["passed"], stable["violations"]
    assert (stable["rsqrt_count"], stable["conforming_rsqrt_count"], stable["layernorm_rsqrt_count"], stable["sharded_qa_rsqrt_count"]) == (3, 1, 1, 1)
    # Flag off: the sharded q-a norm is not an accepted-schedule lineage and never trips the default contract by itself.
    default_only = _validate_rms_accepted_schedule_hlo(parse_hlo_module(HLO.replace("ROOT %rs = f32[32]{0} rsqrt(%var)", "ROOT %rs = f32[32]{0} add(%var, %var)")), enabled=False, layernorm_width=128)
    assert default_only["passed"] and default_only["sharded_qa_rsqrt_count"] == 1


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("wrong_lanes", "f32[32,1,64]", "f32[32,1,32]"),
        ("partial_reduce", "dimensions={0,1,2}, to_apply=%add_combiner", "dimensions={1,2}, to_apply=%add_combiner"),
        ("max_combiner", "reduce(%qsq, %qz), dimensions={0,1,2}, to_apply=%add_combiner", "reduce(%qsq, %qz), dimensions={0,1,2}, to_apply=%max_combiner"),
        ("wrong_scale", "constant(0.00048828125)", "constant(0.0009765625)"),
        ("wrong_epsilon", "%qeps = f32[] constant(1e-05)", "%qeps = f32[] constant(1e-06)"),
        ("not_a_square", "%qsq = f32[32,1,64]{2,1,0:T(1,128)} multiply(%qc, %qc)", "%qsq = f32[32,1,64]{2,1,0:T(1,128)} add(%qc, %qc)"),
        ("not_the_converted_projection", "%qsq = f32[32,1,64]{2,1,0:T(1,128)} multiply(%qc, %qc)", "%qneg = f32[32,1,64]{2,1,0:T(1,128)} negate(%qc)\n  %qsq = f32[32,1,64]{2,1,0:T(1,128)} multiply(%qneg, %qneg)"),
        ("row_scale", "ROOT %qrs = f32[] rsqrt(%qvar)", "ROOT %qrs = f32[2] rsqrt(%qvar)"),
    ),
)
def test_hlo_sharded_qa_mutations_are_refused(label: str, old: str, new: str) -> None:
    assert old in HLO, label
    result = _hlo(HLO.replace(old, new))
    assert not result["passed"], (label, result)
    assert result["sharded_qa_rsqrt_count"] == 0, (label, result)


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("wrong_lanes", "across dimensions = [2] : (tensor<32x1x64xf32>", "across dimensions = [2] : (tensor<32x1x32xf32>"),
        ("wrong_shards", "(tensor<32x1xf32>, tensor<f32>) -> tensor<1xf32>", "(tensor<16x1xf32>, tensor<f32>) -> tensor<1xf32>"),
        ("max_lane_combiner", "applies stablehlo.add across dimensions = [2]", "applies stablehlo.maximum across dimensions = [2]"),
        ("max_shard_combiner", "applies stablehlo.add across dimensions = [0] : (tensor<32x1xf32>", "applies stablehlo.maximum across dimensions = [0] : (tensor<32x1xf32>"),
        ("wrong_scale", "dense<2.048000e+03>", "dense<1.024000e+03>"),
        ("scale_opcode_swapped", "%207 = stablehlo.divide %205, %206", "%207 = stablehlo.multiply %205, %206"),
        ("wrong_epsilon", "%cst_49 = stablehlo.constant dense<9.99999974E-6>", "%cst_49 = stablehlo.constant dense<9.99999997E-7>"),
        ("not_converted_projection", "%203 = chlo.square %202", "%203 = chlo.square %201"),
        ("row_scale", "%210 = stablehlo.rsqrt %209 : tensor<1xf32>", "%210 = stablehlo.rsqrt %209 : tensor<2xf32>"),
    ),
)
def test_stablehlo_sharded_qa_mutations_are_refused(label: str, old: str, new: str) -> None:
    assert old in STABLEHLO, label
    result = _stable(STABLEHLO.replace(old, new, 1))
    assert not result["passed"], (label, result)
    assert result["sharded_qa_rsqrt_count"] == 0, (label, result)
