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
    expected_rms_schedule_census,
    validate_decoder_step_hlo,
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
%kv_reduce (kp: bf16[1,576]) -> f32[] {
  %kp = bf16[1,576]{1,0:T(2,128)(2,1)} parameter(0)
  %ks = bf16[1,512]{1,0:T(2,128)(2,1)} slice(%kp), slice={[0:1], [0:512]}
  %kc = f32[1,512]{1,0:T(1,128)} convert(%ks)
  %ksq = f32[1,512]{1,0:T(1,128)} multiply(%kc, %kc)
  %kz = f32[] constant(0)
  ROOT %ksum = f32[] reduce(%ksq, %kz), dimensions={0,1}, to_apply=%add_combiner
}

%kv_scale (kq: f32[]) -> f32[] {
  %kq = f32[] parameter(0)
  %kinv2 = f32[] constant(0.001953125)
  %kmean2 = f32[] multiply(%kq, %kinv2)
  %keps = f32[] constant(1e-05)
  %kvar = f32[] add(%kmean2, %keps)
  ROOT %krs = f32[] rsqrt(%kvar)
}

%n82_body (bp: (s32[], bf16[32,1,82], bf16[1,6144], u8[32,6144,82], f32[32,48,82])) -> (s32[], bf16[32,1,82], bf16[1,6144], u8[32,6144,82], f32[32,48,82]) {
  %bp = (s32[], bf16[32,1,82]{2,1,0:T(2,128)(2,1)}, bf16[1,6144]{1,0}, u8[32,6144,82]{2,1,0}, f32[32,48,82]{2,1,0}) parameter(0)
  %bi = s32[] get-tuple-element(%bp), index=0
  %bacc = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} get-tuple-element(%bp), index=1
  %bh = bf16[1,6144]{1,0} get-tuple-element(%bp), index=2
  %bpacked = u8[32,6144,82]{2,1,0} get-tuple-element(%bp), index=3
  %bscales = f32[32,48,82]{2,1,0} get-tuple-element(%bp), index=4
  %bzero_i = s32[] constant(0)
  %bwsl = u8[1,6144,82]{2,1,0} dynamic-slice(%bpacked, %bi, %bzero_i, %bzero_i), dynamic_slice_sizes={1,6144,82}
  %bwu8 = u8[6144,82]{1,0} reshape(%bwsl)
  %bwf8 = f8e4m3fn[6144,82]{1,0} bitcast-convert(%bwu8)
  %bwf32 = f32[6144,82]{1,0} convert(%bwf8)
  %bssl = f32[1,48,82]{2,1,0} dynamic-slice(%bscales, %bi, %bzero_i, %bzero_i), dynamic_slice_sizes={1,48,82}
  %bs2 = f32[48,82]{1,0} reshape(%bssl)
  %bsb = f32[48,128,82]{2,1,0} broadcast(%bs2), dimensions={0,2}
  %bsw = f32[6144,82]{1,0} reshape(%bsb)
  %bwdq = f32[6144,82]{1,0} multiply(%bwf32, %bsw)
  %bw = bf16[6144,82]{1,0} convert(%bwdq)
  %bconv = f32[1,82]{1,0} convolution(%bh, %bw), dim_labels=bf_io->bf
  %brow = bf16[1,82]{1,0} convert(%bconv)
  %brow3 = bf16[1,1,82]{2,1,0} reshape(%brow)
  %bzero = s32[] constant(0)
  %bacc2 = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} dynamic-update-slice(%bacc, %brow3, %bi, %bzero, %bzero)
  %bone = s32[] constant(1)
  %bi2 = s32[] add(%bi, %bone)
  ROOT %bout = (s32[], bf16[32,1,82]{2,1,0:T(2,128)(2,1)}, bf16[1,6144]{1,0}, u8[32,6144,82]{2,1,0}, f32[32,48,82]{2,1,0}) tuple(%bi2, %bacc2, %bh, %bpacked, %bscales)
}

%n82_cond (cp: (s32[], bf16[32,1,82], bf16[1,6144], u8[32,6144,82], f32[32,48,82])) -> pred[] {
  %cp = (s32[], bf16[32,1,82]{2,1,0:T(2,128)(2,1)}, bf16[1,6144]{1,0}, u8[32,6144,82]{2,1,0}, f32[32,48,82]{2,1,0}) parameter(0)
  %ci = s32[] get-tuple-element(%cp), index=0
  %climit = s32[] constant(32)
  ROOT %clt = pred[] compare(%ci, %climit), direction=LT
}

ENTRY %main (carry: f32[32,128], key: f32[1,128], hidden: bf16[1,6144], packed: u8[32,6144,82], scales: f32[32,48,82]) -> (f32[32], f32[1]) {
  %hidden = bf16[1,6144]{1,0} parameter(2)
  %packed = u8[32,6144,82]{2,1,0} parameter(3)
  %scales = f32[32,48,82]{2,1,0} parameter(4)
  %izero = s32[] constant(0)
  %accz = bf16[] constant(0)
  %acc0 = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} broadcast(%accz), dimensions={}
  %init = (s32[], bf16[32,1,82]{2,1,0:T(2,128)(2,1)}, bf16[1,6144]{1,0}, u8[32,6144,82]{2,1,0}, f32[32,48,82]{2,1,0}) tuple(%izero, %acc0, %hidden, %packed, %scales)
  %loop = (s32[], bf16[32,1,82]{2,1,0:T(2,128)(2,1)}, bf16[1,6144]{1,0}, u8[32,6144,82]{2,1,0}, f32[32,48,82]{2,1,0}) while(%init), condition=%n82_cond, body=%n82_body
  %qproj = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} get-tuple-element(%loop), index=1
  %kvlanes = bf16[32,1,18]{2,1,0:T(2,128)(2,1)} slice(%qproj), slice={[0:32], [0:1], [64:82]}
  %kvt = bf16[1,32,18]{2,1,0:T(2,128)(2,1)} transpose(%kvlanes), dimensions={1,0,2}
  %kvproj = bf16[1,576]{1,0:T(2,128)(2,1)} reshape(%kvt)
  %kv_sum = f32[] fusion(%kvproj), kind=kLoop, calls=%kv_reduce
  %kv_scale.1 = f32[] fusion(%kv_sum), kind=kLoop, calls=%kv_scale
  %qa_sum = f32[] fusion(%qproj), kind=kLoop, calls=%qa_reduce
  %qa_scale.1 = f32[] fusion(%qa_sum), kind=kLoop, calls=%qa_scale""")

QA_STABLE = """    %c_53 = stablehlo.constant dense<0> : tensor<i32>
    %206 = stablehlo.constant dense<0.000000e+00> : tensor<32x1x82xbf16>
    %700:5 = stablehlo.while(%iterArg = %arg2, %iterArg_2438 = %arg3, %iterArg_2439 = %arg4, %iterArg_2440 = %c_53, %iterArg_2441 = %206) : tensor<32x6144x82xui8>, tensor<32x48x82xf32>, tensor<1x6144xbf16>, tensor<i32>, tensor<32x1x82xbf16>
    cond {
      %c_2442 = stablehlo.constant dense<32> : tensor<i32>
      %17674 = stablehlo.compare LT, %iterArg_2440, %c_2442, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %17674 : tensor<i1>
    } do {
      %17674 = func.call @dynamic_index_in_dim(%iterArg, %iterArg_2440) : (tensor<32x6144x82xui8>, tensor<i32>) -> tensor<6144x82xui8>
      %17675 = func.call @dynamic_index_in_dim_56(%iterArg_2438, %iterArg_2440) : (tensor<32x48x82xf32>, tensor<i32>) -> tensor<48x82xf32>
      %17676 = func.call @closed_call(%iterArg_2439, %17674, %17675) : (tensor<1x6144xbf16>, tensor<6144x82xui8>, tensor<48x82xf32>) -> tensor<1x82xbf16>
      %17677 = func.call @dynamic_update_index_in_dim(%iterArg_2441, %17676, %iterArg_2440) : (tensor<32x1x82xbf16>, tensor<1x82xbf16>, tensor<i32>) -> tensor<32x1x82xbf16>
      %c_2443 = stablehlo.constant dense<1> : tensor<i32>
      %17678 = stablehlo.add %iterArg_2440, %c_2443 : tensor<i32>
      stablehlo.return %iterArg, %iterArg_2438, %iterArg_2439, %17678, %17677 : tensor<32x6144x82xui8>, tensor<32x48x82xf32>, tensor<1x6144xbf16>, tensor<i32>, tensor<32x1x82xbf16>
    }
    %201 = stablehlo.slice %700#4 [0:32, 0:1, 0:64] : (tensor<32x1x82xbf16>) -> tensor<32x1x64xbf16>
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
KV_STABLE = """    %227 = stablehlo.slice %700#4 [0:32, 0:1, 64:82] : (tensor<32x1x82xbf16>) -> tensor<32x1x18xbf16>
    %228 = stablehlo.transpose %227, dims = [1, 0, 2] : (tensor<32x1x18xbf16>) -> tensor<1x32x18xbf16>
    %arg3 = stablehlo.reshape %228 : (tensor<1x32x18xbf16>) -> tensor<1x576xbf16>
    %604 = stablehlo.slice %arg3 [0:1, 0:512] : (tensor<1x576xbf16>) -> tensor<1x512xbf16>
    %605 = stablehlo.convert %604 : (tensor<1x512xbf16>) -> tensor<1x512xf32>
    %606 = chlo.square %605 : tensor<1x512xf32> -> tensor<1x512xf32>
    %cst_170 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %607 = stablehlo.reduce(%606 init: %cst_170) applies stablehlo.add across dimensions = [1] : (tensor<1x512xf32>, tensor<f32>) -> tensor<1xf32>
    %608 = stablehlo.broadcast_in_dim %607, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %cst_171 = stablehlo.constant dense<5.120000e+02> : tensor<f32>
    %609 = stablehlo.broadcast_in_dim %cst_171, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %610 = stablehlo.divide %608, %609 : tensor<1x1xf32>
    %cst_172 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %611 = stablehlo.broadcast_in_dim %cst_172, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %612 = stablehlo.add %610, %611 : tensor<1x1xf32>
    %613 = stablehlo.rsqrt %612 : tensor<1x1xf32>
"""
STABLEHLO = RMS_STABLEHLO.replace(
    "  func.func public @main(%arg0: tensor<32x128xf32>, %arg1: tensor<1x128xf32>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {\n",
    "  func.func public @main(%arg0: tensor<32x128xf32>, %arg1: tensor<1x128xf32>, %arg2: tensor<32x6144x82xui8>, %arg3: tensor<32x48x82xf32>, %arg4: tensor<1x6144xbf16>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {\n" + QA_STABLE + KV_STABLE,
).replace(
    "    return %8, %21 : tensor<32x1xf32>, tensor<1x1xf32>\n  }\n}\n",
    "    return %8, %21 : tensor<32x1xf32>, tensor<1x1xf32>\n  }\n"
    "  func.func private @closed_call(%arg0: tensor<1x6144xbf16>, %arg1: tensor<6144x82xui8>, %arg2: tensor<48x82xf32>) -> tensor<1x82xbf16> {\n"
    "    %0 = stablehlo.bitcast_convert %arg1 : (tensor<6144x82xui8>) -> tensor<6144x82xf8E4M3FN>\n"
    "    %1 = stablehlo.broadcast_in_dim %arg2, dims = [0, 2] : (tensor<48x82xf32>) -> tensor<48x128x82xf32>\n"
    "    %2 = stablehlo.reshape %1 : (tensor<48x128x82xf32>) -> tensor<6144x82xf32>\n"
    "    %3 = stablehlo.convert %0 : (tensor<6144x82xf8E4M3FN>) -> tensor<6144x82xf32>\n"
    "    %4 = stablehlo.multiply %3, %2 : tensor<6144x82xf32>\n"
    "    %5 = stablehlo.convert %4 : (tensor<6144x82xf32>) -> tensor<6144x82xbf16>\n"
    "    %6 = stablehlo.convolution(%arg0, %5) dim_numbers = [b, f]x[i, o]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1x6144xbf16>, tensor<6144x82xbf16>) -> tensor<1x82xf32>\n"
    "    %7 = stablehlo.convert %6 : (tensor<1x82xf32>) -> tensor<1x82xbf16>\n"
    "    return %7 : tensor<1x82xbf16>\n  }\n"
    "  func.func private @dynamic_index_in_dim(%arg0: tensor<32x6144x82xui8>, %arg1: tensor<i32>) -> tensor<6144x82xui8> {\n"
    "    %c = stablehlo.constant dense<0> : tensor<i32>\n"
    "    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c, sizes = [1, 6144, 82] : (tensor<32x6144x82xui8>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x6144x82xui8>\n"
    "    %1 = stablehlo.reshape %0 : (tensor<1x6144x82xui8>) -> tensor<6144x82xui8>\n"
    "    return %1 : tensor<6144x82xui8>\n  }\n"
    "  func.func private @dynamic_index_in_dim_56(%arg0: tensor<32x48x82xf32>, %arg1: tensor<i32>) -> tensor<48x82xf32> {\n"
    "    %c = stablehlo.constant dense<0> : tensor<i32>\n"
    "    %0 = stablehlo.dynamic_slice %arg0, %arg1, %c, %c, sizes = [1, 48, 82] : (tensor<32x48x82xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x48x82xf32>\n"
    "    %1 = stablehlo.reshape %0 : (tensor<1x48x82xf32>) -> tensor<48x82xf32>\n"
    "    return %1 : tensor<48x82xf32>\n  }\n}\n",
)
assert "@closed_call(%arg0" in STABLEHLO


CENSUS = dict(expected_accepted_count=1, expected_sharded_qa_count=1, expected_kv_a_count=1)


def _hlo(text: str, **census) -> dict:
    return _validate_rms_accepted_schedule_hlo(parse_hlo_module(text), enabled=True, layernorm_width=128, **(census or CENSUS))


def _stable(text: str, **census) -> dict:
    return _validate_rms_accepted_schedule_stablehlo(text, enabled=True, layernorm_width=128, **(census or CENSUS))


def test_exact_sharded_qa_norm_is_classified_apart() -> None:
    assert HLO != RMS_HLO and STABLEHLO != RMS_STABLEHLO
    hlo = _hlo(HLO)
    assert hlo["passed"], hlo["violations"]
    assert (hlo["rsqrt_count"], hlo["conforming_rsqrt_count"], hlo["layernorm_rsqrt_count"], hlo["sharded_qa_rsqrt_count"], hlo["kv_a_rsqrt_count"]) == (4, 1, 1, 1, 1)
    stable = _stable(STABLEHLO)
    assert stable["passed"], stable["violations"]
    assert (stable["rsqrt_count"], stable["conforming_rsqrt_count"], stable["layernorm_rsqrt_count"], stable["sharded_qa_rsqrt_count"], stable["kv_a_rsqrt_count"]) == (4, 1, 1, 1, 1)
    # Flag off: the sharded q-a norm is not an accepted-schedule lineage and never trips the default contract by itself.
    default_only = _validate_rms_accepted_schedule_hlo(parse_hlo_module(HLO.replace("ROOT %rs = f32[32]{0} rsqrt(%var)", "ROOT %rs = f32[32]{0} add(%var, %var)")), enabled=False, layernorm_width=128)
    assert default_only["passed"] and default_only["sharded_qa_rsqrt_count"] == 1 and default_only["kv_a_rsqrt_count"] == 1


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


def test_enabled_binders_refuse_omitted_census_expectations() -> None:
    from glm_tpu.greenfield.runtime.decoder import PlanValidationError

    module = parse_hlo_module(HLO)
    for kwargs in ({}, {"expected_accepted_count": 1}, {"expected_sharded_qa_count": 1}, {"expected_accepted_count": 1, "expected_sharded_qa_count": 1}, {"expected_accepted_count": -1, "expected_sharded_qa_count": 1, "expected_kv_a_count": 1}, {"expected_accepted_count": True, "expected_sharded_qa_count": 1, "expected_kv_a_count": 1}):
        with pytest.raises(PlanValidationError):
            _validate_rms_accepted_schedule_hlo(module, enabled=True, layernorm_width=128, **kwargs)
        with pytest.raises(PlanValidationError):
            _validate_rms_accepted_schedule_stablehlo(STABLEHLO, enabled=True, layernorm_width=128, **kwargs)
    # disabled mode does not need the census
    assert "passed" in _validate_rms_accepted_schedule_hlo(module, enabled=False, layernorm_width=128)
    assert "passed" in _validate_rms_accepted_schedule_stablehlo(STABLEHLO, enabled=False, layernorm_width=128)


def test_lineage_census_is_bound_exactly_in_both_representations() -> None:
    assert expected_rms_schedule_census(layers=78, attention_projection_backend="fused_n82_convolution") == (157, 78, 78)
    assert expected_rms_schedule_census(layers=8, attention_projection_backend="separate") == (25, 0, 8)
    with pytest.raises(ValueError):
        expected_rms_schedule_census(layers=-1, attention_projection_backend="separate")
    # missing sharded lineage
    missing = HLO.replace("ROOT %qrs = f32[] rsqrt(%qvar)", "ROOT %qrs = f32[] add(%qvar, %qvar)")
    result = _hlo(missing)
    assert not result["passed"] and any("sharded q-a census drifted: expected 1, found 0" in v for v in result["violations"])
    missing_stable = STABLEHLO.replace("%210 = stablehlo.rsqrt %209 : tensor<1xf32>", "%210 = stablehlo.abs %209 : tensor<1xf32>")
    result = _stable(missing_stable)
    assert not result["passed"] and any("sharded q-a census drifted: expected 1, found 0" in v for v in result["violations"])
    # missing accepted lineage
    missing_rms = HLO.replace("ROOT %rs = f32[32]{0} rsqrt(%var)", "ROOT %rs = f32[32]{0} add(%var, %var)")
    result = _hlo(missing_rms)
    assert not result["passed"] and any("accepted-schedule RMS census drifted: expected 1, found 0" in v for v in result["violations"])
    missing_rms_stable = STABLEHLO.replace("%8 = stablehlo.rsqrt %7 : tensor<32x1xf32>", "%8 = stablehlo.abs %7 : tensor<32x1xf32>")
    result = _stable(missing_rms_stable)
    assert not result["passed"] and any("accepted-schedule RMS census drifted: expected 1, found 0" in v for v in result["violations"])
    # missing kv-a row norm
    missing_kv = HLO.replace("ROOT %krs = f32[] rsqrt(%kvar)", "ROOT %krs = f32[] add(%kvar, %kvar)")
    result = _hlo(missing_kv)
    assert not result["passed"] and any("kv-a row-norm census drifted: expected 1, found 0" in v for v in result["violations"])
    missing_kv_stable = STABLEHLO.replace("%613 = stablehlo.rsqrt %612 : tensor<1x1xf32>", "%613 = stablehlo.abs %612 : tensor<1x1xf32>")
    result = _stable(missing_kv_stable)
    assert not result["passed"] and any("kv-a row-norm census drifted: expected 1, found 0" in v for v in result["violations"])
    # extra lineages (expectation lower than found) and substitution (counts shifted between kinds)
    assert not _hlo(HLO, expected_accepted_count=1, expected_sharded_qa_count=0, expected_kv_a_count=1)["passed"]
    assert not _hlo(HLO, expected_accepted_count=0, expected_sharded_qa_count=1, expected_kv_a_count=1)["passed"]
    assert not _hlo(HLO, expected_accepted_count=2, expected_sharded_qa_count=0, expected_kv_a_count=1)["passed"]
    assert not _hlo(HLO, expected_accepted_count=1, expected_sharded_qa_count=1, expected_kv_a_count=0)["passed"]
    assert not _stable(STABLEHLO, expected_accepted_count=1, expected_sharded_qa_count=0, expected_kv_a_count=1)["passed"]
    assert not _stable(STABLEHLO, expected_accepted_count=0, expected_sharded_qa_count=2, expected_kv_a_count=1)["passed"]
    assert not _stable(STABLEHLO, expected_accepted_count=1, expected_sharded_qa_count=1, expected_kv_a_count=2)["passed"]
    # the decoder-step validator computes the census from its layer count and backend and forwards it to both binders
    import inspect
    source = inspect.getsource(validate_decoder_step_hlo)
    assert "expected_accepted_count, expected_sharded_qa_count, expected_kv_a_count = (" in source
    assert "layers=layers, attention_projection_backend=attention_projection_backend" in source
    assert source.count("expected_accepted_count=expected_accepted_count") == 2
    assert source.count("expected_sharded_qa_count=expected_sharded_qa_count") == 2
    assert source.count("expected_kv_a_count=expected_kv_a_count") == 2
    assert source.count("kv_a_width=kv_lora_rank") == 2
    assert source.count("kv_a_projection_width=kv_lora_rank + qk_rope_head_dim") == 2


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("wrong_width", "f32[1,512]", "f32[1,256]"),
        ("partial_reduce_axis", "reduce(%ksq, %kz), dimensions={0,1}", "reduce(%ksq, %kz), dimensions={0}"),
        ("max_combiner", "reduce(%ksq, %kz), dimensions={0,1}, to_apply=%add_combiner", "reduce(%ksq, %kz), dimensions={0,1}, to_apply=%max_combiner"),
        ("wrong_scale", "constant(0.001953125)", "constant(0.00390625)"),
        ("wrong_epsilon", "%keps = f32[] constant(1e-05)", "%keps = f32[] constant(1e-06)"),
        ("not_a_square", "%ksq = f32[1,512]{1,0:T(1,128)} multiply(%kc, %kc)", "%ksq = f32[1,512]{1,0:T(1,128)} add(%kc, %kc)"),
        ("not_the_converted_latent", "%ksq = f32[1,512]{1,0:T(1,128)} multiply(%kc, %kc)", "%kneg = f32[1,512]{1,0:T(1,128)} negate(%kc)\n  %ksq = f32[1,512]{1,0:T(1,128)} multiply(%kneg, %kneg)"),
        ("row_scale", "ROOT %krs = f32[] rsqrt(%kvar)", "ROOT %krs = f32[2] rsqrt(%kvar)"),
    ),
)
def test_hlo_kv_a_row_norm_mutations_are_refused(label: str, old: str, new: str) -> None:
    assert old in HLO, label
    result = _hlo(HLO.replace(old, new))
    assert not result["passed"], (label, result)
    assert result["kv_a_rsqrt_count"] == 0, (label, result)


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("wrong_width", "across dimensions = [1] : (tensor<1x512xf32>", "across dimensions = [1] : (tensor<1x256xf32>"),
        ("wrong_axis", "across dimensions = [1] : (tensor<1x512xf32>", "across dimensions = [0] : (tensor<1x512xf32>"),
        ("max_combiner", "applies stablehlo.add across dimensions = [1] : (tensor<1x512xf32>", "applies stablehlo.maximum across dimensions = [1] : (tensor<1x512xf32>"),
        ("wrong_scale", "dense<5.120000e+02>", "dense<2.560000e+02>"),
        ("scale_opcode_swapped", "%610 = stablehlo.divide %608, %609", "%610 = stablehlo.multiply %608, %609"),
        ("wrong_epsilon", "%cst_172 = stablehlo.constant dense<9.99999974E-6>", "%cst_172 = stablehlo.constant dense<9.99999997E-7>"),
        ("not_converted_latent", "%606 = chlo.square %605", "%606 = chlo.square %604"),
        ("row_scale", "%613 = stablehlo.rsqrt %612 : tensor<1x1xf32>", "%613 = stablehlo.rsqrt %612 : tensor<2x1xf32>"),
    ),
)
def test_stablehlo_kv_a_row_norm_mutations_are_refused(label: str, old: str, new: str) -> None:
    assert old in STABLEHLO, label
    result = _stable(STABLEHLO.replace(old, new, 1))
    assert not result["passed"], (label, result)
    assert result["kv_a_rsqrt_count"] == 0, (label, result)


# --- kv-a provenance: the latent must be the [0:1, 0:512] slice of the 576-wide projection ---
HLO_KV_SLICE_AFTER_CONVERT = HLO.replace(
    """  %ks = bf16[1,512]{1,0:T(2,128)(2,1)} slice(%kp), slice={[0:1], [0:512]}
  %kc = f32[1,512]{1,0:T(1,128)} convert(%ks)
""",
    """  %kcf = f32[1,576]{1,0:T(1,128)} convert(%kp)
  %kc = f32[1,512]{1,0:T(1,128)} slice(%kcf), slice={[0:1], [0:512]}
""",
)
STABLE_KV_SLICE_AFTER_CONVERT = STABLEHLO.replace(
    """    %604 = stablehlo.slice %arg3 [0:1, 0:512] : (tensor<1x576xbf16>) -> tensor<1x512xbf16>
    %605 = stablehlo.convert %604 : (tensor<1x512xbf16>) -> tensor<1x512xf32>
""",
    """    %604 = stablehlo.convert %arg3 : (tensor<1x576xbf16>) -> tensor<1x576xf32>
    %605 = stablehlo.slice %604 [0:1, 0:512] : (tensor<1x576xf32>) -> tensor<1x512xf32>
""",
)


def test_kv_a_provenance_accepts_both_emitted_orders() -> None:
    assert HLO_KV_SLICE_AFTER_CONVERT != HLO and STABLE_KV_SLICE_AFTER_CONVERT != STABLEHLO
    for text in (HLO, HLO_KV_SLICE_AFTER_CONVERT):
        result = _hlo(text)
        assert result["passed"], result["violations"]
        assert result["kv_a_rsqrt_count"] == 1
    for text in (STABLEHLO, STABLE_KV_SLICE_AFTER_CONVERT):
        result = _stable(text)
        assert result["passed"], result["violations"]
        assert result["kv_a_rsqrt_count"] == 1


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("direct_parameter_no_slice", "  %ks = bf16[1,512]{1,0:T(2,128)(2,1)} slice(%kp), slice={[0:1], [0:512]}\n  %kc = f32[1,512]{1,0:T(1,128)} convert(%ks)\n",
         "  %kc = f32[1,512]{1,0:T(1,128)} convert(%kdirect)\n"),
        ("unrelated_576_parameter", "  %kv_sum = f32[] fusion(%kvproj), kind=kLoop, calls=%kv_reduce", "  %kv_sum = f32[] fusion(%kvparam), kind=kLoop, calls=%kv_reduce"),
        ("caller_substitution_add", "  %kv_sum = f32[] fusion(%kvproj), kind=kLoop, calls=%kv_reduce", "  %kvsum2 = bf16[1,576]{1,0:T(2,128)(2,1)} add(%kvproj, %kvproj)\n  %kv_sum = f32[] fusion(%kvsum2), kind=kLoop, calls=%kv_reduce"),
        ("wrong_lane_slice", "slice(%qproj), slice={[0:32], [0:1], [64:82]}", "slice(%qproj), slice={[0:32], [0:1], [46:64]}"),
        ("wrong_projection_buffer", "  %kvlanes = bf16[32,1,18]{2,1,0:T(2,128)(2,1)} slice(%qproj), slice={[0:32], [0:1], [64:82]}\n", "  %kvlanes = bf16[32,1,18]{2,1,0:T(2,128)(2,1)} slice(%qother), slice={[0:32], [0:1], [64:82]}\n"),
        ("same_shape_entry_buffer", "  %kvlanes = bf16[32,1,18]{2,1,0:T(2,128)(2,1)} slice(%qproj), slice={[0:32], [0:1], [64:82]}\n", "  %kvlanes = bf16[32,1,18]{2,1,0:T(2,128)(2,1)} slice(%qsame), slice={[0:32], [0:1], [64:82]}\n"),
        ("loop_without_n82_convolution", "  %bconv = f32[1,82]{1,0} convolution(%bh, %bw), dim_labels=bf_io->bf\n  %brow = bf16[1,82]{1,0} convert(%bconv)\n", "  %bslice = bf16[1,82]{1,0} slice(%bw), slice={[0:1], [0:82]}\n  %brow = bf16[1,82]{1,0} copy(%bslice)\n"),
        ("loop_wrong_hidden", "6000", "6000"),
        ("dead_convolution_unrelated_update", "  %bacc2 = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} dynamic-update-slice(%bacc, %brow3, %bi, %bzero, %bzero)\n",
         "  %bother = bf16[1,82]{1,0} slice(%bw), slice={[0:1], [0:82]}\n  %bother3 = bf16[1,1,82]{2,1,0} reshape(%bother)\n  %bacc2 = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} dynamic-update-slice(%bacc, %bother3, %bi, %bzero, %bzero)\n"),
        ("accumulator_not_loop_carried", "  %bacc2 = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} dynamic-update-slice(%bacc, %brow3, %bi, %bzero, %bzero)\n",
         "  %bfresh = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} broadcast(%bzero_b), dimensions={}\n  %bacc2 = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} dynamic-update-slice(%bfresh, %brow3, %bi, %bzero, %bzero)\n"),
        ("result_index_not_the_accumulator", "  %qproj = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} get-tuple-element(%loop), index=1\n",
         "  %qproj0 = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} get-tuple-element(%loop), index=1\n  %qproj = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} get-tuple-element(%loop), index=5\n"),
        ("alternate_hidden_operand", "  %bconv = f32[1,82]{1,0} convolution(%bh, %bw), dim_labels=bf_io->bf\n", "  %bhz = bf16[] constant(1)\n  %bhalt = bf16[1,6144]{1,0} broadcast(%bhz), dimensions={}\n  %bconv = f32[1,82]{1,0} convolution(%bhalt, %bw), dim_labels=bf_io->bf\n"),
        ("alternate_weight_operand", "  %bconv = f32[1,82]{1,0} convolution(%bh, %bw), dim_labels=bf_io->bf\n", "  %bwz = bf16[] constant(1)\n  %bwalt = bf16[6144,82]{1,0} broadcast(%bwz), dimensions={}\n  %bconv = f32[1,82]{1,0} convolution(%bh, %bwalt), dim_labels=bf_io->bf\n"),
        ("weight_from_undequantized_scales_only", "  %bwdq = f32[6144,82]{1,0} multiply(%bwf32, %bsw)\n", "  %bwdq = f32[6144,82]{1,0} multiply(%bsw, %bsw)\n"),
        ("hidden_from_wrong_loop_element", "  %bconv = f32[1,82]{1,0} convolution(%bh, %bw), dim_labels=bf_io->bf\n", "  %bhw = bf16[1,6144]{1,0} slice(%bwt), slice={[0:1], [0:6144]}\n  %bconv = f32[1,82]{1,0} convolution(%bhw, %bw), dim_labels=bf_io->bf\n"),
        ("wrong_parent_width", "%kp = bf16[1,576]{1,0:T(2,128)(2,1)} parameter(0)", "%kp = bf16[1,640]{1,0:T(2,128)(2,1)} parameter(0)"),
        ("wrong_slice_bounds", "slice(%kp), slice={[0:1], [0:512]}", "slice(%kp), slice={[0:1], [64:576]}"),
        ("unrelated_512_lineage", "  %kc = f32[1,512]{1,0:T(1,128)} convert(%ks)\n", "  %kc0 = f32[1,512]{1,0:T(1,128)} convert(%ks)\n  %kc = f32[1,512]{1,0:T(1,128)} add(%kc0, %kc0)\n"),
        ("second_slice", "  %kc = f32[1,512]{1,0:T(1,128)} convert(%ks)\n", "  %ks2 = bf16[1,512]{1,0:T(2,128)(2,1)} slice(%ks), slice={[0:1], [0:512]}\n  %kc = f32[1,512]{1,0:T(1,128)} convert(%ks2)\n"),
    ),
)
def test_hlo_kv_a_provenance_substitutions_are_refused(label: str, old: str, new: str) -> None:
    text = HLO
    ENTRY_SIG = "scales: f32[32,48,82]) -> (f32[32], f32[1]) {"
    if label == "direct_parameter_no_slice":
        text = text.replace("%kv_reduce (kp: bf16[1,576]) -> f32[] {\n  %kp = bf16[1,576]{1,0:T(2,128)(2,1)} parameter(0)\n",
                            "%kv_reduce (kp: bf16[1,576], kdirect: bf16[1,512]) -> f32[] {\n  %kp = bf16[1,576]{1,0:T(2,128)(2,1)} parameter(0)\n  %kdirect = bf16[1,512]{1,0:T(2,128)(2,1)} parameter(1)\n")
        text = text.replace("%kv_sum = f32[] fusion(%kvproj), kind=kLoop, calls=%kv_reduce", "%kdirect_in = bf16[1,512]{1,0:T(2,128)(2,1)} parameter(4)\n  %kv_sum = f32[] fusion(%kvproj, %kdirect_in), kind=kLoop, calls=%kv_reduce")
        text = text.replace(ENTRY_SIG, "scales: f32[32,48,82], kdirect_in: bf16[1,512]) -> (f32[32], f32[1]) {")
    if label == "unrelated_576_parameter":
        text = text.replace(ENTRY_SIG, "scales: f32[32,48,82], kvparam: bf16[1,576]) -> (f32[32], f32[1]) {")
        text = text.replace("  %scales = f32[32,48,82]{2,1,0} parameter(4)\n", "  %scales = f32[32,48,82]{2,1,0} parameter(4)\n  %kvparam = bf16[1,576]{1,0:T(2,128)(2,1)} parameter(5)\n")
    if label == "wrong_projection_buffer":
        text = text.replace(ENTRY_SIG, "scales: f32[32,48,82], qother: bf16[32,1,90]) -> (f32[32], f32[1]) {")
        text = text.replace("  %scales = f32[32,48,82]{2,1,0} parameter(4)\n", "  %scales = f32[32,48,82]{2,1,0} parameter(4)\n  %qother = bf16[32,1,90]{2,1,0:T(2,128)(2,1)} parameter(5)\n")
    if label == "same_shape_entry_buffer":
        text = text.replace(ENTRY_SIG, "scales: f32[32,48,82], qsame: bf16[32,1,82]) -> (f32[32], f32[1]) {")
        text = text.replace("  %scales = f32[32,48,82]{2,1,0} parameter(4)\n", "  %scales = f32[32,48,82]{2,1,0} parameter(4)\n  %qsame = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} parameter(5)\n")
    if label == "loop_wrong_hidden":
        text = text.replace("6144", "6000")
    if label == "accumulator_not_loop_carried":
        text = text.replace("  %bzero = s32[] constant(0)\n", "  %bzero = s32[] constant(0)\n  %bzero_b = bf16[] constant(0)\n")
    if label == "hidden_from_wrong_loop_element":
        text = text.replace("  %bconv = f32[1,82]{1,0} convolution(%bh, %bw), dim_labels=bf_io->bf\n", "  %bwt0 = bf16[82,6144]{1,0} transpose(%bw), dimensions={1,0}\n  %bwt = bf16[82,6144]{1,0} copy(%bwt0)\n  %bconv = f32[1,82]{1,0} convolution(%bh, %bw), dim_labels=bf_io->bf\n")
    if label == "result_index_not_the_accumulator":
        text = text.replace("(s32[], bf16[32,1,82]{2,1,0:T(2,128)(2,1)}, bf16[1,6144]{1,0}, u8[32,6144,82]{2,1,0}, f32[32,48,82]{2,1,0})", "(s32[], bf16[32,1,82]{2,1,0:T(2,128)(2,1)}, bf16[1,6144]{1,0}, u8[32,6144,82]{2,1,0}, f32[32,48,82]{2,1,0}, bf16[32,1,82]{2,1,0:T(2,128)(2,1)})")
        text = text.replace("(bp: (s32[], bf16[32,1,82], bf16[1,6144], u8[32,6144,82], f32[32,48,82])) -> (s32[], bf16[32,1,82], bf16[1,6144], u8[32,6144,82], f32[32,48,82])", "(bp: (s32[], bf16[32,1,82], bf16[1,6144], u8[32,6144,82], f32[32,48,82], bf16[32,1,82])) -> (s32[], bf16[32,1,82], bf16[1,6144], u8[32,6144,82], f32[32,48,82], bf16[32,1,82])")
        text = text.replace("(cp: (s32[], bf16[32,1,82], bf16[1,6144], u8[32,6144,82], f32[32,48,82])) -> pred[]", "(cp: (s32[], bf16[32,1,82], bf16[1,6144], u8[32,6144,82], f32[32,48,82], bf16[32,1,82])) -> pred[]")
        text = text.replace("  %bscales = f32[32,48,82]{2,1,0} get-tuple-element(%bp), index=4\n", "  %bscales = f32[32,48,82]{2,1,0} get-tuple-element(%bp), index=4\n  %bextra = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} get-tuple-element(%bp), index=5\n")
        text = text.replace("tuple(%bi2, %bacc2, %bh, %bpacked, %bscales)", "tuple(%bi2, %bacc2, %bh, %bpacked, %bscales, %bextra)")
        text = text.replace("tuple(%izero, %acc0, %hidden, %packed, %scales)", "tuple(%izero, %acc0, %hidden, %packed, %scales, %acc0)")
        text = text.replace("get-tuple-element(%loop), index=4\n", "get-tuple-element(%loop), index=5\n")
    assert old in text, label
    result = _hlo(text.replace(old, new, 1))
    assert not result["passed"], (label, result)
    assert result["kv_a_rsqrt_count"] == 0, (label, result)


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("direct_parameter_no_slice", "    %604 = stablehlo.slice %arg3 [0:1, 0:512] : (tensor<1x576xbf16>) -> tensor<1x512xbf16>\n    %605 = stablehlo.convert %604 : (tensor<1x512xbf16>) -> tensor<1x512xf32>\n",
         "    %605 = stablehlo.convert %arg7 : (tensor<1x512xbf16>) -> tensor<1x512xf32>\n"),
        ("same_shape_entry_buffer", "%227 = stablehlo.slice %700#4 [0:32, 0:1, 64:82]", "%227 = stablehlo.slice %arg8 [0:32, 0:1, 64:82]"),
        ("loop_without_n82_call", "%17676 = func.call @closed_call(%iterArg_2439, %17674, %17675) : (tensor<1x6144xbf16>, tensor<6144x82xui8>, tensor<48x82xf32>) -> tensor<1x82xbf16>", "%17676 = func.call @other_call(%iterArg_2439) : (tensor<1x6144xbf16>) -> tensor<1x82xbf16>"),
        ("loop_wrong_hidden", "6000", "6000"),
        ("dead_call_unrelated_update", "DEAD_CALL_PLACEHOLDER", "DEAD_CALL_PLACEHOLDER"),
        ("accumulator_not_the_iterarg", "func.call @dynamic_update_index_in_dim(%iterArg_2441, %17676, %iterArg_2440)", "func.call @dynamic_update_index_in_dim(%206, %17676, %iterArg_2440)"),
        ("callee_return_not_rooted_in_convolution", "    return %7 : tensor<1x82xbf16>\n", "    %8 = stablehlo.convert %arg2 : (tensor<48x82xf32>) -> tensor<48x82xbf16>\n    %9 = stablehlo.slice %8 [0:1, 0:82] : (tensor<48x82xbf16>) -> tensor<1x82xbf16>\n    return %9 : tensor<1x82xbf16>\n"),
        ("callee_alternate_hidden_operand", "    %6 = stablehlo.convolution(%arg0, %5)", "    %h0 = stablehlo.constant dense<1.000000e+00> : tensor<1x6144xbf16>\n    %6 = stablehlo.convolution(%h0, %5)"),
        ("callee_alternate_weight_operand", "    %6 = stablehlo.convolution(%arg0, %5)", "    %w0 = stablehlo.constant dense<1.000000e+00> : tensor<6144x82xbf16>\n    %6 = stablehlo.convolution(%arg0, %w0)"),
        ("callee_weight_without_packed_argument", "    %4 = stablehlo.multiply %3, %2 : tensor<6144x82xf32>\n", "    %4 = stablehlo.multiply %2, %2 : tensor<6144x82xf32>\n"),
        ("call_hidden_not_iterarg", "func.call @closed_call(%iterArg_2439, %17674, %17675)", "func.call @closed_call(%hz, %17674, %17675)"),
        ("call_weight_not_from_packed_iterarg", "%17674 = func.call @dynamic_index_in_dim(%iterArg, %iterArg_2440)", "%17674 = func.call @dynamic_index_in_dim(%arg9, %iterArg_2440)"),
        ("unrelated_576_parameter", "    %604 = stablehlo.slice %arg3 [0:1, 0:512]", "    %604 = stablehlo.slice %arg5 [0:1, 0:512]"),
        ("caller_substitution_add", "    %604 = stablehlo.slice %arg3 [0:1, 0:512]", "    %arg3b = stablehlo.add %arg3, %arg3 : tensor<1x576xbf16>\n    %604 = stablehlo.slice %arg3b [0:1, 0:512]"),
        ("wrong_lane_slice", "%227 = stablehlo.slice %700#4 [0:32, 0:1, 64:82]", "%227 = stablehlo.slice %700#4 [0:32, 0:1, 46:64]"),
        ("wrong_projection_buffer", "%227 = stablehlo.slice %700#4 [0:32, 0:1, 64:82] : (tensor<32x1x82xbf16>)", "%227 = stablehlo.slice %arg6 [0:32, 0:1, 64:82] : (tensor<32x1x90xbf16>)"),
        ("wrong_parent_width", "(tensor<1x576xbf16>) -> tensor<1x512xbf16>", "(tensor<1x640xbf16>) -> tensor<1x512xbf16>"),
        ("wrong_slice_bounds", "%arg3 [0:1, 0:512] : (tensor<1x576xbf16>)", "%arg3 [0:1, 64:576] : (tensor<1x576xbf16>)"),
        ("unrelated_512_lineage", "    %606 = chlo.square %605 : tensor<1x512xf32> -> tensor<1x512xf32>\n", "    %605b = stablehlo.add %605, %605 : tensor<1x512xf32>\n    %606 = chlo.square %605b : tensor<1x512xf32> -> tensor<1x512xf32>\n"),
    ),
)
def test_stablehlo_kv_a_provenance_substitutions_are_refused(label: str, old: str, new: str) -> None:
    text = STABLEHLO.replace(
        "%arg4: tensor<1x6144xbf16>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {",
        "%arg4: tensor<1x6144xbf16>, %arg7: tensor<1x512xbf16>, %arg5: tensor<1x576xbf16>, %arg6: tensor<32x1x90xbf16>, %arg8: tensor<32x1x82xbf16>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {",
    ).replace("%605 = stablehlo.convert %arg4 : (tensor<1x512xbf16>)", "%605 = stablehlo.convert %arg7 : (tensor<1x512xbf16>)")
    if label == "loop_wrong_hidden":
        text = text.replace("6144", "6000")
    if label == "call_hidden_not_iterarg":
        text = text.replace("    } do {\n", "    } do {\n      %hz = stablehlo.constant dense<1.000000e+00> : tensor<1x6144xbf16>\n")
    if label == "call_weight_not_from_packed_iterarg":
        text = text.replace("%arg8: tensor<32x1x82xbf16>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {", "%arg8: tensor<32x1x82xbf16>, %arg9: tensor<32x6144x82xui8>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {")
    if label == "dead_call_unrelated_update":
        text = text.replace("      %c_2443 = stablehlo.constant dense<1> : tensor<i32>\n", "      %17679 = stablehlo.slice %17675 [0:1, 0:82] : (tensor<48x82xf32>) -> tensor<1x82xf32>\n      %17679b = stablehlo.convert %17679 : (tensor<1x82xf32>) -> tensor<1x82xbf16>\n      %c_2443 = stablehlo.constant dense<1> : tensor<i32>\n")
        old = "func.call @dynamic_update_index_in_dim(%iterArg_2441, %17676, %iterArg_2440)"
        new = "func.call @dynamic_update_index_in_dim(%iterArg_2441, %17679b, %iterArg_2440)"
    assert old in text, label
    result = _stable(text.replace(old, new, 1))
    assert not result["passed"], (label, result)
    assert result["kv_a_rsqrt_count"] == 0, (label, result)


# --- separate layout: the kv-a projection is a dot contracting the hidden size ---
HLO_SEPARATE = RMS_HLO.replace(
    "ENTRY %main (carry: f32[32,128], key: f32[1,128]) -> (f32[32], f32[1]) {",
    """%kv_reduce_s (kp: bf16[1,576]) -> f32[] {
  %kp = bf16[1,576]{1,0:T(2,128)(2,1)} parameter(0)
  %ks = bf16[1,512]{1,0:T(2,128)(2,1)} slice(%kp), slice={[0:1], [0:512]}
  %kc = f32[1,512]{1,0:T(1,128)} convert(%ks)
  %ksq = f32[1,512]{1,0:T(1,128)} multiply(%kc, %kc)
  %kz = f32[] constant(0)
  ROOT %ksum = f32[] reduce(%ksq, %kz), dimensions={0,1}, to_apply=%add_combiner
}

%kv_scale_s (kq: f32[]) -> f32[] {
  %kq = f32[] parameter(0)
  %kinv2 = f32[] constant(0.001953125)
  %kmean2 = f32[] multiply(%kq, %kinv2)
  %keps = f32[] constant(1e-05)
  %kvar = f32[] add(%kmean2, %keps)
  ROOT %krs = f32[] rsqrt(%kvar)
}

ENTRY %main (carry: f32[32,128], key: f32[1,128], hidden_row: f32[6144], kvw: f32[576,6144]) -> (f32[32], f32[1]) {
  %hidden_row = f32[6144]{0} parameter(2)
  %kvw = f32[576,6144]{1,0} parameter(3)
  %kvdot = f32[576]{0} dot(%hidden_row, %kvw), lhs_contracting_dims={0}, rhs_contracting_dims={1}
  %kvbf = bf16[576]{0} convert(%kvdot)
  %kvproj = bf16[1,576]{1,0:T(2,128)(2,1)} reshape(%kvbf)
  %kv_sum = f32[] fusion(%kvproj), kind=kLoop, calls=%kv_reduce_s
  %kv_scale.1 = f32[] fusion(%kv_sum), kind=kLoop, calls=%kv_scale_s""",
)
SEPARATE_CENSUS = dict(expected_accepted_count=1, expected_sharded_qa_count=0, expected_kv_a_count=1)


def _hlo_separate(text: str) -> dict:
    return _validate_rms_accepted_schedule_hlo(
        parse_hlo_module(text), enabled=True, layernorm_width=128, attention_projection_backend="separate", **SEPARATE_CENSUS
    )


def test_separate_layout_kv_a_dot_is_bound_by_its_hidden_contraction() -> None:
    assert HLO_SEPARATE != RMS_HLO
    result = _hlo_separate(HLO_SEPARATE)
    assert result["passed"], result["violations"]
    assert result["kv_a_rsqrt_count"] == 1
    # unrelated P-wide dot: wrong contraction size, or wrong operand rank arrangement
    wrong_contraction = HLO_SEPARATE.replace("f32[6144]", "f32[6000]").replace("f32[576,6144]", "f32[576,6000]")
    assert wrong_contraction != HLO_SEPARATE
    refused = _hlo_separate(wrong_contraction)
    assert not refused["passed"] and refused["kv_a_rsqrt_count"] == 0
    assert any("does not contract the hidden size 6144" in v for v in refused["violations"])
    unrelated_dot = HLO_SEPARATE.replace("%kvdot = f32[576]{0} dot(%hidden_row, %kvw), lhs_contracting_dims={0}, rhs_contracting_dims={1}", "%kvdot0 = f32[576]{0} dot(%hidden_row, %kvw), lhs_contracting_dims={0}, rhs_contracting_dims={1}\n  %kvdot = f32[576]{0} add(%kvdot0, %kvdot0)")
    refused = _hlo_separate(unrelated_dot)
    assert not refused["passed"] and refused["kv_a_rsqrt_count"] == 0
    # the fused rule refuses a dot-produced projection and the separate rule refuses the fused loop
    assert not _hlo(HLO_SEPARATE, expected_accepted_count=1, expected_sharded_qa_count=0, expected_kv_a_count=1)["passed"]
    assert not _validate_rms_accepted_schedule_hlo(parse_hlo_module(HLO), enabled=True, layernorm_width=128, attention_projection_backend="separate", expected_accepted_count=1, expected_sharded_qa_count=1, expected_kv_a_count=1)["passed"]
