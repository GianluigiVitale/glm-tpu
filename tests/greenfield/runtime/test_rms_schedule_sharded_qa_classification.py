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

ENTRY %main (carry: f32[32,128], key: f32[1,128], qproj: bf16[32,1,82], kvproj: bf16[1,576]) -> (f32[32], f32[1]) {
  %qproj = bf16[32,1,82]{2,1,0:T(2,128)(2,1)} parameter(2)
  %kvproj = bf16[1,576]{1,0:T(2,128)(2,1)} parameter(3)
  %kv_sum = f32[] fusion(%kvproj), kind=kLoop, calls=%kv_reduce
  %kv_scale.1 = f32[] fusion(%kv_sum), kind=kLoop, calls=%kv_scale
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
KV_STABLE = """    %604 = stablehlo.slice %arg3 [0:1, 0:512] : (tensor<1x576xbf16>) -> tensor<1x512xbf16>
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
    "  func.func public @main(%arg0: tensor<32x128xf32>, %arg1: tensor<1x128xf32>, %arg2: tensor<32x1x82xbf16>, %arg3: tensor<1x576xbf16>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {\n" + QA_STABLE + KV_STABLE,
)


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
        ("wrong_parent_width", "%kp = bf16[1,576]{1,0:T(2,128)(2,1)} parameter(0)", "%kp = bf16[1,640]{1,0:T(2,128)(2,1)} parameter(0)"),
        ("wrong_slice_bounds", "slice(%kp), slice={[0:1], [0:512]}", "slice(%kp), slice={[0:1], [64:576]}"),
        ("unrelated_512_lineage", "  %kc = f32[1,512]{1,0:T(1,128)} convert(%ks)\n", "  %kc0 = f32[1,512]{1,0:T(1,128)} convert(%ks)\n  %kc = f32[1,512]{1,0:T(1,128)} add(%kc0, %kc0)\n"),
        ("second_slice", "  %kc = f32[1,512]{1,0:T(1,128)} convert(%ks)\n", "  %ks2 = bf16[1,512]{1,0:T(2,128)(2,1)} slice(%ks), slice={[0:1], [0:512]}\n  %kc = f32[1,512]{1,0:T(1,128)} convert(%ks2)\n"),
    ),
)
def test_hlo_kv_a_provenance_substitutions_are_refused(label: str, old: str, new: str) -> None:
    text = HLO
    if label == "direct_parameter_no_slice":
        text = text.replace("%kv_reduce (kp: bf16[1,576]) -> f32[] {\n  %kp = bf16[1,576]{1,0:T(2,128)(2,1)} parameter(0)\n",
                            "%kv_reduce (kp: bf16[1,576], kdirect: bf16[1,512]) -> f32[] {\n  %kp = bf16[1,576]{1,0:T(2,128)(2,1)} parameter(0)\n  %kdirect = bf16[1,512]{1,0:T(2,128)(2,1)} parameter(1)\n")
        text = text.replace("%kv_sum = f32[] fusion(%kvproj), kind=kLoop, calls=%kv_reduce", "%kdirect_in = bf16[1,512]{1,0:T(2,128)(2,1)} parameter(4)\n  %kv_sum = f32[] fusion(%kvproj, %kdirect_in), kind=kLoop, calls=%kv_reduce")
        text = text.replace("kvproj: bf16[1,576]) -> (f32[32], f32[1]) {", "kvproj: bf16[1,576], kdirect_in: bf16[1,512]) -> (f32[32], f32[1]) {")
    assert old in text, label
    result = _hlo(text.replace(old, new, 1))
    assert not result["passed"], (label, result)
    assert result["kv_a_rsqrt_count"] == 0, (label, result)


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("direct_parameter_no_slice", "    %604 = stablehlo.slice %arg3 [0:1, 0:512] : (tensor<1x576xbf16>) -> tensor<1x512xbf16>\n    %605 = stablehlo.convert %604 : (tensor<1x512xbf16>) -> tensor<1x512xf32>\n",
         "    %605 = stablehlo.convert %arg4 : (tensor<1x512xbf16>) -> tensor<1x512xf32>\n"),
        ("wrong_parent_width", "(tensor<1x576xbf16>) -> tensor<1x512xbf16>", "(tensor<1x640xbf16>) -> tensor<1x512xbf16>"),
        ("wrong_slice_bounds", "%arg3 [0:1, 0:512] : (tensor<1x576xbf16>)", "%arg3 [0:1, 64:576] : (tensor<1x576xbf16>)"),
        ("unrelated_512_lineage", "    %606 = chlo.square %605 : tensor<1x512xf32> -> tensor<1x512xf32>\n", "    %605b = stablehlo.add %605, %605 : tensor<1x512xf32>\n    %606 = chlo.square %605b : tensor<1x512xf32> -> tensor<1x512xf32>\n"),
    ),
)
def test_stablehlo_kv_a_provenance_substitutions_are_refused(label: str, old: str, new: str) -> None:
    text = STABLEHLO
    if label == "direct_parameter_no_slice":
        text = text.replace("%arg3: tensor<1x576xbf16>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {", "%arg3: tensor<1x576xbf16>, %arg4: tensor<1x512xbf16>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {")
    assert old in text, label
    result = _stable(text.replace(old, new, 1))
    assert not result["passed"], (label, result)
    assert result["kv_a_rsqrt_count"] == 0, (label, result)
