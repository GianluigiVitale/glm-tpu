"""The DSA key LayerNorm exemption of the RMS schedule contract binds its exact contract.

A synthetic module carries one accepted-schedule RMS lineage (materialized
``f32[32,128]`` carry, fusion-parameter square, last-axis add-reduce, 1/W,
1e-05) and one 128-wide centred LayerNorm (subtract, last-axis add-reduce,
1/128, 1e-06, one row).  With the flag on the module passes only while the
LayerNorm is exactly that; every mutation of its width, axis, scale, combiner,
epsilon, rows or centring must be refused rather than silently exempted.
"""

from __future__ import annotations

import pytest

from glm_tpu.greenfield.runtime.decoder import (
    _validate_rms_accepted_schedule_hlo,
    _validate_rms_accepted_schedule_stablehlo,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

HLO = """HloModule synthetic_rms_layernorm, entry_computation_layout={(f32[32,128]{1,0:T(8,128)}, f32[1,128]{1,0})->(f32[32]{0}, f32[1]{0})}

%add_combiner (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %s = f32[] add(%a, %b)
}

%max_combiner (c: f32[], d: f32[]) -> f32[] {
  %c = f32[] parameter(0)
  %d = f32[] parameter(1)
  ROOT %m = f32[] maximum(%c, %d)
}

%rms_reduce (p: f32[32,128]) -> f32[32] {
  %p = f32[32,128]{1,0:T(8,128)} parameter(0)
  %sq = f32[32,128]{1,0:T(8,128)} multiply(%p, %p)
  %z = f32[] constant(0)
  ROOT %r = f32[32]{0} reduce(%sq, %z), dimensions={1}, to_apply=%add_combiner
}

%rms_scale (q: f32[32]) -> f32[32] {
  %q = f32[32]{0} parameter(0)
  %invw = f32[] constant(0.0078125)
  %invw_b = f32[32]{0} broadcast(%invw), dimensions={}
  %mean = f32[32]{0} multiply(%q, %invw_b)
  %eps = f32[] constant(1e-05)
  %eps_b = f32[32]{0} broadcast(%eps), dimensions={}
  %var = f32[32]{0} add(%mean, %eps_b)
  ROOT %rs = f32[32]{0} rsqrt(%var)
}

ENTRY %main (carry: f32[32,128], key: f32[1,128]) -> (f32[32], f32[1]) {
  %carry = f32[32,128]{1,0:T(8,128)} parameter(0)
  %rms_sum = f32[32]{0} fusion(%carry), kind=kLoop, calls=%rms_reduce
  %rms_scale.1 = f32[32]{0} fusion(%rms_sum), kind=kLoop, calls=%rms_scale
  %key = f32[1,128]{1,0} parameter(1)
  %zero = f32[] constant(0)
  %ksum = f32[1]{0} reduce(%key, %zero), dimensions={1}, to_apply=%add_combiner
  %kinv = f32[] constant(0.0078125)
  %kinv_b = f32[1]{0} broadcast(%kinv), dimensions={}
  %kmean = f32[1]{0} multiply(%ksum, %kinv_b)
  %kmean_b = f32[1,128]{1,0} broadcast(%kmean), dimensions={0}
  %centered = f32[1,128]{1,0} subtract(%key, %kmean_b)
  %csq = f32[1,128]{1,0} multiply(%centered, %centered)
  %vsum = f32[1]{0} reduce(%csq, %zero), dimensions={1}, to_apply=%add_combiner
  %linv = f32[] constant(0.0078125)
  %linv_b = f32[1]{0} broadcast(%linv), dimensions={}
  %vmean = f32[1]{0} multiply(%vsum, %linv_b)
  %leps = f32[] constant(1e-06)
  %leps_b = f32[1]{0} broadcast(%leps), dimensions={}
  %lvar = f32[1]{0} add(%vmean, %leps_b)
  %lrs = f32[1]{0} rsqrt(%lvar)
  ROOT %out = (f32[32]{0}, f32[1]{0}) tuple(%rms_scale.1, %lrs)
}
"""

STABLEHLO = """module @synthetic {
  func.func public @main(%arg0: tensor<32x128xf32>, %arg1: tensor<1x128xf32>) -> (tensor<32x1xf32>, tensor<1x1xf32>) {
    %0 = stablehlo.optimization_barrier %arg0 : tensor<32x128xf32>
    %1 = chlo.square %0 : tensor<32x128xf32> -> tensor<32x128xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<32x128xf32>, tensor<f32>) -> tensor<32xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0] : (tensor<32xf32>) -> tensor<32x1xf32>
    %cst_0 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %4 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<32x1xf32>
    %5 = stablehlo.divide %3, %4 : tensor<32x1xf32>
    %cst_1 = stablehlo.constant dense<9.99999974E-6> : tensor<f32>
    %6 = stablehlo.broadcast_in_dim %cst_1, dims = [] : (tensor<f32>) -> tensor<32x1xf32>
    %7 = stablehlo.add %5, %6 : tensor<32x1xf32>
    %8 = stablehlo.rsqrt %7 : tensor<32x1xf32>
    %9 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %cst_2 = stablehlo.constant dense<1.280000e+02> : tensor<f32>
    %11 = stablehlo.broadcast_in_dim %cst_2, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %12 = stablehlo.divide %10, %11 : tensor<1x1xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<1x128xf32>
    %14 = stablehlo.subtract %arg1, %13 : tensor<1x128xf32>
    %15 = stablehlo.multiply %14, %14 : tensor<1x128xf32>
    %16 = stablehlo.reduce(%15 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0] : (tensor<1xf32>) -> tensor<1x1xf32>
    %18 = stablehlo.divide %17, %11 : tensor<1x1xf32>
    %cst_3 = stablehlo.constant dense<9.99999997E-7> : tensor<f32>
    %19 = stablehlo.broadcast_in_dim %cst_3, dims = [] : (tensor<f32>) -> tensor<1x1xf32>
    %20 = stablehlo.add %18, %19 : tensor<1x1xf32>
    %21 = stablehlo.rsqrt %20 : tensor<1x1xf32>
    return %8, %21 : tensor<32x1xf32>, tensor<1x1xf32>
  }
}
"""


CENSUS = dict(expected_accepted_count=1, expected_sharded_qa_count=0, expected_kv_a_count=0)


def _hlo(text: str, *, enabled: bool = True, width: int = 128) -> dict:
    return _validate_rms_accepted_schedule_hlo(
        parse_hlo_module(text), enabled=enabled, layernorm_width=width, **CENSUS
    )


def _stable(text: str, *, enabled: bool = True, width: int = 128) -> dict:
    return _validate_rms_accepted_schedule_stablehlo(
        text, enabled=enabled, layernorm_width=width, **CENSUS
    )


def test_exact_layernorm_is_classified_apart_and_the_rms_lineage_conforms() -> None:
    hlo = _hlo(HLO)
    assert hlo["passed"], hlo["violations"]
    assert (hlo["rsqrt_count"], hlo["conforming_rsqrt_count"], hlo["layernorm_rsqrt_count"]) == (2, 1, 1)
    assert hlo["tiled_module"] is True
    stable = _stable(STABLEHLO)
    assert stable["passed"], stable["violations"]
    assert (stable["rsqrt_count"], stable["conforming_rsqrt_count"], stable["layernorm_rsqrt_count"]) == (2, 1, 1)
    assert stable["barrier_count"] == 1
    # Flag off: the accepted RMS lineage is refused, the LayerNorm never counts as one.
    assert not _hlo(HLO, enabled=False)["passed"]
    assert not _stable(STABLEHLO, enabled=False)["passed"]
    # The decoder's own key width is what binds the exemption.
    assert not _hlo(HLO, width=64)["passed"]
    assert not _stable(STABLEHLO, width=64)["passed"]


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("wrong_width", "f32[1,128]", "f32[1,256]"),
        ("wrong_axis", "%vsum = f32[1]{0} reduce(%csq, %zero), dimensions={1}", "%vsum = f32[1]{0} reduce(%csq, %zero), dimensions={0}"),
        ("wrong_scale", "%linv = f32[] constant(0.0078125)", "%linv = f32[] constant(0.00390625)"),
        ("scale_opcode_swapped", "%vmean = f32[1]{0} multiply(%vsum, %linv_b)", "%vmean = f32[1]{0} divide(%vsum, %linv_b)"),
        ("max_combiner", "%vsum = f32[1]{0} reduce(%csq, %zero), dimensions={1}, to_apply=%add_combiner", "%vsum = f32[1]{0} reduce(%csq, %zero), dimensions={1}, to_apply=%max_combiner"),
        ("wrong_epsilon", "constant(1e-06)", "constant(1e-07)"),
        ("rms_epsilon_without_centering", "constant(1e-06)", "constant(1e-05)"),
        ("rows_mismatch", "%lrs = f32[1]{0} rsqrt(%lvar)", "%lrs = f32[2]{0} rsqrt(%lvar)"),
        ("uncentered_disguise", "%csq = f32[1,128]{1,0} multiply(%centered, %centered)", "%csq = f32[1,128]{1,0} multiply(%key, %key)"),
        ("not_a_square", "%csq = f32[1,128]{1,0} multiply(%centered, %centered)", "%csq = f32[1,128]{1,0} multiply(%centered, %key)"),
    ),
)
def test_hlo_layernorm_mutations_are_refused(label: str, old: str, new: str) -> None:
    assert old in HLO, label
    result = _hlo(HLO.replace(old, new))
    assert not result["passed"], (label, result)
    assert result["layernorm_rsqrt_count"] == 0, (label, result)


@pytest.mark.parametrize(
    ("label", "old", "new"),
    (
        ("wrong_width", "tensor<1x128xf32>", "tensor<1x256xf32>"),
        ("wrong_axis", "applies stablehlo.add across dimensions = [1] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>\n    %17", "applies stablehlo.add across dimensions = [0] : (tensor<1x128xf32>, tensor<f32>) -> tensor<1xf32>\n    %17"),
        ("wrong_scale", "%cst_2 = stablehlo.constant dense<1.280000e+02>", "%cst_2 = stablehlo.constant dense<6.400000e+01>"),
        ("scale_opcode_swapped", "%18 = stablehlo.divide %17, %11", "%18 = stablehlo.multiply %17, %11"),
        ("max_combiner", "%16 = stablehlo.reduce(%15 init: %cst) applies stablehlo.add", "%16 = stablehlo.reduce(%15 init: %cst) applies stablehlo.maximum"),
        ("wrong_epsilon", "dense<9.99999997E-7>", "dense<9.99999997E-8>"),
        ("rows_mismatch", "%21 = stablehlo.rsqrt %20 : tensor<1x1xf32>", "%21 = stablehlo.rsqrt %20 : tensor<2x1xf32>"),
        ("uncentered_disguise", "%15 = stablehlo.multiply %14, %14", "%15 = stablehlo.multiply %arg1, %arg1"),
        ("not_a_square", "%15 = stablehlo.multiply %14, %14", "%15 = stablehlo.multiply %14, %arg1"),
    ),
)
def test_stablehlo_layernorm_mutations_are_refused(label: str, old: str, new: str) -> None:
    assert old in STABLEHLO, label
    result = _stable(STABLEHLO.replace(old, new))
    assert not result["passed"], (label, result)
    assert result["layernorm_rsqrt_count"] == 0, (label, result)


def test_rms_scale_opcode_is_bound_per_opcode() -> None:
    # multiply by W (instead of 1/W) must not pass as a divide-by-W mean.
    swapped = HLO.replace("%mean = f32[32]{0} multiply(%q, %invw_b)", "%mean = f32[32]{0} divide(%q, %invw_b)")
    assert not _hlo(swapped)["passed"]
    swapped_stable = STABLEHLO.replace("%5 = stablehlo.divide %3, %4", "%5 = stablehlo.multiply %3, %4")
    assert not _stable(swapped_stable)["passed"]
