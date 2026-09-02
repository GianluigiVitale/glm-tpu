"""The one-live-row contracts admit only accepted-schedule RMS carries, bound by lineage.

The 2026-09-02 protected 8K run compiled the full decoder on TPU and failed
closed before execution: every ``f32[32,6144]`` accepted-schedule carry (pad,
square, reduce, normalize fusions) was refused as a dead-row tensor, and the
fused qkv-a q-a norm still used its sharded reduce.  These tests bind the
allowance to conforming lineages on a synthetic module and on the archived
TPU bytes of that run.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from glm_tpu.greenfield.runtime.decoder import (
    _classify_decoder_live_tensor_shapes,
    _classify_scoped_shape_occurrences,
    _rms_accepted_schedule_carry_allowances,
    _rms_schedule_lineages,
    _validate_fused_qkv_a_decoder_association,
    _validate_rms_accepted_schedule_hlo,
    _validate_rms_accepted_schedule_stablehlo,
    expected_rms_schedule_census,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.runtime.test_decoder import _real_8k_decoder_config
from tests.greenfield.runtime.test_rms_schedule_layernorm_classification import HLO as SYNTHETIC_HLO

_RUN_PREFIX = (
    "/home/gianl/glm-run/greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_"
    "token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_"
    "strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_"
)
ARCHIVE = Path(_RUN_PREFIX + "20260902T094725673772375Z/hlo/decoder_78layer_8k_token.optimized_hlo.txt.gz")
# Second refused run (pin 978dc3a9): all 313 norms scheduled; only the q-a producer's BF16 pad remained.
ARCHIVE_2 = Path(_RUN_PREFIX + "20260902T113001516858996Z/hlo/decoder_78layer_8k_token.optimized_hlo.txt.gz")
# Baseline run with every norm on its default schedule (rotary-table refusal run of 02:13Z).
BASELINE = Path(
    "/home/gianl/glm-run/greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_"
    "token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_dr_pregatheredb512_"
    "strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T021346091708582Z/hlo/"
    "decoder_78layer_8k_token.optimized_hlo.txt.gz"
)


FUSED_HLO = """HloModule synthetic_fused_carry, entry_computation_layout={(bf16[1,128]{1,0}, f32[32,128]{1,0:T(8,128)})->f32[1,128]{1,0}}

%add_combiner (a: f32[], b: f32[]) -> f32[] {
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %s = f32[] add(%a, %b)
}

%carry_body (xp: bf16[1,128]) -> f32[32,128] {
  %xp = bf16[1,128]{1,0} parameter(0)
  %zero_b = bf16[] constant(0)
  %pad.1 = bf16[32,128]{1,0:T(8,128)(2,1)} pad(%xp, %zero_b), padding=0_31x0_0
  %cv = f32[32,128]{1,0:T(8,128)} convert(%pad.1)
  %rogue_body_c = f32[] constant(3)
  %rogue_body = f32[32,128]{1,0:T(8,128)} broadcast(%rogue_body_c), dimensions={}
  ROOT %carry_add = f32[32,128]{1,0:T(8,128)} add(%cv, %cv)
}

%rms_reduce (p: f32[32,128], r: f32[32,128]) -> f32[32] {
  %p = f32[32,128]{1,0:T(8,128)} parameter(0)
  %r = f32[32,128]{1,0:T(8,128)} parameter(1)
  %rogue_sq2 = f32[32,128]{1,0:T(8,128)} multiply(%r, %r)
  %sq = f32[32,128]{1,0:T(8,128)} multiply(%p, %p)
  %z = f32[] constant(0)
  ROOT %rsum = f32[32]{0} reduce(%sq, %z), dimensions={1}, to_apply=%add_combiner
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

%norm_body (n0: f32[32,128], n1: f32[32]) -> f32[1,128] {
  %n0 = f32[32,128]{1,0:T(8,128)} parameter(0)
  %n1 = f32[32]{0} parameter(1)
  %nb = f32[32,128]{1,0:T(8,128)} broadcast(%n1), dimensions={0}
  %rogue_n = f32[32,128]{1,0:T(8,128)} multiply(%n0, %n0)
  %nmul = f32[32,128]{1,0:T(8,128)} multiply(%n0, %nb)
  ROOT %nslice = f32[1,128]{1,0} slice(%nmul), slice={[0:1], [0:128]}
}

ENTRY %main (x: bf16[1,128], rogue: f32[32,128]) -> f32[1,128] {
  %x = bf16[1,128]{1,0} parameter(0)
  %rogue = f32[32,128]{1,0:T(8,128)} parameter(1)
  %carry = f32[32,128]{1,0:T(8,128)} fusion(%x), kind=kLoop, calls=%carry_body
  %rms_sum = f32[32]{0} fusion(%carry, %rogue), kind=kLoop, calls=%rms_reduce
  %rms_scale.1 = f32[32]{0} fusion(%rms_sum), kind=kLoop, calls=%rms_scale
  ROOT %normalized = f32[1,128]{1,0} fusion(%carry, %rms_scale.1), kind=kLoop, calls=%norm_body
}
"""


def _unallowed_names(module, allowances: dict, signature: str) -> set[str]:
    names = set()
    for item in module.instructions:
        for shape in item.operand_shapes + item.result_shapes:
            if f"{shape.dtype}[" + ",".join(str(v) for v in shape.dimensions) + "]" != signature:
                continue
            if signature not in allowances.get(item.index, set()):
                names.add(item.name)
    return names


def test_synthetic_carry_is_admitted_and_a_rogue_dead_row_tensor_is_not() -> None:
    rogue = SYNTHETIC_HLO.replace(
        "ENTRY %main (carry: f32[32,128], key: f32[1,128]) -> (f32[32], f32[1]) {\n  %carry = f32[32,128]{1,0:T(8,128)} parameter(0)\n",
        "ENTRY %main (carry: f32[32,128], key: f32[1,128], rogue: f32[32,128]) -> (f32[32], f32[1]) {\n  %carry = f32[32,128]{1,0:T(8,128)} parameter(0)\n  %rogue = f32[32,128]{1,0:T(8,128)} parameter(2)\n  %rogue_sq = f32[32,128]{1,0:T(8,128)} multiply(%rogue, %rogue)\n",
    )
    assert rogue != SYNTHETIC_HLO
    module = parse_hlo_module(rogue)
    by_key, lineages = _rms_schedule_lineages(module, layernorm_width=128)
    allowances, summary = _rms_accepted_schedule_carry_allowances(module, by_key, lineages)
    assert summary["carry_count"] == 1 and summary["widths"] == {"128": 1}
    allowed_repair, allowed_instruction, forbidden, _ = _classify_scoped_shape_occurrences(
        module,
        forbidden_signatures={"f32[32,128]"},
        repair_allowed_signatures=set(),
        prefill_index_repair=False,
        instruction_allowed_signatures=allowances,
    )
    # carry parameter, fusion operand, reduce-fusion parameter and square are admitted ...
    assert allowed_instruction["f32[32,128]"] >= 4
    # ... the rogue parameter and its square (operand + result shapes) are not.
    assert forbidden == {"f32[32,128]": 4}
    assert _unallowed_names(module, allowances, "f32[32,128]") == {"%rogue", "%rogue_sq"}
    assert _validate_rms_accepted_schedule_hlo(module, enabled=True, layernorm_width=128, expected_accepted_count=1, expected_sharded_qa_count=0, expected_kv_a_count=0)["passed"]


def test_rogues_colocated_inside_admitted_fusion_bodies_stay_forbidden() -> None:
    """Exact SSA slices: off-slice tensors inside the carry producer, the reduce fusion and the
    normalize fusion are refused even though they share those computations with the lineage."""

    module = parse_hlo_module(FUSED_HLO)
    contract = _validate_rms_accepted_schedule_hlo(module, enabled=True, layernorm_width=128, expected_accepted_count=1, expected_sharded_qa_count=0, expected_kv_a_count=0)
    assert contract["passed"], contract["violations"]
    assert contract["carry_summary"] == {"carry_count": 1, "allowed_instruction_count": contract["carry_summary"]["allowed_instruction_count"], "widths": {"128": 1}}
    by_key, lineages = _rms_schedule_lineages(module, layernorm_width=128)
    allowances, _ = _rms_accepted_schedule_carry_allowances(module, by_key, lineages)
    unallowed = _unallowed_names(module, allowances, "f32[32,128]")
    assert unallowed == {"%rogue", "%r", "%rogue_sq2", "%rogue_body", "%rogue_n"}, unallowed
    # The admitted set is exactly the lineage: carry body slice, carry, reduce slice, normalize slice.
    admitted = {module.instructions[i].name for i, sigs in allowances.items() if "f32[32,128]" in sigs}
    assert admitted == {"%cv", "%carry_add", "%carry", "%rms_sum", "%p", "%sq", "%rsum", "%normalized", "%n0", "%nb", "%nmul", "%nslice"}, admitted
    # The producer's BF16 pad of the live row (and the convert consuming it) is admitted only there.
    admitted_bf16 = {module.instructions[i].name for i, sigs in allowances.items() if "bf16[32,128]" in sigs}
    assert admitted_bf16 == {"%pad.1", "%cv"}, admitted_bf16
    assert _unallowed_names(module, allowances, "bf16[32,128]") == set()
    rogue_bf16 = FUSED_HLO.replace(
        "  %rogue = f32[32,128]{1,0:T(8,128)} parameter(1)\n",
        "  %rogue = f32[32,128]{1,0:T(8,128)} parameter(1)\n  %rogue_bf16 = bf16[32,128]{1,0:T(8,128)(2,1)} convert(%rogue)\n",
    )
    module_bf16 = parse_hlo_module(rogue_bf16)
    by_key_bf16, lineages_bf16 = _rms_schedule_lineages(module_bf16, layernorm_width=128)
    allowances_bf16, _ = _rms_accepted_schedule_carry_allowances(module_bf16, by_key_bf16, lineages_bf16)
    assert _unallowed_names(module_bf16, allowances_bf16, "bf16[32,128]") == {"%rogue_bf16"}


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archived TPU decoder HLO unavailable")
def test_archived_tpu_decoder_carries_are_all_explained_by_conforming_lineages() -> None:
    module = parse_hlo_module(gzip.open(ARCHIVE, "rt").read())
    config = _real_8k_decoder_config(dsa_score_default_precision=True)
    expected_accepted, expected_sharded, expected_kv = expected_rms_schedule_census(layers=78, attention_projection_backend="fused_n82_convolution")
    assert (expected_accepted, expected_sharded, expected_kv) == (157, 78, 78)
    census = dict(expected_accepted_count=expected_accepted, expected_sharded_qa_count=expected_sharded, expected_kv_a_count=expected_kv)
    rms = _validate_rms_accepted_schedule_hlo(module, enabled=True, layernorm_width=config.index_key_width, **census)
    # First run (pin a49e3ba1): 78 sharded q-a norms (own kind) but the 78 kv-a norms were on the
    # inferred [32,512] schedule — they count as accepted (235) and the kv-a row census is 0, so the
    # bound census REFUSES this module; the per-lineage checks themselves all bind.
    assert (rms["rsqrt_count"], rms["conforming_rsqrt_count"], rms["nonconforming_rsqrt_count"]) == (313, 235, 0)
    assert rms["sharded_qa_rsqrt_count"] == 78 and rms["layernorm_rsqrt_count"] == 0 and rms["kv_a_rsqrt_count"] == 0
    assert not rms["passed"]
    assert any("accepted-schedule RMS census drifted: expected 157, found 235" in v for v in rms["violations"])
    assert any("kv-a row-norm census drifted: expected 78, found 0" in v for v in rms["violations"])
    assert rms["carry_summary"]["carry_count"] == 235
    stable_text = gzip.open(ARCHIVE.with_name("decoder_78layer_8k_token.stablehlo.mlir.gz"), "rt").read()
    stable = _validate_rms_accepted_schedule_stablehlo(
        stable_text, enabled=True, layernorm_width=config.index_key_width, **census
    )
    assert not stable["passed"]
    assert (stable["rsqrt_count"], stable["conforming_rsqrt_count"], stable["sharded_qa_rsqrt_count"], stable["kv_a_rsqrt_count"], stable["barrier_count"]) == (313, 235, 78, 0, 235)
    # With the census this module actually carries (235 accepted, 78 sharded, 0 kv-a rows) both
    # representations pass every lineage check; any single drifted expectation is refused.
    carried = dict(expected_accepted_count=235, expected_sharded_qa_count=78, expected_kv_a_count=0)
    assert _validate_rms_accepted_schedule_hlo(module, enabled=True, layernorm_width=config.index_key_width, **carried)["passed"]
    assert _validate_rms_accepted_schedule_stablehlo(stable_text, enabled=True, layernorm_width=config.index_key_width, **carried)["passed"]
    assert not _validate_rms_accepted_schedule_hlo(module, enabled=True, layernorm_width=config.index_key_width, expected_accepted_count=236, expected_sharded_qa_count=78, expected_kv_a_count=0)["passed"]
    assert not _validate_rms_accepted_schedule_stablehlo(stable_text, enabled=True, layernorm_width=config.index_key_width, expected_accepted_count=235, expected_sharded_qa_count=77, expected_kv_a_count=0)["passed"]
    assert not _validate_rms_accepted_schedule_stablehlo(stable_text, enabled=True, layernorm_width=config.index_key_width, expected_accepted_count=235, expected_sharded_qa_count=78, expected_kv_a_count=1)["passed"]
    assert rms["carry_summary"]["widths"] == {"512": 78, "6144": 157}
    allowances = rms["carry_allowances"]
    common = dict(
        module=module,
        config=config,
        full_indexer_layers=21,
        backend_contract="tpu",
        prefill_index_repair=False,
        dsa_head_key_exact_association=True,
        strategy_nd_attention_projection=True,
    )
    without = _classify_decoder_live_tensor_shapes(**common)
    with_carries = _classify_decoder_live_tensor_shapes(**common, rms_accepted_schedule_carry_allowances=allowances)
    assert len(without["forbidden_shapes"]) == 1570
    assert with_carries["forbidden_shapes"] == []
    assert with_carries["allowed_rms_accepted_schedule_carry_count"] == 1570
    assert len(with_carries["allowed_exact_wk_feature_slices"]) == len(without["allowed_exact_wk_feature_slices"]) == 256
    assert len(with_carries["allowed_strategy_nd_virtual_partials"]) == len(without["allowed_strategy_nd_virtual_partials"])
    text = gzip.open(ARCHIVE, "rt").read()
    fused_without = _validate_fused_qkv_a_decoder_association(text, layers=78, dsa_head_key_exact_association=True, module=module)
    fused_with = _validate_fused_qkv_a_decoder_association(
        text, layers=78, dsa_head_key_exact_association=True, module=module, rms_accepted_schedule_carry_allowances=allowances
    )
    assert fused_without["forbidden_shapes"] == ["f32[32,6144]"]
    assert fused_with["forbidden_shapes"] == [] and fused_with["passed"]
    assert fused_with["allowed_exact_wk_feature_slice_shape_counts"]["f32[32,6144]"] > 256
    # Dropping one lineage's allowance re-exposes exactly that carry cone.
    partial = {index: sigs for index, sigs in allowances.items()}
    victim = next(iter(sorted(partial)))
    del partial[victim]
    exposed = _classify_decoder_live_tensor_shapes(**common, rms_accepted_schedule_carry_allowances=partial)
    assert 1 <= len(exposed["forbidden_shapes"]) <= 4


@pytest.mark.skipif(not ARCHIVE_2.exists(), reason="archived TPU decoder HLO (second run) unavailable")
def test_second_archived_tpu_decoder_schedules_all_313_norms_and_admits_the_bf16_pad() -> None:
    text = gzip.open(ARCHIVE_2, "rt").read()
    module = parse_hlo_module(text)
    config = _real_8k_decoder_config(dsa_score_default_precision=True)
    # This run wrongly scheduled the q-a norm (event 0 flipped): with the bound census the
    # 313-accepted / 0-sharded module must be REFUSED in both representations.
    census = dict(expected_accepted_count=157, expected_sharded_qa_count=78, expected_kv_a_count=78)
    rms = _validate_rms_accepted_schedule_hlo(module, enabled=True, layernorm_width=config.index_key_width, **census)
    assert not rms["passed"]
    assert any("optimized HLO accepted-schedule RMS census drifted: expected 157, found 313" in v for v in rms["violations"])
    assert any("optimized HLO kv-a row-norm census drifted: expected 78, found 0" in v for v in rms["violations"])
    assert any("optimized HLO virtual-TP32 sharded q-a census drifted: expected 78, found 0" in v for v in rms["violations"])
    assert (rms["rsqrt_count"], rms["conforming_rsqrt_count"], rms["nonconforming_rsqrt_count"], rms["sharded_qa_rsqrt_count"]) == (313, 313, 0, 0)
    stable = _validate_rms_accepted_schedule_stablehlo(
        gzip.open(ARCHIVE_2.with_name("decoder_78layer_8k_token.stablehlo.mlir.gz"), "rt").read(),
        enabled=True, layernorm_width=config.index_key_width, **census,
    )
    assert not stable["passed"] and stable["sharded_qa_rsqrt_count"] == 0 and stable["conforming_rsqrt_count"] == 313
    assert any("StableHLO accepted-schedule RMS census drifted: expected 157, found 313" in v for v in stable["violations"])
    assert rms["carry_summary"]["carry_count"] == 313
    assert rms["carry_summary"]["widths"] == {"2048": 78, "512": 78, "6144": 157}
    allowances = rms["carry_allowances"]
    common = dict(
        module=module,
        config=config,
        full_indexer_layers=21,
        backend_contract="tpu",
        prefill_index_repair=False,
        dsa_head_key_exact_association=True,
        strategy_nd_attention_projection=True,
    )
    without = _classify_decoder_live_tensor_shapes(**common)
    with_carries = _classify_decoder_live_tensor_shapes(**common, rms_accepted_schedule_carry_allowances=allowances)
    assert len(without["forbidden_shapes"]) > 1570
    assert with_carries["forbidden_shapes"] == [], with_carries["forbidden_shapes"][:3]
    assert with_carries["passed"]
    fused = _validate_fused_qkv_a_decoder_association(
        text, layers=78, dsa_head_key_exact_association=True, module=module, rms_accepted_schedule_carry_allowances=allowances
    )
    assert fused["passed"] and fused["forbidden_shapes"] == []


@pytest.mark.skipif(not BASELINE.exists(), reason="baseline TPU decoder HLO unavailable")
def test_baseline_archive_binds_78_sharded_qa_and_78_kv_a_row_norms_and_no_accepted_lineage() -> None:
    module = parse_hlo_module(gzip.open(BASELINE, "rt").read())
    config = _real_8k_decoder_config(dsa_score_default_precision=True)
    census = dict(expected_accepted_count=157, expected_sharded_qa_count=78, expected_kv_a_count=78)
    rms = _validate_rms_accepted_schedule_hlo(module, enabled=True, layernorm_width=config.index_key_width, **census)
    # Every q-a and kv-a norm is recognized in its legacy-faithful default form; the 157 hidden-width
    # norms are on the default single-row schedule, so the enabled contract refuses the module.
    assert (rms["rsqrt_count"], rms["sharded_qa_rsqrt_count"], rms["kv_a_rsqrt_count"], rms["conforming_rsqrt_count"], rms["nonconforming_rsqrt_count"]) == (313, 78, 78, 0, 157)
    assert not rms["passed"]
    assert any("accepted-schedule RMS census drifted: expected 157, found 0" in v for v in rms["violations"])
    stable = _validate_rms_accepted_schedule_stablehlo(
        gzip.open(BASELINE.with_name("decoder_78layer_8k_token.stablehlo.mlir.gz"), "rt").read(),
        enabled=True, layernorm_width=config.index_key_width, **census,
    )
    assert (stable["rsqrt_count"], stable["sharded_qa_rsqrt_count"], stable["kv_a_rsqrt_count"], stable["conforming_rsqrt_count"], stable["barrier_count"]) == (313, 78, 78, 0, 0)
    assert not stable["passed"]
    # Disabled mode (the baseline's own flag state) passes: no accepted lineage is present.
    assert _validate_rms_accepted_schedule_hlo(module, enabled=False, layernorm_width=config.index_key_width)["passed"]
