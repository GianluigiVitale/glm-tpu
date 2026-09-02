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
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.runtime.test_decoder import _real_8k_decoder_config
from tests.greenfield.runtime.test_rms_schedule_layernorm_classification import HLO as SYNTHETIC_HLO

ARCHIVE = Path(
    "/home/gianl/glm-run/greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_"
    "token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_"
    "strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T094725673772375Z/hlo/"
    "decoder_78layer_8k_token.optimized_hlo.txt.gz"
)


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
    assert _validate_rms_accepted_schedule_hlo(module, enabled=True, layernorm_width=128)["passed"]


@pytest.mark.skipif(not ARCHIVE.exists(), reason="archived TPU decoder HLO unavailable")
def test_archived_tpu_decoder_carries_are_all_explained_by_conforming_lineages() -> None:
    module = parse_hlo_module(gzip.open(ARCHIVE, "rt").read())
    config = _real_8k_decoder_config(dsa_score_default_precision=True)
    rms = _validate_rms_accepted_schedule_hlo(module, enabled=True, layernorm_width=config.index_key_width)
    # This archive predates the fused q-a norm fix: 78 sharded q-a norms are still unbound.
    assert (rms["rsqrt_count"], rms["conforming_rsqrt_count"], rms["nonconforming_rsqrt_count"]) == (313, 235, 78)
    assert rms["carry_summary"]["carry_count"] == 235
    assert rms["carry_summary"]["widths"] == {"512": 78, "6144": 157}
    allowances = rms["carry_allowances"]
    common = dict(
        module=module,
        config=config,
        full_indexer_layers=3,
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
