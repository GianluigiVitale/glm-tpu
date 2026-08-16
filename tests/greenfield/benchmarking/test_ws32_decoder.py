from __future__ import annotations

import gzip
from hashlib import sha256
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.ws32_decoder import (
    _exact_add_reducer,
    validate_ws32_decoder_hlo,
)
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


FULL_DECODER_HLO = Path(
    "/home/gianl/glm-run/"
    "greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_"
    "downf32_token_splitres_prefill_keyfix_qkva_oracle_dsa_trace2_"
    "20260809T225800140051438Z/hlo/"
    "decoder_78layer_8k_token_dsa_observer.optimized_hlo.txt.gz"
)
WS32_PREFILL_STABLEHLO = Path(
    "/home/gianl/glm-run/"
    "greenfield_ws32_short_decoder_2k_acquire_20260816T004709157408071Z/"
    "hlo/prefill.stablehlo.mlir"
)
WS32_PREFILL_OPTIMIZED_HLO = Path(
    "/home/gianl/glm-run/"
    "greenfield_ws32_short_decoder_2k_acquire_20260816T004709157408071Z/"
    "hlo/prefill.optimized_hlo.txt"
)
TUPLE_ADD_HLO = Path(
    "/home/gianl/glm-run/greenfield_collectives_20260805T135850389312854Z/"
    "hlo/fused_tuple_all_reduce_g4_bfloat16_1x6144.optimized_hlo.txt"
)


def _groups() -> tuple[str, str]:
    feature = ",".join(
        "{" + ",".join(str(value) for value in range(row * 4, row * 4 + 4)) + "}"
        for row in range(8)
    )
    expert = ",".join(
        "{" + ",".join(str(row * 4 + column) for row in range(8)) + "}"
        for column in range(4)
    )
    return feature, expert


def _hlo() -> str:
    feature, expert = _groups()
    return f'''HloModule ws32_complete, num_partitions=32

feature_add {{
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %sum = f32[] add(%a, %b)
}}

expert_add {{
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %sum = f32[] add(%a, %b)
}}

ENTRY main {{
  %input = f32[1,1536] parameter(0)
  %feature = f32[1,1536] all-reduce(%input), replica_groups={{{feature}}}, to_apply=feature_add, use_global_device_ids=true, metadata={{op_name="jit(body)/shard_map/greenfield_ws32_complete_decoder/greenfield_ws32_fused_rmsnorm/feature_square_reduce/psum"}}
  %gather = f32[8,1,1536] all-gather(%feature), dimensions={{0}}, replica_groups={{{expert}}}, use_global_device_ids=true
  ROOT %root = f32[8,1,1536] copy(%gather)
}}
'''


def _report(hlo: str, *, kind: str = "decode"):
    stable = "module @main"
    return validate_ws32_decoder_hlo(
        stable,
        hlo,
        expected_stablehlo_sha256=sha256(stable.encode()).hexdigest(),
        expected_optimized_hlo_sha256=sha256(hlo.encode()).hexdigest(),
        hidden_size=6144,
        kind=kind,
        expected_split_rmsnorm_collective_count=1,
    )


def _prefill_while_hlo() -> str:
    feature, _ = _groups()
    return f'''HloModule ws32_prefill, num_partitions=32

feature_add {{
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %sum = f32[] add(%a, %b)
}}

condition {{
  %state = (s32[], f32[1,1536]) parameter(0)
  %index = s32[] get-tuple-element(%state), index=0
  %limit = s32[] constant(2)
  ROOT %continue = pred[] compare(%index, %limit), direction=LT
}}

body {{
  %state = (s32[], f32[1,1536]) parameter(0)
  %index = s32[] get-tuple-element(%state), index=0
  %input = f32[1,1536] get-tuple-element(%state), index=1
  %one = s32[] constant(1)
  %next = s32[] add(%index, %one)
  %feature = f32[1,1536] all-reduce(%input), replica_groups={{{feature}}}, to_apply=feature_add, use_global_device_ids=true, metadata={{op_name="jit(body)/shard_map/greenfield_ws32_teacher_forced_prefill/greenfield_ws32_fused_rmsnorm/feature_square_reduce/psum"}}
  ROOT %result = (s32[], f32[1,1536]) tuple(%next, %feature)
}}

ENTRY main {{
  %zero = s32[] constant(0)
  %input = f32[1,1536] parameter(0)
  %initial = (s32[], f32[1,1536]) tuple(%zero, %input)
  ROOT %loop = (s32[], f32[1,1536]) while(%initial), condition=condition, body=body
}}
'''


def _tuple_reduce_hlo() -> str:
    _, expert = _groups()
    return f'''HloModule ws32_cache_probe, num_partitions=32

tuple_add {{
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%lhs, %rhs)
}}

ENTRY main {{
  %first = bf16[78,640] parameter(0)
  %second = bf16[21,128] parameter(1)
  ROOT %root = (bf16[78,640], bf16[21,128]) all-reduce(%first, %second), replica_groups={{{expert}}}, to_apply=tuple_add, use_global_device_ids=true, metadata={{op_name="jit(body)/shard_map/greenfield_ws32_cache_probe/root"}}
}}
'''


def _triple_reduce_hlo() -> str:
    _, expert = _groups()
    return f'''HloModule ws32_cache_probe, num_partitions=32

shared_add {{
  %lhs = bf16[] parameter(0)
  %rhs = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%lhs, %rhs)
}}

ENTRY main {{
  %first = bf16[78,640] parameter(0)
  %second = bf16[21,128] parameter(1)
  %third = bf16[1,128] parameter(2)
  ROOT %root = (bf16[78,640], bf16[21,128], bf16[1,128]) all-reduce(%first, %second, %third), replica_groups={{{expert}}}, to_apply=shared_add, use_global_device_ids=true, metadata={{op_name="jit(body)/shard_map/greenfield_ws32_cache_probe/root"}}
}}
'''


def _unused_fusion_operand_hlo() -> str:
    feature, expert = _groups()
    return f'''HloModule ws32_fusion_liveness, num_partitions=32

feature_add {{
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %sum = f32[] add(%a, %b)
}}

expert_add {{
  %a = f32[] parameter(0)
  %b = f32[] parameter(1)
  ROOT %sum = f32[] add(%a, %b)
}}

use_first {{
  %live = f32[1,1536] parameter(0)
  %unused = f32[1,1536] parameter(1)
  ROOT %result = f32[1,1536] copy(%live)
}}

ENTRY main {{
  %input = f32[1,1536] parameter(0)
  %live = f32[1,1536] all-reduce(%input), replica_groups={{{feature}}}, to_apply=feature_add, use_global_device_ids=true, metadata={{op_name="jit(body)/shard_map/greenfield_ws32_complete_decoder/greenfield_ws32_fused_rmsnorm/feature_square_reduce/psum"}}
  %decoy = f32[1,1536] all-reduce(%input), replica_groups={{{expert}}}, to_apply=expert_add, use_global_device_ids=true
  ROOT %root = f32[1,1536] fusion(%live, %decoy), calls=use_first
}}
'''


def test_ws32_complete_hlo_contract_is_live_and_subgroup_only() -> None:
    report = _report(_hlo())
    assert report.passed, report.violations
    assert report.collective_count == report.live_collective_count == 2
    assert report.feature_collective_count == 1
    assert report.expert_collective_count == 1
    assert report.maximum_group_size == 8


def test_ws32_complete_hlo_contract_refuses_structural_mutations() -> None:
    base = _hlo()
    feature, _ = _groups()
    dead = base.replace(
        "  ROOT %root =",
        f"  %dead = f32[1,1536] all-reduce(%input), replica_groups={{{feature}}}, to_apply=feature_add, use_global_device_ids=true\n  ROOT %root =",
    )
    async_value = base.replace(
        "all-reduce(%input)", "all-reduce-start(%input)", 1
    )
    wrong_group = base.replace(
        "{0,1,2,3},{4,5,6,7}", "{0,1,2,4},{3,5,6,7}", 1
    )
    full_hidden = base.replace(
        "%input = f32[1,1536] parameter(0)",
        "%rogue = bf16[32,6144] parameter(1)\n  %input = f32[1,1536] parameter(0)",
    )
    bad_reducer = base.replace(
        "ROOT %sum = f32[] add(%a, %b)",
        "ROOT %sum = f32[] maximum(%a, %b)",
        1,
    )
    rounded_first = base.replace(
        "greenfield_ws32_fused_rmsnorm", "greenfield_ws32_rmsnorm"
    )
    for mutation in (
        dead,
        async_value,
        wrong_group,
        full_hidden,
        bad_reducer,
        rounded_first,
    ):
        report = _report(mutation)
        assert not report.passed


def test_ws32_complete_hlo_binds_exact_scope_to_live_optimized_value() -> None:
    scope = "greenfield_ws32_complete_decoder"
    base = _hlo()
    unscoped = base.replace(scope, "unscoped_decoder")
    dead_decoy = unscoped.replace(
        "  ROOT %root =",
        "  %scope_decoy = f32[1,1536] negate(%input), "
        f'metadata={{op_name="jit(x)/{scope}/decoy"}}\n'
        "  ROOT %root =",
    )
    dead_report = _report(dead_decoy)
    assert not dead_report.passed
    assert dead_report.violations == (
        "optimized HLO lost live exact decode source scope",
    )

    superstring = base.replace(scope, f"{scope}_decoy")
    assert not _report(superstring).passed

    observer_scope = "greenfield_ws32_complete_decoder_dsa_observer"
    observer = base.replace(scope, observer_scope)
    assert _report(observer, kind="observer").passed
    assert not _report(observer, kind="decode").passed

    raw_text_decoy = unscoped.replace(
        'metadata={op_name="jit(body)/shard_map/unscoped_decoder/feature"}',
        'metadata={op_name="jit(body)/shard_map/unscoped_decoder/feature" '
        f'source_file="{scope}"}}',
    )
    assert not _report(raw_text_decoy).passed

    stable_decoy = f'module @main loc("{scope}")'
    stable_report = validate_ws32_decoder_hlo(
        stable_decoy,
        unscoped,
        expected_stablehlo_sha256=sha256(stable_decoy.encode()).hexdigest(),
        expected_optimized_hlo_sha256=sha256(unscoped.encode()).hexdigest(),
        hidden_size=6144,
        kind="decode",
        expected_split_rmsnorm_collective_count=1,
    )
    assert not stable_report.passed
    assert stable_report.violations == (
        "optimized HLO lost live exact decode source scope",
    )


def test_ws32_prefill_hlo_traces_live_while_condition_and_body() -> None:
    base = _prefill_while_hlo()
    report = _report(base, kind="prefill")
    assert report.passed, report.violations
    assert report.collective_count == report.live_collective_count == 1
    dead = base.replace(
        "  ROOT %result =",
        "  %dead = f32[1,1536] negate(%input)\n  ROOT %result =",
    ).replace(
        "  %dead = f32[1,1536] negate(%input)",
        "  %dead = f32[1,1536] all-reduce(%input), "
        f"replica_groups={{{_groups()[0]}}}, to_apply=feature_add, "
        "use_global_device_ids=true",
    )
    refused = _report(dead, kind="prefill")
    assert not refused.passed
    assert any("dead collective decoy" in item for item in refused.violations)


def test_ws32_cache_probe_accepts_only_exact_tuple_add_reducer() -> None:
    exact = _report(_tuple_reduce_hlo(), kind="cache_probe")
    assert exact.passed, exact.violations
    assert exact.expert_collective_count == 1
    assert exact.feature_collective_count == 0
    refused = _report(
        _tuple_reduce_hlo().replace(
            "ROOT %sum = bf16[] add(%lhs, %rhs)",
            "ROOT %sum = bf16[] maximum(%lhs, %rhs)",
        ),
        kind="cache_probe",
    )
    assert not refused.passed
    triple = _report(_triple_reduce_hlo(), kind="cache_probe")
    assert triple.passed, triple.violations
    bad_dtype = _triple_reduce_hlo().replace(
        "%third = bf16[1,128] parameter(2)",
        "%third = f32[1,128] parameter(2)",
    ).replace(
        "bf16[1,128]) all-reduce", "f32[1,128]) all-reduce"
    )
    assert not _report(bad_dtype, kind="cache_probe").passed
    malformed = _triple_reduce_hlo().replace(
        "%rhs = bf16[] parameter(1)", "%rhs = bf16[] parameter(0)"
    )
    assert not _report(malformed, kind="cache_probe").passed


def test_ws32_complete_hlo_refuses_collective_in_unused_fusion_operand() -> None:
    refused = _report(_unused_fusion_operand_hlo())
    assert not refused.passed
    assert refused.collective_count == 2
    assert refused.live_collective_count == 1
    assert any("dead collective decoy" in item for item in refused.violations)


def test_ws32_complete_hlo_requires_explicit_32_partitions_and_no_hidden_gather() -> None:
    no_partitions = _hlo().replace(", num_partitions=32", "")
    report = _report(no_partitions)
    assert not report.passed
    assert any("partitioned 32 ways" in item for item in report.violations)

    feature, expert = _groups()
    hidden_gather = _hlo().replace(
        "%gather = f32[8,1,1536] all-gather(%feature)",
        "%gather = f32[4,1,1536] all-gather(%feature)",
    ).replace(
        f"replica_groups={{{expert}}}, use_global_device_ids=true\n  ROOT %root",
        f"replica_groups={{{feature}}}, use_global_device_ids=true\n  ROOT %root",
    ).replace(
        "ROOT %root = f32[8,1,1536] copy(%gather)",
        "ROOT %root = f32[4,1,1536] copy(%gather)",
    )
    report = _report(hidden_gather)
    assert not report.passed
    assert any("full-pod hidden value" in item for item in report.violations)


@pytest.mark.skipif(
    not FULL_DECODER_HLO.exists(), reason="protected full-decoder HLO unavailable"
)
def test_ws32_liveness_and_tuple_reducer_replay_protected_full_decoder() -> None:
    with gzip.open(FULL_DECODER_HLO, "rt", encoding="utf-8") as stream:
        module = parse_hlo_module(stream.read())
    live = _live_instruction_closure(module.instructions)
    assert len(live) == 177_093
    live_opcodes = {item.raw_opcode for item in live}
    assert {"reduce", "sort", "reduce-window", "scatter"} <= live_opcodes


@pytest.mark.skipif(
    not TUPLE_ADD_HLO.exists(), reason="protected tuple all-reduce HLO unavailable"
)
def test_ws32_tuple_add_reducer_replays_real_tpu_lowering() -> None:
    module = parse_hlo_module(TUPLE_ADD_HLO.read_text(encoding="utf-8"))
    multi = tuple(module.collectives)
    assert len(multi) == 75
    assert {len(item.operand_shapes) for item in multi} == {2}
    assert all(
        _exact_add_reducer(item, module_instructions=module.instructions)
        for item in multi
    )


@pytest.mark.skipif(
    not WS32_PREFILL_STABLEHLO.exists()
    or not WS32_PREFILL_OPTIMIZED_HLO.exists(),
    reason="protected WS32 prefill acquisition HLO unavailable",
)
def test_ws32_prefill_contract_replays_protected_acquisition() -> None:
    report = validate_ws32_decoder_hlo(
        WS32_PREFILL_STABLEHLO.read_text(encoding="utf-8"),
        WS32_PREFILL_OPTIMIZED_HLO.read_text(encoding="utf-8"),
        expected_stablehlo_sha256="0" * 64,
        expected_optimized_hlo_sha256="0" * 64,
        hidden_size=6144,
        kind="prefill",
    )
    assert set(report.violations) == {
        "StableHLO identity drifted",
        "complete WS32 graph lacks the exact split-residual RMSNorm boundary count",
        "complete WS32 graph retains rounded-first RMSNorm boundaries",
        "optimized HLO identity drifted",
    }
    assert report.fused_rmsnorm_collective_count == 0
    assert report.rounded_first_rmsnorm_collective_count == 157
    assert report.instruction_count == 173_829
    assert report.live_instruction_count == 173_229
    assert report.collective_count == report.live_collective_count == 1_289
    assert report.all_reduce_count == 1_226
    assert report.all_gather_count == 63
    assert report.feature_collective_count == 914
    assert report.expert_collective_count == 375
    assert report.maximum_group_size == 8
    assert report.async_collective_count == 0
    assert report.forbidden_full_hidden_values == ()
