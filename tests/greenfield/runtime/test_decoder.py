from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from glm_tpu.greenfield.errors import PlanValidationError


def test_fused_qkv_a_hlo_contract_requires_db502_primitive() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_fused_qkv_a_decoder_association,
    )

    hlo = '''HloModule fused_qkv, replica_count=1, num_partitions=32

ENTRY main {
  %hidden = bf16[1,6144] parameter(0)
  %weight = u8[32,6144,82] parameter(1)
  %scale = f32[32,48,82] parameter(2)
  ROOT %qkv = f32[1,82] convolution(%weight, %scale), dim_labels=bf_io->bf
}
'''
    contract = _validate_fused_qkv_a_decoder_association(hlo, layers=1)
    assert contract["passed"], contract
    assert contract["convolution_count"] == 1

    dead = _validate_fused_qkv_a_decoder_association(
        hlo.replace(
            "%weight = u8[32,6144,82] parameter(1)",
            "%weight = u8[32,6144,82] parameter(1)\n"
            "  %dead = bf16[32,6144] parameter(3)",
        ),
        layers=1,
    )
    assert dead["violations"]
    assert dead["forbidden_shapes"] == ["bf16[32,6144]"]


def _repair_scoped_shape_hlo(*, include_unscoped: bool = False) -> str:
    unrelated = ""
    if include_unscoped:
        unrelated = '''
%unrelated (dead_rows: f32[32,6144], overlay: bf16[2048,6144]) -> bf16[2048,6144] {
  %dead_rows = f32[32,6144] parameter(0)
  %overlay = bf16[2048,6144] parameter(1)
  ROOT %unrelated_root = bf16[2048,6144] copy(%overlay), metadata={op_name="decode/recurrent_overlay"}
}
'''
    return f'''HloModule repair_scope, replica_count=1, num_partitions=32

%repair_weight (wk: f32[128,6144]) -> f32[128,6144] {{
  %wk = f32[128,6144] parameter(0)
  ROOT %weight_root = f32[128,6144] copy(%wk)
}}

%repair (chunk: bf16[2048,6144], rows: f32[32,6144], wk: f32[128,6144]) -> bf16[2048,6144] {{
  %chunk = bf16[2048,6144] parameter(0)
  %rows = f32[32,6144] parameter(1)
  %wk = f32[128,6144] parameter(2)
  %weight = f32[128,6144] fusion(%wk), kind=kLoop, calls=%repair_weight
  ROOT %repair_root = bf16[2048,6144] copy(%chunk)
}}
{unrelated}
ENTRY %main (chunk: bf16[2048,6144], rows: f32[32,6144], wk: f32[128,6144]) -> bf16[2048,6144] {{
  %chunk = bf16[2048,6144] parameter(0)
  %rows = f32[32,6144] parameter(1)
  %wk = f32[128,6144] parameter(2)
  ROOT %repair_call = bf16[2048,6144] fusion(%chunk, %rows, %wk), kind=kLoop, calls=%repair, metadata={{op_name="jit(execute)/shard_map/cond/branch_0_fun/repair_stage_local_prompt_index_cache"}}
}}
'''


def test_prefill_repair_shape_scope_is_not_a_module_wide_exception() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _classify_decoder_live_tensor_shapes,
        _validate_pallas_stage_linear_decoder_calls,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    hlo = _repair_scoped_shape_hlo()
    module = parse_hlo_module(hlo)
    stage = _validate_pallas_stage_linear_decoder_calls(
        hlo,
        layers=0,
        dense_layers=0,
        full_indexer_layers=0,
        dsa_query_backend="reference",
        prefill_index_repair=True,
        module=module,
    )
    assert stage["passed"], stage
    assert set(stage["allowed_prefill_index_repair_shape_counts"]) == {
        "bf16[2048,6144]",
        "f32[128,6144]",
    }

    live = _classify_decoder_live_tensor_shapes(
        module,
        config=_real_8k_decoder_config(),
        full_indexer_layers=0,
        backend_contract="cpu_reference",
        prefill_index_repair=True,
    )
    assert live["passed"], live
    assert live["allowed_prefill_index_repair_shapes"]

    default_stage = _validate_pallas_stage_linear_decoder_calls(
        hlo,
        layers=0,
        dense_layers=0,
        full_indexer_layers=0,
        dsa_query_backend="reference",
    )
    assert not default_stage["passed"]
    assert default_stage["forbidden_decoded_weight_overlays"] == [
        "bf16[2048,6144]",
        "f32[128,6144]",
    ]

    unscoped_hlo = _repair_scoped_shape_hlo(include_unscoped=True)
    unscoped_module = parse_hlo_module(unscoped_hlo)
    unscoped_stage = _validate_pallas_stage_linear_decoder_calls(
        unscoped_hlo,
        layers=0,
        dense_layers=0,
        full_indexer_layers=0,
        dsa_query_backend="reference",
        prefill_index_repair=True,
        module=unscoped_module,
    )
    assert not unscoped_stage["passed"]
    assert unscoped_stage["forbidden_decoded_weight_overlays"] == [
        "bf16[2048,6144]"
    ]

    unscoped_live = _classify_decoder_live_tensor_shapes(
        unscoped_module,
        config=_real_8k_decoder_config(),
        full_indexer_layers=0,
        backend_contract="cpu_reference",
        prefill_index_repair=True,
    )
    assert not unscoped_live["passed"]
    assert any(
        item["op_name"] == "decode/recurrent_overlay"
        or item["computation"].startswith("%unrelated ")
        for item in unscoped_live["forbidden_shapes"]
    )


def test_fused_qkv_dead_row_gate_scopes_prefill_repair_rows() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_fused_qkv_a_decoder_association,
    )

    hlo = '''HloModule fused_qkv_repair, num_partitions=32

%repair_rows (rows: f32[32,6144]) -> f32[32,6144] {
  %rows = f32[32,6144] parameter(0)
  ROOT %rows_root = f32[32,6144] copy(%rows)
}

ENTRY %main (hidden: bf16[1,6144], weight: u8[32,6144,82], scale: f32[32,48,82], rows: f32[32,6144]) -> f32[1,82] {
  %hidden = bf16[1,6144] parameter(0)
  %weight = u8[32,6144,82] parameter(1)
  %scale = f32[32,48,82] parameter(2)
  %rows = f32[32,6144] parameter(3)
  %repair_call = f32[32,6144] fusion(%rows), kind=kLoop, calls=%repair_rows, metadata={op_name="jit(execute)/shard_map/cond/branch_0_fun/repair_stage_local_prompt_index_cache"}
  ROOT %qkv = f32[1,82] convolution(%weight, %scale), dim_labels=bf_io->bf
}
'''
    scoped = _validate_fused_qkv_a_decoder_association(
        hlo,
        layers=1,
        prefill_index_repair=True,
    )
    assert scoped["passed"], scoped
    assert scoped["allowed_prefill_index_repair_shape_counts"][
        "f32[32,6144]"
    ] > 0

    default = _validate_fused_qkv_a_decoder_association(hlo, layers=1)
    assert not default["passed"]
    assert default["forbidden_shapes"] == ["f32[32,6144]"]

    unscoped = hlo.replace(
        "ENTRY %main",
        '''%unrelated_rows (rows: f32[32,6144]) -> f32[32,6144] {
  %rows = f32[32,6144] parameter(0)
  ROOT %root = f32[32,6144] copy(%rows), metadata={op_name="decode/dead_rows"}
}

ENTRY %main''',
    )
    rejected = _validate_fused_qkv_a_decoder_association(
        unscoped,
        layers=1,
        prefill_index_repair=True,
    )
    assert not rejected["passed"]
    assert rejected["forbidden_shapes"] == ["f32[32,6144]"]


def test_fused_qkv_a_runtime_binding_omits_separate_projection_state() -> None:
    from glm_tpu.greenfield.runtime.decoder import _attention_weights

    loaded = []

    def weight(name: str) -> str:
        loaded.append(name)
        return name

    attention = _attention_weights(weight, 0, "fused_n82_convolution")
    assert attention.q_a_bits is None
    assert attention.q_a_scale is None
    assert attention.kv_a_bits is None
    assert attention.kv_a_scale is None
    assert attention.qkv_a_bits == "attention.slot_00.qkv_a.weight_bits"
    assert attention.qkv_a_scale == "attention.slot_00.qkv_a.scale_inv"
    assert "attention.slot_00.q_a.weight_bits" not in loaded
    assert "attention.slot_00.kv_a.weight_bits" not in loaded


def _synthetic_8k_dsa_score_hlo() -> str:
    return '''HloModule dsa_score, replica_count=1, num_partitions=32

%score (q: f32[32,128], key: bf16[2048,128], weight: f32[32]) -> f32[2048] {
  %q = f32[32,128] parameter(0)
  %key = bf16[2048,128] parameter(1)
  %weight = f32[32] parameter(2)
  %weight_broadcast = f32[1,32,2048] broadcast(%weight), dimensions={1}, metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/rh,rhs->rs/dot_general"}
  %contraction = f32[32,2048] convolution(%q, %key), dim_labels=bf_oi->bf, metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/rhd,sd->rhs/dot_general"}
  %scale = f32[] constant(0.0883883461)
  %scale_broadcast = f32[32,2048] broadcast(%scale), dimensions={}, metadata={op_name="jit(mapped_token)/shard_map/broadcast.19787"}
  %scaled = f32[32,2048] multiply(%contraction, %scale_broadcast), metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/mul"}
  %zero = f32[] constant(0)
  %zero_broadcast = f32[32,2048] broadcast(%zero), dimensions={}, metadata={op_name="jit(mapped_token)/shard_map/broadcast.20646"}
  %clamped = f32[32,2048] maximum(%scaled, %zero_broadcast), metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/max"}
  %expanded = f32[1,32,2048] bitcast(%clamped), metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/max"}
  %weighted = f32[1,32,2048] multiply(%weight_broadcast, %expanded), metadata={op_name="jit(mapped_token)/shard_map/multiply.611"}
  ROOT %reduced = f32[2048] reduce(%weighted, %zero), dimensions={0,1}, metadata={op_name="jit(mapped_token)/shard_map/cond/branch_1_fun/rh,rhs->rs/dot_general"}
}
'''


def _real_8k_decoder_config():
    from glm_tpu.greenfield.runtime.decoder import DecoderStepConfig

    return DecoderStepConfig(
        stage_count=8,
        local_parallel_size=4,
        hidden_size=6144,
        selected_width=2048,
        context_capacity=8192,
        maximum_layer_slots=10,
        maximum_full_indexer_slots=3,
        logical_page_size=256,
        local_rows_per_page=64,
        packed_cache_width=192,
        index_key_width=128,
        dsa_indexer_heads=32,
        vocab_size=154880,
    )


def test_8k_dsa_head_score_shape_requires_exact_dataflow() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _classify_decoder_live_tensor_shapes,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    config = _real_8k_decoder_config()
    hlo = _synthetic_8k_dsa_score_hlo()
    record = _classify_decoder_live_tensor_shapes(
        parse_hlo_module(hlo),
        config=config,
        full_indexer_layers=1,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert record["passed"], record

    assert record["score_body_count"] == 1
    assert record["score_dimensions"] == [32, 2048]
    assert len(record["allowed_dsa_score_shapes"]) == 10
    assert record["forbidden_shapes"] == []
    assert record["body_records"][0] == {
        "computation": (
            "%score (q: f32[32,128], key: bf16[2048,128], "
            "weight: f32[32]) -> f32[2048]"
        ),
        "contraction_count": 1,
        "head_weight_broadcast_count": 1,
        "head_weight_multiply_count": 1,
        "opcode_counts": {
            "bitcast": 1,
            "broadcast": 2,
            "convolution": 1,
            "maximum": 1,
            "multiply": 1,
        },
        "reduction_count": 1,
        "score_shape_occurrences": 10,
        "valid": True,
    }

    drifted = _classify_decoder_live_tensor_shapes(
        parse_hlo_module(
            hlo.replace(
                'op_name="jit(mapped_token)/shard_map/cond/'
                'branch_1_fun/mul"',
                'op_name="jit(mapped_token)/shard_map/batch_rows/mul"',
            )
        ),
        config=config,
        full_indexer_layers=1,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not drifted["passed"]
    assert drifted["score_body_count"] == 0
    assert drifted["allowed_dsa_score_shapes"] == []
    assert len(drifted["forbidden_shapes"]) == 10
    assert drifted["violations"] == [
        "decoder local DSA score-body contract drifted: expected=1 observed=0"
    ]


def test_8k_dsa_shape_exception_does_not_admit_dead_rows() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _classify_decoder_live_tensor_shapes,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    hlo = _synthetic_8k_dsa_score_hlo() + '''
ENTRY %main (unrelated: f32[32,2048], wrong_dtype: bf16[32,2048], hidden: bf16[32,6144], dead_query: f32[32,32,128]) -> f32[32,2048] {
  %unrelated = f32[32,2048] parameter(0)
  %wrong_dtype = bf16[32,2048] parameter(1)
  %hidden = bf16[32,6144] parameter(2)
  %dead_query = f32[32,32,128] parameter(3)
  ROOT %root = f32[32,2048] copy(%unrelated), metadata={op_name="batch_rows/copy"}
}
'''
    record = _classify_decoder_live_tensor_shapes(
        parse_hlo_module(hlo),
        config=_real_8k_decoder_config(),
        full_indexer_layers=1,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not record["passed"]
    assert record["score_body_count"] == 1
    assert len(record["allowed_dsa_score_shapes"]) == 10
    assert record["violations"] == []
    forbidden = {
        (
            item["instruction"],
            item["shape"]["dtype"],
            tuple(item["shape"]["dimensions"]),
        )
        for item in record["forbidden_shapes"]
    }
    assert forbidden == {
        ("%unrelated", "f32", (32, 2048)),
        ("%wrong_dtype", "bf16", (32, 2048)),
        ("%hidden", "bf16", (32, 6144)),
        ("%dead_query", "f32", (32, 32, 128)),
        ("%root", "f32", (32, 2048)),
        ("%root", "f32", (32, 2048)),
    }


def test_teacher_forced_prefill_builder_rejects_invalid_contracts() -> None:
    from types import SimpleNamespace

    from glm_tpu.greenfield.errors import PlanValidationError
    from glm_tpu.greenfield.runtime import build_teacher_forced_prefill_program

    config = SimpleNamespace(context_capacity=8, total_devices=32)
    body_only = SimpleNamespace(complete_token_path=False, config=config)
    complete = SimpleNamespace(complete_token_path=True, config=config)

    with pytest.raises(PlanValidationError, match="complete-token decoder"):
        build_teacher_forced_prefill_program(body_only, prompt_length=2)
    for value in (True, 0, -1):
        with pytest.raises(PlanValidationError, match="must be positive"):
            build_teacher_forced_prefill_program(complete, prompt_length=value)
    with pytest.raises(PlanValidationError, match="leave capacity"):
        build_teacher_forced_prefill_program(complete, prompt_length=8)


def test_complete_token_tpu_collective_lowering_is_exact_and_local() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_complete_token_collective_lowering,
    )
    from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4)) for stage in range(8)
    )
    pairs = tuple(
        (groups[stage][slot], groups[(stage + 1) % 8][slot])
        for stage in range(8)
        for slot in range(4)
    )
    group_text = "{" + ",".join(
        "{" + ",".join(map(str, group)) + "}" for group in groups
    ) + "}"
    pair_text = "{" + ",".join(
        "{" + ",".join(map(str, pair)) + "}" for pair in pairs
    ) + "}"
    hlo = f'''HloModule complete_token, replica_count=1, num_partitions=32

%add.1 (x: bf16[], y: bf16[]) -> bf16[] {{
  %x = bf16[] parameter(0)
  %y = bf16[] parameter(1)
  ROOT %sum = bf16[] add(%x, %y)
}}

%add.2 (x: s32[], y: s32[]) -> s32[] {{
  %x = s32[] parameter(0)
  %y = s32[] parameter(1)
  ROOT %sum = s32[] add(%x, %y)
}}

ENTRY %main (scores: bf16[4], ids: s32[4], token: s32[1]) -> s32[1] {{
  %scores = bf16[4] parameter(0)
  %ids = s32[4] parameter(1)
  %token = s32[1] parameter(2)
  %score_exchange = bf16[4] all-reduce(%scores), replica_groups={group_text}, use_global_device_ids=true, to_apply=%add.1
  %id_exchange = s32[4] all-reduce(%ids), replica_groups={group_text}, use_global_device_ids=true, to_apply=%add.2
  ROOT %token_return = (s32[1], s32[1], u32[], u32[]) collective-permute-start(%token), source_target_pairs={pair_text}, metadata={{op_name="jit(mapped_token)/shard_map/ppermute"}}
}}
'''
    module = parse_hlo_module(hlo)
    record = _validate_complete_token_collective_lowering(
        module,
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert record["passed"], record
    assert record["lowering"] == "local_one_hot_all_reduce"
    assert record["score_exchange"][0]["operand_shapes"] == ["bf16[4]"]
    assert record["token_id_exchange"][0]["operand_shapes"] == ["s32[4]"]
    assert record["token_return"][0]["operand_shapes"] == ["s32[1]"]
    assert record["accepted_token_return_op_names"] == [
        "jit(mapped_token)/shard_map/ppermute",
        "jit(execute)/while/body/closed_call/shard_map/ppermute",
    ]

    wide_hlo = hlo.replace("bf16[4]", "bf16[4,16]").replace(
        "s32[4]", "s32[4,16]"
    )
    wide_record = _validate_complete_token_collective_lowering(
        parse_hlo_module(wide_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
        token_observation_candidates=16,
    )
    assert wide_record["passed"], wide_record
    assert wide_record["token_observation_candidates"] == 16
    assert wide_record["score_exchange"][0]["operand_shapes"] == [
        "bf16[4,16]"
    ]
    assert wide_record["token_id_exchange"][0]["operand_shapes"] == [
        "s32[4,16]"
    ]

    wrong_candidate_width = _validate_complete_token_collective_lowering(
        parse_hlo_module(hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
        token_observation_candidates=16,
    )
    assert not wrong_candidate_width["passed"]

    prefill_record = _validate_complete_token_collective_lowering(
        parse_hlo_module(
            hlo.replace(
                "jit(mapped_token)/shard_map/ppermute",
                "jit(execute)/while/body/closed_call/shard_map/ppermute",
            )
        ),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert prefill_record["passed"], prefill_record

    wrong_source = _validate_complete_token_collective_lowering(
        parse_hlo_module(
            hlo.replace(
                "jit(mapped_token)/shard_map/ppermute",
                "jit(unrelated)/shard_map/ppermute",
            )
        ),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not wrong_source["passed"]
    assert any(
        "source operation drifted" in item
        for item in wrong_source["violations"]
    )

    nonlocal_hlo = hlo.replace(
        f"replica_groups={group_text}",
        "replica_groups={{" + ",".join(map(str, range(32))) + "}}",
    )
    rejected = _validate_complete_token_collective_lowering(
        parse_hlo_module(nonlocal_hlo),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not rejected["passed"]
    assert any("escaped PP8 local groups" in item for item in rejected["violations"])

    wrong_return = _validate_complete_token_collective_lowering(
        parse_hlo_module(hlo.replace("token: s32[1]", "token: s32[2]").replace(
            "%token = s32[1]", "%token = s32[2]"
        )),
        expected_groups=groups,
        expected_pairs=pairs,
        backend_contract="tpu_v4_pp8_pallas_feature_linear",
    )
    assert not wrong_return["passed"]
    assert any("exactly one s32[1]" in item for item in wrong_return["violations"])


def test_feature_decoder_hlo_contract_pins_all_raw_kernels_and_overlays() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_pallas_feature_decoder_calls,
    )

    selected = (
        "out = bf16[8,8,6144] custom-call("
        "u8[256,6144,512], u8[256,6144,512], u8[256,512,6144]), "
        'custom_call_target="tpu_custom_call", '
        'metadata={op_name="greenfield_fp8_fused_selected_moe_'
        'r8_g256_h6144_i512"}'
    )
    shared_up = (
        "up = bf16[1,512] custom-call(u8[512,6144]), "
        'custom_call_target="tpu_custom_call", '
        'metadata={op_name="greenfield_fp8_block_up_gate_m8_k6144_n512"}'
    )
    shared_down = (
        "down = bf16[1,6144] custom-call(u8[6144,512]), "
        'custom_call_target="tpu_custom_call", '
        'metadata={op_name="greenfield_fp8_block_matmul_m8_k512_n6144"}'
    )
    hlo = "\n".join((selected, shared_up, shared_down) * 75)
    record = _validate_pallas_feature_decoder_calls(
        hlo,
        sparse_layers=75,
        feature_output_tile=128,
    )
    assert record["passed"], record
    assert record["feature_output_tile"] == 128

    wide_hlo = hlo.replace(
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512",
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256",
    )
    wide = _validate_pallas_feature_decoder_calls(wide_hlo, sparse_layers=75)
    assert wide["passed"], wide
    assert wide["feature_output_tile"] == 256
    fp32_hlo = wide_hlo.replace(
        "out = bf16[8,8,6144]",
        "out = f32[8,8,6144]",
    ).replace(
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256",
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256_downf32",
    )
    fp32_hlo += "\n" + "\n".join(
        "cast = bf16[8,6144] custom-call(f32[8,6144]), "
        'custom_call_target="tpu_custom_call", '
        'metadata={op_name="greenfield_fp32_to_bf16_r8_h6144"}'
        for _ in range(75)
    )
    fp32 = _validate_pallas_feature_decoder_calls(
        fp32_hlo,
        sparse_layers=75,
        reconstruct_down_fp32=True,
    )
    assert fp32["passed"], fp32
    assert fp32["reconstruct_down_fp32"] is True
    stale_fp32_shape = _validate_pallas_feature_decoder_calls(
        wide_hlo.replace(
            "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256",
            "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256_downf32",
        ),
        sparse_layers=75,
        reconstruct_down_fp32=True,
    )
    assert not stale_fp32_shape["passed"]
    with pytest.raises(PlanValidationError, match="incompatible"):
        _validate_pallas_feature_decoder_calls(
            fp32_hlo,
            sparse_layers=75,
            fuse_route_weighting=True,
            reconstruct_down_fp32=True,
        )
    fused_hlo = wide_hlo.replace(
        "out = bf16[8,8,6144]",
        "out = bf16[8,6144]",
    ).replace(
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256",
        "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256_wsum",
    )
    fused = _validate_pallas_feature_decoder_calls(
        fused_hlo,
        sparse_layers=75,
        fuse_route_weighting=True,
    )
    assert fused["passed"], fused
    assert fused["fuse_route_weighting"] is True
    stale_shape = _validate_pallas_feature_decoder_calls(
        wide_hlo.replace(
            "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256",
            "greenfield_fp8_fused_selected_moe_r8_g256_h6144_i512_ot256_wsum",
        ),
        sparse_layers=75,
        fuse_route_weighting=True,
    )
    assert not stale_shape["passed"]
    wrong_fingerprint = _validate_pallas_feature_decoder_calls(
        wide_hlo,
        sparse_layers=75,
        feature_output_tile=128,
    )
    assert not wrong_fingerprint["passed"]

    rejected = _validate_pallas_feature_decoder_calls(
        hlo + "\noverlay = bf16[256,6144,512] parameter(0)",
        sparse_layers=75,
        feature_output_tile=128,
    )
    assert not rejected["passed"]
    assert rejected["forbidden_decoded_expert_overlays"]

    formatted = _validate_pallas_feature_decoder_calls(
        hlo + "\nformatted = f8e4m3fn[512,6144] parameter(0)",
        sparse_layers=75,
        feature_output_tile=128,
    )
    assert not formatted["passed"]
    assert formatted["forbidden_formatted_shared_overlays"]


def test_stage_linear_decoder_hlo_contract_pins_kernels_and_overlays() -> None:
    from glm_tpu.greenfield.runtime.decoder import (
        _validate_pallas_stage_linear_decoder_calls,
    )

    names = (
        "greenfield_fp8_block_matmul_m8_k6144_n2048",
        "greenfield_fp8_block_matmul_m8_k2048_n4096",
        "greenfield_fp8_block_matmul_m8_k6144_n640",
        "greenfield_fp8_block_matmul_m8_k4096_n6144",
        "greenfield_fp8_structured_kv_b_q_absorb_h16_p192_l512",
        "greenfield_fp8_structured_kv_b_value_h16_l512_v256",
        "greenfield_fp8_block_matmul_f32_m8_k2048_n1024",
        "greenfield_fp8_block_matmul_f32_m8_k6144_n128",
    )
    calls = [
        f'%{name} = bf16[1,128] custom-call(u8[1,128]), '
        'custom_call_target="tpu_custom_call", '
        f'metadata={{op_name="{name}"}}'
        for name in names
        for _ in range(
            21 if "block_matmul_f32" in name else 78
        )
    ]
    dense = "greenfield_fp8_fused_block_swiglu_m8_h6144_i3072_o6144"
    calls.extend(
        f'%{dense} = bf16[1,6144] custom-call(u8[3072,6144]), '
        'custom_call_target="tpu_custom_call", '
        f'metadata={{op_name="{dense}"}}'
        for _ in range(3)
    )
    hlo = "\n".join(calls)
    record = _validate_pallas_stage_linear_decoder_calls(
        hlo, layers=78, dense_layers=3, full_indexer_layers=21
    )
    assert record["passed"], record

    reference_dsa_hlo = "\n".join(
        line
        for line in calls
        if "greenfield_fp8_block_matmul_f32_m8_k2048_n1024" not in line
    )
    reference_dsa_hlo += "\n" + "\n".join(
        f"%dsa_owner_{index} = f32[1024,2048] parameter(0)"
        for index in range(21)
    )
    reference_dsa = _validate_pallas_stage_linear_decoder_calls(
        reference_dsa_hlo,
        layers=78,
        dense_layers=3,
        full_indexer_layers=21,
        dsa_query_backend="reference",
    )
    assert reference_dsa["passed"], reference_dsa
    assert reference_dsa["expected_kernel_counts"][
        "greenfield_fp8_block_matmul_f32_m8_k2048_n1024"
    ] == 0

    fused_qkv_hlo = "\n".join(
        line
        for line in calls
        if not any(
            kernel in line
            for kernel in (
                "greenfield_fp8_block_matmul_m8_k6144_n2048",
                "greenfield_fp8_block_matmul_m8_k6144_n640",
            )
        )
    )
    fused_qkv = _validate_pallas_stage_linear_decoder_calls(
        fused_qkv_hlo,
        layers=78,
        dense_layers=3,
        full_indexer_layers=21,
        attention_projection_backend="fused_n82_convolution",
    )
    assert fused_qkv["passed"], fused_qkv
    assert fused_qkv["attention_projection_backend"] == (
        "fused_n82_convolution"
    )
    assert fused_qkv["expected_kernel_counts"][
        "greenfield_fp8_block_matmul_m8_k6144_n2048"
    ] == 0
    assert fused_qkv["expected_kernel_counts"][
        "greenfield_fp8_block_matmul_m8_k6144_n640"
    ] == 0

    rejected = _validate_pallas_stage_linear_decoder_calls(
        hlo + "\noverlay = bf16[2048,6144] parameter(0)",
        layers=78,
        dense_layers=3,
        full_indexer_layers=21,
    )
    assert not rejected["passed"]
    assert rejected["forbidden_decoded_weight_overlays"]

    formatted = _validate_pallas_stage_linear_decoder_calls(
        hlo + "\nformatted = f8e4m3fn[6144,4096] parameter(0)",
        layers=78,
        dense_layers=3,
        full_indexer_layers=21,
    )
    assert not formatted["passed"]
    assert formatted["forbidden_formatted_weight_overlays"]

    from glm_tpu.greenfield.runtime.decoder import (
        _validate_dsa_query_decoder_association,
    )

    association = _validate_dsa_query_decoder_association(
        reference_dsa_hlo,
        full_indexer_layers=21,
        local_parallel_size=4,
        dsa_indexer_heads=32,
        index_key_width=128,
        backend="reference",
    )
    assert association["passed"], association
    global_owner = _validate_dsa_query_decoder_association(
        reference_dsa_hlo + "\n%global = f32[4096,2048] parameter(0)",
        full_indexer_layers=21,
        local_parallel_size=4,
        dsa_indexer_heads=32,
        index_key_width=128,
        backend="reference",
    )
    assert not global_owner["passed"]
    assert global_owner["forbidden_global_shapes"] == ["f32[4096,2048]"]


def test_decoder_sparse_backend_fails_closed_on_layout_mismatch() -> None:
    from dataclasses import replace

    from glm_tpu.greenfield.errors import PlanValidationError
    from glm_tpu.greenfield.model import (
        FEATURE_EXPERT_RUNTIME_LAYOUT,
        build_decoder_feature_fused_qkv_runtime_weight_layout,
        build_decoder_feature_runtime_weight_layout,
        build_decoder_runtime_weight_layout,
        build_decoder_state_layout,
        build_pipeline_schedule,
    )
    from glm_tpu.greenfield.runtime import build_decoder_step_program
    from tests.greenfield.checkpoint.test_runtime_pack import (
        _small_feature_source_plan,
    )

    source_plan = _small_feature_source_plan()
    source_plan = replace(
        source_plan,
        geometry=replace(
            source_plan.geometry,
            hidden_size=128,
            q_lora_rank=128,
            kv_lora_rank=30,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=2,
            moe_intermediate_size=512,
            fp8_block_shape=(128, 128),
        ),
    )
    source_schedule = build_pipeline_schedule(source_plan)
    source_state = build_decoder_state_layout(
        source_plan,
        source_schedule,
        context_capacity=8,
        logical_page_size=8,
        packed_kv_width=8,
    )
    source_layout = build_decoder_runtime_weight_layout(
        source_plan,
        source_schedule,
    )
    feature_plan = replace(
        source_plan,
        expert_layout=FEATURE_EXPERT_RUNTIME_LAYOUT,
    )
    feature_schedule = build_pipeline_schedule(feature_plan)
    feature_state = build_decoder_state_layout(
        feature_plan,
        feature_schedule,
        context_capacity=8,
        logical_page_size=8,
        packed_kv_width=8,
    )
    feature_layout = build_decoder_feature_runtime_weight_layout(
        feature_plan,
        feature_schedule,
        source_layout,
    )
    fused_feature_layout = (
        build_decoder_feature_fused_qkv_runtime_weight_layout(
            feature_plan,
            feature_schedule,
            source_layout,
        )
    )
    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4)) for stage in range(8)
    )
    pairs = tuple(
        (groups[stage][slot], groups[(stage + 1) % 8][slot])
        for stage in range(8)
        for slot in range(4)
    )

    with pytest.raises(PlanValidationError, match="expected 32 devices"):
        build_decoder_step_program(
            feature_plan,
            feature_schedule,
            feature_state,
            fused_feature_layout,
            groups,
            pairs,
            sparse_moe_backend="pallas_feature",
            attention_projection_backend="fused_n82_convolution",
        )

    with pytest.raises(PlanValidationError, match="backend and runtime"):
        build_decoder_step_program(
            feature_plan,
            feature_schedule,
            feature_state,
            feature_layout,
            groups,
            pairs,
        )
    with pytest.raises(PlanValidationError, match="backend and runtime"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            sparse_moe_backend="pallas_feature",
        )
    with pytest.raises(PlanValidationError, match="backend is unknown"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            sparse_moe_backend="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="non-default feature"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            feature_output_tile=256,
        )
    with pytest.raises(PlanValidationError, match="requires pallas_feature"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            feature_reconstruct_down_fp32=True,
        )
    with pytest.raises(PlanValidationError, match="incompatible"):
        build_decoder_step_program(
            feature_plan,
            feature_schedule,
            feature_state,
            feature_layout,
            groups,
            pairs,
            sparse_moe_backend="pallas_feature",
            feature_fuse_route_weighting=True,
            feature_reconstruct_down_fp32=True,
        )
    with pytest.raises(PlanValidationError, match="linear backend is unknown"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            linear_backend="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="DSA query backend is unknown"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            dsa_query_backend="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="attention backend and runtime"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            attention_projection_backend="fused_n82_convolution",
        )
    with pytest.raises(
        PlanValidationError,
        match="attention projection backend is unknown",
    ):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            attention_projection_backend="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="token-path flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="event-observation flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_dsa_events=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="internal-observation flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_dsa_internals=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="complete-token path"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_dsa_events=True,
        )
    with pytest.raises(PlanValidationError, match="residual-observation flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_layer_residuals=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="split residual-state flag"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            split_residual_state=1,  # type: ignore[arg-type]
        )
    with pytest.raises(
        PlanValidationError, match="prefill index-input observation flag"
    ):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_prefill_index_inputs=1,  # type: ignore[arg-type]
        )
    with pytest.raises(PlanValidationError, match="complete-token path"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            observe_prefill_index_inputs=True,
        )
    with pytest.raises(PlanValidationError, match="must be isolated"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            observe_dsa_events=True,
            observe_prefill_index_inputs=True,
        )
    with pytest.raises(PlanValidationError, match="isolated DSA observer"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            observe_layer_residuals=True,
        )
    with pytest.raises(PlanValidationError, match="isolated DSA observer"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            observe_dsa_internals=True,
        )
    with pytest.raises(PlanValidationError, match="must be isolated"):
        build_decoder_step_program(
            source_plan,
            source_schedule,
            source_state,
            source_layout,
            groups,
            pairs,
            complete_token_path=True,
            observe_dsa_events=True,
            observe_dsa_internals=True,
            observe_layer_residuals=True,
        )


def test_complete_small_decoder_token_step_runs_all_stages_on_forced_cpu() -> None:
    program = r'''
import json
from dataclasses import replace
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding
from glm_tpu.greenfield.model import build_decoder_runtime_weight_layout, build_decoder_state_layout, build_pipeline_schedule
from glm_tpu.greenfield.runtime import build_decoder_step_program, build_teacher_forced_prefill_program, validate_decoder_step_hlo, validate_teacher_forced_prefill_hlo
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.checkpoint.test_runtime_pack import _small_plan

source_plan = _small_plan()
plan = replace(
    source_plan,
    geometry=replace(source_plan.geometry, vocab_size=96),
)
schedule = build_pipeline_schedule(plan)
state = build_decoder_state_layout(plan, schedule, context_capacity=8, logical_page_size=8, packed_kv_width=8)
weight_layout = build_decoder_runtime_weight_layout(plan, schedule)
groups = tuple(tuple(stage * 4 + slot for slot in range(4)) for stage in range(8))
pairs = tuple((groups[stage][slot], groups[(stage + 1) % 8][slot]) for stage in range(8) for slot in range(4))
decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True)
explicit_default = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=False, split_residual_state=False)
observer = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=True, observe_layer_residuals=True)
internal_observer = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=True, observe_dsa_internals=True)
split = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, split_residual_state=True)
repair_decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_prefill_index_inputs=True, split_residual_state=True)
split_boundary_regression = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=True, observe_layer_residuals=True, split_residual_state=True)
split_internal_regression = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=True, observe_dsa_internals=True, split_residual_state=True)
prefill = build_teacher_forced_prefill_program(decoder, prompt_length=2)
split_prefill = build_teacher_forced_prefill_program(split, prompt_length=2)
repair_prefill = build_teacher_forced_prefill_program(repair_decoder, prompt_length=2)

weights = {}
weight_specs = decoder.input_specs[0]
for spec in weight_layout.specs:
    shape = (32, *spec.shape)
    if spec.dtype == 'F8_E4M3':
        value = np.zeros(shape, np.uint8)
    elif spec.dtype == 'F32':
        value = np.ones(shape, np.float32) if spec.value_class == 'fp8_scale' else np.zeros(shape, np.float32)
    else:
        value = np.ones(shape, dtype=ml_dtypes.bfloat16)
        if spec.name.endswith('key_norm_bias'):
            value.fill(0)
    weights[spec.name] = jax.device_put(value, NamedSharding(decoder.mesh, weight_specs[spec.name]))

regression_weights = dict(weights)
for spec in weight_layout.specs:
    if spec.dtype == 'F8_E4M3' and spec.name.startswith('dense.'):
        value = np.full((32, *spec.shape), 0x38, np.uint8)
        regression_weights[spec.name] = jax.device_put(
            value, NamedSharding(decoder.mesh, weight_specs[spec.name])
        )

residual_host = np.zeros((32, 1, 8), dtype=ml_dtypes.bfloat16)
initial_row = np.asarray([[0.5, -0.25, 0.75, 1.0, -1.0, 0.125, 0.25, -0.5]], dtype=ml_dtypes.bfloat16)
for rank in groups[0]: residual_host[rank] = initial_row
split_residual_host = np.zeros((32, 2, 1, 8), dtype=ml_dtypes.bfloat16)
split_addend = np.asarray([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]], dtype=ml_dtypes.bfloat16)
for rank in groups[0]:
    split_residual_host[rank, 0] = initial_row
    split_residual_host[rank, 1] = split_addend
kv_host = np.ones((32, 1, 1, 2, 8), dtype=ml_dtypes.bfloat16)
index_host = np.ones((32, 1, 1, 2, 2), dtype=ml_dtypes.bfloat16)
metadata_host = np.full((32, 1, decoder.config.metadata_width), -1, np.int32)
metadata_host[..., decoder.config.count_index] = 0
metadata_host[..., decoder.config.producer_index] = -1
metadata_host[..., decoder.config.visited_index] = 0
metadata_host[..., decoder.config.health_index] = 1
metadata_host[..., decoder.config.active_index] = 0
for rank in groups[0]: metadata_host[rank, 0, decoder.config.active_index] = 1
token_host = np.full((32, 1), -1, np.int32)
for rank in groups[0]: token_host[rank, 0] = 5

put = lambda value, spec: jax.device_put(value, NamedSharding(decoder.mesh, spec))
inputs = (
    weights,
    put(residual_host, decoder.input_specs[1]),
    put(kv_host, decoder.input_specs[2]),
    put(index_host, decoder.input_specs[3]),
    put(metadata_host, decoder.input_specs[4]),
    put(token_host, decoder.input_specs[5]),
    put(np.asarray([0], np.int32), decoder.input_specs[6]),
    put(np.asarray([[0]], np.int32), decoder.input_specs[7]),
    put(np.asarray([1], np.int32), decoder.input_specs[8]),
)
split_inputs = (
    weights,
    put(split_residual_host, split.input_specs[1]),
    *inputs[2:],
)
lowered = jax.jit(decoder.execute).lower(*inputs)
default_stablehlo = lowered.as_text()
explicit_default_stablehlo = jax.jit(explicit_default.execute).lower(*inputs).as_text()
compiled = lowered.compile()
split_compiled = jax.jit(split.execute).lower(*split_inputs).compile()
repair_step_compiled = jax.jit(repair_decoder.execute).lower(*split_inputs).compile()
split_boundary_regression_compiled = jax.jit(split_boundary_regression.execute).lower(*split_inputs).compile()
split_internal_regression_compiled = jax.jit(split_internal_regression.execute).lower(*split_inputs).compile()
observer_compiled = jax.jit(observer.execute).lower(*inputs).compile()
internal_observer_compiled = jax.jit(internal_observer.execute).lower(*inputs).compile()
observed = observer_compiled(*inputs)
internal_observed = internal_observer_compiled(*inputs)
first = compiled(*inputs)
second = compiled(weights, *first)
split_first = split_compiled(*split_inputs)
split_second = split_compiled(weights, *split_first)
regression_inputs = (regression_weights, *split_inputs[1:])
repair_step = repair_step_compiled(*regression_inputs)
split_boundary_observed = split_boundary_regression_compiled(*regression_inputs)
split_internal_observed = split_internal_regression_compiled(*regression_inputs)
prefill_inputs = (
    weights,
    *inputs[1:5],
    jnp.asarray([5, 6], dtype=jnp.int32),
    *inputs[6:9],
)
prefill_compiled = jax.jit(prefill.execute).lower(*prefill_inputs).compile()
prefilled = prefill_compiled(*prefill_inputs)
split_prefill_inputs = (
    weights,
    *split_inputs[1:5],
    jnp.asarray([5, 6], dtype=jnp.int32),
    *split_inputs[6:9],
)
split_prefill_compiled = jax.jit(split_prefill.execute).lower(*split_prefill_inputs).compile()
split_prefilled = split_prefill_compiled(*split_prefill_inputs)
repair_prefill_inputs = (
    weights,
    *split_inputs[1:5],
    jnp.asarray([5, 6], dtype=jnp.int32),
    *split_inputs[6:9],
)
repair_wk_names = repair_decoder.prefill_index_weight_names
repair_wk_bits = tuple(weights[bits_name] for bits_name, _ in repair_wk_names)
repair_wk_scales = tuple(weights[scale_name] for _, scale_name in repair_wk_names)
repair_wk_materializer = repair_decoder.materialize_prefill_index_weights
assert repair_wk_materializer is not None
repair_wk_materialized = jax.jit(repair_wk_materializer)(
    repair_wk_bits, repair_wk_scales
)
repair_prefill_inputs = (*repair_prefill_inputs, repair_wk_materialized)
repair_prefill_compiled = jax.jit(repair_prefill.execute).lower(*repair_prefill_inputs).compile()
repair_prefilled = repair_prefill_compiled(*repair_prefill_inputs)
residual, kv, index, metadata, next_token, next_position, next_blocks, next_lengths = map(np.asarray, jax.device_get(second))
split_residual, split_kv, split_index, split_metadata, split_next_token, split_next_position, split_next_blocks, split_next_lengths = map(np.asarray, jax.device_get(split_second))
prefill_values = list(map(np.asarray, jax.device_get(prefilled)))
split_prefill_values = list(map(np.asarray, jax.device_get(split_prefilled)))
repair_prefill_values = list(map(np.asarray, jax.device_get(repair_prefilled)))
repair_step_history = np.asarray(jax.device_get(repair_step[8]))
split_boundary_history = np.asarray(jax.device_get(split_boundary_observed[10]))
split_internal_history = np.asarray(
    jax.device_get(split_internal_observed[10].normalized_hidden)
)
repair_stage_rows = np.stack([
    repair_step_history[group[0], 0] for group in groups
])
normalized_stage_rows = np.stack([
    split_internal_history[group[0], 0] for group in groups
])
rounded_stage_rows = np.stack([
    split_boundary_history[group[0], stage]
    for stage, group in enumerate(groups)
])
observation = np.asarray(jax.device_get(observed[8]))
token_observation = np.asarray(jax.device_get(observed[9]))
layer_residual_observation = np.asarray(jax.device_get(observed[10]))
internal_observation = {
    name: np.asarray(jax.device_get(value))
    for name, value in zip(
        internal_observed[10]._fields,
        internal_observed[10],
        strict=True,
    )
}
observation_rows = []
for stage, group in enumerate(groups):
    rows = observation[list(group), 0]
    observation_rows.append({
        'all_stage_lanes_equal': bool(np.all(rows == rows[0])),
        'row': rows[0].tolist(),
        'stage': stage,
    })
token_observation_lanes = token_observation[list(groups[-1])]
token_candidate_width = observer.config.token_observation_candidates
token_candidate_scores = token_observation_lanes[
    0, token_candidate_width:
].view(np.float32)
active = np.flatnonzero(metadata[:, 0, decoder.config.active_index] == 1)
module = parse_hlo_module(compiled.as_text())
observer_module = parse_hlo_module(observer_compiled.as_text())
internal_observer_module = parse_hlo_module(internal_observer_compiled.as_text())
hlo_contract = validate_decoder_step_hlo(compiled.as_text(), config=decoder.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference', complete_token_path=True)
split_hlo_contract = validate_decoder_step_hlo(split_compiled.as_text(), config=split.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference', complete_token_path=True, split_residual_state=True)
observer_hlo_contract = validate_decoder_step_hlo(observer_compiled.as_text(), config=observer.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference', complete_token_path=True, token_observation_candidates=observer.config.token_observation_candidates)
internal_observer_hlo_contract = validate_decoder_step_hlo(internal_observer_compiled.as_text(), config=internal_observer.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference', complete_token_path=True, token_observation_candidates=internal_observer.config.token_observation_candidates)
prefill_hlo_contract = validate_teacher_forced_prefill_hlo(prefill_compiled.as_text(), program=prefill, schedule=schedule, backend_contract='cpu_reference')
split_prefill_hlo_contract = validate_teacher_forced_prefill_hlo(split_prefill_compiled.as_text(), program=split_prefill, schedule=schedule, backend_contract='cpu_reference')
repair_prefill_hlo_contract = validate_teacher_forced_prefill_hlo(repair_prefill_compiled.as_text(), program=repair_prefill, schedule=schedule, backend_contract='cpu_reference')
counts = {}
for item in module.collectives: counts[item.opcode] = counts.get(item.opcode, 0) + 1
local_groups = tuple(tuple(group) for group in groups)
collectives_local = all(
    item.replica_groups == local_groups
    for item in module.collectives
    if item.opcode in ('all-gather', 'all-reduce')
)
kv_writes = []
index_writes = []
for stage in range(8):
    owner = groups[stage][0]
    kv_writes.append(bool(np.all(kv[owner, 0, 0, 0] == 0)))
    index_writes.append(bool(np.all(index[owner, 0, 0, 0] == 0)))

print(json.dumps({
    'active': active.tolist(),
    'collectives_local': collectives_local,
    'complete_token_path': decoder.complete_token_path,
    'default_observation_off_stablehlo_identical': default_stablehlo == explicit_default_stablehlo,
    'counts': counts,
    'health': sorted(set(metadata[active, 0, decoder.config.health_index].tolist())),
    'hlo_contract': {key: hlo_contract[key] for key in ('collective_count', 'collective_counts', 'passed', 'violations')},
    'index_writes': index_writes,
    'untargeted_owner_cache_unchanged': bool(np.all(kv[[rank for group in groups for rank in group[1:]], 0, 0, 1] == 1)),
    'kv_writes': kv_writes,
    'next_tokens': sorted(set(next_token[active, 0].tolist())),
    'next_position': next_position.tolist(),
    'next_lengths': next_lengths.tolist(),
    'observer': {
        'collective_counts': {
            opcode: sum(item.opcode == opcode for item in observer_module.collectives)
            for opcode in ('all-gather', 'all-reduce', 'collective-permute')
        },
        'host_callback_absent': all(marker not in observer_compiled.as_text().lower() for marker in ('host_callback', 'outside_compilation', 'xla_ffi_python_cpu_callback', 'xla_python_cpu_callback')),
        'hlo_contract': {key: observer_hlo_contract[key] for key in ('passed', 'token_observation_candidates', 'violations')},
        'production_outputs_exact': all(np.array_equal(np.asarray(jax.device_get(observed[index])), np.asarray(jax.device_get(first[index]))) for index in range(8)),
        'rows': observation_rows,
        'layer_residuals': {
            'dtype': layer_residual_observation.dtype.name,
            'shape': list(layer_residual_observation.shape),
            'stage_lane_replication': all(
                np.array_equal(
                    layer_residual_observation[list(group), stage],
                    np.broadcast_to(
                        layer_residual_observation[group[0], stage],
                        layer_residual_observation[list(group), stage].shape,
                    ),
                )
                for stage, group in enumerate(groups)
            ),
        },
        'token_observation': {
            'candidate_ids': token_observation_lanes[0, :token_candidate_width].tolist(),
            'candidate_scores': token_candidate_scores.tolist(),
            'candidate_width': token_candidate_width,
            'final_stage_lanes_equal': bool(np.all(token_observation_lanes == token_observation_lanes[0])),
            'inactive_lanes_sentinel': bool(np.all(token_observation[:groups[-1][0]] == -1)),
        },
    },
    'internal_observer': {
        'collective_counts': {
            opcode: sum(item.opcode == opcode for item in internal_observer_module.collectives)
            for opcode in ('all-gather', 'all-reduce', 'collective-permute')
        },
        'dtypes': {name: value.dtype.name for name, value in internal_observation.items()},
        'finite': all(np.all(np.isfinite(value)) for name, value in internal_observation.items() if name != 'producer_layer_ids'),
        'hlo_contract': {key: internal_observer_hlo_contract[key] for key in ('passed', 'token_observation_candidates', 'violations')},
        'host_callback_absent': all(marker not in internal_observer_compiled.as_text().lower() for marker in ('host_callback', 'outside_compilation', 'xla_ffi_python_cpu_callback', 'xla_python_cpu_callback')),
        'producer_layer_ids': [
            internal_observation['producer_layer_ids'][group[0], 0].item()
            for group in groups
        ],
        'production_outputs_exact': all(np.array_equal(np.asarray(jax.device_get(internal_observed[index])), np.asarray(jax.device_get(first[index]))) for index in range(8)),
        'program_flag': internal_observer.observe_dsa_internals,
        'shapes': {name: list(value.shape) for name, value in internal_observation.items()},
        'stage_lane_replication': all(
            all(
                np.array_equal(
                    value[list(group)],
                    np.broadcast_to(value[group[0]], value[list(group)].shape),
                )
                for value in internal_observation.values()
            )
            for group in groups
        ),
    },
    'block_tables_unchanged': bool(np.array_equal(next_blocks, np.asarray([[0]], np.int32))),
    'positions': sorted(set(tuple(row) for row in metadata[active, 0, :4].tolist())),
    'producer': sorted(set(metadata[active, 0, decoder.config.producer_index].tolist())),
    'prefill': {
        'next_lengths': prefill_values[7].tolist(),
        'next_position': prefill_values[5].tolist(),
        'next_tokens': sorted(set(prefill_values[4][active, 0].tolist())),
        'positions': sorted(set(tuple(row) for row in prefill_values[3][active, 0, :4].tolist())),
        'valid_counts': sorted(set(prefill_values[3][active, 0, decoder.config.count_index].tolist())),
    },
    'prefill_hlo_contract': {
        'collective_count': prefill_hlo_contract['decoder_contract']['collective_count'],
        'collective_counts': prefill_hlo_contract['decoder_contract']['collective_counts'],
        'dead_prompt_rows': prefill_hlo_contract['dead_prompt_rows'],
        'host_transfer_markers': prefill_hlo_contract['host_transfer_markers'],
        'host_transfer_opcodes': prefill_hlo_contract['host_transfer_opcodes'],
        'loop_count': prefill_hlo_contract['loop_count'],
        'loop_contract': {
            key: prefill_hlo_contract['loop_contract'][key]
            for key in (
                'expected_fused_qkv_internal_loop_count',
                'expected_total_loop_count',
                'fused_qkv_internal_loop_count',
                'loop_count',
                'outer_loop_count',
                'passed',
                'unclassified_loops',
                'violations',
            )
        },
        'outer_loop_count': prefill_hlo_contract['outer_loop_count'],
        'passed': prefill_hlo_contract['passed'],
        'prompt_shape_present': prefill_hlo_contract['prompt_shape_parameter_count'] > 0,
        'violations': prefill_hlo_contract['violations'],
    },
    'prefill_index_repair': {
        'backend': repair_prefill_hlo_contract['index_repair_backend'],
        'decoder_contract_passed': repair_prefill_hlo_contract['decoder_contract']['passed'],
        'exact_projection_operand_count': repair_prefill_hlo_contract['index_repair_contract']['exact_projection_operand_count'],
        'expected_call_count': repair_prefill_hlo_contract['index_repair_contract']['expected_call_count'],
        'forbidden_markers': repair_prefill_hlo_contract['index_repair_contract']['forbidden_markers'],
        'full_pod_history_shapes': repair_prefill_hlo_contract['index_repair_contract']['full_pod_history_shapes'],
        'history_estimated_bytes_per_device': repair_prefill_hlo_contract['index_repair_contract']['history_estimated_bytes_per_device'],
        'history_shapes_compact': bool(
            repair_prefill_hlo_contract['index_repair_contract']['history_shapes']
            and len(repair_prefill_hlo_contract['index_repair_contract']['history_shapes']) <= 8
            and all(
                shape['dtype'] == 'bf16' and 32 not in shape['dimensions']
                for shape in repair_prefill_hlo_contract['index_repair_contract']['history_shapes']
            )
        ),
        'loop_count': repair_prefill_hlo_contract['loop_contract']['prefill_index_repair_loop_count'],
        'passed': repair_prefill_hlo_contract['passed'],
        'production_outputs_exact': all(np.array_equal(repair_prefill_values[index], split_prefill_values[index]) for index in range(8)),
        'projection_count': repair_prefill_hlo_contract['index_repair_contract']['projection_count'],
        'recorded_normalized_input_exact': bool(np.array_equal(repair_stage_rows, normalized_stage_rows)),
        'repair_collectives': repair_prefill_hlo_contract['index_repair_contract']['repair_collectives'],
        'rounded_boundary_is_distinct': bool(np.any(normalized_stage_rows != rounded_stage_rows)),
        'split_residual_state': repair_decoder.split_residual_state,
        'violations': repair_prefill_hlo_contract['violations'],
    },
    'residual_exact': bool(np.array_equal(residual[active], np.ones((4, 1, 8), dtype=ml_dtypes.bfloat16))),
    'split': {
        'active': np.flatnonzero(split_metadata[:, 0, split.config.active_index] == 1).tolist(),
        'collective_counts': split_hlo_contract['collective_counts'],
        'finite': bool(np.all(np.isfinite(split_residual))),
        'hlo_contract': {key: split_hlo_contract[key] for key in ('passed', 'residual_transport_count', 'residual_transport_dimensions', 'residual_transport_dtype', 'split_residual_state', 'violations')},
        'next_lengths': split_next_lengths.tolist(),
        'next_position': split_next_position.tolist(),
        'next_tokens': sorted(set(split_next_token[active, 0].tolist())),
        'prefill': {
            'hlo_passed': split_prefill_hlo_contract['passed'],
            'next_lengths': split_prefill_values[7].tolist(),
            'next_position': split_prefill_values[5].tolist(),
            'next_tokens': sorted(set(split_prefill_values[4][active, 0].tolist())),
            'residual_shape': list(split_prefill_values[0].shape),
            'violations': split_prefill_hlo_contract['violations'],
        },
        'program_flag': split.split_residual_state,
        'residual_shape': list(split_residual.shape),
        'visited': sorted(set(split_metadata[active, 0, split.config.visited_index].tolist())),
    },
    'sparse_moe_backend': decoder.sparse_moe_backend,
    'valid_counts': sorted(set(metadata[active, 0, decoder.config.count_index].tolist())),
    'visited': sorted(set(metadata[active, 0, decoder.config.visited_index].tolist())),
}, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["active"] == [0, 1, 2, 3]
    assert result["residual_exact"]
    assert result["complete_token_path"]
    assert result["default_observation_off_stablehlo_identical"]
    assert result["split"] == {
        "active": [0, 1, 2, 3],
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "finite": True,
        "hlo_contract": {
            "passed": True,
            "residual_transport_count": 8,
            "residual_transport_dimensions": [2, 1, 8],
            "residual_transport_dtype": "f32",
            "split_residual_state": True,
            "violations": [],
        },
        "next_lengths": [3],
        "next_position": [2],
        "next_tokens": [0],
        "prefill": {
            "hlo_passed": True,
            "next_lengths": [3],
            "next_position": [2],
            "next_tokens": [0],
            "residual_shape": [32, 2, 1, 8],
            "violations": [],
        },
        "program_flag": True,
        "residual_shape": [32, 2, 1, 8],
        "visited": [255],
    }
    assert result["next_tokens"] == [0]
    assert result["next_position"] == [2]
    assert result["next_lengths"] == [3]
    assert result["prefill_index_repair"] == {
        "backend": "physical_m64_chunk",
        "decoder_contract_passed": True,
        "exact_projection_operand_count": 8,
        "expected_call_count": 8,
        "forbidden_markers": [],
        "full_pod_history_shapes": [],
        "history_estimated_bytes_per_device": 32,
        "history_shapes_compact": True,
        "loop_count": 8,
        "passed": True,
        "production_outputs_exact": True,
        "projection_count": 8,
        "recorded_normalized_input_exact": True,
        "repair_collectives": [],
        "rounded_boundary_is_distinct": True,
        "split_residual_state": True,
        "violations": [],
    }
    assert result["observer"] == {
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "host_callback_absent": True,
        "hlo_contract": {
            "passed": True,
            "token_observation_candidates": 16,
            "violations": [],
        },
        "production_outputs_exact": True,
        "layer_residuals": {
            "dtype": "bfloat16",
            "shape": [32, 9, 8],
            "stage_lane_replication": True,
        },
        "rows": [
            {
                "all_stage_lanes_equal": True,
                "row": [
                    0,
                    -1,
                    -1,
                    -1,
                    0,
                    -8388608,
                    -8388608,
                    -8388608,
                    1,
                    stage,
                ],
                "stage": stage,
            }
            for stage in range(8)
        ],
        "token_observation": {
            "candidate_ids": list(range(16)),
            "candidate_scores": [8.0] * 16,
            "candidate_width": 16,
            "final_stage_lanes_equal": True,
            "inactive_lanes_sentinel": True,
        },
    }
    assert result["internal_observer"] == {
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "dtypes": {
            "current_key": "float32",
            "head_weights": "float32",
            "normalized_hidden": "bfloat16",
            "producer_layer_ids": "int32",
            "q_a_state": "bfloat16",
            "query": "float32",
        },
        "finite": True,
        "hlo_contract": {
            "passed": True,
            "token_observation_candidates": 16,
            "violations": [],
        },
        "host_callback_absent": True,
        "producer_layer_ids": list(range(8)),
        "production_outputs_exact": True,
        "program_flag": True,
        "shapes": {
            "current_key": [32, 1, 2],
            "head_weights": [32, 1, 4],
            "normalized_hidden": [32, 1, 8],
            "producer_layer_ids": [32, 1],
            "q_a_state": [32, 1, 4],
            "query": [32, 1, 4, 2],
        },
        "stage_lane_replication": True,
    }
    assert result["prefill"] == {
        "next_lengths": [3],
        "next_position": [2],
        "next_tokens": [0],
        "positions": [[0, 1, -1, -1]],
        "valid_counts": [2],
    }
    assert result["prefill_hlo_contract"] == {
        "collective_count": 92,
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "dead_prompt_rows": [],
        "host_transfer_markers": [],
        "host_transfer_opcodes": [],
        "loop_count": 1,
        "loop_contract": {
            "expected_fused_qkv_internal_loop_count": 0,
            "expected_total_loop_count": 1,
            "fused_qkv_internal_loop_count": 0,
            "loop_count": 1,
            "outer_loop_count": 1,
            "passed": True,
            "unclassified_loops": [],
            "violations": [],
        },
        "outer_loop_count": 1,
        "passed": True,
        "prompt_shape_present": True,
        "violations": [],
    }
    assert result["block_tables_unchanged"]
    assert result["sparse_moe_backend"] == "reference"
    assert result["positions"] == [[0, 1, -1, -1]]
    assert result["valid_counts"] == [2]
    assert result["producer"] == [7]
    assert result["visited"] == [255]
    assert result["health"] == [1]
    assert all(result["kv_writes"])
    assert all(result["index_writes"])
    assert result["untargeted_owner_cache_unchanged"]
    assert result["collectives_local"]
    assert result["hlo_contract"] == {
        "collective_count": 92,
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "passed": True,
        "violations": [],
    }
    assert result["counts"] == {
        "all-gather": 58,
        "all-reduce": 17,
        "collective-permute": 17,
    }
