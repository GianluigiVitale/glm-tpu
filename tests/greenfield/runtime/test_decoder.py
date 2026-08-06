from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


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


def test_decoder_sparse_backend_fails_closed_on_layout_mismatch() -> None:
    from dataclasses import replace

    from glm_tpu.greenfield.errors import PlanValidationError
    from glm_tpu.greenfield.model import (
        FEATURE_EXPERT_RUNTIME_LAYOUT,
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
    groups = tuple(
        tuple(stage * 4 + slot for slot in range(4)) for stage in range(8)
    )
    pairs = tuple(
        (groups[stage][slot], groups[(stage + 1) % 8][slot])
        for stage in range(8)
        for slot in range(4)
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


def test_complete_small_decoder_token_step_runs_all_stages_on_forced_cpu() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding
from glm_tpu.greenfield.model import build_decoder_runtime_weight_layout, build_decoder_state_layout, build_pipeline_schedule
from glm_tpu.greenfield.runtime import build_decoder_step_program, build_teacher_forced_prefill_program, validate_decoder_step_hlo, validate_teacher_forced_prefill_hlo
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.checkpoint.test_runtime_pack import _small_plan

plan = _small_plan()
schedule = build_pipeline_schedule(plan)
state = build_decoder_state_layout(plan, schedule, context_capacity=8, logical_page_size=8, packed_kv_width=8)
weight_layout = build_decoder_runtime_weight_layout(plan, schedule)
groups = tuple(tuple(stage * 4 + slot for slot in range(4)) for stage in range(8))
pairs = tuple((groups[stage][slot], groups[(stage + 1) % 8][slot]) for stage in range(8) for slot in range(4))
decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True)
explicit_default = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=False)
observer = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, observe_dsa_events=True)
prefill = build_teacher_forced_prefill_program(decoder, prompt_length=2)

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

residual_host = np.zeros((32, 1, 8), dtype=ml_dtypes.bfloat16)
initial_row = np.asarray([[0.5, -0.25, 0.75, 1.0, -1.0, 0.125, 0.25, -0.5]], dtype=ml_dtypes.bfloat16)
for rank in groups[0]: residual_host[rank] = initial_row
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
lowered = jax.jit(decoder.execute).lower(*inputs)
default_stablehlo = lowered.as_text()
explicit_default_stablehlo = jax.jit(explicit_default.execute).lower(*inputs).as_text()
compiled = lowered.compile()
observer_compiled = jax.jit(observer.execute).lower(*inputs).compile()
observed = observer_compiled(*inputs)
first = compiled(*inputs)
second = compiled(weights, *first)
prefill_inputs = (
    weights,
    *inputs[1:5],
    jnp.asarray([5, 6], dtype=jnp.int32),
    *inputs[6:9],
)
prefill_compiled = jax.jit(prefill.execute).lower(*prefill_inputs).compile()
prefilled = prefill_compiled(*prefill_inputs)
residual, kv, index, metadata, next_token, next_position, next_blocks, next_lengths = map(np.asarray, jax.device_get(second))
prefill_values = list(map(np.asarray, jax.device_get(prefilled)))
observation = np.asarray(jax.device_get(observed[8]))
observation_rows = []
for stage, group in enumerate(groups):
    rows = observation[list(group), 0]
    observation_rows.append({
        'all_stage_lanes_equal': bool(np.all(rows == rows[0])),
        'row': rows[0].tolist(),
        'stage': stage,
    })
active = np.flatnonzero(metadata[:, 0, decoder.config.active_index] == 1)
module = parse_hlo_module(compiled.as_text())
observer_module = parse_hlo_module(observer_compiled.as_text())
hlo_contract = validate_decoder_step_hlo(compiled.as_text(), config=decoder.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference', complete_token_path=True)
prefill_hlo_contract = validate_teacher_forced_prefill_hlo(prefill_compiled.as_text(), program=prefill, schedule=schedule, backend_contract='cpu_reference')
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
        'production_outputs_exact': all(np.array_equal(np.asarray(jax.device_get(observed[index])), np.asarray(jax.device_get(first[index]))) for index in range(8)),
        'rows': observation_rows,
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
        'passed': prefill_hlo_contract['passed'],
        'prompt_shape_present': prefill_hlo_contract['prompt_shape_parameter_count'] > 0,
        'violations': prefill_hlo_contract['violations'],
    },
    'residual_exact': bool(np.array_equal(residual[active], np.ones((4, 1, 8), dtype=ml_dtypes.bfloat16))),
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
    assert result["next_tokens"] == [0]
    assert result["next_position"] == [2]
    assert result["next_lengths"] == [3]
    assert result["observer"] == {
        "collective_counts": {
            "all-gather": 58,
            "all-reduce": 17,
            "collective-permute": 17,
        },
        "host_callback_absent": True,
        "production_outputs_exact": True,
        "rows": [
            {
                "all_stage_lanes_equal": True,
                "row": [0, -1, -1, -1, 1, stage],
                "stage": stage,
            }
            for stage in range(8)
        ],
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
