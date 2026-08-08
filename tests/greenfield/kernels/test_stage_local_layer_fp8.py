from __future__ import annotations

import json
import os
import subprocess
import sys


def test_fused_full_dsa_dense_layer_matches_component_path_on_forced_cpu() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.layer import AttentionFp8Weights, DenseFp8Weights, DsaFp8Weights, MoeFp8Weights, stage_local_transformer_layer_fp8_mapped, stage_local_transformer_layer_fp8_split_mapped
from glm_tpu.greenfield.kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout
from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
from glm_tpu.greenfield.kernels.reference.linear import residual_add
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.kernels.reference.rmsnorm import fused_add_rms_norm, rms_norm
from glm_tpu.greenfield.kernels.stage_local import _stage_fp8_linear, stage_local_dense_fp8_mapped, stage_local_dsa_fp8_mapped, stage_local_index_share_fp8_mapped, stage_local_moe_fp8_mapped

def bits(value):
    return np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)

def draw(seed, shape):
    return np.random.default_rng(seed).normal(0, 0.125, shape).astype(np.float32)

devices = np.asarray(jax.devices(), dtype=object)
mesh = Mesh(devices, ('stage',))
layout = StageLocalKvLayout(logical_page_size=8, local_parallel_size=4, packed_cache_width=8)
dsa_contract = DsaNumericalContract(hidden_size=8, q_lora_rank=4, num_heads=4, head_dim=2, rotary_dim=2, top_k=4)
mla_contract = MlaNumericalContract(num_heads=4, kv_lora_rank=4, qk_nope_head_dim=2, qk_rope_head_dim=2, qk_head_dim=4, v_head_dim=2, packed_cache_width=8, top_k=4)
moe_contract = GlmMoeNumericalContract(hidden_size=8, intermediate_size=8, num_experts=16, top_k=4, stage_size=4, fp8_block_shape=(2, 2))

host = {
    'residual': np.asarray([[0.5, -0.25, 0.75, 1.0, -1.0, 0.125, 0.25, -0.5]], dtype=ml_dtypes.bfloat16),
    'split_residual': np.asarray([[-0.375, 0.5, -0.125, 0.25, 0.75, -0.625, 0.5, 0.25]], dtype=ml_dtypes.bfloat16),
    'kv_cache': np.asarray(draw(1, (4, 2, 2, 8)), dtype=ml_dtypes.bfloat16),
    'index_cache': np.asarray(draw(2, (4, 2, 2, 2)), dtype=ml_dtypes.bfloat16),
    'selected': np.full((1, 4), -1, np.int32),
    'selected_counts': np.zeros((1,), np.int32),
    'position': np.asarray([7], np.int32),
    'tables': np.asarray([[1, 0]], np.int32),
    'lengths': np.asarray([8], np.int32),
    'input_norm': np.ones((8,), dtype=ml_dtypes.bfloat16),
    'post_norm': np.ones((8,), dtype=ml_dtypes.bfloat16),
    'q_a_bits': bits(draw(3, (4, 8))),
    'q_a_scale': np.ones((2, 4), np.float32),
    'q_a_norm': np.ones((4,), dtype=ml_dtypes.bfloat16),
    'q_b_bits': bits(draw(4, (16, 4))),
    'q_b_scale': np.ones((8, 2), np.float32),
    'kv_a_bits': bits(draw(5, (6, 8))),
    'kv_a_scale': np.ones((3, 4), np.float32),
    'kv_a_norm': np.ones((4,), dtype=ml_dtypes.bfloat16),
    'kv_b_bits': bits(draw(6, (16, 4))),
    'kv_b_scale': np.ones((8, 2), np.float32),
    'o_bits': bits(draw(7, (8, 8))),
    'o_scale': np.ones((4, 4), np.float32),
    'wq_bits': bits(draw(8, (8, 4))),
    'wq_scale': np.ones((4, 2), np.float32),
    'wk_bits': bits(draw(9, (2, 8))),
    'wk_scale': np.ones((1, 4), np.float32),
    'key_norm': np.ones((2,), dtype=ml_dtypes.bfloat16),
    'key_bias': np.zeros((2,), dtype=ml_dtypes.bfloat16),
    'head_weight': np.asarray(draw(10, (4, 8)), dtype=ml_dtypes.bfloat16),
    'dense_gate_bits': bits(draw(11, (8, 8))),
    'dense_gate_scale': np.ones((4, 4), np.float32),
    'dense_up_bits': bits(draw(12, (8, 8))),
    'dense_up_scale': np.ones((4, 4), np.float32),
    'dense_down_bits': bits(draw(13, (8, 8))),
    'dense_down_scale': np.ones((4, 4), np.float32),
    'router': np.asarray(draw(14, (16, 8)), dtype=ml_dtypes.bfloat16),
    'bias': np.zeros((16,), np.float32),
    'expert_gate_bits': bits(draw(15, (16, 8, 8))),
    'expert_gate_scale': np.ones((16, 4, 4), np.float32),
    'expert_up_bits': bits(draw(16, (16, 8, 8))),
    'expert_up_scale': np.ones((16, 4, 4), np.float32),
    'expert_down_bits': bits(draw(17, (16, 8, 8))),
    'expert_down_scale': np.ones((16, 4, 4), np.float32),
    'shared_gate_bits': bits(draw(18, (8, 8))),
    'shared_gate_scale': np.ones((4, 4), np.float32),
    'shared_up_bits': bits(draw(19, (8, 8))),
    'shared_up_scale': np.ones((4, 4), np.float32),
    'shared_down_bits': bits(draw(20, (8, 8))),
    'shared_down_scale': np.ones((4, 4), np.float32),
    'health': np.ones((1,), np.bool_),
    'slot': np.arange(4, dtype=np.int32),
}
specs = {name: P() for name in host}
specs.update({
    'kv_cache': P('stage', None, None, None),
    'index_cache': P('stage', None, None, None),
    'q_b_bits': P('stage', None),
    'q_b_scale': P('stage', None),
    'kv_b_bits': P('stage', None),
    'kv_b_scale': P('stage', None),
    'o_bits': P(None, 'stage'),
    'o_scale': P(None, 'stage'),
    'wq_bits': P('stage', None),
    'wq_scale': P('stage', None),
    'head_weight': P('stage', None),
    'dense_gate_bits': P('stage', None),
    'dense_gate_scale': P('stage', None),
    'dense_up_bits': P('stage', None),
    'dense_up_scale': P('stage', None),
    'dense_down_bits': P(None, 'stage'),
    'dense_down_scale': P(None, 'stage'),
    'expert_gate_bits': P('stage', None, None),
    'expert_gate_scale': P('stage', None, None),
    'expert_up_bits': P('stage', None, None),
    'expert_up_scale': P('stage', None, None),
    'expert_down_bits': P('stage', None, None),
    'expert_down_scale': P('stage', None, None),
    'shared_gate_bits': P('stage', None),
    'shared_gate_scale': P('stage', None),
    'shared_up_bits': P('stage', None),
    'shared_up_scale': P('stage', None),
    'shared_down_bits': P(None, 'stage'),
    'shared_down_scale': P(None, 'stage'),
    'slot': P('stage'),
})
values = {
    name: jax.device_put(value, NamedSharding(mesh, specs[name]))
    for name, value in host.items()
}

def weights(x):
    attention = AttentionFp8Weights(x['q_a_bits'], x['q_a_scale'], x['q_a_norm'], x['q_b_bits'], x['q_b_scale'], x['kv_a_bits'], x['kv_a_scale'], x['kv_a_norm'], x['kv_b_bits'], x['kv_b_scale'], x['o_bits'], x['o_scale'])
    dsa = DsaFp8Weights(x['wq_bits'], x['wq_scale'], x['wk_bits'], x['wk_scale'], x['key_norm'], x['key_bias'], x['head_weight'])
    dense = DenseFp8Weights(x['dense_gate_bits'], x['dense_gate_scale'], x['dense_up_bits'], x['dense_up_scale'], x['dense_down_bits'], x['dense_down_scale'])
    return attention, dsa, dense

def moe_weights(x):
    return MoeFp8Weights(x['router'], x['bias'], x['expert_gate_bits'], x['expert_gate_scale'], x['expert_up_bits'], x['expert_up_scale'], x['expert_down_bits'], x['expert_down_scale'], x['shared_gate_bits'], x['shared_gate_scale'], x['shared_up_bits'], x['shared_up_scale'], x['shared_down_bits'], x['shared_down_scale'])

def fused(x):
    attention, dsa, dense = weights(x)
    result = stage_local_transformer_layer_fp8_mapped(
        x['residual'], x['kv_cache'][0], x['index_cache'][0], x['selected'], x['selected_counts'], x['position'], x['tables'], x['lengths'], x['input_norm'], x['post_norm'], attention, dsa, dense, None, x['health'], x['slot'][0], axis_name='stage', indexer_kind='full', mlp_kind='dense', dsa_contract=dsa_contract, mla_contract=mla_contract, moe_contract=moe_contract, cache_layout=layout, block_shape=(2, 2)
    )
    return result.output, result.kv_cache[None], result.index_cache[None], result.selected_positions, result.selected_valid_counts, result.route_indices, result.route_weights, result.contract_valid

def component(x):
    attention, dsa, dense = weights(x)
    selected = stage_local_dsa_fp8_mapped(
        x['residual'], x['index_cache'][0], x['position'], x['tables'], x['lengths'], x['input_norm'], attention.q_a_bits, attention.q_a_scale, attention.q_a_norm_weight, dsa.wq_b_bits, dsa.wq_b_scale, dsa.wk_bits, dsa.wk_scale, dsa.key_norm_weight, dsa.key_norm_bias, dsa.head_weight, x['slot'][0], axis_name='stage', contract=dsa_contract, cache_layout=layout, block_shape=(2, 2)
    )
    attended = stage_local_index_share_fp8_mapped(
        x['residual'], x['kv_cache'][0], selected.selected_positions, selected.valid_counts, x['position'], x['tables'], x['lengths'], x['input_norm'], attention.q_a_bits, attention.q_a_scale, attention.q_a_norm_weight, attention.q_b_bits, attention.q_b_scale, attention.kv_a_bits, attention.kv_a_scale, attention.kv_a_norm_weight, attention.kv_b_bits, attention.kv_b_scale, attention.o_bits, attention.o_scale, x['slot'][0], axis_name='stage', contract=mla_contract, cache_layout=layout, block_shape=(2, 2)
    )
    output = stage_local_dense_fp8_mapped(
        attended.output, x['post_norm'], dense.gate_bits, dense.gate_scale, dense.up_bits, dense.up_scale, dense.down_bits, dense.down_scale, axis_name='stage', block_shape=(2, 2)
    )
    routes = jnp.full((1, 4), -1, jnp.int32)
    route_weights = jnp.zeros((1, 4), jnp.float32)
    return output, attended.cache[None], selected.index_cache[None], selected.selected_positions, selected.valid_counts, routes, route_weights, x['health'] & selected.contract_valid & attended.contract_valid

def fused_sparse(x):
    attention, _, _ = weights(x)
    result = stage_local_transformer_layer_fp8_mapped(
        x['residual'], x['kv_cache'][0], x['index_cache'][0], x['selected'], x['selected_counts'], x['position'], x['tables'], x['lengths'], x['input_norm'], x['post_norm'], attention, None, None, moe_weights(x), x['health'], x['slot'][0], axis_name='stage', indexer_kind='shared', mlp_kind='sparse', dsa_contract=dsa_contract, mla_contract=mla_contract, moe_contract=moe_contract, cache_layout=layout, block_shape=(2, 2)
    )
    return result.output, result.kv_cache[None], result.index_cache[None], result.selected_positions, result.selected_valid_counts, result.route_indices, result.route_weights, result.contract_valid

def component_sparse(x):
    attention, _, _ = weights(x)
    attended = stage_local_index_share_fp8_mapped(
        x['residual'], x['kv_cache'][0], x['selected'], x['selected_counts'], x['position'], x['tables'], x['lengths'], x['input_norm'], attention.q_a_bits, attention.q_a_scale, attention.q_a_norm_weight, attention.q_b_bits, attention.q_b_scale, attention.kv_a_bits, attention.kv_a_scale, attention.kv_a_norm_weight, attention.kv_b_bits, attention.kv_b_scale, attention.o_bits, attention.o_scale, x['slot'][0], axis_name='stage', contract=mla_contract, cache_layout=layout, block_shape=(2, 2)
    )
    moe = moe_weights(x)
    normalized = rms_norm(attended.output, x['post_norm'], epsilon=1e-5)
    update, routes, route_weights = stage_local_moe_fp8_mapped(
        normalized, moe.router_weight, moe.correction_bias, moe.expert_gate_bits, moe.expert_gate_scale, moe.expert_up_bits, moe.expert_up_scale, moe.expert_down_bits, moe.expert_down_scale, moe.shared_gate_bits, moe.shared_gate_scale, moe.shared_up_bits, moe.shared_up_scale, moe.shared_down_bits, moe.shared_down_scale, x['slot'][0], axis_name='stage', contract=moe_contract
    )
    return residual_add(attended.output, update), attended.cache[None], x['index_cache'], x['selected'], x['selected_counts'], routes, route_weights, x['health'] & attended.contract_valid

def split_fused_sparse(x):
    attention, _, _ = weights(x)
    result = stage_local_transformer_layer_fp8_split_mapped(
        x['residual'], x['split_residual'], x['kv_cache'][0], x['index_cache'][0], x['selected'], x['selected_counts'], x['position'], x['tables'], x['lengths'], x['input_norm'], x['post_norm'], attention, None, None, moe_weights(x), x['health'], x['slot'][0], axis_name='stage', indexer_kind='shared', mlp_kind='sparse', dsa_contract=dsa_contract, mla_contract=mla_contract, moe_contract=moe_contract, cache_layout=layout, block_shape=(2, 2)
    )
    return result.hidden_states, result.residual, result.kv_cache[None], result.index_cache[None], result.selected_positions, result.selected_valid_counts, result.route_indices, result.route_weights, result.contract_valid

def split_component_sparse(x):
    attention, _, _ = weights(x)
    normalized, combined = fused_add_rms_norm(x['residual'], x['split_residual'], x['input_norm'], epsilon=1e-5)
    q_residual = rms_norm(_stage_fp8_linear(normalized, attention.q_a_bits, attention.q_a_scale, block_shape=(2, 2), backend='reference', interpret=False), attention.q_a_norm_weight, epsilon=1e-5)
    attended = stage_local_index_share_fp8_mapped(
        combined, x['kv_cache'][0], x['selected'], x['selected_counts'], x['position'], x['tables'], x['lengths'], x['input_norm'], attention.q_a_bits, attention.q_a_scale, attention.q_a_norm_weight, attention.q_b_bits, attention.q_b_scale, attention.kv_a_bits, attention.kv_a_scale, attention.kv_a_norm_weight, attention.kv_b_bits, attention.kv_b_scale, attention.o_bits, attention.o_scale, x['slot'][0], axis_name='stage', contract=mla_contract, cache_layout=layout, block_shape=(2, 2), precomputed_normalized=normalized, precomputed_q_residual=q_residual, add_residual=False
    )
    normalized_mlp, post_residual = fused_add_rms_norm(attended.output, combined, x['post_norm'], epsilon=1e-5)
    moe = moe_weights(x)
    next_hidden, routes, route_weights = stage_local_moe_fp8_mapped(
        normalized_mlp, moe.router_weight, moe.correction_bias, moe.expert_gate_bits, moe.expert_gate_scale, moe.expert_up_bits, moe.expert_up_scale, moe.expert_down_bits, moe.expert_down_scale, moe.shared_gate_bits, moe.shared_gate_scale, moe.shared_up_bits, moe.shared_up_scale, moe.shared_down_bits, moe.shared_down_scale, x['slot'][0], axis_name='stage', contract=moe_contract
    )
    return next_hidden, post_residual, attended.cache[None], x['index_cache'], x['selected'], x['selected_counts'], routes, route_weights, x['health'] & attended.contract_valid

out_specs = (P(), P('stage', None, None, None), P('stage', None, None, None), P(), P(), P(), P(), P())
fused_map = jax.shard_map(fused, mesh=mesh, in_specs=(specs,), out_specs=out_specs, check_vma=False)
component_map = jax.shard_map(component, mesh=mesh, in_specs=(specs,), out_specs=out_specs, check_vma=False)
fused_compiled = jax.jit(fused_map).lower(values).compile()
component_compiled = jax.jit(component_map).lower(values).compile()
actual = fused_compiled(values)
expected = component_compiled(values)
exact = [bool(jnp.array_equal(left, right)) for left, right in zip(actual, expected)]

sparse_values = dict(values)
sparse_values['selected'] = jax.device_put(jnp.asarray([[0, 2, 4, 7]], jnp.int32), NamedSharding(mesh, P()))
sparse_values['selected_counts'] = jax.device_put(jnp.asarray([4], jnp.int32), NamedSharding(mesh, P()))
fused_sparse_map = jax.shard_map(fused_sparse, mesh=mesh, in_specs=(specs,), out_specs=out_specs, check_vma=False)
component_sparse_map = jax.shard_map(component_sparse, mesh=mesh, in_specs=(specs,), out_specs=out_specs, check_vma=False)
fused_sparse_compiled = jax.jit(fused_sparse_map).lower(sparse_values).compile()
component_sparse_compiled = jax.jit(component_sparse_map).lower(sparse_values).compile()
sparse_actual = fused_sparse_compiled(sparse_values)
sparse_expected = component_sparse_compiled(sparse_values)
sparse_exact = [bool(jnp.array_equal(left, right)) for left, right in zip(sparse_actual, sparse_expected)]

split_out_specs = (P(), P(), P('stage', None, None, None), P('stage', None, None, None), P(), P(), P(), P(), P())
split_fused_map = jax.shard_map(split_fused_sparse, mesh=mesh, in_specs=(specs,), out_specs=split_out_specs, check_vma=False)
split_component_map = jax.shard_map(split_component_sparse, mesh=mesh, in_specs=(specs,), out_specs=split_out_specs, check_vma=False)
split_fused_compiled = jax.jit(split_fused_map).lower(sparse_values).compile()
split_component_compiled = jax.jit(split_component_map).lower(sparse_values).compile()
split_actual = split_fused_compiled(sparse_values)
split_expected = split_component_compiled(sparse_values)
split_exact = [bool(jnp.array_equal(left, right)) for left, right in zip(split_actual, split_expected)]

def collectives(hlo):
    return {'ag': hlo.count(' all-gather('), 'ar': hlo.count(' all-reduce('), 'cp': hlo.count(' collective-permute(')}

print(json.dumps({
    'collectives': collectives(fused_compiled.as_text()),
    'component_collectives': collectives(component_compiled.as_text()),
    'exact': exact,
    'health': bool(jnp.all(actual[-1])),
    'route_sentinel': bool(jnp.all(actual[5] == -1)),
    'sparse_collectives': collectives(fused_sparse_compiled.as_text()),
    'sparse_component_collectives': collectives(component_sparse_compiled.as_text()),
    'sparse_exact': sparse_exact,
    'sparse_health': bool(jnp.all(sparse_actual[-1])),
    'split_collectives': collectives(split_fused_compiled.as_text()),
    'split_component_collectives': collectives(split_component_compiled.as_text()),
    'split_exact': split_exact,
    'split_health': bool(jnp.all(split_actual[-1])),
}, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=4".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert all(result["exact"])
    assert result["health"]
    assert result["route_sentinel"]
    assert result["collectives"] == {"ag": 7, "ar": 2, "cp": 0}
    assert result["component_collectives"] == {"ag": 7, "ar": 2, "cp": 0}
    assert all(result["sparse_exact"])
    assert result["sparse_health"]
    assert result["sparse_collectives"] == {"ag": 4, "ar": 2, "cp": 0}
    assert result["sparse_component_collectives"] == {
        "ag": 4,
        "ar": 2,
        "cp": 0,
    }
    assert all(result["split_exact"])
    assert result["split_health"]
    assert result["split_collectives"] == {"ag": 4, "ar": 2, "cp": 0}
    assert result["split_component_collectives"] == {
        "ag": 4,
        "ar": 2,
        "cp": 0,
    }
