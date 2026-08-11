from __future__ import annotations

import json
import os
import subprocess
import sys


def test_cached_dsa_and_index_share_match_oracles_on_forced_cpu() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.benchmarking.gate_c import stage_local_index_share_gate_c
from glm_tpu.greenfield.kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout
from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract, dsa_index_keys, dsa_query_and_head_weights, dsa_scores, exact_topk
from glm_tpu.greenfield.kernels.reference.fp8 import dequantize_fp8_bits_block_weight
from glm_tpu.greenfield.kernels.reference.linear import linear
from glm_tpu.greenfield.kernels.reference.rmsnorm import rms_norm
from glm_tpu.greenfield.kernels.reference.rotary import apply_rotary_fp32_final_round
from glm_tpu.greenfield.kernels.stage_local import stage_local_dsa_fp8_mapped, stage_local_index_share_fp8_mapped

def bits(value):
    return np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)

def draw(seed, shape):
    return np.random.default_rng(seed).normal(0, 0.125, shape).astype(np.float32)

def collectives(hlo):
    return {
        'ag': hlo.count(' all-gather('),
        'ar': hlo.count(' all-reduce('),
        'cp': hlo.count(' collective-permute('),
    }

devices = np.asarray(jax.devices(), dtype=object)
mesh = Mesh(devices, ('stage',))
rep = NamedSharding(mesh, P())
sharding = lambda spec: NamedSharding(mesh, spec)
layout = StageLocalKvLayout(
    logical_page_size=8,
    local_parallel_size=4,
    packed_cache_width=8,
)
residual = jnp.asarray(
    [[0.5, -0.25, 0.75, 1.0, -1.0, 0.125, 0.25, -0.5]],
    jnp.bfloat16,
)
position = jnp.asarray([7], jnp.int32)
lengths = jnp.asarray([8], jnp.int32)
tables = jnp.asarray([[1, 0]], jnp.int32)
bad_tables = jnp.asarray([[9, 0]], jnp.int32)
slot = jnp.arange(4, dtype=jnp.int32)
input_norm = jnp.ones((8,), jnp.bfloat16)

dsa = DsaNumericalContract(
    hidden_size=8,
    q_lora_rank=4,
    num_heads=4,
    head_dim=2,
    rotary_dim=2,
    top_k=4,
)
q_a_bits = bits(draw(1, (4, 8)))
q_a_scale = np.ones((2, 4), np.float32)
q_a_norm = jnp.ones((4,), jnp.bfloat16)
wq_bits = bits(draw(2, (8, 4)))
wq_scale = np.ones((4, 2), np.float32)
wk_bits = bits(draw(3, (2, 8)))
wk_scale = np.ones((1, 4), np.float32)
key_norm = jnp.ones((2,), jnp.bfloat16)
key_bias = jnp.zeros((2,), jnp.bfloat16)
head_weight = jnp.asarray(draw(4, (4, 8)), jnp.bfloat16)
index_cache = np.asarray(
    draw(5, (4, 2, 2, 2)), dtype=ml_dtypes.bfloat16
)

def dsa_mapped(
    row, cache_container, pos, table, valid_lengths, norm, qa, qa_scale,
    qa_norm, wq, wq_scale, wk, wk_scale, key_norm, key_bias,
    head_weight, local_slot,
):
    result = stage_local_dsa_fp8_mapped(
        row,
        cache_container[0],
        pos,
        table,
        valid_lengths,
        norm,
        qa,
        qa_scale,
        qa_norm,
        wq,
        wq_scale,
        wk,
        wk_scale,
        key_norm,
        key_bias,
        head_weight,
        local_slot[0],
        axis_name='stage',
        contract=dsa,
        cache_layout=layout,
        block_shape=(2, 2),
        dsa_score_precision='default',
    )
    return (
        result.index_cache[None],
        result.selected_positions,
        result.valid_counts,
        result.contract_valid,
    )

dsa_map = jax.shard_map(
    dsa_mapped,
    mesh=mesh,
    in_specs=(
        P(), P('stage', None, None, None), P(), P(), P(), P(), P(), P(),
        P(), P('stage', None), P('stage', None), P(), P(), P(), P(),
        P('stage', None), P('stage'),
    ),
    out_specs=(P('stage', None, None, None), P(), P(), P()),
    check_vma=False,
)
dsa_args = (
    jax.device_put(residual, rep),
    jax.device_put(index_cache, sharding(P('stage', None, None, None))),
    jax.device_put(position, rep),
    jax.device_put(tables, rep),
    jax.device_put(lengths, rep),
    jax.device_put(input_norm, rep),
    jax.device_put(q_a_bits, rep),
    jax.device_put(q_a_scale, rep),
    jax.device_put(q_a_norm, rep),
    jax.device_put(wq_bits, sharding(P('stage', None))),
    jax.device_put(wq_scale, sharding(P('stage', None))),
    jax.device_put(wk_bits, rep),
    jax.device_put(wk_scale, rep),
    jax.device_put(key_norm, rep),
    jax.device_put(key_bias, rep),
    jax.device_put(head_weight, sharding(P('stage', None))),
    jax.device_put(slot, sharding(P('stage'))),
)
dsa_compiled = jax.jit(dsa_map).lower(*dsa_args).compile()
got_cache, got_positions, got_counts, dsa_valid = dsa_compiled(*dsa_args)

def dequantize(value, scale):
    return dequantize_fp8_bits_block_weight(
        jnp.asarray(value), jnp.asarray(scale), block_shape=(2, 2)
    )

normalized = rms_norm(residual, input_norm, epsilon=1e-5)
q_residual = rms_norm(
    linear(normalized, dequantize(q_a_bits, q_a_scale)),
    q_a_norm,
    epsilon=1e-5,
)
query, query_head_weight = dsa_query_and_head_weights(
    normalized,
    q_residual,
    dequantize(wq_bits, wq_scale),
    head_weight,
    position,
    contract=dsa,
)
current_key = dsa_index_keys(
    normalized,
    dequantize(wk_bits, wk_scale),
    key_norm,
    key_bias,
    position,
    contract=dsa,
).astype(jnp.bfloat16)
expected_cache = np.array(index_cache, copy=True)
expected_cache[3, 1, 1] = np.asarray(current_key[0])
global_keys = jnp.asarray(
    np.concatenate([expected_cache[owner, 1] for owner in range(4)], axis=0)
)
expected_selection = exact_topk(
    dsa_scores(
        query,
        global_keys,
        query_head_weight,
        precision='default',
    ),
    lengths,
    top_k=4,
)
bad_dsa_args = list(dsa_args)
bad_dsa_args[3] = jax.device_put(bad_tables, rep)
bad_dsa_cache, _, _, bad_dsa_valid = dsa_compiled(*bad_dsa_args)

mla = MlaNumericalContract(
    num_heads=4,
    kv_lora_rank=4,
    qk_nope_head_dim=2,
    qk_rope_head_dim=2,
    qk_head_dim=4,
    v_head_dim=2,
    packed_cache_width=8,
    top_k=4,
)
selected = jnp.asarray([[0, 2, 4, 7]], jnp.int32)
selected_counts = jnp.asarray([4], jnp.int32)
kv_a_norm = jnp.ones((4,), jnp.bfloat16)
attn_q_a_bits = bits(draw(11, (4, 8)))
attn_q_a_scale = np.ones((2, 4), np.float32)
q_b_bits = bits(draw(12, (16, 4)))
q_b_scale = np.ones((8, 2), np.float32)
kv_a_bits = bits(draw(13, (6, 8)))
kv_a_scale = np.ones((3, 4), np.float32)
kv_b_bits = bits(draw(14, (16, 4)))
kv_b_scale = np.ones((8, 2), np.float32)
o_bits = bits(draw(15, (8, 8)))
o_scale = np.ones((4, 4), np.float32)
kv_cache = np.asarray(
    draw(16, (4, 2, 2, 8)), dtype=ml_dtypes.bfloat16
)

def attention_mapped(
    row, cache_container, selection, counts, pos, table, valid_lengths,
    norm, qa, qa_scale, qa_norm, qb, qb_scale, kva, kva_scale, kva_norm,
    kvb, kvb_scale, output_weight, output_scale, local_slot,
):
    result = stage_local_index_share_fp8_mapped(
        row,
        cache_container[0],
        selection,
        counts,
        pos,
        table,
        valid_lengths,
        norm,
        qa,
        qa_scale,
        qa_norm,
        qb,
        qb_scale,
        kva,
        kva_scale,
        kva_norm,
        kvb,
        kvb_scale,
        output_weight,
        output_scale,
        local_slot[0],
        axis_name='stage',
        contract=mla,
        cache_layout=layout,
        block_shape=(2, 2),
    )
    return result.output, result.cache[None], result.contract_valid

def attention_mapped_table(
    row, cache_container, selection, counts, pos, table, valid_lengths,
    norm, qa, qa_scale, qa_norm, qb, qb_scale, kva, kva_scale, kva_norm,
    kvb, kvb_scale, output_weight, output_scale, local_slot, rope_row,
):
    result = stage_local_index_share_fp8_mapped(
        row,
        cache_container[0],
        selection,
        counts,
        pos,
        table,
        valid_lengths,
        norm,
        qa,
        qa_scale,
        qa_norm,
        qb,
        qb_scale,
        kva,
        kva_scale,
        kva_norm,
        kvb,
        kvb_scale,
        output_weight,
        output_scale,
        local_slot[0],
        axis_name='stage',
        contract=mla,
        cache_layout=layout,
        block_shape=(2, 2),
        main_rope_table_row=rope_row,
    )
    return result.output, result.cache[None], result.contract_valid

attention_map = jax.shard_map(
    attention_mapped,
    mesh=mesh,
    in_specs=(
        P(), P('stage', None, None, None), P(), P(), P(), P(), P(), P(),
        P(), P(), P(), P('stage', None), P('stage', None), P(), P(), P(),
        P('stage', None), P('stage', None), P(None, 'stage'), P(None, 'stage'),
        P('stage'),
    ),
    out_specs=(P(), P('stage', None, None, None), P()),
    check_vma=False,
)
attention_table_map = jax.shard_map(
    attention_mapped_table,
    mesh=mesh,
    in_specs=(
        P(), P('stage', None, None, None), P(), P(), P(), P(), P(), P(),
        P(), P(), P(), P('stage', None), P('stage', None), P(), P(), P(),
        P('stage', None), P('stage', None), P(None, 'stage'), P(None, 'stage'),
        P('stage'), P(),
    ),
    out_specs=(P(), P('stage', None, None, None), P()),
    check_vma=False,
)
attention_args = (
    jax.device_put(residual, rep),
    jax.device_put(kv_cache, sharding(P('stage', None, None, None))),
    jax.device_put(selected, rep),
    jax.device_put(selected_counts, rep),
    jax.device_put(position, rep),
    jax.device_put(tables, rep),
    jax.device_put(lengths, rep),
    jax.device_put(input_norm, rep),
    jax.device_put(attn_q_a_bits, rep),
    jax.device_put(attn_q_a_scale, rep),
    jax.device_put(q_a_norm, rep),
    jax.device_put(q_b_bits, sharding(P('stage', None))),
    jax.device_put(q_b_scale, sharding(P('stage', None))),
    jax.device_put(kv_a_bits, rep),
    jax.device_put(kv_a_scale, rep),
    jax.device_put(kv_a_norm, rep),
    jax.device_put(kv_b_bits, sharding(P('stage', None))),
    jax.device_put(kv_b_scale, sharding(P('stage', None))),
    jax.device_put(o_bits, sharding(P(None, 'stage'))),
    jax.device_put(o_scale, sharding(P(None, 'stage'))),
    jax.device_put(slot, sharding(P('stage'))),
)
attention_compiled = jax.jit(attention_map).lower(*attention_args).compile()
got_output, got_kv_cache, attention_valid = attention_compiled(*attention_args)
rope_row = jnp.asarray([0.0, 1.0], jnp.bfloat16)
attention_table_args = (
    *attention_args,
    jax.device_put(rope_row, rep),
)
attention_table_compiled = jax.jit(attention_table_map).lower(
    *attention_table_args
).compile()
table_output, table_kv_cache, table_attention_valid = (
    attention_table_compiled(*attention_table_args)
)
expected_attention = stage_local_index_share_gate_c(
    residual,
    jnp.asarray(kv_cache),
    selected,
    selected_counts,
    position,
    tables,
    lengths,
    input_norm,
    dequantize(attn_q_a_bits, attn_q_a_scale),
    q_a_norm,
    dequantize(q_b_bits, q_b_scale),
    dequantize(kv_a_bits, kv_a_scale),
    kv_a_norm,
    dequantize(kv_b_bits, kv_b_scale),
    dequantize(o_bits, o_scale),
    mesh=mesh,
    contract=mla,
    cache_layout=layout,
)
current_kv = linear(normalized, dequantize(kv_a_bits, kv_a_scale))
current_latent = rms_norm(
    current_kv[..., :4], kv_a_norm, epsilon=1e-5
)
current_rope = apply_rotary_fp32_final_round(
    current_kv[..., 4:6][:, None, :],
    rope_row[:1][None, None, :],
    rope_row[1:][None, None, :],
    interleaved=True,
)[:, 0, :]
expected_table_cache = np.array(kv_cache, copy=True)
expected_table_cache[3, 1, 1] = np.asarray(
    jnp.concatenate(
        (current_latent, current_rope, jnp.zeros((1, 2), jnp.bfloat16)),
        axis=-1,
    )[0]
)
bad_attention_args = list(attention_args)
bad_attention_args[5] = jax.device_put(bad_tables, rep)
bad_output, bad_kv_cache, bad_attention_valid = attention_compiled(
    *bad_attention_args
)

print(json.dumps({
    'attention_cache_error': float(jnp.max(jnp.abs(
        got_kv_cache.astype(jnp.float32)
        - expected_attention.cache_by_owner.astype(jnp.float32)
    ))),
    'attention_hlo': collectives(attention_compiled.as_text()),
    'attention_output_error': float(jnp.max(jnp.abs(
        got_output.astype(jnp.float32)
        - expected_attention.output.astype(jnp.float32)
    ))),
    'attention_valid': bool(jnp.all(attention_valid)),
    'main_rope_table': {
        'cache_exact': bool(jnp.array_equal(
            table_kv_cache, jnp.asarray(expected_table_cache)
        )),
        'dynamic_trig_absent': all(
            marker not in attention_table_compiled.as_text().lower()
            for marker in (' cosine(', ' sine(')
        ),
        'output_finite': bool(jnp.all(jnp.isfinite(table_output))),
        'valid': bool(jnp.all(table_attention_valid)),
    },
    'bad_attention_cache_unchanged': bool(jnp.array_equal(
        bad_kv_cache, jnp.asarray(kv_cache)
    )),
    'bad_attention_output_finite': bool(jnp.all(jnp.isfinite(bad_output))),
    'bad_attention_valid': bool(jnp.any(bad_attention_valid)),
    'bad_dsa_cache_unchanged': bool(jnp.array_equal(
        bad_dsa_cache, jnp.asarray(index_cache)
    )),
    'bad_dsa_valid': bool(jnp.any(bad_dsa_valid)),
    'dsa_cache_error': float(jnp.max(jnp.abs(
        got_cache.astype(jnp.float32) - jnp.asarray(expected_cache).astype(jnp.float32)
    ))),
    'dsa_counts_exact': bool(jnp.array_equal(
        got_counts, expected_selection.valid_counts
    )),
    'dsa_hlo': collectives(dsa_compiled.as_text()),
    'dsa_positions_exact': bool(jnp.array_equal(
        got_positions, expected_selection.positions
    )),
    'dsa_valid': bool(jnp.all(dsa_valid)),
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
    assert result["dsa_cache_error"] == 0.0
    assert result["dsa_positions_exact"]
    assert result["dsa_counts_exact"]
    assert result["dsa_valid"]
    assert result["dsa_hlo"] == {"ag": 3, "ar": 0, "cp": 0}
    assert result["attention_output_error"] == 0.0
    assert result["attention_cache_error"] == 0.0
    assert result["attention_valid"]
    assert result["attention_hlo"] == {"ag": 4, "ar": 1, "cp": 0}
    assert result["main_rope_table"] == {
        "cache_exact": True,
        "dynamic_trig_absent": True,
        "output_finite": True,
        "valid": True,
    }
    assert not result["bad_dsa_valid"]
    assert result["bad_dsa_cache_unchanged"]
    assert not result["bad_attention_valid"]
    assert result["bad_attention_cache_unchanged"]
    assert result["bad_attention_output_finite"]
