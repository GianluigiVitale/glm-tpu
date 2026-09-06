from __future__ import annotations

import json
import os
import subprocess
import sys


def test_ws32_prepare_and_dsa_match_forced_32_reference() -> None:
    program = r'''
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.reference.attention import StageLocalKvLayout
from glm_tpu.greenfield.kernels.reference.dsa import (
    DsaNumericalContract,
    dsa_index_keys,
    dsa_query_and_head_weights,
    dsa_scores,
    local_topk_candidates,
)
from glm_tpu.greenfield.kernels.reference.fp8 import (
    dequantize_fp8_bits_block_weight,
)
from glm_tpu.greenfield.kernels.ws32_layer import (
    Ws32DsaResult,
    Ws32DsaWeights,
    Ws32PreparedAttention,
    Ws32QkvAWeights,
    ws32_dsa_mapped,
    ws32_prepare_attention_mapped,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
    tpu_info.ChipVersion.TPU_V4, 1
)
tpu_info.get_tpu_info.cache_clear()

def bits(value):
    return np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)

def draw(seed, shape, scale=0.125):
    return np.random.default_rng(seed).normal(0, scale, shape).astype(np.float32)

devices = np.asarray(jax.devices(), dtype=object).reshape(8, 4)
mesh = Mesh(devices, ("expert", "feature"))
block = (16, 16)
hidden_size = 128
q_rank = 64
kv_rank = 32
rope_width = 16
contract = DsaNumericalContract(
    hidden_size=hidden_size,
    q_lora_rank=q_rank,
    num_heads=8,
    head_dim=16,
    rotary_dim=8,
    top_k=8,
)
layout = StageLocalKvLayout(
    logical_page_size=16,
    local_parallel_size=8,
    packed_cache_width=48,
)

residual = np.asarray(draw(1, (1, hidden_size), 0.25), dtype=ml_dtypes.bfloat16)
input_norm = np.asarray(
    np.random.default_rng(2).uniform(0.75, 1.25, (hidden_size,)),
    dtype=ml_dtypes.bfloat16,
)
q_a = bits(draw(3, (q_rank, hidden_size)))
q_a_scale = np.random.default_rng(4).uniform(
    0.5, 1.25, (q_rank // 16, hidden_size // 16)
).astype(np.float32)
q_a_norm = np.asarray(
    np.random.default_rng(5).uniform(0.75, 1.25, (q_rank,)),
    dtype=ml_dtypes.bfloat16,
)
kv_a = bits(draw(6, (kv_rank + rope_width, hidden_size)))
kv_a_scale = np.random.default_rng(7).uniform(
    0.5, 1.25, ((kv_rank + rope_width) // 16, hidden_size // 16)
).astype(np.float32)
kv_a_norm = np.asarray(
    np.random.default_rng(8).uniform(0.75, 1.25, (kv_rank,)),
    dtype=ml_dtypes.bfloat16,
)
wq_b = bits(draw(9, (contract.num_heads * contract.head_dim, q_rank)))
wq_b_scale = np.random.default_rng(10).uniform(
    0.5, 1.25, (contract.num_heads * contract.head_dim // 16, q_rank // 16)
).astype(np.float32)
wk = bits(draw(11, (contract.head_dim, hidden_size)))
wk_scale = np.random.default_rng(12).uniform(
    0.5, 1.25, (contract.head_dim // 16, hidden_size // 16)
).astype(np.float32)
key_norm_weight = np.asarray(
    np.random.default_rng(13).uniform(0.75, 1.25, (contract.head_dim,)),
    dtype=ml_dtypes.bfloat16,
)
key_norm_bias = np.asarray(draw(14, (contract.head_dim,), 0.01), dtype=ml_dtypes.bfloat16)
head_weight = np.asarray(
    draw(15, (contract.num_heads, hidden_size), 0.2),
    dtype=ml_dtypes.bfloat16,
)
index_cache = np.asarray(
    draw(16, (1, layout.logical_page_size, contract.head_dim), 0.2),
    dtype=ml_dtypes.bfloat16,
)
position = np.asarray([7], dtype=np.int32)
block_tables = np.asarray([[0]], dtype=np.int32)
context_lengths = np.asarray([8], dtype=np.int32)

qkv_values = Ws32QkvAWeights(
    input_norm, q_a, q_a_scale, q_a_norm, kv_a, kv_a_scale, kv_a_norm
)
qkv_specs = Ws32QkvAWeights(
    P("feature"), P(None, "feature"), P(None, "feature"), P(),
    P(None, "feature"), P(None, "feature"), P(),
)
dsa_values = Ws32DsaWeights(
    wq_b, wq_b_scale, wk, wk_scale,
    key_norm_weight, key_norm_bias, head_weight,
)
dsa_specs = Ws32DsaWeights(
    P("expert", None), P("expert", None),
    P(None, "feature"), P(None, "feature"),
    P(), P(), P("expert", "feature"),
)

def mapped(residual_local, cache_local, pos, tables, lengths, qkv, dsa):
    prepared = ws32_prepare_attention_mapped(
        residual_local,
        qkv,
        hidden_size=hidden_size,
        block_shape=block,
        linear_interpret=True,
    )
    return ws32_dsa_mapped(
        prepared,
        cache_local,
        pos,
        tables,
        lengths,
        dsa,
        contract=contract,
        cache_layout=layout,
        block_shape=block,
        linear_interpret=True,
    )

sharded = jax.shard_map(
    mapped,
    mesh=mesh,
    in_specs=(
        P(None, "feature"), P(None, "expert", None), P(), P(), P(),
        qkv_specs, dsa_specs,
    ),
    out_specs=Ws32DsaResult(
        P(None, "expert", None), P(), P(), P(), P(),
    ),
    check_vma=False,
)

def put(value, spec):
    if isinstance(value, tuple):
        return type(value)(*(put(v, s) for v, s in zip(value, spec)))
    return jax.device_put(value, NamedSharding(mesh, spec))

arguments = (
    put(residual, P(None, "feature")),
    put(index_cache, P(None, "expert", None)),
    put(position, P()),
    put(block_tables, P()),
    put(context_lengths, P()),
    put(qkv_values, qkv_specs),
    put(dsa_values, dsa_specs),
)
compiled = jax.jit(sharded).lower(*arguments).compile()
actual = compiled(*arguments)

prepare_sharded = jax.shard_map(
    lambda value, qkv: ws32_prepare_attention_mapped(
        value, qkv, hidden_size=hidden_size,
        block_shape=block, linear_interpret=True,
    ),
    mesh=mesh,
    in_specs=(P(None, "feature"), qkv_specs),
    out_specs=Ws32PreparedAttention(
        P(None, "feature"), P(None, "feature"), P(), P()
    ),
    check_vma=False,
)
prepared_global = jax.jit(prepare_sharded)(arguments[0], arguments[5])
normalized = jnp.asarray(jax.device_get(prepared_global.normalized_local))
q_residual = jnp.asarray(jax.device_get(prepared_global.q_residual))
wq_weight = dequantize_fp8_bits_block_weight(
    jnp.asarray(wq_b), jnp.asarray(wq_b_scale), block_shape=block
)
wk_weight = dequantize_fp8_bits_block_weight(
    jnp.asarray(wk), jnp.asarray(wk_scale), block_shape=block
)
expected_query, expected_head_weights = dsa_query_and_head_weights(
    normalized,
    q_residual,
    wq_weight,
    jnp.asarray(head_weight),
    jnp.asarray(position),
    contract=contract,
)
current_key = dsa_index_keys(
    normalized,
    wk_weight,
    jnp.asarray(key_norm_weight),
    jnp.asarray(key_norm_bias),
    jnp.asarray(position),
    contract=contract,
).astype(jnp.bfloat16)
expected_cache = jnp.asarray(index_cache).at[0, 7].set(current_key[0])
scores = dsa_scores(
    expected_query,
    expected_cache.reshape(-1, contract.head_dim),
    expected_head_weights,
    precision="highest",
)
expected_scores, expected_positions = local_topk_candidates(
    scores,
    jnp.arange(layout.logical_page_size, dtype=jnp.int32),
    jnp.asarray(context_lengths),
    top_k=contract.top_k,
)

module = parse_hlo_module(compiled.as_text())
print(json.dumps({
    "cache_max_abs": float(jnp.max(jnp.abs(
        actual.index_cache_local.astype(jnp.float32)
        - expected_cache.astype(jnp.float32)
    ))),
    "positions_equal": bool(np.array_equal(
        np.asarray(actual.selected_positions), np.asarray(expected_positions)
    )),
    "score_max_abs": float(jnp.max(jnp.abs(
        actual.selected_scores - expected_scores
    ))),
    "valid_counts": np.asarray(actual.selected_valid_counts).tolist(),
    "contract_valid": np.asarray(actual.contract_valid).tolist(),
    "cache_sharding": str(actual.index_cache_local.sharding.spec),
    "selected_sharding": str(actual.selected_positions.sharding.spec),
    "collective_count": len(module.collectives),
    "collectives": [
        [item.raw_opcode, item.maximum_group_size, item.op_name]
        for item in module.collectives
    ],
    "maximum_group_size": max(
        item.maximum_group_size for item in module.collectives
    ),
    "async_collectives": [
        item.raw_opcode for item in module.collectives
        if item.raw_opcode.endswith(("-start", "-done"))
    ],
    "has_batch32_hidden": (
        "bf16[32,128]" in compiled.as_text()
        or "f32[32,128]" in compiled.as_text()
    ),
}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["cache_max_abs"] <= 0.015625
    assert result["positions_equal"]
    assert result["score_max_abs"] <= 2**-10
    assert result["valid_counts"] == [8]
    assert result["contract_valid"] == [True]
    assert result["cache_sharding"] == "P(None, 'expert')"
    assert result["selected_sharding"] == "P()"
    assert result["collective_count"] == 5, json.dumps(result, sort_keys=True)
    assert [item[:2] for item in result["collectives"]] == [
        ["all-reduce", 4],
        ["all-reduce", 4],
        ["all-gather", 8],
        ["all-gather", 8],
        ["all-gather", 8],
    ]
    assert result["maximum_group_size"] <= 8
    assert result["async_collectives"] == []
    assert not result["has_batch32_hidden"]


def test_ws32_index_share_attention_matches_forced_32_reference() -> None:
    program = r'''
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig
from glm_tpu.greenfield.kernels.reference.attention import (
    MlaNumericalContract,
    StageLocalKvLayout,
    canonicalize_selected_positions,
    gather_paged_selected_kv,
    sparse_mla_attention,
)
from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions
from glm_tpu.greenfield.kernels.reference.fp8 import (
    dequantize_fp8_bits_block_weight,
)
from glm_tpu.greenfield.kernels.reference.linear import linear, residual_add
from glm_tpu.greenfield.kernels.reference.rotary import apply_rotary, rotary_cos_sin
from glm_tpu.greenfield.kernels.ws32_layer import (
    Ws32AttentionResult,
    Ws32AttentionWeights,
    Ws32PreparedAttention,
    ws32_index_share_attention_mapped,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
    tpu_info.ChipVersion.TPU_V4, 1
)
tpu_info.get_tpu_info.cache_clear()

def bits(value):
    return np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)

def draw(seed, shape, scale=0.03125):
    return np.random.default_rng(seed).normal(0, scale, shape).astype(np.float32)

devices = np.asarray(jax.devices(), dtype=object).reshape(8, 4)
mesh = Mesh(devices, ("expert", "feature"))
contract = MlaNumericalContract(top_k=8)
layout = StageLocalKvLayout(local_parallel_size=8)
block = (128, 128)
local_heads = contract.num_heads // layout.local_parallel_size

residual = np.asarray(draw(20, (1, 6144), 0.125), dtype=ml_dtypes.bfloat16)
normalized = np.asarray(draw(21, (1, 6144), 0.125), dtype=ml_dtypes.bfloat16)
q_residual = np.asarray(draw(22, (1, 2048), 0.125), dtype=ml_dtypes.bfloat16)
current_kv = np.asarray(draw(23, (1, 576), 0.125), dtype=ml_dtypes.bfloat16)
cache = np.asarray(
    draw(24, (1, layout.logical_page_size, contract.packed_cache_width), 0.125),
    dtype=ml_dtypes.bfloat16,
)
q_b = bits(draw(25, (contract.num_heads * contract.qk_head_dim, 2048)))
q_b_scale = np.random.default_rng(26).uniform(
    0.25, 0.75, (128, 16)
).astype(np.float32)
kv_b = bits(draw(27, (
    contract.num_heads * (contract.qk_nope_head_dim + contract.v_head_dim),
    contract.kv_lora_rank,
)))
kv_b_scale = np.random.default_rng(28).uniform(
    0.25, 0.75, (224, 4)
).astype(np.float32)
o_weight = bits(draw(29, (6144, contract.num_heads * contract.v_head_dim)))
o_scale = np.random.default_rng(30).uniform(
    0.25, 0.75, (48, 128)
).astype(np.float32)
selected_positions = np.asarray([[7, 0, 6, 1, 5, 2, 4, 3]], dtype=np.int32)
selected_counts = np.asarray([8], dtype=np.int32)
position = np.asarray([7], dtype=np.int32)
block_tables = np.asarray([[0]], dtype=np.int32)
context_lengths = np.asarray([8], dtype=np.int32)

prepared_values = Ws32PreparedAttention(
    normalized, normalized, q_residual, current_kv
)
prepared_specs = Ws32PreparedAttention(
    P(None, "feature"), P(None, "feature"), P(), P()
)
weight_values = Ws32AttentionWeights(
    q_b, q_b_scale, kv_b, kv_b_scale, o_weight, o_scale
)
weight_specs = Ws32AttentionWeights(
    P("expert", None), P("expert", None),
    P("expert", None), P("expert", None),
    P("feature", "expert"), P("feature", "expert"),
)

def mapped(residual_local, prepared, cache_local, positions, counts,
           pos, tables, lengths, weights):
    return ws32_index_share_attention_mapped(
        residual_local,
        prepared,
        cache_local,
        positions,
        counts,
        pos,
        tables,
        lengths,
        weights,
        contract=contract,
        cache_layout=layout,
        block_shape=block,
        sparse_attention_config=SparseMlaConfig(segment_block=8),
        sparse_attention_interpret=True,
        linear_interpret=True,
    )

sharded = jax.shard_map(
    mapped,
    mesh=mesh,
    in_specs=(
        P(None, "feature"), prepared_specs, P(None, "expert", None),
        P(), P(), P(), P(), P(), weight_specs,
    ),
    out_specs=Ws32AttentionResult(
        P(None, "feature"), P(None, "expert", None), P()
    ),
    check_vma=False,
)

def put(value, spec):
    if isinstance(value, tuple):
        return type(value)(*(put(v, s) for v, s in zip(value, spec)))
    return jax.device_put(value, NamedSharding(mesh, spec))

arguments = (
    put(residual, P(None, "feature")),
    put(prepared_values, prepared_specs),
    put(cache, P(None, "expert", None)),
    put(selected_positions, P()),
    put(selected_counts, P()),
    put(position, P()),
    put(block_tables, P()),
    put(context_lengths, P()),
    put(weight_values, weight_specs),
)
compiled = jax.jit(sharded).lower(*arguments).compile()
actual = compiled(*arguments)

q_b_weight = dequantize_fp8_bits_block_weight(
    jnp.asarray(q_b), jnp.asarray(q_b_scale), block_shape=block
)
kv_b_weight = dequantize_fp8_bits_block_weight(
    jnp.asarray(kv_b), jnp.asarray(kv_b_scale), block_shape=block
).reshape(
    contract.num_heads,
    contract.qk_nope_head_dim + contract.v_head_dim,
    contract.kv_lora_rank,
)
o_weight_bf16 = dequantize_fp8_bits_block_weight(
    jnp.asarray(o_weight), jnp.asarray(o_scale), block_shape=block
)
q_states = linear(jnp.asarray(q_residual), q_b_weight).reshape(
    1, contract.num_heads, contract.qk_head_dim
)
q_nope = q_states[..., :contract.qk_nope_head_dim]
q_rope_unrotated = q_states[..., contract.qk_nope_head_dim:]
cos, sin = rotary_cos_sin(
    jnp.asarray(position),
    rotary_dim=contract.qk_rope_head_dim,
    theta=8_000_000.0,
    dtype=q_rope_unrotated.dtype,
)
q_rope = apply_rotary(
    q_rope_unrotated, cos[:, None, :], sin[:, None, :], interleaved=True
)
current_rope = apply_rotary(
    jnp.asarray(current_kv)[:, None, contract.kv_lora_rank:],
    cos[:, None, :],
    sin[:, None, :],
    interleaved=True,
)[:, 0, :]
current_cache_row = jnp.concatenate((
    jnp.asarray(current_kv)[..., :contract.kv_lora_rank],
    current_rope,
    jnp.zeros((1, contract.packed_cache_width - 576), jnp.bfloat16),
), axis=-1).astype(jnp.bfloat16)
expected_cache = jnp.asarray(cache).at[0, 7].set(current_cache_row[0])
local_uk = kv_b_weight[:, :contract.qk_nope_head_dim, :]
local_uv = jnp.transpose(
    kv_b_weight[:, contract.qk_nope_head_dim:, :], (0, 2, 1)
)
q_absorbed = jnp.einsum(
    "rhp,hpl->rhl",
    q_nope.astype(jnp.float32),
    local_uk.astype(jnp.float32),
    preferred_element_type=jnp.float32,
).astype(jnp.bfloat16)
canonical = canonicalize_selected_positions(
    SelectedPositions(jnp.asarray(selected_positions), jnp.asarray(selected_counts))
).selection
segment = gather_paged_selected_kv(
    expected_cache,
    jnp.asarray(block_tables),
    canonical,
    jnp.asarray(context_lengths),
)
attended = sparse_mla_attention(
    q_absorbed, q_rope, segment, contract=contract
).output
value_states = jnp.einsum(
    "rhl,hlv->rhv",
    attended.astype(jnp.float32),
    local_uv.astype(jnp.float32),
    preferred_element_type=jnp.float32,
).astype(jnp.bfloat16)
output_input = value_states.reshape(1, contract.num_heads * contract.v_head_dim)
hidden_chunks = []
for feature in range(4):
    expert_partials = []
    for expert in range(8):
        lhs = output_input[:, expert * 2048:(expert + 1) * 2048]
        rhs = o_weight_bf16[
            feature * 1536:(feature + 1) * 1536,
            expert * 2048:(expert + 1) * 2048,
        ]
        expert_partials.append(lax.dot_general(
            lhs,
            rhs,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ))
    hidden_chunks.append(jnp.sum(
        jnp.stack(expert_partials), axis=0, dtype=jnp.float32
    ).astype(jnp.bfloat16))
expected_output = residual_add(
    jnp.asarray(residual), jnp.concatenate(hidden_chunks, axis=-1)
)

module = parse_hlo_module(compiled.as_text())
print(json.dumps({
    "output_max_abs": float(jnp.max(jnp.abs(
        actual.output_local.astype(jnp.float32)
        - expected_output.astype(jnp.float32)
    ))),
    "cache_bitwise": bool(np.array_equal(
        np.asarray(actual.cache_local).view(np.uint16),
        np.asarray(expected_cache).view(np.uint16),
    )),
    "contract_valid": np.asarray(actual.contract_valid).tolist(),
    "output_sharding": str(actual.output_local.sharding.spec),
    "cache_sharding": str(actual.cache_local.sharding.spec),
    "collectives": [
        [item.raw_opcode, item.maximum_group_size, item.op_name]
        for item in module.collectives
    ],
    "async_collectives": [
        item.raw_opcode for item in module.collectives
        if item.raw_opcode.endswith(("-start", "-done"))
    ],
    "has_batch32_hidden": (
        "bf16[32,6144]" in compiled.as_text()
        or "f32[32,6144]" in compiled.as_text()
    ),
}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["output_max_abs"] <= 0.03125
    assert result["cache_bitwise"]
    assert result["contract_valid"] == [True]
    assert result["output_sharding"] == "P(None, 'feature')"
    assert result["cache_sharding"] == "P(None, 'expert')"
    assert [item[:2] for item in result["collectives"]] == [
        ["all-reduce", 8],
        ["all-reduce", 8],
    ]
    assert result["async_collectives"] == []
    assert not result["has_batch32_hidden"]


def test_ws32_complete_mlp_branches_match_forced_32_references() -> None:
    program = r'''
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.reference.linear import residual_add
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.kernels.ws32 import (
    ws32_dense_fp8_mapped,
    ws32_moe_fp8_from_routes_mapped,
    ws32_rms_norm_mapped,
    ws32_router_from_shards_mapped,
)
from glm_tpu.greenfield.kernels.ws32_layer import (
    Ws32DenseWeights,
    Ws32MlpResult,
    Ws32MoeWeights,
    ws32_mlp_mapped,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
    tpu_info.ChipVersion.TPU_V4, 1
)
tpu_info.get_tpu_info.cache_clear()

def bits(value):
    return np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)

def draw(seed, shape, scale=0.125):
    return np.random.default_rng(seed).normal(0, scale, shape).astype(np.float32)

def put(value, spec):
    if isinstance(value, tuple):
        return type(value)(*(put(v, s) for v, s in zip(value, spec)))
    return jax.device_put(value, NamedSharding(mesh, spec))

devices = np.asarray(jax.devices(), dtype=object).reshape(8, 4)
mesh = Mesh(devices, ("expert", "feature"))
block = (32, 32)
contract = GlmMoeNumericalContract(
    hidden_size=128,
    intermediate_size=128,
    num_experts=16,
    top_k=4,
    stage_size=8,
    fp8_block_shape=block,
)
residual = np.asarray(draw(1, (1, 128), 0.25), dtype=ml_dtypes.bfloat16)
norm_weight = np.asarray(
    np.random.default_rng(2).uniform(0.75, 1.25, (128,)),
    dtype=ml_dtypes.bfloat16,
)

dense_intermediate = 256
dense = Ws32DenseWeights(
    bits(draw(3, (dense_intermediate, 128))),
    np.random.default_rng(4).uniform(0.5, 1.25, (8, 4)).astype(np.float32),
    bits(draw(5, (dense_intermediate, 128))),
    np.random.default_rng(6).uniform(0.5, 1.25, (8, 4)).astype(np.float32),
    bits(draw(7, (128, dense_intermediate))),
    np.random.default_rng(8).uniform(0.5, 1.25, (4, 8)).astype(np.float32),
)
dense_specs = Ws32DenseWeights(
    P("expert", "feature"), P("expert", "feature"),
    P("expert", "feature"), P("expert", "feature"),
    P("feature", "expert"), P("feature", "expert"),
)

moe = Ws32MoeWeights(
    np.asarray(draw(9, (16, 128), 0.2), dtype=ml_dtypes.bfloat16),
    draw(10, (16,), 0.01),
    bits(draw(11, (16, 128, 128))),
    np.random.default_rng(12).uniform(0.5, 1.25, (16, 4, 4)).astype(np.float32),
    bits(draw(13, (16, 128, 128))),
    np.random.default_rng(14).uniform(0.5, 1.25, (16, 4, 4)).astype(np.float32),
    bits(draw(15, (16, 128, 128))),
    np.random.default_rng(16).uniform(0.5, 1.25, (16, 4, 4)).astype(np.float32),
    bits(draw(17, (128, 128))),
    np.random.default_rng(18).uniform(0.5, 1.25, (4, 4)).astype(np.float32),
    bits(draw(19, (128, 128))),
    np.random.default_rng(20).uniform(0.5, 1.25, (4, 4)).astype(np.float32),
    bits(draw(21, (128, 128))),
    np.random.default_rng(22).uniform(0.5, 1.25, (4, 4)).astype(np.float32),
)
moe_specs = Ws32MoeWeights(
    P("expert", "feature"), P("expert"),
    P("expert", None, "feature"), P("expert", None, "feature"),
    P("expert", None, "feature"), P("expert", None, "feature"),
    P("expert", "feature", None), P("expert", "feature", None),
    P(None, "feature"), P(None, "feature"),
    P(None, "feature"), P(None, "feature"),
    P("feature", None), P("feature", None),
)

common = (
    put(residual, P(None, "feature")),
    put(norm_weight, P("feature")),
)
dense_arguments = common + (put(dense, dense_specs),)
moe_arguments = common + (put(moe, moe_specs),)

dense_program = jax.shard_map(
    lambda value, norm, weights: ws32_mlp_mapped(
        value, norm, weights, None,
        mlp_kind="dense", contract=contract, block_shape=block,
        linear_interpret=True,
    ),
    mesh=mesh,
    in_specs=(P(None, "feature"), P("feature"), dense_specs),
    out_specs=Ws32MlpResult(P(None, "feature"), P(), P()),
    check_vma=False,
)
dense_reference = jax.shard_map(
    lambda value, norm, weights: residual_add(
        value,
        ws32_dense_fp8_mapped(
            ws32_rms_norm_mapped(value, norm, global_hidden_size=128),
            *weights,
            block_shape=block,
        ),
    ),
    mesh=mesh,
    in_specs=(P(None, "feature"), P("feature"), dense_specs),
    out_specs=P(None, "feature"),
    check_vma=False,
)
moe_program = jax.shard_map(
    lambda value, norm, weights: ws32_mlp_mapped(
        value, norm, None, weights,
        mlp_kind="sparse", contract=contract, block_shape=block,
        linear_interpret=True,
    ),
    mesh=mesh,
    in_specs=(P(None, "feature"), P("feature"), moe_specs),
    out_specs=Ws32MlpResult(P(None, "feature"), P(), P()),
    check_vma=False,
)
def readable_moe(value, norm, weights):
    normalized = ws32_rms_norm_mapped(value, norm, global_hidden_size=128)
    routes, route_weights = ws32_router_from_shards_mapped(
        normalized,
        weights.router_weight_local,
        weights.correction_bias_local,
        top_k=contract.top_k,
    )
    update = ws32_moe_fp8_from_routes_mapped(
        normalized,
        routes,
        route_weights,
        *weights[2:],
        contract=contract,
    )
    return Ws32MlpResult(
        residual_add(value, update), routes, route_weights
    )
moe_reference = jax.shard_map(
    readable_moe,
    mesh=mesh,
    in_specs=(P(None, "feature"), P("feature"), moe_specs),
    out_specs=Ws32MlpResult(P(None, "feature"), P(), P()),
    check_vma=False,
)

dense_compiled = jax.jit(dense_program).lower(*dense_arguments).compile()
dense_ref_compiled = jax.jit(dense_reference).lower(*dense_arguments).compile()
moe_compiled = jax.jit(moe_program).lower(*moe_arguments).compile()
moe_ref_compiled = jax.jit(moe_reference).lower(*moe_arguments).compile()
dense_actual = dense_compiled(*dense_arguments)
dense_expected = dense_ref_compiled(*dense_arguments)
moe_actual = moe_compiled(*moe_arguments)
moe_expected = moe_ref_compiled(*moe_arguments)

def hlo(text):
    module = parse_hlo_module(text)
    return {
        "count": len(module.collectives),
        "maximum_group_size": max(
            (item.maximum_group_size for item in module.collectives), default=0
        ),
        "async": [
            item.raw_opcode for item in module.collectives
            if item.raw_opcode.endswith(("-start", "-done"))
        ],
        "batch32_hidden": "bf16[32,128]" in text or "f32[32,128]" in text,
    }

print(json.dumps({
    "dense": {
        "max_abs": float(jnp.max(jnp.abs(
            dense_actual.output_local.astype(jnp.float32)
            - dense_expected.astype(jnp.float32)
        ))),
        "sentinel_indices": np.asarray(dense_actual.route_indices).tolist(),
        "sentinel_weights": np.asarray(dense_actual.route_weights).tolist(),
        "sharding": str(dense_actual.output_local.sharding.spec),
        "hlo": hlo(dense_compiled.as_text()),
    },
    "moe": {
        "max_abs": float(jnp.max(jnp.abs(
            moe_actual.output_local.astype(jnp.float32)
            - moe_expected.output_local.astype(jnp.float32)
        ))),
        "indices_equal": bool(np.array_equal(
            np.asarray(moe_actual.route_indices),
            np.asarray(moe_expected.route_indices),
        )),
        "weights_max_abs": float(jnp.max(jnp.abs(
            moe_actual.route_weights - moe_expected.route_weights
        ))),
        "sharding": str(moe_actual.output_local.sharding.spec),
        "hlo": hlo(moe_compiled.as_text()),
    },
}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["dense"]["max_abs"] <= 0.03125
    assert result["dense"]["sentinel_indices"] == [[-1, -1, -1, -1]]
    assert result["dense"]["sentinel_weights"] == [[0.0, 0.0, 0.0, 0.0]]
    assert result["dense"]["hlo"]["count"] == 3
    assert result["moe"]["max_abs"] <= 0.03125
    assert result["moe"]["indices_equal"]
    assert result["moe"]["weights_max_abs"] == 0.0
    # The compiler coalesces the router logits and bias gather in this joined
    # body, leaving one fewer physical collective than the isolated router.
    assert result["moe"]["hlo"]["count"] == 9
    for kind in ("dense", "moe"):
        assert result[kind]["sharding"] == "P(None, 'feature')"
        assert result[kind]["hlo"]["maximum_group_size"] <= 8
        assert result[kind]["hlo"]["async"] == []
        assert not result[kind]["hlo"]["batch32_hidden"]


def test_ws32_complete_dense_layer_compiles_as_one_forced_32_body() -> None:
    program = r'''
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig
from glm_tpu.greenfield.kernels.reference.attention import (
    MlaNumericalContract, StageLocalKvLayout,
)
from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped
from glm_tpu.greenfield.kernels.ws32_layer import (
    Ws32AttentionWeights,
    Ws32DenseWeights,
    Ws32QkvAWeights,
    Ws32TransformerLayerResult,
    ws32_attention_layer_mapped,
    ws32_mlp_mapped,
    ws32_transformer_layer_mapped,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
    tpu_info.ChipVersion.TPU_V4, 1
)
tpu_info.get_tpu_info.cache_clear()

def bits(value):
    return np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)

def draw(seed, shape, scale=0.125):
    return np.random.default_rng(seed).normal(0, scale, shape).astype(np.float32)

devices = np.asarray(jax.devices(), dtype=object).reshape(8, 4)
mesh = Mesh(devices, ("expert", "feature"))
block = (128, 128)
dsa = DsaNumericalContract(
    hidden_size=512, q_lora_rank=2048, num_heads=32,
    head_dim=128, rotary_dim=64, top_k=8,
)
mla = MlaNumericalContract(
    top_k=8,
)
moe = GlmMoeNumericalContract(
    hidden_size=512, intermediate_size=128, num_experts=16,
    top_k=4, stage_size=8, fp8_block_shape=block,
)
layout = StageLocalKvLayout(
    logical_page_size=64, local_parallel_size=8, packed_cache_width=640,
)
attention_config = SparseMlaConfig(segment_block=8)

residual = np.asarray(draw(1, (1, 512), 0.25), dtype=ml_dtypes.bfloat16)
carried = np.asarray(draw(25, (1, 512), 0.25), dtype=ml_dtypes.bfloat16)
cache = np.asarray(draw(2, (1, 64, 640)), dtype=ml_dtypes.bfloat16)
index_cache = np.asarray(draw(3, (1, 64, 128)), dtype=ml_dtypes.bfloat16)
selected = np.asarray([[7, 0, 6, 1, 5, 2, 4, 3]], dtype=np.int32)
counts = np.asarray([8], dtype=np.int32)
scores = draw(4, (1, 8))
position = np.asarray([7], dtype=np.int32)
tables = np.asarray([[0]], dtype=np.int32)
lengths = np.asarray([8], dtype=np.int32)
health = np.asarray([True], dtype=np.bool_)

qkv = Ws32QkvAWeights(
    np.asarray(np.random.default_rng(5).uniform(0.75, 1.25, (512,)), dtype=ml_dtypes.bfloat16),
    bits(draw(6, (2048, 512))),
    np.random.default_rng(7).uniform(0.5, 1.25, (16, 4)).astype(np.float32),
    np.asarray(np.random.default_rng(8).uniform(0.75, 1.25, (2048,)), dtype=ml_dtypes.bfloat16),
    bits(draw(9, (576, 512))),
    np.random.default_rng(10).uniform(0.5, 1.25, (5, 4)).astype(np.float32),
    np.asarray(np.random.default_rng(11).uniform(0.75, 1.25, (512,)), dtype=ml_dtypes.bfloat16),
)
qkv_specs = Ws32QkvAWeights(
    P("feature"), P(None, "feature"), P(None, "feature"), P(),
    P(None, "feature"), P(None, "feature"), P(),
)
attention = Ws32AttentionWeights(
    bits(draw(12, (64 * 256, 2048))),
    np.random.default_rng(13).uniform(0.5, 1.25, (128, 16)).astype(np.float32),
    bits(draw(14, (64 * 448, 512))),
    np.random.default_rng(15).uniform(0.5, 1.25, (224, 4)).astype(np.float32),
    bits(draw(16, (512, 64 * 256))),
    np.random.default_rng(17).uniform(0.5, 1.25, (4, 128)).astype(np.float32),
)
attention_specs = Ws32AttentionWeights(
    P("expert", None), P("expert", None),
    P("expert", None), P("expert", None),
    P("feature", "expert"), P("feature", "expert"),
)
norm = np.asarray(
    np.random.default_rng(18).uniform(0.75, 1.25, (512,)),
    dtype=ml_dtypes.bfloat16,
)
dense = Ws32DenseWeights(
    bits(draw(19, (1024, 512))),
    np.random.default_rng(20).uniform(0.5, 1.25, (8, 4)).astype(np.float32),
    bits(draw(21, (1024, 512))),
    np.random.default_rng(22).uniform(0.5, 1.25, (8, 4)).astype(np.float32),
    bits(draw(23, (512, 1024))),
    np.random.default_rng(24).uniform(0.5, 1.25, (4, 8)).astype(np.float32),
)
dense_specs = Ws32DenseWeights(
    P("expert", "feature"), P("expert", "feature"),
    P("expert", "feature"), P("expert", "feature"),
    P("feature", "expert"), P("feature", "expert"),
)

def mapped(update, residual, kv, index, positions, valid, selected_scores, pos,
           block_tables, context, qkv_weights, attention_weights, norm_weight,
           dense_weights, incoming):
    return ws32_transformer_layer_mapped(
        update, residual, kv, index, positions, valid, selected_scores, pos,
        block_tables, context, qkv_weights, attention_weights, None, norm_weight,
        dense_weights, None, incoming, indexer_kind="shared", mlp_kind="dense",
        dsa_contract=dsa, attention_contract=mla, moe_contract=moe,
        cache_layout=layout, block_shape=block,
        sparse_attention_config=attention_config,
        sparse_attention_interpret=True, linear_interpret=True,
    )

def decomposed(update, residual, kv, index, positions, valid, selected_scores,
               pos, block_tables, context, qkv_weights, attention_weights,
               norm_weight, dense_weights, incoming):
    normalized_input, combined = ws32_fused_add_rms_norm_mapped(
        update, residual, qkv_weights.input_norm_weight_local,
        global_hidden_size=512,
    )
    attention_result = ws32_attention_layer_mapped(
        combined, kv, index, positions, valid, selected_scores, pos, block_tables,
        context, qkv_weights, attention_weights, None,
        dsa_contract=dsa, attention_contract=mla, cache_layout=layout,
        block_shape=block, sparse_attention_config=attention_config,
        sparse_attention_interpret=True, linear_interpret=True,
        precomputed_normalized_local=normalized_input, add_residual=False,
    )
    normalized_mlp, post_attention_residual = ws32_fused_add_rms_norm_mapped(
        attention_result.output_local, combined, norm_weight,
        global_hidden_size=512,
    )
    mlp_result = ws32_mlp_mapped(
        post_attention_residual, norm_weight, dense_weights, None,
        mlp_kind="dense", contract=moe, block_shape=block,
        linear_interpret=True, precomputed_normalized_local=normalized_mlp,
        add_residual=False,
    )
    return Ws32TransformerLayerResult(
        mlp_result.output_local, post_attention_residual,
        normalized_input,
        attention_result.cache_local,
        attention_result.index_cache_local, attention_result.selected_positions,
        attention_result.selected_valid_counts, attention_result.selected_scores,
        mlp_result.route_indices, mlp_result.route_weights,
        incoming & attention_result.contract_valid,
    )

input_specs = (
    P(None, "feature"), P(None, "feature"), P(None, "expert", None),
    P(None, "expert", None), P(), P(), P(), P(), P(), P(), qkv_specs,
    attention_specs, P("feature"), dense_specs, P(),
)
output_specs = Ws32TransformerLayerResult(
    P(None, "feature"), P(None, "feature"), P(None, "feature"),
    P(None, "expert", None),
    P(None, "expert", None), P(), P(), P(), P(), P(), P(),
)
joined = jax.shard_map(
    mapped, mesh=mesh, in_specs=input_specs, out_specs=output_specs,
    check_vma=False,
)
reference = jax.shard_map(
    decomposed, mesh=mesh, in_specs=input_specs, out_specs=output_specs,
    check_vma=False,
)

def put(value, spec):
    if isinstance(value, tuple):
        return type(value)(*(put(v, s) for v, s in zip(value, spec)))
    return jax.device_put(value, NamedSharding(mesh, spec))

values = (
    residual, carried, cache, index_cache, selected, counts, scores, position,
    tables, lengths, qkv, attention, norm, dense, health,
)
arguments = tuple(put(value, spec) for value, spec in zip(values, input_specs))
compiled = jax.jit(joined).lower(*arguments).compile()
reference_compiled = jax.jit(reference).lower(*arguments).compile()
actual = compiled(*arguments)
expected = reference_compiled(*arguments)
module = parse_hlo_module(compiled.as_text())
print(json.dumps({
    "output_bitwise": bool(np.array_equal(
        np.asarray(actual.output_local).view(np.uint16),
        np.asarray(expected.output_local).view(np.uint16),
    )),
    "carried_bitwise": bool(np.array_equal(
        np.asarray(actual.carried_residual_local).view(np.uint16),
        np.asarray(expected.carried_residual_local).view(np.uint16),
    )),
    "cache_bitwise": bool(np.array_equal(
        np.asarray(actual.cache_local).view(np.uint16),
        np.asarray(expected.cache_local).view(np.uint16),
    )),
    "contract_valid": np.asarray(actual.contract_valid).tolist(),
    "output_sharding": str(actual.output_local.sharding.spec),
    "carried_sharding": str(actual.carried_residual_local.sharding.spec),
    "cache_sharding": str(actual.cache_local.sharding.spec),
    "route_indices": np.asarray(actual.route_indices).tolist(),
    "maximum_group_size": max(
        item.maximum_group_size for item in module.collectives
    ),
    "fused_rmsnorm_collectives": sum(
        item.op_name is not None
        and "greenfield_ws32_fused_rmsnorm" in item.op_name.split("/")
        for item in module.collectives
    ),
    "rounded_first_rmsnorm_collectives": sum(
        item.op_name is not None
        and "greenfield_ws32_rmsnorm" in item.op_name.split("/")
        for item in module.collectives
    ),
    "async": [
        item.raw_opcode for item in module.collectives
        if item.raw_opcode.endswith(("-start", "-done"))
    ],
    "batch32_hidden": (
        "bf16[32,512]" in compiled.as_text()
        or "f32[32,512]" in compiled.as_text()
    ),
}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["output_bitwise"]
    assert result["carried_bitwise"]
    assert result["cache_bitwise"]
    assert result["contract_valid"] == [True]
    assert result["output_sharding"] == "P(None, 'feature')"
    assert result["carried_sharding"] == "P(None, 'feature')"
    assert result["cache_sharding"] == "P(None, 'expert')"
    assert result["route_indices"] == [[-1, -1, -1, -1]]
    assert result["maximum_group_size"] <= 8
    assert result["fused_rmsnorm_collectives"] == 2
    assert result["rounded_first_rmsnorm_collectives"] == 0
    assert result["async"] == []
    assert not result["batch32_hidden"]


PRE_A_PRIME_COMMIT = "cb36cb74"


def _strip(lines: list[str]) -> list[str]:
    """Statements only: comments and blank lines are not traced."""
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        kept.append(stripped)
    return kept


def test_the_flag_off_rotary_path_keeps_the_pre_a_prime_statement_order() -> None:
    """Spec §23.8 neutrality: the default-off path must trace as it did before.

    Statement *order*, not statement count: JAX records equations in Python
    execution order, so moving an independent slice across the rotary changes
    the StableHLO text and would raise a false drift alarm against the sealed
    pins at the next acquisition.  The companion test below measures that this
    is true rather than assuming it.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    current = (root / "glm_tpu/greenfield/kernels/ws32_layer.py").read_text(encoding="utf-8")
    previous = subprocess.run(
        ["git", "-C", str(root), "show", f"{PRE_A_PRIME_COMMIT}^:glm_tpu/greenfield/kernels/ws32_layer.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    start = previous.index("    q_rope_unrotated = q_states[")
    end = previous.index("    padding = contract.packed_cache_width", start)
    expected = _strip(previous[start:end].splitlines())
    expected = expected[1:]  # the shared q_rope_unrotated assignment

    marker = "    if main_rope_table_row is None:\n"
    branch_start = current.index(marker) + len(marker)
    branch_end = current.index("\n    else:\n", branch_start)
    observed = _strip(current[branch_start:branch_end].splitlines())

    assert observed == expected, (
        "the default-off rotary path no longer traces in the pre-A' order; "
        "re-acquiring with the flag off would produce different StableHLO"
    )


def test_moving_an_independent_slice_across_the_rotary_changes_the_stablehlo() -> None:
    """The measurement behind the neutrality requirement above."""
    program = r'''
import hashlib
import json

import jax
import jax.numpy as jnp

from glm_tpu.greenfield.kernels.reference.rotary import rotary_cos_sin, apply_rotary

ROTARY_DIM = 16


def hoisted(kv, q, position):
    latent = kv[..., :8]
    rope_input = kv[..., 8 : 8 + ROTARY_DIM][:, None, :]
    cos, sin = rotary_cos_sin(position, rotary_dim=ROTARY_DIM, theta=8e6, dtype=q.dtype)
    rotated_q = apply_rotary(q, cos[:, None, :], sin[:, None, :], interleaved=True)
    rotated = apply_rotary(rope_input, cos[:, None, :], sin[:, None, :], interleaved=True)
    return latent, rotated_q, rotated[:, 0, :]


def in_order(kv, q, position):
    cos, sin = rotary_cos_sin(position, rotary_dim=ROTARY_DIM, theta=8e6, dtype=q.dtype)
    rotated_q = apply_rotary(q, cos[:, None, :], sin[:, None, :], interleaved=True)
    latent = kv[..., :8]
    rope_input = kv[..., 8 : 8 + ROTARY_DIM][:, None, :]
    rotated = apply_rotary(rope_input, cos[:, None, :], sin[:, None, :], interleaved=True)
    return latent, rotated_q, rotated[:, 0, :]


kv = jax.ShapeDtypeStruct((1, 8 + ROTARY_DIM), jnp.bfloat16)
q = jax.ShapeDtypeStruct((1, 4, ROTARY_DIM), jnp.bfloat16)
position = jax.ShapeDtypeStruct((1,), jnp.int32)
digests = {
    name: hashlib.sha256(jax.jit(fn).lower(kv, q, position).as_text().encode()).hexdigest()
    for name, fn in (("hoisted", hoisted), ("in_order", in_order))
}
print(json.dumps(digests))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    digests = json.loads(completed.stdout.strip().splitlines()[-1])
    assert digests["hoisted"] != digests["in_order"], (
        "traced order no longer affects the StableHLO text; the neutrality "
        "argument for the default-off rotary path needs to be re-derived"
    )
