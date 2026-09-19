"""D5 merge semantics and numerical boundary, CPU only."""
import os
import subprocess
import sys

import pytest

from glm_tpu.perf.ws32_decoder_challenger import Ws32PerfOptions


def test_lse_options_fail_closed():
    with pytest.raises(ValueError):
        Ws32PerfOptions(lse_attention=True)
    with pytest.raises(ValueError):
        Ws32PerfOptions(lse_attention=1, bf16_resident=True)


@pytest.mark.parametrize("rows", [1, 3])
def test_lse_attention_cpu_mesh(rows):
    code = r'''
import json
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.perf.lse_attention import lse_attention_mapped, merge_attention_scatter
from glm_tpu.greenfield.kernels.reference.attention import (
    MlaNumericalContract, StageLocalKvLayout, SparseAttentionResult,
    combine_stage_local_attention, gather_stage_local_selected_kv_aligned)
from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions
from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig, pregathered_sparse_mla_pallas
mesh = Mesh(np.asarray(jax.devices(), object), ('expert',))
rng = np.random.default_rng(51)
# First prove reduce-scatter matches the readable LSE merge, including empty
# owners, an entirely empty head, extreme LSE, and replicated invalid health.
out = jnp.asarray(rng.normal(size=(8, 1, 8, 128)), jnp.bfloat16)
lse = rng.normal(size=(8, 1, 8)).astype(np.float32) * 100
lse[0] = -np.inf
lse[:, :, 0] = -np.inf
out = out.at[0].set(0).at[:, :, 0].set(0)
valid = jnp.ones((8, 1), jnp.bool_).at[3, 0].set(False)
fn = jax.jit(jax.shard_map(lambda o, l, v: merge_attention_scatter(SparseAttentionResult(o[0], l[0], v[0])),
    mesh=mesh, in_specs=(P('expert'), P('expert'), P('expert')),
    out_specs=SparseAttentionResult(P(None, 'expert'), P(None, 'expert'), P()), check_vma=False))
a = fn(out, jnp.asarray(lse), valid)
e = combine_stage_local_attention(out, jnp.asarray(lse), valid)
np.testing.assert_allclose(a.output.astype(jnp.float32), e.output.astype(jnp.float32), atol=.008, rtol=.008)
np.testing.assert_allclose(a.logsumexp, e.logsumexp, atol=1e-5)
np.testing.assert_array_equal(a.contract_valid, e.contract_valid)
# Same selected set, cache and head queries in both attention arrangements.
c = MlaNumericalContract(num_heads=8, kv_lora_rank=128, qk_nope_head_dim=64,
    qk_rope_head_dim=64, qk_head_dim=128, v_head_dim=128, packed_cache_width=256, top_k=128)
l = StageLocalKvLayout(logical_page_size=512, local_parallel_size=8, packed_cache_width=256)
q = jnp.asarray(rng.normal(size=(ROWS, 8, 128)) * .3, jnp.bfloat16)
r = jnp.asarray(rng.normal(size=(ROWS, 8, 64)) * .3, jnp.bfloat16)
cache = jnp.asarray(rng.normal(size=(8, 2, 64, 256)), jnp.bfloat16)
tables = jnp.tile(jnp.array([[1, 0]], jnp.int32), (ROWS, 1))
length = jnp.full((ROWS,), 1024, jnp.int32)
config = SparseMlaConfig(segment_block=16)
def body(q, r, cache, positions, count):
    selected = SelectedPositions(positions, count)
    result = lse_attention_mapped(q, r, cache[0], tables, selected, length, contract=c, layout=l, config=config, interpret=True, validate_finite=True)
    aligned = gather_stage_local_selected_kv_aligned(cache[0], tables, selected, length, layout=l, owner_index=jax.lax.axis_index('expert'))
    full = jax.lax.psum(aligned.values, 'expert')
    frozen = pregathered_sparse_mla_pallas(q, r, full, count, contract=replace(c, num_heads=1), config=config, interpret=True, prefill=(ROWS > 1))
    compact = lse_attention_mapped(q, r, cache[0], tables, selected, length, contract=c, layout=l,
        config=config, interpret=True, validate_finite=True, owned_key_capacity=32)
    return result, frozen, compact
fn = jax.jit(jax.shard_map(body, mesh=mesh,
    in_specs=(P(None, 'expert'), P(None, 'expert'), P('expert'), P(), P()),
    out_specs=(SparseAttentionResult(P(None, 'expert'), P(None, 'expert'), P()), P(None, 'expert'), SparseAttentionResult(P(None, 'expert'), P(None, 'expert'), P())), check_vma=False))
errors = []
for positions in [[], [0], list(range(64)), list(range(0, 1024, 8)), rng.choice(1024, 128, replace=False).tolist()]:
    count = len(positions)
    p = jnp.tile(jnp.array([positions + [-1] * (128-count)], jnp.int32), (ROWS, 1))
    counts = jnp.full((ROWS,), count, jnp.int32)
    if ROWS > 1:
        p = p.at[1].set(-1)
        counts = counts.at[1].set(0)
    result, frozen, compact = fn(q, r, cache, p, counts)
    for a,b in zip(result,compact):
        np.testing.assert_array_equal(np.asarray(a).view(np.uint8),np.asarray(b).view(np.uint8))
    assert bool(result.contract_valid.all())
    error = float(jnp.max(jnp.abs(result.output.astype(jnp.float32)-frozen.astype(jnp.float32))))
    errors.append(error)
    assert error <= .015625, (count, error)
    assert bool(jnp.isfinite(result.output).all())
    if not count:
        assert bool((result.output == 0).all()) and bool(jnp.isneginf(result.logsumexp).all())
# Duplicate live positions must propagate failed health across owners.
result, _, compact = fn(q, r, cache, jnp.tile(jnp.array([[0, 0]+[-1]*126], jnp.int32), (ROWS, 1)), jnp.full((ROWS,), 2, jnp.int32))
assert not bool(result.contract_valid[0]) and not bool(compact.contract_valid[0])
# A selected NaN must fail owner health even if softmax masks hide the output.
p = jnp.tile(jnp.array([[0]+[-1]*127], jnp.int32), (ROWS, 1))
counts = jnp.ones((ROWS,), jnp.int32)
poisoned = cache.at[0,1,0,0].set(jnp.nan)
result, _, compact = fn(q, r, poisoned, p, counts)
assert not bool(result.contract_valid.any()) and not bool(compact.contract_valid.any())
# Poison outside the selected set is not a live operand.
poisoned = cache.at[0,1,3,0].set(jnp.nan)
result, _, compact = fn(q, r, poisoned, p, counts)
assert bool(result.contract_valid.all()) and bool(compact.contract_valid.all())
print(json.dumps(dict(max_abs=errors)))
'''
    code = code.replace("ROWS", str(rows))
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=8")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout)
