"""Bitwise shortlist proof including ties, skew and forced fallback."""
import os
import subprocess
import sys


def test_two_stage_matches_frozen_cpu8():
    code = r'''
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.greenfield.kernels.reference.dsa import local_topk_candidates, merge_topk_candidates_with_scores, ScoredSelectedPositions
from glm_tpu.perf.dsa_candidates import two_stage_topk_mapped
mesh=Mesh(np.asarray(jax.devices(), object), ('expert',))
def body(scores, positions, lengths):
    scores, positions = scores[0], positions[0]
    a, fallback = two_stage_topk_mapped(scores, positions, lengths, top_k=16, global_context_size=256, candidates_per_owner=4)
    v, p = local_topk_candidates(scores, positions, lengths, top_k=16)
    b = merge_topk_candidates_with_scores(jax.lax.all_gather(v,'expert'), jax.lax.all_gather(p,'expert'), lengths, top_k=16, global_context_size=256)
    return a,b,fallback
fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P('expert'),P('expert'),P()),out_specs=(ScoredSelectedPositions(P(),P(),P()),ScoredSelectedPositions(P(),P(),P()),P()),check_vma=False))
rng=np.random.default_rng(334)
positions=np.arange(256,dtype=np.int32).reshape(8,32)
paths=set()
for trial in range(60):
    perm=np.stack([rng.permutation(32) for _ in range(8)])
    p=np.take_along_axis(positions,perm,axis=1)
    # Integers exercise exact score ties across and inside owners.
    scores=rng.integers(-8,9,size=(8,3,32)).astype(np.float32)
    if trial%3==0: scores[0]+=100
    if trial%3==1: scores.fill(0)
    lengths=np.array([0,7 if trial%2 else 64,256],np.int32)
    if trial%3==2: lengths=np.array([0,256,256],np.int32)
    if trial == 59:
        # Stripes keep every other owner's omitted -0 below the cutoff.
        # Only the omitted +0 on owner 7 forces fallback; float == misses it.
        p=np.arange(256,dtype=np.int32).reshape(32,8).T
        scores.fill(-0.0)
        scores[7,:,:20] = 0.0
        lengths[:] = 256
    a,b,f=fn(jnp.asarray(scores),jnp.asarray(p),jnp.asarray(lengths))
    for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
    np.testing.assert_array_equal(np.asarray(a.scores).view(np.uint32), np.asarray(b.scores).view(np.uint32))
    paths.add(bool(f))
    if trial == 59: assert bool(f), "signed-zero cut must fall back"
assert paths=={False,True}, paths
print('60 randomized tied/skewed trials, both cut-check branches, bitwise equal')
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=8")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_physical_page_scores_match_logical_key_gather_bitwise():
    import jax.numpy as jnp
    import numpy as np
    from glm_tpu.greenfield.kernels.reference.attention import StageLocalKvLayout
    from glm_tpu.greenfield.kernels.reference.dsa import dsa_scores
    from glm_tpu.perf.dsa_candidates import score_cache_pages

    rng = np.random.default_rng(91)
    layout = StageLocalKvLayout(logical_page_size=128, local_parallel_size=8, packed_cache_width=128)
    query = jnp.asarray(rng.normal(size=(3, 4, 128)), jnp.float32)
    cache = jnp.asarray(rng.normal(size=(4, 16, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(3, 4)), jnp.float32)
    tables = jnp.array([[3, 1, -1, 0]], jnp.int32)
    actual, positions = score_cache_pages(query, cache, weights, tables, layout=layout, owner=jnp.int32(5))
    logical = jnp.take(cache, jnp.clip(tables[0], 0, 3), axis=0).reshape(-1, 128)
    expected = dsa_scores(query, logical, weights, precision="highest")
    np.testing.assert_array_equal(actual, expected)
    expected_positions = np.arange(4)[:, None]*128 + 5*16 + np.arange(16)[None]
    expected_positions[2] = -1
    np.testing.assert_array_equal(positions, expected_positions.reshape(-1))


def test_one_pass_prefill_matches_tiled_frozen_cpu32():
    code = r'''
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.greenfield.kernels.reference.dsa import ScoredSelectedPositions
from glm_tpu.greenfield.kernels.prefill_dsa import ws32_prefill_dsa_from_query_mapped
from glm_tpu.perf.dsa_candidates import prefill_dsa_one_pass_mapped
mesh=Mesh(np.asarray(jax.devices(), object).reshape(8,4), ('expert','feature'))
def body(q,k,w,p,lengths):
    kw=dict(global_context_size=2048,top_k=64)
    a=prefill_dsa_one_pass_mapped(q,k[0],w,p[0],lengths,candidates_per_owner=16,**kw)
    b=ws32_prefill_dsa_from_query_mapped(q,k[0],w,p[0],lengths,key_tile=128,paired_position_sort=True,sorted_local_merge=True,**kw)
    return a,b
result_specs=(ScoredSelectedPositions(P(),P(),P()),P('expert'))
# Health is owner-local, expose one element per owner.
def wrapped(*args):
    a,b=body(*args)
    return (a[0],a[1][None]),(b[0],b[1][None])
fn=jax.jit(jax.shard_map(wrapped,mesh=mesh,in_specs=(P(),P('expert'),P(),P('expert'),P()),out_specs=(result_specs,result_specs),check_vma=False))
rng=np.random.default_rng(36)
q=jnp.asarray(rng.normal(size=(3,32,128)),jnp.float32)
k=jnp.asarray(rng.normal(size=(8,256,128)),jnp.bfloat16)
w=jnp.asarray(rng.normal(size=(3,32)),jnp.float32)
p=np.stack([np.arange(256,dtype=np.int32)*8+i for i in range(8)])
for lengths in ([0,127,2048],[1,511,1984],[0,0,0]):
    a,b=fn(q,k,w,jnp.asarray(p),jnp.array(lengths,jnp.int32))
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)): np.testing.assert_array_equal(x,y)
print('P2 prefill positions, scores, health bitwise equal')
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
