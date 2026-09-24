"""Bitwise shortlist proof including ties, skew and forced fallback."""
import json
import os
import subprocess
import sys


def test_two_stage_matches_frozen_cpu8():
    code = r'''
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.optimized.reference.dsa import local_topk_candidates, merge_topk_candidates_with_scores, ScoredSelectedPositions
from glm_tpu.optimized.dsa_candidates import two_stage_topk_mapped
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
    from glm_tpu.optimized.reference.attention import StageLocalKvLayout
    from glm_tpu.optimized.reference.dsa import dsa_scores
    from glm_tpu.optimized.dsa_candidates import score_cache_pages

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


# Leaf digests (sha256 of dtype | shape | bytes; positions, scores, counts, per-owner health) of the
# frozen tiled prefill DSA selector on the inputs below, recorded from its final run at S2f (it is
# archived at archive/research-20260922); the production one-pass selector produced the same leaves
# bitwise in that run. Recorded with jax/jaxlib 0.10.1 on CPU (the G3 environment).
FROZEN_TILED_SELECTOR = {
    "0,127,2048": (
        "335e89b56642ea31f7442ebceefede8b3eff87667bcb68e460a0850a79c32e07",
        "e30ebfb3327e69ff2cc0cda1417f20113af4ebc48307befb9e311584b7b457d2",
        "8875abbe17e828ccca774f63180a2af2216d2bd09f84a78d3418bc431f816253",
        "6a2cf1002bbb66778d21b8a7b414afd4a09a25dfb2e48269413e8ac2ce195a6b",
    ),
    "1,511,1984": (
        "f8debe139d164bc50eeefbb39284944f42f33d216d894a16e584a498551a2c99",
        "1cc9001ad1d0a726edf93136322468d56bfa021701dd2425e1bf8605010bf391",
        "9c44e32049f0fe6f3a7cb37ff5323a0319232b3549a0a42055119ed20df3f890",
        "6a2cf1002bbb66778d21b8a7b414afd4a09a25dfb2e48269413e8ac2ce195a6b",
    ),
    "0,0,0": (
        "987aaa266b7b9d7ed646e3482f653c99441a6b4ed0647d34b61cf9f616237043",
        "46c0e0c6c0f5616da54ece94b4569c47292a9229f24d183e523df40b2f7e4c62",
        "7f2e106fcfd5fabb886a2f7ceb6b016ae230ae25aec3dd2fa31de827ffe55c39",
        "6a2cf1002bbb66778d21b8a7b414afd4a09a25dfb2e48269413e8ac2ce195a6b",
    ),
}


def test_one_pass_prefill_matches_the_recorded_tiled_selector_cpu32():
    code = r'''
import hashlib, json
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.optimized.reference.dsa import ScoredSelectedPositions
from glm_tpu.optimized.dsa_candidates import prefill_dsa_one_pass_mapped
mesh=Mesh(np.asarray(jax.devices(), object).reshape(8,4), ('expert','feature'))
def body(q,k,w,p,lengths):
    a=prefill_dsa_one_pass_mapped(q,k[0],w,p[0],lengths,candidates_per_owner=16,global_context_size=2048,top_k=64)
    # Health is owner-local, expose one element per owner.
    return a[0],a[1][None]
fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P(),P('expert'),P(),P('expert'),P()),
                         out_specs=(ScoredSelectedPositions(P(),P(),P()),P('expert')),check_vma=False))
rng=np.random.default_rng(36)
q=jnp.asarray(rng.normal(size=(3,32,128)),jnp.float32)
k=jnp.asarray(rng.normal(size=(8,256,128)),jnp.bfloat16)
w=jnp.asarray(rng.normal(size=(3,32)),jnp.float32)
p=np.stack([np.arange(256,dtype=np.int32)*8+i for i in range(8)])
def digest(x):
    x=np.asarray(x); return hashlib.sha256(f"{x.dtype}|{x.shape}|".encode()+x.tobytes()).hexdigest()
out={}
for lengths in ([0,127,2048],[1,511,1984],[0,0,0]):
    out[",".join(map(str,lengths))]=[digest(x) for x in jax.tree.leaves(fn(q,k,w,jnp.asarray(p),jnp.array(lengths,jnp.int32)))]
print(json.dumps(out))
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    assert actual == {key: list(value) for key, value in FROZEN_TILED_SELECTOR.items()}
