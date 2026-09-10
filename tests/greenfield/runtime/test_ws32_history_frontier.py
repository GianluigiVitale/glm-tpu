"""CPU integration only: no original TPU reproduction or hardware admission."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("canonical", [True, False])
def test_history_frontier_cpu32_matches_production_caches_and_tail(canonical):
    source = "CANONICAL=" + repr(canonical) + "\n" + r'''
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield.ws32_history_frontier import build_program, HistoryCaches, LayerBoundary, ProducerRows
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b
from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend() == 'cpu'
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
config,weights,wk=fixture(mesh,panel_geometry=True)
def put(v,s=P()): return jax.device_put(v,NamedSharding(mesh,s))
wk=tuple(put(v) for v in wk)
rope=put(jnp.asarray(build_ws32_main_rope_table(config),jnp.bfloat16))
table=put(jnp.arange(config.page_count,dtype=jnp.int32)[None])
def cache(width):
    return put(jnp.zeros((8,config.page_count,64,width),jnp.bfloat16),P('expert',None,None,None))
caches=HistoryCaches(tuple(cache(640) for _ in range(7)),
                     tuple(cache(128) for _ in range(4)),tuple(cache(128) for _ in range(4)))
first_count=128 if CANONICAL else 32
state=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=first_count+19)
opts=dict(key_tile=512,sparse_attention_interpret=True,linear_interpret=True,
          mlp_window=True,rolled_prefix=True,expert_panels=True,
          paired_position_sort=True,sorted_local_merge=True,canonical_dense=CANONICAL)
def equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and a.dtype==b.dtype
    assert a.tobytes()==b.tobytes(), (a.shape,np.count_nonzero(a!=b))
def owner(v): return np.asarray(v).reshape(config.page_count,8,64,-1).transpose(1,0,2,3)
def original_with_observations():
    original=b.ws32_prefill_layer_window_mapped
    def body(tokens,count,state,weights,wk,rope):
        observed=[]
        def retain(*args,**kwargs):
            result=original(*args,**kwargs)
            observed.append(result)
            return result
        with patch.object(b,'ws32_prefill_layer_window_mapped',retain):
            reference=b.ws32_batched_prefill_mapped(tokens,count,state,weights,wk,rope,config=config,**opts)
        assert len(observed)==8
        boundaries=tuple(LayerBoundary(v.output_local,v.carried_residual_local,
            v.normalized_input_local,v.route_indices,v.route_weights,v.contract_valid[None,None])
            for v in observed[:7])
        producers=tuple(ProducerRows(observed[i].selected_positions,observed[i].selected_valid_counts,
            observed[i].selected_scores) for i in (0,1,2,6))
        return reference,boundaries,producers
    boundary=LayerBoundary(P(None,'feature'),P(None,'feature'),P(None,'feature'),P(),P(),P('expert','feature',None))
    from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_weight_specs
    return jax.jit(jax.shard_map(body,mesh=mesh,
        in_specs=(P(),P(),b.ws32_batched_prefill_state_specs(),ws32_decoder_weight_specs(config),(P(),)*4,P()),
        out_specs=(b.Ws32BatchedPrefillResult(b.ws32_batched_prefill_state_specs(),P()),
                   (boundary,)*7,(ProducerRows(P(),P(),P()),)*4),check_vma=False))
full=original_with_observations()
for rows,count,offset in ((128,first_count,0),(114,19,first_count)):
    tokens=put(jnp.concatenate((jnp.arange(count,dtype=jnp.int32)+30,jnp.full(rows-count,-2147483648,jnp.int32))))
    fn=build_program(mesh,config,block_rows=rows,canonical_dense=CANONICAL,interpret=True)
    args=(tokens,put(jnp.int32(count)),put(jnp.int32(offset)),table,caches,
          weights.embedding_local,weights.layers[:7],wk,rope,put(jnp.bool_(True)))
    compiled=fn.lower(*args).compile()
    actual=compiled(*args);jax.block_until_ready(actual)
    assert bool(actual.healthy)
    assert len(actual.boundaries)==7 and len(actual.producers)==4
    assert all(np.asarray(v.health)[...,:count].all() for v in actual.boundaries)
    reference,boundaries,producers=full(tokens,put(jnp.int32(count)),state,weights,wk,rope)
    jax.block_until_ready((reference,boundaries,producers))
    for x,y in zip(jax.tree.leaves((actual.boundaries,actual.producers)),
                   jax.tree.leaves((boundaries,producers)),strict=True): equal(x,y)
    assert bool(reference.state.decoder.contract_valid[0])
    for layer in range(7): equal(actual.caches.kv[layer],owner(reference.state.decoder.kv_cache_local[layer]))
    # Production promotes only repaired indices on the final block.
    for slot in range(4):
        equal(actual.caches.repaired[slot],owner(reference.state.repaired_index_local[slot]))
        family=actual.caches.repaired if bool(reference.state.finished) else actual.caches.unrepaired
        equal(family[slot],owner(reference.state.decoder.index_cache_local[slot]))
    # Out-of-capacity and incoming unhealthy proposals must not advance caches.
    for invalid in (args[:2]+(put(jnp.int32(config.context_capacity-1)),)+args[3:],
                    args[:-1]+(put(jnp.bool_(False)),)):
        refused=compiled(*invalid);jax.block_until_ready(refused)
        assert not bool(refused.healthy)
        for x,y in zip(jax.tree.leaves(refused.caches),jax.tree.leaves(caches),strict=True): equal(x,y)
    if rows==114:
        clean=compiled(put(jnp.where(jnp.arange(rows)<count,tokens,0)),*args[1:])
        for x,y in zip(jax.tree.leaves(actual),jax.tree.leaves(clean),strict=True): equal(x,y)
    caches,state=actual.caches,reference.state
    print('FRONTIER_MATCH',rows,count,flush=True)
assert bool(state.finished)
for options in (dict(block_rows=32,canonical_dense=True),dict(block_rows=128,canonical_dense=1)):
    try: build_program(mesh,config,interpret=True,**options)
    except ValueError: pass
    else: raise AssertionError('invalid static option accepted')
print('HISTORY_FRONTIER_CPU_PASS',flush=True)
'''
    result = subprocess.run([sys.executable, "-c", source], text=True,
        capture_output=True, timeout=600,
        env=dict(os.environ, JAX_PLATFORMS="cpu",
                 XLA_FLAGS="--xla_force_host_platform_device_count=32"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HISTORY_FRONTIER_CPU_PASS" in result.stdout
