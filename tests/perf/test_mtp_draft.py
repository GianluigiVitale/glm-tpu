"""Native MTP composition/state tests; synthetic CPU proof, not trained quality."""
import os
import subprocess
import sys


def test_native_mtp_projection_transformer_and_indexshare_cpu32():
    code=r'''
import json
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from glm_tpu.greenfield.runtime import ws32_decoder as d
from glm_tpu.perf.bf16_resident import bf16_resident_weights
from glm_tpu.perf.mtp_draft import MtpWeights,mtp_config,build_mtp_draft
from glm_tpu.perf.mtp_projection import MtpProjectionWeights,build_mtp_projection
from glm_tpu.perf.mtp_checkpoint import bind_mtp_arrays,PROJECTION_NAMES
from glm_tpu.perf.speculative_verify import build_verifier,build_prefix_committer
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(x,spec=P()):return jax.device_put(x,NamedSharding(mesh,spec))
target,raw,_=fixture(mesh,panel_geometry=True)
config=mtp_config(target);raw=raw._replace(layers=(raw.layers[6],))
base=bf16_resident_weights(mesh,config,raw)
H=config.geometry.hidden_size;rng=np.random.default_rng(20)
projection=MtpProjectionWeights(put(jnp.ones(H,jnp.bfloat16),P('feature')),
    put(jnp.ones(H,jnp.bfloat16),P('feature')),
    put(jnp.asarray(rng.normal(0,.02,(H,2*H)),jnp.bfloat16),P('feature',None)))
weights=MtpWeights(base,projection)
# Bind the separate native tensor map while retaining the exact target I/O
# arrays. This also checks raw native norm/body names against the strict binder.
arrays=dict(zip(jax.tree.leaves(d.ws32_decoder_weight_names(config)),jax.tree.leaves(raw)))
del arrays['model.embed_tokens.weight'],arrays['lm_head.weight']
arrays.update(dict(zip(PROJECTION_NAMES,projection)))
bound=bind_mtp_arrays(arrays,base,mesh,target)
assert bound.decoder.embedding_local is base.embedding_local
assert bound.decoder.lm_head_local is base.lm_head_local
for a,b in zip(jax.tree.leaves(weights),jax.tree.leaves(bound)):
    np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
weights=bound
rope=put(jnp.asarray(d.build_ws32_main_rope_table(config),jnp.bfloat16))
initial=d.make_ws32_initial_state(mesh,config)
interpret=dict(sparse_attention_interpret=True,linear_interpret=True)
full=build_mtp_draft(mesh,config,**interpret)
shared=build_mtp_draft(mesh,config,index_share=True,**interpret)
commit=build_prefix_committer(mesh,config)
tokens=put(jnp.array([31,32,33],jnp.int32))
h=put(jnp.asarray(rng.normal(0,.2,(3,H)),jnp.bfloat16),P(None,'feature'))
proposal=jax.block_until_ready(full(tokens,h,initial,weights,rope))
assert np.asarray(proposal.contract_valid).all()

def exact(x,y,label):
    assert jax.tree.structure(x)==jax.tree.structure(y),label
    for xx,yy in zip(jax.tree.leaves(x),jax.tree.leaves(y)):
        for xs,ys in zip(xx.addressable_shards,yy.addressable_shards):
            assert xs.index==ys.index and xs.device==ys.device
            a,b=np.ascontiguousarray(xs.data),np.ascontiguousarray(ys.data)
            assert np.array_equal(a.view(np.uint8),b.view(np.uint8)),label

# Independent composition: substitute projected inputs into an embedding
# table, then run the already-tested target verifier's single-layer body.
# Projection itself has a separate independent unsharded numerical proof.
embedded=put(base.embedding_local[tokens],P(None,'feature'))
projected=build_mtp_projection(mesh,hidden_size=H)(embedded,h,put(jnp.arange(3,dtype=jnp.int32)),projection)
replacement=put(jnp.zeros_like(base.embedding_local).at[tokens].set(projected.hidden_local),P('expert','feature'))
verify=build_verifier(mesh,config,canonical_mlp=True,batched_attention=True,
    small_expert_tiles=True,rowwise_dsa=True,**interpret)
reference=jax.block_until_ready(verify(tokens,initial,base._replace(embedding_local=replacement),rope))
exact(proposal,reference,'projected single-layer composition')
state=commit(initial,proposal,put(jnp.int32(3)))
assert int(np.asarray(state.position)[0])==3
# Recurrent IndexShare consumes the last MTP normalized hidden at next position.
recurrent=shared(proposal.predictions[-1:],proposal.normalized_hidden_local[-1:],state,weights,rope)
assert np.asarray(recurrent.contract_valid).all()
exact(recurrent.index_cache_local,state.index_cache_local,'IndexShare does not write index cache')
exact(recurrent.selected_positions,state.selected_positions,'shortlist reuse')
exact(recurrent.selected_valid_counts,state.selected_valid_counts,'count reuse')
exact(recurrent.selected_scores,state.selected_scores,'score reuse')
assert not np.array_equal(np.asarray(recurrent.kv_cache_local),np.asarray(state.kv_cache_local))
# Poison every DSA table: recurrence must remain bitwise unchanged, proving it
# does not invoke a key projection, query score or top-k using those tables.
layer=base.layers[0]
poisoned=layer._replace(dsa=jax.tree.map(lambda x:jnp.full_like(x,jnp.nan),layer.dsa))
bad_dsa=weights._replace(decoder=base._replace(layers=(poisoned,)))
other=shared(proposal.predictions[-1:],proposal.normalized_hidden_local[-1:],state,bad_dsa,rope)
exact(recurrent,other,'DSA tables unused during recurrence')
# A refresh from the committed root recomputes full-index target-hidden rows.
refreshed=full(put(jnp.array([44,45],jnp.int32)),h[:2],state,weights,rope)
assert np.asarray(refreshed.contract_valid).all()
assert not np.array_equal(np.asarray(refreshed.index_cache_local),np.asarray(state.index_cache_local))
# Commit one row and roll every rejected physical row back; native and target
# caches share the proven rollback helper, but never the same array roots.
accepted=commit(state,refreshed,put(jnp.int32(1)))
assert int(np.asarray(accepted.position)[0])==4
np.testing.assert_array_equal(np.asarray(accepted.kv_cache_local)[:,:,4:],np.asarray(state.kv_cache_local)[:,:,4:])
np.testing.assert_array_equal(np.asarray(accepted.index_cache_local)[:,:,4:],np.asarray(state.index_cache_local)[:,:,4:])
# Later projected tokens/hidden cannot affect prior rows of a proposal.
changed=full(tokens.at[2].set(66),h.at[2].set(4),initial,weights,rope)
exact(changed.normalized_hidden_local[:2],proposal.normalized_hidden_local[:2],'causal hidden')
exact(changed.predictions[:2],proposal.predictions[:2],'causal predictions')
# IndexShare cannot initialize an empty draft prompt, and malformed input or
# any live invalid hidden must prevent the entire proposal from being committed.
unseeded=shared(tokens[:1],h[:1],initial,weights,rope)
assert not np.asarray(unseeded.contract_valid).any()
invalid=full(tokens,h.at[1,0].set(jnp.nan),initial,weights,rope)
assert not np.asarray(invalid.contract_valid).all()
refused=commit(initial,invalid,put(jnp.int32(1)))
exact(refused,initial._replace(contract_valid=jnp.zeros_like(initial.contract_valid)),'invalid hidden refusal')
try:shared(tokens,h,initial,weights,rope)
except ValueError:pass
else:raise AssertionError('multi-row recurrent IndexShare must be refused')
# Complete native history orchestration: bootstrap/recurrence/target refresh.
from glm_tpu.perf.mtp_state import (make_native_state,TargetHistory,
    build_native_refresh,build_native_extend)
root=make_native_state(mesh,config)
refresh=build_native_refresh(mesh,config,**interpret)
extend=build_native_extend(mesh,config,**interpret)
history=TargetHistory(initial.position,tokens,h,put(jnp.ones(3,jnp.bool_)))
seed=refresh(root,history,put(jnp.int32(3)),weights,rope)
exact(seed.cache,state,'orchestrated bootstrap')
exact(seed.next_token,proposal.predictions[-1:],'cached first draft')
exact(seed.normalized_hidden_local,proposal.normalized_hidden_local[-1:],'cached native hidden')
exact(extend(seed,weights,rope),recurrent,'orchestrated IndexShare second draft')
from glm_tpu.perf.mtp_state import build_native_inputs
for rows in (1,2,3):
    propose_ids=build_native_inputs(mesh,config,rows=rows,**interpret)
    pending=put(jnp.array([44],jnp.int32))
    ids=propose_ids(pending,seed,weights,rope)
    wanted=np.concatenate((np.asarray(pending),np.asarray(seed.next_token),np.asarray(recurrent.predictions)))[:rows]
    np.testing.assert_array_equal(np.asarray(ids),wanted)
    bad=seed._replace(cache=seed.cache._replace(contract_valid=put(jnp.array([False]))))
    np.testing.assert_array_equal(np.asarray(propose_ids(pending,bad,weights,rope)),np.full(rows,-1))

# Rejected continuation is disposable; full refresh uses target history and old root.
next_history=TargetHistory(state.position,put(jnp.array([44,45],jnp.int32)),h[:2],put(jnp.ones(2,jnp.bool_)))
updated=refresh(seed,next_history,put(jnp.int32(1)),weights,rope)
exact(updated.cache,accepted,'orchestrated prefix refresh')
exact(updated.next_token,refreshed.predictions[:1],'partial refresh draft frontier')
exact(updated.normalized_hidden_local,refreshed.normalized_hidden_local[:1],'partial refresh hidden frontier')
exact(refresh(seed,next_history,put(jnp.int32(0)),weights,rope),seed,'zero prefix preserves metadata')
for wrong in (next_history._replace(position=put(jnp.array([4],jnp.int32))),
              next_history._replace(contract_valid=put(jnp.array([True,False])))):
    refused=refresh(seed,wrong,put(jnp.int32(1)),weights,rope)
    exact(refused,seed._replace(cache=seed.cache._replace(contract_valid=put(jnp.array([False])))),
          'misaligned or unhealthy target history refused atomically')
# Eight-row bootstrap and a partial refresh crossing the first expert owner.
# Start near the boundary by genuinely filling preceding native history.
wide_tokens=put(jnp.arange(8,dtype=jnp.int32)+40)
wide_h=put(jnp.asarray(rng.normal(0,.2,(8,H)),jnp.bfloat16),P(None,'feature'))
wide_history=TargetHistory(seed.cache.position,wide_tokens,wide_h,put(jnp.ones(8,jnp.bool_)))
near=refresh(seed,wide_history,put(jnp.int32(8)),weights,rope)
assert int(np.asarray(near.cache.position)[0])==11
crossing=TargetHistory(near.cache.position,wide_tokens,wide_h,put(jnp.ones(8,jnp.bool_)))
proposal_cross=full(wide_tokens,wide_h,near.cache,weights,rope)
crossed=refresh(near,crossing,put(jnp.int32(6)),weights,rope)
assert int(np.asarray(crossed.cache.position)[0])==17
exact(crossed.cache,commit(near.cache,proposal_cross,put(jnp.int32(6))),'owner-crossing partial refresh')
exact(crossed.next_token,proposal_cross.predictions[5:6],'owner-crossing draft frontier')
# Run the actual guarded session with device proposals and native refresh.
from glm_tpu.greenfield.runtime import ws32_batched_prefill as pre
from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
from glm_tpu.perf.speculative_request import SpeculativeRequestSession
from time import perf_counter
target_prompt=verify(tokens,initial,base,rope)
target_cache=commit(initial,target_prompt,put(jnp.int32(3)))
pending=target_prompt.predictions[-1:]
shifted=jnp.concatenate((tokens[1:],pending))
native_seed=refresh(root,TargetHistory(initial.position,shifted,
    target_prompt.normalized_hidden_local,put(jnp.ones(3,jnp.bool_))),put(jnp.int32(3)),weights,rope)
prestate=pre.make_ws32_batched_prefill_state(mesh,config,prompt_length=3)._replace(
    decoder=target_cache,finished=put(jnp.bool_(True)))
input_functions={n:build_native_inputs(mesh,config,rows=n,**interpret) for n in (1,2,3)}
seen=[]
def propose_live(pending,target,native,rows):
    ids=input_functions[rows](pending,native,weights,rope)
    proposal=verify(ids,target,base,rope)
    seen.append((target,native,proposal))
    return ids,proposal
session=SpeculativeRequestSession(RequestPolicy('native-cpu-integration',0,3,7,
    config.context_capacity,config.geometry.vocab_size,(0,)),native_state=native_seed,
    propose=propose_live,commit=commit,
    refresh=lambda old,history,count:refresh(old,history,count,weights,rope),
    replicate_count=put,fleet_agree=lambda x:True,rows=3,
    decode_step=None,replicate_uniform=put,fleet_all=lambda x:x,deliver=lambda event:None,
    delivery_boundary='CPU test sink',request_started=perf_counter())
session.accept_prefill(pre.Ws32BatchedPrefillResult(prestate,pending))
while not session.finished:
    before=len(session.events)
    session.step()
    old_target,old_native,proposal=seen[-1]
    consumed=len(session.events)-before
    # Teacher-forced refresh from accepted target history must reproduce every
    # cache/metadata bit, including after native proposals are rejected.
    manual=full(proposal.predictions,proposal.normalized_hidden_local,old_native.cache,weights,rope)
    manual_cache=commit(old_native.cache,manual,put(jnp.int32(consumed)))
    exact(session._native.cache,manual_cache,'accepted target history native replay')
    exact(session._native.next_token,manual.predictions[consumed-1:consumed],'accepted native next token')
    assert int(np.asarray(session._state.position)[0])==3+len(session.events)-1
assert not session.failed and len(session.events)>1
session.release()
print(json.dumps(dict(scope='CPU32 synthetic single native MTP layer, three bootstrap/two refresh/one recurrent rows',
    projected_transformer_composition='bitwise all 32 owners',indexshare='KV updated; full DSA skipped; index and shortlist unchanged',
    causality=True,rejected_cache_rollback=True,invalid_hidden_atomic_refusal=True,trained_native_mtp=False)))
'''
    result=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=600)
    assert result.returncode==0,result.stdout+result.stderr
    print(result.stdout)
