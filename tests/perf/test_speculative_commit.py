"""Independent physical-cache oracle for speculative prefix commit."""
import os
import subprocess
import sys


def test_prefix_commit_across_owners_pages_and_all_host_refusal_cpu32():
    code = r'''
from types import SimpleNamespace
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderState,ws32_decoder_state_specs
from glm_tpu.perf.speculative_verify import VerificationProposal,proposal_specs,commit_prefix_mapped
from glm_tpu.perf.speculative_accept import greedy_acceptance
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
config=SimpleNamespace(logical_page_size=512,cache_layout=SimpleNamespace(local_rows_per_page=64))
def put(x,spec):return jax.device_put(x,NamedSharding(mesh,spec))
def place(value,specs):return jax.tree.map(put,value,specs)
def same(a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        x,y=np.ascontiguousarray(x),np.ascontiguousarray(y)
        np.testing.assert_array_equal(x.view(np.uint8),y.view(np.uint8))
def body(state,proposal,n,poison):
    # A single owner refusing must prevent every owner from committing.
    local_bad=poison & (jax.lax.axis_index('expert')==5) & (jax.lax.axis_index('feature')==1)
    proposal=proposal._replace(contract_valid=proposal.contract_valid & ~local_bad)
    return commit_prefix_mapped(state,proposal,n,config=config)
commit=jax.jit(jax.shard_map(body,mesh=mesh,
    in_specs=(ws32_decoder_state_specs(),proposal_specs(),P(),P()),
    out_specs=ws32_decoder_state_specs(),check_vma=False))
pages=np.array([[2,0,1]],np.int32)
rows=5
for position in (61,510):
    kv=jnp.asarray((np.arange(2*3*512*4).reshape(2,3,512,4)%127)-63,jnp.bfloat16)
    ix=jnp.asarray((np.arange(3*512*2).reshape(1,3,512,2)%97)-48,jnp.bfloat16)
    original=Ws32DecoderState(kv,ix,jnp.full((1,4),-1,jnp.int32),
        jnp.zeros((1,),jnp.int32),jnp.full((1,4),-jnp.inf,jnp.float32),
        jnp.array([position],jnp.int32),jnp.asarray(pages),
        jnp.array([position+1],jnp.int32),jnp.ones((1,),jnp.bool_))
    selected=jnp.arange(rows*4,dtype=jnp.int32).reshape(rows,4)
    scores=jnp.arange(rows*4,dtype=jnp.float32).reshape(rows,4)*.125
    counts=jnp.full((rows,),4,jnp.int32)
    def prefix(n):
        # Global host-side physical addressing, independent of the mapped owner calculation.
        k,x=kv,ix
        for i in range(n):
            logical,offset=divmod(position+i,512)
            page=int(pages[0,logical])
            k=k.at[:,page,offset,:].set(i+100)
            x=x.at[:,page,offset,:].set(i+200)
        if not n:return original
        return original._replace(kv_cache_local=k,index_cache_local=x,
            selected_positions=selected[n-1:n],selected_valid_counts=counts[n-1:n],
            selected_scores=scores[n-1:n],position=original.position+n,
            context_lengths=original.context_lengths+n)
    full=prefix(rows)
    proposal=VerificationProposal(full.kv_cache_local,full.index_cache_local,selected,counts,scores,
        jnp.arange(rows,dtype=jnp.int32)+11,jnp.zeros((rows,8),jnp.bfloat16),
        jnp.zeros((rows,8),jnp.bfloat16),jnp.ones((rows,),jnp.bool_))
    state=place(original,ws32_decoder_state_specs()); proposed=place(proposal,proposal_specs())
    for n in range(rows+1):same(commit(state,proposed,jnp.int32(n),False),prefix(n))
    refused=original._replace(contract_valid=jnp.zeros((1,),jnp.bool_))
    for n in (-1,rows+1):same(commit(state,proposed,jnp.int32(n),False),refused)
    same(commit(state,proposed,jnp.int32(3),True),refused)
    # Acceptance supplies the consumed-input count, including correction/bonus.
    for rejection in range(rows):
        tokens=jnp.concatenate((jnp.array([10],jnp.int32),proposal.predictions[:-1]))
        if rejection<rows-1:tokens=tokens.at[rejection+1].set(255)
        for budget in (0,1,3,rows):
            choice=greedy_acceptance(tokens,proposal.predictions,jnp.int32(budget),eos_token_ids=(13,))
            n=min(rejection+1,budget,3)
            assert int(choice.emitted_count)==n
            same(commit(state,proposed,choice.emitted_count,False),prefix(n))
print('Physical prefix commit, owner/page crossings, EOS/budget and one-owner refusal are bitwise exact')
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
        env=dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=32'),
        timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
