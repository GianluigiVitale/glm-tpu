"""Compare the distributed adaptation to a full selected-cache CPU expression."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('rows', [1, 2, 3, 4, 32])
def test_global_max_attention_with_empty_owners_and_causal_rows(rows):
    code = r'''
from dataclasses import replace
from pathlib import Path
import hashlib,json,os
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,PartitionSpec as P
from glm_tpu.perf.global_max_attention import global_max_attention_mapped
from glm_tpu.greenfield.kernels.reference.attention import (
    MlaNumericalContract,StageLocalKvLayout,SparseAttentionResult,gather_stage_local_selected_kv_aligned)
from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions
from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig,pregathered_sparse_mla_pallas
mesh=Mesh(np.asarray(jax.devices(),object),('expert',))
c=MlaNumericalContract(num_heads=8,kv_lora_rank=128,qk_nope_head_dim=64,
    qk_rope_head_dim=64,qk_head_dim=128,v_head_dim=128,packed_cache_width=256,top_k=128)
layout=StageLocalKvLayout(logical_page_size=512,local_parallel_size=8,packed_cache_width=256)
rng=np.random.default_rng(610)
q=jnp.asarray(rng.normal(size=(ROWS,8,128))*.3,jnp.bfloat16)
rope=jnp.asarray(rng.normal(size=(ROWS,8,64))*.3,jnp.bfloat16)
cache=jnp.asarray(rng.normal(size=(8,2,64,256)),jnp.bfloat16)
tables=jnp.tile(jnp.array([[1,0]],jnp.int32),(ROWS,1))
lengths=jnp.arange(ROWS,dtype=jnp.int32)+1024-ROWS
def body(q,r,cache,pos,count,length):
    selected=SelectedPositions(pos,count)
    actual=global_max_attention_mapped(q,r,cache[0],tables,selected,length,contract=c,layout=layout)
    aligned=gather_stage_local_selected_kv_aligned(cache[0],tables,selected,length,layout=layout,
        owner_index=jax.lax.axis_index('expert'))
    keys=jax.lax.psum(aligned.values,'expert')
    packed=jnp.concatenate((q,r,jnp.zeros((ROWS,1,64),jnp.bfloat16)),axis=-1)
    scores=jnp.einsum('rhd,rkd->rhk',packed,keys,preferred_element_type=jnp.float32)*c.softmax_scale
    live=jnp.arange(c.top_k)[None,None,:]<count[:,None,None]
    scores=jnp.where(live,scores,-jnp.inf)
    maximum=jnp.max(scores,axis=-1,keepdims=True)
    e=jnp.where(live,jnp.exp(scores-jnp.where(jnp.isfinite(maximum),maximum,0.)),0.)
    den=e.sum(-1)
    num=jnp.einsum('rhk,rkd->rhd',e.astype(jnp.bfloat16),keys[...,:128],preferred_element_type=jnp.float32)
    expected=(num/jnp.where(den>0,den,1.)[...,None]).astype(jnp.bfloat16)
    frozen=pregathered_sparse_mla_pallas(q,r,keys,count,contract=replace(c,num_heads=1),
        config=SparseMlaConfig(segment_block=16),interpret=True,prefill=ROWS>1)
    return actual,expected,frozen
fn=jax.jit(jax.shard_map(body,mesh=mesh,
    in_specs=(P(None,'expert'),P(None,'expert'),P('expert'),P(),P(),P()),
    out_specs=(SparseAttentionResult(P(None,'expert'),P(None,'expert'),P()),P(None,'expert'),P(None,'expert')),check_vma=False))
errors=[]
for chosen in ([],[0],list(range(64)),list(range(0,992,8)),rng.choice(980,128,replace=False).tolist()):
    count=len(chosen)
    pos=jnp.tile(jnp.array([chosen+[-1]*(128-count)],jnp.int32),(ROWS,1))
    counts=jnp.full((ROWS,),count,jnp.int32)
    if ROWS>1:pos=pos.at[1].set(-1);counts=counts.at[1].set(0)
    actual,expected,frozen=fn(q,rope,cache,pos,counts,lengths)
    assert bool(actual.contract_valid.all()) and bool(jnp.isfinite(actual.output).all())
    # Sum grouping can round one BF16 ulp differently; not an exactness proof.
    np.testing.assert_allclose(actual.output.astype(jnp.float32),expected.astype(jnp.float32),atol=.0078125,rtol=.008)
    frozen_error=float(jnp.max(jnp.abs(actual.output.astype(jnp.float32)-frozen.astype(jnp.float32))))
    errors.append(dict(selected_count=count,max_abs_vs_global_expression=float(jnp.max(jnp.abs(
        actual.output.astype(jnp.float32)-expected.astype(jnp.float32)))),max_abs_vs_frozen=frozen_error))
    assert frozen_error<=.015625,(count,frozen_error)
    empty=np.asarray(counts)==0
    assert np.all(np.asarray(actual.output)[empty]==0)
    assert np.isneginf(np.asarray(actual.logsumexp)[empty]).all()
pos=jnp.tile(jnp.array([[0]+[-1]*127],jnp.int32),(ROWS,1)); counts=jnp.ones(ROWS,jnp.int32)
poison=cache.at[0,1,0,0].set(jnp.nan)
actual,_,_=fn(q,rope,poison,pos,counts,lengths)
assert not bool(actual.contract_valid.any())
poison=cache.at[0,1,3,0].set(jnp.nan)
actual,_,_=fn(q,rope,poison,pos,counts,lengths)
assert bool(actual.contract_valid.all())
# A live future token is a contract error, not a key to silently include.
future=pos.at[:,0].set(lengths)
actual,_,_=fn(q,rope,cache,future,counts,lengths)
assert not bool(actual.contract_valid.any())
duplicate=pos.at[:,1].set(0)
actual,_,_=fn(q,rope,cache,duplicate,jnp.full(ROWS,2,jnp.int32),lengths)
assert not bool(actual.contract_valid.any())
# A poisoned query on one head owner invalidates every owner's result.
actual,_,_=fn(q.at[0,7,0].set(jnp.nan),rope,cache,pos,counts,lengths)
assert not bool(actual.contract_valid[0])
if os.environ.get('GLM_GLOBAL_MAX_CPU_REPORT_DIR'):
    output=Path(os.environ['GLM_GLOBAL_MAX_CPU_REPORT_DIR']);output.mkdir(parents=True,exist_ok=True)
    source=Path('glm_tpu/perf/global_max_attention.py')
    (output/f'rows{ROWS}.json').write_text(json.dumps(dict(rows=ROWS,errors=errors,jax=jax.__version__,
        backend='cpu',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),exact=False,
        scope='synthetic 8-owner small geometry; no trained or TPU admission'),indent=2)+'\n')
print('global-max distributed numerical and health checks passed')
'''.replace('ROWS', str(rows))
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
        env=dict(os.environ, JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_force_host_platform_device_count=8'),
        timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
