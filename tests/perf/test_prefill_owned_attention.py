"""Bounded owner buffers preserve complete B114/B128 prefill state bit patterns."""
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program


@pytest.mark.parametrize('lse,bound',[(False,128),(True,True),(True,0),(True,192)])
def test_owned_prefill_options_fail_before_build(lse,bound):
    config=SimpleNamespace(sparse_segment_block=128,geometry=SimpleNamespace(dsa_top_k=512))
    with pytest.raises(ValueError,match='owned key capacity'):
        build_ws32_prefill_challenger_program(None,config,lse_attention=lse,owned_key_capacity=bound)


@pytest.mark.parametrize('rows',[114,128])
def test_complete_owned_prefill_is_bitwise_with_full_buffer(rows):
    code=r'''
from dataclasses import replace
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
from glm_tpu.greenfield.runtime import ws32_batched_prefill as pre,ws32_decoder as dec
from glm_tpu.perf.bf16_resident import bf16_resident_weights
from glm_tpu.perf.prefill_challenger import build_ws32_prefill_challenger_program
import glm_tpu.perf.prefill_attention as attention
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
def put(x):return jax.device_put(x,NamedSharding(mesh,P()))
config,raw,wk=fixture(mesh,panel_geometry=True)
config=replace(config,geometry=replace(config.geometry,dsa_top_k=512))
weights=bf16_resident_weights(mesh,config,raw)
wk=tuple(put(x) for x in wk)
rope=put(jnp.asarray(dec.build_ws32_main_rope_table(config),jnp.bfloat16))
initial=pre.make_ws32_batched_prefill_state(mesh,config,prompt_length=3)
tokens=put(jnp.array([30,31,32]+[-1]*(ROWS-3),jnp.int32));count=put(jnp.int32(3))
opts=dict(lse_attention=True,bf16_resident=True,block_rows=ROWS,key_tile=128,
    mlp_window=True,rolled_prefix=True,expert_panels=True,canonical_dense=True,
    paired_position_sort=True,sorted_local_merge=True,sparse_attention_interpret=True,linear_interpret=True)
original=attention.lse_attention_mapped
full=build_ws32_prefill_challenger_program(mesh,config,**opts)
small=build_ws32_prefill_challenger_program(mesh,config,owned_key_capacity=128,**opts)
assert attention.lse_attention_mapped is original
x=full.execute(tokens,count,initial,weights,wk,rope)
y=small.execute(tokens,count,initial,weights,wk,rope)
assert bool(np.asarray(y.state.finished)) and bool(np.asarray(y.state.decoder.contract_valid).all())
for a,b in zip(jax.tree.leaves(x),jax.tree.leaves(y)):
    np.testing.assert_array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))
# Finished state refuses another append and keeps every owned cache/frontier.
z=small.execute(tokens,count,y.state,weights,wk,rope)
assert not bool(np.asarray(z.state.decoder.contract_valid).all())
for field in ('kv_cache_local','index_cache_local','position','context_lengths'):
    np.testing.assert_array_equal(np.asarray(getattr(z.state.decoder,field)),np.asarray(getattr(y.state.decoder,field)))
assert attention.lse_attention_mapped is original
'''.replace('ROWS',str(rows))
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,
        env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32'),timeout=600)
    assert p.returncode==0,p.stdout+p.stderr
