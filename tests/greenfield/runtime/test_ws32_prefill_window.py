"""Window composition and rollback on CPU32; not TPU numerical promotion."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("rolled_components", [False, True])
def test_window_matches_tiled_decoder_and_atomic_rollback_cpu32(rolled_components):
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b
from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table,ws32_decoder_weight_specs
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
config,weights,wk=fixture(mesh,panel_geometry=ROLLED)
def put(v,s=P()):return jax.device_put(v,NamedSharding(mesh,s))
wk=tuple(put(v) for v in wk)
rope=put(jnp.asarray(build_ws32_main_rope_table(config),jnp.bfloat16))
base=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=761)
rng=np.random.default_rng(347)
ds=base.decoder
ds=ds._replace(
    kv_cache_local=put(jnp.asarray(rng.normal(0,.1,ds.kv_cache_local.shape),jnp.bfloat16),P(None,None,'expert',None)),
    index_cache_local=put(jnp.asarray(rng.normal(0,.1,ds.index_cache_local.shape),jnp.bfloat16),P(None,None,'expert',None)),
    position=put(jnp.array([505],jnp.int32)),context_lengths=put(jnp.array([506],jnp.int32)),
    block_tables=put(jnp.array([[2,0,1]],jnp.int32)),
)
base=base._replace(decoder=ds,repaired_index_local=jnp.full_like(base.repaired_index_local,7))
opts=dict(key_tile=128,sparse_attention_interpret=True,linear_interpret=True)
window_opts=dict(mlp_window=True,rolled_prefix=ROLLED,expert_panels=ROLLED,sorted_local_merge=ROLLED,paired_position_sort=ROLLED)
wide=b.build_ws32_batched_prefill_program(mesh,config,block_rows=128,**window_opts,**opts)
small=b.build_ws32_batched_prefill_program(mesh,config,block_rows=32,**opts)
assert wide.mlp_window and not small.mlp_window
assert wide.rolled_prefix==ROLLED and wide.expert_panels==ROLLED and wide.sorted_local_merge==ROLLED
tokens=put(jnp.arange(128,dtype=jnp.int32)+30)
compiled=wide.execute.lower(tokens,put(jnp.int32(128)),base,weights,wk,rope).compile()
print('compiled B128 eight-layer window',flush=True)
ops=[x for x in parse_hlo_module(compiled.as_text()).instructions if x.is_collective]
fg=tuple(tuple(range(e*4,e*4+4)) for e in range(8))
eg=tuple(tuple(e*4+f for e in range(8)) for f in range(4))
assert ops and all(x.replica_groups in (fg,eg) for x in ops)
if ROLLED:
    loops=[line for line in compiled.as_text().splitlines() if ' while(' in line and 'greenfield_ws32_prefill_rolled_prefix/while"' in line]
    assert len(loops)==8,loops

def equal_tree(a,c):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(c)):
        np.testing.assert_array_equal(x,y)
def reference(t,n,s,r):
    result=None
    for start in range(0,n,32):
        count=min(32,n-start)
        result=small.execute(t[start:start+32],put(jnp.int32(count)),s,weights,wk,r)
        s=result.state
    return result
def failed(result,old):
    assert not np.asarray(result.state.decoder.contract_valid).any()
    assert result.next_token.tolist()==[-1]
    equal_tree(result.state,old._replace(decoder=old.decoder._replace(contract_valid=put(jnp.zeros(1,jnp.bool_)))))

# Two128 windows, all8 layers (full producers0/1/2/6 and shared2->3/6->7).
first=compiled(tokens,put(jnp.int32(128)),base,weights,wk,rope)
assert np.asarray(first.state.decoder.contract_valid).all()
assert first.state.decoder.position.tolist()==[633] and not bool(first.state.finished)
assert first.next_token.tolist()==[-1]
assert not np.array_equal(first.state.decoder.index_cache_local,first.state.repaired_index_local)
equal_tree(first,reference(tokens,128,base,rope))
second=compiled(tokens,put(jnp.int32(128)),first.state,weights,wk,rope)
equal_tree(second,reference(tokens,128,first.state,rope))
assert bool(second.state.finished)
b.finish_ws32_batched_prefill(second)
np.testing.assert_array_equal(second.state.decoder.index_cache_local,second.state.repaired_index_local)
altered=first.state._replace(repaired_index_local=jnp.full_like(first.state.repaired_index_local,-9))
changed=compiled(tokens,put(jnp.int32(128)),altered,weights,wk,rope)
for name in ('kv_cache_local','selected_positions','selected_valid_counts','selected_scores'):
    np.testing.assert_array_equal(getattr(changed.state.decoder,name),getattr(second.state.decoder,name))
np.testing.assert_array_equal(changed.next_token,second.next_token)
print('two windows equal explicit4x32 decoder composition',flush=True)

# Tail boundaries, poisoned padding and empty trailing tiles close to capacity.
for n in (31,32,33,127):
    offset=1500 if n<=33 else 505
    s=base._replace(prompt_length=put(jnp.int32(offset+n)),decoder=ds._replace(
        position=put(jnp.array([offset],jnp.int32)),context_lengths=put(jnp.array([offset+1],jnp.int32))))
    t=tokens.at[n:].set(-2147483648)
    r=rope.at[offset+n:].set(jnp.nan)
    result=compiled(t,put(jnp.int32(n)),s,weights,wk,r)
    assert np.asarray(result.state.decoder.contract_valid).all(),n
    equal_tree(result,reference(t,n,s,r))
    assert bool(result.state.finished) and result.state.decoder.position.tolist()==[offset+n]
    print('tail',n,'pass',flush=True)

# A live failure in the last tile must roll back all earlier proposed writes.
failed(compiled(tokens.at[110].set(-1),put(jnp.int32(128)),base,weights,wk,rope),base)
for n in (0,-1,129,2147483647):
    failed(compiled(tokens,put(jnp.int32(n)),base,weights,wk,rope),base)
for offset in (-1,2147483647):
    invalid=base._replace(decoder=base.decoder._replace(position=put(jnp.array([offset],jnp.int32))))
    failed(compiled(tokens,put(jnp.int32(128)),invalid,weights,wk,rope),invalid)
failed(compiled(tokens,put(jnp.int32(128)),second.state,weights,wk,rope),second.state)

# Inject single-owner late-row layer health AFTER its prefix wrote all proposed
# caches. This is an intervention in the real composition, not caller health.
original=b.ws32_prefill_layer_window_mapped
def poison_layer(*args,**kwargs):
    result=original(*args,**kwargs)
    owner_bad=(lax.axis_index('expert')==3)&(lax.axis_index('feature')==2)
    return result._replace(contract_valid=result.contract_valid.at[110].set(
        result.contract_valid[110]&~owner_bad))
b.ws32_prefill_layer_window_mapped=poison_layer
poison=b.build_ws32_batched_prefill_program(mesh,config,block_rows=128,**window_opts,**opts)
final_base=base._replace(prompt_length=put(jnp.int32(633)))
failed(poison.execute(tokens,put(jnp.int32(128)),final_base,weights,wk,rope),final_base)
b.ws32_prefill_layer_window_mapped=original
print('CPU32_PREFILL_WINDOW_PASS',flush=True)
"""
    code = code.replace("ROLLED", repr(rolled_components))
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        text=True,
        capture_output=True,
        timeout=480,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU32_PREFILL_WINDOW_PASS" in result.stdout


@pytest.mark.parametrize(
    "field", ["rolled_prefix", "expert_panels", "sorted_local_merge"]
)
def test_new_window_flags_are_static_and_require_opt_in(field):
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import _require_window_options
    from glm_tpu.greenfield.errors import PlanValidationError

    options = dict(
        mlp_window=True,
        rolled_prefix=False,
        expert_panels=False,
        sorted_local_merge=False,
    )
    for bad in (1, None, "true"):
        with pytest.raises(PlanValidationError):
            _require_window_options(**{**options, field: bad})
    with pytest.raises(PlanValidationError):
        _require_window_options(**{**options, "mlp_window": False, field: True})
