"""Frozen B114 executable with91live rows; CPU semantics, not TPU admission."""

import os
import subprocess
import sys


def test_frozen_b114_live91_near_capacity_cpu32():
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
from glm_tpu.greenfield.runtime import ws32_batched_prefill as b
from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object)[::-1].reshape(8,4),('expert','feature'))
config,weights,wk=fixture(mesh,panel_geometry=True)
def put(value,spec=P()):return jax.device_put(value,NamedSharding(mesh,spec))
def same(a,e):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(e),strict=True):
        np.testing.assert_array_equal(x,y)
wk=tuple(put(w) for w in wk)
rope=put(jnp.asarray(build_ws32_main_rope_table(config),jnp.bfloat16))
start=config.context_capacity-128
state=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=start+91)
state=state._replace(decoder=state.decoder._replace(
    position=put(jnp.array([start],jnp.int32)),
    context_lengths=put(jnp.array([start+1],jnp.int32))))
opts=dict(key_tile=128,sparse_attention_interpret=True,linear_interpret=True,
    mlp_window=True,rolled_prefix=True,expert_panels=True,
    paired_position_sort=True,sorted_local_merge=True)
wide=b.build_ws32_batched_prefill_program(mesh,config,block_rows=114,**opts)
small=b.build_ws32_batched_prefill_program(mesh,config,block_rows=32,**opts)
tokens=put(jnp.concatenate((jnp.arange(91,dtype=jnp.int32)+30,jnp.zeros(23,jnp.int32))))
actual=wide.execute(tokens,put(jnp.int32(91)),state,weights,wk,rope)
jax.block_until_ready(actual)
assert bool(actual.state.finished) and np.asarray(actual.state.decoder.contract_valid).all()
assert actual.state.decoder.position.tolist()==[start+91]
assert actual.state.decoder.context_lengths.tolist()==[start+92]
b.finish_ws32_batched_prefill(actual)
print('FROZEN_TAIL_BASELINE_PASS',flush=True)
# Poison all23 padded token IDs and all inactive rotary rows, including the
# completely empty fourth prefix tile. Live outputs/health/cache must not move.
poison=wide.execute(tokens.at[91:].set(-2147483648),put(jnp.int32(91)),
    state,weights,wk,rope.at[start+91:].set(jnp.nan))
same(actual,poison)
print('FROZEN_TAIL_POISON_PASS',flush=True)
# Shape-matched32-row prefix reference; do not compare with a staticM27 scorer.
current=state
for offset,count in ((0,32),(32,32),(64,27)):
    ids=put(jnp.concatenate((tokens[offset:offset+count],jnp.full(32-count,-1,jnp.int32))))
    expected=small.execute(ids,put(jnp.int32(count)),current,weights,wk,rope)
    current=expected.state
same(actual,expected)
print('FROZEN_TAIL_CONTROL_PASS',flush=True)
# Last live row90 is NOT row113: invalid live input must poison health and
# atomically restore the entire initial state, including both index caches.
bad=wide.execute(tokens.at[90].set(-1),put(jnp.int32(91)),state,weights,wk,rope)
assert not np.asarray(bad.state.decoder.contract_valid).any()
assert bad.next_token.tolist()==[-1]
want=state._replace(decoder=state.decoder._replace(contract_valid=put(jnp.zeros(1,jnp.bool_))))
same(bad.state,want)
print('FROZEN_TAIL_ROLLBACK_PASS',flush=True)
# Diagnostic geometry: physicalB128/live32 repeatedly, then physicalB114/live27.
# Last physical windows extend beyond capacity; only live rows may write/rotate.
main=b.build_ws32_batched_prefill_program(mesh,config,block_rows=128,**opts)
start=config.context_capacity-96
initial=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=start+91)
initial=initial._replace(decoder=initial.decoder._replace(
    position=put(jnp.array([start],jnp.int32)),
    context_lengths=put(jnp.array([start+1],jnp.int32))))
current=control=initial
for offset,count,physical,program in ((0,32,128,main),(32,32,128,main),(64,27,114,wide)):
    ids=put(jnp.concatenate((jnp.arange(count,dtype=jnp.int32)+30+offset,
                            jnp.zeros(physical-count,jnp.int32))))
    actual=program.execute(ids,put(jnp.int32(count)),current,weights,wk,rope)
    poison=program.execute(ids.at[count:].set(-2147483648),put(jnp.int32(count)),
                          current,weights,wk,rope.at[start+offset+count:].set(jnp.nan))
    same(actual,poison)
    narrow=put(jnp.concatenate((ids[:count],jnp.zeros(32-count,jnp.int32))))
    expected=small.execute(narrow,put(jnp.int32(count)),control,weights,wk,rope)
    same(actual,expected)
    if count==27:
        bad=program.execute(ids.at[26].set(-1),put(jnp.int32(count)),current,weights,wk,rope)
        want=current._replace(decoder=current.decoder._replace(contract_valid=put(jnp.zeros(1,jnp.bool_))))
        same(bad.state,want)
        assert bad.next_token.tolist()==[-1]
    current,control=actual.state,expected.state
assert current.decoder.position.tolist()==[start+91] and bool(current.finished)
b.finish_ws32_batched_prefill(actual)
print('FROZEN_LIVE32_CAPACITY_CONTROL_POISON_ROLLBACK_PASS',flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        text=True,
        capture_output=True,
        timeout=420,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FROZEN_TAIL_ROLLBACK_PASS" in result.stdout
    assert "FROZEN_LIVE32_CAPACITY_CONTROL_POISON_ROLLBACK_PASS" in result.stdout
