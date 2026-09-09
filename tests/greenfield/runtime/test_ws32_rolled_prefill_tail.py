"""Exercise static33->64 prefix padding, not33 live rows in a B128 graph."""

import os
import subprocess
import sys


def test_static33_rolled_window_near_capacity_cpu32():
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
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
wk=tuple(put(w) for w in wk)
rope=put(jnp.asarray(build_ws32_main_rope_table(config),jnp.bfloat16))
start=config.context_capacity-36
state=b.make_ws32_batched_prefill_state(mesh,config,prompt_length=start+33)
# Synthetic populated-prefix state, not a checkpoint-resume claim.
state=state._replace(decoder=state.decoder._replace(
    position=put(jnp.array([start],jnp.int32)),
    context_lengths=put(jnp.array([start+1],jnp.int32))))
opts=dict(key_tile=128,sparse_attention_interpret=True,linear_interpret=True)
wide=b.build_ws32_batched_prefill_program(mesh,config,block_rows=33,
    mlp_window=True,rolled_prefix=True,expert_panels=True,
    paired_position_sort=True,sorted_local_merge=True,**opts)
small=b.build_ws32_batched_prefill_program(mesh,config,block_rows=32,**opts)
tokens=put(jnp.arange(33,dtype=jnp.int32)+30)
actual=wide.execute(tokens,put(jnp.int32(33)),state,weights,wk,rope)
first=small.execute(tokens[:32],put(jnp.int32(32)),state,weights,wk,rope)
# Match the actual second scan tile: B32 with one live row, not a changed M1
# scorer realization. Literal M1 changes FP32 score rounding (~2.38e-7 observed).
tail=put(jnp.concatenate((tokens[32:],jnp.full(31,-2147483648,jnp.int32))))
expected=small.execute(tail,put(jnp.int32(1)),first.state,weights,wk,rope)
for a,e in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
    np.testing.assert_array_equal(a,e)
assert bool(actual.state.finished) and np.asarray(actual.state.decoder.contract_valid).all()
assert actual.state.decoder.position.tolist()==[start+33]
b.finish_ws32_batched_prefill(actual)
# Bad last live row rolls back both prefix tiles, including their cache writes.
bad=wide.execute(tokens.at[32].set(-1),put(jnp.int32(33)),state,weights,wk,rope)
assert not np.asarray(bad.state.decoder.contract_valid).any()
assert bad.next_token.tolist()==[-1]
want=state._replace(decoder=state.decoder._replace(contract_valid=put(jnp.zeros(1,jnp.bool_))))
for a,e in zip(jax.tree.leaves(bad.state),jax.tree.leaves(want)):
    np.testing.assert_array_equal(a,e)
print('STATIC33_ROLLED_NEAR_CAPACITY_PASS')
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
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "STATIC33_ROLLED_NEAR_CAPACITY_PASS" in result.stdout
