"""The production prefill program on the CPU mesh: health, its one-pass selector, atomic refusal of a
finished state and exclusive state donation.

Until S2f this test also compared every leaf with the independent frozen FP8 prefill program (raw
path bitwise, BF16-resident path within rtol 0.02 / atol 0.0625). That oracle is archived at
``archive/research-20260922``; its final green run is recorded in the S2f commit message, and the
reference model (``tests/reference``, ``VALIDATION.md``) plus the G1/G3 goldens carry the evidence.
"""

import os
import subprocess
import sys


CODE = r"""
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.layers.attention import dsa_indexer  # the production DSA selects through its one-pass selector
seen=[]
one_pass=dsa_indexer.prefill_dsa_one_pass
def observed_selector(*args,**kwargs):
    seen.append(True)
    return one_pass(*args,**kwargs)
dsa_indexer.prefill_dsa_one_pass=observed_selector
from glm_tpu.models.glm_moe_dsa.prefill import build_prefill_program
from glm_tpu.runner.programs import build_program_set
from tests.fixtures.tiny_model import cpu_mesh, engine_inputs
mesh=cpu_mesh()
def put(v): return jax.device_put(v,NamedSharding(mesh,P()))
inputs=engine_inputs(mesh,panel_geometry=True)
config=inputs.config
initialize=build_program_set(mesh,config,interpret=True).cache_init.fn
# the production program hard-wires the admitted profile (window, rolled prefix, panels, canonical
# dense, one-pass selector); only the block size and the interpret flags are arguments
program=build_prefill_program(mesh,config,block_rows=128,sparse_attention_interpret=True,linear_interpret=True)
tokens,count=put(jnp.array([30,31,32]+[-1]*125,jnp.int32)),put(jnp.int32(3))
y=program.execute(tokens,count,initialize(put(np.int32(3))),inputs.weights,inputs.wk,inputs.rope)
assert bool(np.asarray(y.state.decoder.contract_valid).all())
assert bool(np.asarray(y.state.finished))
assert int(np.asarray(y.next_token)[0])>=0
assert len(seen) >= len(config.full_index_slots), 'the program did not use the one-pass selector'
# An already-finished state must refuse another append atomically.
committed=y.state
z=program.execute(put(jnp.array([33]+[-1]*127,jnp.int32)),put(jnp.int32(1)),committed,inputs.weights,inputs.wk,
                  inputs.rope)
for field in ('kv_cache_local','index_cache_local','position','context_lengths'):
    np.testing.assert_array_equal(np.asarray(getattr(committed.decoder,field)),np.asarray(getattr(z.state.decoder,field)))
assert not bool(np.asarray(z.state.decoder.contract_valid).all())
assert np.asarray(z.next_token).tolist()==[-1]
print('complete prefill: healthy, one-pass selector, atomic refusal of a finished state')
# Long-context ownership uses the SAME body with exclusive state donation.
owned=jax.jit(program.execute,donate_argnums=(2,))
plain=program.execute(tokens,count,initialize(put(np.int32(3))),inputs.weights,inputs.wk,inputs.rope)
donated=owned(tokens,count,initialize(put(np.int32(3))),inputs.weights,inputs.wk,inputs.rope)
jax.block_until_ready((plain,donated))
for expected,actual in zip(jax.tree.leaves(plain),jax.tree.leaves(donated)):
    np.testing.assert_array_equal(np.asarray(expected),np.asarray(actual))
print('Exclusive prefill state donation preserves every output leaf')
"""


def test_prefill_program_health_selector_refusal_and_donation_cpu32():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
