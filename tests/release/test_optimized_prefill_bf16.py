"""Resident projection refusals and the canonical dense row placement of the production prefill.

Until S2f ``test_bf16_canonical_dense_cpu32`` also compared the BF16-resident canonical dense MLP with
the frozen FP8 one (output within rtol 0.02 / atol 0.01, routing and live masks equal); that oracle
is archived at ``archive/research-20260922`` and its final green run is recorded in the S2f commit
message. The production behaviour it checked stays: B114 and B128 both run, dead rows stay zero and
every owner reports healthy.
"""
import os
import subprocess
import sys

import pytest
import jax.numpy as jnp

from glm_tpu.optimized import prefill_bf16 as resident


def test_resident_projection_rejects_raw_bits_or_scales():
    lhs = jnp.ones((2,128),jnp.bfloat16)
    for weight, scale in ((jnp.ones((128,128),jnp.uint8),None),
                          (jnp.ones((128,128),jnp.bfloat16),jnp.ones((1,1),jnp.float32))):
        with pytest.raises(ValueError): resident.resident_matmul(lhs,weight,scale)


def test_bf16_canonical_dense_cpu32():
    code=r'''
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.optimized.bf16_resident import bf16_resident_weights, bf16_weight_specs
from glm_tpu.optimized.prefill_dense_canonical import ws32_prefill_dense_canonical_mapped as canonical
from tests.fixtures.tiny_model import cpu_mesh, fixture
mesh=cpu_mesh()
config,weights,_=fixture(mesh)
bf16=bf16_resident_weights(mesh,config,weights)
rng=np.random.default_rng(1926)
for rows in (114,128):
    # Late live rows expose accidentally using ordinary B128 placement.
    x=jnp.asarray(rng.normal(size=(rows,config.geometry.hidden_size)),jnp.bfloat16)
    live=jnp.arange(rows)%3!=0
    def body(x,live,resident):
        return canonical(x,live,resident.layers[0].dense,moe_contract=config.moe_contract,linear_interpret=True)
    fn=jax.jit(jax.shard_map(body,mesh=mesh,in_specs=(P(None,'feature'),P(),bf16_weight_specs(config)),
                             out_specs=(P(None,'feature'),P(),P(),P()),check_vma=False))
    b=fn(x,live,bf16)
    assert np.asarray(b[0]).shape==(rows,config.geometry.hidden_size)
    assert np.isfinite(np.asarray(b[0]).astype(np.float32)).all()
    assert np.asarray(b[-1]).all()
    assert not np.asarray(b[0])[~np.asarray(live)].any()
print('BF16 canonical B114/B128 output, live masks and health checked')
'''
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32')
    p=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,timeout=300)
    assert p.returncode==0,p.stdout+p.stderr
