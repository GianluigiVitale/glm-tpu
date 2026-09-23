"""Resident projection refusals and the canonical dense row placement of the production prefill."""
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
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.optimized.bf16_resident import bf16_resident_weights, bf16_weight_specs
from glm_tpu.optimized.prefill_dense_canonical import ws32_prefill_dense_canonical_mapped as canonical
from glm_tpu.greenfield.kernels.ws32_prefill_dense_canonical import ws32_prefill_dense_canonical_mapped
from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_weight_specs
from tests.greenfield.runtime.ws32_prefill_cpu_fixture import fixture
mesh=Mesh(np.asarray(jax.devices()).reshape(8,4),('expert','feature'))
config,weights,_=fixture(mesh)
bf16=bf16_resident_weights(mesh,config,weights)
assert canonical is not ws32_prefill_dense_canonical_mapped
rng=np.random.default_rng(1926)
for rows in (114,128):
    # Late live rows expose accidentally using ordinary B128 placement.
    x=jnp.asarray(rng.normal(size=(rows,config.geometry.hidden_size)),jnp.bfloat16)
    live=jnp.arange(rows)%3!=0
    def body(x,live,raw,resident):
        expected=ws32_prefill_dense_canonical_mapped(x,live,raw.layers[0].dense,
                   moe_contract=config.moe_contract,linear_interpret=True)
        actual=canonical(x,live,resident.layers[0].dense,
                   moe_contract=config.moe_contract,linear_interpret=True)
        return expected,actual
    fn=jax.jit(jax.shard_map(body,mesh=mesh,
        in_specs=(P(None,'feature'),P(),ws32_decoder_weight_specs(config),bf16_weight_specs(config)),
        out_specs=((P(None,'feature'),P(),P(),P()),)*2,check_vma=False))
    a,b=fn(x,live,weights,bf16)
    np.testing.assert_allclose(np.asarray(a[0]).astype(np.float32),np.asarray(b[0]).astype(np.float32),rtol=.02,atol=.01)
    for aa,bb in zip(a[1:],b[1:]): np.testing.assert_array_equal(np.asarray(aa),np.asarray(bb))
    assert np.asarray(b[-1]).all()
    assert not np.asarray(b[0])[~np.asarray(live)].any()
print('BF16 canonical B114/B128 output, routing, live masks and health checked')
'''
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=32')
    p=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True,timeout=300)
    assert p.returncode==0,p.stdout+p.stderr
