"""Diagnostic math/ownership checks on CPU, not original TPU reproduction."""

import os
import subprocess
import sys

import pytest


def test_scalar_prefix_carries_only_its_own_kv_slot():
    import jax.numpy as jnp
    from scripts.greenfield.prefill_router_boundary import scalar_prefix_inputs

    values = (
        jnp.zeros((17, 64)),
        jnp.zeros((17, 64)),
        jnp.zeros((8, 2, 64, 640)),
        jnp.zeros((8, 2, 64, 128)),
        jnp.zeros((8, 2, 64, 128)),
        jnp.zeros((17, 2048)),
        jnp.ones((17,)),
        jnp.zeros((17, 2048)),
        jnp.asarray(505),
        jnp.asarray(17),
        jnp.asarray([[1, 0]]),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        jnp.ones((8, 4, 17), jnp.bool_),
        jnp.zeros((17, 64)),
    )
    previous = [None] * 12
    previous[10] = jnp.ones_like(values[2])
    one = scalar_prefix_inputs(values, 8, tuple(previous))
    assert one[2] is previous[10]
    assert one[3] is values[3] and one[4] is values[4]
    assert int(one[8]) == 513 and one[0].shape == (1, 64)
    assert scalar_prefix_inputs(values, 0)[2] is values[2]
    previous[10] = jnp.ones((17, 256))
    with pytest.raises(ValueError, match="cache result"):
        scalar_prefix_inputs(values, 8, tuple(previous))


def test_router_boundary_reuses_both_router_arithmetic_on_cpu32():
    code = r"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh,PartitionSpec as P
from scripts.greenfield.prefill_router_boundary import build_router_replay_program
from glm_tpu.greenfield.kernels.ws32 import ws32_router_from_shards_mapped
from glm_tpu.greenfield.kernels.ws32_prefill_layer import ws32_prefill_router_mapped
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
rng=np.random.RandomState(1907)
hidden=jnp.asarray(rng.normal(size=(17,64)),jnp.bfloat16)
weight=jnp.asarray(rng.normal(size=(256,64))*.1,jnp.bfloat16)
bias=jnp.asarray(rng.normal(size=256)*.05,jnp.float32)
live=jnp.arange(17)<11
hidden=hidden.at[11:].set(jnp.nan)
diag=build_router_replay_program(mesh)
actual=jax.jit(diag)(hidden,weight,bias,live)
batch=jax.shard_map(lambda h,w,b,l:ws32_prefill_router_mapped(h,w,b,l)[:2],mesh=mesh,in_specs=(P(None,'feature'),P('expert','feature'),P('expert'),P()),out_specs=(P(),P()),check_vma=False)
ids,weights=jax.jit(batch)(hidden,weight,bias,live)
np.testing.assert_array_equal(actual[5],ids)
np.testing.assert_array_equal(actual[6],weights)
assert np.isfinite(np.asarray(actual[0])).all() and not np.any(np.asarray(actual[6])[11:])
np.testing.assert_array_equal(actual[3],bias)
partial=np.asarray(actual[1])
assert partial.shape==(8,4,17,32)
summed=partial.sum(axis=1).transpose(1,0,2).reshape(17,256)
np.testing.assert_allclose(actual[2],summed,atol=2e-6,rtol=2e-6)
scalar=jax.shard_map(lambda h,w,b:ws32_router_from_shards_mapped(h,w,b,top_k=8),mesh=mesh,in_specs=(P(None,'feature'),P('expert','feature'),P('expert')),out_specs=(P(),P()),check_vma=False)
one=hidden[4:5]
d=jax.jit(diag)(one,weight,bias,jnp.ones((1,),jnp.bool_))
ids,weights=jax.jit(scalar)(one,weight,bias)
np.testing.assert_array_equal(d[5],ids)
np.testing.assert_array_equal(d[6],weights)
print('ROUTER_BOUNDARY_CPU32_PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=90,
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ROUTER_BOUNDARY_CPU32_PASS" in result.stdout
