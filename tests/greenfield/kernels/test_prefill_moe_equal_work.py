"""Actual forced32 JAX interfaces at B16/B128, reduced weights, CPU only."""

import os
import subprocess
import sys


def test_real_mapped_equal_work_and_metadata_interfaces():
    program = r"""
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.benchmarking.ws32_one_layer import WS32_ONE_LAYER_INPUT_SPECS
from glm_tpu.greenfield.benchmarking import REAL_LAYER_OUTPUT_TOLERANCE, compare_bounded_tensor
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from scripts.greenfield.probe_ws32_prefill_moe import build_mapped
from scripts.greenfield.prefill_moe_scaling import split_equal_work, device_active_tiles, occupancy
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend() == 'cpu' and len(jax.devices()) == 32
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8,4), ('expert','feature'))
contract = GlmMoeNumericalContract(hidden_size=128, intermediate_size=32,
    num_experts=256, top_k=8, stage_size=8, fp8_block_shape=(32,32))
rng = np.random.default_rng(937)
def bits(shape):
    return np.asarray(rng.normal(0,.1,shape), ml_dtypes.float8_e4m3fn).view(np.uint8)
def scales(shape):
    return rng.uniform(.25,1.5,shape).astype(np.float32)
weights = (bits((256,32,128)),scales((256,1,4)),bits((256,32,128)),scales((256,1,4)),
           bits((256,128,32)),scales((256,4,1)),bits((32,128)),scales((1,4)),
           bits((32,128)),scales((1,4)),bits((128,32)),scales((4,1)))
hidden = np.asarray(rng.normal(0,.15,(128,128)), ml_dtypes.bfloat16)
hidden[4] = 0
rw = rng.uniform(.01,1.,(128,8)).astype(np.float32)
rw /= rw.sum(axis=1, keepdims=True)
batch, one = build_mapped(mesh, contract=contract, interpret=True, fp32_route_sum=True)
replicated = NamedSharding(mesh, P())
metadata = jax.jit(device_active_tiles, in_shardings=replicated, out_shardings=replicated)
compiled = {}
for case in ('distributed', 'concentrated'):
    routes = np.stack([rng.choice(256,8,replace=False) if case == 'distributed'
                       else rng.permutation(np.arange(128,136)) for _ in range(128)]).astype(np.int32)
    values = tuple(jax.device_put(v, NamedSharding(mesh,s)) for v,s in zip(
        (hidden,routes,rw,*weights), WS32_ONE_LAYER_INPUT_SPECS, strict=True))
    small = split_equal_work(values)
    jax.block_until_ready((values,small))
    if not compiled:
        compiled['wide'] = batch.lower(*values).compile()
        compiled['small'] = batch.lower(*small[0]).compile()
        compiled['one'] = one.lower(*[v[:1] for v in values[:3]], *values[3:]).compile()
    wide, health = compiled['wide'](*values)
    controls = [compiled['small'](*v) for v in small]
    reference = compiled['one'](*[v[:1] for v in values[:3]], *values[3:])
    assert hasattr(reference, 'addressable_shards')  # not tuple-return API
    np.testing.assert_array_equal(np.asarray(wide).view(np.uint16),
        np.asarray(jnp.concatenate([v[0] for v in controls])).view(np.uint16))
    # The baseline intentionally uses the existing bounded M1 contract (DB583),
    # not bitwise equality to the scalar path's differently associated route sum.
    assert compare_bounded_tensor(np.asarray(wide[:1]), np.asarray(reference), REAL_LAYER_OUTPUT_TOLERANCE)['passed']
    assert np.asarray(health).all() and all(np.asarray(v[1]).all() for v in controls)
    assert all(s.data.shape == (1,1) for s in health.addressable_shards)
    for v in (values,*small):
        actual = metadata(v[1])
        jax.block_until_ready(actual)
        occupancy(np.asarray(v[1]), np.asarray(actual))
    feature = tuple(tuple(range(e*4,e*4+4)) for e in range(8))
    expert = tuple(tuple(e*4+f for e in range(8)) for f in range(4))
    for name in ('wide','small'):
        cs = [i for i in parse_hlo_module(compiled[name].as_text()).instructions if i.is_collective]
        assert sorted(i.maximum_group_size for i in cs) == [4,8]
        assert all(i.opcode == 'all-reduce' and i.replica_groups in (feature,expert) for i in cs)
print('equal128 actual JAX/sharding/metadata/scalar interfaces PASS')
"""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32"
    ).strip()
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "interfaces PASS" in result.stdout
