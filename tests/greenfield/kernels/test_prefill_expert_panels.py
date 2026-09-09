"""CPU mechanism admission; not TPU allocation, arithmetic or speed proof."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.pallas.mosaic import tpu_info

from glm_tpu.greenfield.kernels.prefill_expert_panels import (
    build_expert_panels,
    pack_expert_panel_rows,
    unpack_expert_panel_rows,
)
from glm_tpu.greenfield.kernels.pallas.prefill_panel_fp8 import prefill_panel_fp8_matmul
from glm_tpu.greenfield.kernels.pallas.prefill_grouped_fp8 import (
    prefill_grouped_fp8_matmul,
)


@pytest.fixture(autouse=True)
def cpu_tpu_interpretation(monkeypatch):
    assert jax.default_backend() == "cpu"
    monkeypatch.setitem(
        tpu_info.registry,
        "cpu",
        lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1),
    )
    tpu_info.get_tpu_info.cache_clear()
    yield
    tpu_info.get_tpu_info.cache_clear()


@pytest.mark.parametrize("offset", [0, 2, 4])
def test_exclusive_panels_restore_every_owned_row_once(offset):
    counts = np.array([1, 31, 32, 33, 0, 7], np.int32)
    m = int(counts.sum())
    x = jnp.arange(m * 3, dtype=jnp.float32).reshape(m, 3)
    plan = jax.jit(lambda c, o: build_expert_panels(c, o, rows=m, local_groups=2))(
        jnp.asarray(counts), jnp.int32(offset)
    )
    assert bool(plan.valid)
    assert int(plan.active_panels) == sum(
        (int(c) + 31) // 32 for c in counts[offset : offset + 2]
    )
    packed = pack_expert_panel_rows(x, plan)
    live = np.arange(32)[None, :] < np.asarray(plan.live_counts)[:, None]
    np.testing.assert_array_equal(np.asarray(packed)[~live], 0)
    owned = (np.repeat(np.arange(6), counts) >= offset) & (
        np.repeat(np.arange(6), counts) < offset + 2
    )
    indices = np.asarray(plan.restore_indices)[owned]
    assert len(np.unique(indices)) == int(counts[offset : offset + 2].sum())
    assert live.reshape(-1)[indices].all()
    poisoned = jnp.where(jnp.asarray(live)[..., None], packed, jnp.nan)
    expected = np.where(owned[:, None], np.asarray(x), 0)
    np.testing.assert_array_equal(unpack_expert_panel_rows(poisoned, plan), expected)


def test_invalid_metadata_and_empty_owner_skip_all_work():
    m = 19
    x = jnp.full((m, 128), jnp.nan, jnp.bfloat16)
    w = jnp.full((2, 256, 128), 127, jnp.uint8)  # FP8 NaNs
    scales = jnp.full((2, 2, 1), jnp.nan, jnp.float32)
    base = jnp.array([19, 0, 0, 0], jnp.int32)

    @jax.jit
    def run(c, o):
        p = build_expert_panels(c, o, rows=m, local_groups=2)
        return prefill_panel_fp8_matmul(x, w, scales, p, interpret=True)

    value, valid = run(base, jnp.int32(2))
    assert bool(valid)
    np.testing.assert_array_equal(value, 0)
    for c, o in (
        (base, -1),
        (base, 3),
        (base, 2147483647),
        (base.at[1].set(-1), 0),
        (base.at[0].set(18), 0),
        (base.at[0].set(2147483647), 0),
    ):
        value, valid = run(c, jnp.int32(o))
        assert not bool(valid)
        np.testing.assert_array_equal(value, 0)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_full_k_scales_and_n128_stripes_match_original(dtype):
    # Ninth K block catches accidental reuse of _scale_value's ki%8 slab.
    rng = np.random.default_rng(911)
    counts = jnp.array([1, 33, 7, 0], jnp.int32)
    m, k, n = 41, 1152, 512
    x = jnp.asarray(rng.normal(0, 0.2, (m, k)), jnp.bfloat16)
    bits = lax.bitcast_convert_type(
        jnp.asarray(rng.normal(0, 0.15, (2, n, k)), jnp.float8_e4m3fn), jnp.uint8
    )
    scales = jnp.asarray(rng.uniform(0.1, 2.0, (2, 4, 9)), jnp.float32)
    scales = scales.at[0, 1, 8].set(0).at[1, 3, 8].set(3.25)

    @jax.jit
    def compare(x, bits, scales):
        plan = build_expert_panels(counts, jnp.int32(1), rows=m, local_groups=2)
        new, ok = prefill_panel_fp8_matmul(
            x, bits, scales, plan, result_dtype=dtype, interpret=True
        )
        old, old_ok = prefill_grouped_fp8_matmul(
            x, bits, scales, counts, jnp.int32(1), result_dtype=dtype, interpret=True
        )
        return new, old, ok & old_ok

    new, old, ok = compare(x, bits, scales)
    assert bool(ok)
    view = np.uint32 if dtype == jnp.float32 else np.uint16
    np.testing.assert_array_equal(
        np.asarray(new).view(view), np.asarray(old).view(view)
    )


def test_all_rows_one_expert_capacity_and_scale_zero():
    counts = jnp.array([0, 129, 0, 0], jnp.int32)
    plan = build_expert_panels(counts, jnp.int32(0), rows=129, local_groups=4)
    assert int(plan.active_panels) == 5
    np.testing.assert_array_equal(np.asarray(plan.live_counts)[:5], [32, 32, 32, 32, 1])
    x = jnp.ones((129, 128), jnp.bfloat16)
    w = jnp.full((4, 256, 128), 56, jnp.uint8)  # 1.0
    s = jnp.ones((4, 2, 1), jnp.float32).at[1, 1, 0].set(0)
    value, ok = jax.jit(
        lambda x: prefill_panel_fp8_matmul(x, w, s, plan, interpret=True)
    )(x)
    assert bool(ok)
    np.testing.assert_array_equal(np.asarray(value)[:, :128], 128)
    np.testing.assert_array_equal(np.asarray(value)[:, 128:], 0)


def test_bad_static_geometry_refuses():
    c = jnp.array([1, 0], jnp.int32)
    for offset in (jnp.array([0], jnp.int32), jnp.float32(0)):
        with pytest.raises(ValueError):
            build_expert_panels(c, offset, rows=1, local_groups=2)
    with pytest.raises(ValueError):
        build_expert_panels(c, jnp.int32(0), rows=True, local_groups=2)


def test_mapped_moe_preserves_route_sum_feature_groups_and_empty_owners():
    import os
    import subprocess
    import sys

    code = r"""
import jax, jax.numpy as jnp, numpy as np, ml_dtypes
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from glm_tpu.greenfield.kernels.ws32_prefill_moe import ws32_prefill_moe_from_routes_mapped
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
assert jax.default_backend()=='cpu'
tpu_info.registry['cpu']=lambda:tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4,1)
tpu_info.get_tpu_info.cache_clear()
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
c=GlmMoeNumericalContract(hidden_size=1024,intermediate_size=256,num_experts=16,
                        top_k=2,stage_size=8,fp8_block_shape=(128,128))
rng=np.random.default_rng(913)
B=17
def bits(shape):return np.asarray(rng.normal(0,.07,shape),ml_dtypes.float8_e4m3fn).view(np.uint8)
def scales(shape):return rng.uniform(.5,1.5,shape).astype(np.float32)
weights=(bits((16,256,1024)),scales((16,2,8)),bits((16,256,1024)),scales((16,2,8)),
         bits((16,1024,256)),scales((16,8,2)),bits((256,1024)),scales((2,8)),
         bits((256,1024)),scales((2,8)),bits((1024,256)),scales((8,2)))
specs=(P(None,'feature'),P(),P(),P('expert',None,'feature'),P('expert',None,'feature'),
       P('expert',None,'feature'),P('expert',None,'feature'),P('expert','feature',None),
       P('expert','feature',None),P(None,'feature'),P(None,'feature'),P(None,'feature'),
       P(None,'feature'),P('feature',None),P('feature',None))
def build(flag):
 def mapped(*args):
  y,ok=ws32_prefill_moe_from_routes_mapped(*args,contract=c,interpret=True,
                                         fp32_route_sum=True,expert_panels=flag)
  return y,ok[None,None]
 return jax.jit(jax.shard_map(mapped,mesh=mesh,in_specs=specs,
                out_specs=(P(None,'feature'),P('expert','feature')),check_vma=False))
old,new=build(False),build(True)
x=np.asarray(rng.normal(0,.1,(B,1024)),ml_dtypes.bfloat16)
for concentrated in (False,True):
 routes=np.tile(np.array([6,7],np.int32),(B,1)) if concentrated else np.stack([rng.choice(16,2,replace=False) for _ in range(B)]).astype(np.int32)
 rw=rng.uniform(.1,1,(B,2)).astype(np.float32);rw/=rw.sum(axis=1,keepdims=True)
 rw[3]=0
 args=tuple(jax.device_put(v,NamedSharding(mesh,p)) for v,p in zip((x,routes,rw,*weights),specs))
 expected,eo=old(*args)
 compiled=new.lower(*args).compile()
 actual,ok=compiled(*args)
 assert np.asarray(eo).all() and np.asarray(ok).all()
 np.testing.assert_array_equal(np.asarray(actual).view(np.uint16),np.asarray(expected).view(np.uint16))
 hlo=parse_hlo_module(compiled.as_text())
 cs=[i for i in hlo.instructions if i.is_collective]
 feature=tuple(tuple(range(e*4,e*4+4)) for e in range(8))
 expert=tuple(tuple(e*4+f for e in range(8)) for f in range(4))
 assert cs and all(i.opcode=='all-reduce' and i.replica_groups in (feature,expert) for i in cs)
 bad=list(args);bad[2]=bad[2].at[0,0].set(-1)
 _,ok=new(*bad);assert not np.asarray(ok).any()
print('PANEL_MOE_CPU32_PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PANEL_MOE_CPU32_PASS" in result.stdout
