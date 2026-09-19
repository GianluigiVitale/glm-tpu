"""Route-grouped FP8 projection and grouped MoE: bitwise equal to the frozen bodies.

CPU interpret-mode semantics only.  No TPU timing, HLO admission or memory
claim is made here; see docs/perf/REFERENCE_LOWHANGING_FRUIT_20260919.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.pallas.fp8_matmul import (
    Fp8BlockMatmulConfig,
    fp8_block_matmul,
    fp8_block_matmul_f32,
)
from glm_tpu.perf.fp8_routed_experts import (
    RoutedProjectionConfig,
    fp8_routed_projection,
)


def _interpret_on_cpu() -> None:
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V4, 1
    )
    tpu_info.get_tpu_info.cache_clear()


def _bits(rng, shape):
    return jnp.asarray(
        np.asarray(rng.normal(0, 0.125, shape), ml_dtypes.float8_e4m3fn).view(np.uint8)
    )


def _scales(rng, shape):
    return jnp.asarray(rng.uniform(0.5, 1.25, shape), jnp.float32)


@pytest.mark.parametrize(
    "config",
    [
        dict(output_tile=0),
        dict(contraction_tile=48),
        dict(output_tile=64),
        dict(block_shape=(32, 32), contraction_tile=96),
        dict(block_shape=(32, 32), contraction_tile=512),
        dict(write_empty_slot=1),
    ],
)
def test_tile_config_refuses_non_block_geometry(config):
    with pytest.raises(ValueError):
        RoutedProjectionConfig(**config)


def test_tile_config_accepts_slab_aligned_multiples():
    for blocks in (1, 2, 4, 8):
        RoutedProjectionConfig(contraction_tile=128 * blocks, output_tile=256)
    frozen = RoutedProjectionConfig.frozen_tiles((32, 32))
    assert (frozen.output_tile, frozen.contraction_tile) == (32, 32)


@pytest.mark.parametrize('write_empty_slot',[False,True])
def test_routed_projection_all_unowned_is_all_zeros(write_empty_slot):
    _interpret_on_cpu()
    rng = np.random.default_rng(2)
    bits, scale = _bits(rng, (2, 64, 64)), _scales(rng, (2, 2, 2))
    lhs = jnp.ones((3, 64), jnp.bfloat16)
    out = fp8_routed_projection(
        lhs, ((bits, scale),), jnp.zeros((3,), jnp.int32), jnp.zeros((3,), bool),
        config=RoutedProjectionConfig(block_shape=(32,32),output_tile=32,contraction_tile=32,
                                      write_empty_slot=write_empty_slot), interpret=True,
    )
    assert out.shape == (3, 1, 64) and not np.any(np.asarray(out))


@pytest.mark.parametrize(
    ("tiles", "result_dtype"),
    [
        ((32, 32), jnp.float32),
        ((64, 64), jnp.float32),
        ((256, 128), jnp.float32),
        ((64, 32), jnp.bfloat16),
        ((128, 128), jnp.bfloat16),
    ],
)
@pytest.mark.parametrize('write_empty_slot',[False,True])
def test_routed_projection_matches_frozen_kernel_bitwise(tiles, result_dtype,write_empty_slot):
    """Owned slots equal fp8_block_matmul[_f32] bit for bit; unowned are zeros."""

    _interpret_on_cpu()
    rng = np.random.default_rng(0)
    experts, output, contraction, slots = 6, 256, 128, 8
    block = (32, 32)
    tables = tuple(
        (_bits(rng, (experts, output, contraction)), _scales(rng, (experts, 8, 4)))
        for _ in range(2)
    )
    lhs = jnp.asarray(rng.normal(0, 0.25, (slots, contraction)), jnp.bfloat16)
    ids = jnp.asarray([0, 5, 2, 5, 1, 3, 3, 4], jnp.int32)
    owned = jnp.asarray([1, 0, 1, 1, 0, 0, 1, 1], bool)
    config = RoutedProjectionConfig(
        block_shape=block, output_tile=tiles[0], contraction_tile=tiles[1],write_empty_slot=write_empty_slot
    )
    out = jax.jit(
        lambda l, t, i, o: fp8_routed_projection(
            l, t, i, o, config=config, result_dtype=result_dtype, interpret=True
        )
    )(lhs, tables, ids, owned)
    assert out.shape == (slots, 2, output) and out.dtype == result_dtype
    frozen_config = Fp8BlockMatmulConfig(
        block_shape=block, output_tile=block[0], contraction_tile=block[1]
    )
    frozen = fp8_block_matmul_f32 if result_dtype == jnp.float32 else fp8_block_matmul
    view = np.uint32 if result_dtype == jnp.float32 else np.uint16
    for slot in range(slots):
        for table, (bits, scale) in enumerate(tables):
            actual = np.asarray(out[slot, table])
            if bool(owned[slot]):
                expected = frozen(
                    lhs[slot : slot + 1], bits[int(ids[slot])], scale[int(ids[slot])],
                    config=frozen_config, interpret=True,
                )[0]
                assert np.array_equal(actual.view(view), np.asarray(expected).view(view))
            else:
                assert not np.any(actual.astype(np.float32))


def test_routed_projection_refuses_contract_drift():
    _interpret_on_cpu()
    rng = np.random.default_rng(1)
    bits, scale = _bits(rng, (2, 64, 64)), _scales(rng, (2, 2, 2))
    lhs = jnp.zeros((3, 64), jnp.bfloat16)
    ids = jnp.zeros((3,), jnp.int32)
    owned = jnp.ones((3,), bool)
    config = RoutedProjectionConfig.frozen_tiles((32, 32))
    good = dict(config=config, interpret=True)
    fp8_routed_projection(lhs, ((bits, scale),), ids, owned, **good)
    with pytest.raises(ValueError):
        fp8_routed_projection(lhs.astype(jnp.float32), ((bits, scale),), ids, owned, **good)
    with pytest.raises(ValueError):
        fp8_routed_projection(lhs, ((bits, scale[:, :1]),), ids, owned, **good)
    with pytest.raises(ValueError):
        fp8_routed_projection(lhs, ((bits, scale),), ids[:2], owned, **good)
    with pytest.raises(ValueError):
        fp8_routed_projection(lhs, ((bits, scale),), ids, owned.astype(jnp.int32), **good)
    with pytest.raises(ValueError):
        fp8_routed_projection(
            lhs, ((bits, scale),), ids, owned,
            config=RoutedProjectionConfig(block_shape=(32, 32), output_tile=96,
                                          contraction_tile=32),
            interpret=True,
        )
    with pytest.raises(ValueError):
        fp8_routed_projection(lhs, ((bits, scale),), ids, owned, result_dtype=jnp.int32, **good)


def test_grouped_moe_matches_frozen_pallas_moe_on_forced_32_cpu_devices():
    """Same fixture as tests/greenfield/kernels/test_ws32_pallas.py, plus two more
    route patterns; bitwise equality and the collective census (6 -> 2)."""

    program = r'''
import json
from collections import Counter
import jax, jax.numpy as jnp, ml_dtypes, numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.kernels.ws32 import ws32_moe_pallas_from_routes_mapped
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.perf.fp8_routed_experts import RoutedProjectionConfig, ws32_moe_grouped_routes_mapped
tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
def bits(v): return np.asarray(v, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)
def draw(seed, shape): return np.random.default_rng(seed).normal(0, .125, shape).astype(np.float32)
def scales(seed, shape): return np.random.default_rng(seed).uniform(.5, 1.25, shape).astype(np.float32)
mesh = Mesh(np.asarray(jax.devices(), dtype=object).reshape(8, 4), ("expert", "feature"))
contract = GlmMoeNumericalContract(hidden_size=128, intermediate_size=128, num_experts=16,
                                   top_k=4, stage_size=8, fp8_block_shape=(32, 32))
hidden = np.asarray(np.linspace(-.5, .75, 128, dtype=np.float32)[None, :], dtype=ml_dtypes.bfloat16)
cases = {"distributed": [[0, 5, 10, 15]], "concentrated": [[4, 5, 6, 7]],
         "two_per_owner": [[2, 3, 12, 13]], "repeated_expert": [[9, 9, 0, 1]]}
values = [hidden, None, np.asarray([[.4, .3, .2, .1]], np.float32),
          bits(draw(1, (16, 128, 128))), scales(4, (16, 4, 4)), bits(draw(2, (16, 128, 128))), scales(5, (16, 4, 4)),
          bits(draw(3, (16, 128, 128))), scales(6, (16, 4, 4)), bits(draw(7, (128, 128))), scales(10, (4, 4)),
          bits(draw(8, (128, 128))), scales(11, (4, 4)), bits(draw(9, (128, 128))), scales(12, (4, 4))]
specs = (P(None, "feature"), P(), P(), P("expert", None, "feature"), P("expert", None, "feature"),
         P("expert", None, "feature"), P("expert", None, "feature"), P("expert", "feature", None),
         P("expert", "feature", None), P(None, "feature"), P(None, "feature"), P(None, "feature"),
         P(None, "feature"), P("feature", None), P("feature", None))
def inputs(routes):
    v = list(values); v[1] = np.asarray(routes, np.int32)
    return tuple(jax.device_put(x, NamedSharding(mesh, s)) for x, s in zip(v, specs))
def program(fn, **kw):
    return jax.jit(jax.shard_map(lambda *v: fn(*v, contract=contract, interpret=True, **kw),
                                 mesh=mesh, in_specs=specs, out_specs=P(None, "feature"), check_vma=False))
frozen = program(ws32_moe_pallas_from_routes_mapped)
report = {"cases": {}, "hlo": {}}
for tiles in ((32, 32),):  # local hidden is 32 wide here; multi-block tiles are covered above
    cfg = RoutedProjectionConfig(block_shape=(32, 32), output_tile=tiles[0], contraction_tile=tiles[1])
    grouped = program(ws32_moe_grouped_routes_mapped, config=cfg)
    for name, routes in cases.items():
        a, b = frozen(*inputs(routes)), grouped(*inputs(routes))
        report["cases"][f"{tiles}/{name}"] = dict(
            bitwise_mismatches=int(np.count_nonzero(np.asarray(a).view(np.uint16) != np.asarray(b).view(np.uint16))),
            shape=list(b.shape), sharding=str(b.sharding.spec))
    if tiles == (32, 32):
        for label, fn in (("frozen", frozen), ("grouped", grouped)):
            text = fn.lower(*inputs(cases["distributed"])).compile().as_text()
            counts = Counter(x.opcode for x in parse_hlo_module(text).instructions)
            fg = tuple(tuple(range(e * 4, e * 4 + 4)) for e in range(8))
            eg = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))
            collectives = [x for x in parse_hlo_module(text).instructions if x.is_collective]
            report["hlo"][label] = dict(all_reduce=counts.get("all-reduce", 0), all_gather=counts.get("all-gather", 0),
                                        maximum_group_size=max(x.maximum_group_size for x in collectives),
                                        local_groups=all(x.replica_groups in (fg, eg) for x in collectives))
print(json.dumps(report, sort_keys=True))
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS=(os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32").strip())
    completed = subprocess.run([sys.executable, "-c", program], env=env, text=True,
                               capture_output=True, check=False, timeout=600)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    for name, case in report["cases"].items():
        assert case["bitwise_mismatches"] == 0, (name, case)
        assert case["shape"] == [1, 128] and case["sharding"] == "P(None, 'feature')"
    assert report["hlo"]["frozen"]["all_reduce"] == 6
    assert report["hlo"]["grouped"]["all_reduce"] == 2
    assert report["hlo"]["grouped"]["all_gather"] == 0
    assert report["hlo"]["grouped"]["maximum_group_size"] == 8
    # The frozen one-layer MoE HLO policy (top_k+1 feature reductions in the
    # greenfield_ws32_moe scope) does not describe the grouped body by design;
    # what must hold is that every collective stays in a feature-4/expert-8 group.
    assert report["hlo"]["grouped"]["local_groups"] and report["hlo"]["frozen"]["local_groups"]
