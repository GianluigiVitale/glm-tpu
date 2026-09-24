from __future__ import annotations

import json
import os
import subprocess
import sys


def test_final_layout_pallas_moe_matches_fallback_on_four_cpu_devices() -> None:
    """Exercise routed/shared composition and the one-collective contract."""

    program = r'''
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.pallas.fp8_matmul import Fp8BlockMatmulConfig
from glm_tpu.optimized.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.kernels.stage_local import (
    stage_local_moe_fp8_mapped,
    stage_local_moe_pallas_mapped,
)

tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
    tpu_info.ChipVersion.TPU_V4, 1
)
tpu_info.get_tpu_info.cache_clear()

def bits(value):
    return np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)

def draw(seed, shape):
    return np.random.default_rng(seed).normal(0, 0.125, shape).astype(np.float32)

def scales(seed, shape):
    return np.random.default_rng(seed).uniform(0.5, 1.25, shape).astype(np.float32)

devices = np.asarray(jax.devices(), dtype=object)
mesh = Mesh(devices, ("stage",))
rep = NamedSharding(mesh, P())
expert_sharding = NamedSharding(mesh, P("stage", None, None))
shared_out_sharding = NamedSharding(mesh, P("stage", None))
shared_down_sharding = NamedSharding(mesh, P(None, "stage"))
slot_sharding = NamedSharding(mesh, P("stage"))
contract = GlmMoeNumericalContract(
    hidden_size=8,
    intermediate_size=8,
    num_experts=16,
    top_k=4,
    stage_size=4,
    fp8_block_shape=(2, 2),
)
config = Fp8BlockMatmulConfig(
    block_shape=(2, 2),
    row_tile=8,
    output_tile=2,
    contraction_tile=2,
)

hidden = np.asarray(
    [[0.5, -0.25, 0.75, 1.0, -1.0, 0.125, 0.25, -0.5]],
    dtype=ml_dtypes.bfloat16,
)
router = np.asarray(draw(1, (16, 8)), dtype=ml_dtypes.bfloat16)
distributed_bias = np.asarray(
    [0.9, 0.0, 0.8, 0.0, 0.7, 0.0, 0.6, 0.0] + [0.0] * 8,
    dtype=np.float32,
)
concentrated_bias = np.asarray(
    [0.0] * 8 + [0.9, 0.8, 0.7, 0.6] + [0.0] * 4,
    dtype=np.float32,
)

expert_gate_nk = bits(draw(2, (16, 8, 8)))
expert_up_nk = bits(draw(3, (16, 8, 8)))
expert_down_nk = bits(draw(4, (16, 8, 8)))
expert_gate_kn = np.ascontiguousarray(np.transpose(expert_gate_nk, (0, 2, 1)))
expert_up_kn = np.ascontiguousarray(np.transpose(expert_up_nk, (0, 2, 1)))
expert_down_kn = np.ascontiguousarray(np.transpose(expert_down_nk, (0, 2, 1)))
expert_gate_scale = scales(5, (16, 4, 4))
expert_up_scale = scales(6, (16, 4, 4))
expert_down_scale = scales(7, (16, 4, 4))
shared_gate = bits(draw(8, (8, 8)))
shared_up = bits(draw(9, (8, 8)))
shared_down = bits(draw(10, (8, 8)))
shared_gate_scale = scales(11, (4, 4))
shared_up_scale = scales(12, (4, 4))
shared_down_scale = scales(13, (4, 4))
slots = np.arange(4, dtype=np.int32)

def put(value, sharding):
    return jax.device_put(jnp.asarray(value), sharding)

common = (
    put(hidden, rep),
    put(router, rep),
    put(distributed_bias, rep),
)
reference_tail = (
    put(expert_gate_nk, expert_sharding),
    put(expert_gate_scale, expert_sharding),
    put(expert_up_nk, expert_sharding),
    put(expert_up_scale, expert_sharding),
    put(expert_down_nk, expert_sharding),
    put(expert_down_scale, expert_sharding),
    put(shared_gate, shared_out_sharding),
    put(shared_gate_scale, shared_out_sharding),
    put(shared_up, shared_out_sharding),
    put(shared_up_scale, shared_out_sharding),
    put(shared_down, shared_down_sharding),
    put(shared_down_scale, shared_down_sharding),
    put(slots, slot_sharding),
)
pallas_tail = (
    put(expert_gate_kn, expert_sharding),
    put(expert_gate_scale, expert_sharding),
    put(expert_up_kn, expert_sharding),
    put(expert_up_scale, expert_sharding),
    put(expert_down_kn, expert_sharding),
    put(expert_down_scale, expert_sharding),
    *reference_tail[6:],
)
in_specs = (
    P(), P(), P(),
    P("stage", None, None), P("stage", None, None),
    P("stage", None, None), P("stage", None, None),
    P("stage", None, None), P("stage", None, None),
    P("stage", None), P("stage", None),
    P("stage", None), P("stage", None),
    P(None, "stage"), P(None, "stage"), P("stage"),
)

reference_map = jax.shard_map(
    lambda *values: stage_local_moe_fp8_mapped(
        *values[:-1],
        values[-1][0],
        axis_name="stage",
        contract=contract,
    ),
    mesh=mesh,
    in_specs=in_specs,
    out_specs=(P(), P(), P()),
    check_vma=False,
)
pallas_map = jax.shard_map(
    lambda *values: stage_local_moe_pallas_mapped(
        *values[:-1],
        values[-1][0],
        axis_name="stage",
        contract=contract,
        config=config,
        interpret=True,
    ),
    mesh=mesh,
    in_specs=in_specs,
    out_specs=(P(), P(), P()),
    check_vma=False,
)
reference_compiled = jax.jit(reference_map).lower(*(common + reference_tail)).compile()
pallas_compiled = jax.jit(pallas_map).lower(*(common + pallas_tail)).compile()

def run_case(bias):
    bias = put(bias, rep)
    reference = reference_compiled(common[0], common[1], bias, *reference_tail)
    actual = pallas_compiled(common[0], common[1], bias, *pallas_tail)
    return {
        "max_error": float(jnp.max(jnp.abs(
            actual[0].astype(jnp.float32) - reference[0].astype(jnp.float32)
        ))),
        "routes_exact": bool(jnp.array_equal(actual[1], reference[1])),
        "weights_exact": bool(jnp.array_equal(actual[2], reference[2])),
        "routes": np.asarray(actual[1]).tolist(),
    }

hlo = pallas_compiled.as_text()
print(json.dumps({
    "distributed": run_case(distributed_bias),
    "concentrated": run_case(concentrated_bias),
    "collectives": {
        "all_gather": hlo.count(" all-gather("),
        "all_reduce": hlo.count(" all-reduce("),
        "collective_permute": hlo.count(" collective-permute("),
    },
}, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=4".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    for name in ("distributed", "concentrated"):
        assert result[name]["max_error"] <= 0.03125
        assert result[name]["routes_exact"]
        assert result[name]["weights_exact"]
    assert set(result["distributed"]["routes"][0]) == {0, 2, 4, 6}
    assert set(result["concentrated"]["routes"][0]) == {8, 9, 10, 11}
    assert result["collectives"] == {
        "all_gather": 0,
        "all_reduce": 1,
        "collective_permute": 0,
    }
