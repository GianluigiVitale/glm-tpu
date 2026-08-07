from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    (
        "routed_output_tile",
        "fuse_route_weighting",
        "reconstruct_down_fp32",
    ),
    (
        (128, False, False),
        (256, False, False),
        (256, True, False),
        (256, False, True),
    ),
)
def test_feature_sharded_pallas_moe_matches_complete_expert_reference(
    routed_output_tile: int,
    fuse_route_weighting: bool,
    reconstruct_down_fp32: bool,
) -> None:
    """Prove balanced all-route feature shards and one local combine."""

    program = r'''
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.pallas import Fp8BlockMatmulConfig
from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.kernels.stage_local import (
    stage_local_moe_fp8_mapped,
    stage_local_moe_pallas_feature_mapped,
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
feature_up_sharding = NamedSharding(mesh, P(None, None, "stage"))
feature_up_scale_sharding = NamedSharding(mesh, P(None, "stage", None))
feature_down_sharding = NamedSharding(mesh, P(None, "stage", None))
feature_down_scale_sharding = NamedSharding(mesh, P(None, None, "stage"))
shared_out_sharding = NamedSharding(mesh, P("stage", None))
shared_down_sharding = NamedSharding(mesh, P(None, "stage"))
slot_sharding = NamedSharding(mesh, P("stage"))
contract = GlmMoeNumericalContract(
    hidden_size=256,
    intermediate_size=1024,
    num_experts=16,
    top_k=4,
    stage_size=4,
    fp8_block_shape=(128, 128),
)
config = Fp8BlockMatmulConfig(
    block_shape=(128, 128),
    row_tile=8,
    output_tile=__ROUTED_OUTPUT_TILE__,
    contraction_tile=128,
)

hidden = np.asarray(
    np.linspace(-0.5, 0.75, 256, dtype=np.float32)[None, :],
    dtype=ml_dtypes.bfloat16,
)
router = np.asarray(draw(1, (16, 256)), dtype=ml_dtypes.bfloat16)
distributed_bias = np.asarray(
    [0.9, 0.0, 0.8, 0.0, 0.7, 0.0, 0.6, 0.0] + [0.0] * 8,
    dtype=np.float32,
)
concentrated_bias = np.asarray(
    [0.0] * 8 + [0.9, 0.8, 0.7, 0.6] + [0.0] * 4,
    dtype=np.float32,
)

expert_gate_nk = bits(draw(2, (16, 1024, 256)))
expert_up_nk = bits(draw(3, (16, 1024, 256)))
expert_down_nk = bits(draw(4, (16, 256, 1024)))
expert_gate_kn = np.ascontiguousarray(np.transpose(expert_gate_nk, (0, 2, 1)))
expert_up_kn = np.ascontiguousarray(np.transpose(expert_up_nk, (0, 2, 1)))
expert_down_kn = np.ascontiguousarray(np.transpose(expert_down_nk, (0, 2, 1)))
expert_gate_scale = scales(5, (16, 8, 2))
expert_up_scale = scales(6, (16, 8, 2))
expert_down_scale = scales(7, (16, 2, 8))
shared_gate = bits(draw(8, (1024, 256)))
shared_up = bits(draw(9, (1024, 256)))
shared_down = bits(draw(10, (256, 1024)))
shared_gate_scale = scales(11, (8, 2))
shared_up_scale = scales(12, (8, 2))
shared_down_scale = scales(13, (2, 8))
slots = np.arange(4, dtype=np.int32)

def put(value, sharding):
    return jax.device_put(jnp.asarray(value), sharding)

common = (put(hidden, rep), put(router, rep))
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
feature_tail = (
    put(expert_gate_kn, feature_up_sharding),
    put(expert_gate_scale, feature_up_scale_sharding),
    put(expert_up_kn, feature_up_sharding),
    put(expert_up_scale, feature_up_scale_sharding),
    put(expert_down_kn, feature_down_sharding),
    put(expert_down_scale, feature_down_scale_sharding),
    *reference_tail[6:],
)
reference_specs = (
    P(), P(), P(),
    P("stage", None, None), P("stage", None, None),
    P("stage", None, None), P("stage", None, None),
    P("stage", None, None), P("stage", None, None),
    P("stage", None), P("stage", None),
    P("stage", None), P("stage", None),
    P(None, "stage"), P(None, "stage"), P("stage"),
)
feature_specs = (
    P(), P(), P(),
    P(None, None, "stage"), P(None, "stage", None),
    P(None, None, "stage"), P(None, "stage", None),
    P(None, "stage", None), P(None, None, "stage"),
    P("stage", None), P("stage", None),
    P("stage", None), P("stage", None),
    P(None, "stage"), P(None, "stage"), P("stage"),
)

reference_map = jax.shard_map(
    lambda *values: stage_local_moe_fp8_mapped(
        *values[:-1], values[-1][0], axis_name="stage", contract=contract
    ),
    mesh=mesh,
    in_specs=reference_specs,
    out_specs=(P(), P(), P()),
    check_vma=False,
)
feature_map = jax.shard_map(
    lambda *values: stage_local_moe_pallas_feature_mapped(
        *values[:-1],
        values[-1][0],
        axis_name="stage",
        contract=contract,
        config=config,
        fuse_route_weighting=__FUSE_ROUTE_WEIGHTING__,
        reconstruct_down_fp32=__RECONSTRUCT_DOWN_FP32__,
        interpret=True,
    ),
    mesh=mesh,
    in_specs=feature_specs,
    out_specs=(P(), P(), P()),
    check_vma=False,
)

def run_case(bias):
    bias = put(bias, rep)
    reference = jax.jit(reference_map)(
        common[0], common[1], bias, *reference_tail
    )
    actual = jax.jit(feature_map)(
        common[0], common[1], bias, *feature_tail
    )
    return {
        "max_error": float(jnp.max(jnp.abs(
            actual[0].astype(jnp.float32) - reference[0].astype(jnp.float32)
        ))),
        "routes_exact": bool(jnp.array_equal(actual[1], reference[1])),
        "weights_exact": bool(jnp.array_equal(actual[2], reference[2])),
        "routes": np.asarray(actual[1]).tolist(),
    }

lowered = jax.jit(feature_map).lower(
    common[0], common[1], put(distributed_bias, rep), *feature_tail
)
hlo = lowered.compile().as_text()
all_reduce_lines = [
    line.strip() for line in hlo.splitlines() if " all-reduce(" in line
]
print(json.dumps({
    "distributed": run_case(distributed_bias),
    "concentrated": run_case(concentrated_bias),
    "collectives": {
        "all_gather": hlo.count(" all-gather("),
        "all_reduce": hlo.count(" all-reduce("),
        "collective_permute": hlo.count(" collective-permute("),
    },
    "all_reduce_lines": all_reduce_lines,
}, sort_keys=True))
'''
    program = program.replace(
        "__ROUTED_OUTPUT_TILE__", str(routed_output_tile)
    )
    program = program.replace(
        "__FUSE_ROUTE_WEIGHTING__", repr(fuse_route_weighting)
    )
    program = program.replace(
        "__RECONSTRUCT_DOWN_FP32__", repr(reconstruct_down_fp32)
    )
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
        assert result[name]["max_error"] <= 0.0625
        assert result[name]["routes_exact"]
        assert result[name]["weights_exact"]
    assert set(result["distributed"]["routes"][0]) == {0, 2, 4, 6}
    assert set(result["concentrated"]["routes"][0]) == {8, 9, 10, 11}
    assert result["collectives"] == {
        "all_gather": 0,
        "all_reduce": 2 if reconstruct_down_fp32 else 1,
        "collective_permute": 0,
    }
    if reconstruct_down_fp32:
        assert any(
            "f32[4,256]" in line
            for line in result["all_reduce_lines"]
        ), result["all_reduce_lines"]
        assert not any(
            "bf16[4,256]" in line
            for line in result["all_reduce_lines"]
        ), result["all_reduce_lines"]
