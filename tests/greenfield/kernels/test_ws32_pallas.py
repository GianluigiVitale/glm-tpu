from __future__ import annotations

import json
import os
import subprocess
import sys


def test_ws32_pallas_matches_reference_on_forced_32_cpu_devices() -> None:
    """Prove the challenger reuses sealed [out,in] owners without transpose."""

    program = r'''
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.kernels.ws32 import (
    ws32_moe_fp8_from_routes_mapped,
    ws32_moe_pallas_from_routes_mapped,
)
from glm_tpu.greenfield.sharding.ws32 import validate_ws32_repeated_hlo

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

devices = np.asarray(jax.devices(), dtype=object).reshape(8, 4)
mesh = Mesh(devices, ("expert", "feature"))
contract = GlmMoeNumericalContract(
    hidden_size=128,
    intermediate_size=128,
    num_experts=16,
    top_k=4,
    stage_size=8,
    fp8_block_shape=(32, 32),
)
hidden = np.asarray(
    np.linspace(-0.5, 0.75, 128, dtype=np.float32)[None, :],
    dtype=ml_dtypes.bfloat16,
)
distributed_routes = np.asarray([[0, 5, 10, 15]], dtype=np.int32)
concentrated_routes = np.asarray([[4, 5, 6, 7]], dtype=np.int32)
route_weights = np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
expert_gate = bits(draw(1, (16, 128, 128)))
expert_up = bits(draw(2, (16, 128, 128)))
expert_down = bits(draw(3, (16, 128, 128)))
expert_gate_scale = scales(4, (16, 4, 4))
expert_up_scale = scales(5, (16, 4, 4))
expert_down_scale = scales(6, (16, 4, 4))
shared_gate = bits(draw(7, (128, 128)))
shared_up = bits(draw(8, (128, 128)))
shared_down = bits(draw(9, (128, 128)))
shared_gate_scale = scales(10, (4, 4))
shared_up_scale = scales(11, (4, 4))
shared_down_scale = scales(12, (4, 4))

shardings = (
    NamedSharding(mesh, P(None, "feature")),
    NamedSharding(mesh, P()),
    NamedSharding(mesh, P()),
    NamedSharding(mesh, P("expert", None, "feature")),
    NamedSharding(mesh, P("expert", None, "feature")),
    NamedSharding(mesh, P("expert", None, "feature")),
    NamedSharding(mesh, P("expert", None, "feature")),
    NamedSharding(mesh, P("expert", "feature", None)),
    NamedSharding(mesh, P("expert", "feature", None)),
    NamedSharding(mesh, P(None, "feature")),
    NamedSharding(mesh, P(None, "feature")),
    NamedSharding(mesh, P(None, "feature")),
    NamedSharding(mesh, P(None, "feature")),
    NamedSharding(mesh, P("feature", None)),
    NamedSharding(mesh, P("feature", None)),
)
in_specs = (
    P(None, "feature"), P(), P(),
    P("expert", None, "feature"), P("expert", None, "feature"),
    P("expert", None, "feature"), P("expert", None, "feature"),
    P("expert", "feature", None), P("expert", "feature", None),
    P(None, "feature"), P(None, "feature"),
    P(None, "feature"), P(None, "feature"),
    P("feature", None), P("feature", None),
)

def inputs(routes):
    values = (
        hidden, routes, route_weights,
        expert_gate, expert_gate_scale, expert_up, expert_up_scale,
        expert_down, expert_down_scale,
        shared_gate, shared_gate_scale, shared_up, shared_up_scale,
        shared_down, shared_down_scale,
    )
    return tuple(jax.device_put(value, sharding) for value, sharding in zip(values, shardings))

reference = jax.shard_map(
    lambda *values: ws32_moe_fp8_from_routes_mapped(*values, contract=contract),
    mesh=mesh,
    in_specs=in_specs,
    out_specs=P(None, "feature"),
    check_vma=False,
)
pallas = jax.shard_map(
    lambda *values: ws32_moe_pallas_from_routes_mapped(
        *values, contract=contract, interpret=True
    ),
    mesh=mesh,
    in_specs=in_specs,
    out_specs=P(None, "feature"),
    check_vma=False,
)

distributed_inputs = inputs(distributed_routes)
reference_compiled = jax.jit(reference).lower(*distributed_inputs).compile()
pallas_compiled = jax.jit(pallas).lower(*distributed_inputs).compile()
report = validate_ws32_repeated_hlo(
    pallas_compiled.as_text(),
    kind="moe",
    hidden_size=128,
    moe_intermediate_size=128,
    top_k=4,
    expected_result_dtype=None,
)
results = {}
for name, routes in (("distributed", distributed_routes), ("concentrated", concentrated_routes)):
    values = inputs(routes)
    expected = reference_compiled(*values)
    actual = pallas_compiled(*values)
    expected_bits = np.asarray(expected).view(np.uint16)
    actual_bits = np.asarray(actual).view(np.uint16)
    results[name] = {
        "bitwise_mismatches": int(np.count_nonzero(expected_bits != actual_bits)),
        "max_abs_error": float(jnp.max(jnp.abs(
            expected.astype(jnp.float32) - actual.astype(jnp.float32)
        ))),
        "output_shape": list(actual.shape),
        "output_sharding": str(actual.sharding.spec),
    }
print(json.dumps({
    "cases": results,
    "hlo": {
        "all_reduce_count": report.all_reduce_count,
        "maximum_group_size": report.maximum_group_size,
        "valid": report.valid,
        "violations": report.violations,
    },
}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    existing = environment.get("XLA_FLAGS", "").strip()
    environment["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["hlo"] == {
        "all_reduce_count": 6,
        "maximum_group_size": 8,
        "valid": True,
        "violations": [],
    }
    for case in ("distributed", "concentrated"):
        assert result["cases"][case]["output_shape"] == [1, 128]
        assert result["cases"][case]["output_sharding"] == "P(None, 'feature')"
        assert result["cases"][case]["max_abs_error"] <= 0.03125
