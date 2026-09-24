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

from glm_tpu.optimized.reference.moe import GlmMoeNumericalContract
from glm_tpu.greenfield.kernels.ws32 import (
    ws32_moe_fp8_from_routes_mapped,
    ws32_moe_pallas_from_routes_mapped,
)
from glm_tpu.optimized.mesh import validate_ws32_repeated_hlo

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


def test_ws32_complete_decoder_primitives_match_forced_32_references() -> None:
    program = r'''
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.optimized.reference.moe import route_glm_noaux_tc_logits
from glm_tpu.greenfield.kernels.ws32 import (
    ws32_dense_fp8_mapped,
    ws32_dense_pallas_mapped,
    ws32_rms_norm_mapped,
    ws32_router_from_shards_mapped,
)
from glm_tpu.optimized.hlo_contract import parse_hlo_module
from glm_tpu.optimized.mesh import validate_ws32_repeated_hlo

tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
    tpu_info.ChipVersion.TPU_V4, 1
)
tpu_info.get_tpu_info.cache_clear()

def bits(value):
    return np.asarray(value, dtype=ml_dtypes.float8_e4m3fn).view(np.uint8)

def draw(seed, shape, scale=0.125):
    return np.random.default_rng(seed).normal(0, scale, shape).astype(np.float32)

devices = np.asarray(jax.devices(), dtype=object).reshape(8, 4)
mesh = Mesh(devices, ("expert", "feature"))
hidden = np.asarray(draw(1, (1, 128), 0.25), dtype=ml_dtypes.bfloat16)
hidden_sharding = NamedSharding(mesh, P(None, "feature"))

dense_size = 256
block = (32, 32)
gate = bits(draw(2, (dense_size, 128)))
up = bits(draw(3, (dense_size, 128)))
down = bits(draw(4, (128, dense_size)))
gate_scale = np.random.default_rng(5).uniform(0.5, 1.25, (8, 4)).astype(np.float32)
up_scale = np.random.default_rng(6).uniform(0.5, 1.25, (8, 4)).astype(np.float32)
down_scale = np.random.default_rng(7).uniform(0.5, 1.25, (4, 8)).astype(np.float32)
dense_specs = (
    P(None, "feature"),
    P("expert", "feature"), P("expert", "feature"),
    P("expert", "feature"), P("expert", "feature"),
    P("feature", "expert"), P("feature", "expert"),
)
dense_shardings = tuple(NamedSharding(mesh, spec) for spec in dense_specs)
dense_values = tuple(
    jax.device_put(value, sharding)
    for value, sharding in zip(
        (hidden, gate, gate_scale, up, up_scale, down, down_scale),
        dense_shardings,
    )
)
readable_dense = jax.shard_map(
    lambda *values: ws32_dense_fp8_mapped(*values, block_shape=block),
    mesh=mesh,
    in_specs=dense_specs,
    out_specs=P(None, "feature"),
    check_vma=False,
)
pallas_dense = jax.shard_map(
    lambda *values: ws32_dense_pallas_mapped(
        *values, block_shape=block, interpret=True
    ),
    mesh=mesh,
    in_specs=dense_specs,
    out_specs=P(None, "feature"),
    check_vma=False,
)
readable_compiled = jax.jit(readable_dense).lower(*dense_values).compile()
pallas_compiled = jax.jit(pallas_dense).lower(*dense_values).compile()
expected_dense = readable_compiled(*dense_values)
actual_dense = pallas_compiled(*dense_values)
dense_contract = validate_ws32_repeated_hlo(
    pallas_compiled.as_text(),
    kind="dense",
    hidden_size=128,
    dense_intermediate_size=dense_size,
)

norm_weight = np.asarray(
    np.random.default_rng(8).uniform(0.75, 1.25, (128,)),
    dtype=ml_dtypes.bfloat16,
)
norm_map = jax.shard_map(
    lambda value, weight: ws32_rms_norm_mapped(
        value, weight, global_hidden_size=128
    ),
    mesh=mesh,
    in_specs=(P(None, "feature"), P("feature")),
    out_specs=P(None, "feature"),
    check_vma=False,
)
norm_values = (
    jax.device_put(hidden, hidden_sharding),
    jax.device_put(norm_weight, NamedSharding(mesh, P("feature"))),
)
norm_compiled = jax.jit(norm_map).lower(*norm_values).compile()
actual_norm = norm_compiled(*norm_values)
hidden_f32 = jnp.asarray(hidden).astype(jnp.float32)
inverse = lax.rsqrt(jnp.sum(lax.square(hidden_f32), axis=-1, keepdims=True)
                    / jnp.float32(128) + jnp.float32(1e-5))
expected_norm = (
    (hidden_f32 * inverse).astype(jnp.bfloat16)
    * jnp.asarray(norm_weight)
).astype(jnp.bfloat16)

num_experts = 16
router_weight = np.asarray(draw(9, (num_experts, 128), 0.2), dtype=ml_dtypes.bfloat16)
router_bias = draw(10, (num_experts,), 0.01)
router_map = jax.shard_map(
    lambda value, weight, bias: ws32_router_from_shards_mapped(
        value, weight, bias, top_k=4
    ),
    mesh=mesh,
    in_specs=(P(None, "feature"), P("expert", "feature"), P("expert")),
    out_specs=(P(), P()),
    check_vma=False,
)
router_values = (
    jax.device_put(hidden, hidden_sharding),
    jax.device_put(router_weight, NamedSharding(mesh, P("expert", "feature"))),
    jax.device_put(router_bias, NamedSharding(mesh, P("expert"))),
)
router_compiled = jax.jit(router_map).lower(*router_values).compile()
actual_indices, actual_weights = router_compiled(*router_values)
hidden_chunks = jnp.asarray(hidden).astype(jnp.float32).reshape(1, 4, 32)
weight_chunks = jnp.asarray(router_weight).astype(jnp.float32).reshape(8, 2, 4, 32)
logit_rows = []
for expert_row in range(8):
    partials = []
    for feature in range(4):
        partials.append(lax.dot_general(
            hidden_chunks[:, feature, :],
            weight_chunks[expert_row, :, feature, :],
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        ))
    logit_rows.append(jnp.sum(jnp.stack(partials), axis=0, dtype=jnp.float32))
expected_indices, expected_weights = route_glm_noaux_tc_logits(
    jnp.concatenate(logit_rows, axis=1), jnp.asarray(router_bias), top_k=4
)

def collective_summary(text):
    module = parse_hlo_module(text)
    return {
        "count": len(module.collectives),
        "maximum_group_size": max(
            (item.maximum_group_size for item in module.collectives), default=0
        ),
        "opcodes": sorted(item.raw_opcode for item in module.collectives),
    }

print(json.dumps({
    "dense": {
        "max_abs": float(jnp.max(jnp.abs(
            actual_dense.astype(jnp.float32) - expected_dense.astype(jnp.float32)
        ))),
        "sharding": str(actual_dense.sharding.spec),
        "contract_valid": dense_contract.valid,
        "violations": dense_contract.violations,
        "collectives": collective_summary(pallas_compiled.as_text()),
    },
    "norm": {
        "max_abs": float(jnp.max(jnp.abs(
            actual_norm.astype(jnp.float32) - expected_norm.astype(jnp.float32)
        ))),
        "sharding": str(actual_norm.sharding.spec),
        "collectives": collective_summary(norm_compiled.as_text()),
    },
    "router": {
        "indices_equal": bool(np.array_equal(
            np.asarray(actual_indices), np.asarray(expected_indices)
        )),
        "weights_max_abs": float(jnp.max(jnp.abs(
            actual_weights - expected_weights
        ))),
        "index_sharding": str(actual_indices.sharding.spec),
        "weight_sharding": str(actual_weights.sharding.spec),
        "collectives": collective_summary(router_compiled.as_text()),
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
    assert result["dense"]["max_abs"] <= 0.03125
    assert result["dense"]["sharding"] == "P(None, 'feature')"
    assert result["dense"]["contract_valid"], result["dense"]["violations"]
    assert result["dense"]["collectives"]["count"] == 2
    assert result["norm"]["max_abs"] <= 0.015625
    assert result["norm"]["sharding"] == "P(None, 'feature')"
    assert result["norm"]["collectives"]["count"] == 1
    assert result["router"]["indices_equal"]
    assert result["router"]["weights_max_abs"] == 0.0
    assert result["router"]["index_sharding"] == "P()"
    assert result["router"]["weight_sharding"] == "P()"
    assert result["router"]["collectives"]["count"] == 3
    for section in ("dense", "norm", "router"):
        assert result[section]["collectives"]["maximum_group_size"] <= 8
        assert not any(
            opcode.endswith(("-start", "-done"))
            for opcode in result[section]["collectives"]["opcodes"]
        )
