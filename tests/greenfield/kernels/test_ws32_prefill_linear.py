"""Forced-32 CPU admission; no TPU or prefill performance claim."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import jax.numpy as jnp
import pytest

from glm_tpu.greenfield.kernels.ws32_prefill_linear import (
    ws32_prefill_dense_mapped,
    ws32_prefill_linear_mapped,
)


def test_prefill_linear_and_dense_match_one_row_paths_on_cpu32() -> None:
    program = r"""
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax._src.pallas.mosaic import tpu_info
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from glm_tpu.greenfield.kernels.ws32 import (
    ws32_dense_pallas_mapped, ws32_fp8_feature_linear_pallas_mapped,
    ws32_fp8_expert_linear_pallas_mapped,
)
from glm_tpu.greenfield.kernels.ws32_prefill_linear import (
    ws32_prefill_dense_mapped, ws32_prefill_linear_mapped,
)
from glm_tpu.optimized.hlo_contract import parse_hlo_module

tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
    tpu_info.ChipVersion.TPU_V4, 1
)
tpu_info.get_tpu_info.cache_clear()
assert jax.default_backend() == "cpu"
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8, 4), ("expert", "feature"))
rng = np.random.default_rng(172)
hidden = np.asarray(rng.normal(0, 0.15, (17, 128)), ml_dtypes.bfloat16)
hidden[7] = 0
def bits(shape):
    return np.asarray(rng.normal(0, 0.125, shape), ml_dtypes.float8_e4m3fn).view(np.uint8)
gate, up, down = bits((256, 128)), bits((256, 128)), bits((128, 256))
gs = rng.uniform(0.5, 1.5, (8, 4)).astype(np.float32)
us = rng.uniform(0.5, 1.5, (8, 4)).astype(np.float32)
ds = rng.uniform(0.5, 1.5, (4, 8)).astype(np.float32)
specs = (P(None, "feature"), P("expert", "feature"), P("expert", "feature"),
         P("expert", "feature"), P("expert", "feature"),
         P("feature", "expert"), P("feature", "expert"))
arrays = tuple(jax.device_put(v, NamedSharding(mesh, s))
               for v, s in zip((hidden, gate, gs, up, us, down, ds), specs))

def mapped(fn, in_specs, out_spec):
    return jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=in_specs,
                               out_specs=out_spec, check_vma=False))
def check(batch_fn, one_fn, values, expected_groups):
    compiled = batch_fn.lower(*values).compile()
    actual = compiled(*values)
    parts = [one_fn(values[0][i:i+1], *values[1:]) for i in range(17)]
    expected = jnp.concatenate(parts)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint16),
                                  np.asarray(expected).view(np.uint16))
    assert actual.shape[0] == 17 and actual.dtype == jnp.bfloat16
    module = parse_hlo_module(compiled.as_text())
    collectives = [x for x in module.instructions if x.is_collective]
    assert all(x.opcode == "all-reduce" for x in collectives)
    assert sorted(x.maximum_group_size for x in collectives) == expected_groups
    feature_groups = tuple(tuple(range(e * 4, e * 4 + 4)) for e in range(8))
    expert_groups = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))
    assert all(x.replica_groups in (feature_groups, expert_groups) for x in collectives)
    return {"shape": list(actual.shape), "groups": expected_groups}

batch = mapped(lambda *v: ws32_prefill_dense_mapped(*v, block_shape=(32,32), interpret=True),
               specs, P(None, "feature"))
one = mapped(lambda *v: ws32_dense_pallas_mapped(*v, block_shape=(32,32), interpret=True),
             specs, P(None, "feature"))
report = {"dense": check(batch, one, arrays, [4, 8])}

for axis, reference, values, linear_specs, output_spec, group in (
    ("feature", ws32_fp8_feature_linear_pallas_mapped, arrays[:3], specs[:3], P(None,"expert"), 4),
    ("expert", ws32_fp8_expert_linear_pallas_mapped,
     (jax.device_put(np.asarray(rng.normal(0,0.15,(17,256)), ml_dtypes.bfloat16),
                     NamedSharding(mesh,P(None,"expert"))), arrays[5], arrays[6]),
     (P(None,"expert"), specs[5], specs[6]), P(None,"feature"), 8),
):
    batch = mapped(lambda *v: ws32_prefill_linear_mapped(
        *v, reduction_axis=axis, block_shape=(32,32), interpret=True),
        linear_specs, output_spec)
    one = mapped(lambda *v: reference(*v, block_shape=(32,32), interpret=True),
                 linear_specs, output_spec)
    report[axis] = check(batch, one, values, [group])
print(json.dumps(report))
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = (
        environment.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32"
    ).strip()
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "dense": {"shape": [17, 128], "groups": [4, 8]},
        "feature": {"shape": [17, 256], "groups": [4]},
        "expert": {"shape": [17, 128], "groups": [8]},
    }


def test_prefill_linear_rejects_invalid_rows_dtype_and_axis() -> None:
    bits = jnp.zeros((128, 128), jnp.uint8)
    scales = jnp.ones((1, 1), jnp.float32)
    for value in (jnp.ones((128,), jnp.bfloat16), jnp.ones((0, 128), jnp.bfloat16)):
        with pytest.raises(ValueError, match="nonempty"):
            ws32_prefill_linear_mapped(value, bits, scales, reduction_axis="feature")
    with pytest.raises(ValueError, match="bfloat16"):
        ws32_prefill_linear_mapped(
            jnp.ones((8, 128), jnp.float32), bits, scales, reduction_axis="feature"
        )
    with pytest.raises(ValueError, match="feature or expert"):
        ws32_prefill_linear_mapped(
            jnp.ones((8, 128), jnp.bfloat16), bits, scales, reduction_axis="model"
        )


def test_prefill_dense_rejects_mismatched_weight_geometry() -> None:
    hidden = jnp.ones((8, 128), jnp.bfloat16)
    bits = jnp.zeros((128, 128), jnp.uint8)
    scale = jnp.ones((1, 1), jnp.float32)
    with pytest.raises(ValueError, match="gate/up"):
        ws32_prefill_dense_mapped(hidden, bits, scale, bits[:64], scale, bits, scale)
    with pytest.raises(ValueError, match="down geometry"):
        ws32_prefill_dense_mapped(hidden, bits, scale, bits, scale, bits[:64], scale)
