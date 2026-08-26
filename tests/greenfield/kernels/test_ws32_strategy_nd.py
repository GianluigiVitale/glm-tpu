from __future__ import annotations

import json
import os
import subprocess
import sys


def test_ws32_strategy_nd_dense_reduce_matches_full_row_with_lp8_only() -> None:
    program = r'''
import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.stage_local import (
    _strategy_nd_row0_bf16_reduce,
)
from glm_tpu.greenfield.kernels.ws32 import (
    ws32_strategy_nd_dense_down_reduce_mapped,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module

devices = np.asarray(jax.devices(), dtype=object).reshape(8, 4)
mesh = Mesh(devices, ("expert", "feature"))
rng = np.random.default_rng(20260826)
model_partials = np.asarray(
    rng.normal(0.0, 0.2, (32, 1, 6144)),
    dtype=ml_dtypes.bfloat16,
)
owner_partials = model_partials.reshape(8, 4, 1, 6144)

program = jax.shard_map(
    lambda values: ws32_strategy_nd_dense_down_reduce_mapped(values[0]),
    mesh=mesh,
    in_specs=P("expert", None, None, "feature"),
    out_specs=P(None, "feature"),
    check_vma=False,
)
sharded = jax.device_put(
    owner_partials,
    NamedSharding(mesh, P("expert", None, None, "feature")),
)
compiled = jax.jit(program).lower(sharded).compile()
actual = np.asarray(compiled(sharded))
expected = np.asarray(
    _strategy_nd_row0_bf16_reduce(jnp.asarray(model_partials))
)
module = parse_hlo_module(compiled.as_text())
collectives = [
    {
        "opcode": item.raw_opcode,
        "group": item.maximum_group_size,
        "operand_dims": [list(shape.dimensions) for shape in item.operand_shapes],
    }
    for item in module.collectives
]
print(json.dumps({
    "bitwise": bool(np.array_equal(
        actual.view(np.uint16), expected.view(np.uint16)
    )),
    "collectives": collectives,
    "output_shape": list(actual.shape),
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
    assert result["bitwise"]
    assert result["output_shape"] == [1, 6144]
    assert result["collectives"] == [
        {
            "group": 8,
            "opcode": "all-gather",
            "operand_dims": [[1, 4, 1, 1536]],
        }
    ]
