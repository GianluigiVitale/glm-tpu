from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp

from glm_tpu.greenfield.kernels.stage_local import (
    _virtual_dense_final_layout_convolution_down_partials,
)


def test_ws32_strategy_nd_four_rank_partial_geometry() -> None:
    shape = jax.ShapeDtypeStruct
    result = jax.eval_shape(
        lambda hidden, merged_bits, merged_scale, down_bits, down_scale: (
            _virtual_dense_final_layout_convolution_down_partials(
                hidden,
                merged_bits,
                merged_scale,
                down_bits,
                down_scale,
                block_shape=(128, 128),
                compile_rows=1,
                virtual_shards=4,
                output_size=1536,
            )
        ),
        shape((1, 6144), jnp.bfloat16),
        shape((4, 6144, 768), jnp.uint8),
        shape((4, 48, 768), jnp.float32),
        shape((4, 384, 1536), jnp.uint8),
        shape((4, 3, 1536), jnp.float32),
    )
    assert result.shape == (4, 1, 1536)
    assert result.dtype == jnp.bfloat16


def test_ws32_strategy_nd_dense_stablehlo_uses_lp4_then_lp8() -> None:
    program = r'''
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.kernels.ws32 import (
    ws32_strategy_nd_dense_final_layout_mapped,
)

mesh = Mesh(
    np.asarray(jax.devices(), dtype=object).reshape(8, 4),
    ("expert", "feature"),
)
def abstract(shape, dtype, spec):
    return jax.ShapeDtypeStruct(
        shape, dtype, sharding=NamedSharding(mesh, spec)
    )
mapped = jax.shard_map(
    ws32_strategy_nd_dense_final_layout_mapped,
    mesh=mesh,
    in_specs=(
        P(None, "feature"),
        P("expert", None, None),
        P("expert", None, None),
        P("expert", None, "feature"),
        P("expert", None, "feature"),
    ),
    out_specs=P(None, "feature"),
    check_vma=False,
)
arguments = (
    abstract((1, 6144), jnp.bfloat16, P(None, "feature")),
    abstract((32, 6144, 768), jnp.uint8, P("expert", None, None)),
    abstract((32, 48, 768), jnp.float32, P("expert", None, None)),
    abstract((32, 384, 6144), jnp.uint8, P("expert", None, "feature")),
    abstract((32, 3, 6144), jnp.float32, P("expert", None, "feature")),
)
stablehlo = str(
    jax.jit(mapped).lower(*arguments).compiler_ir(dialect="stablehlo")
)
gathers = [
    line.strip()
    for line in stablehlo.splitlines()
    if "stablehlo.all_gather" in line
]
print(json.dumps({
    "all_reduce_count": stablehlo.count("stablehlo.all_reduce"),
    "gathers": gathers,
}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    import json

    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["all_reduce_count"] == 0
    assert len(result["gathers"]) == 2
    assert "tensor<8x4xi64>" in result["gathers"][0]
    assert "tensor<1x1536xbf16>) -> tensor<1x6144xbf16>" in result["gathers"][0]
    assert "tensor<4x8xi64>" in result["gathers"][1]
    assert (
        "tensor<1x4x1x1536xbf16>) -> tensor<8x4x1x1536xbf16>"
        in result["gathers"][1]
    )
