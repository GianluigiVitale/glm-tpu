from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import jax.numpy as jnp
import pytest

from glm_tpu.greenfield.kernels.pallas import (
    StageRemoteCopyConfig,
    stage_value_remote_copy_kernel_name,
)
from glm_tpu.greenfield.kernels.pallas.stage_remote_copy import _canonical_targets


def test_stage_remote_copy_config_and_pairs_fail_closed() -> None:
    with pytest.raises(ValueError, match="at least two"):
        StageRemoteCopyConfig(total_devices=1)
    with pytest.raises(ValueError, match="permute"):
        _canonical_targets(((0, 1), (1, 1)), 2)
    with pytest.raises(ValueError, match="self"):
        _canonical_targets(((0, 0), (1, 1)), 2)
    assert (
        stage_value_remote_copy_kernel_name(
            jnp.zeros((1, 7), dtype=jnp.bfloat16)
        )
        == "greenfield_stage_remote_copy_bf16_1x7"
    )
    with pytest.raises(ValueError, match="BF16"):
        stage_value_remote_copy_kernel_name(jnp.zeros((1, 7), dtype=jnp.int16))


def test_stage_remote_copy_rejects_dtype_and_shape_drift() -> None:
    from glm_tpu.greenfield.kernels.pallas import stage_remote_copy_pallas

    config = StageRemoteCopyConfig(
        total_devices=4,
        hidden_width=7,
        metadata_width=5,
    )
    residual = jnp.zeros((1, 7), dtype=jnp.bfloat16)
    metadata = jnp.zeros((1, 5), dtype=jnp.int32)
    destination = jnp.asarray(1, dtype=jnp.int32)
    with pytest.raises(ValueError, match="BF16"):
        stage_remote_copy_pallas(
            residual.astype(jnp.float32),
            metadata,
            destination,
            config=config,
            interpret=True,
        )
    with pytest.raises(ValueError, match="compact shape"):
        stage_remote_copy_pallas(
            residual,
            metadata[:, :-1],
            destination,
            config=config,
            interpret=True,
        )


def test_interpreted_remote_copy_matches_reference_with_odd_tails() -> None:
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
    program = r'''
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from glm_tpu.greenfield.kernels.pallas import (
    StageRemoteCopyConfig,
    stage_remote_copy_kernel,
)

pairs = ((0, 2), (2, 1), (1, 3), (3, 0))
config = StageRemoteCopyConfig(
    total_devices=4,
    hidden_width=7,
    metadata_width=5,
)
mesh = Mesh(np.asarray(jax.devices(), dtype=object), ("device",))

def make(backend):
    def mapped(residual, metadata):
        output = stage_remote_copy_kernel(
            residual[0],
            metadata[0],
            axis_name="device",
            pairs=pairs,
            backend=backend,
            config=config,
            interpret=backend == "pallas",
        )
        return output[0][None, ...], output[1][None, ...]
    return jax.jit(jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=(P("device", None, None), P("device", None, None)),
        out_specs=(P("device", None, None), P("device", None, None)),
        check_vma=False,
    ))

residual = jnp.arange(4 * 7, dtype=jnp.float32).reshape(4, 1, 7).astype(jnp.bfloat16)
metadata = jnp.arange(4 * 5, dtype=jnp.int32).reshape(4, 1, 5)
reference = make("reference")(residual, metadata)
observed = make("pallas")(residual, metadata)
np.testing.assert_array_equal(np.asarray(observed[0]), np.asarray(reference[0]))
np.testing.assert_array_equal(np.asarray(observed[1]), np.asarray(reference[1]))
'''
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
