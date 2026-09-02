from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.gate_d_projection_contraction_pp16_host_rope import (
    GateDProjectionHostRopePp16Result,
    build_gate_d_projection_host_rope_pp16,
)
from glm_tpu.greenfield.errors import PlanValidationError

ROOT = Path(__file__).parents[3]


@dataclass(frozen=True)
class _FakeDevice:
    id: int
    platform: str
    device_kind: str = "TPU v4"
    process_index: int = 0
    coords: tuple[int, int, int] = (0, 0, 0)
    core_on_chip: int = 0


def test_host_rope_builder_rejects_non_pp16_or_non_tpu_devices_before_mesh() -> None:
    with pytest.raises(PlanValidationError, match="two distinct"):
        build_gate_d_projection_host_rope_pp16(devices=(_FakeDevice(0, "tpu"),))
    with pytest.raises(PlanValidationError, match="exact PP16"):
        build_gate_d_projection_host_rope_pp16(
            devices=(_FakeDevice(0, "cpu"), _FakeDevice(1, "cpu"))
        )
    stage_zero = (
        _FakeDevice(0, "tpu", coords=(0, 0, 0)),
        _FakeDevice(1, "tpu", coords=(1, 0, 0)),
    )
    with pytest.raises(PlanValidationError, match="axis name"):
        build_gate_d_projection_host_rope_pp16(devices=stage_zero, axis_name="model")


def test_host_rope_result_contract_roots_only_the_causal_frontier() -> None:
    assert GateDProjectionHostRopePp16Result._fields == (
        "normalized_hidden_owners",
        "projected_key_owners",
        "current_key_owners",
    )


def test_host_rope_builder_source_has_no_device_transcendentals() -> None:
    source = (
        ROOT
        / "glm_tpu/greenfield/benchmarking/gate_d_projection_contraction_pp16_host_rope.py"
    ).read_text(encoding="utf-8")
    assert "rotary_cos_sin(" not in source
    assert "jnp.cos" not in source and "jnp.sin" not in source
    assert "dsa_index_keys_from_projection_host_rope(" in source
    assert 'key_norm_mode="divide_sqrt"' in source
    assert "lax.optimization_barrier(normalized_hidden_owner[0])" in source


def test_two_cpu_device_host_rope_shard_map_is_abstractly_constructable() -> None:
    code = """
import json
import jax
import jax.numpy as jnp
from glm_tpu.greenfield.benchmarking.gate_d_projection_contraction_pp16_host_rope import _build_shard_map
devices = tuple(jax.devices())
assert jax.default_backend() == 'cpu' and len(devices) == 2
replay = _build_shard_map(devices=devices, axis_name='feature')
outputs = jax.eval_shape(
    replay,
    jax.ShapeDtypeStruct((2, 1, 6144), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 128, 6144), jnp.float32),
    jax.ShapeDtypeStruct((2, 128), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 128), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 64), jnp.float32),
)
text = replay.lower(
    jax.ShapeDtypeStruct((2, 1, 6144), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 128, 6144), jnp.float32),
    jax.ShapeDtypeStruct((2, 128), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 128), jnp.bfloat16),
    jax.ShapeDtypeStruct((2, 64), jnp.float32),
).as_text()
print(json.dumps({
    'outputs': [{'shape': list(value.shape), 'dtype': str(value.dtype)} for value in outputs],
    'has_cosine': ('stablehlo.cosine' in text) or ('stablehlo.sine' in text),
    'has_collective': any(token in text for token in ('stablehlo.all_', 'stablehlo.collective_')),
    'has_dot': 'stablehlo.dot_general' in text,
}))
"""
    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
    }
    result = subprocess.run(
        ["/home/gianl/vllm-env/bin/python", "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["outputs"] == [
        {"dtype": "bfloat16", "shape": [2, 1, 6144]},
        {"dtype": "float32", "shape": [2, 1, 128]},
        {"dtype": "float32", "shape": [2, 1, 128]},
    ]
    assert report["has_cosine"] is False
    assert report["has_collective"] is False
    assert report["has_dot"] is True
