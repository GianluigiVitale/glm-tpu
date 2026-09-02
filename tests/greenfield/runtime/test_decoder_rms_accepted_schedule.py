"""Forced-CPU equivalence and HLO contract of the accepted decode-step RMS schedule.

The flag changes only the variance reduce shape (a 32-row barrier-carried
operand instead of a single live row), so on CPU the decoder step must keep the
default step's tokens and metadata and stay within BF16 rounding on state,
while its compiled step carries only row-wise ``rsqrt`` scales.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_small_decoder_with_accepted_rms_schedule_matches_default_on_forced_cpu() -> None:
    program = r"""
import json
from dataclasses import replace
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding
from glm_tpu.greenfield.model import build_decoder_runtime_weight_layout, build_decoder_state_layout, build_pipeline_schedule
from glm_tpu.greenfield.runtime import build_decoder_step_program
from glm_tpu.greenfield.runtime.decoder import _validate_rms_accepted_schedule_hlo, _validate_rms_accepted_schedule_stablehlo
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.checkpoint.test_runtime_pack import _small_plan

source_plan = _small_plan()
plan = replace(source_plan, geometry=replace(source_plan.geometry, vocab_size=96))
schedule = build_pipeline_schedule(plan)
state = build_decoder_state_layout(plan, schedule, context_capacity=8, logical_page_size=8, packed_kv_width=8)
weight_layout = build_decoder_runtime_weight_layout(plan, schedule)
groups = tuple(tuple(stage * 4 + slot for slot in range(4)) for stage in range(8))
pairs = tuple((groups[stage][slot], groups[(stage + 1) % 8][slot]) for stage in range(8) for slot in range(4))
decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True)
schedule_decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, rms_accepted_schedule=True)
assert decoder.rms_accepted_schedule is False and schedule_decoder.rms_accepted_schedule is True
assert len(schedule_decoder.input_specs) == len(decoder.input_specs)

weights = {}
weight_specs = decoder.input_specs[0]
rng = np.random.default_rng(8155)
for spec in weight_layout.specs:
    shape = (32, *spec.shape)
    if spec.dtype == 'F8_E4M3':
        value = rng.integers(0x30, 0x40, size=shape, dtype=np.uint8)
    elif spec.dtype == 'F32':
        value = np.ones(shape, np.float32) if spec.value_class == 'fp8_scale' else np.zeros(shape, np.float32)
    else:
        value = np.ones(shape, dtype=ml_dtypes.bfloat16)
        if spec.name.endswith('key_norm_bias'):
            value.fill(0)
        if 'head_weight' in spec.name or 'wq' in spec.name or 'wk' in spec.name:
            value = (rng.standard_normal(shape) * 0.5).astype(ml_dtypes.bfloat16)
    weights[spec.name] = jax.device_put(value, NamedSharding(decoder.mesh, weight_specs[spec.name]))

residual_host = np.zeros((32, 1, 8), dtype=ml_dtypes.bfloat16)
initial_row = np.asarray([[0.5, -0.25, 0.75, 1.0, -1.0, 0.125, 0.25, -0.5]], dtype=ml_dtypes.bfloat16)
for rank in groups[0]: residual_host[rank] = initial_row
kv_host = np.ones((32, 1, 1, 2, 8), dtype=ml_dtypes.bfloat16)
index_host = (rng.standard_normal((32, 1, 1, 2, 2)) * 0.5).astype(ml_dtypes.bfloat16)
metadata_host = np.full((32, 1, decoder.config.metadata_width), -1, np.int32)
metadata_host[..., decoder.config.count_index] = 0
metadata_host[..., decoder.config.producer_index] = -1
metadata_host[..., decoder.config.visited_index] = 0
metadata_host[..., decoder.config.health_index] = 1
metadata_host[..., decoder.config.active_index] = 0
for rank in groups[0]: metadata_host[rank, 0, decoder.config.active_index] = 1
token_host = np.full((32, 1), -1, np.int32)
for rank in groups[0]: token_host[rank, 0] = 5
put = lambda value, spec: jax.device_put(value, NamedSharding(decoder.mesh, spec))
inputs = (
    weights,
    put(residual_host, decoder.input_specs[1]),
    put(kv_host, decoder.input_specs[2]),
    put(index_host, decoder.input_specs[3]),
    put(metadata_host, decoder.input_specs[4]),
    put(token_host, decoder.input_specs[5]),
    put(np.asarray([5], np.int32), decoder.input_specs[6]),
    put(np.asarray([[0]], np.int32), decoder.input_specs[7]),
    put(np.asarray([6], np.int32), decoder.input_specs[8]),
)
default_lowered = jax.jit(decoder.execute).lower(*inputs)
schedule_lowered = jax.jit(schedule_decoder.execute).lower(*inputs)
default_stablehlo, schedule_stablehlo = default_lowered.as_text(), schedule_lowered.as_text()
default_compiled = default_lowered.compile()
schedule_compiled = schedule_lowered.compile()
first = default_compiled(*inputs)
schedule_first = schedule_compiled(*inputs)

def host(values):
    return [np.asarray(jax.device_get(value)) for value in values]
default_out, schedule_out = host(first), host(schedule_first)
default_module = parse_hlo_module(default_compiled.as_text())
schedule_module = parse_hlo_module(schedule_compiled.as_text())
report = {
    'output_count': len(default_out),
    'metadata_equal': bool(np.array_equal(default_out[3], schedule_out[3])),
    'tokens_equal': bool(np.array_equal(default_out[4], schedule_out[4])) if len(default_out) > 4 else None,
    'index_max_abs': float(np.abs(default_out[2].astype(np.float32) - schedule_out[2].astype(np.float32)).max()),
    'residual_max_abs': float(np.abs(default_out[0].astype(np.float32) - schedule_out[0].astype(np.float32)).max()),
    'default_contract': _validate_rms_accepted_schedule_hlo(default_module, enabled=False),
    'schedule_contract': _validate_rms_accepted_schedule_hlo(schedule_module, enabled=True),
    'default_as_schedule': _validate_rms_accepted_schedule_hlo(default_module, enabled=True)['passed'],
    'schedule_as_default': _validate_rms_accepted_schedule_hlo(schedule_module, enabled=False)['passed'],
    'default_stablehlo': _validate_rms_accepted_schedule_stablehlo(default_stablehlo, enabled=False),
    'schedule_stablehlo': _validate_rms_accepted_schedule_stablehlo(schedule_stablehlo, enabled=True),
    'default_stablehlo_as_schedule': _validate_rms_accepted_schedule_stablehlo(default_stablehlo, enabled=True)['passed'],
    'schedule_stablehlo_as_default': _validate_rms_accepted_schedule_stablehlo(schedule_stablehlo, enabled=False)['passed'],
}
print(json.dumps(report))
"""
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=32"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=1500,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["metadata_equal"] is True
    assert report["output_count"] > 4
    assert report["tokens_equal"] is True
    assert report["index_max_abs"] <= 2.0**-7
    assert report["residual_max_abs"] <= 2.0**-6
    assert report["default_contract"]["passed"], report["default_contract"]
    assert report["schedule_contract"]["passed"], report["schedule_contract"]
    assert report["schedule_contract"]["nonconforming_rsqrt_count"] == 0
    assert report["schedule_contract"]["conforming_rsqrt_count"] >= 1
    assert report["default_contract"]["conforming_rsqrt_count"] == 0
    assert report["default_contract"]["nonconforming_rsqrt_count"] >= 1
    assert report["default_as_schedule"] is False
    assert report["schedule_as_default"] is False
    assert report["default_stablehlo"]["passed"], report["default_stablehlo"]
    assert report["schedule_stablehlo"]["passed"], report["schedule_stablehlo"]
    assert report["schedule_stablehlo"]["conforming_rsqrt_count"] == report["schedule_contract"]["conforming_rsqrt_count"]
    assert report["schedule_stablehlo"]["barrier_count"] >= report["schedule_stablehlo"]["conforming_rsqrt_count"]
    assert report["default_stablehlo"]["conforming_rsqrt_count"] == 0
    assert report["default_stablehlo_as_schedule"] is False
    assert report["schedule_stablehlo_as_default"] is False
    # The DSA indexer key LayerNorm is the only non-RMS rsqrt and is unchanged by the flag.
    assert report["schedule_contract"]["layernorm_rsqrt_count"] >= 1
    assert report["schedule_contract"]["layernorm_rsqrt_count"] == report["default_contract"]["layernorm_rsqrt_count"]
    assert report["schedule_stablehlo"]["layernorm_rsqrt_count"] == report["default_stablehlo"]["layernorm_rsqrt_count"] >= 1
    assert report["schedule_contract"]["rsqrt_count"] == report["schedule_contract"]["conforming_rsqrt_count"] + report["schedule_contract"]["layernorm_rsqrt_count"]
