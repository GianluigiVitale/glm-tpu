"""Forced-CPU equivalence of the decoder with host DSA rotary rows.

On CPU the on-device ``cos``/``sin`` are accurate, so a decoder step fed with
the host FP32 DSA table must reproduce the default step's tokens, metadata and
selected positions and keep index-cache keys within BF16 rounding; with both
host tables enabled no transcendental may remain in the compiled step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_small_decoder_with_host_dsa_rope_table_matches_default_on_forced_cpu() -> None:
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
dsa_rope_decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, dsa_rope_table_enabled=True)
both_decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs, complete_token_path=True, main_rope_table_enabled=True, dsa_rope_table_enabled=True)
assert decoder.dsa_rope_table_host is None and decoder.dsa_rope_table_sha256 is None and decoder.dsa_rope_table_bytes_per_device == 0
assert dsa_rope_decoder.dsa_rope_table_host is not None and dsa_rope_decoder.dsa_rope_table_host.dtype == np.float32
assert dsa_rope_decoder.dsa_rope_table_host.shape == (8, dsa_rope_decoder.config.dsa_rope_table_width)
assert dsa_rope_decoder.config.dsa_rope_table_width % 2 == 0
assert len(dsa_rope_decoder.input_specs) == len(decoder.input_specs) + 1
assert len(both_decoder.input_specs) == len(decoder.input_specs) + 2

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
dsa_table = put(dsa_rope_decoder.dsa_rope_table_host, dsa_rope_decoder.input_specs[-1])
main_table = put(both_decoder.main_rope_table_host, both_decoder.input_specs[-2])
default_compiled = jax.jit(decoder.execute).lower(*inputs).compile()
dsa_lowered = jax.jit(dsa_rope_decoder.execute).lower(*inputs, dsa_table)
dsa_compiled = dsa_lowered.compile()
both_lowered = jax.jit(both_decoder.execute).lower(*inputs, main_table, dsa_table)
both_compiled = both_lowered.compile()
first = default_compiled(*inputs)
dsa_first = dsa_compiled(*inputs, dsa_table)
both_first = both_compiled(*inputs, main_table, dsa_table)

def host(values):
    return [np.asarray(jax.device_get(value)) for value in values]
default_out, dsa_out, both_out = host(first), host(dsa_first), host(both_first)
report = {
    'output_count': len(default_out),
    'dsa_metadata_equal': bool(np.array_equal(default_out[3], dsa_out[3])),
    'dsa_index_max_abs': float(np.abs(default_out[2].astype(np.float32) - dsa_out[2].astype(np.float32)).max()),
    'dsa_residual_max_abs': float(np.abs(default_out[0].astype(np.float32) - dsa_out[0].astype(np.float32)).max()),
    'dsa_tokens_equal': bool(np.array_equal(default_out[4], dsa_out[4])) if len(default_out) > 4 else None,
    'both_metadata_equal': bool(np.array_equal(default_out[3], both_out[3])),
    'both_tokens_equal': bool(np.array_equal(default_out[4], both_out[4])) if len(default_out) > 4 else None,
    'both_index_max_abs': float(np.abs(default_out[2].astype(np.float32) - both_out[2].astype(np.float32)).max()),
}
both_module = parse_hlo_module(both_compiled.as_text())
report['both_transcendental_count'] = sum(1 for i in both_module.instructions if i.opcode in ('cosine', 'sine', 'power'))
dsa_module = parse_hlo_module(dsa_compiled.as_text())
report['dsa_lookup_count'] = sum(1 for i in dsa_module.instructions if i.op_name and 'greenfield_dsa_rope_table_lookup' in i.op_name)
default_module = parse_hlo_module(default_compiled.as_text())
report['default_transcendental_count'] = sum(1 for i in default_module.instructions if i.opcode in ('cosine', 'sine', 'power'))
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
    assert report["dsa_metadata_equal"] is True
    assert report["both_metadata_equal"] is True
    assert report["output_count"] > 4
    assert report["dsa_tokens_equal"] is True
    assert report["both_tokens_equal"] is True
    assert report["dsa_index_max_abs"] <= 2.0**-7
    assert report["both_index_max_abs"] <= 2.0**-7
    assert report["dsa_residual_max_abs"] <= 2.0**-6
    assert report["dsa_lookup_count"] >= 1
    assert report["both_transcendental_count"] == 0
    assert report["default_transcendental_count"] >= 1
