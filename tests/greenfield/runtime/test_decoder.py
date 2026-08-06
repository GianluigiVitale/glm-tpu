from __future__ import annotations

import json
import os
import subprocess
import sys


def test_complete_small_decoder_step_runs_all_stages_on_forced_cpu() -> None:
    program = r'''
import json
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import NamedSharding
from glm_tpu.greenfield.model import build_decoder_runtime_weight_layout, build_decoder_state_layout, build_pipeline_schedule
from glm_tpu.greenfield.runtime import build_decoder_step_program, validate_decoder_step_hlo
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from tests.greenfield.checkpoint.test_runtime_pack import _small_plan

plan = _small_plan()
schedule = build_pipeline_schedule(plan)
state = build_decoder_state_layout(plan, schedule, context_capacity=8, logical_page_size=8, packed_kv_width=8)
weight_layout = build_decoder_runtime_weight_layout(plan, schedule)
groups = tuple(tuple(stage * 4 + slot for slot in range(4)) for stage in range(8))
pairs = tuple((groups[stage][slot], groups[(stage + 1) % 8][slot]) for stage in range(8) for slot in range(4))
decoder = build_decoder_step_program(plan, schedule, state, weight_layout, groups, pairs)

weights = {}
weight_specs = decoder.input_specs[0]
for spec in weight_layout.specs:
    shape = (32, *spec.shape)
    if spec.dtype == 'F8_E4M3':
        value = np.zeros(shape, np.uint8)
    elif spec.dtype == 'F32':
        value = np.ones(shape, np.float32) if spec.value_class == 'fp8_scale' else np.zeros(shape, np.float32)
    else:
        value = np.ones(shape, dtype=ml_dtypes.bfloat16)
        if spec.name.endswith('key_norm_bias'):
            value.fill(0)
    weights[spec.name] = jax.device_put(value, NamedSharding(decoder.mesh, weight_specs[spec.name]))

residual_host = np.zeros((32, 1, 8), dtype=ml_dtypes.bfloat16)
initial_row = np.asarray([[0.5, -0.25, 0.75, 1.0, -1.0, 0.125, 0.25, -0.5]], dtype=ml_dtypes.bfloat16)
for rank in groups[0]: residual_host[rank] = initial_row
kv_host = np.ones((32, 1, 1, 2, 8), dtype=ml_dtypes.bfloat16)
index_host = np.ones((32, 1, 1, 2, 2), dtype=ml_dtypes.bfloat16)
metadata_host = np.full((32, 1, decoder.config.metadata_width), -1, np.int32)
metadata_host[..., decoder.config.count_index] = 0
metadata_host[..., decoder.config.producer_index] = -1
metadata_host[..., decoder.config.visited_index] = 0
metadata_host[..., decoder.config.health_index] = 1
metadata_host[..., decoder.config.active_index] = 0
for rank in groups[0]: metadata_host[rank, 0, decoder.config.active_index] = 1

put = lambda value, spec: jax.device_put(value, NamedSharding(decoder.mesh, spec))
inputs = (
    weights,
    put(residual_host, decoder.input_specs[1]),
    put(kv_host, decoder.input_specs[2]),
    put(index_host, decoder.input_specs[3]),
    put(metadata_host, decoder.input_specs[4]),
    put(np.asarray([0], np.int32), decoder.input_specs[5]),
    put(np.asarray([[0]], np.int32), decoder.input_specs[6]),
    put(np.asarray([1], np.int32), decoder.input_specs[7]),
)
compiled = jax.jit(decoder.execute).lower(*inputs).compile()
residual, kv, index, metadata = map(np.asarray, jax.device_get(compiled(*inputs)))
active = np.flatnonzero(metadata[:, 0, decoder.config.active_index] == 1)
module = parse_hlo_module(compiled.as_text())
hlo_contract = validate_decoder_step_hlo(compiled.as_text(), config=decoder.config, schedule=schedule, groups=groups, pairs=pairs, backend_contract='cpu_reference')
counts = {}
for item in module.collectives: counts[item.opcode] = counts.get(item.opcode, 0) + 1
local_groups = tuple(tuple(group) for group in groups)
collectives_local = all(
    item.replica_groups == local_groups
    for item in module.collectives
    if item.opcode in ('all-gather', 'all-reduce')
)
kv_writes = []
index_writes = []
for stage in range(8):
    owner = groups[stage][0]
    kv_writes.append(bool(np.all(kv[owner, 0, 0, 0] == 0)))
    index_writes.append(bool(np.all(index[owner, 0, 0, 0] == 0)))

print(json.dumps({
    'active': active.tolist(),
    'collectives_local': collectives_local,
    'counts': counts,
    'health': sorted(set(metadata[active, 0, decoder.config.health_index].tolist())),
    'hlo_contract': {key: hlo_contract[key] for key in ('collective_count', 'collective_counts', 'passed', 'violations')},
    'index_writes': index_writes,
    'inactive_cache_unchanged': bool(np.all(kv[1:, 0, 0, 1] == 1)),
    'kv_writes': kv_writes,
    'positions': sorted(set(tuple(row) for row in metadata[active, 0, :4].tolist())),
    'producer': sorted(set(metadata[active, 0, decoder.config.producer_index].tolist())),
    'residual_exact': bool(np.array_equal(residual[active], np.repeat(initial_row[None], 4, axis=0))),
    'valid_counts': sorted(set(metadata[active, 0, decoder.config.count_index].tolist())),
    'visited': sorted(set(metadata[active, 0, decoder.config.visited_index].tolist())),
}, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = (
        f"{existing} --xla_force_host_platform_device_count=32".strip()
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
    assert result["active"] == [0, 1, 2, 3]
    assert result["residual_exact"]
    assert result["positions"] == [[0, -1, -1, -1]]
    assert result["valid_counts"] == [1]
    assert result["producer"] == [7]
    assert result["visited"] == [255]
    assert result["health"] == [1]
    assert all(result["kv_writes"])
    assert all(result["index_writes"])
    assert result["inactive_cache_unchanged"]
    assert result["collectives_local"]
    assert result["hlo_contract"] == {
        "collective_count": 88,
        "collective_counts": {
            "all-gather": 56,
            "all-reduce": 16,
            "collective-permute": 16,
        },
        "passed": True,
        "violations": [],
    }
    assert result["counts"] == {
        "all-gather": 56,
        "all-reduce": 16,
        "collective-permute": 16,
    }
