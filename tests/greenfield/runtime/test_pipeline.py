from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.runtime.pipeline import PipelineSkeletonConfig
from glm_tpu.greenfield.runtime.pipeline import _canonical_groups, _canonical_pairs


GROUPS = tuple(tuple(stage * 4 + slot for slot in range(4)) for stage in range(8))
PAIRS = tuple(
    (GROUPS[stage][slot], GROUPS[(stage + 1) % 8][slot])
    for stage in range(8)
    for slot in range(4)
)


def test_pipeline_topology_contract_refuses_lane_or_group_drift() -> None:
    config = PipelineSkeletonConfig(8, 4, 8, 4)
    assert _canonical_groups(GROUPS, config) == GROUPS
    assert _canonical_pairs(PAIRS, GROUPS, config) == PAIRS
    bad = list(PAIRS)
    bad[0] = (bad[0][0], GROUPS[1][1])
    with pytest.raises(PlanValidationError, match="lanes"):
        _canonical_pairs(bad, GROUPS, config)
    with pytest.raises(PlanValidationError, match="partition"):
        _canonical_groups((*GROUPS[:-1], GROUPS[-2]), config)


def test_one_live_row_pipeline_executes_each_stage_once_on_forced_cpu() -> None:
    program = r'''
import json
import numpy as np
import jax
from glm_tpu.greenfield.runtime.pipeline import PipelineSkeletonConfig, build_pipeline_skeleton

groups = tuple(tuple(stage * 4 + slot for slot in range(4)) for stage in range(8))
pairs = tuple(
    (groups[stage][slot], groups[(stage + 1) % 8][slot])
    for stage in range(8)
    for slot in range(4)
)
config = PipelineSkeletonConfig(8, 4, 8, 4)
pipeline = build_pipeline_skeleton(config, groups, pairs, (2, 18, 30, 38, 46, 58, 66, 74))
residual, metadata = map(np.asarray, jax.device_get(pipeline.compiled(pipeline.residual, pipeline.metadata)))
active = metadata[:, 0, -1]
active_ranks = np.flatnonzero(active == 1).tolist()
result = {
    "active_ranks": active_ranks,
    "contract": {k: pipeline.hlo_contract[k] for k in ("all_reduce_count", "collective_permute_count", "passed", "violations")},
    "inactive_metadata_zero": bool(np.all(metadata[4:, 0, -3:] == np.asarray([-1, 0, 0], np.int32))),
    "producer": sorted(set(metadata[active_ranks, 0, -3].tolist())),
    "selected": sorted(set(tuple(row) for row in metadata[active_ranks, 0, :4].tolist())),
    "visited": sorted(set(metadata[active_ranks, 0, -2].tolist())),
    "residual_first": sorted(set(residual[active_ranks, 0, 0].astype(np.float32).tolist())),
}
print(json.dumps(result, sort_keys=True))
'''
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    existing = env.get("XLA_FLAGS", "").strip()
    env["XLA_FLAGS"] = f"{existing} --xla_force_host_platform_device_count=32".strip()
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["contract"] == {
        "all_reduce_count": 8,
        "collective_permute_count": 16,
        "passed": True,
        "violations": [],
    }
    assert result["active_ranks"] == [0, 1, 2, 3]
    assert result["inactive_metadata_zero"]
    assert result["producer"] == [74]
    assert result["selected"] == [[3, 2, 1, 0]]
    assert result["visited"] == [255]
    assert result["residual_first"] == [35.5]
