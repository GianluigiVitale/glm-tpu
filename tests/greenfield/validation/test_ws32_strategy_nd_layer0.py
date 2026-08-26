from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.greenfield.probe_ws32_strategy_nd_layer0 import (
    TENSOR_NAMES,
    _manifest_records,
)


REPO = Path(__file__).resolve().parents[3]


def _manifest() -> dict[str, object]:
    files = []
    for expert in range(8):
        ranks = list(range(expert * 4, expert * 4 + 4))
        for feature in range(4):
            tensors = {
                name: {"sha256": f"{expert:02d}-{name}"}
                for name in TENSOR_NAMES
            }
            files.append(
                {
                    "expert_coordinate": expert,
                    "feature_coordinate": feature,
                    "model_ranks": ranks,
                    "tensors": tensors,
                }
            )
    return {"files": files}


def test_bounded_layer_manifest_requires_exact_owner_and_rank_coverage() -> None:
    manifest = _manifest()
    assert len(_manifest_records(manifest)) == 32

    missing = deepcopy(manifest)
    missing["files"].pop()  # type: ignore[union-attr]
    with pytest.raises(RuntimeError, match="owner coverage"):
        _manifest_records(missing)

    wrong_rank = deepcopy(manifest)
    wrong_rank["files"][0]["model_ranks"][0] = 31  # type: ignore[index]
    with pytest.raises(RuntimeError, match="model-rank ownership"):
        _manifest_records(wrong_rank)


def test_bounded_layer_manifest_requires_gate_up_feature_replicas() -> None:
    manifest = _manifest()
    manifest["files"][1]["tensors"][TENSOR_NAMES[0]]["sha256"] = "drift"  # type: ignore[index]
    with pytest.raises(RuntimeError, match="feature replicas"):
        _manifest_records(manifest)


def test_bounded_layer_wrapper_is_default_off_and_same_region() -> None:
    script = (
        REPO / "scripts/greenfield/run_ws32_strategy_nd_layer0.sh"
    ).read_text(encoding="utf-8")
    assert "GLM_GREENFIELD_WS32_STRATEGY_ND_LAYER0:-0" in script
    assert "APPROVED_BUCKET=gs://driftbench-dsv4-uc" in script
    assert "US-CENTRAL2" in script
    assert "/home/gianl/.glm-tpu-rsync.lock" in script
    assert "/home/gianl/glm-run/.glm_pod_workload.lock" in script
    assert "performance_claim':False" in script
    assert "results_db_run_id':None" in script
    assert "driftbench-storage" not in script
