from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.greenfield.probe_ws32_strategy_nd_layer0 import (
    TENSOR_NAMES,
    _lowering_contract,
    _manifest_records,
)


REPO = Path(__file__).resolve().parents[3]
TPU_HLO = """HloModule x
ENTRY %main (p0: bf16[1,1536], p1: bf16[4,1,1536]) -> (bf16[1,6144], bf16[32,1,1536]) {
  %p0 = bf16[1,1536]{1,0} parameter(0)
  %p1 = bf16[4,1,1536]{2,1,0} parameter(1)
  %ag0 = bf16[1,6144]{1,0} all-gather(%p0), channel_id=1, replica_groups={{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15},{16,17,18,19},{20,21,22,23},{24,25,26,27},{28,29,30,31}}, dimensions={1}, use_global_device_ids=true, metadata={op_name="jit(x)/greenfield_ws32_strategy_nd_dense/hidden_gather/all_gather"}
  %ag1 = bf16[32,1,1536]{2,1,0} all-gather(%p1), channel_id=2, replica_groups={{0,4,8,12,16,20,24,28},{1,5,9,13,17,21,25,29},{2,6,10,14,18,22,26,30},{3,7,11,15,19,23,27,31}}, dimensions={0}, use_global_device_ids=true, metadata={op_name="jit(x)/greenfield_ws32_strategy_nd_dense/expert_gather/all_gather"}
  ROOT %out = (bf16[1,6144], bf16[32,1,1536]) tuple(%ag0,%ag1)
}
"""


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


def test_bounded_layer_preserves_graph_before_structural_refusal() -> None:
    script = (
        REPO / "scripts/greenfield/probe_ws32_strategy_nd_layer0.py"
    ).read_text(encoding="utf-8")
    graph_write = script.index(
        '(args.hlo_dir / "layer0.optimized_hlo.txt").write_text(optimized_hlo)'
    )
    refusal = script.index('if not contract["passed"]:', graph_write)
    assert graph_write < refusal
    assert '"status": "HLO_REFUSED"' in script[refusal:]


def test_bounded_layer_accepts_only_exact_tpu_collective_geometry() -> None:
    assert _lowering_contract(TPU_HLO)["passed"]
    assert not _lowering_contract(
        TPU_HLO.replace("bf16[4,1,1536]", "bf16[1,4,1,1536]")
    )["passed"]
    assert not _lowering_contract(
        TPU_HLO.replace("{0,4,8,12,16,20,24,28}", "{0,1,2,3,4,5,6,7}")
    )["passed"]
    assert not _lowering_contract(
        TPU_HLO.replace("expert_gather/all_gather", "wrong_scope/all_gather")
    )["passed"]
