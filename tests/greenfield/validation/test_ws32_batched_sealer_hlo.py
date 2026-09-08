"""Replay acquired companion graphs through the actual batched sealer entry."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.validation.ws32_prefill_admission import (
    SHORT_PROFILE,
    short_acquisition,
)
from scripts.greenfield import seal_short_decoder_ws32 as sealer


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "graph", ["cache_probe", "decode", "observer", "exact_materialize", "exact_promote"]
)
def test_original_companion_graph_through_batched_sealer(graph):
    receipt = short_acquisition(ROOT)
    run = Path("/home/gianl/glm-run") / receipt["tag"]
    paths = [
        run / "hlo" / f"{graph}.{suffix}"
        for suffix in ("stablehlo.mlir", "optimized_hlo.txt")
    ]
    if (
        not all(path.exists() for path in paths)
        or not (run / "runner.rank0.json").exists()
    ):
        pytest.skip("original acquisition unavailable")
    pins = receipt["graphs"][graph]
    args = SimpleNamespace(
        batched_prefill_profile=SHORT_PROFILE,
        exact_dsa=True,
        strategy_nd_dense=True,
        host_main_rope_table=True,
        **{f"expected_{graph}_{name}": value for name, value in pins.items()},
    )
    result = sealer._replay_batched_graph(
        paths[0].read_text(),
        paths[1].read_text(),
        graph=graph,
        args=args,
    )
    identity = result.pop("source_location_identity")
    assert identity["raw_optimized_hlo_sha256"] == pins["optimized_hlo_sha256"]
    original = json.loads((run / "runner.rank0.json").read_text())["graphs"][graph]
    assert result == {**original, "passed": True, "violations": []}


@pytest.mark.parametrize("graph", ["prefill_chunk", "prefill_tail"])
def test_numerical_prefill_replay_matches_serialized_fleet(graph):
    """Exercise the real JSON boundary missed by in-memory admission tests."""
    run = Path("/home/gianl/glm-run") / (
        "greenfield_ws32_short_decoder_2k_numerical_c17_hrope_bp1_"
        "20260908T090441883274696Z"
    )
    if not (run / "fleet" / "runner.rank0.json").exists():
        pytest.skip("original numerical fleet evidence unavailable")
    pins = short_acquisition(ROOT)["graphs"][graph]
    args = SimpleNamespace(
        batched_prefill_profile=SHORT_PROFILE,
        exact_dsa=True,
        strategy_nd_dense=True,
        host_main_rope_table=True,
        **{f"expected_{graph}_{name}": value for name, value in pins.items()},
    )
    stable = run / "fleet_hlo" / f"{graph}.rank0.stablehlo.mlir"
    optimized = run / "fleet_hlo" / f"{graph}.rank0.optimized_hlo.txt"
    replay = sealer._replay_batched_graph(
        stable.read_text(), optimized.read_text(), graph=graph, args=args
    )
    assert sealer._same(replay, json.loads(json.dumps(replay)))
    for rank in range(8):
        record = json.loads((run / "fleet" / f"runner.rank{rank}.json").read_text())
        original = record["graphs"][graph]
        assert sealer._same(replay, original), rank
        changed = json.loads(json.dumps(original))
        changed["atomic_commit_proof"]["rollback_inputs"]["0"] = 3
        assert not sealer._same(replay, changed), rank
        changed["atomic_commit_proof"]["rollback_inputs"]["0"] = 2.0
        assert not sealer._same(replay, changed), rank
