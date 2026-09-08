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
