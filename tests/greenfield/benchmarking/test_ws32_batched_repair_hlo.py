"""Actual original repair provenance and same-shape cross-producer mutations."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_batched_repair_hlo import (
    check_batched_repair_lineage,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module


@pytest.fixture(scope="module", params=[("prefill_tail", 11), ("prefill_chunk", 17)])
def original(request):
    graph, rows = request.param
    root = Path(__file__).resolve().parents[3]
    receipt = json.loads(
        (
            root
            / "docs/artifacts/prefill-batched-seven-graph-acquisition-20260908.json"
        ).read_text()
    )
    path = (
        Path("/home/gianl/glm-run")
        / receipt["tag"]
        / "hlo"
        / (graph + ".optimized_hlo.txt")
    )
    if not path.exists():
        pytest.skip("original optimized HLO unavailable")
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == receipt["graphs"][graph]["optimized_hlo_sha256"]
    index = PrefillHloIndex(parse_hlo_module(raw.decode()))
    baseline = check_batched_repair_lineage(index, block_rows=rows)
    assert baseline["passed"], baseline
    return index, rows, baseline


def rewrite(index, op, **changes):
    new = replace(op, **changes)
    return PrefillHloIndex(
        replace(
            index.module,
            instructions=tuple(
                new if p.index == op.index else p for p in index.module.instructions
            ),
        )
    )


def rewire(index, op, position, name):
    names = list(op.operand_names)
    old = names[position]
    names[position] = name
    # The replacements here are distinct same-shaped inputs, so unchanged
    # recorded operand shapes still describe the mutated raw graph faithfully.
    return rewrite(
        index, op, operand_names=tuple(names), raw_line=op.raw_line.replace(old, name)
    )


def test_original_producers(original):
    _, _, result = original
    assert len(result["producers"]) == 21
    assert [r["wk_entry_leaf"] for r in result["producers"]] == list(range(2324, 2345))
    assert "HEALTH_IMPLICATION" in result["not_proven"]
    assert result["dependency_nodes"] < 10000


@pytest.mark.parametrize(
    "case", ["wk", "normalized_under_label", "repeat_source", "writer_key", "precision"]
)
def test_cross_producer_mutations(original, case):
    index, rows, baseline = original
    entry = index.computations["ENTRY"]
    a, b = baseline["producers"][:2]
    pa, pb = entry[a["projection"]], entry[b["projection"]]
    if case == "wk":
        mutated = rewire(index, pa, 1, pb.operand_names[1])
    elif case == "repeat_source":
        mutated = rewire(index, pa, 0, pb.operand_names[0])
    elif case == "normalized_under_label":
        ga, gb = entry[a["normalized_gather"]], entry[b["normalized_gather"]]
        mutated = rewire(index, ga, 0, gb.operand_names[0])
    elif case == "writer_key":
        wa, wb = entry[a["writer"]], entry[b["writer"]]
        ta, tb = index.operand(wa, 2), index.operand(wb, 2)
        mutated = rewire(index, ta, 3, tb.operand_names[3])
    else:
        comp = index.callee(pa, "calls")
        dot = next(
            p for p in index.computations[comp].values() if p.opcode == "convolution"
        )
        assert "operand_precision={default,highest}" in dot.raw_line
        mutated = rewrite(
            index,
            dot,
            raw_line=dot.raw_line.replace(
                "operand_precision={default,highest}",
                "operand_precision={default,default}",
            ),
        )
    result = check_batched_repair_lineage(mutated, block_rows=rows)
    assert not result["passed"], (case, result)


@pytest.mark.parametrize("rows", [1, 12, 32, True, 17.0])
def test_unknown_row_profile(rows):
    with pytest.raises(ValueError, match="B17/B11"):
        check_batched_repair_lineage(None, block_rows=rows)
