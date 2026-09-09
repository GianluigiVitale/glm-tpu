"""Retained DB602 own-WK/input/writer provenance; no TPU execution."""

import pytest

from tests.greenfield.benchmarking.test_ws32_rolled_prefill_hlo import original, rewrite
from glm_tpu.greenfield.benchmarking.ws32_batched_repair_hlo import check_batched_repair_lineage
from glm_tpu.greenfield.benchmarking.ws32_prefill_hlo_identity import Value, attribute
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_repair_hlo import check_rolled_repair_lineage


@pytest.mark.parametrize("rows", [114, 128])
def test_historical_repair_guard_remains_closed(rows):
    with pytest.raises(ValueError, match="B17/B11"):
        check_batched_repair_lineage(None, block_rows=rows)


@pytest.fixture(scope="module")
def repair(original):
    index, rows, live, adapter, loops = original
    report = check_rolled_repair_lineage(index, block_rows=rows, live_instructions=live)
    assert report["passed"], report
    return index, rows, live, adapter, loops, report


def test_actual_own_repair_producers(repair):
    _, rows, _, _, _, report = repair
    assert len(report["producers"]) == 21
    assert [r["wk_entry_leaf"] for r in report["producers"]] == list(range(2324, 2345))
    assert report["dependency_nodes"] == (4120 if rows == 128 else 4116)
    assert report["scope"] == "ROLLED_OWN_WK_NORMALIZED_INPUT_AND_REPAIRED_WRITER_PROVENANCE"


@pytest.mark.parametrize("case", ["wrong_wk", "cross_normalized", "writer_dependency", "precision"])
def test_actual_repair_mutations_refuse(repair, case):
    index, rows, live, adapter, loops, report = repair
    first = report["producers"][0]
    body = index.callee(loops[0].loop, "body")
    nodes = index.computations[body]
    projection = nodes[first["projection"]]
    if case == "wrong_wk":
        # Change the actual ENTRY WK leaf, not merely a scope label or the
        # invariant slot number (tail legitimately forwards equal-value aliases).
        op = next(op for op in index.computations["ENTRY"].values()
                  if op.opcode == "get-tuple-element"
                  and attribute(op, "index") == "2324")
        changed = rewrite(index, op, raw_line=op.raw_line.replace("index=2324", "index=2325"))
    elif case == "cross_normalized":
        gather = nodes[first["normalized_gather"]]
        actual = index.operand(gather, 0)
        other = next(op for op in nodes.values()
                     if op.result_shapes == actual.result_shapes
                     and not adapter.ssa.same(Value(op), Value(actual)))
        changed = rewrite(index, gather, operand_names=(other.name,))
    elif case == "writer_dependency":
        writer = nodes[first["writer"]]
        arguments = index.operand(writer, 2)
        key = index.operand(arguments, 3)
        other = next(op for op in nodes.values()
                     if op.result_shapes == key.result_shapes
                     and not adapter.ssa.same(Value(op), Value(key)))
        operands = list(arguments.operand_names)
        operands[3] = other.name
        changed = rewrite(index, arguments, operand_names=tuple(operands))
    else:
        dot = next(op for op in index.computations[index.callee(projection, "calls")].values()
                   if op.opcode == "convolution")
        assert "operand_precision={default,highest}" in dot.raw_line
        changed = rewrite(index, dot, raw_line=dot.raw_line.replace(
            "operand_precision={default,highest}", "operand_precision={default,default}"
        ))
    result = check_rolled_repair_lineage(changed, block_rows=rows, live_instructions=live)
    assert not result["passed"], (case, result)
