"""Retained rolled operand guards and actual-SSA hostile mutations (CPU only)."""

import pytest

from tests.greenfield.benchmarking.test_ws32_rolled_prefill_hlo import original, rewrite
from glm_tpu.greenfield.benchmarking.ws32_batched_operand_health_hlo import check_batched_operand_health
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_operand_health_hlo import (
    OperandTileHealth, TileBooleanFactors, TileOperands, check_rolled_operand_health,
)


@pytest.fixture(scope="module")
def operand(original):
    index, rows, live, adapter, loops = original
    report = check_rolled_operand_health(index, block_rows=rows, live_instructions=live)
    assert report["passed"], report
    health = OperandTileHealth(adapter)
    health.boolean = TileBooleanFactors(index, live_rows=32, live_mask=health.live_mask,
                                       nonempty_live=False)
    proof = TileOperands(adapter, loops[0], health)
    return index, rows, live, adapter, loops, report, proof


def test_actual_operand_guards(operand):
    *_, report, proof = operand
    assert len(report["layers"]) == 78
    assert len(report["routers"]) == 75
    assert len(report["panels"]) == 225
    assert all(r["activity_is_not_required_for_commit"] for r in report["panels"])
    assert report["scope"] == "ROLLED_OPERAND_FINITE_AND_PANEL_VALIDITY_GUARDS"
    assert proof.finite


@pytest.mark.parametrize("case", ["normalized_guard", "query_guard", "cache_layout",
                                  "absorbed_copy_layout", "panel_and", "panel_grid_gt",
                                  "panel_wrong_grid"])
def test_actual_operand_mutations(operand, case):
    index, rows, live, adapter, loops, report, p = operand
    body = index.computations[index.callee(loops[0].loop, "body")]
    b = p.boolean
    if case in ("normalized_guard", "query_guard"):
        key = "normalized" if case == "normalized_guard" else "prepared_query"
        call = b.ref(body[report["layers"][0][key]])
        data = p.arg(call, 0)
        guards = [b.ops[ref[0]] for ref, _ in p.known if not ref[1]
                  and b.ops[ref[0]].opcode == "is-finite" and p.arg(ref, 0) == data]
        assert guards
        changed = index
        for op in guards:
            changed = rewrite(changed, op, opcode="constant", raw_opcode="constant",
                              operand_names=(), operand_shapes=(),
                              raw_line=op.raw_line.split("is-finite(", 1)[0] + "constant(true)")
    elif case == "cache_layout":
        call = b.ref(body[report["layers"][0]["sparse"]])
        op = b.ops[p.arg(call, 2)[0]]
        assert "{3,2,1,0:" in op.raw_line
        changed = rewrite(index, op, raw_line=op.raw_line.replace("{3,2,1,0:", "{2,3,1,0:"))
    elif case == "absorbed_copy_layout":
        view = next(ref for ref in p.finite
                    if b.shape(ref).dimensions == (32, 8, 4, 128)
                    and b.ops[ref[0]].opcode == "bitcast")
        op = b.ops[b.operand(view, 0)[0]]
        assert op.opcode == "copy" and "{3,0,1,2:" in op.raw_line
        changed = rewrite(index, op, raw_line=op.raw_line.replace("{3,0,1,2:", "{3,2,1,0:"))
    else:
        from glm_tpu.greenfield.benchmarking.ws32_batched_operand_health_hlo import OperandHealthProof
        outer = OperandHealthProof(index, rows)
        cond = index.computations["ENTRY"][report["panels"][0]["validity_guard"]]
        pred = outer.health.predicate(outer.boolean.ref(cond))
        op = outer.boolean.ops[pred[0]]
        assert op.opcode == "and"
        if case == "panel_and":
            changed = rewrite(index, op, opcode="or", raw_opcode="or",
                              raw_line=op.raw_line.replace(" and(", " or("))
        else:
            op = next(outer.boolean.ops[outer.arg(pred, j)[0]] for j in (0, 1)
                      if outer.boolean.ops[outer.arg(pred, j)[0]].opcode == "compare")
            assert "direction=GT" in op.raw_line
            if case == "panel_grid_gt":
                changed = rewrite(index, op, raw_line=op.raw_line.replace("direction=GT", "direction=GE"))
            else:
                grid = index.operand(op, 0)
                other = next(v for v in index.computations["ENTRY"].values()
                             if v.result_shapes == grid.result_shapes
                             and outer.boolean.resolve(outer.boolean.ref(v))
                             != outer.boolean.resolve(outer.boolean.ref(grid)))
                changed = rewrite(index, op, operand_names=(other.name, op.operand_names[1]))
    result = check_rolled_operand_health(changed, block_rows=rows, live_instructions=live)
    assert not result["passed"], (case, result)


@pytest.mark.parametrize("rows", [114, 128])
def test_historical_operand_guard_stays_closed(rows):
    with pytest.raises(ValueError, match="B17/B11"):
        check_batched_operand_health(None, block_rows=rows)
