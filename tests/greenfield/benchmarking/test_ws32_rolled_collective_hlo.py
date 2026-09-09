"""Actual DB602 collective schedule and physical-placement mutations, CPU only."""

import pytest

from tests.greenfield.benchmarking.test_ws32_rolled_prefill_hlo import original, rewrite
from glm_tpu.greenfield.benchmarking.ws32_batched_collective_hlo import (
    check_batched_collectives,
)
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import _computation_base
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_collective_hlo import (
    check_rolled_collectives,
)


@pytest.mark.parametrize("rows", [114, 128])
def test_historical_collective_public_guard_remains_closed(rows):
    with pytest.raises(ValueError, match="only for B17/B11"):
        check_batched_collectives(None, block_rows=rows, live_instructions=())


def test_actual_rolled_collective_inventory(original):
    index, rows, live, _, _ = original
    report = check_rolled_collectives(index, block_rows=rows, live_instructions=live)
    assert report["passed"], report
    assert report["static_collective_count"] == 787
    assert report["static_placement_counts"] == {
        "prefix": 474, "suffix": 306, "outer": 7,
    }
    assert report["reduction_operand_leaves"] == 898
    assert report["gather_output_leaves"] == 159
    assert report["four_iteration_prefix_schedule_count"] == 1896
    assert len(report["health_reductions"]) == 2
    assert "MEASURED_DYNAMIC_EXECUTION_COUNTS" in report["not_proven"]
    assert report["missing"] == report["unexpected"] == []
    # Boundary layers make the compiler's NEXT-router-bias fusion explicit:
    # layer2 carries the first bias, layer77 has no next router.
    for layer, arity in ((2, 2), (77, 1)):
        matching = [
            op for op in index.module.collectives
            if _computation_base(op.computation) == "ENTRY"
            and f"/layer_{layer}/" in (op.op_name or "")
            and op.raw_opcode == "all-reduce"
            and op.result_shapes[0].dtype == "bf16"
            and op.result_shapes[0].dimensions == (rows, 1536)
        ]
        assert len(matching) == 1
        assert len(matching[0].operand_names) == arity


@pytest.mark.parametrize("case", [
    "wrong_groups", "dead_collective", "gather_axis", "tuple_order",
    "prefix_layer_placement",
])
def test_actual_collective_mutations_refuse(original, case):
    index, rows, live, _, loops = original
    body = index.callee(loops[0].loop, "body")
    prefix = [
        op for op in index.module.collectives
        if _computation_base(op.computation) == body
    ]
    assert prefix
    if case == "wrong_groups":
        op = prefix[0]
        groups = [list(group) for group in op.replica_groups]
        groups[0][0], groups[1][0] = groups[1][0], groups[0][0]
        changed = rewrite(index, op, replica_groups=tuple(tuple(g) for g in groups))
    elif case == "dead_collective":
        op = prefix[0]
        changed = index
        live = tuple(p for p in live if p.index != op.index)
    elif case == "gather_axis":
        op = next(op for op in prefix if op.raw_opcode == "all-gather"
                  and "dimensions={0}" in op.raw_line)
        changed = rewrite(index, op, raw_line=op.raw_line.replace(
            "dimensions={0}", "dimensions={1}"
        ))
    elif case == "tuple_order":
        op = next(op for op in prefix if len(op.operand_names) == 4)
        # Permute producer references and matching input/output leaf shapes
        # together; the physical arity check still passes, exact ordering must not.
        order = (1, 0, 2, 3)
        changed = rewrite(
            index, op,
            operand_names=tuple(op.operand_names[i] for i in order),
            operand_shapes=tuple(op.operand_shapes[i] for i in order),
            result_shapes=tuple(op.result_shapes[i] for i in order),
        )
    else:
        op = prefix[0]
        assert "/layer_0/" in op.op_name
        changed = rewrite(index, op, op_name=op.op_name.replace("/layer_0/", "/layer_1/"))
    report = check_rolled_collectives(changed, block_rows=rows, live_instructions=live)
    assert not report["passed"], (case, report)
    if case == "prefix_layer_placement":
        assert "actual prefix body" in report["error"]
