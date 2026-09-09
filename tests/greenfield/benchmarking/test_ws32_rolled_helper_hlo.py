"""Actual rolled annotation/copy/scratch refusals on retained DB602 originals."""

from dataclasses import replace
import re

import pytest

from tests.greenfield.benchmarking.test_ws32_rolled_prefill_hlo import original, rewrite
from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import (
    _target, check_batched_helpers,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_prefill_hlo_identity import PrefillIdentity, Value
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_helper_hlo import check_rolled_helpers


@pytest.mark.parametrize("rows", [114, 128])
def test_historical_helper_guard_stays_closed(rows):
    with pytest.raises(ValueError, match="only for B17/B11"):
        check_batched_helpers(None, block_rows=rows, live_instructions=())


@pytest.fixture(scope="module")
def helpers(original):
    index, rows, live, _, _ = original
    report = check_rolled_helpers(index, block_rows=rows, live_instructions=live)
    assert report["passed"], report
    return index, rows, live, report


def test_actual_helpers_and_initialized_scratch(helpers):
    _, rows, _, report = helpers
    assert report["helper_count"] == (2705 if rows == 128 else 2771)
    pairs = report["scratch_pairs"]
    assert sum(p["kind"] == "panel_search_pair" for p in pairs) == 150
    assert sum(p["kind"] == "merge_complete_halves" for p in pairs) == 42
    assert report["missing"] == report["unexpected"] == []
    assert "NUMERICAL_OR_MEMORY_ADMISSION" in report["not_proven"]


@pytest.mark.parametrize("case", [
    "annotation_scope", "copy_overlap", "scratch_geometry", "scratch_escape",
    "half_initialization", "duplicate_partial_binding",
])
def test_actual_helper_mutations_refuse(helpers, case):
    index, rows, live, report = helpers
    custom = [op for op in index.module.instructions if op.opcode == "custom-call"]
    if case == "annotation_scope":
        op = next(op for op in custom if _target(op) == "AssumeGatherIndicesInBound")
        assert op.op_name.endswith("/gather")
        changed = rewrite(index, op, op_name=op.op_name + "/not_a_gather")
    elif case == "copy_overlap":
        concat = next(op for op in custom if _target(op) == "ConcatBitcast")
        first = index.operand(index.operand(concat, 0), 0)
        second = index.operand(index.operand(concat, 1), 0)
        ranges = re.search(r"slice=\{([^}]*)\}", second.raw_line)[1]
        raw, count = re.subn(r"slice=\{[^}]*\}", "slice={" + ranges + "}", first.raw_line)
        assert count == 1 and raw != first.raw_line
        changed = rewrite(index, first, raw_line=raw)
    elif case in ("scratch_geometry", "scratch_escape"):
        op = next(op for op in custom if _target(op) == "AllocateBuffer"
                  and op.result_shapes[0].dtype == "u32")
        if case == "scratch_geometry":
            shape = op.result_shapes[0]
            changed = rewrite(index, op, result_shapes=(replace(
                shape, dimensions=(shape.dimensions[0] + 1,)
            ),))
        else:
            # A second direct scratch consumer must fail even if it is not
            # itself a helper/custom call and does not alter helper counts.
            extra = replace(
                op, index=max(p.index for p in index.module.instructions) + 1,
                name="%adversarial_scratch_escape", opcode="copy", raw_opcode="copy",
                operand_names=(op.name,), operand_shapes=op.result_shapes,
                raw_line=f"  %adversarial_scratch_escape = u32[{op.result_shapes[0].dimensions[0]}] copy({op.name})",
            )
            changed = PrefillHloIndex(replace(index.module, instructions=(
                *index.module.instructions, extra,
            )))
    else:
        record = next(p for p in report["scratch_pairs"] if p["kind"] == "merge_complete_halves")
        completed = next(op for op in index.module.instructions if op.name == record["completion"])
        ssa = PrefillIdentity(index)
        second = ssa.resolve(Value(completed), stop_at_shape_change=True)
        assert second.op.opcode == "dynamic-update-slice"
        if case == "half_initialization":
            offset = ssa.resolve(ssa.operand(second, 3))
            assert ssa.constant(offset, "s32", 1)
            changed = rewrite(index, offset.op, raw_line=offset.op.raw_line.replace(
                "constant(1)", "constant(0)"
            ))
        else:
            # Duplicate an existing partial-buffer binding. Instruction-user
            # deduplication alone cannot distinguish one from two bindings.
            base = ssa.resolve(ssa.operand(second, 0), stop_at_shape_change=True)
            caller = next(
                index.operand(completed, i)
                for i in range(len(completed.operand_names))
                if ssa.same(ssa.operand(Value(completed), i), base)
            )
            assert caller.opcode == "fusion"
            assert completed.operand_names.count(caller.name) == 1
            changed = rewrite(
                index, completed,
                operand_names=(*completed.operand_names, caller.name),
                operand_shapes=(*completed.operand_shapes, caller.result_shapes[0]),
            )
    result = check_rolled_helpers(changed, block_rows=rows, live_instructions=live)
    assert not result["passed"], (case, result)
