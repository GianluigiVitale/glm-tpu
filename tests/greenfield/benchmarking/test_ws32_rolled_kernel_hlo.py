"""Saved DB602 Pallas interfaces and actual placement, without TPU execution."""

from dataclasses import replace
import re

import pytest

from tests.greenfield.benchmarking.test_ws32_rolled_prefill_hlo import original, rewrite
from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import _target
from glm_tpu.greenfield.benchmarking.ws32_batched_kernel_hlo import check_batched_kernels
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import _computation_base
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_kernel_hlo import check_rolled_kernels


@pytest.mark.parametrize("rows", [114, 128])
def test_historical_kernel_guard_stays_closed(rows):
    with pytest.raises(ValueError, match="only for B17/B11"):
        check_batched_kernels(None, block_rows=rows, live_instructions=())


def test_actual_rolled_pallas_inventory(original):
    index, rows, live, _, _ = original
    report = check_rolled_kernels(index, block_rows=rows, live_instructions=live)
    assert report["passed"], report
    assert report["kernel_count"] == 1047
    assert report["family_counts"] == {
        "raw": 588, "panels": 225, "structured": 156, "sparse": 78,
    }
    assert report["static_placement_counts"] == {"prefix": 588, "suffix": 459}
    assert report["four_iteration_prefix_schedule_count"] == 2352
    assert report["missing"] == report["unexpected"] == []
    assert "OPAQUE_KERNEL_ARITHMETIC" in report["not_proven"]


@pytest.mark.parametrize("case", [
    "alias", "panel_count", "output_dtype", "prefix_placement", "side_effect",
    "panel_false_branch",
])
def test_actual_rolled_kernel_mutations_refuse(original, case):
    index, rows, live, _, loops = original
    calls = [op for op in index.module.instructions
             if op.opcode == "custom-call" and _target(op) == "tpu_custom_call"]
    panel = next(op for op in calls if
                 "/greenfield_prefill_expert_panel_raw_fp8/pallas_call" in op.op_name)
    if case == "alias":
        raw, count = re.subn(
            r"(output_to_operand_aliasing=\{\{\}:\s*\()5(,\s*\{\}\)\})",
            r"\g<1>0\2", panel.raw_line,
        )
        assert count == 1
        changed = rewrite(index, panel, raw_line=raw)
    elif case == "panel_count":
        shape = panel.result_shapes[0]
        assert shape.dimensions[0] == (63 if rows == 128 else 60)
        changed = rewrite(index, panel, result_shapes=(replace(
            shape, dimensions=(shape.dimensions[0] + 1, *shape.dimensions[1:])
        ),))
    elif case == "output_dtype":
        shape = panel.result_shapes[0]
        changed = rewrite(index, panel, result_shapes=(replace(
            shape, dtype="bf16" if shape.dtype == "f32" else "f32"
        ),))
    elif case == "prefix_placement":
        body = index.callee(loops[0].loop, "body")
        op = next(op for op in calls if _computation_base(op.computation) == body)
        assert "/layer_0/" in op.op_name
        changed = rewrite(index, op, op_name=op.op_name.replace("/layer_0/", "/layer_1/"))
    elif case == "side_effect":
        raw = panel.raw_line
        if "custom_call_has_side_effect=false" in raw:
            raw = raw.replace("custom_call_has_side_effect=false", "custom_call_has_side_effect=true")
        else:
            assert "custom_call_has_side_effect" not in raw
            raw = raw.replace(
                'custom_call_target="tpu_custom_call"',
                'custom_call_target="tpu_custom_call", custom_call_has_side_effect=true',
            )
        changed = rewrite(index, panel, raw_line=raw)
    else:
        body = _computation_base(panel.computation)
        owners = []
        for op in index.module.instructions:
            if op.opcode != "conditional":
                continue
            match = re.search(r"branch_computations=\{([^}]*)\}", op.raw_line)
            branches = [x.strip() for x in match[1].split(",")] if match else []
            if branches and branches[-1].lstrip("%") == body:
                owners.append((op, branches))
        assert len(owners) == 1
        op, branches = owners[0]
        changed = rewrite(index, op, raw_line=op.raw_line.replace(
            "branch_computations={" + ", ".join(branches) + "}",
            "branch_computations={" + ", ".join(reversed(branches)) + "}",
        ))
        assert changed.module.instructions[op.index].raw_line != op.raw_line
    report = check_rolled_kernels(changed, block_rows=rows, live_instructions=live)
    assert not report["passed"], (case, report)
