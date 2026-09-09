"""Actual reduced TPU originals and negative copy-structure regressions.

The source run stopped before all nine calls: these tests admit compiler
structure only, never numerical reproduction or model correctness.
"""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re

import pytest

from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import _target
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from scripts.greenfield import ws32_dense_frontier_admission as admission

ROOT = Path(
    "/home/gianl/glm-run/greenfield_fp8_ws32_dense_frontier_d01_20260909T151822927153523Z/rank0"
)
OPTIMIZED = {
    "dense01": "dbfc050ed769dab8f000669d175065664e39025dd53b9fbe03edf0ae788bf3e5",
    "wk_decode": "ad52d17fb9af62e0a172bea26d660cf9ebbad0d17c91c593e86c4eead3d283a2",
    "wk_promote": "da005257550e6c84c0116256e55857829a62abc1cbcb511961641cb0b522a62c",
}
FAMILIES = [
    ("bf16", (1024, 640), 1),
    ("u8", (1536, 2048), 2),
    ("u8", (3584, 512), 4),
    ("u8", (512, 2048), 4),
]


@pytest.fixture(scope="module")
def originals():
    runner = json.loads((ROOT / "runner.json").read_text())
    assert runner["code_hash"] == "849e6dbd5915aaea93ef2840250ff98912d5dd35"
    assert runner["status"] == "DIAGNOSTIC_FAILED"
    assert runner["current_phase"] == "dense/admission"
    assert runner["call_evidence"] == []
    graphs = {}
    for name, digest in OPTIMIZED.items():
        raw = (ROOT / f"{name}.optimized_hlo.txt").read_bytes()
        assert sha256(raw).hexdigest() == digest
        stable = (ROOT / f"{name}.stablehlo.mlir").read_bytes()
        assert (len(stable), sha256(stable).hexdigest()) == admission.RAW[name]
        graphs[name] = (stable.decode(), raw.decode())
    return runner, graphs


@pytest.mark.parametrize("name", list(OPTIMIZED))
def test_all_three_original_programs(originals, name):
    runner, graphs = originals
    result = admission.inspect_program(
        name, *graphs[name], runner["programs"][name]["compiled_memory"]
    )
    assert result["passed"] and not result["numerical_promotion"]
    assert not result["performance_claim"]
    if name == "dense01":
        assert result["structure"]["helpers"]["helper_count"] == 44


@pytest.fixture(scope="module")
def dense(originals):
    module = parse_hlo_module(originals[1]["dense01"][1])
    return PrefillHloIndex(module), _live_instruction_closure(module.instructions)


def copies(index, dtype, dims):
    return [
        op
        for op in index.module.instructions
        if op.raw_opcode == "custom-call"
        and _target(op) == "ConcatBitcast"
        and (op.result_shapes[0].dtype, op.result_shapes[0].dimensions) == (dtype, dims)
    ]


@pytest.mark.parametrize("dtype,dims,count", FAMILIES)
@pytest.mark.parametrize(
    "case", ["span", "source", "escape", "incomplete", "side_effect", "unknown", "dead"]
)
def test_actual_copy_mutations_refuse(dense, dtype, dims, count, case):
    index, live = dense
    selected = copies(index, dtype, dims)
    assert len(selected) == count
    op = selected[0]
    done = index.operand(op, 0)
    start = index.operand(done, 0)
    source = index.operand(start, 0)
    changes, extra = {}, []

    def change(old, **kwargs):
        changes[old.index] = replace(old, **kwargs)

    if case == "span":
        changed = re.sub(r"(slice=\{\[)\d+:", r"\g<1>1:", start.raw_line, count=1)
        assert changed != start.raw_line
        change(start, raw_line=changed)
    elif case == "source":
        # Same-shaped different source for only one of the four pieces.
        other = replace(
            source,
            name="%test_different_source",
            index=max(p.index for p in index.module.instructions) + 1,
        )
        extra.append(other)
        change(start, operand_names=(other.name,))
    elif case == "escape":
        extra.append(
            replace(
                done,
                name="%test_escaping_slice",
                index=max(p.index for p in index.module.instructions) + 1,
            )
        )
    elif case == "incomplete":
        change(done, raw_opcode="slice-start")
    elif case == "side_effect":
        change(op, raw_line=op.raw_line + ", custom_call_has_side_effect=true")
    elif case == "unknown":
        change(op, raw_line=op.raw_line.replace("ConcatBitcast", "UnregisteredCopy"))
    else:
        live = tuple(p for p in live if p.index != op.index)
    mutated = replace(
        index.module,
        instructions=tuple(changes.get(p.index, p) for p in index.module.instructions)
        + tuple(extra),
    )
    with pytest.raises(ValueError, match="dense helper interfaces/completion differ"):
        admission.check_helpers(PrefillHloIndex(mutated), live)


@pytest.mark.parametrize("dtype,dims,count", FAMILIES)
def test_actual_family_count_cap_is_enforced(dense, monkeypatch, dtype, dims, count):
    index, live = dense
    check = admission._check_helper_schedule

    def lower_one_cap(*args, **kwargs):
        # Original graph unchanged; make its observed count exceed the cap.
        kwargs["copy_limits"] = dict(kwargs["copy_limits"])
        kwargs["copy_limits"][("ConcatBitcast", dtype, dims)] = count - 1
        return check(*args, **kwargs)

    monkeypatch.setattr(admission, "_check_helper_schedule", lower_one_cap)
    with pytest.raises(ValueError, match="compiler copy count exceeds bound"):
        admission.check_helpers(index, live)
