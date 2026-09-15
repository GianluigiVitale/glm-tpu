"""Fixed diagnostic guards: actual retained WK and synthetic loop mutations.

No synthetic inventory is a TPU compilation or a numerical reproduction result.
"""

from collections import Counter
from dataclasses import replace
import json

from pathlib import Path

import pytest

from scripts.greenfield import ws32_dense_frontier_admission as admission
from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import (
    _check_helper_schedule,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _live_instruction_closure,
)
from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
# Local sealed originals of the retired rolled-prefill worker test (b667f00f).
ROOT = Path("/home/gianl/glm-run") / (
    "greenfield_fp8_ws32_prefill_expert_panel_phase_l6_20260909T033031628458338Z"
)


@pytest.mark.parametrize("name", ["wk_decode", "wk_promote"])
def test_retained_wk_structure(name):
    root = ROOT / "fleet/rank0"
    original = (root / f"{name}.optimized_hlo.txt").read_text()
    report = admission.inspect_structure(name, original)
    assert report["collective_count"] == (2 if name == "wk_decode" else 0)
    assert report["helpers"]["annotation_count"] == (2 if name == "wk_decode" else 0)
    memory = json.loads((root / "runner.json").read_text())["programs"][name][
        "compiled_memory"
    ]
    admission.validate_memory(name, memory)


@pytest.fixture(scope="module")
def wk():
    text = (ROOT / "fleet/rank0/wk_decode.optimized_hlo.txt").read_text()
    module = parse_hlo_module(text)
    return PrefillHloIndex(module), _live_instruction_closure(module.instructions)


@pytest.mark.parametrize(
    "case", ["unknown", "side_effect", "shape", "operand", "scope", "missing", "dead"]
)
def test_wk_helper_refusals(wk, case):
    index, live = wk
    op = next(p for p in index.module.instructions if p.raw_opcode == "custom-call")
    if case == "dead":
        live = tuple(p for p in live if p.index != op.index)
    else:
        changes = {
            "unknown": dict(
                raw_line=op.raw_line.replace("AssumeGatherIndicesInBound", "Unknown")
            ),
            "side_effect": dict(
                raw_line=op.raw_line.replace(
                    ", metadata=", ", custom_call_has_side_effect=true, metadata="
                )
            ),
            "shape": dict(result_shapes=(replace(op.result_shapes[0], dtype="u32"),)),
            "operand": dict(operand_names=()),
            "scope": dict(op_name="unrelated"),
        }
        instructions = (
            tuple(p for p in index.module.instructions if p.index != op.index)
            if case == "missing"
            else tuple(
                replace(p, **changes[case]) if p.index == op.index else p
                for p in index.module.instructions
            )
        )
        index = PrefillHloIndex(replace(index.module, instructions=instructions))
    with pytest.raises(ValueError, match="WK annotation"):
        admission.check_wk_helpers(index, live, "wk_decode")


def annotation_fixture(count):
    lines = [
        "HloModule annotation, num_partitions=32",
        "ENTRY %main {",
        "%x = s32[1024] parameter(0)",
    ]
    for i in range(count):
        lines.append(
            f'%a{i} = s32[1024] custom-call(%x), custom_call_target="AssumeGatherIndicesInBound", metadata={{op_name="jit/gather"}}'
        )
    leaves = ", ".join(f"%a{i}" for i in range(count))
    types = ", ".join("s32[1024]" for _ in range(count))
    lines.append(f"ROOT %out = ({types}) tuple({leaves})\n}}")
    module = parse_hlo_module("\n".join(lines))
    return PrefillHloIndex(module), _live_instruction_closure(module.instructions)


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_bounded_annotations_do_not_widen_historical_exact_counts(count):
    index, live = annotation_fixture(count)
    key = ("AssumeGatherIndicesInBound", "s32", (1024,))
    expected = Counter({key: 2})
    args = dict(
        block_rows=128,
        live_instructions=live,
        expected=expected,
        scratch_check=lambda *a: [],
    )  # no scratch in this fixture
    exact = _check_helper_schedule(index, **args)
    assert exact["passed"] == (count == 2)
    bounded = _check_helper_schedule(index, count_limits={key: 2}, **args)
    assert bounded["passed"] == (count <= 2)
    assert expected[key] == 2  # caller's fixed contract must not be mutated
    for wrong in (True, -1, 2.0):
        assert not _check_helper_schedule(index, count_limits={key: wrong}, **args)[
            "passed"
        ]
    assert not _check_helper_schedule(
        index, count_limits={("Unknown", "s32", (1024,)): 2}, **args
    )["passed"]


def loop_fixture():
    lines = ["HloModule loops, num_partitions=32"]
    for layer in (0, 1):
        lines.extend(
            [
                f"%body{layer} {{",
                "%p = (s32[], f32[]) parameter(0)",
                "%i = s32[] get-tuple-element(%p), index=0",
                "%x = f32[] get-tuple-element(%p), index=1",
                "%one = s32[] constant(1)",
                "%next = s32[] add(%i, %one)",
                "ROOT %out = (s32[], f32[]) tuple(%next, %x)",
                "}",
                f"%cond{layer} {{",
                "%p = (s32[], f32[]) parameter(0)",
                "%i = s32[] get-tuple-element(%p), index=0",
                "%four = s32[] constant(4)",
                "ROOT %test = pred[] compare(%i, %four), direction=LT",
                "}",
            ]
        )
    lines.extend(
        [
            "ENTRY %main {",
            "%zero = s32[] constant(0)",
            "%x = f32[] parameter(0)",
            "%init = (s32[], f32[]) tuple(%zero, %x)",
        ]
    )
    for layer in (0, 1):
        lines.append(
            f'%loop{layer} = (s32[], f32[]) while(%init), condition=%cond{layer}, body=%body{layer}, metadata={{op_name="greenfield_ws32_batched_prefill/layer_{layer}/greenfield_ws32_prefill_rolled_prefix/while"}}'
        )
    lines.append(
        "ROOT %out = ((s32[], f32[]), (s32[], f32[])) tuple(%loop0, %loop1)\n}"
    )
    return "\n".join(lines)


def test_two_loop_induction_and_refusals():
    text = loop_fixture()

    def check(raw):
        m = parse_hlo_module(raw)
        return admission.prefix_bodies(
            PrefillHloIndex(m), _live_instruction_closure(m.instructions)
        )

    assert check(text) == {"%body0": 0, "%body1": 1}
    for old, new in [
        ("constant(4)", "constant(5)"),
        ("constant(1)", "constant(2)"),
        ("constant(0)", "constant(1)"),
        ("direction=LT", "direction=LE"),
        ("add(%i, %one)", "add(%one, %one)"),
        ("compare(%i, %four)", "compare(%four, %i)"),
        ("layer_1/", "layer_0/"),
        ("body=%body1", "body=%body0"),
    ]:
        assert old in text
        with pytest.raises(ValueError):
            check(text.replace(old, new, 1))


def test_fixed_inventory_memory_and_raw_refusals():
    expected = admission.expected_collectives()
    assert sum(expected.values()) == 29
    assert sum(n * (4 if k[0] == "prefix" else 1) for k, n in expected.items()) == 101
    for name in admission.PROGRAMS:
        admission.validate_memory(name, admission.MEMORY_CAPS)
        for key in admission.MEMORY_CAPS:
            for value in (True, -1, admission.MEMORY_CAPS[key] + 1):
                with pytest.raises(ValueError):
                    admission.validate_memory(
                        name, dict(admission.MEMORY_CAPS, **{key: value})
                    )
        with pytest.raises(ValueError, match="raw graph"):
            admission.inspect_program(
                name, "changed raw graph", "", admission.MEMORY_CAPS
            )
    with pytest.raises(ValueError):
        admission.validate_memory("dense01", {})
