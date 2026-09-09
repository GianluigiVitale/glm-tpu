"""Saved DB607 compiler regression/mutations, not another TPU acquisition."""

from hashlib import sha256
import json
from pathlib import Path
import re

import pytest

from scripts.greenfield import ws32_dense_canonical_admission as c

ROOT = Path(
    "/home/gianl/glm-run/greenfield_fp8_ws32_dense_canonical_compile_20260909T185544706417041Z/rank0"
)
OPTIMIZED_SHA = "401c3b2c9dbf55932057651f46230cb903704ff9edd9b7e5d272d789d13dc754"


@pytest.fixture(scope="module")
def original():
    if not ROOT.is_dir():
        pytest.skip("requires generation-bound DB607 original compiler evidence")
    text = (ROOT / "dense01_canonical.optimized_hlo.txt").read_text()
    assert sha256(text.encode()).hexdigest() == OPTIMIZED_SHA
    return text


def inspect(text):
    return c.a.inspect_structure("dense01", text, suffix_check=c.canonical_suffix)


def replace_op(text, op, before, after):
    plain = re.sub(r"/\*index=\d+\*/", "", op.raw_line)
    assert before != after and before in plain and text.count(op.raw_line) == 1
    return text.replace(op.raw_line, plain.replace(before, after, 1), 1)


def test_actual_graph_with_actual_compiled_memory_and_json(original):
    journal = [
        json.loads(l) for l in (ROOT / "compile_journal.jsonl").read_text().splitlines()
    ]
    memory = next(x["compiled_memory"] for x in journal if "compiled_memory" in x)
    report = c.inspect_program(
        c.protocol.GRAPH,
        (ROOT / "dense01_canonical.stablehlo.mlir").read_text(),
        original,
        memory,
    )
    assert report == json.loads(json.dumps(report))
    s = report["structure"]
    assert s["kernels"]["kernel_count"] == 24
    assert s["helpers"]["helper_count"] == 44
    assert s["collectives"]["static_instruction_count"] == 23
    assert s["collectives"]["four_iteration_schedule_leaf_pairs"] == 113
    assert len(s["canonical_suffix"]["loops"]) == 2
    assert not report["numerical_promotion"] and not report["performance_claim"]


def test_old_inspector_does_not_silently_admit_new_suffix(original):
    with pytest.raises(ValueError, match="not ENTRY"):
        c.a.inspect_structure("dense01", original)


@pytest.mark.parametrize(
    "case",
    [
        "trip_count",
        "counter_update",
        "counter_initial",
        "immutable_carry",
        "write_counter",
        "write_zero_index",
        "write_old_stack",
        "partial_write",
        "reduction_slice",
        "output_swap",
        "output_initial",
        "route_swap",
        "health_bypass",
        "health_or",
        "suffix_owner",
        "loop_owner",
        "host",
        "partitions",
    ],
)
def test_actual_structural_mutations(original, case):
    a = c.a
    index = a.PrefillHloIndex(a.parse_hlo_module(original))
    live = a._live_instruction_closure(index.module.instructions)
    bodies = a.prefix_bodies(index, live, loop_suffix=c.LOOP)
    body = next(b for b, layer in bodies.items() if layer == 0)
    loop = next(
        o
        for o in index.module.instructions
        if o.opcode == "while" and index.callee(o, "body") == body
    )
    root, entry = index.roots[body], index.roots["ENTRY"]
    initial = index.operand(loop, 0)
    t = a.RolledTransitions(index, 128)
    if case == "trip_count":
        cond = index.roots[index.callee(loop, "condition")]
        op = t.arg(t.node(a.Value(cond), "compare", 2), 1).op
        text = replace_op(original, op, "constant(4)", "constant(3)")
    elif case == "counter_update":
        op = t.node(t.ssa.leaf(a.Value(root), 0), "add", 2).op
        text = replace_op(original, op, " add(", " subtract(")
    elif case == "counter_initial":
        op = t.resolve(t.ssa.leaf(a.Value(initial), 0)).op
        text = replace_op(original, op, "constant(0)", "constant(1)")
    elif case == "immutable_carry":
        text = replace_op(
            original, root, index.operand(root, 8).name, index.operand(root, 10).name
        )
    elif case in (
        "write_counter",
        "write_zero_index",
        "write_old_stack",
        "partial_write",
    ):
        write = t.node(t.ssa.leaf(a.Value(root), 1), "dynamic-update-slice", 5).op
        if case == "write_counter":
            text = replace_op(
                original, write, write.operand_names[2], write.operand_names[3]
            )
        elif case == "write_zero_index":
            text = replace_op(
                original, write, write.operand_names[3], write.operand_names[2]
            )
        elif case == "partial_write":
            op = index.operand(write, 1)
            text = replace_op(original, op, "bf16[1,32,1536]", "bf16[1,16,1536]")
        else:
            # Same-shaped immutable input cannot stand in for the old output stack.
            old = index.operand(root, 1)  # fusion taking own stack as operand0
            carry = index.operand(root, 6)
            text = replace_op(original, old, old.operand_names[0], carry.name)
    elif case == "reduction_slice":
        write = t.node(t.ssa.leaf(a.Value(root), 1), "dynamic-update-slice", 5)
        tile = t.node(t.arg(write, 1), "bitcast", 1)
        selected = t.node(t.arg(tile, 0), "select", 3)
        op = t.node(t.arg(selected, 1), "slice", 1).op
        text = replace_op(original, op, "[0:32]", "[32:64]")
    elif case in ("output_swap", "route_swap", "health_bypass"):
        slot = {"output_swap": 0, "route_swap": 8, "health_bypass": 10}[case]
        text = replace_op(
            original,
            entry,
            index.operand(entry, slot).name,
            index.operand(entry, slot + 12).name,
        )
    elif case == "output_initial":
        out = t.node(t.ssa.leaf(a.Value(entry), 0), "select", 3)
        flat = t.resolve(t.arg(out, 1))
        completed = t.arg(flat, 0)
        assert completed.op == loop and completed.path == (1,)
        # Redirect the actual ENTRY GTE only; do not modify the loop itself.
        op = next(
            o
            for o in index.computations["ENTRY"].values()
            if o.opcode == "get-tuple-element"
            and o.operand_names == (loop.name,)
            and a.attribute(o, "index") == "1"
        )
        text = replace_op(original, op, loop.name, initial.name)
    elif case == "health_or":
        op = t.node(t.ssa.leaf(a.Value(entry), 10), "and", 2).op
        text = replace_op(original, op, " and(", " or(")
    elif case == "suffix_owner":
        op = next(
            o for o in index.computations[body].values() if o.opcode == "all-reduce"
        )
        text = replace_op(original, op, "/layer_0/", "/layer_1/")
    elif case == "loop_owner":
        text = replace_op(original, loop, "/layer_0/", "/layer_1/")
    elif case == "host":
        text = original.replace(
            entry.raw_line, "%bad = s32[] infeed()\n" + entry.raw_line
        )
    else:
        text = original.replace("num_partitions=32", "num_partitions=16", 1)
    with pytest.raises(ValueError):
        inspect(text)


@pytest.mark.parametrize("case", ["raw", "role", "memory", "alias"])
def test_raw_role_and_memory_fail_closed(original, case):
    stable = (ROOT / "dense01_canonical.stablehlo.mlir").read_text()
    name, memory = c.protocol.GRAPH, dict(c.a.MEMORY_CAPS)
    if case == "raw":
        stable += "\n"
    elif case == "role":
        name = "dense01"
    elif case == "memory":
        memory["temp_size_in_bytes"] += 1
    else:
        memory["alias_size_in_bytes"] = 1
    with pytest.raises(ValueError):
        c.inspect_program(name, stable, original, memory)
