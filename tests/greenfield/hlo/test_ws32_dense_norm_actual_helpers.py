"""Replay preserved actual TPU graphs; mutations are structural tests, not TPU runs."""

from hashlib import sha256
import json
import re
from pathlib import Path

import pytest

from scripts.greenfield import ws32_dense_frontier_admission as old
from scripts.greenfield import ws32_dense_norm_admission as admission
from scripts.greenfield import ws32_dense_norm_helpers as helpers

ROOT = Path(
    "/home/gianl/glm-run/greenfield_fp8_ws32_dense_norm_d01_20260909T173035450074269Z/rank0"
)
HASHES = {
    "dense01_norm": "226a23450d34472bb3da4f25e1862a22a18a22626705106cef538a9201ff624e",
    "dense_suffix": "429b2c97f5d1a0ce4aea692a1ba5328206c568e15602c90b8b82c174a98600e8",
    "wk_decode": "95ba3433eef746fa60ca87951b8956c6b5bd182a84bf17fad657b8925aecb563",
    "wk_promote": "1e9b0b3fa2f303a1a8d7708563dcdd0de31d82349a7a9f951f02bda27e0c0611",
}


@pytest.fixture(scope="module")
def originals():
    if not ROOT.is_dir():
        pytest.skip("requires preserved generation-bound norm TPU acquisition")
    result = {}
    for name, digest in HASHES.items():
        text = (ROOT / (name + ".optimized_hlo.txt")).read_text()
        assert sha256(text.encode()).hexdigest() == digest
        result[name] = text
    return result


def change(text, op, before, after):
    # HLO printer interleaves /*index=N*/ comments in tuple operands.
    plain = re.sub(r"/\*index=\d+\*/", "", op.raw_line)
    assert before in plain and before != after
    new = plain.replace(before, after, 1)
    assert text.count(op.raw_line) == 1
    return text.replace(op.raw_line, new, 1)


@pytest.mark.parametrize("name", HASHES)
def test_all_four_actual_programs_and_json(originals, name):
    journal = [
        json.loads(l) for l in (ROOT / "compile_journal.jsonl").read_text().splitlines()
    ]
    memory = next(
        x["compiled_memory"]
        for x in journal
        if x.get("graph") == name and "compiled_memory" in x
    )
    result = admission.inspect_program(
        name, (ROOT / (name + ".stablehlo.mlir")).read_text(), originals[name], memory
    )
    assert result["passed"] and result == json.loads(json.dumps(result))
    assert not result["numerical_promotion"] and not result["performance_claim"]


def test_historical_capture_profile_still_refuses_new_stacks(originals):
    with pytest.raises(ValueError, match="unregistered helper signature"):
        old.inspect_structure("dense01", originals["dense01_norm"])


@pytest.mark.parametrize(
    "case",
    [
        "duplicate_alloc",
        "partial_write",
        "wrong_counter",
        "old_stack_update",
        "old_stack_extra_use",
        "wrong_root_slot",
        "output_escape",
        "wrong_output",
        "nonzero_reduce",
        "wrong_reduce",
        "short_loop",
    ],
)
def test_observation_stack_mutations(originals, case):
    text = originals["dense01_norm"]
    index = old.PrefillHloIndex(old.parse_hlo_module(text))
    loop = next(
        o
        for o in index.module.instructions
        if o.opcode == "while"
        and (o.op_name or "").endswith(old.LOOP)
        and "layer_0/" in o.op_name
    )
    body = index.callee(loop, "body")
    t = old.RolledTransitions(index, 128)
    rt = index.roots[body]
    initial = index.operand(loop, 0)
    write = t.node(t.ssa.leaf(old.Value(rt), 13), "dynamic-update-slice", 5).op
    if case in ("duplicate_alloc", "swapped_initial"):
        x, y = initial.operand_names[13:15]
        args = list(initial.operand_names)
        args[14] = x
        if case == "swapped_initial":
            args[13] = y
        text = change(
            text,
            initial,
            "tuple(" + ", ".join(initial.operand_names) + ")",
            "tuple(" + ", ".join(args) + ")",
        )
    elif case == "partial_write":
        update = index.operand(write, 1)
        text = change(text, update, "f32[1,32,1]", "f32[1,16,1]")
    elif case == "wrong_counter":
        text = change(
            text, write, write.operand_names[2] + ",", write.operand_names[3] + ","
        )
    elif case == "old_stack_update":
        text = change(
            text, write, write.operand_names[1] + ",", write.operand_names[0] + ","
        )
    elif case == "old_stack_extra_use":
        gte = helpers.tuple_leaf(index, t.parameter(body).op, 13)
        text = change(text, rt, rt.operand_names[4] + ",", gte.name + ",")
    elif case == "wrong_root_slot":
        args = list(rt.operand_names)
        args[13], args[14] = args[14], args[13]
        text = change(
            text,
            rt,
            "tuple(" + ", ".join(rt.operand_names) + ")",
            "tuple(" + ", ".join(args) + ")",
        )
    elif case in ("output_escape", "wrong_output"):
        root = index.roots["ENTRY"]
        args = list(root.operand_names)
        args[28] = args[27]
        if case == "wrong_output":
            args[27] = root.operand_names[28]
        text = change(
            text,
            root,
            "tuple(" + ", ".join(root.operand_names) + ")",
            "tuple(" + ", ".join(args) + ")",
        )
    elif case in ("nonzero_reduce", "wrong_reduce"):
        reduced = index.operand(index.operand(index.roots["ENTRY"], 27), 0)
        if case == "nonzero_reduce":
            zero = index.operand(index.operand(reduced, 1), 0)
            text = change(text, zero, "constant(-0)", "constant(1)")
        else:
            reducer = index.roots[index.callee(reduced, "to_apply")]
            text = change(text, reducer, "add(", "maximum(")
    else:
        cond = index.roots[index.callee(loop, "condition")]
        limit = index.operand(cond, 1)
        text = change(text, limit, "constant(4)", "constant(3)")
    new = old.PrefillHloIndex(old.parse_hlo_module(text))
    live = old._live_instruction_closure(new.module.instructions)
    with pytest.raises(ValueError):
        helpers.capture_helpers(new, live)


@pytest.mark.parametrize(
    "case",
    [
        "slice_start",
        "slice_end",
        "pad_offset",
        "pad_value",
        "clamp_min",
        "clamp_max",
        "gather_slice",
        "gather_map",
        "index_escape",
        "index_source",
    ],
)
def test_padded_gather_mutations(originals, case):
    text = originals["dense_suffix"]
    index = old.PrefillHloIndex(old.parse_hlo_module(text))
    ann = next(
        o
        for o in index.module.instructions
        if o.raw_opcode == "custom-call"
        and old._target(o) == "AssumeGatherIndicesInBound"
    )
    comp = old._computation_base(ann.computation)
    if case.startswith("slice_"):
        op = next(o for o in index.computations[comp].values() if o.opcode == "slice")
        text = change(
            text, op, "[0:128]", "[1:129]" if case == "slice_start" else "[0:127]"
        )
    elif case.startswith("pad_") or case.startswith("clamp_"):
        caller = next(
            o
            for o in index.module.instructions
            if o.opcode == "fusion" and index.callee(o, "calls") == comp
        )
        t = old.RolledTransitions(index, 128)
        clamp = t.node(old.Value(index.operand(caller, 1)), "clamp", 3)
        pad = t.node(t.arg(clamp, 1), "pad", 2)
        if case == "pad_offset":
            text = change(text, pad.op, "padding=0_896", "padding=1_895")
        elif case == "pad_value":
            text = change(text, t.arg(pad, 1).op, "constant(2147483647)", "constant(0)")
        else:
            j = 0 if case == "clamp_min" else 2
            broadcast = t.node(t.arg(clamp, j), "broadcast", 1)
            value = t.arg(broadcast, 0).op
            text = change(
                text,
                value,
                "constant(0)" if j == 0 else "constant(127)",
                "constant(1)" if j == 0 else "constant(128)",
            )
    elif case.startswith("gather_"):
        op = next(o for o in index.computations[comp].values() if o.opcode == "gather")
        text = change(
            text,
            op,
            "slice_sizes={1,1536}" if case == "gather_slice" else "start_index_map={0}",
            "slice_sizes={2,1536}" if case == "gather_slice" else "start_index_map={1}",
        )
    elif case == "index_escape":
        root = index.roots[comp]
        text = change(text, root, root.operand_names[0], ann.name)
    else:
        op = next(o for o in index.computations[comp].values() if o.opcode == "gather")
        text = change(text, op, op.operand_names[1], ann.name)
    with pytest.raises(ValueError):
        admission.inspect_suffix(text)
