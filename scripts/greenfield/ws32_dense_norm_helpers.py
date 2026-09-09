"""Closed observation-buffer uses from the preserved norm diagnostic graphs.

Only four layer0 observation stacks and one padded row-gather annotation are
new. No model arithmetic is interpreted; numerical relevance is still mandatory.
"""

from __future__ import annotations

from typing import Any
import re

from scripts.greenfield import ws32_dense_frontier_admission as a


def users(index: Any, op: Any) -> list[tuple[Any, int]]:
    return [
        (u, n)
        for u in index.computations[a._computation_base(op.computation)].values()
        for n, name in enumerate(u.operand_names)
        if name == op.name
    ]


def only_use(index: Any, source: Any, consumer: Any, position: int) -> None:
    a._require(
        [(u.index, n) for u, n in users(index, source)] == [(consumer.index, position)],
        "norm observation has an extra, displaced or missing use",
    )


def tuple_leaf(index: Any, source: Any, slot: int) -> Any:
    """Require exactly one extraction of this leaf, not a whole-tuple escape."""
    selected = []
    for u, n in users(index, source):
        a._require(
            u.opcode == "get-tuple-element" and n == 0,
            "norm observed tuple escapes without leaf selection",
        )
        if int(a.attribute(u, "index")) == slot:
            selected.append(u)
    a._require(len(selected) == 1, "norm observation leaf extraction differs")
    return selected[0]


def observation_stacks(index: Any, allocations: list, live: tuple) -> list[dict]:
    bodies = a.prefix_bodies(index, live)  # own counter starts0, +1, tests<4
    body = next(b for b, layer in bodies.items() if layer == 0)
    loop = next(
        o
        for o in index.module.instructions
        if o.opcode == "while" and index.callee(o, "body") == body
    )
    initial = index.operand(loop, 0)
    root, entry = index.roots[body], index.roots["ENTRY"]
    t = a.RolledTransitions(index, 128)
    parameter = t.parameter(body)
    counter = t.ssa.leaf(parameter, 0)
    only_use(index, initial, loop, 0)
    seen, records = set(), []
    for slot, output, width in ((13, 27, 1), (14, 28, 1), (17, 31, 1), (18, 32, 1536)):
        allocation = index.operand(initial, slot)
        a._require(
            allocation in allocations
            and allocation.index not in seen
            and a._computation_base(allocation.computation) == "ENTRY"
            and a._shape(a.Value(allocation), "f32", (4, 32, width)),
            "norm observation allocation/slot differs",
        )
        seen.add(allocation.index)
        only_use(index, allocation, initial, slot)
        old = tuple_leaf(index, parameter.op, slot)
        leaf = index.operand(root, slot)
        only_use(index, leaf, root, slot)
        # The observed writes are either a scalar fusion output or one selected
        # member of a two-output fusion. Bind both sides of that exact boundary.
        fusion = leaf
        fused_slot = None
        if leaf.opcode == "get-tuple-element":
            fusion = index.operand(leaf, 0)
            fused_slot = int(a.attribute(leaf, "index"))
            a._require(
                tuple_leaf(index, fusion, fused_slot) == leaf,
                "norm write fusion leaf differs",
            )
        a._require(fusion.opcode == "fusion", "norm stack write must be a fusion")
        fc = index.callee(fusion, "calls")
        callers = [
            o
            for o in index.module.instructions
            if o.opcode == "fusion" and index.callee(o, "calls") == fc
        ]
        a._require(callers == [fusion], "norm stack fusion has another caller")
        positions = [
            j for j, name in enumerate(fusion.operand_names) if name == old.name
        ]
        a._require(len(positions) == 1, "norm old stack fusion binding differs")
        only_use(index, old, fusion, positions[0])
        fp = [
            o
            for o in index.computations[fc].values()
            if o.opcode == "parameter"
            and f"parameter({positions[0]})" in a._callee_attribute_text(o.raw_line)
        ]
        a._require(len(fp) == 1, "norm stack fusion parameter differs")
        write = t.node(t.ssa.leaf(a.Value(root), slot), "dynamic-update-slice", 5)
        a._require(
            a._computation_base(write.op.computation) == fc,
            "norm stack write outside its own fusion",
        )
        # This exclusive base use also rules out reading an uninitialized stack
        # in the update, indices, model arithmetic or another observation stack.
        only_use(index, fp[0], write.op, 0)
        a._require(
            a._shape(write, "f32", (4, 32, width))
            and a._shape(t.arg(write, 1), "f32", (1, 32, width))
            and t.ssa.same(t.arg(write, 0), t.ssa.leaf(parameter, slot))
            and t.ssa.same(t.arg(write, 2), counter)
            and all(t.ssa.constant(t.arg(write, j), "s32", 0) for j in (3, 4)),
            "norm observation write is partial, displaced or non-own",
        )
        fr = index.roots[fc]
        if fused_slot is None:
            a._require(
                fr == write.op and not users(index, write.op),
                "norm scalar write escapes fusion root",
            )
        else:
            a._require(
                fr.opcode == "tuple" and index.operand(fr, fused_slot) == write.op,
                "norm tuple write output differs",
            )
            only_use(index, write.op, fr, fused_slot)
        completed = tuple_leaf(index, loop, slot)
        emitted = index.operand(entry, output)
        only_use(index, emitted, entry, output)
        a._require(
            a._shape(a.Value(emitted), "f32", (1, 1, 128, width))
            and len(emitted.operand_names) == 1,
            "norm packet output interface differs",
        )
        if width == 1536:
            a._require(
                emitted.opcode == "bitcast" and index.operand(emitted, 0) == completed,
                "norm completed sum output differs",
            )
            only_use(index, completed, emitted, 0)
        else:
            reduced = index.operand(emitted, 0)
            a._require(
                emitted.opcode == "reshape"
                and reduced.opcode == "reduce"
                and len(reduced.operand_names) == 2
                and a._shape(a.Value(reduced), "f32", (4, 32))
                and index.operand(reduced, 0) == completed
                and "dimensions={2}" in a._callee_attribute_text(reduced.raw_line),
                "norm singleton observation forwarding differs",
            )
            only_use(index, completed, reduced, 0)
            only_use(index, reduced, emitted, 0)
            index.scalar_add(reduced)
            zero = t.node(a.Value(index.operand(reduced, 1)), "broadcast", 1)
            a._require(
                a._shape(zero, "f32", ())
                and a._shape(t.arg(zero, 0), "f32", ())
                and t.arg(zero, 0).op.opcode == "constant"
                and re.search(
                    r"\bconstant\(-0\)",
                    a._callee_attribute_text(t.arg(zero, 0).op.raw_line),
                ),
                "norm singleton observation reduction is not zero-initialized",
            )
        records.append(
            dict(
                layer=0,
                stack_slot=slot,
                packet_slot=output,
                allocation=allocation.name,
                complete_slice_writes=4,
            )
        )
    a._require(seen == {o.index for o in allocations}, "extra norm observation scratch")
    return records


def capture_helpers(index: Any, live: tuple) -> dict:
    def scratch(index: Any, allocations: list, rows: int, live_ids: set) -> list:
        obs = [o for o in allocations if o.result_shapes[0].dtype == "f32"]
        merge = [o for o in allocations if o.result_shapes[0].dtype != "f32"]
        return a._merge_scratch(
            index, merge, live_ids, layer_ids=(0, 1)
        ) + observation_stacks(index, obs, live)

    return a.check_helpers(
        index,
        live,
        extra_allocations={
            ("AllocateBuffer", "f32", (4, 32, 1)): 3,
            ("AllocateBuffer", "f32", (4, 32, 1536)): 1,
        },
        scratch_check=scratch,
    )


def padded_gather(index: Any, annotation: Any) -> dict:
    """Bound compiler-only index padding, not a new gather/model algorithm."""
    comp = a._computation_base(annotation.computation)
    calls = [
        o
        for o in index.module.instructions
        if o.opcode == "fusion" and index.callee(o, "calls") == comp
    ]
    a._require(
        len(calls) == 1 and a._computation_base(calls[0].computation) == "ENTRY",
        "norm padded gather must have one ENTRY caller",
    )
    call = calls[0]
    a._require(len(call.operand_names) == 2, "norm padded gather caller arity differs")
    ssa = a.RolledTransitions(index, 128)
    param = index.operand(annotation, 0)
    a._require(
        param.opcode == "parameter"
        and "parameter(1)" in a._callee_attribute_text(param.raw_line),
        "norm padded gather input binding differs",
    )
    only_use(index, param, annotation, 0)
    ann_users = users(index, annotation)
    a._require(len(ann_users) == 1, "norm padded indices escape")
    sliced, position = ann_users[0]
    a._require(
        position == 0
        and sliced.opcode == "slice"
        and a._shape(a.Value(sliced), "s32", (128,))
        and "slice={[0:128]}" in a._callee_attribute_text(sliced.raw_line),
        "norm padded gather must consume only original128 indices",
    )
    # The acquired consumer contains only this annotation, slice, identity
    # reshapes/transposes and the one original 128x1536 row gather.
    gathers = [o for o in index.computations[comp].values() if o.opcode == "gather"]
    a._require(len(gathers) == 1, "norm padded gather count differs")
    gather = gathers[0]
    for text in (
        "offset_dims={1}",
        "collapsed_slice_dims={0}",
        "start_index_map={0}",
        "index_vector_dim=1",
        "slice_sizes={1,1536}",
    ):
        a._require(
            text in a._callee_attribute_text(gather.raw_line),
            "norm padded row gather dimensions differ",
        )
    a._require(
        a._shape(a.Value(gather), "bf16", (128, 1536))
        and a._shape(a.Value(index.operand(gather, 0)), "bf16", (128, 1536)),
        "norm padded row gather shape differs",
    )
    value = index.operand(gather, 1)
    chain = []
    while value.opcode in ("reshape", "transpose"):
        a._require(
            a._shape(a.Value(value), "s32", (128,)) and len(value.operand_names) == 1,
            "norm gather index forwarding shape differs",
        )
        if value.opcode == "transpose":
            a._require(
                "dimensions={0}" in a._callee_attribute_text(value.raw_line),
                "norm gather index permutation differs",
            )
        chain.append(value)
        value = index.operand(value, 0)
    a._require(value == sliced, "norm row gather does not consume bounded slice")
    next_op, next_pos = gather, 1
    for value in chain:
        only_use(index, value, next_op, next_pos)
        next_op, next_pos = value, 0
    only_use(index, sliced, next_op, next_pos)
    clamp = ssa.node(a.Value(index.operand(call, 1)), "clamp", 3)
    for j, bound in ((0, 0), (2, 127)):
        broadcast = ssa.node(ssa.arg(clamp, j), "broadcast", 1)
        a._require(
            a._shape(broadcast, "s32", (1024,))
            and ssa.ssa.constant(ssa.arg(broadcast, 0), "s32", bound),
            "norm padded gather clamp bound differs",
        )
    pad = ssa.node(ssa.arg(clamp, 1), "pad", 2)
    a._require(
        a._shape(clamp, "s32", (1024,))
        and a._shape(pad, "s32", (1024,))
        and a._shape(ssa.arg(pad, 0), "s32", (128,))
        and "padding=0_896" in a._callee_attribute_text(pad.op.raw_line)
        and ssa.ssa.constant(ssa.arg(pad, 1), "s32", 2147483647),
        "norm row padding differs",
    )
    return dict(
        original_rows=128,
        padded_indices=1024,
        consumed_indices=128,
        clamp_min=0,
        clamp_max=127,
    )
