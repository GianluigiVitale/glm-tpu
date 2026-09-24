"""DB609 full-model dense-loop interfaces, not numerical admission.

Reuse DB607's fixed-four induction and complete-stack-write checks. Full-model
dead output elimination leaves only output/health stacks, not diagnostic routes.
The 78 prefix/cache and 75 MoE proofs remain independent requirements.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ...optimized.hlo_contract import HloInstruction
from .ws32_batched_commit_hlo import _require, _shape
from .ws32_batched_health_hlo import WriterHealthProof
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _callee_attribute_text, _computation_base
from .ws32_prefill_fixed_loops import fixed_loop_bodies
from .ws32_prefill_hlo_identity import Value, attribute
from .ws32_hlo_boolean_factors import dimensions
from .ws32_rolled_prefill_health_hlo import _value, stacked_health_leaves
from .ws32_rolled_prefill_hlo import RolledTransitions, _rows

LOOP = "greenfield_ws32_prefill_dense_canonical/while"
STACKS = (("bf16", (4, 32, 1536)), ("pred", (4, 32)))


def _slice(value: Value, ranges: str) -> bool:
    found = re.findall(
        r"\bslice=\{([^}]*)\}", _callee_attribute_text(value.op.raw_line)
    )
    return len(found) == 1 and re.sub(r"\s", "", found[0]) == ranges


def _masked_source(
    t: RolledTransitions, value: Value, source: Value, rows: int
) -> Value:
    value = t.resolve(value)
    if rows == 114:
        unstack = t.node(value, "bitcast", 1)
        _require(_shape(unstack, "bf16", (4, 32, 1536)), "canonical tail stack shape")
        pad = t.node(t.arg(unstack, 0), "pad", 2)
        _require(
            _shape(pad, "bf16", (128, 1536))
            and re.findall(
                r"\bpadding=([^,\s]+)", _callee_attribute_text(pad.op.raw_line)
            )
            == ["0_14x0_0"]
            and t.ssa.constant(t.arg(pad, 1), "bf16", 0),
            "canonical tail stack must zero-pad final14 rows",
        )
        value = t.arg(pad, 0)
    selected = t.node(value, "select", 3)
    dims = (4, 32, 1536) if rows == 128 else (114, 1536)
    _require(_shape(selected, "bf16", dims), "canonical selected stack shape")
    zero = t.node(t.arg(selected, 2), "broadcast", 1)
    _require(
        _shape(zero, "bf16", dims) and t.ssa.constant(t.arg(zero, 0), "bf16", 0),
        "canonical masked stack false branch is not zero",
    )
    actual = t.arg(selected, 1)
    if rows == 114:
        trim = t.node(actual, "slice", 1)
        _require(
            _shape(trim, "bf16", (114, 1536)) and _slice(trim, "[0:114],[0:1536]"),
            "canonical tail output crop differs",
        )
        flat = t.node(t.arg(trim, 0), "bitcast", 1)
        _require(_shape(flat, "bf16", (128, 1536)), "canonical tail flatten shape")
        actual = t.arg(flat, 0)
    _require(t.ssa.same(actual, source), "canonical stack is not own completed output")
    mask = t.node(t.arg(selected, 0), "broadcast", 1)
    _require(
        _shape(mask, "pred", dims)
        and dimensions(mask.op) == ((0, 1) if rows == 128 else (0,)),
        "canonical row mask broadcast differs",
    )
    return t.arg(mask, 0)


def _boolean_ref(proof: WriterHealthProof, value: Value) -> tuple:
    b = proof.boolean
    bindings = tuple(_boolean_ref(proof, child) for child in value.bindings)
    if bindings not in b.frame_ids:
        b.frame_ids[bindings] = len(b.frames)
        b.frames.append(bindings)
    return (value.op.index, value.path, b.frame_ids[bindings])


def _live_mask(t: RolledTransitions, value: Value, rows: int) -> None:
    proof = WriterHealthProof(t.index, rows)
    if rows == 114:
        _require(
            proof.live_mask(_boolean_ref(proof, value)) == 0,
            "canonical tail mask is not actual live rows",
        )
        return
    compare = t.node(value, "compare", 2)
    _require(
        _shape(compare, "pred", (4, 32)) and attribute(compare.op, "direction") == "LT",
        "canonical stacked mask comparison differs",
    )
    position = t.node(t.arg(compare, 0), "add", 2)
    bound = t.node(t.arg(compare, 1), "broadcast", 1)
    _require(
        _shape(position, "s32", (4, 32))
        and _shape(bound, "s32", (4, 32))
        and dimensions(bound.op) == ()
        and proof.count(_boolean_ref(proof, t.arg(bound, 0))),
        "canonical stacked mask does not use bounded actual count",
    )

    def iota(v: Value, axis: int) -> bool:
        return (
            v.op.opcode == "iota"
            and _shape(v, "s32", (4, 32))
            and attribute(v.op, "iota_dimension") == str(axis)
        )

    matched = False
    for j in (0, 1):
        inner, outer = t.arg(position, j), t.arg(position, 1 - j)
        if (
            not iota(inner, 1)
            or outer.op.opcode != "multiply"
            or len(outer.op.operand_names) != 2
        ):
            continue
        for k in (0, 1):
            row, scale = t.arg(outer, k), t.arg(outer, 1 - k)
            if (
                iota(row, 0)
                and scale.op.opcode == "broadcast"
                and _shape(scale, "s32", (4, 32))
                and dimensions(scale.op) == ()
                and t.ssa.constant(t.arg(scale, 0), "s32", 32)
            ):
                matched = True
    _require(matched, "canonical stacked mask is not row-major32*i+j")


def _carried_mask(t: RolledTransitions, value: Value, mask: Value, rows: int) -> None:
    value = t.resolve(value)
    if rows == 114:
        stacked = t.node(value, "reshape", 1)
        _require(_shape(stacked, "pred", (4, 32)), "canonical live carry stack shape")
        pad = t.node(t.arg(stacked, 0), "pad", 2)
        _require(
            _shape(pad, "pred", (128,))
            and re.findall(
                r"\bpadding=([^,\s]+)", _callee_attribute_text(pad.op.raw_line)
            )
            == ["0_14"]
            and t.ssa.constant(t.arg(pad, 1), "pred", 0),
            "canonical live carry must pad14 false predicates",
        )
        value = t.arg(pad, 0)
    _require(
        t.ssa.same(value, mask), "canonical immutable live carry uses another mask"
    )


def _stacks(t: RolledTransitions, loop: HloInstruction, body: str) -> None:
    initial = t.resolve(Value(t.index.operand(loop, 0)))
    root, parameter = Value(t.index.roots[body]), t.parameter(body)
    _require(len(loop.result_shapes) == 15, "fullmodel canonical loop arity differs")
    for slot in range(3, 15):
        _require(
            t.ssa.same(t.ssa.leaf(root, slot), t.ssa.leaf(parameter, slot)),
            "canonical immutable carry changed",
        )
    # Only after proving the loop invariant may subsequent bindings unwrap it.
    t.ssa.invariants[loop.index] = frozenset(range(3, 15))
    for slot, (dtype, dims) in enumerate(STACKS, 1):
        shape = loop.result_shapes[slot]
        _require(
            (shape.dtype, shape.dimensions) == (dtype, dims),
            "canonical stack interface differs",
        )
        zero = t.node(t.ssa.leaf(initial, slot), "broadcast", 1)
        _require(
            _shape(zero, dtype, dims) and t.ssa.constant(t.arg(zero, 0), dtype, 0),
            "canonical stack is not zero-initialized",
        )
        write = t.node(t.ssa.leaf(root, slot), "dynamic-update-slice", len(dims) + 2)
        _require(
            _shape(write, dtype, dims)
            and _shape(t.arg(write, 1), dtype, (1, *dims[1:]))
            and t.ssa.same(t.arg(write, 0), t.ssa.leaf(parameter, slot))
            and t.ssa.same(t.arg(write, 2), t.ssa.leaf(parameter, 0))
            and all(
                t.ssa.constant(t.arg(write, j), "s32", 0)
                for j in range(3, len(dims) + 2)
            ),
            "canonical stack write is partial, displaced or non-own",
        )
        if slot == 1:
            tile = t.node(t.arg(write, 1), "bitcast", 1)
            selected = t.node(t.arg(tile, 0), "select", 3)
            sliced = t.node(t.arg(selected, 1), "slice", 1)
            summed = t.arg(sliced, 0)
            _require(
                _shape(selected, "bf16", (32, 1536))
                and _slice(sliced, "[0:32],[0:1536]")
                and _shape(summed, "bf16", (128, 1536))
                and summed.op.opcode == "all-reduce"
                and _computation_base(summed.op.computation) == body
                and (summed.op.op_name or "").endswith("expert_down_reduce/psum"),
                "canonical output must take first32 of own dense reduction",
            )


def _commit_health(
    t: RolledTransitions, loops: dict[int, HloInstruction], rows: int
) -> list[dict]:
    proof = WriterHealthProof(t.index, rows)
    vote = proof.frontier()
    boolean = proof.boolean
    by_id = {loop.index: layer for layer, loop in loops.items()}
    records = {}
    for ref, axis in boolean.factors(vote):
        if axis not in (None, 0):
            continue
        value = _value(boolean, ref)
        if not _shape(value, "pred", (rows,)):
            continue
        trim = None
        if rows == 114:
            if value.op.opcode != "slice" or not _slice(value, "[0:114]"):
                continue
            trim = value.op.name
            value = t.arg(value, 0)
        if value.op.opcode != "reshape" or not _shape(value, "pred", (128,)):
            continue
        for source in stacked_health_leaves(
            t, t.ssa.operand(value, 0), conjunction=True
        ):
            if source.op.index not in by_id or source.path != (2,) or source.bindings:
                continue
            layer = by_id[source.op.index]
            records[layer] = dict(
                layer=layer,
                loop=source.op.name,
                health_slot=2,
                flatten=value.op.name,
                trim=trim,
                domain="ALL" if axis is None else "LIVE",
            )
    _require(
        set(records) == {0, 1, 2}, "global commit lacks own canonical health stacks"
    )
    return [records[layer] for layer in range(3)]


def check_canonical_dense_loops(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    _rows(block_rows)
    report = dict(
        passed=False,
        scope="FULLMODEL_CANONICAL_DENSE_STACKS_AND_HEALTH",
        arithmetic_proof=False,
        numerical_admission=False,
    )
    try:
        bodies = fixed_loop_bodies(
            index, live_instructions, loop_suffix=LOOP, expected_layers=(0, 1, 2)
        )
        t = RolledTransitions(index, block_rows)
        prefixes = t.all_loops(live_instructions)
        loops = {
            bodies[index.callee(op, "body")]: op
            for op in index.module.instructions
            if op.opcode == "while" and (op.op_name or "").endswith(LOOP)
        }
        for layer, loop in sorted(loops.items()):
            _stacks(t, loop, index.callee(loop, "body"))
            initial = t.resolve(Value(index.operand(loop, 0)))
            mask = _masked_source(
                t,
                t.ssa.leaf(initial, 4),
                t.ssa.leaf(Value(prefixes[layer].loop), 4),
                block_rows,
            )
            next_slot = (12, 15, 10)[layer]
            output_mask = _masked_source(
                t,
                t.ssa.leaf(prefixes[layer + 1].initial, next_slot),
                t.ssa.leaf(Value(loop), 1),
                block_rows,
            )
            _require(
                t.ssa.same(mask, output_mask),
                "canonical output and input live masks differ",
            )
            _live_mask(t, mask, block_rows)
            _carried_mask(t, t.ssa.leaf(initial, 3), mask, block_rows)
        report.update(
            passed=True,
            bodies=bodies,
            complete_stack_writes=6,
            immutable_slots=list(range(3, 15)),
            health=_commit_health(t, loops, block_rows),
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report


def inspect_ws32_canonical_prefill_hlo(
    stablehlo: str,
    optimized_hlo: str,
    *,
    block_rows: int,
    expected_stablehlo_sha256: str,
    expected_optimized_hlo_sha256: str,
) -> dict[str, Any]:
    """All historical obligations plus three canonical dense loops, default-off.

    Remains UNREGISTERED: this report alone never enables numerical execution.
    """
    from .ws32_rolled_prefill import _inspect_hlo

    return _inspect_hlo(
        stablehlo,
        optimized_hlo,
        block_rows=block_rows,
        expected_stablehlo_sha256=expected_stablehlo_sha256,
        expected_optimized_hlo_sha256=expected_optimized_hlo_sha256,
        canonical_dense=True,
    )
