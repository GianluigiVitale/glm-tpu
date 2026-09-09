"""Actual DB602 rolled-loop transitions, separate from historical B17/B11 gates.

This module does not authorize numerical execution. A while leaf is a transition,
never the identity of its initial value. Reuse existing SSA and boundary proofs;
do not interpret attention, FP8, routing or score arithmetic here.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from ..sharding.hlo_contract import HloInstruction, HloShape
from .ws32_batched_commit_hlo import _check_commit, _require, _shape
from .ws32_batched_moe_hlo import PrefillHloIndex, _check_moe_route_sums
from .ws32_batched_helper_hlo import (
    _concat_structure,
    _slice_users,
    _target,
    _paired_copy_limits,
)
from .ws32_pallas_one_layer import _callee_attribute_text, _computation_base
from .ws32_prefill_hlo_identity import PrefillIdentity, Value, attribute
from .ws32_batched_health_hlo import WriterHealthProof
from .ws32_hlo_boolean_factors import Ref


PROFILE = "ws32-rolled-b128-b114-transitions-v1"
_SCOPE = "greenfield_ws32_prefill_rolled_prefix/while"
_LAYER = re.compile(r"(?:^|/)greenfield_ws32_batched_prefill/layer_(\d+)(?:/|$)")
_FULL_INDEX = frozenset((0, 1, 2, *range(6, 78, 4)))


def _rows(rows: int) -> None:
    if type(rows) is not int or rows not in (114, 128):
        raise ValueError("rolled profile requires exactly B128 or B114")


def check_rolled_route_sums(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    """Reuse the full75-layer FP32 combine proof, without widening old profiles."""
    _rows(block_rows)
    return _check_moe_route_sums(
        index,
        block_rows=block_rows,
        live_instructions=live_instructions,
        router_bias_tuple=True,
    )


def check_rolled_commit(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
    anchors: dict[str, Value] | None = None,
) -> dict[str, Any]:
    """Atomic rollback/schedule only; selected-row/cache transitions are separate."""
    _rows(block_rows)
    transitions = RolledTransitions(index, block_rows)
    try:
        transitions.all_loops(live_instructions)
    except (ValueError, KeyError, IndexError) as error:
        return dict(passed=False, error=str(error), scope="ROLLED_ATOMIC_COMMIT_ONLY")
    return _check_commit(
        index,
        block_rows=block_rows,
        live_instructions=live_instructions,
        anchors=anchors,
        identity=transitions.ssa,
    )


@dataclass(frozen=True)
class RolledLoop:
    """Actual value anchors, for later cache and health assembly checks."""

    layer: int
    loop: HloInstruction
    parameter: Value
    initial: Value
    root: Value
    cache_slots: tuple[int, ...]
    stack_slots: tuple[int, ...]
    health_slot: int


class RolledIdentity(PrefillIdentity):
    """Only proven immutable while leaves forward; caches/stacks stay opaque."""

    def __init__(
        self, index: PrefillHloIndex, *, canonical_dense: bool = False
    ) -> None:
        super().__init__(index)
        _require(type(canonical_dense) is bool, "canonical dense option must be bool")
        self.canonical_dense = canonical_dense
        self.invariants: dict[int, frozenset[int]] = {}
        self._copy_slice_users: dict | None = None

    def _reconstruct(self, value: Value) -> Value:
        # The tail rematerializes immutable WK/weights through known four-slice
        # copy scaffolding. Reuse its closed-source proof, additionally demand
        # ascending order before treating it as identity. No unknown helper.
        if value.op.result_shapes == (HloShape("bf16", (21, 16, 64, 128)),):
            if self.canonical_dense and self._canonical_cache_copy(value):
                return self.operand(self.operand(self.operand(value, 0), 0), 0)
            return super()._reconstruct(value)
        _require(len(value.op.result_shapes) == 1, "rolled copy must have one result")
        shape = value.op.result_shapes[0]
        # Rolled inputs add these explicit stacked-row and single-layer cache
        # geometries to historical copies. This is copy identity only, never
        # identity of a mutable cache before and after its while transition.
        stacked = {
            HloShape("s32", (4, 32, 2048)),
            HloShape("f32", (4, 32, 2048)),
            HloShape("s32", (4, 32)),
            HloShape("pred", (4, 32)),
            HloShape("bf16", (4, 32, 1536)),
            HloShape("bf16", (4, 32, 64)),
            HloShape("bf16", (16, 64, 640)),
            HloShape("bf16", (16, 64, 128)),
        }
        _require(
            _target(value.op) == "ConcatBitcast"
            and (
                ("ConcatBitcast", shape.dtype, shape.dimensions)
                in _paired_copy_limits()
                or shape in stacked
            ),
            f"rolled invariant forwarding is not a known copy shape: {value.op.name} {shape}",
        )
        if self._copy_slice_users is None:
            concats = [
                o
                for o in self.index.module.instructions
                if o.opcode == "custom-call" and _target(o) == "ConcatBitcast"
            ]
            self._copy_slice_users = _slice_users(self.index, concats)
        _concat_structure(self.index, value.op, self._copy_slice_users)
        dims = shape.dimensions
        axis = 1 if (shape.dtype, dims) == ("bf16", (8192, 64)) else 0
        spans = (
            ((0, 20), (20, 40), (40, 60), (60, 78))
            if dims == (78, 16, 64, 640)
            else tuple(
                (j * dims[axis] // 4, (j + 1) * dims[axis] // 4) for j in range(4)
            )
        )
        first = None
        for j in range(4):
            done = self.operand(value, j)
            start = self.operand(done, 0)
            ranges = re.findall(
                r"\bslice=\{([^}]*)\}", _callee_attribute_text(start.op.raw_line)
            )
            expected = [(0, d) for d in dims]
            expected[axis] = spans[j]
            _require(
                len(ranges) == 1
                and re.sub(r"\s", "", ranges[0])
                == ",".join(f"[{start}:{end}]" for start, end in expected),
                "rolled invariant copy slices reordered",
            )
            if j == 0:
                first = self.operand(start, 0)
        assert first is not None
        return first

    def _canonical_cache_copy(self, value: Value) -> bool:
        """DB609 main's one acquired cache helper rotation, not generic concat.

        The canonical-only caller also proves own cache history/source. Exact
        helper target, all four paired handles, disjoint complete spans and
        exclusive uses are retained. Other permutations still refuse through
        the historical ascending checker. No model arithmetic is inferred.
        """
        if _target(value.op) != "ConcatBitcast" or len(value.op.operand_names) != 4:
            return False
        expected = ((12, 18), (18, 21), (0, 6), (6, 12))
        for j, (start, end) in enumerate(expected):
            done = self.operand(value, j)
            if len(done.op.operand_names) != 1:
                return False
            part = self.operand(done, 0)
            spans = re.findall(
                r"\bslice=\{([^}]*)\}", _callee_attribute_text(part.op.raw_line)
            )
            if (
                len(spans) != 1
                or re.sub(r"\s", "", spans[0])
                != f"[{start}:{end}],[0:16],[0:64],[0:128]"
            ):
                return False
        if self._copy_slice_users is None:
            concats = [
                o
                for o in self.index.module.instructions
                if o.opcode == "custom-call" and _target(o) == "ConcatBitcast"
            ]
            self._copy_slice_users = _slice_users(self.index, concats)
        _concat_structure(self.index, value.op, self._copy_slice_users)
        return True

    def resolve(self, value: Value, **kwargs: Any) -> Value:
        # Shape-changing operations are opaque, not assumed identity. The same
        # exact SSA result can still initialize two immutable carried leaves.
        kwargs.setdefault("stop_at_shape_change", True)
        for _ in range(80):
            value = super().resolve(value, **kwargs)
            if (
                value.op.opcode == "while"
                and len(value.path) == 1
                and value.path[0] in self.invariants.get(value.op.index, ())
            ):
                initial = self.operand(Value(value.op, bindings=value.bindings), 0)
                value = self.leaf(initial, value.path[0])
            else:
                return value
        raise ValueError("unbounded proven loop-invariant forwarding")


class RolledTransitions:
    def __init__(
        self, index: PrefillHloIndex, rows: int, *, canonical_dense: bool = False
    ) -> None:
        _rows(rows)
        self.index, self.rows = index, rows
        self.ssa = RolledIdentity(index, canonical_dense=canonical_dense)

    def resolve(self, value: Value) -> Value:
        return self.ssa.resolve(value, stop_at_shape_change=True)

    def arg(self, value: Value, position: int) -> Value:
        return self.resolve(self.ssa.operand(value, position))

    def node(self, value: Value, opcode: str, arity: int) -> Value:
        value = self.resolve(value)
        _require(
            not value.path
            and value.op.opcode == opcode
            and len(value.op.operand_names) == arity,
            f"rolled expected {opcode}/{arity}, got {value.op.name}:{value.op.opcode}{value.path}",
        )
        return value

    def parameter(self, name: str) -> Value:
        params = [
            p for p in self.index.computations[name].values() if p.opcode == "parameter"
        ]
        _require(
            len(params) == 1
            and re.findall(
                r"\bparameter\((\d+)\)", _callee_attribute_text(params[0].raw_line)
            )
            == ["0"],
            "rolled computation must have one tuple parameter0",
        )
        return Value(params[0])

    @staticmethod
    def integer_node(opcode: str, *args: tuple) -> tuple:
        if opcode in {"minimum", "maximum", "multiply"}:
            args = tuple(sorted(args, key=repr))
        return (opcode, *args)

    def integer_expression(
        self, value: Value, loop: RolledLoop, depth: int = 0
    ) -> tuple | None:
        """Recognize only the S32 tile-count expression; unknown values stay opaque.

        No algebraic reassociation or overflow assumption. The accepted tree
        clips the external count before subtracting 32 times the proven 0..3
        induction variable, so all its arithmetic is in the S32 range.
        """
        if depth >= 24:
            return None
        value = self.resolve(value)
        op = value.op
        recurse = lambda v: self.integer_expression(v, loop, depth + 1)
        if op.index == loop.parameter.op.index and len(value.path) == 1:
            slot = value.path[0]
            if not value.bindings and slot == 0:
                return ("iteration",)
            if not value.bindings and slot in self.ssa.invariants[loop.loop.index]:
                return recurse(self.ssa.leaf(loop.initial, slot))
            return None
        if self.ssa.input(value, 1):
            return (
                ("valid_rows",) if op.result_shapes[1] == HloShape("s32", ()) else None
            )
        if value.path or op.result_shapes not in (
            (HloShape("s32", ()),),
            (HloShape("s32", (1,)),),
        ):
            return None
        if op.opcode == "constant":
            for number in (0, 32, self.rows):
                if self.ssa.constant(value, "s32", number):
                    return ("constant", number)
            return None
        if op.opcode in {"minimum", "maximum", "subtract", "multiply"}:
            if len(op.operand_names) != 2:
                return None
            args = tuple(recurse(self.ssa.operand(value, j)) for j in range(2))
            if any(arg is None for arg in args):
                return None
            return self.integer_node(op.opcode, *args)
        if op.opcode == "dynamic-slice" and len(op.operand_names) == 2:
            source = self.arg(value, 0)
            if source.op.index == loop.parameter.op.index and len(source.path) == 1:
                slot = source.path[0]
                if slot not in self.ssa.invariants[loop.loop.index]:
                    return None
                source = self.resolve(self.ssa.leaf(loop.initial, slot))
            if (
                not source.path
                and source.op.opcode == "iota"
                and source.op.result_shapes == (HloShape("s32", (4,)),)
                and attribute(source.op, "iota_dimension") == "0"
                and recurse(self.ssa.operand(value, 1)) == ("iteration",)
                and re.findall(
                    r"\bdynamic_slice_sizes=\{([^}]*)\}",
                    _callee_attribute_text(op.raw_line),
                )
                == ["1"]
            ):
                return ("iteration",)
        return None

    def tile_counts(self, loop: RolledLoop) -> tuple[Value, ...]:
        """Bind actual body nodes for clip(clip(count,0,B)-32*i,0,32).

        These anchors are inputs to the subsequent writer/health proof, not
        evidence by themselves that each consumer uses the correct count.
        """
        n = self.integer_node
        c = lambda v: ("constant", v)
        expected = n(
            "minimum",
            c(32),
            n(
                "maximum",
                c(0),
                n(
                    "subtract",
                    n("minimum", c(self.rows), n("maximum", c(0), ("valid_rows",))),
                    n("multiply", c(32), ("iteration",)),
                ),
            ),
        )
        candidates = tuple(
            Value(op)
            for op in self.index.computations[
                self.index.callee(loop.loop, "body")
            ].values()
            if op.opcode == "minimum"
            and op.result_shapes == (HloShape("s32", ()),)
            and self.integer_expression(Value(op), loop) == expected
        )
        _require(
            bool(candidates), f"layer{loop.layer}: missing exact tile-count recurrence"
        )
        return candidates

    def inspect(self, loop: HloInstruction) -> RolledLoop:
        layers = _LAYER.findall(loop.op_name or "")
        _require(len(layers) == 1, "rolled loop lacks exact layer scope")
        layer = int(layers[0])
        _require(0 <= layer < 78, "rolled layer out of range")
        _require(
            loop.opcode == "while"
            and len(loop.operand_names) == 1
            and _computation_base(loop.computation) == "ENTRY",
            "rolled loop must be a direct device ENTRY transition",
        )
        body, condition = (
            self.index.callee(loop, name) for name in ("body", "condition")
        )
        parameter, cond_param = self.parameter(body), self.parameter(condition)
        initial = self.resolve(Value(self.index.operand(loop, 0)))
        root = Value(self.index.roots[body])
        _require(
            root.op.opcode == initial.op.opcode == "tuple"
            and len(root.op.operand_names)
            == len(initial.op.operand_names)
            == len(loop.result_shapes)
            and root.op.result_shapes
            == initial.op.result_shapes
            == loop.result_shapes
            == parameter.op.result_shapes
            == cond_param.op.result_shapes,
            "rolled tuple input/body/condition/output shape mismatch",
        )
        _require(
            self.ssa.constant(self.ssa.leaf(initial, 0), "s32", 0),
            "rolled induction must start0",
        )
        counter = self.ssa.leaf(parameter, 0)
        update = self.node(self.ssa.leaf(root, 0), "add", 2)
        _require(
            _shape(update, "s32", ())
            and any(
                self.ssa.same(self.arg(update, j), counter)
                and self.ssa.constant(self.arg(update, 1 - j), "s32", 1)
                for j in (0, 1)
            ),
            "rolled induction must add1 to its own counter",
        )
        cond = self.node(Value(self.index.roots[condition]), "compare", 2)
        _require(
            _shape(cond, "pred", ())
            and attribute(cond.op, "direction") == "LT"
            and self.ssa.same(self.arg(cond, 0), self.ssa.leaf(cond_param, 0))
            and self.ssa.constant(self.arg(cond, 1), "s32", 4),
            "rolled condition must be own counter<4",
        )
        caches = (1, 2, 3) if layer in _FULL_INDEX else (1,)
        health = 6 if layer < 2 else 9 if layer in _FULL_INDEX else 4
        stacks = tuple(range(caches[-1] + 1, health + 1))
        expected = [HloShape("s32", ()), HloShape("bf16", (16, 64, 640))]
        if layer in _FULL_INDEX:
            expected += [HloShape("bf16", (16, 64, 128))] * 2
        expected += [HloShape("bf16", (4, 32, 1536))] * 2
        if layer in _FULL_INDEX and layer >= 2:
            expected += [
                HloShape("s32", (4, 32, 2048)),
                HloShape("s32", (4, 32)),
                HloShape("f32", (4, 32, 2048)),
            ]
        expected += [HloShape("pred", (4, 32))]
        _require(
            tuple(expected) == loop.result_shapes[: health + 1],
            "rolled specialized tuple family drift",
        )
        for slot in stacks:
            shape = expected[slot]
            write = self.node(
                self.ssa.leaf(root, slot),
                "dynamic-update-slice",
                2 + len(shape.dimensions),
            )
            _require(
                write.op.result_shapes == (shape,)
                and self.ssa.operand(write, 1).op.result_shapes
                == (HloShape(shape.dtype, (1, *shape.dimensions[1:])),)
                and self.ssa.same(self.arg(write, 0), self.ssa.leaf(parameter, slot))
                and _shape(self.arg(write, 1), shape.dtype, (1, *shape.dimensions[1:]))
                and self.ssa.same(self.arg(write, 2), counter)
                and all(
                    self.ssa.constant(self.arg(write, j), "s32", 0)
                    for j in range(3, len(write.op.operand_names))
                ),
                f"layer{layer} stack{slot}: non-own, incomplete or displaced row write",
            )
        for slot in caches:
            value = self.resolve(self.ssa.leaf(root, slot))
            _require(
                value.op.opcode == "conditional",
                f"layer{layer} cache{slot}: missing guarded transition",
            )
            _require(
                self.ssa.same(
                    self.ssa.branch(value, 0), self.ssa.leaf(parameter, slot)
                ),
                f"layer{layer} cache{slot}: refusal does not preserve its own prior cache",
            )
        # These leaves are hoisted immutable inputs/constants. Proving identity
        # here permits later recurrence checks to bind them to the initial tuple.
        for slot in range(health + 1, len(loop.result_shapes)):
            forwarded = self.resolve(self.ssa.leaf(root, slot))
            _require(
                forwarded.op.index == parameter.op.index
                and not forwarded.bindings
                and len(forwarded.path) == 1
                and health < forwarded.path[0] < len(loop.result_shapes)
                and self.ssa.same(
                    self.ssa.leaf(initial, slot),
                    self.ssa.leaf(initial, forwarded.path[0]),
                ),
                f"layer{layer} invariant{slot}: changed loop invariant",
            )
        # A copy from another invariant is valid only with identical INITIAL
        # values and no source in the mutable partition. Thus induction proves
        # every such leaf immutable, including mutually forwarded duplicates.
        self.ssa.invariants[loop.index] = frozenset(
            range(health + 1, len(loop.result_shapes))
        )
        return RolledLoop(layer, loop, parameter, initial, root, caches, stacks, health)

    def all_loops(
        self, live_instructions: Sequence[HloInstruction]
    ) -> tuple[RolledLoop, ...]:
        loops = [
            op
            for op in self.index.module.instructions
            if op.opcode == "while" and (op.op_name or "").endswith(_SCOPE)
        ]
        live = {(op.computation, op.name) for op in live_instructions}
        _require(
            len(loops) == 78 and all((o.computation, o.name) in live for o in loops),
            "expected78 live rolled loops",
        )
        result = tuple(sorted((self.inspect(o) for o in loops), key=lambda v: v.layer))
        _require(
            [v.layer for v in result] == list(range(78)),
            "rolled loop layers duplicate or missing",
        )
        return result


class RolledTileHealth(WriterHealthProof):
    """Local implication CONDITIONAL on a nonempty tile and its live health.

    This is not a whole-window commit proof. The outer bridge must establish
    that committed live rows imply the corresponding stacked tile health; an
    empty tile requires its own no-write/rollback argument. Never treat these
    conditional local results as unconditional all-four-iterations permission.
    """

    def __init__(self, transitions: RolledTransitions) -> None:
        super().__init__(transitions.index, 32)
        self.transitions = transitions
        self.loop: RolledLoop | None = None
        self.expected_count: tuple | None = None

    def boolean_ref(self, value: Value) -> Ref:
        b = self.boolean
        bindings = tuple(self.boolean_ref(v) for v in value.bindings)
        if bindings not in b.frame_ids:
            b.frame_ids[bindings] = len(b.frames)
            b.frames.append(bindings)
        return (value.op.index, value.path, b.frame_ids[bindings])

    def value_ref(self, value: Ref) -> Value:
        b = self.boolean
        return Value(
            b.ops[value[0]],
            value[1],
            tuple(self.value_ref(v) for v in b.frames[value[2]]),
        )

    def count(self, value: Ref) -> bool:
        _require(self.loop is not None, "tile health lacks a bound loop")
        return (
            self.transitions.integer_expression(self.value_ref(value), self.loop)
            == self.expected_count
        )

    def inspect(self, loop: RolledLoop) -> dict[str, Any]:
        t, b = self.transitions, self.boolean
        self.loop = loop
        self.expected_count = t.integer_expression(t.tile_counts(loop)[0], loop)
        # Normalized SSA is context-independent; required-true factors are not:
        # the live-mask callback now refers to THIS loop's count and induction.
        b.memo.clear()
        b.nonempty_live = True  # Explicit hypothesis of this local implication.
        stack = t.resolve(t.ssa.leaf(loop.root, loop.health_slot))
        update = t.arg(stack, 1)
        known = b.factors(self.boolean_ref(update), axis=1)
        writers = []
        for slot in loop.cache_slots:
            writer = self.boolean_ref(t.resolve(t.ssa.leaf(loop.root, slot)))
            conditional = (writer[0], (), writer[2])
            predicate = self.predicate(conditional)
            passed = b.implies(known, predicate)
            writers.append(
                dict(
                    slot=slot,
                    conditional=b.ops[writer[0]].name,
                    predicate=b.ops[predicate[0]].name,
                    passed=passed,
                )
            )
        return dict(
            layer=loop.layer,
            passed=all(w["passed"] for w in writers),
            health_update=update.op.name,
            writers=writers,
        )


def check_rolled_tile_health(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    """Check actual local writer predicates, retaining explicit assumptions."""
    _rows(block_rows)
    report: dict[str, Any] = dict(
        passed=False,
        scope="NONEMPTY_TILE_LIVE_HEALTH_IMPLIES_OWN_WRITER_PREDICATES_ONLY",
        assumptions=["TILE_COUNT_GT_ZERO", "ALL_LIVE_TILE_HEALTH_TRUE"],
        not_proven=[
            "GLOBAL_COMMIT_TO_TILE_HEALTH",
            "EMPTY_TILE_NO_WRITES",
            "WRITER_ACCEPTED_VALUE_LINEAGE",
            "NUMERICAL_OR_MEMORY_ADMISSION",
        ],
    )
    try:
        transitions = RolledTransitions(index, block_rows)
        loops = transitions.all_loops(live_instructions)
        proof = RolledTileHealth(transitions)
        layers = [proof.inspect(loop) for loop in loops]
        report.update(layers=layers, passed=all(layer["passed"] for layer in layers))
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report


def check_rolled_loops(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    """Prove loop count, own rollback and complete stack writes, not arithmetic."""
    _rows(block_rows)
    report: dict[str, Any] = dict(
        passed=False,
        profile=PROFILE,
        scope="ROLLED_INDUCTION_STACK_WRITES_CACHE_ROLLBACK_ONLY",
        not_proven=[
            "TILE_COUNT_TO_WRITER_AND_HEALTH_CONSUMERS",
            "CACHE_ACCEPTED_WRITE_LINEAGE",
            "STACK_TO_COMMIT_HEALTH",
            "FINAL_SELECTED_ROW",
            "NUMERICAL_CORRECTNESS_OR_MEMORY",
        ],
    )
    try:
        transitions = RolledTransitions(index, block_rows)
        loops = transitions.all_loops(live_instructions)
        counts = {v.layer: transitions.tile_counts(v) for v in loops}
        report.update(
            passed=True,
            block_rows=block_rows,
            iterations=4,
            layers=[
                dict(
                    layer=v.layer,
                    loop=v.loop.name,
                    cache_slots=list(v.cache_slots),
                    stack_slots=list(v.stack_slots),
                    health_slot=v.health_slot,
                    tile_count_nodes=[node.op.name for node in counts[v.layer]],
                )
                for v in loops
            ],
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
