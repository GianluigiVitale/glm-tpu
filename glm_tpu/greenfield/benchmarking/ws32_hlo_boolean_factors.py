"""Conservative required-true factors with explicit live-row domains.

This is boolean implication, not operand ancestry or an arithmetic interpreter.
Unknown operations stay opaque. Fusion frames are interned integer identities,
so selected tuple leaves retain their actual caller without recursive keys.
"""

from __future__ import annotations

import re
from typing import Callable

from ...optimized.hlo_contract import HloInstruction, HloShape
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _callee_attribute_text
from .ws32_prefill_hlo_identity import attribute


Ref = tuple[int, tuple[int, ...], int]
Factor = tuple[Ref, int | None]


def dimensions(op: HloInstruction) -> tuple[int, ...]:
    matches = re.findall(
        r"\bdimensions=\{([^}]*)\}", _callee_attribute_text(op.raw_line)
    )
    if len(matches) != 1:
        raise ValueError("missing/ambiguous boolean dimensions")
    return tuple(int(x.strip()) for x in matches[0].split(",") if x.strip())


class BooleanFactors:
    """Necessary conjuncts; ALL=None, LIVE=axis of an authenticated row mask.

    A factor (value, axis) means every element on LIVE rows is true, not every
    padded element. Only a caller-proven nonempty live set permits a broadcast
    scalar to become an ALL obligation. ``live_mask`` must identify the exact
    mask and its row axis from operands, not names; it does not authorize runs.
    """

    def __init__(
        self,
        index: PrefillHloIndex,
        *,
        live_rows: int,
        live_mask: Callable[[Ref], int | None],
        nonempty_live: bool,
    ) -> None:
        self.index = index
        self.ops = {op.index: op for op in index.module.instructions}
        self.frames: list[tuple[Ref, ...]] = [()]
        self.frame_ids: dict[tuple[Ref, ...], int] = {(): 0}
        self.normal: dict[Ref, Ref] = {}
        self.memo: dict[Factor, frozenset[Factor]] = {}
        self.live_rows, self.live_mask = live_rows, live_mask
        self.nonempty_live = nonempty_live

    def ref(self, op: HloInstruction, path: tuple[int, ...] = ()) -> Ref:
        return (op.index, path, 0)

    def operand(self, value: Ref, number: int) -> Ref:
        return (self.index.operand(self.ops[value[0]], number).index, (), value[2])

    def resolve(self, value: Ref) -> Ref:
        trail = []
        seen = set()
        while value not in self.normal:
            if value in seen or len(trail) >= 10000:
                raise ValueError("cyclic/unbounded boolean forwarding")
            seen.add(value)
            trail.append(value)
            op, path, frame = self.ops[value[0]], value[1], value[2]
            if op.opcode == "tuple" and path:
                arg = self.operand(value, path[0])
                value = (arg[0], path[1:], arg[2])
            elif op.opcode == "get-tuple-element":
                arg = self.operand(value, 0)
                value = (arg[0], (int(attribute(op, "index")),) + path, arg[2])
            elif op.opcode == "parameter" and frame:
                number = re.findall(
                    r"\bparameter\((\d+)\)", _callee_attribute_text(op.raw_line)
                )
                if len(number) != 1:
                    raise ValueError("ambiguous boolean parameter")
                arg = self.frames[frame][int(number[0])]
                value = (arg[0], arg[1] + path, arg[2])
            elif op.opcode == "fusion":
                bindings = tuple(
                    self.operand(value, j) for j in range(len(op.operand_names))
                )
                if bindings not in self.frame_ids:
                    self.frame_ids[bindings] = len(self.frames)
                    self.frames.append(bindings)
                root = self.index.roots[self.index.callee(op, "calls")]
                value = (root.index, path, self.frame_ids[bindings])
            elif not path and op.opcode == "copy" and len(op.operand_names) == 1:
                arg = self.operand(value, 0)
                if self.shape(value) != self.shape(arg):
                    raise ValueError("boolean copy shape changed")
                value = arg
            elif not path and op.opcode == "copy-done":
                start = self.operand(value, 0)
                sop = self.ops[start[0]]
                if sop.opcode != "copy-start" or len(sop.operand_names) != 1:
                    raise ValueError("unfinished boolean copy")
                arg = self.operand(start, 0)
                if self.shape(value) != self.shape(arg):
                    raise ValueError("boolean asynchronous copy shape changed")
                value = arg
            else:
                self.normal[value] = value
                break
        result = self.normal[value]
        for old in trail:
            self.normal[old] = result
        return result

    def shape(self, value: Ref) -> HloShape:
        op, path = self.ops[value[0]], value[1]
        if not path and len(op.result_shapes) == 1:
            return op.result_shapes[0]
        # Acquired ENTRY and opaque conditionals have flat tuples. Do not
        # infer nested tuple flattening from the parser's flattened shapes.
        header = op.raw_line.split("=", 1)[1].split(op.raw_opcode + "(", 1)[0]
        header = re.sub(r"\{[^}]*\}", "", header)
        if len(path) == 1 and header.count("(") == 1:
            return op.result_shapes[path[0]]
        raise ValueError(
            f"boolean shape needs a resolved scalar/array leaf: {op.name}{path}"
        )

    def scalar_and_reducer(self, op: HloInstruction) -> None:
        callee = self.index.callee(op, "to_apply")
        nodes = list(self.index.computations[callee].values())
        root = self.index.roots[callee]
        params = [p for p in nodes if p.opcode == "parameter"]
        if not (
            len(nodes) == 3
            and len(params) == 2
            and root.opcode == "and"
            and set(root.operand_names) == {p.name for p in params}
            and len(root.operand_names) == 2
            and all(p.result_shapes == (HloShape("pred", ()),) for p in nodes)
            and sorted(re.findall(r"\bparameter\((\d+)\)", p.raw_line) for p in params)
            == [["0"], ["1"]]
        ):
            raise ValueError("boolean reduction is not exact scalar AND")

    def _key(self, value: Ref, axis: int | None) -> Factor:
        value = self.resolve(value)
        shape = self.shape(value)
        if shape.dtype != "pred" or shape.element_count == 0:
            raise ValueError("boolean obligation requires nonempty PRED")
        if axis is not None and (
            not 0 <= axis < len(shape.dimensions)
            or shape.dimensions[axis] != self.live_rows
        ):
            raise ValueError("live-domain row axis drifted")
        return (value, axis)

    def _rule(self, key: Factor) -> tuple[str, list[Factor]]:
        value, axis = key
        op = self.ops[value[0]]
        shape = self.shape(value)
        arg = lambda j: self.resolve(self.operand(value, j))
        child = lambda j, domain=axis: self._key(arg(j), domain)
        if value[1]:
            return "atom", []
        if op.opcode in ("and", "or") and len(op.operand_names) == 2:
            if any(self.shape(arg(j)) != shape for j in range(2)):
                raise ValueError("boolean binary shape drift")
            if op.opcode == "or":
                for j in range(2):
                    mask = arg(j)
                    maskop = self.ops[mask[0]]
                    if maskop.opcode == "not" and len(maskop.operand_names) == 1:
                        live_axis = self.live_mask(self.resolve(self.operand(mask, 0)))
                        if live_axis is not None and axis in (None, live_axis):
                            return "union", [child(1 - j, live_axis)]
                return "intersection", [child(0), child(1)]
            return "union", [child(0), child(1)]
        if op.opcode == "select" and len(op.operand_names) == 3:
            if all(self.shape(arg(j)) == shape for j in (1, 2)):
                return "intersection", [child(1), child(2)]
        if op.opcode == "broadcast" and len(op.operand_names) == 1:
            source = self.shape(arg(0))
            mapping = dimensions(op)
            if not (
                source.dtype == "pred"
                and len(mapping) == len(source.dimensions)
                and len(set(mapping)) == len(mapping)
                and all(0 <= d < len(shape.dimensions) for d in mapping)
                and tuple(shape.dimensions[d] for d in mapping) == source.dimensions
            ):
                raise ValueError("boolean broadcast does not cover its full input")
            if axis is None:
                return "union", [child(0, None)]
            if axis in mapping:
                return "union", [child(0, mapping.index(axis))]
            if self.nonempty_live:
                return "union", [child(0, None)]
            return "atom", []
        if op.opcode == "reduce" and len(op.operand_names) == 2:
            self.scalar_and_reducer(op)
            source, initial = self.shape(arg(0)), arg(1)
            initop = self.ops[initial[0]]
            reduced = dimensions(op)
            kept = tuple(j for j in range(len(source.dimensions)) if j not in reduced)
            if not (
                source.dtype == "pred"
                and source.element_count > 0
                and len(set(reduced)) == len(reduced)
                and all(0 <= d < len(source.dimensions) for d in reduced)
                and tuple(source.dimensions[j] for j in kept) == shape.dimensions
                and self.shape(initial) == HloShape("pred", ())
                and initop.opcode == "constant"
                and re.findall(
                    r"\bconstant\(([^)]*)\)", _callee_attribute_text(initop.raw_line)
                )
                == ["true"]
            ):
                raise ValueError("boolean reduction initializer/axes drift")
            return "union", [child(0, None if axis is None else kept[axis])]
        if (
            op.opcode in ("reshape", "bitcast", "transpose")
            and len(op.operand_names) == 1
        ):
            source = self.shape(arg(0))
            if source.dtype != "pred" or source.element_count != shape.element_count:
                raise ValueError("boolean shape forwarding lost elements")
            if op.opcode == "bitcast" and source.element_count != 1:
                # Only pred[1]→pred[] is used by the acquired vote. Equal
                # logical counts alone do not exclude physical padding lanes.
                return "atom", []
            if op.opcode == "transpose":
                mapping = dimensions(op)
                if (
                    sorted(mapping) != list(range(len(source.dimensions)))
                    or tuple(source.dimensions[d] for d in mapping) != shape.dimensions
                ):
                    raise ValueError("boolean transpose permutation drift")
                return "union", [child(0, None if axis is None else mapping[axis])]
            if axis is None:
                return "union", [child(0, None)]
            # LIVE shape forwarding is not ALL forwarding: only unchanged
            # logical shape is admitted here; general repartition stays opaque.
            if source.dimensions == shape.dimensions and op.opcode == "reshape":
                return "union", [child(0)]
        return "atom", []

    def factors(self, value: Ref, axis: int | None = None) -> frozenset[Factor]:
        target = self._key(value, axis)
        stack = [target]
        active: set[Factor] = set()
        while stack:
            key = stack[-1]
            if key in self.memo:
                stack.pop()
                active.discard(key)
                continue
            if len(self.memo) + len(stack) > 100000:
                raise ValueError("boolean proof exceeds bounded profile")
            rule, children = self._rule(key)
            if rule == "intersection":
                ref = key[0]
                op = self.ops[ref[0]]
                # Array OR/vector-select can stitch together true elements
                # from different branches. Whole-array common implications
                # need not be pointwise common after transpose/reshape.
                scalar_choice = (
                    self.shape(ref).dimensions == ()
                    if op.opcode == "or"
                    else self.shape(self.resolve(self.operand(ref, 0))).dimensions == ()
                )
                if not scalar_choice:
                    rule, children = "atom", []
            missing = [c for c in children if c not in self.memo]
            if missing:
                active.add(key)
                if missing[0] in active:
                    raise ValueError("cyclic boolean implication")
                stack.append(missing[0])
                continue
            if rule == "atom":
                result = frozenset({key})
            elif rule == "union":
                result = frozenset().union(*(self.memo[c] for c in children))
            else:
                result = self.memo[children[0]].intersection(
                    *(self.memo[c] for c in children[1:])
                )
            # The expression itself is also required. In particular OR only
            # exposes common necessary children, never replaces its own atom.
            self.memo[key] = result | {key}
        return self.memo[target]

    def implies(
        self, known: frozenset[Factor], value: Ref, axis: int | None = None
    ) -> bool:
        """Prove a consequent, not a subset of its merely necessary factors.

        For example, the necessary-factor intersection of p OR q is empty;
        that does NOT mean any antecedent implies p OR q. A consequent OR needs
        one proven branch, AND needs both. Unknowns require exact known atoms.
        """
        target = self._key(value, axis)
        memo: dict[Factor, bool] = {}
        stack = [target]
        active: set[Factor] = set()
        while stack:
            key = stack[-1]
            if key in memo:
                stack.pop()
                active.discard(key)
                continue
            if len(memo) + len(stack) > 100000:
                raise ValueError("boolean consequent exceeds bounded profile")
            ref, domain = key
            op = self.ops[ref[0]]
            if key in known or (domain is not None and (ref, None) in known):
                memo[key] = True
                continue
            if op.opcode == "constant" and re.findall(
                r"\bconstant\(([^)]*)\)", _callee_attribute_text(op.raw_line)
            ) == ["true"]:
                memo[key] = True
                continue
            rule, children = self._rule(key)
            if rule == "atom":
                memo[key] = False
                continue
            missing = [c for c in children if c not in memo]
            if missing:
                active.add(key)
                if missing[0] in active:
                    raise ValueError("cyclic boolean consequent")
                stack.append(missing[0])
                continue
            # Forwarding/reduction/exact mask rules are equivalences in their
            # declared domains. For select both possible branches must hold.
            memo[key] = (
                any(memo[c] for c in children)
                if rule == "intersection" and op.opcode == "or"
                else all(memo[c] for c in children)
            )
        return memo[target]
