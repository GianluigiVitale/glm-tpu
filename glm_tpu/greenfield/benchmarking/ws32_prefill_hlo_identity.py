"""Bounded SSA identity for the acquired prefill commit boundary, not a HLO interpreter."""

from __future__ import annotations

from dataclasses import dataclass
import re

from ...optimized.hlo_contract import HloInstruction
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _callee_attribute_text, _computation_base


@dataclass(frozen=True)
class Value:
    op: HloInstruction
    path: tuple[int, ...] = ()
    bindings: tuple["Value", ...] = ()


def attribute(op: HloInstruction, name: str) -> str:
    matches = re.findall(
        r"\b" + name + r"=([\w.-]+)", _callee_attribute_text(op.raw_line)
    )
    if len(matches) != 1:
        raise ValueError(f"missing/ambiguous {name} on {op.name}")
    return matches[0]


class PrefillIdentity:
    """Follow exact tuple bindings/copies; leave all nonidentity arithmetic opaque.

    No general reshape equivalence or custom-call allowance. The sole custom
    identity is the acquired index-cache ConcatBitcast of four ordered slices.
    """

    def __init__(self, index: PrefillHloIndex) -> None:
        self.index = index

    def operand(self, value: Value, position: int) -> Value:
        return Value(self.index.operand(value.op, position), bindings=value.bindings)

    def leaf(self, value: Value, position: int) -> Value:
        return Value(value.op, value.path + (position,), value.bindings)

    def branch(self, value: Value, branch: int) -> Value:
        op = value.op
        matches = re.findall(
            r"\bbranch_computations=\{([^}]*)\}", _callee_attribute_text(op.raw_line)
        )
        if (
            op.opcode != "conditional"
            or len(op.operand_names) != 3
            or len(matches) != 1
        ):
            raise ValueError("expected exactly two conditional branches")
        names = [name.strip() for name in matches[0].split(",")]
        if len(names) != 2 or branch not in (0, 1):
            raise ValueError("invalid conditional branch binding")
        return Value(
            self.index.roots[names[branch]],
            value.path,
            (self.operand(value, branch + 1),),
        )

    @staticmethod
    def _identity_shape(output: HloInstruction, source: HloInstruction) -> bool:
        if len(output.result_shapes) != 1 or len(source.result_shapes) != 1:
            return False
        a, b = output.result_shapes[0], source.result_shapes[0]
        return a.dtype == b.dtype and (
            a.dimensions == b.dimensions
            or (a.dimensions in ((), (1,)) and b.dimensions in ((), (1,)))
        )

    def resolve(
        self,
        value: Value,
        *,
        reconstruct: bool = True,
        stop_at_shape_change: bool = False,
    ) -> Value:
        for _ in range(128):
            op, path = value.op, value.path
            if op.opcode == "tuple" and path:
                arg = self.operand(value, path[0])
                value = Value(arg.op, path[1:], arg.bindings)
            elif op.opcode == "get-tuple-element":
                arg = self.operand(value, 0)
                value = Value(
                    arg.op, (int(attribute(op, "index")),) + path, arg.bindings
                )
            elif op.opcode == "parameter" and value.bindings:
                numbers = re.findall(
                    r"\bparameter\((\d+)\)", _callee_attribute_text(op.raw_line)
                )
                if len(numbers) != 1:
                    raise ValueError("invalid parameter binding")
                arg = value.bindings[int(numbers[0])]
                value = Value(arg.op, arg.path + path, arg.bindings)
            elif op.opcode == "fusion":
                value = Value(
                    self.index.roots[self.index.callee(op, "calls")],
                    path,
                    tuple(self.operand(value, i) for i in range(len(op.operand_names))),
                )
            elif not path and op.opcode in {"copy", "bitcast", "reshape"}:
                arg = self.operand(value, 0)
                if len(op.operand_names) != 1 or not self._identity_shape(op, arg.op):
                    if stop_at_shape_change and op.opcode in {"bitcast", "reshape"}:
                        return value
                    raise ValueError("nonidentity copy/shape forwarding")
                if op.opcode == "bitcast" and (
                    op.result_shapes[0].dimensions not in ((), (1,))
                    or arg.op.result_shapes[0].dimensions not in ((), (1,))
                ):
                    if stop_at_shape_change:
                        return value
                    raise ValueError("array-layout bitcast is not proven identity")
                value = arg
            elif not path and op.opcode == "copy-done":
                start = self.operand(value, 0)
                if start.op.opcode != "copy-start" or len(start.op.operand_names) != 1:
                    raise ValueError("invalid asynchronous copy")
                arg = self.operand(start, 0)
                if not self._identity_shape(op, arg.op):
                    raise ValueError("asynchronous copy shape changed")
                value = arg
            elif not path and op.opcode == "custom-call" and reconstruct:
                value = self._reconstruct(value)
            else:
                return value
        raise ValueError("unbounded identity forwarding")

    def _reconstruct(self, value: Value) -> Value:
        op = value.op
        # Only the actual attribute before metadata/config, never a quoted label.
        header = op.raw_line.split(", metadata=", 1)[0].split(", backend_config=", 1)[0]
        if re.findall(r'\bcustom_call_target="([^"]+)"', header) != ["ConcatBitcast"]:
            raise ValueError("unsupported custom-call in identity path")
        if (
            len(op.result_shapes) != 1
            or (
                op.result_shapes[0].dtype,
                op.result_shapes[0].dimensions,
            )
            != ("bf16", (21, 16, 64, 128))
            or len(op.operand_names) != 4
        ):
            raise ValueError("unacquired cache reconstruction shape")
        source = None
        for i, (begin, end) in enumerate(((0, 6), (6, 12), (12, 18), (18, 21))):
            part = self.resolve(self.operand(value, i), reconstruct=False)
            if (
                part.op.opcode != "slice-done"
                or part.path
                or len(part.op.operand_names) != 1
            ):
                raise ValueError("cache reconstruction is not completed slices")
            start = self.operand(part, 0)
            if start.op.opcode != "slice-start" or len(start.op.operand_names) != 1:
                raise ValueError("invalid asynchronous slice")
            ranges = re.findall(
                r"\bslice=\{([^}]*)\}", _callee_attribute_text(start.op.raw_line)
            )
            expected = f"[{begin}:{end}],[0:16],[0:64],[0:128]"
            if len(ranges) != 1 or re.sub(r"\s", "", ranges[0]) != expected:
                raise ValueError("cache reconstruction slices reordered/incomplete")
            if len(part.op.result_shapes) != 1 or (
                part.op.result_shapes[0].dtype,
                part.op.result_shapes[0].dimensions,
            ) != ("bf16", (end - begin, 16, 64, 128)):
                raise ValueError("cache reconstruction slice shape drifted")
            arg = self.operand(start, 0)
            if not self._identity_shape(op, arg.op):
                raise ValueError("cache reconstruction input shape drifted")
            arg = self.resolve(arg)
            if source is not None and self.key(arg) != self.key(source):
                raise ValueError("cache reconstruction mixes distinct inputs")
            source = arg
        if source is None:
            raise ValueError("empty cache reconstruction")
        return source

    def key(self, value: Value) -> tuple:
        def bound_key(v: Value) -> tuple:
            return (v.op.index, v.path, tuple(bound_key(arg) for arg in v.bindings))

        return bound_key(self.resolve(value))

    def same(self, a: Value, b: Value) -> bool:
        return self.key(a) == self.key(b)

    def input(self, value: Value, number: int) -> bool:
        v = self.resolve(value)
        return (
            _computation_base(v.op.computation) == "ENTRY"
            and v.op.opcode == "parameter"
            and v.path == (number,)
            and not v.bindings
            and re.findall(
                r"\bparameter\((\d+)\)", _callee_attribute_text(v.op.raw_line)
            )
            == ["0"]
        )

    def constant(self, value: Value, dtype: str, number: int) -> bool:
        v = self.resolve(value)
        spellings = [[str(number)], ["{" + str(number) + "}"]]
        if dtype == "pred" and number in (0, 1):
            spellings.append(["true" if number else "false"])
        return (
            not v.path
            and v.op.opcode == "constant"
            and len(v.op.result_shapes) == 1
            and v.op.result_shapes[0].dtype == dtype
            and v.op.result_shapes[0].dimensions in ((), (1,))
            and re.findall(
                r"\bconstant\(([^)]*)\)", _callee_attribute_text(v.op.raw_line)
            )
            in spellings
        )
