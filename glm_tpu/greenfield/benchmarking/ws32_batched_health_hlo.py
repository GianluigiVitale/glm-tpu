"""Actual batched commit vote must imply all accepted cache-writer predicates.

This narrow structural proof does not certify opaque arithmetic or establish
that every possible model-health check exists. Full numerical admission remains
separate. ALL and LIVE-row obligations must never be conflated.
"""

from __future__ import annotations

from typing import Any

from ..sharding.hlo_contract import HloShape
from .ws32_batched_cache_hlo import IndexCachePaths
from .ws32_batched_commit_hlo import _minimum, _require, _shape
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_batched_helper_hlo import _concat_structure, _slice_users, _target
from .ws32_hlo_boolean_factors import BooleanFactors, Ref, dimensions
from .ws32_pallas_one_layer import _computation_base, _callee_attribute_text
from .ws32_prefill_hlo_identity import Value, attribute
import re


class WriterHealthProof:
    def __init__(self, index: PrefillHloIndex, rows: int) -> None:
        self.index, self.rows = index, rows
        self.boolean = BooleanFactors(
            index, live_rows=rows, live_mask=self.live_mask, nonempty_live=False
        )

    def node(self, value: Ref, opcode: str, arity: int) -> Ref:
        value = self.boolean.resolve(value)
        op = self.boolean.ops[value[0]]
        _require(
            not value[1] and op.opcode == opcode and len(op.operand_names) == arity,
            f"health expected {opcode}/{arity}, got {op.name}:{op.opcode}{value[1]}",
        )
        return value

    def arg(self, value: Ref, number: int) -> Ref:
        return self.boolean.resolve(self.boolean.operand(value, number))

    def scalar(self, value: Ref) -> Ref:
        for _ in range(16):
            value = self.boolean.resolve(value)
            op = self.boolean.ops[value[0]]
            if not value[1] and op.opcode in ("bitcast", "reshape"):
                source = self.arg(value, 0)
                a, b = self.boolean.shape(value), self.boolean.shape(source)
                _require(
                    a.dtype == b.dtype and a.element_count == b.element_count == 1,
                    "health scalar forwarding changes values",
                )
                value = source
            else:
                return value
        raise ValueError("unbounded health scalar forwarding")

    def input(self, value: Ref, leaf: int) -> bool:
        value = self.scalar(value)
        op = self.boolean.ops[value[0]]
        return (
            value[1] == (leaf,)
            and value[2] == 0
            and op.opcode == "parameter"
            and _computation_base(op.computation) == "ENTRY"
            and re.findall(r"\bparameter\((\d+)\)", _callee_attribute_text(op.raw_line))
            == ["0"]
        )

    def constant(self, value: Ref, number: int) -> bool:
        value = self.scalar(value)
        op = self.boolean.ops[value[0]]
        return (
            not value[1]
            and op.opcode == "constant"
            and self.boolean.shape(value) == HloShape("s32", ())
            and re.findall(
                r"\bconstant\(([^)]*)\)", _callee_attribute_text(op.raw_line)
            )
            == [str(number)]
        )

    def count(self, value: Ref) -> bool:
        value = self.scalar(value)
        op = self.boolean.ops[value[0]]
        if value[1] or op.opcode != "minimum" or len(op.operand_names) != 2:
            return False
        for j in range(2):
            maximum = self.arg(value, 1 - j)
            mop = self.boolean.ops[maximum[0]]
            if (
                self.constant(self.arg(value, j), self.rows)
                and not maximum[1]
                and mop.opcode == "maximum"
                and len(mop.operand_names) == 2
            ):
                if any(
                    self.input(self.arg(maximum, k), 1)
                    and self.constant(self.arg(maximum, 1 - k), 0)
                    for k in range(2)
                ):
                    return True
        return False

    def live_mask(self, value: Ref) -> int | None:
        value = self.boolean.resolve(value)
        op = self.boolean.ops[value[0]]
        if value[1]:
            return None
        if op.opcode == "broadcast" and len(op.operand_names) == 1:
            source = self.arg(value, 0)
            axis = self.live_mask(source)
            mapping = dimensions(op)
            a, b = self.boolean.shape(source), self.boolean.shape(value)
            if (
                axis is not None
                and len(mapping) == len(a.dimensions)
                and len(set(mapping)) == len(mapping)
                and all(0 <= d < len(b.dimensions) for d in mapping)
                and tuple(b.dimensions[d] for d in mapping) == a.dimensions
            ):
                return mapping[axis]
            return None
        if (
            op.opcode != "compare"
            or len(op.operand_names) != 2
            or attribute(op, "direction") != "LT"
        ):
            return None
        iota, bound = self.arg(value, 0), self.arg(value, 1)
        io, bo = self.boolean.ops[iota[0]], self.boolean.ops[bound[0]]
        if (
            io.opcode != "iota"
            or bo.opcode != "broadcast"
            or len(bo.operand_names) != 1
        ):
            return None
        axis = int(attribute(io, "iota_dimension"))
        shape = self.boolean.shape(value)
        if (
            not iota[1]
            and not bound[1]
            and shape.dtype == "pred"
            and 0 <= axis < len(shape.dimensions)
            and shape.dimensions[axis] == self.rows
            and self.boolean.shape(iota)
            == self.boolean.shape(bound)
            == HloShape("s32", shape.dimensions)
            and dimensions(bo) == ()
            and self.count(self.arg(bound, 0))
        ):
            return axis
        return None

    def predicate(self, value: Ref) -> Ref:
        value = self.node(value, "conditional", 3)
        converted = self.node(self.arg(value, 0), "convert", 1)
        result = self.arg(converted, 0)
        _require(
            self.boolean.shape(converted) == HloShape("s32", ())
            and self.boolean.shape(result) == HloShape("pred", ()),
            "writer selector is not scalar PRED to S32",
        )
        return result

    def frontier(self) -> Ref:
        b = self.boolean
        first = b.resolve(b.ref(self.index.roots["ENTRY"], (0,)))
        _require(first[1] == (0,), "missing actual atomic KV output")
        commit = (first[0], (), first[2])
        consensus = self.node(self.predicate(commit), "compare", 2)
        _require(
            attribute(b.ops[consensus[0]], "direction") == "NE"
            and self.constant(self.arg(consensus, 1), 0),
            "health consensus drift",
        )
        expert = self.node(self.arg(consensus, 0), "all-reduce", 1)
        feature = self.node(self.arg(expert, 0), "all-reduce", 1)
        _minimum(self.index, b.ops[expert[0]], "expert")
        _minimum(self.index, b.ops[feature[0]], "feature")
        converted = self.node(self.arg(feature, 0), "convert", 1)
        vote = self.arg(converted, 0)
        _require(
            b.shape(converted) == HloShape("s32", ())
            and b.shape(vote) == HloShape("pred", ()),
            "local health vote is not scalar PRED",
        )
        # Establish nonempty LIVE from the ACTUAL vote's direct conjuncts,
        # before interpreting any inactive-row exemption or broadcast.
        stack, leaves = [vote], []
        while stack:
            value = b.resolve(stack.pop())
            op = b.ops[value[0]]
            if not value[1] and op.opcode == "and":
                _require(
                    len(op.operand_names) == 2
                    and b.shape(value) == HloShape("pred", ()),
                    "span conjunction is not scalar",
                )
                stack.extend(self.arg(value, j) for j in range(2))
            else:
                leaves.append(value)
            _require(len(leaves) + len(stack) < 10000, "unbounded span vote")
        for direction, limit in (("GT", 0), ("LE", self.rows)):
            _require(
                any(
                    not v[1]
                    and b.ops[v[0]].opcode == "compare"
                    and attribute(b.ops[v[0]], "direction") == direction
                    and self.input(self.arg(v, 0), 1)
                    and self.constant(self.arg(v, 1), limit)
                    for v in leaves
                ),
                "commit does not require original 0<count<=B",
            )
        row_votes = [
            v
            for v in leaves
            if not v[1]
            and b.ops[v[0]].opcode == "reduce"
            and b.shape(self.arg(v, 0)) == HloShape("pred", (self.rows,))
        ]
        _require(len(row_votes) == 1, "missing/ambiguous actual final row vote")
        row_vote = row_votes[0]
        _require(dimensions(b.ops[row_vote[0]]) == (0,), "final row vote axis drift")
        # Validate AND reducer/true initializer through the same strict rule.
        b._rule(b._key(row_vote, None))
        masked = self.node(self.arg(row_vote, 0), "or", 2)
        exemptions = []
        for j in range(2):
            invert = self.arg(masked, j)
            invop = b.ops[invert[0]]
            if (
                not invert[1]
                and invop.opcode == "not"
                and len(invop.operand_names) == 1
            ):
                if self.live_mask(self.arg(invert, 0)) == 0:
                    exemptions.append(j)
        _require(
            len(exemptions) == 1,
            "final row vote does not mask exact original live prefix",
        )
        b.nonempty_live = True
        return vote

    def writers(self) -> list[tuple[str, int, Ref]]:
        paths = IndexCachePaths(self.index, self.rows)
        result = []
        for family, records in paths.accepted_stacks().items():
            for record in records:
                op = record["conditional"].op
                _require(
                    _computation_base(op.computation) == "ENTRY",
                    "non-ENTRY index writer",
                )
                result.append((family, record["layer"], self.boolean.ref(op)))
        root = Value(self.index.roots["ENTRY"])
        first = paths.resolve(paths.ssa.leaf(root, 0))
        accepted = paths.ssa.branch(Value(first.op, bindings=first.bindings), 1)
        value = paths.ssa.leaf(accepted, 0)

        def unwrap(value: Value) -> Value:
            value = paths.ssa.resolve(
                value, reconstruct=False, stop_at_shape_change=True
            )
            if value.op.opcode == "custom-call":
                _require(
                    _target(value.op) == "ConcatBitcast"
                    and _shape(value, "bf16", (78, 16, 64, 640)),
                    "unknown KV stack scaffold",
                )
                _concat_structure(
                    self.index, value.op, _slice_users(self.index, [value.op])
                )
                done = self.index.operand(value.op, 0)
                start = self.index.operand(done, 0)
                value = Value(self.index.operand(start, 0))
            return value

        for layer in reversed(range(78)):
            value = unwrap(value)
            value = paths.node(value, "dynamic-update-slice", 6)
            _require(_shape(value, "bf16", (78, 16, 64, 640)), "KV stack shape drift")
            for operand, number in ((2, layer), (3, 0), (4, 0), (5, 0)):
                _require(
                    paths.ssa.constant(
                        paths.ssa.operand(value, operand), "s32", number
                    ),
                    "KV stack slot/axis drift",
                )
            writer = paths.resolve(paths.ssa.operand(value, 1))
            _require(
                writer.path == (1,)
                and writer.op.opcode == "conditional"
                and _computation_base(writer.op.computation) == "ENTRY"
                and writer.op.result_shapes
                == (HloShape("bf16", (1024, 640)), HloShape("bf16", (1, 16, 64, 640))),
                "KV slot not actual paired writer leaf1",
            )
            result.append(("kv", layer, self.boolean.ref(writer.op)))
            value = paths.ssa.resolve(
                paths.ssa.operand(value, 0),
                reconstruct=False,
                stop_at_shape_change=True,
            )
        value = unwrap(value)
        _require(paths.ssa.input(value, 2), "KV stack not based on original input2")
        _require(
            len(result) == 120 and len({v[2] for v in result}) == 120,
            "writer coverage drift",
        )
        return result


def check_batched_writer_health(
    index: PrefillHloIndex, *, block_rows: int
) -> dict[str, Any]:
    if type(block_rows) is not int or block_rows not in (11, 17):
        raise ValueError("writer health profile requires B17/B11")
    report: dict[str, Any] = dict(
        passed=False,
        scope="SHORT_PREFILL_ALL_CACHE_WRITERS_MUST_GATE",
        not_proven=[
            "ALL_MODEL_HEALTH_CHECKS_EXIST",
            "OPAQUE_ARITHMETIC",
            "NUMERICAL_OR_MEMORY_ADMISSION",
        ],
    )
    try:
        proof = WriterHealthProof(index, block_rows)
        vote = proof.frontier()
        known = proof.boolean.factors(vote)
        writers = proof.writers()
        missing, records = [], []
        for family, layer, writer in writers:
            predicate = proof.predicate(writer)
            passed = proof.boolean.implies(known, predicate)
            records.append(
                dict(
                    family=family,
                    layer=layer,
                    writer=proof.boolean.ops[writer[0]].name,
                    passed=passed,
                )
            )
            if not passed:
                missing.append(records[-1])
        report.update(
            writers=records,
            missing=missing,
            factor_count=len(known),
            evaluated_values=len(proof.boolean.memo),
            passed=not missing,
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
