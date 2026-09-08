"""Acquired short-prefill consensus, atomic rollback and final-token SSA proof.

This does NOT prove layer arithmetic, all layer-health inputs, proposed cache
writes or memory feasibility. Those remain separate full-profile obligations.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ..sharding.hlo_contract import HloInstruction
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_decoder import _group_family
from .ws32_pallas_one_layer import _callee_attribute_text
from .ws32_prefill_hlo_identity import PrefillIdentity, Value, attribute


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _shape(value: Value, dtype: str, dimensions: tuple[int, ...]) -> bool:
    return (
        not value.path
        and len(value.op.result_shapes) == 1
        and value.op.result_shapes[0].dtype == dtype
        and value.op.result_shapes[0].dimensions == dimensions
    )


def _minimum(index: PrefillHloIndex, op: HloInstruction, family: str) -> None:
    _require(
        op.raw_opcode == "all-reduce"
        and _shape(Value(op), "s32", ())
        and _group_family(op) == family
        and len(op.replica_groups) == (8 if family == "feature" else 4)
        and op.use_global_device_ids
        and len(op.operand_names) == 1
        and _shape(Value(index.operand(op, 0)), "s32", ()),
        "health reduction groups/scalar payload drifted",
    )
    callee = index.callee(op, "to_apply")
    nodes, root = list(index.computations[callee].values()), index.roots[callee]
    params = [p for p in nodes if p.opcode == "parameter"]
    _require(
        len(nodes) == 3
        and len(params) == 2
        and root.opcode == "minimum"
        and len(root.operand_names) == 2
        and set(root.operand_names) == {p.name for p in params}
        and all(_shape(Value(p), "s32", ()) for p in nodes)
        and sorted(
            re.findall(r"\bparameter\((\d+)\)", _callee_attribute_text(p.raw_line))
            for p in params
        )
        == [["0"], ["1"]],
        "health reducer is not exact scalar S32 MIN",
    )


def check_batched_commit(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    """Bind the acquired capacity8192 main/tail ENTRY leaves to actual decisions."""
    if type(block_rows) is not int or not 1 <= block_rows <= 32:
        raise ValueError("commit proof requires1..32 rows")
    report: dict[str, Any] = dict(
        passed=False,
        scope="SHORT_PREFILL_ATOMIC_COMMIT_ONLY",
        not_proven=[
            "ALL_LAYER_HEALTH_CONTRIBUTIONS",
            "PROPOSED_CACHE_WRITES_AND_UNREPAIRED_PROVENANCE",
            "HEAD_SAMPLING_ARITHMETIC",
            "PHYSICAL_ALIASING_OR_MEMORY_FEASIBILITY",
        ],
    )
    ssa = PrefillIdentity(index)

    def node(v: Value, opcode: str, arity: int) -> Value:
        v = ssa.resolve(v)
        _require(
            not v.path and v.op.opcode == opcode and len(v.op.operand_names) == arity,
            f"expected {opcode}/{arity}, got {v.op.name}:{v.op.opcode}{v.path}",
        )
        return v

    def arg(v: Value, i: int) -> Value:
        return ssa.resolve(ssa.operand(v, i))

    def predicate(v: Value) -> Value:
        v = node(v, "conditional", 3)
        converted = node(arg(v, 0), "convert", 1)
        result = arg(converted, 0)
        _require(
            _shape(converted, "s32", ()) and _shape(result, "pred", ()),
            "conditional selector is not scalar pred→S32",
        )
        return result

    def pair(v: Value, opcode: str, a: Value, b: Value) -> None:
        v = node(v, opcode, 2)
        _require(
            (ssa.same(arg(v, 0), a) and ssa.same(arg(v, 1), b))
            or (ssa.same(arg(v, 0), b) and ssa.same(arg(v, 1), a)),
            f"disconnected {opcode} operands",
        )

    def scalar_expr(v: Value, depth: int = 0) -> tuple:
        _require(depth < 16, "unbounded schedule expression")
        v = ssa.resolve(v)
        for leaf in (1, 7, 12, 13):
            if ssa.input(v, leaf):
                return ("input", leaf)
        for constant in (0, 1, block_rows, 8191, 8192):
            if ssa.constant(v, "s32", constant):
                return ("constant", constant)
        _require(
            not v.path
            and _shape(v, "s32", ())
            and v.op.opcode in {"minimum", "maximum", "add", "subtract"}
            and len(v.op.operand_names) == 2,
            "unexpected safe-end arithmetic",
        )
        operands = [scalar_expr(arg(v, i), depth + 1) for i in range(2)]
        if v.op.opcode != "subtract":
            operands.sort(key=repr)
        return (v.op.opcode, *operands)

    try:
        root = Value(index.roots["ENTRY"])
        _require(
            root.op.opcode == "tuple" and len(root.op.operand_names) == 13,
            "prefill ENTRY must return12 state leaves and one token",
        )
        outputs = [ssa.resolve(ssa.leaf(root, i)) for i in range(13)]
        commit = outputs[0]
        _require(commit.path == (0,), "KV does not come from atomic commit slot0")
        commit = Value(commit.op, bindings=commit.bindings)
        node(commit, "conditional", 3)
        live = {(op.computation, op.name) for op in live_instructions}
        _require(
            (commit.op.computation, commit.op.name) in live, "dead commit conditional"
        )
        # Every mutable output must come from this SAME conditional and slot.
        output_to_slot = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 6, 8: 7, 9: 8, 11: 9}
        for output, slot in output_to_slot.items():
            _require(
                ssa.same(outputs[output], ssa.leaf(commit, slot)),
                f"output{output} bypasses/reorders atomic commit",
            )
        _require(
            ssa.input(outputs[6], 8) and ssa.input(outputs[10], 12),
            "page table or prompt length changed",
        )

        consensus = node(predicate(commit), "compare", 2)
        _require(
            attribute(consensus.op, "direction") == "NE"
            and ssa.constant(arg(consensus, 1), "s32", 0),
            "consensus is not MIN!=0",
        )
        expert = node(arg(consensus, 0), "all-reduce", 1)
        feature = node(arg(expert, 0), "all-reduce", 1)
        for v, family in ((feature, "feature"), (expert, "expert")):
            _minimum(index, v.op, family)
            _require((v.op.computation, v.op.name) in live, "dead health reduction")
        scalar_reductions = [
            op for op in index.module.collectives if _shape(Value(op), "s32", ())
        ]
        _require(
            {op.index for op in scalar_reductions}
            == {feature.op.index, expert.op.index}
            and len(scalar_reductions) == 2,
            "unexpected scalar health collective",
        )
        local_convert = node(arg(feature, 0), "convert", 1)
        local_health = arg(local_convert, 0)
        _require(
            _shape(local_convert, "s32", ()) and _shape(local_health, "pred", ()),
            "local health is not a boolean vote",
        )

        refused = ssa.branch(commit, 0)
        accepted = ssa.branch(commit, 1)
        _require(
            ssa.resolve(refused).op.opcode == "tuple"
            and len(ssa.resolve(refused).op.operand_names) == 10
            and ssa.resolve(accepted).op.opcode == "tuple"
            and len(ssa.resolve(accepted).op.operand_names) == 10,
            "commit branch state shape drifted",
        )
        rollback = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 9, 8: 11, 9: 13}
        for slot, input_leaf in rollback.items():
            _require(
                ssa.input(ssa.leaf(refused, slot), input_leaf),
                f"rollback slot{slot} is not original input{input_leaf}",
            )
        _require(
            ssa.constant(ssa.leaf(refused, 7), "pred", 0), "rollback health not false"
        )
        _require(
            ssa.same(ssa.leaf(accepted, 7), consensus),
            "committed health is not consensus",
        )

        final = node(ssa.leaf(accepted, 9), "and", 2)
        terms = [arg(final, i) for i in range(2)]
        comparisons = [v for v in terms if v.op.opcode == "compare"]
        inversions = [v for v in terms if v.op.opcode == "not"]
        _require(
            len(comparisons) == len(inversions) == 1, "final lacks end/finished checks"
        )
        eq = node(comparisons[0], "compare", 2)
        inv = node(inversions[0], "not", 1)
        _require(
            attribute(eq.op, "direction") == "EQ"
            and ssa.input(arg(eq, 1), 12)
            and ssa.input(arg(inv, 0), 13),
            "final does not test original prompt/finished",
        )
        end = arg(eq, 0)

        def binary(op: str, a: tuple, b: tuple) -> tuple:
            return (op, *sorted((a, b), key=repr)) if op != "subtract" else (op, a, b)

        c = lambda n: ("constant", n)
        offset = binary("minimum", c(8191), binary("maximum", ("input", 7), c(0)))
        count = binary("minimum", c(block_rows), binary("maximum", ("input", 1), c(0)))
        safe_count = binary("minimum", count, binary("subtract", c(8192), offset))
        _require(
            scalar_expr(end) == binary("add", offset, safe_count),
            "safe-end schedule drifted",
        )
        _require(
            ssa.same(ssa.leaf(accepted, 5), end),
            "commit position differs from final end",
        )
        context = node(ssa.leaf(accepted, 6), "add", 2)
        _require(
            ssa.same(arg(context, 0), end) and ssa.constant(arg(context, 1), "s32", 1),
            "committed context length is not end+1",
        )

        promotion = node(ssa.leaf(accepted, 1), "conditional", 3)
        _require(
            ssa.same(predicate(promotion), final),
            "index promotion uses different final predicate",
        )
        _require(
            ssa.same(ssa.branch(promotion, 1), ssa.leaf(accepted, 8)),
            "final active cache is not proposed repaired cache",
        )
        _require(
            not ssa.same(ssa.branch(promotion, 0), ssa.leaf(accepted, 8)),
            "nonfinal block directly forwards proposed repaired cache",
        )
        # Necessary refusal only: unresolved SSA distinction is NOT proof of
        # unrepaired provenance or numerical inequality. The cache/helper
        # ownership profile must independently prove the nonfinal producer.

        token = node(outputs[12], "select", 3)
        _require(
            _shape(token, "s32", (1,)) and ssa.constant(arg(token, 2), "s32", -1),
            "token refusal sentinel is not s32[-1]",
        )
        pair(arg(token, 0), "and", consensus, final)
        head_token = arg(token, 1)
        _require(
            head_token.op.opcode == "conditional" and head_token.path == (0,),
            "token is not conditional head output0",
        )
        head = Value(head_token.op, bindings=head_token.bindings)
        _require(
            ssa.same(predicate(head), final),
            "head executes on different final predicate",
        )
        head_false = ssa.branch(head, 0)
        _require(
            ssa.constant(ssa.leaf(head_false, 0), "s32", -1)
            and ssa.constant(ssa.leaf(head_false, 1), "pred", 1),
            "nonfinal head branch is not (-1,true)",
        )
        # Require actual head-health leaf as a conjunct of the vote. This does
        # not claim every model layer contributes to the other conjunct.
        local_health = node(local_health, "and", 2)
        _require(
            any(ssa.same(arg(local_health, i), ssa.leaf(head, 1)) for i in range(2)),
            "head health disconnected from owner vote",
        )
        report.update(
            passed=True,
            commit=commit.op.name,
            health_reductions=[feature.op.name, expert.op.name],
            final_predicate=final.op.name,
            head=head.op.name,
            # Report keys must survive the worker's JSON boundary unchanged.
            # Keep integer slots above for SSA checks, strings only on the wire.
            rollback_inputs={str(slot): leaf for slot, leaf in rollback.items()},
            capacity=8192,
            block_rows=block_rows,
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
