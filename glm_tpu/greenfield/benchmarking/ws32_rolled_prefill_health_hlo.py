"""Actual outer commit vote to rolled live-tile health and writer predicates."""

from __future__ import annotations

import re
from typing import Any, Sequence

from ...optimized.hlo_contract import HloInstruction
from .ws32_batched_commit_hlo import _require, _shape
from .ws32_batched_health_hlo import WriterHealthProof
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_hlo_boolean_factors import BooleanFactors, Ref
from .ws32_pallas_one_layer import _callee_attribute_text
from .ws32_prefill_hlo_identity import Value
from .ws32_rolled_prefill_hlo import RolledTransitions, RolledTileHealth, _rows


def _value(boolean: BooleanFactors, ref: Ref) -> Value:
    return Value(
        boolean.ops[ref[0]],
        ref[1],
        tuple(_value(boolean, child) for child in boolean.frames[ref[2]]),
    )


def stacked_health_leaves(
    transitions: RolledTransitions, value: Value, *, conjunction: bool = False
) -> list[Value]:
    """Exact same-shape AND under the already checked row-major flatten.

    ALL/LIVE domain is supplied by the caller. No OR/select, transpose or
    arbitrary reshape may turn unrelated predicate bits into a health witness.
    """
    pending, leaves = [value], []
    visited = set()
    while pending:
        source = transitions.resolve(pending.pop())
        key = (source.op.index, source.path, source.bindings)
        if key in visited:
            continue
        visited.add(key)
        _require(len(visited) <= 16, "unbounded stacked health conjunction")
        shape = (
            source.op.result_shapes[source.path[0]]
            if len(source.path) == 1
            else (
                source.op.result_shapes[0]
                if not source.path and len(source.op.result_shapes) == 1
                else None
            )
        )
        _require(
            shape is not None and (shape.dtype, shape.dimensions) == ("pred", (4, 32)),
            "stacked health conjunction shape differs",
        )
        if (
            conjunction
            and not source.path
            and source.op.opcode == "and"
            and len(source.op.operand_names) == 2
        ):
            pending.extend(transitions.arg(source, j) for j in (0, 1))
        else:
            leaves.append(source)
    return leaves


def check_rolled_commit_health(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
    canonical_dense: bool = False,
) -> dict[str, Any]:
    """Bind all78 health stacks through exact flatten/trim to the actual vote.

    For every admitted outer count, global row32*i+j is live iff j is below
    clip(count-32*i,0,32). A proven complete stack update at i therefore maps
    exactly to that iteration's live-health antecedent. Empty tiles impose no
    writer-health requirement; their masked no-write proof is separate.
    """
    _rows(block_rows)
    _require(type(canonical_dense) is bool, "canonical dense option must be bool")
    report: dict[str, Any] = dict(
        passed=False,
        scope="GLOBAL_COMMIT_TO_ALL_NONEMPTY_TILE_WRITER_HEALTH",
        not_proven=[
            "EMPTY_TILE_NO_WRITES",
            "ACCEPTED_CACHE_WRITE_LINEAGE",
            "ALL_MODEL_OPERAND_HEALTH_EXISTS",
            "NUMERICAL_OR_MEMORY_ADMISSION",
        ],
    )
    try:
        transitions = RolledTransitions(index, block_rows)
        loops = transitions.all_loops(live_instructions)
        by_loop = {loop.loop.index: loop for loop in loops}
        global_proof = WriterHealthProof(index, block_rows)
        vote = global_proof.frontier()
        b = global_proof.boolean
        known = b.factors(vote)
        records: dict[int, dict[str, Any]] = {}
        for ref, axis in known:
            if axis not in (None, 0):
                continue
            value = _value(b, ref)
            if not _shape(value, "pred", (block_rows,)):
                continue
            trim = None
            if block_rows == 114:
                if value.op.opcode != "slice" or len(value.op.operand_names) != 1:
                    continue
                ranges = re.findall(
                    r"\bslice=\{([^}]*)\}", _callee_attribute_text(value.op.raw_line)
                )
                if len(ranges) != 1 or re.sub(r"\s", "", ranges[0]) != "[0:114]":
                    continue
                trim = value.op.name
                value = transitions.arg(value, 0)
            if (
                value.op.opcode != "reshape"
                or len(value.op.operand_names) != 1
                or not _shape(value, "pred", (128,))
            ):
                continue
            source = transitions.ssa.operand(value, 0)
            if not _shape(source, "pred", (4, 32)):
                continue
            for source in stacked_health_leaves(
                transitions, source, conjunction=canonical_dense
            ):
                loop = by_loop.get(source.op.index)
                if (
                    loop is None
                    or source.path != (loop.health_slot,)
                    or source.bindings
                ):
                    continue
                records[loop.layer] = dict(
                    layer=loop.layer,
                    loop=loop.loop.name,
                    health_slot=loop.health_slot,
                    flatten=value.op.name,
                    trim=trim,
                    global_factor=b.ops[ref[0]].name,
                    domain="ALL" if axis is None else "LIVE",
                )
        _require(
            set(records) == set(range(78)),
            f"global vote lacks exact rolled health stacks:{sorted(set(range(78))-set(records))}",
        )
        tile = RolledTileHealth(transitions)
        local = [tile.inspect(loop) for loop in loops]
        _require(
            all(r["passed"] for r in local),
            "global health does not imply a local writer",
        )
        report.update(
            passed=True,
            local_vote=b.ops[vote[0]].name,
            stacks=[records[i] for i in range(78)],
            layers=local,
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
