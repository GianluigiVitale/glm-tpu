"""Bind actual committed metadata reads to the final live prompt row.

This checks addressing, not the arithmetic or padding values of the arrays
being read. TPU optimized text elides some tail padding constants; source pins
and numerical interventions remain necessary for those values.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ..sharding.hlo_contract import HloInstruction
from .ws32_batched_commit_hlo import _require, _shape
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_batched_health_hlo import WriterHealthProof
from .ws32_pallas_one_layer import _callee_attribute_text
from .ws32_prefill_hlo_identity import Value, attribute
from .ws32_rolled_prefill_hlo import RolledTransitions, _rows, check_rolled_commit


class RolledLastRow:
    def __init__(self, transitions: RolledTransitions) -> None:
        self.t = transitions

    def index(self, value: Value, loop: Any) -> Value:
        """Prove normalize(max(clip(input_count,0,B)-1,0), B).

        The caller proves via WriterHealthProof.frontier that accepted commit
        requires 0<count<=B. Thus this expression
        is exactly count-1, and dynamic-slice's clamping cannot change it.
        """
        t = self.t
        value = t.node(value, "select", 3)
        _require(_shape(value, "s32", ()), "last-row index must be S32 scalar")
        condition, added, last = (t.arg(value, j) for j in range(3))
        condition = t.node(condition, "compare", 2)
        added = t.node(added, "add", 2)
        _require(
            attribute(condition.op, "direction") == "LT"
            and _shape(condition, "pred", ())
            and t.ssa.same(t.arg(condition, 0), last)
            and t.ssa.constant(t.arg(condition, 1), "s32", 0)
            and t.ssa.same(t.arg(added, 0), last)
            and t.ssa.constant(t.arg(added, 1), "s32", t.rows),
            "last-row negative-index normalization drift",
        )
        last = t.node(last, "maximum", 2)
        difference, zero = t.arg(last, 0), t.arg(last, 1)
        if t.ssa.constant(difference, "s32", 0):
            difference, zero = zero, difference
        _require(t.ssa.constant(zero, "s32", 0), "last-row lower clamp drift")
        difference = t.node(difference, "add", 2)
        count, minus_one = t.arg(difference, 0), t.arg(difference, 1)
        if t.ssa.constant(count, "s32", -1):
            count, minus_one = minus_one, count
        _require(
            t.ssa.constant(minus_one, "s32", -1), "last-row is not count minus one"
        )
        n = t.integer_node
        expected = n(
            "minimum",
            ("constant", t.rows),
            n("maximum", ("constant", 0), ("valid_rows",)),
        )
        _require(
            t.integer_expression(count, loop) == expected,
            "last-row count is not clipped ENTRY valid_rows",
        )
        return value

    def read(self, value: Value, *, dtype: str, width: int | None, loop: Any) -> dict:
        t = self.t
        output = (1,) if width is None else (1, width)
        source = (t.rows,) if width is None else (t.rows, width)
        value = t.resolve(value)
        masks: list[str] = []
        # B114 compiler sinks three shared-layer padding selects after the
        # selected-position/score read. Record them; do NOT infer their truth
        # from an elided constant or treat either branch as generic identity.
        if t.rows == 114 and width is not None:
            for _ in range(3):
                value = t.node(value, "select", 3)
                _require(_shape(value, dtype, output), "tail padding shape drift")
                mask = t.arg(value, 0)
                _require(_shape(mask, "pred", output), "tail padding mask shape")
                _require(
                    _shape(t.ssa.operand(value, 2), dtype, output),
                    "tail padding alternate shape",
                )
                masks.append(mask.op.name)
                value = t.arg(value, 1)
        value = t.node(value, "dynamic-slice", 2 if width is None else 3)
        _require(
            _shape(value, dtype, output)
            and _shape(t.ssa.operand(value, 0), dtype, source),
            "last-row metadata read geometry drift",
        )
        sizes = re.findall(
            r"\bdynamic_slice_sizes=\{([^}]*)\}",
            _callee_attribute_text(value.op.raw_line),
        )
        _require(
            len(sizes) == 1
            and re.sub(r"\s", "", sizes[0]) == ",".join(map(str, output)),
            "last-row slice extent drift",
        )
        if width is not None:
            _require(
                t.ssa.constant(t.arg(value, 2), "s32", 0),
                "last-row column offset drift",
            )
        index = self.index(t.arg(value, 1), loop)
        return dict(
            read=value.op.name,
            index=index.op.name,
            padding_masks_not_interpreted=masks,
        )


def check_rolled_last_row_indices(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    """Actual accepted selected positions/counts/scores use count-1 addressing."""
    _rows(block_rows)
    report: dict[str, Any] = dict(
        passed=False,
        scope="COMMITTED_METADATA_LAST_LIVE_ROW_ADDRESSING",
        not_proven=[
            "SELECTED_ARRAY_VALUES_AND_PADDING_PREDICATES",
            "FINAL_HEAD_INPUT_ROW",
            "NUMERICAL_OR_MEMORY_ADMISSION",
        ],
    )
    try:
        t = RolledTransitions(index, block_rows)
        loops = t.all_loops(live_instructions)
        anchors: dict[str, Value] = {}
        commit = check_rolled_commit(
            index,
            block_rows=block_rows,
            live_instructions=live_instructions,
            anchors=anchors,
        )
        _require(commit["passed"], str(commit.get("error", "atomic commit refused")))
        # Atomic state/schedule checks alone do not establish valid count.
        # Reuse the actual global vote's direct scalar span conjunction proof.
        health = WriterHealthProof(index, block_rows)
        vote = health.frontier()
        checker = RolledLastRow(t)
        records = [
            checker.read(
                t.ssa.leaf(anchors["accepted"], slot),
                dtype=dtype,
                width=width,
                loop=loops[0],
            )
            for slot, dtype, width in (
                (2, "s32", 2048),
                (3, "s32", None),
                (4, "f32", 2048),
            )
        ]
        report.update(
            passed=True,
            reads=records,
            count_guard_vote=health.boolean.ops[vote[0]].name,
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
