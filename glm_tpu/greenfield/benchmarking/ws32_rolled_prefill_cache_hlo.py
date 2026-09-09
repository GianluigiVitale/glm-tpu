"""Own-layer rolled cache transitions in saved production B128/B114 HLO.

Reuse admitted copy/layout and scalar replacement-scatter rules. No generic
while identity, address arithmetic interpretation or numerical authorization.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from ..sharding.hlo_contract import HloInstruction, HloShape
from .ws32_batched_cache_hlo import IndexCachePaths, LAYERS
from .ws32_batched_commit_hlo import _require, _shape
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _callee_attribute_text
from .ws32_prefill_hlo_identity import Value, attribute
from .ws32_hlo_boolean_factors import dimensions
from .ws32_rolled_prefill_hlo import (
    RolledLoop,
    RolledTransitions,
    _rows,
    check_rolled_commit,
)


class RolledCachePaths(IndexCachePaths):
    def __init__(self, transitions: RolledTransitions) -> None:
        # Inner writers always have32 rows; the main/tail block count is kept
        # by the transitions object, never substituted for the row scatter size.
        super().__init__(transitions.index, 32)
        self.transitions = transitions
        self.ssa = transitions.ssa

    def own_initial(self, loop: RolledLoop, slot: int) -> None:
        width = 640 if slot == 1 else 128
        layers = tuple(range(78)) if slot == 1 else LAYERS
        outer_slot = layers.index(loop.layer)
        original = {1: 2, 2: 3, 3: 11}[slot]
        dims = (len(layers), 16, 64, width)
        initial = self.ssa.leaf(loop.initial, slot)
        source = self.node(
            self.bridge(initial, (16, 64, width), (1, 16, 64, width)), "slice", 1
        )
        expected = f"[{outer_slot}:{outer_slot+1}],[0:16],[0:64],[0:{width}]"
        ranges = re.findall(
            r"\bslice=\{([^}]*)\}", _callee_attribute_text(source.op.raw_line)
        )
        _require(
            len(ranges) == 1 and re.sub(r"\s", "", ranges[0]) == expected,
            "rolled initial cache reads another layer",
        )
        base = self.arg(source, 0)
        for _ in range(len(layers) + 1):
            if self.ssa.input(base, original):
                return
            base = self.node(base, "dynamic-update-slice", 6)
            _require(
                _shape(base, "bf16", dims)
                and _shape(self.ssa.operand(base, 1), "bf16", (1, 16, 64, width))
                and any(
                    self.ssa.constant(self.ssa.operand(base, 2), "s32", prior)
                    for prior in range(outer_slot)
                )
                and all(
                    self.ssa.constant(self.ssa.operand(base, j), "s32", 0)
                    for j in (3, 4, 5)
                ),
                "rolled initial cache slot overwritten or preceding write displaced",
            )
            base = self.arg(base, 0)
        raise ValueError("unbounded rolled initial cache history")

    def writer(self, loop: RolledLoop, slot: int) -> dict[str, Any]:
        self.own_initial(loop, slot)
        width = 640 if slot == 1 else 128
        dims, flat = (16, 64, width), (1024, width)
        conditional = self.resolve(self.ssa.leaf(loop.root, slot))
        _require(conditional.op.opcode == "conditional", "rolled writer is not guarded")
        old = self.resolve(self.ssa.branch(conditional, 0))
        _require(
            self.ssa.same(old, self.ssa.leaf(loop.parameter, slot)),
            "rolled writer refusal changes own cache",
        )
        accepted = self.ssa.branch(conditional, 1)
        scatter = self.node(self.bridge(accepted, dims, flat), "scatter", 3)
        self.replacement_scatter(scatter, width=width)
        prior = self.bridge(self.ssa.operand(scatter, 0), flat, dims)
        _require(self.ssa.same(prior, old), "rolled accepted write changes cache base")
        self.inactive_rows_drop(loop, self.ssa.operand(scatter, 1))
        return dict(
            layer=loop.layer,
            slot=slot,
            conditional=conditional,
            scatter=scatter,
            indices=self.ssa.operand(scatter, 1),
            updates=self.ssa.operand(scatter, 2),
        )

    def vector_constant(self, value: Value, number: int) -> bool:
        value = self.resolve(value)
        return (
            not value.path
            and value.op.opcode == "broadcast"
            and len(value.op.operand_names) == 1
            and _shape(value, "s32", (32,))
            and dimensions(value.op) == ()
            and self.ssa.constant(self.ssa.operand(value, 0), "s32", number)
        )

    def inactive_rows_drop(self, loop: RolledLoop, indices: Value) -> None:
        """Bind inactive indices to positive OOB1024, including negative normalization.

        Address arithmetic for ACTIVE rows remains a separate source/numerical
        obligation. This checks the exact mask supplied to the actual scatter.
        """
        value = self.resolve(indices)
        if value.op.opcode == "transpose":
            _require(
                _shape(value, "s32", (32,)) and dimensions(value.op) == (0,),
                "scatter index transpose changes row mapping",
            )
            value = self.arg(value, 0)
        normalized = self.node(value, "select", 3)
        _require(
            _shape(normalized, "s32", (32,)), "scatter normalized index shape drift"
        )
        negative = self.node(self.arg(normalized, 0), "compare", 2)
        added = self.node(self.arg(normalized, 1), "add", 2)
        masked = self.node(self.arg(normalized, 2), "select", 3)
        _require(
            _shape(negative, "pred", (32,))
            and attribute(negative.op, "direction") == "LT"
            and self.ssa.same(self.arg(negative, 0), masked)
            and self.vector_constant(self.arg(negative, 1), 0)
            and _shape(added, "s32", (32,))
            and any(
                self.ssa.same(self.arg(added, j), masked)
                and self.vector_constant(self.arg(added, 1 - j), 1024)
                for j in (0, 1)
            )
            and _shape(masked, "s32", (32,))
            and self.vector_constant(self.arg(masked, 2), 1024),
            "inactive scatter sentinel or negative-index normalization drift",
        )
        t = self.transitions
        expected = t.integer_expression(t.tile_counts(loop)[0], loop)
        # Source write_prefill_cache_block clips an already bounded tile count.
        # At most two repeated clips; each is identity because expected is0..32.
        allowed = [expected]
        for _ in range(2):
            allowed.append(
                t.integer_node(
                    "minimum",
                    ("constant", 32),
                    t.integer_node("maximum", ("constant", 0), allowed[-1]),
                )
            )
        pending = [self.arg(masked, 0)]
        for _ in range(32):
            if not pending:
                break
            mask = self.resolve(pending.pop())
            _require(_shape(mask, "pred", (32,)), "scatter row mask shape drift")
            if mask.op.opcode == "and" and len(mask.op.operand_names) == 2:
                pending.extend(self.arg(mask, j) for j in (0, 1))
            elif mask.op.opcode == "compare" and len(mask.op.operand_names) == 2:
                first, bound = self.arg(mask, 0), self.arg(mask, 1)
                if (
                    attribute(mask.op, "direction") == "LT"
                    and not first.path
                    and first.op.opcode == "iota"
                    and _shape(first, "s32", (32,))
                    and attribute(first.op, "iota_dimension") == "0"
                    and not bound.path
                    and bound.op.opcode == "broadcast"
                    and _shape(bound, "s32", (32,))
                    and dimensions(bound.op) == ()
                    and len(bound.op.operand_names) == 1
                    and t.integer_expression(self.arg(bound, 0), loop) in allowed
                ):
                    return
        raise ValueError("scatter mask does not require its own live tile row")

    def loop_leaf(self, value: Value, loop: RolledLoop, slot: int, width: int) -> None:
        value = self.node(value, "bitcast", 1)
        source = self.ssa.operand(value, 0)
        self.layout(value, (1, 16, 64, width))
        self.layout(source, (16, 64, width))
        source = self.resolve(source)
        _require(
            source.op.index == loop.loop.index
            and source.path == (slot,)
            and not source.bindings,
            "outer cache update does not consume its own loop's final cache",
        )

    def stack(
        self,
        value: Value,
        loops: Sequence[RolledLoop],
        *,
        original: int,
        cache_slot: int,
    ) -> list[dict[str, Any]]:
        width = 640 if cache_slot == 1 else 128
        layers = tuple(range(78)) if cache_slot == 1 else LAYERS
        dims = (len(layers), 16, 64, width)
        records = []
        for outer_slot in reversed(range(len(layers))):
            loop = loops[layers[outer_slot]]
            value = self.node(value, "dynamic-update-slice", 6)
            _require(_shape(value, "bf16", dims), "rolled outer cache shape drift")
            for arg, number in ((2, outer_slot), (3, 0), (4, 0), (5, 0)):
                _require(
                    self.ssa.constant(self.ssa.operand(value, arg), "s32", number),
                    "rolled outer cache update slot displaced",
                )
            self.loop_leaf(self.ssa.operand(value, 1), loop, cache_slot, width)
            records.append(
                dict(
                    layer=loop.layer,
                    outer_slot=outer_slot,
                    cache_slot=cache_slot,
                    update=value.op.name,
                )
            )
            value = self.arg(value, 0)
        _require(
            self.ssa.input(value, original),
            "rolled outer cache has wrong original base",
        )
        return list(reversed(records))

    def accepted_stacks(self, accepted: Value, loops: Sequence[RolledLoop]) -> dict:
        promotion = self.node(self.ssa.leaf(accepted, 1), "conditional", 3)
        return dict(
            kv=self.stack(self.ssa.leaf(accepted, 0), loops, original=2, cache_slot=1),
            unrepaired=self.stack(
                self.ssa.branch(promotion, 0), loops, original=3, cache_slot=2
            ),
            repaired=self.stack(
                self.ssa.leaf(accepted, 8), loops, original=11, cache_slot=3
            ),
        )


def check_rolled_cache_paths(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
    canonical_dense: bool = False,
) -> dict[str, Any]:
    _rows(block_rows)
    report: dict[str, Any] = dict(
        passed=False,
        scope="ROLLED_ACCEPTED_CACHE_REPLACEMENT_AND_OWN_OUTER_ASSEMBLY",
        not_proven=[
            "ACTIVE_ROW_ADDRESS_ARITHMETIC",
            "WRITTEN_KEY_PROVENANCE",
            "GLOBAL_COMMIT_HEALTH",
            "NUMERICAL_OR_MEMORY_ADMISSION",
        ],
    )
    try:
        transitions = RolledTransitions(
            index, block_rows, canonical_dense=canonical_dense
        )
        loops = transitions.all_loops(live_instructions)
        anchors: dict[str, Value] = {}
        commit = check_rolled_commit(
            index,
            block_rows=block_rows,
            live_instructions=live_instructions,
            anchors=anchors,
        )
        _require(
            commit["passed"], f"rolled cache lacks atomic commit: {commit.get('error')}"
        )
        paths = RolledCachePaths(transitions)
        writers = [
            paths.writer(loop, slot) for loop in loops for slot in loop.cache_slots
        ]
        stacks = paths.accepted_stacks(anchors["accepted"], loops)
        report.update(
            passed=True,
            stacks=stacks,
            writers=[
                dict(
                    layer=w["layer"],
                    slot=w["slot"],
                    conditional=w["conditional"].op.name,
                    scatter=w["scatter"].op.name,
                )
                for w in writers
            ],
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
