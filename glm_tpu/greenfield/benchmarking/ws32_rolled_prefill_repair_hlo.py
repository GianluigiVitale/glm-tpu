"""Reuse completed-WK and actual repair-writer lineage inside validated loops."""

from __future__ import annotations

from typing import Any, Sequence

from ..sharding.hlo_contract import HloInstruction
from .ws32_batched_cache_hlo import LAYERS
from .ws32_batched_commit_hlo import _require
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_batched_repair_hlo import _check_repair_lineage
from .ws32_prefill_hlo_identity import Value
from .ws32_rolled_prefill_cache_hlo import RolledCachePaths
from .ws32_rolled_prefill_hlo import RolledTransitions, _rows


def check_rolled_repair_lineage(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    _rows(block_rows)
    try:
        transitions = RolledTransitions(index, block_rows)
        loops = transitions.all_loops(live_instructions)

        class Paths(RolledCachePaths):
            def accepted_stacks(self) -> dict:
                return {"repaired": [self.writer(loops[layer], 3) for layer in LAYERS]}

        paths = Paths(transitions)
        ssa = transitions.ssa

        def wk_origin(value: Value, layer: int) -> Value:
            value = ssa.resolve(value)
            loop = loops[layer]
            _require(
                value.op.index == loop.parameter.op.index
                and len(value.path) == 1
                and not value.bindings
                and value.path[0] in ssa.invariants[loop.loop.index],
                "repair WK is not its own immutable loop argument",
            )
            # all_loops proves both direct invariant forwarding and aliases
            # of immutable slots with the exact same initial SSA value.
            return ssa.leaf(loop.initial, value.path[0])

        report = _check_repair_lineage(
            index,
            block_rows=32,
            paths=paths,
            producer_computations={
                layer: index.callee(loops[layer].loop, "body") for layer in LAYERS
            },
            wk_origin=wk_origin,
            unpadded_input=True,
        )
        return {
            **report,
            "scope": "ROLLED_OWN_WK_NORMALIZED_INPUT_AND_REPAIRED_WRITER_PROVENANCE",
        }
    except (ValueError, KeyError, IndexError) as error:
        return dict(passed=False, error=str(error))
