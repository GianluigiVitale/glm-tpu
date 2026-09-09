"""Rolled compiler annotations, bounded copies and initialized local scratch."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from ..sharding.hlo_contract import HloInstruction
from .ws32_batched_cache_hlo import LAYERS
from .ws32_batched_commit_hlo import _require, _shape
from .ws32_batched_helper_hlo import (
    _check_helper_schedule,
    _scratch_pairs,
    _signature,
    _LAYER,
)
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _computation_base
from .ws32_prefill_hlo_identity import PrefillIdentity, Value
from .ws32_rolled_prefill_hlo import _rows


def _copy_limits() -> dict[tuple, int]:
    # Bounded maxima from BOTH original graphs. Each present copy must still
    # pass the same-source, complete-span, exclusive-consumer structural guard.
    # This is NOT copy-as-identity; SSA forwarding separately requires order.
    return {
        ("ConcatBitcast", "bf16", (21, 16, 64, 128)): 5,
        ("ConcatBitcast", "bf16", (8192, 64)): 1,
        ("ConcatBitcast", "bf16", (16, 64, 640)): 1,
        ("ConcatBitcast", "bf16", (19360, 1536)): 1,
        ("ConcatBitcast", "u8", (1536, 2048)): 153,
        ("ConcatBitcast", "u8", (2048, 1536)): 152,
        ("ConcatBitcast", "u8", (2048, 2048)): 78,
        ("ConcatBitcast", "u8", (3584, 512)): 78,
        ("ConcatBitcast", "u8", (1536, 1536)): 8,
        ("ConcatBitcast", "u8", (512, 2048)): 2,
        ("ConcatBitcast", "s32", (4, 32, 2048)): 39,
        ("ConcatBitcast", "f32", (4, 32, 2048)): 20,
        ("ConcatBitcast", "f32", (128, 6144)): 2,
    }


def _expected(rows: int) -> Counter:
    _rows(rows)
    panels = (8 * rows + 31) // 32 + 31
    return Counter(
        {
            ("AssumeGatherIndicesInBound", "s32", (1024,)): 1241,
            ("AssumeGatherIndicesInBound", "s32", (2048,)): 225,
            ("AssumeGatherIndicesInBound", "s32", (65536,)): 99,
            ("AssumeGatherIndicesInBound", "u32", (1024,)): 150,
            ("GatherScatterIndicesBitpacked", "s32", (rows, 8, 2)): 75,
            ("GatherScatterIndicesBitpacked", "s32", (32, 2048, 2)): 99,
            ("AllocateBuffer", "u32", (8 * rows,)): 150,
            ("AllocateBuffer", "u32", (panels,)): 150,
            ("AllocateBuffer", "s32", (32, 2, 2, 512)): 42,
            **_copy_limits(),
        }
    )


def _merge_scratch(
    index: PrefillHloIndex, allocations: Sequence[HloInstruction], live: set
) -> list[dict[str, Any]]:
    """Both half writes must complete before allocated merge storage escapes."""
    _require(len(allocations) == 42, "expected42 merge scratch allocations")
    computations = {_computation_base(op.computation) for op in allocations}
    callers: dict[str, list[HloInstruction]] = defaultdict(list)
    for op in index.module.instructions:
        if op.opcode == "fusion":
            callee = index.callee(op, "calls")
            if callee in computations:
                callers[callee].append(op)
    cached_users: dict[str, dict[str, list[HloInstruction]]] = {}

    def users(op: HloInstruction) -> list[HloInstruction]:
        comp = _computation_base(op.computation)
        if comp not in cached_users:
            table: dict[str, list[HloInstruction]] = defaultdict(list)
            for node in index.computations[comp].values():
                for name in set(node.operand_names):
                    table[name].append(node)
            cached_users[comp] = table
        return cached_users[comp][op.name]

    ssa = PrefillIdentity(index)
    dims, half = (32, 2, 2, 512), (32, 1, 2, 512)

    def write(value: Value, offsets: tuple[int, ...]) -> None:
        _require(
            value.op.opcode == "dynamic-update-slice"
            and not value.path
            and len(value.op.operand_names) == 6
            and _shape(value, "s32", dims)
            and _shape(ssa.operand(value, 1), "s32", half),
            "merge scratch half-write geometry drift",
        )
        _require(
            all(
                ssa.constant(ssa.operand(value, 2 + j), "s32", offset)
                for j, offset in enumerate(offsets)
            ),
            "merge scratch offsets leave a hole",
        )

    records = []
    for allocation in allocations:
        comp = _computation_base(allocation.computation)
        _require(
            _signature(allocation) == (("s32", dims),), "merge allocation shape drift"
        )
        uses = users(allocation)
        _require(
            len(uses) == 1 and uses[0].index == index.roots[comp].index,
            "uninitialized merge storage escapes its root write",
        )
        first = Value(uses[0])
        write(first, (0, 0, 0, 0))
        _require(
            ssa.operand(first, 0).op.index == allocation.index, "merge write base drift"
        )
        owners = callers[comp]
        _require(len(owners) == 1, "merge allocation lacks unique fusion caller")
        caller = owners[0]
        consumers = users(caller)
        _require(
            len(consumers) == 1
            and consumers[0].opcode == "fusion"
            and (caller.computation, caller.name) in live,
            "partly initialized merge storage escapes",
        )
        completed = consumers[0]
        _require(
            (completed.computation, completed.name) in live
            and completed.operand_names.count(caller.name) == 1,
            "merge completion is dead or repeats partial-buffer binding",
        )
        second = ssa.resolve(Value(completed), stop_at_shape_change=True)
        write(second, (0, 1, 0, 0))
        _require(
            ssa.same(ssa.operand(second, 0), Value(caller)),
            "merge halves use different bases",
        )
        # Binding the called root is insufficient if another body operation
        # consumes the half-initialized argument first. Its only use is DUS base.
        second_comp = index.callee(completed, "calls")
        base = index.operand(second.op, 0)
        _require(
            base.opcode == "parameter"
            and users(base) == [second.op]
            and index.roots[second_comp].index == second.op.index,
            "merge partial base has an escaping body consumer",
        )
        layers = _LAYER.findall(second.op.op_name or "")
        _require(
            len(layers) == 1 and int(layers[0]) in LAYERS,
            "merge scratch lacks own indexer scope",
        )
        records.append(
            dict(
                kind="merge_complete_halves",
                layer=int(layers[0]),
                allocation=allocation.name,
                completion=completed.name,
            )
        )
    _require(
        Counter(r["layer"] for r in records) == Counter({layer: 2 for layer in LAYERS}),
        "expected two complete merge buffers per indexer",
    )
    return records


def _scratch(
    index: PrefillHloIndex, allocations: Sequence[HloInstruction], rows: int, live: set
) -> list[dict[str, Any]]:
    search = [op for op in allocations if op.result_shapes[0].dtype == "u32"]
    merge = [op for op in allocations if op.result_shapes[0].dtype == "s32"]
    _require(len(search) + len(merge) == len(allocations), "unknown scratch dtype")
    pairs = _scratch_pairs(index, search, rows, live, panel_search=True)
    return [dict(kind="panel_search_pair", **r) for r in pairs] + _merge_scratch(
        index, merge, live
    )


def check_rolled_helpers(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    _rows(block_rows)
    report = _check_helper_schedule(
        index,
        block_rows=block_rows,
        live_instructions=live_instructions,
        expected=_expected(block_rows),
        copy_limits=_copy_limits(),
        scratch_check=_scratch,
    )
    return {**report, "scope": "ROLLED_COMPILER_HELPER_AND_SCRATCH_STRUCTURE_ONLY"}
