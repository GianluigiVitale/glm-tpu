"""Exact DB611 helper interfaces and initialized scratch, not model admission.

Reuse the original helper/copy/search/merge checks. Fixed acquired exceptions
stay local: exact-decode concatenate annotations, axis2 complete copies, B128
completed panel copies, and fully written observer 32-row buffers. Source/HLO
pins, physical collectives, kernels and loop admission belong to the caller.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import copy
from dataclasses import replace
import re
from typing import Any, Sequence

from glm_tpu.greenfield.sharding.hlo_contract import HloInstruction
from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import (
    _check_helper_schedule, _require, _scratch_pairs, _signature, _slice_users, _target,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import _callee_attribute_text, _computation_base
from glm_tpu.greenfield.benchmarking.ws32_prefill_hlo_identity import PrefillIdentity, Value
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_helper_hlo import _merge_scratch

PROGRAMS = ("candidate_b128", "candidate_b114", "control_b128", "control_b114",
            "exact_decode", "exact_promote", "observer")
AXIS2 = ("ConcatBitcast", "u8", (32, 6144, 82))
OBSERVER_SCRATCH = ("AllocateBuffer", "bf16", (32, 1, 1536))


def expected_helpers(name: str) -> Counter:
    """Exact acquired multiplicities, including compiler copies; no upper bounds."""
    _require(name in PROGRAMS, "unregistered history helper program")
    if name in PROGRAMS[:4]:
        rows = int(name.rsplit("b", 1)[1])
        copies = (("u8", (1536, 2048), 11), ("u8", (2048, 1536), 8), ("bf16", (8192, 64), 1))
        if rows == 128:
            copies += (("u8", (3584, 512), 14), ("u8", (512, 2048), 8),
                       ("u8", (1536, 1536), 3), ("bf16", (16, 64, 640), 7))
        else:
            copies += (("u8", (2048, 2048), 7), ("u8", (3584, 512), 7),
                       ("s32", (4, 32, 2048), 5), ("f32", (4, 32, 2048), 5),
                       ("bf16", (16, 64, 640), 4))
            if name == "control_b114":
                copies += (("u8", (1536, 1536), 6),)
        return Counter({
            ("AssumeGatherIndicesInBound", "s32", (1024,)): 91,
            ("AssumeGatherIndicesInBound", "s32", (65536,)): 11,
            ("AssumeGatherIndicesInBound", "s32", (2048,)): 12,
            ("AssumeGatherIndicesInBound", "u32", (1024,)): 8,
            ("GatherScatterIndicesBitpacked", "s32", (32, 2048, 2)): 11,
            ("GatherScatterIndicesBitpacked", "s32", (rows, 8, 2)): 4,
            ("AllocateBuffer", "s32", (32, 2, 2, 512)): 8,
            ("AllocateBuffer", "u32", (8 * rows,)): 8,
            ("AllocateBuffer", "u32", ((8 * rows + 31) // 32 + 31,)): 8,
            **{("ConcatBitcast", dtype, dims): n for dtype, dims, n in copies},
        })
    if name == "exact_decode":
        return Counter({
            ("AssumeGatherIndicesInBound", "s32", (2097152,)): 4,
            ("AssumeGatherIndicesInBound", "s32", (786432,)): 4,
            ("GatherScatterIndicesBitpacked", "s32", (1024, 2048, 2)): 4,
            ("GatherScatterIndicesBitpacked", "s32", (128, 6144, 2)): 4,
            ("ConcatBitcast", "s32", (128, 16, 8, 128)): 3,
            ("ConcatBitcast", "s32", (16, 48, 8, 128)): 3,
            ("ConcatBitcast", "u8", (2048, 1536)): 4,
        })
    if name == "exact_promote":
        return Counter({("ConcatBitcast", "f32", (1024, 2048)): 4, AXIS2: 3})
    return Counter({
        ("AssumeGatherIndicesInBound", "s32", (1024,)): 27,
        ("AssumeGatherIndicesInBound", "s32", (2048,)): 15,
        ("AssumeGatherIndicesInBound", "s32", (16384,)): 8,
        OBSERVER_SCRATCH: 3,
        **{("ConcatBitcast", dtype, dims): n for dtype, dims, n in (
            ("f32", (1024, 2048), 12), ("f32", (128, 6144), 4),
            ("bf16", (1024, 640), 3), ("u8", (2048, 2048), 7),
            ("u8", (3584, 512), 7), ("u8", (1536, 2048), 11),
            ("u8", (4, 6144, 768), 3), ("u8", (4, 384, 1536), 3),
            ("u8", (32, 6144, 82), 3), ("u8", (2048, 1536), 11),
        )},
    })


def _axis2_copy(index: PrefillHloIndex, op: HloInstruction, users: dict) -> dict:
    """The acquired nonuniform 21/21/21/19 byte spans, in original operand order."""
    full = (("u8", (32, 6144, 82)),)
    spans = ((0, 21), (21, 42), (42, 63), (63, 82))
    _require(_signature(op) == full and len(op.operand_names) == len(op.operand_shapes) == 4
             and len(set(op.operand_names)) == 4, "history axis2 copy interface differs")
    sources = set()
    for i, (lo, hi) in enumerate(spans):
        done = index.operand(op, i)
        part = (("u8", (32, 6144, hi - lo)),)
        _require(done.raw_opcode == "slice-done" and len(done.operand_names) == 1
                 and _signature(done) == part
                 and (op.operand_shapes[i].dtype, op.operand_shapes[i].dimensions) == part[0],
                 "history axis2 copy operand is not the ordered completed span")
        start = index.operand(done, 0)
        _require(start.raw_opcode == "slice-start" and len(start.operand_names) == 1
                 and _signature(start) == full + part + (("s32", ()),)
                 and tuple((s.dtype, s.dimensions) for s in start.operand_shapes) == full
                 and done.operand_shapes == start.result_shapes,
                 "history axis2 copy asynchronous handle differs")
        source = index.operand(start, 0)
        _require(_signature(source) == full, "history axis2 copy source shape differs")
        sources.add((source.computation, source.name))
        spans_text = re.findall(r"\bslice=\{([^}]*)\}", _callee_attribute_text(start.raw_line))
        _require(len(spans_text) == 1 and re.sub(r"\s", "", spans_text[0])
                 == f"[0:32],[0:6144],[{lo}:{hi}]", "history axis2 copy span/order differs")
        _require(users[(start.computation, start.name)] == [done]
                 and users[(done.computation, done.name)] == [op],
                 "history axis2 copy has an escaping consumer")
    _require(len(sources) == 1, "history axis2 copy mixes sources")
    return dict(kind="axis2_complete_ordered_copy", helper=op.name, axis=2,
                spans=[list(v) for v in spans])


def _observer_scratch(index: PrefillHloIndex, allocations: Sequence[HloInstruction], live: set) -> list[dict]:
    """All 32 row writes complete before any observer temporary leaves its chain."""
    _require(len(allocations) == 3, "history observer requires three scratch buffers")
    users = defaultdict(list)
    callers = defaultdict(list)
    for op in index.module.instructions:
        comp = _computation_base(op.computation)
        for name in set(op.operand_names):
            users[(comp, name)].append(op)
        if op.opcode == "fusion":
            callers[index.callee(op, "calls")].append(op)
    ssa = PrefillIdentity(index)
    full, row_shape = (("bf16", (32, 1, 1536)),), (("bf16", (1, 1, 1536)),)

    def uses(op):
        return users[(_computation_base(op.computation), op.name)]

    def write(op, row):
        _require(op.raw_opcode == "dynamic-update-slice" and len(op.operand_names) == 5
                 and _signature(op) == full and _signature(index.operand(op, 0)) == full
                 and _signature(index.operand(op, 1)) == row_shape
                 and tuple((s.dtype, s.dimensions) for s in op.operand_shapes)
                 == full + row_shape + (("s32", ()),) * 3,
                 "history observer scratch row-write geometry differs")
        _require(all(ssa.constant(Value(index.operand(op, j + 2)), "s32", offset)
                     for j, offset in enumerate((row, 0, 0))),
                 "history observer scratch row offsets leave a hole")
        _require((op.computation, op.name) in live, "history observer scratch write is dead")

    records = []
    for allocation in allocations:
        _require(_signature(allocation) == full and not allocation.operand_names
                 and not allocation.operand_shapes, "history observer allocation interface differs")
        comp = _computation_base(allocation.computation)
        root = index.roots[comp]
        write(root, 0)
        _require(uses(allocation) == [root] and root.operand_names[0] == allocation.name
                 and root.operand_names.count(allocation.name) == 1,
                 "history observer uninitialized storage escapes")
        _require(len(callers[comp]) == 1, "history observer scratch initializer caller differs")
        current = callers[comp][0]
        chain = [current.name]
        _require(_signature(current) == full and (current.computation, current.name) in live,
                 "history observer scratch initializer is dead or malformed")
        for row in range(1, 32):
            consumers = uses(current)
            _require(len(consumers) == 1 and consumers[0].opcode == "fusion",
                     "history observer partial scratch escapes")
            completed = consumers[0]
            _require(completed.operand_names.count(current.name) == 1
                     and _signature(completed) == full and (completed.computation, completed.name) in live,
                     "history observer partial scratch repeats/dead binding")
            body = index.callee(completed, "calls")
            _require(callers[body] == [completed], "history observer scratch completion has multiple callers")
            root = index.roots[body]
            write(root, row)
            base = index.operand(root, 0)
            number = re.findall(r"\bparameter\((\d+)\)", base.raw_line)
            _require(base.opcode == "parameter" and len(number) == 1
                     and int(number[0]) < len(completed.operand_names)
                     and completed.operand_names[int(number[0])] == current.name
                     and completed.operand_shapes[int(number[0])] == current.result_shapes[0]
                     and uses(base) == [root] and root.operand_names.count(base.name) == 1,
                     "history observer partial parameter escapes or changes base")
            current = completed
            chain.append(current.name)
        final = uses(current)
        _require(len(final) == 1 and final[0].opcode == "reduce"
                 and final[0].operand_names.count(current.name) == 1
                 and (final[0].computation, final[0].name) in live,
                 "history observer completed scratch consumer differs")
        records.append(dict(kind="observer_complete_rows", allocation=allocation.name,
                            rows=32, chain=chain, consumer=final[0].name))
    return records


def _history_search(index: PrefillHloIndex, allocations: Sequence[HloInstruction], rows: int, live: set) -> list[dict]:
    """Eight B128 small panels use closed completed copies; B114 stays direct."""
    users = defaultdict(list)
    for op in index.module.instructions:
        for name in set(op.operand_names):
            users[(_computation_base(op.computation), name)].append(op)

    def uses(op):
        return users[(_computation_base(op.computation), op.name)]

    panels = (8 * rows + 31) // 32 + 31
    forwarded, effective = {}, []
    for allocation in allocations:
        if rows == 114 or _signature(allocation) != (("u32", (panels,)),):
            effective.append(allocation)
            continue
        consumers = uses(allocation)
        _require(len(consumers) == 1 and consumers[0].raw_opcode == "copy-start",
                 "history panel allocation escapes its copy")
        start = consumers[0]
        full = _signature(allocation)
        _require(start.operand_names == (allocation.name,) and _signature(start) == full + full + (("u32", ()),)
                 and tuple((s.dtype, s.dimensions) for s in start.operand_shapes) == full,
                 "history panel copy handle/source differs")
        consumers = uses(start)
        _require(len(consumers) == 1 and consumers[0].raw_opcode == "copy-done",
                 "history panel copy handle escapes")
        done = consumers[0]
        _require(done.operand_names == (start.name,) and done.operand_shapes == start.result_shapes
                 and _signature(done) == full
                 and all((op.computation, op.name) in live for op in (start, done)),
                 "history panel copy incomplete/dead")
        forwarded[done.name] = dict(allocation=allocation.name, start=start.name, done=done.name)
        effective.append(done)
    _require(len(forwarded) == (8 if rows == 128 else 0), "history panel completed-copy count differs")
    pairs = _scratch_pairs(index, effective, rows, live, panel_search=True, layer_ids=(3, 4, 5, 6))
    for row in pairs:
        row["scratch_copies"] = [forwarded[n] for n in row["allocations"] if n in forwarded]
        row["allocations"] = sorted(forwarded[n]["allocation"] if n in forwarded else n for n in row["allocations"])
    return [dict(kind="panel_search_pair", **r) for r in pairs]


def check_history_helpers(index: PrefillHloIndex, *, name: str,
                          live_instructions: Sequence[HloInstruction]) -> dict[str, Any]:
    """Require exact inventory first; every exception and remaining helper is checked."""
    report = dict(passed=False, scope="HISTORY_COMPILER_HELPERS_AND_INITIALIZED_SCRATCH_ONLY",
                  numerical_promotion=False, performance_claim=False,
                  not_proven=["MODEL_OR_CACHE_SEMANTICS", "NUMERICAL_OR_MEMORY_ADMISSION"])
    try:
        expected = expected_helpers(name)
        live = {(op.computation, op.name) for op in live_instructions}
        custom = [op for op in index.module.instructions if op.raw_opcode == "custom-call"
                  and _target(op) != "tpu_custom_call"]
        observed = Counter()
        for op in custom:
            _require(len(op.result_shapes) == 1 and (op.computation, op.name) in live,
                     "history helper shape/liveness differs")
            flags = re.findall(r"\bcustom_call_has_side_effect=([\w]+)", _callee_attribute_text(op.raw_line))
            _require(flags in ([], ["false"]), "history helper side effect differs")
            observed[(_target(op), *_signature(op)[0])] += 1
        _require(observed == expected, "history exact helper inventory differs")
        special, proofs = [], []
        if name == "exact_decode":
            annotations = [op for op in custom if _target(op) == "GatherScatterIndicesBitpacked"]
            _require({op.name for op in annotations} == {f"%custom-call.{i}" for i in range(1, 16, 2)},
                     "history exact-decode annotation inventory differs")
            scopes = Counter()
            for op in annotations:
                dims = op.result_shapes[0].dimensions
                role = "tuple4_query" if dims == (1024, 2048, 2) else "wk_decode"
                scope = f"jit(decode_body)/shard_map/greenfield_ws32_exact_dsa_materializer/{role}/concatenate"
                _require(op.op_name == scope and len(op.operand_names) == 1
                         and op.operand_shapes == op.result_shapes
                         and index.operand(op, 0).result_shapes == op.result_shapes,
                         "history exact-decode annotation operand/scope differs")
                scopes[role] += 1
            _require(scopes == Counter(tuple4_query=4, wk_decode=4), "history decode annotation role coverage differs")
            special.extend(annotations)
            proofs.extend(dict(kind="exact_decode_concatenate_annotation", helper=op.name) for op in annotations)
        if name in ("exact_promote", "observer"):
            copies = [op for op in custom if (_target(op), *_signature(op)[0]) == AXIS2]
            first = 4 if name == "exact_promote" else 112
            _require({op.name for op in copies} == {f"%custom-call.{i}" for i in range(first, first + 3)}
                     and all(_computation_base(op.computation) == "ENTRY" for op in copies),
                     "history axis2 helper inventory/scope differs")
            users = _slice_users(index, copies)
            proofs.extend(_axis2_copy(index, op, users) for op in copies)
            special.extend(copies)
        if name == "observer":
            allocations = [op for op in custom if _target(op) == "AllocateBuffer"]
            _require({op.name for op in allocations} == {f"%custom-call.{i}" for i in range(103, 106)},
                     "history observer allocation inventory differs")
            proofs.extend(_observer_scratch(index, allocations, live))
            special.extend(allocations)

        def scratch(_view, allocations, rows, live_set):
            if name not in PROGRAMS[:4]:
                _require(not allocations, "unexpected history generic scratch")
                return []
            search = [op for op in allocations if op.result_shapes[0].dtype == "u32"]
            merge = [op for op in allocations if op.result_shapes[0].dtype == "s32"]
            _require(len(search) + len(merge) == len(allocations), "unknown history scratch dtype")
            return _history_search(index, search, rows, live_set) + _merge_scratch(
                    index, merge, live_set, layer_ids=(0, 1, 2, 6))

        # Do not remove exception operands/users from the generic copy scan.
        # Only bypass its target classification for the already-proved exact
        # subset. Original computation/root indexes and every use remain intact.
        checked = {op.index for op in special}
        view = copy(index)
        view.module = replace(index.module, instructions=tuple(
            replace(op, raw_opcode="history-checked-helper") if op.index in checked else op
            for op in index.module.instructions))
        remaining = expected - Counter((_target(op), *_signature(op)[0]) for op in special)
        generic = _check_helper_schedule(view, block_rows=int(name.rsplit("b", 1)[1]) if name in PROGRAMS[:4] else 1,
            live_instructions=live_instructions, expected=remaining, scratch_check=scratch)
        _require(generic["passed"], f"history generic helper structure differs:{generic}")
        report.update(passed=True, helper_count=len(custom), generic=generic, exceptions=proofs,
                      scratch_pairs=generic["scratch_pairs"] + [p for p in proofs if p["kind"] == "observer_complete_rows"])
    except (ValueError, KeyError, IndexError, TypeError) as error:
        report["error"] = str(error)
    return report
