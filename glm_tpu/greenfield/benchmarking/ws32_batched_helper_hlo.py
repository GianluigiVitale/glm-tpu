"""Acquired compiler helper structure, not numerical or cache identity proof.

Adapts the single-layer annotation/scratch checks using computation-local user
indexes. ConcatBitcast must not be interpreted as an ordinary concatenation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Callable, Sequence

from ..sharding.hlo_contract import HloInstruction
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _callee_attribute_text, _computation_base


_LAYER = re.compile(r"(?:^|/)greenfield_ws32_batched_prefill/layer_(\d+)(?:/|$)")
_SEARCH = "jit(searchsorted)/jit(_searchsorted_scan_impl)"


def _target(op: HloInstruction) -> str:
    header = op.raw_line.split(", metadata=", 1)[0].split(", backend_config=", 1)[0]
    names = re.findall(r'\bcustom_call_target="([^"\\]+)"', header)
    if len(names) != 1:
        raise ValueError(f"{op.name}: missing/ambiguous actual custom-call target")
    return names[0]


def _signature(op: HloInstruction) -> tuple:
    return tuple((s.dtype, s.dimensions) for s in op.result_shapes)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _expected(rows: int, *, paired_position_sort: bool = False) -> Counter:
    expected = Counter(
        {
            ("AssumeGatherIndicesInBound", "s32", (1024,)): 652 if rows == 17 else 651,
            ("AssumeGatherIndicesInBound", "s32", (4096,)): 42,
            ("AssumeGatherIndicesInBound", "s32", (rows * 2048,)): 99,
            ("GatherScatterIndicesBitpacked", "s32", (rows, 8, 2)): 75,
            ("GatherScatterIndicesBitpacked", "s32", (rows, 2048, 2)): 120,
            ("GatherScatterIndicesBitpacked", "s32", (rows, 4096, 2)): 42,
            ("GatherScatterIndicesBitpacked", "s32", (rows, 16384, 2)): 42,
            ("AllocateBuffer", "u32", (256,)): 450,
            ("ConcatBitcast", "bf16", (21, 16, 64, 128)): 5,
            ("ConcatBitcast", "bf16", (78, 16, 64, 640)): 1,
            ("ConcatBitcast", "bf16", (1024, 640)): 77,
            ("ConcatBitcast", "bf16", (8192, 64)): 1,
            ("ConcatBitcast", "f32", (128, 6144)): 14 if rows == 17 else 21,
            ("ConcatBitcast", "u8", (512, 2048)): 21,
            ("ConcatBitcast", "u8", (2048, 1536)): 228,
            ("ConcatBitcast", "u8", (2048, 2048)): 78,
            ("ConcatBitcast", "u8", (3584, 512)): 99,
            ("ConcatBitcast", "u8", (1536, 1536)): 9 if rows == 17 else 8,
            ("ConcatBitcast", "u8", (1536, 2048)): 153 if rows == 17 else 152,
            **({("ConcatBitcast", "s32", (278528,)): 20} if rows == 17 else {}),
        }
    )
    if paired_position_sort:
        # DB595→DB596 actual prefix removes two permutation-index helpers per
        # merge. The full engine has21 index producers, each with local+union
        # merges. B17's split union permutations also lose their concat helper.
        del expected[("GatherScatterIndicesBitpacked", "s32", (rows, 4096, 2))]
        del expected[("GatherScatterIndicesBitpacked", "s32", (rows, 16384, 2))]
        if rows == 17:
            del expected[("ConcatBitcast", "s32", (278528,))]
    return expected


def _paired_copy_limits() -> dict[tuple, int]:
    """Optional copies, bounded by maxima in the two original compiler graphs.

    Every present copy still requires same-source, complete-span, exclusive-use
    validation. This does not establish model ownership: the enclosing inspector
    separately requires kernel, cache, repair and health proofs. Removed s32
    permutation scaffolding is NOT optional and must remain absent.
    """
    return {
        key: count
        for key, count in (_expected(17) | _expected(11)).items()
        if key[0] == "ConcatBitcast" and key[1] != "s32"
    }


def _scratch_pairs(
    index: PrefillHloIndex,
    allocations: Sequence[HloInstruction],
    rows: int,
    live: set,
    *,
    panel_search: bool = False,
) -> list[dict[str, Any]]:
    """Six exact local allocations in three eight-leaf scan inits per MoE layer.

    Build users once per allocation-bearing computation, not 450 whole-module
    scans. This proves scratch containment, not the search algorithm itself.
    """
    total = 300 if panel_search else 450
    _require(len(allocations) == total, f"expected{total} local scratch allocations")
    by_computation: dict[tuple[str, str | None], list[HloInstruction]] = defaultdict(
        list
    )
    user_tables: dict[str, dict[str, list[HloInstruction]]] = {}

    def local_users(computation: str) -> dict[str, list[HloInstruction]]:
        if computation not in user_tables:
            table: dict[str, list[HloInstruction]] = defaultdict(list)
            for node in index.computations[computation].values():
                for name in set(node.operand_names):
                    table[name].append(node)
            user_tables[computation] = table
        return user_tables[computation]

    for op in allocations:
        computation = _computation_base(op.computation)
        init_name = None
        if panel_search:
            uses = local_users(computation)[op.name]
            _require(
                len(uses) == 1 and uses[0].opcode == "tuple",
                "panel scratch escapes initializer",
            )
            init_name = uses[0].name
        by_computation[(computation, init_name)].append(op)
    expected = (
        ("s32", ()),
        ("u32", (256,)),
        ("u32", (256,)),
        ("f32", (256,)),
        ("pred", (256,)),
        ("f32", (rows + 1,)),
        ("s32", ()),
        ("s32", ()),
    )
    pairs = []
    for (computation, _), local_allocations in by_computation.items():
        _require(
            len(local_allocations) == 2, "scratch must pair within one computation"
        )
        users = local_users(computation)
        inits = []
        for allocation in local_allocations:
            uses = users[allocation.name]
            _require(
                len(uses) == 1 and uses[0].opcode == "tuple",
                "scratch escapes its scan initializer",
            )
            inits.append(uses[0])
        init = inits[0]
        names = {op.name for op in local_allocations}
        if panel_search:
            size = local_allocations[0].result_shapes[0].dimensions[0]
            panels = (8 * rows + 31) // 32 + 31
            _require(
                size in (8 * rows, panels), "unregistered panel-search scratch size"
            )
            expected = (
                ("s32", ()),
                ("u32", (size,)),
                ("u32", (size,)),
                ("s32", (256 if size == 8 * rows else 32,)),
                ("s32", ()),
                ("s32", ()),
                ("s32", ()),
            )
        _require(
            inits[1].name == init.name
            and len(init.operand_names) == len(expected)
            and set(init.operand_names[1:3]) == names
            and not (set(init.operand_names[:1] + init.operand_names[3:]) & names)
            and _signature(init) == expected,
            "scratch scan pair/tuple slots drifted",
        )
        loops = users[init.name]
        _require(len(loops) == 1, "scratch initializer has extra users")
        loop = loops[0]
        _require(
            loop.raw_opcode == "while"
            and loop.operand_names == (init.name,)
            and loop.result_shapes == init.result_shapes
            and (loop.computation, loop.name) in live
            and (init.computation, init.name) in live
            and _SEARCH in (loop.op_name or ""),
            "scratch is not the acquired live searchsorted while",
        )
        layers = _LAYER.findall(loop.op_name or "")
        _require(
            len(layers) == 1 and 3 <= int(layers[0]) < 78,
            "scratch loop lacks one MoE layer",
        )
        index.callee(loop, "condition")
        index.callee(loop, "body")
        pairs.append(
            dict(
                layer=int(layers[0]),
                computation=computation,
                initializer=init.name,
                loop=loop.name,
                allocations=sorted(names),
                **({"scratch_size": size} if panel_search else {}),
            )
        )
    _require(
        Counter(p["layer"] for p in pairs)
        == Counter({layer: 2 if panel_search else 3 for layer in range(3, 78)}),
        "unexpected scratch loops per MoE layer",
    )
    if panel_search:
        _require(
            Counter((p["layer"], p["scratch_size"]) for p in pairs)
            == Counter(
                (layer, size)
                for layer in range(3, 78)
                for size in (8 * rows, (8 * rows + 31) // 32 + 31)
            ),
            "panel-search pair coverage drift",
        )
    return sorted(pairs, key=lambda p: (p["layer"], p["computation"], p["initializer"]))


def check_batched_helpers(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
    paired_position_sort: bool = False,
) -> dict[str, Any]:
    """Inspect acquired B17/B11 helper operands; cannot authorize execution."""
    if type(block_rows) is not int or block_rows not in (11, 17):
        raise ValueError("helper profile is registered only for B17/B11")
    if type(paired_position_sort) is not bool:
        raise ValueError("paired position sort must be a static bool")
    return _check_helper_schedule(
        index,
        block_rows=block_rows,
        live_instructions=live_instructions,
        expected=_expected(block_rows, paired_position_sort=paired_position_sort),
        copy_limits=_paired_copy_limits() if paired_position_sort else None,
    )


def _check_helper_schedule(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
    expected: Counter,
    copy_limits: dict[tuple, int] | None = None,
    count_limits: dict[tuple, int] | None = None,
    scratch_check: Callable[..., list[dict[str, Any]]] = _scratch_pairs,
) -> dict[str, Any]:
    """Reuse exact interfaces/completion checks with a fixed helper schedule.

    Distinct diagnostic profiles may pre-register bounded annotation counts.
    Historical profiles leave count_limits unset and retain exact counts.
    """
    report: dict[str, Any] = dict(
        passed=False,
        scope="SHORT_PREFILL_COMPILER_HELPER_STRUCTURE_ONLY",
        not_proven=[
            "PALLAS_KERNEL_SEMANTICS",
            "CACHE_RECONSTRUCTION_IDENTITY_OR_OWNERSHIP",
            "SEARCH_ALGORITHM_CORRECTNESS",
            "NUMERICAL_OR_MEMORY_ADMISSION",
        ],
    )
    live = {(op.computation, op.name) for op in live_instructions}
    expected, observed = expected.copy(), Counter()
    allocations = []
    concats = []
    try:
        for op in index.module.instructions:
            if op.raw_opcode != "custom-call":
                continue
            target = _target(op)
            if target == "tpu_custom_call":
                continue  # separate Pallas contract, not silently admitted here
            _require(
                len(op.result_shapes) == 1 and (op.computation, op.name) in live,
                "helper shape/liveness drifted",
            )
            flags = re.findall(
                r"\bcustom_call_has_side_effect=([\w]+)",
                _callee_attribute_text(op.raw_line),
            )
            _require(
                flags in ([], ["false"]),
                "helper side-effect attribute is not absent/false",
            )
            shape = op.result_shapes[0]
            key = (target, shape.dtype, shape.dimensions)
            _require(key in expected, f"unregistered helper signature:{key}")
            observed[key] += 1
            if target in (
                "AssumeGatherIndicesInBound",
                "GatherScatterIndicesBitpacked",
            ):
                _require(
                    len(op.operand_names) == 1
                    and op.operand_shapes == op.result_shapes
                    and index.operand(op, 0).result_shapes == op.result_shapes
                    and bool(op.op_name and op.op_name.endswith("/gather")),
                    "index annotation operand/scope drifted",
                )
            elif target == "AllocateBuffer":
                _require(
                    not op.operand_names and not op.operand_shapes,
                    "scratch allocation has inputs",
                )
                allocations.append(op)
            else:
                concats.append(op)
        users = _slice_users(index, concats)
        for op in concats:
            _concat_structure(index, op, users)
        report["scratch_pairs"] = scratch_check(index, allocations, block_rows, live)

        if count_limits is not None:
            report["bounded_annotation_counts"] = []
            for key, maximum in count_limits.items():
                _require(
                    key in expected
                    and key[0]
                    in ("AssumeGatherIndicesInBound", "GatherScatterIndicesBitpacked")
                    and type(maximum) is int
                    and maximum >= 0,
                    "only registered index annotation counts may be bounded",
                )
                count = observed[key]
                _require(count <= maximum, f"annotation count exceeds bound:{key}")
                report["bounded_annotation_counts"].append(
                    dict(
                        target=key[0],
                        dtype=key[1],
                        dimensions=list(key[2]),
                        count=count,
                        maximum=maximum,
                    )
                )
                expected[key] = count

        if copy_limits is not None:
            report["bounded_copy_counts"] = []
            for key, maximum in copy_limits.items():
                count = observed[key]
                _require(count <= maximum, f"compiler copy count exceeds bound:{key}")
                report["bounded_copy_counts"].append(
                    dict(
                        target=key[0],
                        dtype=key[1],
                        dimensions=list(key[2]),
                        count=count,
                        maximum=maximum,
                    )
                )
                # Only multiplicity is compiler-dependent. All other helper
                # families keep their exact counts, including removed gathers.
                expected[key] = count

        def records(counter):
            return [
                dict(target=k[0], dtype=k[1], dimensions=k[2], count=v)
                for k, v in sorted(counter.items())
            ]

        report.update(
            passed=observed == expected,
            helper_count=sum(observed.values()),
            missing=records(expected - observed),
            unexpected=records(observed - expected),
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report


def _slice_users(index: PrefillHloIndex, concats: Sequence[HloInstruction]) -> dict:
    """Index only watched slice handles, once, without all-module user storage."""
    watched = set()
    for op in concats:
        for i in range(len(op.operand_names)):
            done = index.operand(op, i)
            watched.add((done.computation, done.name))
            if len(done.operand_names) == 1:
                start = index.operand(done, 0)
                watched.add((start.computation, start.name))
    users: dict[tuple, list] = defaultdict(list)
    for op in index.module.instructions:
        for name in set(op.operand_names):
            key = (op.computation, name)
            if key in watched:
                users[key].append(op)
    return users


def _concat_structure(index: PrefillHloIndex, op: HloInstruction, users: dict) -> None:
    """Adapt existing closed WK slice scaffolding to the acquired helper families.

    Slice attributes carry placement. Nonascending operand order is present in
    historical protected graphs AND both new captures. This checks that same
    compiler mechanism; it does not resolve an arbitrary concat as SSA identity.
    """
    full = _signature(op)
    _require(
        len(full) == 1
        and len(op.operand_names) == len(op.operand_shapes) == 4
        and len(set(op.operand_names)) == 4,
        "ConcatBitcast needs four distinct slices",
    )
    dtype, dims = full[0]
    axis = 1 if (dtype, dims) == ("bf16", (8192, 64)) else 0
    if dims == (21, 16, 64, 128):
        spans = ((0, 6), (6, 12), (12, 18), (18, 21))
    elif dims == (78, 16, 64, 640):
        spans = ((0, 20), (20, 40), (40, 60), (60, 78))
    else:
        _require(dims[axis] % 4 == 0, "ConcatBitcast split not in acquired families")
        width = dims[axis] // 4
        spans = tuple((i * width, (i + 1) * width) for i in range(4))
    expected_ranges = set()
    for span in spans:
        ranges = [(0, d) for d in dims]
        ranges[axis] = span
        expected_ranges.add(tuple(ranges))
    seen, sources = set(), set()
    for i in range(4):
        done = index.operand(op, i)
        _require(
            done.raw_opcode == "slice-done"
            and len(done.operand_names) == 1
            and _signature(done)
            == ((op.operand_shapes[i].dtype, op.operand_shapes[i].dimensions),),
            "ConcatBitcast operand is not a completed slice",
        )
        start = index.operand(done, 0)
        part = _signature(done)
        _require(
            start.raw_opcode == "slice-start"
            and len(start.operand_names) == 1
            and _signature(start) == full + part + (("s32", ()),)
            and tuple((s.dtype, s.dimensions) for s in start.operand_shapes) == full
            and done.operand_shapes == start.result_shapes,
            "asynchronous slice handle/source/result shape drifted",
        )
        source = index.operand(start, 0)
        _require(_signature(source) == full, "ConcatBitcast source shape drifted")
        sources.add((source.computation, source.name))
        attrs = re.findall(
            r"\bslice=\{([^}]*)\}", _callee_attribute_text(start.raw_line)
        )
        _require(len(attrs) == 1, "missing/ambiguous slice spans")
        text = re.sub(r"\s", "", attrs[0])
        _require(
            re.fullmatch(r"\[\d+:\d+\](?:,\[\d+:\d+\])*", text) is not None,
            "unsupported slice stride/range syntax",
        )
        ranges = tuple(
            (int(a), int(b)) for a, b in re.findall(r"\[(\d+):(\d+)\]", text)
        )
        _require(
            ranges in expected_ranges
            and ranges not in seen
            and part == ((dtype, tuple(b - a for a, b in ranges)),),
            "overlapping/incomplete/wrong-axis slice coverage",
        )
        seen.add(ranges)
        _require(
            users[(start.computation, start.name)] == [done]
            and users[(done.computation, done.name)] == [op],
            "slice scaffolding has an escaping consumer",
        )
    _require(
        len(sources) == 1 and seen == expected_ranges,
        "ConcatBitcast mixes sources or lacks complete coverage",
    )
