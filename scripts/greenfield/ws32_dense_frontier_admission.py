"""Fixed two-layer diagnostic graph contract, not full-model admission.

Reuse physical groups/reducers, Pallas interfaces and narrow helper completion
checks. No MoE, route or full-decoder symbolic proof is required for two dense
layers. Numerical attribution additionally requires exact DB604 byte reproduction.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
from typing import Any, Callable, Mapping

from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex
from glm_tpu.greenfield.benchmarking.ws32_batched_commit_hlo import _require, _shape
from glm_tpu.greenfield.benchmarking.ws32_batched_collective_hlo import (
    _physical_records,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_kernel_hlo import (
    _check_kernel_schedule,
)
from glm_tpu.greenfield.benchmarking.ws32_batched_helper_hlo import (
    _check_helper_schedule,
    _target,
)
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    _computation_base,
    _live_instruction_closure,
    _callee_attribute_text,
)
from glm_tpu.greenfield.benchmarking.ws32_prefill_hlo_identity import Value, attribute
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_hlo import RolledTransitions
from glm_tpu.greenfield.benchmarking.ws32_prefill_fixed_loops import fixed_loop_bodies
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_collective_hlo import (
    _expected as rolled_collectives,
)
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_kernel_hlo import (
    _expected as rolled_kernels,
)
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_helper_hlo import (
    _merge_scratch,
)
from scripts.greenfield.prefill_window_admission import (
    expected_collectives as wk_collectives,
)
from scripts.greenfield.prefill_layer_hlo import FEATURE, EXPERT

# Verbatim (b667f00f) constants of the retired ws32_dense_frontier_worker.py.
GRAPH = "dense01"
PROGRAMS = ("wk_decode", "wk_promote", GRAPH)

PROFILE = "ws32-dense01-db604-original-reproduction-v1"
RAW = {
    "wk_decode": (
        10725,
        "8eeefbb0cbc3518ac223b49e1bade70dc1aa78c965c983ebef83c0140284c362",
    ),
    "wk_promote": (
        802,
        "7b277bb821af372bd03687010b1db3630533dc9db46747863dd08cc0742006e5",
    ),
    "dense01": (
        653497,
        "4f2ec6ed56ac1485bfda976882c74211e168a667018b6caf06be0ae5f4eccb60",
    ),
}
MEMORY_CAPS = dict(
    argument_size_in_bytes=512 << 20,
    output_size_in_bytes=32 << 20,
    alias_size_in_bytes=0,
    temp_size_in_bytes=512 << 20,
    generated_code_size_in_bytes=128 << 20,
)
LAYER = re.compile(r"(?:^|/)greenfield_ws32_batched_prefill/layer_(\d+)(?:/|$)")
LOOP = "greenfield_ws32_prefill_rolled_prefix/while"


def prefix_bodies(
    index: PrefillHloIndex, live: tuple, *, loop_suffix: str = LOOP
) -> dict[str, int]:
    """Only the two original outer-loop counters, not model arithmetic."""
    return fixed_loop_bodies(
        index, live, loop_suffix=loop_suffix, expected_layers=(0, 1)
    )


def _flatten_placed(counter: Counter) -> Counter:
    result = Counter()
    for key, count in counter.items():
        _require(len(key[-2]) == len(key[-1]), "dense tuple leaf arity differs")
        for left, right in zip(key[-2], key[-1], strict=True):
            result[(*key[:-2], left, right)] += count
    return result


def expected_collectives() -> Counter:
    """Reuse the existing source schedule; remove every non-dense/head/vote op."""
    expected = Counter(
        {
            key: count
            for key, count in rolled_collectives(128).items()
            if key[1] in (0, 1)
            or key
            == (
                "outer",
                -1,
                "all-reduce",
                "expert",
                "add",
                -1,
                (("bf16", (128, 1536)),),
                (("bf16", (128, 1536)),),
            )
        }
    )
    return _flatten_placed(expected)


def check_collectives(
    index: PrefillHloIndex,
    live: tuple,
    bodies: dict[str, int],
    *,
    suffix_bodies: dict[str, int] | None = None,
) -> dict:
    records, votes = _physical_records(index, live)
    _require(not votes, "dense diagnostic has no scalar health-vote collectives")
    actual = Counter()
    for op, key in records:
        comp, layer = _computation_base(op.computation), key[0]
        if comp in bodies:
            _require(bodies[comp] == layer, "dense collective outside own prefix body")
            place = "prefix"
        elif suffix_bodies is not None and layer != -1:
            _require(
                suffix_bodies.get(comp) == layer,
                "dense collective outside own canonical suffix body",
            )
            place = "suffix"
        else:
            _require(comp == "ENTRY", "dense non-prefix collective is not ENTRY")
            place = "outer" if layer == -1 else "suffix"
        actual[(place, *key)] += 1
    flat = _flatten_placed(actual)
    _require(
        flat == expected_collectives(),
        "dense physical collective leaf schedule differs",
    )
    return dict(
        static_instruction_count=len(records),
        static_leaf_pairs=sum(flat.values()),
        four_iteration_schedule_leaf_pairs=sum(
            n
            * (
                4
                if k[0] == "prefix" or (suffix_bodies is not None and k[0] == "suffix")
                else 1
            )
            for k, n in flat.items()
        ),
        physical_groups=dict(feature=FEATURE, expert=EXPERT),
        measured_dynamic_count_claim=False,
    )


def check_kernels(
    index: PrefillHloIndex,
    live: tuple,
    bodies: dict[str, int],
    *,
    suffix_bodies: dict[str, int] | None = None,
) -> dict:
    all_expected, places = rolled_kernels(128)
    expected = Counter({k: v for k, v in all_expected.items() if k[0] in (0, 1)})

    def placement(op: Any, key: tuple) -> None:
        _require(key in expected, "unregistered dense kernel interface")
        comp = _computation_base(op.computation)
        suffix_ok = (
            comp == "ENTRY"
            if suffix_bodies is None
            else suffix_bodies.get(comp) == key[0]
        )
        _require(
            bodies.get(comp) == key[0] if places[key] == "prefix" else suffix_ok,
            "dense kernel outside own prefix or wide suffix",
        )

    report = _check_kernel_schedule(
        index,
        live_instructions=live,
        expected=expected,
        placement_check=placement,
        families=("raw", "structured", "sparse"),
    )
    _require(report["passed"], f"dense kernel schedule differs:{report}")
    return report


def check_helpers(
    index: PrefillHloIndex,
    live: tuple,
    *,
    extra_allocations: Mapping[tuple, int] | None = None,
    scratch_check: Callable | None = None,
) -> dict:
    # Fixed shape-specific upper bounds. The first reduced compilation exposed
    # four already-known closed-copy families omitted/undercounted here; see the
    # preserved 20260909T151822927153523Z refusal. No new opaque operation class.
    # Optional annotations/copies are not model operations; each present helper
    # still passes original operand, side-effect, complete-span or scratch checks.
    annotations = {
        ("AssumeGatherIndicesInBound", "s32", (1024,)): 128,
        ("AssumeGatherIndicesInBound", "s32", (65536,)): 8,
        ("GatherScatterIndicesBitpacked", "s32", (32, 2048, 2)): 8,
    }
    copies = {
        ("ConcatBitcast", dtype, dims): maximum
        for dtype, dims, maximum in (
            ("bf16", (8192, 64), 1),
            ("bf16", (19360, 1536), 1),
            ("bf16", (16, 64, 640), 2),
            ("bf16", (1024, 640), 2),
            ("bf16", (16, 64, 128), 4),
            ("u8", (2048, 1536), 4),
            ("u8", (1536, 2048), 2),
            ("u8", (2048, 2048), 2),
            ("u8", (3584, 512), 4),
            ("u8", (1536, 1536), 6),
            ("u8", (512, 2048), 4),
            ("s32", (4, 32, 2048), 4),
            ("f32", (4, 32, 2048), 4),
            ("f32", (128, 6144), 2),
        )
    }
    expected = Counter(
        {**annotations, **copies, ("AllocateBuffer", "s32", (32, 2, 2, 512)): 4}
    )
    if extra_allocations is not None:
        _require(
            all(
                k[0] == "AllocateBuffer" and k not in expected
                for k in extra_allocations
            )
            and scratch_check is not None,
            "extra diagnostic allocations require a distinct completion checker",
        )
        expected.update(extra_allocations)

    def scratch(index, allocations, rows, live):
        return _merge_scratch(index, allocations, live, layer_ids=(0, 1))

    report = _check_helper_schedule(
        index,
        block_rows=128,
        live_instructions=live,
        expected=expected,
        copy_limits=copies,
        count_limits=annotations,
        scratch_check=scratch if scratch_check is None else scratch_check,
    )
    _require(report["passed"], f"dense helper interfaces/completion differ:{report}")
    return report


def validate_memory(name: str, memory: Mapping[str, Any]) -> None:
    _require(
        name in PROGRAMS
        and set(memory) == set(MEMORY_CAPS)
        and all(type(v) is int and 0 <= v <= MEMORY_CAPS[k] for k, v in memory.items()),
        "dense compiled memory inventory/caps differ",
    )


def check_wk_helpers(index: PrefillHloIndex, live: tuple, name: str) -> dict:
    """Original WK's two index annotations; no opaque compute or allocation.

    The bitpacked annotation is attached to concatenate, not gather, in the
    retained WK compiler graph. Keep this exception local to this fixed role.
    """
    expected = (
        Counter(
            {
                ("AssumeGatherIndicesInBound", "s32", (786432,), "/gather"): 1,
                (
                    "GatherScatterIndicesBitpacked",
                    "s32",
                    (128, 6144, 2),
                    "/concatenate",
                ): 1,
            }
        )
        if name == "wk_decode"
        else Counter()
    )
    seen = Counter()
    live_ids = {op.index for op in live}
    for op in index.module.instructions:
        if op.raw_opcode != "custom-call":
            continue
        _require(
            len(op.result_shapes) == len(op.operand_names) == 1
            and op.index in live_ids
            and op.operand_shapes == op.result_shapes
            and index.operand(op, 0).result_shapes == op.result_shapes,
            "WK annotation shape/operand/liveness differs",
        )
        flags = re.findall(
            r"\bcustom_call_has_side_effect=([\w]+)",
            _callee_attribute_text(op.raw_line),
        )
        _require(flags in ([], ["false"]), "WK annotation side effect forbidden")
        s = op.result_shapes[0]
        target = _target(op)
        suffix = "/gather" if target == "AssumeGatherIndicesInBound" else "/concatenate"
        key = (target, s.dtype, s.dimensions, suffix)
        _require(
            key in expected and bool(op.op_name and op.op_name.endswith(suffix)),
            "WK annotation signature/scope differs",
        )
        seen[key] += 1
    _require(seen == expected, "WK annotation count differs")
    return dict(annotation_count=sum(seen.values()))


def inspect_structure(
    name: str,
    optimized: str,
    *,
    helper_check: Callable | None = None,
    suffix_check: Callable | None = None,
) -> dict:
    module = parse_hlo_module(optimized)
    index = PrefillHloIndex(module)
    live = _live_instruction_closure(module.instructions)
    _require(module.num_partitions == 32, "dense requires32 physical partitions")
    _require(
        not any(
            op.opcode in ("infeed", "outfeed", "send", "recv")
            or re.search(
                r'custom_call_target="[^"]*(?:host|io|python)[^"]*callback', op.raw_line
            )
            for op in module.instructions
        ),
        "dense host transport forbidden",
    )
    _require(
        not any(
            s.dtype in ("bf16", "f32") and s.element_count >= 32 * 2048 * 1536
            for op in module.instructions
            for s in op.result_shapes
        ),
        "dense full floating-weight expansion forbidden",
    )
    if name == "dense01":
        bodies = prefix_bodies(index, live)
        suffix = None if suffix_check is None else suffix_check(index, live)
        placed = {} if suffix is None else dict(suffix_bodies=suffix["bodies"])
        result = dict(
            prefix_bodies=bodies,
            collectives=check_collectives(index, live, bodies, **placed),
            kernels=check_kernels(index, live, bodies, **placed),
            helpers=(check_helpers if helper_check is None else helper_check)(
                index, live
            ),
        )
        if suffix is not None:
            result["canonical_suffix"] = suffix
        return result
    _require(name in PROGRAMS[:2], "unregistered dense graph")
    records, votes = _physical_records(index, live)
    _require(not votes, "WK scalar votes forbidden")
    actual = Counter(
        (
            op.opcode,
            op.replica_groups,
            tuple((s.dtype, s.dimensions) for s in op.operand_shapes),
            tuple((s.dtype, s.dimensions) for s in op.result_shapes),
        )
        for op, _ in records
    )
    _require(
        actual == wk_collectives(name), "dense WK physical collective schedule differs"
    )
    helpers = check_wk_helpers(index, live, name)
    return dict(
        collective_count=len(records),
        physical_groups=dict(feature=FEATURE, expert=EXPERT),
        helpers=helpers,
    )


def inspect_program(
    name: str, stable: str, optimized: str, memory: Mapping[str, Any]
) -> dict:
    _require(
        name in RAW
        and (len(stable.encode()), sha256(stable.encode()).hexdigest()) == RAW[name],
        "dense preregistered raw graph differs",
    )
    validate_memory(name, memory)
    structure = inspect_structure(name, optimized)
    return json.loads(
        json.dumps(
            dict(
                profile=PROFILE,
                graph=name,
                passed=True,
                stablehlo_sha256=RAW[name][1],
                optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
                compiled_memory=dict(memory),
                structure=structure,
                scope="BOUNDED_DENSE_DIAGNOSTIC_REQUIRES_LIVE_MEMORY_AND_DB604_REPRODUCTION",
                numerical_promotion=False,
                performance_claim=False,
            ),
            allow_nan=False,
        )
    )
