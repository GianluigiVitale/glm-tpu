"""Actual rolled B32 prefix and wide expert-panel Pallas interface schedule."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Sequence

from ..sharding.hlo_contract import HloInstruction
from .ws32_batched_commit_hlo import _require
from .ws32_batched_kernel_hlo import _check_kernel_schedule, _LAYER
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _computation_base, _callee_attribute_text
from .ws32_rolled_prefill_hlo import RolledTransitions, _rows


def _expected(rows: int) -> tuple[Counter, dict[tuple, str]]:
    _rows(rows)
    wide = ((rows + 7) // 8) * 8
    # Source planner: ceil(route_rows/32) + local_groups - 1.
    panels = (8 * rows + 31) // 32 + 31
    result: Counter = Counter()
    placements: dict[tuple, str] = {}

    def add(place, layer, family, name, ins, output, alias=-1, count=1):
        key = (layer, family, name, ins, (output,), alias)
        result[key] += count
        _require(
            key not in placements or placements[key] == place,
            "ambiguous kernel placement",
        )
        placements[key] = place

    def raw(place, layer, k, n, dtype="f32", count=1):
        m = 32 if place == "prefix" else wide
        name = f"greenfield_fp8_block_matmul_{'f32_' if dtype == 'f32' else ''}m{m}_k{k}_n{n}"
        add(
            place,
            layer,
            "raw",
            name,
            (("bf16", (m, k)), ("u8", (n, k)), ("f32", (16, 128))),
            (dtype, (m, n)),
            count=count,
        )

    for layer in range(78):
        raw("prefix", layer, 1536, 2048)
        raw("prefix", layer, 1536, 640)
        raw("prefix", layer, 2048, 2048, "bf16")
        raw("prefix", layer, 2048, 1536)
        if layer in (0, 1, 2) or (layer >= 6 and layer % 4 == 2):
            raw("prefix", layer, 1536, 128)
            raw("prefix", layer, 2048, 512)
        if layer < 3:
            raw("suffix", layer, 1536, 1536, count=3)
        else:
            raw("suffix", layer, 1536, 2048, count=2)
            raw("suffix", layer, 2048, 1536, "bf16")
            for k, n, dtype, count in ((1536, 2048, "f32", 2), (2048, 1536, "bf16", 1)):
                output = (dtype, (panels, 32, n))
                add(
                    "suffix",
                    layer,
                    "panels",
                    "greenfield_prefill_expert_panel_raw_fp8",
                    (
                        ("s32", ()),
                        ("s32", (panels,)),
                        ("bf16", (panels, 32, k)),
                        ("u8", (32, n, k)),
                        ("f32", (32, 16, 128)),
                        output,
                    ),
                    output,
                    alias=5,
                    count=count,
                )
        weight = (("u8", (3584, 512)),)
        add(
            "prefix",
            layer,
            "structured",
            "greenfield_fp8_structured_kv_b_q_absorb_h8_p192_l512_prefill_m32",
            (("bf16", (16, 32, 128)),) + weight + (("f32", (64, 8, 128)),),
            ("bf16", (32, 32, 128)),
        )
        add(
            "prefix",
            layer,
            "structured",
            "greenfield_fp8_structured_kv_b_value_h8_l512_v256_prefill_m32",
            (("bf16", (8, 32, 512)),) + weight + (("f32", (96, 8, 128)),),
            ("bf16", (24, 32, 128)),
        )
        add(
            "prefix",
            layer,
            "sparse",
            "greenfield_pregathered_sparse_mla_h8_k2048_b512_w640_prefill_m32",
            (("s32", (32,)), ("bf16", (32, 8, 640)), ("bf16", (32, 4, 512, 640))),
            ("bf16", (32, 8, 512)),
        )
    return result, placements


def check_rolled_kernels(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    _rows(block_rows)
    try:
        t = RolledTransitions(index, block_rows)
        loops = t.all_loops(live_instructions)
        bodies = {index.callee(loop.loop, "body"): loop.layer for loop in loops}
        expected, placements = _expected(block_rows)
        counts: Counter = Counter()
        live = {op.index for op in live_instructions}
        callers: dict[str, list[tuple[HloInstruction, int]]] = defaultdict(list)
        for op in index.module.instructions:
            if op.opcode != "conditional":
                continue
            groups = re.findall(
                r"\bbranch_computations=\{([^}]*)\}",
                _callee_attribute_text(op.raw_line),
            )
            _require(len(groups) == 1, "ambiguous kernel conditional branches")
            branches = [name.strip() for name in groups[0].split(",")]
            _require(
                len(branches) == 2 and len(op.operand_names) == 3,
                "kernel conditional arity",
            )
            for branch, name in enumerate(branches):
                callers[name].append((op, branch))

        def placement(op: HloInstruction, key: tuple) -> None:
            place = placements.get(key)
            _require(place is not None, f"unregistered rolled kernel interface:{key}")
            comp = _computation_base(op.computation)
            if place == "prefix":
                _require(
                    bodies.get(comp) == key[0],
                    "kernel is not in own actual prefix body",
                )
            elif key[1] == "panels":
                owners = callers.get(comp, [])
                _require(len(owners) == 1, "panel computation lacks unique caller")
                owner, branch = owners[0]
                _require(
                    branch == 1
                    and owner.index in live
                    and _computation_base(owner.computation) == "ENTRY"
                    and _LAYER.findall(owner.op_name or "") == [str(key[0])],
                    "panel is not own wide-suffix accepted branch",
                )
            else:
                _require(comp == "ENTRY", "wide suffix kernel is not ENTRY")
            counts[place] += 1

        report = _check_kernel_schedule(
            index,
            live_instructions=live_instructions,
            expected=expected,
            placement_check=placement,
            families=("raw", "panels", "structured", "sparse"),
        )
        return {
            **report,
            "scope": "ROLLED_PALLAS_INTERFACE_AND_PLACEMENT_ONLY",
            "static_placement_counts": dict(counts),
            "four_iteration_prefix_schedule_count": 4 * counts["prefix"],
            "dynamic_count_caveat": "Schedule expansion only, not measured branch execution",
        }
    except (ValueError, KeyError, IndexError) as error:
        return dict(
            passed=False,
            scope="ROLLED_PALLAS_INTERFACE_AND_PLACEMENT_ONLY",
            error=str(error),
        )
