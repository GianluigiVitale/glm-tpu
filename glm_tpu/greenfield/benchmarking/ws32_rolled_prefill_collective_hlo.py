"""Exact heterogeneous rolled-prefix/wide-suffix physical communication.

Expected schedule derives from the pinned prefix/MLP source, including actual
ordered tuple fusion. Counts are static unless explicitly identified as body
schedule expansion; conditional head operations are never declared unconditional.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Mapping, Sequence

from ...optimized.hlo_contract import HloInstruction
from .ws32_batched_collective_hlo import _physical_records, _records
from .ws32_batched_commit_hlo import _require
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _computation_base
from .ws32_rolled_prefill_hlo import RolledTransitions, _rows
from .ws32_prefill_fixed_loops import fixed_loop_bodies


def _expected(rows: int, *, canonical_dense: bool = False, nucleus_head: bool = False,
              nucleus_gather_lowered: bool = False) -> Counter:
    """B32 causal prefix, B128/B114 MLP, source-derived local groups and leaves."""
    _rows(rows)
    _require(type(nucleus_gather_lowered) is bool and
             (not nucleus_gather_lowered or nucleus_head),
             "lowered vocabulary exchange requires explicit nucleus profile")
    result: Counter = Counter()

    def add(place, layer, family, ins, outs=None, *, count=1, kind="add", axis=-1):
        opcode = "all-gather" if kind == "gather" else "all-reduce"
        result[(place, layer, opcode, family, kind, axis, ins, outs or ins)] += count

    def summed(place, layer, family, dims, *, count=1, dtype="f32", output="bf16"):
        add(place, layer, family, ((dtype, dims),), ((output, dims),), count=count)

    for layer in range(78):
        full = layer in (0, 1, 2) or (layer >= 6 and layer % 4 == 2)
        # Both norms surround the prefix attention before the wide MLP.
        summed("prefix", layer, "feature", (32, 1), count=2, output="f32")
        dims = (
            ((32, 576), (32, 128), (32, 2048), (32, 4))
            if full
            else ((32, 576), (32, 2048))
        )
        dtypes = ("bf16", "f32", "bf16", "f32") if full else ("bf16", "bf16")
        add(
            "prefix",
            layer,
            "feature",
            tuple(("f32", d) for d in dims),
            tuple(zip(dtypes, dims)),
        )
        summed("prefix", layer, "expert", (32, 2048, 640), dtype="bf16")
        summed("prefix", layer, "expert", (32, 1536))
        if full:
            add(
                "prefix",
                layer,
                "feature",
                (("bf16", (32, 1536)),),
                (("bf16", (32, 6144)),),
                kind="gather",
                axis=1,
            )
            for dtype, width in (("f32", 516), ("f32", 2048), ("s32", 2048)):
                add(
                    "prefix",
                    layer,
                    "expert",
                    ((dtype, (32, width)),),
                    ((dtype, (256, width)),),
                    kind="gather",
                    axis=0,
                )
        if layer < 3:
            summed(
                "canonical_dense" if canonical_dense else "suffix",
                layer,
                "feature",
                (2, 128 if canonical_dense else rows, 1536),
            )
        else:
            summed("suffix", layer, "feature", (rows, 32), output="f32")
            add(
                "suffix",
                layer,
                "expert",
                (("f32", (rows, 32)),),
                (("f32", (rows, 256)),),
                kind="gather",
                axis=1,
            )
            dims = ((2, rows, 2048), (2, 8 * rows, 2048))
            add(
                "suffix",
                layer,
                "feature",
                tuple(("f32", d) for d in dims),
                tuple(("bf16", d) for d in dims),
            )
        # XLA fuses the NEXT layer's router-bias reconstruction onto the
        # preceding suffix output (layers2..76), not onto its named router.
        if canonical_dense and layer < 3:
            summed("canonical_dense", layer, "expert", (128, 1536))
        elif 2 <= layer < 77:
            add(
                "suffix",
                layer,
                "expert",
                (("f32", (rows, 1536)), ("f32", (256,))),
                (("bf16", (rows, 1536)), ("f32", (256,))),
            )
        else:
            summed("suffix", layer, "expert", (rows, 1536))
    if canonical_dense:
        # This exact standalone interface is separately source/consumer-bound
        # by the canonical router-bias proof, not admitted by its shape alone.
        summed("outer", -1, "expert", (256,), output="f32")
    summed("outer", -1, "expert", (rows, 1536), dtype="bf16")
    summed("outer", -1, "feature", (1, 1), output="f32")
    summed("outer", -1, "feature", (1, 19360))
    if nucleus_head:
        if nucleus_gather_lowered:
            summed("outer", -1, "expert", (154880,), dtype="bf16")
        else:
            add("outer", -1, "expert", (("bf16", (1, 19360)),),
                (("bf16", (1, 154880)),), kind="gather", axis=1)
    else:
        for dtype in ("bf16", "s32"):
            summed("outer", -1, "expert", (8,), dtype=dtype, output=dtype)
    for family in ("feature", "expert"):
        add("outer", -1, family, (("s32", ()),), kind="minimum")
    return result


def _nucleus_exchange(index: PrefillHloIndex, records: list) -> dict[str, Any]:
    """Recognize the observed compiler interface, not an operand-value proof.

    Native caller binds frozen source and complete RAW before reaching here.
    TPU lowers its final all_gather to zero-pad/DUS/expert8 ADD. The optimized
    text hides the partition lookup table as {...}; do NOT claim to replay its
    values or opaque producer math. Those retain the existing compiler/source
    trust boundary. No historical profile or layer collective is broadened.
    """
    shape = (("bf16", (154880,)),)
    key = (-1, "all-reduce", "expert", "add", -1, shape, shape)
    candidates = [op for op, actual in records if actual == key]
    if not candidates:
        return dict(form="all_gather", lowered=False)
    _require(len(candidates) == 1, "ambiguous final vocabulary exchange")
    op = candidates[0]

    def shaped(value: HloInstruction, dtype: str, dims: tuple) -> bool:
        return (len(value.result_shapes) == 1 and
                value.result_shapes[0].dtype == dtype and
                value.result_shapes[0].dimensions == dims)

    update = index.operand(op, 0)
    _require(update.raw_opcode == "dynamic-update-slice" and
             len(update.operand_names) == 3 and shaped(update, "bf16", (154880,)),
             "vocabulary sum must consume one full zero-padded DUS")
    base, fragment, offset = (index.operand(update, n) for n in range(3))
    _require(base.raw_opcode == "broadcast" and len(base.operand_names) == 1 and
             shaped(base, "bf16", (154880,)), "vocabulary DUS base differs")
    zero = index.operand(base, 0)
    _require(zero.raw_opcode == "constant" and shaped(zero, "bf16", ()) and
             re.search(r"\bconstant\(0\)", zero.raw_line) is not None,
             "vocabulary padding is not positive zero")
    _require(shaped(fragment, "bf16", (19360,)), "vocabulary shard width/dtype differs")
    _require(offset.raw_opcode == "bitcast" and len(offset.operand_names) == 1 and
             shaped(offset, "u32", ()), "vocabulary offset scalar differs")
    product = index.operand(offset, 0)
    _require(product.raw_opcode == "multiply" and len(product.operand_names) == 2 and
             shaped(product, "u32", (1,)), "vocabulary offset product differs")
    lookup, stride = (index.operand(product, n) for n in range(2))
    _require(stride.raw_opcode == "constant" and shaped(stride, "u32", (1,)) and
             re.search(r"\bconstant\(\{19360\}\)", stride.raw_line) is not None,
             "vocabulary offset stride differs")
    _require(lookup.raw_opcode == "dynamic-slice" and len(lookup.operand_names) == 2 and
             shaped(lookup, "u32", (1,)), "vocabulary rank lookup differs")
    table = index.operand(lookup, 0)
    _require(table.raw_opcode == "constant" and shaped(table, "u32", (32,)),
             "vocabulary rank lookup table shape differs")
    return dict(form="zero_padded_expert8_all_reduce", lowered=True,
                computation=op.computation, instruction=op.name,
                scope="SOURCE_RAW_BOUND_PHYSICAL_VOCAB_EXCHANGE_INTERFACE",
                not_proven=["OPAQUE_PARTITION_TABLE_VALUES", "FRAGMENT_VALUES",
                            "NUMERICAL_OR_MEMORY_ADMISSION"])


def _placed_records(counter: Counter) -> list[dict[str, Any]]:
    return [
        dict(placement=key[0], **_records(Counter({key[1:]: count}))[0])
        for key, count in sorted(counter.items())
    ]


def check_rolled_collectives(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
    canonical_dense: bool = False,
) -> dict[str, Any]:
    """Reuse old physical/reducer guards, with actual loop-body placement."""
    return _check_rolled_collectives(
        index, block_rows=block_rows, live_instructions=live_instructions,
        canonical_dense=canonical_dense,
    )


def _check_rolled_collectives(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
    canonical_dense: bool = False,
    prefix_bodies: Mapping[str, int] | None = None,
    nucleus_head: bool = False,
) -> dict[str, Any]:
    """Shared physical schedule; long callers separately prove fixed loops."""
    _rows(block_rows)
    _require(type(canonical_dense) is bool, "canonical dense option must be bool")
    _require(type(nucleus_head) is bool, "nucleus head option must be bool")
    report: dict[str, Any] = dict(
        passed=False,
        scope="ROLLED_PHYSICAL_COLLECTIVE_INVENTORY_AND_PLACEMENT",
        not_proven=[
            "OPERAND_OWNERSHIP",
            "NUMERICAL_OR_MEMORY_ADMISSION",
            "MEASURED_DYNAMIC_EXECUTION_COUNTS",
        ],
    )
    try:
        if prefix_bodies is None:
            transitions = RolledTransitions(index, block_rows)
            loops = transitions.all_loops(live_instructions)
            bodies = {index.callee(loop.loop, "body"): loop.layer for loop in loops}
        else:
            bodies = prefix_bodies
        _require(len(bodies) == 78, "rolled collective body ownership is ambiguous")
        dense_bodies = (
            fixed_loop_bodies(
                index,
                live_instructions,
                loop_suffix="greenfield_ws32_prefill_dense_canonical/while",
                expected_layers=(0, 1, 2),
            )
            if canonical_dense
            else {}
        )
        records, votes = _physical_records(index, live_instructions)
        observed: Counter = Counter()
        for op, key in records:
            layer = key[0]
            computation = _computation_base(op.computation)
            if computation in bodies:
                _require(
                    layer == bodies[computation],
                    "collective layer differs from actual prefix body",
                )
                place = "prefix"
            elif computation in dense_bodies:
                _require(
                    layer == dense_bodies[computation],
                    "collective layer differs from own canonical body",
                )
                place = "canonical_dense"
            elif layer == -1:
                place = "outer"
            else:
                _require(computation == "ENTRY", "wide suffix collective is not ENTRY")
                place = "suffix"
            observed[(place, *key)] += 1
        exchange = _nucleus_exchange(index, records) if nucleus_head else None
        expected = _expected(block_rows, canonical_dense=canonical_dense, nucleus_head=nucleus_head,
                             nucleus_gather_lowered=bool(exchange and exchange["lowered"]))
        placements = Counter()
        for key, count in observed.items():
            placements[key[0]] += count
        report.update(
            passed=observed == expected,
            static_collective_count=sum(observed.values()),
            static_placement_counts=dict(placements),
            reduction_operand_leaves=sum(
                len(k[6]) * n for k, n in observed.items() if k[2] == "all-reduce"
            ),
            gather_output_leaves=sum(
                len(k[7]) * n for k, n in observed.items() if k[2] == "all-gather"
            ),
            four_iteration_prefix_schedule_count=4 * placements["prefix"],
            dynamic_count_caveat="Body schedule expansion, not measured branch execution; outer head remains conditional",
            health_reductions=[
                {**v, "computation": _computation_base(v["computation"])} for v in votes
            ],
            missing=_placed_records(expected - observed),
            unexpected=_placed_records(observed - expected),
        )
        if canonical_dense:
            report["four_iteration_dense_schedule_count"] = (
                4 * placements["canonical_dense"]
            )
        if exchange is not None:
            report["nucleus_exchange"] = exchange
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
