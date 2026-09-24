"""Exact acquired short-prefill communication inventory, not execution admission.

Counts physical operations AND their ordered input/output leaves per layer.
This adapts the one-layer payload guard to the full B17/B11 graphs. Metadata
scope assigns inventory buckets, never proves data ownership or consumer use.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import re
from typing import Any, Sequence

from ...optimized.hlo_contract import HloInstruction
from .ws32_batched_commit_hlo import _minimum
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_decoder import _exact_add_reducer, _group_family
from .ws32_pallas_one_layer import _callee_attribute_text


_LAYER = re.compile(r"(?:^|/)greenfield_ws32_batched_prefill/layer_(\d+)(?:/|$)")
_FULL = frozenset((0, 1, 2, *range(6, 78, 4)))


def _expected(rows: int) -> Counter:
    """Source schedule plus observed tuple fusion, no gather/reduce alternatives."""
    expected: Counter = Counter()

    def add(layer, family, inputs, outputs=None, *, count=1, kind="add", axis=-1):
        opcode = "all-gather" if kind == "gather" else "all-reduce"
        expected[
            (layer, opcode, family, kind, axis, inputs, outputs or inputs)
        ] += count

    def shape(dtype, *dims):
        return (dtype, dims)

    def summed(layer, family, *dims, dtype="f32", output="bf16", count=1):
        add(layer, family, (shape(dtype, *dims),), (shape(output, *dims),), count=count)

    for layer in range(78):
        summed(layer, "feature", rows, 1, output="f32", count=2)  # two RMSNorms
        # q/kv-a feature sums, merged with producer key/head projections.
        dims = (
            ((rows, 576), (rows, 128), (rows, 2048), (rows, 4))
            if layer in _FULL
            else ((rows, 576), (rows, 2048))
        )
        out = ("bf16", "f32", "bf16", "f32") if layer in _FULL else ("bf16", "bf16")
        if layer == 74:
            # Both captured graphs order the last producer's tuple differently.
            # Preserve exact paired leaves, not a shape/order-wide allowance.
            dims = ((rows, 128), (rows, 4), (rows, 2048), (rows, 576))
            out = ("f32", "f32", "bf16", "bf16")
        add(layer, "feature", tuple(("f32", d) for d in dims), tuple(zip(out, dims)))
        summed(layer, "expert", rows, 2048, 640, dtype="bf16")  # selected KV
        if layer < 3:
            summed(layer, "expert", rows, 1536)  # attention output
            summed(layer, "feature", 2, rows, 1536)  # dense gate/up
            summed(layer, "expert", rows, 1536)  # dense output
        else:
            # Compiler merges attention output with zero-insert router-bias ADD.
            add(
                layer,
                "expert",
                (("f32", (rows, 1536)), ("f32", (256,))),
                (("bf16", (rows, 1536)), ("f32", (256,))),
            )
            summed(layer, "feature", rows, 32, output="f32")
            add(
                layer,
                "expert",
                (("f32", (1, 32, rows)),),
                (("f32", (8, 32, rows)),),
                kind="gather",
                axis=0,
            )
            dims = ((2, rows, 2048), (2, 8 * rows, 2048))
            add(
                layer,
                "feature",
                tuple(("f32", d) for d in dims),
                tuple(("bf16", d) for d in dims),
            )
            summed(layer, "expert", rows, 1536)  # routed output; shared adds locally
        if layer in _FULL:
            add(
                layer,
                "feature",
                (("bf16", (rows, 1536)),),
                (("bf16", (rows, 6144)),),
                kind="gather",
                axis=1,
            )
            for dtype, width in (("f32", 516), ("f32", 2048), ("s32", 2048)):
                add(
                    layer,
                    "expert",
                    ((dtype, (1, rows, width)),),
                    ((dtype, (8, rows, width)),),
                    kind="gather",
                    axis=0,
                )

    # Outside the repeated layer scope: one embedding, final head and two votes.
    summed(-1, "expert", rows, 1536, dtype="bf16")
    summed(-1, "feature", 1, 1, output="f32")
    summed(-1, "feature", 1, 19360)
    for dtype in ("bf16", "s32"):
        summed(-1, "expert", 8, dtype=dtype, output=dtype)
    for family in ("feature", "expert"):
        add(-1, family, (("s32", ()),), kind="minimum")
    return expected


def _physical_records(
    index: PrefillHloIndex, live_instructions: Sequence[HloInstruction]
) -> tuple[list[tuple[HloInstruction, tuple]], list[dict[str, Any]]]:
    """Shared exact physical validation; no expected schedule is inferred here."""
    if index.module.num_partitions != 32:
        raise ValueError("expected32 physical partitions")
    live = {(op.computation, op.name) for op in live_instructions}
    computations = {
        name.lstrip("%"): tuple(nodes.values())
        for name, nodes in index.computations.items()
    }
    records, votes = [], []
    for op in index.module.collectives:
        family = _group_family(op)
        if not (
            family in ("feature", "expert")
            and len(op.replica_groups) == (8 if family == "feature" else 4)
            and op.use_global_device_ids
            and op.raw_opcode in ("all-reduce", "all-gather")
            and (op.computation, op.name) in live
            and len(op.operand_names) == len(op.operand_shapes) == len(op.result_shapes)
            and all(
                index.operand(op, i).result_shapes == (s,)
                for i, s in enumerate(op.operand_shapes)
            )
        ):
            raise ValueError(f"{op.name}: groups/opcode/liveness/operand arity drift")
        scopes = _LAYER.findall(op.op_name or "")
        if len(scopes) > 1:
            raise ValueError("ambiguous layer scope")
        layer = int(scopes[0]) if scopes else -1
        attrs = _callee_attribute_text(op.raw_line)
        axis = -1
        if op.raw_opcode == "all-gather":
            axes = re.findall(r"\bdimensions=\{(\d+)\}", attrs)
            if len(axes) != 1:
                raise ValueError(f"{op.name}: ambiguous/missing gather axis")
            axis, kind = int(axes[0]), "gather"
        else:
            index.callee(op, "to_apply")
            if _exact_add_reducer(
                replace(op, raw_line=attrs),
                module_instructions=index.module.instructions,
                instructions_by_computation=computations,
            ):
                kind = "add"
            else:
                _minimum(index, op, family)
                kind = "minimum"
                votes.append(
                    dict(computation=op.computation, name=op.name, family=family)
                )
        ins = tuple((s.dtype, s.dimensions) for s in op.operand_shapes)
        outs = tuple((s.dtype, s.dimensions) for s in op.result_shapes)
        records.append((op, (layer, op.raw_opcode, family, kind, axis, ins, outs)))
    return records, votes


def _records(counter: Counter) -> list[dict[str, Any]]:
    return [
        dict(
            layer=k[0],
            opcode=k[1],
            family=k[2],
            reducer=k[3],
            gather_axis=k[4],
            inputs=k[5],
            outputs=k[6],
            count=v,
        )
        for k, v in sorted(counter.items())
    ]


def check_batched_collectives(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    """Prove exact B17/B11 physical inventory; other compiled shapes unregistered.

    Shared scalar ADD reducers may have mixed F32/BF16 output leaves after XLA
    post-sum casts. The only MINs are one scalar per subgroup; the separate
    atomic-commit proof must bind those two actual operations to state outputs.
    """
    if type(block_rows) is not int or block_rows not in (11, 17):
        raise ValueError("collective profile is registered only for B17/B11")
    report: dict[str, Any] = dict(
        passed=False,
        scope="SHORT_PREFILL_PHYSICAL_COLLECTIVE_INVENTORY_ONLY",
        not_proven=[
            "OPERAND_OWNERSHIP_AND_CONSUMER_LINEAGE",
            "ALL_LAYER_HEALTH",
            "COMPILER_HELPERS",
            "NUMERICAL_OR_PERFORMANCE_ADMISSION",
        ],
    )
    observed: Counter = Counter()
    expected = _expected(block_rows)
    try:
        records, votes = _physical_records(index, live_instructions)
        observed.update(key for _, key in records)
        report.update(
            collective_count=sum(observed.values()),
            reduction_operand_leaves=sum(
                len(k[5]) * n for k, n in observed.items() if k[1] == "all-reduce"
            ),
            gather_output_leaves=sum(
                len(k[6]) * n for k, n in observed.items() if k[1] == "all-gather"
            ),
            health_reductions=votes,
            missing=_records(expected - observed),
            unexpected=_records(observed - expected),
            passed=observed == expected,
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
