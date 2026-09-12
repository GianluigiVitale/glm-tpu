"""Exact acquired Pallas call interface/schedule, not kernel arithmetic proof."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Callable, Sequence

from ..sharding.hlo_contract import HloInstruction, HloShape
from .ws32_batched_helper_hlo import _target
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_pallas_one_layer import _callee_attribute_text


_LAYER = re.compile(r"(?:^|/)greenfield_ws32_batched_prefill/layer_(\d+)(?:/|$)")


def _expected(rows: int) -> Counter:
    """Raw stored U8/F32 interfaces of the already-admitted layer primitives."""
    padded = ((rows + 7) // 8) * 8
    routes, metadata = rows * 8, rows + 255
    result: Counter = Counter()

    def add(layer, family, name, inputs, output, alias=-1, count=1):
        result[(layer, family, name, inputs, (output,), alias)] += count

    def raw(layer, k, n, dtype="f32", count=1):
        name = f"greenfield_fp8_block_matmul_{'f32_' if dtype == 'f32' else ''}m{padded}_k{k}_n{n}"
        add(
            layer,
            "raw",
            name,
            (("bf16", (padded, k)), ("u8", (n, k)), ("f32", (16, 128))),
            (dtype, (padded, n)),
            count=count,
        )

    for layer in range(78):
        raw(layer, 1536, 2048)  # q-a
        raw(layer, 1536, 640)  # kv-a, including output padding
        raw(layer, 2048, 2048, "bf16")  # q-b
        raw(layer, 2048, 1536)  # attention output
        if layer in (0, 1, 2) or (layer >= 6 and (layer - 6) % 4 == 0):
            raw(layer, 1536, 128)  # index key
            raw(layer, 2048, 512)  # index query
        if layer < 3:
            raw(layer, 1536, 1536, count=3)  # dense gate/up/down
        else:
            raw(layer, 1536, 2048, count=2)  # shared gate/up
            raw(layer, 2048, 1536, "bf16")  # shared down
            for k, n, dtype, count in ((1536, 2048, "f32", 2), (2048, 1536, "bf16", 1)):
                inputs = (
                    ("s32", ()),
                    ("s32", (257,)),
                    ("s32", (metadata,)),
                    ("s32", (metadata,)),
                    ("s32", (1,)),
                    ("bf16", (routes, k)),
                    ("u8", (32, n, k)),
                    ("f32", (32, 16, 128)),
                    (dtype, (routes, n)),
                )
                # XLA adds the dynamic grid scalar before the source's8 inputs.
                add(
                    layer,
                    "grouped",
                    "greenfield_prefill_grouped_raw_fp8",
                    inputs,
                    (dtype, (routes, n)),
                    alias=8,
                    count=count,
                )
        weight = (("u8", (3584, 512)),)
        add(
            layer,
            "structured",
            f"greenfield_fp8_structured_kv_b_q_absorb_h8_p192_l512_prefill_m{rows}",
            (("bf16", (16, padded, 128)),) + weight + (("f32", (64, 8, 128)),),
            ("bf16", (32, padded, 128)),
        )
        add(
            layer,
            "structured",
            f"greenfield_fp8_structured_kv_b_value_h8_l512_v256_prefill_m{rows}",
            (("bf16", (8, padded, 512)),) + weight + (("f32", (96, 8, 128)),),
            ("bf16", (24, padded, 128)),
        )
        add(
            layer,
            "sparse",
            f"greenfield_pregathered_sparse_mla_h8_k2048_b512_w640_prefill_m{rows}",
            (("s32", (rows,)), ("bf16", (rows, 8, 640)), ("bf16", (rows, 4, 512, 640))),
            ("bf16", (rows, 8, 512)),
        )
    return result


def _records(counter: Counter) -> list[dict[str, Any]]:
    return [
        dict(
            layer=k[0],
            family=k[1],
            kernel=k[2],
            inputs=k[3],
            outputs=k[4],
            alias_operand=k[5],
            count=v,
        )
        for k, v in sorted(counter.items())
    ]


def check_batched_kernels(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    """Bind exact live call shapes, local raw storage and output aliases.

    The source pin and full HLO digest bind implementation bytes separately.
    This inventory does not inspect opaque kernel bodies or establish which
    model leaf supplied a same-shaped weight. Real layer and §21 evidence remain
    indispensable; this report cannot authorize model execution on its own.
    """
    if type(block_rows) is not int or block_rows not in (11, 17):
        raise ValueError("kernel profile is registered only for B17/B11")
    return _check_kernel_schedule(
        index, live_instructions=live_instructions, expected=_expected(block_rows)
    )


def _check_kernel_schedule(
    index: PrefillHloIndex,
    *,
    live_instructions: Sequence[HloInstruction],
    expected: Counter,
    placement_check: Callable[[HloInstruction, tuple], None] | None = None,
    families: tuple[str, ...] = ("raw", "grouped", "structured", "sparse"),
    layer_resolver: Callable[[HloInstruction], int] | None = None,
    large_float_guard: Callable[[HloInstruction, HloShape], bool] | None = None,
) -> dict[str, Any]:
    """Shared interface/alias/size guards; caller supplies a fixed source schedule."""
    report: dict[str, Any] = dict(
        passed=False,
        scope="SHORT_PREFILL_PALLAS_INTERFACE_AND_SCHEDULE_ONLY",
        not_proven=[
            "OPAQUE_KERNEL_ARITHMETIC",
            "MODEL_LEAF_OWNERSHIP",
            "ROUTE_SCHEDULE_VALUE_CORRECTNESS",
            "NUMERICAL_OR_PERFORMANCE_ADMISSION",
        ],
    )
    observed: Counter = Counter()
    live = {(op.computation, op.name) for op in live_instructions}
    try:
        if index.module.num_partitions != 32:
            raise ValueError("expected32 physical partitions")
        for op in index.module.instructions:
            # Short caches are below this full-local-expert expansion size.
            # This is NOT a long-capacity shape policy or a peak HBM estimate.
            for shape in op.result_shapes:
                if shape.dtype in ("bf16", "f32") and shape.element_count >= 32 * 2048 * 1536:
                    if large_float_guard is None or not large_float_guard(op, shape):
                        raise ValueError(
                            f"{op.name}: short graph contains full-size floating weight expansion"
                        )
            if op.raw_opcode != "custom-call" or _target(op) != "tpu_custom_call":
                continue
            if not (
                (op.computation, op.name) in live
                and len(op.operand_names) == len(op.operand_shapes)
                and all(
                    index.operand(op, i).result_shapes == (s,)
                    for i, s in enumerate(op.operand_shapes)
                )
            ):
                raise ValueError(f"{op.name}: dead or mismatched kernel operands")
            attrs = _callee_attribute_text(op.raw_line)
            if re.findall(r"\bcustom_call_has_side_effect=(\w+)", attrs) not in (
                [],
                ["false"],
            ):
                raise ValueError("Pallas side effects unregistered")
            scope = op.op_name or ""
            layers = _LAYER.findall(scope)
            kernel = re.findall(r"/([^/]+)/pallas_call$", scope)
            if len(kernel) != 1 or (layer_resolver is None and len(layers) != 1):
                raise ValueError("kernel lacks exact layer/call scope")
            layer = int(layers[0]) if layer_resolver is None else layer_resolver(op)
            if type(layer) is not int:
                raise ValueError("kernel layer resolver returned non-integer")
            name = kernel[0]
            family = (
                "raw"
                if name.startswith("greenfield_fp8_block_matmul_")
                else (
                    "grouped"
                    if name == "greenfield_prefill_grouped_raw_fp8"
                    else (
                        "structured"
                        if name.startswith("greenfield_fp8_structured_kv_b_")
                        else (
                            "sparse"
                            if name.startswith("greenfield_pregathered_sparse_mla_")
                            else "unknown"
                        )
                    )
                )
            )
            if name == "greenfield_prefill_expert_panel_raw_fp8":
                family = "panels"
            alias = -1
            if "output_to_operand_aliasing" in attrs:
                matches = re.findall(
                    r"\boutput_to_operand_aliasing=\{\{\}:\s*\((\d+),\s*\{\}\)\}", attrs
                )
                if len(matches) != 1 or attrs.count("output_to_operand_aliasing") != 1:
                    raise ValueError("unregistered output alias syntax")
                alias = int(matches[0])
            ins = tuple((s.dtype, s.dimensions) for s in op.operand_shapes)
            outs = tuple((s.dtype, s.dimensions) for s in op.result_shapes)
            key = (layer, family, name, ins, outs, alias)
            if placement_check is not None:
                placement_check(op, key)
            observed[key] += 1
        report.update(
            passed=observed == expected,
            kernel_count=sum(observed.values()),
            family_counts=dict(
                Counter(
                    {
                        family: sum(n for k, n in observed.items() if k[1] == family)
                        for family in families
                    }
                )
            ),
            missing=_records(expected - observed),
            unexpected=_records(observed - expected),
        )
    except (ValueError, KeyError, IndexError) as error:
        report["error"] = str(error)
    return report
