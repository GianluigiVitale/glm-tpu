"""Fail-closed source-derived inventory for the distinct full-layer candidate.

Tuple-merged reductions are compared by payload, never by a guessed count of
source psums. Actual optimized HLO is checked before executing the candidate.
This is structural admission, not a trace, numerical or performance proof.
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from scripts.greenfield.prefill_moe_precision_hlo import check_fp32_route_sum

FEATURE = tuple(tuple(range(e * 4, e * 4 + 4)) for e in range(8))
EXPERT = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))


def _compiler_helpers(calls, *, layer: int) -> dict[str, Any]:
    """Bind TPU layout/index annotations separately from model Pallas calls.

    Layer0 signatures acquired at 4a15234c, candidate HLO dc1d5a94...a13d9f3.
    Same compiler mechanisms as one_layer.py and runtime/decoder.py guards.
    Layer3 remains unregistered until its own pre-execution HLO is inspected.
    """
    observed = Counter()
    operands_valid = True
    for op in calls:
        match = re.search(r'custom_call_target="([^"]+)"', op.raw_line)
        target = match.group(1) if match else "<missing>"
        if (
            len(op.result_shapes) != 1
            or "custom_call_has_side_effect=true" in op.raw_line
        ):
            operands_valid = False
            continue
        shape = op.result_shapes[0]
        observed[(target, shape.dtype, shape.dimensions)] += 1
        if target in ("AssumeGatherIndicesInBound", "GatherScatterIndicesBitpacked"):
            operands_valid &= (
                len(op.operand_names) == 1
                and op.operand_shapes == op.result_shapes
                and shape.dtype == "s32"
                and bool(op.op_name and op.op_name.endswith("/gather"))
            )
        elif target == "ConcatBitcast":
            operands_valid &= (
                shape.dtype == "u8"
                and len(shape.dimensions) == 2
                and len(op.operand_names) == len(op.operand_shapes) == 4
                and all(
                    s.dtype == "u8"
                    and s.dimensions == (shape.dimensions[0] // 4, shape.dimensions[1])
                    for s in op.operand_shapes
                )
            )
        else:
            operands_valid = False
    expected = (
        Counter(
            {
                ("AssumeGatherIndicesInBound", "s32", (1024,)): 7,
                ("AssumeGatherIndicesInBound", "s32", (34816,)): 2,
                ("GatherScatterIndicesBitpacked", "s32", (17, 4096, 2)): 2,
                ("GatherScatterIndicesBitpacked", "s32", (17, 16384, 2)): 2,
                ("GatherScatterIndicesBitpacked", "s32", (17, 2048, 2)): 3,
                ("ConcatBitcast", "u8", (2048, 2048)): 1,
                ("ConcatBitcast", "u8", (3584, 512)): 1,
                ("ConcatBitcast", "u8", (1536, 2048)): 1,
                ("ConcatBitcast", "u8", (1536, 1536)): 3,
            }
        )
        if layer == 0
        else None
    )
    return dict(
        passed=expected is not None and observed == expected and operands_valid,
        registered=expected is not None,
        operands_valid=operands_valid,
        signatures=[
            dict(target=k[0], dtype=k[1], dimensions=k[2], count=v)
            for k, v in sorted(observed.items())
        ],
    )


def check_layer_hlo(hlo: str, *, layer: int) -> dict[str, Any]:
    if layer not in (0, 3):
        raise ValueError("unregistered layer HLO")
    module = parse_hlo_module(hlo)
    collectives = [op for op in module.instructions if op.is_collective]
    payloads = Counter()
    valid_groups = True
    for op in collectives:
        if op.replica_groups not in (FEATURE, EXPERT) or op.opcode not in (
            "all-reduce",
            "all-gather",
        ):
            valid_groups = False
        # TPU can fold the post-reduction BF16 cast into an all-reduce with
        # FP32 operands/reducer. Bind actual inputs for sums, outputs for gathers.
        shapes = op.operand_shapes if op.opcode == "all-reduce" else op.result_shapes
        for shape in shapes:
            payloads[
                (op.opcode, op.maximum_group_size, shape.dtype, shape.element_count)
            ] += 1
    expected = Counter()

    def add(op, group, dtype, elements, count=1):
        expected[(op, group, dtype, elements)] += count

    # Two fused-add RMSNorms, q-a and kv-a feature partials.
    add("all-reduce", 4, "f32", 17, 2)
    add("all-reduce", 4, "f32", 17 * 2048)
    add("all-reduce", 4, "f32", 17 * 576)
    add("all-reduce", 8, "bf16", 17 * 2048 * 640)
    add("all-reduce", 8, "f32", 17 * 1536, 2)
    if layer == 0:
        add("all-reduce", 4, "f32", 17 * 4)
        add("all-reduce", 4, "f32", 17 * 128)
        add("all-reduce", 4, "f32", 2 * 17 * 1536)
        add("all-gather", 8, "f32", 8 * 17 * (4 * 128 + 4))
        add("all-gather", 8, "s32", 8 * 17 * 2048)
        add("all-gather", 8, "f32", 8 * 17 * 2048)
        add("all-gather", 4, "bf16", 17 * 6144)
    else:
        add("all-reduce", 4, "f32", 17 * 32)
        add("all-reduce", 4, "f32", 2 * 136 * 2048)
        add("all-reduce", 4, "f32", 2 * 17 * 2048)
        add("all-gather", 8, "f32", 17 * 256)
        add("all-gather", 8, "f32", 256)
    calls = [op for op in module.instructions if op.opcode == "custom-call"]
    pallas = [
        op for op in calls if 'custom_call_target="tpu_custom_call"' in op.raw_line
    ]
    helpers = _compiler_helpers([op for op in calls if op not in pallas], layer=layer)
    raw = [op for op in pallas if "greenfield_fp8_block_matmul" in op.raw_line]
    grouped = [
        op for op in pallas if "greenfield_prefill_grouped_raw_fp8" in op.raw_line
    ]
    structured = [
        op for op in pallas if "greenfield_fp8_structured_kv_b" in op.raw_line
    ]
    sparse = [
        op
        for op in pallas
        if "greenfield_pregathered_sparse_mla_" in op.raw_line
        and "prefill_m17" in op.raw_line
    ]
    # BF16/F32 full expert expansion is >100M elements. The selected-cache
    # tile is <23M. Inspect all intermediate result shapes, not metadata labels.
    overlays = [
        (op.name, shape.to_dict())
        for op in module.instructions
        for shape in op.result_shapes
        if shape.dtype in ("bf16", "f32") and shape.element_count >= 32 * 2048 * 1536
    ]
    precision = (
        check_fp32_route_sum(
            module, expert_scope="greenfield_ws32_prefill_moe/expert_reduce"
        )
        if layer == 3
        else None
    )
    checks = dict(
        physical_groups=valid_groups,
        exact_collective_payload_inventory=payloads == expected,
        only_declared_pallas_calls=len(pallas) == (12 if layer == 0 else 13),
        exact_compiler_helpers=helpers["passed"],
        raw_calls=len(raw) == (9 if layer == 0 else 7),
        grouped_calls=len(grouped) == (0 if layer == 0 else 3),
        structured_calls=len(structured) == 2,
        sparse_calls=len(sparse) == 1,
        raw_u8_operands=all(
            any(s.dtype == "u8" for s in op.operand_shapes)
            for op in raw + grouped + structured
        ),
        no_full_weight_expansion=not overlays,
        fp32_route_sum=precision is None or precision["passed"],
        no_host_transport=not any(
            op.opcode in ("infeed", "outfeed", "send", "recv")
            for op in module.instructions
        ),
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        collective_payloads=[
            dict(opcode=k[0], group_size=k[1], dtype=k[2], elements=k[3], count=v)
            for k, v in sorted(payloads.items())
        ],
        expected_payloads=[
            dict(opcode=k[0], group_size=k[1], dtype=k[2], elements=k[3], count=v)
            for k, v in sorted(expected.items())
        ],
        collectives=[op.to_dict() for op in collectives],
        custom_calls=[op.to_dict() for op in calls],
        compiler_helpers=helpers,
        full_weight_overlays=overlays,
        fp32_route_sum_proof=precision,
        performance_claim=False,
    )
