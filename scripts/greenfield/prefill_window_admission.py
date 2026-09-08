"""Fixed DB590 graph admission for one layer6 numerical discriminator.

Raw bytes, including metadata and opaque Pallas bodies, stay exact. This is not
a general symbolic model proof, independent canonical DSA proof, full-model
admission, timing permission or a launcher. Runtime memory and completed WK
boundaries must pass separately before any layer call.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.validation.ws32_prefill_memory import budget_resident_execution
from scripts.greenfield.prefill_layer_hlo import EXPERT, FEATURE
from scripts.greenfield.prefill_moe_precision_hlo import check_fp32_route_sum

PROFILE = "ws32-layer6-window-db590-fixed-graphs-v1"
RECEIPT = Path(__file__).resolve().parents[2] / (
    "docs/artifacts/prefill-window-layer6-four-graph-acquisition-20260908.json"
)
RECEIPT_SHA = "b5e336dadfd2a99c80c5b76cd77d2d11ce5a15d81bc053bc4a6377cfdf94607e"
PROGRAMS = ("wk_decode", "wk_promote", "candidate", "control")
REQUIRED_RESERVE_BYTES = 1 << 30


def registered_programs() -> dict[str, Any]:
    """The acquisition receipt supplies fixed facts, never an operator verdict."""
    raw = RECEIPT.read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA:
        raise ValueError("DB590 admission receipt content changed")
    return json.loads(raw)["programs"]


def _shape(shape: Any) -> tuple[str, tuple[int, ...]]:
    return shape.dtype, shape.dimensions


def expected_collectives(name: str) -> Counter:
    """Source schedule, including DB590's tuple-fused ordered payload leaves."""
    result = Counter()

    def add(op, groups, inputs, outputs=None, count=1):
        result[
            (op, groups, tuple(inputs), tuple(inputs if outputs is None else outputs))
        ] += count

    def f(*dims):
        return ("f32", dims)

    def b(*dims):
        return ("bf16", dims)

    if name == "wk_promote":
        return result
    if name == "wk_decode":
        add("all-reduce", FEATURE, [f(48)])
        add("all-gather", FEATURE, [("u8", (128, 1536))], [("u8", (128, 6144))])
        return result
    if name not in ("candidate", "control"):
        raise ValueError("unknown window graph")
    tiles, rows = (4, 128) if name == "candidate" else (1, 32)
    add("all-reduce", FEATURE, [f(32, 1)] * tiles, count=2)
    add("all-gather", FEATURE, [b(32, 1536)], [b(32, 6144)], tiles)
    widths = (
        [576, 128, 2048, 576, 576, 576, 128, 128, 128, 4, 2048, 4, 2048, 4, 2048, 4]
        if tiles == 4
        else [576, 128, 2048, 4]
    )
    add(
        "all-reduce",
        FEATURE,
        [f(32, n) for n in widths],
        [b(32, n) if n in (576, 2048) else f(32, n) for n in widths],
    )
    add("all-gather", EXPERT, [f(32, 516)], [f(256, 516)], tiles)
    for dtype in ("s32", "f32"):
        add("all-gather", EXPERT, [(dtype, (32, 2048))], [(dtype, (256, 2048))], tiles)
    add("all-reduce", EXPERT, [b(32, 2048, 640)], count=tiles)
    add(
        "all-reduce",
        EXPERT,
        [f(32, 1536)] * tiles + [f(256)],
        [b(32, 1536)] * tiles + [f(256)],
    )
    add("all-reduce", FEATURE, [f(rows, 32)])
    add("all-gather", EXPERT, [f(rows, 32)], [f(rows, 256)])
    add(
        "all-reduce",
        FEATURE,
        [f(2, rows, 2048), f(2, rows * 8, 2048)],
        [b(2, rows, 2048), b(2, rows * 8, 2048)],
    )
    add("all-reduce", EXPERT, [f(rows, 1536)], [b(rows, 1536)])
    return result


def inspect_program(
    name: str, stablehlo: str, optimized_hlo: str, compiled_memory: Mapping[str, Any]
) -> dict[str, Any]:
    """Refuse any raw-byte/role/allocation drift before inspecting the graph."""
    pins = registered_programs()
    if name not in pins:
        raise ValueError("unknown window graph role")
    pin = pins[name]
    stable_sha = sha256(stablehlo.encode()).hexdigest()
    hlo_sha = sha256(optimized_hlo.encode()).hexdigest()
    if (stable_sha, hlo_sha) != (pin["stablehlo_sha256"], pin["optimized_hlo_sha256"]):
        raise ValueError(f"{name}: raw acquired graph identity changed")
    if (
        set(compiled_memory) != set(pin["compiled_memory"])
        or any(type(v) is not int for v in compiled_memory.values())
        or dict(compiled_memory) != pin["compiled_memory"]
    ):
        raise ValueError(f"{name}: actual compiler allocation changed")
    module = parse_hlo_module(optimized_hlo)
    collectives = [op for op in module.instructions if op.is_collective]
    payloads = Counter(
        (
            op.opcode,
            op.replica_groups,
            tuple(map(_shape, op.operand_shapes)),
            tuple(map(_shape, op.result_shapes)),
        )
        for op in collectives
    )
    calls = [op for op in module.instructions if op.opcode == "custom-call"]
    targets = Counter()
    for op in calls:
        match = re.search(r'custom_call_target="([^"]+)"', op.raw_line)
        targets[match[1] if match else "<missing>"] += 1
    expected_targets = {
        "candidate": {
            "tpu_custom_call": 42,
            "AllocateBuffer": 6,
            "AssumeGatherIndicesInBound": 46,
            "GatherScatterIndicesBitpacked": 29,
            "ConcatBitcast": 18,
        },
        "control": {
            "tpu_custom_call": 15,
            "AllocateBuffer": 6,
            "AssumeGatherIndicesInBound": 16,
            "GatherScatterIndicesBitpacked": 8,
            "ConcatBitcast": 10,
        },
        "wk_decode": {
            "AssumeGatherIndicesInBound": 1,
            "GatherScatterIndicesBitpacked": 1,
        },
        "wk_promote": {},
    }[name]
    precision = None
    if name in ("candidate", "control"):
        precision = check_fp32_route_sum(
            module,
            rows=128 if name == "candidate" else 32,
            expert_scope="greenfield_ws32_prefill_moe/expert_reduce",
        )
    checks = dict(
        raw_graph_pair_exact=True,
        compiler_allocations_exact=True,
        paired_collective_payloads=payloads == expected_collectives(name),
        collective_count=len(collectives) == pin["collective_count"],
        custom_call_families=targets == Counter(expected_targets),
        custom_call_count=len(calls) == pin["custom_call_count"],
        fp32_route_sum=precision is None or precision["passed"],
        no_host_transport=not any(
            op.opcode in ("infeed", "outfeed", "send", "recv")
            for op in module.instructions
        ),
        no_full_weight_expansion=not any(
            s.dtype in ("bf16", "f32") and s.element_count >= 32 * 2048 * 1536
            for op in module.instructions
            for s in op.result_shapes
        ),
    )
    if not all(checks.values()):
        raise ValueError(f"{name}: window graph checks failed: {checks}")
    return dict(
        profile=PROFILE,
        graph=name,
        passed=True,
        checks=checks,
        stablehlo_sha256=stable_sha,
        optimized_hlo_sha256=hlo_sha,
        compiled_memory=dict(compiled_memory),
        fp32_route_sum=precision,
        collective_payloads=[
            dict(
                opcode=k[0],
                groups=[list(group) for group in k[1]],
                inputs=[[dtype, list(dims)] for dtype, dims in k[2]],
                outputs=[[dtype, list(dims)] for dtype, dims in k[3]],
                count=v,
            )
            for k, v in sorted(payloads.items())
        ],
        custom_call_targets=dict(sorted(targets.items())),
        numerical_execution_authorized=False,
        scope="FIXED_GRAPH_ONLY_REQUIRES_RUNTIME_MEMORY_WK_AND_FLEET_ADMISSION",
    )


def validate_program_report(
    report: Mapping[str, Any],
    name: str,
    stablehlo: str,
    optimized_hlo: str,
    compiled_memory: Mapping[str, Any],
) -> None:
    """Re-derive the complete published report, including nested types/values."""
    expected = inspect_program(name, stablehlo, optimized_hlo, compiled_memory)
    serialized = json.dumps(report, sort_keys=True, allow_nan=False)
    if report != json.loads(serialized) or serialized != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("window graph report differs from original evidence replay")


def memory_budget(
    census: Mapping[str, Any],
    compiled_memory: Mapping[str, Mapping[str, Any]],
    *,
    active_graph: str,
) -> dict[str, Any]:
    """Budget each completed-call boundary with all four programs resident.

    Caller captures named weights/WK/inputs/retained observations PLUS all live
    arrays before dispatch. Thus candidate results and earlier control caches
    still in use cannot disappear from later budgets. No alias subtraction.
    Fleet ownership and post-call peaks are separate compulsory worker checks.
    """
    pins = registered_programs()
    if set(compiled_memory) != set(PROGRAMS) or active_graph not in PROGRAMS:
        raise ValueError("window memory requires all four fixed graph roles")
    for name in PROGRAMS:
        actual = compiled_memory[name]
        if (
            any(type(v) is not int for v in actual.values())
            or dict(actual) != pins[name]["compiled_memory"]
        ):
            raise ValueError(f"{name}: window compiler memory drift")
    return budget_resident_execution(
        census,
        compiled_memory,
        active_graph=active_graph,
        resident_graphs=PROGRAMS,
        required_reserve_bytes=REQUIRED_RESERVE_BYTES,
    )
