"""Fixed DB593 graphs and simultaneous completed-window memory admission.

No numerical entry point, new precision policy or inherited DB590 admission.
Seven existing host coordinates may move; every other graph byte stays exact.
This is a selected-layer prerequisite, not full-layer/DSA or performance proof.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.validation.ws32_prefill_memory import budget_resident_execution
from scripts.greenfield.prefill_layer_hlo import EXPERT, FEATURE
from scripts.greenfield.prefill_moe_precision_hlo import check_fp32_route_sum
from scripts.greenfield.prefill_window_acquisition import COMPLETED_PROGRAMS as PROGRAMS
from scripts.greenfield.prefill_window_admission import (
    expected_collectives as wk_collectives,
)
from scripts.greenfield.prefill_window_locations import location_identity

PROFILE = "ws32-layer6-completed-window-db593-v1"
RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "docs/artifacts/prefill-completed-window-five-graph-acquisition-20260908.json"
)
RECEIPT_SHA = "11164d7feef5d20040809c1fce67bd4af488106afca47c657d8c67912989da2c"
REQUIRED_RESERVE_BYTES = 1 << 30
HOST_EQUIVALENCE = {
    "wk_decode": "53d67b61f249f29eada9fbe503162ad75c63a0dacb3bcb435e53cee19d897129",
    "wk_promote": "644d04b179711e94148c9a1f1c6cadd99f641b9d7b8fb55a2293cf5ff11586b4",
    "prefix": "d0f6bf5a6f66b1665a8a61b27a3386f17a628b799da46024fc09249754ddce9e",
    "candidate": "2b4e57c5feb39bf0f91cd75263eac621673783dfcd9191b4925d03f0d2492b61",
    "control": "3e7b217c8bc8e265504f46a2a49625057d05331dad34569cedcbebfcf8fa7eb1",
}


def registered_programs() -> dict[str, Any]:
    raw = RECEIPT.read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA:
        raise ValueError("DB593 acquisition receipt changed")
    return json.loads(raw)["programs"]


def expected_collectives(name: str) -> Counter:
    """Physical paired payloads from the actual prefix/suffix source schedule."""
    if name in ("wk_decode", "wk_promote"):
        return wk_collectives(name)
    result = Counter()

    def add(op, groups, inputs, outputs=None, count=1):
        result[
            (op, groups, tuple(inputs), tuple(inputs if outputs is None else outputs))
        ] += count

    def f(*shape):
        return "f32", shape

    def b(*shape):
        return "bf16", shape

    if name == "prefix":
        add("all-reduce", FEATURE, [f(32, 1)], count=2)
        add("all-gather", FEATURE, [b(32, 1536)], [b(32, 6144)])
        add(
            "all-reduce",
            FEATURE,
            [f(32, 576), f(32, 128), f(32, 2048), f(32, 4)],
            [b(32, 576), f(32, 128), b(32, 2048), f(32, 4)],
        )
        add("all-gather", EXPERT, [f(32, 516)], [f(256, 516)])
        for dtype in ("s32", "f32"):
            add("all-gather", EXPERT, [(dtype, (32, 2048))], [(dtype, (256, 2048))])
        add("all-reduce", EXPERT, [b(32, 2048, 640)])
        add("all-reduce", EXPERT, [f(32, 1536)], [b(32, 1536)])
    elif name in ("candidate", "control"):
        rows = 128 if name == "candidate" else 32
        add("all-reduce", EXPERT, [f(256)])
        add("all-reduce", FEATURE, [f(rows, 32)])
        add("all-gather", EXPERT, [f(rows, 32)], [f(rows, 256)])
        add(
            "all-reduce",
            FEATURE,
            [f(2, rows, 2048), f(2, rows * 8, 2048)],
            [b(2, rows, 2048), b(2, rows * 8, 2048)],
        )
        add("all-reduce", EXPERT, [f(rows, 1536)], [b(rows, 1536)])
    else:
        raise ValueError("unknown completed-window program")
    return result


def _validate_memory(name: str, actual: Mapping[str, Any], pin: dict) -> None:
    if (
        any(type(v) is not int for v in actual.values())
        or dict(actual) != pin["compiled_memory"]
    ):
        raise ValueError(f"{name}: DB593 compiler allocation changed")


def inspect_program(
    name: str, stablehlo: str, optimized_hlo: str, compiled_memory: Mapping[str, Any]
) -> dict[str, Any]:
    pins = registered_programs()
    if name not in pins:
        raise ValueError("unknown completed-window graph role")
    pin = pins[name]
    identity = location_identity(optimized_hlo)
    stable_sha = sha256(stablehlo.encode()).hexdigest()
    if (
        stable_sha != pin["stablehlo_sha256"]
        or identity["host_location_equivalence_sha256"] != HOST_EQUIVALENCE[name]
    ):
        raise ValueError(f"{name}: DB593 acquired graph identity changed")
    _validate_memory(name, compiled_memory, pin)
    checks, precision = inspect_structure(name, optimized_hlo, pin)
    return dict(
        profile=PROFILE,
        graph=name,
        passed=True,
        checks=checks,
        stablehlo_sha256=stable_sha,
        optimized_hlo_sha256=identity["raw_optimized_hlo_sha256"],
        raw_graph_pair_exact=identity["raw_optimized_hlo_sha256"]
        == pin["optimized_hlo_sha256"],
        host_coordinate_identity=identity,
        compiled_memory=dict(compiled_memory),
        fp32_route_sum=precision,
        numerical_execution_authorized=False,
        scope="FIXED_COMPLETED_GRAPH_REQUIRES_LIVE_MEMORY_WK_FLEET_AND_NUMERICAL_PROTOCOL",
    )


def inspect_structure(name: str, optimized_hlo: str, pin: Mapping[str, Any]) -> tuple:
    """Shared physical checks; callers separately bind their own graph identity."""
    module = parse_hlo_module(optimized_hlo)
    collectives = [op for op in module.instructions if op.is_collective]
    payloads = Counter(
        (
            op.opcode,
            op.replica_groups,
            tuple((s.dtype, s.dimensions) for s in op.operand_shapes),
            tuple((s.dtype, s.dimensions) for s in op.result_shapes),
        )
        for op in collectives
    )
    precision = None
    if name in ("candidate", "control"):
        precision = check_fp32_route_sum(
            module,
            rows=128 if name == "candidate" else 32,
            expert_scope="greenfield_ws32_prefill_moe/expert_reduce",
        )
    checks = dict(
        paired_collective_payloads=payloads == expected_collectives(name),
        physical_collective_count=len(collectives) == pin["physical_collective_count"],
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
        raise ValueError(f"{name}: completed-window structure differs: {checks}")
    return checks, precision


def validate_program_report(
    report: Mapping[str, Any], *args: Any, **kwargs: Any
) -> None:
    expected = inspect_program(*args, **kwargs)
    serialized = json.dumps(report, sort_keys=True, allow_nan=False)
    if report != json.loads(serialized) or serialized != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError(
            "completed-window report differs from original evidence replay"
        )


def memory_budget(
    census: Mapping[str, Any],
    compiled_memory: Mapping[str, Mapping[str, Any]],
    *,
    active_graph: str,
) -> dict[str, Any]:
    """Five resident executables plus ALL live prefix/cache/assembly outputs.

    Caller must take the existing complete live-array census before each call,
    after device concatenation is complete. Old caches/prefixes remain visible;
    no alias subtraction or summed duplicate weight arguments. This estimate
    is not a measured numerical peak and cannot authorize dispatch by itself.
    """
    pins = registered_programs()
    if set(compiled_memory) != set(PROGRAMS) or active_graph not in PROGRAMS:
        raise ValueError("completed-window memory requires all five graph roles")
    for name in PROGRAMS:
        _validate_memory(name, compiled_memory[name], pins[name])
    return budget_resident_execution(
        census,
        compiled_memory,
        active_graph=active_graph,
        resident_graphs=PROGRAMS,
        required_reserve_bytes=REQUIRED_RESERVE_BYTES,
    )
