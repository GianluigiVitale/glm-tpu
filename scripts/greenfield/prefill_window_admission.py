"""DB590 window admission facts reused by the native path.

The retained WK program admission reuses this module's physical groups and
expected collective schedule, and BudgetedCalls' default budgeter and evidence
replay use its receipt-bound compiler-memory pins and reserve. The window
campaign's own optimized-HLO inspection (host-coordinate identity, route-sum
proof) retired with the campaign at b667f00f. This is not a general symbolic
model proof, full-model admission, timing permission or a launcher.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from glm_tpu.greenfield.validation.ws32_prefill_memory import budget_resident_execution

# Verbatim (b667f00f) physical groups of the retired prefill_layer_hlo.py.
FEATURE = tuple(tuple(range(e * 4, e * 4 + 4)) for e in range(8))
EXPERT = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))

PROFILE = "ws32-layer6-window-db590-host-coordinates-v2"
RECEIPT = Path(__file__).resolve().parents[2] / (
    "docs/artifacts/prefill-window-layer6-four-graph-acquisition-20260908.json"
)
RECEIPT_SHA = "b5e336dadfd2a99c80c5b76cd77d2d11ce5a15d81bc053bc4a6377cfdf94607e"
PROGRAMS = ("wk_decode", "wk_promote", "candidate", "control")
REQUIRED_RESERVE_BYTES = 1 << 30
# Computed from the raw-SHA-bound DB590 originals, not from a candidate run.
# Retain the receipt's raw hashes independently. Only 28 host coordinates differ.


def registered_programs() -> dict[str, Any]:
    """The acquisition receipt supplies fixed facts, never an operator verdict."""
    raw = RECEIPT.read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA:
        raise ValueError("DB590 admission receipt content changed")
    return json.loads(raw)["programs"]


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
