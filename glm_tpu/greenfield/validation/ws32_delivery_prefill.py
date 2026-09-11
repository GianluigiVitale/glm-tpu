"""Fixed §26 batched long-workload geometry, not execution admission.

Reuse §23.5's sealed workload registry and the existing B128/B114 program
builders. These plans certify neither checkpoint payloads nor HLO/HBM fit.
The worker/sealer must still admit the actual capacity-specific programs.
"""

from __future__ import annotations

from .long_context_oracle import WS32_LONG_CONTEXT_PROFILES
from .ws32_prefill import BatchedPrefillPlan


def long_plan(context_label: str) -> BatchedPrefillPlan:
    """Return the frozen batched geometry for one registered L7/L8 workload.

    L7's three live tail rows use the existing physical B114 implementation.
    E0's final 128 live rows require B128, never B114 or prompt truncation.
    This API has no arbitrary length/capacity/row-size override.
    """
    if type(context_label) is not str or context_label not in WS32_LONG_CONTEXT_PROFILES:
        raise ValueError("delivery prefill requires a registered long-context label")
    entry = WS32_LONG_CONTEXT_PROFILES[context_label]
    e0 = context_label == "256k_e0"
    expected_kind, expected_length = ("e0", 262144) if e0 else ("passkey", 127363)
    if (
        entry["kind"] != expected_kind
        or type(entry["prompt_token_count"]) is not int
        or entry["prompt_token_count"] != expected_length
    ):
        raise ValueError("delivery long-context workload registry differs")
    return BatchedPrefillPlan(
        prompt_length=entry["prompt_token_count"],
        block_rows=128,
        context_capacity=262656 if e0 else 131072,
        mlp_window=True,
        tail_graph_rows=128 if e0 else 114,
    )
