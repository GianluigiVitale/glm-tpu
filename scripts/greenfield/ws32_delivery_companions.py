"""Reuse capacity-specific decoder HLO checks, not serial model execution.

DB571/572 originals are retained separately from current literal RAW hashes.
CPU TPU-target lowering found only embedded debug-information changes in
observer/decode; full non-debug Mosaic bodies and surrounding RAW agree.
That comparison is a local diagnostic, NOT a runtime normalizer. Fresh numerical
optimized text is inspected in full by the unchanged decoder/materializer
validators; no debug-byte normalizer or inherited optimized PASS is used.
This component does not authorize a request, memory fit or task-quality claim.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from glm_tpu.greenfield.benchmarking.ws32_decoder import (
    validate_ws32_decoder_hlo, validate_ws32_exact_dsa_materializer_hlo,
)
from scripts.greenfield import ws32_delivery_hlo as delivery

ROLES = ("exact_materialize", "exact_promote", "observer", "decode", "cache_probe")
ORIGINALS = {
    131072: (571, "greenfield_ws32_short_decoder_8k_numerical_cap131072_hrope_20260906T191737720789480Z",
             "f02c1e5ab53765dc72952e4b70090dc9a10a103535da551e4614b7bac4fcaa5f"),
    262656: (572, "greenfield_ws32_short_decoder_8k_numerical_cap262656_hrope_20260906T211233754818996Z",
             "f2f7a954775db60063a7864a941d7119496eb8be6eb57141a7f9a3b70ff36b19"),
}
_COMMON = {
    "exact_materialize": "1d925d96f4770e6edd5cef41e1c6f8f4071e039ad93b1cf7d96380fbfdf6f36e",
    "exact_promote": "e38eb7a45472107a383114c25039b9cfa58c2120fe98592f62d8f47996d3ffff",
}
_CAPACITY_RAW = {
    131072: {
        "observer": "ccbf007d26721750fb65b843eddcf25c639b2a54e61014b757792be85b2edca4",
        "decode": "ba2600e6f00ca403c03e7ead69874a1482fc642803c5a2b3a56b06762d51c740",
        "cache_probe": "81bd74150384d062cef5761da6702669a574d40483ccc5843bced757c4eeead8",
    },
    262656: {
        "observer": "7b15165c8af5811bde06540c5d951d490378f8f776aa3d73744a16a908cfec91",
        "decode": "9b05cf6ec103ffa78683bcd3781c8e03fe6906f1958819181c09e9fa0d4f19e8",
        "cache_probe": "c050dc266cba14f6c3e97c279a843085b2f5747f295c1f97c8c14ab499390dd0",
    },
}
_CURRENT_RAW = {
    131072: {**_CAPACITY_RAW[131072],
        "observer": "b62f899d42ab85b2ccce7739ddbdddc6e87f620d07d7d9029792bc6969eed5af",
        "decode": "d5ee604cf9aa2eb8e10ae0fb620ccbd5cf665fecbb16c7f1270c6e1e8b6865b3"},
    262656: {**_CAPACITY_RAW[262656],
        "observer": "e6632e0284a2f37b7211f06b7627625a46e86efd6e47c42157d11ed790fad9c7",
        "decode": "1400eb19d678707625f4ed2b0688783557ca53fae4c130aeffaee7fa2ff999a7"},
}


def raw_registration(context_label: str, *, historical: bool = False) -> dict[str, str]:
    """Current literal pins by default; old pins only for retained replay."""
    capacity = delivery.programs.long_plan(context_label).context_capacity
    return {**_COMMON, **(_CAPACITY_RAW if historical else _CURRENT_RAW)[capacity]}


def inspect_hlo(
    stable: str, optimized: str, *, repo: Path, context_label: str, graph: str,
    expected_stable: str, expected_optimized: str,
) -> dict[str, Any]:
    """Worker and sealer inspect the same original bytes independently."""
    raw = raw_registration(context_label)
    historical = raw_registration(context_label, historical=True)
    # Actual requests require CURRENT pins in require_request. This inspector
    # also replays explicitly named originals without relabelling their bytes.
    if (graph not in raw or expected_stable not in (raw[graph], historical[graph])
            or sha256(stable.encode()).hexdigest() != expected_stable):
        raise ValueError("long companion RAW graph/capacity differs")
    delivery.programs.require_source(repo)
    actual, policy = delivery.optimized_identity(optimized, expected_optimized)
    pins = dict(expected_stablehlo_sha256=expected_stable,
                expected_optimized_hlo_sha256=actual)
    if graph.startswith("exact_"):
        report = validate_ws32_exact_dsa_materializer_hlo(
            stable, optimized, kind=graph, **pins).to_dict()
    else:
        report = validate_ws32_decoder_hlo(
            stable, optimized, kind=graph, hidden_size=6144, exact_dsa=True,
            strategy_nd_dense=True, host_main_rope_table=True, **pins).to_dict()
    if report.get("passed") is not True:
        raise ValueError(f"long companion structural refusal: {graph}: {report.get('violations')}")
    return {**report, "source_location_identity": {
        "schema": "ws32_delivery_companion_identity_v1", "profile": delivery.PROFILE,
        "context_capacity": delivery.programs.long_plan(context_label).context_capacity,
        "raw_stablehlo_sha256": expected_stable, "raw_optimized_hlo_sha256": actual,
        "raw_registration_basis": "current" if expected_stable == raw[graph] else "historical_replay_only",
        "optimized_identity_policy": policy, "numerical_inheritance": False,
        "dispatch_authorized": False,
    }}
