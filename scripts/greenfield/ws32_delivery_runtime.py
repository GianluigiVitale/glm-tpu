"""Fixed long-prefill execution boundaries, not full worker launch authority.

Reuse acquired graph allocations, the original host loop and32-owner memory
validator. Phase loading and companion/outer evidence remain caller obligations.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from glm_tpu.greenfield.validation.ws32_prefill_memory import MEMORY_FIELDS
from glm_tpu.greenfield.validation.ws32_prefill_admission import SHORT_RESERVE_BYTES, SHORT_DEVICE_LIMIT_BYTES
from scripts.greenfield import ws32_delivery_hlo as hlo
from scripts.greenfield import ws32_delivery_programs as programs
from scripts.greenfield.ws32_phase_weights import PhaseWeights

PROFILE = hlo.PROFILE
RESERVE = SHORT_RESERVE_BYTES
DEVICE_LIMIT = SHORT_DEVICE_LIMIT_BYTES
ROLES = ("prefill_chunk", "prefill_tail")


def budget_seconds(context_label: str) -> int:
    """Existing §23 long-run ceilings, NOT speed targets or timing forecasts."""
    programs.long_plan(context_label)
    return 54000 if context_label == "256k_e0" else 27000


def registration(repo: Path, context_label: str) -> dict[str, dict[str, Any]]:
    """Read SHA-bound original allocations; never accept a worker-chosen cap."""
    programs.require_source(repo)
    raw = programs.raw_registration(context_label)
    if context_label == "256k_e0":
        receipt = json.loads((repo / "docs/artifacts/prefill-capture-barrier-db615-sealed-20260912.json").read_text())
        return {role: dict(stablehlo_sha256=raw[role][1],
                           optimized_hlo_sha256=receipt["optimized_hlo_sha256"],
                           memory=dict(receipt["memory"])) for role in ROLES}
    receipt = json.loads((repo / "docs/artifacts/prefill-delivery-long-compile-oom-20260911.json").read_text())
    result = {}
    for role in ROLES:
        name = "prefill_128k_" + role.removeprefix("prefill_")
        originals = [worker["programs"][name] for worker in receipt["workers"]]
        first = originals[0]
        value = dict(stablehlo_sha256=raw[role][1],
                     optimized_hlo_sha256=first["optimized_hlo_sha256"],
                     memory=dict(first["compiled_memory"]))
        if len(originals) != 8 or any(
            row["stablehlo_sha256"] != value["stablehlo_sha256"]
            or row["optimized_hlo_sha256"] != value["optimized_hlo_sha256"]
            or row["compiled_memory"] != value["memory"] for row in originals
        ):
            raise ValueError("long original graph fleet disagrees")
        result[role] = value
    return result


def require_memory(analysis: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if (set(analysis) != set(MEMORY_FIELDS)
            or any(type(value) is not int or value < 0 for value in analysis.values())
            or dict(analysis) != dict(expected)):
        raise ValueError("long compiled allocation differs from original")


def require_inputs(
    *, repo: Path, context_label: str, args: Any, plan: Any, config: Any,
    compiled: Mapping[str, Any], reports: Mapping[str, Any], owner: PhaseWeights,
    weights: Any, wk: Any,
) -> None:
    """Caller runs this inside existing voted preflight before cache allocation.

    Graph reports are worker-local prior inspection, not external authorization.
    Bind the actual compiled text; sealer independently replays the same bytes.
    This does not replace the all-live census or prove no other borrowers.
    """
    if (args.batched_prefill_profile != PROFILE or plan != programs.long_plan(context_label)
            or config.context_capacity != plan.context_capacity
            or type(args.prefill_memory_reserve_bytes) is not int
            or args.prefill_memory_reserve_bytes != RESERVE
            or type(args.prefill_budget_seconds) not in (int, float)
            or args.prefill_budget_seconds != budget_seconds(context_label)):
        raise ValueError("long prefill request/profile/budget differs")
    if (not isinstance(owner, PhaseWeights) or owner.phase != "prefill"
            or owner.raw_config != config or owner.raw_weights is not weights
            or owner.wk is not wk or len(wk) != len(config.full_index_slots)
            or owner.decode_weights is not None):
        raise ValueError("long prefill requires its completed raw/WK phase owner")
    expected = registration(repo, context_label)
    if set(compiled) != set(ROLES) or not isinstance(reports, Mapping) or set(reports) != set(ROLES):
        raise ValueError("long prefill requires inspected original graph pair")
    if context_label == "256k_e0" and compiled[ROLES[0]] is not compiled[ROLES[1]]:
        raise ValueError("E0 must reuse one compiled object for both roles")
    for role in ROLES:
        report = reports[role]
        entry = expected[role]
        for form in ("stablehlo_sha256", "optimized_hlo_sha256"):
            if getattr(args, f"expected_{role}_{form}") != entry[form] or report.get(form) != entry[form]:
                raise ValueError("long graph pin/report differs from original")
        if (report.get("schema") != "ws32_delivery_long_structural_v1"
                or report.get("passed") is not True
                or report.get("context_capacity") != plan.context_capacity
                or report.get("block_rows") != dict(plan.graph_rows)[role]
                or report.get("state_ownership_contract") != programs.state_ownership(context_label)
                or report.get("dispatch_authorized") is not False):
            raise ValueError("long prefill graph inspection is incomplete")
        if sha256(compiled[role].as_text().encode()).hexdigest() != entry["optimized_hlo_sha256"]:
            raise ValueError("long actual executable differs from inspected original")
        analysis = compiled[role].memory_analysis()
        require_memory({key: int(getattr(analysis, key)) for key in MEMORY_FIELDS}, entry["memory"])


def identity(context_label: str) -> dict[str, Any]:
    return programs.long_plan(context_label).identity(
        state_ownership_contract=programs.state_ownership(context_label))


def require_role_record(admission: Mapping[str, Any], context_label: str) -> None:
    """E0's compile-once contract must survive publication and sealer replay."""
    programs.long_plan(context_label)
    if context_label == "256k_e0" and admission.get("executable_roles") != dict.fromkeys(ROLES, ROLES[0]):
        raise ValueError("E0 memory record must identify one shared compiled object")
