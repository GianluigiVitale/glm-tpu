"""Fixed long-prefill execution boundaries, not full worker launch authority.

Reuse acquired graph allocations, the original host loop and32-owner memory
validator. Phase loading and companion/outer evidence remain caller obligations.
"""

from __future__ import annotations

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
        if (getattr(args, f"expected_{role}_stablehlo_sha256") != entry["stablehlo_sha256"]
                or report.get("stablehlo_sha256") != entry["stablehlo_sha256"]):
            raise ValueError("long RAW pin/report differs from original")
        optimized_pin = getattr(args, f"expected_{role}_optimized_hlo_sha256")
        if optimized_pin not in (entry["optimized_hlo_sha256"], hlo.FRESH_OPTIMIZED_MARKER):
            raise ValueError("long optimized request is neither original nor fresh profile")
        actual_sha, policy = hlo.optimized_identity(compiled[role].as_text(), optimized_pin)
        if (report.get("optimized_hlo_sha256") != actual_sha
                or (optimized_pin == hlo.FRESH_OPTIMIZED_MARKER
                    and report.get("optimized_identity_policy") != policy)):
            raise ValueError("long actual executable lacks its own inspected identity")
        if (report.get("schema") != "ws32_delivery_long_structural_v1"
                or report.get("passed") is not True
                or report.get("context_capacity") != plan.context_capacity
                or report.get("block_rows") != dict(plan.graph_rows)[role]
                or report.get("state_ownership_contract") != programs.state_ownership(context_label)
                or report.get("dispatch_authorized") is not False):
            raise ValueError("long prefill graph inspection is incomplete")
        analysis = compiled[role].memory_analysis()
        require_memory({key: int(getattr(analysis, key)) for key in MEMORY_FIELDS}, entry["memory"])


def identity(context_label: str) -> dict[str, Any]:
    return programs.long_plan(context_label).identity(
        state_ownership_contract=programs.state_ownership(context_label))


def numerical_identity(context_label: str) -> dict[str, Any]:
    """Fixed request/journal/runner identity, not a success or launch permit."""
    from glm_tpu.greenfield.validation.long_context_oracle import WS32_LONG_CONTEXT_PROFILES
    from glm_tpu.greenfield.validation.ws32_prefill import PREFILL_MODE
    from glm_tpu.greenfield.validation.ws32_prefill_admission import ROLLED_SHORT_PROFILE, short_program_options

    plan = programs.long_plan(context_label)
    options = {**short_program_options(ROLLED_SHORT_PROFILE), "canonical_dense": True}
    if context_label == "256k_e0":
        options.update(pending_cache_rows=True, flat_pending_rows=True, capture_barrier=True)
    return dict(
        prefill_mode=PREFILL_MODE, batched_prefill_profile=PROFILE,
        delivery_context_label=context_label,
        validation_contract="s26_long_s23_5_task_state_cache_v1",
        batched_prefill_plan=identity(context_label),
        batched_prefill_program_options=options,
        batched_prefill_acquisition=dict(original_receipts=dict(programs.PREREQUISITES),
            optimized_graphs_acquired_in_numerical_run=True, numerical_inheritance=False),
        long_workload=dict(WS32_LONG_CONTEXT_PROFILES[context_label]),
        context_capacity=plan.context_capacity,
        prefill_memory_reserve_bytes=RESERVE, prefill_budget_seconds=budget_seconds(context_label),
    )


def require_request(args: Any, *, context_label: str, prompt_length: int, repo: Path) -> None:
    """Pre-load fixed workload checks shared by future worker/sealer entry.

    No TPU or payload access. All fourteen pins are explicit: original RAW plus
    fresh-optimized markers, never arbitrary worker-chosen graph acceptance.
    Existing metadata/overlay/topology/clean-source guards remain mandatory.
    """
    from glm_tpu.greenfield.validation.ws32_prefill import PREFILL_MODE
    from glm_tpu.greenfield.validation.long_context_oracle import WS32_LONG_CONTEXT_PROFILES
    from scripts.greenfield import ws32_delivery_companions as companions

    plan = programs.long_plan(context_label)
    entry = WS32_LONG_CONTEXT_PROFILES[context_label]
    fixed = dict(prefill_chunk=128, context_capacity=plan.context_capacity, exact_dsa=1,
        strategy_nd_dense=1, host_main_rope_table=1, rotary_diagnostic=0,
        observer_steps=14 if context_label == "256k_e0" else 20,
        warmup=2, iterations=256 if context_label == "256k_e0" else 10, trace_steps=2,
        prefill_memory_reserve_bytes=RESERVE)
    if (args.prefill_mode != PREFILL_MODE or args.batched_prefill_profile != PROFILE
            or getattr(args, "compile_only", 0) != 0
            or type(prompt_length) is not int or prompt_length != plan.prompt_length
            or any(type(getattr(args, k, None)) is not int or getattr(args, k) != v for k, v in fixed.items())
            or type(args.prefill_budget_seconds) not in (int, float)
            or args.prefill_budget_seconds != budget_seconds(context_label)):
        raise ValueError("long delivery request geometry/options/budget differs")
    if (args.long_context != entry["kind"]
            or args.long_context_manifest_sha256 != entry["manifest_sha256"]
            or args.long_context_success_sha256 != entry["success_sha256"]
            or args.dsa_adjudication_record is not None
            or args.dsa_adjudication_sha256 != hlo.FRESH_OPTIMIZED_MARKER
            or getattr(args, "delivery_context_label", context_label) != context_label):
        raise ValueError("long delivery sealed workload/depth/contract differs")
    raw = {role: values[1] for role, values in programs.raw_registration(context_label).items()}
    raw.update(companions.raw_registration(context_label))
    for graph, digest in raw.items():
        if (getattr(args, f"expected_{graph}_stablehlo_sha256", None) != digest
                or getattr(args, f"expected_{graph}_optimized_hlo_sha256", None) != hlo.FRESH_OPTIMIZED_MARKER):
            raise ValueError("long delivery requires fixed RAW and explicit fresh optimized pins")
    programs.require_source(repo)


def require_role_record(admission: Mapping[str, Any], context_label: str) -> None:
    """E0's compile-once contract must survive publication and sealer replay."""
    programs.long_plan(context_label)
    if context_label == "256k_e0" and admission.get("executable_roles") != dict.fromkeys(ROLES, ROLES[0]):
        raise ValueError("E0 memory record must identify one shared compiled object")
