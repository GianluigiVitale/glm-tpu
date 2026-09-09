"""Panel B128 versus unchanged DB594 B32; preregistered bounded experiment.

No baseline acceptance changes. Raw TPU-target suffix is fixed before execution;
actual compiled allocations and all-nine live budgets remain mandatory.
"""

from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from scripts.greenfield import prefill_completed_window_admission as original
from scripts.greenfield import prefill_paired_sort_admission as paired

PROFILE = "ws32-layer6-expert-panels-b128-original-b32-phase-v1"
PROGRAMS = original.PROGRAMS
REQUIRED_RESERVE_BYTES = original.REQUIRED_RESERVE_BYTES
CANDIDATE_SHA = "b3dc3996d7ab44384ad15f51a89acdc443e134f915757ffec782661b2fef24d5"
CANDIDATE_BYTES = 123770
REPO = Path(__file__).resolve().parents[2]
TARGET_BINDINGS = {
    "configs/prefill-performance-targets-v1.json": "5f7b99ce09154dfda12258128b6fa96a7cb157b74323424543676c5b473aedba",
    "docs/greenfield/PREFILL_PERFORMANCE_TARGETS.md": "0a8d99ac497a6b0722d744423d73d487be32064c44f0fda2819d74047af70fa4",
    "docs/artifacts/prefill-paired-sort-phase-db596-sealed-20260908.json": "103b65970e5dfda98d58f04b047785c10dfa92d96d179d5efa0b52cf2d035863",
}


def target_registration() -> dict:
    """Bound before fleet TPU initialization and rechecked by graph replay."""
    for name, digest in TARGET_BINDINGS.items():
        path = REPO / name
        if path.is_symlink() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"panel target/control bytes differ: {name}")
    return dict(
        bound_files=dict(TARGET_BINDINGS),
        control_db_run_id=596,
        required_wide_suffix_p50_ratio_max=0.9,
        required_wide_partial_phase_sum_p50_ratio_max=1.0,
        comparison_scope="HISTORICAL_B128_COMPONENT_NOT_MODEL_TTFT",
        model_performance_claim=False,
    )


def registered_programs() -> dict:
    target_registration()
    pins = paired.registered_programs()
    pins["candidate"]["stablehlo_sha256"] = CANDIDATE_SHA
    return pins


def _validate_memory(name: str, actual: Mapping[str, Any], pin: dict) -> None:
    if name != "candidate":
        paired._validate_memory(name, actual, pin)
        return
    expected = pin["compiled_memory"]
    if set(actual) != set(expected) or any(
        type(v) is not int or v < 0 for v in actual.values()
    ):
        raise ValueError("panel compiler memory schema differs")
    caps = {"temp_size_in_bytes": 128 << 20, "generated_code_size_in_bytes": 32 << 20}
    for key, value in actual.items():
        outside = value > caps[key] if key in caps else value != expected[key]
        if outside:
            raise ValueError(f"panel compiler allocation refused: {key}")


def inspect_program(
    name: str, stablehlo: str, optimized_hlo: str, compiled_memory: Mapping[str, Any]
) -> dict:
    pins = registered_programs()
    if (
        name not in pins
        or sha256(stablehlo.encode()).hexdigest() != pins[name]["stablehlo_sha256"]
    ):
        raise ValueError("panel preregistered StableHLO differs")
    if name == "candidate" and len(stablehlo.encode()) != CANDIDATE_BYTES:
        raise ValueError("panel candidate StableHLO size differs")
    _validate_memory(name, compiled_memory, pins[name])
    checks, precision = original.inspect_structure(name, optimized_hlo, pins[name])
    return dict(
        profile=PROFILE,
        target_registration=target_registration(),
        graph=name,
        passed=True,
        checks=checks,
        stablehlo_sha256=sha256(stablehlo.encode()).hexdigest(),
        optimized_hlo_sha256=sha256(optimized_hlo.encode()).hexdigest(),
        compiled_memory=dict(compiled_memory),
        fp32_route_sum=precision,
        numerical_execution_authorized=False,
        scope="PANEL_B128_REQUIRES_ORIGINAL_B32_BOUNDED_OUTPUT_AND_EXACT_OTHER_WITNESSES",
    )


def validate_program_report(
    report: Mapping[str, Any], *args: Any, **kwargs: Any
) -> None:
    if json.dumps(report, sort_keys=True, allow_nan=False) != json.dumps(
        inspect_program(*args, **kwargs), sort_keys=True, allow_nan=False
    ):
        raise ValueError("panel graph report differs from actual replay")


def memory_budget(census: Any, analyses: dict, *, active_graph: str) -> dict:
    from scripts.greenfield import prefill_completed_window_assembly as assembly

    return assembly.memory_budget(
        census, analyses, active_graph=active_graph, admission=sys.modules[__name__]
    )
