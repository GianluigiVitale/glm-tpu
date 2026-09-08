"""Paired-position-sort phase admission; never changes DB593/595 acceptance.

The prefix StableHLO is preregistered from production abstract inputs, lowering
for TPU-v4 with Mosaic v13. That offline path reproduces all five DB593 originals
byte-for-byte with the flag off. Actual TPU allocations are bounded, not guessed
equal to the old prefix, and all nine resident programs enter the live budget.
Exact DB594 original-array reproduction is required before measured samples.
"""

from hashlib import sha256
import json
from typing import Any, Mapping

from scripts.greenfield import prefill_completed_window_admission as original

PROFILE = "ws32-layer6-paired-position-sort-phase-v1"
PROGRAMS = original.PROGRAMS
REQUIRED_RESERVE_BYTES = original.REQUIRED_RESERVE_BYTES
PREFIX_SHA = "7cd48f470d086072efbb86906b1d691b596f2be137800aaa33967e21db083c5c"
PREFIX_BYTES = 273757


def registered_programs() -> dict:
    pins = original.registered_programs()
    pins["prefix"]["stablehlo_sha256"] = PREFIX_SHA
    return pins


def _validate_memory(name: str, actual: Mapping[str, Any], pin: dict) -> None:
    if name != "prefix":
        original._validate_memory(name, actual, pin)
        return
    expected = pin["compiled_memory"]
    if set(actual) != set(expected) or any(
        type(v) is not int or v < 0 for v in actual.values()
    ):
        raise ValueError("paired prefix compiler memory schema differs")
    # Identical signature and no added aliases. Scratch/code may change under
    # the new lowering, but cannot exceed the preregistered bounded experiment.
    caps = {"temp_size_in_bytes": 128 << 20, "generated_code_size_in_bytes": 32 << 20}
    for key, value in actual.items():
        outside = value > caps[key] if key in caps else value != expected[key]
        if outside:
            raise ValueError(f"paired prefix compiler allocation refused: {key}")


def inspect_program(
    name: str, stablehlo: str, optimized_hlo: str, compiled_memory: Mapping[str, Any]
) -> dict:
    pins = registered_programs()
    if (
        name not in pins
        or sha256(stablehlo.encode()).hexdigest() != pins[name]["stablehlo_sha256"]
    ):
        raise ValueError("paired phase preregistered StableHLO differs")
    if name == "prefix" and len(stablehlo.encode()) != PREFIX_BYTES:
        raise ValueError("paired prefix StableHLO size differs")
    _validate_memory(name, compiled_memory, pins[name])
    checks, precision = original.inspect_structure(name, optimized_hlo, pins[name])
    return dict(
        profile=PROFILE,
        graph=name,
        passed=True,
        checks=checks,
        stablehlo_sha256=sha256(stablehlo.encode()).hexdigest(),
        optimized_hlo_sha256=sha256(optimized_hlo.encode()).hexdigest(),
        compiled_memory=dict(compiled_memory),
        fp32_route_sum=precision,
        optimized_identity_scope="ACTUAL_TPU_GRAPH_STRUCTURAL_CHECK_NOT_DB593_BYTE_IDENTITY",
        numerical_execution_authorized=False,
        scope="PAIRED_SORT_REQUIRES_NINE_GRAPH_LIVE_BUDGET_WK_AND_DB594_ORIGINAL_REPRODUCTION",
    )


def validate_program_report(
    report: Mapping[str, Any], *args: Any, **kwargs: Any
) -> None:
    if json.dumps(report, sort_keys=True, allow_nan=False) != json.dumps(
        inspect_program(*args, **kwargs), sort_keys=True, allow_nan=False
    ):
        raise ValueError("paired phase report differs from actual graph replay")


def memory_budget(census: Any, analyses: dict, *, active_graph: str) -> dict:
    import sys
    from scripts.greenfield import prefill_completed_window_assembly as assembly

    return assembly.memory_budget(
        census, analyses, active_graph=active_graph, admission=sys.modules[__name__]
    )
