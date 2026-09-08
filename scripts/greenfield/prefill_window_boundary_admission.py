"""Fixed DB591 boundary graphs and memory; no numerical launch authority.

Reuse DB590's unchanged physical schedule and seven outer host coordinates,
not its model hashes or allocations. Capture changes are bound to DB591.
Original-signature reproduction and authenticated fleet execution are separate.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.validation.ws32_prefill_memory import budget_resident_execution
from scripts.greenfield.prefill_moe_precision_hlo import check_fp32_route_sum
from scripts.greenfield.prefill_window_admission import (
    PROGRAMS,
    REQUIRED_RESERVE_BYTES,
    expected_collectives,
)
from scripts.greenfield.prefill_window_boundary import validate_output_schema
from scripts.greenfield.prefill_window_locations import location_identity

PROFILE = "ws32-layer6-boundary-db591-host-coordinates-v1"
RECEIPT = Path(__file__).resolve().parents[2] / (
    "docs/artifacts/prefill-window-boundary-four-graph-acquisition-20260908.json"
)
RECEIPT_SHA = "dc8a9572539a1670a526e74d7969c299aa616a754f3df51156fefb6ce4b9b931"
# Derived from the raw-SHA-bound acquisition, before numerical capture.
HOST_EQUIVALENCE = {
    "candidate": "3184adbcd7fdd64ceb6c1067af98f82c8126c54b837cf38c8aa2240bab92e138",
    "control": "591e94b2a7d30450f2306bb8c5cb6652dee103d81b55f6a230b5452470c5f387",
    "wk_decode": "53d67b61f249f29eada9fbe503162ad75c63a0dacb3bcb435e53cee19d897129",
    "wk_promote": "644d04b179711e94148c9a1f1c6cadd99f641b9d7b8fb55a2293cf5ff11586b4",
}


def registered_programs() -> dict[str, Any]:
    raw = RECEIPT.read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA:
        raise ValueError("DB591 admission receipt content changed")
    return json.loads(raw)["programs"]


def _validate_memory(name: str, actual: Mapping[str, Any], pin: dict) -> None:
    if (
        any(type(v) is not int for v in actual.values())
        or dict(actual) != pin["compiled_memory"]
    ):
        raise ValueError(f"{name}: DB591 compiler allocation changed")


def inspect_program(
    name: str,
    stablehlo: str,
    optimized_hlo: str,
    compiled_memory: Mapping[str, Any],
    *,
    output_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Exact acquired bytes except seven host coordinates; exact output schema.

    Raw identity also binds helper/kernel bodies and model debug locations.
    Structural replay documents the physical schedule and actual FP32 combine;
    it deliberately does not expand into a new symbolic arithmetic proof.
    """
    pins = registered_programs()
    if name not in pins:
        raise ValueError("unknown boundary graph role")
    pin = pins[name]
    stable_sha = sha256(stablehlo.encode()).hexdigest()
    identity = location_identity(optimized_hlo)
    if (
        stable_sha != pin["stablehlo_sha256"]
        or identity["host_location_equivalence_sha256"] != HOST_EQUIVALENCE[name]
    ):
        raise ValueError(f"{name}: DB591 acquired graph identity changed")
    _validate_memory(name, compiled_memory, pin)
    schema_sha = None
    if name in ("candidate", "control"):
        if output_schema is None:
            raise ValueError("boundary graph requires actual compiled output schema")
        validate_output_schema(output_schema, name=name)
        schema_sha = sha256(
            json.dumps(output_schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if schema_sha != pin["compiler_output_schema_sha256"]:
            raise ValueError("DB591 output schema changed")
    elif output_schema is not None:
        raise ValueError("WK graph has no boundary capture schema")

    module = parse_hlo_module(optimized_hlo)

    def shape(value: Any) -> tuple:
        return value.dtype, value.dimensions

    collectives = [op for op in module.instructions if op.is_collective]
    payloads = Counter(
        (
            op.opcode,
            op.replica_groups,
            tuple(map(shape, op.operand_shapes)),
            tuple(map(shape, op.result_shapes)),
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
    if (
        payloads != expected_collectives(name)
        or len(collectives) != pin["physical_collective_count"]
        or (precision is not None and not precision["passed"])
    ):
        raise ValueError("DB591 physical schedule/FP32 combine differs")
    return dict(
        profile=PROFILE,
        graph=name,
        passed=True,
        stablehlo_sha256=stable_sha,
        optimized_hlo_sha256=identity["raw_optimized_hlo_sha256"],
        raw_graph_pair_exact=(
            identity["raw_optimized_hlo_sha256"] == pin["optimized_hlo_sha256"]
        ),
        host_coordinate_identity=identity,
        compiled_memory=dict(compiled_memory),
        compiler_output_schema_sha256=schema_sha,
        physical_collective_count=len(collectives),
        paired_collective_payloads_exact=True,
        fp32_route_sum=precision,
        numerical_execution_authorized=False,
        scope="BOUNDARY_GRAPH_ONLY_REQUIRES_RUNTIME_MEMORY_WK_FLEET_AND_ORIGINALS",
    )


def validate_program_report(
    report: Mapping[str, Any], *args: Any, **kwargs: Any
) -> None:
    """Replay full JSON-native report; equality alone would accept bool as int."""
    expected = inspect_program(*args, **kwargs)
    serialized = json.dumps(report, sort_keys=True, allow_nan=False)
    if report != json.loads(serialized) or serialized != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError("boundary graph report differs from original evidence replay")


def memory_budget(
    census: Mapping[str, Any],
    compiled_memory: Mapping[str, Mapping[str, Any]],
    *,
    active_graph: str,
) -> dict[str, Any]:
    """All live arrays (including captures/cache generations), four resident codes.

    Caller must measure live buffers before EVERY dispatch; this static helper
    does not establish ownership or measured numerical peak. No alias deduction.
    """
    pins = registered_programs()
    if set(compiled_memory) != set(PROGRAMS) or active_graph not in PROGRAMS:
        raise ValueError("boundary memory requires four fixed graph roles")
    for name in PROGRAMS:
        _validate_memory(name, compiled_memory[name], pins[name])
    return budget_resident_execution(
        census,
        compiled_memory,
        active_graph=active_graph,
        resident_graphs=PROGRAMS,
        required_reserve_bytes=REQUIRED_RESERVE_BYTES,
    )
