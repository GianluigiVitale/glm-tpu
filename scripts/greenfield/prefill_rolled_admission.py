"""Preregistered combined layer, actual-compiler checks and live memory budget.

Untimed retained-reference discriminator only, never full-model promotion.
No optional-copy counts or debug-coordinate predictions. Existing independent
numerical comparisons remain mandatory; shape/group checks are not arithmetic.
"""

from collections import Counter
from hashlib import sha256
import json
import re
from typing import Any, Mapping

from glm_tpu.greenfield.sharding.hlo_contract import parse_hlo_module
from glm_tpu.greenfield.validation.ws32_prefill_memory import budget_resident_execution
from scripts.greenfield import prefill_completed_window_admission as completed
from scripts.greenfield import prefill_paired_sort_admission as paired
from scripts.greenfield import prefill_rolled_window as protocol
from scripts.greenfield.prefill_moe_precision_hlo import check_fp32_route_sum

PROFILE = "ws32-layer6-rolled128-retained-db600-v1"
PROGRAMS = protocol.PROGRAMS
REQUIRED_RESERVE_BYTES = 1 << 30
CANDIDATE_SHA = "f4eed4e7f25abfaf24f22cb69fcbd7f331f1d6d082e237a4049ed1b0f7e58426"
CANDIDATE_BYTES = 441963
MEMORY_CAPS = {
    "argument_size_in_bytes": 512 << 20,
    "output_size_in_bytes": 32 << 20,
    "alias_size_in_bytes": 0,
    "temp_size_in_bytes": 512 << 20,
    "generated_code_size_in_bytes": 128 << 20,
}


def registered_programs() -> dict:
    protocol.originals._json_bound(protocol.SEAL, protocol.SEAL_SHA)
    pins = paired.registered_programs()
    return {
        **{name: pins[name] for name in PROGRAMS[:2]},
        "candidate": dict(
            stablehlo_sha256=CANDIDATE_SHA, stablehlo_bytes=CANDIDATE_BYTES
        ),
    }


def validate_memory(name: str, memory: Mapping[str, Any]) -> None:
    if name not in PROGRAMS:
        raise ValueError("unregistered rolled graph")
    if name != "candidate":
        paired._validate_memory(name, memory, registered_programs()[name])
        return
    if set(memory) != set(MEMORY_CAPS) or any(
        type(v) is not int or v < 0 or v > MEMORY_CAPS[k] for k, v in memory.items()
    ):
        raise ValueError("rolled candidate compiler allocations exceed registered caps")


def _flatten(counter: Counter) -> Counter:
    """Tuple fusion may change operation count, never ordered leaf pair semantics."""
    result = Counter()
    for (opcode, groups, inputs, outputs), count in counter.items():
        if len(inputs) != len(outputs):
            raise ValueError("rolled collective input/output leaf count differs")
        for left, right in zip(inputs, outputs, strict=True):
            result[(opcode, groups, left, right)] += count
    return result


def inspect_structure(optimized: str) -> dict:
    module = parse_hlo_module(optimized)
    collectives = [op for op in module.instructions if op.is_collective]
    actual = Counter(
        (
            op.opcode,
            op.replica_groups,
            tuple((s.dtype, s.dimensions) for s in op.operand_shapes),
            tuple((s.dtype, s.dimensions) for s in op.result_shapes),
        )
        for op in collectives
    )
    # The prefix body appears ONCE in the rolled graph, executing four times;
    # the wide suffix executes once. Do not report static count as dynamic count.
    expected = completed.expected_collectives(
        "prefix"
    ) + completed.expected_collectives("candidate")
    loops = [
        op
        for op in module.instructions
        if op.opcode == "while"
        and (op.op_name or "").endswith("greenfield_ws32_prefill_rolled_prefix/while")
    ]
    precision = check_fp32_route_sum(
        module, rows=128, expert_scope="greenfield_ws32_prefill_moe/expert_reduce"
    )
    checks = dict(
        collective_leaf_payloads=_flatten(actual) == _flatten(expected),
        one_outer_rolled_prefix=len(loops) == 1,
        fp32_route_sum=precision["passed"],
        no_host_transport=not any(
            op.opcode in ("infeed", "outfeed", "send", "recv")
            or bool(
                re.search(
                    r'custom_call_target="[^"]*(?:host|io|python)[^"]*callback',
                    op.raw_line,
                )
            )
            for op in module.instructions
        ),
        no_full_weight_expansion=not any(
            s.dtype in ("bf16", "f32") and s.element_count >= 32 * 2048 * 1536
            for op in module.instructions
            for s in op.result_shapes
        ),
    )
    if not all(checks.values()):
        raise ValueError(f"rolled candidate actual structure differs: {checks}")
    return dict(
        checks=checks,
        fp32_route_sum=precision,
        static_collective_count=len(collectives),
        rolled_loops=[op.raw_line for op in loops],
        dynamic_prefix_iterations=4,
        numerical_boundary_identity_claim=False,
    )


def inspect_program(
    name: str, stable: str, optimized: str, memory: Mapping[str, Any]
) -> dict:
    pins = registered_programs()
    if (
        name not in pins
        or sha256(stable.encode()).hexdigest() != pins[name]["stablehlo_sha256"]
    ):
        raise ValueError("rolled preregistered raw graph differs")
    validate_memory(name, memory)
    if name != "candidate":
        structure = paired.inspect_program(name, stable, optimized, memory)
    else:
        if len(stable.encode()) != CANDIDATE_BYTES:
            raise ValueError("rolled raw graph bytes differ")
        structure = inspect_structure(optimized)
    return dict(
        profile=PROFILE,
        graph=name,
        passed=True,
        stablehlo_sha256=sha256(stable.encode()).hexdigest(),
        optimized_hlo_sha256=sha256(optimized.encode()).hexdigest(),
        compiled_memory=dict(memory),
        structure=structure,
        source_seal_sha256=protocol.SEAL_SHA,
        scope="UNTIMED_RETAINED_REFERENCE_LAYER_INTEGRATION_NOT_MODEL_ADMISSION",
    )


def validate_program_report(report: Mapping[str, Any], *args: Any) -> None:
    if json.dumps(report, sort_keys=True, allow_nan=False) != json.dumps(
        inspect_program(*args), sort_keys=True, allow_nan=False
    ):
        raise ValueError("rolled graph report differs from actual replay")


def memory_budget(
    census: Mapping[str, Any], analyses: Mapping[str, Any], *, active_graph: str
) -> dict:
    if set(analyses) != set(PROGRAMS) or active_graph not in PROGRAMS:
        raise ValueError("rolled live budget requires all three programs")
    for name in PROGRAMS:
        validate_memory(name, analyses[name])
    return budget_resident_execution(
        census,
        analyses,
        active_graph=active_graph,
        resident_graphs=PROGRAMS,
        required_reserve_bytes=REQUIRED_RESERVE_BYTES,
    )
