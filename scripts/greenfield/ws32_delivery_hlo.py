"""Source-bound long-prefill structural checks using the original HLO tools.

No numerical dispatch authority. Kernel math, bounded update values/addresses,
and health semantics rest on the frozen source and retained semantic tests;
actual load/state/DSA, all-live HBM, peak counters and task results remain required.
This does not claim the short profile's twelve symbolic proofs at new capacity.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Sequence

from glm_tpu.optimized.hlo_contract import HloInstruction, parse_hlo_module
from glm_tpu.greenfield.benchmarking.ws32_batched_moe_hlo import PrefillHloIndex, _check_moe_route_sums
from glm_tpu.greenfield.benchmarking.ws32_decoder import _HOST_MARKERS
from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import _live_instruction_closure
from glm_tpu.greenfield.benchmarking.ws32_prefill_fixed_loops import fixed_loop_bodies
from glm_tpu.greenfield.benchmarking.ws32_long_prefill_cache_hlo import prove_large_cache_storage
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_collective_hlo import _check_rolled_collectives
from glm_tpu.greenfield.benchmarking.ws32_rolled_prefill_kernel_hlo import _check_rolled_kernels
from scripts.greenfield import ws32_delivery_programs as programs

PROFILE = "ws32_delivery_long_phase_v1"
FRESH_OPTIMIZED_MARKER = "0" * 64
FRESH_POLICY = "FROZEN_SOURCE_RAW_AND_FRESH_ACTUAL_STRUCTURAL_CHECKS"


def optimized_identity(optimized: str, expected: str) -> tuple[str, str]:
    """Bind actual compiler bytes, without normalizing away caller metadata.

    The explicit long profile may request fresh optimized inspection, as the
    short paired profile already does. Source and registered RAW are checked
    by the enclosing inspector; a digest alone is NOT graph authorization.
    Literal callers still require their exact original digest.
    """
    if (type(expected) is not str or re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or not isinstance(optimized, str) or not optimized.strip()):
        raise ValueError("long optimized pin/text must be nonempty SHA-bound evidence")
    actual = sha256(optimized.encode()).hexdigest()
    if expected != FRESH_OPTIMIZED_MARKER and actual != expected:
        raise ValueError("long original optimized graph bytes differ")
    return actual, FRESH_POLICY if expected == FRESH_OPTIMIZED_MARKER else "LITERAL_OPTIMIZED_SHA256"


def check_index(
    index: PrefillHloIndex, *, context_capacity: int, block_rows: int,
    live_instructions: Sequence[HloInstruction],
    nucleus_head: bool = False,
) -> dict[str, Any]:
    """Structural component only; outer inspector must authenticate source/RAW."""
    if type(nucleus_head) is not bool:
        raise ValueError("nucleus head must be an explicit bool")
    allowed = (((262656, 128), (262656, 114), (166912, 128), (166912, 114)) if nucleus_head else
               ((131072, 128), (131072, 114), (262656, 128)))
    if (type(block_rows) is not int or type(context_capacity) is not int
            or (context_capacity, block_rows) not in allowed):
        raise ValueError("unregistered long structural geometry")
    if index.module.num_partitions != 32:
        raise ValueError("long executable requires32 partitions")
    if any(op.opcode in {"infeed", "outfeed", "send", "recv"}
           or any(marker in op.raw_line.lower() for marker in _HOST_MARKERS)
           for op in index.module.instructions):
        raise ValueError("long executable contains host transport")
    bodies = fixed_loop_bodies(
        index, live_instructions,
        loop_suffix="greenfield_ws32_prefill_rolled_prefix/while",
        expected_layers=tuple(range(78)),
    )
    cache, approved = prove_large_cache_storage(index, context_capacity=context_capacity,
                                               native_benchmark=nucleus_head)
    common = dict(block_rows=block_rows, live_instructions=live_instructions,
                  canonical_dense=True, prefix_bodies=bodies)
    collectives = _check_rolled_collectives(index, **common, nucleus_head=nucleus_head)
    kernels = _check_rolled_kernels(
        index, **common,
        large_float_guard=lambda op, shape: op.index in approved,
    )
    moe = _check_moe_route_sums(index, block_rows=block_rows,
                              live_instructions=live_instructions, router_bias_tuple=True)
    checks = dict(large_cache_storage=cache, collectives=collectives,
                  kernels=kernels, fp32_moe_route_sums=moe)
    return dict(
        passed=all(value.get("passed") is True for value in checks.values()),
        checks=checks, prefix_loop_count=len(bodies),
        context_capacity=context_capacity, block_rows=block_rows,
        instruction_count=len(index.module.instructions),
        live_instruction_count=len(live_instructions),
        maximum_group_size=max(op.maximum_group_size for op in index.module.collectives),
        scope="LONG_LOCAL_SCHEDULE_KERNEL_INTERFACE_AND_FULL_CACHE_BASE_LINEAGE",
        not_proven=["OPAQUE_KERNEL_ARITHMETIC", "BOUNDED_UPDATE_VALUES_AND_ADDRESSES",
                    "COMPLETE_HEALTH_AND_COMMIT_IMPLICATION", "RUNTIME_HBM", "TASK_QUALITY"],
    )


def inspect_hlo(
    stablehlo: str, optimized_hlo: str, *, repo: Path, context_label: str,
    role: str, expected_stablehlo_sha256: str, expected_optimized_sha256: str,
) -> dict[str, Any]:
    """Bind actual bytes and fixed current source before parsing the long graph."""
    plan = programs.long_plan(context_label)
    pins = programs.raw_registration(context_label)
    if role not in pins:
        raise ValueError("unregistered long graph role")
    if expected_stablehlo_sha256 != pins[role][1]:
        raise ValueError("long StableHLO pin differs from fixed registration")
    programs.require_source(repo)
    raw = stablehlo.encode()
    raw_sha = sha256(raw).hexdigest()
    if (len(raw), raw_sha) != pins[role]:
        raise ValueError("long original graph bytes differ")
    optimized_sha, policy = optimized_identity(optimized_hlo, expected_optimized_sha256)
    if any(marker in stablehlo.lower() for marker in _HOST_MARKERS):
        raise ValueError("long StableHLO contains host execution marker")
    module = parse_hlo_module(optimized_hlo)
    report = check_index(
        PrefillHloIndex(module), context_capacity=plan.context_capacity,
        block_rows=dict(plan.graph_rows)[role],
        live_instructions=_live_instruction_closure(module.instructions),
    )
    # Producer JSON and independently computed sealer reports must have the
    # same representation, including nested tuples and integer-keyed mappings.
    return json.loads(json.dumps({
        **report, "schema": "ws32_delivery_long_structural_v1",
        "stablehlo_sha256": raw_sha, "optimized_hlo_sha256": optimized_sha,
        **({"optimized_identity_policy": policy} if policy == FRESH_POLICY else {}),
        "state_ownership_contract": programs.state_ownership(context_label),
        "dispatch_authorized": False, "numerical_claim": False, "performance_claim": False,
    }, allow_nan=False))
