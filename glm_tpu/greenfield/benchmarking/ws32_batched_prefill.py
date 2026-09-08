"""First-acquisition inventory for the actual §24 complete prefill graphs.

This deliberately cannot authorize execution. Actual graphs now support narrow
route, commit, collective, helper, kernel, cache/repair and selected health proofs;
memory admission and numerical wiring remain open. Reuse the acquired originals
instead of guessing a broad allowlist from single-layer compiler products.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import re
from typing import Any

from .ws32_decoder import _group_family, _exact_add_reducer, _HOST_MARKERS
from .ws32_pallas_one_layer import _computation_base, _live_instruction_closure
from ..sharding.hlo_contract import parse_hlo_module
from .ws32_batched_moe_hlo import PrefillHloIndex, check_batched_moe_route_sums
from .ws32_batched_commit_hlo import check_batched_commit
from .ws32_batched_collective_hlo import check_batched_collectives
from .ws32_batched_helper_hlo import check_batched_helpers
from .ws32_batched_kernel_hlo import check_batched_kernels
from .ws32_batched_cache_hlo import check_batched_index_cache_stacks
from .ws32_batched_repair_hlo import check_batched_repair_lineage
from .ws32_batched_health_hlo import check_batched_writer_health
from .ws32_batched_operand_health_hlo import check_batched_operand_health


UNREGISTERED = "batched prefill production HLO/allocation profile is not registered"


def inspect_ws32_batched_prefill_hlo(
    stablehlo: str,
    optimized_hlo: str,
    *,
    block_rows: int,
    expected_stablehlo_sha256: str,
    expected_optimized_hlo_sha256: str,
    paired_position_sort: bool = False,
) -> dict[str, Any]:
    """Acquire, never approve, real graph payloads without embedding HLO blobs."""
    if type(block_rows) is not int or not 1 <= block_rows <= 32:
        raise ValueError("batched HLO requires1..32 live rows")
    if type(paired_position_sort) is not bool:
        raise ValueError("paired position sort must be a static bool")
    for value in (expected_stablehlo_sha256, expected_optimized_hlo_sha256):
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("batched HLO pin must be lowercase SHA256")
    stable_sha = sha256(stablehlo.encode()).hexdigest()
    optimized_sha = sha256(optimized_hlo.encode()).hexdigest()
    module = parse_hlo_module(optimized_hlo)
    computations = defaultdict(list)
    for op in module.instructions:
        computations[op.computation.split(" ", 1)[0].lstrip("%")].append(op)
    computations = {name: tuple(ops) for name, ops in computations.items()}
    live = _live_instruction_closure(module.instructions)
    live_keys = {(op.computation, op.name) for op in live}
    payloads, calls = Counter(), Counter()
    exceptional_reducers = []
    violations = []
    index = PrefillHloIndex(module)
    route_proof = check_batched_moe_route_sums(
        index, block_rows=block_rows, live_instructions=live
    )
    if not route_proof["passed"]:
        violations.append("batched per-layer FP32 MoE route-sum proof failed")
    commit_proof = check_batched_commit(
        index, block_rows=block_rows, live_instructions=live
    )
    if not commit_proof["passed"]:
        violations.append("batched health/atomic commit proof failed")
    collective_proof = (
        check_batched_collectives(index, block_rows=block_rows, live_instructions=live)
        if block_rows in (11, 17)
        else {"passed": False, "error": "unregistered collective row count"}
    )
    if not collective_proof["passed"]:
        violations.append("batched exact physical collective inventory failed")
    helper_proof = (
        check_batched_helpers(
            index,
            block_rows=block_rows,
            live_instructions=live,
            paired_position_sort=paired_position_sort,
        )
        if block_rows in (11, 17)
        else {"passed": False, "error": "unregistered helper row count"}
    )
    if not helper_proof["passed"]:
        violations.append("batched compiler helper structure failed")
    kernel_proof = (
        check_batched_kernels(index, block_rows=block_rows, live_instructions=live)
        if block_rows in (11, 17)
        else {"passed": False, "error": "unregistered kernel row count"}
    )
    if not kernel_proof["passed"]:
        violations.append("batched Pallas interface/schedule failed")
    cache_proof = (
        check_batched_index_cache_stacks(index, block_rows=block_rows)
        if block_rows in (11, 17)
        else {"passed": False, "error": "unregistered cache row count"}
    )
    if not cache_proof["passed"]:
        violations.append("batched index-cache storage ownership failed")
    repair_proof = (
        check_batched_repair_lineage(index, block_rows=block_rows)
        if block_rows in (11, 17)
        else {"passed": False, "error": "unregistered repair row count"}
    )
    if not repair_proof["passed"]:
        violations.append("batched repair input/writer provenance failed")
    writer_health_proof = (
        check_batched_writer_health(index, block_rows=block_rows)
        if block_rows in (11, 17)
        else {"passed": False, "error": "unregistered writer health row count"}
    )
    if not writer_health_proof["passed"]:
        violations.append("batched cache writers do not all gate commit")
    operand_health_proof = (
        check_batched_operand_health(index, block_rows=block_rows)
        if block_rows in (11, 17)
        else {"passed": False, "error": "unregistered operand health row count"}
    )
    if not operand_health_proof["passed"]:
        violations.append("batched operand finite/grouped validity guards failed")
    if stable_sha != expected_stablehlo_sha256:
        violations.append("StableHLO identity drifted")
    if optimized_sha != expected_optimized_hlo_sha256:
        violations.append("optimized HLO identity drifted")
    if module.num_partitions != 32:
        violations.append("batched executable is not partitioned32 ways")
    layers = set()
    for op in live:
        layers.update(
            int(v)
            for v in re.findall(
                r"greenfield_ws32_batched_prefill/layer_(\d+)(?:/|$)", op.op_name or ""
            )
        )
    if layers != set(range(78)):
        violations.append("batched HLO lacks complete78-layer live scope inventory")
    if any(marker in stablehlo.lower() for marker in _HOST_MARKERS) or any(
        op.opcode in {"infeed", "outfeed", "send", "recv"} for op in module.instructions
    ):
        violations.append("batched HLO contains host transport/execution")
    for op in module.collectives:
        family = _group_family(op)
        if (
            family is None
            or not op.use_global_device_ids
            or op.raw_opcode not in {"all-reduce", "all-gather"}
        ):
            violations.append(f"batched collective group/opcode drifted: {op.name}")
        if (op.computation, op.name) not in live_keys:
            violations.append(f"batched collective is dead: {op.name}")
        shapes = (
            op.operand_shapes if op.raw_opcode == "all-reduce" else op.result_shapes
        )
        for shape in shapes:
            payloads[
                (op.raw_opcode, family or "unknown", shape.dtype, shape.dimensions)
            ] += 1
        if op.raw_opcode == "all-reduce" and not _exact_add_reducer(
            op,
            module_instructions=module.instructions,
            instructions_by_computation=computations,
        ):
            # Includes the two intended health MINs. They are NOT authorized
            # by their names: subsequent admission must prove scalar reducer
            # AND feature→expert→actual commit-predicate lineage.
            exceptional_reducers.append(
                {
                    "computation": _computation_base(op.computation),
                    "name": op.name,
                    "scope": op.op_name,
                    "group_family": family,
                    "operand_shapes": [s.to_dict() for s in op.operand_shapes],
                }
            )
    for op in module.instructions:
        if op.opcode == "custom-call":
            target = re.search(r'custom_call_target="([^"]+)"', op.raw_line)
            calls[
                (
                    target[1] if target else "missing",
                    tuple((s.dtype, s.dimensions) for s in op.result_shapes),
                )
            ] += 1
    # Nothing in this diagnostic can make the numerical graph pass, including
    # matching hashes, valid groups, familiar labels or successful CPU tests.
    violations.append(UNREGISTERED)
    return {
        **({"paired_position_sort": True} if paired_position_sort else {}),
        "kind": "batched_prefill",
        "block_rows": block_rows,
        "stablehlo_sha256": stable_sha,
        "optimized_hlo_sha256": optimized_sha,
        "instruction_count": len(module.instructions),
        "live_instruction_count": len(live),
        "collective_count": len(module.collectives),
        "maximum_group_size": max(
            (op.maximum_group_size for op in module.collectives), default=0
        ),
        "live_layer_ids": sorted(layers),
        "collective_payloads": [
            dict(opcode=k[0], family=k[1], dtype=k[2], dimensions=list(k[3]), count=v)
            for k, v in sorted(payloads.items())
        ],
        "custom_call_inventory": [
            dict(
                target=k[0],
                results=[dict(dtype=d, dimensions=list(s)) for d, s in k[1]],
                count=v,
            )
            for k, v in sorted(calls.items())
        ],
        "non_add_reducers_requiring_lineage": exceptional_reducers,
        "profile_registered": False,
        "moe_route_sum_proof": route_proof,
        "atomic_commit_proof": commit_proof,
        "collective_inventory_proof": collective_proof,
        "compiler_helper_proof": helper_proof,
        "pallas_interface_proof": kernel_proof,
        "index_cache_storage_proof": cache_proof,
        "repair_lineage_proof": repair_proof,
        "writer_health_proof": writer_health_proof,
        "operand_health_proof": operand_health_proof,
        "passed": False,
        "violations": violations,
        "performance_claim": False,
    }
