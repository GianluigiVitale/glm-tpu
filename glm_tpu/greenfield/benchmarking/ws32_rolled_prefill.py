"""Complete rolled B128/B114 structural report, not numerical authorization.

One parsed index/live closure serves all narrow proofs. Actual source/raw pins,
all-live HBM admission and the candidate's own numerical run remain required.
Historical B17/B11 inspection is not widened.
"""

from __future__ import annotations

from hashlib import sha256
import re
from typing import Any, Sequence

from ..sharding.hlo_contract import HloInstruction, parse_hlo_module
from .ws32_batched_moe_hlo import PrefillHloIndex
from .ws32_batched_prefill import UNREGISTERED
from .ws32_decoder import _HOST_MARKERS
from .ws32_pallas_one_layer import _live_instruction_closure
from .ws32_rolled_prefill_hlo import (
    _rows,
    check_rolled_loops,
    check_rolled_commit,
    check_rolled_route_sums,
)
from .ws32_rolled_prefill_cache_hlo import check_rolled_cache_paths
from .ws32_rolled_prefill_health_hlo import check_rolled_commit_health
from .ws32_rolled_prefill_last_row_hlo import check_rolled_last_row_indices
from .ws32_rolled_prefill_collective_hlo import check_rolled_collectives
from .ws32_rolled_prefill_helper_hlo import check_rolled_helpers
from .ws32_rolled_prefill_kernel_hlo import check_rolled_kernels
from .ws32_rolled_prefill_repair_hlo import check_rolled_repair_lineage
from .ws32_rolled_prefill_operand_health_hlo import check_rolled_operand_health


CHECKS = (
    ("rolled_transition_proof", check_rolled_loops),
    ("moe_route_sum_proof", check_rolled_route_sums),
    ("atomic_commit_proof", check_rolled_commit),
    ("collective_inventory_proof", check_rolled_collectives),
    ("compiler_helper_proof", check_rolled_helpers),
    ("pallas_interface_proof", check_rolled_kernels),
    ("index_cache_storage_proof", check_rolled_cache_paths),
    ("repair_lineage_proof", check_rolled_repair_lineage),
    ("writer_health_proof", check_rolled_commit_health),
    ("last_live_row_proof", check_rolled_last_row_indices),
    ("operand_health_proof", check_rolled_operand_health),
)


def _inspect_index(
    index: PrefillHloIndex,
    *,
    block_rows: int,
    live_instructions: Sequence[HloInstruction],
) -> dict[str, Any]:
    _rows(block_rows)
    proofs = {
        name: check(index, block_rows=block_rows, live_instructions=live_instructions)
        for name, check in CHECKS
    }
    violations = [
        f"rolled {name} failed"
        for name, value in proofs.items()
        if value.get("passed") is not True
    ]
    if index.module.num_partitions != 32:
        violations.append("rolled executable requires32 partitions")
    if any(
        op.opcode in {"infeed", "outfeed", "send", "recv"}
        for op in index.module.instructions
    ):
        violations.append("rolled executable contains host transport")
    return {
        **proofs,
        "kind": "batched_prefill",
        "structural_profile": "rolled_b128_b114_v1",
        "block_rows": block_rows,
        "paired_position_sort": True,
        "rolled_prefix": True,
        "expert_panels": True,
        "sorted_local_merge": True,
        "key_tile": 512,
        "instruction_count": len(index.module.instructions),
        "live_instruction_count": len(live_instructions),
        "collective_count": len(index.module.collectives),
        "maximum_group_size": max(
            (o.maximum_group_size for o in index.module.collectives), default=0
        ),
        "profile_registered": False,
        "passed": False,
        "violations": [*violations, UNREGISTERED],
        "numerical_claim": False,
        "performance_claim": False,
        "remaining_obligations": [
            "SOURCE_AND_RAW_GRAPH_IDENTITY",
            "TAIL_PADDING_AND_HEAD_INPUT_SOURCE_NUMERICAL_CHECKS",
            "ACTUAL_ALL_LIVE_MEMORY_AND_MEASURED_PEAK",
            "OWN_SHORT_NUMERICAL_AND_REQUEST_WALL",
        ],
    }


def inspect_ws32_rolled_prefill_hlo(
    stablehlo: str,
    optimized_hlo: str,
    *,
    block_rows: int,
    expected_stablehlo_sha256: str,
    expected_optimized_hlo_sha256: str,
) -> dict[str, Any]:
    _rows(block_rows)
    for value in (expected_stablehlo_sha256, expected_optimized_hlo_sha256):
        if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("rolled HLO pin must be lowercase SHA256")
    stable_sha = sha256(stablehlo.encode()).hexdigest()
    optimized_sha = sha256(optimized_hlo.encode()).hexdigest()
    if (stable_sha, optimized_sha) != (
        expected_stablehlo_sha256,
        expected_optimized_hlo_sha256,
    ):
        raise ValueError("rolled raw HLO bytes differ from supplied pins")
    module = parse_hlo_module(optimized_hlo)
    report = _inspect_index(
        PrefillHloIndex(module),
        block_rows=block_rows,
        live_instructions=_live_instruction_closure(module.instructions),
    )
    if any(marker in stablehlo.lower() for marker in _HOST_MARKERS):
        report["violations"].insert(
            0, "rolled StableHLO contains host execution marker"
        )
    return {
        **report,
        "stablehlo_sha256": stable_sha,
        "optimized_hlo_sha256": optimized_sha,
    }
