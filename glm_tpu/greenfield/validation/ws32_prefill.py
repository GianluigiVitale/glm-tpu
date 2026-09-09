"""Mode-bound §24 prefill identity and evidence accounting (no JAX import)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from typing import Any, Mapping


SERIAL_PREFILL_MODE = "serial_teacher_forced_v1"
PREFILL_MODE = "layer_major_raw_v1"
PREFILL_GRAPH_KIND = "batched_prefill"
GRAPHS = ("prefill_chunk", "prefill_tail")


@dataclass(frozen=True, slots=True)
class BatchedPrefillPlan:
    prompt_length: int
    block_rows: int
    context_capacity: int
    mlp_window: bool = False

    def __post_init__(self) -> None:
        for value in (self.prompt_length, self.block_rows, self.context_capacity):
            if type(value) is not int:
                raise ValueError("batched prefill plan requires integer geometry")
        if type(self.mlp_window) is not bool:
            raise ValueError("batched prefill window option must be a bool")
        if not 1 <= self.block_rows <= (128 if self.mlp_window else 32):
            raise ValueError("batched prefill block exceeds explicit window mode")
        if not 0 < self.prompt_length < self.context_capacity:
            raise ValueError("batched prefill prompt must leave decode capacity")

    @property
    def split(self) -> tuple[int, int]:
        full = (self.prompt_length - 1) // self.block_rows
        return full, self.prompt_length - full * self.block_rows

    @property
    def graph_rows(self) -> tuple[tuple[str, int], ...]:
        return ((GRAPHS[0], self.block_rows), (GRAPHS[1], self.split[1]))

    def identity(self) -> dict[str, Any]:
        full, tail = self.split
        return {
            **({"mlp_window": True} if self.mlp_window else {}),
            "mode": PREFILL_MODE,
            "graph_kind": PREFILL_GRAPH_KIND,
            "block_rows": self.block_rows,
            "prompt_length": self.prompt_length,
            "context_capacity": self.context_capacity,
            "full_chunks": full,
            "tail_length": tail,
            "raw_prefill_exact_aliases": False,
            "raw_prefill_strategy_nd_dense": False,
            "host_main_rope_table": True,
            "donate_argnums": [],
            "repair_promotion": "final_healthy_commit_only",
            "head_execution": "final_live_row_only",
        }


PREFILL_MODES = (SERIAL_PREFILL_MODE, PREFILL_MODE)


def require_prefill_mode(mode: str) -> str:
    if type(mode) is not str or mode not in PREFILL_MODES:
        raise ValueError("unknown WS32 prefill mode")
    return mode


def prefill_graph_kind(graph: str, mode: str = SERIAL_PREFILL_MODE) -> str:
    require_prefill_mode(mode)
    if graph in GRAPHS:
        return PREFILL_GRAPH_KIND if mode == PREFILL_MODE else "prefill"
    if graph.startswith("prefill"):
        raise ValueError("unknown WS32 prefill graph")
    return graph


def require_batched_profile(
    mode: str,
    *,
    exact_dsa: bool,
    host_main_rope_table: bool,
    block_rows: int,
    long_context: str | None,
    adjudication_record: Any,
    adjudication_sha256: str,
) -> None:
    """Current admission scope: no inherited serial adjudication or long run.

    The first short numerical run records its own observations. Register its
    first divergence under §21 before adding a reviewed mode-specific record.
    Long contexts remain refused until short-context admission/targets exist.
    """
    require_prefill_mode(mode)
    if mode != PREFILL_MODE:
        return
    if exact_dsa is not True or host_main_rope_table is not True:
        raise ValueError("batched prefill requires exact decode and host main RoPE")
    if type(block_rows) is not int or not 1 <= block_rows <= 32:
        raise ValueError("batched prefill requires1..32 live rows")
    if long_context is not None:
        raise ValueError(
            "batched long context has no protected short-model admission yet"
        )
    if adjudication_record is not None or adjudication_sha256 != "0" * 64:
        raise ValueError(
            "batched prefill requires OWN adjudication; serial records refused"
        )


def require_fleet_prefill_mode(records: list[Mapping[str, Any]]) -> str:
    if not records:
        raise ValueError("prefill evidence has no runner records")
    modes = [
        require_prefill_mode(r.get("prefill_mode", SERIAL_PREFILL_MODE))
        for r in records
    ]
    if any(mode != modes[0] for mode in modes):
        raise ValueError("runner records disagree on prefill mode")
    return modes[0]


def validate_execution_record(
    record: Mapping[str, Any], plan: BatchedPrefillPlan
) -> None:
    """Shared worker/sealer structural check; not numerical/HLO authorization."""
    expected = {
        "identity",
        "budget_seconds",
        "cache_initialization_seconds",
        "memory_admission_seconds",
        "memory_admission",
        "input_transfer_seconds",
        "block_wall_seconds",
        "projected_total_seconds_max",
        "request_prefill_seconds",
        "final_frontier",
        "finished_healthy",
        "repaired_index_installed",
        "first_token_ready",
        "ttft_measured",
        "timing_scope",
    }
    if set(record) != expected or json.dumps(
        record["identity"], sort_keys=True, allow_nan=False
    ) != json.dumps(plan.identity(), sort_keys=True, allow_nan=False):
        raise ValueError("batched prefill execution identity/schema differs")
    from .ws32_prefill_memory import validate_prefill_memory_record

    validate_prefill_memory_record(record["memory_admission"])
    if (
        record["finished_healthy"] is not True
        or record["repaired_index_installed"] is not True
    ):
        raise ValueError("batched prefill has not committed repaired healthy state")
    if (
        type(record["final_frontier"]) is not int
        or record["final_frontier"] != plan.prompt_length
    ):
        raise ValueError("batched prefill final frontier differs")
    if type(record["first_token_ready"]) is not int or record["first_token_ready"] < 0:
        raise ValueError("batched prefill has no live first token")
    if (
        record["ttft_measured"] is not False
        or record["timing_scope"]
        != "input_ids_ready_to_prefill_token_ready_not_delivery"
    ):
        raise ValueError("batched prefill incorrectly claims delivered TTFT")
    samples = []
    for key in ("input_transfer_seconds", "block_wall_seconds"):
        if type(record[key]) is not list or len(record[key]) != plan.split[0] + 1:
            raise ValueError("batched prefill timing vector does not cover every block")
        samples.extend(record[key])
    scalars = [
        record[k]
        for k in (
            "budget_seconds",
            "cache_initialization_seconds",
            "memory_admission_seconds",
            "projected_total_seconds_max",
            "request_prefill_seconds",
        )
    ]
    if any(
        type(v) not in (int, float) or not isfinite(v) or v < 0
        for v in samples + scalars
    ):
        raise ValueError("batched prefill timings must be finite nonnegative numbers")
    if (
        record["budget_seconds"] <= 0
        or max(record["projected_total_seconds_max"], record["request_prefill_seconds"])
        > record["budget_seconds"]
    ):
        raise ValueError("batched prefill exceeds declared wall budget")
    if (
        sum(samples)
        + record["cache_initialization_seconds"]
        + record["memory_admission_seconds"]
        > record["request_prefill_seconds"] + 1e-6
    ):
        raise ValueError("batched prefill component wall exceeds request wall")
