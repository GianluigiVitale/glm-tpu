#!/usr/bin/env python3
"""Bind real runtime weights and compile/run the 78-layer PP8 decoder body."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking.one_layer import (  # noqa: E402
    TensorTolerance,
    compare_bounded_tensor,
)
from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    FeatureRuntimeCheckpointLoadExpectation,
    RuntimeCheckpointLoadExpectation,
    load_runtime_checkpoint,
    verify_feature_runtime_packed_checkpoint,
    verify_runtime_packed_checkpoint,
)
from glm_tpu.greenfield.model import (  # noqa: E402
    build_decoder_state_layout,
    build_pipeline_schedule,
)
from glm_tpu.greenfield.runtime import (  # noqa: E402
    build_decoder_step_program,
    build_teacher_forced_prefill_program,
    validate_decoder_step_hlo,
    validate_teacher_forced_prefill_hlo,
)
from glm_tpu.greenfield.topology import (  # noqa: E402
    discover_physical_topology,
    validate_target_v4_64,
)
from glm_tpu.greenfield.validation import (  # noqa: E402
    inspect_short_context_dsa_oracle,
    inspect_short_context_oracle,
)
from scripts.greenfield.pack_feature_runtime_checkpoint import (  # noqa: E402
    _build_context as _build_feature_context,
)
from scripts.greenfield.pack_runtime_checkpoint import (  # noqa: E402
    _build_context as _build_reference_context,
)


DSA_CROSS_BACKEND_SCORE_TOLERANCE = TensorTolerance(
    max_abs=0.125,
    p99_abs=0.03125,
    mean_abs=0.01,
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write one diagnostic tensor bundle without exposing a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_stats(device: Any) -> dict[str, int] | None:
    value = device.memory_stats()
    if value is None:
        return None
    return {
        str(name): int(number)
        for name, number in value.items()
        if isinstance(number, int) and not isinstance(number, bool)
    }


def _fleet_digest(
    multihost_utils: Any,
    digest_hex: str,
    *,
    num_processes: int,
) -> list[str]:
    digest = np.frombuffer(bytes.fromhex(digest_hex), dtype=np.uint8)
    fleet = np.asarray(multihost_utils.process_allgather(digest)).reshape(
        num_processes,
        len(digest),
    )
    values = [row.tobytes().hex() for row in fleet]
    if len(set(values)) != 1:
        raise RuntimeError(f"hosts disagree on optimized HLO: {values}")
    return values


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _materialize_global_array(
    jax: Any,
    multihost_utils: Any,
    value: Any,
) -> np.ndarray:
    """Materialize a global array without fetching non-addressable shards."""
    if value.is_fully_addressable:
        materialized = jax.device_get(value)
    else:
        materialized = multihost_utils.process_allgather(value, tiled=True)
    host = np.asarray(materialized)
    if host.shape != tuple(value.shape):
        raise RuntimeError(
            "global array gather changed shape: "
            f"expected={tuple(value.shape)} actual={host.shape}"
        )
    return host


def _validate_completed_step_selected_states(
    active_metadata: np.ndarray,
    *,
    selected_width: int,
    count_index: int,
    next_position: int,
    next_context_length: int,
) -> dict[str, Any]:
    """Validate DSA state against the just-completed, not next, position."""

    if selected_width <= 0 or count_index < selected_width:
        raise ValueError("selected-state geometry is invalid")
    if active_metadata.ndim != 2 or active_metadata.shape[1] <= count_index:
        raise ValueError("active selected-state metadata shape is invalid")
    position_context_aligned = (
        next_position >= 0 and next_context_length == next_position + 1
    )
    expected_valid_count = min(next_position, selected_width)
    rows_valid = []
    for row in active_metadata:
        count = int(row[count_index])
        positions = row[:selected_width]
        valid = positions[:count]
        tail = positions[count:]
        rows_valid.append(
            position_context_aligned
            and count == expected_valid_count
            and bool(np.all(valid >= 0))
            # The output position is the exclusive bound for the DSA state
            # produced during this step. The incremented context length also
            # includes the slot to be consumed by the following step.
            and bool(np.all(valid < next_position))
            and len(set(valid.tolist())) == count
            and bool(np.all(tail == -1))
        )
    return {
        "expected_valid_count": expected_valid_count,
        "next_context_length": next_context_length,
        "next_position": next_position,
        "position_context_aligned": position_context_aligned,
        "rows_valid": rows_valid,
    }


def _load_short_context_oracle(
    oracle_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Load exact prompt/output IDs only after the sealed oracle passes."""

    from safetensors import safe_open

    manifest = inspect_short_context_oracle(oracle_dir)
    if manifest["manifest_sha256"] != expected_manifest_sha256:
        raise RuntimeError(
            "short-context oracle manifest differs from the protected pin"
        )
    tensor_path = oracle_dir / manifest["files"]["tokens"]["filename"]
    with safe_open(tensor_path, framework="np") as handle:
        prompt = np.asarray(
            handle.get_tensor("prompt_token_ids"), dtype=np.int32
        ).copy()
        generated = np.asarray(
            handle.get_tensor("generated_token_ids"), dtype=np.int32
        ).copy()
    return manifest, prompt, generated


def _load_short_context_dsa_oracle(
    oracle_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load exact all-event DSA tensors only after sealed inspection."""

    from safetensors import safe_open

    manifest = inspect_short_context_dsa_oracle(oracle_dir)
    if manifest["manifest_sha256"] != expected_manifest_sha256:
        raise RuntimeError(
            "short-context DSA oracle manifest differs from the protected pin"
        )
    tensor_path = oracle_dir / manifest["files"]["tensors"]["filename"]
    expected_names = {
        "decode_positions",
        "producer_layer_ids",
        "selected_positions",
        "selected_scores",
        "valid_counts",
    }
    with safe_open(tensor_path, framework="np") as handle:
        if set(handle.keys()) != expected_names:
            raise RuntimeError("short-context DSA oracle tensor keys drifted")
        tensors = {
            name: np.asarray(handle.get_tensor(name)).copy()
            for name in expected_names
        }
    return manifest, tensors


def _stage_dsa_producer_layer_ids(schedule: Any) -> tuple[tuple[int, ...], ...]:
    """Return full-indexer producers in executable stage/slot order."""

    return tuple(
        tuple(
            int(layer.layer_id)
            for layer in stage.layers
            if layer.indexer_kind == "full"
        )
        for stage in schedule.stages
    )


def _float32_topk_descending_key(values: np.ndarray) -> np.ndarray:
    """Unsigned key whose ascending order is ``lax.top_k`` FP32 order."""

    scores = np.ascontiguousarray(values, dtype=np.float32)
    bits = scores.view(np.uint32)
    ascending = np.where(
        (bits >> np.uint32(31)) == np.uint32(0),
        bits ^ np.uint32(0x80000000),
        ~bits,
    ).astype(np.uint32, copy=False)
    return np.bitwise_not(ascending)


def _validate_dsa_observation_step(
    observation: np.ndarray,
    *,
    groups: tuple[tuple[int, ...], ...],
    stage_producer_layer_ids: tuple[tuple[int, ...], ...],
    selected_width: int,
    expected_positions: np.ndarray,
    expected_scores: np.ndarray,
    expected_valid_counts: np.ndarray,
    expected_producer_layer_ids: np.ndarray,
    decode_position: int,
) -> dict[str, Any]:
    """Gate exact device sets/ties and diagnose position-aligned legacy scores.

    The numerical contract deliberately does not require total score-rank
    identity across independent TPU programs. It does require an exact
    selected set, an exact ``-1``/``-inf`` tail, canonical lowest-position
    ties for the scores produced by this executing greenfield program. The
    accepted Gate C score bounds apply only when both programs consume the
    same captured hidden input; after full-network reduction reassociation,
    position-aligned legacy scores are retained as a non-gating diagnostic.
    """

    observed = np.asarray(observation)
    expected_positions = np.asarray(expected_positions)
    expected_scores = np.asarray(expected_scores)
    expected_valid_counts = np.asarray(expected_valid_counts)
    expected_producer_layer_ids = np.asarray(expected_producer_layer_ids)
    if selected_width <= 0:
        raise ValueError("DSA selected width must be positive")
    if len(groups) != len(stage_producer_layer_ids) or not groups:
        raise ValueError("DSA stage/group mapping is invalid")
    flat_ranks = tuple(rank for group in groups for rank in group)
    if sorted(flat_ranks) != list(range(len(flat_ranks))):
        raise ValueError("DSA groups must cover every rank exactly once")
    maximum_slots = max(len(values) for values in stage_producer_layer_ids)
    observation_width = 2 * selected_width + 2
    expected_shape = (len(flat_ranks), maximum_slots, observation_width)
    if observed.shape != expected_shape or observed.dtype != np.dtype(np.int32):
        raise ValueError(
            "DSA observer tensor contract drifted: "
            f"expected={expected_shape}/int32 "
            f"observed={observed.shape}/{observed.dtype}"
        )
    flat_producers = tuple(
        producer
        for stage_values in stage_producer_layer_ids
        for producer in stage_values
    )
    event_count = len(flat_producers)
    if (
        expected_positions.shape != (event_count, selected_width)
        or expected_scores.shape != (event_count, selected_width)
        or expected_valid_counts.shape != (event_count,)
        or expected_producer_layer_ids.shape != (event_count,)
    ):
        raise ValueError("DSA oracle step tensor contract drifted")
    if expected_producer_layer_ids.tolist() != list(flat_producers):
        raise ValueError(
            "DSA oracle producer mapping differs from the executable schedule"
        )

    lane_mismatch_stages: list[int] = []
    padded_slot_mismatches: list[dict[str, int]] = []
    producer_mismatches: list[dict[str, int]] = []
    count_mismatches: list[dict[str, int]] = []
    selected_set_mismatches: list[dict[str, Any]] = []
    tail_mismatches: list[dict[str, Any]] = []
    score_contract_mismatches: list[dict[str, Any]] = []
    legacy_order_mismatch_count = 0
    first_legacy_order_mismatch = None
    position_aligned_observed_scores: list[np.ndarray] = []
    position_aligned_expected_scores: list[np.ndarray] = []
    expected_live_score_count = 0
    event = 0
    for stage, (group, stage_producers) in enumerate(
        zip(groups, stage_producer_layer_ids, strict=True)
    ):
        lanes = observed[list(group)]
        if not np.all(lanes == lanes[0]):
            lane_mismatch_stages.append(stage)
        canonical = lanes[0]
        for slot, producer in enumerate(stage_producers):
            row = canonical[slot]
            observed_positions = row[:selected_width]
            observed_scores = np.ascontiguousarray(
                row[selected_width : 2 * selected_width]
            ).view(np.float32)
            observed_count = int(row[2 * selected_width])
            observed_producer = int(row[2 * selected_width + 1])
            expected_count = int(expected_valid_counts[event])
            if observed_producer != producer:
                producer_mismatches.append(
                    {
                        "event_index": event,
                        "expected": producer,
                        "observed": observed_producer,
                    }
                )
            if observed_count != expected_count:
                count_mismatches.append(
                    {
                        "event_index": event,
                        "expected": expected_count,
                        "observed": observed_count,
                    }
                )
            order_mismatches = np.flatnonzero(
                observed_positions != expected_positions[event]
            )
            legacy_order_mismatch_count += int(order_mismatches.size)
            if (
                order_mismatches.size
                and first_legacy_order_mismatch is None
            ):
                offset = int(order_mismatches[0])
                first_legacy_order_mismatch = {
                    "event_index": event,
                    "expected": int(expected_positions[event, offset]),
                    "observed": int(observed_positions[offset]),
                    "producer_layer_id": producer,
                    "selected_offset": offset,
                }

            safe_observed_count = min(max(observed_count, 0), selected_width)
            safe_expected_count = min(max(expected_count, 0), selected_width)
            observed_live = observed_positions[:safe_observed_count]
            observed_live_scores = observed_scores[:safe_observed_count]
            expected_live = expected_positions[event, :safe_expected_count]
            expected_live_scores = expected_scores[event, :safe_expected_count]
            expected_live_score_count += int(safe_expected_count)
            expected_only = np.setdiff1d(expected_live, observed_live)
            observed_only = np.setdiff1d(observed_live, expected_live)
            if (
                observed_count != expected_count
                or expected_only.size
                or observed_only.size
            ):
                selected_set_mismatches.append(
                    {
                        "event_index": event,
                        "expected_only_count": int(expected_only.size),
                        "expected_only_first": expected_only[:8].tolist(),
                        "observed_only_count": int(observed_only.size),
                        "observed_only_first": observed_only[:8].tolist(),
                        "producer_layer_id": producer,
                    }
                )

            observed_tail = observed_positions[safe_observed_count:]
            observed_score_tail = observed_scores[safe_observed_count:]
            if (
                observed_count < 0
                or observed_count > selected_width
                or np.any(observed_tail != -1)
                or np.any(~np.isneginf(observed_score_tail))
            ):
                tail_mismatches.append(
                    {
                        "event_index": event,
                        "position_tail_mismatch_count": int(
                            np.count_nonzero(observed_tail != -1)
                        ),
                        "producer_layer_id": producer,
                        "score_tail_mismatch_count": int(
                            np.count_nonzero(~np.isneginf(observed_score_tail))
                        ),
                    }
                )

            invalid_position_count = int(
                np.count_nonzero(
                    (observed_live < 0)
                    | (observed_live > int(decode_position))
                )
            )
            duplicate_position_count = int(
                observed_live.size - np.unique(observed_live).size
            )
            nonfinite_score_count = int(
                np.count_nonzero(~np.isfinite(observed_live_scores))
            )
            canonical_order = np.lexsort(
                (
                    observed_live.astype(np.int64, copy=False),
                    _float32_topk_descending_key(observed_live_scores),
                )
            )
            order_is_canonical = bool(
                np.array_equal(
                    canonical_order,
                    np.arange(observed_live.size, dtype=np.int64),
                )
            )
            if (
                invalid_position_count
                or duplicate_position_count
                or nonfinite_score_count
                or not order_is_canonical
            ):
                first_wrong_offset = None
                if not order_is_canonical and canonical_order.size:
                    wrong = np.flatnonzero(
                        canonical_order
                        != np.arange(canonical_order.size, dtype=np.int64)
                    )
                    if wrong.size:
                        first_wrong_offset = int(wrong[0])
                score_contract_mismatches.append(
                    {
                        "duplicate_position_count": duplicate_position_count,
                        "event_index": event,
                        "first_noncanonical_offset": first_wrong_offset,
                        "invalid_position_count": invalid_position_count,
                        "nonfinite_score_count": nonfinite_score_count,
                        "producer_layer_id": producer,
                    }
                )

            if (
                observed_count == expected_count
                and not expected_only.size
                and not observed_only.size
                and safe_observed_count
            ):
                observed_order = np.argsort(observed_live, kind="stable")
                expected_order = np.argsort(expected_live, kind="stable")
                position_aligned_observed_scores.append(
                    observed_live_scores[observed_order]
                )
                position_aligned_expected_scores.append(
                    expected_live_scores[expected_order]
                )
            event += 1
        for slot in range(len(stage_producers), maximum_slots):
            if np.any(canonical[slot] != -1):
                padded_slot_mismatches.append({"slot": slot, "stage": stage})

    if position_aligned_observed_scores:
        aligned_observed = np.concatenate(position_aligned_observed_scores)
        aligned_expected = np.concatenate(position_aligned_expected_scores)
        score_comparison = compare_bounded_tensor(
            aligned_observed,
            aligned_expected,
            DSA_CROSS_BACKEND_SCORE_TOLERANCE,
        )
    else:
        score_comparison = {
            "error": None,
            "observed_range": None,
            "passed": False,
            "reason": "no exact position-aligned live scores",
            "reference_range": None,
            "shape": [0],
            "tolerance": DSA_CROSS_BACKEND_SCORE_TOLERANCE.to_dict(),
        }
    aligned_position_count = int(
        sum(values.size for values in position_aligned_observed_scores)
    )
    score_comparison["aligned_position_count"] = aligned_position_count
    score_comparison["expected_position_count"] = expected_live_score_count
    score_comparison["coverage_complete"] = bool(
        aligned_position_count == expected_live_score_count
    )
    score_comparison["passed"] = bool(
        score_comparison["passed"] and score_comparison["coverage_complete"]
    )
    passed = not any(
        (
            lane_mismatch_stages,
            padded_slot_mismatches,
            producer_mismatches,
            count_mismatches,
            selected_set_mismatches,
            tail_mismatches,
            score_contract_mismatches,
        )
    )
    return {
        "count_mismatches": count_mismatches,
        "decode_position": int(decode_position),
        "event_count": event_count,
        "actual_device_score_order_and_ties": not score_contract_mismatches,
        "exact_selected_set_and_tail": not (
            selected_set_mismatches or tail_mismatches
        ),
        "first_legacy_order_mismatch": first_legacy_order_mismatch,
        "lane_mismatch_stages": lane_mismatch_stages,
        "legacy_order_mismatch_count": legacy_order_mismatch_count,
        "legacy_total_order_match": legacy_order_mismatch_count == 0,
        "padded_slot_mismatches": padded_slot_mismatches,
        "passed": passed,
        "producer_mismatches": producer_mismatches,
        "score_contract_mismatches": score_contract_mismatches,
        "legacy_score_bounded_comparison": score_comparison,
        "selected_set_mismatches": selected_set_mismatches,
        "tail_mismatches": tail_mismatches,
    }


def _validate_token_observation_step(
    observation: np.ndarray,
    *,
    groups: tuple[tuple[int, ...], ...],
    candidate_width: int,
    expected_token_id: int,
    observed_token_id: int,
    vocab_size: int,
) -> dict[str, Any]:
    """Validate compact global logit candidates from the observer executable."""

    observed = np.asarray(observation)
    ranks = tuple(rank for group in groups for rank in group)
    expected_shape = (len(ranks), 2 * candidate_width)
    if (
        candidate_width <= 1
        or observed.shape != expected_shape
        or observed.dtype != np.dtype(np.int32)
        or sorted(ranks) != list(range(len(ranks)))
    ):
        raise ValueError(
            "token observer tensor contract drifted: "
            f"expected={expected_shape}/int32 "
            f"observed={observed.shape}/{observed.dtype}"
        )
    producer_group = groups[-1]
    lanes = observed[list(producer_group)]
    lane_replication = bool(np.all(lanes == lanes[0]))
    inactive_ranks = sorted(set(ranks) - set(producer_group))
    inactive_rows_are_sentinel = bool(
        np.all(observed[inactive_ranks] == np.int32(-1))
    )
    row = lanes[0]
    candidate_ids = row[:candidate_width]
    candidate_scores = np.ascontiguousarray(
        row[candidate_width:]
    ).view(np.float32)
    ids_valid = bool(
        np.all((candidate_ids >= 0) & (candidate_ids < vocab_size))
        and np.unique(candidate_ids).size == candidate_width
    )
    scores_finite = bool(np.all(np.isfinite(candidate_scores)))
    canonical_order = np.lexsort(
        (
            candidate_ids.astype(np.int64, copy=False),
            _float32_topk_descending_key(candidate_scores),
        )
    )
    order_and_ties_valid = bool(
        np.array_equal(
            canonical_order,
            np.arange(candidate_width, dtype=np.int64),
        )
    )
    expected_offsets = np.flatnonzero(candidate_ids == expected_token_id)
    expected_offset = (
        int(expected_offsets[0]) if expected_offsets.size else None
    )
    expected_rank = (
        expected_offset + 1 if expected_offset is not None else None
    )
    expected_score = (
        float(candidate_scores[expected_offset])
        if expected_offset is not None
        else None
    )
    winner_matches_output = bool(
        int(candidate_ids[0]) == int(observed_token_id)
    )
    return {
        "candidate_ids": candidate_ids.tolist(),
        "candidate_scores": candidate_scores.tolist(),
        "candidate_width": candidate_width,
        "expected_in_candidate_set": expected_offset is not None,
        "expected_offset": expected_offset,
        "expected_rank": expected_rank,
        "expected_score": expected_score,
        "expected_token_id": int(expected_token_id),
        "inactive_rows_are_sentinel": inactive_rows_are_sentinel,
        "ids_valid": ids_valid,
        "lane_replication": lane_replication,
        "observed_token_id": int(observed_token_id),
        "order_and_ties_valid": order_and_ties_valid,
        "passed": bool(
            lane_replication
            and inactive_rows_are_sentinel
            and ids_valid
            and scores_finite
            and order_and_ties_valid
            and winner_matches_output
        ),
        "scores_finite": scores_finite,
        "top1_top2_margin": float(
            candidate_scores[0] - candidate_scores[1]
        ),
        "top1_expected_margin": (
            float(candidate_scores[0] - expected_score)
            if expected_score is not None
            else None
        ),
        "winner_matches_output": winner_matches_output,
    }


def _observer_hlo_isolation_contract(
    optimized_hlo: str,
    *,
    production_contract: dict[str, Any],
    observer_contract: dict[str, Any],
) -> dict[str, Any]:
    """Prove the diagnostic executable is callback-free and non-donating."""

    lowered = optimized_hlo.lower()
    callback_markers = tuple(
        marker
        for marker in (
            "host_callback",
            "outside_compilation",
            "xla_ffi_python_cpu_callback",
            "xla_python_cpu_callback",
        )
        if marker in lowered
    )
    header = optimized_hlo.splitlines()[0] if optimized_hlo else ""
    input_output_alias_present = bool(
        re.search(r"\binput_output_alias\s*=", header)
    )
    compared_fields = (
        "collective_count",
        "collective_counts",
        "all_reduce_component_count",
        "all_reduce_arity_counts",
    )
    scalar_collective_contract_matches = all(
        observer_contract.get(name) == production_contract.get(name)
        for name in compared_fields
    )

    def without_token_exchange_shapes(
        contract: dict[str, Any],
    ) -> dict[str, int]:
        counts = {
            str(name): int(count)
            for name, count in contract.get(
                "all_reduce_result_shape_counts", {}
            ).items()
        }
        token_contract = contract.get(
            "complete_token_collective_contract", {}
        )
        for exchange_name in ("score_exchange", "token_id_exchange"):
            exchanges = token_contract.get(exchange_name, [])
            if len(exchanges) != 1:
                return {"__invalid_token_exchange__": 1}
            result_shapes = exchanges[0].get("result_shapes", [])
            if len(result_shapes) != 1:
                return {"__invalid_token_exchange__": 1}
            shape = str(result_shapes[0])
            if counts.get(shape, 0) <= 0:
                return {"__invalid_token_exchange__": 1}
            counts[shape] -= 1
            if counts[shape] == 0:
                del counts[shape]
        return dict(sorted(counts.items()))

    production_non_token_shapes = without_token_exchange_shapes(
        production_contract
    )
    observer_non_token_shapes = without_token_exchange_shapes(
        observer_contract
    )
    non_token_result_shapes_match = bool(
        production_non_token_shapes == observer_non_token_shapes
        and "__invalid_token_exchange__" not in production_non_token_shapes
        and "__invalid_token_exchange__" not in observer_non_token_shapes
    )
    collective_contract_matches = bool(
        scalar_collective_contract_matches and non_token_result_shapes_match
    )
    return {
        "callback_markers": list(callback_markers),
        "collective_contract_matches_production": collective_contract_matches,
        "compared_collective_fields": list(compared_fields),
        "donate_argnums": [],
        "input_output_alias_present": input_output_alias_present,
        "non_token_result_shapes_match": non_token_result_shapes_match,
        "observer_non_token_result_shape_counts": (
            observer_non_token_shapes
        ),
        "passed": bool(
            production_contract.get("passed")
            and observer_contract.get("passed")
            and collective_contract_matches
            and not callback_markers
            and not input_output_alias_present
        ),
        "production_non_token_result_shape_counts": (
            production_non_token_shapes
        ),
        "token_exchange_shape_difference_allowed": True,
    }


def _raw_token_sequence_contract(
    observed: list[int],
    expected: np.ndarray,
) -> dict[str, Any]:
    """Compare one autoregressive prefix without decoding or normalization."""

    actual = np.asarray(observed, dtype=np.int32)
    reference = np.asarray(expected, dtype=np.int32)
    if actual.ndim != 1 or reference.ndim != 1:
        raise ValueError("raw token sequences must be one-dimensional")
    if actual.size <= 0 or actual.size > reference.size:
        raise ValueError("raw token comparison length exceeds the oracle")
    expected_prefix = reference[: actual.size]
    mismatches = np.flatnonzero(actual != expected_prefix)
    return {
        "compared_token_count": int(actual.size),
        "exact_prefix_match": bool(mismatches.size == 0),
        "expected_token_ids": expected_prefix.tolist(),
        "first_mismatch_index": (
            None if mismatches.size == 0 else int(mismatches[0])
        ),
        "observed_token_ids": actual.tolist(),
        "oracle_token_count": int(reference.size),
    }


def _make_global_array(
    jax: Any,
    mesh: Any,
    partition_spec: Any,
    global_shape: tuple[int, ...],
    builder: Callable[[int, tuple[int, ...]], np.ndarray],
) -> Any:
    from jax.sharding import NamedSharding

    sharding = NamedSharding(mesh, partition_spec)
    indices = sharding.addressable_devices_indices_map(global_shape)
    arrays = []
    for device, index in indices.items():
        shard_shape = tuple(
            (
                (dimension if item.stop is None else item.stop)
                - (0 if item.start is None else item.start)
            )
            if isinstance(item, slice)
            else 1
            for dimension, item in zip(global_shape, index, strict=True)
        )
        rank = (
            0 if index[0].start is None else index[0].start
        ) if isinstance(index[0], slice) else int(index[0])
        host = builder(rank, shard_shape)
        if tuple(host.shape) != shard_shape:
            raise RuntimeError(
                f"state builder shape drift: expected={shard_shape} "
                f"observed={host.shape}"
            )
        array = jax.device_put(host, device)
        array.block_until_ready()
        arrays.append(array)
    return jax.make_array_from_single_device_arrays(
        global_shape,
        sharding,
        arrays,
    )


def _delete_arrays(values: tuple[Any, ...]) -> None:
    for value in values:
        try:
            value.delete()
        except (AttributeError, RuntimeError):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinator-address", required=True)
    parser.add_argument("--num-processes", type=int, default=8)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument(
        "--runtime-kind",
        choices=("reference", "pallas_feature", "pallas_feature_linear"),
        default="reference",
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--source-runtime-root", type=Path)
    parser.add_argument("--source-runtime-manifest-sha256")
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-packed-manifest-sha256", required=True)
    parser.add_argument("--context-capacity", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--trace-steps", type=int, default=0)
    parser.add_argument(
        "--feature-output-tile",
        type=int,
        choices=(128, 256),
        default=None,
    )
    parser.add_argument(
        "--feature-fuse-route-weighting",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--complete-token-path",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument("--short-context-oracle-dir", type=Path)
    parser.add_argument("--short-context-oracle-manifest-sha256")
    parser.add_argument("--short-context-dsa-oracle-dir", type=Path)
    parser.add_argument("--short-context-dsa-oracle-manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.feature_output_tile is None:
        args.feature_output_tile = (
            128 if args.runtime_kind == "reference" else 256
        )
    args.feature_fuse_route_weighting = bool(
        args.feature_fuse_route_weighting
    )
    args.complete_token_path = bool(args.complete_token_path)
    oracle_mode = args.short_context_oracle_dir is not None
    if oracle_mode != (
        args.short_context_oracle_manifest_sha256 is not None
    ):
        raise ValueError(
            "short-context oracle directory and manifest pin are required together"
        )
    if oracle_mode and not args.complete_token_path:
        raise ValueError("short-context oracle requires the complete-token path")
    dsa_oracle_mode = args.short_context_dsa_oracle_dir is not None
    if dsa_oracle_mode != (
        args.short_context_dsa_oracle_manifest_sha256 is not None
    ):
        raise ValueError(
            "short-context DSA oracle directory and manifest pin are required "
            "together"
        )
    if dsa_oracle_mode and not oracle_mode:
        raise ValueError("short-context DSA oracle requires the token oracle")
    if args.num_processes != 8 or not 0 <= args.process_id < 8:
        raise ValueError("protected decoder compile requires process ids 0..7")
    if args.context_capacity != 2048:
        raise ValueError("first production compile is fixed to 2K")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("decoder warmup/iterations must be positive")
    if args.trace_steps not in (0, 2):
        raise ValueError("decoder diagnostic trace requires zero or two steps")
    if (args.trace_root is None) != (args.trace_steps == 0):
        raise ValueError("decoder trace root and trace steps must be enabled together")
    oracle_manifest = None
    prompt_token_ids = None
    oracle_generated_token_ids = None
    dsa_oracle_manifest = None
    dsa_oracle_tensors = None
    if oracle_mode:
        assert args.short_context_oracle_dir is not None
        assert args.short_context_oracle_manifest_sha256 is not None
        (
            oracle_manifest,
            prompt_token_ids,
            oracle_generated_token_ids,
        ) = _load_short_context_oracle(
            args.short_context_oracle_dir,
            expected_manifest_sha256=(
                args.short_context_oracle_manifest_sha256
            ),
        )
        compared_tokens = 1 + args.warmup + args.iterations
        if compared_tokens > oracle_generated_token_ids.size:
            raise ValueError(
                "decoder correctness window exceeds the sealed token oracle"
            )
        recurrent_steps = args.warmup + args.iterations + args.trace_steps
        if prompt_token_ids.size + recurrent_steps > args.context_capacity:
            raise ValueError(
                "prompt plus recurrent/trace steps exceeds context capacity"
            )
    if dsa_oracle_mode:
        assert args.short_context_dsa_oracle_dir is not None
        assert args.short_context_dsa_oracle_manifest_sha256 is not None
        assert oracle_manifest is not None
        assert prompt_token_ids is not None
        assert oracle_generated_token_ids is not None
        (
            dsa_oracle_manifest,
            dsa_oracle_tensors,
        ) = _load_short_context_dsa_oracle(
            args.short_context_dsa_oracle_dir,
            expected_manifest_sha256=(
                args.short_context_dsa_oracle_manifest_sha256
            ),
        )
        if dsa_oracle_manifest["token_oracle"]["manifest_sha256"] != (
            oracle_manifest["manifest_sha256"]
        ):
            raise ValueError("DSA oracle is not linked to the pinned token oracle")
        decode_positions = dsa_oracle_tensors["decode_positions"]
        if (
            decode_positions[0] != prompt_token_ids.size
            or decode_positions[-1] >= args.context_capacity
        ):
            raise ValueError("DSA oracle positions do not align with the 2K prompt")
        dsa_steps = int(decode_positions.size)
        if args.warmup + args.iterations + args.trace_steps != dsa_steps:
            raise ValueError(
                "production warmup/timing/trace window must cover the exact DSA "
                "oracle window"
            )
        if 1 + dsa_steps > oracle_generated_token_ids.size:
            raise ValueError("DSA observer token window exceeds the token oracle")
    if args.runtime_kind in ("pallas_feature", "pallas_feature_linear") and (
        args.source_runtime_root is None
        or args.source_runtime_manifest_sha256 is None
    ):
        raise ValueError(
            "feature runtime requires its source runtime root and manifest"
        )
    if args.runtime_kind == "reference" and args.feature_output_tile != 128:
        raise ValueError(
            "a non-default feature output tile requires a feature runtime"
        )
    if args.runtime_kind == "reference" and args.feature_fuse_route_weighting:
        raise ValueError(
            "feature route-weight fusion requires a feature runtime"
        )
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"
        )
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    runtime_manifest = json.loads(
        (args.runtime_root / "runtime_manifest.json").read_text()
    )
    if not isinstance(runtime_manifest, dict):
        raise RuntimeError("runtime manifest is not an object")
    if args.runtime_kind == "reference":
        context_args = SimpleNamespace(
            source_checkpoint_root=args.source_checkpoint_root,
            source_packed_manifest_sha256=args.source_packed_manifest_sha256,
            destination=runtime_manifest["destination"],
        )
        pack_context = _build_reference_context(
            context_args,
            runtime_manifest["pack_code_hash"],
        )
        from glm_tpu.greenfield.types import ExecutionPlan

        execution_plan = ExecutionPlan.from_dict(
            pack_context.source_checkpoint.layout["plan_manifest"][
                "execution_plan"
            ]
        )
        expectation = RuntimeCheckpointLoadExpectation(
            runtime_manifest_sha256=args.runtime_manifest_sha256,
            runtime_layout_manifest_sha256=runtime_manifest[
                "runtime_layout_manifest_sha256"
            ],
            runtime_layout_hash=runtime_manifest["runtime_layout_hash"],
            source_packed_manifest_sha256=runtime_manifest[
                "source_packed_manifest_sha256"
            ],
            source_layout_manifest_sha256=runtime_manifest[
                "source_layout_manifest_sha256"
            ],
            plan_hash=runtime_manifest["plan_hash"],
            schedule_hash=runtime_manifest["schedule_hash"],
            pack_code_hash=runtime_manifest["pack_code_hash"],
            destination=runtime_manifest["destination"],
            source_destination=runtime_manifest[
                "source_checkpoint_destination"
            ],
            plan_id=runtime_manifest["plan_id"],
            model_id=runtime_manifest["model_id"],
        )
        verified_runtime = verify_runtime_packed_checkpoint(
            args.runtime_root,
            expectation,
            pack_context.layout,
            pack_context.source_checkpoint,
        )
        sparse_moe_backend = "reference"
        linear_backend = "reference"
        hlo_backend_contract = "tpu_v4_pp8_reference"
    else:
        context_args = SimpleNamespace(
            source_checkpoint_root=args.source_checkpoint_root,
            source_packed_manifest_sha256=args.source_packed_manifest_sha256,
            source_runtime_root=args.source_runtime_root,
            source_runtime_manifest_sha256=(
                args.source_runtime_manifest_sha256
            ),
            destination=runtime_manifest["destination"],
        )
        pack_context = _build_feature_context(
            context_args,
            runtime_manifest["pack_code_hash"],
        )
        execution_plan = pack_context.target_plan
        expectation = FeatureRuntimeCheckpointLoadExpectation(
            runtime_manifest_sha256=args.runtime_manifest_sha256,
            runtime_layout_manifest_sha256=runtime_manifest[
                "runtime_layout_manifest_sha256"
            ],
            runtime_layout_hash=runtime_manifest["runtime_layout_hash"],
            source_runtime_manifest_sha256=runtime_manifest[
                "source_runtime_manifest_sha256"
            ],
            source_runtime_layout_manifest_sha256=runtime_manifest[
                "source_runtime_layout_manifest_sha256"
            ],
            source_runtime_layout_hash=runtime_manifest[
                "source_runtime_layout_hash"
            ],
            plan_hash=runtime_manifest["plan_hash"],
            schedule_hash=runtime_manifest["schedule_hash"],
            pack_code_hash=runtime_manifest["pack_code_hash"],
            destination=runtime_manifest["destination"],
            source_destination=runtime_manifest[
                "source_checkpoint_destination"
            ],
            plan_id=runtime_manifest["plan_id"],
            model_id=runtime_manifest["model_id"],
        )
        verified_runtime = verify_feature_runtime_packed_checkpoint(
            args.runtime_root,
            expectation,
            pack_context.layout,
            pack_context.source_runtime_checkpoint,
        )
        sparse_moe_backend = "pallas_feature"
        linear_backend = (
            "pallas" if args.runtime_kind == "pallas_feature_linear" else "reference"
        )
        hlo_backend_contract = (
            "tpu_v4_pp8_pallas_feature_linear"
            if linear_backend == "pallas"
            else "tpu_v4_pp8_pallas_feature"
        )
    schedule = build_pipeline_schedule(execution_plan)
    stage_dsa_producers = _stage_dsa_producer_layer_ids(schedule)
    if dsa_oracle_mode:
        assert dsa_oracle_tensors is not None
        expected_dsa_producers = tuple(
            producer
            for stage_values in stage_dsa_producers
            for producer in stage_values
        )
        if dsa_oracle_tensors["producer_layer_ids"].tolist() != list(
            expected_dsa_producers
        ):
            raise ValueError(
                "DSA oracle producer IDs differ from the executable schedule"
            )
    state_layout = build_decoder_state_layout(
        execution_plan,
        schedule,
        context_capacity=args.context_capacity,
    )
    import jax
    import ml_dtypes
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    loaded = None
    state_values: tuple[Any, ...] = ()
    auxiliary_values: tuple[Any, ...] = ()
    try:
        if (
            jax.process_count() != 8
            or jax.local_device_count() != 4
            or jax.device_count() != 32
        ):
            raise RuntimeError(
                "protected decoder requires 8 processes, 4 local chips, and 32 chips"
            )
        local_ids = np.asarray(
            [device.id for device in jax.local_devices()],
            dtype=np.int32,
        )
        fleet_local_ids = np.asarray(
            multihost_utils.process_allgather(local_ids)
        ).reshape(args.num_processes, jax.local_device_count())
        observed_local_order = {
            int(device_id): local_index
            for process_row in fleet_local_ids
            for local_index, device_id in enumerate(process_row.tolist())
        }
        live_topology = discover_physical_topology(
            jax.devices(),
            slice_name="db-v4-64-od",
            observed_local_order=observed_local_order,
        )
        validate_target_v4_64(live_topology)
        if live_topology.topology_hash != execution_plan.topology.topology_hash:
            raise RuntimeError("live topology differs from the runtime checkpoint plan")
        by_id = {int(device.id): device for device in jax.devices()}
        runtime_devices = tuple(
            by_id[device.device_id] for device in pack_context.layout.devices
        )
        groups = tuple(
            tuple(range(stage * 4, stage * 4 + 4)) for stage in range(8)
        )
        pairs = tuple(
            (groups[stage][slot], groups[(stage + 1) % 8][slot])
            for stage in range(8)
            for slot in range(4)
        )
        decoder = build_decoder_step_program(
            execution_plan,
            schedule,
            state_layout,
            pack_context.layout,
            groups,
            pairs,
            devices=runtime_devices,
            sparse_moe_backend=sparse_moe_backend,
            feature_output_tile=args.feature_output_tile,
            feature_fuse_route_weighting=(
                args.feature_fuse_route_weighting
            ),
            linear_backend=linear_backend,
            complete_token_path=args.complete_token_path,
        )
        dsa_observer = None
        if dsa_oracle_mode:
            dsa_observer = build_decoder_step_program(
                execution_plan,
                schedule,
                state_layout,
                pack_context.layout,
                groups,
                pairs,
                devices=runtime_devices,
                sparse_moe_backend=sparse_moe_backend,
                feature_output_tile=args.feature_output_tile,
                feature_fuse_route_weighting=(
                    args.feature_fuse_route_weighting
                ),
                linear_backend=linear_backend,
                complete_token_path=True,
                observe_dsa_events=True,
            )
        prefill = None
        if oracle_mode:
            assert prompt_token_ids is not None
            prefill = build_teacher_forced_prefill_program(
                decoder,
                prompt_length=int(prompt_token_ids.size),
            )
        multihost_utils.sync_global_devices("greenfield-short-decoder-load-start")
        load_started = time.monotonic()
        loaded = load_runtime_checkpoint(
            verified_runtime,
            expectation,
            pack_context.layout,
            decoder.mesh,
            verify_device_roundtrip=False,
        )
        load_seconds = time.monotonic() - load_started
        multihost_utils.sync_global_devices("greenfield-short-decoder-load-end")

        total_devices = decoder.config.total_devices
        kv_shape = state_layout.stages[0].padded_kv_cache_shape
        index_shape = state_layout.stages[0].padded_indexer_cache_shape
        residual_shape = (total_devices, 1, execution_plan.geometry.hidden_size)
        kv_global_shape = (total_devices, *kv_shape)
        index_global_shape = (total_devices, *index_shape)
        metadata_shape = (total_devices, 1, decoder.config.metadata_width)
        token_shape = (total_devices, 1)
        initial_row = np.linspace(
            -0.5,
            0.5,
            execution_plan.geometry.hidden_size,
            dtype=np.float32,
        ).astype(ml_dtypes.bfloat16)[None, None, :]

        def residual_builder(rank: int, shape: tuple[int, ...]) -> np.ndarray:
            value = np.zeros(shape, dtype=ml_dtypes.bfloat16)
            if not oracle_mode and rank in groups[0]:
                value[...] = initial_row
            return value

        def zero_bf16(_: int, shape: tuple[int, ...]) -> np.ndarray:
            return np.zeros(shape, dtype=ml_dtypes.bfloat16)

        def metadata_builder(rank: int, shape: tuple[int, ...]) -> np.ndarray:
            value = np.full(shape, -1, dtype=np.int32)
            value[..., decoder.config.count_index] = 0
            value[..., decoder.config.producer_index] = -1
            value[..., decoder.config.visited_index] = 0
            value[..., decoder.config.health_index] = 1
            value[..., decoder.config.active_index] = int(rank in groups[0])
            return value

        def token_builder(rank: int, shape: tuple[int, ...]) -> np.ndarray:
            value = np.full(shape, -1, dtype=np.int32)
            if rank in groups[0]:
                value[...] = np.int32(
                    prompt_token_ids[0]
                    if oracle_mode and prompt_token_ids is not None
                    else 1
                )
            return value

        residual = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[1],
            residual_shape,
            residual_builder,
        )
        kv = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[2],
            kv_global_shape,
            zero_bf16,
        )
        index = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[3],
            index_global_shape,
            zero_bf16,
        )
        metadata = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[4],
            metadata_shape,
            metadata_builder,
        )
        token = None
        if args.complete_token_path:
            token = _make_global_array(
                jax,
                decoder.mesh,
                decoder.input_specs[5],
                token_shape,
                token_builder,
            )
        prompt = None
        if oracle_mode:
            assert prompt_token_ids is not None
            prompt = jax.device_put(
                prompt_token_ids,
                NamedSharding(decoder.mesh, P()),
            )
        auxiliary_values = tuple(
            value for value in (token, prompt) if value is not None
        )
        position = jax.device_put(
            np.asarray([0], dtype=np.int32),
            NamedSharding(decoder.mesh, P()),
        )
        pages = state_layout.stages[0].physical_page_count
        block_tables = jax.device_put(
            np.arange(pages, dtype=np.int32)[None, :],
            NamedSharding(decoder.mesh, P()),
        )
        context_lengths = jax.device_put(
            np.asarray([1], dtype=np.int32),
            NamedSharding(decoder.mesh, P()),
        )
        common_inputs = (
            loaded.weights,
            residual,
            kv,
            index,
            metadata,
        )
        if args.complete_token_path:
            assert token is not None
            inputs = (
                *common_inputs,
                token,
                position,
                block_tables,
                context_lengths,
            )
            state_values = (
                residual,
                kv,
                index,
                metadata,
                token,
                position,
                block_tables,
                context_lengths,
            )
            donate_argnums = (1, 2, 3, 4, 5)
        else:
            inputs = (
                *common_inputs,
                position,
                block_tables,
                context_lengths,
            )
            state_values = (
                residual,
                kv,
                index,
                metadata,
                position,
                block_tables,
                context_lengths,
            )
            donate_argnums = (1, 2, 3, 4)
        prefill_inputs = None
        if oracle_mode:
            assert prefill is not None and prompt is not None
            prefill_inputs = (
                loaded.weights,
                residual,
                kv,
                index,
                metadata,
                prompt,
                position,
                block_tables,
                context_lengths,
            )
        multihost_utils.sync_global_devices("greenfield-short-decoder-compile-start")
        compile_started = time.monotonic()
        lowered = jax.jit(
            decoder.execute,
            donate_argnums=donate_argnums,
        ).lower(*inputs)
        compiled = lowered.compile()
        compile_seconds = time.monotonic() - compile_started
        multihost_utils.sync_global_devices("greenfield-short-decoder-compile-end")
        optimized_hlo = compiled.as_text()
        hlo_sha256 = sha256(optimized_hlo.encode("utf-8")).hexdigest()
        fleet_hlo_hashes = _fleet_digest(
            multihost_utils,
            hlo_sha256,
            num_processes=args.num_processes,
        )
        hlo_contract = validate_decoder_step_hlo(
            optimized_hlo,
            config=decoder.config,
            schedule=schedule,
            groups=groups,
            pairs=pairs,
            backend_contract=hlo_backend_contract,
            feature_output_tile=decoder.feature_output_tile,
            feature_fuse_route_weighting=(
                decoder.feature_fuse_route_weighting
            ),
            complete_token_path=decoder.complete_token_path,
        )
        if jax.process_index() == 0:
            hlo_dir = args.output.parent / "hlo"
            hlo_dir.mkdir(parents=True, exist_ok=True)
            hlo_stem = (
                "decoder_78layer_2k_token"
                if args.complete_token_path
                else "decoder_78layer_2k"
            )
            with gzip.open(
                hlo_dir / f"{hlo_stem}.optimized_hlo.txt.gz",
                "wt",
                encoding="utf-8",
            ) as stream:
                stream.write(optimized_hlo)
            _atomic_json(
                hlo_dir / f"{hlo_stem}.hlo_contract.json",
                hlo_contract,
            )
        del optimized_hlo
        if not hlo_contract["passed"]:
            raise RuntimeError(
                "decoder HLO contract failed before execution: "
                f"{hlo_contract['violations']}"
            )
        compiled_dsa_observer = None
        dsa_observer_compile_seconds = None
        dsa_observer_hlo_sha256 = None
        fleet_dsa_observer_hlo_hashes = None
        dsa_observer_hlo_contract = None
        dsa_observer_isolation_contract = None
        if dsa_oracle_mode:
            assert dsa_observer is not None
            multihost_utils.sync_global_devices(
                "greenfield-short-dsa-observer-compile-start"
            )
            observer_compile_started = time.monotonic()
            # Deliberately no donate_argnums: the observer must replay first
            # while preserving the prefill result for production timing.
            lowered_dsa_observer = jax.jit(dsa_observer.execute).lower(*inputs)
            compiled_dsa_observer = lowered_dsa_observer.compile()
            dsa_observer_compile_seconds = (
                time.monotonic() - observer_compile_started
            )
            multihost_utils.sync_global_devices(
                "greenfield-short-dsa-observer-compile-end"
            )
            optimized_dsa_observer_hlo = compiled_dsa_observer.as_text()
            dsa_observer_hlo_sha256 = sha256(
                optimized_dsa_observer_hlo.encode("utf-8")
            ).hexdigest()
            fleet_dsa_observer_hlo_hashes = _fleet_digest(
                multihost_utils,
                dsa_observer_hlo_sha256,
                num_processes=args.num_processes,
            )
            dsa_observer_hlo_contract = validate_decoder_step_hlo(
                optimized_dsa_observer_hlo,
                config=dsa_observer.config,
                schedule=schedule,
                groups=groups,
                pairs=pairs,
                backend_contract=hlo_backend_contract,
                feature_output_tile=dsa_observer.feature_output_tile,
                feature_fuse_route_weighting=(
                    dsa_observer.feature_fuse_route_weighting
                ),
                complete_token_path=True,
                token_observation_candidates=(
                    dsa_observer.config.token_observation_candidates
                ),
            )
            dsa_observer_isolation_contract = (
                _observer_hlo_isolation_contract(
                    optimized_dsa_observer_hlo,
                    production_contract=hlo_contract,
                    observer_contract=dsa_observer_hlo_contract,
                )
            )
            if jax.process_index() == 0:
                hlo_dir = args.output.parent / "hlo"
                with gzip.open(
                    hlo_dir
                    / "decoder_78layer_2k_token_dsa_observer.optimized_hlo.txt.gz",
                    "wt",
                    encoding="utf-8",
                ) as stream:
                    stream.write(optimized_dsa_observer_hlo)
                _atomic_json(
                    hlo_dir
                    / "decoder_78layer_2k_token_dsa_observer.hlo_contract.json",
                    dsa_observer_hlo_contract,
                )
                _atomic_json(
                    hlo_dir
                    / "decoder_78layer_2k_token_dsa_observer.isolation_contract.json",
                    dsa_observer_isolation_contract,
                )
            del optimized_dsa_observer_hlo
            if not dsa_observer_hlo_contract["passed"]:
                raise RuntimeError(
                    "DSA observer HLO contract failed before execution: "
                    f"{dsa_observer_hlo_contract['violations']}"
                )
            if not dsa_observer_isolation_contract["passed"]:
                raise RuntimeError(
                    "DSA observer isolation contract failed before execution: "
                    f"{dsa_observer_isolation_contract}"
                )
        compiled_prefill = None
        prefill_compile_seconds = None
        prefill_hlo_sha256 = None
        fleet_prefill_hlo_hashes = None
        prefill_hlo_contract = None
        if oracle_mode:
            assert prefill is not None and prefill_inputs is not None
            multihost_utils.sync_global_devices(
                "greenfield-short-prefill-compile-start"
            )
            prefill_compile_started = time.monotonic()
            lowered_prefill = jax.jit(
                prefill.execute,
                donate_argnums=(1, 2, 3, 4),
            ).lower(*prefill_inputs)
            compiled_prefill = lowered_prefill.compile()
            prefill_compile_seconds = (
                time.monotonic() - prefill_compile_started
            )
            multihost_utils.sync_global_devices(
                "greenfield-short-prefill-compile-end"
            )
            optimized_prefill_hlo = compiled_prefill.as_text()
            prefill_hlo_sha256 = sha256(
                optimized_prefill_hlo.encode("utf-8")
            ).hexdigest()
            fleet_prefill_hlo_hashes = _fleet_digest(
                multihost_utils,
                prefill_hlo_sha256,
                num_processes=args.num_processes,
            )
            prefill_hlo_contract = validate_teacher_forced_prefill_hlo(
                optimized_prefill_hlo,
                program=prefill,
                schedule=schedule,
                backend_contract=hlo_backend_contract,
            )
            if jax.process_index() == 0:
                hlo_dir = args.output.parent / "hlo"
                with gzip.open(
                    hlo_dir / "prefill_78layer_2k.optimized_hlo.txt.gz",
                    "wt",
                    encoding="utf-8",
                ) as stream:
                    stream.write(optimized_prefill_hlo)
                _atomic_json(
                    hlo_dir / "prefill_78layer_2k.hlo_contract.json",
                    prefill_hlo_contract,
                )
            del optimized_prefill_hlo
            if not prefill_hlo_contract["passed"]:
                raise RuntimeError(
                    "prefill HLO contract failed before execution: "
                    f"{prefill_hlo_contract['violations']}"
                )

        generated_tokens: list[int] = []
        prefill_wall_ms = None
        prefill_token_oracle_contract = None
        dsa_observer_contract = None
        if oracle_mode:
            assert compiled_prefill is not None and prefill_inputs is not None
            prefill_started = time.perf_counter_ns()
            output = compiled_prefill(*prefill_inputs)
            output[3].block_until_ready()
            prefill_wall_ms = (
                time.perf_counter_ns() - prefill_started
            ) / 1_000_000
            first_token = _materialize_global_array(
                jax,
                multihost_utils,
                output[4],
            )[list(groups[0]), 0]
            if not np.all(first_token == first_token[0]):
                raise RuntimeError("prefill token lanes disagree")
            generated_tokens.append(int(first_token[0]))
            if dsa_oracle_mode:
                assert oracle_generated_token_ids is not None
                prefill_token_oracle_contract = _raw_token_sequence_contract(
                    generated_tokens,
                    oracle_generated_token_ids,
                )
                if jax.process_index() == 0:
                    _atomic_json(
                        args.output.parent
                        / "dsa_observer"
                        / "prefill_token_observation.json",
                        prefill_token_oracle_contract,
                    )
                if not prefill_token_oracle_contract["exact_prefix_match"]:
                    raise RuntimeError(
                        "prefill first-token oracle contract failed: "
                        f"{prefill_token_oracle_contract}"
                    )
        else:
            output = compiled(*inputs)
            output[3].block_until_ready()
        if dsa_oracle_mode:
            assert compiled_dsa_observer is not None
            assert dsa_oracle_manifest is not None
            assert dsa_oracle_tensors is not None
            assert oracle_generated_token_ids is not None
            assert prompt_token_ids is not None
            observer_current = tuple(output)
            observer_owns_current = False
            observer_step_records = []
            observer_artifacts: list[dict[str, Any]] = []
            observer_tokens: list[int] = []
            try:
                for step, decode_position in enumerate(
                    dsa_oracle_tensors["decode_positions"].tolist()
                ):
                    observer_result = compiled_dsa_observer(
                        loaded.weights, *observer_current
                    )
                    observer_result[8].block_until_ready()
                    observation_host = _materialize_global_array(
                        jax,
                        multihost_utils,
                        observer_result[8],
                    )
                    token_observation_host = _materialize_global_array(
                        jax,
                        multihost_utils,
                        observer_result[9],
                    )
                    observer_token = _materialize_global_array(
                        jax,
                        multihost_utils,
                        observer_result[4],
                    )[list(groups[0]), 0]
                    if not np.all(observer_token == observer_token[0]):
                        raise RuntimeError("DSA observer token lanes disagree")
                    observed_token_id = int(observer_token[0])
                    expected_token_id = int(
                        oracle_generated_token_ids[step + 1]
                    )
                    token_observation_contract = (
                        _validate_token_observation_step(
                            token_observation_host,
                            groups=groups,
                            candidate_width=(
                                dsa_observer.config.token_observation_candidates
                            ),
                            expected_token_id=expected_token_id,
                            observed_token_id=observed_token_id,
                            vocab_size=decoder.config.vocab_size,
                        )
                    )
                    observation_sha256 = sha256(
                        np.ascontiguousarray(observation_host).tobytes()
                    ).hexdigest()
                    token_observation_sha256 = sha256(
                        np.ascontiguousarray(
                            token_observation_host
                        ).tobytes()
                    ).hexdigest()
                    artifact_name = f"step_{step:02d}_position_{decode_position}.npz"
                    if jax.process_index() == 0:
                        _atomic_npz(
                            args.output.parent / "dsa_observer" / artifact_name,
                            decode_position=np.asarray(
                                [decode_position], dtype=np.int32
                            ),
                            observation=np.asarray(
                                observation_host, dtype=np.int32
                            ),
                            token_observation=np.asarray(
                                token_observation_host, dtype=np.int32
                            ),
                        )
                    observer_artifacts.append(
                        {
                            "filename": artifact_name,
                            "observation_sha256": observation_sha256,
                            "token_observation_sha256": (
                                token_observation_sha256
                            ),
                        }
                    )
                    step_contract = _validate_dsa_observation_step(
                        observation_host,
                        groups=groups,
                        stage_producer_layer_ids=stage_dsa_producers,
                        selected_width=decoder.config.selected_width,
                        expected_positions=dsa_oracle_tensors[
                            "selected_positions"
                        ][step],
                        expected_scores=dsa_oracle_tensors[
                            "selected_scores"
                        ][step],
                        expected_valid_counts=dsa_oracle_tensors[
                            "valid_counts"
                        ][step],
                        expected_producer_layer_ids=dsa_oracle_tensors[
                            "producer_layer_ids"
                        ],
                        decode_position=int(decode_position),
                    )
                    next_position = np.asarray(
                        jax.device_get(observer_result[5])
                    )
                    step_contract["next_position"] = next_position.tolist()
                    step_contract["observation_sha256"] = observation_sha256
                    step_contract["token_observation_sha256"] = (
                        token_observation_sha256
                    )
                    step_contract["token_observation"] = (
                        token_observation_contract
                    )
                    step_contract["position_passed"] = bool(
                        next_position.tolist() == [int(decode_position) + 1]
                    )
                    step_contract["passed"] = bool(
                        step_contract["passed"]
                        and step_contract["position_passed"]
                        and token_observation_contract["passed"]
                    )
                    observer_step_records.append(step_contract)
                    if not step_contract["passed"]:
                        raise RuntimeError(
                            "DSA observer device-score/set/tie contract failed: "
                            f"{step_contract}"
                        )
                    observer_tokens.append(observed_token_id)
                    if observer_owns_current:
                        _delete_arrays(observer_current)
                    observer_current = tuple(observer_result[:8])
                    observer_owns_current = True
                    _delete_arrays((observer_result[8], observer_result[9]))
                preserved_prefill_position = np.asarray(
                    jax.device_get(output[5])
                )
                preserved_prefill_token = _materialize_global_array(
                    jax,
                    multihost_utils,
                    output[4],
                )[list(groups[0]), 0]
                prefill_state_preserved = bool(
                    preserved_prefill_position.tolist()
                    == [int(prompt_token_ids.size)]
                    and np.all(preserved_prefill_token == generated_tokens[0])
                )
                observer_token_contract = _raw_token_sequence_contract(
                    observer_tokens,
                    oracle_generated_token_ids[
                        1 : 1 + len(observer_tokens)
                    ],
                )
                dsa_observer_contract = {
                    "all_steps_passed": all(
                        record["passed"] for record in observer_step_records
                    ),
                    "decode_step_count": len(observer_step_records),
                    "event_count": int(
                        dsa_oracle_tensors["producer_layer_ids"].size
                    ),
                    "manifest_sha256": dsa_oracle_manifest[
                        "manifest_sha256"
                    ],
                    "observer_executed_before_production": True,
                    "observer_hlo_sha256": dsa_observer_hlo_sha256,
                    "observation_artifacts": observer_artifacts,
                    "prefill_state_preserved_without_donation": (
                        prefill_state_preserved
                    ),
                    "prefill_token_sequence": prefill_token_oracle_contract,
                    "production_executable_observer_enabled": False,
                    "score_comparison": {
                        "compared": True,
                        "cross_backend_total_order_is_gate": False,
                        "executing_score_order_and_ties_are_gate": True,
                        "legacy_scores_use_position_aligned_bounded_gate": False,
                        "legacy_scores_use_position_aligned_diagnostic": True,
                        "legacy_score_tolerance": (
                            DSA_CROSS_BACKEND_SCORE_TOLERANCE.to_dict()
                        ),
                        "oracle_sha256": dsa_oracle_manifest["arrays"][
                            "selected_scores"
                        ]["sha256"],
                        "selected_set_against_legacy_is_exact_gate": True,
                    },
                    "step_records": observer_step_records,
                    "token_oracle_offset": 1,
                    "token_observation_candidates": (
                        dsa_observer.config.token_observation_candidates
                    ),
                    "token_sequence": observer_token_contract,
                }
                dsa_observer_contract["passed"] = bool(
                    dsa_observer_contract["all_steps_passed"]
                    and prefill_state_preserved
                    and observer_token_contract["exact_prefix_match"]
                )
                if not dsa_observer_contract["passed"]:
                    raise RuntimeError(
                        "DSA observer replay contract failed: "
                        f"{dsa_observer_contract}"
                    )
            finally:
                if observer_owns_current:
                    _delete_arrays(observer_current)
        state_values = (
            tuple(output)
            if args.complete_token_path
            else (*output, position, block_tables, context_lengths)
        )
        current = output

        def run_step(values: tuple[Any, ...]) -> tuple[Any, ...]:
            if args.complete_token_path:
                return compiled(loaded.weights, *values)
            return compiled(
                loaded.weights,
                *values,
                position,
                block_tables,
                context_lengths,
            )

        def materialize_active_token(
            values: tuple[Any, ...], *, phase: str
        ) -> int:
            token_values = _materialize_global_array(
                jax,
                multihost_utils,
                values[4],
            )[list(groups[0]), 0]
            if not np.all(token_values == token_values[0]):
                raise RuntimeError(
                    f"complete token lanes disagree after {phase} step"
                )
            return int(token_values[0])

        for _ in range(args.warmup):
            current = run_step(current)
            current[3].block_until_ready()
            if oracle_mode:
                generated_tokens.append(
                    materialize_active_token(current, phase="warmup")
                )
        samples = []
        if not oracle_mode:
            generated_tokens = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            current = run_step(current)
            current[3].block_until_ready()
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
            if args.complete_token_path:
                generated_tokens.append(
                    materialize_active_token(current, phase="timed")
                )
        trace_record = None
        if args.trace_root is not None:
            args.trace_root.mkdir(parents=True, exist_ok=False)
            options = jax.profiler.ProfileOptions()
            options.python_tracer_level = 0
            tracing = False
            multihost_utils.sync_global_devices(
                "greenfield-short-decoder-trace-ready"
            )
            try:
                jax.profiler.start_trace(
                    str(args.trace_root),
                    profiler_options=options,
                )
                tracing = True
                multihost_utils.sync_global_devices(
                    "greenfield-short-decoder-trace-started"
                )
                for step in range(args.trace_steps):
                    with jax.profiler.TraceAnnotation(
                        "greenfield_short_decoder_body_step",
                        step_num=step,
                    ):
                        current = run_step(current)
                        current[3].block_until_ready()
                jax.profiler.stop_trace()
                tracing = False
            finally:
                if tracing:
                    jax.profiler.stop_trace()
            multihost_utils.sync_global_devices(
                "greenfield-short-decoder-trace-stopped"
            )
            xplanes = tuple(sorted(args.trace_root.rglob("*.xplane.pb")))
            if len(xplanes) != 1:
                raise RuntimeError(
                    "expected one local decoder XPlane, "
                    f"found {len(xplanes)}"
                )
            trace_record = {
                "files": [
                    {
                        "path": str(xplanes[0]),
                        "sha256": _sha256_file(xplanes[0]),
                        "size_bytes": xplanes[0].stat().st_size,
                    }
                ],
                "profiler_started_after_profiler_free_timing": True,
                "steps": args.trace_steps,
            }
        state_values = (
            tuple(current)
            if args.complete_token_path
            else (*current, position, block_tables, context_lengths)
        )
        metadata_host = _materialize_global_array(
            jax,
            multihost_utils,
            current[3],
        )
        active = np.flatnonzero(
            metadata_host[:, 0, decoder.config.active_index] == 1
        )
        expected_producer = schedule.stages[-1].layers[-1].index_state_producer_layer
        active_metadata = metadata_host[active, 0]
        metadata_contract = {
            "active_ranks": active.tolist(),
            "health": sorted(
                set(active_metadata[:, decoder.config.health_index].tolist())
            ),
            "producer": sorted(
                set(active_metadata[:, decoder.config.producer_index].tolist())
            ),
            "selected_prefix": sorted(
                set(
                    tuple(row)
                    for row in active_metadata[:, :8].tolist()
                )
            ),
            "valid_counts": sorted(
                set(active_metadata[:, decoder.config.count_index].tolist())
            ),
            "visited": sorted(
                set(active_metadata[:, decoder.config.visited_index].tolist())
            ),
        }
        base_metadata_passed = (
            metadata_contract["active_ranks"] == list(groups[0])
            and metadata_contract["health"] == [1]
            and metadata_contract["producer"] == [expected_producer]
            and metadata_contract["visited"] == [255]
        )
        if args.complete_token_path:
            next_position = np.asarray(jax.device_get(current[5]))
            next_context_lengths = np.asarray(
                jax.device_get(current[7])
            )
            selected_state_contract = (
                _validate_completed_step_selected_states(
                    active_metadata,
                    selected_width=decoder.config.selected_width,
                    count_index=decoder.config.count_index,
                    next_position=int(next_position[0]),
                    next_context_length=int(next_context_lengths[0]),
                )
            )
            metadata_contract["selected_state_contract"] = (
                selected_state_contract
            )
            metadata_passed = bool(
                base_metadata_passed
                and all(selected_state_contract["rows_valid"])
            )
        else:
            metadata_passed = bool(
                base_metadata_passed
                and metadata_contract["valid_counts"] == [1]
                and all(
                    row[0] == 0
                    for row in metadata_contract["selected_prefix"]
                )
            )
        token_contract = None
        token_passed = True
        if args.complete_token_path:
            token_host = _materialize_global_array(
                jax,
                multihost_utils,
                current[4],
            )
            active_tokens = token_host[active, 0]
            raw_sequence = None
            if oracle_mode:
                assert oracle_manifest is not None
                assert oracle_generated_token_ids is not None
                assert prompt_token_ids is not None
                raw_sequence = _raw_token_sequence_contract(
                    generated_tokens,
                    oracle_generated_token_ids,
                )
            expected_next_position = (
                int(prompt_token_ids.size)
                + args.warmup
                + args.iterations
                + args.trace_steps
                if oracle_mode and prompt_token_ids is not None
                else None
            )
            token_contract = {
                "active_tokens": sorted(set(active_tokens.tolist())),
                "correctness_window_tokens": (
                    generated_tokens if oracle_mode else None
                ),
                "profiler_free_window_tokens": generated_tokens[
                    -args.iterations:
                ],
                "all_active_lanes_equal": bool(
                    active_tokens.size == len(groups[0])
                    and np.all(active_tokens == active_tokens[0])
                ),
                "all_in_vocabulary": bool(
                    np.all(active_tokens >= 0)
                    and np.all(
                        active_tokens < execution_plan.geometry.vocab_size
                    )
                ),
                "next_position": next_position.tolist(),
                "next_context_lengths": next_context_lengths.tolist(),
                "expected_next_position": expected_next_position,
                "position_contract_passed": bool(
                    expected_next_position is None
                    or (
                        next_position.tolist() == [expected_next_position]
                        and next_context_lengths.tolist()
                        == [expected_next_position + 1]
                    )
                ),
                "prefill_used": oracle_mode,
                "raw_token_sequence": raw_sequence,
                "short_context_oracle": (
                    {
                        "generated_token_ids_sha256": oracle_manifest[
                            "generated_token_ids_sha256"
                        ],
                        "manifest_sha256": oracle_manifest[
                            "manifest_sha256"
                        ],
                        "prompt_token_count": int(prompt_token_ids.size),
                        "prompt_token_ids_sha256": oracle_manifest[
                            "prompt_token_ids_sha256"
                        ],
                        "source": oracle_manifest["source"],
                    }
                    if oracle_mode
                    and oracle_manifest is not None
                    and prompt_token_ids is not None
                    else None
                ),
                "synthetic_initial_state": not oracle_mode,
            }
            token_passed = bool(
                token_contract["all_active_lanes_equal"]
                and token_contract["all_in_vocabulary"]
                and token_contract["position_contract_passed"]
                and (
                    raw_sequence is None
                    or raw_sequence["exact_prefix_match"]
                )
            )
            metadata_passed = bool(metadata_passed and token_passed)
        if dsa_oracle_mode:
            if dsa_observer_contract is None:
                raise RuntimeError("DSA observer contract was not produced")
            metadata_passed = bool(
                metadata_passed and dsa_observer_contract["passed"]
            )
        local_kv_nonzero = []
        local_index_nonzero = []
        for shard in current[1].addressable_shards:
            host = np.asarray(jax.device_get(shard.data))
            local_kv_nonzero.append(int(np.count_nonzero(host[:, 0, 0, 0])))
        for shard in current[2].addressable_shards:
            host = np.asarray(jax.device_get(shard.data))
            local_index_nonzero.append(int(np.count_nonzero(host[:, 0, 0, 0])))
        record = {
            "artifact_kind": (
                "greenfield_real_78layer_2k_decoder_"
                + (
                    "token_oracle_"
                    if oracle_mode
                    else ("token_" if args.complete_token_path else "body_")
                )
                + f"{args.runtime_kind}"
            ),
            "body_only": not args.complete_token_path,
            "code_hash": code_hash,
            "compile_seconds": compile_seconds,
            "decoder_compile_seconds": compile_seconds,
            "context_capacity": args.context_capacity,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "device_memory_after_execute": [
                _memory_stats(device) for device in jax.local_devices()
            ],
            "dsa_observer_compile_seconds": dsa_observer_compile_seconds,
            "dsa_observer_contract": dsa_observer_contract,
            "dsa_observer_hlo_contract": dsa_observer_hlo_contract,
            "dsa_observer_hlo_sha256": dsa_observer_hlo_sha256,
            "dsa_observer_isolation_contract": (
                dsa_observer_isolation_contract
            ),
            "fleet_hlo_hashes": fleet_hlo_hashes,
            "fleet_dsa_observer_hlo_hashes": (
                fleet_dsa_observer_hlo_hashes
            ),
            "fleet_prefill_hlo_hashes": fleet_prefill_hlo_hashes,
            "fleet_local_device_ids_in_runtime_order": fleet_local_ids.tolist(),
            "hlo_contract": hlo_contract,
            "hostname": socket.gethostname(),
            "iterations": args.iterations,
            "jax_process_index": jax.process_index(),
            "launch_process_id": args.process_id,
            "load_record": loaded.load_record,
            "load_seconds": load_seconds,
            "local_index_nonzero_counts": local_index_nonzero,
            "local_kv_nonzero_counts": local_kv_nonzero,
            "metadata_contract": metadata_contract,
            "metadata_passed": metadata_passed,
            "optimized_hlo_sha256": hlo_sha256,
            "plan_hash": execution_plan.plan_hash,
            "profiler_free_body_wall": _percentiles(samples),
            "profiler_free_complete_step_wall": (
                _percentiles(samples) if args.complete_token_path else None
            ),
            "prefill_compile_seconds": prefill_compile_seconds,
            "prefill_hlo_contract": prefill_hlo_contract,
            "prefill_hlo_sha256": prefill_hlo_sha256,
            "prefill_wall_ms": prefill_wall_ms,
            "prefill_used": oracle_mode,
            "raw_token_claim": bool(
                oracle_mode
                and token_contract is not None
                and token_contract["raw_token_sequence"] is not None
                and token_contract["raw_token_sequence"]["exact_prefix_match"]
            ),
            "runtime_layout_hash": pack_context.layout.layout_hash,
            "runtime_manifest_sha256": expectation.runtime_manifest_sha256,
            "runtime_kind": args.runtime_kind,
            "schedule_hash": schedule.schedule_hash,
            "schema_version": 6,
            "state_layout": state_layout.to_dict(),
            "state_layout_hash": state_layout.state_layout_hash,
            "sparse_moe_backend": decoder.sparse_moe_backend,
            "feature_output_tile": decoder.feature_output_tile,
            "feature_fuse_route_weighting": (
                decoder.feature_fuse_route_weighting
            ),
            "linear_backend": decoder.linear_backend,
            "complete_token_path": decoder.complete_token_path,
            "token_contract": token_contract,
            "token_passed": token_passed,
            "topology_hash": live_topology.topology_hash,
            "trace": trace_record,
            "trace_and_timing_use_observer_free_production_executable": bool(
                not decoder.observe_dsa_events
            ),
            "transformer_body_timing_only": not args.complete_token_path,
            "warmup": args.warmup,
        }
        _atomic_json(args.output, record)
        if not token_passed:
            raise RuntimeError(
                f"decoder raw-token contract failed: {token_contract}"
            )
        if not metadata_passed:
            raise RuntimeError(f"decoder metadata contract failed: {metadata_contract}")
        print(
            "GREENFIELD_SHORT_DECODER_HOST_OK "
            f"launch_process={args.process_id} jax_process={jax.process_index()} "
            f"compile_s={compile_seconds:.3f} "
            f"p50_step_ms={record['profiler_free_body_wall']['p50_ms']:.6f} "
            f"hlo={hlo_sha256}",
            flush=True,
        )
        return 0
    finally:
        _delete_arrays(state_values)
        _delete_arrays(auxiliary_values)
        if loaded is not None:
            loaded.close()
        jax.distributed.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
