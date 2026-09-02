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
    FINAL_DENSE_CONVOLUTION_RUNTIME_LAYOUT,
    FUSED_QKV_A_N82_RUNTIME_LAYOUT,
    SEPARATE_QKV_A_RUNTIME_LAYOUT,
    build_decoder_state_layout,
    build_pipeline_schedule,
)
from glm_tpu.greenfield.kernels.stage_local import (  # noqa: E402
    _reduce_strategy_nd_row0_bf16_partials,
)
from glm_tpu.greenfield.runtime import (  # noqa: E402
    LAYER0_INGREDIENT_NAMES,
    build_decoder_step_program,
    build_teacher_forced_prefill_program,
    validate_decoder_step_hlo,
    validate_dsa_query_weight_materializer_hlo,
    validate_layer0_residual_discriminator_hlo,
    validate_layer0_ingredients_observer_hlo,
    validate_prefill_index_weight_materialization_hlo,
    validate_teacher_forced_prefill_hlo,
)
from glm_tpu.greenfield.sharding.hlo_contract import (  # noqa: E402
    parse_hlo_module,
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
LAYER1_CURRENT_NORMALIZED_HIDDEN_SHA256 = (
    "787c9ba7b39d6fd43b59876f713052b64a3ac2d3ecbc6074795ba5364abe153b"
)
LAYER1_MAIN_ROPE_NORMALIZED_HIDDEN_SHA256 = (
    "6c54c09a773e622fef35e753dc99929a83b73d9d89157fd732a5ace903149bca"
)
STRATEGY_ND_CANARY_MODEL_TO_PHYSICAL = (
    0, 8, 16, 24, 2, 10, 18, 26,
    4, 12, 20, 28, 6, 14, 22, 30,
    1, 9, 17, 25, 3, 11, 19, 27,
    5, 13, 21, 29, 7, 15, 23, 31,
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


def _runtime_pipeline_groups(
    schedule: Any,
    runtime_layout: Any,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]:
    """Map physical stage ownership to ranks in runtime-layout order.

    ``build_decoder_step_program`` consumes ranks into its explicitly ordered
    runtime-device tuple, while the execution plan records physical device
    ids. Deriving this mapping from the final checkpoint layout keeps PP8 and
    PP16 on the same code path without assuming either 8x4 or 16x2 geometry.
    """

    rank_by_device_id: dict[int, int] = {}
    for rank, device in enumerate(runtime_layout.devices):
        device_id = int(device.device_id)
        if device_id in rank_by_device_id:
            raise ValueError(
                f"runtime layout repeats physical device id {device_id}"
            )
        rank_by_device_id[device_id] = rank
    groups = []
    for stage in schedule.stages:
        try:
            group = tuple(
                rank_by_device_id[int(device_id)]
                for device_id in stage.assignment.device_ids
            )
        except KeyError as exc:
            raise ValueError(
                "pipeline stage names a device absent from the runtime layout"
            ) from exc
        groups.append(group)
    canonical_groups = tuple(groups)
    flattened = tuple(rank for group in canonical_groups for rank in group)
    if tuple(sorted(flattened)) != tuple(range(len(runtime_layout.devices))):
        raise ValueError(
            "pipeline groups do not cover every runtime-layout rank exactly once"
        )
    if not canonical_groups:
        raise ValueError("pipeline schedule has no stages")
    local_parallel_sizes = {len(group) for group in canonical_groups}
    if len(local_parallel_sizes) != 1 or not next(iter(local_parallel_sizes)):
        raise ValueError("pipeline groups have inconsistent local widths")
    local_parallel_size = len(canonical_groups[0])
    pairs = tuple(
        (
            canonical_groups[stage][slot],
            canonical_groups[(stage + 1) % len(canonical_groups)][slot],
        )
        for stage in range(len(canonical_groups))
        for slot in range(local_parallel_size)
    )
    return canonical_groups, pairs


def _expected_stage_visit_mask(stage_count: int) -> int:
    """Return the exact all-stages-visited mask for the execution plan."""

    if not isinstance(stage_count, int) or isinstance(stage_count, bool):
        raise TypeError("stage count must be an integer")
    if stage_count <= 0 or stage_count >= 32:
        raise ValueError("stage count must fit the signed int32 metadata mask")
    return (1 << stage_count) - 1


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


def _encode_bfloat16_bits(value: np.ndarray) -> np.ndarray:
    """Return a portable little-endian uint16 representation of BF16 bits."""

    observed = np.asarray(value)
    if observed.dtype.name != "bfloat16":
        raise ValueError(
            "BF16 bit encoding requires bfloat16 input, "
            f"got {observed.dtype}"
        )
    native_bits = np.ascontiguousarray(observed).view(np.uint16)
    return native_bits.astype(np.dtype("<u2"), copy=False)


def _validate_layer0_ingredients(
    observed: dict[str, np.ndarray],
    *,
    groups: tuple[tuple[int, ...], ...],
    expected_positions: np.ndarray,
    expected_scores: np.ndarray,
    expected_count: int,
    logical_page_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Validate stage-0 ownership/replication and encode a compact artifact."""

    if tuple(observed) != LAYER0_INGREDIENT_NAMES:
        raise RuntimeError("layer-0 ingredient result names drifted")
    active_rows = np.asarray(groups[0], dtype=np.int32)
    inactive_rows = np.asarray(
        sorted(set(range(len(groups) * len(groups[0]))) - set(groups[0])),
        dtype=np.int32,
    )
    active = {name: np.asarray(value[active_rows]) for name, value in observed.items()}

    sentinel_passed = True
    negative_infinity_names = {
        "selected_scores",
        "sparse_partial_logsumexp",
        "combined_attention_logsumexp",
    }
    negative_one_names = {
        "selected_positions",
        "owner_selected_positions",
    }
    for name, value in observed.items():
        inactive = np.asarray(value[inactive_rows])
        if name in negative_infinity_names:
            sentinel_passed &= bool(np.all(np.isneginf(inactive)))
        elif name in negative_one_names:
            sentinel_passed &= bool(np.all(inactive == -1))
        elif inactive.dtype.name == "bfloat16":
            sentinel_passed &= bool(np.all(_encode_bfloat16_bits(inactive) == 0))
        else:
            sentinel_passed &= bool(np.all(inactive == 0))

    replicated_names = (
        "selected_positions",
        "selected_scores",
        "selected_valid_counts",
        "normalized_input",
        "combined_residual",
        "current_cache_row",
        "combined_attention_output",
        "combined_attention_logsumexp",
        "combined_attention_valid",
        "attention_reduced_update",
        "normalized_mlp",
        "post_attention_residual",
        "dense_reduced_update",
        "next_hidden",
        "layer1_normalized",
        "contract_valid",
    )
    replication_passed = all(
        all(np.array_equal(value[0], row) for row in value[1:])
        for value in (active[name] for name in replicated_names)
    )
    selection_exact = bool(
        np.array_equal(active["selected_positions"][0], expected_positions)
        and np.array_equal(active["selected_scores"][0], expected_scores)
        and int(active["selected_valid_counts"][0, 0]) == expected_count
    )

    owner_positions = active["owner_selected_positions"]
    owner_counts = active["owner_selected_valid_counts"][:, 0].astype(np.int64)
    owner_cache = active["owner_selected_cache_values"]
    local_rows_per_page = logical_page_size // len(groups[0])
    owner_contracts = []
    union: list[int] = []
    for owner, count in enumerate(owner_counts.tolist()):
        live = owner_positions[owner, :count]
        tail = owner_positions[owner, count:]
        owners = (live % logical_page_size) // local_rows_per_page
        union.extend(int(value) for value in live.tolist())
        owner_contracts.append(
            {
                "cache_tail_zero": bool(np.all(owner_cache[owner, count:] == 0)),
                "count": int(count),
                "owner": owner,
                "positions_have_expected_owner": bool(np.all(owners == owner)),
                "positions_strictly_ascending": bool(
                    live.size < 2 or np.all(live[1:] > live[:-1])
                ),
                "tail_is_negative_one": bool(np.all(tail == -1)),
            }
        )
    expected_union = np.sort(expected_positions[:expected_count])
    owner_union_exact = bool(
        len(union) == expected_count
        and np.array_equal(np.sort(np.asarray(union, dtype=np.int32)), expected_union)
    )
    owner_partition_passed = bool(
        owner_union_exact
        and all(
            record["cache_tail_zero"]
            and record["positions_have_expected_owner"]
            and record["positions_strictly_ascending"]
            and record["tail_is_negative_one"]
            for record in owner_contracts
        )
    )
    health_passed = bool(
        np.all(active["contract_valid"])
        and np.all(active["owner_selected_cache_valid"])
        and np.all(active["sparse_partial_valid"])
        and np.all(active["combined_attention_valid"])
    )
    finite_names = tuple(
        name
        for name, value in active.items()
        if (
            value.dtype.name == "bfloat16"
            or np.issubdtype(value.dtype, np.inexact)
        )
        and name not in negative_infinity_names
    )
    finite_passed = bool(
        all(np.all(np.isfinite(active[name])) for name in finite_names)
    )
    selected_cache_nonzero = bool(np.any(owner_cache != 0))

    artifact: dict[str, np.ndarray] = {}
    arrays: dict[str, dict[str, Any]] = {}
    for name, value in active.items():
        if value.dtype.name == "bfloat16":
            artifact_name = f"{name}_bfloat16_bits"
            encoded = _encode_bfloat16_bits(value)
        else:
            artifact_name = name
            encoded = np.ascontiguousarray(value)
        artifact[artifact_name] = encoded
        arrays[artifact_name] = {
            "dtype": str(encoded.dtype),
            "sha256": sha256(encoded.tobytes(order="C")).hexdigest(),
            "shape": list(encoded.shape),
        }
    contract = {
        "active_rows": active_rows.tolist(),
        "arrays": arrays,
        "finite_names": list(finite_names),
        "finite_passed": finite_passed,
        "health_passed": health_passed,
        "inactive_rows_are_sentinel": sentinel_passed,
        "lane_replication_passed": replication_passed,
        "owner_partition_passed": owner_partition_passed,
        "owner_records": owner_contracts,
        "owner_union_exact": owner_union_exact,
        "selected_cache_nonzero": selected_cache_nonzero,
        "selection_exact": selection_exact,
    }
    contract["passed"] = bool(
        finite_passed
        and health_passed
        and sentinel_passed
        and replication_passed
        and owner_partition_passed
        and selected_cache_nonzero
        and selection_exact
    )
    return artifact, contract


def _canonicalize_layer_residual_observation(
    observation: np.ndarray,
    *,
    groups: tuple[tuple[int, ...], ...],
    schedule: Any,
    hidden_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Collapse replicated stage writers into one exact boundary sequence."""

    observed = np.asarray(observation)
    scheduled_layers = tuple(
        (layer.layer_id, stage.assignment.stage_id)
        for stage in schedule.stages
        for layer in stage.layers
    )
    layer_count = len(scheduled_layers)
    layer_to_stage = dict(scheduled_layers)
    flat_ranks = tuple(rank for group in groups for rank in group)
    expected_shape = (
        len(flat_ranks),
        layer_count + 1,
        hidden_size,
    )
    if (
        observed.ndim != 3
        or observed.shape != expected_shape
        or observed.dtype.name != "bfloat16"
        or len(layer_to_stage) != layer_count
        or tuple(sorted(layer_to_stage)) != tuple(range(layer_count))
        or tuple(sorted(flat_ranks)) != tuple(range(len(flat_ranks)))
    ):
        raise RuntimeError(
            "layer residual observer tensor contract drifted: "
            f"expected={expected_shape}/bfloat16 "
            f"observed={observed.shape}/{observed.dtype}"
        )

    canonical = np.empty(
        (layer_count + 1, observed.shape[-1]), dtype=observed.dtype
    )
    boundary_records: list[dict[str, Any]] = []
    lane_mismatches: list[int] = []
    writer_mismatches: list[int] = []
    nonwriter_nonzero: list[int] = []
    all_ranks = set(flat_ranks)
    for boundary in range(layer_count + 1):
        writer_stages = set()
        if boundary > 0:
            writer_stages.add(layer_to_stage[boundary - 1])
        if boundary < layer_count:
            writer_stages.add(layer_to_stage[boundary])
        writer_rows = []
        writer_ranks: set[int] = set()
        for stage_id in sorted(writer_stages):
            group = tuple(groups[stage_id])
            writer_ranks.update(group)
            lanes = observed[list(group), boundary]
            if not all(np.array_equal(lanes[0], lane) for lane in lanes[1:]):
                lane_mismatches.append(boundary)
            writer_rows.append(lanes[0])
        if not writer_rows:
            raise RuntimeError(f"layer boundary {boundary} has no writer")
        if not all(
            np.array_equal(writer_rows[0], row) for row in writer_rows[1:]
        ):
            writer_mismatches.append(boundary)
        canonical[boundary] = writer_rows[0]
        inactive = sorted(all_ranks - writer_ranks)
        if inactive and np.any(observed[inactive, boundary] != 0):
            nonwriter_nonzero.append(boundary)
        boundary_records.append(
            {
                "boundary": boundary,
                "writer_stages": sorted(writer_stages),
            }
        )
    canonical_bits = _encode_bfloat16_bits(canonical)
    digest = sha256(canonical_bits.tobytes()).hexdigest()
    contract = {
        "boundary_count": layer_count + 1,
        "boundary_records": boundary_records,
        "canonical_sha256": digest,
        "dtype": observed.dtype.name,
        "hidden_size": int(observed.shape[-1]),
        "lane_mismatch_boundaries": sorted(set(lane_mismatches)),
        "nonwriter_nonzero_boundaries": sorted(set(nonwriter_nonzero)),
        "passed": bool(
            not lane_mismatches
            and not writer_mismatches
            and not nonwriter_nonzero
        ),
        "storage_byte_order": "little",
        "storage_dtype": canonical_bits.dtype.str,
        "storage_field": "residual_bfloat16_bits",
        "writer_mismatch_boundaries": sorted(set(writer_mismatches)),
    }
    return canonical, contract


def _canonicalize_dsa_internal_observation(
    observation: Any,
    *,
    groups: tuple[tuple[int, ...], ...],
    stage_producer_layer_ids: tuple[tuple[int, ...], ...],
    hidden_size: int,
    q_lora_rank: int,
    num_heads: int,
    head_dim: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collapse exact stage replicas into the ordered full-indexer stream."""

    fields = {
        "normalized_hidden": np.asarray(observation.normalized_hidden),
        "q_a_state": np.asarray(observation.q_a_state),
        "query": np.asarray(observation.query),
        "head_weights": np.asarray(observation.head_weights),
        "current_key": np.asarray(observation.current_key),
        "producer_layer_ids": np.asarray(observation.producer_layer_ids),
    }
    flat_ranks = tuple(rank for group in groups for rank in group)
    maximum_slots = max(len(values) for values in stage_producer_layer_ids)
    shapes = {
        "normalized_hidden": (len(flat_ranks), maximum_slots, hidden_size),
        "q_a_state": (len(flat_ranks), maximum_slots, q_lora_rank),
        "query": (
            len(flat_ranks),
            maximum_slots,
            num_heads,
            head_dim,
        ),
        "head_weights": (len(flat_ranks), maximum_slots, num_heads),
        "current_key": (len(flat_ranks), maximum_slots, head_dim),
        "producer_layer_ids": (len(flat_ranks), maximum_slots),
    }
    expected_dtypes = {
        "normalized_hidden": "bfloat16",
        "q_a_state": "bfloat16",
        "query": "float32",
        "head_weights": "float32",
        "current_key": "float32",
        "producer_layer_ids": "int32",
    }
    for name, value in fields.items():
        if value.shape != shapes[name] or value.dtype.name != expected_dtypes[name]:
            raise RuntimeError(
                f"DSA internal observer {name} contract drifted: "
                f"expected={shapes[name]}/{expected_dtypes[name]} "
                f"observed={value.shape}/{value.dtype}"
            )
    if sorted(flat_ranks) != list(range(len(flat_ranks))) or len(groups) != len(
        stage_producer_layer_ids
    ):
        raise RuntimeError("DSA internal observer stage mapping drifted")

    canonical_lists: dict[str, list[np.ndarray]] = {
        name: [] for name in fields if name != "producer_layer_ids"
    }
    canonical_producers: list[int] = []
    lane_mismatches: list[dict[str, Any]] = []
    padded_slot_mismatches: list[dict[str, Any]] = []
    for stage, (group, producers) in enumerate(
        zip(groups, stage_producer_layer_ids, strict=True)
    ):
        for slot in range(maximum_slots):
            live = slot < len(producers)
            expected_producer = producers[slot] if live else -1
            for name, value in fields.items():
                lanes = value[list(group), slot]
                if not np.all(lanes == lanes[0]):
                    lane_mismatches.append(
                        {"field": name, "slot": slot, "stage": stage}
                    )
            observed_producer = int(fields["producer_layer_ids"][group[0], slot])
            if observed_producer != expected_producer:
                padded_slot_mismatches.append(
                    {
                        "expected_producer": expected_producer,
                        "observed_producer": observed_producer,
                        "slot": slot,
                        "stage": stage,
                    }
                )
            if live:
                canonical_producers.append(expected_producer)
                for name in canonical_lists:
                    canonical_lists[name].append(fields[name][group[0], slot])
            else:
                for name in canonical_lists:
                    if np.any(fields[name][group[0], slot] != 0):
                        padded_slot_mismatches.append(
                            {"field": name, "slot": slot, "stage": stage}
                        )

    canonical = {
        name: np.stack(values, axis=0)
        for name, values in canonical_lists.items()
    }
    canonical["producer_layer_ids"] = np.asarray(
        canonical_producers, dtype=np.int32
    )
    field_records = {}
    for name, value in canonical.items():
        contiguous = np.ascontiguousarray(value)
        field_records[name] = {
            "dtype": value.dtype.name,
            "sha256": sha256(contiguous.tobytes(order="C")).hexdigest(),
            "shape": list(value.shape),
        }
    contract = {
        "event_count": len(canonical_producers),
        "field_records": field_records,
        "lane_mismatches": lane_mismatches,
        "padded_slot_mismatches": padded_slot_mismatches,
        "passed": not lane_mismatches and not padded_slot_mismatches,
        "producer_layer_ids": canonical_producers,
    }
    return canonical, contract


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


def _validate_strategy_nd_canary_hlo(
    optimized_hlo: str,
    *,
    groups: tuple[tuple[int, ...], ...],
) -> dict[str, Any]:
    """Require one exact-shape LP4 gather in the DB533 canary executable."""

    canonical_groups = tuple(tuple(int(rank) for rank in group) for group in groups)
    module = parse_hlo_module(optimized_hlo)
    collectives = module.collectives

    def in_scope(op_name: str | None, scope: str) -> bool:
        return bool(op_name) and scope in op_name.split("/")

    shaped = tuple(
        item
        for item in collectives
        if item.opcode == "all-gather"
        and any(
            shape.dtype == "bf16"
            and shape.dimensions in (
                (8, 1, 6144),
                (1, 8, 1, 6144),
            )
            for shape in item.operand_shapes
        )
        and any(
            shape.dtype == "bf16"
            and shape.dimensions in ((4, 8, 1, 6144), (32, 1, 6144))
            for shape in item.result_shapes
        )
    )
    scoped = tuple(
        item
        for item in shaped
        if in_scope(item.op_name, "greenfield_strategy_nd_row0_canary")
        and in_scope(
            item.op_name,
            "greenfield_strategy_nd_row0_association_gather",
        )
    )
    violations: list[str] = []
    if module.num_partitions not in (None, 32):
        violations.append("StrategyND canary partition count drifted")
    if len(collectives) != 1:
        violations.append(
            "StrategyND canary must contain exactly one collective: "
            f"{len(collectives)}"
        )
    if len(shaped) != 1 or len(scoped) != 1:
        violations.append(
            "StrategyND canary lost its exact scoped partial gather"
        )
    if any(
        item.replica_groups != canonical_groups
        or item.maximum_group_size != 4
        for item in collectives
    ):
        violations.append("StrategyND canary gather escaped exact LP4 groups")
    host_markers = tuple(
        marker
        for marker in (
            "host_callback",
            "outside_compilation",
            "xla_ffi_python_cpu_callback",
            "xla_python_cpu_callback",
        )
        if marker in optimized_hlo
    )
    if host_markers:
        violations.append("StrategyND canary contains host execution")
    return {
        "collective_count": len(collectives),
        "host_markers": list(host_markers),
        "passed": not violations,
        "scoped_shaped_gather_count": len(scoped),
        "shaped_gather_count": len(shaped),
        "violations": violations,
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
        if isinstance(value, (tuple, list)):
            _delete_arrays(tuple(value))
            continue
        try:
            value.delete()
        except (AttributeError, RuntimeError):
            pass


def _protected_short_context_label(context_capacity: int) -> str:
    """Return the evidence label for an admitted short-context capacity."""

    labels = {2048: "2k", 8192: "8k"}
    try:
        return labels[context_capacity]
    except KeyError as exc:
        raise ValueError(
            "protected short decoder capacity must be 2048 or 8192"
        ) from exc


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
    parser.add_argument(
        "--verify-device-roundtrip",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument("--source-runtime-root", type=Path)
    parser.add_argument("--source-runtime-manifest-sha256")
    parser.add_argument("--source-feature-runtime-root", type=Path)
    parser.add_argument("--source-feature-runtime-manifest-sha256")
    parser.add_argument(
        "--feature-source-metadata-only",
        type=int,
        choices=(0, 1),
        default=0,
    )
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
        "--feature-reconstruct-down-fp32",
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
    parser.add_argument(
        "--split-residual-state",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--prefill-index-repair",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--dsa-query-exact-association",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--dsa-head-key-exact-association",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--dsa-score-default-precision",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--main-rope-table",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--dsa-rope-table",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Feed host FP32 DSA rotary rows gathered by position instead of "
            "evaluating cos/sin on the accelerator (default off)."
        ),
    )
    parser.add_argument(
        "--rms-accepted-schedule",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Reduce every decode-step RMS variance over the accepted 32-row "
            "f32[32,W] operand behind an FP32 barrier instead of a single-row "
            "reduce (default off)."
        ),
    )
    parser.add_argument(
        "--pregathered-b512-attention",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--strategy-nd-attention-projection",
        type=int,
        choices=(0, 1),
        default=int(
            os.environ.get(
                "GLM_GREENFIELD_STRATEGY_ND_ATTENTION_PROJECTION", "0"
            )
        ),
    )
    parser.add_argument(
        "--dense-final-layout-convolution",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument("--short-context-oracle-dir", type=Path)
    parser.add_argument("--short-context-oracle-manifest-sha256")
    parser.add_argument("--short-context-dsa-oracle-dir", type=Path)
    parser.add_argument("--short-context-dsa-oracle-manifest-sha256")
    parser.add_argument(
        "--observe-layer-residuals",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--observe-dsa-internals",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--observe-layer0-residual-variants",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--observe-layer0-subshard-variants",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--observe-layer0-attention-schedule-variants",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--observe-layer0-attention-output-association-variants",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--observe-layer0-strategy-nd-row0-association",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument("--strategy-nd-canary-input-bits", type=Path)
    parser.add_argument("--strategy-nd-canary-input-sha256")
    parser.add_argument("--strategy-nd-canary-output-bits", type=Path)
    parser.add_argument("--strategy-nd-canary-output-sha256")
    parser.add_argument(
        "--observe-layer0-ingredients",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument("--layer1-internal-reference-npz", type=Path)
    parser.add_argument("--layer1-internal-reference-sha256")
    parser.add_argument(
        "--dsa-internal-baseline-observation-npz", type=Path
    )
    parser.add_argument("--dsa-internal-baseline-observation-sha256")
    parser.add_argument("--dsa-internal-layer0-reference-npz", type=Path)
    parser.add_argument("--dsa-internal-layer0-reference-sha256")
    parser.add_argument("--layer-residual-position", type=int, default=2044)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    greenfield_run_tag = os.environ.get("GLM_GREENFIELD_RUN_TAG", "").strip()
    context_label = _protected_short_context_label(args.context_capacity)
    if args.feature_output_tile is None:
        args.feature_output_tile = (
            128 if args.runtime_kind == "reference" else 256
        )
    args.feature_fuse_route_weighting = bool(
        args.feature_fuse_route_weighting
    )
    args.feature_reconstruct_down_fp32 = bool(
        args.feature_reconstruct_down_fp32
    )
    args.verify_device_roundtrip = bool(args.verify_device_roundtrip)
    args.complete_token_path = bool(args.complete_token_path)
    args.split_residual_state = bool(args.split_residual_state)
    args.prefill_index_repair = bool(args.prefill_index_repair)
    args.dsa_query_exact_association = bool(
        args.dsa_query_exact_association
    )
    args.dsa_head_key_exact_association = bool(
        args.dsa_head_key_exact_association
    )
    args.dsa_score_default_precision = bool(
        args.dsa_score_default_precision
    )
    args.main_rope_table = bool(args.main_rope_table)
    args.dsa_rope_table = bool(args.dsa_rope_table)
    args.rms_accepted_schedule = bool(args.rms_accepted_schedule)
    args.pregathered_b512_attention = bool(
        args.pregathered_b512_attention
    )
    args.strategy_nd_attention_projection = bool(
        args.strategy_nd_attention_projection
    )
    args.dense_final_layout_convolution = bool(
        args.dense_final_layout_convolution
    )
    args.observe_layer_residuals = bool(args.observe_layer_residuals)
    args.observe_dsa_internals = bool(args.observe_dsa_internals)
    args.observe_layer0_residual_variants = bool(
        args.observe_layer0_residual_variants
    )
    args.observe_layer0_subshard_variants = bool(
        args.observe_layer0_subshard_variants
    )
    args.observe_layer0_attention_schedule_variants = bool(
        args.observe_layer0_attention_schedule_variants
    )
    args.observe_layer0_attention_output_association_variants = bool(
        args.observe_layer0_attention_output_association_variants
    )
    args.observe_layer0_strategy_nd_row0_association = bool(
        args.observe_layer0_strategy_nd_row0_association
    )
    args.observe_layer0_ingredients = bool(args.observe_layer0_ingredients)
    if (
        sum(
            (
                args.observe_layer0_residual_variants,
                args.observe_layer0_subshard_variants,
                args.observe_layer0_attention_schedule_variants,
                args.observe_layer0_attention_output_association_variants,
                args.observe_layer0_strategy_nd_row0_association,
                args.observe_layer0_ingredients,
            )
        )
        > 1
    ):
        raise ValueError(
            "layer-0 diagnostic replays must remain mutually isolated"
        )
    observe_layer0_discriminator = bool(
        args.observe_layer0_residual_variants
        or args.observe_layer0_subshard_variants
        or args.observe_layer0_attention_schedule_variants
        or args.observe_layer0_attention_output_association_variants
        or args.observe_layer0_strategy_nd_row0_association
    )
    if args.dsa_rope_table and (
        args.observe_layer0_ingredients or observe_layer0_discriminator
    ):
        raise ValueError(
            "DSA host rotary table is not plumbed through layer-0 discriminators"
        )
    if args.rms_accepted_schedule and (
        args.observe_layer0_ingredients or observe_layer0_discriminator
    ):
        raise ValueError(
            "RMS accepted schedule is not plumbed through layer-0 discriminators"
        )
    if args.observe_layer0_subshard_variants:
        layer0_discriminator_kind = "virtual_tp32"
    elif args.observe_layer0_attention_schedule_variants:
        layer0_discriminator_kind = "attention_schedule"
    elif args.observe_layer0_attention_output_association_variants:
        layer0_discriminator_kind = "attention_output_association"
    elif args.observe_layer0_strategy_nd_row0_association:
        layer0_discriminator_kind = "strategy_nd_row0_association"
    else:
        layer0_discriminator_kind = "combine_precision"
    main_rope_layer0_discriminator = bool(
        args.observe_layer0_attention_schedule_variants
        or args.observe_layer0_attention_output_association_variants
        or args.observe_layer0_strategy_nd_row0_association
    )
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
    if args.prefill_index_repair and not dsa_oracle_mode:
        raise ValueError(
            "prefill index repair requires the sealed DSA/token oracle"
        )
    if args.prefill_index_repair and args.context_capacity != 8192:
        raise ValueError("prefill index repair is admitted only for protected 8K")
    if args.prefill_index_repair and not args.split_residual_state:
        raise ValueError(
            "prefill index repair requires the accepted split residual state"
        )
    if args.dsa_head_key_exact_association and (
        not args.dsa_query_exact_association
        or not args.prefill_index_repair
    ):
        raise ValueError(
            "exact DSA head/key association requires exact query and the "
            "completed prefill wk materializer"
        )
    # The DSA-internal path is a separately compiled, non-donating executable
    # whose selected output must match a pinned production observation.  The
    # residual observer instead changes the returned recurrent boundary and
    # remains incompatible with the repaired prefill state.
    if args.prefill_index_repair and args.observe_layer_residuals:
        raise ValueError(
            "prefill index repair must remain isolated from residual observation"
        )
    if args.observe_layer_residuals and not dsa_oracle_mode:
        raise ValueError(
            "layer residual observation requires the sealed DSA/token oracle"
        )
    internal_prerequisites = (
        args.dsa_internal_baseline_observation_npz,
        args.dsa_internal_baseline_observation_sha256,
        args.dsa_internal_layer0_reference_npz,
        args.dsa_internal_layer0_reference_sha256,
    )
    internal_prerequisite_presence = tuple(
        value is not None for value in internal_prerequisites
    )
    if (
        any(internal_prerequisite_presence)
        and not all(internal_prerequisite_presence)
    ) or args.observe_dsa_internals != all(internal_prerequisite_presence):
        raise ValueError(
            "DSA internal observation requires both pinned diagnostic inputs"
        )
    if args.observe_dsa_internals and not dsa_oracle_mode:
        raise ValueError(
            "DSA internal observation requires the sealed DSA/token oracle"
        )
    if args.observe_dsa_internals and args.observe_layer_residuals:
        raise ValueError("DSA internal and residual observers must be isolated")
    layer0_variant_reference_fields = (
        args.layer1_internal_reference_npz,
        args.layer1_internal_reference_sha256,
    )
    if observe_layer0_discriminator != all(
        value is not None for value in layer0_variant_reference_fields
    ) or (
        not observe_layer0_discriminator
        and any(value is not None for value in layer0_variant_reference_fields)
    ):
        raise ValueError(
            "layer-0 discriminator requires both pinned layer-1 references"
        )
    strategy_nd_canary_fields = (
        args.strategy_nd_canary_input_bits,
        args.strategy_nd_canary_input_sha256,
        args.strategy_nd_canary_output_bits,
        args.strategy_nd_canary_output_sha256,
    )
    strategy_nd_canary_present = tuple(
        value is not None for value in strategy_nd_canary_fields
    )
    if (
        any(strategy_nd_canary_present)
        and not all(strategy_nd_canary_present)
    ) or args.observe_layer0_strategy_nd_row0_association != all(
        strategy_nd_canary_present
    ):
        raise ValueError(
            "StrategyND row-zero discriminator requires both sealed DB533 "
            "canary arrays and hashes"
        )
    if observe_layer0_discriminator and not dsa_oracle_mode:
        raise ValueError(
            "layer-0 discriminator requires the sealed DSA/token oracle"
        )
    if args.observe_layer0_ingredients and not dsa_oracle_mode:
        raise ValueError(
            "layer-0 ingredient capture requires the sealed DSA/token oracle"
        )
    if args.observe_layer0_ingredients and not greenfield_run_tag:
        raise ValueError(
            "layer-0 ingredient capture requires GLM_GREENFIELD_RUN_TAG"
        )
    if observe_layer0_discriminator and (
        args.observe_dsa_internals or args.observe_layer_residuals
    ):
        raise ValueError(
            "layer-0 discriminator must remain isolated from other observers"
        )
    if args.observe_layer0_ingredients and (
        args.observe_dsa_internals or args.observe_layer_residuals
    ):
        raise ValueError(
            "layer-0 ingredient capture must remain isolated from other observers"
        )
    if observe_layer0_discriminator and not (
        args.prefill_index_repair
        and args.dsa_query_exact_association
        and args.dsa_head_key_exact_association
        and args.dsa_score_default_precision
    ):
        raise ValueError(
            "layer-0 discriminator requires every proven 8K DSA correction"
        )
    if args.observe_layer0_attention_schedule_variants and not (
        args.main_rope_table
        and args.runtime_kind == "pallas_feature_linear"
        and args.complete_token_path
        and args.split_residual_state
    ):
        raise ValueError(
            "layer-0 attention-schedule discriminator requires the protected "
            "table-on production path"
        )
    if args.observe_layer0_attention_output_association_variants and not (
        args.main_rope_table
        and args.runtime_kind == "pallas_feature_linear"
        and args.complete_token_path
        and args.split_residual_state
    ):
        raise ValueError(
            "layer-0 attention-output association discriminator requires "
            "the protected table-on production path"
        )
    if args.observe_layer0_strategy_nd_row0_association and not (
        args.main_rope_table
        and args.runtime_kind == "pallas_feature_linear"
        and args.complete_token_path
        and args.split_residual_state
    ):
        raise ValueError(
            "layer-0 StrategyND row-zero discriminator requires the protected "
            "table-on production path"
        )
    if args.observe_layer0_ingredients and not (
        args.prefill_index_repair
        and args.dsa_query_exact_association
        and args.dsa_head_key_exact_association
        and args.dsa_score_default_precision
        and args.main_rope_table
        and args.runtime_kind == "pallas_feature_linear"
        and args.complete_token_path
        and args.split_residual_state
    ):
        raise ValueError(
            "layer-0 ingredient capture requires the proven table-on 8K "
            "production path"
        )
    if args.pregathered_b512_attention and not (
        args.context_capacity == 8192
        and dsa_oracle_mode
        and args.prefill_index_repair
        and args.dsa_query_exact_association
        and args.dsa_head_key_exact_association
        and args.dsa_score_default_precision
        and args.main_rope_table
        and args.runtime_kind == "pallas_feature_linear"
        and args.complete_token_path
        and args.split_residual_state
        and not observe_layer0_discriminator
        and not args.observe_layer0_ingredients
    ):
        raise ValueError(
            "pregathered-B512 attention requires the protected 8K exact "
            "production path without a layer-0 diagnostic"
        )
    if args.strategy_nd_attention_projection and not (
        args.pregathered_b512_attention
        and args.context_capacity == 8192
        and dsa_oracle_mode
        and not observe_layer0_discriminator
        and not args.observe_layer0_ingredients
    ):
        raise ValueError(
            "StrategyND attention projection requires the protected 8K "
            "pregathered-B512 production path"
        )
    if args.dense_final_layout_convolution and not (
        args.strategy_nd_attention_projection
        and args.context_capacity == 8192
        and dsa_oracle_mode
        and not observe_layer0_discriminator
        and not args.observe_layer0_ingredients
    ):
        raise ValueError(
            "dense final-layout convolution requires the protected 8K "
            "StrategyND production path"
        )
    if args.num_processes != 8 or not 0 <= args.process_id < 8:
        raise ValueError("protected decoder compile requires process ids 0..7")
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
    dsa_internal_baseline = None
    dsa_internal_layer0_reference = None
    layer1_normalized_hidden_reference = None
    strategy_nd_canary_input_bits = None
    strategy_nd_canary_output_bits = None
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
            raise ValueError(
                "DSA oracle positions do not align with the protected prompt"
            )
        dsa_steps = int(decode_positions.size)
        if args.warmup + args.iterations + args.trace_steps != dsa_steps:
            raise ValueError(
                "production warmup/timing/trace window must cover the exact DSA "
                "oracle window"
            )
        if 1 + dsa_steps > oracle_generated_token_ids.size:
            raise ValueError("DSA observer token window exceeds the token oracle")
        if args.observe_layer_residuals and args.layer_residual_position not in set(
            int(value) for value in decode_positions.tolist()
        ):
            raise ValueError(
                "layer residual position is outside the sealed observer window"
            )
    if args.observe_dsa_internals:
        assert args.dsa_internal_baseline_observation_npz is not None
        assert args.dsa_internal_baseline_observation_sha256 is not None
        assert args.dsa_internal_layer0_reference_npz is not None
        assert args.dsa_internal_layer0_reference_sha256 is not None
        if _sha256_file(args.dsa_internal_baseline_observation_npz) != (
            args.dsa_internal_baseline_observation_sha256
        ):
            raise ValueError("DSA internal baseline observation hash drifted")
        if _sha256_file(args.dsa_internal_layer0_reference_npz) != (
            args.dsa_internal_layer0_reference_sha256
        ):
            raise ValueError("DSA internal layer-0 reference hash drifted")
        with np.load(
            args.dsa_internal_baseline_observation_npz, allow_pickle=False
        ) as payload:
            if "observation" not in payload.files:
                raise ValueError("DSA internal baseline has no observation")
            dsa_internal_baseline = np.asarray(
                payload["observation"], dtype=np.int32
            )
        with np.load(
            args.dsa_internal_layer0_reference_npz, allow_pickle=False
        ) as payload:
            required = {
                "actual__normalized_hidden",
                "actual__q_a_state",
                "actual__query",
                "actual__head_weights",
                "actual__current_key",
            }
            if not required.issubset(payload.files):
                raise ValueError("DSA internal layer-0 reference is incomplete")
            dsa_internal_layer0_reference = {
                name.removeprefix("actual__"): np.asarray(payload[name])
                for name in sorted(required)
            }
    if observe_layer0_discriminator:
        assert args.layer1_internal_reference_npz is not None
        assert args.layer1_internal_reference_sha256 is not None
        if _sha256_file(args.layer1_internal_reference_npz) != (
            args.layer1_internal_reference_sha256
        ):
            raise ValueError("layer-1 internal reference hash drifted")
        with np.load(
            args.layer1_internal_reference_npz, allow_pickle=False
        ) as payload:
            required = {
                "accepted__normalized_hidden",
                "event_index",
                "layer_id",
                "position",
            }
            if not required.issubset(payload.files):
                raise ValueError("layer-1 internal reference is incomplete")
            if int(payload["event_index"]) != 1 or int(payload["layer_id"]) != 1:
                raise ValueError("layer-1 internal reference identity drifted")
            if int(payload["position"]) != int(
                dsa_oracle_tensors["decode_positions"][0]
            ):
                raise ValueError("layer-1 internal reference position drifted")
            layer1_normalized_hidden_reference = np.asarray(
                payload["accepted__normalized_hidden"], dtype=np.uint16
            )
            if layer1_normalized_hidden_reference.shape != (6144,):
                raise ValueError("layer-1 normalized-hidden shape drifted")
    if args.observe_layer0_strategy_nd_row0_association:
        assert args.strategy_nd_canary_input_bits is not None
        assert args.strategy_nd_canary_input_sha256 is not None
        assert args.strategy_nd_canary_output_bits is not None
        assert args.strategy_nd_canary_output_sha256 is not None
        if _sha256_file(args.strategy_nd_canary_input_bits) != (
            args.strategy_nd_canary_input_sha256
        ):
            raise ValueError("DB533 StrategyND canary input hash drifted")
        if _sha256_file(args.strategy_nd_canary_output_bits) != (
            args.strategy_nd_canary_output_sha256
        ):
            raise ValueError("DB533 StrategyND canary output hash drifted")
        strategy_nd_canary_input_bits = np.load(
            args.strategy_nd_canary_input_bits,
            allow_pickle=False,
        )
        strategy_nd_canary_output_bits = np.load(
            args.strategy_nd_canary_output_bits,
            allow_pickle=False,
        )
        if (
            strategy_nd_canary_input_bits.shape != (32, 32, 6144)
            or strategy_nd_canary_input_bits.dtype != np.dtype("<u2")
            or strategy_nd_canary_output_bits.shape != (32, 32, 6144)
            or strategy_nd_canary_output_bits.dtype != np.dtype("<u2")
        ):
            raise ValueError("DB533 StrategyND canary array contract drifted")
    if args.runtime_kind in ("pallas_feature", "pallas_feature_linear") and (
        args.source_runtime_root is None
        or args.source_runtime_manifest_sha256 is None
    ):
        raise ValueError(
            "feature runtime requires its source runtime root and manifest"
        )
    if (args.source_feature_runtime_root is None) != (
        args.source_feature_runtime_manifest_sha256 is None
    ):
        raise ValueError(
            "source feature runtime root and manifest must be provided together"
        )
    if args.source_feature_runtime_root is not None and (
        args.runtime_kind not in ("pallas_feature", "pallas_feature_linear")
        or not args.feature_source_metadata_only
    ):
        raise ValueError(
            "a source feature runtime requires a feature backend and metadata-only parent lineage"
        )
    if args.feature_source_metadata_only and args.runtime_kind not in (
        "pallas_feature",
        "pallas_feature_linear",
    ):
        raise ValueError(
            "metadata-only source lineage requires a feature runtime"
        )
    if args.runtime_kind == "reference" and args.feature_output_tile != 128:
        raise ValueError(
            "a non-default feature output tile requires a feature runtime"
        )
    if args.runtime_kind == "reference" and args.feature_fuse_route_weighting:
        raise ValueError(
            "feature route-weight fusion requires a feature runtime"
        )
    if (
        args.runtime_kind == "reference"
        and args.feature_reconstruct_down_fp32
    ):
        raise ValueError(
            "feature FP32 reconstruction requires a feature runtime"
        )
    if (
        args.feature_reconstruct_down_fp32
        and args.feature_fuse_route_weighting
    ):
        raise ValueError(
            "feature FP32 reconstruction is incompatible with fused route "
            "weighting"
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
    attention_projection_layout = runtime_manifest.get(
        "attention_projection_layout",
        SEPARATE_QKV_A_RUNTIME_LAYOUT,
    )
    if attention_projection_layout == SEPARATE_QKV_A_RUNTIME_LAYOUT:
        attention_projection_backend = "separate"
    elif attention_projection_layout == FUSED_QKV_A_N82_RUNTIME_LAYOUT:
        attention_projection_backend = "fused_n82_convolution"
    else:
        raise ValueError("runtime attention projection layout is unknown")
    if args.runtime_kind == "reference":
        if attention_projection_backend != "separate":
            raise ValueError(
                "reference runtime cannot consume fused qkv-a state"
            )
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
        hlo_backend_contract = {
            "PP8_LP4": "tpu_v4_pp8_reference",
            "PP16_LP2": "tpu_v4_pp16_reference",
        }[execution_plan.name.value]
    else:
        context_args = SimpleNamespace(
            source_checkpoint_root=args.source_checkpoint_root,
            source_packed_manifest_sha256=args.source_packed_manifest_sha256,
            source_runtime_root=args.source_runtime_root,
            source_runtime_manifest_sha256=(
                args.source_runtime_manifest_sha256
            ),
            source_feature_runtime_root=args.source_feature_runtime_root,
            source_feature_runtime_manifest_sha256=(
                args.source_feature_runtime_manifest_sha256
            ),
            destination=runtime_manifest["destination"],
            fused_qkv_a=(
                attention_projection_backend == "fused_n82_convolution"
            ),
            dense_convolution=args.dense_final_layout_convolution,
            source_metadata_only=bool(args.feature_source_metadata_only),
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
        if args.dense_final_layout_convolution != (
            runtime_manifest.get("dense_projection_layout")
            == FINAL_DENSE_CONVOLUTION_RUNTIME_LAYOUT
        ):
            raise ValueError(
                "dense final-layout flag and runtime manifest disagree"
            )
        sparse_moe_backend = "pallas_feature"
        linear_backend = (
            "pallas" if args.runtime_kind == "pallas_feature_linear" else "reference"
        )
        hlo_backend_contract = {
            ("PP8_LP4", False): "tpu_v4_pp8_pallas_feature",
            ("PP8_LP4", True): "tpu_v4_pp8_pallas_feature_linear",
            ("PP16_LP2", False): "tpu_v4_pp16_pallas_feature",
            ("PP16_LP2", True): "tpu_v4_pp16_pallas_feature_linear",
        }[(execution_plan.name.value, linear_backend == "pallas")]
    # DB499 proves that accepted M=1 DSA queries require a complete local
    # FP32 wq_b owner shard. Keep every other projection on its selected
    # runtime backend; only this exactness boundary uses the reference path.
    dsa_query_backend = "reference"
    if args.dsa_query_exact_association and (
        attention_projection_backend != "fused_n82_convolution"
    ):
        raise ValueError(
            "exact DSA query association requires the proven fused qkv-a "
            "runtime"
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
    materialized_prefill_index_weights: tuple[Any, ...] | None = None
    materialized_dsa_query_weights: tuple[Any, ...] | None = None
    dsa_query_weight_aliases: tuple[tuple[Any, ...], ...] | None = None
    main_rope_table = None
    main_rope_table_local_hashes: list[str] = []
    dsa_rope_table = None
    dsa_rope_table_local_hashes: list[str] = []
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
        groups, pairs = _runtime_pipeline_groups(
            schedule,
            pack_context.layout,
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
            feature_reconstruct_down_fp32=(
                args.feature_reconstruct_down_fp32
            ),
            linear_backend=linear_backend,
            dsa_query_backend=dsa_query_backend,
            dsa_query_exact_association=(
                args.dsa_query_exact_association
            ),
            dsa_head_key_exact_association=(
                args.dsa_head_key_exact_association
            ),
            dsa_score_default_precision=(
                args.dsa_score_default_precision
            ),
            main_rope_table_enabled=args.main_rope_table,
            dsa_rope_table_enabled=args.dsa_rope_table,
            rms_accepted_schedule=args.rms_accepted_schedule,
            pregathered_b512_attention=(
                args.pregathered_b512_attention
            ),
            strategy_nd_attention_projection=(
                args.strategy_nd_attention_projection
            ),
            dense_final_layout_convolution=(
                args.dense_final_layout_convolution
            ),
            attention_projection_backend=attention_projection_backend,
            complete_token_path=args.complete_token_path,
            build_layer0_residual_discriminator=(
                observe_layer0_discriminator
            ),
            layer0_residual_discriminator_kind=layer0_discriminator_kind,
            build_layer0_ingredients_observer=(
                args.observe_layer0_ingredients
            ),
            split_residual_state=args.split_residual_state,
        )
        dsa_observer = None
        if dsa_oracle_mode and not (
            observe_layer0_discriminator or args.observe_layer0_ingredients
        ):
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
                feature_reconstruct_down_fp32=(
                    args.feature_reconstruct_down_fp32
                ),
                linear_backend=linear_backend,
                dsa_query_backend=dsa_query_backend,
                dsa_query_exact_association=(
                    args.dsa_query_exact_association
                ),
                dsa_head_key_exact_association=(
                    args.dsa_head_key_exact_association
                ),
                dsa_score_default_precision=(
                    args.dsa_score_default_precision
                ),
                main_rope_table_enabled=args.main_rope_table,
                dsa_rope_table_enabled=args.dsa_rope_table,
            rms_accepted_schedule=args.rms_accepted_schedule,
                pregathered_b512_attention=(
                    args.pregathered_b512_attention
                ),
                strategy_nd_attention_projection=(
                    args.strategy_nd_attention_projection
                ),
                dense_final_layout_convolution=(
                    args.dense_final_layout_convolution
                ),
                attention_projection_backend=(
                    attention_projection_backend
                ),
                complete_token_path=True,
                observe_dsa_events=True,
                observe_dsa_internals=args.observe_dsa_internals,
                observe_layer_residuals=args.observe_layer_residuals,
                split_residual_state=args.split_residual_state,
            )
        prefill = None
        prefill_decoder = None
        if oracle_mode:
            assert prompt_token_ids is not None
            prefill_decoder = decoder
            if args.prefill_index_repair:
                prefill_decoder = build_decoder_step_program(
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
                    feature_reconstruct_down_fp32=(
                        args.feature_reconstruct_down_fp32
                    ),
                    linear_backend=linear_backend,
                    dsa_query_backend=dsa_query_backend,
                    dsa_query_exact_association=(
                        args.dsa_query_exact_association
                    ),
                    dsa_head_key_exact_association=(
                        args.dsa_head_key_exact_association
                    ),
                    dsa_score_default_precision=(
                        args.dsa_score_default_precision
                    ),
                    main_rope_table_enabled=args.main_rope_table,
                    dsa_rope_table_enabled=args.dsa_rope_table,
            rms_accepted_schedule=args.rms_accepted_schedule,
                    pregathered_b512_attention=(
                        args.pregathered_b512_attention
                    ),
                    strategy_nd_attention_projection=(
                        args.strategy_nd_attention_projection
                    ),
                    dense_final_layout_convolution=(
                        args.dense_final_layout_convolution
                    ),
                    attention_projection_backend=(
                        attention_projection_backend
                    ),
                    complete_token_path=True,
                    observe_prefill_index_inputs=True,
                    split_residual_state=args.split_residual_state,
                )
            prefill = build_teacher_forced_prefill_program(
                prefill_decoder,
                prompt_length=int(prompt_token_ids.size),
            )
        multihost_utils.sync_global_devices("greenfield-short-decoder-load-start")
        load_started = time.monotonic()
        loaded = load_runtime_checkpoint(
            verified_runtime,
            expectation,
            pack_context.layout,
            decoder.mesh,
            verify_device_roundtrip=args.verify_device_roundtrip,
        )
        load_seconds = time.monotonic() - load_started
        multihost_utils.sync_global_devices("greenfield-short-decoder-load-end")

        if args.main_rope_table:
            if (
                decoder.main_rope_table_host is None
                or decoder.main_rope_table_sha256 is None
                or decoder.main_rope_table_bytes_per_device <= 0
            ):
                raise RuntimeError("decoder main-RoPE table asset is unavailable")
            for related in (dsa_observer, prefill_decoder):
                if related is not None and (
                    not related.main_rope_table_enabled
                    or related.main_rope_table_sha256
                    != decoder.main_rope_table_sha256
                    or related.main_rope_table_host is None
                    or related.main_rope_table_host.shape
                    != decoder.main_rope_table_host.shape
                    or related.main_rope_table_bytes_per_device
                    != decoder.main_rope_table_bytes_per_device
                ):
                    raise RuntimeError("decoder main-RoPE table identities disagree")
            main_rope_table = jax.device_put(
                decoder.main_rope_table_host,
                NamedSharding(decoder.mesh, P()),
            )
            main_rope_table.block_until_ready()
            main_rope_table_local_hashes = [
                sha256(
                    np.ascontiguousarray(
                        np.asarray(jax.device_get(shard.data))
                    ).view(np.uint16).tobytes()
                ).hexdigest()
                for shard in main_rope_table.addressable_shards
            ]
            if set(main_rope_table_local_hashes) != {
                decoder.main_rope_table_sha256
            }:
                raise RuntimeError("device main-RoPE table identity drifted")
        elif any(
            value is not None
            for value in (
                decoder.main_rope_table_host,
                decoder.main_rope_table_sha256,
            )
        ) or decoder.main_rope_table_bytes_per_device != 0:
            raise RuntimeError("default decoder materialized a main-RoPE table")

        if args.dsa_rope_table:
            if (
                decoder.dsa_rope_table_host is None
                or decoder.dsa_rope_table_sha256 is None
                or decoder.dsa_rope_table_bytes_per_device <= 0
            ):
                raise RuntimeError("decoder DSA host rotary table asset is unavailable")
            for related in (dsa_observer, prefill_decoder):
                if related is not None and (
                    not related.dsa_rope_table_enabled
                    or related.dsa_rope_table_sha256
                    != decoder.dsa_rope_table_sha256
                    or related.dsa_rope_table_host is None
                    or related.dsa_rope_table_host.shape
                    != decoder.dsa_rope_table_host.shape
                    or related.dsa_rope_table_bytes_per_device
                    != decoder.dsa_rope_table_bytes_per_device
                ):
                    raise RuntimeError(
                        "decoder DSA host rotary table identities disagree"
                    )
            dsa_rope_table = jax.device_put(
                decoder.dsa_rope_table_host,
                NamedSharding(decoder.mesh, P()),
            )
            dsa_rope_table.block_until_ready()
            dsa_rope_table_local_hashes = [
                sha256(
                    np.ascontiguousarray(
                        np.asarray(jax.device_get(shard.data))
                    ).astype(np.float32).tobytes()
                ).hexdigest()
                for shard in dsa_rope_table.addressable_shards
            ]
            if set(dsa_rope_table_local_hashes) != {
                decoder.dsa_rope_table_sha256
            }:
                raise RuntimeError("device DSA host rotary table identity drifted")
        elif any(
            value is not None
            for value in (
                decoder.dsa_rope_table_host,
                decoder.dsa_rope_table_sha256,
            )
        ) or decoder.dsa_rope_table_bytes_per_device != 0:
            raise RuntimeError("default decoder materialized a DSA host rotary table")

        dsa_query_materialization_compile_seconds = None
        dsa_query_materialization_execute_seconds = None
        dsa_query_materialization_hlo_sha256 = None
        fleet_dsa_query_materialization_hlo_hashes = None
        dsa_query_materialization_hlo_contract = None
        dsa_query_materialization_state = None
        if args.dsa_query_exact_association:
            query_materializer = decoder.materialize_dsa_query_weights_fp32
            if query_materializer is None:
                raise RuntimeError("exact DSA query materializer is unavailable")
            names = decoder.dsa_query_weight_names
            if len(names) != decoder.config.maximum_full_indexer_slots:
                raise RuntimeError("DSA query weight-name count drifted")
            raw_query_bits = tuple(
                loaded.weights[bits_name] for bits_name, _ in names
            )
            raw_query_scales = tuple(
                loaded.weights[scale_name] for _, scale_name in names
            )
            multihost_utils.sync_global_devices(
                "greenfield-dsa-query-materialization-compile-start"
            )
            started = time.monotonic()
            lowered_query_materializer = jax.jit(query_materializer).lower(
                raw_query_bits, raw_query_scales
            )
            compiled_query_materializer = (
                lowered_query_materializer.compile()
            )
            dsa_query_materialization_compile_seconds = (
                time.monotonic() - started
            )
            optimized_query_materializer_hlo = (
                compiled_query_materializer.as_text()
            )
            dsa_query_materialization_hlo_sha256 = sha256(
                optimized_query_materializer_hlo.encode("utf-8")
            ).hexdigest()
            fleet_dsa_query_materialization_hlo_hashes = _fleet_digest(
                multihost_utils,
                dsa_query_materialization_hlo_sha256,
                num_processes=args.num_processes,
            )
            dsa_query_materialization_hlo_contract = (
                validate_dsa_query_weight_materializer_hlo(
                    optimized_query_materializer_hlo,
                    full_indexer_slots=(
                        decoder.config.maximum_full_indexer_slots
                    ),
                    local_output_width=(
                        decoder.config.dsa_indexer_heads
                        * decoder.config.index_key_width
                        // decoder.config.local_parallel_size
                    ),
                    q_lora_rank=execution_plan.geometry.q_lora_rank,
                    total_devices=decoder.config.total_devices,
                )
            )
            if not dsa_query_materialization_hlo_contract["passed"]:
                raise RuntimeError(
                    "DSA query materializer HLO contract failed: "
                    f"{dsa_query_materialization_hlo_contract['violations']}"
                )
            multihost_utils.sync_global_devices(
                "greenfield-dsa-query-materialization-execute-start"
            )
            started = time.monotonic()
            materialized_dsa_query_weights = tuple(
                compiled_query_materializer(
                    raw_query_bits, raw_query_scales
                )
            )
            jax.block_until_ready(materialized_dsa_query_weights)
            dsa_query_materialization_execute_seconds = (
                time.monotonic() - started
            )
            multihost_utils.sync_global_devices(
                "greenfield-dsa-query-materialization-execute-end"
            )
            dsa_query_weight_aliases = (
                materialized_dsa_query_weights,
            ) * 4
            local_query_records = []
            for slot, value in enumerate(materialized_dsa_query_weights):
                for shard in value.addressable_shards:
                    host = np.ascontiguousarray(
                        np.asarray(jax.device_get(shard.data), dtype=np.float32)
                    )
                    expected_shape = (
                        1,
                        decoder.config.dsa_indexer_heads
                        * decoder.config.index_key_width
                        // decoder.config.local_parallel_size,
                        execution_plan.geometry.q_lora_rank,
                    )
                    if host.shape != expected_shape or not np.all(
                        np.isfinite(host)
                    ):
                        raise RuntimeError(
                            "materialized DSA query owner shape/finite "
                            "contract failed"
                        )
                    local_query_records.append(
                        {
                            "byte_count": int(host.nbytes),
                            "device_id": int(shard.device.id),
                            "sha256": sha256(
                                host.tobytes(order="C")
                            ).hexdigest(),
                            "slot": slot,
                        }
                    )
            dsa_query_materialization_state = {
                "input_alias_count": 4,
                "local_shards": local_query_records,
                "materialized_bytes_per_device": sum(
                    record["byte_count"] for record in local_query_records
                )
                // jax.local_device_count(),
                "slot_count": len(materialized_dsa_query_weights),
                "source": "completed_stage_local_raw_fp8_to_fp32",
            }
            if jax.process_index() == 0:
                hlo_dir = args.output.parent / "hlo"
                hlo_dir.mkdir(parents=True, exist_ok=True)
                with gzip.open(
                    hlo_dir
                    / "dsa_query_weight_materializer.optimized_hlo.txt.gz",
                    "wt",
                    encoding="utf-8",
                ) as stream:
                    stream.write(optimized_query_materializer_hlo)
                _atomic_json(
                    hlo_dir
                    / "dsa_query_weight_materializer.hlo_contract.json",
                    dsa_query_materialization_hlo_contract,
                )
            del optimized_query_materializer_hlo

        prefill_wk_materialization_compile_seconds = None
        prefill_wk_materialization_execute_seconds = None
        prefill_wk_materialization_hlo_sha256 = None
        fleet_prefill_wk_materialization_hlo_hashes = None
        prefill_wk_materialization_hlo_contract = None
        prefill_wk_materialization_state = None
        if args.prefill_index_repair:
            assert prefill_decoder is not None
            bf16_decoder = prefill_decoder.decode_prefill_index_weights_bf16
            fp32_promoter = prefill_decoder.promote_prefill_index_weights_fp32
            if bf16_decoder is None or fp32_promoter is None:
                raise RuntimeError(
                    "prefill index repair split weight adapter is unavailable"
                )
            names = prefill_decoder.prefill_index_weight_names
            if len(names) != prefill_decoder.config.maximum_full_indexer_slots:
                raise RuntimeError(
                    "prefill index repair weight-name count drifted"
                )
            raw_wk_bits = tuple(
                loaded.weights[bits_name] for bits_name, _ in names
            )
            raw_wk_scales = tuple(
                loaded.weights[scale_name] for _, scale_name in names
            )
            decoder_inputs = (raw_wk_bits, raw_wk_scales)
            multihost_utils.sync_global_devices(
                "greenfield-prefill-wk-materialization-compile-start"
            )
            materializer_compile_started = time.monotonic()
            lowered_bf16_decoder = jax.jit(bf16_decoder).lower(
                *decoder_inputs
            )
            compiled_bf16_decoder = lowered_bf16_decoder.compile()
            optimized_bf16_decoder_hlo = compiled_bf16_decoder.as_text()
            bf16_decoder_contract = (
                validate_prefill_index_weight_materialization_hlo(
                    optimized_bf16_decoder_hlo,
                    decoder=prefill_decoder,
                    phase="decode_bf16",
                )
            )
            if not bf16_decoder_contract["passed"]:
                raise RuntimeError(
                    "prefill wk BF16 decode HLO contract failed before "
                    f"execution: {bf16_decoder_contract['violations']}"
                )
            multihost_utils.sync_global_devices(
                "greenfield-prefill-wk-bf16-decode-execute-start"
            )
            bf16_execute_started = time.monotonic()
            decoded_prefill_index_weights = tuple(
                compiled_bf16_decoder(*decoder_inputs)
            )
            jax.block_until_ready(decoded_prefill_index_weights)
            bf16_execute_seconds = time.monotonic() - bf16_execute_started
            multihost_utils.sync_global_devices(
                "greenfield-prefill-wk-bf16-decode-execute-end"
            )

            lowered_fp32_promoter = jax.jit(fp32_promoter).lower(
                decoded_prefill_index_weights
            )
            compiled_fp32_promoter = lowered_fp32_promoter.compile()
            optimized_fp32_promoter_hlo = compiled_fp32_promoter.as_text()
            fp32_promoter_contract = (
                validate_prefill_index_weight_materialization_hlo(
                    optimized_fp32_promoter_hlo,
                    decoder=prefill_decoder,
                    phase="promote_fp32",
                )
            )
            if not fp32_promoter_contract["passed"]:
                raise RuntimeError(
                    "prefill wk FP32 promotion HLO contract failed before "
                    f"execution: {fp32_promoter_contract['violations']}"
                )
            prefill_wk_materialization_compile_seconds = (
                time.monotonic() - materializer_compile_started
            )
            multihost_utils.sync_global_devices(
                "greenfield-prefill-wk-materialization-compile-end"
            )
            fp32_execute_started = time.monotonic()
            materialized_prefill_index_weights = tuple(
                compiled_fp32_promoter(decoded_prefill_index_weights)
            )
            jax.block_until_ready(materialized_prefill_index_weights)
            prefill_wk_materialization_execute_seconds = (
                bf16_execute_seconds
                + time.monotonic()
                - fp32_execute_started
            )
            multihost_utils.sync_global_devices(
                "greenfield-prefill-wk-materialization-execute-end"
            )
            bf16_hlo_sha256 = sha256(
                optimized_bf16_decoder_hlo.encode("utf-8")
            ).hexdigest()
            fp32_hlo_sha256 = sha256(
                optimized_fp32_promoter_hlo.encode("utf-8")
            ).hexdigest()
            prefill_wk_materialization_hlo_sha256 = sha256(
                f"{bf16_hlo_sha256}:{fp32_hlo_sha256}".encode()
            ).hexdigest()
            fleet_prefill_wk_materialization_hlo_hashes = _fleet_digest(
                multihost_utils,
                prefill_wk_materialization_hlo_sha256,
                num_processes=args.num_processes,
            )
            prefill_wk_materialization_hlo_contract = {
                "bf16_decode": bf16_decoder_contract,
                "bf16_decode_hlo_sha256": bf16_hlo_sha256,
                "fp32_promote": fp32_promoter_contract,
                "fp32_promote_hlo_sha256": fp32_hlo_sha256,
                "passed": bool(
                    bf16_decoder_contract["passed"]
                    and fp32_promoter_contract["passed"]
                ),
            }
            if jax.process_index() == 0:
                hlo_dir = args.output.parent / "hlo"
                hlo_dir.mkdir(parents=True, exist_ok=True)
                for filename, hlo in (
                    ("prefill_index_wk_decode_bf16", optimized_bf16_decoder_hlo),
                    ("prefill_index_wk_promote_fp32", optimized_fp32_promoter_hlo),
                ):
                    with gzip.open(
                        hlo_dir / f"{filename}.optimized_hlo.txt.gz",
                        "wt",
                        encoding="utf-8",
                    ) as stream:
                        stream.write(hlo)
                _atomic_json(
                    hlo_dir / "prefill_index_wk_materialization.hlo_contract.json",
                    prefill_wk_materialization_hlo_contract,
                )
            del optimized_bf16_decoder_hlo, optimized_fp32_promoter_hlo
            local_materialized_records = []
            for slot, value in enumerate(materialized_prefill_index_weights):
                for shard in value.addressable_shards:
                    host = np.ascontiguousarray(
                        np.asarray(jax.device_get(shard.data), dtype=np.float32)
                    )
                    if host.shape != (
                        1,
                        execution_plan.geometry.dsa_indexer_head_dim,
                        execution_plan.geometry.hidden_size,
                    ) or not np.all(np.isfinite(host)):
                        raise RuntimeError(
                            "materialized prefill wk shape/finite contract failed"
                        )
                    local_materialized_records.append(
                        {
                            "byte_count": int(host.nbytes),
                            "byte_sum": int(host.view(np.uint8).sum(dtype=np.uint64)),
                            "device_id": int(shard.device.id),
                            "sha256": sha256(host.tobytes(order="C")).hexdigest(),
                            "slot": slot,
                        }
                    )
            prefill_wk_materialization_state = {
                "local_shards": local_materialized_records,
                "materialized_bytes_per_device": (
                    prefill_decoder.config.maximum_full_indexer_slots
                    * execution_plan.geometry.dsa_indexer_head_dim
                    * execution_plan.geometry.hidden_size
                    * np.dtype(np.float32).itemsize
                ),
                "source": "completed_stage_local_raw_fp8_to_bf16_to_fp32",
            }

        total_devices = decoder.config.total_devices
        kv_shape = state_layout.stages[0].padded_kv_cache_shape
        index_shape = state_layout.stages[0].padded_indexer_cache_shape
        residual_shape = (
            (
                total_devices,
                2,
                1,
                execution_plan.geometry.hidden_size,
            )
            if args.split_residual_state
            else (
                total_devices,
                1,
                execution_plan.geometry.hidden_size,
            )
        )
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
                if args.split_residual_state:
                    value[:, 0, ...] = initial_row
                else:
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

        state_spec_offset = 2 if args.dsa_query_exact_association else 1
        residual = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[state_spec_offset],
            residual_shape,
            residual_builder,
        )
        kv = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[state_spec_offset + 1],
            kv_global_shape,
            zero_bf16,
        )
        index = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[state_spec_offset + 2],
            index_global_shape,
            zero_bf16,
        )
        metadata = _make_global_array(
            jax,
            decoder.mesh,
            decoder.input_specs[state_spec_offset + 3],
            metadata_shape,
            metadata_builder,
        )
        token = None
        if args.complete_token_path:
            token = _make_global_array(
                jax,
                decoder.mesh,
                decoder.input_specs[state_spec_offset + 4],
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
            value
            for value in (token, prompt, main_rope_table, dsa_rope_table)
            if value is not None
        ) + (
            tuple(materialized_prefill_index_weights)
            if materialized_prefill_index_weights is not None
            else ()
        ) + (
            tuple(materialized_dsa_query_weights)
            if materialized_dsa_query_weights is not None
            else ()
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
        if args.dsa_query_exact_association:
            if dsa_query_weight_aliases is None:
                raise RuntimeError("exact DSA query aliases were not built")
            exact_dsa_weights: Any = dsa_query_weight_aliases
            if args.dsa_head_key_exact_association:
                if materialized_prefill_index_weights is None:
                    raise RuntimeError(
                        "exact DSA head/key execution lost FP32 wk owners"
                    )
                exact_dsa_weights = (
                    dsa_query_weight_aliases,
                    materialized_prefill_index_weights,
                )
            common_inputs = (
                loaded.weights,
                exact_dsa_weights,
                residual,
                kv,
                index,
                metadata,
            )
        runtime_prefix = (
            common_inputs[:2]
            if args.dsa_query_exact_association
            else common_inputs[:1]
        )
        donation_shift = int(args.dsa_query_exact_association)
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
            donate_argnums = tuple(
                value + donation_shift for value in (1, 2, 3, 4, 5)
            )
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
            donate_argnums = tuple(
                value + donation_shift for value in (1, 2, 3, 4)
            )
        if args.main_rope_table:
            if main_rope_table is None:
                raise RuntimeError("main-RoPE table input was not materialized")
            inputs = (*inputs, main_rope_table)
        if args.dsa_rope_table:
            if dsa_rope_table is None:
                raise RuntimeError(
                    "DSA host rotary table input was not materialized"
                )
            inputs = (*inputs, dsa_rope_table)
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
            if args.prefill_index_repair:
                if materialized_prefill_index_weights is None:
                    raise RuntimeError(
                        "prefill repair lost its materialized wk inputs"
                    )
                prefill_inputs = (
                    *prefill_inputs,
                    materialized_prefill_index_weights,
                )
            if args.dsa_query_exact_association:
                if dsa_query_weight_aliases is None:
                    raise RuntimeError(
                        "prefill lost exact DSA query aliases"
                    )
                if not args.prefill_index_repair:
                    prefill_inputs = (*prefill_inputs, None)
                prefill_inputs = (
                    *prefill_inputs,
                    dsa_query_weight_aliases,
                )
            if args.main_rope_table:
                if main_rope_table is None:
                    raise RuntimeError("prefill main-RoPE table is unavailable")
                if not args.prefill_index_repair and not (
                    args.dsa_query_exact_association
                ):
                    prefill_inputs = (*prefill_inputs, None)
                if not args.dsa_query_exact_association:
                    prefill_inputs = (*prefill_inputs, None)
                prefill_inputs = (*prefill_inputs, main_rope_table)
            if args.dsa_rope_table:
                if dsa_rope_table is None:
                    raise RuntimeError("prefill DSA host rotary table is unavailable")
                # Positional optional prefill inputs: materialized wk, query
                # aliases, main table, then the DSA table.
                if not args.prefill_index_repair and not (
                    args.dsa_query_exact_association or args.main_rope_table
                ):
                    prefill_inputs = (*prefill_inputs, None)
                if not args.dsa_query_exact_association and not (
                    args.main_rope_table
                ):
                    prefill_inputs = (*prefill_inputs, None)
                if not args.main_rope_table:
                    prefill_inputs = (*prefill_inputs, None)
                prefill_inputs = (*prefill_inputs, dsa_rope_table)
        multihost_utils.sync_global_devices("greenfield-short-decoder-compile-start")
        compile_started = time.monotonic()
        lowered = jax.jit(
            decoder.execute,
            donate_argnums=donate_argnums,
        ).lower(*inputs)
        decoder_stablehlo = lowered.as_text()
        stablehlo_sha256 = sha256(
            decoder_stablehlo.encode("utf-8")
        ).hexdigest()
        fleet_stablehlo_hashes = _fleet_digest(
            multihost_utils,
            stablehlo_sha256,
            num_processes=args.num_processes,
        )
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
            stablehlo=decoder_stablehlo,
            config=decoder.config,
            schedule=schedule,
            groups=groups,
            pairs=pairs,
            backend_contract=hlo_backend_contract,
            feature_output_tile=decoder.feature_output_tile,
            feature_fuse_route_weighting=(
                decoder.feature_fuse_route_weighting
            ),
            feature_reconstruct_down_fp32=(
                decoder.feature_reconstruct_down_fp32
            ),
            dsa_query_backend=decoder.dsa_query_backend,
            dsa_query_exact_association=(
                decoder.dsa_query_exact_association
            ),
            dsa_head_key_exact_association=(
                decoder.dsa_head_key_exact_association
            ),
            attention_projection_backend=(
                decoder.attention_projection_backend
            ),
            complete_token_path=decoder.complete_token_path,
            split_residual_state=decoder.split_residual_state,
            main_rope_table_enabled=decoder.main_rope_table_enabled,
            dsa_rope_table_enabled=decoder.dsa_rope_table_enabled,
            rms_accepted_schedule=decoder.rms_accepted_schedule,
            pregathered_b512_attention=(
                decoder.pregathered_b512_attention
            ),
            strategy_nd_attention_projection=(
                decoder.strategy_nd_attention_projection
            ),
            dense_final_layout_convolution=(
                decoder.dense_final_layout_convolution
            ),
        )
        if jax.process_index() == 0:
            hlo_dir = args.output.parent / "hlo"
            hlo_dir.mkdir(parents=True, exist_ok=True)
            hlo_stem = (
                f"decoder_78layer_{context_label}_token"
                if args.complete_token_path
                else f"decoder_78layer_{context_label}"
            )
            with gzip.open(
                hlo_dir / f"{hlo_stem}.optimized_hlo.txt.gz",
                "wt",
                encoding="utf-8",
            ) as stream:
                stream.write(optimized_hlo)
            with gzip.open(
                hlo_dir / f"{hlo_stem}.stablehlo.mlir.gz",
                "wt",
                encoding="utf-8",
            ) as stream:
                stream.write(decoder_stablehlo)
            _atomic_json(
                hlo_dir / f"{hlo_stem}.hlo_contract.json",
                hlo_contract,
            )
        del optimized_hlo
        del decoder_stablehlo
        if not hlo_contract["passed"]:
            raise RuntimeError(
                "decoder HLO contract failed before execution: "
                f"{hlo_contract['violations']}"
            )
        compiled_dsa_observer = None
        dsa_observer_compile_seconds = None
        dsa_observer_hlo_sha256 = None
        dsa_observer_stablehlo_sha256 = None
        fleet_dsa_observer_hlo_hashes = None
        fleet_dsa_observer_stablehlo_hashes = None
        dsa_observer_hlo_contract = None
        dsa_observer_isolation_contract = None
        if dsa_oracle_mode and not (
            observe_layer0_discriminator or args.observe_layer0_ingredients
        ):
            assert dsa_observer is not None
            multihost_utils.sync_global_devices(
                "greenfield-short-dsa-observer-compile-start"
            )
            observer_compile_started = time.monotonic()
            # Deliberately no donate_argnums: the observer must replay first
            # while preserving the prefill result for production timing.
            lowered_dsa_observer = jax.jit(dsa_observer.execute).lower(*inputs)
            dsa_observer_stablehlo = lowered_dsa_observer.as_text()
            dsa_observer_stablehlo_sha256 = sha256(
                dsa_observer_stablehlo.encode("utf-8")
            ).hexdigest()
            fleet_dsa_observer_stablehlo_hashes = _fleet_digest(
                multihost_utils,
                dsa_observer_stablehlo_sha256,
                num_processes=args.num_processes,
            )
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
                stablehlo=dsa_observer_stablehlo,
                config=dsa_observer.config,
                schedule=schedule,
                groups=groups,
                pairs=pairs,
                backend_contract=hlo_backend_contract,
                feature_output_tile=dsa_observer.feature_output_tile,
                feature_fuse_route_weighting=(
                    dsa_observer.feature_fuse_route_weighting
                ),
                feature_reconstruct_down_fp32=(
                    dsa_observer.feature_reconstruct_down_fp32
                ),
                dsa_query_backend=dsa_observer.dsa_query_backend,
                dsa_query_exact_association=(
                    dsa_observer.dsa_query_exact_association
                ),
                dsa_head_key_exact_association=(
                    dsa_observer.dsa_head_key_exact_association
                ),
                attention_projection_backend=(
                    dsa_observer.attention_projection_backend
                ),
                complete_token_path=True,
                token_observation_candidates=(
                    dsa_observer.config.token_observation_candidates
                ),
                split_residual_state=dsa_observer.split_residual_state,
                main_rope_table_enabled=(
                    dsa_observer.main_rope_table_enabled
                ),
                dsa_rope_table_enabled=dsa_observer.dsa_rope_table_enabled,
                rms_accepted_schedule=dsa_observer.rms_accepted_schedule,
                pregathered_b512_attention=(
                    dsa_observer.pregathered_b512_attention
                ),
                strategy_nd_attention_projection=(
                    dsa_observer.strategy_nd_attention_projection
                ),
                dense_final_layout_convolution=(
                    dsa_observer.dense_final_layout_convolution
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
                    / (
                        f"decoder_78layer_{context_label}_token_dsa_observer"
                        ".optimized_hlo.txt.gz"
                    ),
                    "wt",
                    encoding="utf-8",
                ) as stream:
                    stream.write(optimized_dsa_observer_hlo)
                with gzip.open(
                    hlo_dir
                    / (
                        f"decoder_78layer_{context_label}_token_dsa_observer"
                        ".stablehlo.mlir.gz"
                    ),
                    "wt",
                    encoding="utf-8",
                ) as stream:
                    stream.write(dsa_observer_stablehlo)
                _atomic_json(
                    hlo_dir
                    / (
                        f"decoder_78layer_{context_label}_token_dsa_observer"
                        ".hlo_contract.json"
                    ),
                    dsa_observer_hlo_contract,
                )
                _atomic_json(
                    hlo_dir
                    / (
                        f"decoder_78layer_{context_label}_token_dsa_observer"
                        ".isolation_contract.json"
                    ),
                    dsa_observer_isolation_contract,
                )
            del optimized_dsa_observer_hlo
            del dsa_observer_stablehlo
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
        compiled_layer0_residual_discriminators: tuple[
            tuple[str, Any], ...
        ] = ()
        layer0_residual_discriminator_compile_seconds: dict[str, float] = {}
        layer0_residual_discriminator_hlo_sha256: dict[str, str] = {}
        layer0_residual_discriminator_stablehlo_sha256: dict[str, str] = {}
        layer0_residual_discriminator_barrier_counts: dict[str, int] = {}
        fleet_layer0_residual_discriminator_hlo_hashes: dict[
            str, list[str]
        ] = {}
        fleet_layer0_residual_discriminator_stablehlo_hashes: dict[
            str, list[str]
        ] = {}
        layer0_residual_discriminator_hlo_contracts: dict[
            str, dict[str, Any]
        ] = {}
        layer0_residual_discriminator_suite_contract = None
        if observe_layer0_discriminator:
            if args.observe_layer0_strategy_nd_row0_association:
                expected_variant_names = (
                    "strategy_nd_row0_control",
                    "strategy_nd_row0_both",
                )
            elif args.observe_layer0_subshard_variants:
                expected_variant_names = (
                    "dcp_then_model_sequential_bf16",
                    "dcp_then_model_pairwise_bf16",
                    "model_then_dcp_sequential_bf16",
                    "model_then_dcp_pairwise_bf16",
                )
            elif args.observe_layer0_attention_output_association_variants:
                expected_variant_names = (
                    "attention_output_association_control",
                    "attention_output_dcp_then_model_sequential_bf16",
                    "attention_output_dcp_then_model_pairwise_bf16",
                    "attention_output_model_then_dcp_sequential_bf16",
                    "attention_output_model_then_dcp_pairwise_bf16",
                )
            elif args.observe_layer0_attention_schedule_variants:
                expected_variant_names = (
                    "attention_schedule_control",
                    "replicated_monolithic_attention",
                )
            else:
                expected_variant_names = (
                    "baseline_bf16",
                    "attention_output_fp32",
                    "dense_down_fp32",
                    "attention_output_and_dense_down_fp32",
                )
            observed_variant_names = tuple(
                name for name, _ in decoder.layer0_residual_discriminators
            )
            if observed_variant_names != expected_variant_names:
                raise RuntimeError(
                    "layer-0 residual discriminator program set drifted: "
                    f"{observed_variant_names}"
                )
            compiled_discriminators = []
            for variant_name, discriminator in (
                decoder.layer0_residual_discriminators
            ):
                multihost_utils.sync_global_devices(
                    "greenfield-layer0-residual-discriminator-"
                    f"{variant_name}-compile-start"
                )
                discriminator_compile_started = time.monotonic()
                lowered_discriminator = jax.jit(discriminator).lower(*inputs)
                discriminator_stablehlo = lowered_discriminator.as_text()
                stablehlo_digest = sha256(
                    discriminator_stablehlo.encode("utf-8")
                ).hexdigest()
                layer0_residual_discriminator_stablehlo_sha256[
                    variant_name
                ] = stablehlo_digest
                fleet_layer0_residual_discriminator_stablehlo_hashes[
                    variant_name
                ] = _fleet_digest(
                    multihost_utils,
                    stablehlo_digest,
                    num_processes=args.num_processes,
                )
                layer0_residual_discriminator_barrier_counts[
                    variant_name
                ] = discriminator_stablehlo.count(
                    "stablehlo.optimization_barrier"
                )
                compiled_discriminator = lowered_discriminator.compile()
                layer0_residual_discriminator_compile_seconds[
                    variant_name
                ] = time.monotonic() - discriminator_compile_started
                multihost_utils.sync_global_devices(
                    "greenfield-layer0-residual-discriminator-"
                    f"{variant_name}-compile-end"
                )
                optimized_discriminator_hlo = compiled_discriminator.as_text()
                hlo_digest = sha256(
                    optimized_discriminator_hlo.encode("utf-8")
                ).hexdigest()
                layer0_residual_discriminator_hlo_sha256[
                    variant_name
                ] = hlo_digest
                fleet_layer0_residual_discriminator_hlo_hashes[
                    variant_name
                ] = _fleet_digest(
                    multihost_utils,
                    hlo_digest,
                    num_processes=args.num_processes,
                )
                hlo_contract = validate_layer0_residual_discriminator_hlo(
                    optimized_discriminator_hlo,
                    config=decoder.config,
                    groups=groups,
                    variant_name=variant_name,
                    main_rope_table_enabled=args.main_rope_table,
                )
                layer0_residual_discriminator_hlo_contracts[
                    variant_name
                ] = hlo_contract
                if jax.process_index() == 0:
                    hlo_dir = args.output.parent / "hlo"
                    with gzip.open(
                        hlo_dir
                        / (
                            "layer0_residual_discriminator."
                            f"{variant_name}.optimized_hlo.txt.gz"
                        ),
                        "wt",
                        encoding="utf-8",
                    ) as stream:
                        stream.write(optimized_discriminator_hlo)
                    with gzip.open(
                        hlo_dir
                        / (
                            "layer0_residual_discriminator."
                            f"{variant_name}.stablehlo.txt.gz"
                        ),
                        "wt",
                        encoding="utf-8",
                    ) as stream:
                        stream.write(discriminator_stablehlo)
                    _atomic_json(
                        hlo_dir
                        / (
                            "layer0_residual_discriminator."
                            f"{variant_name}.hlo_contract.json"
                        ),
                        hlo_contract,
                    )
                del optimized_discriminator_hlo
                del discriminator_stablehlo
                compiled_discriminators.append(
                    (variant_name, compiled_discriminator)
                )
            compiled_layer0_residual_discriminators = tuple(
                compiled_discriminators
            )
            strategy_nd_stablehlo_contract = {
                "applicable": bool(
                    args.observe_layer0_strategy_nd_row0_association
                ),
                "barrier_counts": dict(
                    layer0_residual_discriminator_barrier_counts
                ),
                "expected_candidate_minus_control": 164,
                "passed": True,
            }
            if args.observe_layer0_strategy_nd_row0_association:
                barrier_delta = (
                    layer0_residual_discriminator_barrier_counts[
                        "strategy_nd_row0_both"
                    ]
                    - layer0_residual_discriminator_barrier_counts[
                        "strategy_nd_row0_control"
                    ]
                )
                strategy_nd_stablehlo_contract["observed_delta"] = (
                    barrier_delta
                )
                strategy_nd_stablehlo_contract["passed"] = bool(
                    barrier_delta == 164
                )

            strategy_nd_canary_contract = None
            if args.observe_layer0_strategy_nd_row0_association:
                assert strategy_nd_canary_input_bits is not None
                assert strategy_nd_canary_output_bits is not None
                axis_names = tuple(decoder.mesh.axis_names)
                if axis_names != ("device",):
                    raise RuntimeError(
                        "StrategyND canary requires the explicit device axis"
                    )
                model_to_physical = np.asarray(
                    STRATEGY_ND_CANARY_MODEL_TO_PHYSICAL,
                    dtype=np.int32,
                )
                canary_model_input_bits = np.ascontiguousarray(
                    strategy_nd_canary_input_bits[:, model_to_physical]
                )

                def make_canary_input(trial: int) -> Any:
                    def builder(
                        rank: int,
                        shard_shape: tuple[int, ...],
                    ) -> np.ndarray:
                        slot = rank % 4
                        model_bits = canary_model_input_bits[trial]
                        local_bits = np.ascontiguousarray(
                            model_bits[slot * 8 : (slot + 1) * 8]
                        )
                        local = local_bits.view(ml_dtypes.bfloat16)[
                            None, :, None, :
                        ]
                        if tuple(local.shape) != shard_shape:
                            raise RuntimeError(
                                "StrategyND canary shard shape drifted"
                            )
                        return local

                    return _make_global_array(
                        jax,
                        decoder.mesh,
                        P("device", None, None, None),
                        (32, 8, 1, 6144),
                        builder,
                    )

                def canary_mapped(local_partials: Any) -> Any:
                    with jax.named_scope(
                        "greenfield_strategy_nd_row0_canary"
                    ):
                        reduced = _reduce_strategy_nd_row0_bf16_partials(
                            local_partials[0],
                            axis_name="device",
                            groups=groups,
                        )
                    return reduced[None, ...]

                canary_execute = jax.shard_map(
                    canary_mapped,
                    mesh=decoder.mesh,
                    in_specs=P("device", None, None, None),
                    out_specs=P("device", None, None),
                    check_vma=False,
                )
                first_canary_input = make_canary_input(0)
                multihost_utils.sync_global_devices(
                    "greenfield-strategy-nd-row0-canary-compile-start"
                )
                canary_compile_started = time.monotonic()
                lowered_canary = jax.jit(canary_execute).lower(
                    first_canary_input
                )
                canary_stablehlo = lowered_canary.as_text()
                canary_barrier_count = canary_stablehlo.count(
                    "stablehlo.optimization_barrier"
                )
                compiled_canary = lowered_canary.compile()
                canary_compile_seconds = (
                    time.monotonic() - canary_compile_started
                )
                multihost_utils.sync_global_devices(
                    "greenfield-strategy-nd-row0-canary-compile-end"
                )
                optimized_canary_hlo = compiled_canary.as_text()
                canary_hlo_sha256 = sha256(
                    optimized_canary_hlo.encode("utf-8")
                ).hexdigest()
                fleet_canary_hlo_hashes = _fleet_digest(
                    multihost_utils,
                    canary_hlo_sha256,
                    num_processes=args.num_processes,
                )
                canary_hlo_contract = _validate_strategy_nd_canary_hlo(
                    optimized_canary_hlo,
                    groups=groups,
                )
                mismatch_count = 0
                first_mismatch = None
                actual_row0_sha256: list[str] = []
                for trial in range(32):
                    canary_input = (
                        first_canary_input
                        if trial == 0
                        else make_canary_input(trial)
                    )
                    canary_output = compiled_canary(canary_input)
                    canary_output.block_until_ready()
                    actual = _materialize_global_array(
                        jax,
                        multihost_utils,
                        canary_output,
                    )
                    actual_bits = _encode_bfloat16_bits(actual[:, 0])
                    expected_bits = np.asarray(
                        strategy_nd_canary_output_bits[trial, 0]
                    )
                    mismatches = np.argwhere(
                        actual_bits != expected_bits[None, :]
                    )
                    mismatch_count += int(mismatches.shape[0])
                    if first_mismatch is None and mismatches.size:
                        first_mismatch = {
                            "column": int(mismatches[0, 1]),
                            "lane": int(mismatches[0, 0]),
                            "trial": trial,
                        }
                    actual_row0_sha256.append(
                        sha256(actual_bits[0].tobytes()).hexdigest()
                    )
                    canary_output.delete()
                    canary_input.delete()
                strategy_nd_canary_contract = {
                    "actual_row0_sha256": actual_row0_sha256,
                    "compile_seconds": canary_compile_seconds,
                    "first_mismatch": first_mismatch,
                    "fleet_hlo_sha256": fleet_canary_hlo_hashes,
                    "hlo_contract": canary_hlo_contract,
                    "hlo_sha256": canary_hlo_sha256,
                    "input_file_sha256": (
                        args.strategy_nd_canary_input_sha256
                    ),
                    "lane_count": 32,
                    "mismatch_count": mismatch_count,
                    "output_file_sha256": (
                        args.strategy_nd_canary_output_sha256
                    ),
                    "stablehlo_optimization_barrier_count": (
                        canary_barrier_count
                    ),
                    "trial_count": 32,
                }
                strategy_nd_canary_contract["passed"] = bool(
                    canary_barrier_count == 82
                    and canary_hlo_contract["passed"]
                    and mismatch_count == 0
                    and len(set(actual_row0_sha256)) == 32
                )
                if jax.process_index() == 0:
                    hlo_dir = args.output.parent / "hlo"
                    with gzip.open(
                        hlo_dir / "strategy_nd_row0_canary.optimized_hlo.txt.gz",
                        "wt",
                        encoding="utf-8",
                    ) as stream:
                        stream.write(optimized_canary_hlo)
                    _atomic_json(
                        hlo_dir / "strategy_nd_row0_canary.contract.json",
                        strategy_nd_canary_contract,
                    )
                del optimized_canary_hlo
                del canary_stablehlo
                del canary_model_input_bits
                if not strategy_nd_canary_contract["passed"]:
                    raise RuntimeError(
                        "StrategyND DB533 canary failed before model execution: "
                        f"{strategy_nd_canary_contract}"
                    )
            layer0_residual_discriminator_suite_contract = {
                "discriminator_kind": layer0_discriminator_kind,
                "all_hlo_contracts_pass": all(
                    contract["passed"]
                    for contract in (
                        layer0_residual_discriminator_hlo_contracts.values()
                    )
                ),
                "all_distinct_hlo_modules": (
                    len(
                        set(
                            layer0_residual_discriminator_hlo_sha256.values()
                        )
                    )
                    == len(expected_variant_names)
                ),
                # Retained for compatibility with the two completed four-arm
                # diagnostic schemas. New consumers use the generic field.
                "four_distinct_hlo_modules": (
                    len(
                        set(
                            layer0_residual_discriminator_hlo_sha256.values()
                        )
                    )
                    == len(expected_variant_names)
                ),
                "program_count": len(
                    compiled_layer0_residual_discriminators
                ),
                "strategy_nd_canary_passed": (
                    strategy_nd_canary_contract is not None
                    and strategy_nd_canary_contract["passed"]
                    if args.observe_layer0_strategy_nd_row0_association
                    else True
                ),
                "strategy_nd_stablehlo_contract": (
                    strategy_nd_stablehlo_contract
                ),
                "variant_names": list(observed_variant_names),
            }
            layer0_residual_discriminator_suite_contract["passed"] = bool(
                layer0_residual_discriminator_suite_contract[
                    "all_hlo_contracts_pass"
                ]
                and layer0_residual_discriminator_suite_contract[
                    "all_distinct_hlo_modules"
                ]
                and layer0_residual_discriminator_suite_contract[
                    "program_count"
                ]
                == len(expected_variant_names)
                and layer0_residual_discriminator_suite_contract[
                    "strategy_nd_canary_passed"
                ]
                and strategy_nd_stablehlo_contract["passed"]
            )
            if jax.process_index() == 0:
                _atomic_json(
                    args.output.parent
                    / "hlo"
                    / "layer0_residual_discriminator.suite_contract.json",
                    layer0_residual_discriminator_suite_contract,
                )
            if not layer0_residual_discriminator_suite_contract["passed"]:
                raise RuntimeError(
                    "layer-0 residual discriminator isolated-HLO suite "
                    "failed: "
                    f"{layer0_residual_discriminator_suite_contract}; "
                    f"arms={layer0_residual_discriminator_hlo_contracts}"
                )
        compiled_layer0_ingredients_observer = None
        layer0_ingredients_compile_seconds = None
        layer0_ingredients_hlo_sha256 = None
        fleet_layer0_ingredients_hlo_hashes = None
        layer0_ingredients_hlo_contract = None
        if args.observe_layer0_ingredients:
            if decoder.layer0_ingredients_observer is None:
                raise RuntimeError(
                    "layer-0 ingredient observer executable is unavailable"
                )
            multihost_utils.sync_global_devices(
                "greenfield-layer0-ingredients-compile-start"
            )
            ingredients_compile_started = time.monotonic()
            compiled_layer0_ingredients_observer = jax.jit(
                decoder.layer0_ingredients_observer
            ).lower(*inputs).compile()
            layer0_ingredients_compile_seconds = (
                time.monotonic() - ingredients_compile_started
            )
            multihost_utils.sync_global_devices(
                "greenfield-layer0-ingredients-compile-end"
            )
            optimized_ingredients_hlo = (
                compiled_layer0_ingredients_observer.as_text()
            )
            layer0_ingredients_hlo_sha256 = sha256(
                optimized_ingredients_hlo.encode("utf-8")
            ).hexdigest()
            fleet_layer0_ingredients_hlo_hashes = _fleet_digest(
                multihost_utils,
                layer0_ingredients_hlo_sha256,
                num_processes=args.num_processes,
            )
            layer0_ingredients_hlo_contract = (
                validate_layer0_ingredients_observer_hlo(
                    optimized_ingredients_hlo,
                    config=decoder.config,
                    groups=groups,
                    main_rope_table_enabled=args.main_rope_table,
                )
            )
            if jax.process_index() == 0:
                hlo_dir = args.output.parent / "hlo"
                with gzip.open(
                    hlo_dir / "layer0_ingredients.optimized_hlo.txt.gz",
                    "wt",
                    encoding="utf-8",
                ) as stream:
                    stream.write(optimized_ingredients_hlo)
                _atomic_json(
                    hlo_dir / "layer0_ingredients.hlo_contract.json",
                    layer0_ingredients_hlo_contract,
                )
            del optimized_ingredients_hlo
            if not layer0_ingredients_hlo_contract["passed"]:
                raise RuntimeError(
                    "layer-0 ingredient observer HLO contract failed before "
                    f"execution: {layer0_ingredients_hlo_contract}"
                )
        compiled_prefill = None
        prefill_compile_seconds = None
        prefill_hlo_sha256 = None
        prefill_stablehlo_sha256 = None
        fleet_prefill_hlo_hashes = None
        fleet_prefill_stablehlo_hashes = None
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
            prefill_stablehlo = lowered_prefill.as_text()
            prefill_stablehlo_sha256 = sha256(
                prefill_stablehlo.encode("utf-8")
            ).hexdigest()
            fleet_prefill_stablehlo_hashes = _fleet_digest(
                multihost_utils,
                prefill_stablehlo_sha256,
                num_processes=args.num_processes,
            )
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
                stablehlo=prefill_stablehlo,
                program=prefill,
                schedule=schedule,
                backend_contract=hlo_backend_contract,
            )
            if jax.process_index() == 0:
                hlo_dir = args.output.parent / "hlo"
                with gzip.open(
                    hlo_dir
                    / f"prefill_78layer_{context_label}.optimized_hlo.txt.gz",
                    "wt",
                    encoding="utf-8",
                ) as stream:
                    stream.write(optimized_prefill_hlo)
                with gzip.open(
                    hlo_dir
                    / f"prefill_78layer_{context_label}.stablehlo.mlir.gz",
                    "wt",
                    encoding="utf-8",
                ) as stream:
                    stream.write(prefill_stablehlo)
                _atomic_json(
                    hlo_dir
                    / f"prefill_78layer_{context_label}.hlo_contract.json",
                    prefill_hlo_contract,
                )
            del optimized_prefill_hlo
            del prefill_stablehlo
            if not prefill_hlo_contract["passed"]:
                raise RuntimeError(
                    "prefill HLO contract failed before execution: "
                    f"{prefill_hlo_contract['violations']}"
                )

        generated_tokens: list[int] = []
        prefill_wall_ms = None
        prefill_token_oracle_contract = None
        dsa_observer_contract = None
        layer_residual_contract = None
        dsa_internal_contract = None
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
        if observe_layer0_discriminator:
            assert compiled_layer0_residual_discriminators
            assert layer0_residual_discriminator_suite_contract is not None
            assert layer1_normalized_hidden_reference is not None
            assert dsa_oracle_tensors is not None
            active_rows = np.asarray(groups[0], dtype=np.int32)
            inactive_rows = np.asarray(
                sorted(set(range(decoder.config.total_devices)) - set(groups[0])),
                dtype=np.int32,
            )
            expected_positions = np.asarray(
                dsa_oracle_tensors["selected_positions"][0, 0],
                dtype=np.int32,
            )
            expected_scores = np.asarray(
                dsa_oracle_tensors["selected_scores"][0, 0],
                dtype=np.float32,
            )
            expected_count = int(
                dsa_oracle_tensors["valid_counts"][0, 0]
            )
            reference_value = (
                np.ascontiguousarray(layer1_normalized_hidden_reference)
                .view(np.dtype("<u2"))
                .view(ml_dtypes.bfloat16)
                .astype(np.float32)
            )
            variant_names = expected_variant_names
            comparisons: dict[str, dict[str, Any]] = {}
            arm_contracts: dict[str, dict[str, Any]] = {}
            variant_bits_by_name: dict[str, np.ndarray] = {}
            selection_by_name: dict[
                str, tuple[np.ndarray, np.ndarray, np.ndarray]
            ] = {}
            for variant_name, compiled_discriminator in (
                compiled_layer0_residual_discriminators
            ):
                multihost_utils.sync_global_devices(
                    "greenfield-layer0-residual-discriminator-"
                    f"{variant_name}-execute-start"
                )
                discriminator_inputs = (*runtime_prefix, *tuple(output))
                if args.main_rope_table:
                    discriminator_inputs = (
                        *discriminator_inputs,
                        main_rope_table,
                    )
                discriminator_result = compiled_discriminator(
                    *discriminator_inputs
                )
                discriminator_result[3].block_until_ready()
                (
                    positions_host,
                    scores_host,
                    counts_host,
                    normalized_host,
                    valid_host,
                ) = (
                    _materialize_global_array(jax, multihost_utils, value)
                    for value in discriminator_result
                )
                multihost_utils.sync_global_devices(
                    "greenfield-layer0-residual-discriminator-"
                    f"{variant_name}-execute-end"
                )
                active_positions = positions_host[active_rows]
                active_scores = scores_host[active_rows]
                active_counts = counts_host[active_rows]
                active_normalized = normalized_host[active_rows]
                active_valid = valid_host[active_rows]
                normalized_bits = _encode_bfloat16_bits(active_normalized)
                actual_bits = np.ascontiguousarray(normalized_bits[0])
                variant_bits_by_name[variant_name] = actual_bits
                selection_by_name[variant_name] = (
                    np.asarray(active_positions[0], dtype=np.int32),
                    np.asarray(active_scores[0], dtype=np.float32),
                    np.asarray(active_counts[0], dtype=np.int32),
                )
                actual_value = active_normalized[0].astype(np.float32)
                delta = actual_value - reference_value
                comparisons[variant_name] = {
                    "actual_sha256": sha256(
                        actual_bits.tobytes(order="C")
                    ).hexdigest(),
                    "elementwise_exact": bool(
                        np.array_equal(
                            actual_bits, layer1_normalized_hidden_reference
                        )
                    ),
                    "max_abs": float(np.max(np.abs(delta), initial=0.0)),
                    "mean_abs": float(np.mean(np.abs(delta))),
                    "mismatch_count": int(
                        np.count_nonzero(
                            actual_bits != layer1_normalized_hidden_reference
                        )
                    ),
                    "signed_mean": float(np.mean(delta)),
                }
                arm_contracts[variant_name] = {
                    "active_contract_valid": bool(np.all(active_valid)),
                    "hlo_contract": (
                        layer0_residual_discriminator_hlo_contracts[
                            variant_name
                        ]
                    ),
                    "inactive_rows_are_sentinel": bool(
                        np.all(positions_host[inactive_rows] == -1)
                        and np.all(np.isneginf(scores_host[inactive_rows]))
                        and np.all(counts_host[inactive_rows] == 0)
                        and np.all(
                            _encode_bfloat16_bits(
                                normalized_host[inactive_rows]
                            )
                            == 0
                        )
                        and not np.any(valid_host[inactive_rows])
                    ),
                    "lane_replication": bool(
                        np.all(active_positions == active_positions[0])
                        and np.all(active_scores == active_scores[0])
                        and np.all(active_counts == active_counts[0])
                        and np.all(
                            normalized_bits == normalized_bits[0][None, ...]
                        )
                    ),
                    "selection_exact": bool(
                        np.array_equal(
                            active_positions[0], expected_positions
                        )
                        and np.array_equal(active_scores[0], expected_scores)
                        and int(active_counts[0, 0]) == expected_count
                    ),
                }
                arm_contracts[variant_name]["passed"] = bool(
                    arm_contracts[variant_name]["active_contract_valid"]
                    and arm_contracts[variant_name][
                        "inactive_rows_are_sentinel"
                    ]
                    and arm_contracts[variant_name]["lane_replication"]
                    and arm_contracts[variant_name]["selection_exact"]
                    and arm_contracts[variant_name]["hlo_contract"]["passed"]
                )
            variants_bits = np.stack(
                tuple(variant_bits_by_name[name] for name in variant_names),
                axis=0,
            )
            discriminator_contract = {
                "arm_contracts": arm_contracts,
                "compile_seconds_by_variant": (
                    layer0_residual_discriminator_compile_seconds
                ),
                "decode_position": int(
                    dsa_oracle_tensors["decode_positions"][0]
                ),
                "fleet_hlo_hashes": (
                    fleet_layer0_residual_discriminator_hlo_hashes
                ),
                "fleet_stablehlo_hashes": (
                    fleet_layer0_residual_discriminator_stablehlo_hashes
                ),
                "hlo_sha256_by_variant": (
                    layer0_residual_discriminator_hlo_sha256
                ),
                "stablehlo_sha256_by_variant": (
                    layer0_residual_discriminator_stablehlo_sha256
                ),
                "hlo_suite_contract": (
                    layer0_residual_discriminator_suite_contract
                ),
                "strategy_nd_canary_contract": (
                    strategy_nd_canary_contract
                ),
                "current_baseline_expected_sha256": (
                    LAYER1_MAIN_ROPE_NORMALIZED_HIDDEN_SHA256
                    if main_rope_layer0_discriminator
                    else LAYER1_CURRENT_NORMALIZED_HIDDEN_SHA256
                ),
                "discriminator_kind": layer0_discriminator_kind,
                "layer1_reference_sha256": (
                    args.layer1_internal_reference_sha256
                ),
                "reproduces_current_baseline": (
                    bool(
                        comparisons[
                            (
                                "strategy_nd_row0_control"
                                if args.observe_layer0_strategy_nd_row0_association
                                else (
                                    "attention_output_association_control"
                                    if args.observe_layer0_attention_output_association_variants
                                    else "attention_schedule_control"
                                )
                            )
                            if main_rope_layer0_discriminator
                            else "baseline_bf16"
                        ]["actual_sha256"]
                        == (
                            LAYER1_MAIN_ROPE_NORMALIZED_HIDDEN_SHA256
                            if main_rope_layer0_discriminator
                            else LAYER1_CURRENT_NORMALIZED_HIDDEN_SHA256
                        )
                    )
                    if (
                        args.observe_layer0_residual_variants
                        or main_rope_layer0_discriminator
                    )
                    else None
                ),
                "sealed_independent_baseline": (
                    None
                    if args.observe_layer0_residual_variants
                    else (
                        {
                            "code_hash": (
                                "b5ba20dd4768d62743494511df22f3cd5935bd46"
                            ),
                            "normalized_hidden_sha256": (
                                LAYER1_MAIN_ROPE_NORMALIZED_HIDDEN_SHA256
                            ),
                            "observer_contract_sha256": (
                                "7600e22f3682b8263a5a1771968331f04e1f6875f4cf01ad6b7ba74823063829"
                            ),
                            "observer_npz_sha256": (
                                "96fe8d9bf0e8fa43a3f2ab92735854b8416f49a0082201515f6c720fd077af05"
                            ),
                            "run_tag": (
                                "greenfield_short_decoder_compile_pp8_8k_"
                                "pallas_feature_linear_ot256_downf32_token_"
                                "splitres_prefill_keyfix_queryexact_"
                                "headkeyexact_scoredefault_mainrope_oracle_"
                                "dsa_dsa_internal_trace2_"
                                "20260811T113139003786245Z"
                            ),
                        }
                        if main_rope_layer0_discriminator
                        else {
                            "code_hash": (
                                "12315aa1daccab67f7eaff709c291425a73b4006"
                            ),
                            "normalized_hidden_sha256": (
                                LAYER1_CURRENT_NORMALIZED_HIDDEN_SHA256
                            ),
                            "run_tag": (
                                "greenfield_short_decoder_compile_pp8_8k_"
                                "pallas_feature_linear_ot256_downf32_token_"
                                "splitres_prefill_keyfix_queryexact_"
                                "headkeyexact_scoredefault_oracle_dsa_"
                                "layer0_residual_variants_trace2_"
                                "20260810T221121969164909Z"
                            ),
                        }
                    )
                ),
                "variant_comparisons": comparisons,
                "variant_names": list(variant_names),
            }
            discriminator_contract["passed"] = bool(
                all(contract["passed"] for contract in arm_contracts.values())
                and (
                    discriminator_contract["reproduces_current_baseline"]
                    is not False
                )
                and layer0_residual_discriminator_suite_contract["passed"]
            )
            if jax.process_index() == 0:
                discriminator_dir = (
                    args.output.parent / "layer0_residual_discriminator"
                )
                _atomic_npz(
                    discriminator_dir
                    / (
                        "position_"
                        f"{discriminator_contract['decode_position']}_variants.npz"
                    ),
                    normalized_hidden_bfloat16_bits=np.asarray(
                        variants_bits, dtype=np.dtype("<u2")
                    ),
                    selected_positions=np.asarray(
                        selection_by_name[variant_names[0]][0],
                        dtype=np.int32,
                    ),
                    selected_scores=np.asarray(
                        selection_by_name[variant_names[0]][1],
                        dtype=np.float32,
                    ),
                    selected_valid_count=np.asarray(
                        selection_by_name[variant_names[0]][2],
                        dtype=np.int32,
                    ),
                )
                _atomic_json(
                    discriminator_dir / "contract.json",
                    discriminator_contract,
                )
            multihost_utils.sync_global_devices(
                "greenfield-layer0-residual-discriminator-complete"
            )
            if not discriminator_contract["passed"]:
                raise RuntimeError(
                    "layer-0 residual discriminator contract failed: "
                    f"{discriminator_contract}"
                )
            raise RuntimeError(
                "layer-0 residual discriminator completed; diagnostic-only "
                f"comparisons={comparisons}"
            )
        if args.observe_layer0_ingredients:
            assert compiled_layer0_ingredients_observer is not None
            assert layer0_ingredients_hlo_contract is not None
            assert layer0_ingredients_hlo_sha256 is not None
            assert dsa_oracle_tensors is not None
            multihost_utils.sync_global_devices(
                "greenfield-layer0-ingredients-execute-start"
            )
            ingredients_inputs = (*runtime_prefix, *tuple(output))
            if args.main_rope_table:
                ingredients_inputs = (*ingredients_inputs, main_rope_table)
            ingredients_result = compiled_layer0_ingredients_observer(
                *ingredients_inputs
            )
            ingredients_result[-1].block_until_ready()
            observed_ingredients = {
                name: _materialize_global_array(
                    jax, multihost_utils, value
                )
                for name, value in zip(
                    LAYER0_INGREDIENT_NAMES,
                    ingredients_result,
                    strict=True,
                )
            }
            multihost_utils.sync_global_devices(
                "greenfield-layer0-ingredients-execute-end"
            )
            expected_positions = np.asarray(
                dsa_oracle_tensors["selected_positions"][0, 0],
                dtype=np.int32,
            )
            expected_scores = np.asarray(
                dsa_oracle_tensors["selected_scores"][0, 0],
                dtype=np.float32,
            )
            expected_count = int(dsa_oracle_tensors["valid_counts"][0, 0])
            artifact_arrays, ingredients_contract = (
                _validate_layer0_ingredients(
                    observed_ingredients,
                    groups=groups,
                    expected_positions=expected_positions,
                    expected_scores=expected_scores,
                    expected_count=expected_count,
                    logical_page_size=state_layout.logical_page_size,
                )
            )
            decode_position = int(dsa_oracle_tensors["decode_positions"][0])
            ingredients_contract.update(
                {
                    "code_hash": code_hash,
                    "compile_seconds": layer0_ingredients_compile_seconds,
                    "decode_position": decode_position,
                    "fleet_hlo_hashes": fleet_layer0_ingredients_hlo_hashes,
                    "hlo_contract": layer0_ingredients_hlo_contract,
                    "hlo_sha256": layer0_ingredients_hlo_sha256,
                    "ingredient_names": list(LAYER0_INGREDIENT_NAMES),
                    "main_rope_table_enabled": args.main_rope_table,
                    "main_rope_table_sha256": decoder.main_rope_table_sha256,
                    "run_tag": greenfield_run_tag,
                    "source_state": "post_teacher_forced_prefill",
                }
            )
            ingredients_contract["passed"] = bool(
                ingredients_contract["passed"]
                and layer0_ingredients_hlo_contract["passed"]
            )
            if jax.process_index() == 0:
                ingredients_dir = args.output.parent / "layer0_ingredients"
                _atomic_npz(
                    ingredients_dir
                    / f"position_{decode_position}_ingredients.npz",
                    decode_position=np.asarray(
                        [decode_position], dtype=np.int32
                    ),
                    **artifact_arrays,
                )
                _atomic_json(
                    ingredients_dir / "contract.json", ingredients_contract
                )
            multihost_utils.sync_global_devices(
                "greenfield-layer0-ingredients-complete"
            )
            _delete_arrays(ingredients_result)
            if not ingredients_contract["passed"]:
                raise RuntimeError(
                    "layer-0 ingredient observer contract failed: "
                    f"{ingredients_contract}"
                )
            raise RuntimeError(
                "layer-0 ingredient observer completed; diagnostic-only "
                f"position={decode_position}"
            )
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
            teacher_forced_input_tokens: list[int] = []
            try:
                for step, decode_position in enumerate(
                    dsa_oracle_tensors["decode_positions"].tolist()
                ):
                    observer_inputs = (*runtime_prefix, *observer_current)
                    if args.main_rope_table:
                        observer_inputs = (*observer_inputs, main_rope_table)
                    if args.dsa_rope_table:
                        observer_inputs = (*observer_inputs, dsa_rope_table)
                    observer_result = compiled_dsa_observer(*observer_inputs)
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
                    if args.observe_dsa_internals and step == 0:
                        assert dsa_internal_baseline is not None
                        assert dsa_internal_layer0_reference is not None
                        internal_value = observer_result[10]
                        internal_host = SimpleNamespace(
                            **{
                                name: _materialize_global_array(
                                    jax, multihost_utils, value
                                )
                                for name, value in zip(
                                    internal_value._fields,
                                    internal_value,
                                    strict=True,
                                )
                            }
                        )
                        (
                            canonical_dsa_internals,
                            dsa_internal_contract,
                        ) = _canonicalize_dsa_internal_observation(
                            internal_host,
                            groups=groups,
                            stage_producer_layer_ids=stage_dsa_producers,
                            hidden_size=execution_plan.geometry.hidden_size,
                            q_lora_rank=execution_plan.geometry.q_lora_rank,
                            num_heads=execution_plan.geometry.dsa_indexer_heads,
                            head_dim=(
                                execution_plan.geometry.dsa_indexer_head_dim
                            ),
                        )
                        baseline_exact = bool(
                            observation_host.shape == dsa_internal_baseline.shape
                            and np.array_equal(
                                observation_host, dsa_internal_baseline
                            )
                        )
                        dsa_internal_contract["baseline_observation"] = {
                            "actual_sha256": sha256(
                                np.ascontiguousarray(
                                    observation_host
                                ).tobytes(order="C")
                            ).hexdigest(),
                            "elementwise_exact": baseline_exact,
                            "expected_sha256": sha256(
                                np.ascontiguousarray(
                                    dsa_internal_baseline
                                ).tobytes(order="C")
                            ).hexdigest(),
                            "shape": list(observation_host.shape),
                        }
                        layer0_comparisons = {}
                        for name in (
                            "normalized_hidden",
                            "q_a_state",
                            "query",
                            "head_weights",
                            "current_key",
                        ):
                            actual = canonical_dsa_internals[name][0]
                            reference = dsa_internal_layer0_reference[name]
                            if actual.dtype.name == "bfloat16":
                                actual_storage = _encode_bfloat16_bits(actual)
                            else:
                                actual_storage = np.ascontiguousarray(actual)
                            reference_storage = np.ascontiguousarray(reference)
                            shape_equal = actual_storage.shape == reference_storage.shape
                            exact = bool(
                                shape_equal
                                and actual_storage.dtype == reference_storage.dtype
                                and np.array_equal(
                                    actual_storage, reference_storage
                                )
                            )
                            if shape_equal:
                                delta = (
                                    actual.astype(np.float32)
                                    - (
                                        reference_storage.view(
                                            np.dtype("<u2")
                                        ).view(ml_dtypes.bfloat16).astype(
                                            np.float32
                                        )
                                        if actual.dtype.name == "bfloat16"
                                        else reference_storage.astype(np.float32)
                                    )
                                )
                                max_abs = float(
                                    np.max(np.abs(delta), initial=0.0)
                                )
                                mismatch_count = int(
                                    np.count_nonzero(
                                        actual_storage != reference_storage
                                    )
                                )
                            else:
                                max_abs = None
                                mismatch_count = None
                            layer0_comparisons[name] = {
                                "actual_sha256": sha256(
                                    actual_storage.tobytes(order="C")
                                ).hexdigest(),
                                "elementwise_exact": exact,
                                "expected_sha256": sha256(
                                    reference_storage.tobytes(order="C")
                                ).hexdigest(),
                                "max_abs": max_abs,
                                "mismatch_count": mismatch_count,
                                "shape": list(actual_storage.shape),
                            }
                        dsa_internal_contract["layer0_reference"] = (
                            layer0_comparisons
                        )
                        dsa_internal_contract["decode_position"] = int(
                            decode_position
                        )
                        dsa_internal_contract["baseline_observation_exact"] = (
                            baseline_exact
                        )
                        dsa_internal_contract["layer0_query_exact"] = bool(
                            layer0_comparisons["query"]["elementwise_exact"]
                        )
                        dsa_internal_contract["passed"] = bool(
                            dsa_internal_contract["passed"]
                            and baseline_exact
                            and dsa_internal_contract["layer0_query_exact"]
                        )
                        if jax.process_index() == 0:
                            internal_dir = (
                                args.output.parent / "dsa_internal_observer"
                            )
                            _atomic_npz(
                                internal_dir
                                / f"position_{int(decode_position)}_internals.npz",
                                normalized_hidden_bfloat16_bits=(
                                    _encode_bfloat16_bits(
                                        canonical_dsa_internals[
                                            "normalized_hidden"
                                        ]
                                    )
                                ),
                                q_a_state_bfloat16_bits=(
                                    _encode_bfloat16_bits(
                                        canonical_dsa_internals["q_a_state"]
                                    )
                                ),
                                query=np.asarray(
                                    canonical_dsa_internals["query"],
                                    dtype=np.float32,
                                ),
                                head_weights=np.asarray(
                                    canonical_dsa_internals["head_weights"],
                                    dtype=np.float32,
                                ),
                                current_key=np.asarray(
                                    canonical_dsa_internals["current_key"],
                                    dtype=np.float32,
                                ),
                                producer_layer_ids=np.asarray(
                                    canonical_dsa_internals[
                                        "producer_layer_ids"
                                    ],
                                    dtype=np.int32,
                                ),
                                decode_position=np.asarray(
                                    [decode_position], dtype=np.int32
                                ),
                            )
                            _atomic_json(
                                internal_dir / "contract.json",
                                dsa_internal_contract,
                            )
                    if (
                        args.observe_layer_residuals
                        and int(decode_position) == args.layer_residual_position
                    ):
                        residual_observation_host = _materialize_global_array(
                            jax,
                            multihost_utils,
                            observer_result[10],
                        )
                        (
                            canonical_layer_residuals,
                            layer_residual_contract,
                        ) = _canonicalize_layer_residual_observation(
                            residual_observation_host,
                            groups=groups,
                            schedule=schedule,
                            hidden_size=execution_plan.geometry.hidden_size,
                        )
                        layer_residual_contract["decode_position"] = int(
                            decode_position
                        )
                        layer_residual_contract["teacher_forced"] = True
                        if not layer_residual_contract["passed"]:
                            raise RuntimeError(
                                "layer residual observer replication contract "
                                f"failed: {layer_residual_contract}"
                            )
                        if jax.process_index() == 0:
                            residual_dir = (
                                args.output.parent / "layer_residual_observer"
                            )
                            _atomic_npz(
                                residual_dir
                                / (
                                    "position_"
                                    f"{int(decode_position)}_boundaries.npz"
                                ),
                                boundary_layer_ids=np.arange(
                                    canonical_layer_residuals.shape[0],
                                    dtype=np.int32,
                                ),
                                decode_position=np.asarray(
                                    [decode_position], dtype=np.int32
                                ),
                                residual_bfloat16_bits=(
                                    _encode_bfloat16_bits(
                                        canonical_layer_residuals
                                    )
                                ),
                            )
                            _atomic_json(
                                residual_dir / "contract.json",
                                layer_residual_contract,
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
                    step_contract["dsa_internal_observation"] = (
                        dsa_internal_contract
                        if args.observe_dsa_internals and step == 0
                        else None
                    )
                    step_contract["position_passed"] = bool(
                        next_position.tolist() == [int(decode_position) + 1]
                    )
                    step_contract["passed"] = bool(
                        step_contract["passed"]
                        and step_contract["position_passed"]
                        and token_observation_contract["passed"]
                        and (
                            not args.observe_dsa_internals
                            or step != 0
                            or (
                                dsa_internal_contract is not None
                                and dsa_internal_contract["passed"]
                            )
                        )
                    )
                    observer_step_records.append(step_contract)
                    if not step_contract["passed"]:
                        raise RuntimeError(
                            "DSA observer device-score/set/tie contract failed: "
                            f"{step_contract}"
                        )
                    observer_tokens.append(observed_token_id)
                    next_observer_values = list(observer_result[:8])
                    if args.observe_layer_residuals:
                        forced_token_id = expected_token_id

                        def teacher_token_builder(
                            rank: int, shape: tuple[int, ...]
                        ) -> np.ndarray:
                            value = np.full(shape, -1, dtype=np.int32)
                            if rank in groups[0]:
                                value[...] = np.int32(forced_token_id)
                            return value

                        forced_token = _make_global_array(
                            jax,
                            dsa_observer.mesh,
                            dsa_observer.input_specs[state_spec_offset + 4],
                            (dsa_observer.config.total_devices, 1),
                            teacher_token_builder,
                        )
                        next_observer_values[4] = forced_token
                        teacher_forced_input_tokens.append(forced_token_id)
                    if observer_owns_current:
                        _delete_arrays(observer_current)
                    if args.observe_layer_residuals:
                        _delete_arrays((observer_result[4],))
                    observer_current = tuple(next_observer_values)
                    observer_owns_current = True
                    _delete_arrays(tuple(observer_result[8:]))
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
                    "layer_residual_observation": layer_residual_contract,
                    "dsa_internal_observation": dsa_internal_contract,
                    "teacher_forced_input_token_ids": (
                        teacher_forced_input_tokens
                        if args.observe_layer_residuals
                        else None
                    ),
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
                    and (
                        not args.observe_layer_residuals
                        or (
                            layer_residual_contract is not None
                            and layer_residual_contract["passed"]
                            and teacher_forced_input_tokens
                            == oracle_generated_token_ids[
                                1 : 1 + len(teacher_forced_input_tokens)
                            ].tolist()
                        )
                    )
                    and (
                        not args.observe_dsa_internals
                        or (
                            dsa_internal_contract is not None
                            and dsa_internal_contract["passed"]
                        )
                    )
                )
                if not dsa_observer_contract["passed"]:
                    if args.observe_layer_residuals and jax.process_index() == 0:
                        _atomic_json(
                            args.output.parent
                            / "layer_residual_observer"
                            / "observer_contract.json",
                            dsa_observer_contract,
                        )
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
                step_inputs = (*runtime_prefix, *values)
            else:
                step_inputs = (
                *runtime_prefix,
                *values,
                position,
                block_tables,
                context_lengths,
                )
            if args.main_rope_table:
                step_inputs = (*step_inputs, main_rope_table)
            if args.dsa_rope_table:
                step_inputs = (*step_inputs, dsa_rope_table)
            return compiled(*step_inputs)

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
            and metadata_contract["visited"]
            == [_expected_stage_visit_mask(execution_plan.pipeline_stages)]
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
                f"greenfield_real_78layer_{context_label}_decoder_"
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
            "dsa_observer_stablehlo_sha256": (
                dsa_observer_stablehlo_sha256
            ),
            "dsa_observer_isolation_contract": (
                dsa_observer_isolation_contract
            ),
            "dsa_query_exact_association": (
                decoder.dsa_query_exact_association
            ),
            "dsa_head_key_exact_association": (
                decoder.dsa_head_key_exact_association
            ),
            "dsa_score_default_precision": (
                decoder.dsa_score_default_precision
            ),
            "dsa_query_materialization_compile_seconds": (
                dsa_query_materialization_compile_seconds
            ),
            "dsa_query_materialization_execute_seconds": (
                dsa_query_materialization_execute_seconds
            ),
            "dsa_query_materialization_hlo_contract": (
                dsa_query_materialization_hlo_contract
            ),
            "dsa_query_materialization_hlo_sha256": (
                dsa_query_materialization_hlo_sha256
            ),
            "dsa_query_materialization_state": (
                dsa_query_materialization_state
            ),
            "fleet_hlo_hashes": fleet_hlo_hashes,
            "fleet_stablehlo_hashes": fleet_stablehlo_hashes,
            "fleet_dsa_observer_hlo_hashes": (
                fleet_dsa_observer_hlo_hashes
            ),
            "fleet_dsa_observer_stablehlo_hashes": (
                fleet_dsa_observer_stablehlo_hashes
            ),
            "fleet_dsa_query_materialization_hlo_hashes": (
                fleet_dsa_query_materialization_hlo_hashes
            ),
            "fleet_prefill_hlo_hashes": fleet_prefill_hlo_hashes,
            "fleet_prefill_stablehlo_hashes": (
                fleet_prefill_stablehlo_hashes
            ),
            "fleet_prefill_wk_materialization_hlo_hashes": (
                fleet_prefill_wk_materialization_hlo_hashes
            ),
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
            "main_rope_table_enabled": decoder.main_rope_table_enabled,
            "pregathered_b512_attention": (
                decoder.pregathered_b512_attention
            ),
            "strategy_nd_attention_projection": (
                decoder.strategy_nd_attention_projection
            ),
            "dense_final_layout_convolution": (
                decoder.dense_final_layout_convolution
            ),
            "main_rope_table_sha256": decoder.main_rope_table_sha256,
            "main_rope_table_shape": (
                list(decoder.main_rope_table_host.shape)
                if decoder.main_rope_table_host is not None
                else None
            ),
            "main_rope_table_bytes_per_device": (
                decoder.main_rope_table_bytes_per_device
            ),
            "main_rope_table_local_device_sha256": (
                main_rope_table_local_hashes
            ),
            "rms_accepted_schedule": decoder.rms_accepted_schedule,
            "rms_accepted_schedule_prerequisite": (
                "bounded TPU replay gate_d_layer1_rms_schedule_20260902T061305905714981Z: "
                "accepted-schedule arm reproduced the accepted layer-1 row exactly"
                if decoder.rms_accepted_schedule
                else None
            ),
            "dsa_rope_table_enabled": decoder.dsa_rope_table_enabled,
            "dsa_rope_table_sha256": decoder.dsa_rope_table_sha256,
            "dsa_rope_table_shape": (
                list(decoder.dsa_rope_table_host.shape)
                if decoder.dsa_rope_table_host is not None
                else None
            ),
            "dsa_rope_table_bytes_per_device": (
                decoder.dsa_rope_table_bytes_per_device
            ),
            "dsa_rope_table_local_device_sha256": (
                dsa_rope_table_local_hashes
            ),
            "dsa_rope_table_prerequisite": (
                {
                    "bounded_proof_run_tag": (
                        "gate_d_projection_contraction_pp16_numerical_"
                        "20260902T004306002075694Z"
                    ),
                    "position_8155_row_sha256": (
                        "748aa6122b8d83cfcf65928d33d1ac10392b7cc968617f324a75a62168d3a9c0"
                    ),
                    "terminal_marker_sha256": (
                        "f26cde1a4ebe459cb038d06c2c1afc39db4416537d8f1f50c2a3bf648d48064b"
                    ),
                }
                if decoder.dsa_rope_table_enabled
                else None
            ),
            "main_rope_table_prerequisite": (
                {
                    "code_hash": (
                        "ed7c74f423e811f1acfbeb3c075fb8fad37d4f51"
                    ),
                    "position_8155_row_sha256": (
                        "67b01e3cab682d5ffd04ac9c8043e7e6825ee1275023428c41f9a1ae412dea1d"
                    ),
                    "results_db_item_row_id": 1816,
                    "results_db_run_id": 531,
                    "runner_sha256": (
                        "d32d03576faa99c882913cf289a32d9aaa7f258a6b83588464058f36a8ad7370"
                    ),
                    "success_sha256": (
                        "8f9763e4e8106db0fe30b9fa3820873f736ede2215edfc8b950fcd311d08b6d7"
                    ),
                }
                if decoder.main_rope_table_enabled
                else None
            ),
            "metadata_contract": metadata_contract,
            "metadata_passed": metadata_passed,
            "optimized_hlo_sha256": hlo_sha256,
            "stablehlo_sha256": stablehlo_sha256,
            "plan_hash": execution_plan.plan_hash,
            "profiler_free_body_wall": _percentiles(samples),
            "profiler_free_complete_step_wall": (
                _percentiles(samples) if args.complete_token_path else None
            ),
            "prefill_compile_seconds": prefill_compile_seconds,
            "prefill_hlo_contract": prefill_hlo_contract,
            "prefill_hlo_sha256": prefill_hlo_sha256,
            "prefill_stablehlo_sha256": prefill_stablehlo_sha256,
            "prefill_index_repair": args.prefill_index_repair,
            "prefill_index_repair_backend": (
                prefill.index_repair_backend if prefill is not None else "none"
            ),
            "prefill_wk_materialization_compile_seconds": (
                prefill_wk_materialization_compile_seconds
            ),
            "prefill_wk_materialization_execute_seconds": (
                prefill_wk_materialization_execute_seconds
            ),
            "prefill_wk_materialization_hlo_contract": (
                prefill_wk_materialization_hlo_contract
            ),
            "prefill_wk_materialization_hlo_sha256": (
                prefill_wk_materialization_hlo_sha256
            ),
            "prefill_wk_materialization_state": (
                prefill_wk_materialization_state
            ),
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
            "schema_version": 18,
            "state_layout": state_layout.to_dict(),
            "state_layout_hash": state_layout.state_layout_hash,
            "sparse_moe_backend": decoder.sparse_moe_backend,
            "feature_output_tile": decoder.feature_output_tile,
            "feature_source_verification_mode": (
                "feature_runtime_with_metadata_parent"
                if args.source_feature_runtime_root is not None
                else (
                    "metadata_lineage"
                    if args.feature_source_metadata_only
                    else "complete_payload"
                )
            ),
            "feature_fuse_route_weighting": (
                decoder.feature_fuse_route_weighting
            ),
            "feature_reconstruct_down_fp32": (
                decoder.feature_reconstruct_down_fp32
            ),
            "split_residual_state": decoder.split_residual_state,
            "residual_transport_components": (
                2 if decoder.split_residual_state else 1
            ),
            "residual_transport_bytes_per_stage": (
                (2 if decoder.split_residual_state else 1)
                * execution_plan.geometry.hidden_size
                * np.dtype(ml_dtypes.bfloat16).itemsize
            ),
            "split_residual_extra_bytes_per_device": (
                execution_plan.geometry.hidden_size
                * np.dtype(ml_dtypes.bfloat16).itemsize
                if decoder.split_residual_state
                else 0
            ),
            "linear_backend": decoder.linear_backend,
            "dsa_query_backend": decoder.dsa_query_backend,
            "attention_projection_backend": (
                decoder.attention_projection_backend
            ),
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
