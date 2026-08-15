"""Protected token/DSA/cache contracts for the complete WS32 decoder."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .short_context_dsa_oracle import inspect_short_context_dsa_oracle
from .short_context_oracle import inspect_short_context_oracle


@dataclass(frozen=True, slots=True)
class Ws32ShortContextOracle:
    token_manifest: Mapping[str, Any]
    dsa_manifest: Mapping[str, Any]
    token_success_sha256: str
    dsa_success_sha256: str
    prompt_token_ids: np.ndarray
    generated_token_ids: np.ndarray
    decode_positions: np.ndarray
    producer_layer_ids: np.ndarray
    selected_positions: np.ndarray
    selected_scores: np.ndarray
    valid_counts: np.ndarray


def load_ws32_short_context_oracle(
    token_oracle_dir: Path,
    dsa_oracle_dir: Path,
    *,
    expected_token_manifest_sha256: str,
    expected_dsa_manifest_sha256: str,
    expected_token_success_sha256: str,
    expected_dsa_success_sha256: str,
) -> Ws32ShortContextOracle:
    """Load only fully inspected, mutually bound legacy oracle tensors."""

    from safetensors import safe_open

    token_oracle_dir = Path(token_oracle_dir)
    dsa_oracle_dir = Path(dsa_oracle_dir)
    token_manifest = inspect_short_context_oracle(token_oracle_dir)
    dsa_manifest = inspect_short_context_dsa_oracle(dsa_oracle_dir)
    if token_manifest.get("manifest_sha256") != expected_token_manifest_sha256:
        raise ValueError("WS32 token oracle identity drifted")
    if dsa_manifest.get("manifest_sha256") != expected_dsa_manifest_sha256:
        raise ValueError("WS32 DSA oracle identity drifted")
    if dsa_manifest.get("token_oracle", {}).get("manifest_sha256") != (
        expected_token_manifest_sha256
    ):
        raise ValueError("WS32 DSA oracle is not bound to the token oracle")

    def verify_success(
        oracle_dir: Path,
        *,
        expected_sha256: str,
        artifact_kind: str,
        manifest_sha256: str,
    ) -> None:
        success = oracle_dir.parent / "SUCCESS"
        raw = success.read_bytes()
        if sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("WS32 oracle terminal SUCCESS identity drifted")
        fields: dict[str, str] = {}
        for line in raw.decode("utf-8").splitlines():
            if line.count("=") != 1:
                raise ValueError("WS32 oracle terminal SUCCESS schema drifted")
            key, value = line.split("=", 1)
            if not key or key in fields:
                raise ValueError("WS32 oracle terminal SUCCESS schema drifted")
            fields[key] = value
        if fields.get("artifact_kind") != artifact_kind or (
            fields.get("manifest_sha256") != manifest_sha256
        ) or not fields.get("remote_prefix", "").startswith(
            "gs://driftbench-dsv4-uc/"
        ):
            raise ValueError("WS32 oracle terminal SUCCESS binding drifted")

    verify_success(
        token_oracle_dir,
        expected_sha256=expected_token_success_sha256,
        artifact_kind="greenfield_short_context_legacy_oracle",
        manifest_sha256=expected_token_manifest_sha256,
    )
    verify_success(
        dsa_oracle_dir,
        expected_sha256=expected_dsa_success_sha256,
        artifact_kind="greenfield_short_context_legacy_dsa_oracle",
        manifest_sha256=expected_dsa_manifest_sha256,
    )
    with safe_open(
        token_oracle_dir / "tokens.safetensors", framework="np"
    ) as handle:
        tokens = {name: handle.get_tensor(name) for name in handle.keys()}
    with safe_open(
        dsa_oracle_dir / "dsa_events.safetensors", framework="np"
    ) as handle:
        dsa = {name: handle.get_tensor(name) for name in handle.keys()}
    prompt = np.asarray(tokens["prompt_token_ids"], dtype=np.int32)
    generated = np.asarray(tokens["generated_token_ids"], dtype=np.int32)
    if prompt.ndim != 1 or generated.ndim != 1 or not prompt.size or (
        not generated.size
    ):
        raise ValueError("WS32 token oracle arrays are empty or non-vector")
    if int(dsa["decode_positions"][0]) != prompt.size:
        raise ValueError("WS32 token and DSA oracle positions disagree")
    return Ws32ShortContextOracle(
        token_manifest=token_manifest,
        dsa_manifest=dsa_manifest,
        token_success_sha256=expected_token_success_sha256,
        dsa_success_sha256=expected_dsa_success_sha256,
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        decode_positions=np.asarray(dsa["decode_positions"], dtype=np.int32),
        producer_layer_ids=np.asarray(
            dsa["producer_layer_ids"], dtype=np.int32
        ),
        selected_positions=np.asarray(
            dsa["selected_positions"], dtype=np.int32
        ),
        selected_scores=np.asarray(dsa["selected_scores"], dtype=np.float32),
        valid_counts=np.asarray(dsa["valid_counts"], dtype=np.int32),
    )


def compare_ws32_raw_tokens(
    observed: list[int] | tuple[int, ...] | np.ndarray,
    oracle: Ws32ShortContextOracle,
) -> dict[str, Any]:
    actual = np.asarray(observed, dtype=np.int32)
    if actual.ndim != 1 or actual.size > oracle.generated_token_ids.size:
        raise ValueError("WS32 raw-token observation geometry drifted")
    expected = oracle.generated_token_ids[: actual.size]
    mismatches = np.flatnonzero(actual != expected)
    return {
        "compared_count": int(actual.size),
        "exact_prefix_match": not bool(mismatches.size),
        "expected_sha256": sha256(expected.astype("<i4").tobytes()).hexdigest(),
        "first_mismatch": None
        if not mismatches.size
        else {
            "index": int(mismatches[0]),
            "expected": int(expected[mismatches[0]]),
            "observed": int(actual[mismatches[0]]),
        },
        "observed_sha256": sha256(actual.astype("<i4").tobytes()).hexdigest(),
    }


def _descending_float32_key(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    ascending = np.where(
        bits & np.uint32(0x80000000), ~bits, bits ^ np.uint32(0x80000000)
    )
    return ~ascending


def compare_ws32_dsa_step(
    *,
    producer_layer_ids: np.ndarray,
    selected_positions: np.ndarray,
    selected_valid_counts: np.ndarray,
    selected_scores: np.ndarray,
    oracle: Ws32ShortContextOracle,
    step: int,
) -> dict[str, Any]:
    """Gate exact selected sets/tails and the executing program's tie order."""

    if not 0 <= step < oracle.decode_positions.size:
        raise ValueError("WS32 DSA step is outside the oracle")
    producer_layer_ids = np.asarray(producer_layer_ids, dtype=np.int32)
    positions = np.asarray(selected_positions, dtype=np.int32)
    counts = np.asarray(selected_valid_counts, dtype=np.int32)
    scores = np.asarray(selected_scores, dtype=np.float32)
    events = oracle.producer_layer_ids.size
    width = oracle.selected_positions.shape[-1]
    if producer_layer_ids.shape != (events,) or positions.shape != (
        events,
        1,
        width,
    ) or counts.shape != (events, 1) or scores.shape != (events, 1, width):
        raise ValueError("WS32 DSA observation geometry drifted")
    positions = positions[:, 0]
    counts = counts[:, 0]
    scores = scores[:, 0]
    producer_exact = bool(
        np.array_equal(producer_layer_ids, oracle.producer_layer_ids)
    )
    count_mismatches = []
    set_mismatches = []
    tail_mismatches = []
    score_contract_mismatches = []
    legacy_order_mismatches = int(
        np.count_nonzero(positions != oracle.selected_positions[step])
    )
    aligned_actual = []
    aligned_expected = []
    decode_position = int(oracle.decode_positions[step])
    for event in range(events):
        observed_count = int(counts[event])
        expected_count = int(oracle.valid_counts[step, event])
        if observed_count != expected_count:
            count_mismatches.append(event)
        safe_count = min(max(observed_count, 0), width)
        expected_safe_count = min(max(expected_count, 0), width)
        live = positions[event, :safe_count]
        live_scores = scores[event, :safe_count]
        expected_live = oracle.selected_positions[
            step, event, :expected_safe_count
        ]
        expected_scores = oracle.selected_scores[
            step, event, :expected_safe_count
        ]
        expected_only = np.setdiff1d(expected_live, live)
        observed_only = np.setdiff1d(live, expected_live)
        if observed_count != expected_count or expected_only.size or observed_only.size:
            set_mismatches.append(
                {
                    "event_index": event,
                    "expected_only": expected_only[:8].tolist(),
                    "observed_only": observed_only[:8].tolist(),
                }
            )
        tail_positions = positions[event, safe_count:]
        tail_scores = scores[event, safe_count:]
        if observed_count < 0 or observed_count > width or (
            np.any(tail_positions != -1) or np.any(~np.isneginf(tail_scores))
        ):
            tail_mismatches.append(event)
        canonical = np.lexsort(
            (live.astype(np.int64, copy=False), _descending_float32_key(live_scores))
        )
        if (
            np.any(live < 0)
            or np.any(live > decode_position)
            or np.unique(live).size != live.size
            or np.any(~np.isfinite(live_scores))
            or not np.array_equal(canonical, np.arange(live.size))
        ):
            score_contract_mismatches.append(event)
        if (
            observed_count == expected_count
            and not expected_only.size
            and not observed_only.size
        ):
            observed_order = np.argsort(live, kind="stable")
            expected_order = np.argsort(expected_live, kind="stable")
            aligned_actual.append(live_scores[observed_order])
            aligned_expected.append(expected_scores[expected_order])
    if aligned_actual:
        score_actual = np.concatenate(aligned_actual).astype(np.float32)
        score_expected = np.concatenate(aligned_expected).astype(np.float32)
        delta = np.abs(score_actual - score_expected)
        score_diagnostic = {
            "aligned_count": int(delta.size),
            "max_abs": float(np.max(delta, initial=0.0)),
            "mean_abs": float(np.mean(delta)) if delta.size else 0.0,
            "p99_abs": float(np.percentile(delta, 99)) if delta.size else 0.0,
        }
    else:
        score_diagnostic = {
            "aligned_count": 0,
            "max_abs": None,
            "mean_abs": None,
            "p99_abs": None,
        }
    passed = not any(
        (
            not producer_exact,
            count_mismatches,
            set_mismatches,
            tail_mismatches,
            score_contract_mismatches,
        )
    )
    return {
        "actual_device_score_order_and_ties": not score_contract_mismatches,
        "count_mismatch_events": count_mismatches,
        "decode_position": decode_position,
        "exact_selected_set_and_tail": not (set_mismatches or tail_mismatches),
        "legacy_order_mismatch_count": legacy_order_mismatches,
        "legacy_total_order_match": legacy_order_mismatches == 0,
        "passed": passed,
        "producer_layer_ids_exact": producer_exact,
        "score_contract_mismatch_events": score_contract_mismatches,
        "score_diagnostic": score_diagnostic,
        "selected_set_mismatches": set_mismatches,
        "tail_mismatch_events": tail_mismatches,
    }


def validate_ws32_cache_probe(
    *,
    position: np.ndarray,
    kv_rows: np.ndarray,
    index_rows: np.ndarray,
    contract_valid: np.ndarray,
    expected_position: int,
    num_layers: int,
    full_indexer_count: int,
    packed_cache_width: int,
    index_width: int,
) -> dict[str, Any]:
    """Validate the compact all-layer write witness returned by the TPU."""

    position = np.asarray(position)
    kv_rows = np.asarray(kv_rows)
    index_rows = np.asarray(index_rows)
    contract_valid = np.asarray(contract_valid)
    shapes_exact = bool(
        position.shape == (1,)
        and kv_rows.shape == (num_layers, packed_cache_width)
        and index_rows.shape == (full_indexer_count, index_width)
        and contract_valid.shape == (1,)
    )
    finite = bool(np.all(np.isfinite(kv_rows)) and np.all(np.isfinite(index_rows)))
    kv_nonzero = (
        []
        if not shapes_exact
        else np.flatnonzero(np.any(kv_rows != 0, axis=-1)).tolist()
    )
    index_nonzero = (
        []
        if not shapes_exact
        else np.flatnonzero(np.any(index_rows != 0, axis=-1)).tolist()
    )
    passed = bool(
        shapes_exact
        and position.tolist() == [expected_position]
        and contract_valid.tolist() == [True]
        and finite
        and kv_nonzero == list(range(num_layers))
        and index_nonzero == list(range(full_indexer_count))
    )
    return {
        "contract_valid": contract_valid.tolist(),
        "finite": finite,
        "index_nonzero_slots": index_nonzero,
        "index_rows_sha256": sha256(
            np.ascontiguousarray(index_rows).view(np.uint8).tobytes()
        ).hexdigest(),
        "kv_nonzero_layers": kv_nonzero,
        "kv_rows_sha256": sha256(
            np.ascontiguousarray(kv_rows).view(np.uint8).tobytes()
        ).hexdigest(),
        "passed": passed,
        "position": position.tolist(),
        "shapes_exact": shapes_exact,
    }
