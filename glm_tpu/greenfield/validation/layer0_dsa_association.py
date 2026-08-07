"""Validation for the bounded sealed layer-0 DSA association probe."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np


ARTIFACT_KIND = "greenfield_layer0_dsa_association_input"
FORMAT_VERSION = 2
SUPPORTED_FORMAT_VERSIONS = (1, FORMAT_VERSION)
MODEL_ID = "zai-org/GLM-5.2-FP8"


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _manifest_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    return sha256(_canonical_json(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_record(name: str, value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "byte_count": int(contiguous.nbytes),
        "dtype": str(contiguous.dtype),
        "name": name,
        "sha256": sha256(contiguous.tobytes(order="C")).hexdigest(),
        "shape": list(contiguous.shape),
    }


def _require_digest(value: Any, *, field: str) -> None:
    if not isinstance(value, str) or len(value) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"layer-0 DSA {field} is not a lowercase digest")


def inspect_layer0_dsa_association_input(
    input_dir: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Verify the append-only real-input artifact and return copied arrays."""

    from safetensors import safe_open

    input_dir = Path(input_dir)
    manifest = json.loads((input_dir / "manifest.json").read_text())
    format_version = manifest.get("format_version")
    if manifest.get("artifact_kind") != ARTIFACT_KIND or (
        format_version not in SUPPORTED_FORMAT_VERSIONS
    ):
        raise ValueError("unsupported layer-0 DSA association input")
    if manifest.get("model_id") != MODEL_ID or (
        manifest.get("diagnostic_only") is not True
    ):
        raise ValueError("layer-0 DSA artifact scope/model drifted")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ValueError("layer-0 DSA manifest checksum mismatch")
    if expected_manifest_sha256 is not None and (
        manifest["manifest_sha256"] != expected_manifest_sha256
    ):
        raise ValueError("layer-0 DSA expected manifest identity drifted")
    for field in (
        "code_hash",
        "dsa_oracle_manifest_sha256",
        "token_oracle_manifest_sha256",
    ):
        _require_digest(manifest.get(field), field=field)
    if manifest.get("event_index") != 0 or (
        manifest.get("producer_layer_id") != 0
    ):
        raise ValueError("layer-0 DSA event identity drifted")

    file_record = manifest.get("file", {})
    if file_record.get("filename") != "layer0_dsa_input.safetensors":
        raise ValueError("layer-0 DSA tensor filename drifted")
    artifact_path = input_dir / file_record["filename"]
    if artifact_path.stat().st_size != file_record.get("byte_count") or (
        _sha256_file(artifact_path) != file_record.get("sha256")
    ):
        raise ValueError("layer-0 DSA tensor file integrity failed")

    unique = int(manifest.get("unique_token_count", -1))
    if unique <= 0:
        raise ValueError("layer-0 DSA unique-token count drifted")
    expected_contract = {
        "current_token_id": ((1,), np.dtype(np.int32)),
        "decode_position": ((1,), np.dtype(np.int32)),
        "expected_selected_count": ((1,), np.dtype(np.int32)),
        "expected_selected_positions": ((2048,), np.dtype(np.int32)),
        "expected_selected_scores": ((2048,), np.dtype(np.float32)),
        "input_layernorm__weight": ((6144,), np.dtype(np.uint16)),
        "prompt_token_ids": ((8155,), np.dtype(np.int32)),
        "self_attn__indexer__k_norm__bias": ((128,), np.dtype(np.uint16)),
        "self_attn__indexer__k_norm__weight": ((128,), np.dtype(np.uint16)),
        "self_attn__indexer__weights_proj__weight": (
            (32, 6144),
            np.dtype(np.uint16),
        ),
        "self_attn__indexer__wk__weight": ((128, 6144), np.dtype(np.uint8)),
        "self_attn__indexer__wk__weight_scale_inv": (
            (1, 48),
            np.dtype(np.float32),
        ),
        "self_attn__indexer__wq_b__weight": (
            (4096, 2048),
            np.dtype(np.uint8),
        ),
        "self_attn__indexer__wq_b__weight_scale_inv": (
            (32, 16),
            np.dtype(np.float32),
        ),
        "self_attn__q_a_layernorm__weight": (
            (2048,),
            np.dtype(np.uint16),
        ),
        "self_attn__q_a_proj__weight": ((2048, 6144), np.dtype(np.uint8)),
        "self_attn__q_a_proj__weight_scale_inv": (
            (16, 48),
            np.dtype(np.float32),
        ),
        "unique_embedding_bfloat16_bits": (
            (unique, 6144),
            np.dtype(np.uint16),
        ),
        "unique_token_ids": ((unique,), np.dtype(np.int32)),
    }
    if format_version >= 2:
        expected_contract.update(
            {
                "self_attn__kv_a_proj_with_mqa__weight": (
                    (576, 6144),
                    np.dtype(np.uint8),
                ),
                "self_attn__kv_a_proj_with_mqa__weight_scale_inv": (
                    (5, 48),
                    np.dtype(np.float32),
                ),
            }
        )
    with safe_open(artifact_path, framework="np") as handle:
        if handle.metadata() != {
            "artifact_kind": ARTIFACT_KIND,
            "format_version": str(format_version),
        }:
            raise ValueError("layer-0 DSA tensor metadata drifted")
        if set(handle.keys()) != set(expected_contract):
            raise ValueError("layer-0 DSA tensor key set drifted")
        arrays = {
            name: handle.get_tensor(name).copy() for name in handle.keys()
        }
    for name, (shape, dtype) in expected_contract.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"layer-0 DSA array contract drifted: {name}")
        if manifest.get("arrays", {}).get(name) != _array_record(name, value):
            raise ValueError(f"layer-0 DSA array checksum drifted: {name}")

    unique_ids = arrays["unique_token_ids"]
    if not np.array_equal(unique_ids, np.unique(unique_ids)):
        raise ValueError("layer-0 DSA unique token IDs are not canonical")
    prompt_ids = arrays["prompt_token_ids"]
    current_id = arrays["current_token_id"]
    if not np.isin(prompt_ids, unique_ids).all() or (
        not np.isin(current_id, unique_ids).all()
    ):
        raise ValueError("layer-0 DSA embedding-row coverage is incomplete")
    if arrays["decode_position"].tolist() != [8155] or (
        arrays["expected_selected_count"].tolist() != [2048]
    ):
        raise ValueError("layer-0 DSA protected event geometry drifted")
    positions = arrays["expected_selected_positions"]
    scores = arrays["expected_selected_scores"]
    if np.unique(positions).size != 2048 or np.any(positions < 0) or (
        np.any(positions > 8155)
    ):
        raise ValueError("layer-0 DSA sealed positions are not unique/causal")
    if not np.isfinite(scores).all():
        raise ValueError("layer-0 DSA sealed scores are non-finite")
    canonical = np.lexsort((positions, -scores))
    if not np.array_equal(canonical, np.arange(2048)):
        raise ValueError("layer-0 DSA sealed score/tie order drifted")

    source_index = manifest.get("source_index", {})
    if source_index.get("filename") != "model.safetensors.index.json":
        raise ValueError("layer-0 DSA source index identity drifted")
    _require_digest(source_index.get("sha256"), field="source_index.sha256")
    expected_source_records = 14 if format_version >= 2 else 12
    if len(manifest.get("source_records", [])) != expected_source_records:
        raise ValueError("layer-0 DSA source tensor coverage drifted")
    return manifest, arrays


def compare_dsa_association_scores(
    scores: np.ndarray,
    expected_positions: np.ndarray,
    expected_scores: np.ndarray,
) -> dict[str, Any]:
    """Compare one complete score row with exact sealed set/tie semantics."""

    scores = np.asarray(scores, dtype=np.float32)
    expected_positions = np.asarray(expected_positions, dtype=np.int32)
    expected_scores = np.asarray(expected_scores, dtype=np.float32)
    if scores.ndim != 1 or expected_positions.ndim != 1 or (
        expected_scores.shape != expected_positions.shape
    ):
        raise ValueError("DSA association comparison shapes drifted")
    top_k = int(expected_positions.size)
    if top_k <= 0 or scores.size < top_k:
        raise ValueError("DSA association comparison width is invalid")
    if not np.isfinite(scores).all() or not np.isfinite(expected_scores).all():
        raise ValueError("DSA association comparison requires finite scores")
    if np.any(expected_positions < 0) or np.any(expected_positions >= scores.size):
        raise ValueError("DSA association expected position is out of range")

    positions = np.arange(scores.size, dtype=np.int32)
    selected = np.lexsort((positions, -scores))[:top_k].astype(np.int32)
    selected_set = set(selected.tolist())
    expected_set = set(expected_positions.tolist())
    expected_aligned = scores[expected_positions]
    delta = expected_aligned - expected_scores
    absolute = np.abs(delta)
    actual_std = float(expected_aligned.astype(np.float64).std())
    expected_std = float(expected_scores.astype(np.float64).std())
    if actual_std == 0.0 or expected_std == 0.0:
        correlation = float(np.array_equal(expected_aligned, expected_scores))
    else:
        correlation = float(
            np.corrcoef(
                expected_aligned.astype(np.float64),
                expected_scores.astype(np.float64),
            )[0, 1]
        )
    set_exact = selected_set == expected_set
    order_exact = bool(np.array_equal(selected, expected_positions))
    return {
        "context": int(scores.size),
        "top_k": top_k,
        "selected_set_exact": set_exact,
        "selected_order_exact": order_exact,
        "position_mismatch_count": int(
            np.count_nonzero(selected != expected_positions)
        ),
        "swapped_position_count": int(len(selected_set - expected_set)),
        "score_all_finite": True,
        "score_max_abs": float(absolute.max(initial=0.0)),
        "score_mean_abs": float(absolute.mean()),
        "score_p99_abs": float(np.percentile(absolute, 99)),
        "score_mean_signed": float(delta.mean()),
        "score_correlation": correlation,
        "actual_cutoff_score": float(scores[selected[-1]]),
        "expected_cutoff_score": float(expected_scores[-1]),
        "actual_top_positions": selected[:16].tolist(),
        "passed": bool(set_exact and order_exact),
    }
