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


def inspect_distributed_q_a_norm_artifact(
    artifact_dir: Path,
    *,
    expected_manifest_sha256: str,
    expected_code_hash: str,
    expected_input_manifest_sha256: str,
) -> tuple[dict[str, Any], np.ndarray]:
    """Verify and load the immutable DB491 distributed q-a result."""

    from safetensors import safe_open

    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    if manifest.get("artifact_kind") != "greenfield_distributed_q_a_norm" or (
        manifest.get("format_version") != 1
        or manifest.get("diagnostic_only") is not True
    ):
        raise ValueError("unsupported distributed q-a norm artifact")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest) or (
        manifest.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise ValueError("distributed q-a norm manifest checksum mismatch")
    if manifest.get("code_hash") != expected_code_hash or (
        manifest.get("input_manifest_sha256")
        != expected_input_manifest_sha256
    ):
        raise ValueError("distributed q-a norm provenance drifted")
    if manifest.get("model_config") != {
        "path": "reference/hf-repo/config.json",
        "sha256": (
            "22e49334abf8562fecf70ca3292ba3f5b33f5602fb2bf10b52dd64a66cfe65ff"
        ),
        "rms_norm_eps": 1e-5,
    } or manifest.get("numerical_geometry") != {
        "input_rms_norm_epsilon": 1e-5,
        "q_a_rms_norm_epsilon": 1e-5,
        "key_layer_norm_epsilon": 1e-6,
    }:
        raise ValueError("distributed q-a norm numerical provenance drifted")
    file_record = manifest.get("file", {})
    if file_record.get("filename") != "distributed_q_a_norm.safetensors":
        raise ValueError("distributed q-a norm filename drifted")
    tensor_path = artifact_dir / file_record["filename"]
    if tensor_path.stat().st_size != file_record.get("byte_count") or (
        _sha256_file(tensor_path) != file_record.get("sha256")
    ):
        raise ValueError("distributed q-a norm tensor integrity failed")
    with safe_open(tensor_path, framework="np") as handle:
        if handle.metadata() != {
            "artifact_kind": "greenfield_distributed_q_a_norm",
            "format_version": "1",
        } or list(handle.keys()) != ["q_residual_bfloat16_bits"]:
            raise ValueError("distributed q-a norm tensor metadata drifted")
        q_bits = handle.get_tensor("q_residual_bfloat16_bits").copy()
    q_record = manifest.get("q_residual", {})
    if q_bits.shape != (32, 2048) or q_bits.dtype != np.uint16 or (
        q_record.get("shape") != [32, 2048]
        or q_record.get("dtype") != "bfloat16"
        or q_record.get("byte_count") != q_bits.nbytes
        or q_record.get("sha256")
        != sha256(q_bits.view(np.uint8)).hexdigest()
    ):
        raise ValueError("distributed q-a norm tensor contract drifted")
    return manifest, q_bits


def inspect_greenfield_layer0_dsa_internal_observation(
    artifact_dir: Path,
    *,
    expected_contract_sha256: str,
    expected_tensor_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Verify the sealed all-event observer and its exact layer-0 boundary."""

    artifact_dir = Path(artifact_dir)
    contract_path = artifact_dir / "contract.json"
    tensor_path = artifact_dir / "position_8155_internals.npz"
    if _sha256_file(contract_path) != expected_contract_sha256:
        raise ValueError("greenfield DSA internal contract identity drifted")
    if _sha256_file(tensor_path) != expected_tensor_sha256:
        raise ValueError("greenfield DSA internal tensor identity drifted")
    contract = json.loads(contract_path.read_text())
    expected_keys = {
        "current_key",
        "decode_position",
        "head_weights",
        "normalized_hidden_bfloat16_bits",
        "producer_layer_ids",
        "q_a_state_bfloat16_bits",
        "query",
    }
    with np.load(tensor_path, allow_pickle=False) as bundle:
        if set(bundle.files) != expected_keys:
            raise ValueError("greenfield DSA internal tensor keys drifted")
        arrays = {name: np.asarray(bundle[name]).copy() for name in bundle.files}
    event_count = 21
    storage_contract = {
        "normalized_hidden": (
            "normalized_hidden_bfloat16_bits",
            (event_count, 6144),
            np.dtype(np.uint16),
            "bfloat16",
        ),
        "q_a_state": (
            "q_a_state_bfloat16_bits",
            (event_count, 2048),
            np.dtype(np.uint16),
            "bfloat16",
        ),
        "query": (
            "query",
            (event_count, 32, 128),
            np.dtype(np.float32),
            "float32",
        ),
        "head_weights": (
            "head_weights",
            (event_count, 32),
            np.dtype(np.float32),
            "float32",
        ),
        "current_key": (
            "current_key",
            (event_count, 128),
            np.dtype(np.float32),
            "float32",
        ),
        "producer_layer_ids": (
            "producer_layer_ids",
            (event_count,),
            np.dtype(np.int32),
            "int32",
        ),
    }
    if contract.get("decode_position") != 8155 or (
        contract.get("event_count") != event_count
        or arrays["decode_position"].shape != (1,)
        or arrays["decode_position"].dtype != np.dtype(np.int32)
        or arrays["decode_position"].tolist() != [8155]
    ):
        raise ValueError("greenfield DSA internal event geometry drifted")
    for logical_name, (
        storage_name,
        shape,
        dtype,
        logical_dtype,
    ) in storage_contract.items():
        value = np.ascontiguousarray(arrays[storage_name])
        record = contract.get("field_records", {}).get(logical_name, {})
        if value.shape != shape or value.dtype != dtype or record != {
            "dtype": logical_dtype,
            "sha256": sha256(value.tobytes(order="C")).hexdigest(),
            "shape": list(shape),
        }:
            raise ValueError(
                f"greenfield DSA internal field drifted: {logical_name}"
            )
    producers = arrays["producer_layer_ids"]
    if producers.tolist() != contract.get("producer_layer_ids") or (
        np.flatnonzero(producers == 0).tolist() != [0]
    ):
        raise ValueError("greenfield DSA internal producer identity drifted")
    layer0 = contract.get("layer0_reference", {})
    for logical_name, (storage_name, _, _, _) in storage_contract.items():
        if logical_name == "producer_layer_ids":
            continue
        value = np.ascontiguousarray(arrays[storage_name][0])
        digest = sha256(value.tobytes(order="C")).hexdigest()
        reference = layer0.get(logical_name, {})
        if (
            reference.get("elementwise_exact") is not True
            or reference.get("mismatch_count") != 0
            or reference.get("max_abs") != 0.0
            or reference.get("actual_sha256") != digest
            or reference.get("expected_sha256") != digest
            or reference.get("shape") != list(value.shape)
        ):
            raise ValueError(
                f"greenfield layer-0 DSA reference is not exact: {logical_name}"
            )
    if contract.get("layer0_query_exact") is not True or (
        contract.get("lane_mismatches") != []
        or contract.get("padded_slot_mismatches") != []
    ):
        raise ValueError("greenfield layer-0 DSA observer contract drifted")
    return contract, arrays


def inspect_greenfield_layer0_dsa_selected_observation(
    artifact_path: Path,
    *,
    expected_sha256: str,
    local_parallel_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the current protected layer-0 positions/scores after validation."""

    artifact_path = Path(artifact_path)
    if _sha256_file(artifact_path) != expected_sha256:
        raise ValueError("greenfield selected-score observation identity drifted")
    if (
        not isinstance(local_parallel_size, int)
        or isinstance(local_parallel_size, bool)
        or local_parallel_size <= 0
    ):
        raise ValueError("local parallel size must be positive")
    with np.load(artifact_path, allow_pickle=False) as bundle:
        if set(bundle.files) != {
            "decode_position",
            "observation",
            "token_observation",
        }:
            raise ValueError("greenfield selected-score fields drifted")
        decode_position = np.asarray(bundle["decode_position"])
        observation = np.asarray(bundle["observation"])
        token_observation = np.asarray(bundle["token_observation"])
    if (
        decode_position.shape != (1,)
        or decode_position.dtype != np.dtype(np.int32)
        or decode_position.tolist() != [8155]
        or observation.shape != (32, 5, 4098)
        or observation.dtype != np.dtype(np.int32)
        or token_observation.shape != (32, 32)
        or token_observation.dtype != np.dtype(np.int32)
    ):
        raise ValueError("greenfield selected-score tensor contract drifted")
    lanes = observation[:local_parallel_size]
    if not np.all(lanes == lanes[0]):
        raise ValueError("greenfield layer-0 selected-score lanes disagree")
    row = lanes[0, 0]
    positions = row[:2048].copy()
    scores = np.ascontiguousarray(row[2048:4096]).view(np.float32).copy()
    if int(row[4096]) != 2048 or int(row[4097]) != 0:
        raise ValueError("greenfield layer-0 selected-score identity drifted")
    if (
        np.unique(positions).size != positions.size
        or np.any(positions < 0)
        or np.any(positions > 8155)
        or not np.isfinite(scores).all()
        or not np.array_equal(
            np.lexsort((positions, -scores)),
            np.arange(positions.size),
        )
    ):
        raise ValueError("greenfield layer-0 selected-score order drifted")
    return positions, scores


def pack_stage_local_index_keys(
    prompt_bfloat16_bits: np.ndarray,
    current_key: np.ndarray,
    *,
    logical_page_size: int = 512,
    local_parallel_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append the current key and pack exact LP4 page-striped cache lanes."""

    import ml_dtypes

    prompt = np.asarray(prompt_bfloat16_bits)
    current = np.asarray(current_key, dtype=np.float32)
    if (
        prompt.ndim != 2
        or prompt.dtype != np.dtype(np.uint16)
        or current.shape != (prompt.shape[1],)
        or not np.isfinite(current).all()
        or not isinstance(logical_page_size, int)
        or isinstance(logical_page_size, bool)
        or not isinstance(local_parallel_size, int)
        or isinstance(local_parallel_size, bool)
        or logical_page_size <= 0
        or local_parallel_size <= 0
        or logical_page_size % local_parallel_size
    ):
        raise ValueError("stage-local index-key packing contract drifted")
    current_bits = np.ascontiguousarray(
        current.astype(ml_dtypes.bfloat16)
    ).view(np.uint16)
    global_bits = np.concatenate((prompt, current_bits[None, :]), axis=0)
    context = global_bits.shape[0]
    local_rows = logical_page_size // local_parallel_size
    page_count = (context + logical_page_size - 1) // logical_page_size
    lane_width = page_count * local_rows
    row = np.arange(local_rows, dtype=np.int32)
    lanes = np.zeros(
        (local_parallel_size, lane_width, prompt.shape[1]),
        dtype=np.uint16,
    )
    positions = np.full(
        (local_parallel_size, lane_width), -1, dtype=np.int32
    )
    for lane in range(local_parallel_size):
        global_positions = (
            np.arange(page_count, dtype=np.int32)[:, None]
            * np.int32(logical_page_size)
            + np.int32(lane * local_rows)
            + row[None, :]
        ).reshape(-1)
        valid = global_positions < context
        positions[lane, valid] = global_positions[valid]
        lanes[lane, valid] = global_bits[global_positions[valid]]
    live_positions = positions[positions >= 0]
    if not np.array_equal(np.sort(live_positions), np.arange(context)):
        raise ValueError("stage-local index-key packing lost context positions")
    return global_bits, lanes, positions


def stitch_stage_local_scores(
    lane_scores: np.ndarray,
    lane_positions: np.ndarray,
    *,
    context: int,
) -> np.ndarray:
    """Stitch fixed-width LP4 score lanes into one logical score row."""

    scores = np.asarray(lane_scores, dtype=np.float32)
    positions = np.asarray(lane_positions)
    if (
        scores.ndim != 2
        or positions.shape != scores.shape
        or positions.dtype != np.dtype(np.int32)
        or not isinstance(context, int)
        or isinstance(context, bool)
        or context <= 0
    ):
        raise ValueError("stage-local score stitching contract drifted")
    valid = positions >= 0
    live_positions = positions[valid]
    if (
        np.any(positions[~valid] != -1)
        or np.any(live_positions >= context)
        or not np.isfinite(scores[valid]).all()
        or not np.array_equal(np.sort(live_positions), np.arange(context))
    ):
        raise ValueError("stage-local score positions are incomplete")
    logical = np.empty((context,), dtype=np.float32)
    logical[live_positions] = scores[valid]
    return logical


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
