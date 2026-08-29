"""Strict offline classifier for the PP16 layer-0 position-113 observer."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

from ..errors import BenchmarkValidationError
from ..validation.prompt_index_cache import inspect_legacy_prompt_index_cache

PP16_FEATURE2_POSITION113 = 113
PP16_FEATURE2_SEALED_REJECTION_SHA256 = (
    "e514fc28e9d8c30bc7de9d70f01ae04002666494446e2f4a06a7fe6c66901e65"
)
PP16_FEATURE2_POSITION113_ORACLE_SHA256 = (
    "a2ef16a7a55876099124d0ac4bd139f86c6318b27c0e48fef5d64193ed0a3023"
)
PP16_FEATURE2_DB518_PROMPT_CACHE_MANIFEST_SHA256 = (
    "acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab"
)
PP16_FEATURE2_DB518_PROMPT_CACHE_SHA256 = (
    "3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1"
)
PP16_FEATURE2_DB518_PROMPT_CACHE_TENSOR_SHA256 = (
    "36303f0638661b4a56d3c9a1d4023a9b39eb19dbfd6d48e0718c29a45c41c07a"
)
_ORDINARY_FIELDS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "event1_positions": ((1, 2048), np.dtype(np.int32)),
    "event1_valid_counts": ((1,), np.dtype(np.int32)),
    "event1_scores": ((1, 2048), np.dtype(np.float32)),
    "current_carried_halves_bfloat16_bits": ((2, 1, 3072), np.dtype(np.uint16)),
    "current_attention_query_owners_bfloat16_bits": (
        (2, 1, 32, 256),
        np.dtype(np.uint16),
    ),
    "current_kv_a_bfloat16_bits": ((1, 576), np.dtype(np.uint16)),
    "current_normalized_hidden_owners_bfloat16_bits": (
        (2, 1, 6144),
        np.dtype(np.uint16),
    ),
    "current_q_a_state_owners_bfloat16_bits": (
        (2, 1, 2048),
        np.dtype(np.uint16),
    ),
    "current_dsa_query_owners": ((2, 1, 32, 128), np.dtype(np.float32)),
    "current_dsa_head_weights_owners": ((2, 1, 32), np.dtype(np.float32)),
    "layer0_kv_cache_owners_bfloat16_bits": (
        (2, 16, 256, 640),
        np.dtype(np.uint16),
    ),
    "layer0_index_cache_owners_bfloat16_bits": (
        (2, 16, 256, 128),
        np.dtype(np.uint16),
    ),
    "layer1_index_cache_owners_bfloat16_bits": (
        (2, 16, 256, 128),
        np.dtype(np.uint16),
    ),
    "carried_liveness_digest_owners": ((2, 2), np.dtype(np.uint32)),
    "contract_valid": ((1,), np.dtype(np.bool_)),
}
_OBSERVER_FIELDS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "position113_normalized_hidden_owners_bfloat16_bits": (
        (2, 1, 6144),
        np.dtype(np.uint16),
    ),
    "position113_q_a_state_owners_bfloat16_bits": (
        (2, 1, 2048),
        np.dtype(np.uint16),
    ),
    "position113_dsa_query_owners": ((2, 1, 32, 128), np.dtype(np.float32)),
    "position113_dsa_head_weights_owners": (
        (2, 1, 32),
        np.dtype(np.float32),
    ),
    "position113_current_key_owners": ((2, 1, 128), np.dtype(np.float32)),
    "position113_selected_positions_owners": (
        (2, 1, 2048),
        np.dtype(np.int32),
    ),
    "position113_selected_valid_counts_owners": (
        (2, 1),
        np.dtype(np.int32),
    ),
    "position113_selected_scores_owners": (
        (2, 1, 2048),
        np.dtype(np.float32),
    ),
    "position113_observation_count_owners": ((2, 1), np.dtype(np.int32)),
}
_ORACLE_FIELDS: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {
    "accepted_cache_row_bfloat16_bits": ((128,), np.dtype(np.uint16)),
    "accepted_post_rope_key": ((128,), np.dtype(np.float32)),
    "accepted_projection_input": ((6144,), np.dtype(np.float32)),
}


def feature2_position113_capture_schema() -> dict[
    str, tuple[tuple[int, ...], np.dtype[Any]]
]:
    """Return a defensive copy of the exact ordinary-plus-observer schema."""

    return {**_ORDINARY_FIELDS, **_OBSERVER_FIELDS}


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.ascontiguousarray(left).tobytes()
        == np.ascontiguousarray(right).tobytes()
    )


def _bitwise_mismatch(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise BenchmarkValidationError(
            "position-113 bitwise comparison geometry drifted"
        )
    item_bytes = left.dtype.itemsize
    left_bytes = np.ascontiguousarray(left).view(np.uint8).reshape(-1, item_bytes)
    right_bytes = np.ascontiguousarray(right).view(np.uint8).reshape(-1, item_bytes)
    unequal = np.any(left_bytes != right_bytes, axis=1)
    indices = np.flatnonzero(unequal)
    return {
        "first_mismatch_flat_index": None if not indices.size else int(indices[0]),
        "mismatch_count": int(indices.size),
    }


def _load_exact_npz(
    path: str | Path,
    expected: dict[str, tuple[tuple[int, ...], np.dtype[Any]]],
) -> tuple[dict[str, np.ndarray], str]:
    source = Path(path)
    raw = source.read_bytes()
    with np.load(BytesIO(raw), allow_pickle=False) as handle:
        if set(handle.files) != set(expected):
            raise BenchmarkValidationError(
                f"position-113 NPZ schema drifted: expected={sorted(expected)} "
                f"observed={sorted(handle.files)}"
            )
        arrays = {name: np.ascontiguousarray(handle[name]) for name in sorted(expected)}
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise BenchmarkValidationError(
                f"position-113 array {name!r} drifted: "
                f"expected={shape}/{dtype} observed={value.shape}/{value.dtype}"
            )
    return arrays, sha256(raw).hexdigest()


def _compare_feature2_position113_capture(
    capture_path: str | Path,
    *,
    sealed_rejection_path: str | Path,
    accepted_prompt_key_path: str | Path,
    accepted_prompt_cache_dir: str | Path | None,
    require_ordinary_unchanged: bool,
    require_exact_boundary: bool,
) -> dict[str, Any]:
    """Authenticate and classify either the observer or corrected DB518 boundary."""

    if require_ordinary_unchanged == require_exact_boundary:
        raise ValueError("position-113 comparison mode must be unique")
    if require_exact_boundary != (accepted_prompt_cache_dir is not None):
        raise ValueError("DB518 exact boundary requires the accepted full prompt cache")

    capture, capture_sha = _load_exact_npz(
        capture_path, feature2_position113_capture_schema()
    )
    baseline, baseline_sha = _load_exact_npz(sealed_rejection_path, _ORDINARY_FIELDS)
    oracle_all, oracle_sha = _load_exact_npz(
        accepted_prompt_key_path,
        {
            **_ORACLE_FIELDS,
            "accepted_post_rope_bfloat16_bits": (
                (128,),
                np.dtype(np.uint16),
            ),
            "accepted_pre_layer_norm_key": ((128,), np.dtype(np.float32)),
            "accepted_pre_rope_key": ((128,), np.dtype(np.float32)),
            "greenfield_cache_row_bfloat16_bits": (
                (128,),
                np.dtype(np.uint16),
            ),
            "greenfield_post_rope_bfloat16_bits": (
                (128,),
                np.dtype(np.uint16),
            ),
            "greenfield_post_rope_key": ((128,), np.dtype(np.float32)),
            "greenfield_pre_layer_norm_key": ((128,), np.dtype(np.float32)),
            "greenfield_pre_rope_key": ((128,), np.dtype(np.float32)),
            "greenfield_projection_input": ((6144,), np.dtype(np.float32)),
        },
    )
    if baseline_sha != PP16_FEATURE2_SEALED_REJECTION_SHA256:
        raise BenchmarkValidationError(
            "position-113 sealed rejection identity drifted: "
            f"expected={PP16_FEATURE2_SEALED_REJECTION_SHA256} "
            f"observed={baseline_sha}"
        )
    if oracle_sha != PP16_FEATURE2_POSITION113_ORACLE_SHA256:
        raise BenchmarkValidationError(
            "position-113 accepted oracle identity drifted: "
            f"expected={PP16_FEATURE2_POSITION113_ORACLE_SHA256} "
            f"observed={oracle_sha}"
        )
    accepted_prompt_cache: np.ndarray | None = None
    accepted_prompt_cache_manifest: dict[str, Any] | None = None
    if accepted_prompt_cache_dir is not None:
        accepted_prompt_cache_manifest, accepted_prompt_cache = (
            inspect_legacy_prompt_index_cache(
                Path(accepted_prompt_cache_dir),
                expected_manifest_sha256=(
                    PP16_FEATURE2_DB518_PROMPT_CACHE_MANIFEST_SHA256
                ),
            )
        )
        if (
            accepted_prompt_cache_manifest.get("prompt_index_key_bfloat16_sha256")
            != PP16_FEATURE2_DB518_PROMPT_CACHE_SHA256
            or accepted_prompt_cache_manifest.get("tensor_file", {}).get("sha256")
            != PP16_FEATURE2_DB518_PROMPT_CACHE_TENSOR_SHA256
            or _sha256_array(accepted_prompt_cache)
            != PP16_FEATURE2_DB518_PROMPT_CACHE_SHA256
        ):
            raise BenchmarkValidationError(
                "position-113 accepted DB518 prompt cache identity drifted"
            )
    ordinary_equal = {
        name: _bitwise_equal(capture[name], baseline[name]) for name in _ORDINARY_FIELDS
    }
    if require_ordinary_unchanged and not all(ordinary_equal.values()):
        changed = sorted(name for name, equal in ordinary_equal.items() if not equal)
        raise BenchmarkValidationError(
            f"position-113 observer changed sealed rejection outputs: {changed}"
        )

    owner_equal = {
        name: _bitwise_equal(value[0], value[1])
        for name, value in capture.items()
        if name in _OBSERVER_FIELDS
    }
    if not all(owner_equal.values()):
        changed = sorted(name for name, equal in owner_equal.items() if not equal)
        raise BenchmarkValidationError(
            f"position-113 observer owners disagree: {changed}"
        )
    counts = capture["position113_observation_count_owners"]
    if not np.array_equal(counts, np.ones((2, 1), dtype=np.int32)):
        raise BenchmarkValidationError(
            f"position-113 observer occurrence count drifted: {counts.tolist()}"
        )
    valid_counts = capture["position113_selected_valid_counts_owners"]
    if not np.array_equal(valid_counts, np.full((2, 1), 114, dtype=np.int32)):
        raise BenchmarkValidationError(
            f"position-113 selected valid count drifted: {valid_counts.tolist()}"
        )
    selected_positions = capture["position113_selected_positions_owners"][0, 0]
    selected_live = selected_positions[:114]
    selected_tail = selected_positions[114:]
    if not np.array_equal(np.sort(selected_live), np.arange(114, dtype=np.int32)):
        raise BenchmarkValidationError(
            "position-113 selected live set is not exactly positions 0..113"
        )
    if np.unique(selected_live).size != 114:
        raise BenchmarkValidationError(
            "position-113 selected live prefix contains duplicate positions"
        )
    if not np.array_equal(
        selected_tail, np.full(selected_tail.shape, -1, dtype=np.int32)
    ):
        raise BenchmarkValidationError(
            "position-113 selected position tail is not exactly -1"
        )
    selected_scores = capture["position113_selected_scores_owners"][0, 0]
    if not np.all(np.isfinite(selected_scores[:114])) or not np.all(
        np.isneginf(selected_scores[114:])
    ):
        raise BenchmarkValidationError(
            "position-113 selected score live/tail contract drifted"
        )

    normalized_bits = capture["position113_normalized_hidden_owners_bfloat16_bits"][
        0, 0
    ]
    accepted_projection = oracle_all["accepted_projection_input"]
    accepted_projection_bf16_bits = np.ascontiguousarray(
        accepted_projection.astype(ml_dtypes.bfloat16)
    ).view(np.uint16)
    normalized_matches_accepted_round = _bitwise_equal(
        normalized_bits, accepted_projection_bf16_bits
    )

    current_key = capture["position113_current_key_owners"][0, 0]
    current_key_matches_accepted_fp32 = _bitwise_equal(
        current_key, oracle_all["accepted_post_rope_key"]
    )
    current_key_bits = np.ascontiguousarray(
        current_key.astype(ml_dtypes.bfloat16)
    ).view(np.uint16)
    current_key_matches_accepted_bf16 = _bitwise_equal(
        current_key_bits, oracle_all["accepted_cache_row_bfloat16_bits"]
    )
    position_row = PP16_FEATURE2_POSITION113 % 512
    position_owner = position_row // 256
    position_local_row = position_row % 256
    position_local_page = PP16_FEATURE2_POSITION113 // 512
    candidate_cache_row = capture["layer0_index_cache_owners_bfloat16_bits"][
        position_owner, position_local_page, position_local_row
    ]
    if not _bitwise_equal(current_key_bits, candidate_cache_row):
        raise BenchmarkValidationError(
            "position-113 observed key does not reproduce the sealed candidate cache row"
        )

    if not normalized_matches_accepted_round:
        classification = "FIRST_DIVERGENCE_AT_OR_BEFORE_NORMALIZED_BF16"
    elif not current_key_matches_accepted_fp32:
        classification = "FIRST_DIVERGENCE_AFTER_NORMALIZED_AT_OR_BEFORE_CURRENT_KEY"
    elif not current_key_matches_accepted_bf16:
        classification = "FIRST_DIVERGENCE_AT_CURRENT_KEY_BF16_ROUND"
    else:
        classification = "OBSERVED_NORMALIZED_AND_CURRENT_KEY_PATH_EXACT"

    observer_hashes = {
        name: _sha256_array(capture[name]) for name in sorted(_OBSERVER_FIELDS)
    }
    prompt_cache_exact: bool | None = None
    prompt_cache_diff: dict[str, Any] | None = None
    candidate_prompt_cache_sha256: str | None = None
    if accepted_prompt_cache is not None:
        positions = np.arange(accepted_prompt_cache.shape[0], dtype=np.int64)
        rows = positions % 512
        candidate_prompt_cache = np.ascontiguousarray(
            capture["layer0_index_cache_owners_bfloat16_bits"][
                rows // 256, positions // 512, rows % 256
            ]
        )
        prompt_cache_diff = _bitwise_mismatch(
            candidate_prompt_cache, accepted_prompt_cache
        )
        prompt_cache_exact = prompt_cache_diff["mismatch_count"] == 0
        candidate_prompt_cache_sha256 = _sha256_array(candidate_prompt_cache)
    boundary_exact = bool(
        normalized_matches_accepted_round
        and current_key_matches_accepted_fp32
        and current_key_matches_accepted_bf16
        and prompt_cache_exact is not False
    )
    if require_exact_boundary:
        return {
            "accepted_oracle_scope": {
                "current_key_bfloat16": True,
                "current_key_float32": True,
                "normalized_bfloat16_round_of_fp32_projection_input": True,
                "q_a_query_head_selected_set": False,
            },
            "accepted_prompt_key_npz_sha256": oracle_sha,
            "artifact_kind": "greenfield_pp16_feature2_db518_position113_comparison",
            "baseline_capture_npz_sha256": baseline_sha,
            "boundary_exact": boundary_exact,
            "cache_ownership": {
                "live_prompt_row_count": int(accepted_prompt_cache.shape[0]),
                "owner": position_owner,
                "owner_local_page": position_local_page,
                "owner_local_row": position_local_row,
                "page_rows": 512,
                "rows_per_owner": 256,
            },
            "candidate_prompt_cache_sha256": candidate_prompt_cache_sha256,
            "capture_npz_sha256": capture_sha,
            "classification": classification,
            "current_key_matches_accepted_bfloat16": (
                current_key_matches_accepted_bf16
            ),
            "current_key_matches_accepted_float32": current_key_matches_accepted_fp32,
            "current_key_bfloat16_diff": _bitwise_mismatch(
                current_key_bits, oracle_all["accepted_cache_row_bfloat16_bits"]
            ),
            "current_key_float32_diff": _bitwise_mismatch(
                current_key, oracle_all["accepted_post_rope_key"]
            ),
            "normalized_matches_accepted_bfloat16_round": (
                normalized_matches_accepted_round
            ),
            "normalized_bfloat16_diff": _bitwise_mismatch(
                normalized_bits, accepted_projection_bf16_bits
            ),
            "numerical_exactness_claim": False,
            "observer_array_sha256": observer_hashes,
            "observer_owner_bitwise_equal": owner_equal,
            "ordinary_output_bitwise_equal_to_sealed_rejection": ordinary_equal,
            "performance_claim": False,
            "position": PP16_FEATURE2_POSITION113,
            "prompt_cache_diff": prompt_cache_diff,
            "prompt_cache_matches_accepted": prompt_cache_exact,
            "accepted_prompt_cache_manifest_sha256": (
                PP16_FEATURE2_DB518_PROMPT_CACHE_MANIFEST_SHA256
            ),
            "accepted_prompt_cache_sha256": (PP16_FEATURE2_DB518_PROMPT_CACHE_SHA256),
            "accepted_prompt_cache_tensor_sha256": (
                PP16_FEATURE2_DB518_PROMPT_CACHE_TENSOR_SHA256
            ),
            "status": (
                "DB518_POSITION113_EXACT"
                if boundary_exact
                else "DB518_POSITION113_REJECTED"
            ),
        }
    return {
        "accepted_oracle_scope": {
            "current_key_bfloat16": True,
            "current_key_float32": True,
            "normalized_bfloat16_round_of_fp32_projection_input": True,
            "q_a_query_head_selected_set": False,
        },
        "accepted_prompt_key_npz_sha256": oracle_sha,
        "baseline_capture_npz_sha256": baseline_sha,
        "capture_npz_sha256": capture_sha,
        "classification": classification,
        "current_key_matches_accepted_bfloat16": (current_key_matches_accepted_bf16),
        "current_key_matches_accepted_float32": current_key_matches_accepted_fp32,
        "current_key_bfloat16_diff": _bitwise_mismatch(
            current_key_bits, oracle_all["accepted_cache_row_bfloat16_bits"]
        ),
        "current_key_float32_diff": _bitwise_mismatch(
            current_key, oracle_all["accepted_post_rope_key"]
        ),
        "normalized_matches_accepted_bfloat16_round": (
            normalized_matches_accepted_round
        ),
        "normalized_bfloat16_diff": _bitwise_mismatch(
            normalized_bits, accepted_projection_bf16_bits
        ),
        "numerical_exactness_claim": False,
        "observer_array_sha256": observer_hashes,
        "observer_owner_bitwise_equal": owner_equal,
        "ordinary_output_bitwise_equal": ordinary_equal,
        "performance_claim": False,
        "position": PP16_FEATURE2_POSITION113,
        "status": "POSITION113_CAPTURE_CLASSIFIED",
    }


def compare_feature2_position113_capture(
    capture_path: str | Path,
    *,
    sealed_rejection_path: str | Path,
    accepted_prompt_key_path: str | Path,
) -> dict[str, Any]:
    """Classify p113 while proving the observer did not affect ordinary outputs."""

    return _compare_feature2_position113_capture(
        capture_path,
        sealed_rejection_path=sealed_rejection_path,
        accepted_prompt_key_path=accepted_prompt_key_path,
        accepted_prompt_cache_dir=None,
        require_ordinary_unchanged=True,
        require_exact_boundary=False,
    )


def compare_feature2_db518_position113_capture(
    capture_path: str | Path,
    *,
    sealed_rejection_path: str | Path,
    accepted_prompt_key_path: str | Path,
    accepted_prompt_cache_dir: str | Path,
) -> dict[str, Any]:
    """Require the corrected DB518 p113 key/cache boundary to be bitwise exact."""

    return _compare_feature2_position113_capture(
        capture_path,
        sealed_rejection_path=sealed_rejection_path,
        accepted_prompt_key_path=accepted_prompt_key_path,
        accepted_prompt_cache_dir=accepted_prompt_cache_dir,
        require_ordinary_unchanged=False,
        require_exact_boundary=True,
    )
