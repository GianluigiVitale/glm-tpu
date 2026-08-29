"""Fail-closed exact comparator for one PP16 feature2 numerical capture."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np

from ..errors import BenchmarkValidationError
from ..kernels.stage_local import STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE
from ..validation.short_context_dsa_oracle import inspect_short_context_dsa_oracle
from .pp16_dense_boundary import derive_expected_dense_boundary_bits
from .pp16_feature2_acquisition import inspect_feature2_event1_lineage
from .pp16_feature_sharded_state import validate_feature2_accepted_state

_BASE_CAPTURE_SCHEMA = {
    "event1_positions": ((1, 2048), np.dtype(np.int32)),
    "event1_valid_counts": ((1,), np.dtype(np.int32)),
    "event1_scores": ((1, 2048), np.dtype(np.float32)),
    "current_carried_halves_bfloat16_bits": ((2, 1, 3072), np.dtype(np.uint16)),
    "current_attention_query_owners_bfloat16_bits": (
        (2, 1, 32, 256),
        np.dtype(np.uint16),
    ),
    "current_kv_a_bfloat16_bits": ((1, 576), np.dtype(np.uint16)),
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
_SEALED_BOUNDARY_CAPTURE_SCHEMA = {
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
}
_FULL_WIDTH_CAPTURE_SCHEMA = {
    **_BASE_CAPTURE_SCHEMA,
    **_SEALED_BOUNDARY_CAPTURE_SCHEMA,
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _bitwise_mismatches(candidate: np.ndarray, expected: np.ndarray) -> int:
    if candidate.shape != expected.shape or candidate.dtype != expected.dtype:
        raise BenchmarkValidationError("feature2 exact-comparison geometry drifted")
    byte_width = candidate.dtype.itemsize
    candidate_words = np.ascontiguousarray(candidate).view(f"V{byte_width}")
    expected_words = np.ascontiguousarray(expected).view(f"V{byte_width}")
    return int(np.count_nonzero(candidate_words != expected_words))


def _compare_feature2_numerical_capture(
    capture_path: Path,
    *,
    token_oracle_dir: Path,
    dsa_oracle_dir: Path,
    layer1_internal_reference: Path,
    db529_internal_dir: Path,
    db550_boundary: Path,
    require_sealed_boundaries: bool,
) -> dict[str, Any]:
    """Authenticate sources and classify a raw capture without JAX."""

    capture_path = Path(capture_path)
    capture_sha256 = _sha256_file(capture_path)
    lineage = inspect_feature2_event1_lineage(
        token_oracle_dir=token_oracle_dir,
        dsa_oracle_dir=dsa_oracle_dir,
        layer1_internal_reference=layer1_internal_reference,
        db529_internal_dir=db529_internal_dir,
    )
    capture_schema = (
        _FULL_WIDTH_CAPTURE_SCHEMA
        if require_sealed_boundaries
        else _BASE_CAPTURE_SCHEMA
    )
    with np.load(capture_path, allow_pickle=False) as handle:
        if set(handle.files) != set(capture_schema):
            raise BenchmarkValidationError("feature2 numerical capture keys drifted")
        captured = {name: np.ascontiguousarray(handle[name]) for name in handle.files}
    if _sha256_file(capture_path) != capture_sha256:
        raise BenchmarkValidationError("feature2 numerical capture changed while read")
    for name, (shape, dtype) in capture_schema.items():
        value = captured[name]
        if value.shape != shape or value.dtype != dtype:
            raise BenchmarkValidationError(
                f"feature2 numerical capture field {name!r} drifted"
            )

    dsa_manifest = inspect_short_context_dsa_oracle(Path(dsa_oracle_dir))
    from safetensors import safe_open

    with safe_open(
        Path(dsa_oracle_dir) / dsa_manifest["files"]["tensors"]["filename"],
        framework="np",
    ) as handle:
        expected_positions = np.ascontiguousarray(
            handle.get_tensor("selected_positions")[0, 1][None, :]
        )
        expected_scores = np.ascontiguousarray(
            handle.get_tensor("selected_scores")[0, 1][None, :]
        )
        expected_valid = np.ascontiguousarray(
            handle.get_tensor("valid_counts")[0, 1][None]
        )
    if (
        _sha256_array(expected_positions[0]) != lineage["expected_positions_sha256"]
        or _sha256_array(expected_scores[0]) != lineage["expected_scores_sha256"]
        or int(expected_valid[0]) != lineage["expected_valid_count"]
    ):
        raise BenchmarkValidationError(
            "feature2 DSA target changed after authentication"
        )

    with np.load(layer1_internal_reference, allow_pickle=False) as handle:
        expected_current_key = np.ascontiguousarray(handle["accepted__current_key"])
        if require_sealed_boundaries:
            expected_normalized_hidden = np.ascontiguousarray(
                handle["accepted__normalized_hidden"]
            )
            expected_q_a_state = np.ascontiguousarray(handle["accepted__q_a_state"])
            expected_query = np.ascontiguousarray(handle["accepted__query"])
            expected_head_weights = np.ascontiguousarray(
                handle["accepted__head_weights"]
            )
    if require_sealed_boundaries:
        authenticated_layer1 = {
            "layer1_normalized_hidden_sha256": expected_normalized_hidden,
            "layer1_q_a_state_sha256": expected_q_a_state,
            "layer1_query_sha256": expected_query,
            "layer1_head_weights_sha256": expected_head_weights,
            "layer1_current_key_sha256": expected_current_key,
        }
        if any(
            _sha256_array(value) != lineage[name]
            for name, value in authenticated_layer1.items()
        ):
            raise BenchmarkValidationError(
                "feature2 layer-1 sealed boundary changed after authentication"
            )
    elif _sha256_array(expected_current_key) != lineage["layer1_current_key_sha256"]:
        raise BenchmarkValidationError(
            "feature2 layer-1 current key changed after authentication"
        )
    # Position 8155 maps to logical page 15, owner 1, owner-local row 219.
    current_key_bits = np.ascontiguousarray(
        captured["layer1_index_cache_owners_bfloat16_bits"][1, 15, 219]
    )
    expected_current_key_bits = np.ascontiguousarray(
        np.asarray(expected_current_key, dtype=ml_dtypes.bfloat16).view(np.uint16)
    )

    with np.load(db550_boundary, allow_pickle=False) as handle:
        model_axis_device_ids = tuple(
            int(item)
            for item in np.argsort(
                np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
            )
        )
        dense, expected_carried = derive_expected_dense_boundary_bits(
            handle["dense_virtual_partials_bfloat16_bits"],
            handle["post_attention_residual_bfloat16_bits"],
            model_axis_device_ids,
        )
        input_residual = np.ascontiguousarray(
            handle["post_attention_residual_bfloat16_bits"]
        )
    validate_feature2_accepted_state(dense, input_residual)
    carried = np.ascontiguousarray(
        np.concatenate(tuple(captured["current_carried_halves_bfloat16_bits"]), axis=1)
    )

    mismatch_counts = {
        "carried_bfloat16_bits": _bitwise_mismatches(carried, expected_carried),
        "contract_valid": int(not bool(captured["contract_valid"][0])),
        "event1_positions": _bitwise_mismatches(
            captured["event1_positions"], expected_positions
        ),
        "event1_scores": _bitwise_mismatches(
            captured["event1_scores"], expected_scores
        ),
        "event1_valid_counts": _bitwise_mismatches(
            captured["event1_valid_counts"], expected_valid
        ),
        "layer1_current_key_bfloat16_bits": _bitwise_mismatches(
            current_key_bits, expected_current_key_bits
        ),
    }
    if require_sealed_boundaries:
        expected_normalized_owners = np.ascontiguousarray(
            np.broadcast_to(expected_normalized_hidden, (2, 1, 6144))
        )
        expected_q_a_owners = np.ascontiguousarray(
            np.broadcast_to(expected_q_a_state, (2, 1, 2048))
        )
        expected_query_owners = np.ascontiguousarray(
            np.broadcast_to(expected_query, (2, 1, 32, 128))
        )
        expected_head_weight_owners = np.ascontiguousarray(
            np.broadcast_to(expected_head_weights, (2, 1, 32))
        )
        mismatch_counts.update(
            {
                "layer1_normalized_hidden_bfloat16_bits": _bitwise_mismatches(
                    captured["current_normalized_hidden_owners_bfloat16_bits"],
                    expected_normalized_owners,
                ),
                "layer1_q_a_state_bfloat16_bits": _bitwise_mismatches(
                    captured["current_q_a_state_owners_bfloat16_bits"],
                    expected_q_a_owners,
                ),
                "layer1_dsa_query_float32": _bitwise_mismatches(
                    captured["current_dsa_query_owners"],
                    expected_query_owners,
                ),
                "layer1_dsa_head_weights_float32": _bitwise_mismatches(
                    captured["current_dsa_head_weights_owners"],
                    expected_head_weight_owners,
                ),
            }
        )
    exact = all(value == 0 for value in mismatch_counts.values())
    if not require_sealed_boundaries:
        # This object is immutable historical evidence. Keep its report byte-for-byte
        # compatible with the sealed rejection even while v2 evolves independently.
        return {
            "artifact_kind": "greenfield_pp16_feature2_numerical_comparison",
            "capture_sha256": capture_sha256,
            "captured_array_sha256": {
                name: _sha256_array(value) for name, value in sorted(captured.items())
            },
            "claim_scope": (
                "event-1 DSA positions/scores/valid-count plus sealed current-key, "
                "carried-boundary, and contract exactness; no Gate-D, token-rate, "
                "or performance claim"
            ),
            "event1_target_lineage": lineage,
            "exact": exact,
            "expected_sha256": {
                "carried_bfloat16_bits": _sha256_array(expected_carried),
                "event1_positions": _sha256_array(expected_positions),
                "event1_scores": _sha256_array(expected_scores),
                "layer1_current_key_bfloat16_bits": _sha256_array(
                    expected_current_key_bits
                ),
                "layer1_current_key_float32_source": _sha256_array(
                    expected_current_key
                ),
            },
            "mismatch_counts": mismatch_counts,
            "performance_claim": False,
            "status": "NUMERICAL_EXACT" if exact else "NUMERICAL_REJECTED",
        }
    return {
        "artifact_kind": "greenfield_pp16_feature2_full_width_numerical_comparison",
        "capture_sha256": capture_sha256,
        "captured_array_sha256": {
            name: _sha256_array(value) for name, value in sorted(captured.items())
        },
        "claim_scope": (
            "event-1 DSA positions/scores/valid-count plus exact two-owner "
            "normalized-hidden, q-a-state, DSA-query and head-weight boundaries, "
            "current key, carried boundary and contract; no Gate-D, token-rate, "
            "or performance claim"
        ),
        "comparison_schema": "full_width_sealed_boundaries_v2",
        "event1_target_lineage": lineage,
        "exact": exact,
        "expected_sha256": {
            "carried_bfloat16_bits": _sha256_array(expected_carried),
            "event1_positions": _sha256_array(expected_positions),
            "event1_scores": _sha256_array(expected_scores),
            "layer1_current_key_bfloat16_bits": _sha256_array(
                expected_current_key_bits
            ),
            "layer1_current_key_float32_source": _sha256_array(expected_current_key),
            "layer1_normalized_hidden_bfloat16_bits": _sha256_array(
                expected_normalized_hidden
            ),
            "layer1_q_a_state_bfloat16_bits": _sha256_array(expected_q_a_state),
            "layer1_dsa_query_float32": _sha256_array(expected_query),
            "layer1_dsa_head_weights_float32": _sha256_array(expected_head_weights),
        },
        "mismatch_counts": mismatch_counts,
        "performance_claim": False,
        "sealed_boundary_comparisons_required": True,
        "status": "NUMERICAL_EXACT" if exact else "NUMERICAL_REJECTED",
    }


def compare_feature2_numerical_capture(
    capture_path: Path,
    *,
    token_oracle_dir: Path,
    dsa_oracle_dir: Path,
    layer1_internal_reference: Path,
    db529_internal_dir: Path,
    db550_boundary: Path,
) -> dict[str, Any]:
    """Re-evaluate the immutable historical feature2 capture schema."""

    return _compare_feature2_numerical_capture(
        capture_path,
        token_oracle_dir=token_oracle_dir,
        dsa_oracle_dir=dsa_oracle_dir,
        layer1_internal_reference=layer1_internal_reference,
        db529_internal_dir=db529_internal_dir,
        db550_boundary=db550_boundary,
        require_sealed_boundaries=False,
    )


def compare_feature2_full_width_numerical_capture(
    capture_path: Path,
    *,
    token_oracle_dir: Path,
    dsa_oracle_dir: Path,
    layer1_internal_reference: Path,
    db529_internal_dir: Path,
    db550_boundary: Path,
) -> dict[str, Any]:
    """Require every authenticated successor boundary from both LP2 owners."""

    return _compare_feature2_numerical_capture(
        capture_path,
        token_oracle_dir=token_oracle_dir,
        dsa_oracle_dir=dsa_oracle_dir,
        layer1_internal_reference=layer1_internal_reference,
        db529_internal_dir=db529_internal_dir,
        db550_boundary=db550_boundary,
        require_sealed_boundaries=True,
    )
