from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.pp16_feature2_position113 import (
    compare_feature2_position113_capture,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError

BASELINE = Path(
    "/home/gianl/glm-run/greenfield_pp16_feature2_prefill_numerical_"
    "20260829T051119686986506Z/result.npz"
)
ORACLE = Path(
    "/home/gianl/gcs-models/oracles/greenfield/glm52/prompt_projection_input/8k/"
    "greenfield_legacy_layer0_prompt_projection_input_p113_"
    "20260809T050055956585082Z/prompt_projection_input_comparison/"
    "prompt_key_internal_comparison.npz"
)
PROTECTED_RUN = Path(
    "/home/gianl/glm-run/"
    "greenfield_pp16_feature2_position113_numerical_"
    "20260829T085708974224501Z"
)
COMPACT_ARTIFACT = Path(__file__).parents[3] / (
    "docs/artifacts/pp16-feature2-position113-numerical.json"
)


def _write_candidate(path: Path) -> None:
    with np.load(BASELINE, allow_pickle=False) as handle:
        arrays = {name: np.ascontiguousarray(handle[name]) for name in handle.files}
    with np.load(ORACLE, allow_pickle=False) as handle:
        accepted_projection = np.ascontiguousarray(handle["accepted_projection_input"])
    normalized_bits = np.ascontiguousarray(
        accepted_projection.astype(ml_dtypes.bfloat16)
    ).view(np.uint16)
    candidate_key_bits = arrays["layer0_index_cache_owners_bfloat16_bits"][0, 0, 113]
    candidate_key = candidate_key_bits.view(ml_dtypes.bfloat16).astype(np.float32)
    selected = np.full((2, 1, 2048), -1, dtype=np.int32)
    selected[:, 0, :114] = np.arange(114, dtype=np.int32)
    selected_scores = np.full((2, 1, 2048), -np.inf, dtype=np.float32)
    selected_scores[:, 0, :114] = np.float32(0.0)
    arrays.update(
        {
            "position113_normalized_hidden_owners_bfloat16_bits": np.broadcast_to(
                normalized_bits, (2, 1, 6144)
            ).copy(),
            "position113_q_a_state_owners_bfloat16_bits": np.zeros(
                (2, 1, 2048), dtype=np.uint16
            ),
            "position113_dsa_query_owners": np.zeros((2, 1, 32, 128), dtype=np.float32),
            "position113_dsa_head_weights_owners": np.zeros(
                (2, 1, 32), dtype=np.float32
            ),
            "position113_current_key_owners": np.broadcast_to(
                candidate_key, (2, 1, 128)
            ).copy(),
            "position113_selected_positions_owners": selected,
            "position113_selected_valid_counts_owners": np.full(
                (2, 1), 114, dtype=np.int32
            ),
            "position113_selected_scores_owners": selected_scores,
            "position113_observation_count_owners": np.ones((2, 1), dtype=np.int32),
        }
    )
    np.savez(path, **arrays)


@pytest.mark.skipif(
    not BASELINE.is_file() or not ORACLE.is_file(),
    reason="sealed PP16 rejection or accepted p113 oracle is unavailable",
)
def test_position113_classifier_preserves_baseline_and_limits_claim(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture.npz"
    _write_candidate(capture)
    report = compare_feature2_position113_capture(
        capture,
        sealed_rejection_path=BASELINE,
        accepted_prompt_key_path=ORACLE,
    )
    assert report["status"] == "POSITION113_CAPTURE_CLASSIFIED"
    assert report["classification"] == (
        "FIRST_DIVERGENCE_AFTER_NORMALIZED_AT_OR_BEFORE_CURRENT_KEY"
    )
    assert report["normalized_matches_accepted_bfloat16_round"] is True
    assert report["current_key_matches_accepted_bfloat16"] is False
    assert report["current_key_matches_accepted_float32"] is False
    assert report["current_key_bfloat16_diff"] == {
        "first_mismatch_flat_index": 35,
        "mismatch_count": 1,
    }
    assert report["normalized_bfloat16_diff"] == {
        "first_mismatch_flat_index": None,
        "mismatch_count": 0,
    }
    assert report["numerical_exactness_claim"] is False
    assert report["performance_claim"] is False
    assert report["accepted_oracle_scope"]["q_a_query_head_selected_set"] is False
    assert all(report["ordinary_output_bitwise_equal"].values())
    assert all(report["observer_owner_bitwise_equal"].values())


@pytest.mark.skipif(
    not all(
        path.is_file()
        for path in (
            PROTECTED_RUN / "result.npz",
            PROTECTED_RUN / "comparison.json",
            PROTECTED_RUN / "summary.json",
            PROTECTED_RUN / "POSITION113_CAPTURE_CLASSIFIED",
            BASELINE,
            ORACLE,
            COMPACT_ARTIFACT,
        )
    ),
    reason="protected position-113 result is unavailable",
)
def test_position113_compact_artifact_matches_protected_result() -> None:
    artifact = json.loads(COMPACT_ARTIFACT.read_text())
    result = PROTECTED_RUN / "result.npz"
    comparison_path = PROTECTED_RUN / "comparison.json"
    summary_path = PROTECTED_RUN / "summary.json"
    terminal_path = PROTECTED_RUN / "POSITION113_CAPTURE_CLASSIFIED"
    report = compare_feature2_position113_capture(
        result,
        sealed_rejection_path=BASELINE,
        accepted_prompt_key_path=ORACLE,
    )
    comparison = json.loads(comparison_path.read_text())
    summary = json.loads(summary_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    assert report == comparison
    assert artifact["status"] == report["status"]
    assert artifact["code_hash"] == summary["code_hash"]
    assert artifact["classification"]["classification"] == report["classification"]
    assert artifact["classification"]["ordinary_outputs_all_bitwise_equal"] is all(
        report["ordinary_output_bitwise_equal"].values()
    )
    assert artifact["classification"]["observer_owners_all_bitwise_equal"] is all(
        report["observer_owner_bitwise_equal"].values()
    )
    assert artifact["integrity"] == {
        **artifact["integrity"],
        "comparison_sha256": sha256(comparison_path.read_bytes()).hexdigest(),
        "result_npz_sha256": sha256(result.read_bytes()).hexdigest(),
        "summary_sha256": sha256(summary_path.read_bytes()).hexdigest(),
    }
    assert (
        artifact["archive"]["terminal_file_sha256"]
        == sha256(terminal_path.read_bytes()).hexdigest()
    )
    assert (
        artifact["archive"]["terminal_marker_self_sha256"]
        == terminal["marker_self_sha256"]
    )
    with (
        np.load(result, allow_pickle=False) as capture,
        np.load(ORACLE, allow_pickle=False) as oracle,
    ):
        candidate = np.ascontiguousarray(
            capture["position113_current_key_owners"][0, 0].astype(ml_dtypes.bfloat16)
        ).view(np.uint16)
        accepted = np.ascontiguousarray(oracle["accepted_cache_row_bfloat16_bits"])
        greenfield = np.ascontiguousarray(oracle["greenfield_cache_row_bfloat16_bits"])
    assert artifact["classification"]["candidate_current_key_bfloat16_sha256"] == (
        sha256(candidate.tobytes()).hexdigest()
    )
    assert artifact["classification"]["accepted_current_key_bfloat16_sha256"] == (
        sha256(accepted.tobytes()).hexdigest()
    )
    assert artifact["classification"]["greenfield_current_key_bfloat16_sha256"] == (
        sha256(greenfield.tobytes()).hexdigest()
    )
    assert np.array_equal(candidate, greenfield)
    mismatches = np.flatnonzero(candidate != accepted)
    assert mismatches.tolist() == [35]
    assert artifact["classification"][
        "candidate_vs_accepted_current_key_bfloat16_values"
    ] == [int(candidate[35]), int(accepted[35])]


@pytest.mark.skipif(
    not BASELINE.is_file() or not ORACLE.is_file(),
    reason="sealed PP16 rejection or accepted p113 oracle is unavailable",
)
def test_position113_classifier_rejects_observer_effect(tmp_path: Path) -> None:
    capture = tmp_path / "capture.npz"
    _write_candidate(capture)
    with np.load(capture, allow_pickle=False) as handle:
        arrays = {name: np.ascontiguousarray(handle[name]) for name in handle.files}
    arrays["event1_scores"][0, 0] = np.nextafter(
        arrays["event1_scores"][0, 0], np.float32(np.inf)
    )
    np.savez(capture, **arrays)
    with pytest.raises(BenchmarkValidationError, match="changed sealed rejection"):
        compare_feature2_position113_capture(
            capture,
            sealed_rejection_path=BASELINE,
            accepted_prompt_key_path=ORACLE,
        )


@pytest.mark.skipif(
    not BASELINE.is_file() or not ORACLE.is_file(),
    reason="sealed PP16 rejection or accepted p113 oracle is unavailable",
)
def test_position113_classifier_rejects_signed_zero_owner_drift(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "capture.npz"
    _write_candidate(capture)
    with np.load(capture, allow_pickle=False) as handle:
        arrays = {name: np.ascontiguousarray(handle[name]) for name in handle.files}
    arrays["position113_dsa_query_owners"][1, 0, 0, 0] = np.float32(-0.0)
    np.savez(capture, **arrays)
    with pytest.raises(BenchmarkValidationError, match="owners disagree"):
        compare_feature2_position113_capture(
            capture,
            sealed_rejection_path=BASELINE,
            accepted_prompt_key_path=ORACLE,
        )


@pytest.mark.skipif(
    not BASELINE.is_file() or not ORACLE.is_file(),
    reason="sealed PP16 rejection or accepted p113 oracle is unavailable",
)
@pytest.mark.parametrize("source", ("baseline", "oracle"))
def test_position113_classifier_rejects_same_schema_source_redefinition(
    tmp_path: Path,
    source: str,
) -> None:
    capture = tmp_path / "capture.npz"
    _write_candidate(capture)
    original = BASELINE if source == "baseline" else ORACLE
    rewritten = tmp_path / f"rewritten_{source}.npz"
    with np.load(original, allow_pickle=False) as handle:
        np.savez(
            rewritten,
            **{
                name: np.ascontiguousarray(handle[name])
                for name in reversed(handle.files)
            },
        )
    assert rewritten.read_bytes() != original.read_bytes()
    with pytest.raises(BenchmarkValidationError, match="identity drifted"):
        compare_feature2_position113_capture(
            capture,
            sealed_rejection_path=(rewritten if source == "baseline" else BASELINE),
            accepted_prompt_key_path=(rewritten if source == "oracle" else ORACLE),
        )


@pytest.mark.skipif(
    not BASELINE.is_file() or not ORACLE.is_file(),
    reason="sealed PP16 rejection or accepted p113 oracle is unavailable",
)
@pytest.mark.parametrize("source", ("baseline", "oracle"))
def test_position113_classifier_uses_the_exact_bytes_it_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    capture = tmp_path / "capture.npz"
    _write_candidate(capture)
    raced_baseline = tmp_path / "baseline.npz"
    raced_oracle = tmp_path / "oracle.npz"
    raced_baseline.write_bytes(BASELINE.read_bytes())
    raced_oracle.write_bytes(ORACLE.read_bytes())
    raced_source = raced_baseline if source == "baseline" else raced_oracle
    attacked = tmp_path / f"attacked_{source}.npz"
    with np.load(raced_source, allow_pickle=False) as handle:
        arrays = {name: np.ascontiguousarray(handle[name]) for name in handle.files}
    if source == "baseline":
        arrays["event1_scores"][0, 0] = np.nextafter(
            arrays["event1_scores"][0, 0], np.float32(np.inf)
        )
    else:
        arrays["accepted_projection_input"][0] = np.float32(1.0)
    np.savez(attacked, **arrays)
    attacked_bytes = attacked.read_bytes()
    original_read_bytes = Path.read_bytes
    replaced = False

    def replace_after_read(path: Path) -> bytes:
        nonlocal replaced
        raw = original_read_bytes(path)
        if path == raced_source and not replaced:
            raced_source.write_bytes(attacked_bytes)
            replaced = True
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    report = compare_feature2_position113_capture(
        capture,
        sealed_rejection_path=raced_baseline,
        accepted_prompt_key_path=raced_oracle,
    )
    assert replaced is True
    assert original_read_bytes(raced_source) == attacked_bytes
    assert report["classification"] == (
        "FIRST_DIVERGENCE_AFTER_NORMALIZED_AT_OR_BEFORE_CURRENT_KEY"
    )


@pytest.mark.skipif(
    not BASELINE.is_file() or not ORACLE.is_file(),
    reason="sealed PP16 rejection or accepted p113 oracle is unavailable",
)
@pytest.mark.parametrize("attack", ("wrong_set", "duplicate", "tail", "score_tail"))
def test_position113_classifier_rejects_invalid_selected_prefix(
    tmp_path: Path,
    attack: str,
) -> None:
    capture = tmp_path / "capture.npz"
    _write_candidate(capture)
    with np.load(capture, allow_pickle=False) as handle:
        arrays = {name: np.ascontiguousarray(handle[name]) for name in handle.files}
    if attack == "wrong_set":
        arrays["position113_selected_positions_owners"][:, 0, :114] = np.arange(
            1, 115, dtype=np.int32
        )
    elif attack == "duplicate":
        arrays["position113_selected_positions_owners"][:, 0, 1] = 0
    elif attack == "tail":
        arrays["position113_selected_positions_owners"][:, 0, 114] = 5
    else:
        arrays["position113_selected_scores_owners"][:, 0, 114] = np.float32(0.0)
    np.savez(capture, **arrays)
    with pytest.raises(BenchmarkValidationError, match="selected"):
        compare_feature2_position113_capture(
            capture,
            sealed_rejection_path=BASELINE,
            accepted_prompt_key_path=ORACLE,
        )
