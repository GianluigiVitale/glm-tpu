"""Unit coverage for the CPU-only event-1 prompt-cache localization helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/greenfield/diagnose_event1_prompt_index_cache_offline.py"
)


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("diagnose_event1", SCRIPT)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded


def test_prompt_cache_rows_follow_db518_owner_page_row_layout(module):
    bits = np.zeros((2, 16, 256, 128), dtype=np.uint16)
    for position in (0, 113, 255, 256, 511, 512, 8154):
        local = position % 512
        bits[local // 256, position // 512, local % 256, :] = position
    rows = module.prompt_cache_rows(bits, rows=8155)
    assert rows.shape == (8155, 128)
    for position in (0, 113, 255, 256, 511, 512, 8154):
        assert int(rows[position, 0]) == position
    assert int(rows[1, 0]) == 0


def test_prompt_cache_rows_reject_geometry_drift(module):
    with pytest.raises(ValueError):
        module.prompt_cache_rows(np.zeros((2, 16, 128, 128), dtype=np.uint16), rows=10)
    with pytest.raises(ValueError):
        module.prompt_cache_rows(np.zeros((2, 1, 256, 128), dtype=np.uint16), rows=600)
    with pytest.raises(ValueError):
        module.prompt_cache_rows(np.zeros((2, 16, 256, 128), dtype=np.float32), rows=10)


def test_default_precision_scores_apply_bf16_pass_relu_and_signed_weights(module):
    rng = np.random.default_rng(7)
    query = rng.normal(size=(2, 128)).astype(np.float32)
    keys = rng.normal(size=(5, 128)).astype(np.float32)
    weights = np.array([1.5, -0.5], dtype=np.float32)
    scores = module.default_precision_dsa_scores(query, keys, weights)
    q16 = query.astype(ml_dtypes.bfloat16).astype(np.float64)
    k16 = keys.astype(ml_dtypes.bfloat16).astype(np.float64)
    per_head = (q16 @ k16.T).astype(np.float32).astype(np.float64) * np.float64(
        np.float32(128**-0.5)
    )
    expected = (weights.astype(np.float64)[:, None] * np.maximum(per_head, 0.0)).sum(0)
    np.testing.assert_allclose(scores, expected, rtol=0, atol=0)
    # A query change below BF16 resolution must not change the emulated score.
    nudged = query.copy()
    nudged[0, 0] = np.nextafter(nudged[0, 0], np.float32(np.inf))
    assert np.array_equal(module.default_precision_dsa_scores(nudged, keys, weights), scores)


def test_default_precision_scores_reject_shape_drift(module):
    with pytest.raises(ValueError):
        module.default_precision_dsa_scores(
            np.zeros((2, 64), np.float32), np.zeros((3, 128), np.float32), np.zeros(2, np.float32)
        )
    with pytest.raises(ValueError):
        module.default_precision_dsa_scores(
            np.zeros((2, 128), np.float32), np.zeros((3, 128), np.float32), np.zeros(3, np.float32)
        )


def test_observer_event_decodes_positions_scores_and_producer(module):
    width = module.SELECTED_WIDTH
    observation = np.full((4, 3, 2 * width + 2), -1, dtype=np.int32)
    row = observation[0][1]
    row[:3] = [5, 9, 2]
    row[width : width + 3] = np.array([1.5, -2.0, 0.25], np.float32).view(np.int32)
    row[2 * width] = 3
    row[2 * width + 1] = 1
    decoded = module.observer_event(observation, 1)
    assert decoded == {5: 1.5, 9: -2.0, 2: 0.25}
    row[2 * width + 1] = 0
    with pytest.raises(ValueError):
        module.observer_event(observation, 1)


def test_compare_sets_reports_swaps_and_score_deltas(module):
    reference = {0: 1.0, 1: 0.5, 2: 0.25}
    candidate_scores = np.array([1.0, 0.5, 0.0, 0.75], dtype=np.float64)
    result = module.compare_sets(reference, candidate_scores)
    assert result["expected_only"] == [2]
    assert result["observed_only"] == [3]
    assert result["score_delta_nonzero"] == 1
    assert result["score_delta_max_abs"] == pytest.approx(0.25)
