"""Small CPU diagnostics tests; no model correctness/performance admission."""

import numpy as np
import pytest

from scripts.greenfield.analyze_prefill_window_capture import (
    align_scores,
    delta,
    dsa_fp64,
    ordered_ids,
    row_changes,
)


def test_canonical_selection_checks_all_experts_and_lower_index_ties():
    scores = np.zeros((2, 256), np.float32)
    scores[1, 255] = 1
    np.testing.assert_array_equal(ordered_ids(scores)[0], np.arange(8))
    np.testing.assert_array_equal(ordered_ids(scores)[1], [255, 0, 1, 2, 3, 4, 5, 6])


@pytest.mark.parametrize(
    "bad", [np.array([[np.nan]]), np.array([[np.inf]]), np.zeros(4)]
)
def test_nonfinite_or_wrong_rank_selection_refuses(bad):
    with pytest.raises(ValueError):
        ordered_ids(bad)


def test_position_alignment_and_padding():
    np.testing.assert_array_equal(
        align_scores(np.array([2, 0, 1, -1]), np.array([4.0, 8.0, 6.0, -np.inf]), 3),
        [8, 6, 4],
    )


@pytest.mark.parametrize("positions", [[0, 0, 2], [0, 2, -1], [0, 1, 3]])
def test_missing_duplicate_or_noncausal_position_refuses(positions):
    with pytest.raises(ValueError):
        align_scores(np.array(positions), np.ones(3), 3)


def test_signed_dsa_relu_scaling_and_query_precision():
    query = np.array([[2.0, -1.0], [-1.0, 3.0]], np.float32)
    keys = np.array([[1.0, 0.0], [0.0, 1.0], [-2.0, -1.0]], np.float32)
    weights = np.array([0.5, -0.25], np.float32)
    expected = np.array([1.0, -0.75, 0.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(dsa_fp64(query, keys, weights), expected, rtol=1e-15)
    query[0, 0] = np.float32(2.0001)
    assert dsa_fp64(query, keys, weights)[0] != expected[0]


def test_delta_has_no_pass_threshold_and_rows_not_elements():
    a, b = np.zeros((4, 2)), np.zeros((4, 2))
    a[2] = [2, -2]
    r = delta(a, b)
    assert r["max_abs"] == 2 and r["mean"] == 0 and r["changed_elements"] == 2
    assert "passed" not in r
    assert row_changes(a, b) == [2]
