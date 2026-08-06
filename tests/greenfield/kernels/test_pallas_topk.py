from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.pallas import (
    DsaTopKConfig,
    local_topk_candidates_kernel,
    local_topk_candidates_pallas,
    merge_topk_candidates_kernel,
    merge_topk_candidates_pallas,
)
from glm_tpu.greenfield.kernels.reference.dsa import (
    local_topk_candidates,
    merge_topk_candidates,
)


@pytest.mark.parametrize("context", [17, 130, 257])
def test_local_topk_pallas_interpret_matches_exact_reference(context: int) -> None:
    source_positions = np.arange(context, dtype=np.int32)
    positions = source_positions[(source_positions * 73 + 19) % context]
    scores = np.cos(source_positions.astype(np.float32) * np.float32(0.31))
    scores[source_positions % 11 == 0] = np.float32(5.0)
    local_scores = jnp.asarray(scores[None, :], dtype=jnp.float32)
    global_positions = jnp.asarray(positions, dtype=jnp.int32)
    valid_lengths = jnp.asarray([max(context - 3, 0)], dtype=jnp.int32)
    config = DsaTopKConfig(selection_width=4, local_block_size=128)

    actual_scores, actual_positions = local_topk_candidates_pallas(
        local_scores,
        global_positions,
        valid_lengths,
        config=config,
        interpret=True,
    )
    expected_scores, expected_positions = local_topk_candidates(
        local_scores, global_positions, valid_lengths, top_k=4
    )

    np.testing.assert_array_equal(
        np.asarray(actual_scores), np.asarray(expected_scores)
    )
    np.testing.assert_array_equal(
        np.asarray(actual_positions), np.asarray(expected_positions)
    )


def test_local_topk_pallas_preserves_ties_and_exact_minus_one_tail() -> None:
    scores = jnp.asarray([[7.0, 7.0, 6.0]], dtype=jnp.float32)
    positions = jnp.asarray([2, 0, 1], dtype=jnp.int32)
    valid_lengths = jnp.asarray([2], dtype=jnp.int32)
    config = DsaTopKConfig(selection_width=5, local_block_size=128)

    actual_scores, actual_positions = local_topk_candidates_pallas(
        scores,
        positions,
        valid_lengths,
        config=config,
        interpret=True,
    )
    expected_scores, expected_positions = local_topk_candidates(
        scores, positions, valid_lengths, top_k=5
    )

    np.testing.assert_array_equal(
        np.asarray(actual_scores), np.asarray(expected_scores)
    )
    np.testing.assert_array_equal(
        np.asarray(actual_positions), np.asarray(expected_positions)
    )
    np.testing.assert_array_equal(np.asarray(actual_positions), [[0, 1, -1, -1, -1]])


def test_candidate_merge_pallas_is_exact_for_odd_group_count_and_duplicates() -> None:
    scores = jnp.asarray(
        [
            [[9.0, 7.0, 5.0, 3.0]],
            [[9.0, 8.0, 5.0, 2.0]],
            [[9.0, 6.0, 5.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    positions = jnp.asarray(
        [
            [[4, 3, 5, 8]],
            [[1, 2, 5, 9]],
            [[0, 6, 5, 7]],
        ],
        dtype=jnp.int32,
    )
    valid_lengths = jnp.asarray([10], dtype=jnp.int32)
    config = DsaTopKConfig(selection_width=5, local_block_size=128)

    actual = merge_topk_candidates_pallas(
        scores,
        positions,
        valid_lengths,
        global_context_size=10,
        config=config,
        interpret=True,
    )
    expected = merge_topk_candidates(
        scores,
        positions,
        valid_lengths,
        top_k=5,
        global_context_size=10,
    )
    reversed_actual = merge_topk_candidates_pallas(
        scores[::-1],
        positions[::-1],
        valid_lengths,
        global_context_size=10,
        config=config,
        interpret=True,
    )

    np.testing.assert_array_equal(
        np.asarray(actual.positions), np.asarray(expected.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(actual.valid_counts), np.asarray(expected.valid_counts)
    )
    np.testing.assert_array_equal(
        np.asarray(reversed_actual.positions), np.asarray(expected.positions)
    )


def test_topk_dispatch_is_default_off() -> None:
    scores = jnp.asarray([[2.0, 3.0, 1.0]], dtype=jnp.float32)
    positions = jnp.asarray([2, 0, 1], dtype=jnp.int32)
    valid_lengths = jnp.asarray([3], dtype=jnp.int32)
    expected_scores, expected_positions = local_topk_candidates(
        scores, positions, valid_lengths, top_k=2
    )

    actual_scores, actual_positions = local_topk_candidates_kernel(
        scores, positions, valid_lengths, top_k=2
    )
    np.testing.assert_array_equal(np.asarray(actual_scores), np.asarray(expected_scores))
    np.testing.assert_array_equal(
        np.asarray(actual_positions), np.asarray(expected_positions)
    )
    with pytest.raises(ValueError, match="unsupported DSA top-k backend"):
        local_topk_candidates_kernel(
            scores,
            positions,
            valid_lengths,
            top_k=2,
            backend="unknown",  # type: ignore[arg-type]
        )

    candidate_scores = expected_scores[None, ...]
    candidate_positions = expected_positions[None, ...]
    expected_merge = merge_topk_candidates(
        candidate_scores,
        candidate_positions,
        valid_lengths,
        top_k=2,
        global_context_size=3,
    )
    actual_merge = merge_topk_candidates_kernel(
        candidate_scores,
        candidate_positions,
        valid_lengths,
        top_k=2,
        global_context_size=3,
    )
    np.testing.assert_array_equal(
        np.asarray(actual_merge.positions), np.asarray(expected_merge.positions)
    )


def test_topk_pallas_rejects_shape_dtype_and_config_drift() -> None:
    config = DsaTopKConfig(selection_width=2, local_block_size=128)
    scores = jnp.ones((1, 3), dtype=jnp.float32)
    positions = jnp.arange(3, dtype=jnp.int32)
    valid_lengths = jnp.asarray([3], dtype=jnp.int32)

    with pytest.raises(ValueError, match="one exact score row"):
        local_topk_candidates_pallas(
            jnp.ones((2, 3), dtype=jnp.float32),
            positions,
            valid_lengths,
            config=config,
            interpret=True,
        )
    with pytest.raises(ValueError, match="scores must remain FP32"):
        local_topk_candidates_pallas(
            scores.astype(jnp.bfloat16),
            positions,
            valid_lengths,
            config=config,
            interpret=True,
        )
    with pytest.raises(ValueError, match="must be int32"):
        local_topk_candidates_pallas(
            scores,
            positions.astype(jnp.int16),
            valid_lengths,
            config=config,
            interpret=True,
        )
    with pytest.raises(ValueError, match="match the Pallas selection width"):
        local_topk_candidates_kernel(
            scores,
            positions,
            valid_lengths,
            top_k=3,
            backend="pallas",
            config=config,
            interpret=True,
        )
    with pytest.raises(ValueError, match="multiple of 128"):
        DsaTopKConfig(local_block_size=64)
    with pytest.raises(ValueError, match="tiles must contain 128"):
        DsaTopKConfig(tile_size=64)
