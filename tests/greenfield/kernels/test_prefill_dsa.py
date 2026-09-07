"""CPU-only causal/tie/coverage admission; no TPU or speed claim."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.prefill_dsa import causal_dsa_local_candidates
from glm_tpu.greenfield.kernels.reference.dsa import dsa_scores, local_topk_candidates


def inputs(context=333):
    rng = np.random.default_rng(287)
    query = jnp.asarray(rng.normal(0, 0.2, (5, 32, 128)), jnp.float32)
    keys = jnp.asarray(rng.normal(0, 0.3, (context, 128)), jnp.bfloat16)
    # Signed weights matter: scores are not all nonnegative.
    weights = jnp.asarray(rng.normal(0, 0.2, (5, 32)), jnp.float32)
    positions = jnp.arange(context, dtype=jnp.int32) * 8 + 3
    lengths = jnp.asarray([0, 4, 1017, 2048, context * 8], jnp.int32)
    return query, keys, weights, positions, lengths


@pytest.mark.parametrize("tile", [128, 256])
def test_causal_local_selection_matches_own_scores_with_tails_and_holes(tile):
    query, keys, weights, positions, lengths = inputs()
    positions = positions.at[20].set(-1).at[155].set(-1)
    call = jax.jit(
        lambda *v: causal_dsa_local_candidates(
            *v, global_context_size=2664, top_k=64, key_tile=tile
        )
    )
    actual = call(query, keys, weights, positions, lengths)
    assert bool(actual.valid)
    # Independently materialize this test's small executing tiled score rows.
    # This is an oracle/test only, never the production streaming implementation.
    scores = jnp.concatenate(
        [dsa_scores(query, keys[i : i + tile], weights) for i in range(0, 333, tile)],
        axis=1,
    )
    expected_scores, expected_positions = local_topk_candidates(
        scores, positions, lengths, top_k=64
    )
    np.testing.assert_array_equal(actual.positions, expected_positions)
    np.testing.assert_allclose(actual.scores, expected_scores, rtol=1e-6, atol=1e-6)
    for row, length in zip(np.asarray(actual.positions), np.asarray(lengths)):
        assert np.all((row == -1) | (row < length))
        assert len(np.unique(row[row >= 0])) == (row >= 0).sum()


def test_lowest_global_position_ties_cross_tiles_and_unused_future_keys():
    query, keys, weights, positions, lengths = inputs()
    weights = jnp.zeros_like(weights)
    # Future-only second/third tiles are poisoned; they must not be scored.
    keys = keys.at[128:].set(jnp.nan)
    lengths = jnp.asarray([0, 4, 12, 512, 1024], jnp.int32)
    actual = jax.jit(
        lambda *v: causal_dsa_local_candidates(
            *v, global_context_size=2664, top_k=64, key_tile=128
        )
    )(query, keys, weights, positions, lengths)
    assert bool(actual.valid)
    _, expected = local_topk_candidates(
        jnp.zeros((5, 333), jnp.float32), positions, lengths, top_k=64
    )
    np.testing.assert_array_equal(actual.positions, expected)
    # Now expose a second tile without poison: ties must not prefer latest tile.
    keys = jnp.zeros_like(keys)
    actual = causal_dsa_local_candidates(
        query,
        keys,
        weights,
        positions,
        jnp.full((5,), 2664, jnp.int32),
        global_context_size=2664,
        top_k=200,
        key_tile=128,
    )
    np.testing.assert_array_equal(
        actual.positions, np.tile(np.arange(200) * 8 + 3, (5, 1))
    )


def test_invalid_metadata_or_visible_nonfinite_cannot_report_healthy():
    q, keys, hw, pos, lens = inputs()
    call = jax.jit(
        lambda *v: causal_dsa_local_candidates(
            *v, global_context_size=2664, top_k=64, key_tile=128
        )
    )
    for positions, lengths in (
        (pos.at[2].set(pos[1]), lens),
        (pos.at[0].set(-2), lens),
        (pos, lens.at[0].set(-1)),
        (pos, lens.at[0].set(2665)),
    ):
        result = call(q, keys, hw, positions, lengths)
        assert not bool(result.valid)
        np.testing.assert_array_equal(result.positions, -1)
    result = call(q, keys.at[0].set(jnp.nan), hw, pos, lens)
    assert not bool(result.valid)


def test_static_budget_and_dtype_contracts():
    args = inputs()
    for kwargs in (
        {"key_tile": 8192},
        {"key_tile": 129},
        {"top_k": 4096},
        {"global_context_size": -1},
    ):
        with pytest.raises(ValueError):
            causal_dsa_local_candidates(
                *args, **({"global_context_size": 2664} | kwargs)
            )
    with pytest.raises(ValueError, match="F32"):
        causal_dsa_local_candidates(
            args[0].astype(jnp.bfloat16), *args[1:], global_context_size=2664
        )
