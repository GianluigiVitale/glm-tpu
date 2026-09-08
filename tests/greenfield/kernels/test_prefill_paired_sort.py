"""DB595 permutation-gather replacement: exact CPU semantics, not TPU speed."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference.dsa import merge_topk_candidates_with_scores
from glm_tpu.greenfield.kernels.prefill_dsa import causal_dsa_local_candidates


def assert_same_bits(actual, expected):
    for got, want in zip(actual, expected):
        got, want = np.asarray(got), np.asarray(want)
        assert got.shape == want.shape and got.dtype == want.dtype
        assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize(
    "groups,rows,candidates,top_k",
    [(2, 5, 64, 64), (8, 32, 2048, 2048), (8, 1, 64, 64)],
)
def test_paired_sort_preserves_all_returned_bits(groups, rows, candidates, top_k):
    rng = np.random.default_rng(595)
    shape = (groups, rows, candidates)
    # Many score ties and duplicate positions, unequal scores at duplicate keys,
    # shuffled group order, repeated padding, signed zeros and infinities.
    positions = rng.integers(-1, groups * candidates // 2, shape, dtype=np.int32)
    scores = rng.integers(-3, 4, shape).astype(np.float32)
    scores.flat[::7] = np.float32(-0.0)
    scores.flat[::13] = np.float32(0.0)
    positions.flat[::19] = -1
    scores[positions == -1] = -np.inf
    scores.flat[::23] = np.inf
    order = rng.permutation(groups)
    lengths = np.resize(
        np.asarray([0, 1, top_k - 1, top_k, groups * candidates], np.int32), rows
    )
    args = (
        jnp.asarray(scores[order]),
        jnp.asarray(positions[order]),
        jnp.asarray(lengths),
    )
    fn = partial(
        merge_topk_candidates_with_scores,
        top_k=top_k,
        global_context_size=groups * candidates,
    )
    expected = jax.jit(fn)(*args)
    actual = jax.jit(partial(fn, paired_position_sort=True))(*args)
    assert_same_bits(actual, expected)
    # Default is still the old reference realization.
    assert_same_bits(jax.jit(partial(fn, paired_position_sort=False))(*args), expected)


def test_paired_sort_causal_tile_merge_matches_old_bits():
    rng = np.random.default_rng(596)
    q = jnp.asarray(rng.normal(size=(5, 32, 128)), jnp.float32)
    keys = jnp.asarray(rng.normal(size=(333, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(5, 32)), jnp.float32).at[4].set(0)
    positions = (jnp.arange(333, dtype=jnp.int32) * 8).at[20].set(-1)
    lengths = jnp.asarray([0, 1, 100, 1400, 2664], jnp.int32)
    fn = partial(
        causal_dsa_local_candidates, global_context_size=2664, top_k=64, key_tile=128
    )
    args = q, keys, weights, positions, lengths
    assert_same_bits(
        jax.jit(partial(fn, paired_position_sort=True))(*args), jax.jit(fn)(*args)
    )


def test_production_union_lowering_removes_two_permutation_gathers():
    args = (
        jax.ShapeDtypeStruct((8, 32, 2048), jnp.float32),
        jax.ShapeDtypeStruct((8, 32, 2048), jnp.int32),
        jax.ShapeDtypeStruct((32,), jnp.int32),
    )
    fn = partial(
        merge_topk_candidates_with_scores, top_k=2048, global_context_size=4096
    )
    old = str(jax.jit(fn).lower(*args).compiler_ir("stablehlo"))
    new = str(
        jax.jit(partial(fn, paired_position_sort=True))
        .lower(*args)
        .compiler_ir("stablehlo")
    )
    # One gather remains: final selected positions. No claim about TPU lowering.
    assert old.count('"stablehlo.gather"') == 3
    assert new.count('"stablehlo.gather"') == 1
    assert "is_stable = true" in new


def test_flag_is_static_bool_not_truthy_integer():
    with pytest.raises(ValueError, match="static bool"):
        merge_topk_candidates_with_scores(
            jnp.zeros((2, 1, 2)),
            jnp.zeros((2, 1, 2), jnp.int32),
            jnp.ones((1,), jnp.int32),
            top_k=1,
            global_context_size=4,
            paired_position_sort=1,
        )
