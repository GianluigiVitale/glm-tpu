"""Sorted-candidate CPU exactness/structure only; not TPU performance proof."""

from functools import partial
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.prefill_dsa import causal_dsa_local_candidates
from glm_tpu.greenfield.kernels.prefill_sorted_merge import merge_sorted_candidate_pair
from glm_tpu.optimized.reference.dsa import merge_topk_candidates_with_scores


def same_bits(got, expected):
    for a, b in zip(got, expected, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()


def ordered(scores, positions, width):
    """Independent finite IEEE total-order/position oracle (no JAX selector)."""
    raw = np.asarray(scores, np.float32).view(np.int32)
    keys = np.where(raw < 0, raw ^ np.int32(0x7FFFFFFF), raw)
    order = np.lexsort((positions, -keys.astype(np.int64)))[:width]
    return np.asarray(scores)[order], np.asarray(positions)[order]


@pytest.mark.parametrize("width", [1, 2, 8, 64, 2048])
def test_sorted_pair_exact_bits_and_total_order(width):
    rng = np.random.default_rng(598)
    rows = 32 if width == 2048 else 9
    lhs, rhs, lp, rp, want_s, want_p = [], [], [], [], [], []
    for row in range(rows):
        # Disjoint arbitrarily interleaved owners, not concatenation tie order.
        pos = rng.permutation(2 * width).astype(np.int32)
        score = rng.integers(-4, 5, 2 * width).astype(np.float32)
        score[::5] = np.float32(-0.0)
        score[::7] = np.float32(+0.0)
        if row % 7 == 0:
            score[:] = 1  # all tied, including across owner/tile boundary
        elif row % 7 == 1:
            score = rng.normal(size=2 * width).astype(np.float32)
        elif row % 7 == 2:
            score[:] = -np.inf
            pos[:] = -1
        elif row % 7 in (3, 4):
            score[width // 2 :] = -np.inf
            pos[width // 2 :] = -1
        elif row % 7 == 5:
            # Includes subnormals, signed zeros, adjacent and extreme finite bits.
            score[:] = np.resize(
                np.asarray(
                    [
                        0,
                        0x80000000,
                        1,
                        0x80000001,
                        0x3F800000,
                        0x3F800001,
                        0x7F7FFFFF,
                        0xFF7FFFFF,
                    ],
                    np.uint32,
                ).view(np.float32),
                2 * width,
            )
        a, b = ordered(score[:width], pos[:width], width), ordered(
            score[width:], pos[width:], width
        )
        lhs.append(a[0])
        lp.append(a[1])
        rhs.append(b[0])
        rp.append(b[1])
        expected = ordered(score, pos, width)
        want_s.append(expected[0])
        want_p.append(expected[1])
    args = tuple(
        jnp.asarray(x)
        for x in (np.array(lhs), np.array(lp), np.array(rhs), np.array(rp))
    )
    actual = jax.jit(merge_sorted_candidate_pair)(*args)
    same_bits(actual, (np.array(want_s), np.array(want_p)))
    # The preserved executing reference must agree as well, not just our oracle.
    reference = jax.jit(
        partial(
            merge_topk_candidates_with_scores,
            top_k=width,
            global_context_size=2 * width,
            paired_position_sort=True,
        )
    )(
        jnp.stack((args[0], args[2])),
        jnp.stack((args[1], args[3])),
        jnp.full((rows,), 2 * width, jnp.int32),
    )
    same_bits(actual, (reference.scores, reference.positions))


def test_all_small_tied_sequences():
    # Exhaust all four-element score assignments, including empty entries.
    left, right, lp, rp, expected = [], [], [], [], []
    for values in product((-np.inf, -1.0, 0.0, 1.0), repeat=4):
        scores = np.asarray(values, np.float32)
        pos = np.asarray([3, 0, 2, 1], np.int32)
        pos[np.isneginf(scores)] = -1
        a, b = ordered(scores[:2], pos[:2], 2), ordered(scores[2:], pos[2:], 2)
        left.append(a[0])
        lp.append(a[1])
        right.append(b[0])
        rp.append(b[1])
        expected.append(ordered(scores, pos, 2))
    call = jax.jit(merge_sorted_candidate_pair)
    for start in range(0, len(expected), 32):
        stop = start + 32
        actual = call(*(jnp.asarray(x[start:stop]) for x in (left, lp, right, rp)))
        same_bits(
            actual,
            tuple(np.stack([v[i] for v in expected[start:stop]]) for i in (0, 1)),
        )


@pytest.mark.parametrize(
    "width,tile,precision", [(64, 128, "highest"), (2048, 512, "default")]
)
def test_actual_causal_tile_loop_matches_preserved_baseline(width, tile, precision):
    rng = np.random.default_rng(5981)
    n = tile * 3 + 17
    q = jnp.asarray(rng.normal(size=(5, 32, 128)), jnp.float32)
    keys = jnp.asarray(rng.normal(size=(n, 128)), jnp.bfloat16)
    hw = jnp.asarray(rng.normal(size=(5, 32)), jnp.float32).at[-1].set(0)
    positions = (
        (jnp.arange(n, dtype=jnp.int32) * 8 + 2).at[7].set(-1).at[tile + 5].set(-1)
    )
    lengths = jnp.asarray([0, 3, tile * 8 - 3, n * 8, n * 8], jnp.int32)
    baseline = partial(
        causal_dsa_local_candidates,
        global_context_size=n * 8,
        top_k=width,
        key_tile=tile,
        precision=precision,
        paired_position_sort=True,
    )
    args = q, keys, hw, positions, lengths
    control = jax.jit(baseline)
    candidate = jax.jit(partial(baseline, sorted_local_merge=True))
    same_bits(candidate(*args), control(*args))
    assert bool(candidate(*args).valid)
    # Entire future tiles may contain NaNs without being executed.
    future = (
        q,
        keys.at[tile:].set(jnp.nan),
        hw,
        positions,
        jnp.full((5,), tile * 8 - 3, jnp.int32),
    )
    same_bits(candidate(*future), control(*future))
    assert bool(candidate(*future).valid)
    for bad in (
        (q, keys.at[0].set(jnp.nan), hw, positions, lengths),
        (q, keys, hw, positions.at[2].set(positions[1]), lengths),
        (q, keys, hw, positions, lengths.at[0].set(-1)),
    ):
        assert not bool(candidate(*bad).valid)


def test_production_pair_is_vector_network_without_sort_gather_or_scalar_loop():
    args = (
        jax.ShapeDtypeStruct((32, 2048), jnp.float32),
        jax.ShapeDtypeStruct((32, 2048), jnp.int32),
    ) * 2
    hlo = str(
        jax.jit(merge_sorted_candidate_pair).lower(*args).compiler_ir("stablehlo")
    )
    for forbidden in (
        "stablehlo.sort",
        "stablehlo.gather",
        "stablehlo.while",
        "chlo.top_k",
    ):
        assert forbidden not in hlo
    assert "stablehlo.reverse" in hlo


def test_specialized_contract_refuses_unsupported_shapes_and_flags():
    for shape in ((1, 3), (33, 8), (1, 4096), (0, 8)):
        s, p = jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.int32)
        with pytest.raises(ValueError, match="sorted merge"):
            merge_sorted_candidate_pair(s, p, s, p)
    s, p = jnp.zeros((1, 8), jnp.float32), jnp.zeros((1, 8), jnp.int32)
    with pytest.raises(ValueError, match="sorted merge"):
        merge_sorted_candidate_pair(s.astype(jnp.bfloat16), p, s, p)
    q, k, h = (
        jnp.zeros((1, 32, 128)),
        jnp.zeros((128, 128), jnp.bfloat16),
        jnp.zeros((1, 32)),
    )
    for kwargs, match in (
        ({"sorted_local_merge": 1}, "static bool"),
        ({"sorted_local_merge": True, "top_k": 3}, "power-of-two"),
    ):
        with pytest.raises(ValueError, match=match):
            causal_dsa_local_candidates(
                q,
                k,
                h,
                jnp.arange(128, dtype=jnp.int32),
                jnp.array([128], jnp.int32),
                global_context_size=128,
                **kwargs
            )
