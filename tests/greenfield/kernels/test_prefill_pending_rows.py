"""Pending rows preserve existing writer addresses; CPU, not TPU fit evidence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.optimized.prefill_cache import write_prefill_cache_block
from glm_tpu.greenfield.kernels.prefill_pending_rows import (
    apply_prefill_pending_rows,
    capture_prefill_pending_rows,
    prefill_pending_addresses,
)
from glm_tpu.optimized.reference.attention import StageLocalKvLayout


LAYOUT = StageLocalKvLayout(local_parallel_size=8, packed_cache_width=4)


def addresses(table, offset, count, owner, rows=128):
    return prefill_pending_addresses(
        jnp.asarray(table, jnp.int32),
        jnp.int32(offset),
        jnp.int32(count),
        jnp.int32(owner),
        physical_pages=3,
        window_rows=rows,
        layout=LAYOUT,
    )


def test_all_stripe_offsets_and_page_permutations_match_independent_addresses():
    table = np.array([[2, 0, 1]], np.int32)
    offsets = np.arange(512, dtype=np.int32)
    counts = np.array([0, 1, 3, 32, 64, 114, 128], np.int32)
    cases = np.array(
        [(o, c, e) for o in offsets for c in counts for e in range(8)], np.int32
    )
    run = jax.jit(jax.vmap(lambda case: addresses(table, *case)))
    out = run(jnp.asarray(cases))
    pos = cases[:, 0, None] + np.arange(128)[None, :]
    mask = (np.arange(128)[None, :] < cases[:, 1, None]) & (
        (pos % 512) // 64 == cases[:, 2, None]
    )
    expected = np.where(mask, table[0][pos // 512] * 64 + pos % 64, 192)
    np.testing.assert_array_equal(out.targets, expected)
    assert np.asarray(out.valid).all()


@pytest.mark.parametrize(
    "offset,count", [(0, 128), (57, 128), (505, 128), (633, 3), (1020, 114), (1533, 3)]
)
def test_full_stack_reconstruction_equals_original_writer(offset, count):
    table = jnp.array([[2, 0, 1]], jnp.int32)
    initial = (
        jnp.arange(3 * 3 * 64 * 4, dtype=jnp.float32)
        .reshape(3, 3, 64, 4)
        .astype(jnp.bfloat16)
    )
    values = (jnp.arange(128 * 4).reshape(128, 4) % 31).astype(jnp.bfloat16)
    values = jnp.where(jnp.arange(128)[:, None] < count, values, jnp.nan)

    @jax.jit
    def replay(owner):
        proposals = []
        for layer in range(3):
            cache = initial[layer]
            for start in range(0, 128, 32):
                n = min(max(count - start, 0), 32)
                # Same empty-tile clamping as the current layer window.
                at = min(offset + start, 1535)
                cache = write_prefill_cache_block(
                    cache,
                    values[start : start + 32],
                    table,
                    jnp.int32(at),
                    jnp.int32(n),
                    owner,
                    layout=LAYOUT,
                ).cache
            proposals.append(cache)
        plan = addresses(table, offset, count, owner)
        pending = jnp.stack(
            [capture_prefill_pending_rows(p, plan.targets) for p in proposals]
        )
        return (
            jnp.stack(proposals),
            apply_prefill_pending_rows(initial, plan.targets, pending),
            pending,
        )

    expected, actual, pending = jax.vmap(replay)(jnp.arange(8, dtype=jnp.int32))
    assert np.asarray(expected).tobytes() == np.asarray(actual).tobytes()
    assert np.isfinite(np.asarray(pending).astype(np.float32)).all()
    assert pending.shape == (8, 3, 128, 4)


@pytest.mark.parametrize(
    "offset,count,owner,table",
    [
        (-1, 1, 0, [[0, 1, 2]]),
        (2147483647, 128, 0, [[0, 1, 2]]),
        (-2147483648, 1, 0, [[0, 1, 2]]),
        (1535, 2, 7, [[0, 1, 2]]),
        (0, -1, 0, [[0, 1, 2]]),
        (0, 129, 0, [[0, 1, 2]]),
        (0, 128, -1, [[0, 1, 2]]),
        (0, 128, 8, [[0, 1, 2]]),
        (505, 128, 0, [[0, 0, 2]]),
        (505, 128, 0, [[-1, 1, 2]]),
        (505, 128, 0, [[0, 3, 2]]),
    ],
)
def test_invalid_metadata_drops_every_target(offset, count, owner, table):
    result = jax.jit(addresses)(table, offset, count, owner)
    assert not bool(result.valid)
    np.testing.assert_array_equal(result.targets, np.full(128, 192))


def test_unused_pages_and_empty_at_capacity_are_valid():
    assert bool(addresses([[0, -1, 999]], 0, 3, 0).valid)
    empty = addresses([[2, 0, 1]], 1536, 0, 0)
    assert bool(empty.valid)
    np.testing.assert_array_equal(empty.targets, np.full(128, 192))


def test_capture_sanitizes_drop_rows_and_commit_does_not_wrap_negative():
    old = jnp.full((2, 3, 64, 4), 7, jnp.bfloat16)
    target = jnp.array([0, -1, 192, 191, 2147483647], jnp.int32)
    changed = old[0].at[0, 0].set(3).at[2, 63].set(4)
    pending = capture_prefill_pending_rows(changed, target)
    np.testing.assert_array_equal(pending[[1, 2, 4], :], np.zeros((3, 4)))
    out = apply_prefill_pending_rows(old, target, jnp.stack([pending, pending]))
    expected = old.at[:, 0, 0].set(3).at[:, 2, 63].set(4)
    assert np.asarray(out).tobytes() == np.asarray(expected).tobytes()


def test_only_compact_proposals_cross_the_conditional_boundary():
    # A small real lowered transaction, not a prediction of production TPU XLA.
    def body(old, rows, targets, healthy):
        return jax.lax.cond(
            healthy,
            lambda _: apply_prefill_pending_rows(old, targets, rows),
            lambda _: old,
            None,
        )

    old = jnp.zeros((3, 3, 64, 4), jnp.bfloat16)
    rows = jnp.ones((3, 128, 4), jnp.bfloat16)
    target = addresses([[2, 0, 1]], 505, 3, 0).targets
    fn = jax.jit(body, donate_argnums=(0,))
    saved = np.asarray(old).tobytes()
    out = fn(old, rows, target, jnp.bool_(False))
    assert np.asarray(out).tobytes() == saved and old.is_deleted()
    raw = str(
        fn.lower(
            jax.ShapeDtypeStruct((3, 3, 64, 4), jnp.bfloat16),
            rows,
            target,
            jnp.bool_(True),
        ).compiler_ir()
    )
    assert "stablehlo.case" in raw and "stablehlo.scatter" in raw


@pytest.mark.parametrize("bad", [0, 129, True, 1.5])
def test_bad_static_window_refused(bad):
    with pytest.raises(ValueError, match="1..128"):
        addresses([[0, 1, 2]], 0, 1, 0, rows=bad)
