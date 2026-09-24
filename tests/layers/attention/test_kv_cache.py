"""CPU address/causality tests; no TPU or cache-integrity promotion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.layers.attention.kv_cache import write_prefill_cache_block
from glm_tpu.layers.attention._s3_attention import StageLocalKvLayout


@pytest.mark.parametrize("offset,count", [(60, 17), (504, 17), (1020, 4), (64, 0)])
def test_block_writes_equal_scalar_reference_across_stripes_pages_and_tails(
    offset, count
):
    layout = StageLocalKvLayout(local_parallel_size=8, packed_cache_width=128)
    table = jnp.asarray([[2, 0, 1]], jnp.int32)
    rows = jnp.arange(17 * 128, dtype=jnp.float32).reshape(17, 128).astype(jnp.bfloat16)
    rows = rows.at[count:].set(jnp.nan)
    initial = jnp.full((3, 64, 128), -1, jnp.bfloat16)
    call = jax.jit(
        lambda cache, owner: write_prefill_cache_block(
            cache,
            rows,
            table,
            jnp.int32(offset),
            jnp.int32(count),
            owner,
            layout=layout,
        )
    )
    for owner in range(8):
        out = call(initial, jnp.int32(owner))
        expected = np.asarray(initial).copy()
        for i in range(count):
            p = offset + i
            if (p % 512) // 64 == owner:
                expected[int(table[0, p // 512]), p % 64] = np.asarray(rows[i])
        np.testing.assert_array_equal(np.asarray(out.cache), expected)
        assert bool(out.valid)
        np.testing.assert_array_equal(
            np.asarray(out.causal_lengths),
            [offset + i + 1 if i < count else 0 for i in range(17)],
        )
        np.testing.assert_array_equal(np.asarray(out.row_valid), np.arange(17) < count)


@pytest.mark.parametrize(
    "offset,count,owner,table,bad_live",
    [
        (512, 17, 0, [[0, 0, 2]], False),  # alias earlier, otherwise untouched page
        (512, 17, 0, [[-1, 1, 2]], False),
        (512, 17, 0, [[3, 1, 2]], False),
        (2147483647, 17, 0, [[0, 1, 2]], False),
        (-2147483648, 17, 0, [[0, 1, 2]], False),
        (0, 2147483647, 0, [[0, 1, 2]], False),
        (0, -1, 0, [[0, 1, 2]], False),
        (0, 17, 8, [[0, 1, 2]], False),
        (0, 17, -1, [[0, 1, 2]], False),
        (1530, 17, 0, [[0, 1, 2]], False),
        (0, 17, 0, [[0, 1, 2]], True),
    ],
)
def test_invalid_block_never_changes_cache(offset, count, owner, table, bad_live):
    layout = StageLocalKvLayout(local_parallel_size=8, packed_cache_width=128)
    cache = jnp.full((3, 64, 128), 7, jnp.bfloat16)
    rows = jnp.ones((17, 128), jnp.bfloat16)
    if bad_live:
        rows = rows.at[0, 0].set(jnp.inf)
    out = jax.jit(
        lambda c, r, t, o, n, e: write_prefill_cache_block(
            c, r, t, o, n, e, layout=layout
        )
    )(
        cache,
        rows,
        jnp.asarray(table, jnp.int32),
        jnp.int32(offset),
        jnp.int32(count),
        jnp.int32(owner),
    )
    assert not bool(out.valid)
    np.testing.assert_array_equal(np.asarray(out.cache), np.asarray(cache))
    assert (
        not np.asarray(out.row_valid).any() and not np.asarray(out.causal_lengths).any()
    )


def test_empty_at_capacity_ignores_padding_but_validates_live_prefix():
    layout = StageLocalKvLayout(local_parallel_size=8, packed_cache_width=640)
    cache = jnp.ones((1, 64, 640), jnp.bfloat16)
    out = write_prefill_cache_block(
        cache,
        jnp.full((1, 640), jnp.nan, jnp.bfloat16),
        jnp.asarray([[0]], jnp.int32),
        jnp.int32(512),
        jnp.int32(0),
        jnp.int32(0),
        layout=layout,
    )
    assert bool(out.valid)
    np.testing.assert_array_equal(np.asarray(out.cache), np.asarray(cache))
