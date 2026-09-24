"""CPU lossless route grouping; no TPU performance assertion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.layers.moe.routed_experts import (
    group_prefill_routes,
    gather_prefill_route_rows,
    restore_prefill_route_rows,
)


@pytest.mark.parametrize("rows", [1, 17, 128, 512])
@pytest.mark.parametrize("concentrated", [False, True])
def test_routes_round_trip_and_skew(rows, concentrated):
    rng = np.random.default_rng(183)
    count = 32 if concentrated else 256
    indices = np.stack([rng.choice(count, 8, replace=False) for _ in range(rows)]).astype(np.int32)
    routes = jax.jit(group_prefill_routes)(jnp.asarray(indices))
    assert bool(routes.valid)
    expected = np.argsort(indices.reshape(-1), kind="stable")
    np.testing.assert_array_equal(routes.sorted_flat_ids, expected)
    np.testing.assert_array_equal(routes.group_sizes, np.bincount(indices.reshape(-1), minlength=256))
    assert int(routes.group_sizes.sum()) == rows * 8
    if concentrated:
        assert int(routes.group_sizes[:32].sum()) == rows * 8
        assert int(routes.group_sizes[32:].sum()) == 0
    hidden = jnp.arange(rows * 4, dtype=jnp.float32).reshape(rows, 4).astype(jnp.bfloat16)
    gathered = gather_prefill_route_rows(hidden, routes, top_k=8)
    np.testing.assert_array_equal(gathered, np.asarray(hidden)[expected // 8])
    # Distinct route slots are essential: duplicating hidden rows alone cannot
    # expose inverse-permutation mistakes within one token.
    values = jnp.arange(rows * 8 * 3, dtype=jnp.float32).reshape(rows * 8, 3)
    restored = restore_prefill_route_rows(values[routes.sorted_flat_ids], routes, top_k=8)
    np.testing.assert_array_equal(restored, np.asarray(values).reshape(rows, 8, 3))


@pytest.mark.parametrize("bad", [-1, 256, 0])
def test_dynamic_invalid_routes_are_not_admitted(bad):
    values = jnp.arange(8, dtype=jnp.int32)[None, :].at[0, 7].set(bad)
    routes = jax.jit(group_prefill_routes)(values)
    assert not bool(routes.valid)


def test_wrong_route_dtype_is_rejected():
    with pytest.raises(ValueError, match="int32"):
        group_prefill_routes(jnp.zeros((2, 8), jnp.float32))
