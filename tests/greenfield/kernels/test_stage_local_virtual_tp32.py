from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from glm_tpu.greenfield.kernels.stage_local import (
    _sum_virtual_dcp_bf16_partials,
)


def _bf16_add(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.asarray(
        left.astype(np.float32) + right.astype(np.float32),
        dtype=jnp.bfloat16,
    )


def test_virtual_dcp_bf16_trees_are_explicit_and_numerically_distinct() -> None:
    source = np.asarray(
        [
            14080.0,
            -1.9140625,
            -2.875,
            2496.0,
            -2496.0,
            -0.86328125,
            0.00927734375,
            540.0,
        ],
        dtype=jnp.bfloat16,
    ).reshape(8, 1, 1)

    sequential = source[0]
    for value in source[1:]:
        sequential = _bf16_add(sequential, value)
    pairwise = tuple(source[index] for index in range(8))
    while len(pairwise) > 1:
        pairwise = tuple(
            _bf16_add(pairwise[index], pairwise[index + 1])
            for index in range(0, len(pairwise), 2)
        )

    actual_sequential = np.asarray(
        jax.jit(
            lambda value: _sum_virtual_dcp_bf16_partials(
                value, pairwise=False
            )
        )(jnp.asarray(source))
    )
    actual_pairwise = np.asarray(
        jax.jit(
            lambda value: _sum_virtual_dcp_bf16_partials(
                value, pairwise=True
            )
        )(jnp.asarray(source))
    )
    np.testing.assert_array_equal(actual_sequential, sequential)
    np.testing.assert_array_equal(actual_pairwise, pairwise[0])
    assert not np.array_equal(
        actual_sequential.view(np.uint16),
        actual_pairwise.view(np.uint16),
    )

    sequential_hlo = jax.jit(
        lambda value: _sum_virtual_dcp_bf16_partials(
            value, pairwise=False
        )
    ).lower(jnp.asarray(source)).as_text()
    pairwise_hlo = jax.jit(
        lambda value: _sum_virtual_dcp_bf16_partials(
            value, pairwise=True
        )
    ).lower(jnp.asarray(source)).as_text()
    assert sequential_hlo.count("stablehlo.optimization_barrier") == 7
    assert pairwise_hlo.count("stablehlo.optimization_barrier") == 7
    assert sequential_hlo != pairwise_hlo


def test_virtual_dcp_bf16_sum_rejects_unrounded_or_wrong_shard_count() -> None:
    with np.testing.assert_raises_regex(ValueError, "eight rank-two"):
        _sum_virtual_dcp_bf16_partials(
            jnp.zeros((7, 1, 4), dtype=jnp.bfloat16), pairwise=False
        )
    with np.testing.assert_raises_regex(ValueError, "rounded BF16"):
        _sum_virtual_dcp_bf16_partials(
            jnp.zeros((8, 1, 4), dtype=jnp.float32), pairwise=True
        )
