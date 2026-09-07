"""CPU semantic admission for reusing raw-FP8 tiles in batched prefill.

This is deliberately not a TPU performance test or full-layer admission. It
checks the existing primitive before new prefill wrappers are implemented.
The same comparisons must run on real TPU inputs before production adoption.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from glm_tpu.greenfield.kernels.pallas.fp8_matmul import (
    Fp8BlockMatmulConfig,
    fp8_block_matmul,
    fp8_block_matmul_f32,
)


@pytest.mark.parametrize("rows", [8, 17, 32])
@pytest.mark.parametrize("zero_output_tail", [False, True])
def test_multirow_fp8_matches_individual_rows_with_tails(
    rows: int, zero_output_tail: bool
) -> None:
    """Changing the row geometry must not mix tokens or expose padded rows."""

    # Both contraction and output cross a 128-wide tile with a partial tail.
    # Distinct rows, a zero row and zero scale blocks catch cross-row/boundary
    # mistakes that duplicating one token across the row dimension would miss.
    rng = np.random.default_rng(9017)
    lhs = jnp.asarray(rng.normal(0, 0.25, (rows, 130)), jnp.bfloat16)
    lhs = lhs.at[rows // 2].set(0)
    weight = jnp.asarray(rng.normal(0, 0.2, (135, 130)), jnp.float8_e4m3fn)
    bits = lax.bitcast_convert_type(weight, jnp.uint8)
    tail_scale = [0.0, 0.0] if zero_output_tail else [1.5, 0.5]
    scale = jnp.asarray([[0.75, 0.0], tail_scale], jnp.float32)
    config = Fp8BlockMatmulConfig()

    def project(value: jax.Array) -> jax.Array:
        return fp8_block_matmul_f32(
            value, bits, scale, config=config, interpret=True
        )

    batch = jax.jit(project)(lhs)
    one = jax.jit(project)
    serial = jnp.concatenate([one(lhs[i : i + 1]) for i in range(rows)])
    np.testing.assert_array_equal(
        np.asarray(batch).view(np.uint32), np.asarray(serial).view(np.uint32)
    )
    assert batch.shape == (rows, 135)
    assert batch.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(batch[rows // 2]), 0)
    if zero_output_tail:
        np.testing.assert_array_equal(np.asarray(batch[:, 128:]), 0)
    else:
        assert np.any(np.asarray(batch[:, 128:]) != 0)

    rounded = fp8_block_matmul(lhs, bits, scale, config=config, interpret=True)
    assert rounded.shape == (rows, 135)
    assert rounded.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(rounded).view(np.uint16),
        np.asarray(serial.astype(jnp.bfloat16)).view(np.uint16),
    )

    # Permuting real rows must only permute their outputs, not change expert
    # or matrix ownership. This does not test routing, which has its own gate.
    order = np.random.default_rng(44).permutation(rows)
    permuted = jax.jit(project)(lhs[order])
    np.testing.assert_array_equal(
        np.asarray(permuted).view(np.uint32),
        np.asarray(batch[order]).view(np.uint32),
    )
