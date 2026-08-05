from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference import (
    apply_rotary,
    dense_swiglu,
    embedding_lookup,
    final_norm,
    linear,
    residual_add,
    rms_norm,
    rotary_cos_sin,
    vocabulary_logits,
)


def test_rms_norm_matches_glm_fp32_then_activation_rounding() -> None:
    hidden = jnp.asarray([[1.0, -2.0, 3.0, -4.0]], dtype=jnp.bfloat16)
    weight = jnp.asarray([1.0, 0.5, -0.25, 2.0], dtype=jnp.bfloat16)
    got = rms_norm(hidden, weight, epsilon=1e-5)
    value = hidden.astype(jnp.float32)
    normalized = value * jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + 1e-5)
    expected = normalized.astype(jnp.bfloat16) * weight
    assert got.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))
    np.testing.assert_array_equal(
        np.asarray(final_norm(hidden, weight, epsilon=1e-5)), np.asarray(got)
    )


def test_rms_norm_refuses_shape_dtype_and_epsilon_drift() -> None:
    with pytest.raises(ValueError, match="weight"):
        rms_norm(jnp.ones((1, 4)), jnp.ones((3,)), epsilon=1e-5)
    with pytest.raises(ValueError, match="epsilon"):
        rms_norm(jnp.ones((1, 4)), jnp.ones((4,)), epsilon=0)
    with pytest.raises(ValueError, match="inexact"):
        rms_norm(jnp.ones((1, 4), jnp.int32), jnp.ones((4,)), epsilon=1e-5)


def test_linear_preserves_checkpoint_out_in_orientation_and_leading_shape() -> None:
    hidden = jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3)
    weight = jnp.asarray([[1, 2, 3], [-1, 0, 1]], dtype=jnp.float32)
    bias = jnp.asarray([0.5, -0.5], dtype=jnp.float32)
    got = linear(hidden, weight, bias)
    expected = np.asarray(hidden) @ np.asarray(weight).T + np.asarray(bias)
    np.testing.assert_array_equal(np.asarray(got), expected)
    assert got.shape == (2, 2, 2)


def test_linear_bf16_has_explicit_bf16_output_boundary() -> None:
    hidden = jnp.asarray([[1.0, -0.5, 0.25]], dtype=jnp.bfloat16)
    weight = jnp.asarray([[2.0, 1.0, -1.0]], dtype=jnp.bfloat16)
    got = jax.jit(linear)(hidden, weight)
    assert got.dtype == jnp.bfloat16
    assert got.shape == (1, 1)
    assert float(got[0, 0]) == 1.25


def test_linear_refuses_silent_shape_broadcasts() -> None:
    with pytest.raises(ValueError, match="input width"):
        linear(jnp.ones((1, 3)), jnp.ones((2, 4)))
    with pytest.raises(ValueError, match="bias"):
        linear(jnp.ones((1, 3)), jnp.ones((2, 3)), jnp.ones((1,)))


def test_dense_swiglu_residual_embedding_and_logits_are_explicit() -> None:
    hidden = jnp.asarray([[0.5, -1.0]], dtype=jnp.float32)
    gate = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32)
    up = jnp.asarray([[0.5, 0.0], [0.0, 0.5], [1.0, -1.0]], dtype=jnp.float32)
    down = jnp.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]], dtype=jnp.float32)
    got = dense_swiglu(hidden, gate, up, down)
    gate_value = np.asarray(hidden) @ np.asarray(gate).T
    up_value = np.asarray(hidden) @ np.asarray(up).T
    activated = gate_value / (1.0 + np.exp(-gate_value))
    expected = (activated * up_value) @ np.asarray(down).T
    np.testing.assert_allclose(np.asarray(got), expected, rtol=2e-7, atol=2e-7)
    np.testing.assert_array_equal(
        np.asarray(residual_add(hidden, got)), np.asarray(hidden + got)
    )

    table = jnp.arange(12, dtype=jnp.bfloat16).reshape(6, 2)
    ids = jnp.asarray([[5, 0]], dtype=jnp.int32)
    embeddings = embedding_lookup(ids, table)
    np.testing.assert_array_equal(np.asarray(embeddings), np.asarray(table)[[5, 0]][None])
    logits = vocabulary_logits(embeddings, table)
    assert logits.shape == (1, 2, 6)
    assert logits.dtype == jnp.bfloat16


def test_residual_add_refuses_broadcast_or_dtype_promotion() -> None:
    with pytest.raises(ValueError, match="shapes"):
        residual_add(jnp.ones((1, 4)), jnp.ones((4,)))
    with pytest.raises(ValueError, match="dtypes"):
        residual_add(jnp.ones((1, 4), jnp.bfloat16), jnp.ones((1, 4), jnp.float32))


def test_rotary_tables_use_glm_frequency_formula() -> None:
    positions = jnp.asarray([0, 1, 3], dtype=jnp.int32)
    cos, sin = rotary_cos_sin(
        positions, rotary_dim=4, theta=100.0, dtype=jnp.float32
    )
    frequencies = np.asarray([1.0, 0.1], dtype=np.float32)
    angles = np.asarray([0.0, 1.0, 3.0], dtype=np.float32)[:, None] * frequencies
    np.testing.assert_allclose(np.asarray(cos), np.cos(angles), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sin), np.sin(angles), rtol=1e-6, atol=1e-6)


def test_rotary_pair_layouts_are_explicit_and_dot_preserving() -> None:
    value = jnp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float32)
    cos = jnp.asarray([[0.0, 0.0]], dtype=jnp.float32)
    sin = jnp.asarray([[1.0, 1.0]], dtype=jnp.float32)
    interleaved = apply_rotary(value, cos, sin, interleaved=True)
    half_split = apply_rotary(value, cos, sin, interleaved=False)
    np.testing.assert_array_equal(np.asarray(interleaved), [[-2.0, 1.0, -4.0, 3.0]])
    np.testing.assert_array_equal(np.asarray(half_split), [[-3.0, -4.0, 1.0, 2.0]])
    np.testing.assert_allclose(
        np.asarray(jnp.sum(interleaved * interleaved, axis=-1)),
        np.asarray(jnp.sum(value * value, axis=-1)),
        rtol=0,
        atol=0,
    )


def test_rotary_refuses_odd_width_and_noninteger_positions() -> None:
    with pytest.raises(ValueError, match="even"):
        rotary_cos_sin(jnp.asarray([0]), rotary_dim=3, theta=10.0)
    with pytest.raises(ValueError, match="integer"):
        rotary_cos_sin(jnp.asarray([0.0]), rotary_dim=4, theta=10.0)
    with pytest.raises(ValueError, match="pair dimension"):
        apply_rotary(
            jnp.ones((1, 4)), jnp.ones((1, 1)), jnp.ones((1, 1)), interleaved=True
        )
