from __future__ import annotations

from hashlib import sha256

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.optimized.reference.rotary import (
    apply_rotary,
    apply_rotary_fp32_final_round,
    build_rotary_table_host,
    rotary_cos_sin,
    rotary_table_sha256,
)
from glm_tpu.optimized.reference.linear import (
    dense_swiglu,
    embedding_lookup,
    linear,
    residual_add,
    vocabulary_logits,
)
from glm_tpu.optimized.reference.rmsnorm import final_norm, fused_add_rms_norm, rms_norm


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


def test_fused_add_rms_norm_uses_unrounded_fp32_sum() -> None:
    rng = np.random.default_rng(0)
    hidden = jnp.asarray(
        rng.normal(size=(1, 64)) * 3.0, dtype=jnp.bfloat16
    )
    residual = jnp.asarray(
        rng.normal(size=(1, 64)) * 3.0, dtype=jnp.bfloat16
    )
    weight = jnp.asarray(
        rng.normal(loc=1.0, scale=0.1, size=(64,)), dtype=jnp.bfloat16
    )

    normalized, carried = fused_add_rms_norm(
        hidden, residual, weight, epsilon=1e-5
    )
    summed = hidden.astype(jnp.float32) + residual.astype(jnp.float32)
    expected_carried = summed.astype(jnp.bfloat16)
    expected_normalized = (
        (
            summed
            * jax.lax.rsqrt(
                jnp.mean(summed * summed, axis=-1, keepdims=True) + 1e-5
            )
        ).astype(jnp.bfloat16)
        * weight
    ).astype(jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(carried), np.asarray(expected_carried))
    np.testing.assert_array_equal(
        np.asarray(normalized), np.asarray(expected_normalized)
    )

    rounded_first = rms_norm(expected_carried, weight, epsilon=1e-5)
    mismatch_count = int(
        jnp.count_nonzero(
            jax.lax.bitcast_convert_type(normalized, jnp.uint16)
            != jax.lax.bitcast_convert_type(rounded_first, jnp.uint16)
        )
    )
    assert mismatch_count == 18


def test_fused_add_rms_norm_refuses_state_contract_drift() -> None:
    hidden = jnp.ones((1, 4), dtype=jnp.bfloat16)
    residual = jnp.ones((1, 4), dtype=jnp.bfloat16)
    weight = jnp.ones((4,), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="shapes"):
        fused_add_rms_norm(hidden, residual[:, :3], weight, epsilon=1e-5)
    with pytest.raises(ValueError, match="dtypes"):
        fused_add_rms_norm(
            hidden, residual.astype(jnp.float32), weight, epsilon=1e-5
        )
    with pytest.raises(ValueError, match="weight"):
        fused_add_rms_norm(hidden, residual, weight[:3], epsilon=1e-5)
    with pytest.raises(ValueError, match="epsilon"):
        fused_add_rms_norm(hidden, residual, weight, epsilon=0)


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


def test_host_rotary_table_matches_protected_accepted_row() -> None:
    table = build_rotary_table_host(
        8192, rotary_dim=64, theta=8_000_000.0
    )
    assert table.shape == (8192, 64)
    assert table.dtype.name == "bfloat16"
    assert table.flags.c_contiguous
    row_bits = np.ascontiguousarray(table[8155]).view(np.uint16)
    assert rotary_table_sha256(table) == (
        "6a22140fc2aec475399738c6fc0f29be2a6c419feb0249aee35681c607c80701"
    )
    assert rotary_table_sha256(table) == rotary_table_sha256(table.copy())
    assert sha256(row_bits.tobytes()).hexdigest() == (
        "67b01e3cab682d5ffd04ac9c8043e7e6825ee1275023428c41f9a1ae412dea1d"
    )


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


def test_rotary_fp32_final_round_matches_accepted_layer0_cache_row() -> None:
    def bfloat16_from_hex(value: str, shape: tuple[int, ...]) -> jax.Array:
        bits = np.frombuffer(bytes.fromhex(value), dtype="<u2").reshape(shape)
        return jax.lax.bitcast_convert_type(jnp.asarray(bits), jnp.bfloat16)

    pre_rope = bfloat16_from_hex(
        "febe8fbe263ec7bf443fc9be2e3f6d3fb3be8c3f14bf1a3f"
        "5abf78beb93de0bf2b4028bfc43e36be7f3f823f873f693f"
        "d4bf1e3f103f133e343fb13fa83f3a3f8abf843e9c3ff23e"
        "3abe633dd2be76bfe8be543f9ebf19bd253e9d3e333e1b3e"
        "fdbf893f3c3f8fbfe23db73efb3f864052befebf4cbf7abe"
        "7bc03fc08ebf303e",
        (1, 64),
    )
    cos = bfloat16_from_hex(
        "573fbc3e3cbf7bbf7d3fa4be533f4b3f53bf183f793f80bf"
        "12bf793f8b3dd23c78bf3bbef73e4c3f6c3f793f7d3f7f3f"
        "803f803f803f803f803f803f803f803f",
        (1, 32),
    )
    sin = bfloat16_from_hex(
        "0bbf6ebf2ebf483e23be723f11bf1c3f103f4ebf6c3e913b"
        "523f683e7f3f80bf833e7c3f603f1b3fc53e743e153eb63d"
        "5e3d073da43c483cf43b943b343bdc3a",
        (1, 32),
    )
    expected = np.frombuffer(
        bytes.fromhex(
            "11bf0f3db1bf39bf54bf6ebe59bf46bf30be913fc4be3dbf"
            "57bf913e923fabbfebbf0340ad3dd4be3c3f9c3f88bf68bf"
            "e03edbbf043f893eaabf4b3f433fa6bf7a3f07bf30bf8f3f"
            "0bbe07be833e82bf3dbf173f99bfa9beeb3da73e253e2a3e"
            "02c0773f453f8cbfd33db83ef43f874043befebf4cbf7ebe"
            "7ac040c08ebf2e3e"
        ),
        dtype="<u2",
    ).reshape(1, 64)

    got = apply_rotary_fp32_final_round(
        pre_rope, cos, sin, interleaved=True
    )
    got_bits = np.asarray(
        jax.lax.bitcast_convert_type(got, jnp.uint16)
    )
    np.testing.assert_array_equal(got_bits, expected)

    rounded_products = apply_rotary(pre_rope, cos, sin, interleaved=True)
    rounded_bits = np.asarray(
        jax.lax.bitcast_convert_type(rounded_products, jnp.uint16)
    )
    assert int(np.count_nonzero(rounded_bits != expected)) == 16


def test_rotary_refuses_odd_width_and_noninteger_positions() -> None:
    with pytest.raises(ValueError, match="capacity"):
        build_rotary_table_host(0, rotary_dim=64, theta=8_000_000.0)
    with pytest.raises(ValueError, match="dimension"):
        build_rotary_table_host(8, rotary_dim=63, theta=8_000_000.0)
    with pytest.raises(ValueError, match="theta"):
        build_rotary_table_host(8, rotary_dim=64, theta=0.0)
    with pytest.raises(ValueError, match="two-dimensional BF16"):
        rotary_table_sha256(np.zeros((8, 64), dtype=np.float32))
    with pytest.raises(ValueError, match="even"):
        rotary_cos_sin(jnp.asarray([0]), rotary_dim=3, theta=10.0)
    with pytest.raises(ValueError, match="integer"):
        rotary_cos_sin(jnp.asarray([0.0]), rotary_dim=4, theta=10.0)
    with pytest.raises(ValueError, match="pair dimension"):
        apply_rotary(
            jnp.ones((1, 4)), jnp.ones((1, 1)), jnp.ones((1, 1)), interleaved=True
        )
    with pytest.raises(ValueError, match="inexact"):
        apply_rotary_fp32_final_round(
            jnp.ones((1, 4), dtype=jnp.int32),
            jnp.ones((1, 2), dtype=jnp.float32),
            jnp.ones((1, 2), dtype=jnp.float32),
            interleaved=True,
        )
