"""Tests of :mod:`glm_tpu.layers.rope`."""

from __future__ import annotations

from hashlib import sha256

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.layers.rope import (
    apply_rotary,
    apply_rotary_fp32_final_round,
    build_rotary_table_host,
    rotary_cos_sin,
    rotary_table_sha256,
)


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
