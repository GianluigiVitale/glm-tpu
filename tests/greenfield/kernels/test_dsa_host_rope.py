from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference.dsa import (
    DsaNumericalContract,
    dsa_index_keys_from_projection,
)
from glm_tpu.greenfield.kernels.reference.dsa_host_rope import (
    dsa_index_keys_from_projection_host_rope,
    rotary_cos_sin_from_rows,
    rotate_dsa_query_host_rope,
)
from glm_tpu.greenfield.kernels.reference.rotary import apply_rotary, rotary_cos_sin
from glm_tpu.greenfield.kernels.reference.rotary_table import (
    build_dsa_rotary_table_host,
    dsa_inverse_frequencies_host,
    dsa_rotary_row_host,
    dsa_rotary_table_sha256,
)

CONTRACT = DsaNumericalContract()
POSITIONS = (0, 1, 7, 8155, 65535, 262143)


def _jax_rows(positions: tuple[int, ...]) -> np.ndarray:
    cos, sin = rotary_cos_sin(
        jnp.asarray(positions, dtype=jnp.int32),
        rotary_dim=CONTRACT.rotary_dim,
        theta=CONTRACT.theta,
        dtype=jnp.float32,
    )
    return np.concatenate((np.asarray(cos), np.asarray(sin)), axis=-1)


def test_host_inverse_frequencies_are_bit_identical_to_on_device_form() -> None:
    host = dsa_inverse_frequencies_host(
        rotary_dim=CONTRACT.rotary_dim, theta=CONTRACT.theta
    )
    device_form = np.asarray(
        jnp.power(
            jnp.float32(CONTRACT.theta),
            -jnp.arange(0, CONTRACT.rotary_dim, 2, dtype=jnp.float32)
            / jnp.float32(CONTRACT.rotary_dim),
        )
    )
    assert host.dtype == np.float32
    assert np.array_equal(host, device_form)


def test_host_table_rows_match_cpu_reference_within_one_ulp() -> None:
    table = build_dsa_rotary_table_host(
        262144, rotary_dim=CONTRACT.rotary_dim, theta=CONTRACT.theta
    )
    assert table.shape == (262144, CONTRACT.rotary_dim) and table.dtype == np.float32
    reference = _jax_rows(POSITIONS)
    rows = table[list(POSITIONS)]
    # libm versus XLA-CPU polynomial: at most one FP32 ulp of a unit-magnitude value.
    assert np.abs(rows - reference).max() <= 2.0**-24 * 1.01
    for position in POSITIONS:
        row = dsa_rotary_row_host(
            position, rotary_dim=CONTRACT.rotary_dim, theta=CONTRACT.theta
        )
        assert np.array_equal(row, table[position])
    assert dsa_rotary_table_sha256(table) == dsa_rotary_table_sha256(table.copy())
    assert dsa_rotary_table_sha256(table[8155]) != dsa_rotary_table_sha256(table[0])
    assert np.all(
        np.abs(
            table[:, : CONTRACT.rotary_dim // 2] ** 2
            + table[:, CONTRACT.rotary_dim // 2 :] ** 2
            - 1.0
        )
        < 1e-6
    )


def test_host_rows_reproduce_reference_keys_and_isolate_rotary_only() -> None:
    generator = np.random.default_rng(8155)
    projected = jnp.asarray(
        generator.standard_normal((3, CONTRACT.head_dim), dtype=np.float32) * 2.5
    )
    weight = jnp.asarray(
        1.0 + 0.1 * generator.standard_normal(CONTRACT.head_dim), dtype=jnp.float32
    )
    bias = jnp.asarray(
        0.05 * generator.standard_normal(CONTRACT.head_dim), dtype=jnp.float32
    )
    positions = jnp.asarray([8155, 0, 262143], dtype=jnp.int32)
    reference = dsa_index_keys_from_projection(
        projected, weight, bias, positions, key_norm_mode="divide_sqrt"
    )
    # Rows built from the on-device formula reproduce the established path exactly.
    same_rows = jnp.asarray(_jax_rows((8155, 0, 262143)))
    exact = dsa_index_keys_from_projection_host_rope(
        projected, weight, bias, positions, same_rows, key_norm_mode="divide_sqrt"
    )
    assert np.array_equal(np.asarray(exact), np.asarray(reference))
    host_rows = jnp.asarray(
        np.stack(
            [
                dsa_rotary_row_host(
                    int(p), rotary_dim=CONTRACT.rotary_dim, theta=CONTRACT.theta
                )
                for p in (8155, 0, 262143)
            ]
        )
    )
    host = np.asarray(
        dsa_index_keys_from_projection_host_rope(
            projected, weight, bias, positions, host_rows, key_norm_mode="divide_sqrt"
        )
    )
    reference_np = np.asarray(reference)
    assert np.array_equal(
        host[:, CONTRACT.rotary_dim :], reference_np[:, CONTRACT.rotary_dim :]
    )
    assert np.abs(host - reference_np).max() < 1e-6


def test_query_rotation_matches_reference_pairing() -> None:
    generator = np.random.default_rng(3)
    query = jnp.asarray(
        generator.standard_normal(
            (2, CONTRACT.num_heads, CONTRACT.head_dim), dtype=np.float32
        )
    )
    positions = jnp.asarray([8155, 12], dtype=jnp.int32)
    rows = jnp.asarray(_jax_rows((8155, 12)))
    rotated = rotate_dsa_query_host_rope(query, positions, rows)
    cos, sin = rotary_cos_sin(
        positions,
        rotary_dim=CONTRACT.rotary_dim,
        theta=CONTRACT.theta,
        dtype=jnp.float32,
    )
    expected = jnp.concatenate(
        (
            apply_rotary(
                query[..., : CONTRACT.rotary_dim],
                cos[:, None, :],
                sin[:, None, :],
                interleaved=CONTRACT.interleaved_rotary,
            ),
            query[..., CONTRACT.rotary_dim :],
        ),
        axis=-1,
    )
    assert np.array_equal(np.asarray(rotated), np.asarray(expected))


def test_host_rope_helpers_fail_loudly_on_shape_dtype_and_argument_drift() -> None:
    positions = jnp.asarray([8155], dtype=jnp.int32)
    good = jnp.asarray(_jax_rows((8155,)))
    cos, sin = rotary_cos_sin_from_rows(good, positions, rotary_dim=CONTRACT.rotary_dim)
    assert cos.shape == sin.shape == (1, CONTRACT.rotary_dim // 2)
    with pytest.raises(ValueError, match="match positions"):
        rotary_cos_sin_from_rows(
            good[:, :32], positions, rotary_dim=CONTRACT.rotary_dim
        )
    with pytest.raises(ValueError, match="FP32"):
        rotary_cos_sin_from_rows(
            good.astype(jnp.bfloat16), positions, rotary_dim=CONTRACT.rotary_dim
        )
    with pytest.raises(ValueError, match="even"):
        rotary_cos_sin_from_rows(good, positions, rotary_dim=63)
    with pytest.raises(TypeError):
        rotary_cos_sin_from_rows(good, positions, rotary_dim=True)  # type: ignore[arg-type]
    for bad in (
        lambda: build_dsa_rotary_table_host(0, rotary_dim=64, theta=8e6),
        lambda: build_dsa_rotary_table_host(4, rotary_dim=63, theta=8e6),
        lambda: build_dsa_rotary_table_host(4, rotary_dim=64, theta=0),
        lambda: dsa_rotary_row_host(-1, rotary_dim=64, theta=8e6),
        lambda: dsa_rotary_table_sha256(np.zeros((2, 2), dtype=np.float64)),
    ):
        with pytest.raises(ValueError):
            bad()
    projected = jnp.zeros((1, CONTRACT.head_dim), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="FP32"):
        dsa_index_keys_from_projection_host_rope(
            projected,
            jnp.ones((128,), jnp.float32),
            jnp.zeros((128,), jnp.float32),
            positions,
            good,
        )
    assert jax.default_backend() == "cpu"
