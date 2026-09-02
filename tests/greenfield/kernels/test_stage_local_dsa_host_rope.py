from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from glm_tpu.greenfield.kernels.reference.dsa import DsaNumericalContract
from glm_tpu.greenfield.kernels.reference.rotary import rotary_cos_sin
from glm_tpu.greenfield.kernels.reference.rotary_table import dsa_rotary_row_host
from glm_tpu.greenfield.kernels.stage_local import (
    _local_dsa_current_key_from_projection,
    _local_dsa_query_from_projection,
)

CONTRACT = DsaNumericalContract()


def _jax_row(position: int) -> np.ndarray:
    cos, sin = rotary_cos_sin(
        jnp.asarray([position], dtype=jnp.int32),
        rotary_dim=CONTRACT.rotary_dim,
        theta=CONTRACT.theta,
        dtype=jnp.float32,
    )
    return np.concatenate((np.asarray(cos)[0], np.asarray(sin)[0]))


def test_stage_local_dsa_query_and_key_take_host_rows_without_changing_math() -> None:
    generator = np.random.default_rng(8155)
    projected_query = jnp.asarray(
        generator.standard_normal((1, 4 * CONTRACT.head_dim), dtype=np.float32)
    )
    normalized = jnp.asarray(
        generator.standard_normal((1, CONTRACT.hidden_size)).astype(np.float32)
    ).astype(jnp.bfloat16)
    head_weight = jnp.asarray(
        generator.standard_normal((4, CONTRACT.hidden_size)).astype(np.float32)
    ).astype(jnp.bfloat16)
    projected_key = jnp.asarray(
        generator.standard_normal((1, CONTRACT.head_dim), dtype=np.float32) * 2
    )
    weight = jnp.ones((CONTRACT.head_dim,), jnp.bfloat16)
    bias = jnp.zeros((CONTRACT.head_dim,), jnp.bfloat16)
    position = jnp.asarray([8155], dtype=jnp.int32)
    reference_query, reference_weights = _local_dsa_query_from_projection(
        projected_query, normalized, head_weight, position, contract=CONTRACT
    )
    same_row = jnp.asarray(_jax_row(8155))
    query, weights = _local_dsa_query_from_projection(
        projected_query,
        normalized,
        head_weight,
        position,
        contract=CONTRACT,
        dsa_rope_table_row=same_row,
    )
    assert np.array_equal(np.asarray(query), np.asarray(reference_query))
    assert np.array_equal(np.asarray(weights), np.asarray(reference_weights))
    host_row = jnp.asarray(
        dsa_rotary_row_host(8155, rotary_dim=CONTRACT.rotary_dim, theta=CONTRACT.theta)
    )
    host_query, _ = _local_dsa_query_from_projection(
        projected_query,
        normalized,
        head_weight,
        position,
        contract=CONTRACT,
        dsa_rope_table_row=host_row,
    )
    assert np.abs(np.asarray(host_query) - np.asarray(reference_query)).max() < 1e-6
    for mode in ("divide_sqrt", "multiply_rsqrt"):
        reference_key = _local_dsa_current_key_from_projection(
            projected_key,
            weight,
            bias,
            position,
            contract=CONTRACT,
            key_norm_mode=mode,
            dsa_rope_table_row=None,
        )
        key = _local_dsa_current_key_from_projection(
            projected_key,
            weight,
            bias,
            position,
            contract=CONTRACT,
            key_norm_mode=mode,
            dsa_rope_table_row=same_row,
        )
        assert np.array_equal(np.asarray(key), np.asarray(reference_key))
        assert key.dtype == jnp.float32
        host_key = _local_dsa_current_key_from_projection(
            projected_key,
            weight,
            bias,
            position,
            contract=CONTRACT,
            key_norm_mode=mode,
            dsa_rope_table_row=host_row,
        )
        assert np.array_equal(
            np.asarray(host_key)[:, CONTRACT.rotary_dim :],
            np.asarray(reference_key)[:, CONTRACT.rotary_dim :],
        )
        assert np.abs(np.asarray(host_key) - np.asarray(reference_key)).max() < 1e-6
