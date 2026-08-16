from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from glm_tpu.greenfield.kernels.reference.dsa import (
    DsaNumericalContract,
    dsa_index_keys_from_projection,
    dsa_query_and_head_weights,
)
from glm_tpu.greenfield.kernels.reference.linear import linear
from glm_tpu.greenfield.kernels.ws32_layer import (
    ws32_exact_dsa_current_key,
    ws32_grouped_dsa_query_and_head,
)


def _contract() -> DsaNumericalContract:
    return DsaNumericalContract(
        hidden_size=8,
        q_lora_rank=4,
        num_heads=32,
        head_dim=128,
        rotary_dim=64,
        top_k=4,
    )


def _draw(seed: int, shape: tuple[int, ...], scale: float = 0.125) -> jnp.ndarray:
    value = np.random.default_rng(seed).normal(0.0, scale, shape)
    return jnp.asarray(value, dtype=jnp.float32)


def _candidate_query(
    *,
    alias_count: int,
    q_residual: jnp.ndarray,
    normalized: jnp.ndarray,
    full_query_weight: jnp.ndarray,
    full_head_weight: jnp.ndarray,
    position: jnp.ndarray,
    contract: DsaNumericalContract,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    local_heads = contract.num_heads // alias_count
    query_parts = []
    head_parts = []
    for owner in range(alias_count):
        query_start = owner * local_heads * contract.head_dim
        query_weight = full_query_weight[
            query_start : query_start + local_heads * contract.head_dim
        ]
        head_weight = full_head_weight[
            owner * local_heads : (owner + 1) * local_heads
        ]
        query, head = ws32_grouped_dsa_query_and_head(
            q_residual,
            normalized,
            tuple(query_weight for _ in range(alias_count)),
            head_weight,
            position,
            contract=contract,
        )
        query_parts.append(query)
        head_parts.append(head)
    return jnp.concatenate(query_parts, axis=1), jnp.concatenate(head_parts, axis=1)


def test_ws32_tuple4_and_tuple8_query_candidates_match_reference() -> None:
    contract = _contract()
    q_residual = _draw(1, (1, contract.q_lora_rank)).astype(jnp.bfloat16)
    normalized = _draw(2, (1, contract.hidden_size)).astype(jnp.bfloat16)
    query_weight = _draw(
        3, (contract.num_heads * contract.head_dim, contract.q_lora_rank)
    )
    head_weight = _draw(
        4, (contract.num_heads, contract.hidden_size)
    ).astype(jnp.bfloat16)
    position = jnp.asarray([8155], dtype=jnp.int32)

    expected_query, expected_head = dsa_query_and_head_weights(
        normalized,
        q_residual,
        query_weight,
        head_weight,
        position,
        contract=contract,
    )
    for alias_count in (4, 8):
        actual_query, actual_head = _candidate_query(
            alias_count=alias_count,
            q_residual=q_residual,
            normalized=normalized,
            full_query_weight=query_weight,
            full_head_weight=head_weight,
            position=position,
            contract=contract,
        )
        np.testing.assert_array_equal(np.asarray(actual_query), np.asarray(expected_query))
        np.testing.assert_array_equal(np.asarray(actual_head), np.asarray(expected_head))


def test_ws32_exact_current_key_matches_divide_sqrt_reference() -> None:
    contract = _contract()
    normalized = _draw(5, (1, contract.hidden_size)).astype(jnp.bfloat16)
    wk = _draw(6, (contract.head_dim, contract.hidden_size))
    norm_weight = _draw(7, (contract.head_dim,), 0.02).astype(jnp.bfloat16)
    norm_bias = _draw(8, (contract.head_dim,), 0.01).astype(jnp.bfloat16)
    position = jnp.asarray([8155], dtype=jnp.int32)

    expected = dsa_index_keys_from_projection(
        linear(normalized, wk, output_dtype=jnp.float32),
        norm_weight,
        norm_bias,
        position,
        contract=contract,
        key_norm_mode="divide_sqrt",
    ).astype(jnp.float32)
    actual = ws32_exact_dsa_current_key(
        normalized,
        wk,
        norm_weight,
        norm_bias,
        position,
        contract=contract,
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_ws32_grouped_query_rejects_non_16kib_geometry() -> None:
    contract = _contract()
    q_residual = jnp.zeros((1, contract.q_lora_rank), dtype=jnp.bfloat16)
    normalized = jnp.zeros((1, contract.hidden_size), dtype=jnp.bfloat16)
    wrong_owner = jnp.zeros((128, contract.q_lora_rank), dtype=jnp.float32)
    head_weight = jnp.zeros((1, contract.hidden_size), dtype=jnp.bfloat16)

    try:
        ws32_grouped_dsa_query_and_head(
            q_residual,
            normalized,
            (wrong_owner,) * 4,
            head_weight,
            jnp.asarray([0], dtype=jnp.int32),
            contract=contract,
        )
    except ValueError as error:
        assert "retain 32 heads" in str(error)
    else:
        raise AssertionError("non-16-KiB grouped query geometry was accepted")
