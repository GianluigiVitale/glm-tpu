from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference import (
    DsaNumericalContract,
    distributed_exact_topk_reference,
    dsa_index_keys,
    dsa_index_keys_from_projection,
    dsa_query_and_head_weights,
    dsa_scores,
    exact_topk,
    local_topk_candidates,
    linear,
    merge_topk_candidates,
)


def small_contract() -> DsaNumericalContract:
    return DsaNumericalContract(
        hidden_size=6,
        q_lora_rank=4,
        num_heads=2,
        head_dim=4,
        rotary_dim=2,
        top_k=4,
        theta=100.0,
    )


def test_glm_dsa_contract_is_exact() -> None:
    contract = DsaNumericalContract()
    assert contract.hidden_size == 6144
    assert contract.q_lora_rank == 2048
    assert contract.num_heads == 32
    assert contract.head_dim == 128
    assert contract.rotary_dim == 64
    assert contract.top_k == 2048
    assert contract.theta == 8_000_000.0
    assert contract.padding_sentinel == -1


def test_dsa_contract_refuses_semantic_drift() -> None:
    with pytest.raises(ValueError, match="rotary_dim"):
        DsaNumericalContract(head_dim=4, rotary_dim=6)
    with pytest.raises(ValueError, match="FP32"):
        DsaNumericalContract(score_dtype="bfloat16")
    with pytest.raises(ValueError, match="interleaved"):
        DsaNumericalContract(interleaved_rotary=False)
    with pytest.raises(ValueError, match="tie policy"):
        DsaNumericalContract(tie_policy="score_only")
    with pytest.raises(ValueError, match="sentinel"):
        DsaNumericalContract(padding_sentinel=0)


def test_query_key_and_score_math_matches_direct_fp32_formula() -> None:
    contract = small_contract()
    hidden_query = jnp.asarray(
        [[0.5, -1.0, 0.25, 0.75, -0.5, 1.25]], dtype=jnp.bfloat16
    )
    q_residual = jnp.asarray([[0.5, -0.25, 1.0, 0.75]], dtype=jnp.bfloat16)
    hidden_keys = jnp.asarray(
        [
            [0.5, 0.25, -0.5, 1.0, -1.0, 0.75],
            [-0.25, 1.0, 0.5, -0.75, 0.25, 1.25],
            [1.0, -0.5, 0.75, 0.25, 0.5, -1.0],
        ],
        dtype=jnp.bfloat16,
    )
    query_weight = (
        jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 31.0 - 0.5
    ).astype(jnp.bfloat16)
    head_weight = jnp.asarray(
        [[0.5, 0.25, -0.5, 1.0, 0.0, -0.25],
         [-0.5, 0.75, 0.25, 0.0, 1.0, 0.5]],
        dtype=jnp.bfloat16,
    )
    key_weight = (
        jnp.arange(24, dtype=jnp.float32).reshape(4, 6) / 23.0 - 0.5
    ).astype(jnp.bfloat16)
    norm_weight = jnp.asarray([1.0, 0.5, 1.5, -0.5], dtype=jnp.float32)
    norm_bias = jnp.asarray([0.0, 0.1, -0.2, 0.3], dtype=jnp.float32)
    query, weights = dsa_query_and_head_weights(
        hidden_query,
        q_residual,
        query_weight,
        head_weight,
        jnp.asarray([2], dtype=jnp.int32),
        contract=contract,
    )
    keys = dsa_index_keys(
        hidden_keys,
        key_weight,
        norm_weight,
        norm_bias,
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        contract=contract,
    )
    projected_keys = linear(
        hidden_keys, key_weight, output_dtype=jnp.float32
    )
    keys_from_projection = dsa_index_keys_from_projection(
        projected_keys,
        norm_weight,
        norm_bias,
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        contract=contract,
    )
    got = dsa_scores(query, keys, weights)
    expected_per_head = np.maximum(
        np.einsum("rhd,sd->rhs", np.asarray(query), np.asarray(keys))
        * contract.head_dim**-0.5,
        0.0,
    )
    expected = np.einsum("rh,rhs->rs", np.asarray(weights), expected_per_head)
    assert query.shape == (1, 2, 4)
    assert keys.shape == (3, 4)
    assert got.shape == (1, 3)
    assert got.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(keys_from_projection), np.asarray(keys)
    )
    np.testing.assert_allclose(np.asarray(got), expected, rtol=2e-7, atol=2e-7)


def test_dsa_scorer_pins_highest_dot_precision() -> None:
    query = jnp.ones((1, 2, 4), dtype=jnp.float32)
    keys = jnp.ones((3, 4), dtype=jnp.float32)
    weights = jnp.ones((1, 2), dtype=jnp.float32)
    hlo = jax.jit(dsa_scores).lower(query, keys, weights).compile().as_text()
    assert "operand_precision={highest,highest}" in hlo


def test_exact_topk_has_lowest_position_ties_and_minus_one_tail() -> None:
    scores = jnp.asarray(
        [[5.0, 5.0, 4.0, 3.0, 2.0], [9.0, 8.0, 7.0, 6.0, 5.0]],
        dtype=jnp.float32,
    )
    selected = exact_topk(
        scores, jnp.asarray([5, 2], dtype=jnp.int32), top_k=4
    )
    np.testing.assert_array_equal(
        np.asarray(selected.positions), [[0, 1, 2, 3], [0, 1, -1, -1]]
    )
    np.testing.assert_array_equal(np.asarray(selected.valid_counts), [4, 2])


def test_exact_topk_pads_context_shorter_than_selection_width() -> None:
    selected = exact_topk(
        jnp.asarray([[3.0, 1.0]], dtype=jnp.float32),
        jnp.asarray([2], dtype=jnp.int32),
        top_k=4,
    )
    np.testing.assert_array_equal(np.asarray(selected.positions), [[0, 1, -1, -1]])
    np.testing.assert_array_equal(np.asarray(selected.valid_counts), [2])
    empty = exact_topk(
        jnp.asarray([[3.0, 1.0]], dtype=jnp.float32),
        jnp.asarray([-1], dtype=jnp.int32),
        top_k=4,
    )
    np.testing.assert_array_equal(np.asarray(empty.positions), [[-1, -1, -1, -1]])
    np.testing.assert_array_equal(np.asarray(empty.valid_counts), [0])


def test_distributed_selection_matches_flat_including_ties_and_group_order() -> None:
    flat_scores = jnp.asarray(
        [[10.0, 10.0, 8.0, 8.0, 7.0, 7.0, 6.0, 6.0]],
        dtype=jnp.float32,
    )
    expected = exact_topk(
        flat_scores, jnp.asarray([8], dtype=jnp.int32), top_k=5
    )
    # Two striped context owners, deliberately presented in reverse owner order.
    shard_positions = jnp.asarray([[1, 3, 5, 7], [0, 2, 4, 6]], dtype=jnp.int32)
    shard_scores = jnp.stack(
        (flat_scores[:, [1, 3, 5, 7]], flat_scores[:, [0, 2, 4, 6]]), axis=0
    )
    got = distributed_exact_topk_reference(
        shard_scores,
        shard_positions,
        jnp.asarray([8], dtype=jnp.int32),
        top_k=5,
        global_context_size=8,
    )
    np.testing.assert_array_equal(np.asarray(got.positions), np.asarray(expected.positions))
    np.testing.assert_array_equal(
        np.asarray(got.valid_counts), np.asarray(expected.valid_counts)
    )


def test_local_candidate_width_keeps_hot_shard_winners() -> None:
    scores = jnp.asarray([[100.0, 99.0, 98.0, 97.0, 1.0]], dtype=jnp.float32)
    values, positions = local_topk_candidates(
        scores,
        jnp.asarray([0, 2, 4, 6, 8], dtype=jnp.int32),
        jnp.asarray([9], dtype=jnp.int32),
        top_k=4,
    )
    np.testing.assert_array_equal(np.asarray(positions), [[0, 2, 4, 6]])
    np.testing.assert_array_equal(np.asarray(values), [[100.0, 99.0, 98.0, 97.0]])


def test_local_ties_use_global_position_not_input_order() -> None:
    values, positions = local_topk_candidates(
        jnp.asarray([[5.0, 5.0, 5.0, 4.0]], dtype=jnp.float32),
        jnp.asarray([7, 1, 3, 5], dtype=jnp.int32),
        jnp.asarray([8], dtype=jnp.int32),
        top_k=2,
    )
    np.testing.assert_array_equal(np.asarray(positions), [[1, 3]])
    np.testing.assert_array_equal(np.asarray(values), [[5.0, 5.0]])


def test_merge_is_invariant_to_candidate_concatenation_order() -> None:
    scores = jnp.asarray(
        [[[5.0, 4.0, 3.0]], [[5.0, 4.0, 3.0]]], dtype=jnp.float32
    )
    positions = jnp.asarray(
        [[[1, 3, 5]], [[0, 2, 4]]], dtype=jnp.int32
    )
    valid = jnp.asarray([6], dtype=jnp.int32)
    first = merge_topk_candidates(
        scores, positions, valid, top_k=4, global_context_size=6
    )
    second = merge_topk_candidates(
        scores[::-1], positions[::-1], valid, top_k=4, global_context_size=6
    )
    np.testing.assert_array_equal(np.asarray(first.positions), [[0, 1, 2, 3]])
    np.testing.assert_array_equal(np.asarray(second.positions), np.asarray(first.positions))


def test_decode_reference_has_one_live_row_not_batch_32() -> None:
    selected = jax.jit(lambda value: exact_topk(value, jnp.asarray([7]), top_k=4))(
        jnp.arange(7, dtype=jnp.float32)[None, :]
    )
    assert selected.positions.shape == (1, 4)
    assert selected.valid_counts.shape == (1,)


def test_dsa_shape_contracts_fail_loudly() -> None:
    contract = small_contract()
    with pytest.raises(ValueError, match="query_weight"):
        dsa_query_and_head_weights(
            jnp.ones((1, 6)),
            jnp.ones((1, 4)),
            jnp.ones((7, 4)),
            jnp.ones((2, 6)),
            jnp.asarray([0]),
            contract=contract,
        )
    with pytest.raises(ValueError, match="ranks"):
        dsa_scores(jnp.ones((1, 4)), jnp.ones((3, 4)), jnp.ones((1, 2)))
    with pytest.raises(ValueError, match="positive"):
        exact_topk(jnp.ones((1, 4)), jnp.asarray([4]), top_k=0)


def test_contract_can_change_static_fixture_sizes_without_changing_semantics() -> None:
    contract = replace(small_contract(), top_k=2)
    assert contract.tie_policy == "descending_score_then_lowest_global_position"
    assert contract.interleaved_rotary is True
