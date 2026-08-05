from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference import (
    MlaNumericalContract,
    SelectedKvSegment,
    SelectedPositions,
    StageLocalKvLayout,
    canonicalize_selected_positions,
    gather_paged_selected_kv,
    gather_stage_local_selected_kv,
    selected_positions_for_owner,
    sparse_mla_attention,
    stage_local_sparse_mla_reference,
)


def small_contract() -> MlaNumericalContract:
    return MlaNumericalContract(
        num_heads=2,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        qk_head_dim=4,
        packed_cache_width=8,
        top_k=4,
    )


def small_layout() -> StageLocalKvLayout:
    return StageLocalKvLayout(
        logical_page_size=4,
        local_parallel_size=2,
        packed_cache_width=8,
    )


def paged_fixture() -> tuple[jax.Array, jax.Array, jax.Array]:
    """Eight logical tokens mapped to physical pages 2 then 0."""

    token_rows = np.arange(8 * 8, dtype=np.float32).reshape(8, 8) / 16.0
    cache = np.full((3, 4, 8), np.nan, dtype=np.float32)
    block_table = np.asarray([[2, 0]], dtype=np.int32)
    cache[2] = token_rows[:4]
    cache[0] = token_rows[4:]
    return jnp.asarray(cache), jnp.asarray(block_table), jnp.asarray(token_rows)


def local_cache_fixture() -> tuple[jax.Array, jax.Array, jax.Array]:
    global_cache, block_tables, token_rows = paged_fixture()
    layout = small_layout()
    local = np.full(
        (
            layout.local_parallel_size,
            global_cache.shape[0],
            layout.local_rows_per_page,
            layout.packed_cache_width,
        ),
        np.nan,
        dtype=np.float32,
    )
    table = np.asarray(block_tables)[0]
    rows = np.asarray(token_rows)
    for position in range(rows.shape[0]):
        logical_block = position // layout.logical_page_size
        within = position % layout.logical_page_size
        owner = within // layout.local_rows_per_page
        local_row = within % layout.local_rows_per_page
        local[owner, table[logical_block], local_row] = rows[position]
    return jnp.asarray(local), block_tables, token_rows


def test_glm_mla_contract_and_stage_local_layout_are_exact() -> None:
    contract = MlaNumericalContract()
    layout = StageLocalKvLayout()
    assert contract.num_heads == 64
    assert contract.kv_lora_rank == 512
    assert contract.qk_rope_head_dim == 64
    assert contract.qk_head_dim == 256
    assert contract.packed_cache_width == 640
    assert contract.top_k == 2048
    assert contract.softmax_scale == 256**-0.5
    assert layout.logical_page_size == 512
    assert layout.local_rows_per_page == 128


def test_attention_contract_refuses_layout_or_arithmetic_drift() -> None:
    with pytest.raises(ValueError, match="nope plus RoPE"):
        MlaNumericalContract(qk_head_dim=255)
    with pytest.raises(ValueError, match="truncates"):
        MlaNumericalContract(packed_cache_width=575)
    with pytest.raises(ValueError, match="softmax"):
        MlaNumericalContract(score_dtype="bfloat16")
    with pytest.raises(ValueError, match="divide"):
        StageLocalKvLayout(logical_page_size=5, local_parallel_size=2)


def test_canonical_attention_copy_preserves_state_and_sorts_live_prefix() -> None:
    selected = SelectedPositions(
        jnp.asarray([[6, 1, 4, -1]], dtype=jnp.int32),
        jnp.asarray([3], dtype=jnp.int32),
    )
    canonical = canonicalize_selected_positions(selected)
    np.testing.assert_array_equal(np.asarray(selected.positions), [[6, 1, 4, -1]])
    np.testing.assert_array_equal(
        np.asarray(canonical.selection.positions), [[1, 4, 6, -1]]
    )
    np.testing.assert_array_equal(np.asarray(canonical.contract_valid), [True])


def test_paged_gather_maps_absolute_positions_and_zeros_tail() -> None:
    cache, block_tables, token_rows = paged_fixture()
    selected = SelectedPositions(
        jnp.asarray([[6, 1, 4, -1]], dtype=jnp.int32),
        jnp.asarray([3], dtype=jnp.int32),
    )
    segment = gather_paged_selected_kv(
        cache,
        block_tables,
        selected,
        jnp.asarray([8], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(np.asarray(segment.positions), [[1, 4, 6, -1]])
    np.testing.assert_array_equal(
        np.asarray(segment.values[0, :3]), np.asarray(token_rows)[[1, 4, 6]]
    )
    np.testing.assert_array_equal(np.asarray(segment.values[0, 3]), np.zeros(8))
    np.testing.assert_array_equal(np.asarray(segment.contract_valid), [True])


def test_gather_health_catches_malformed_or_noncausal_state_without_oob_read() -> None:
    cache, block_tables, _ = paged_fixture()
    malformed = SelectedPositions(
        jnp.asarray([[1, -1, 4, -1]], dtype=jnp.int32),
        jnp.asarray([2], dtype=jnp.int32),
    )
    bad_tail = gather_paged_selected_kv(
        cache,
        block_tables,
        malformed,
        jnp.asarray([8], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(np.asarray(bad_tail.contract_valid), [False])
    stale = gather_paged_selected_kv(
        cache,
        block_tables,
        SelectedPositions(
            jnp.asarray([[6, -1, -1, -1]], dtype=jnp.int32),
            jnp.asarray([1], dtype=jnp.int32),
        ),
        jnp.asarray([4], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(np.asarray(stale.contract_valid), [False])
    np.testing.assert_array_equal(np.asarray(stale.values), np.zeros((1, 4, 8)))
    for invalid_length in (-1, 9):
        invalid = gather_paged_selected_kv(
            cache,
            block_tables,
            SelectedPositions(
                jnp.full((1, 4), -1, dtype=jnp.int32),
                jnp.asarray([0], dtype=jnp.int32),
            ),
            jnp.asarray([invalid_length], dtype=jnp.int32),
        )
        np.testing.assert_array_equal(
            np.asarray(invalid.contract_valid), [False]
        )


def test_owner_subsets_are_disjoint_and_local_gather_matches_logical_rows() -> None:
    local_caches, block_tables, token_rows = local_cache_fixture()
    layout = small_layout()
    selected = SelectedPositions(
        jnp.asarray([[7, 2, 0, 5]], dtype=jnp.int32),
        jnp.asarray([4], dtype=jnp.int32),
    )
    subsets = [
        selected_positions_for_owner(selected, layout=layout, owner_index=owner)
        for owner in range(layout.local_parallel_size)
    ]
    live = []
    for subset in subsets:
        row = np.asarray(subset.selection.positions[0])
        live.extend(row[row >= 0].tolist())
    assert sorted(live) == [0, 2, 5, 7]
    assert len(live) == len(set(live))

    for owner, subset in enumerate(subsets):
        segment = gather_stage_local_selected_kv(
            local_caches[owner],
            block_tables,
            selected,
            jnp.asarray([8], dtype=jnp.int32),
            layout=layout,
            owner_index=owner,
        )
        count = int(np.asarray(subset.selection.valid_counts[0]))
        positions = np.asarray(segment.positions[0, :count])
        np.testing.assert_array_equal(
            np.asarray(segment.values[0, :count]), np.asarray(token_rows)[positions]
        )
        np.testing.assert_array_equal(np.asarray(segment.contract_valid), [True])


def test_sparse_mla_matches_direct_fp32_selected_attention() -> None:
    contract = small_contract()
    values = jnp.asarray(
        [
            [
                [0.5, -0.25, 1.0, 0.75, 0.1, 0.2, 99.0, -99.0],
                [1.0, 0.5, -0.5, 0.25, -0.2, 0.3, 88.0, -88.0],
                [-0.5, 0.75, 0.25, 1.0, 0.4, -0.1, 77.0, -77.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=jnp.float32,
    )
    segment = SelectedKvSegment(
        values,
        jnp.asarray([[0, 2, 5, -1]], dtype=jnp.int32),
        jnp.asarray([3], dtype=jnp.int32),
        jnp.asarray([True]),
    )
    q_nope = jnp.asarray(
        [[[0.5, 0.25, -0.5, 1.0], [-0.25, 1.0, 0.5, 0.75]]],
        dtype=jnp.float32,
    )
    q_rope = jnp.asarray([[[0.2, -0.1], [0.5, 0.25]]], dtype=jnp.float32)
    got = sparse_mla_attention(q_nope, q_rope, segment, contract=contract)

    v = np.asarray(values)[0, :3, :4]
    k_rope = np.asarray(values)[0, :3, 4:6]
    scores = (
        np.einsum("hd,kd->hk", np.asarray(q_nope)[0], v)
        + np.einsum("hd,kd->hk", np.asarray(q_rope)[0], k_rope)
    ) * contract.softmax_scale
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = np.einsum("hk,kd->hd", probabilities, v)
    np.testing.assert_allclose(
        np.asarray(got.output[0]), expected, rtol=2e-7, atol=2e-7
    )
    np.testing.assert_allclose(
        np.asarray(got.logsumexp[0]),
        np.log(np.exp(scores).sum(axis=-1)),
        rtol=2e-7,
        atol=2e-7,
    )


def test_empty_selected_row_is_zero_even_when_garbage_contains_nan() -> None:
    contract = small_contract()
    segment = SelectedKvSegment(
        jnp.full((1, 4, 8), jnp.nan, dtype=jnp.float32),
        jnp.full((1, 4), -1, dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([True]),
    )
    got = sparse_mla_attention(
        jnp.ones((1, 2, 4), dtype=jnp.float32),
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        segment,
        contract=contract,
    )
    np.testing.assert_array_equal(np.asarray(got.output), np.zeros((1, 2, 4)))
    assert np.all(np.isneginf(np.asarray(got.logsumexp)))


def test_bfloat16_attention_rounds_unnormalized_weights_before_pv() -> None:
    contract = small_contract()
    values = jnp.asarray(
        np.arange(32, dtype=np.float32).reshape(1, 4, 8) / 13.0,
        dtype=jnp.bfloat16,
    )
    segment = SelectedKvSegment(
        values,
        jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32),
        jnp.asarray([4], dtype=jnp.int32),
        jnp.asarray([True]),
    )
    q_nope = jnp.asarray(
        [[[0.25, -0.5, 0.75, 1.0], [1.25, 0.5, -0.25, 0.75]]],
        dtype=jnp.bfloat16,
    )
    q_rope = jnp.asarray(
        [[[0.5, -0.25], [-0.75, 0.5]]], dtype=jnp.bfloat16
    )
    got = sparse_mla_attention(q_nope, q_rope, segment, contract=contract)

    scores = (
        jnp.einsum(
            "rhd,rkd->rhk",
            q_nope,
            values[..., :4],
            preferred_element_type=jnp.float32,
        )
        + jnp.einsum(
            "rhd,rkd->rhk",
            q_rope,
            values[..., 4:6],
            preferred_element_type=jnp.float32,
        )
    ) * jnp.float32(contract.softmax_scale)
    maximum = jnp.max(scores, axis=-1, keepdims=True)
    unnormalized = jnp.exp(scores - maximum)
    denominator = jnp.sum(unnormalized, axis=-1, keepdims=True)
    expected = jnp.einsum(
        "rhk,rkd->rhd",
        unnormalized.astype(jnp.bfloat16),
        values[..., :4],
        preferred_element_type=jnp.float32,
    ) / denominator
    expected = expected.astype(jnp.bfloat16)
    normalized_first = jnp.einsum(
        "rhk,rkd->rhd",
        (unnormalized / denominator).astype(jnp.bfloat16),
        values[..., :4],
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)

    assert got.output.dtype == jnp.bfloat16
    assert got.logsumexp.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(got.output), np.asarray(expected))
    assert np.any(np.asarray(got.output) != np.asarray(normalized_first))


def test_stage_local_lse_merge_matches_unsharded_selected_attention() -> None:
    contract = small_contract()
    layout = small_layout()
    cache, block_tables, _ = paged_fixture()
    local_caches, _, _ = local_cache_fixture()
    selected = SelectedPositions(
        jnp.asarray([[7, 2, 0, 5]], dtype=jnp.int32),
        jnp.asarray([4], dtype=jnp.int32),
    )
    lengths = jnp.asarray([8], dtype=jnp.int32)
    q_nope = jnp.asarray(
        [[[0.5, 0.25, -0.5, 1.0], [-0.25, 1.0, 0.5, 0.75]]],
        dtype=jnp.float32,
    )
    q_rope = jnp.asarray([[[0.2, -0.1], [0.5, 0.25]]], dtype=jnp.float32)
    unsharded = sparse_mla_attention(
        q_nope,
        q_rope,
        gather_paged_selected_kv(cache, block_tables, selected, lengths),
        contract=contract,
    )
    stage_local = stage_local_sparse_mla_reference(
        q_nope,
        q_rope,
        local_caches,
        block_tables,
        selected,
        lengths,
        layout=layout,
        contract=contract,
    )
    np.testing.assert_allclose(
        np.asarray(stage_local.output),
        np.asarray(unsharded.output),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.asarray(stage_local.logsumexp),
        np.asarray(unsharded.logsumexp),
        rtol=2e-6,
        atol=2e-6,
    )
    np.testing.assert_array_equal(np.asarray(stage_local.contract_valid), [True])


def test_decode_reference_has_one_row_and_no_collective_or_dead_bucket() -> None:
    contract = small_contract()
    layout = small_layout()
    local_caches, block_tables, _ = local_cache_fixture()
    selected = SelectedPositions(
        jnp.asarray([[7, 2, 0, 5]], dtype=jnp.int32),
        jnp.asarray([4], dtype=jnp.int32),
    )

    def run(q_nope: jax.Array, q_rope: jax.Array) -> jax.Array:
        return stage_local_sparse_mla_reference(
            q_nope,
            q_rope,
            local_caches,
            block_tables,
            selected,
            jnp.asarray([8], dtype=jnp.int32),
            layout=layout,
            contract=contract,
        ).output

    q_nope = jnp.ones((1, 2, 4), dtype=jnp.float32)
    q_rope = jnp.ones((1, 2, 2), dtype=jnp.float32)
    compiled = jax.jit(run).lower(q_nope, q_rope).compile()
    assert compiled(q_nope, q_rope).shape == (1, 2, 4)
    hlo = compiled.as_text().lower()
    assert "all-reduce" not in hlo
    assert "[32," not in hlo


def test_attention_shape_and_dtype_contracts_fail_loudly() -> None:
    contract = small_contract()
    segment = SelectedKvSegment(
        jnp.zeros((1, 4, 8), dtype=jnp.bfloat16),
        jnp.full((1, 4), -1, dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([True]),
    )
    with pytest.raises(ValueError, match="share the cache dtype"):
        sparse_mla_attention(
            jnp.ones((1, 2, 4), dtype=jnp.float32),
            jnp.ones((1, 2, 2), dtype=jnp.float32),
            segment,
            contract=contract,
        )
    malformed = SelectedKvSegment(
        jnp.zeros((1, 4, 8), dtype=jnp.float32),
        jnp.asarray([[1, 0, -1, -1]], dtype=jnp.int32),
        jnp.asarray([2], dtype=jnp.int32),
        jnp.asarray([True]),
    )
    malformed_result = sparse_mla_attention(
        jnp.ones((1, 2, 4), dtype=jnp.float32),
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        malformed,
        contract=contract,
    )
    np.testing.assert_array_equal(
        np.asarray(malformed_result.contract_valid), [False]
    )
    with pytest.raises(ValueError, match="one row"):
        stage_local_sparse_mla_reference(
            jnp.ones((2, 2, 4), dtype=jnp.float32),
            jnp.ones((2, 2, 2), dtype=jnp.float32),
            jnp.ones((2, 3, 2, 8), dtype=jnp.float32),
            jnp.ones((2, 2), dtype=jnp.int32),
            SelectedPositions(
                jnp.full((2, 4), -1, dtype=jnp.int32),
                jnp.zeros((2,), dtype=jnp.int32),
            ),
            jnp.zeros((2,), dtype=jnp.int32),
            layout=small_layout(),
            contract=contract,
        )
