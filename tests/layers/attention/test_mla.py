"""CPU multirow attention primitives; not TPU arithmetic or speed proof.

The prefill attention block (``prefill_index_share_lse``) writes its rows' keys once and attends each query
within its own causal bound: every row equals the block computed with only the rows up to it live, a later row
never reaches an earlier one, padding and empty blocks are exact no-ops, and malformed selections, metadata or
non-finite operands fail health; a non-finite key to be written refuses the whole block's write (ported from the
research package's ``kernels/test_ws32_prefill_attention.py``, ``archive/research-20260922``; on 32 forced CPU
devices with the frozen fixture's first layer).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import jax

from glm_tpu.kernels.sparse_mla.kernel import pregathered_sparse_mla_pallas, SparseMlaConfig, sparse_mla_attention
from glm_tpu.layers.contracts import MlaNumericalContract, StageLocalKvLayout, SelectedPositions
from glm_tpu.layers.attention.kv_cache import (
    SelectedKvSegment,
    canonicalize_selected_positions,
    gather_stage_local_selected_kv,
    gather_stage_local_selected_kv_aligned,
    selected_positions_for_owner,
)
from tests.reference.attention import gather_paged_selected_kv, stage_local_sparse_mla_reference


def test_multirow_sparse_attention_counts_and_scratch_do_not_leak_between_queries():
    rng = np.random.default_rng(443)
    rows, heads, latent, rope, topk = 17, 4, 8, 4, 16
    contract = MlaNumericalContract(
        num_heads=heads,
        kv_lora_rank=latent,
        qk_nope_head_dim=4,
        qk_rope_head_dim=rope,
        qk_head_dim=8,
        v_head_dim=4,
        packed_cache_width=16,
        top_k=topk,
    )
    config = SparseMlaConfig(segment_block=4)
    q = jnp.asarray(rng.normal(size=(rows, heads, latent)), jnp.bfloat16)
    r = jnp.asarray(rng.normal(size=(rows, heads, rope)), jnp.bfloat16)
    cache = jnp.asarray(rng.normal(size=(rows, topk, 16)), jnp.bfloat16)
    counts = jnp.asarray([0, 1, 3, 4, 5, 15, 16, 0, 16, 1, 0, 9, 3, 0, 16, 7, 0], jnp.int32)
    cache = jnp.where(jnp.arange(topk)[None, :, None] < counts[:, None, None], cache, 0)

    def run(q, r, c, n, prefill):
        return pregathered_sparse_mla_pallas(
            q,
            r,
            c,
            n,
            contract=contract,
            config=config,
            prefill=prefill,
            interpret=True,
        )

    actual = run(q, r, cache, counts, True)
    expected = jnp.concatenate(
        [run(q[i : i + 1], r[i : i + 1], cache[i : i + 1], counts[i : i + 1], False) for i in range(rows)]
    )
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint16), np.asarray(expected).view(np.uint16))
    assert np.all(np.asarray(actual)[np.asarray(counts) == 0] == 0)
    order = jnp.asarray(rng.permutation(rows))
    permuted = run(q[order], r[order], cache[order], counts[order], True)
    np.testing.assert_array_equal(np.asarray(permuted), np.asarray(actual[order]))
    with pytest.raises(ValueError, match="prefill=True"):
        run(q, r, cache, counts, False)


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
    np.testing.assert_array_equal(np.asarray(canonical.selection.positions), [[1, 4, 6, -1]])
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
    np.testing.assert_array_equal(np.asarray(segment.values[0, :3]), np.asarray(token_rows)[[1, 4, 6]])
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
        np.testing.assert_array_equal(np.asarray(invalid.contract_valid), [False])


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
        np.testing.assert_array_equal(np.asarray(segment.values[0, :count]), np.asarray(token_rows)[positions])
        np.testing.assert_array_equal(np.asarray(segment.contract_valid), [True])


def test_owner_subset_accepts_jitted_scalar_stage_index() -> None:
    layout = small_layout()
    selected = SelectedPositions(
        jnp.asarray([[7, 2, 0, 5]], dtype=jnp.int32),
        jnp.asarray([4], dtype=jnp.int32),
    )
    mapped = jax.jit(
        lambda owner: (
            selected_positions_for_owner(
                selected,
                layout=layout,
                owner_index=owner,
            ).selection.positions
        )
    )
    np.testing.assert_array_equal(
        np.asarray(mapped(jnp.asarray(1, dtype=jnp.int32))),
        np.asarray([[2, 7, -1, -1]], dtype=np.int32),
    )


def test_aligned_owner_segments_sum_to_exact_global_selected_segment() -> None:
    local_caches, block_tables, _ = local_cache_fixture()
    layout = small_layout()
    selected = SelectedPositions(
        jnp.asarray([[7, 2, 0, 5]], dtype=jnp.int32),
        jnp.asarray([4], dtype=jnp.int32),
    )
    expected = gather_paged_selected_kv(
        jnp.concatenate(tuple(local_caches), axis=1),
        block_tables,
        selected,
        jnp.asarray([8], dtype=jnp.int32),
    )
    aligned = [
        gather_stage_local_selected_kv_aligned(
            local_caches[owner],
            block_tables,
            selected,
            jnp.asarray([8], dtype=jnp.int32),
            layout=layout,
            owner_index=owner,
        )
        for owner in range(layout.local_parallel_size)
    ]
    np.testing.assert_array_equal(
        np.asarray(sum((item.values for item in aligned), jnp.zeros_like(aligned[0].values))),
        np.asarray(expected.values),
    )
    for item in aligned:
        np.testing.assert_array_equal(item.positions, expected.positions)
        np.testing.assert_array_equal(item.valid_counts, expected.valid_counts)
        np.testing.assert_array_equal(item.contract_valid, expected.contract_valid)


def test_aligned_owner_segment_zeros_unowned_rows_and_propagates_health() -> None:
    local_caches, block_tables, _ = local_cache_fixture()
    layout = small_layout()
    selected = SelectedPositions(
        jnp.asarray([[7, 2, 0, 5]], dtype=jnp.int32),
        jnp.asarray([4], dtype=jnp.int32),
    )
    segment = gather_stage_local_selected_kv_aligned(
        local_caches[0],
        block_tables,
        selected,
        jnp.asarray([8], dtype=jnp.int32),
        layout=layout,
        owner_index=jnp.asarray(0, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(segment.positions, [[0, 2, 5, 7]])
    np.testing.assert_array_equal(np.asarray(segment.values[0, [1, 3]]), 0)
    assert np.all(np.any(np.asarray(segment.values[0, [0, 2]]) != 0, axis=1))
    np.testing.assert_array_equal(segment.contract_valid, [True])
    invalid = gather_stage_local_selected_kv_aligned(
        local_caches[0],
        block_tables,
        selected,
        jnp.asarray([5], dtype=jnp.int32),
        layout=layout,
        owner_index=0,
    )
    np.testing.assert_array_equal(invalid.contract_valid, [False])


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
        np.einsum("hd,kd->hk", np.asarray(q_nope)[0], v) + np.einsum("hd,kd->hk", np.asarray(q_rope)[0], k_rope)
    ) * contract.softmax_scale
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = np.einsum("hk,kd->hd", probabilities, v)
    np.testing.assert_allclose(np.asarray(got.output[0]), expected, rtol=2e-7, atol=2e-7)
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
    q_rope = jnp.asarray([[[0.5, -0.25], [-0.75, 0.5]]], dtype=jnp.bfloat16)
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
    expected = (
        jnp.einsum(
            "rhk,rkd->rhd",
            unnormalized.astype(jnp.bfloat16),
            values[..., :4],
            preferred_element_type=jnp.float32,
        )
        / denominator
    )
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
    np.testing.assert_array_equal(np.asarray(malformed_result.contract_valid), [False])
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


PREFILL_ATTENTION = r"""
import json
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from jax._src.pallas.mosaic import tpu_info
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
from glm_tpu.kernels.sparse_mla.kernel import SparseMlaConfig
from glm_tpu.layers.attention.mla import PreparedAttention, prefill_index_share_lse, prefill_prepare_attention
from glm_tpu.models.glm_moe_dsa.weights import bf16_weight_specs
from glm_tpu.runner.hlo_utils import parse_hlo_module
from tests.fixtures.tiny_model import cpu_mesh, engine_inputs
mesh = cpu_mesh()
inputs = engine_inputs(mesh, panel_geometry=True)
config = inputs.config
contract = config.attention_contract
layer, spec = inputs.weights.layers[0], bf16_weight_specs(config).layers[0]
rng = np.random.default_rng(915)
R = 17


def bf(shape, scale=0.1):
    return np.asarray(jnp.asarray(rng.normal(0, scale, shape), jnp.bfloat16))


def put(value, sharding=P()):
    return jax.device_put(value, NamedSharding(mesh, sharding))


def bits(value):
    return np.asarray(value).tobytes()


residual = put(bf((R, config.geometry.hidden_size)), P(None, 'feature'))
prepared = PreparedAttention(residual, residual, put(bf((R, config.geometry.q_lora_rank))),
                             put(bf((R, contract.kv_lora_rank + contract.qk_rope_head_dim))))
prep_spec = PreparedAttention(P(None, 'feature'), P(None, 'feature'), P(), P())
cache = put(bf((config.page_count, config.logical_page_size, contract.packed_cache_width)), P(None, 'expert', None))
table = put(np.asarray([[2, 0, 1]], np.int32))
angles = rng.normal(size=(R, contract.qk_rope_head_dim // 2))
rope = put(np.asarray(jnp.asarray(np.concatenate((np.cos(angles), np.sin(angles)), axis=1), jnp.bfloat16)))
kwargs = dict(contract=contract, cache_layout=config.cache_layout, linear_interpret=True,
              sparse_attention_config=SparseMlaConfig(segment_block=config.sparse_segment_block),
              sparse_attention_interpret=True)


def body(r, p, c, s, n, offset, count, t, w, h):
    out = prefill_index_share_lse(r, p, c, s, n, offset, count, t, w, main_rope_table_rows=h, **kwargs)
    return out.output_local, out.cache_local, out.contract_valid[None, None]


block = jax.jit(jax.shard_map(
    body, mesh=mesh,
    in_specs=(P(None, 'feature'), prep_spec, P(None, 'expert', None), P(), P(), P(), P(), P(), spec.attention, P()),
    out_specs=(P(None, 'feature'), P(None, 'expert', None), P('expert', 'feature', None)), check_vma=False))


def selections(offset, count):
    positions, counts = np.full((R, contract.top_k), -1, np.int32), np.zeros(R, np.int32)
    for i in range(count):
        end = offset + i + 1
        counts[i] = min(end, contract.top_k)
        positions[i, : counts[i]] = np.arange(end - counts[i], end)  # ascending: the consumer canonicalizes
    return put(positions), put(counts)


def run(offset, count, p=prepared, h=rope, s=None, n=None, c=cache, r=residual):
    if s is None:
        s, n = selections(offset, count)
    return jax.block_until_ready(block(r, p, c, s, n, put(np.int32(offset)), put(np.int32(count)), table,
                                       layer.attention, h))


def with_rows(offset, count, written):
    # the input cache with the rows of positions offset..offset+count-1 taken from ``written``
    expected, pages = np.asarray(cache).copy(), np.asarray(table)[0]
    for position in range(offset, offset + count):
        expected[pages[position // 512], position % 512] = written[pages[position // 512], position % 512]
    return expected


report = {}
for offset, count in ((55, 17), (505, 17), (0, 11)):
    out, end, health = (np.asarray(v) for v in run(offset, count))
    assert health.all() and not out[count:].any(), (offset, count)
    assert bits(end) == bits(with_rows(offset, count, end)), (offset, count)  # only the live rows' positions
    # each row equals the block computed with only the rows up to it live: its causal prefix, same shape
    for i in range(count):
        o, e, h = (np.asarray(v) for v in run(offset, i + 1))
        assert h.all() and bits(o[: i + 1]) == bits(out[: i + 1]), (offset, i)
        assert bits(e) == bits(with_rows(offset, i + 1, end)), (offset, i)
    report[f'{offset}+{count}'] = 'causal'
base = run(55, 17)
# a later row's query and key never reach an earlier row
changed = prepared._replace(q_residual=prepared.q_residual.at[-1].set(2), current_kv=prepared.current_kv.at[-1].set(3))
future = run(55, 17, p=changed)
assert bits(np.asarray(future[0])[:-1]) == bits(np.asarray(base[0])[:-1])
assert bits(np.asarray(future[0])[-1]) != bits(np.asarray(base[0])[-1])
# padded garbage is ignored; a zero-live block is a healthy exact cache no-op with zero output
poisoned = run(0, 11, p=jax.tree.map(lambda v: v.at[11:].set(jnp.nan), prepared), h=rope.at[11:].set(jnp.nan),
               r=residual.at[11:].set(jnp.nan))
assert all(bits(a) == bits(b) for a, b in zip(poisoned, run(0, 11)))
empty = run(0, 0, p=jax.tree.map(lambda v: jnp.full_like(v, jnp.nan), prepared))
assert np.asarray(empty[2]).all() and not np.asarray(empty[0]).any() and bits(empty[1]) == bits(cache)
# a future selection (its key is physically present in the block) or a short count: row 0 unhealthy
s, n = selections(55, 17)
for label, result in (('future', run(55, 17, s=s.at[0, 55].set(56), n=n)), ('count', run(55, 17, s=s, n=n.at[0].set(55)))):
    assert not np.asarray(result[2])[:, :, 0].any(), label
# a bad offset, a NaN live key or a NaN live rotary row (both reach the written key): the whole block
# unhealthy and the cache unchanged
bad_key = prepared._replace(current_kv=prepared.current_kv.at[0, 0].set(jnp.nan))
for label, result in (('offset', run(2147483647, 17, s=s, n=n)), ('key', run(55, 17, p=bad_key)),
                      ('rope', run(55, 17, h=rope.at[0, 0].set(jnp.nan)))):
    assert not np.asarray(result[2]).any() and bits(result[1]) == bits(cache), label
# a NaN query: that row unhealthy, the others not; a NaN in an old cache row every row selects (position 40:
# physical page 2): every row unhealthy
bad_query = prepared._replace(q_residual=prepared.q_residual.at[0, 0].set(jnp.nan))
health = np.asarray(run(55, 17, p=bad_query)[2])
assert not health[:, :, 0].any() and health[:, :, 1:].all()
assert not np.asarray(run(55, 17, c=cache.at[2, 40, 0].set(jnp.nan))[2]).any()
# the block's only collectives are the expert-8 attention exchanges
compiled = block.lower(residual, prepared, cache, s, n, put(np.int32(55)), put(np.int32(17)), table, layer.attention,
                       rope).compile()
collectives = parse_hlo_module(compiled.as_text()).collectives
expert = tuple(tuple(e * 4 + f for e in range(8)) for f in range(4))
assert collectives and all(c.replica_groups == expert for c in collectives), {c.replica_groups for c in collectives}
report['collectives'] = sorted({c.opcode for c in collectives})
# preparation has no cross-row arithmetic: changing one row leaves every other row bitwise unchanged
prepare = jax.jit(jax.shard_map(lambda r, w: prefill_prepare_attention(r, w, linear_interpret=True), mesh=mesh,
                                in_specs=(P(None, 'feature'), spec.qkv_a), out_specs=prep_spec, check_vma=False))
for x, y in zip(prepare(residual, layer.qkv_a), prepare(residual.at[5].set(3), layer.qkv_a)):
    x, y = np.asarray(x), np.asarray(y)
    assert bits(np.delete(x, 5, axis=0)) == bits(np.delete(y, 5, axis=0)) and bits(x[5]) != bits(y[5])
print(json.dumps(report))
"""  # noqa: E501 (child program text)


@pytest.mark.cpu32
def test_prefill_attention_is_causal_within_its_block_cpu32() -> None:
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=32")
    result = subprocess.run(
        [sys.executable, "-c", PREFILL_ATTENTION], env=env, capture_output=True, text=True, timeout=900
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    blocks = ("55+17", "505+17", "0+11")
    assert {key: report[key] for key in blocks} == dict.fromkeys(blocks, "causal")
