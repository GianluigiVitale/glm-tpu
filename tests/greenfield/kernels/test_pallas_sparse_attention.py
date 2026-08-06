from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.pallas import (
    SparseMlaConfig,
    stage_local_sparse_mla_kernel,
    stage_local_sparse_mla_pallas,
)
from glm_tpu.greenfield.kernels.reference.attention import (
    MlaNumericalContract,
    StageLocalKvLayout,
    combine_stage_local_attention,
    stage_local_sparse_mla_reference,
)
from glm_tpu.greenfield.kernels.reference.dsa import SelectedPositions


LAYOUT = StageLocalKvLayout(
    logical_page_size=8,
    local_parallel_size=4,
    packed_cache_width=8,
)
CONTRACT = MlaNumericalContract(
    num_heads=4,
    kv_lora_rank=4,
    qk_nope_head_dim=2,
    qk_rope_head_dim=2,
    qk_head_dim=4,
    v_head_dim=2,
    packed_cache_width=8,
    top_k=8,
)
CONFIG = SparseMlaConfig(segment_block=4)


def _fixture(dtype: jnp.dtype) -> tuple[jnp.ndarray, ...]:
    rng = np.random.default_rng(771)
    global_cache = rng.normal(size=(8, 8, 8)).astype(np.float32)
    local_caches = np.stack(
        [
            global_cache[:, owner * 2 : (owner + 1) * 2, :]
            for owner in range(4)
        ]
    )
    query_nope = jnp.asarray(rng.normal(size=(1, 4, 4)), dtype)
    query_rope = jnp.asarray(rng.normal(size=(1, 4, 2)), dtype)
    block_tables = jnp.asarray([[6, 1, 7, 2]], dtype=jnp.int32)
    positions = jnp.asarray(
        [[27, 2, 9, 7, 18, 30, 15, 1]], dtype=jnp.int32
    )
    counts = jnp.asarray([8], dtype=jnp.int32)
    lengths = jnp.asarray([31], dtype=jnp.int32)
    return (
        query_nope,
        query_rope,
        jnp.asarray(local_caches, dtype),
        block_tables,
        positions,
        counts,
        lengths,
    )


def _assert_lse_close(actual: jnp.ndarray, expected: jnp.ndarray) -> None:
    actual_np = np.asarray(actual)
    expected_np = np.asarray(expected)
    np.testing.assert_array_equal(np.isneginf(actual_np), np.isneginf(expected_np))
    finite = np.isfinite(expected_np)
    np.testing.assert_allclose(
        actual_np[finite], expected_np[finite], rtol=0, atol=2e-5
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_fused_selected_kv_attention_interpret_matches_stage_reference(
    dtype: jnp.dtype,
) -> None:
    q_nope, q_rope, caches, tables, positions, counts, lengths = _fixture(dtype)
    selected = SelectedPositions(positions, counts)
    expected = stage_local_sparse_mla_reference(
        q_nope,
        q_rope,
        caches,
        tables,
        selected,
        lengths,
        layout=LAYOUT,
        contract=CONTRACT,
    )
    partial_outputs = []
    partial_lse = []
    partial_valid = []
    for owner in range(LAYOUT.local_parallel_size):
        reference = stage_local_sparse_mla_kernel(
            q_nope,
            q_rope,
            caches[owner],
            tables,
            selected,
            lengths,
            layout=LAYOUT,
            owner_index=owner,
            contract=CONTRACT,
        )
        actual = stage_local_sparse_mla_pallas(
            q_nope,
            q_rope,
            caches[owner],
            tables,
            selected,
            lengths,
            layout=LAYOUT,
            owner_index=jnp.asarray(owner, dtype=jnp.int32),
            contract=CONTRACT,
            config=CONFIG,
            interpret=True,
        )
        np.testing.assert_allclose(
            np.asarray(actual.output, dtype=np.float32),
            np.asarray(reference.output, dtype=np.float32),
            rtol=0,
            atol=2.5e-3 if dtype == jnp.bfloat16 else 2e-6,
        )
        _assert_lse_close(actual.logsumexp, reference.logsumexp)
        np.testing.assert_array_equal(
            np.asarray(actual.contract_valid), np.asarray(reference.contract_valid)
        )
        partial_outputs.append(actual.output)
        partial_lse.append(actual.logsumexp)
        partial_valid.append(actual.contract_valid)

    combined = combine_stage_local_attention(
        jnp.stack(partial_outputs),
        jnp.stack(partial_lse),
        jnp.stack(partial_valid),
    )
    np.testing.assert_allclose(
        np.asarray(combined.output, dtype=np.float32),
        np.asarray(expected.output, dtype=np.float32),
        rtol=0,
        atol=4e-3 if dtype == jnp.bfloat16 else 3e-6,
    )
    _assert_lse_close(combined.logsumexp, expected.logsumexp)
    np.testing.assert_array_equal(np.asarray(combined.contract_valid), [True])


def test_fused_attention_zeros_dma_tail_before_poison_can_propagate() -> None:
    q_nope, q_rope, caches, tables, positions, counts, lengths = _fixture(
        jnp.float32
    )
    # Owner zero has two live rows, so its four-row DMA tile fetches safe row
    # zero for two tail lanes. Poison that row to prove the lanes are zeroed
    # before QK and PV instead of relying on 0*NaN behavior.
    cache = caches[0].at[0, 0, :].set(jnp.nan)
    selected = SelectedPositions(positions, counts)
    actual = stage_local_sparse_mla_pallas(
        q_nope,
        q_rope,
        cache,
        tables,
        selected,
        lengths,
        layout=LAYOUT,
        owner_index=0,
        contract=CONTRACT,
        config=CONFIG,
        interpret=True,
    )
    assert bool(jnp.all(jnp.isfinite(actual.output)))
    assert bool(jnp.all(jnp.isfinite(actual.logsumexp)))
    assert bool(jnp.all(actual.contract_valid))


def test_fused_attention_rejects_duplicate_and_bad_page_metadata_in_health() -> None:
    q_nope, q_rope, caches, tables, positions, counts, lengths = _fixture(
        jnp.float32
    )
    duplicate = positions.at[0, 1].set(positions[0, 0])
    duplicate_result = stage_local_sparse_mla_pallas(
        q_nope,
        q_rope,
        caches[1],
        tables,
        SelectedPositions(duplicate, counts),
        lengths,
        layout=LAYOUT,
        owner_index=1,
        contract=CONTRACT,
        config=CONFIG,
        interpret=True,
    )
    assert not bool(jnp.all(duplicate_result.contract_valid))

    bad_tables = tables.at[0, 0].set(99)
    bad_page_result = stage_local_sparse_mla_pallas(
        q_nope,
        q_rope,
        caches[1],
        bad_tables,
        SelectedPositions(positions, counts),
        lengths,
        layout=LAYOUT,
        owner_index=1,
        contract=CONTRACT,
        config=CONFIG,
        interpret=True,
    )
    assert not bool(jnp.all(bad_page_result.contract_valid))
    assert bool(jnp.all(jnp.isfinite(bad_page_result.output)))


def test_sparse_attention_dispatch_is_default_off_and_refuses_unknown_backend() -> None:
    q_nope, q_rope, caches, tables, positions, counts, lengths = _fixture(
        jnp.float32
    )
    selected = SelectedPositions(positions, counts)
    expected = stage_local_sparse_mla_kernel(
        q_nope,
        q_rope,
        caches[3],
        tables,
        selected,
        lengths,
        layout=LAYOUT,
        owner_index=3,
        contract=CONTRACT,
        backend="reference",
    )
    actual = stage_local_sparse_mla_kernel(
        q_nope,
        q_rope,
        caches[3],
        tables,
        selected,
        lengths,
        layout=LAYOUT,
        owner_index=3,
        contract=CONTRACT,
    )
    np.testing.assert_array_equal(np.asarray(actual.output), np.asarray(expected.output))
    with pytest.raises(ValueError, match="unsupported sparse-MLA backend"):
        stage_local_sparse_mla_kernel(
            q_nope,
            q_rope,
            caches[3],
            tables,
            selected,
            lengths,
            layout=LAYOUT,
            owner_index=3,
            contract=CONTRACT,
            backend="unknown",  # type: ignore[arg-type]
        )


def test_sparse_attention_config_and_compiled_shape_contracts_fail_loudly() -> None:
    with pytest.raises(ValueError, match="segment_block"):
        SparseMlaConfig(segment_block=0)
    with pytest.raises(ValueError, match="sort tiles"):
        SparseMlaConfig(sort_tile=64)
    q_nope, q_rope, caches, tables, positions, counts, lengths = _fixture(
        jnp.float32
    )
    with pytest.raises(ValueError, match="segment blocks"):
        stage_local_sparse_mla_pallas(
            q_nope,
            q_rope,
            caches[0],
            tables,
            SelectedPositions(positions, counts),
            lengths,
            layout=LAYOUT,
            owner_index=0,
            contract=CONTRACT,
            config=CONFIG,
            interpret=False,
        )
