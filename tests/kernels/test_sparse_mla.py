from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.kernels.sparse_mla.kernel import SparseMlaConfig, pregathered_sparse_mla_pallas, sparse_mla_attention
from glm_tpu.layers.contracts import MlaNumericalContract, SelectedPositions
from tests.reference.attention import gather_paged_selected_kv


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


def test_pregathered_attention_interpret_matches_full_segment_reference() -> None:
    q_nope, q_rope, caches, tables, positions, counts, lengths = _fixture(
        jnp.bfloat16
    )
    global_cache = jnp.concatenate(tuple(caches[owner] for owner in range(4)), axis=1)
    segment = gather_paged_selected_kv(
        global_cache,
        tables,
        SelectedPositions(positions, counts),
        lengths,
    )
    expected = sparse_mla_attention(
        q_nope, q_rope, segment, contract=CONTRACT
    )
    actual = pregathered_sparse_mla_pallas(
        q_nope,
        q_rope,
        segment.values,
        segment.valid_counts,
        contract=CONTRACT,
        config=CONFIG,
        interpret=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected.output, dtype=np.float32),
        rtol=0,
        atol=4e-3,
    )


def test_pregathered_attention_rejects_shape_and_dtype_drift() -> None:
    q_nope, q_rope, caches, tables, positions, counts, lengths = _fixture(
        jnp.bfloat16
    )
    global_cache = jnp.concatenate(tuple(caches[owner] for owner in range(4)), axis=1)
    segment = gather_paged_selected_kv(
        global_cache,
        tables,
        SelectedPositions(positions, counts),
        lengths,
    )
    with pytest.raises(ValueError, match="selected cache shape"):
        pregathered_sparse_mla_pallas(
            q_nope,
            q_rope,
            segment.values[:, :-1],
            segment.valid_counts,
            contract=CONTRACT,
            config=CONFIG,
            interpret=True,
        )
    with pytest.raises(ValueError, match="operands must be BF16"):
        pregathered_sparse_mla_pallas(
            q_nope.astype(jnp.float32),
            q_rope,
            segment.values,
            segment.valid_counts,
            contract=CONTRACT,
            config=CONFIG,
            interpret=True,
        )


def test_sparse_attention_config_refuses_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="segment_block"):
        SparseMlaConfig(segment_block=0)
    with pytest.raises(ValueError, match="sort tiles"):
        SparseMlaConfig(sort_tile=64)
