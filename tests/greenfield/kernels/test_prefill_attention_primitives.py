"""CPU multirow attention primitives; not TPU arithmetic or speed proof."""

import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.optimized.sparse_attention import pregathered_sparse_mla_pallas, SparseMlaConfig
from glm_tpu.optimized.reference.attention import MlaNumericalContract


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
    counts = jnp.asarray(
        [0, 1, 3, 4, 5, 15, 16, 0, 16, 1, 0, 9, 3, 0, 16, 7, 0], jnp.int32
    )
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
        [
            run(q[i : i + 1], r[i : i + 1], cache[i : i + 1], counts[i : i + 1], False)
            for i in range(rows)
        ]
    )
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint16), np.asarray(expected).view(np.uint16)
    )
    assert np.all(np.asarray(actual)[np.asarray(counts) == 0] == 0)
    order = jnp.asarray(rng.permutation(rows))
    permuted = run(q[order], r[order], cache[order], counts[order], True)
    np.testing.assert_array_equal(np.asarray(permuted), np.asarray(actual[order]))
    with pytest.raises(ValueError, match="prefill=True"):
        run(q, r, cache, counts, False)
