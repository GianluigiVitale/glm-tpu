"""CPU multirow attention primitives; not TPU arithmetic or speed proof."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.pallas.fp8_matmul import (
    fp8_structured_kv_b_q_absorb,
    fp8_structured_kv_b_value,
)
from glm_tpu.optimized.sparse_attention import pregathered_sparse_mla_pallas, SparseMlaConfig
from glm_tpu.optimized.reference.attention import MlaNumericalContract


@pytest.mark.parametrize("rows", [1, 17, 32])
def test_structured_batched_queries_and_values_match_old_row_path(rows):
    from jax._src.pallas.mosaic import tpu_info

    tpu_info.registry["cpu"] = lambda: tpu_info.get_tpu_info_for_chip(
        tpu_info.ChipVersion.TPU_V4, 1
    )
    tpu_info.get_tpu_info.cache_clear()
    rng = np.random.default_rng(778)
    # Both head offsets, nonsymmetric raw weights and every scale block distinct.
    bits = jnp.asarray(
        np.asarray(rng.normal(0, 0.1, (896, 512)), ml_dtypes.float8_e4m3fn).view(
            np.uint8
        )
    )
    scales = jnp.asarray(rng.uniform(0.25, 1, (7, 4)), jnp.float32)
    for fn, width in (
        (fp8_structured_kv_b_q_absorb, 192),
        (fp8_structured_kv_b_value, 512),
    ):
        x = jnp.asarray(rng.normal(0, 0.2, (rows, 2, width)), jnp.bfloat16)
        if rows > 1:
            x = x.at[4].set(0)
        actual = fn(x, bits, scales, prefill=True, interpret=True)
        expected = jnp.concatenate(
            [fn(x[i : i + 1], bits, scales, interpret=True) for i in range(rows)]
        )
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint16), np.asarray(expected).view(np.uint16)
        )
        if rows > 1:
            with pytest.raises(ValueError, match="one decode row"):
                fn(x, bits, scales, prefill=False, interpret=True)
        # One call over the full row tile, not R calls/scanned weight reloads.
        graph = str(
            jax.make_jaxpr(lambda v: fn(v, bits, scales, prefill=True, interpret=True))(
                x
            )
        )
        assert graph.count("name=greenfield_fp8_structured") == 1


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
