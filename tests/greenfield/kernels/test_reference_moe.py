from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.optimized.reference.moe import (
    GlmMoeNumericalContract,
    dequantize_fp8_block_weight,
    route_glm_noaux_tc_logits,
)


def test_glm_52_contract_is_exact() -> None:
    contract = GlmMoeNumericalContract()
    assert contract.hidden_size == 6144
    assert contract.intermediate_size == 2048
    assert contract.num_experts == 256
    assert contract.top_k == 8
    assert contract.stage_size == 4
    assert contract.local_experts == 64
    assert contract.local_shared_intermediate == 512
    assert contract.routed_scaling_factor == 2.5
    assert contract.fp8_block_shape == (128, 128)
    assert contract.router_dtype == "float32"


def test_contract_refuses_nonlocal_expert_partition() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        GlmMoeNumericalContract(num_experts=255)


def test_block_dequant_uses_checkpoint_out_in_blocks() -> None:
    weight = jnp.arange(1, 25, dtype=jnp.float32).reshape(4, 6)
    scale = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    got = dequantize_fp8_block_weight(
        weight, scale, block_shape=(2, 3), output_dtype=jnp.float32
    )
    expanded = np.asarray(
        [
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(weight) * expanded)


def test_noaux_bias_selects_but_does_not_weight() -> None:
    logits = jnp.asarray([[3.0, 2.0, 1.0, 0.0]], dtype=jnp.float32)
    bias = jnp.asarray([[0.0, 0.0, 100.0, 99.0]], dtype=jnp.float32)[0]
    indices, weights = route_glm_noaux_tc_logits(logits, bias, top_k=2)
    np.testing.assert_array_equal(np.asarray(indices), [[2, 3]])
    unbiased = jax.nn.sigmoid(logits)[0, jnp.asarray([2, 3])]
    expected = unbiased / unbiased.sum()
    np.testing.assert_allclose(np.asarray(weights[0]), np.asarray(expected), rtol=0, atol=0)


def test_router_ties_choose_lowest_expert_ids() -> None:
    logits = jnp.zeros((1, 16), dtype=jnp.float32)
    indices, weights = route_glm_noaux_tc_logits(
        logits, jnp.zeros((16,), dtype=jnp.float32), top_k=8
    )
    np.testing.assert_array_equal(np.asarray(indices), [list(range(8))])
    np.testing.assert_array_equal(
        np.asarray(weights), np.full((1, 8), 0.125, dtype=np.float32)
    )
