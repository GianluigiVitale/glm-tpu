"""The fused qkv-a q-a norm honours the accepted RMS schedule (default off)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.kernels.reference.qkv_a import FusedQkvAContract, one_row_fused_qkv_a_convolution


def _inputs(rng: np.random.Generator, contract: FusedQkvAContract):
    hidden = jnp.asarray(rng.normal(size=(1, contract.hidden_size)).astype(ml_dtypes.bfloat16))
    bits = jnp.asarray(
        rng.integers(0x20, 0x48, size=(contract.virtual_shards, contract.hidden_size, contract.packed_width_per_shard), dtype=np.uint8)
    )
    scale = jnp.asarray(
        rng.uniform(0.005, 0.02, size=(contract.virtual_shards, contract.scale_rows, contract.packed_width_per_shard)).astype(np.float32)
    )
    weight = jnp.asarray(rng.uniform(0.75, 1.25, size=(contract.q_lora_rank,)).astype(ml_dtypes.bfloat16))
    return hidden, bits, scale, weight


def test_accepted_schedule_matches_default_within_bf16_rounding_and_is_default_off() -> None:
    contract = FusedQkvAContract(hidden_size=256, q_lora_rank=64, kv_a_width=32, virtual_shards=4, quant_block=128)
    rng = np.random.default_rng(31)
    hidden, bits, scale, weight = _inputs(rng, contract)
    default = one_row_fused_qkv_a_convolution(hidden, bits, scale, weight, contract=contract)
    explicit_off = one_row_fused_qkv_a_convolution(hidden, bits, scale, weight, contract=contract, accepted_schedule=False)
    accepted = one_row_fused_qkv_a_convolution(hidden, bits, scale, weight, contract=contract, accepted_schedule=True)
    np.testing.assert_array_equal(np.asarray(default.q_residual).view(np.uint16), np.asarray(explicit_off.q_residual).view(np.uint16))
    np.testing.assert_array_equal(np.asarray(default.kv_a_projection).view(np.uint16), np.asarray(accepted.kv_a_projection).view(np.uint16))
    d = np.asarray(default.q_residual).astype(np.float32)
    a = np.asarray(accepted.q_residual).astype(np.float32)
    assert a.shape == d.shape == (1, contract.q_lora_rank)
    assert np.abs(a - d).max() <= np.abs(d).max() * 2.0**-7
    with pytest.raises(TypeError):
        one_row_fused_qkv_a_convolution(hidden, bits, scale, weight, contract=contract, accepted_schedule=1)


def test_accepted_schedule_lowers_the_barrier_carried_32_row_reduce() -> None:
    contract = FusedQkvAContract(hidden_size=256, q_lora_rank=64, kv_a_width=32, virtual_shards=4, quant_block=128)
    hidden, bits, scale, weight = _inputs(np.random.default_rng(32), contract)
    lowered = jax.jit(
        lambda h, b, s, w: one_row_fused_qkv_a_convolution(h, b, s, w, contract=contract, accepted_schedule=True)
    ).lower(hidden, bits, scale, weight).as_text()
    assert "stablehlo.optimization_barrier" in lowered and "tensor<32x64xf32>" in lowered
    assert "stablehlo.rsqrt" in lowered and "tensor<32x1xf32>" in lowered
    default_lowered = jax.jit(
        lambda h, b, s, w: one_row_fused_qkv_a_convolution(h, b, s, w, contract=contract)
    ).lower(hidden, bits, scale, weight).as_text()
    assert "tensor<32x64xf32>" not in default_lowered
