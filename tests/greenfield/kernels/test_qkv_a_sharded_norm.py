"""The fused qkv-a q-a norm keeps the legacy-faithful virtual-TP32 sharded reduction.

Replacing it with the 32-row accepted schedule flipped a DSA top-k boundary at
event 0 on TPU (2026-09-02); the sharded form was bit-exact there in every
protected run, so the accepted-schedule flag must not reach this norm.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from glm_tpu.greenfield.kernels.reference.qkv_a import FusedQkvAContract, one_row_fused_qkv_a_convolution


def test_fused_q_a_norm_has_no_schedule_switch_and_lowers_the_sharded_two_stage_reduce() -> None:
    assert "accepted_schedule" not in inspect.signature(one_row_fused_qkv_a_convolution).parameters
    contract = FusedQkvAContract(hidden_size=256, q_lora_rank=64, kv_a_width=32, virtual_shards=4, quant_block=128)
    rng = np.random.default_rng(31)
    hidden = jnp.asarray(rng.normal(size=(1, contract.hidden_size)).astype(ml_dtypes.bfloat16))
    bits = jnp.asarray(rng.integers(0x20, 0x48, size=(4, contract.hidden_size, contract.packed_width_per_shard), dtype=np.uint8))
    scale = jnp.asarray(rng.uniform(0.005, 0.02, size=(4, contract.scale_rows, contract.packed_width_per_shard)).astype(np.float32))
    weight = jnp.asarray(rng.uniform(0.75, 1.25, size=(contract.q_lora_rank,)).astype(ml_dtypes.bfloat16))
    lowered = jax.jit(
        lambda h, b, s, w: one_row_fused_qkv_a_convolution(h, b, s, w, contract=contract)
    ).lower(hidden, bits, scale, weight).as_text()
    # per-shard lane sums, then the shard sum, then a scalar-row rsqrt; no 32-row carry.
    assert "across dimensions = [2] : (tensor<4x1x16xf32>, tensor<f32>) -> tensor<4x1xf32>" in lowered
    assert "across dimensions = [0] : (tensor<4x1xf32>, tensor<f32>) -> tensor<1xf32>" in lowered
    assert "stablehlo.rsqrt" in lowered and ": tensor<1xf32>" in lowered
    assert "tensor<32x64xf32>" not in lowered and "optimization_barrier" not in lowered
