"""Exact one-row fused q-a/kv-a projection for the greenfield decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True, slots=True)
class FusedQkvAContract:
    """Static final-layout contract proven by protected DB502."""

    hidden_size: int = 6144
    q_lora_rank: int = 2048
    kv_a_width: int = 576
    virtual_shards: int = 32
    quant_block: int = 128
    epsilon: float = 1e-5

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_size,
            self.q_lora_rank,
            self.kv_a_width,
            self.virtual_shards,
            self.quant_block,
        )
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in dimensions
        ):
            raise ValueError("fused qkv-a dimensions must be positive integers")
        if self.q_lora_rank % self.virtual_shards or (
            self.kv_a_width % self.virtual_shards
        ):
            raise ValueError("fused qkv-a outputs must divide over virtual shards")
        if self.hidden_size % self.quant_block:
            raise ValueError("fused qkv-a contraction must contain full scale blocks")
        if self.epsilon <= 0:
            raise ValueError("fused qkv-a epsilon must be positive")

    @property
    def q_width_per_shard(self) -> int:
        return self.q_lora_rank // self.virtual_shards

    @property
    def kv_width_per_shard(self) -> int:
        return self.kv_a_width // self.virtual_shards

    @property
    def packed_width_per_shard(self) -> int:
        return self.q_width_per_shard + self.kv_width_per_shard

    @property
    def scale_rows(self) -> int:
        return self.hidden_size // self.quant_block


class FusedQkvAProjection(NamedTuple):
    """Normalized q-a state and the unnormalized fused kv-a companion."""

    q_residual: Any
    kv_a_projection: Any


def one_row_fused_qkv_a_convolution(
    normalized_hidden: Any,
    packed_weight_bits: Any,
    packed_scale: Any,
    q_a_norm_weight: Any,
    *,
    contract: FusedQkvAContract = FusedQkvAContract(),
) -> FusedQkvAProjection:
    """Project one live row through the protected shard-major N82 boundary.

    The 32 output shards are virtual arithmetic partitions inside one
    topology-local stage device. They are not token rows and create no
    collective. The packed tensors must already be in final checkpoint layout;
    this function never repacks q-a or kv-a weights during decode.
    """

    expected_shapes = {
        "normalized_hidden": (1, contract.hidden_size),
        "packed_weight_bits": (
            contract.virtual_shards,
            contract.hidden_size,
            contract.packed_width_per_shard,
        ),
        "packed_scale": (
            contract.virtual_shards,
            contract.scale_rows,
            contract.packed_width_per_shard,
        ),
        "q_a_norm_weight": (contract.q_lora_rank,),
    }
    values = {
        "normalized_hidden": normalized_hidden,
        "packed_weight_bits": packed_weight_bits,
        "packed_scale": packed_scale,
        "q_a_norm_weight": q_a_norm_weight,
    }
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(
                f"fused qkv-a {name} shape drifted: "
                f"expected={expected} found={values[name].shape}"
            )
    if normalized_hidden.dtype != jnp.bfloat16:
        raise ValueError("fused qkv-a hidden row must be BF16")
    if packed_weight_bits.dtype != jnp.uint8:
        raise ValueError("fused qkv-a weights must be raw uint8 FP8 bits")
    if packed_scale.dtype != jnp.float32:
        raise ValueError("fused qkv-a scales must be FP32")
    if q_a_norm_weight.dtype != jnp.bfloat16:
        raise ValueError("fused qkv-a norm weight must be BF16")

    def project(weight_scale: tuple[Any, Any]) -> Any:
        weight_bits, scale = weight_scale
        weight = lax.bitcast_convert_type(weight_bits, jnp.float8_e4m3fn)
        expanded_scale = jnp.repeat(
            scale, contract.quant_block, axis=0
        )[: contract.hidden_size]
        decoded = (
            weight.astype(jnp.float32) * expanded_scale.astype(jnp.float32)
        ).astype(jnp.bfloat16)
        return lax.conv_general_dilated(
            normalized_hidden,
            decoded,
            window_strides=(),
            padding=(),
            dimension_numbers=("NC", "IO", "NC"),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)

    with jax.named_scope(
        f"one_row_fused_qkv_a_n{contract.packed_width_per_shard}_convolution"
    ):
        projected = lax.map(project, (packed_weight_bits, packed_scale))

    local_q = projected[:, :, : contract.q_width_per_shard]
    # Legacy-faithful virtual-TP32 q-a RMS norm: per-shard sums of squares over
    # the 64 local lanes, then the sum over the 32 shards, then one scalar
    # rsqrt.  This mirrors the oracle's sharded q_a_layernorm and was bit-exact
    # at DSA event 0 in every protected 8K run; replacing it with the 32-row
    # accepted schedule (2026-09-02) flipped a top-k boundary at event 0, so the
    # accepted schedule deliberately does not apply here.
    local_square_sum = jnp.sum(
        jnp.square(local_q.astype(jnp.float32)), axis=-1
    )
    global_square_sum = jnp.sum(local_square_sum, axis=0)
    inverse_rms = lax.rsqrt(
        global_square_sum / jnp.float32(contract.q_lora_rank)
        + jnp.float32(contract.epsilon)
    )
    local_normalized = (
        local_q.astype(jnp.float32) * inverse_rms[None, :, None]
    ).astype(jnp.bfloat16)
    logical_q = jnp.transpose(local_normalized, (1, 0, 2)).reshape(
        1, contract.q_lora_rank
    )
    q_residual = (logical_q * q_a_norm_weight).astype(jnp.bfloat16)
    kv_a_projection = jnp.transpose(
        projected[:, :, contract.q_width_per_shard :], (1, 0, 2)
    ).reshape(1, contract.kv_a_width)
    return FusedQkvAProjection(q_residual, kv_a_projection)
