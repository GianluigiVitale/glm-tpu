"""Numerical contracts and state types shared by the layers.

The absorbed-MLA, DSA-indexer and MoE contracts and the stage-local KV-cache layout (configuration: defined in
:mod:`glm_tpu.config.cache`, re-exported here), the compact DSA selection carried between IndexShare layers, the
shape validators of their inputs, and the BF16 resident weight groups the attention, indexer and dense layer bodies
take (built by :mod:`glm_tpu.models.glm_moe_dsa.weights`).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

# The contracts are configuration (glm_tpu.config.cache defines them and imports no JAX); the layers import them
# from here.
from glm_tpu.config.cache import (
    DsaNumericalContract as DsaNumericalContract,
    GlmMoeNumericalContract as GlmMoeNumericalContract,
    MlaNumericalContract as MlaNumericalContract,
    StageLocalKvLayout as StageLocalKvLayout,
)


def require_shape(name: str, value: jax.Array, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


def require_int32(name: str, value: jax.Array) -> None:
    if value.dtype != jnp.int32:
        raise ValueError(f"{name} must have dtype int32, got {value.dtype}")


class SelectedPositions(NamedTuple):
    """Compact DSA/IndexShare state; decode shape is ``[1, top_k]``."""

    positions: jax.Array
    valid_counts: jax.Array


# ----------------------------------------------------------------------------- BF16 resident weight groups
# The non-routed tables the attention, indexer and dense layer bodies take, decoded once at load time
# (built by glm_tpu.models.glm_moe_dsa.weights).
class Bf16QkvAWeights(NamedTuple):
    input_norm_weight_local: Any
    q_a_local: Any  # bf16 [q_lora, local_hidden]
    q_a_norm_weight: Any
    kv_a_local: Any  # bf16 [kv_lora + rope, local_hidden]
    kv_a_norm_weight: Any


class Bf16AttentionWeights(NamedTuple):
    q_b_local: Any  # bf16 [local_heads * qk_head, q_lora]
    kv_b_local: Any  # bf16 [local_heads * 448, 512]
    o_local: Any  # bf16 [local_hidden, local_heads * v_head]


class Bf16DsaWeights(NamedTuple):
    wq_b_local: Any  # bf16 [local_dsa_heads * 128, q_lora]
    wk_local: Any  # bf16 [128, local_hidden]
    key_norm_weight: Any
    key_norm_bias: Any
    head_weight_local: Any


class Bf16DenseWeights(NamedTuple):
    gate_local: Any
    up_local: Any
    down_local: Any
