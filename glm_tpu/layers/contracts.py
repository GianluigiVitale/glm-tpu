"""Numerical contracts and state types shared by the layers.

The absorbed-MLA, DSA-indexer and MoE contracts, the stage-local KV-cache layout, the compact DSA
selection carried between IndexShare layers, and the shape validators of their inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


@dataclass(frozen=True, slots=True)
class MlaNumericalContract:
    """Compile-relevant GLM-5.2 absorbed-MLA arithmetic contract."""

    num_heads: int = 64
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256
    v_head_dim: int = 256
    packed_cache_width: int = 640
    top_k: int = 2048
    score_dtype: str = "float32"
    probability_rounding: str = "unnormalized_cache_dtype_before_pv"
    position_order: str = "ascending_global_position"

    def __post_init__(self) -> None:
        for field in (
            "num_heads",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "qk_head_dim",
            "v_head_dim",
            "packed_cache_width",
            "top_k",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.qk_nope_head_dim + self.qk_rope_head_dim != self.qk_head_dim:
            raise ValueError("qk_head_dim must equal nope plus RoPE dimensions")
        if self.kv_lora_rank + self.qk_rope_head_dim > self.packed_cache_width:
            raise ValueError("packed cache width truncates latent or RoPE state")
        if self.score_dtype != "float32":
            raise ValueError("sparse MLA scores and softmax must be FP32")
        if self.probability_rounding != "unnormalized_cache_dtype_before_pv":
            raise ValueError("sparse MLA probability-rounding contract drifted")
        if self.position_order != "ascending_global_position":
            raise ValueError("sparse MLA accumulation order must be position canonical")

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim**-0.5


@dataclass(frozen=True, slots=True)
class StageLocalKvLayout:
    """Context striping for one attention layer's topology-local cache."""

    logical_page_size: int = 512
    local_parallel_size: int = 4
    packed_cache_width: int = 640

    def __post_init__(self) -> None:
        for field in (
            "logical_page_size",
            "local_parallel_size",
            "packed_cache_width",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.logical_page_size % self.local_parallel_size:
            raise ValueError("logical page size must divide over the local stage group")

    @property
    def local_rows_per_page(self) -> int:
        return self.logical_page_size // self.local_parallel_size

    def owner(self, positions: jax.Array) -> jax.Array:
        """Return the local chip owning each non-negative absolute position."""

        return (positions.astype(jnp.int32) % jnp.int32(self.logical_page_size)) // jnp.int32(self.local_rows_per_page)


def _require_shape(name: str, value: jax.Array, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


def _require_int32(name: str, value: jax.Array) -> None:
    if value.dtype != jnp.int32:
        raise ValueError(f"{name} must have dtype int32, got {value.dtype}")


@dataclass(frozen=True, slots=True)
class DsaNumericalContract:
    """Compile-relevant semantic contract for the GLM lightning indexer."""

    hidden_size: int = 6144
    q_lora_rank: int = 2048
    num_heads: int = 32
    head_dim: int = 128
    rotary_dim: int = 64
    top_k: int = 2048
    theta: float = 8_000_000.0
    key_layer_norm_epsilon: float = 1e-6
    interleaved_rotary: bool = True
    score_dtype: str = "float32"
    tie_policy: str = "descending_score_then_lowest_global_position"
    padding_sentinel: int = -1

    def __post_init__(self) -> None:
        for field in (
            "hidden_size",
            "q_lora_rank",
            "num_heads",
            "head_dim",
            "rotary_dim",
            "top_k",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.rotary_dim > self.head_dim or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be even and no larger than head_dim")
        if (
            isinstance(self.theta, bool)
            or isinstance(self.key_layer_norm_epsilon, bool)
            or self.theta <= 0
            or self.key_layer_norm_epsilon <= 0
        ):
            raise ValueError("theta and key LayerNorm epsilon must be positive")
        if self.interleaved_rotary is not True:
            raise ValueError("the exact GLM DSA contract requires interleaved rotary")
        if self.score_dtype != "float32":
            raise ValueError("the exact GLM DSA scorer requires FP32 scores")
        if self.tie_policy != "descending_score_then_lowest_global_position":
            raise ValueError("the exact GLM DSA tie policy drifted")
        if self.padding_sentinel != -1:
            raise ValueError("selected-position padding sentinel must be -1")


class SelectedPositions(NamedTuple):
    """Compact DSA/IndexShare state; decode shape is ``[1, top_k]``."""

    positions: jax.Array
    valid_counts: jax.Array


@dataclass(frozen=True, slots=True)
class GlmMoeNumericalContract:
    """Compile-relevant arithmetic contract for one sparse GLM layer."""

    hidden_size: int = 6144
    intermediate_size: int = 2048
    num_experts: int = 256
    top_k: int = 8
    stage_size: int = 4
    routed_scaling_factor: float = 2.5
    fp8_block_shape: tuple[int, int] = (128, 128)
    activation_dtype: str = "bfloat16"
    router_dtype: str = "float32"
    fp8_scale_dtype: str = "float32"
    scoring_function: str = "sigmoid"
    routing_method: str = "noaux_tc"
    routing_tie_policy: str = "jax_lax_top_k_lowest_expert_id"
    reduction_association: str = "top_k_axis_then_stage_psum; routed_and_shared_stacked_not_mixed"

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "stage_size",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.num_experts % self.stage_size:
            raise ValueError("num_experts must divide evenly over the stage")
        if self.intermediate_size % self.stage_size:
            raise ValueError("shared intermediate size must divide over the stage")
        if self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be positive")
        if len(self.fp8_block_shape) != 2 or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in self.fp8_block_shape
        ):
            raise ValueError("fp8_block_shape must contain two positive integers")
        if self.scoring_function != "sigmoid":
            raise ValueError("GLM-5.2 requires sigmoid MoE scoring")
        if self.routing_method != "noaux_tc":
            raise ValueError("GLM-5.2 requires noaux_tc routing")

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.stage_size

    @property
    def local_shared_intermediate(self) -> int:
        return self.intermediate_size // self.stage_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_dtype": self.activation_dtype,
            "fp8_block_shape": list(self.fp8_block_shape),
            "fp8_scale_dtype": self.fp8_scale_dtype,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "reduction_association": self.reduction_association,
            "routed_scaling_factor": self.routed_scaling_factor,
            "router_dtype": self.router_dtype,
            "routing_method": self.routing_method,
            "routing_tie_policy": self.routing_tie_policy,
            "scoring_function": self.scoring_function,
            "stage_size": self.stage_size,
            "top_k": self.top_k,
        }
