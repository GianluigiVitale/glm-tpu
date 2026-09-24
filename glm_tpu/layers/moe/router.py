"""Exactness-first GLM-5.2 MoE reference implementation.

This module restates the model contract directly in JAX.  It imports no
legacy model or TPU-inference code.  The accepted engine is only an oracle for
captured tensors and numerical comparisons.

The PP8/PP16 forms own all 256 routed experts inside one topology-local stage:
64 complete experts per PP8 chip or 128 per PP16 chip. The shared expert is
tensor-sharded over its intermediate dimension. Routed and shared partials
are stacked before one stage-local psum; the two value domains remain
separate and routed scale 2.5 is applied only after reduction. Consequently
the only collective is the combine over the two- or four-chip expert axis.

The implementation deliberately targets a true batch-one decode row.  It is
a readable fallback and correctness oracle; optimized GMM/Pallas kernels may
replace its selected-expert loop only after matching it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
from jax import lax
import jax.numpy as jnp


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
    reduction_association: str = (
        "top_k_axis_then_stage_psum; routed_and_shared_stacked_not_mixed"
    )

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
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in self.fp8_block_shape
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


def _require_shape(name: str, value: jax.Array, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


def dequantize_fp8_block_weight(
    weight: jax.Array,
    scale: jax.Array,
    *,
    block_shape: tuple[int, int] = (128, 128),
    output_dtype: jnp.dtype = jnp.bfloat16,
) -> jax.Array:
    """Fold FP8 inverse block scales into one checkpoint-oriented weight.

    ``weight`` is ``[out, in]`` and ``scale`` is
    ``[ceil(out/block_out), ceil(in/block_in)]``.  GLM dimensions are exact
    multiples of 128; tails are supported so corruption tests can exercise
    non-production shapes without silently changing ownership.
    """

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("weight and scale must both be rank two")
    if len(block_shape) != 2 or any(
        not isinstance(item, int) or isinstance(item, bool) or item <= 0
        for item in block_shape
    ):
        raise ValueError("block_shape must contain two positive integers")
    expected_scale = tuple(
        (dimension + block - 1) // block
        for dimension, block in zip(weight.shape, block_shape, strict=True)
    )
    if scale.shape != expected_scale:
        raise ValueError(
            f"scale must have shape {expected_scale} for weight {weight.shape}, "
            f"got {scale.shape}"
        )
    expanded = jnp.repeat(scale.astype(jnp.float32), block_shape[0], axis=0)
    expanded = jnp.repeat(expanded, block_shape[1], axis=1)
    expanded = expanded[: weight.shape[0], : weight.shape[1]]
    return (weight.astype(jnp.float32) * expanded).astype(output_dtype)


def route_glm_noaux_tc_logits(
    router_logits: jax.Array,
    correction_bias: jax.Array,
    *,
    top_k: int = 8,
    _observe: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return selected expert ids and unbiased normalized sigmoid weights.

    The correction bias participates only in selection.  The selected weights
    are gathered from the unbiased sigmoid scores and normalized in FP32.
    Routed scaling is intentionally absent; the accepted v4 path applies it to
    the reduced routed output.
    """

    if router_logits.ndim != 2:
        raise ValueError("router_logits must have shape [tokens, experts]")
    _require_shape("correction_bias", correction_bias, (router_logits.shape[1],))
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not (0 < top_k <= router_logits.shape[1])
    ):
        raise ValueError("top_k must be in [1, num_experts]")
    scores = jax.nn.sigmoid(router_logits.astype(jnp.float32))
    biased_scores = scores + correction_bias.astype(jnp.float32)[None, :]
    _, indices = lax.top_k(biased_scores, top_k)
    weights = jnp.take_along_axis(scores, indices, axis=-1)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True, dtype=jnp.float32)
    if _observe is not None:
        _observe(
            "router_selection",
            dict(
                scores=scores,
                biased_scores=biased_scores,
                indices=indices,
                weights=weights,
            ),
        )
    return indices.astype(jnp.int32), weights.astype(jnp.float32)
