"""Readable exactness-first JAX reference kernels."""

from .linear import (
    dense_swiglu,
    embedding_lookup,
    linear,
    residual_add,
    silu,
    vocabulary_logits,
)
from .moe import (
    GlmMoeNumericalContract,
    dequantize_fp8_block_weight,
    reference_moe_from_routes,
    route_glm_noaux_tc,
    route_glm_noaux_tc_logits,
    stage_local_moe,
    stage_local_moe_from_routes,
)
from .rmsnorm import final_norm, rms_norm
from .rotary import apply_rotary, rotary_cos_sin

__all__ = (
    "GlmMoeNumericalContract",
    "apply_rotary",
    "dequantize_fp8_block_weight",
    "dense_swiglu",
    "embedding_lookup",
    "final_norm",
    "linear",
    "reference_moe_from_routes",
    "residual_add",
    "rms_norm",
    "rotary_cos_sin",
    "route_glm_noaux_tc",
    "route_glm_noaux_tc_logits",
    "silu",
    "stage_local_moe",
    "stage_local_moe_from_routes",
    "vocabulary_logits",
)
