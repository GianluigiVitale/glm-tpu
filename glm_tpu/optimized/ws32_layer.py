"""Complete batch-one WS32 attention, DSA, and IndexShare layer bodies.

These functions execute inside one ``shard_map`` over the exact
``expert=8 x feature=4`` mesh.  The residual stays ``[1, hidden/4]`` on the
feature axis.  Context pages and complete attention/DSA heads stay on the
expert axis and are replicated only across feature columns.  Repeated
collectives therefore have physical size four or eight; no operation
materializes a full hidden row on all 32 chips.

Moved verbatim at S2f from ``glm_tpu/greenfield/kernels/ws32_layer.py`` (its production definitions; the
research remainder is archived at ``archive/research-20260922``).
"""
from __future__ import annotations

from typing import Any, NamedTuple


class Ws32QkvAWeights(NamedTuple):
    input_norm_weight_local: Any
    q_a_bits_local: Any
    q_a_scale_local: Any
    q_a_norm_weight: Any
    kv_a_bits_local: Any
    kv_a_scale_local: Any
    kv_a_norm_weight: Any


class Ws32DsaWeights(NamedTuple):
    wq_b_bits_local: Any
    wq_b_scale_local: Any
    wk_bits_local: Any
    wk_scale_local: Any
    key_norm_weight: Any
    key_norm_bias: Any
    head_weight_local: Any


class Ws32AttentionWeights(NamedTuple):
    q_b_bits_local: Any
    q_b_scale_local: Any
    kv_b_bits_local: Any
    kv_b_scale_local: Any
    o_bits_local: Any
    o_scale_local: Any


class Ws32DenseWeights(NamedTuple):
    gate_bits_local: Any
    gate_scale_local: Any
    up_bits_local: Any
    up_scale_local: Any
    down_bits_local: Any
    down_scale_local: Any


class Ws32StrategyNdDenseWeights(NamedTuple):
    """Four ordered legacy-rank shards in their final WS32 ownership."""

    merged_bits_in_out_local: Any
    merged_scale_in_out_local: Any
    down_bits_in_out_local: Any
    down_scale_in_out_local: Any


class Ws32MoeWeights(NamedTuple):
    router_weight_local: Any
    correction_bias_local: Any
    expert_gate_bits_local: Any
    expert_gate_scale_local: Any
    expert_up_bits_local: Any
    expert_up_scale_local: Any
    expert_down_bits_local: Any
    expert_down_scale_local: Any
    shared_gate_bits_local: Any
    shared_gate_scale_local: Any
    shared_up_bits_local: Any
    shared_up_scale_local: Any
    shared_down_bits_local: Any
    shared_down_scale_local: Any


class Ws32PreparedAttention(NamedTuple):
    normalized_local: Any
    normalized_for_exact_dsa: Any
    q_residual: Any
    current_kv: Any


class Ws32DsaResult(NamedTuple):
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    contract_valid: Any


class Ws32AttentionResult(NamedTuple):
    output_local: Any
    cache_local: Any
    contract_valid: Any


class Ws32AttentionLayerResult(NamedTuple):
    output_local: Any
    cache_local: Any
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    contract_valid: Any


class Ws32TransformerLayerResult(NamedTuple):
    output_local: Any
    carried_residual_local: Any
    normalized_input_local: Any
    cache_local: Any
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    route_indices: Any
    route_weights: Any
    contract_valid: Any


class Ws32MlpResult(NamedTuple):
    output_local: Any
    route_indices: Any
    route_weights: Any
