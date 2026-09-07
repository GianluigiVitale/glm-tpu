"""Opt-in complete causal multirow WS32 transformer layer.

Composition only: promoted decode remains untouched. The caller owns block
frontiers, populated prefixes, final repaired-index promotion and all-chip
commit/refusal. New prefill numerics require independent TPU/decoder admission.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .pallas import SparseMlaConfig
from .reference.attention import MlaNumericalContract
from .reference.dsa import DsaNumericalContract
from .reference.moe import GlmMoeNumericalContract, route_glm_noaux_tc_logits
from .ws32 import ws32_fused_add_rms_norm_mapped
from .ws32_layer import (
    Ws32AttentionWeights,
    Ws32DenseWeights,
    Ws32DsaWeights,
    Ws32MoeWeights,
    Ws32QkvAWeights,
)
from .ws32_prefill_attention import (
    ws32_prefill_index_share_attention_mapped,
    ws32_prefill_prepare_attention_mapped,
)
from .ws32_prefill_dsa import ws32_prefill_dsa_mapped
from .ws32_prefill_linear import ws32_prefill_dense_mapped
from .ws32_prefill_moe import ws32_prefill_moe_from_routes_mapped


class Ws32PrefillLayerResult(NamedTuple):
    output_local: Any
    carried_residual_local: Any
    cache_local: Any
    unrepaired_index_cache: Any
    repaired_index_cache: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    route_indices: Any
    route_weights: Any
    contract_valid: Any


def ws32_prefill_router_mapped(
    hidden_local: Any,
    router_weight_local: Any,
    correction_bias_local: Any,
    live: Any,
    *,
    top_k: int = 8,
) -> tuple[Any, Any, Any]:
    """Exact noaux_tc selection of this multirow FP32 router's own logits.

    Correction bias affects IDs only, never mixture weights. Padded rows return
    valid placeholder IDs/zero weights for the grouped kernel; they do not claim
    zero execution cost. Final-block executables should use narrow static rows.
    """
    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("prefill router requires WS32 expert8/feature4 mesh")
    if (
        hidden_local.ndim != 2
        or not 1 <= hidden_local.shape[0] <= 32
        or hidden_local.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill router requires1..32 BF16 feature rows")
    rows = hidden_local.shape[0]
    if live.shape != (rows,) or live.dtype != jnp.bool_:
        raise ValueError("prefill router requires boolean live rows")
    if (
        router_weight_local.ndim != 2
        or router_weight_local.shape[1] != hidden_local.shape[1]
        or (
            correction_bias_local.shape != (router_weight_local.shape[0],)
            or router_weight_local.dtype != jnp.bfloat16
            or correction_bias_local.dtype != jnp.float32
        )
    ):
        raise ValueError("prefill router owner geometry/dtype drifted")
    clean = jnp.where(live[:, None], hidden_local, 0)
    partial = lax.dot_general(
        clean.astype(jnp.float32),
        router_weight_local.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    with jax.named_scope("greenfield_ws32_prefill_router/feature_reduce"):
        local_logits = lax.psum(partial, "feature")
    with jax.named_scope("greenfield_ws32_prefill_router/expert_gather"):
        logits = lax.all_gather(local_logits, "expert", axis=1, tiled=True)
        bias = lax.all_gather(correction_bias_local, "expert", axis=0, tiled=True)
    indices, weights = route_glm_noaux_tc_logits(logits, bias, top_k=top_k)
    valid = ~live | (
        jnp.all(jnp.isfinite(clean), axis=1)
        & jnp.all(jnp.isfinite(logits), axis=1)
        & jnp.all(jnp.isfinite(bias))
        & jnp.all(jnp.isfinite(weights), axis=1)
    )
    return indices, jnp.where(live[:, None], weights, 0), valid


def ws32_prefill_transformer_layer_mapped(
    hidden_update_local: Any,
    carried_residual_local: Any,
    cache_local: Any,
    unrepaired_index_cache: Any,
    repaired_index_cache: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    selected_scores: Any,
    position_offset: Any,
    valid_rows: Any,
    block_table: Any,
    qkv_a_weights: Ws32QkvAWeights,
    attention_weights: Ws32AttentionWeights,
    dsa_weights: Ws32DsaWeights | None,
    materialized_wk: Any | None,
    post_attention_norm_weight_local: Any,
    dense_weights: Ws32DenseWeights | None,
    moe_weights: Ws32MoeWeights | None,
    incoming_contract_valid: Any,
    *,
    main_rope_table_rows: Any,
    dsa_contract: DsaNumericalContract = DsaNumericalContract(),
    attention_contract: MlaNumericalContract = MlaNumericalContract(),
    moe_contract: GlmMoeNumericalContract = GlmMoeNumericalContract(stage_size=8),
    rms_norm_epsilon: float = 1e-5,
    key_tile: int = 4096,
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(segment_block=512),
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> Ws32PrefillLayerResult:
    """Execute one full/shared-indexer × dense/MoE layer on a prompt block.

    Static weight presence selects branches; inactive branches do not trace model
    compute. Carry split residuals through BOTH norms without first rounding their
    FP32 sum. Metadata and local health stay on device; no Python per-row dispatch.
    Failed health forbids committing any returned cache/residual on any chip.
    """
    if (
        hidden_update_local.ndim != 2
        or not 1 <= hidden_update_local.shape[0] <= 32
        or (
            hidden_update_local.dtype != jnp.bfloat16
            or carried_residual_local.shape != hidden_update_local.shape
            or carried_residual_local.dtype != jnp.bfloat16
            or hidden_update_local.shape[1] * 4 != dsa_contract.hidden_size
        )
    ):
        raise ValueError("prefill split residual geometry drifted")
    rows = hidden_update_local.shape[0]
    if (
        incoming_contract_valid.shape != (rows,)
        or incoming_contract_valid.dtype != jnp.bool_
    ):
        raise ValueError("prefill layer requires per-row boolean incoming health")
    if valid_rows.shape != () or valid_rows.dtype != jnp.int32:
        raise ValueError("prefill live count must be int32 scalar")
    if (dsa_weights is None) != (materialized_wk is None):
        raise ValueError(
            "full indexer requires both raw DSA owners and completed repair wk"
        )
    if (dense_weights is None) == (moe_weights is None):
        raise ValueError("prefill layer requires exactly one dense or MoE branch")
    if (
        moe_contract.hidden_size != dsa_contract.hidden_size
        or moe_contract.stage_size != 8
        or (dsa_contract.top_k != attention_contract.top_k)
    ):
        raise ValueError("prefill layer numerical contracts disagree")
    if (
        selected_positions.shape != (rows, dsa_contract.top_k)
        or selected_positions.dtype != jnp.int32
        or (
            selected_valid_counts.shape != (rows,)
            or selected_valid_counts.dtype != jnp.int32
            or selected_scores.shape != selected_positions.shape
            or selected_scores.dtype != jnp.float32
        )
    ):
        raise ValueError("prefill IndexShare metadata geometry drifted")
    live = jnp.arange(rows, dtype=jnp.int32) < jnp.clip(valid_rows, 0, rows)
    update = jnp.where(live[:, None], hidden_update_local, 0)
    residual = jnp.where(live[:, None], carried_residual_local, 0)
    normalized, combined = ws32_fused_add_rms_norm_mapped(
        update,
        residual,
        qkv_a_weights.input_norm_weight_local,
        global_hidden_size=dsa_contract.hidden_size,
        epsilon=rms_norm_epsilon,
    )
    prepared = ws32_prefill_prepare_attention_mapped(
        combined,
        qkv_a_weights,
        precomputed_normalized_local=normalized,
        rms_norm_epsilon=rms_norm_epsilon,
        linear_interpret=linear_interpret,
    )
    dsa_valid = jnp.bool_(True)
    if dsa_weights is not None:
        dsa = ws32_prefill_dsa_mapped(
            prepared,
            unrepaired_index_cache,
            repaired_index_cache,
            position_offset,
            valid_rows,
            block_table,
            dsa_weights,
            materialized_wk,
            contract=dsa_contract,
            key_tile=key_tile,
            linear_interpret=linear_interpret,
        )
        unrepaired_index_cache, repaired_index_cache = (
            dsa.unrepaired_index_cache,
            dsa.repaired_index_cache,
        )
        selected_positions, selected_valid_counts, selected_scores = (
            dsa.selected_positions,
            dsa.selected_valid_counts,
            dsa.selected_scores,
        )
        dsa_valid = dsa.contract_valid
    attention = ws32_prefill_index_share_attention_mapped(
        combined,
        prepared,
        cache_local,
        selected_positions,
        selected_valid_counts,
        position_offset,
        valid_rows,
        block_table,
        attention_weights,
        main_rope_table_rows=main_rope_table_rows,
        contract=attention_contract,
        sparse_attention_config=sparse_attention_config,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
        add_residual=False,
    )
    normalized_mlp, post_residual = ws32_fused_add_rms_norm_mapped(
        attention.output_local,
        combined,
        post_attention_norm_weight_local,
        global_hidden_size=moe_contract.hidden_size,
        epsilon=rms_norm_epsilon,
    )
    normalized_mlp = jnp.where(live[:, None], normalized_mlp, 0)
    mlp_valid = jnp.ones((rows,), jnp.bool_)
    if dense_weights is not None:
        output = ws32_prefill_dense_mapped(
            normalized_mlp, *dense_weights, interpret=linear_interpret
        )
        route_indices = jnp.full((rows, moe_contract.top_k), -1, jnp.int32)
        route_weights = jnp.zeros((rows, moe_contract.top_k), jnp.float32)
    else:
        if moe_weights.router_weight_local.shape[0] * 8 != moe_contract.num_experts:
            raise ValueError("prefill router and grouped expert counts disagree")
        route_indices, route_weights, router_valid = ws32_prefill_router_mapped(
            normalized_mlp,
            moe_weights.router_weight_local,
            moe_weights.correction_bias_local,
            live,
            top_k=moe_contract.top_k,
        )
        output, grouped_valid = ws32_prefill_moe_from_routes_mapped(
            normalized_mlp,
            route_indices,
            route_weights,
            *moe_weights[2:],
            contract=moe_contract,
            interpret=linear_interpret,
            fp32_route_sum=True,
        )
        mlp_valid = router_valid & grouped_valid
    output = jnp.where(live[:, None], output, 0)
    post_residual = jnp.where(live[:, None], post_residual, 0)
    selected_live = (
        jnp.arange(dsa_contract.top_k)[None] < selected_valid_counts[:, None]
    )
    valid = (
        incoming_contract_valid
        & dsa_valid
        & attention.contract_valid
        & mlp_valid
        & (
            ~live
            | (
                jnp.all(jnp.isfinite(normalized), axis=1)
                & jnp.all(jnp.isfinite(normalized_mlp), axis=1)
                & jnp.all(jnp.isfinite(output), axis=1)
                & jnp.all(jnp.isfinite(post_residual), axis=1)
                & jnp.all(~selected_live | jnp.isfinite(selected_scores), axis=1)
            )
        )
    )
    return Ws32PrefillLayerResult(
        output,
        post_residual,
        attention.cache_local,
        unrepaired_index_cache,
        repaired_index_cache,
        selected_positions,
        selected_valid_counts,
        selected_scores,
        route_indices,
        route_weights,
        valid,
    )
