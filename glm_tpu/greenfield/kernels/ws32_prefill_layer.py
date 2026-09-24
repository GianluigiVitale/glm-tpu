"""Opt-in complete causal multirow WS32 transformer layer.

Composition only: promoted decode remains untouched. The caller owns block
frontiers, populated prefixes, final repaired-index promotion and all-chip
commit/refusal. New prefill numerics require independent TPU/decoder admission.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .pallas.sparse_attention import SparseMlaConfig
from ...optimized.reference.attention import MlaNumericalContract
from ...optimized.reference.dsa import DsaNumericalContract
from ...optimized.reference.moe import GlmMoeNumericalContract, route_glm_noaux_tc_logits
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
from glm_tpu.optimized.prefill_layer import (
    Ws32PrefillLayerResult,
    Ws32PrefillPrefixResult,
    ws32_prefill_router_mapped,
)  # S2f: moved to production


def ws32_prefill_mlp_mapped(
    normalized_mlp: Any,
    live: Any,
    dense_weights: Ws32DenseWeights | None,
    moe_weights: Ws32MoeWeights | None,
    *,
    moe_contract: GlmMoeNumericalContract,
    linear_interpret: bool = False,
    expert_panels: bool = False,
    _observe: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Shared suffix for a small layer or a <=128-row MLP window.

    No normalization or residual addition here: the prefix has already applied
    both split-residual norms. The caller must retain prefix health and state.
    """
    if (
        normalized_mlp.ndim != 2
        or not 1 <= normalized_mlp.shape[0] <= 128
        or normalized_mlp.dtype != jnp.bfloat16
        or normalized_mlp.shape[1] * 4 != moe_contract.hidden_size
        or live.shape != (normalized_mlp.shape[0],)
        or live.dtype != jnp.bool_
        or (dense_weights is None) == (moe_weights is None)
    ):
        raise ValueError("prefill MLP boundary geometry/branch drifted")
    rows = normalized_mlp.shape[0]
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
            _observe=_observe,
        )
        output, grouped_valid = ws32_prefill_moe_from_routes_mapped(
            normalized_mlp,
            route_indices,
            route_weights,
            *moe_weights[2:],
            contract=moe_contract,
            interpret=linear_interpret,
            fp32_route_sum=True,
            expert_panels=expert_panels,
        )
        mlp_valid = router_valid & grouped_valid
    return output, route_indices, route_weights, mlp_valid


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
    paired_position_sort: bool = False,
    sorted_local_merge: bool = False,
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(segment_block=512),
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    prefix_only: bool = False,
    _observe: Callable[[str, dict[str, Any]], None] | None = None,
) -> Ws32PrefillLayerResult | Ws32PrefillPrefixResult:
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
            paired_position_sort=paired_position_sort,
            sorted_local_merge=sorted_local_merge,
            linear_interpret=linear_interpret,
            _observe=_observe,
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
        _observe=_observe,
    )
    normalized_mlp = jnp.where(live[:, None], normalized_mlp, 0)
    if _observe is not None:
        _observe(
            "attention_mlp_boundary",
            dict(
                attention_update=attention.output_local,
                combined=combined,
                normalized_mlp=normalized_mlp,
                live=live,
            ),
        )
    if prefix_only:
        post_residual = jnp.where(live[:, None], post_residual, 0)
        selected_live = (
            jnp.arange(dsa_contract.top_k)[None] < selected_valid_counts[:, None]
        )
        prefix_valid = (
            incoming_contract_valid
            & dsa_valid
            & attention.contract_valid
            & (
                ~live
                | (
                    jnp.all(jnp.isfinite(normalized), axis=1)
                    & jnp.all(jnp.isfinite(normalized_mlp), axis=1)
                    & jnp.all(jnp.isfinite(post_residual), axis=1)
                    & jnp.all(~selected_live | jnp.isfinite(selected_scores), axis=1)
                )
            )
        )
        return Ws32PrefillPrefixResult(
            normalized_mlp,
            post_residual,
            attention.cache_local,
            unrepaired_index_cache,
            repaired_index_cache,
            selected_positions,
            selected_valid_counts,
            selected_scores,
            prefix_valid,
            normalized,
        )
    output, route_indices, route_weights, mlp_valid = ws32_prefill_mlp_mapped(
        normalized_mlp,
        live,
        dense_weights,
        moe_weights,
        moe_contract=moe_contract,
        linear_interpret=linear_interpret,
        _observe=_observe,
    )
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
        normalized,
    )
