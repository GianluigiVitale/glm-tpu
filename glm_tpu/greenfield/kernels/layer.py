"""Complete raw-FP8 transformer-layer composition inside one local stage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import jax.numpy as jnp

from .pallas import Fp8BlockMatmulConfig
from .reference.attention import MlaNumericalContract, StageLocalKvLayout
from .reference.dsa import DsaNumericalContract
from .reference.linear import residual_add
from .reference.moe import GlmMoeNumericalContract
from .reference.rmsnorm import fused_add_rms_norm, rms_norm
from .stage_local import (
    StageLinearBackend,
    _stage_fp8_linear,
    stage_local_dense_fp8_mapped,
    stage_local_dsa_fp8_mapped,
    stage_local_index_share_fp8_mapped,
    stage_local_moe_fp8_mapped,
    stage_local_moe_pallas_feature_mapped,
)

SparseMoeBackend = Literal["reference", "pallas_feature"]


class AttentionFp8Weights(NamedTuple):
    q_a_bits: Any
    q_a_scale: Any
    q_a_norm_weight: Any
    q_b_bits: Any
    q_b_scale: Any
    kv_a_bits: Any
    kv_a_scale: Any
    kv_a_norm_weight: Any
    kv_b_bits: Any
    kv_b_scale: Any
    o_bits: Any
    o_scale: Any


class DsaFp8Weights(NamedTuple):
    wq_b_bits: Any
    wq_b_scale: Any
    wk_bits: Any
    wk_scale: Any
    key_norm_weight: Any
    key_norm_bias: Any
    head_weight: Any


class DenseFp8Weights(NamedTuple):
    gate_bits: Any
    gate_scale: Any
    up_bits: Any
    up_scale: Any
    down_bits: Any
    down_scale: Any


class MoeFp8Weights(NamedTuple):
    router_weight: Any
    correction_bias: Any
    expert_gate_bits: Any
    expert_gate_scale: Any
    expert_up_bits: Any
    expert_up_scale: Any
    expert_down_bits: Any
    expert_down_scale: Any
    shared_gate_bits: Any
    shared_gate_scale: Any
    shared_up_bits: Any
    shared_up_scale: Any
    shared_down_bits: Any
    shared_down_scale: Any


class StageLocalLayerFp8Result(NamedTuple):
    output: Any
    kv_cache: Any
    index_cache: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    route_indices: Any
    route_weights: Any
    contract_valid: Any


class StageLocalSplitLayerFp8Result(NamedTuple):
    hidden_states: Any
    residual: Any
    kv_cache: Any
    index_cache: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    route_indices: Any
    route_weights: Any
    contract_valid: Any


def stage_local_transformer_layer_fp8_mapped(
    residual: Any,
    kv_cache: Any,
    index_cache: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    input_norm_weight: Any,
    post_attention_norm_weight: Any,
    attention: AttentionFp8Weights,
    dsa: DsaFp8Weights | None,
    dense: DenseFp8Weights | None,
    moe: MoeFp8Weights | None,
    incoming_contract_valid: Any,
    local_slot: Any,
    *,
    axis_name: str,
    indexer_kind: str,
    mlp_kind: str,
    dsa_contract: DsaNumericalContract = DsaNumericalContract(),
    mla_contract: MlaNumericalContract = MlaNumericalContract(),
    moe_contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    block_shape: tuple[int, int] = (128, 128),
    rms_norm_epsilon: float = 1e-5,
    lora_norm_epsilon: float = 1e-5,
    rope_theta: float = 8_000_000.0,
    sparse_moe_backend: SparseMoeBackend = "reference",
    pallas_moe_config: Fp8BlockMatmulConfig | None = None,
    pallas_moe_fuse_route_weighting: bool = False,
    pallas_moe_reconstruct_down_fp32: bool = False,
    linear_backend: StageLinearBackend = "reference",
    dsa_query_backend: StageLinearBackend | None = None,
    linear_interpret: bool = False,
) -> StageLocalLayerFp8Result:
    """Execute exact DSA/IndexShare, sparse MLA, and dense or MoE MLP."""

    if indexer_kind not in ("full", "shared"):
        raise ValueError("layer indexer kind must be full or shared")
    if mlp_kind not in ("dense", "sparse"):
        raise ValueError("layer MLP kind must be dense or sparse")
    if sparse_moe_backend not in ("reference", "pallas_feature"):
        raise ValueError("layer sparse MoE backend is unknown")
    if not isinstance(pallas_moe_fuse_route_weighting, bool):
        raise ValueError("layer route-weight fusion flag must be boolean")
    if pallas_moe_fuse_route_weighting and (
        sparse_moe_backend != "pallas_feature"
    ):
        raise ValueError("route-weight fusion requires feature-Pallas MoE")
    if not isinstance(pallas_moe_reconstruct_down_fp32, bool):
        raise ValueError("FP32 routed-down reconstruction flag must be boolean")
    if pallas_moe_reconstruct_down_fp32 and (
        sparse_moe_backend != "pallas_feature"
    ):
        raise ValueError(
            "FP32 routed-down reconstruction requires feature-Pallas MoE"
        )
    if (
        pallas_moe_reconstruct_down_fp32
        and pallas_moe_fuse_route_weighting
    ):
        raise ValueError(
            "FP32 routed-down reconstruction is incompatible with fused "
            "route weighting"
        )
    if linear_backend not in ("reference", "pallas"):
        raise ValueError("layer FP8 linear backend is unknown")
    if dsa_query_backend not in (None, "reference", "pallas"):
        raise ValueError("layer DSA query backend is unknown")
    if (dsa is None) != (indexer_kind == "shared"):
        raise ValueError("full DSA weights must exist only for a full indexer")
    if (dense is None) != (mlp_kind == "sparse"):
        raise ValueError("dense weights must exist only for a dense layer")
    if (moe is None) != (mlp_kind == "dense"):
        raise ValueError("MoE weights must exist only for a sparse layer")
    if incoming_contract_valid.shape != (1,) or (
        incoming_contract_valid.dtype != jnp.bool_
    ):
        raise ValueError("incoming layer health must be one boolean row")
    if residual.shape != (1, dsa_contract.hidden_size):
        raise ValueError("layer residual disagrees with DSA hidden size")
    if residual.shape[1] != moe_contract.hidden_size:
        raise ValueError("layer residual disagrees with MoE hidden size")
    if input_norm_weight.shape != residual.shape[1:] or (
        post_attention_norm_weight.shape != residual.shape[1:]
    ):
        raise ValueError("layer norm weights disagree with hidden size")

    normalized_input = rms_norm(
        residual, input_norm_weight, epsilon=rms_norm_epsilon
    )
    q_residual = rms_norm(
        _stage_fp8_linear(
            normalized_input,
            attention.q_a_bits,
            attention.q_a_scale,
            block_shape=block_shape,
            backend=linear_backend,
            interpret=linear_interpret,
        ),
        attention.q_a_norm_weight,
        epsilon=lora_norm_epsilon,
    )

    if indexer_kind == "full":
        assert dsa is not None
        dsa_result = stage_local_dsa_fp8_mapped(
            residual,
            index_cache,
            position,
            block_tables,
            context_lengths,
            input_norm_weight,
            attention.q_a_bits,
            attention.q_a_scale,
            attention.q_a_norm_weight,
            dsa.wq_b_bits,
            dsa.wq_b_scale,
            dsa.wk_bits,
            dsa.wk_scale,
            dsa.key_norm_weight,
            dsa.key_norm_bias,
            dsa.head_weight,
            local_slot,
            axis_name=axis_name,
            contract=dsa_contract,
            cache_layout=cache_layout,
            axis_index_groups=axis_index_groups,
            block_shape=block_shape,
            rms_norm_epsilon=rms_norm_epsilon,
            lora_norm_epsilon=lora_norm_epsilon,
            precomputed_normalized=normalized_input,
            precomputed_q_residual=q_residual,
            linear_backend=linear_backend,
            dsa_query_backend=dsa_query_backend,
            linear_interpret=linear_interpret,
        )
        index_cache = dsa_result.index_cache
        selected_positions = dsa_result.selected_positions
        selected_valid_counts = dsa_result.valid_counts
        selected_scores = dsa_result.selected_scores
        dsa_valid = dsa_result.contract_valid
    else:
        selected_scores = jnp.full(
            selected_positions.shape,
            -jnp.inf,
            dtype=jnp.float32,
        )
        dsa_valid = jnp.ones((1,), dtype=jnp.bool_)

    attention_result = stage_local_index_share_fp8_mapped(
        residual,
        kv_cache,
        selected_positions,
        selected_valid_counts,
        position,
        block_tables,
        context_lengths,
        input_norm_weight,
        attention.q_a_bits,
        attention.q_a_scale,
        attention.q_a_norm_weight,
        attention.q_b_bits,
        attention.q_b_scale,
        attention.kv_a_bits,
        attention.kv_a_scale,
        attention.kv_a_norm_weight,
        attention.kv_b_bits,
        attention.kv_b_scale,
        attention.o_bits,
        attention.o_scale,
        local_slot,
        axis_name=axis_name,
        contract=mla_contract,
        cache_layout=cache_layout,
        axis_index_groups=axis_index_groups,
        block_shape=block_shape,
        rms_norm_epsilon=rms_norm_epsilon,
        lora_norm_epsilon=lora_norm_epsilon,
        rope_theta=rope_theta,
        precomputed_normalized=normalized_input,
        precomputed_q_residual=q_residual,
        linear_backend=linear_backend,
        linear_interpret=linear_interpret,
    )
    residual = attention_result.output
    if mlp_kind == "dense":
        assert dense is not None
        output = stage_local_dense_fp8_mapped(
            residual,
            post_attention_norm_weight,
            dense.gate_bits,
            dense.gate_scale,
            dense.up_bits,
            dense.up_scale,
            dense.down_bits,
            dense.down_scale,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups,
            block_shape=block_shape,
            epsilon=rms_norm_epsilon,
            linear_backend=linear_backend,
            linear_interpret=linear_interpret,
        )
        route_indices = jnp.full(
            (1, moe_contract.top_k), jnp.int32(-1), dtype=jnp.int32
        )
        route_weights = jnp.zeros(
            (1, moe_contract.top_k), dtype=jnp.float32
        )
    else:
        assert moe is not None
        normalized = rms_norm(
            residual,
            post_attention_norm_weight,
            epsilon=rms_norm_epsilon,
        )
        moe_args = (
            normalized,
            moe.router_weight,
            moe.correction_bias,
            moe.expert_gate_bits,
            moe.expert_gate_scale,
            moe.expert_up_bits,
            moe.expert_up_scale,
            moe.expert_down_bits,
            moe.expert_down_scale,
            moe.shared_gate_bits,
            moe.shared_gate_scale,
            moe.shared_up_bits,
            moe.shared_up_scale,
            moe.shared_down_bits,
            moe.shared_down_scale,
            local_slot,
        )
        if sparse_moe_backend == "reference":
            update, route_indices, route_weights = stage_local_moe_fp8_mapped(
                *moe_args,
                axis_name=axis_name,
                contract=moe_contract,
                axis_index_groups=axis_index_groups,
            )
        else:
            if pallas_moe_config is None:
                raise ValueError(
                    "feature-Pallas MoE requires an explicit tile config"
                )
            update, route_indices, route_weights = (
                stage_local_moe_pallas_feature_mapped(
                    *moe_args,
                    axis_name=axis_name,
                    contract=moe_contract,
                    axis_index_groups=axis_index_groups,
                    config=pallas_moe_config,
                    fuse_route_weighting=pallas_moe_fuse_route_weighting,
                    reconstruct_down_fp32=(
                        pallas_moe_reconstruct_down_fp32
                    ),
                )
            )
        output = residual_add(residual, update)
    return StageLocalLayerFp8Result(
        output,
        attention_result.cache,
        index_cache,
        selected_positions,
        selected_valid_counts,
        selected_scores,
        route_indices,
        route_weights,
        (
            incoming_contract_valid
            & dsa_valid
            & attention_result.contract_valid
        ),
    )


def stage_local_transformer_layer_fp8_split_mapped(
    hidden_states: Any,
    residual: Any,
    kv_cache: Any,
    index_cache: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    input_norm_weight: Any,
    post_attention_norm_weight: Any,
    attention: AttentionFp8Weights,
    dsa: DsaFp8Weights | None,
    dense: DenseFp8Weights | None,
    moe: MoeFp8Weights | None,
    incoming_contract_valid: Any,
    local_slot: Any,
    *,
    axis_name: str,
    indexer_kind: str,
    mlp_kind: str,
    dsa_contract: DsaNumericalContract = DsaNumericalContract(),
    mla_contract: MlaNumericalContract = MlaNumericalContract(),
    moe_contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    block_shape: tuple[int, int] = (128, 128),
    rms_norm_epsilon: float = 1e-5,
    lora_norm_epsilon: float = 1e-5,
    rope_theta: float = 8_000_000.0,
    sparse_moe_backend: SparseMoeBackend = "reference",
    pallas_moe_config: Fp8BlockMatmulConfig | None = None,
    pallas_moe_fuse_route_weighting: bool = False,
    pallas_moe_reconstruct_down_fp32: bool = False,
    linear_backend: StageLinearBackend = "reference",
    dsa_query_backend: StageLinearBackend | None = None,
    linear_interpret: bool = False,
) -> StageLocalSplitLayerFp8Result:
    """Execute one layer while preserving legacy hidden/residual association."""

    if indexer_kind not in ("full", "shared"):
        raise ValueError("layer indexer kind must be full or shared")
    if mlp_kind not in ("dense", "sparse"):
        raise ValueError("layer MLP kind must be dense or sparse")
    if sparse_moe_backend not in ("reference", "pallas_feature"):
        raise ValueError("layer sparse MoE backend is unknown")
    if not isinstance(pallas_moe_fuse_route_weighting, bool):
        raise ValueError("layer route-weight fusion flag must be boolean")
    if pallas_moe_fuse_route_weighting and sparse_moe_backend != "pallas_feature":
        raise ValueError("route-weight fusion requires feature-Pallas MoE")
    if not isinstance(pallas_moe_reconstruct_down_fp32, bool):
        raise ValueError("FP32 routed-down reconstruction flag must be boolean")
    if (
        pallas_moe_reconstruct_down_fp32
        and sparse_moe_backend != "pallas_feature"
    ):
        raise ValueError(
            "FP32 routed-down reconstruction requires feature-Pallas MoE"
        )
    if pallas_moe_reconstruct_down_fp32 and pallas_moe_fuse_route_weighting:
        raise ValueError(
            "FP32 routed-down reconstruction is incompatible with fused "
            "route weighting"
        )
    if linear_backend not in ("reference", "pallas"):
        raise ValueError("layer FP8 linear backend is unknown")
    if dsa_query_backend not in (None, "reference", "pallas"):
        raise ValueError("layer DSA query backend is unknown")
    if (dsa is None) != (indexer_kind == "shared"):
        raise ValueError("full DSA weights must exist only for a full indexer")
    if (dense is None) != (mlp_kind == "sparse"):
        raise ValueError("dense weights must exist only for a dense layer")
    if (moe is None) != (mlp_kind == "dense"):
        raise ValueError("MoE weights must exist only for a sparse layer")
    if incoming_contract_valid.shape != (1,) or (
        incoming_contract_valid.dtype != jnp.bool_
    ):
        raise ValueError("incoming layer health must be one boolean row")
    expected_shape = (1, dsa_contract.hidden_size)
    if hidden_states.shape != expected_shape or residual.shape != expected_shape:
        raise ValueError("split layer state disagrees with DSA hidden size")
    if hidden_states.dtype != residual.dtype:
        raise ValueError("split layer state dtypes must match")
    if hidden_states.shape[1] != moe_contract.hidden_size:
        raise ValueError("split layer state disagrees with MoE hidden size")
    if input_norm_weight.shape != hidden_states.shape[1:] or (
        post_attention_norm_weight.shape != hidden_states.shape[1:]
    ):
        raise ValueError("layer norm weights disagree with hidden size")

    normalized_input, combined_residual = fused_add_rms_norm(
        hidden_states,
        residual,
        input_norm_weight,
        epsilon=rms_norm_epsilon,
    )
    q_residual = rms_norm(
        _stage_fp8_linear(
            normalized_input,
            attention.q_a_bits,
            attention.q_a_scale,
            block_shape=block_shape,
            backend=linear_backend,
            interpret=linear_interpret,
        ),
        attention.q_a_norm_weight,
        epsilon=lora_norm_epsilon,
    )

    if indexer_kind == "full":
        assert dsa is not None
        dsa_result = stage_local_dsa_fp8_mapped(
            combined_residual,
            index_cache,
            position,
            block_tables,
            context_lengths,
            input_norm_weight,
            attention.q_a_bits,
            attention.q_a_scale,
            attention.q_a_norm_weight,
            dsa.wq_b_bits,
            dsa.wq_b_scale,
            dsa.wk_bits,
            dsa.wk_scale,
            dsa.key_norm_weight,
            dsa.key_norm_bias,
            dsa.head_weight,
            local_slot,
            axis_name=axis_name,
            contract=dsa_contract,
            cache_layout=cache_layout,
            axis_index_groups=axis_index_groups,
            block_shape=block_shape,
            rms_norm_epsilon=rms_norm_epsilon,
            lora_norm_epsilon=lora_norm_epsilon,
            precomputed_normalized=normalized_input,
            precomputed_q_residual=q_residual,
            linear_backend=linear_backend,
            dsa_query_backend=dsa_query_backend,
            linear_interpret=linear_interpret,
        )
        index_cache = dsa_result.index_cache
        selected_positions = dsa_result.selected_positions
        selected_valid_counts = dsa_result.valid_counts
        selected_scores = dsa_result.selected_scores
        dsa_valid = dsa_result.contract_valid
    else:
        selected_scores = jnp.full(
            selected_positions.shape, -jnp.inf, dtype=jnp.float32
        )
        dsa_valid = jnp.ones((1,), dtype=jnp.bool_)

    attention_result = stage_local_index_share_fp8_mapped(
        combined_residual,
        kv_cache,
        selected_positions,
        selected_valid_counts,
        position,
        block_tables,
        context_lengths,
        input_norm_weight,
        attention.q_a_bits,
        attention.q_a_scale,
        attention.q_a_norm_weight,
        attention.q_b_bits,
        attention.q_b_scale,
        attention.kv_a_bits,
        attention.kv_a_scale,
        attention.kv_a_norm_weight,
        attention.kv_b_bits,
        attention.kv_b_scale,
        attention.o_bits,
        attention.o_scale,
        local_slot,
        axis_name=axis_name,
        contract=mla_contract,
        cache_layout=cache_layout,
        axis_index_groups=axis_index_groups,
        block_shape=block_shape,
        rms_norm_epsilon=rms_norm_epsilon,
        lora_norm_epsilon=lora_norm_epsilon,
        rope_theta=rope_theta,
        precomputed_normalized=normalized_input,
        precomputed_q_residual=q_residual,
        linear_backend=linear_backend,
        linear_interpret=linear_interpret,
        add_residual=False,
    )
    normalized_mlp, post_attention_residual = fused_add_rms_norm(
        attention_result.output,
        combined_residual,
        post_attention_norm_weight,
        epsilon=rms_norm_epsilon,
    )
    if mlp_kind == "dense":
        assert dense is not None
        next_hidden = stage_local_dense_fp8_mapped(
            post_attention_residual,
            post_attention_norm_weight,
            dense.gate_bits,
            dense.gate_scale,
            dense.up_bits,
            dense.up_scale,
            dense.down_bits,
            dense.down_scale,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups,
            block_shape=block_shape,
            epsilon=rms_norm_epsilon,
            linear_backend=linear_backend,
            linear_interpret=linear_interpret,
            precomputed_normalized=normalized_mlp,
            add_residual=False,
        )
        route_indices = jnp.full(
            (1, moe_contract.top_k), jnp.int32(-1), dtype=jnp.int32
        )
        route_weights = jnp.zeros(
            (1, moe_contract.top_k), dtype=jnp.float32
        )
    else:
        assert moe is not None
        moe_args = (
            normalized_mlp,
            moe.router_weight,
            moe.correction_bias,
            moe.expert_gate_bits,
            moe.expert_gate_scale,
            moe.expert_up_bits,
            moe.expert_up_scale,
            moe.expert_down_bits,
            moe.expert_down_scale,
            moe.shared_gate_bits,
            moe.shared_gate_scale,
            moe.shared_up_bits,
            moe.shared_up_scale,
            moe.shared_down_bits,
            moe.shared_down_scale,
            local_slot,
        )
        if sparse_moe_backend == "reference":
            next_hidden, route_indices, route_weights = (
                stage_local_moe_fp8_mapped(
                    *moe_args,
                    axis_name=axis_name,
                    contract=moe_contract,
                    axis_index_groups=axis_index_groups,
                )
            )
        else:
            if pallas_moe_config is None:
                raise ValueError(
                    "feature-Pallas MoE requires an explicit tile config"
                )
            next_hidden, route_indices, route_weights = (
                stage_local_moe_pallas_feature_mapped(
                    *moe_args,
                    axis_name=axis_name,
                    contract=moe_contract,
                    axis_index_groups=axis_index_groups,
                    config=pallas_moe_config,
                    fuse_route_weighting=pallas_moe_fuse_route_weighting,
                    reconstruct_down_fp32=pallas_moe_reconstruct_down_fp32,
                )
            )
    return StageLocalSplitLayerFp8Result(
        next_hidden,
        post_attention_residual,
        attention_result.cache,
        index_cache,
        selected_positions,
        selected_valid_counts,
        selected_scores,
        route_indices,
        route_weights,
        incoming_contract_valid & dsa_valid & attention_result.contract_valid,
    )
