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
from .reference.qkv_a import (
    FusedQkvAContract,
    one_row_fused_qkv_a_convolution,
)
from .reference.rmsnorm import (
    fused_add_rms_norm,
    fused_add_rms_norm_with_auxiliary,
    rms_norm,
)
# Keep the sealed tuple-candidate import statement unchanged for its AST authority.
from .reference.rmsnorm import fused_add_rms_norm_with_compensated_auxiliary
from .stage_local import (
    STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
    StageLocalDenseFp8Ingredients,
    StageLocalDsaFp8Internals,
    StageLocalIndexShareFp8Ingredients,
    StageLocalIndexShareFp8ObservedResult,
    StageLinearBackend,
    VirtualTp32ReductionAssociation,
    _stage_fp8_linear,
    stage_local_dense_fp8_mapped,
    stage_local_dsa_fp8_mapped,
    stage_local_index_share_fp8_mapped,
    stage_local_moe_fp8_mapped,
    stage_local_moe_pallas_feature_mapped,
)

SparseMoeBackend = Literal["reference", "pallas_feature"]
AttentionProjectionBackend = Literal["separate", "fused_n82_convolution"]


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
    qkv_a_bits: Any | None = None
    qkv_a_scale: Any | None = None


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
    up_bits: Any | None
    up_scale: Any | None
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
    dsa_internals: StageLocalDsaFp8Internals


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
    dsa_internals: StageLocalDsaFp8Internals


class StageLocalSplitLayerFp8Ingredients(NamedTuple):
    """Layer-0 primitive boundaries returned only by an isolated observer."""

    normalized_input: Any
    combined_residual: Any
    attention: StageLocalIndexShareFp8Ingredients
    normalized_mlp: Any
    post_attention_residual: Any
    dense: StageLocalDenseFp8Ingredients
    next_hidden: Any


class StageLocalSplitLayerFp8ObservedResult(NamedTuple):
    """Ordinary split-layer result paired with diagnostic-only ingredients."""

    result: StageLocalSplitLayerFp8Result
    ingredients: StageLocalSplitLayerFp8Ingredients


class StageLocalSplitLayerFp8AuxiliaryResult(NamedTuple):
    """Default-off layer result retaining the device-local FP32 RMS input."""

    result: StageLocalSplitLayerFp8Result
    input_rms_fp32: Any


class StageLocalSplitLayerFp8CompensatedAuxiliaryResult(NamedTuple):
    """Layer result plus reconstructed FP32 value pending equality proof."""

    result: StageLocalSplitLayerFp8Result
    restored_input_rms_fp32: Any


def _empty_dsa_internals(
    normalized_input: Any,
    q_residual: Any,
    contract: DsaNumericalContract,
) -> StageLocalDsaFp8Internals:
    """Shape-stable sentinel for a layer that reuses IndexShare state."""

    return StageLocalDsaFp8Internals(
        normalized_input,
        q_residual,
        jnp.zeros(
            (1, contract.num_heads, contract.head_dim), dtype=jnp.float32
        ),
        jnp.zeros((1, contract.num_heads), dtype=jnp.float32),
        jnp.zeros((1, contract.head_dim), dtype=jnp.float32),
    )


def _project_attention_qkv_a(
    normalized_input: Any,
    attention: AttentionFp8Weights,
    *,
    backend: AttentionProjectionBackend,
    dsa_contract: DsaNumericalContract,
    mla_contract: MlaNumericalContract,
    block_shape: tuple[int, int],
    epsilon: float,
    linear_backend: StageLinearBackend,
    linear_interpret: bool,
    rms_accepted_schedule: bool = False,
) -> tuple[Any, Any | None]:
    """Produce q-a and, for the DB502 path, its fused kv-a companion."""

    if backend == "separate":
        if attention.qkv_a_bits is not None or attention.qkv_a_scale is not None:
            raise ValueError("separate attention cannot consume fused qkv-a state")
        if (
            attention.q_a_bits is None
            or attention.q_a_scale is None
            or attention.kv_a_bits is None
            or attention.kv_a_scale is None
        ):
            raise ValueError("separate attention requires q-a and kv-a state")
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
            epsilon=epsilon,
            accepted_schedule=rms_accepted_schedule,
        )
        return q_residual, None
    if backend != "fused_n82_convolution":
        raise ValueError("layer attention projection backend is unknown")
    if attention.qkv_a_bits is None or attention.qkv_a_scale is None:
        raise ValueError("fused attention requires packed qkv-a state")
    if block_shape != (128, 128):
        raise ValueError("fused attention requires 128x128 FP8 scale blocks")
    if any(
        value is not None
        for value in (
            attention.q_a_bits,
            attention.q_a_scale,
            attention.kv_a_bits,
            attention.kv_a_scale,
        )
    ):
        raise ValueError("fused attention must not retain separate q-a/kv-a state")
    projected = one_row_fused_qkv_a_convolution(
        normalized_input,
        attention.qkv_a_bits,
        attention.qkv_a_scale,
        attention.q_a_norm_weight,
        contract=FusedQkvAContract(
            hidden_size=dsa_contract.hidden_size,
            q_lora_rank=dsa_contract.q_lora_rank,
            kv_a_width=(
                mla_contract.kv_lora_rank
                + mla_contract.qk_rope_head_dim
            ),
            epsilon=epsilon,
        ),
    )
    return projected.q_residual, projected.kv_a_projection


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
    main_rope_table_row: Any | None = None,
    dsa_rope_table_row: Any | None = None,
    rms_accepted_schedule: bool = False,
    sparse_moe_backend: SparseMoeBackend = "reference",
    pallas_moe_config: Fp8BlockMatmulConfig | None = None,
    pallas_moe_fuse_route_weighting: bool = False,
    pallas_moe_reconstruct_down_fp32: bool = False,
    linear_backend: StageLinearBackend = "reference",
    dsa_query_backend: StageLinearBackend | None = None,
    dsa_query_weight_aliases: tuple[Any, Any, Any, Any] | None = None,
    dsa_precomputed_wk_weight: Any | None = None,
    dsa_head_key_exact_association: bool = False,
    dsa_score_precision: Literal["default", "highest"] = "highest",
    attention_projection_backend: AttentionProjectionBackend = "separate",
    linear_interpret: bool = False,
    pregathered_b512_attention: bool = False,
) -> StageLocalLayerFp8Result:
    """Execute exact DSA/IndexShare, sparse MLA, and dense or MoE MLP."""

    if indexer_kind not in ("full", "shared"):
        raise ValueError("layer indexer kind must be full or shared")
    if mlp_kind not in ("dense", "sparse"):
        raise ValueError("layer MLP kind must be dense or sparse")
    if not isinstance(pregathered_b512_attention, bool):
        raise ValueError("layer pregathered-B512 attention flag must be boolean")
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
    if not isinstance(dsa_head_key_exact_association, bool):
        raise ValueError("layer exact DSA head/key flag must be boolean")
    if dsa_rope_table_row is not None and (
        dsa_rope_table_row.shape != (dsa_contract.rotary_dim,)
        or dsa_rope_table_row.dtype != jnp.float32
    ):
        raise ValueError("layer DSA host rotary row is invalid")
    if indexer_kind == "full" and (
        dsa_head_key_exact_association
        != (dsa_precomputed_wk_weight is not None)
    ):
        raise ValueError("full indexer exact head/key wk state drifted")
    if indexer_kind == "shared" and dsa_precomputed_wk_weight is not None:
        raise ValueError("shared indexer cannot consume an external wk owner")
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
        residual,
        input_norm_weight,
        epsilon=rms_norm_epsilon,
        accepted_schedule=rms_accepted_schedule,
    )
    q_residual, current_kv = _project_attention_qkv_a(
        normalized_input,
        attention,
        backend=attention_projection_backend,
        dsa_contract=dsa_contract,
        mla_contract=mla_contract,
        block_shape=block_shape,
        epsilon=lora_norm_epsilon,
        linear_backend=linear_backend,
        linear_interpret=linear_interpret,
        rms_accepted_schedule=rms_accepted_schedule,
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
            rms_accepted_schedule=rms_accepted_schedule,
            precomputed_q_residual=q_residual,
            linear_backend=linear_backend,
            dsa_query_backend=dsa_query_backend,
            dsa_query_weight_aliases=dsa_query_weight_aliases,
            precomputed_wk_weight=dsa_precomputed_wk_weight,
            dsa_head_key_exact_association=(
                dsa_head_key_exact_association
            ),
            dsa_score_precision=dsa_score_precision,
            linear_interpret=linear_interpret,
            dsa_rope_table_row=dsa_rope_table_row,
        )
        index_cache = dsa_result.index_cache
        selected_positions = dsa_result.selected_positions
        selected_valid_counts = dsa_result.valid_counts
        selected_scores = dsa_result.selected_scores
        dsa_valid = dsa_result.contract_valid
        dsa_internals = dsa_result.internals
    else:
        selected_scores = jnp.full(
            selected_positions.shape,
            -jnp.inf,
            dtype=jnp.float32,
        )
        dsa_valid = jnp.ones((1,), dtype=jnp.bool_)
        dsa_internals = _empty_dsa_internals(
            normalized_input, q_residual, dsa_contract
        )

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
        main_rope_table_row=main_rope_table_row,
        precomputed_normalized=normalized_input,
        precomputed_q_residual=q_residual,
        precomputed_kv_a=current_kv,
        linear_backend=linear_backend,
        linear_interpret=linear_interpret,
        rms_accepted_schedule=rms_accepted_schedule,
        pregathered_b512_attention=pregathered_b512_attention,
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
            rms_accepted_schedule=rms_accepted_schedule,
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
            accepted_schedule=rms_accepted_schedule,
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
        dsa_internals,
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
    main_rope_table_row: Any | None = None,
    dsa_rope_table_row: Any | None = None,
    rms_accepted_schedule: bool = False,
    sparse_moe_backend: SparseMoeBackend = "reference",
    pallas_moe_config: Fp8BlockMatmulConfig | None = None,
    pallas_moe_fuse_route_weighting: bool = False,
    pallas_moe_reconstruct_down_fp32: bool = False,
    linear_backend: StageLinearBackend = "reference",
    dsa_query_backend: StageLinearBackend | None = None,
    dsa_query_weight_aliases: tuple[Any, Any, Any, Any] | None = None,
    dsa_precomputed_wk_weight: Any | None = None,
    dsa_head_key_exact_association: bool = False,
    dsa_score_precision: Literal["default", "highest"] = "highest",
    attention_projection_backend: AttentionProjectionBackend = "separate",
    linear_interpret: bool = False,
    reconstruct_attention_output_fp32: bool = False,
    reconstruct_dense_down_fp32: bool = False,
    virtual_tp32_reduction_association: (
        VirtualTp32ReductionAssociation | None
    ) = None,
    virtual_tp32_attention_only: bool = False,
    replicated_monolithic_attention: bool = False,
    pregathered_b512_attention: bool = False,
    dense_final_layout_convolution: bool = False,
    capture_ingredients: bool = False,
    retain_input_rms_auxiliary: bool = False,
    retain_input_rms_compensated_auxiliary: bool = False,
) -> (
    StageLocalSplitLayerFp8Result
    | StageLocalSplitLayerFp8ObservedResult
    | StageLocalSplitLayerFp8AuxiliaryResult
    | StageLocalSplitLayerFp8CompensatedAuxiliaryResult
):
    """Execute one layer while preserving legacy hidden/residual association."""

    if indexer_kind not in ("full", "shared"):
        raise ValueError("layer indexer kind must be full or shared")
    if mlp_kind not in ("dense", "sparse"):
        raise ValueError("layer MLP kind must be dense or sparse")
    if not isinstance(reconstruct_attention_output_fp32, bool):
        raise ValueError("layer attention FP32 reconstruction flag must be boolean")
    if not isinstance(reconstruct_dense_down_fp32, bool):
        raise ValueError("layer dense FP32 reconstruction flag must be boolean")
    if not isinstance(replicated_monolithic_attention, bool):
        raise ValueError(
            "layer replicated-monolithic attention flag must be boolean"
        )
    if not isinstance(pregathered_b512_attention, bool):
        raise ValueError("layer pregathered-B512 attention flag must be boolean")
    if not isinstance(dense_final_layout_convolution, bool):
        raise ValueError("layer dense final-layout flag must be boolean")
    if not isinstance(virtual_tp32_attention_only, bool):
        raise ValueError("layer virtual-TP32 attention-only flag must be boolean")
    if not isinstance(capture_ingredients, bool):
        raise ValueError("layer ingredient-capture flag must be boolean")
    if not isinstance(retain_input_rms_auxiliary, bool):
        raise ValueError("layer RMS auxiliary-retention flag must be boolean")
    if not isinstance(retain_input_rms_compensated_auxiliary, bool):
        raise ValueError(
            "layer compensated RMS auxiliary-retention flag must be boolean"
        )
    if retain_input_rms_auxiliary and retain_input_rms_compensated_auxiliary:
        raise ValueError("layer RMS auxiliary modes are mutually exclusive")
    if (
        retain_input_rms_auxiliary or retain_input_rms_compensated_auxiliary
    ) and capture_ingredients:
        raise ValueError(
            "layer RMS auxiliary retention must remain isolated from ingredient capture"
        )
    if reconstruct_dense_down_fp32 and mlp_kind != "dense":
        raise ValueError("dense FP32 reconstruction requires a dense layer")
    if virtual_tp32_reduction_association is not None and (
        reconstruct_attention_output_fp32 or reconstruct_dense_down_fp32
    ):
        raise ValueError(
            "virtual TP32 reduction cannot be combined with FP32 reconstruction"
        )
    if (
        virtual_tp32_attention_only
        and virtual_tp32_reduction_association is None
    ):
        raise ValueError(
            "attention-only virtual TP32 requires an explicit association"
        )
    if (
        virtual_tp32_reduction_association is not None
        and mlp_kind != "dense"
        and not virtual_tp32_attention_only
    ):
        raise ValueError(
            "dense virtual TP32 reduction requires a dense layer"
        )
    if capture_ingredients and (
        mlp_kind != "dense"
        or reconstruct_attention_output_fp32
        or reconstruct_dense_down_fp32
        or virtual_tp32_reduction_association is not None
        or linear_backend != "pallas"
    ):
        raise ValueError(
            "layer ingredient capture requires the production dense BF16 Pallas path"
        )
    strategy_nd_attention_projection = (
        virtual_tp32_attention_only
        and virtual_tp32_reduction_association
        == STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION
    )
    if pregathered_b512_attention and (
        replicated_monolithic_attention
        or capture_ingredients
        or reconstruct_attention_output_fp32
        or (
            virtual_tp32_reduction_association is not None
            and not strategy_nd_attention_projection
        )
    ):
        raise ValueError(
            "layer pregathered-B512 attention must remain isolated from "
            "diagnostic attention/output variants"
        )
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
    if not isinstance(dsa_head_key_exact_association, bool):
        raise ValueError("layer exact DSA head/key flag must be boolean")
    if dsa_rope_table_row is not None and (
        dsa_rope_table_row.shape != (dsa_contract.rotary_dim,)
        or dsa_rope_table_row.dtype != jnp.float32
    ):
        raise ValueError("layer DSA host rotary row is invalid")
    if indexer_kind == "full" and (
        dsa_head_key_exact_association
        != (dsa_precomputed_wk_weight is not None)
    ):
        raise ValueError("full indexer exact head/key wk state drifted")
    if indexer_kind == "shared" and dsa_precomputed_wk_weight is not None:
        raise ValueError("shared indexer cannot consume an external wk owner")
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

    if retain_input_rms_auxiliary:
        rms_candidate = fused_add_rms_norm_with_auxiliary(
            hidden_states,
            residual,
            input_norm_weight,
            epsilon=rms_norm_epsilon,
        )
        normalized_input = rms_candidate.output
        combined_residual = rms_candidate.carried_residual
        input_rms_fp32 = rms_candidate.rms_input_fp32
        restored_input_rms_fp32 = None
    elif retain_input_rms_compensated_auxiliary:
        rms_candidate = fused_add_rms_norm_with_compensated_auxiliary(
            hidden_states,
            residual,
            input_norm_weight,
            epsilon=rms_norm_epsilon,
        )
        normalized_input = rms_candidate.output
        combined_residual = rms_candidate.carried_residual
        input_rms_fp32 = None
        restored_input_rms_fp32 = rms_candidate.restored_rms_input_fp32
    else:
        normalized_input, combined_residual = fused_add_rms_norm(
            hidden_states,
            residual,
            input_norm_weight,
            epsilon=rms_norm_epsilon,
            accepted_schedule=rms_accepted_schedule,
        )
        input_rms_fp32 = None
        restored_input_rms_fp32 = None
    q_residual, current_kv = _project_attention_qkv_a(
        normalized_input,
        attention,
        backend=attention_projection_backend,
        dsa_contract=dsa_contract,
        mla_contract=mla_contract,
        block_shape=block_shape,
        epsilon=lora_norm_epsilon,
        linear_backend=linear_backend,
        linear_interpret=linear_interpret,
        rms_accepted_schedule=rms_accepted_schedule,
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
            rms_accepted_schedule=rms_accepted_schedule,
            precomputed_q_residual=q_residual,
            linear_backend=linear_backend,
            dsa_query_backend=dsa_query_backend,
            dsa_query_weight_aliases=dsa_query_weight_aliases,
            precomputed_wk_weight=dsa_precomputed_wk_weight,
            dsa_head_key_exact_association=(
                dsa_head_key_exact_association
            ),
            dsa_score_precision=dsa_score_precision,
            linear_interpret=linear_interpret,
            dsa_rope_table_row=dsa_rope_table_row,
        )
        index_cache = dsa_result.index_cache
        selected_positions = dsa_result.selected_positions
        selected_valid_counts = dsa_result.valid_counts
        selected_scores = dsa_result.selected_scores
        dsa_valid = dsa_result.contract_valid
        dsa_internals = dsa_result.internals
    else:
        selected_scores = jnp.full(
            selected_positions.shape, -jnp.inf, dtype=jnp.float32
        )
        dsa_valid = jnp.ones((1,), dtype=jnp.bool_)
        dsa_internals = _empty_dsa_internals(
            normalized_input, q_residual, dsa_contract
        )

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
        main_rope_table_row=main_rope_table_row,
        precomputed_normalized=normalized_input,
        precomputed_q_residual=q_residual,
        precomputed_kv_a=current_kv,
        linear_backend=linear_backend,
        linear_interpret=linear_interpret,
        rms_accepted_schedule=rms_accepted_schedule,
        add_residual=False,
        reconstruct_output_fp32=reconstruct_attention_output_fp32,
        virtual_tp32_reduction_association=(
            virtual_tp32_reduction_association
        ),
        replicated_monolithic_attention=replicated_monolithic_attention,
        pregathered_b512_attention=pregathered_b512_attention,
        capture_ingredients=capture_ingredients,
    )
    if capture_ingredients:
        assert isinstance(
            attention_result, StageLocalIndexShareFp8ObservedResult
        )
        attention_ingredients = attention_result.ingredients
        attention_result = attention_result.result
    else:
        attention_ingredients = None
    normalized_mlp, post_attention_residual = fused_add_rms_norm(
        attention_result.output,
        combined_residual,
        post_attention_norm_weight,
        epsilon=rms_norm_epsilon,
        accepted_schedule=rms_accepted_schedule,
    )
    if mlp_kind == "dense":
        assert dense is not None
        dense_result = stage_local_dense_fp8_mapped(
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
            rms_accepted_schedule=rms_accepted_schedule,
            linear_backend=linear_backend,
            linear_interpret=linear_interpret,
            precomputed_normalized=normalized_mlp,
            add_residual=False,
            reconstruct_down_fp32=reconstruct_dense_down_fp32,
            virtual_tp32_reduction_association=(
                STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION
                if dense_final_layout_convolution
                else None
                if virtual_tp32_attention_only
                else virtual_tp32_reduction_association
            ),
            final_layout_convolution=dense_final_layout_convolution,
            capture_ingredients=capture_ingredients,
        )
        if capture_ingredients:
            next_hidden = dense_result.output
            dense_ingredients = dense_result.ingredients
        else:
            next_hidden = dense_result
            dense_ingredients = None
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
    result = StageLocalSplitLayerFp8Result(
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
        dsa_internals,
    )
    if retain_input_rms_auxiliary:
        assert input_rms_fp32 is not None
        return StageLocalSplitLayerFp8AuxiliaryResult(result, input_rms_fp32)
    if retain_input_rms_compensated_auxiliary:
        assert restored_input_rms_fp32 is not None
        return StageLocalSplitLayerFp8CompensatedAuxiliaryResult(
            result,
            restored_input_rms_fp32,
        )
    if not capture_ingredients:
        return result
    assert attention_ingredients is not None
    assert dense_ingredients is not None
    return StageLocalSplitLayerFp8ObservedResult(
        result,
        StageLocalSplitLayerFp8Ingredients(
            normalized_input,
            combined_residual,
            attention_ingredients,
            normalized_mlp,
            post_attention_residual,
            dense_ingredients,
            next_hidden,
        ),
    )
