"""P3 private wide IndexShare prefix and cache-write mirrors.

Only the row guards differ from the frozen functions. Wider execution is limited
to sparse IndexShare prefixes: full DSA producers/M64 repair and dense layers keep
the original <=32-row schedule. Explicit AST checks pin these narrow changes.
"""
from typing import Any, Callable
import jax.numpy as jnp
from jax import lax
from ..greenfield.kernels.pallas import SparseMlaConfig
from ..greenfield.kernels.reference.attention import MlaNumericalContract, StageLocalKvLayout
from ..greenfield.kernels.reference.dsa import DsaNumericalContract
from ..greenfield.kernels.reference.moe import GlmMoeNumericalContract
from ..greenfield.kernels.prefill_cache import PrefillCacheWrite
from ..greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped
from ..greenfield.kernels.ws32_layer import (Ws32AttentionWeights, Ws32DenseWeights,
    Ws32DsaWeights, Ws32MoeWeights, Ws32QkvAWeights)
from ..greenfield.kernels.ws32_prefill_layer import (Ws32PrefillLayerResult,
    Ws32PrefillPrefixResult, ws32_prefill_mlp_mapped)
from ..greenfield.kernels.ws32_prefill_attention import (
    ws32_prefill_prepare_attention_mapped, ws32_prefill_index_share_attention_mapped)
from ..greenfield.kernels.ws32_prefill_dsa import ws32_prefill_dsa_mapped


def _require_block(value: Any) -> int:
    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("prefill attention requires WS32 expert8/feature4 mesh")
    if (
        value.ndim != 2
        or not 1 <= value.shape[0] <= 128
        or value.shape[1] <= 0
        or value.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill attention requires1..128 BF16 feature rows")
    return value.shape[0]


def write_prefill_cache_block(
    cache_local: Any,
    rows: Any,
    block_table: Any,
    position_offset: Any,
    valid_rows: Any,
    owner_index: Any,
    *,
    layout: StageLocalKvLayout,
) -> PrefillCacheWrite:
    """Write owned live rows, or leave the entire cache unchanged on bad input.

    Inputs use a single shared page table. Empty blocks are healthy no-ops if
    metadata/prefix mapping is valid. Nonfinite padded rows are ignored. Caller
    must bind offset to its committed append frontier and gate all-chip health.
    Apply separately to KV/unrepaired index destinations. Never alias/promote a
    repaired destination into the active prompt index cache before prefill ends.
    """
    if (
        cache_local.ndim != 3
        or min(cache_local.shape) <= 0
        or cache_local.shape[1:]
        != (layout.local_rows_per_page, layout.packed_cache_width)
        or cache_local.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill cache must match BF16 striped owner layout")
    if (
        rows.ndim != 2
        or not 1 <= rows.shape[0] <= 128
        or rows.shape[1] != layout.packed_cache_width
        or rows.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill cache block requires1..128 BF16 rows")
    if (
        block_table.ndim != 2
        or block_table.shape[0] != 1
        or block_table.shape[1] <= 0
        or block_table.dtype != jnp.int32
    ):
        raise ValueError("prefill cache needs one shared int32 page table")
    for value in (position_offset, valid_rows, owner_index):
        if value.shape != () or value.dtype != jnp.int32:
            raise ValueError("prefill offset/count/owner must be int32 scalars")
    capacity = block_table.shape[1] * layout.logical_page_size
    flat_count = cache_local.shape[0] * layout.local_rows_per_page
    if max(capacity, flat_count) >= 2147483647:
        raise ValueError("prefill cache address range must fit positive int32")
    count = jnp.clip(valid_rows, 0, rows.shape[0])
    # Safe end/addition even for adversarial INT_MIN/INT_MAX metadata.
    safe_start = jnp.clip(position_offset, 0, capacity)
    count_ok = (valid_rows >= 0) & (valid_rows <= rows.shape[0])
    span_ok = (position_offset >= 0) & (position_offset <= capacity - count)
    safe_count = jnp.minimum(count, capacity - safe_start)
    end = safe_start + safe_count
    required_pages = end // layout.logical_page_size + (
        end % layout.logical_page_size != 0
    ).astype(jnp.int32)
    page_live = jnp.arange(block_table.shape[1], dtype=jnp.int32) < required_pages
    page_ids = block_table[0]
    pages_ok = jnp.all(
        ~page_live | ((page_ids >= 0) & (page_ids < cache_local.shape[0]))
    )
    ordered = jnp.sort(jnp.where(page_live, page_ids, jnp.iinfo(jnp.int32).max))
    unique = jnp.all(
        jnp.where(
            jnp.arange(1, block_table.shape[1]) < required_pages,
            ordered[1:] != ordered[:-1],
            True,
        )
    )
    live = jnp.arange(rows.shape[0], dtype=jnp.int32) < count
    finite = jnp.all(jnp.isfinite(rows) | ~live[:, None])
    valid = (
        count_ok
        & span_ok
        & pages_ok
        & unique
        & finite
        & (owner_index >= 0)
        & (owner_index < layout.local_parallel_size)
    )
    # Padded rows have a safe address without offset+row overflow.
    delta = jnp.minimum(
        jnp.arange(rows.shape[0], dtype=jnp.int32), jnp.maximum(safe_count - 1, 0)
    )
    positions = jnp.minimum(safe_start, capacity - 1) + delta
    pages = page_ids[positions // layout.logical_page_size]
    local_row = positions % layout.local_rows_per_page
    owned = live & (layout.owner(positions) == owner_index)
    flat_index = (
        jnp.clip(pages, 0, cache_local.shape[0] - 1) * layout.local_rows_per_page
        + local_row
    )
    # Positive sentinel; mode=drop is explicit and masked indices may repeat.
    targets = jnp.where(owned, flat_index, flat_count)
    clean_rows = jnp.where(live[:, None], rows, jnp.zeros((), rows.dtype))

    def write(cache):
        flat = cache.reshape(flat_count, layout.packed_cache_width)
        return flat.at[targets].set(clean_rows, mode="drop").reshape(cache.shape)

    updated = lax.cond(valid, write, lambda cache: cache, cache_local)
    row_valid = live & valid
    causal_lengths = jnp.where(row_valid, positions + 1, 0)
    return PrefillCacheWrite(updated, valid, causal_lengths, row_valid)


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
        or not 1 <= hidden_update_local.shape[0] <= 128
        or (
            hidden_update_local.dtype != jnp.bfloat16
            or carried_residual_local.shape != hidden_update_local.shape
            or carried_residual_local.dtype != jnp.bfloat16
            or hidden_update_local.shape[1] * 4 != dsa_contract.hidden_size
        )
    ):
        raise ValueError("prefill split residual geometry drifted")
    rows = hidden_update_local.shape[0]
    if rows > 32 and (not prefix_only or dsa_weights is not None or dense_weights is not None):
        raise ValueError('wide rows require a sparse IndexShare prefix')
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
