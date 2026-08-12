"""Raw-FP8 batch-one layer bodies for a topology-local stage group.

These bodies run *inside* a surrounding global ``shard_map``.  They therefore
accept already-local final-owner shards and an explicit local slot rather than
constructing a nested mesh.  Every repeated collective is restricted to the
provided stage-local axis groups.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, NamedTuple, Sequence

import jax
import jax.numpy as jnp
from jax import lax

from .pallas import (
    Fp8BlockMatmulConfig,
    SparseMlaConfig,
    fp8_block_matmul,
    fp8_block_matmul_f32,
    fp8_block_up_gate,
    fp8_fused_block_swiglu,
    fp8_fused_selected_moe,
    fp32_to_bf16_pallas_boundary,
    fp8_structured_kv_b_q_absorb,
    fp8_structured_kv_b_value,
    pregathered_sparse_mla_pallas,
    stage_local_sparse_mla_kernel,
)
from .reference.attention import (
    MlaNumericalContract,
    SparseAttentionResult,
    StageLocalKvLayout,
    canonicalize_selected_positions,
    combine_stage_local_attention,
    gather_stage_local_selected_kv,
    gather_stage_local_selected_kv_aligned,
)
from .reference.dsa import (
    DsaNumericalContract,
    SelectedPositions,
    dsa_index_keys,
    dsa_index_keys_from_projection,
    dsa_scores,
    local_topk_candidates,
    merge_topk_candidates_with_scores,
)
from .reference.fp8 import dequantize_fp8_bits_block_weight
from .reference.linear import linear, residual_add, silu
from .reference.moe import GlmMoeNumericalContract, route_glm_noaux_tc
from .reference.rmsnorm import rms_norm
from .reference.rotary import (
    apply_rotary,
    apply_rotary_fp32_final_round,
    rotary_cos_sin,
)


class StageLocalDsaFp8Result(NamedTuple):
    """Updated local key cache and compact exact DSA selection."""

    index_cache: Any
    selected_positions: Any
    valid_counts: Any
    selected_scores: Any
    contract_valid: Any
    internals: "StageLocalDsaFp8Internals"


class StageLocalDsaFp8Internals(NamedTuple):
    """Already-live full-indexer values exposed only to diagnostics."""

    normalized_hidden: Any
    q_a_state: Any
    query: Any
    head_weights: Any
    current_key: Any


class StageLocalIndexShareFp8Result(NamedTuple):
    """Updated local KV cache and post-attention residual."""

    output: Any
    cache: Any
    contract_valid: Any


class StageLocalIndexShareFp8Ingredients(NamedTuple):
    """Unsaturated attention boundaries exposed only by a diagnostic replay."""

    current_cache_row: Any
    owner_selected_positions: Any
    owner_selected_valid_counts: Any
    owner_selected_cache_values: Any
    owner_selected_cache_valid: Any
    sparse_partial_output: Any
    sparse_partial_logsumexp: Any
    sparse_partial_valid: Any
    combined_attention_output: Any
    combined_attention_logsumexp: Any
    combined_attention_valid: Any
    value_states: Any
    output_input: Any
    virtual_output_partials: Any
    local_output_update: Any
    reduced_output_update: Any


class StageLocalIndexShareFp8ObservedResult(NamedTuple):
    """Ordinary attention result paired with diagnostic-only ingredients."""

    result: StageLocalIndexShareFp8Result
    ingredients: StageLocalIndexShareFp8Ingredients


class StageLocalDenseFp8Ingredients(NamedTuple):
    """Unsaturated dense-MLP boundaries exposed only by a diagnostic replay."""

    normalized_input: Any
    virtual_down_partials: Any
    local_down_update: Any
    reduced_down_update: Any


class StageLocalDenseFp8ObservedResult(NamedTuple):
    """Ordinary dense result paired with diagnostic-only ingredients."""

    output: Any
    ingredients: StageLocalDenseFp8Ingredients


StageLinearBackend = Literal["reference", "pallas"]
VirtualTp32ReductionAssociation = Literal[
    "dcp_then_model_sequential_bf16",
    "dcp_then_model_pairwise_bf16",
    "model_then_dcp_sequential_bf16",
    "model_then_dcp_pairwise_bf16",
    "strategy_nd_row0_bf16",
]
VIRTUAL_TP32_REDUCTION_ASSOCIATIONS: tuple[
    VirtualTp32ReductionAssociation, ...
] = (
    "dcp_then_model_sequential_bf16",
    "dcp_then_model_pairwise_bf16",
    "model_then_dcp_sequential_bf16",
    "model_then_dcp_pairwise_bf16",
)
STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION: VirtualTp32ReductionAssociation = (
    "strategy_nd_row0_bf16"
)
_SUPPORTED_VIRTUAL_TP32_REDUCTION_ASSOCIATIONS = (
    *VIRTUAL_TP32_REDUCTION_ASSOCIATIONS,
    STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION,
)
_VIRTUAL_DCP_SHARDS_PER_PP8_OWNER = 8

# DB533 recovered the accepted M32 model-axis placement.  Indexing the 32
# model-ordered projection partials by this inverse yields physical device-id
# order, where device_id = x + 2*y + 8*z on the accepted 2x4x4 slice.
_STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE = (
    0,
    16,
    4,
    20,
    8,
    24,
    12,
    28,
    1,
    17,
    5,
    21,
    9,
    25,
    13,
    29,
    2,
    18,
    6,
    22,
    10,
    26,
    14,
    30,
    3,
    19,
    7,
    23,
    11,
    27,
    15,
    31,
)


def _virtual_attention_output_partials(
    output_input: Any,
    o_bits: Any,
    o_scale: Any,
    *,
    block_shape: tuple[int, int],
    linear_interpret: bool,
) -> Any:
    """Return eight already-rounded K512 output-projection partials."""

    hidden = o_bits.shape[0]
    if (
        hidden != 6144
        or output_input.shape != (1, 4096)
        or block_shape != (128, 128)
    ):
        raise ValueError(
            "virtual attention partials require the exact GLM PP8 geometry"
        )
    virtual_contraction = (
        output_input.shape[1] // _VIRTUAL_DCP_SHARDS_PER_PP8_OWNER
    )
    virtual_scale_contraction = virtual_contraction // block_shape[1]
    return jnp.stack(
        tuple(
            fp8_block_matmul(
                output_input[
                    :,
                    shard * virtual_contraction : (shard + 1)
                    * virtual_contraction,
                ],
                o_bits[
                    :,
                    shard * virtual_contraction : (shard + 1)
                    * virtual_contraction,
                ],
                o_scale[
                    :,
                    shard
                    * virtual_scale_contraction : (shard + 1)
                    * virtual_scale_contraction,
                ],
                config=Fp8BlockMatmulConfig(
                    block_shape=block_shape,
                    output_tile=block_shape[0],
                    contraction_tile=block_shape[1],
                ),
                interpret=linear_interpret,
            )
            for shard in range(_VIRTUAL_DCP_SHARDS_PER_PP8_OWNER)
        ),
        axis=0,
    )


def _virtual_dense_down_partials(
    normalized: Any,
    gate_bits: Any,
    gate_scale: Any,
    up_bits: Any,
    up_scale: Any,
    down_bits: Any,
    down_scale: Any,
    *,
    block_shape: tuple[int, int],
    linear_interpret: bool,
) -> Any:
    """Return eight already-rounded I384 fused-SwiGLU down partials."""

    if (
        normalized.shape != (1, 6144)
        or gate_bits.shape != (3072, 6144)
        or block_shape != (128, 128)
    ):
        raise ValueError(
            "virtual dense partials require the exact GLM PP8 geometry"
        )
    virtual_intermediate = (
        gate_bits.shape[0] // _VIRTUAL_DCP_SHARDS_PER_PP8_OWNER
    )
    virtual_scale_intermediate = virtual_intermediate // block_shape[0]
    return jnp.stack(
        tuple(
            fp8_fused_block_swiglu(
                normalized,
                gate_bits[
                    shard * virtual_intermediate : (shard + 1)
                    * virtual_intermediate,
                    :,
                ],
                gate_scale[
                    shard
                    * virtual_scale_intermediate : (shard + 1)
                    * virtual_scale_intermediate,
                    :,
                ],
                up_bits[
                    shard * virtual_intermediate : (shard + 1)
                    * virtual_intermediate,
                    :,
                ],
                up_scale[
                    shard
                    * virtual_scale_intermediate : (shard + 1)
                    * virtual_scale_intermediate,
                    :,
                ],
                down_bits[
                    :,
                    shard * virtual_intermediate : (shard + 1)
                    * virtual_intermediate,
                ],
                down_scale[
                    :,
                    shard
                    * virtual_scale_intermediate : (shard + 1)
                    * virtual_scale_intermediate,
                ],
                config=Fp8BlockMatmulConfig(
                    block_shape=block_shape,
                    output_tile=block_shape[0],
                    contraction_tile=block_shape[1],
                ),
                interpret=linear_interpret,
            )
            for shard in range(_VIRTUAL_DCP_SHARDS_PER_PP8_OWNER)
        ),
        axis=0,
    )


def _sum_virtual_dcp_bf16_partials(
    partials: Any,
    *,
    pairwise: bool,
) -> Any:
    """Sum eight already-rounded legacy DCP partials in a pinned BF16 tree."""

    if partials.ndim != 3 or partials.shape[0] != (
        _VIRTUAL_DCP_SHARDS_PER_PP8_OWNER
    ):
        raise ValueError("virtual TP32 reduction requires eight rank-two partials")
    if partials.dtype != jnp.bfloat16:
        raise ValueError("virtual TP32 partials must already be rounded BF16")
    if not isinstance(pairwise, bool):
        raise ValueError("virtual TP32 reduction-tree flag must be boolean")

    def add(left: Any, right: Any) -> Any:
        # The explicit barrier prevents XLA from flattening the requested
        # legacy-association discriminator into one wider reduction.
        return lax.optimization_barrier(
            (left + right).astype(jnp.bfloat16)
        )

    values = tuple(partials[index] for index in range(partials.shape[0]))
    if not pairwise:
        result = values[0]
        for value in values[1:]:
            result = add(result, value)
        return result

    while len(values) > 1:
        if len(values) % 2:
            raise ValueError("pairwise virtual TP32 reduction lost a power-of-two level")
        values = tuple(
            add(values[index], values[index + 1])
            for index in range(0, len(values), 2)
        )
    return values[0]


def _strategy_nd_row0_bf16_reduce(model_partials: Any) -> Any:
    """Replay DB533's exact accepted M32 reduction for live decode row zero."""

    if model_partials.shape != (32, 1, 6144):
        raise ValueError(
            "StrategyND row-zero reduction requires 32 model partials"
        )
    if model_partials.dtype != jnp.bfloat16:
        raise ValueError(
            "StrategyND row-zero partials must already be rounded BF16"
        )

    def add(left: Any, right: Any) -> Any:
        return lax.optimization_barrier(
            (left + right).astype(jnp.bfloat16)
        )

    def reduce_four(values: Any, *, cross: bool) -> Any:
        if cross:
            return add(add(values[0], values[3]), add(values[1], values[2]))
        return add(add(values[0], values[1]), add(values[2], values[3]))

    physical = jnp.stack(
        tuple(
            model_partials[model_position]
            for model_position in (
                _STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE
            )
        ),
        axis=0,
    )
    # Physical ids are x-fastest in [z, y, x].  DB533 row zero uses phases
    # y -> x -> z, hence the explicit transpose to [y, x, z, row, hidden].
    physical_y_x_z = jnp.transpose(
        physical.reshape(4, 4, 2, 1, 6144),
        (1, 2, 0, 3, 4),
    )
    y_reduced = jnp.concatenate(
        (
            reduce_four(physical_y_x_z[..., :2048], cross=False),
            reduce_four(
                physical_y_x_z[..., 2048:4096], cross=True
            ),
            reduce_four(physical_y_x_z[..., 4096:], cross=False),
        ),
        axis=-1,
    )
    x_reduced = add(y_reduced[0], y_reduced[1])
    return jnp.concatenate(
        tuple(
            reduce_four(
                x_reduced[..., start : start + 256],
                cross=bool((start // 256) % 2),
            )
            for start in range(0, 6144, 256)
        ),
        axis=-1,
    )


def _reduce_strategy_nd_row0_bf16_partials(
    local_partials: Any,
    *,
    axis_name: str,
    groups: tuple[tuple[int, ...], ...],
) -> Any:
    """Gather four PP8 owners, then replay the accepted row-zero M32 tree."""

    if local_partials.shape != (8, 1, 6144):
        raise ValueError(
            "StrategyND row-zero reduction requires eight local partials"
        )
    if local_partials.dtype != jnp.bfloat16:
        raise ValueError(
            "StrategyND row-zero local partials must be rounded BF16"
        )
    with jax.named_scope("greenfield_strategy_nd_row0_association"):
        with jax.named_scope(
            "greenfield_strategy_nd_row0_association_gather"
        ):
            gathered = lax.all_gather(
                local_partials,
                axis_name=axis_name,
                axis=0,
                tiled=False,
                axis_index_groups=groups,
            )
        return _strategy_nd_row0_bf16_reduce(
            gathered.reshape(32, 1, 6144)
        )


def _reduce_virtual_tp32_bf16_partials(
    local_partials: Any,
    *,
    axis_name: str,
    groups: tuple[tuple[int, ...], ...] | None,
    association: VirtualTp32ReductionAssociation,
) -> Any:
    """Reduce 8 virtual DCP shards x 4 physical model owners, locally only."""

    if association not in _SUPPORTED_VIRTUAL_TP32_REDUCTION_ASSOCIATIONS:
        raise ValueError("virtual TP32 reduction association is unknown")
    if groups is None or any(len(group) != 4 for group in groups):
        raise ValueError("virtual TP32 reduction requires explicit LP4 groups")
    if association == STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION:
        return _reduce_strategy_nd_row0_bf16_partials(
            local_partials,
            axis_name=axis_name,
            groups=groups,
        )
    dcp_first = association.startswith("dcp_then_model_")
    pairwise = association.endswith("_pairwise_bf16")
    with jax.named_scope(f"greenfield_virtual_tp32_{association}"):
        if dcp_first:
            local_value = _sum_virtual_dcp_bf16_partials(
                local_partials,
                pairwise=pairwise,
            )
            return lax.psum(
                local_value,
                axis_name=axis_name,
                axis_index_groups=groups,
            )
        model_partials = lax.psum(
            local_partials,
            axis_name=axis_name,
            axis_index_groups=groups,
        )
        return _sum_virtual_dcp_bf16_partials(
            model_partials,
            pairwise=pairwise,
        )


def _stage_fp8_linear(
    hidden: Any,
    weight_bits: Any,
    scale: Any,
    *,
    block_shape: tuple[int, int],
    backend: StageLinearBackend,
    interpret: bool,
) -> Any:
    """Apply one final-owner FP8 linear without changing the fallback."""

    if backend == "reference":
        return linear(
            hidden,
            dequantize_fp8_bits_block_weight(
                weight_bits, scale, block_shape=block_shape
            ),
        )
    if backend != "pallas":
        raise ValueError("stage-local FP8 linear backend is unknown")
    return fp8_block_matmul(
        hidden,
        weight_bits,
        scale,
        config=Fp8BlockMatmulConfig(
            block_shape=block_shape,
            output_tile=block_shape[0],
            contraction_tile=block_shape[1],
        ),
        interpret=interpret,
    )


def _axis_groups(
    groups: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...] | None:
    if groups is None:
        return None
    value = tuple(tuple(int(rank) for rank in group) for group in groups)
    if not value or any(not group for group in value):
        raise ValueError("stage-local axis groups must be non-empty")
    return value


def _require_decode_metadata(
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    local_slot: Any,
    *,
    layout: StageLocalKvLayout,
    physical_page_count: int,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return safe current-row indices plus a device-resident health bit."""

    if position.shape != (1,) or not jnp.issubdtype(position.dtype, jnp.integer):
        raise ValueError("decode position must be one integer row")
    if block_tables.ndim != 2 or block_tables.shape[0] != 1:
        raise ValueError("block tables must contain one decode row")
    if block_tables.shape[1] == 0 or block_tables.dtype != jnp.int32:
        raise ValueError("block tables must expose int32 logical pages")
    if context_lengths.shape != (1,) or context_lengths.dtype != jnp.int32:
        raise ValueError("context lengths must contain one int32 row")
    if local_slot.shape != () or not jnp.issubdtype(local_slot.dtype, jnp.integer):
        raise ValueError("local_slot must be an integer scalar")
    if physical_page_count <= 0:
        raise ValueError("physical page count must be positive")

    capacity = block_tables.shape[1] * layout.logical_page_size
    current = position[0].astype(jnp.int32)
    length = context_lengths[0]
    safe_current = jnp.clip(current, jnp.int32(0), jnp.int32(capacity - 1))
    logical_page = safe_current // jnp.int32(layout.logical_page_size)
    safe_logical_page = jnp.clip(
        logical_page, jnp.int32(0), jnp.int32(block_tables.shape[1] - 1)
    )
    physical_page = block_tables[0, safe_logical_page]
    physical_ok = (physical_page >= 0) & (physical_page < physical_page_count)
    safe_physical_page = jnp.clip(
        physical_page, jnp.int32(0), jnp.int32(physical_page_count - 1)
    )
    within_page = safe_current % jnp.int32(layout.logical_page_size)
    target_owner = within_page // jnp.int32(layout.local_rows_per_page)
    local_row = within_page % jnp.int32(layout.local_rows_per_page)

    page_ids = block_tables[0]
    page_slots = jnp.arange(block_tables.shape[1], dtype=jnp.int32)
    required_pages = (
        jnp.maximum(length, jnp.int32(0))
        + jnp.int32(layout.logical_page_size - 1)
    ) // jnp.int32(layout.logical_page_size)
    live_pages = page_slots < required_pages
    page_table_ok = jnp.all(
        jnp.where(
            live_pages,
            (page_ids >= 0) & (page_ids < physical_page_count),
            True,
        )
    )
    ordered_live_pages = jnp.sort(
        jnp.where(live_pages, page_ids, jnp.iinfo(jnp.int32).max)
    )
    adjacent_slots = jnp.arange(
        1, block_tables.shape[1], dtype=jnp.int32
    )
    page_table_unique = jnp.all(
        jnp.where(
            adjacent_slots < required_pages,
            ordered_live_pages[1:] != ordered_live_pages[:-1],
            True,
        )
    )
    metadata_valid = (
        (length > 0)
        & (length <= capacity)
        & (current == length - 1)
        & (current >= 0)
        & (local_slot >= 0)
        & (local_slot < layout.local_parallel_size)
        & physical_ok
        & page_table_ok
        & page_table_unique
    )
    return (
        safe_physical_page,
        local_row,
        target_owner,
        safe_current,
        metadata_valid[None],
    )


def _local_dsa_query_from_projection(
    projected_query: Any,
    normalized: Any,
    head_weight: Any,
    position: Any,
    *,
    contract: DsaNumericalContract,
) -> tuple[Any, Any]:
    if projected_query.ndim != 2 or projected_query.shape[0] != 1:
        raise ValueError("local DSA query projection must contain one row")
    if projected_query.shape[1] % contract.head_dim:
        raise ValueError("local DSA query width must divide into exact heads")
    local_heads = projected_query.shape[1] // contract.head_dim
    if projected_query.dtype != jnp.float32:
        raise ValueError("local DSA query projection must remain FP32")
    if head_weight.shape != (local_heads, contract.hidden_size):
        raise ValueError("local DSA head-weight shard has an invalid shape")
    with jax.default_matmul_precision("highest"):
        query = projected_query.reshape(1, local_heads, contract.head_dim)
        weights = linear(
            normalized, head_weight, output_dtype=jnp.float32
        ) * jnp.float32(contract.num_heads**-0.5)
    cos, sin = rotary_cos_sin(
        position,
        rotary_dim=contract.rotary_dim,
        theta=contract.theta,
        dtype=jnp.float32,
    )
    rotated = apply_rotary(
        query[..., : contract.rotary_dim],
        cos[:, None, :],
        sin[:, None, :],
        interleaved=contract.interleaved_rotary,
    )
    return (
        jnp.concatenate(
            (rotated, query[..., contract.rotary_dim :]), axis=-1
        ).astype(jnp.float32),
        weights.astype(jnp.float32),
    )


def _local_dsa_query(
    q_residual: Any,
    normalized: Any,
    query_weight: Any,
    head_weight: Any,
    position: Any,
    *,
    contract: DsaNumericalContract,
) -> tuple[Any, Any]:
    local_heads = query_weight.shape[0] // contract.head_dim
    if query_weight.shape != (
        local_heads * contract.head_dim,
        contract.q_lora_rank,
    ):
        raise ValueError("local DSA query shard has an invalid shape")
    with jax.default_matmul_precision("highest"):
        projected_query = linear(
            q_residual, query_weight, output_dtype=jnp.float32
        )
    return _local_dsa_query_from_projection(
        projected_query,
        normalized,
        head_weight,
        position,
        contract=contract,
    )


def _local_dsa_query_tuple4_exact(
    q_residual: Any,
    normalized: Any,
    query_weight_aliases: tuple[Any, Any, Any, Any],
    head_weight: Any,
    position: Any,
    *,
    contract: DsaNumericalContract,
) -> tuple[Any, Any]:
    """Preserve the accepted TPU-v4 four-reduction query association."""

    expected_shape = (
        head_weight.shape[0] * contract.head_dim,
        contract.q_lora_rank,
    )
    if len(query_weight_aliases) != 4:
        raise ValueError("exact DSA query association requires four aliases")
    if any(
        weight.shape != expected_shape or weight.dtype != jnp.float32
        for weight in query_weight_aliases
    ):
        raise ValueError("exact DSA query aliases must be local FP32 owners")
    # DB522 proves the fused q-a producer must complete its BF16 boundary.
    query_input = lax.optimization_barrier(q_residual)
    # DB525 proves that four entry aliases and one grouped result barrier make
    # TPU v4 retain the accepted 16-KiB reduction association. Only the first
    # result is live; the other three are association anchors, not extra state.
    grouped = jnp.stack(
        tuple(
            linear(query_input, weight, output_dtype=jnp.float32)
            for weight in query_weight_aliases
        )
    )
    projected_query = lax.optimization_barrier(grouped)[0]
    return _local_dsa_query_from_projection(
        projected_query,
        normalized,
        head_weight,
        position,
        contract=contract,
    )


def _reshape_gathered_heads(
    value: Any,
    *,
    num_heads: int,
    head_width: int,
) -> Any:
    return jnp.transpose(value, (1, 0, 2, 3)).reshape(
        1, num_heads, head_width
    )


def stage_local_dsa_fp8_mapped(
    residual: Any,
    index_cache: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    input_norm_weight: Any,
    q_a_bits: Any,
    q_a_scale: Any,
    q_a_norm_weight: Any,
    wq_b_bits: Any,
    wq_b_scale: Any,
    wk_bits: Any,
    wk_scale: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    head_weight: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: DsaNumericalContract = DsaNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    block_shape: tuple[int, int] = (128, 128),
    rms_norm_epsilon: float = 1e-5,
    lora_norm_epsilon: float = 1e-5,
    precomputed_normalized: Any | None = None,
    precomputed_q_residual: Any | None = None,
    linear_backend: StageLinearBackend = "reference",
    dsa_query_backend: StageLinearBackend | None = None,
    dsa_query_weight_aliases: tuple[Any, Any, Any, Any] | None = None,
    precomputed_wk_weight: Any | None = None,
    dsa_head_key_exact_association: bool = False,
    dsa_score_precision: Literal["default", "highest"] = "highest",
    linear_interpret: bool = False,
) -> StageLocalDsaFp8Result:
    """Write one BF16 index key, score local pages, and merge exact top-k.

    The function is intended for a surrounding global ``shard_map``.  It
    dequantizes only the already-local final-owner query shard, communicates
    inside explicit stage groups, and never materializes a global key cache.
    Invalid page metadata is clipped before every read/write and propagated as
    a device-resident false health predicate.
    """

    groups = _axis_groups(axis_index_groups)
    query_backend = (
        linear_backend if dsa_query_backend is None else dsa_query_backend
    )
    if query_backend not in ("reference", "pallas"):
        raise ValueError("stage-local DSA query backend is unknown")
    if not isinstance(dsa_head_key_exact_association, bool):
        raise ValueError("exact DSA head/key flag must be boolean")
    if dsa_head_key_exact_association != (precomputed_wk_weight is not None):
        raise ValueError(
            "exact DSA head/key execution requires one external FP32 wk owner"
        )
    if residual.shape != (1, contract.hidden_size):
        raise ValueError("DSA residual must contain one exact hidden row")
    if index_cache.ndim != 3 or index_cache.shape[1:] != (
        cache_layout.local_rows_per_page,
        contract.head_dim,
    ):
        raise ValueError("local DSA key cache has an invalid shape")
    if residual.dtype != jnp.bfloat16 or index_cache.dtype != jnp.bfloat16:
        raise ValueError("DSA residual and key cache must remain BF16")
    if cache_layout.local_parallel_size <= 0 or (
        contract.num_heads % cache_layout.local_parallel_size
    ):
        raise ValueError("DSA heads must divide over the local stage")
    local_heads = contract.num_heads // cache_layout.local_parallel_size
    if input_norm_weight.shape != (contract.hidden_size,):
        raise ValueError("DSA input norm shape is invalid")
    if (q_a_bits is None) != (q_a_scale is None):
        raise ValueError("DSA q_a FP8 weight and scale must be supplied together")
    if q_a_bits is not None and q_a_bits.shape != (
        contract.q_lora_rank,
        contract.hidden_size,
    ):
        raise ValueError("DSA q_a FP8 shape is invalid")
    if q_a_norm_weight.shape != (contract.q_lora_rank,):
        raise ValueError("DSA q_a norm shape is invalid")
    if wq_b_bits.shape != (
        local_heads * contract.head_dim,
        contract.q_lora_rank,
    ):
        raise ValueError("DSA local wq_b FP8 shape is invalid")
    if wk_bits.shape != (contract.head_dim, contract.hidden_size):
        raise ValueError("DSA wk FP8 shape is invalid")
    if precomputed_wk_weight is not None and (
        precomputed_wk_weight.shape
        != (contract.head_dim, contract.hidden_size)
        or precomputed_wk_weight.dtype != jnp.float32
    ):
        raise ValueError("external DSA wk owner must be exact local FP32")
    if key_norm_weight.shape != (contract.head_dim,) or key_norm_bias.shape != (
        contract.head_dim,
    ):
        raise ValueError("DSA key norm shapes are invalid")
    if head_weight.shape != (local_heads, contract.hidden_size):
        raise ValueError("DSA local head-weight shape is invalid")

    (
        physical_page,
        local_row,
        target_owner,
        _,
        metadata_valid,
    ) = _require_decode_metadata(
        position,
        block_tables,
        context_lengths,
        local_slot,
        layout=cache_layout,
        physical_page_count=index_cache.shape[0],
    )
    if (precomputed_normalized is None) != (precomputed_q_residual is None):
        raise ValueError("DSA shared q_a intermediates must be supplied together")
    if precomputed_normalized is None:
        if q_a_bits is None or q_a_scale is None:
            raise ValueError("DSA q_a FP8 state is required without intermediates")
        normalized = rms_norm(
            residual, input_norm_weight, epsilon=rms_norm_epsilon
        )
        q_residual = rms_norm(
            _stage_fp8_linear(
                normalized,
                q_a_bits,
                q_a_scale,
                block_shape=block_shape,
                backend=linear_backend,
                interpret=linear_interpret,
            ),
            q_a_norm_weight,
            epsilon=lora_norm_epsilon,
        )
    else:
        normalized = precomputed_normalized
        q_residual = precomputed_q_residual
        if normalized.shape != residual.shape or normalized.dtype != residual.dtype:
            raise ValueError("DSA precomputed normalized residual is invalid")
        if q_residual.shape != (1, contract.q_lora_rank) or (
            q_residual.dtype != residual.dtype
        ):
            raise ValueError("DSA precomputed q residual is invalid")
    indexer_normalized = (
        lax.optimization_barrier(normalized)
        if dsa_head_key_exact_association
        else normalized
    )
    if dsa_query_weight_aliases is not None:
        if query_backend != "reference":
            raise ValueError(
                "exact DSA query aliases require the reference backend"
            )
        local_query, local_head_weights = _local_dsa_query_tuple4_exact(
            q_residual,
            indexer_normalized,
            dsa_query_weight_aliases,
            head_weight,
            position,
            contract=contract,
        )
    elif query_backend == "reference":
        # DB499 proves the accepted M=1 association only when the complete
        # topology-local owner shard exists as FP32 before projection. The
        # boundary is 8 MiB for PP8 and never reconstructs the global weight.
        local_wq_b_weight = lax.optimization_barrier(
            dequantize_fp8_bits_block_weight(
                wq_b_bits,
                wq_b_scale,
                block_shape=block_shape,
                output_dtype=jnp.float32,
            )
        )
        projected_query = linear(
            q_residual, local_wq_b_weight, output_dtype=jnp.float32
        )
        local_query, local_head_weights = _local_dsa_query_from_projection(
            projected_query,
            indexer_normalized,
            head_weight,
            position,
            contract=contract,
        )
    elif query_backend == "pallas":
        projected_query = fp8_block_matmul_f32(
            q_residual,
            wq_b_bits,
            wq_b_scale,
            config=Fp8BlockMatmulConfig(
                block_shape=block_shape,
                output_tile=block_shape[0],
                contraction_tile=block_shape[1],
            ),
            interpret=linear_interpret,
        )
        local_query, local_head_weights = _local_dsa_query_from_projection(
            projected_query,
            indexer_normalized,
            head_weight,
            position,
            contract=contract,
        )
    query_width = local_heads * contract.head_dim
    packed_query = jnp.concatenate(
        (local_query.reshape(1, query_width), local_head_weights), axis=-1
    )
    gathered_query = lax.all_gather(
        packed_query,
        axis_name=axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=groups,
    )
    query = jnp.transpose(
        gathered_query[..., :query_width], (1, 0, 2)
    ).reshape(1, contract.num_heads, contract.head_dim)
    gathered_head_weights = jnp.transpose(
        gathered_query[..., query_width:], (1, 0, 2)
    ).reshape(1, contract.num_heads)

    if dsa_head_key_exact_association:
        assert precomputed_wk_weight is not None
        projected_key = linear(
            indexer_normalized,
            precomputed_wk_weight,
            output_dtype=jnp.float32,
        )
        current_key_f32 = dsa_index_keys_from_projection(
            projected_key,
            key_norm_weight,
            key_norm_bias,
            position,
            contract=contract,
            key_norm_mode="divide_sqrt",
        ).astype(jnp.float32)
    elif linear_backend == "reference":
        wk_weight = dequantize_fp8_bits_block_weight(
            wk_bits, wk_scale, block_shape=block_shape
        )
        current_key_f32 = dsa_index_keys(
            indexer_normalized,
            wk_weight,
            key_norm_weight,
            key_norm_bias,
            position,
            contract=contract,
        ).astype(jnp.float32)
    else:
        projected_key = fp8_block_matmul_f32(
            indexer_normalized,
            wk_bits,
            wk_scale,
            config=Fp8BlockMatmulConfig(
                block_shape=block_shape,
                output_tile=block_shape[0],
                contraction_tile=block_shape[1],
            ),
            interpret=linear_interpret,
        )
        current_key_f32 = dsa_index_keys_from_projection(
            projected_key,
            key_norm_weight,
            key_norm_bias,
            position,
            contract=contract,
        ).astype(jnp.float32)
    current_key = current_key_f32.astype(index_cache.dtype)

    def write_current(value: Any) -> Any:
        return value.at[physical_page, local_row].set(current_key[0])

    index_cache = lax.cond(
        metadata_valid[0] & (local_slot == target_owner),
        write_current,
        lambda value: value,
        index_cache,
    )

    page_ids = block_tables[0]
    page_ok = (page_ids >= 0) & (page_ids < index_cache.shape[0])
    safe_pages = jnp.clip(
        page_ids, jnp.int32(0), jnp.int32(index_cache.shape[0] - 1)
    )
    logical_cache = jnp.take(index_cache, safe_pages, axis=0)
    logical_pages = jnp.arange(block_tables.shape[1], dtype=jnp.int32)
    local_rows = jnp.arange(
        cache_layout.local_rows_per_page, dtype=jnp.int32
    )
    global_positions = (
        logical_pages[:, None] * jnp.int32(cache_layout.logical_page_size)
        + local_slot.astype(jnp.int32)
        * jnp.int32(cache_layout.local_rows_per_page)
        + local_rows[None, :]
    )
    global_positions = jnp.where(
        page_ok[:, None], global_positions, jnp.int32(-1)
    ).reshape(-1)
    local_keys = logical_cache.reshape(-1, contract.head_dim)
    local_scores = dsa_scores(
        query,
        local_keys,
        gathered_head_weights,
        precision=dsa_score_precision,
    )
    candidate_scores, candidate_positions = local_topk_candidates(
        local_scores,
        global_positions,
        context_lengths,
        top_k=contract.top_k,
    )
    gathered_scores = lax.all_gather(
        candidate_scores,
        axis_name=axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=groups,
    )
    gathered_positions = lax.all_gather(
        candidate_positions,
        axis_name=axis_name,
        axis=0,
        tiled=False,
        axis_index_groups=groups,
    )
    selected = merge_topk_candidates_with_scores(
        gathered_scores,
        gathered_positions,
        context_lengths,
        top_k=contract.top_k,
        global_context_size=(
            block_tables.shape[1] * cache_layout.logical_page_size
        ),
    )
    selection_valid = canonicalize_selected_positions(
        selected
    ).contract_valid
    return StageLocalDsaFp8Result(
        index_cache,
        selected.positions,
        selected.valid_counts,
        selected.scores,
        metadata_valid & selection_valid,
        StageLocalDsaFp8Internals(
            normalized,
            q_residual,
            query,
            gathered_head_weights,
            current_key_f32,
        ),
    )


def stage_local_index_share_fp8_mapped(
    residual: Any,
    cache: Any,
    selected_positions: Any,
    selected_valid_counts: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    input_norm_weight: Any,
    q_a_bits: Any,
    q_a_scale: Any,
    q_a_norm_weight: Any,
    q_b_bits: Any,
    q_b_scale: Any,
    kv_a_bits: Any,
    kv_a_scale: Any,
    kv_a_norm_weight: Any,
    kv_b_bits: Any,
    kv_b_scale: Any,
    o_bits: Any,
    o_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: MlaNumericalContract = MlaNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    block_shape: tuple[int, int] = (128, 128),
    rms_norm_epsilon: float = 1e-5,
    lora_norm_epsilon: float = 1e-5,
    rope_theta: float = 8_000_000.0,
    main_rope_table_row: Any | None = None,
    precomputed_normalized: Any | None = None,
    precomputed_q_residual: Any | None = None,
    precomputed_kv_a: Any | None = None,
    sparse_attention_backend: Literal["reference", "pallas"] = "reference",
    sparse_attention_config: SparseMlaConfig = SparseMlaConfig(),
    sparse_attention_interpret: bool = False,
    linear_backend: StageLinearBackend = "reference",
    linear_interpret: bool = False,
    add_residual: bool = True,
    reconstruct_output_fp32: bool = False,
    virtual_tp32_reduction_association: (
        VirtualTp32ReductionAssociation | None
    ) = None,
    replicated_monolithic_attention: bool = False,
    pregathered_b512_attention: bool = False,
    capture_ingredients: bool = False,
) -> StageLocalIndexShareFp8Result | StageLocalIndexShareFp8ObservedResult:
    """Consume compact DSA state and execute raw-FP8 stage-local sparse MLA."""

    groups = _axis_groups(axis_index_groups)
    if residual.ndim != 2 or residual.shape[0] != 1:
        raise ValueError("IndexShare residual must contain exactly one row")
    hidden = residual.shape[1]
    if cache.ndim != 3 or cache.shape[1:] != (
        cache_layout.local_rows_per_page,
        contract.packed_cache_width,
    ):
        raise ValueError("local IndexShare cache has an invalid shape")
    if residual.dtype != jnp.bfloat16 or cache.dtype != jnp.bfloat16:
        raise ValueError("IndexShare residual and cache must remain BF16")
    if not isinstance(add_residual, bool):
        raise ValueError("IndexShare residual-add flag must be boolean")
    if not isinstance(reconstruct_output_fp32, bool):
        raise ValueError("IndexShare FP32 output reconstruction flag must be boolean")
    if not isinstance(replicated_monolithic_attention, bool):
        raise ValueError(
            "IndexShare replicated-monolithic attention flag must be boolean"
        )
    if not isinstance(pregathered_b512_attention, bool):
        raise ValueError("IndexShare pregathered-B512 flag must be boolean")
    if not isinstance(capture_ingredients, bool):
        raise ValueError("IndexShare ingredient-capture flag must be boolean")
    if main_rope_table_row is not None and (
        main_rope_table_row.shape != (2 * (contract.qk_rope_head_dim // 2),)
        or main_rope_table_row.dtype != jnp.bfloat16
    ):
        raise ValueError("IndexShare main-RoPE table row is invalid")
    if virtual_tp32_reduction_association is not None and (
        virtual_tp32_reduction_association
        not in _SUPPORTED_VIRTUAL_TP32_REDUCTION_ASSOCIATIONS
    ):
        raise ValueError("IndexShare virtual TP32 association is unknown")
    if virtual_tp32_reduction_association is not None and (
        reconstruct_output_fp32 or linear_backend != "pallas"
    ):
        raise ValueError(
            "IndexShare virtual TP32 association requires the BF16 Pallas path"
        )
    if replicated_monolithic_attention and capture_ingredients:
        raise ValueError(
            "replicated-monolithic attention must remain isolated from "
            "ingredient capture"
        )
    if pregathered_b512_attention and (
        replicated_monolithic_attention
        or capture_ingredients
        or reconstruct_output_fp32
        or virtual_tp32_reduction_association is not None
    ):
        raise ValueError(
            "pregathered-B512 attention must remain isolated from diagnostic "
            "attention/output variants"
        )
    if pregathered_b512_attention and cache_layout.local_parallel_size != 4:
        raise ValueError("pregathered-B512 attention is protected only for PP8 LP4")
    if capture_ingredients and (
        reconstruct_output_fp32
        or virtual_tp32_reduction_association is not None
        or linear_backend != "pallas"
    ):
        raise ValueError(
            "IndexShare ingredient capture requires the production BF16 Pallas path"
        )
    if contract.num_heads % cache_layout.local_parallel_size:
        raise ValueError("attention heads must divide over the local stage")
    local_heads = contract.num_heads // cache_layout.local_parallel_size
    if selected_positions.shape != (1, contract.top_k):
        raise ValueError("IndexShare selected-position shape is invalid")
    if selected_valid_counts.shape != (1,):
        raise ValueError("IndexShare selected-count shape is invalid")
    if input_norm_weight.shape != (hidden,):
        raise ValueError("IndexShare input norm shape is invalid")
    if (q_a_bits is None) != (q_a_scale is None):
        raise ValueError(
            "IndexShare q_a FP8 weight and scale must be supplied together"
        )
    if q_a_bits is not None and (
        q_a_bits.shape[1:] != (hidden,) or q_a_bits.ndim != 2
    ):
        raise ValueError("IndexShare q_a FP8 shape is invalid")
    q_lora_rank = q_a_norm_weight.shape[0]
    if q_a_norm_weight.shape != (q_lora_rank,):
        raise ValueError("IndexShare q_a norm shape is invalid")
    if q_b_bits.shape != (
        local_heads * contract.qk_head_dim,
        q_lora_rank,
    ):
        raise ValueError("IndexShare local q_b FP8 shape is invalid")
    if (kv_a_bits is None) != (kv_a_scale is None):
        raise ValueError(
            "IndexShare kv_a FP8 weight and scale must be supplied together"
        )
    if kv_a_bits is not None and kv_a_bits.shape != (
        contract.kv_lora_rank + contract.qk_rope_head_dim,
        hidden,
    ):
        raise ValueError("IndexShare kv_a FP8 shape is invalid")
    if kv_a_norm_weight.shape != (contract.kv_lora_rank,):
        raise ValueError("IndexShare kv_a norm shape is invalid")
    combined_width = contract.qk_nope_head_dim + contract.v_head_dim
    if kv_b_bits.shape != (
        local_heads * combined_width,
        contract.kv_lora_rank,
    ):
        raise ValueError("IndexShare local kv_b FP8 shape is invalid")
    if o_bits.shape != (hidden, local_heads * contract.v_head_dim):
        raise ValueError("IndexShare local output FP8 shape is invalid")

    (
        physical_page,
        local_row,
        target_owner,
        _,
        metadata_valid,
    ) = _require_decode_metadata(
        position,
        block_tables,
        context_lengths,
        local_slot,
        layout=cache_layout,
        physical_page_count=cache.shape[0],
    )
    if (precomputed_normalized is None) != (precomputed_q_residual is None):
        raise ValueError(
            "IndexShare shared q_a intermediates must be supplied together"
        )
    if precomputed_normalized is None:
        if q_a_bits is None or q_a_scale is None:
            raise ValueError(
                "IndexShare q_a FP8 state is required without intermediates"
            )
        normalized = rms_norm(
            residual, input_norm_weight, epsilon=rms_norm_epsilon
        )
        q_residual = rms_norm(
            _stage_fp8_linear(
                normalized,
                q_a_bits,
                q_a_scale,
                block_shape=block_shape,
                backend=linear_backend,
                interpret=linear_interpret,
            ),
            q_a_norm_weight,
            epsilon=lora_norm_epsilon,
        )
    else:
        normalized = precomputed_normalized
        q_residual = precomputed_q_residual
        if normalized.shape != residual.shape or normalized.dtype != residual.dtype:
            raise ValueError(
                "IndexShare precomputed normalized residual is invalid"
            )
        if q_residual.shape != (1, q_lora_rank) or (
            q_residual.dtype != residual.dtype
        ):
            raise ValueError("IndexShare precomputed q residual is invalid")
    q_states = _stage_fp8_linear(
        q_residual,
        q_b_bits,
        q_b_scale,
        block_shape=block_shape,
        backend=linear_backend,
        interpret=linear_interpret,
    ).reshape(
        1, local_heads, contract.qk_head_dim
    )
    q_nope = q_states[..., : contract.qk_nope_head_dim]
    q_rope_unrotated = q_states[..., contract.qk_nope_head_dim :]
    if main_rope_table_row is None:
        cos, sin = rotary_cos_sin(
            position,
            rotary_dim=contract.qk_rope_head_dim,
            theta=rope_theta,
            dtype=q_rope_unrotated.dtype,
        )
        q_rope = apply_rotary(
            q_rope_unrotated,
            cos[:, None, :],
            sin[:, None, :],
            interleaved=True,
        )
    else:
        half = contract.qk_rope_head_dim // 2
        with jax.named_scope("greenfield_main_rope_table"):
            cos = main_rope_table_row[:half][None, :]
            sin = main_rope_table_row[half:][None, :]
            q_rope = apply_rotary_fp32_final_round(
                q_rope_unrotated,
                cos[:, None, :],
                sin[:, None, :],
                interleaved=True,
            )

    if precomputed_kv_a is None:
        if kv_a_bits is None or kv_a_scale is None:
            raise ValueError(
                "IndexShare kv_a FP8 state is required without an intermediate"
            )
        current_kv = _stage_fp8_linear(
            normalized,
            kv_a_bits,
            kv_a_scale,
            block_shape=block_shape,
            backend=linear_backend,
            interpret=linear_interpret,
        )
    else:
        current_kv = precomputed_kv_a
        if current_kv.shape != (
            1,
            contract.kv_lora_rank + contract.qk_rope_head_dim,
        ) or current_kv.dtype != residual.dtype:
            raise ValueError("IndexShare precomputed kv_a projection is invalid")
    current_latent = rms_norm(
        current_kv[..., : contract.kv_lora_rank],
        kv_a_norm_weight,
        epsilon=lora_norm_epsilon,
    )
    current_rope_input = current_kv[
        ...,
        contract.kv_lora_rank : contract.kv_lora_rank
        + contract.qk_rope_head_dim,
    ][:, None, :]
    if main_rope_table_row is None:
        current_rope = apply_rotary(
            current_rope_input,
            cos[:, None, :],
            sin[:, None, :],
            interleaved=True,
        )[:, 0, :]
    else:
        with jax.named_scope("greenfield_main_rope_table"):
            current_rope = apply_rotary_fp32_final_round(
                current_rope_input,
                cos[:, None, :],
                sin[:, None, :],
                interleaved=True,
            )[:, 0, :]
    padding = contract.packed_cache_width - (
        contract.kv_lora_rank + contract.qk_rope_head_dim
    )
    current_cache_row = jnp.concatenate(
        (
            current_latent,
            current_rope,
            jnp.zeros((1, padding), dtype=normalized.dtype),
        ),
        axis=-1,
    ).astype(cache.dtype)

    def write_current(value: Any) -> Any:
        return value.at[physical_page, local_row].set(current_cache_row[0])

    cache = lax.cond(
        metadata_valid[0] & (local_slot == target_owner),
        write_current,
        lambda value: value,
        cache,
    )
    selected = SelectedPositions(selected_positions, selected_valid_counts)
    owner_selected_cache = (
        gather_stage_local_selected_kv(
            cache,
            block_tables,
            selected,
            context_lengths,
            layout=cache_layout,
            owner_index=local_slot,
        )
        if capture_ingredients
        else None
    )

    local_weight_uv = None
    if linear_backend == "reference":
        local_kv_b = dequantize_fp8_bits_block_weight(
            kv_b_bits, kv_b_scale, block_shape=block_shape
        ).reshape(local_heads, combined_width, contract.kv_lora_rank)
        local_weight_uk = local_kv_b[:, : contract.qk_nope_head_dim, :]
        local_weight_uv = jnp.transpose(
            local_kv_b[:, contract.qk_nope_head_dim :, :], (0, 2, 1)
        )
        q_absorbed_local = jnp.einsum(
            "rhp,hpl->rhl",
            q_nope.astype(jnp.float32),
            local_weight_uk.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        ).astype(cache.dtype)
    elif linear_backend == "pallas":
        q_absorbed_local = fp8_structured_kv_b_q_absorb(
            q_nope,
            kv_b_bits,
            kv_b_scale,
            config=Fp8BlockMatmulConfig(
                block_shape=block_shape,
                output_tile=block_shape[0],
                contraction_tile=block_shape[1],
            ),
            interpret=linear_interpret,
        )
    else:
        raise ValueError("stage-local FP8 linear backend is unknown")
    if pregathered_b512_attention:
        aligned = gather_stage_local_selected_kv_aligned(
            cache,
            block_tables,
            selected,
            context_lengths,
            layout=cache_layout,
            owner_index=local_slot,
        )
        with jax.named_scope("greenfield_selected_cache_lp4_exchange"):
            selected_cache = lax.psum(
                aligned.values,
                axis_name=axis_name,
                axis_index_groups=groups,
            )
        with jax.named_scope("greenfield_pregathered_b512_attention"):
            attended_local = pregathered_sparse_mla_pallas(
                q_absorbed_local,
                q_rope,
                selected_cache,
                aligned.valid_counts,
                contract=replace(contract, num_heads=local_heads),
                config=SparseMlaConfig(segment_block=512),
                interpret=sparse_attention_interpret,
            )
        attention_contract_valid = aligned.contract_valid
        partial = None
        combined = None
    else:
        packed_query = jnp.concatenate((q_absorbed_local, q_rope), axis=-1)
        gathered_query = lax.all_gather(
            packed_query,
            axis_name=axis_name,
            axis=0,
            tiled=False,
            axis_index_groups=groups,
        )
        full_query = _reshape_gathered_heads(
            gathered_query,
            num_heads=contract.num_heads,
            head_width=contract.kv_lora_rank + contract.qk_rope_head_dim,
        )
        q_absorbed = full_query[..., : contract.kv_lora_rank]
        full_q_rope = full_query[..., contract.kv_lora_rank :]
    if not pregathered_b512_attention and replicated_monolithic_attention:
        with jax.named_scope(
            "greenfield_replicated_monolithic_attention_cache_gather"
        ):
            gathered_cache = lax.all_gather(
                cache,
                axis_name=axis_name,
                axis=0,
                tiled=False,
                axis_index_groups=groups,
            )
        monolithic_cache = jnp.transpose(
            gathered_cache, (1, 0, 2, 3)
        ).reshape(
            cache.shape[0],
            cache_layout.logical_page_size,
            cache_layout.packed_cache_width,
        )
        monolithic_layout = StageLocalKvLayout(
            logical_page_size=cache_layout.logical_page_size,
            local_parallel_size=1,
            packed_cache_width=cache_layout.packed_cache_width,
        )
        with jax.named_scope("greenfield_replicated_monolithic_attention"):
            combined = stage_local_sparse_mla_kernel(
                q_absorbed,
                full_q_rope,
                monolithic_cache,
                block_tables,
                selected,
                context_lengths,
                layout=monolithic_layout,
                owner_index=jnp.int32(0),
                contract=contract,
                backend=sparse_attention_backend,
                config=sparse_attention_config,
                interpret=sparse_attention_interpret,
            )
        partial = combined
    elif not pregathered_b512_attention:
        partial = stage_local_sparse_mla_kernel(
            q_absorbed,
            full_q_rope,
            cache,
            block_tables,
            selected,
            context_lengths,
            layout=cache_layout,
            owner_index=local_slot,
            contract=contract,
            backend=sparse_attention_backend,
            config=sparse_attention_config,
            interpret=sparse_attention_interpret,
        )
        with jax.named_scope("greenfield_owner_split_attention_output_gather"):
            gathered_outputs = lax.all_gather(
                partial.output,
                axis_name=axis_name,
                axis=0,
                tiled=False,
                axis_index_groups=groups,
            )
        with jax.named_scope("greenfield_owner_split_attention_lse_gather"):
            gathered_lse = lax.all_gather(
                partial.logsumexp,
                axis_name=axis_name,
                axis=0,
                tiled=False,
                axis_index_groups=groups,
            )
        with jax.named_scope("greenfield_owner_split_attention_validity_gather"):
            gathered_validity = lax.all_gather(
                partial.contract_valid,
                axis_name=axis_name,
                axis=0,
                tiled=False,
                axis_index_groups=groups,
            )
        combined = combine_stage_local_attention(
            gathered_outputs, gathered_lse, gathered_validity
        )
    if not pregathered_b512_attention:
        assert combined is not None
        attended_local = lax.dynamic_slice_in_dim(
            combined.output,
            local_slot.astype(jnp.int32) * jnp.int32(local_heads),
            local_heads,
            axis=1,
        )
        attention_contract_valid = combined.contract_valid
    if linear_backend == "reference":
        assert local_weight_uv is not None
        value_states = jnp.einsum(
            "rhl,hlv->rhv",
            attended_local.astype(jnp.float32),
            local_weight_uv.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        ).astype(residual.dtype)
    else:
        value_states = fp8_structured_kv_b_value(
            attended_local,
            kv_b_bits,
            kv_b_scale,
            qk_nope_head_dim=contract.qk_nope_head_dim,
            config=Fp8BlockMatmulConfig(
                block_shape=block_shape,
                output_tile=block_shape[0],
                contraction_tile=block_shape[1],
            ),
            interpret=linear_interpret,
        )
    output_input = value_states.reshape(
        1, local_heads * contract.v_head_dim
    )
    diagnostic_virtual_partials = (
        _virtual_attention_output_partials(
            output_input,
            o_bits,
            o_scale,
            block_shape=block_shape,
            linear_interpret=linear_interpret,
        )
        if capture_ingredients
        else None
    )
    if virtual_tp32_reduction_association is not None:
        local_partials = _virtual_attention_output_partials(
            output_input,
            o_bits,
            o_scale,
            block_shape=block_shape,
            linear_interpret=linear_interpret,
        )
        if (
            virtual_tp32_reduction_association
            == STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION
        ):
            with jax.named_scope(
                "greenfield_strategy_nd_row0_attention_output"
            ):
                update = _reduce_virtual_tp32_bf16_partials(
                    local_partials,
                    axis_name=axis_name,
                    groups=groups,
                    association=virtual_tp32_reduction_association,
                )
        else:
            update = _reduce_virtual_tp32_bf16_partials(
                local_partials,
                axis_name=axis_name,
                groups=groups,
                association=virtual_tp32_reduction_association,
            )
    elif reconstruct_output_fp32:
        if linear_backend == "reference":
            local_update = lax.dot_general(
                output_input,
                dequantize_fp8_bits_block_weight(
                    o_bits, o_scale, block_shape=block_shape
                ),
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        elif linear_backend == "pallas":
            local_update = fp8_block_matmul_f32(
                output_input,
                o_bits,
                o_scale,
                config=Fp8BlockMatmulConfig(
                    block_shape=block_shape,
                    output_tile=block_shape[0],
                    contraction_tile=block_shape[1],
                ),
                interpret=linear_interpret,
            )
        else:
            raise ValueError("stage-local FP8 linear backend is unknown")
    else:
        local_update = _stage_fp8_linear(
            output_input,
            o_bits,
            o_scale,
            block_shape=block_shape,
            backend=linear_backend,
            interpret=linear_interpret,
        )
    if virtual_tp32_reduction_association is None:
        update = lax.psum(
            local_update,
            axis_name=axis_name,
            axis_index_groups=groups,
        )
    if reconstruct_output_fp32:
        if linear_backend == "pallas":
            # A plain cast is commuted into the psum by TPU XLA, changing
            # the requested FP32 collective into a BF16-result collective.
            # Keep the cast behind the same opaque boundary already used by
            # the feature-MoE FP32 reconstruction path.
            update = fp32_to_bf16_pallas_boundary(
                update,
                row_tile=8,
                output_tile=block_shape[0],
                interpret=linear_interpret,
            )
        else:
            update = update.astype(residual.dtype)
    output = residual_add(residual, update) if add_residual else update
    result = StageLocalIndexShareFp8Result(
        output,
        cache,
        metadata_valid & attention_contract_valid,
    )
    if not capture_ingredients:
        return result
    assert owner_selected_cache is not None
    assert diagnostic_virtual_partials is not None
    assert partial is not None
    assert combined is not None
    return StageLocalIndexShareFp8ObservedResult(
        result,
        StageLocalIndexShareFp8Ingredients(
            current_cache_row,
            owner_selected_cache.positions,
            owner_selected_cache.valid_counts,
            owner_selected_cache.values,
            owner_selected_cache.contract_valid,
            partial.output,
            partial.logsumexp,
            partial.contract_valid,
            combined.output,
            combined.logsumexp,
            combined.contract_valid,
            value_states,
            output_input,
            diagnostic_virtual_partials,
            local_update,
            update,
        ),
    )


def stage_local_dense_fp8_mapped(
    residual: Any,
    norm_weight: Any,
    gate_bits: Any,
    gate_scale: Any,
    up_bits: Any,
    up_scale: Any,
    down_bits: Any,
    down_scale: Any,
    *,
    axis_name: str,
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    block_shape: tuple[int, int] = (128, 128),
    epsilon: float = 1e-5,
    linear_backend: StageLinearBackend = "reference",
    linear_interpret: bool = False,
    precomputed_normalized: Any | None = None,
    add_residual: bool = True,
    reconstruct_down_fp32: bool = False,
    virtual_tp32_reduction_association: (
        VirtualTp32ReductionAssociation | None
    ) = None,
    capture_ingredients: bool = False,
) -> Any | StageLocalDenseFp8ObservedResult:
    """Execute one dense SwiGLU from local raw shards and one local combine."""

    groups = _axis_groups(axis_index_groups)
    if residual.ndim != 2 or residual.shape[0] != 1:
        raise ValueError("dense decode residual must contain exactly one row")
    hidden = residual.shape[1]
    if norm_weight.shape != (hidden,):
        raise ValueError("dense norm shape disagrees with hidden size")
    if gate_bits.shape != up_bits.shape or gate_bits.shape[1] != hidden:
        raise ValueError("dense local gate/up shards are invalid")
    if down_bits.shape != (hidden, gate_bits.shape[0]):
        raise ValueError("dense local down shard is invalid")
    if not isinstance(add_residual, bool):
        raise ValueError("dense residual-add flag must be boolean")
    if not isinstance(reconstruct_down_fp32, bool):
        raise ValueError("dense FP32 down reconstruction flag must be boolean")
    if not isinstance(capture_ingredients, bool):
        raise ValueError("dense ingredient-capture flag must be boolean")
    if virtual_tp32_reduction_association is not None and (
        virtual_tp32_reduction_association
        not in _SUPPORTED_VIRTUAL_TP32_REDUCTION_ASSOCIATIONS
    ):
        raise ValueError("dense virtual TP32 association is unknown")
    if virtual_tp32_reduction_association is not None and (
        reconstruct_down_fp32 or linear_backend != "pallas"
    ):
        raise ValueError(
            "dense virtual TP32 association requires the BF16 Pallas path"
        )
    if capture_ingredients and (
        reconstruct_down_fp32
        or virtual_tp32_reduction_association is not None
        or linear_backend != "pallas"
    ):
        raise ValueError(
            "dense ingredient capture requires the production BF16 Pallas path"
        )
    if precomputed_normalized is None:
        normalized = rms_norm(residual, norm_weight, epsilon=epsilon)
    else:
        normalized = precomputed_normalized
        if normalized.shape != residual.shape or (
            normalized.dtype != residual.dtype
        ):
            raise ValueError("dense precomputed normalized input is invalid")
    diagnostic_virtual_partials = (
        _virtual_dense_down_partials(
            normalized,
            gate_bits,
            gate_scale,
            up_bits,
            up_scale,
            down_bits,
            down_scale,
            block_shape=block_shape,
            linear_interpret=linear_interpret,
        )
        if capture_ingredients
        else None
    )
    if virtual_tp32_reduction_association is not None:
        local_partials = _virtual_dense_down_partials(
            normalized,
            gate_bits,
            gate_scale,
            up_bits,
            up_scale,
            down_bits,
            down_scale,
            block_shape=block_shape,
            linear_interpret=linear_interpret,
        )
        if (
            virtual_tp32_reduction_association
            == STRATEGY_ND_ROW0_REDUCTION_ASSOCIATION
        ):
            with jax.named_scope(
                "greenfield_strategy_nd_row0_dense_down"
            ):
                update = _reduce_virtual_tp32_bf16_partials(
                    local_partials,
                    axis_name=axis_name,
                    groups=groups,
                    association=virtual_tp32_reduction_association,
                )
        else:
            update = _reduce_virtual_tp32_bf16_partials(
                local_partials,
                axis_name=axis_name,
                groups=groups,
                association=virtual_tp32_reduction_association,
            )
    elif linear_backend == "reference":
        gate_weight = dequantize_fp8_bits_block_weight(
            gate_bits, gate_scale, block_shape=block_shape
        )
        up_weight = dequantize_fp8_bits_block_weight(
            up_bits, up_scale, block_shape=block_shape
        )
        down_weight = dequantize_fp8_bits_block_weight(
            down_bits, down_scale, block_shape=block_shape
        )
        activated = (
            silu(linear(normalized, gate_weight))
            * linear(normalized, up_weight)
        ).astype(normalized.dtype)
        if reconstruct_down_fp32:
            local_update = lax.dot_general(
                activated,
                down_weight,
                dimension_numbers=(((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
        else:
            local_update = linear(activated, down_weight)
    elif linear_backend == "pallas":
        local_update = fp8_fused_block_swiglu(
            normalized,
            gate_bits,
            gate_scale,
            up_bits,
            up_scale,
            down_bits,
            down_scale,
            config=Fp8BlockMatmulConfig(
                block_shape=block_shape,
                output_tile=block_shape[0],
                contraction_tile=block_shape[1],
            ),
            down_result_dtype=(
                jnp.float32 if reconstruct_down_fp32 else jnp.bfloat16
            ),
            interpret=linear_interpret,
        )
    else:
        raise ValueError("stage-local FP8 linear backend is unknown")
    if virtual_tp32_reduction_association is None:
        update = lax.psum(
            local_update,
            axis_name=axis_name,
            axis_index_groups=groups,
        )
    if reconstruct_down_fp32:
        if linear_backend == "pallas":
            update = fp32_to_bf16_pallas_boundary(
                update,
                row_tile=8,
                output_tile=block_shape[0],
                interpret=linear_interpret,
            )
        else:
            update = update.astype(residual.dtype)
    output = residual_add(residual, update) if add_residual else update
    if not capture_ingredients:
        return output
    assert diagnostic_virtual_partials is not None
    return StageLocalDenseFp8ObservedResult(
        output,
        StageLocalDenseFp8Ingredients(
            normalized,
            diagnostic_virtual_partials,
            local_update,
            update,
        ),
    )


def stage_local_moe_fp8_mapped(
    hidden_states: Any,
    router_weight: Any,
    correction_bias: Any,
    expert_gate_bits: Any,
    expert_gate_scale: Any,
    expert_up_bits: Any,
    expert_up_scale: Any,
    expert_down_bits: Any,
    expert_down_scale: Any,
    shared_gate_bits: Any,
    shared_gate_scale: Any,
    shared_up_bits: Any,
    shared_up_scale: Any,
    shared_down_bits: Any,
    shared_down_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
) -> tuple[Any, Any, Any]:
    """Route and execute selected raw-FP8 experts without a BF16 overlay."""

    groups = _axis_groups(axis_index_groups)
    if hidden_states.shape != (1, contract.hidden_size):
        raise ValueError("MoE decode hidden state must contain one exact row")
    if local_slot.shape != () or not jnp.issubdtype(local_slot.dtype, jnp.integer):
        raise ValueError("local_slot must be an integer scalar")
    local_expert_shape = (
        contract.local_experts,
        contract.intermediate_size,
        contract.hidden_size,
    )
    if expert_gate_bits.shape != local_expert_shape or expert_up_bits.shape != local_expert_shape:
        raise ValueError("local expert gate/up FP8 shapes are invalid")
    if expert_down_bits.shape != (
        contract.local_experts,
        contract.hidden_size,
        contract.intermediate_size,
    ):
        raise ValueError("local expert down FP8 shape is invalid")
    route_indices, route_weights = route_glm_noaux_tc(
        hidden_states,
        router_weight,
        correction_bias,
        top_k=contract.top_k,
    )
    expert_start = local_slot.astype(jnp.int32) * jnp.int32(
        contract.local_experts
    )
    routed_parts = []
    for position in range(contract.top_k):
        global_expert = route_indices[0, position]
        owns_expert = (
            (global_expert >= expert_start)
            & (global_expert < expert_start + contract.local_experts)
        )
        local_expert = jnp.clip(
            global_expert - expert_start, 0, contract.local_experts - 1
        )

        def compute(_: None) -> Any:
            gate = dequantize_fp8_bits_block_weight(
                lax.dynamic_index_in_dim(
                    expert_gate_bits, local_expert, axis=0, keepdims=False
                ),
                lax.dynamic_index_in_dim(
                    expert_gate_scale, local_expert, axis=0, keepdims=False
                ),
                block_shape=contract.fp8_block_shape,
            )
            up = dequantize_fp8_bits_block_weight(
                lax.dynamic_index_in_dim(
                    expert_up_bits, local_expert, axis=0, keepdims=False
                ),
                lax.dynamic_index_in_dim(
                    expert_up_scale, local_expert, axis=0, keepdims=False
                ),
                block_shape=contract.fp8_block_shape,
            )
            down = dequantize_fp8_bits_block_weight(
                lax.dynamic_index_in_dim(
                    expert_down_bits, local_expert, axis=0, keepdims=False
                ),
                lax.dynamic_index_in_dim(
                    expert_down_scale, local_expert, axis=0, keepdims=False
                ),
                block_shape=contract.fp8_block_shape,
            )
            activated = (
                silu(linear(hidden_states, gate))
                * linear(hidden_states, up)
            ).astype(hidden_states.dtype)
            output = linear(activated, down)
            return output * route_weights[0, position].astype(hidden_states.dtype)

        routed_parts.append(
            lax.cond(
                owns_expert,
                compute,
                lambda _: jnp.zeros_like(hidden_states),
                operand=None,
            )
        )
    local_routed = jnp.sum(
        jnp.stack(routed_parts, axis=0),
        axis=0,
        dtype=hidden_states.dtype,
    )
    shared_gate = dequantize_fp8_bits_block_weight(
        shared_gate_bits,
        shared_gate_scale,
        block_shape=contract.fp8_block_shape,
    )
    shared_up = dequantize_fp8_bits_block_weight(
        shared_up_bits,
        shared_up_scale,
        block_shape=contract.fp8_block_shape,
    )
    shared_down = dequantize_fp8_bits_block_weight(
        shared_down_bits,
        shared_down_scale,
        block_shape=contract.fp8_block_shape,
    )
    shared_activated = (
        silu(linear(hidden_states, shared_gate))
        * linear(hidden_states, shared_up)
    ).astype(hidden_states.dtype)
    local_shared = linear(shared_activated, shared_down)
    combined = lax.psum(
        jnp.stack((local_routed, local_shared), axis=0),
        axis_name=axis_name,
        axis_index_groups=groups,
    )
    scale = jnp.asarray(
        contract.routed_scaling_factor, dtype=hidden_states.dtype
    )
    output = (combined[0] * scale + combined[1]).astype(hidden_states.dtype)
    return output, route_indices, route_weights


def stage_local_moe_pallas_from_routes_mapped(
    hidden_states: Any,
    route_indices: Any,
    route_weights: Any,
    expert_gate_bits: Any,
    expert_gate_scale: Any,
    expert_up_bits: Any,
    expert_up_scale: Any,
    expert_down_bits: Any,
    expert_down_scale: Any,
    shared_gate_bits: Any,
    shared_gate_scale: Any,
    shared_up_bits: Any,
    shared_up_scale: Any,
    shared_down_bits: Any,
    shared_down_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(
        contraction_tile=512
    ),
    interpret: bool = False,
) -> Any:
    """Execute final-layout raw-FP8 MoE work and one local combine.

    Routed tables are already packed in the Pallas MXU access order:
    gate/up ``[local_experts, hidden, intermediate]`` and down
    ``[local_experts, intermediate, hidden]``. Shared-expert shards retain
    checkpoint ``[out, in]`` orientation because their feature sharding is
    directly compatible with the standalone Pallas matmuls. No complete
    expert table is transposed or decoded by the runtime.

    The routed top-k reduction is performed in BF16 before the stage-local
    collective. Routed and shared partials occupy separate leading rows of
    the same ``psum`` and the routed factor is applied only after reduction.
    """

    groups = _axis_groups(axis_index_groups)
    if hidden_states.shape != (1, contract.hidden_size):
        raise ValueError("Pallas MoE hidden state must contain one exact row")
    if hidden_states.dtype != jnp.bfloat16:
        raise ValueError("Pallas MoE hidden state must be BF16")
    if route_indices.shape != (1, contract.top_k) or (
        route_indices.dtype != jnp.int32
    ):
        raise ValueError("Pallas MoE routes must be one exact int32 top-k row")
    if route_weights.shape != (1, contract.top_k) or (
        route_weights.dtype != jnp.float32
    ):
        raise ValueError("Pallas MoE route weights must be one FP32 top-k row")
    if local_slot.shape != () or local_slot.dtype != jnp.int32:
        raise ValueError("Pallas MoE local_slot must be an int32 scalar")
    if config.block_shape != contract.fp8_block_shape:
        raise ValueError(
            "Pallas tile block shape must match the MoE numerical contract"
        )

    expected_gate_shape = (
        contract.local_experts,
        contract.hidden_size,
        contract.intermediate_size,
    )
    expected_down_shape = (
        contract.local_experts,
        contract.intermediate_size,
        contract.hidden_size,
    )
    if expert_gate_bits.shape != expected_gate_shape or (
        expert_up_bits.shape != expected_gate_shape
    ):
        raise ValueError(
            "Pallas routed gate/up tables must use final [G,K,N] layout"
        )
    if expert_down_bits.shape != expected_down_shape:
        raise ValueError(
            "Pallas routed down table must use final [G,K,N] layout"
        )
    shared_gate_shape = (
        contract.local_shared_intermediate,
        contract.hidden_size,
    )
    if shared_gate_bits.shape != shared_gate_shape or (
        shared_up_bits.shape != shared_gate_shape
    ):
        raise ValueError(
            "Pallas shared gate/up shards must retain [local_out,in] layout"
        )
    if shared_down_bits.shape != (
        contract.hidden_size,
        contract.local_shared_intermediate,
    ):
        raise ValueError(
            "Pallas shared down shard must retain [out,local_in] layout"
        )

    expert_start = local_slot * jnp.int32(contract.local_experts)
    routed_outputs = fp8_fused_selected_moe(
        hidden_states,
        route_indices[0],
        expert_start,
        expert_gate_bits,
        expert_gate_scale,
        expert_up_bits,
        expert_up_scale,
        expert_down_bits,
        expert_down_scale,
        config=config,
        interpret=interpret,
    )
    weighted_routed = (
        routed_outputs
        * route_weights[0, :, None].astype(hidden_states.dtype)
    ).astype(hidden_states.dtype)
    local_routed = jnp.sum(
        weighted_routed,
        axis=0,
        dtype=hidden_states.dtype,
    )[None, :]

    # Selected routed kernels explicitly apply every 128-wide scale block
    # inside their wider contraction tile.  The standalone shared kernels
    # consume one scalar scale per contraction tile, so keep that tile at one
    # checkpoint scale block even when routed experts use k=512.
    shared_config = Fp8BlockMatmulConfig(
        block_shape=config.block_shape,
        row_tile=config.row_tile,
        output_tile=config.block_shape[0],
        contraction_tile=config.block_shape[1],
        output_dtype=config.output_dtype,
        accumulator_dtype=config.accumulator_dtype,
    )
    shared_gate, shared_up = fp8_block_up_gate(
        hidden_states,
        shared_gate_bits,
        shared_gate_scale,
        shared_up_bits,
        shared_up_scale,
        config=shared_config,
        interpret=interpret,
    )
    shared_activated = (silu(shared_gate) * shared_up).astype(
        hidden_states.dtype
    )
    local_shared = fp8_block_matmul(
        shared_activated,
        shared_down_bits,
        shared_down_scale,
        config=shared_config,
        interpret=interpret,
    )

    combined = lax.psum(
        jnp.stack((local_routed, local_shared), axis=0),
        axis_name=axis_name,
        axis_index_groups=groups,
    )
    scale = jnp.asarray(
        contract.routed_scaling_factor, dtype=hidden_states.dtype
    )
    return (combined[0] * scale + combined[1]).astype(hidden_states.dtype)


def stage_local_moe_pallas_feature_from_routes_mapped(
    hidden_states: Any,
    route_indices: Any,
    route_weights: Any,
    expert_gate_bits: Any,
    expert_gate_scale: Any,
    expert_up_bits: Any,
    expert_up_scale: Any,
    expert_down_bits: Any,
    expert_down_scale: Any,
    shared_gate_bits: Any,
    shared_gate_scale: Any,
    shared_up_bits: Any,
    shared_up_scale: Any,
    shared_down_bits: Any,
    shared_down_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(
        contraction_tile=512
    ),
    fuse_route_weighting: bool = False,
    reconstruct_down_fp32: bool = False,
    interpret: bool = False,
) -> Any:
    """Execute all top-k routes on one stage-local feature shard.

    Every stage chip owns the same expert identities but a disjoint quarter
    of each expert's intermediate dimension. Gate/up output shards and the
    reciprocal down contraction shards are therefore local, route work is
    balanced independently of expert ids, and the existing single local
    combine reconstructs the exact hidden-width partials.
    """

    groups = _axis_groups(axis_index_groups)
    if hidden_states.shape != (1, contract.hidden_size) or (
        hidden_states.dtype != jnp.bfloat16
    ):
        raise ValueError(
            "feature-sharded Pallas MoE hidden must be one BF16 row"
        )
    if route_indices.shape != (1, contract.top_k) or (
        route_indices.dtype != jnp.int32
    ):
        raise ValueError(
            "feature-sharded Pallas routes must be one exact int32 top-k row"
        )
    if route_weights.shape != (1, contract.top_k) or (
        route_weights.dtype != jnp.float32
    ):
        raise ValueError(
            "feature-sharded Pallas route weights must be one FP32 top-k row"
        )
    if local_slot.shape != () or local_slot.dtype != jnp.int32:
        raise ValueError(
            "feature-sharded Pallas local_slot must be an int32 scalar"
        )
    if config.block_shape != contract.fp8_block_shape:
        raise ValueError(
            "Pallas tile block shape must match the MoE numerical contract"
        )
    if not isinstance(reconstruct_down_fp32, bool):
        raise ValueError("FP32 routed-down reconstruction flag must be boolean")
    if reconstruct_down_fp32 and fuse_route_weighting:
        raise ValueError(
            "FP32 routed-down reconstruction is incompatible with fused "
            "route weighting"
        )

    local_intermediate = contract.local_shared_intermediate
    expected_gate_shape = (
        contract.num_experts,
        contract.hidden_size,
        local_intermediate,
    )
    expected_down_shape = (
        contract.num_experts,
        local_intermediate,
        contract.hidden_size,
    )
    if expert_gate_bits.shape != expected_gate_shape or (
        expert_up_bits.shape != expected_gate_shape
    ):
        raise ValueError(
            "feature-sharded routed gate/up must use [E,H,I/stage]"
        )
    if expert_down_bits.shape != expected_down_shape:
        raise ValueError(
            "feature-sharded routed down must use [E,I/stage,H]"
        )
    expected_gate_scale = (
        contract.num_experts,
        local_intermediate // contract.fp8_block_shape[0],
        contract.hidden_size // contract.fp8_block_shape[1],
    )
    expected_down_scale = (
        contract.num_experts,
        contract.hidden_size // contract.fp8_block_shape[0],
        local_intermediate // contract.fp8_block_shape[1],
    )
    if expert_gate_scale.shape != expected_gate_scale or (
        expert_up_scale.shape != expected_gate_scale
    ):
        raise ValueError("feature-sharded routed gate/up scale shape drifted")
    if expert_down_scale.shape != expected_down_scale:
        raise ValueError("feature-sharded routed down scale shape drifted")

    shared_gate_shape = (local_intermediate, contract.hidden_size)
    if shared_gate_bits.shape != shared_gate_shape or (
        shared_up_bits.shape != shared_gate_shape
    ):
        raise ValueError(
            "feature-sharded shared gate/up must retain [local_out,in]"
        )
    if shared_down_bits.shape != (
        contract.hidden_size,
        local_intermediate,
    ):
        raise ValueError(
            "feature-sharded shared down must retain [out,local_in]"
        )

    routed_outputs = fp8_fused_selected_moe(
        hidden_states,
        route_indices[0],
        jnp.asarray(0, dtype=jnp.int32),
        expert_gate_bits,
        expert_gate_scale,
        expert_up_bits,
        expert_up_scale,
        expert_down_bits,
        expert_down_scale,
        route_weights=(route_weights[0] if fuse_route_weighting else None),
        down_result_dtype=(
            jnp.float32 if reconstruct_down_fp32 else jnp.bfloat16
        ),
        config=config,
        interpret=interpret,
    )
    if fuse_route_weighting:
        local_routed = routed_outputs
    elif reconstruct_down_fp32:
        complete_routed_fp32 = lax.psum(
            routed_outputs,
            axis_name=axis_name,
            axis_index_groups=groups,
        )
        # Keep the conversion behind an opaque device custom-call.  A plain
        # cast, even after lax.optimization_barrier, is commuted into the psum
        # by TPU XLA and lowers a BF16-result all-reduce.
        complete_routed = fp32_to_bf16_pallas_boundary(
            complete_routed_fp32,
            row_tile=contract.top_k,
            output_tile=config.block_shape[0],
            interpret=interpret,
        )
        route_owners = route_indices[0] // jnp.int32(
            contract.local_experts
        )
        owned_routes = jnp.where(
            (route_owners == local_slot)[:, None],
            complete_routed,
            jnp.zeros_like(complete_routed),
        )
        weighted_routed = (
            owned_routes
            * route_weights[0, :, None].astype(hidden_states.dtype)
        ).astype(hidden_states.dtype)
        local_routed = jnp.sum(
            weighted_routed,
            axis=0,
            dtype=hidden_states.dtype,
        )[None, :]
    else:
        weighted_routed = (
            routed_outputs
            * route_weights[0, :, None].astype(hidden_states.dtype)
        ).astype(hidden_states.dtype)
        local_routed = jnp.sum(
            weighted_routed,
            axis=0,
            dtype=hidden_states.dtype,
        )[None, :]

    shared_config = Fp8BlockMatmulConfig(
        block_shape=config.block_shape,
        row_tile=config.row_tile,
        output_tile=config.block_shape[0],
        contraction_tile=config.block_shape[1],
        output_dtype=config.output_dtype,
        accumulator_dtype=config.accumulator_dtype,
    )
    shared_gate, shared_up = fp8_block_up_gate(
        hidden_states,
        shared_gate_bits,
        shared_gate_scale,
        shared_up_bits,
        shared_up_scale,
        config=shared_config,
        interpret=interpret,
    )
    shared_activated = (silu(shared_gate) * shared_up).astype(
        hidden_states.dtype
    )
    local_shared = fp8_block_matmul(
        shared_activated,
        shared_down_bits,
        shared_down_scale,
        config=shared_config,
        interpret=interpret,
    )

    combined = lax.psum(
        jnp.stack((local_routed, local_shared), axis=0),
        axis_name=axis_name,
        axis_index_groups=groups,
    )
    scale = jnp.asarray(
        contract.routed_scaling_factor, dtype=hidden_states.dtype
    )
    return (combined[0] * scale + combined[1]).astype(hidden_states.dtype)


def stage_local_moe_pallas_feature_mapped(
    hidden_states: Any,
    router_weight: Any,
    correction_bias: Any,
    expert_gate_bits: Any,
    expert_gate_scale: Any,
    expert_up_bits: Any,
    expert_up_scale: Any,
    expert_down_bits: Any,
    expert_down_scale: Any,
    shared_gate_bits: Any,
    shared_gate_scale: Any,
    shared_up_bits: Any,
    shared_up_scale: Any,
    shared_down_bits: Any,
    shared_down_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(
        contraction_tile=512
    ),
    fuse_route_weighting: bool = False,
    reconstruct_down_fp32: bool = False,
    interpret: bool = False,
) -> tuple[Any, Any, Any]:
    """Route once and execute the stage-local expert-feature challenger."""

    if router_weight.shape != (contract.num_experts, contract.hidden_size):
        raise ValueError("feature-sharded Pallas router shape is invalid")
    if correction_bias.shape != (contract.num_experts,):
        raise ValueError("feature-sharded Pallas correction bias is invalid")
    route_indices, route_weights = route_glm_noaux_tc(
        hidden_states,
        router_weight,
        correction_bias,
        top_k=contract.top_k,
    )
    output = stage_local_moe_pallas_feature_from_routes_mapped(
        hidden_states,
        route_indices,
        route_weights,
        expert_gate_bits,
        expert_gate_scale,
        expert_up_bits,
        expert_up_scale,
        expert_down_bits,
        expert_down_scale,
        shared_gate_bits,
        shared_gate_scale,
        shared_up_bits,
        shared_up_scale,
        shared_down_bits,
        shared_down_scale,
        local_slot,
        axis_name=axis_name,
        contract=contract,
        axis_index_groups=axis_index_groups,
        config=config,
        fuse_route_weighting=fuse_route_weighting,
        reconstruct_down_fp32=reconstruct_down_fp32,
        interpret=interpret,
    )
    return output, route_indices, route_weights


def stage_local_moe_pallas_mapped(
    hidden_states: Any,
    router_weight: Any,
    correction_bias: Any,
    expert_gate_bits: Any,
    expert_gate_scale: Any,
    expert_up_bits: Any,
    expert_up_scale: Any,
    expert_down_bits: Any,
    expert_down_scale: Any,
    shared_gate_bits: Any,
    shared_gate_scale: Any,
    shared_up_bits: Any,
    shared_up_scale: Any,
    shared_down_bits: Any,
    shared_down_scale: Any,
    local_slot: Any,
    *,
    axis_name: str,
    contract: GlmMoeNumericalContract = GlmMoeNumericalContract(),
    axis_index_groups: Sequence[Sequence[int]] | None = None,
    config: Fp8BlockMatmulConfig = Fp8BlockMatmulConfig(
        contraction_tile=512
    ),
    interpret: bool = False,
) -> tuple[Any, Any, Any]:
    """Route once, execute the final-layout Pallas MoE, and return routing."""

    if router_weight.shape != (contract.num_experts, contract.hidden_size):
        raise ValueError("Pallas MoE router weight shape is invalid")
    if correction_bias.shape != (contract.num_experts,):
        raise ValueError("Pallas MoE correction bias shape is invalid")
    route_indices, route_weights = route_glm_noaux_tc(
        hidden_states,
        router_weight,
        correction_bias,
        top_k=contract.top_k,
    )
    output = stage_local_moe_pallas_from_routes_mapped(
        hidden_states,
        route_indices,
        route_weights,
        expert_gate_bits,
        expert_gate_scale,
        expert_up_bits,
        expert_up_scale,
        expert_down_bits,
        expert_down_scale,
        shared_gate_bits,
        shared_gate_scale,
        shared_up_bits,
        shared_up_scale,
        shared_down_bits,
        shared_down_scale,
        local_slot,
        axis_name=axis_name,
        contract=contract,
        axis_index_groups=axis_index_groups,
        config=config,
        interpret=interpret,
    )
    return output, route_indices, route_weights
