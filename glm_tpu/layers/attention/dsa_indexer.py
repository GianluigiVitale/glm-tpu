"""The DSA indexer: key projection and scoring, and exact top-k shortlists with a global cut check
and a full-width fallback, for decode and prefill."""

from __future__ import annotations

from typing import Any, NamedTuple, Literal

from jax import lax
import jax.numpy as jnp
import jax

from glm_tpu.layers.attention.kv_cache import (
    require_decode_metadata,
    canonicalize_selected_positions,
    write_prefill_cache_block,
)
from glm_tpu.layers.attention.mla import PreparedAttention
from glm_tpu.layers.contracts import DsaNumericalContract, SelectedPositions, StageLocalKvLayout, require_shape
from glm_tpu.layers.linear import dot_f32, head_weight_partial, resident_matmul_f32
from glm_tpu.layers.norm import affine_layer_norm, affine_key_layer_norm
from glm_tpu.layers.rope import apply_rotary, rotary_cos_sin, rotary_cos_sin_from_rows
from glm_tpu.models.glm_moe_dsa.weights import Bf16DsaWeights


# Keep module import backend-neutral: multi-host callers must be able to call
# ``jax.distributed.initialize`` after importing the reference contract.
_NEGATIVE_INFINITY = float("-inf")


def two_stage_topk(
    scores: Any,
    positions: Any,
    valid_lengths: Any,
    *,
    top_k: int,
    global_context_size: int,
    candidates_per_owner: int = 512,
    expert_axis: str = "expert",
    positions_in_order: bool = False,
):
    """Return frozen score/position pairs and whether the exact fallback ran.

    Every owner supplies its best L+1 pairs. The first L enter the merge; the
    extra pair proves no omitted item outranks the global Kth pair, including
    lowest-position ties. Any owner's failed check makes the whole group redo
    the frozen full-K selection. No score arithmetic is changed.
    """
    if type(candidates_per_owner) is not int or candidates_per_owner <= 0:
        raise ValueError("candidates_per_owner must be a positive integer")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if type(positions_in_order) is not bool:
        raise ValueError("positions_in_order must be a static boolean")
    if scores.ndim != 2 or scores.shape[1] <= 0 or positions.shape != (scores.shape[1],):
        raise ValueError("scores/positions geometry drifted")
    if valid_lengths.shape != (scores.shape[0],):
        raise ValueError("valid lengths must have one value per score row")
    if positions.dtype != jnp.int32 or valid_lengths.dtype != jnp.int32:
        raise ValueError("positions and lengths must be int32")
    width = min(candidates_per_owner, top_k)

    def local(k):
        if not positions_in_order:
            return local_topk_candidates(scores, positions, valid_lengths, top_k=k)
        # Cache-page scoring below emits ascending logical positions. Invalid
        # pages may occur anywhere; they are masked before top-k's index tie rule.
        masked = jnp.where((positions[None] >= 0) & (positions[None] < valid_lengths[:, None]), scores, -jnp.inf)
        pad = max(0, k - scores.shape[1])
        masked = jnp.pad(masked, ((0, 0), (0, pad)), constant_values=-jnp.inf)
        pos = jnp.pad(positions, ((0, pad),), constant_values=-1)
        values, index = lax.top_k(masked, k)
        return values, jnp.where(values == -jnp.inf, -1, pos[index]).astype(jnp.int32)

    def merge(values, indices):
        values = lax.all_gather(values, expert_axis, axis=0, tiled=False)
        indices = lax.all_gather(indices, expert_axis, axis=0, tiled=False)
        return merge_topk_candidates_with_scores(
            values,
            indices,
            valid_lengths,
            top_k=top_k,
            global_context_size=global_context_size,
            paired_position_sort=True,
        )

    def full(_):
        values, indices = local(top_k)
        return merge(values, indices)

    values, indices = local(width + 1)
    # Pad the union locally if L*owners<K, so a short configuration remains
    # correct through fallback rather than assuming a particular mesh size.
    union_width = width * lax.axis_size(expert_axis)
    pad = max(0, (top_k + lax.axis_size(expert_axis) - 1) // lax.axis_size(expert_axis) - width)
    short = merge(
        jnp.pad(values[:, :width], ((0, 0), (0, pad)), constant_values=-jnp.inf),
        jnp.pad(indices[:, :width], ((0, 0), (0, pad)), constant_values=-1),
    )
    cutoff_score = short.scores[:, -1]
    cutoff_position = short.positions[:, -1]
    omitted_score, omitted_position = values[:, width], indices[:, width]
    omitted_live = omitted_position >= 0

    # lax.top_k has a total FP32 order: +0 outranks -0. Float comparisons
    # alone collapse those keys and can incorrectly accept an incomplete cut.
    def score_key(value):
        bits = lax.bitcast_convert_type(value, jnp.int32)
        return jnp.where(bits < 0, bits ^ jnp.int32(0x7FFFFFFF), bits)

    omitted_key, cutoff_key = score_key(omitted_score), score_key(cutoff_score)
    outranks = omitted_live & (
        (omitted_key > cutoff_key)
        | ((omitted_key == cutoff_key) & ((cutoff_position < 0) | (omitted_position < cutoff_position)))
    )
    # Nonfinite input scores use the frozen path without relying on comparisons
    # involving NaN. Negative infinities are already the frozen masked sentinel.
    uncertain = jnp.any(jnp.isnan(scores) | jnp.isposinf(scores))
    fallback = lax.pmax((jnp.any(outranks) | uncertain | (union_width < top_k)).astype(jnp.int32), expert_axis).astype(
        jnp.bool_
    )
    return lax.cond(fallback, full, lambda _: short, operand=None), fallback


def score_cache_pages(query, cache, head_weights, block_tables, *, layout, owner):
    """Score physical keys in place; gather scalar scores into logical order.

    The projection/FP32 DSA arithmetic is the frozen dsa_scores. Page mapping
    moves one scalar per key rather than its full 128-component cache vector.
    Return ascending logical positions for the sort-free local top-k path.
    """

    physical_scores = dsa_scores(query, cache.reshape(-1, cache.shape[-1]), head_weights, precision="highest")
    pages = block_tables[0]
    page_ok = (pages >= 0) & (pages < cache.shape[0])
    safe_pages = jnp.clip(pages, 0, cache.shape[0] - 1)
    local_rows = jnp.arange(layout.local_rows_per_page, dtype=jnp.int32)
    indices = safe_pages[:, None] * layout.local_rows_per_page + local_rows[None]
    scores = jnp.take(physical_scores, indices.reshape(-1), axis=1)
    positions = (
        jnp.arange(pages.shape[0], dtype=jnp.int32)[:, None] * layout.logical_page_size
        + owner * layout.local_rows_per_page
        + local_rows[None]
    )
    positions = jnp.where(page_ok[:, None], positions, -1).reshape(-1)
    return scores, positions


def prefill_dsa_one_pass(
    query,
    keys,
    head_weights,
    positions,
    valid_lengths,
    *,
    global_context_size,
    top_k=2048,
    precision="highest",
    candidates_per_owner=512,
):
    """P2: one score row and one shortlist per prefill tile, with frozen health.

    Removes the repeated score/top-k/merge chain over 512-key blocks. Temporary
    per-head scores are rows*32*local_context FP32 values (64 MiB at 128K/M32).
    This is opt-in and keeps causal lengths and finite-score admission intact.
    """
    from glm_tpu.layers.attention.kv_cache import canonicalize_selected_positions
    from glm_tpu.layers.contracts import SelectedPositions

    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("prefill DSA requires expert8/feature4")
    if query.ndim != 3 or not 1 <= query.shape[0] <= 32 or query.shape[1:] != (32, 128):
        raise ValueError("prefill DSA requires 1..32 rows and 32x128 queries")
    if query.dtype != jnp.float32 or head_weights.dtype != jnp.float32 or keys.dtype != jnp.bfloat16:
        raise ValueError("prefill DSA requires F32 queries/weights and BF16 keys")
    if keys.ndim != 2 or keys.shape[0] <= 0 or keys.shape[1] != 128 or positions.shape != (keys.shape[0],):
        raise ValueError("prefill DSA key/position geometry drifted")
    if positions.dtype != jnp.int32 or valid_lengths.dtype != jnp.int32:
        raise ValueError("prefill DSA metadata must be int32")
    if type(top_k) is not int or not 0 < top_k <= 2048:
        raise ValueError("prefill DSA top_k must be 1..2048")
    if type(global_context_size) is not int or not 0 < global_context_size < 2147483647:
        raise ValueError("prefill DSA context must fit positive int32")
    seen = lax.associative_scan(jnp.maximum, positions)
    previous = jnp.concatenate((jnp.full((1,), -1, jnp.int32), seen[:-1]))
    metadata_ok = (
        jnp.all((positions >= -1) & (positions < global_context_size))
        & jnp.all((positions == -1) | (positions > previous))
        & jnp.all((valid_lengths >= 0) & (valid_lengths <= global_context_size))
    )
    scores = dsa_scores(query, keys, head_weights, precision=precision)
    visible = (positions[None] >= 0) & (positions[None] < valid_lengths[:, None])
    healthy = (
        metadata_ok
        & jnp.all(jnp.isfinite(query))
        & jnp.all(jnp.isfinite(head_weights))
        & jnp.all(jnp.isfinite(scores) | ~visible)
    )
    selected, _ = two_stage_topk(
        scores,
        positions,
        valid_lengths,
        top_k=top_k,
        global_context_size=global_context_size,
        candidates_per_owner=candidates_per_owner,
        positions_in_order=True,
    )
    canonical = canonicalize_selected_positions(SelectedPositions(selected.positions, selected.valid_counts))
    live = jnp.arange(top_k)[None] < selected.valid_counts[:, None]
    selected_ok = jnp.all(canonical.contract_valid) & jnp.all(
        jnp.where(live, jnp.isfinite(selected.scores) & (selected.positions < valid_lengths[:, None]), True)
    )
    return selected, healthy & selected_ok


def prompt_index_key_chunk(
    normalized_chunk: Any,
    positions: Any,
    wk_weight: Any,
    key_norm_weight: Any,
    key_norm_bias: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    physical_rows: int = 64,
    rope_table_rows: Any | None = None,
) -> Any:
    """Project exact normalized inputs with the accepted M64 association.

    The teacher-forced scan records the exact BF16 normalization output already
    consumed by DSA.  The public chunk stays logical and stage-local; only the
    key projection and affine key LayerNorm execute in explicit physical-row
    partitions.  Recurrent ``decode_batch1`` remains one row.
    """

    if not isinstance(physical_rows, int) or isinstance(physical_rows, bool) or physical_rows <= 0:
        raise ValueError("physical prompt-key rows must be positive")
    if normalized_chunk.ndim != 2 or normalized_chunk.shape[1] != (contract.hidden_size):
        raise ValueError("prompt-key normalized chunk has an invalid shape")
    chunk_rows = normalized_chunk.shape[0]
    if chunk_rows <= 0 or chunk_rows % physical_rows:
        raise ValueError("prompt-key chunk must divide into physical rows")
    if positions.shape != (chunk_rows,) or not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("prompt-key positions must be one integer row per token")
    if normalized_chunk.dtype != jnp.bfloat16:
        raise ValueError("prompt-key normalized chunk must remain BF16")
    if wk_weight.shape != (contract.head_dim, contract.hidden_size) or (wk_weight.dtype != jnp.float32):
        raise ValueError("prompt-key wk must be the adapted FP32 owner weight")
    if key_norm_weight.shape != (contract.head_dim,) or (key_norm_bias.shape != key_norm_weight.shape):
        raise ValueError("prompt-key affine norm shapes are invalid")
    if key_norm_weight.dtype != jnp.bfloat16 or (key_norm_bias.dtype != jnp.bfloat16):
        raise ValueError("prompt-key affine norm parameters must remain BF16")

    partitions = normalized_chunk.reshape(
        chunk_rows // physical_rows,
        physical_rows,
        contract.hidden_size,
    )

    def project_and_normalize(partition: Any) -> Any:
        projected = lax.dot_general(
            partition.astype(jnp.float32),
            wk_weight,
            dimension_numbers=(((1,), (1,)), ((), ())),
            precision=(lax.Precision.DEFAULT, lax.Precision.HIGHEST),
            preferred_element_type=jnp.float32,
        )
        projected = projected.astype(jnp.float32)
        return affine_key_layer_norm(
            projected,
            key_norm_weight,
            key_norm_bias,
            epsilon=contract.key_layer_norm_epsilon,
            mode="divide_sqrt",
        )

    keys = lax.map(project_and_normalize, partitions).reshape(chunk_rows, contract.head_dim)
    if rope_table_rows is None:
        cos, sin = rotary_cos_sin(
            positions,
            rotary_dim=contract.rotary_dim,
            theta=contract.theta,
            dtype=jnp.float32,
        )
    else:
        # Host FP32 cos|sin rows gathered by position: on-device cos/sin at
        # large rotary angles are inaccurate on TPU (protected V2/V3 replays).
        cos, sin = rotary_cos_sin_from_rows(rope_table_rows, positions, rotary_dim=contract.rotary_dim)
    rotated = apply_rotary(
        keys[:, : contract.rotary_dim],
        cos,
        sin,
        interleaved=contract.interleaved_rotary,
    )
    return jnp.concatenate((rotated, keys[:, contract.rotary_dim :]), axis=-1).astype(jnp.float32)


class ScoredSelectedPositions(NamedTuple):
    """Exact compact selection plus the executing score for every slot."""

    positions: jax.Array
    valid_counts: jax.Array
    scores: jax.Array


def dsa_index_keys_from_projection(
    projected: jax.Array,
    key_norm_weight: jax.Array,
    key_norm_bias: jax.Array,
    positions: jax.Array,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    key_norm_mode: Literal["divide_sqrt", "multiply_rsqrt"] = "multiply_rsqrt",
) -> jax.Array:
    """Normalize and rotate an already-computed FP32 DSA key projection."""

    tokens = projected.shape[0] if projected.ndim == 2 else -1
    require_shape("projected", projected, (tokens, contract.head_dim))
    require_shape("key_norm_weight", key_norm_weight, (contract.head_dim,))
    require_shape("key_norm_bias", key_norm_bias, (contract.head_dim,))
    require_shape("positions", positions, (tokens,))
    if projected.dtype != jnp.float32:
        raise ValueError("DSA key projection must remain FP32")
    if not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("DSA positions must have an integer dtype")

    keys = affine_layer_norm(
        projected,
        key_norm_weight,
        key_norm_bias,
        epsilon=contract.key_layer_norm_epsilon,
        mode=key_norm_mode,
    )
    cos, sin = rotary_cos_sin(
        positions,
        rotary_dim=contract.rotary_dim,
        theta=contract.theta,
        dtype=jnp.float32,
    )
    rotated = apply_rotary(
        keys[..., : contract.rotary_dim],
        cos,
        sin,
        interleaved=contract.interleaved_rotary,
    )
    return jnp.concatenate((rotated, keys[..., contract.rotary_dim :]), axis=-1)


def dsa_scores(
    query: jax.Array,
    index_keys: jax.Array,
    head_weights: jax.Array,
    *,
    precision: Literal["default", "highest"] = "highest",
) -> jax.Array:
    """Compute signed FP32 DSA scores ``[query_rows, context]``.

    ReLU is applied to every per-head query/key dot before the signed head
    weighting, exactly as in GLM-5.2. The key cache contains one shared
    128-wide key per historical token.
    """

    if query.ndim != 3 or index_keys.ndim != 2 or head_weights.ndim != 2:
        raise ValueError("DSA query/key/head-weight ranks must be 3/2/2")
    if precision not in ("default", "highest"):
        raise ValueError(f"unsupported DSA score precision {precision!r}")
    rows, heads, head_dim = query.shape
    require_shape("index_keys", index_keys, (index_keys.shape[0], head_dim))
    require_shape("head_weights", head_weights, (rows, heads))

    def compute() -> jax.Array:
        per_head = jnp.einsum(
            "rhd,sd->rhs",
            query.astype(jnp.float32),
            index_keys.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        ) * jnp.float32(head_dim**-0.5)
        per_head = jnp.maximum(per_head, jnp.float32(0.0))
        scores = jnp.einsum(
            "rh,rhs->rs",
            head_weights.astype(jnp.float32),
            per_head,
            preferred_element_type=jnp.float32,
        )
        return scores.astype(jnp.float32)

    if precision == "default":
        return compute()
    with jax.default_matmul_precision("highest"):
        return compute()


def local_topk_candidates(
    local_scores: jax.Array,
    global_positions: jax.Array,
    valid_lengths: jax.Array,
    *,
    top_k: int,
) -> tuple[jax.Array, jax.Array]:
    """Return one context shard's exact candidate score/global-position pairs."""

    if local_scores.ndim != 2:
        raise ValueError("local scores must have shape [rows, local_context]")
    rows, local_context = local_scores.shape
    require_shape("global_positions", global_positions, (local_context,))
    require_shape("valid_lengths", valid_lengths, (rows,))
    if not jnp.issubdtype(global_positions.dtype, jnp.integer):
        raise ValueError("global positions must have an integer dtype")
    if not jnp.issubdtype(valid_lengths.dtype, jnp.integer):
        raise ValueError("valid lengths must have an integer dtype")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    # Establish the global tie order before top-k.  Do not assume striped or
    # contiguous cache owners happen to present their positions in ascending
    # order.
    position_order = jnp.argsort(global_positions, stable=True)
    ordered_positions = global_positions.astype(jnp.int32)[position_order]
    ordered_scores = jnp.take(local_scores.astype(jnp.float32), position_order, axis=1)
    valid = (ordered_positions[None, :] >= 0) & (ordered_positions[None, :] < valid_lengths[:, None])
    masked = jnp.where(valid, ordered_scores, _NEGATIVE_INFINITY)
    padded_width = max(local_context, top_k)
    if padded_width != local_context:
        masked = jnp.pad(
            masked,
            ((0, 0), (0, padded_width - local_context)),
            constant_values=_NEGATIVE_INFINITY,
        )
    values, local_indices = lax.top_k(masked, top_k)
    safe_indices = jnp.minimum(local_indices, max(local_context - 1, 0))
    if local_context == 0:
        positions = jnp.full((rows, top_k), -1, dtype=jnp.int32)
    else:
        positions = jnp.take(ordered_positions, safe_indices, axis=0)
    positions = jnp.where(values == _NEGATIVE_INFINITY, jnp.int32(-1), positions)
    return values.astype(jnp.float32), positions.astype(jnp.int32)


def _merge_topk_candidates_scored(
    candidate_scores: jax.Array,
    candidate_positions: jax.Array,
    valid_lengths: jax.Array,
    *,
    top_k: int,
    global_context_size: int,
    paired_position_sort: bool = False,
) -> ScoredSelectedPositions:
    """Merge candidates and retain the scores selected by the same top-k.

    Inputs are ``[local_group, rows, candidates]``. A stable ascending-global-
    position pre-sort makes the final ``lax.top_k`` tie break identical to a
    flat global score row even if an all-gather returns groups in another order.
    """

    if candidate_scores.ndim != 3 or candidate_positions.shape != candidate_scores.shape:
        raise ValueError("candidate scores/positions must share rank-three shape")
    groups, rows, candidates = candidate_scores.shape
    require_shape("valid_lengths", valid_lengths, (rows,))
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(global_context_size, int) or global_context_size < 0:
        raise ValueError("global_context_size must be a non-negative integer")
    if groups * candidates < top_k:
        raise ValueError("candidate union is narrower than top_k")

    scores = jnp.transpose(candidate_scores, (1, 0, 2)).reshape(rows, groups * candidates)
    positions = jnp.transpose(candidate_positions, (1, 0, 2)).reshape(rows, groups * candidates)
    if type(paired_position_sort) is not bool:
        raise ValueError("paired_position_sort must be a static bool")
    if paired_position_sort:
        # Position is the ONLY key. Scores are bit-preserving payloads, so equal
        # positions retain input order exactly as in the reference argsort.
        # Avoid two row-wise permutation gathers (DB595's dominant prefix cost).
        sorted_positions, sorted_scores = lax.sort((positions, scores), dimension=1, is_stable=True, num_keys=1)
    else:
        position_order = jnp.argsort(positions, axis=1, stable=True)
        sorted_scores = jnp.take_along_axis(scores, position_order, axis=1)
        sorted_positions = jnp.take_along_axis(positions, position_order, axis=1)
    selected_scores, selected_slots = lax.top_k(sorted_scores, top_k)
    selected = jnp.take_along_axis(sorted_positions, selected_slots, axis=1)
    valid_counts = jnp.clip(
        valid_lengths.astype(jnp.int32),
        jnp.int32(0),
        jnp.int32(min(top_k, global_context_size)),
    )
    slots = lax.broadcasted_iota(jnp.int32, (rows, top_k), 1)
    live = slots < valid_counts[:, None]
    selected = jnp.where(live, selected, jnp.int32(-1))
    selected_scores = jnp.where(
        live,
        selected_scores.astype(jnp.float32),
        jnp.float32(_NEGATIVE_INFINITY),
    )
    return ScoredSelectedPositions(
        selected.astype(jnp.int32),
        valid_counts,
        selected_scores,
    )


def merge_topk_candidates_with_scores(
    candidate_scores: jax.Array,
    candidate_positions: jax.Array,
    valid_lengths: jax.Array,
    *,
    top_k: int,
    global_context_size: int,
    paired_position_sort: bool = False,
) -> ScoredSelectedPositions:
    """Return exact positions and their executing-device FP32 scores.

    This is an observer surface, not a second selector: the scores are the
    values emitted by the same ``lax.top_k`` that chose ``positions``.
    ``paired_position_sort`` is a default-off prefill experiment; it changes
    only how the stable position permutation is applied, not score arithmetic.
    """

    return _merge_topk_candidates_scored(
        candidate_scores,
        candidate_positions,
        valid_lengths,
        top_k=top_k,
        global_context_size=global_context_size,
        paired_position_sort=paired_position_sort,
    )


class PrefillDsaInputs(NamedTuple):
    query: Any
    head_weights: Any
    keys: Any
    normalized_full: Any
    contract_valid: Any


class PrefillDsaResult(NamedTuple):
    unrepaired_index_cache: Any
    repaired_index_cache: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    contract_valid: Any


def prefill_dsa_inputs(
    prepared: PreparedAttention,
    positions: Any,
    live: Any,
    weights: Bf16DsaWeights,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    linear_interpret: bool = False,
) -> PrefillDsaInputs:
    """Produce multirow queries/heads and unrepaired keys from final owners.

    Returned full normalization crosses feature4 only, for the existing M64 repair
    leaf. It is exactly the BF16 normalization used for the unrepaired producer.
    Padded rows are sanitized before arithmetic; live nonfinite operands fail health.
    """
    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("prefill DSA requires WS32 expert8/feature4 mesh")
    normalized = prepared.normalized_local
    if (
        normalized.ndim != 2
        or not 1 <= normalized.shape[0] <= 32
        or (
            normalized.shape[1] * 4 != contract.hidden_size
            or normalized.dtype != jnp.bfloat16
            or contract.num_heads != 32
            or contract.head_dim != 128
        )
    ):
        raise ValueError("prefill DSA normalized/contract geometry drifted")
    rows, hidden = normalized.shape
    if prepared.q_residual.shape != (rows, contract.q_lora_rank) or prepared.q_residual.dtype != jnp.bfloat16:
        raise ValueError("prefill DSA q residual geometry drifted")
    if positions.shape != (rows,) or positions.dtype != jnp.int32 or (live.shape != (rows,) or live.dtype != jnp.bool_):
        raise ValueError("prefill DSA requires int32 positions and boolean live rows")
    heads = contract.num_heads // 8
    if weights.wq_b_local.shape != (
        heads * contract.head_dim,
        contract.q_lora_rank,
    ) or (
        weights.wk_local.shape != (contract.head_dim, hidden)
        or weights.head_weight_local.shape != (heads, hidden)
        or weights.head_weight_local.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill DSA final weight owner geometry drifted")
    if any(
        w.shape != (contract.head_dim,) or w.dtype != jnp.bfloat16
        for w in (
            weights.key_norm_weight,
            weights.key_norm_bias,
        )
    ):
        raise ValueError("prefill DSA affine norm parameters must be BF16 owners")
    normalized = jnp.where(live[:, None], normalized, 0)
    q_input = jnp.where(live[:, None], prepared.q_residual, 0)
    positions = jnp.where(live, positions, 0)
    projected_q = resident_matmul_f32(q_input, weights.wq_b_local, interpret=linear_interpret).reshape(
        rows, heads, contract.head_dim
    )
    head_partial = lax.dot_general(
        normalized.astype(jnp.float32),
        weights.head_weight_local.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    with jax.named_scope("prefill_dsa/head_feature_reduce"):
        head = lax.psum(head_partial, "feature") * jnp.float32(contract.num_heads**-0.5)
    cos, sin = rotary_cos_sin(
        positions,
        rotary_dim=contract.rotary_dim,
        theta=contract.theta,
        dtype=jnp.float32,
    )
    rotated = apply_rotary(
        projected_q[..., : contract.rotary_dim],
        cos[:, None],
        sin[:, None],
        interleaved=True,
    )
    query = jnp.concatenate((rotated, projected_q[..., contract.rotary_dim :]), axis=-1)
    packed = jnp.concatenate((query.reshape(rows, -1), head), axis=-1)
    with jax.named_scope("prefill_dsa/query_expert_gather"):
        gathered = lax.all_gather(packed, "expert", axis=0, tiled=False)
    query_width = heads * contract.head_dim
    query = gathered[..., :query_width].transpose(1, 0, 2).reshape(rows, contract.num_heads, contract.head_dim)
    head = gathered[..., query_width:].transpose(1, 0, 2).reshape(rows, contract.num_heads)
    key_partial = resident_matmul_f32(normalized, weights.wk_local, interpret=linear_interpret)
    with jax.named_scope("prefill_dsa/key_feature_reduce"):
        projected_key = lax.psum(key_partial, "feature")
    keys = dsa_index_keys_from_projection(
        projected_key,
        weights.key_norm_weight,
        weights.key_norm_bias,
        positions,
        contract=contract,
        key_norm_mode="divide_sqrt",
    ).astype(jnp.bfloat16)
    with jax.named_scope("prefill_dsa/repair_normalized_feature_gather"):
        normalized_full = lax.all_gather(normalized, "feature", axis=1, tiled=True)
    valid = ~live | (
        (positions >= 0)
        & jnp.all(jnp.isfinite(normalized_full), axis=1)
        & jnp.all(jnp.isfinite(q_input), axis=1)
        & jnp.all(jnp.isfinite(projected_key), axis=1)
        & jnp.all(jnp.isfinite(keys), axis=1)
        & jnp.all(jnp.isfinite(query), axis=(1, 2))
        & jnp.all(jnp.isfinite(head), axis=1)
    )
    return PrefillDsaInputs(query, head, keys, normalized_full, valid)


def prefill_dsa(
    prepared: PreparedAttention,
    unrepaired_index_cache: Any,
    repaired_index_cache: Any,
    position_offset: Any,
    valid_rows: Any,
    block_table: Any,
    weights: Bf16DsaWeights,
    materialized_wk: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    linear_interpret: bool = False,
) -> PrefillDsaResult:
    """Append both key versions; score exclusively from UNREPAIRED storage.

    The FP32 repair weight must already be materialized/completed by the loader.
    Neither this function nor a successful health result promotes repaired keys.
    Caller binds offset to its committed append frontier, populated prefix and
    page-table identity, consumes all32 health flags and rejects all proposed state
    on failure. Distinct cache state/alias lifetime is a decoder-level obligation.
    """
    if unrepaired_index_cache.shape != repaired_index_cache.shape:
        raise ValueError("prefill dual index caches must share owner geometry")
    if block_table.ndim != 2 or block_table.shape[0] != 1 or block_table.shape[1] <= 0:
        raise ValueError("prefill DSA requires one nonempty shared page table")
    if any(v.shape != () or v.dtype != jnp.int32 for v in (position_offset, valid_rows)):
        raise ValueError("prefill DSA offset/count must be int32 scalars")
    rows = prepared.normalized_local.shape[0]
    layout = StageLocalKvLayout(local_parallel_size=8, packed_cache_width=contract.head_dim)
    capacity = block_table.shape[1] * layout.logical_page_size
    if capacity >= 2147483647:
        raise ValueError("prefill DSA capacity must fit positive int32")
    live = jnp.arange(rows, dtype=jnp.int32) < jnp.clip(valid_rows, 0, rows)
    safe_start = jnp.clip(position_offset, 0, capacity - 1)
    positions = safe_start + jnp.minimum(jnp.arange(rows, dtype=jnp.int32), capacity - 1 - safe_start)
    inputs = prefill_dsa_inputs(
        prepared,
        positions,
        live,
        weights,
        contract=contract,
        linear_interpret=linear_interpret,
    )
    owner = lax.axis_index("expert")
    write = write_prefill_cache_block(
        unrepaired_index_cache,
        inputs.keys,
        block_table,
        position_offset,
        valid_rows,
        owner,
        layout=layout,
    )
    # Repeat last live row AND position, including final partial blocks. Zero
    # live uses finite sanitized input0/position0 and the writer remains a no-op.
    repeat = jnp.minimum(
        jnp.arange(64, dtype=jnp.int32),
        jnp.maximum(jnp.clip(valid_rows, 0, rows) - 1, 0),
    )
    repair_positions = jnp.where(jnp.any(live), positions[repeat], 0)
    with jax.named_scope("prefill_dsa/repair"):
        repair_keys = prompt_index_key_chunk(
            inputs.normalized_full[repeat],
            repair_positions,
            materialized_wk,
            weights.key_norm_weight,
            weights.key_norm_bias,
            contract=contract,
        )[:rows].astype(jnp.bfloat16)
    repair = write_prefill_cache_block(
        repaired_index_cache,
        repair_keys,
        block_table,
        position_offset,
        valid_rows,
        owner,
        layout=layout,
    )
    page_ids = block_table[0]
    safe_pages = jnp.clip(page_ids, 0, unrepaired_index_cache.shape[0] - 1)
    keys = write.cache[safe_pages].reshape(-1, contract.head_dim)
    logical_positions = (
        jnp.arange(block_table.shape[1], dtype=jnp.int32)[:, None] * layout.logical_page_size
        + owner * layout.local_rows_per_page
        + jnp.arange(layout.local_rows_per_page, dtype=jnp.int32)[None]
    ).reshape(-1)
    # Full live-prefix page range/uniqueness was checked by the block writer.
    # Invalid metadata supplies no keys/lengths, so selection cannot read an
    # unsafe mapping and health remains false regardless of empty results.
    logical_positions = jnp.where(write.valid, logical_positions, -1)
    selected, selector_ok = prefill_dsa_one_pass(
        inputs.query,
        keys,
        inputs.head_weights,
        logical_positions,
        write.causal_lengths,
        global_context_size=capacity,
        top_k=contract.top_k,
        precision="default",
        candidates_per_owner=512,
    )
    valid = write.valid & repair.valid & jnp.all(inputs.contract_valid) & selector_ok
    return PrefillDsaResult(
        write.cache,
        repair.cache,
        selected.positions,
        selected.valid_counts,
        selected.scores,
        valid,
    )


class DsaResult(NamedTuple):
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    contract_valid: Any


def dsa_bf16(
    prepared: PreparedAttention,
    index_cache_local: Any,
    position: Any,
    block_tables: Any,
    context_lengths: Any,
    weights: Bf16DsaWeights,
    *,
    expert_axis: str,
    feature_axis: str,
    contract: DsaNumericalContract,
    cache_layout: StageLocalKvLayout,
) -> DsaResult:
    """Mirror of ``ws32_dsa_mapped`` (raw path) with BF16 wq_b / wk tables and the two-stage selection."""

    normalized = prepared.normalized_local
    local_heads = contract.num_heads // cache_layout.local_parallel_size
    owner = lax.axis_index(expert_axis)
    physical_page, local_row, target_owner, _, metadata_valid = require_decode_metadata(
        position,
        block_tables,
        context_lengths,
        owner,
        layout=cache_layout,
        physical_page_count=index_cache_local.shape[0],
    )
    projected_query = dot_f32(prepared.q_residual, weights.wq_b_local)
    query = projected_query.reshape(1, local_heads, contract.head_dim)
    with jax.named_scope("bf16_dsa/head_weight_feature_reduce"):
        reduced_head = lax.psum(head_weight_partial(normalized, weights), axis_name=feature_axis)
    local_head_weights = reduced_head * jnp.float32(contract.num_heads**-0.5)
    cos, sin = rotary_cos_sin(position, rotary_dim=contract.rotary_dim, theta=contract.theta, dtype=jnp.float32)
    rotated = apply_rotary(
        query[..., : contract.rotary_dim], cos[:, None, :], sin[:, None, :], interleaved=contract.interleaved_rotary
    )
    local_query = jnp.concatenate((rotated, query[..., contract.rotary_dim :]), axis=-1).astype(jnp.float32)
    packed_query = jnp.concatenate((local_query.reshape(1, -1), local_head_weights), axis=-1)
    local_query_width = local_heads * contract.head_dim
    with jax.named_scope("bf16_dsa/query_expert_gather"):
        gathered_query = lax.all_gather(packed_query, axis_name=expert_axis, axis=0, tiled=False)
    query = jnp.transpose(gathered_query[..., :local_query_width], (1, 0, 2)).reshape(
        1, contract.num_heads, contract.head_dim
    )
    head_weights = jnp.transpose(gathered_query[..., local_query_width:], (1, 0, 2)).reshape(1, contract.num_heads)
    key_partial = dot_f32(normalized, weights.wk_local)
    with jax.named_scope("bf16_dsa/key_feature_reduce"):
        projected_key = lax.psum(key_partial, axis_name=feature_axis)
    current_key_f32 = dsa_index_keys_from_projection(
        projected_key,
        weights.key_norm_weight,
        weights.key_norm_bias,
        position,
        contract=contract,
    ).astype(jnp.float32)
    current_key = current_key_f32.astype(index_cache_local.dtype)

    def write_current(value: Any) -> Any:
        return value.at[physical_page, local_row].set(current_key[0])

    index_cache_local = lax.cond(
        metadata_valid[0] & (owner == target_owner), write_current, lambda v: v, index_cache_local
    )
    from glm_tpu.layers.attention.dsa_indexer import score_cache_pages, two_stage_topk

    local_scores, global_positions = score_cache_pages(
        query,
        index_cache_local,
        head_weights,
        block_tables,
        layout=cache_layout,
        owner=owner,
    )
    selected, _ = two_stage_topk(
        local_scores,
        global_positions,
        context_lengths,
        top_k=contract.top_k,
        global_context_size=block_tables.shape[1] * cache_layout.logical_page_size,
        positions_in_order=True,
        expert_axis=expert_axis,
    )
    selection_valid = canonicalize_selected_positions(
        SelectedPositions(selected.positions, selected.valid_counts)
    ).contract_valid
    return DsaResult(
        index_cache_local, selected.positions, selected.valid_counts, selected.scores, metadata_valid & selection_valid
    )
