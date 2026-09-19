"""P6 exact candidate payload sort, an opt-in alternative to top-k plus gather.

The frozen merge's validation and count/sentinel handling are preserved. This
candidate is unwired until CPU proof and TPU measurement establish its boundary.
"""
import jax
from jax import lax
import jax.numpy as jnp
from ..greenfield.kernels.reference.dsa import (
    _merge_topk_candidates_scored, _require_shape, _NEGATIVE_INFINITY, ScoredSelectedPositions,
)


def sort_payload_candidate_merge(
    candidate_scores: jax.Array,
    candidate_positions: jax.Array,
    valid_lengths: jax.Array,
    *,
    top_k: int,
    global_context_size: int,
    paired_position_sort: bool = False,
) -> ScoredSelectedPositions:
    """Sort descending FP32 score keys and ascending positions with raw payloads.

    Inputs are ``[local_group, rows, candidates]``. NaN/+inf inputs use the
    frozen merge; finite scores, signed zeros and -inf sentinels use the exact
    integer score ordering without a selected-position gather.
    """

    if candidate_scores.ndim != 3 or candidate_positions.shape != candidate_scores.shape:
        raise ValueError("candidate scores/positions must share rank-three shape")
    groups, rows, candidates = candidate_scores.shape
    _require_shape("valid_lengths", valid_lengths, (rows,))
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(global_context_size, int) or global_context_size < 0:
        raise ValueError("global_context_size must be a non-negative integer")
    if groups * candidates < top_k:
        raise ValueError("candidate union is narrower than top_k")

    scores = jnp.transpose(candidate_scores, (1, 0, 2)).reshape(
        rows, groups * candidates
    )
    positions = jnp.transpose(candidate_positions, (1, 0, 2)).reshape(
        rows, groups * candidates
    )
    if type(paired_position_sort) is not bool:
        raise ValueError("paired_position_sort must be a static bool")
    if candidate_scores.dtype != jnp.float32 or candidate_positions.dtype != jnp.int32:
        raise ValueError('payload-sort merge requires FP32 scores and int32 positions')
    # Preserve top_k's FP32 total order, including +0 ahead of -0. Position
    # is the second key; scores remain untouched payload bits. Sorting the
    # payload directly removes the final row-wise selected-position gather.
    bits = lax.bitcast_convert_type(scores,jnp.int32)
    score_key = jnp.where(bits<0,bits ^ jnp.int32(0x7FFFFFFF),bits)
    def ordered(_):
        _, chosen_positions, chosen_scores = lax.sort(
            (~score_key,positions,scores),dimension=1,is_stable=True,num_keys=2)
        return chosen_positions[:,:top_k],chosen_scores[:,:top_k]
    def unusual(_):
        original = _merge_topk_candidates_scored(candidate_scores,candidate_positions,valid_lengths,
            top_k=top_k,global_context_size=global_context_size,paired_position_sort=paired_position_sort)
        return original.positions,original.scores
    selected, selected_scores = lax.cond(jnp.any(jnp.isnan(scores) | jnp.isposinf(scores)),
        unusual,ordered,None)
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
