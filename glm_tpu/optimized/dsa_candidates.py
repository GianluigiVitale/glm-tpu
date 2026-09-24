"""D10 exact shortlists with a global cut check and full-width fallback."""
from typing import Any

from jax import lax
import jax.numpy as jnp

from .reference.dsa import local_topk_candidates, merge_topk_candidates_with_scores


def two_stage_topk_mapped(
    scores: Any, positions: Any, valid_lengths: Any, *, top_k: int,
    global_context_size: int, candidates_per_owner: int = 512,
    expert_axis: str = "expert", positions_in_order: bool = False,
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
        pad = max(0, k-scores.shape[1])
        masked = jnp.pad(masked, ((0, 0), (0, pad)), constant_values=-jnp.inf)
        pos = jnp.pad(positions, ((0, pad),), constant_values=-1)
        values, index = lax.top_k(masked, k)
        return values, jnp.where(values == -jnp.inf, -1, pos[index]).astype(jnp.int32)

    def merge(values, indices):
        values = lax.all_gather(values, expert_axis, axis=0, tiled=False)
        indices = lax.all_gather(indices, expert_axis, axis=0, tiled=False)
        return merge_topk_candidates_with_scores(
            values, indices, valid_lengths, top_k=top_k,
            global_context_size=global_context_size, paired_position_sort=True,
        )

    def full(_):
        values, indices = local(top_k)
        return merge(values, indices)

    values, indices = local(width+1)
    # Pad the union locally if L*owners<K, so a short configuration remains
    # correct through fallback rather than assuming a particular mesh size.
    union_width = width * lax.axis_size(expert_axis)
    pad = max(0, (top_k + lax.axis_size(expert_axis)-1)//lax.axis_size(expert_axis)-width)
    short = merge(jnp.pad(values[:, :width], ((0, 0), (0, pad)), constant_values=-jnp.inf),
                  jnp.pad(indices[:, :width], ((0, 0), (0, pad)), constant_values=-1))
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
    outranks = omitted_live & ((omitted_key > cutoff_key) |
        ((omitted_key == cutoff_key) & ((cutoff_position < 0) | (omitted_position < cutoff_position))))
    # Nonfinite input scores use the frozen path without relying on comparisons
    # involving NaN. Negative infinities are already the frozen masked sentinel.
    uncertain = jnp.any(jnp.isnan(scores) | jnp.isposinf(scores))
    fallback = lax.pmax((jnp.any(outranks) | uncertain | (union_width < top_k)).astype(jnp.int32), expert_axis).astype(jnp.bool_)
    return lax.cond(fallback, full, lambda _: short, operand=None), fallback


def score_cache_pages(query, cache, head_weights, block_tables, *, layout, owner):
    """Score physical keys in place; gather scalar scores into logical order.

    The projection/FP32 DSA arithmetic is the frozen dsa_scores. Page mapping
    moves one scalar per key rather than its full 128-component cache vector.
    Return ascending logical positions for the sort-free local top-k path.
    """
    from .reference.dsa import dsa_scores

    physical_scores = dsa_scores(query, cache.reshape(-1, cache.shape[-1]), head_weights, precision="highest")
    pages = block_tables[0]
    page_ok = (pages >= 0) & (pages < cache.shape[0])
    safe_pages = jnp.clip(pages, 0, cache.shape[0]-1)
    local_rows = jnp.arange(layout.local_rows_per_page, dtype=jnp.int32)
    indices = safe_pages[:, None] * layout.local_rows_per_page + local_rows[None]
    scores = jnp.take(physical_scores, indices.reshape(-1), axis=1)
    positions = (jnp.arange(pages.shape[0], dtype=jnp.int32)[:, None] * layout.logical_page_size
                 + owner * layout.local_rows_per_page + local_rows[None])
    positions = jnp.where(page_ok[:, None], positions, -1).reshape(-1)
    return scores, positions


def prefill_dsa_one_pass_mapped(query, keys, head_weights, positions, valid_lengths, *,
                                 global_context_size, top_k=2048, precision="highest",
                                 candidates_per_owner=512):
    """P2: one score row and one shortlist per prefill tile, with frozen health.

    Removes the repeated score/top-k/merge chain over 512-key blocks. Temporary
    per-head scores are rows*32*local_context FP32 values (64 MiB at 128K/M32).
    This is opt-in and keeps causal lengths and finite-score admission intact.
    """
    from .reference.attention import canonicalize_selected_positions
    from .reference.dsa import SelectedPositions, dsa_scores

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
    metadata_ok = (jnp.all((positions >= -1) & (positions < global_context_size))
        & jnp.all((positions == -1) | (positions > previous))
        & jnp.all((valid_lengths >= 0) & (valid_lengths <= global_context_size)))
    scores = dsa_scores(query, keys, head_weights, precision=precision)
    visible = (positions[None] >= 0) & (positions[None] < valid_lengths[:, None])
    healthy = metadata_ok & jnp.all(jnp.isfinite(query)) & jnp.all(jnp.isfinite(head_weights)) & jnp.all(jnp.isfinite(scores) | ~visible)
    selected, _ = two_stage_topk_mapped(scores, positions, valid_lengths, top_k=top_k,
        global_context_size=global_context_size, candidates_per_owner=candidates_per_owner, positions_in_order=True)
    canonical = canonicalize_selected_positions(SelectedPositions(selected.positions, selected.valid_counts))
    live = jnp.arange(top_k)[None] < selected.valid_counts[:, None]
    selected_ok = jnp.all(canonical.contract_valid) & jnp.all(jnp.where(live,
        jnp.isfinite(selected.scores) & (selected.positions < valid_lengths[:, None]), True))
    return selected, healthy & selected_ok
