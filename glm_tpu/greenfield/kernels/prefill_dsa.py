"""Unwired, bounded causal multirow DSA selection reference for prefill.

Tiles over keys, not over complete decoder steps. Reuses the existing scorer
and exact candidate selector; only [rows, heads, key_tile] score intermediates
and [rows, top_k] carried candidates are needed. Performance is not admitted.
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .reference.dsa import (
    SelectedPositions,
    dsa_scores,
    local_topk_candidates,
    merge_topk_candidates_with_scores,
)
from .reference.attention import canonicalize_selected_positions


class PrefillDsaCandidates(NamedTuple):
    scores: Any
    positions: Any
    valid: Any


def causal_dsa_local_candidates(
    query: Any,
    index_keys: Any,
    head_weights: Any,
    global_positions: Any,
    valid_lengths: Any,
    *,
    global_context_size: int,
    top_k: int = 2048,
    key_tile: int = 4096,
    precision: Literal["default", "highest"] = "highest",
) -> PrefillDsaCandidates:
    """Select each row's exact local candidates from its own causal score row.

    Keys include prior and current prompt-block UNREPAIRED index rows. Positions
    must be increasing among live entries, with -1 for holes; they describe ONE
    context owner. Lengths are exclusive absolute-position bounds per query row.
    No cache writes/repair or collective occurs here. The caller merges these
    candidates over its validated local context group and must gate all health.
    Future-only tiles skip scorer and selector work. Shape/tile checks bound the
    largest per-head temporary to 32*32*4096 F32 elements (16MiB).
    """
    if query.ndim != 3 or index_keys.ndim != 2 or head_weights.ndim != 2:
        raise ValueError("prefill DSA requires query/key/head-weight ranks3/2/2")
    rows, heads, dim = query.shape
    context = index_keys.shape[0]
    if not (1 <= rows <= 32 and heads == 32 and dim == 128 and context > 0):
        raise ValueError(
            "prefill DSA requires1..32 rows,32 heads,128 dimensions,nonempty keys"
        )
    if index_keys.shape != (context, dim) or head_weights.shape != (rows, heads):
        raise ValueError("prefill DSA query/key/head-weight geometry differs")
    if (
        query.dtype != jnp.float32
        or head_weights.dtype != jnp.float32
        or index_keys.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill DSA requires F32 queries/weights and BF16 keys")
    if global_positions.shape != (context,) or valid_lengths.shape != (rows,):
        raise ValueError("prefill DSA position/length geometry differs")
    if global_positions.dtype != jnp.int32 or valid_lengths.dtype != jnp.int32:
        raise ValueError("prefill DSA positions/lengths must be int32")
    if type(global_context_size) is not int or not 0 < global_context_size < 2147483647:
        raise ValueError("prefill DSA global context must fit positive int32")
    if type(top_k) is not int or not 0 < top_k <= 2048:
        raise ValueError("prefill DSA top_k must be1..2048")
    if type(key_tile) is not int or not 128 <= key_tile <= 4096 or key_tile % 128:
        raise ValueError("prefill DSA key tile must be a multiple of128 up to4096")
    if precision not in ("default", "highest"):
        raise ValueError("unsupported prefill DSA precision")

    seen = lax.associative_scan(jnp.maximum, global_positions)
    previous = jnp.concatenate((jnp.full((1,), -1, jnp.int32), seen[:-1]))
    metadata_ok = (
        jnp.all((global_positions >= -1) & (global_positions < global_context_size))
        & jnp.all((global_positions == -1) | (global_positions > previous))
        & jnp.all((valid_lengths >= 0) & (valid_lengths <= global_context_size))
    )
    health = (
        metadata_ok & jnp.all(jnp.isfinite(query)) & jnp.all(jnp.isfinite(head_weights))
    )
    padded = ((context + key_tile - 1) // key_tile) * key_tile
    keys = jnp.pad(index_keys, ((0, padded - context), (0, 0)))
    positions = jnp.pad(global_positions, ((0, padded - context),), constant_values=-1)
    initial_scores = jnp.full((rows, top_k), -jnp.inf, jnp.float32)
    initial_positions = jnp.full((rows, top_k), -1, jnp.int32)

    def tile_body(tile, carried):
        block_positions = lax.dynamic_slice(positions, (tile * key_tile,), (key_tile,))
        visible = (block_positions[None, :] >= 0) & (
            block_positions[None, :] < valid_lengths[:, None]
        )

        def compute(current):
            previous_scores, previous_positions, ok = current
            block_keys = lax.dynamic_slice(keys, (tile * key_tile, 0), (key_tile, dim))
            scores = dsa_scores(query, block_keys, head_weights, precision=precision)
            ok = ok & jnp.all(jnp.isfinite(scores) | ~visible)
            new_scores, new_positions = local_topk_candidates(
                scores, block_positions, valid_lengths, top_k=top_k
            )
            merged = merge_topk_candidates_with_scores(
                jnp.stack((previous_scores, new_scores)),
                jnp.stack((previous_positions, new_positions)),
                valid_lengths,
                top_k=top_k,
                global_context_size=global_context_size,
            )
            # This is local coverage, not a global valid-count declaration.
            return merged.scores, merged.positions, ok

        return lax.cond(
            metadata_ok & jnp.any(visible), compute, lambda current: current, carried
        )

    scores, selected_positions, valid = lax.fori_loop(
        0, padded // key_tile, tile_body, (initial_scores, initial_positions, health)
    )
    return PrefillDsaCandidates(scores, selected_positions, valid)


def ws32_prefill_dsa_from_query_mapped(
    query: Any,
    index_keys_local: Any,
    head_weights: Any,
    global_positions_local: Any,
    valid_lengths: Any,
    *,
    global_context_size: int,
    top_k: int = 2048,
    key_tile: int = 4096,
    precision: Literal["default", "highest"] = "highest",
) -> tuple[Any, Any]:
    """Expert8 candidate exchange, returning scored positions and local health.

    Caller supplies a validated WS32 expert8/feature4 mesh, queries/lengths
    replicated across both axes, and context keys/positions sharded over expert.
    Only compact [rows,top_k] candidates travel; no cache/hidden reconstruction.
    All32 local health flags must be consumed by serving admission.
    Disjoint complete cache coverage across owners must be separately validated
    by the caller's cache/page contract. Selected-candidate checks cannot detect
    absent or duplicated keys that never enter the candidate set.
    """
    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("prefill DSA requires WS32 expert8/feature4 mesh")
    local = causal_dsa_local_candidates(
        query,
        index_keys_local,
        head_weights,
        global_positions_local,
        valid_lengths,
        global_context_size=global_context_size,
        top_k=top_k,
        key_tile=key_tile,
        precision=precision,
    )
    with jax.named_scope("greenfield_ws32_prefill_dsa/candidates"):
        scores = lax.all_gather(local.scores, "expert", axis=0, tiled=False)
        positions = lax.all_gather(local.positions, "expert", axis=0, tiled=False)
    selected = merge_topk_candidates_with_scores(
        scores,
        positions,
        valid_lengths,
        top_k=top_k,
        global_context_size=global_context_size,
    )
    canonical = canonicalize_selected_positions(
        SelectedPositions(selected.positions, selected.valid_counts)
    )
    live = jnp.arange(top_k)[None, :] < selected.valid_counts[:, None]
    selected_ok = jnp.all(canonical.contract_valid) & jnp.all(
        jnp.where(
            live,
            jnp.isfinite(selected.scores)
            & (selected.positions < valid_lengths[:, None]),
            True,
        )
    )
    return selected, local.valid & selected_ok
