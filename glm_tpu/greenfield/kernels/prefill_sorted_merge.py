"""Exact sorted-pair half merge for prefill; no TPU performance admission.

This is NOT a replacement for the generic unordered-candidate selector. Each
input must already be score-descending/position-ascending, with disjoint live
positions and canonical (-inf, -1) padding. The causal key-tile loop establishes
these preconditions. Its existing health checks still reject nonfinite scores.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax


def merge_sorted_candidate_pair(
    left_scores: jax.Array,
    left_positions: jax.Array,
    right_scores: jax.Array,
    right_positions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the best K pairs from two sorted [rows,K] lists, K a power of two.

    Reverse the right list to form a bitonic sequence. One half-cleaner retains
    its best K elements; log2(K) vector compare/exchange stages order that half.
    No K-step scalar selection loop, general sort, or permutation gather occurs.

    Integer score keys preserve all F32 payload bits and the pinned top_k's
    finite total order, including +0 before -0; identical keys use lower global
    position. NaNs/infinite *live* scores are outside the caller's healthy domain.
    This specialized primitive does not validate sortedness/coverage at runtime;
    it must not consume arbitrary lists or replace global coverage checks.
    """
    shape = left_scores.shape
    if (
        len(shape) != 2
        or not 1 <= shape[0] <= 32
        or not 1 <= shape[1] <= 2048
        or shape[1] & (shape[1] - 1)
        or any(
            x.shape != shape for x in (left_positions, right_scores, right_positions)
        )
        or left_scores.dtype != jnp.float32
        or right_scores.dtype != jnp.float32
        or left_positions.dtype != jnp.int32
        or right_positions.dtype != jnp.int32
    ):
        raise ValueError(
            "sorted merge needs F32/S32 [1..32,power-of-two K<=2048] pairs"
        )

    def score_key(scores):
        bits = lax.bitcast_convert_type(scores, jnp.int32)
        return jnp.where(bits < 0, bits ^ jnp.int32(0x7FFFFFFF), bits)

    def choose_left(ak, ap, bk, bp):
        return (ak > bk) | ((ak == bk) & (ap <= bp))

    left_key = score_key(left_scores)
    right_key = lax.rev(score_key(right_scores), (1,))
    right_pos = lax.rev(right_positions, (1,))
    take_left = choose_left(left_key, left_positions, right_key, right_pos)
    keys = jnp.where(take_left, left_key, right_key)
    positions = jnp.where(take_left, left_positions, right_pos)

    rows, width = shape
    stride = width // 2
    while stride:
        # Static reshape/slices, not data-dependent gather indices. Both halves
        # remain in descending order after the final stride=1 exchange.
        key_blocks = keys.reshape(rows, width // (2 * stride), 2, stride)
        pos_blocks = positions.reshape(rows, width // (2 * stride), 2, stride)
        ak, bk = key_blocks[:, :, 0, :], key_blocks[:, :, 1, :]
        ap, bp = pos_blocks[:, :, 0, :], pos_blocks[:, :, 1, :]
        take_left = choose_left(ak, ap, bk, bp)
        keys = jnp.stack(
            (jnp.where(take_left, ak, bk), jnp.where(take_left, bk, ak)), axis=2
        ).reshape(shape)
        positions = jnp.stack(
            (jnp.where(take_left, ap, bp), jnp.where(take_left, bp, ap)), axis=2
        ).reshape(shape)
        stride //= 2

    bits = jnp.where(keys < 0, keys ^ jnp.int32(0x7FFFFFFF), keys)
    return lax.bitcast_convert_type(bits, jnp.float32), positions
