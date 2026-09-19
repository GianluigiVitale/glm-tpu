"""Candidate-set nucleus sampling at the WS32 output boundary (opt-in challenger).

The frozen ``nucleus_sample`` performs one stable two-key ``lax.sort`` of the
whole 154,880-entry vocabulary row plus two full-length ``cumsum``s per
generated token.  XLA sorts are slow on TPU: the reference GLM-5.3-Flash engine
notes that its sort-based sampler cost about 2 ms per token on v5e and
replaces it with a candidate top-k (``glm53/engine.py::device_sample``).  On
this pod the sampled request path (DB621) measures about 18 ms/token more
than the greedy path (DB610) with the same decoder body; the full-vocabulary
sort and cumsums are the only work that differs.

This module keeps the frozen sampler's exact rule ("descending score, ascending
token id; include the token that crosses ``top_p``; sample proportionally
inside that prefix with one replayable FP32 uniform") but evaluates it on a
candidate set: the top ``candidates_per_shard`` logits of every expert-8
vocabulary shard, all-gathered and ordered by the same key.  The candidate set
is a superset of the global top-``candidates_per_shard``, and the nucleus is
provably inside it whenever no shard's last candidate lies inside the kept
prefix.  That condition is evaluated on device from replicated data; when it
fails, or when the health checks fail, the SAME call falls back to the frozen
full-vocabulary sampler through ``lax.cond`` so the result is always defined.

Numerical boundary (documented, not hidden): the softmax normalizer is the
psum of per-shard sums instead of a descending-order serial sum.  The
candidate prefix, its ordering and every probability ratio are otherwise the
frozen sampler's; a keep-set difference can only occur when ``previous_mass``
lands within one FP32 ulp of ``top_p``.  ``tests/perf`` finds zero token
differences over its randomized draws, and the fallback predicate is exercised
on flat distributions.
"""

from __future__ import annotations

from typing import Any

import jax
from jax import lax
import jax.numpy as jnp

from ..greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped
from ..greenfield.kernels.ws32_io import (
    Ws32GreedySampleResult,
    Ws32SplitGreedySampleResult,
    ws32_logits_mapped,
)
from ..greenfield.kernels.ws32_sampling import NucleusConfig, ws32_nucleus_sample_mapped


def _candidate_token(
    scores: Any, ids: Any, shard_last: Any, normalizer: Any, uniform: Any,
    *, config: NucleusConfig,
) -> tuple[Any, Any]:
    """Frozen nucleus rule over ordered candidates -> (token, nucleus_inside)."""

    weights = jnp.exp((scores - scores[0]) / jnp.float32(config.temperature))
    probabilities = weights / normalizer
    previous_mass = jnp.concatenate(
        (jnp.zeros((1,), jnp.float32), jnp.cumsum(probabilities)[:-1])
    )
    keep = (previous_mass < jnp.float32(config.top_p)) & (weights > 0)
    mass = jnp.cumsum(jnp.where(keep, weights, 0))
    draw = jnp.clip(uniform, 0, 1) * mass[-1]
    index = jnp.minimum(jnp.sum(mass <= draw), jnp.sum(keep) - 1)
    # Every shard's weakest candidate must fall OUTSIDE the kept prefix;
    # otherwise that shard may hide tokens that belong to the nucleus.
    inside = ~jnp.any(keep & shard_last)
    return ids[index], inside


def ws32_nucleus_sample_candidates_mapped(
    local_logits: Any,
    uniform: Any,
    *,
    vocab_size: int,
    config: NucleusConfig,
    candidates_per_shard: int = 256,
    expert_axis: str = "expert",
) -> Ws32GreedySampleResult:
    """Sample one token from expert-sharded logits via per-shard candidates.

    Exactly the frozen ``ws32_nucleus_sample_mapped`` interface.  Falls back
    to that function (one replicated ``lax.cond`` branch) whenever the
    candidate set cannot be proven to contain the nucleus or the inputs are
    unhealthy, so unhealthy inputs still fail closed with token -1.
    """

    if not isinstance(config, NucleusConfig):
        raise ValueError("explicit NucleusConfig required")
    if (local_logits.ndim != 2 or local_logits.shape[0] != 1
            or local_logits.dtype != jnp.bfloat16 or local_logits.shape[1] * 8 != vocab_size):
        raise ValueError("WS32 candidate sampler requires one BF16 expert-8 vocabulary shard")
    if uniform.shape != () or uniform.dtype != jnp.float32:
        raise ValueError("sampling uniform must be one FP32 scalar")
    local_vocab = local_logits.shape[1]
    if (type(candidates_per_shard) is not int or candidates_per_shard <= 0):
        raise ValueError("candidates_per_shard must be a positive integer")
    k = min(candidates_per_shard, local_vocab)

    finite = jnp.isfinite(local_logits[0])
    scores = jnp.where(finite, local_logits[0], 0).astype(jnp.float32)
    owner = lax.axis_index(expert_axis).astype(jnp.int32)
    ids = owner * jnp.int32(local_vocab) + jnp.arange(local_vocab, dtype=jnp.int32)
    # lax.top_k returns the lowest index among equal scores: ascending token id
    # within the shard, the frozen tie rule.
    local_scores, local_slots = lax.top_k(scores, k)
    local_ids = jnp.take(ids, local_slots)
    last = jnp.arange(k, dtype=jnp.int32) == jnp.int32(k - 1)
    with jax.named_scope("glm_perf_nucleus_candidates/expert_candidate_exchange"):
        gathered_scores = lax.all_gather(local_scores[None], expert_axis, axis=0, tiled=False)
        gathered_ids = lax.all_gather(local_ids[None], expert_axis, axis=0, tiled=False)
        all_finite = lax.pmin(jnp.all(finite).astype(jnp.int32), expert_axis) != 0
    candidate_scores = gathered_scores.reshape(-1)
    candidate_ids = gathered_ids.reshape(-1)
    shard_last = jnp.tile(last, (8,))
    negative, ordered_ids, ordered_last = lax.sort(
        (-candidate_scores, candidate_ids, shard_last.astype(jnp.int32)),
        dimension=0, num_keys=2, is_stable=True,
    )
    ordered_scores = -negative
    # The global maximum is the first ordered candidate on every chip; the
    # normalizer covers the WHOLE vocabulary like the frozen sampler.
    local_weight_sum = jnp.sum(
        jnp.exp((scores - ordered_scores[0]) / jnp.float32(config.temperature))
    )
    with jax.named_scope("glm_perf_nucleus_candidates/normalizer_expert_reduce"):
        normalizer = lax.psum(local_weight_sum, expert_axis)
    token, inside = _candidate_token(
        ordered_scores, ordered_ids, ordered_last.astype(jnp.bool_), normalizer, uniform,
        config=config,
    )
    healthy = all_finite & jnp.isfinite(uniform) & (uniform >= 0) & (uniform < 1)
    use_candidates = healthy & (inside | (k == local_vocab))

    def fast(_: None) -> Ws32GreedySampleResult:
        return Ws32GreedySampleResult(token[None].astype(jnp.int32), healthy[None])

    def exact(_: None) -> Ws32GreedySampleResult:
        return ws32_nucleus_sample_mapped(
            local_logits, uniform, vocab_size=vocab_size, config=config,
            expert_axis=expert_axis,
        )

    # use_candidates derives only from all-gathered/psum data: identical on
    # every chip, so the collectives inside the exact branch stay uniform.
    return lax.cond(use_candidates, fast, exact, operand=None)


def ws32_split_nucleus_sample_candidates_mapped(
    hidden_update_local: Any, carried_residual_local: Any,
    final_norm_weight_local: Any, lm_head_local: Any, uniform: Any, *,
    hidden_size: int, vocab_size: int, config: NucleusConfig,
    candidates_per_shard: int = 256,
    feature_axis: str = "feature", expert_axis: str = "expert",
    rms_norm_epsilon: float = 1e-5,
) -> Ws32SplitGreedySampleResult:
    """Frozen fused norm/logits head with the candidate-set sampler."""

    normalized, final_residual = ws32_fused_add_rms_norm_mapped(
        hidden_update_local, carried_residual_local, final_norm_weight_local,
        global_hidden_size=hidden_size, feature_axis=feature_axis,
        epsilon=rms_norm_epsilon,
    )
    logits = ws32_logits_mapped(normalized, lm_head_local, vocab_size=vocab_size,
                               feature_axis=feature_axis)
    sampled = ws32_nucleus_sample_candidates_mapped(
        logits, uniform, vocab_size=vocab_size, config=config,
        candidates_per_shard=candidates_per_shard, expert_axis=expert_axis,
    )
    return Ws32SplitGreedySampleResult(sampled.token_id, sampled.contract_valid, final_residual)
