"""Greedy speculative acceptance; target predictions alone authorize output."""
from typing import Any, NamedTuple

import jax.numpy as jnp


class GreedyAcceptance(NamedTuple):
    # Output token storage is target_predictions[:emitted_count]. Consuming that
    # many verifier input rows leaves the last emitted token pending for decode.
    emitted_count: Any
    accepted_draft_count: Any
    accepted_by_position: Any
    stopped_on_eos: Any
    stopped_on_length: Any
    valid: Any


def greedy_acceptance(input_tokens, target_predictions, remaining, *, eos_token_ids=()):
    """Choose the matching draft prefix plus one correction/bonus prediction.

    input_tokens[0] is already the target's pending token. Draft proposals begin
    at index 1. This helper does no sampling, cache mutation or token delivery.
    """
    if (input_tokens.ndim != 1 or not 1 <= input_tokens.size <= 8
            or target_predictions.shape != input_tokens.shape
            or input_tokens.dtype != jnp.int32 or target_predictions.dtype != jnp.int32
            or remaining.shape != () or remaining.dtype != jnp.int32
            or not isinstance(eos_token_ids, tuple)
            or any(type(t) is not int or not 0 <= t < 2**31 for t in eos_token_ids)):
        raise ValueError('greedy speculation requires matching int32 rows, scalar budget and static EOS IDs')
    valid = (remaining >= 0) & jnp.all(input_tokens >= 0) & jnp.all(target_predictions >= 0)
    matched = jnp.cumprod((input_tokens[1:] == target_predictions[:-1]).astype(jnp.int32))
    accepted = jnp.sum(matched, dtype=jnp.int32)
    count = jnp.minimum(accepted + 1, jnp.maximum(remaining, 0))
    positions = jnp.arange(input_tokens.size, dtype=jnp.int32)
    eos = jnp.zeros_like(target_predictions, dtype=jnp.bool_)
    for token in eos_token_ids:
        eos |= target_predictions == token
    first_eos = jnp.min(jnp.where(eos & (positions < count), positions, input_tokens.size))
    stopped_eos = first_eos < count
    count = jnp.minimum(count, first_eos + 1)
    count = jnp.where(valid, count, 0)
    return GreedyAcceptance(count, jnp.minimum(accepted, count),
        (jnp.arange(input_tokens.size - 1) < accepted) & (jnp.arange(input_tokens.size - 1) < count),
        valid & stopped_eos, valid & ~stopped_eos & (count == remaining), valid)
