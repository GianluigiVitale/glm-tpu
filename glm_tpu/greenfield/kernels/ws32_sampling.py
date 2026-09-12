"""Opt-in request sampling at the existing WS32 output boundary.

No model-layer or greedy-path changes. Nucleus sampling gathers one vocabulary
row over expert-8 only, after the existing feature-4 logit reduction. This is
not a layer hidden-state gather. Performance and full-model integration are
not admitted by this module's CPU tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp

from .ws32 import ws32_fused_add_rms_norm_mapped
from .ws32_io import Ws32GreedySampleResult, Ws32SplitGreedySampleResult, ws32_logits_mapped


@dataclass(frozen=True, slots=True)
class NucleusConfig:
    temperature: float = 1.0
    top_p: float = 0.95

    def __post_init__(self) -> None:
        for value in (self.temperature, self.top_p):
            if type(value) is not float or not math.isfinite(value):
                raise ValueError("sampling parameters require finite floats")
        # TPU FP32 can flush subnormals: reject host values that would turn a
        # valid-looking positive parameter into zero or infinity on device.
        minimum = 1.1754943508222875e-38
        if not minimum <= self.temperature <= 3.4028234663852886e38 or not minimum <= self.top_p <= 1:
            raise ValueError("temperature/top_p must fit positive normal FP32; top_p <= 1")


def request_uniform(*, seed: int, request_id: str, token_index: int) -> float:
    """Stateless replayable SHA256/24-bit uniform in [0,1), exactly FP32.

    Persist this algorithm identity, seed, request id and next token index with
    the whole request state. Resume must not reset the counter. One draw per
    delivered/generated token, including the first token after prefill. This
    protocol is explicit and does not claim the model card's unspecified RNG.
    """
    if any(type(x) is not int or not 0 <= x < 2**64 for x in (seed, token_index)):
        raise ValueError("seed and token index must be uint64 integers")
    if type(request_id) is not str or not request_id:
        raise ValueError("a nonempty request id is required")
    identity = sha256(request_id.encode("utf-8")).digest()
    digest = sha256(b"glm-ws32-request-uniform-v1\0" + seed.to_bytes(8, "big")
                    + identity + token_index.to_bytes(8, "big")).digest()
    return int.from_bytes(digest[:3], "big") / 2**24


def nucleus_sample(logits: Any, uniform: Any, *, config: NucleusConfig) -> Ws32GreedySampleResult:
    """One-row FP32 nucleus CDF, descending score then ascending token-id ties.

    Include the token that crosses top_p and sample proportionally within that
    prefix. Invalid inputs fail closed with token -1 and false health; they must
    never be emitted by the request controller. No batched padding/dead rows.
    """
    if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] <= 0:
        raise ValueError("sampling requires exactly one nonempty vocabulary row")
    if logits.dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError("sampling requires BF16/FP32 logits")
    if uniform.shape != () or uniform.dtype != jnp.float32:
        raise ValueError("sampling uniform must be one FP32 scalar")
    healthy = jnp.all(jnp.isfinite(logits)) & jnp.isfinite(uniform) & (uniform >= 0) & (uniform < 1)
    scores = jnp.where(jnp.isfinite(logits[0]), logits[0], 0).astype(jnp.float32)
    ids = jnp.arange(scores.size, dtype=jnp.int32)
    negative, ids = lax.sort((-scores, ids), dimension=0, num_keys=2, is_stable=True)
    scores = -negative
    weights = jnp.exp((scores - scores[0]) / jnp.float32(config.temperature))
    probabilities = weights / jnp.sum(weights)
    previous_mass = jnp.concatenate((jnp.zeros((1,), jnp.float32), jnp.cumsum(probabilities)[:-1]))
    keep = (previous_mass < jnp.float32(config.top_p)) & (weights > 0)
    mass = jnp.cumsum(jnp.where(keep, weights, 0))
    draw = jnp.clip(uniform, 0, 1) * mass[-1]
    # Multiplication can round a draw just below total to total. Clamp to the
    # final positive retained mass, never an excluded/underflowed tail token.
    index = jnp.minimum(jnp.sum(mass <= draw), jnp.sum(keep) - 1)
    token = jnp.where(healthy, ids[index], jnp.int32(-1))
    return Ws32GreedySampleResult(token[None], healthy[None])


def ws32_nucleus_sample_mapped(
    local_logits: Any, uniform: Any, *, vocab_size: int,
    config: NucleusConfig, expert_axis: str = "expert",
) -> Ws32GreedySampleResult:
    """Gather only expert-sharded final vocabulary scores; return one token."""
    if (local_logits.ndim != 2 or local_logits.shape[0] != 1
            or local_logits.dtype != jnp.bfloat16 or local_logits.shape[1] * 8 != vocab_size):
        raise ValueError("WS32 sampler requires one BF16 expert-8 vocabulary shard")
    with jax.named_scope("greenfield_ws32_nucleus/final_vocab_exchange"):
        logits = lax.all_gather(local_logits, expert_axis, axis=1, tiled=True)
    return nucleus_sample(logits, uniform, config=config)


def ws32_split_nucleus_sample_mapped(
    hidden_update_local: Any, carried_residual_local: Any,
    final_norm_weight_local: Any, lm_head_local: Any, uniform: Any, *,
    hidden_size: int, vocab_size: int, config: NucleusConfig,
    feature_axis: str = "feature", expert_axis: str = "expert",
    rms_norm_epsilon: float = 1e-5,
) -> Ws32SplitGreedySampleResult:
    """Use the original fused norm/logits, not re-normalized rounded residuals."""
    normalized, final_residual = ws32_fused_add_rms_norm_mapped(
        hidden_update_local, carried_residual_local, final_norm_weight_local,
        global_hidden_size=hidden_size, feature_axis=feature_axis,
        epsilon=rms_norm_epsilon,
    )
    logits = ws32_logits_mapped(normalized, lm_head_local, vocab_size=vocab_size,
                               feature_axis=feature_axis)
    sampled = ws32_nucleus_sample_mapped(logits, uniform, vocab_size=vocab_size,
                                        config=config, expert_axis=expert_axis)
    return Ws32SplitGreedySampleResult(sampled.token_id, sampled.contract_valid, final_residual)
