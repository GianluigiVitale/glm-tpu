"""Exact one-row GLM-5.2 DSA scorer and distributed selection reference.

This is the readable semantic path that optimized XLA/Pallas kernels must
match. It keeps query rows explicit (decode uses exactly one), pins indexer
dot precision to ``highest``, preserves the accepted interleaved RoPE layout,
and defines exact lowest-global-position tie order for local and distributed
top-k. No approximate selector is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from .linear import linear
from .rotary import apply_rotary, rotary_cos_sin


_NEGATIVE_INFINITY = jnp.float32(float("-inf"))


@dataclass(frozen=True, slots=True)
class DsaNumericalContract:
    """Compile-relevant semantic contract for the GLM lightning indexer."""

    hidden_size: int = 6144
    q_lora_rank: int = 2048
    num_heads: int = 32
    head_dim: int = 128
    rotary_dim: int = 64
    top_k: int = 2048
    theta: float = 8_000_000.0
    key_layer_norm_epsilon: float = 1e-6
    interleaved_rotary: bool = True
    score_dtype: str = "float32"
    tie_policy: str = "descending_score_then_lowest_global_position"
    padding_sentinel: int = -1

    def __post_init__(self) -> None:
        for field in (
            "hidden_size",
            "q_lora_rank",
            "num_heads",
            "head_dim",
            "rotary_dim",
            "top_k",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.rotary_dim > self.head_dim or self.rotary_dim % 2:
            raise ValueError("rotary_dim must be even and no larger than head_dim")
        if (
            isinstance(self.theta, bool)
            or isinstance(self.key_layer_norm_epsilon, bool)
            or self.theta <= 0
            or self.key_layer_norm_epsilon <= 0
        ):
            raise ValueError("theta and key LayerNorm epsilon must be positive")
        if self.interleaved_rotary is not True:
            raise ValueError("the exact GLM DSA contract requires interleaved rotary")
        if self.score_dtype != "float32":
            raise ValueError("the exact GLM DSA scorer requires FP32 scores")
        if self.tie_policy != "descending_score_then_lowest_global_position":
            raise ValueError("the exact GLM DSA tie policy drifted")
        if self.padding_sentinel != -1:
            raise ValueError("selected-position padding sentinel must be -1")


class SelectedPositions(NamedTuple):
    """Compact DSA/IndexShare state; decode shape is ``[1, top_k]``."""

    positions: jax.Array
    valid_counts: jax.Array


def _require_shape(name: str, value: jax.Array, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


def _affine_layer_norm(
    value: jax.Array,
    weight: jax.Array,
    bias: jax.Array,
    *,
    epsilon: float,
) -> jax.Array:
    """FP32 biased LayerNorm used only by the 128-wide indexer key."""

    _require_shape("key LayerNorm weight", weight, (value.shape[-1],))
    _require_shape("key LayerNorm bias", bias, (value.shape[-1],))
    value_f32 = value.astype(jnp.float32)
    mean = jnp.mean(value_f32, axis=-1, keepdims=True)
    variance = jnp.mean(lax.square(value_f32 - mean), axis=-1, keepdims=True)
    normalized = (value_f32 - mean) * lax.rsqrt(
        variance + jnp.float32(epsilon)
    )
    return (
        normalized * weight.astype(jnp.float32)
        + bias.astype(jnp.float32)
    )


def dsa_query_and_head_weights(
    hidden_states: jax.Array,
    q_residual: jax.Array,
    query_weight_out_in: jax.Array,
    head_weight_out_in: jax.Array,
    positions: jax.Array,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
) -> tuple[jax.Array, jax.Array]:
    """Project current rows to rotated indexer queries and signed head weights."""

    rows = hidden_states.shape[0] if hidden_states.ndim == 2 else -1
    _require_shape("hidden_states", hidden_states, (rows, contract.hidden_size))
    _require_shape("q_residual", q_residual, (rows, contract.q_lora_rank))
    _require_shape(
        "query_weight_out_in",
        query_weight_out_in,
        (contract.num_heads * contract.head_dim, contract.q_lora_rank),
    )
    _require_shape(
        "head_weight_out_in",
        head_weight_out_in,
        (contract.num_heads, contract.hidden_size),
    )
    _require_shape("positions", positions, (rows,))
    if not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("DSA positions must have an integer dtype")

    with jax.default_matmul_precision("highest"):
        query = linear(
            q_residual,
            query_weight_out_in,
            output_dtype=jnp.float32,
        ).reshape(rows, contract.num_heads, contract.head_dim)
        head_weights = linear(
            hidden_states,
            head_weight_out_in,
            output_dtype=jnp.float32,
        ) * jnp.float32(contract.num_heads**-0.5)
    cos, sin = rotary_cos_sin(
        positions,
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
    query = jnp.concatenate((rotated, query[..., contract.rotary_dim :]), axis=-1)
    return query.astype(jnp.float32), head_weights.astype(jnp.float32)


def dsa_index_keys(
    hidden_states: jax.Array,
    key_weight_out_in: jax.Array,
    key_norm_weight: jax.Array,
    key_norm_bias: jax.Array,
    positions: jax.Array,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
) -> jax.Array:
    """Project, normalize, and rotate the cache-resident DSA index keys."""

    tokens = hidden_states.shape[0] if hidden_states.ndim == 2 else -1
    _require_shape("hidden_states", hidden_states, (tokens, contract.hidden_size))
    _require_shape(
        "key_weight_out_in",
        key_weight_out_in,
        (contract.head_dim, contract.hidden_size),
    )
    _require_shape("key_norm_weight", key_norm_weight, (contract.head_dim,))
    _require_shape("key_norm_bias", key_norm_bias, (contract.head_dim,))
    _require_shape("positions", positions, (tokens,))
    if not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("DSA positions must have an integer dtype")

    with jax.default_matmul_precision("highest"):
        projected = linear(
            hidden_states,
            key_weight_out_in,
            output_dtype=jnp.float32,
        )
    keys = _affine_layer_norm(
        projected,
        key_norm_weight,
        key_norm_bias,
        epsilon=contract.key_layer_norm_epsilon,
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
) -> jax.Array:
    """Compute signed FP32 DSA scores ``[query_rows, context]``.

    ReLU is applied to every per-head query/key dot before the signed head
    weighting, exactly as in GLM-5.2. The key cache contains one shared
    128-wide key per historical token.
    """

    if query.ndim != 3 or index_keys.ndim != 2 or head_weights.ndim != 2:
        raise ValueError("DSA query/key/head-weight ranks must be 3/2/2")
    rows, heads, head_dim = query.shape
    _require_shape("index_keys", index_keys, (index_keys.shape[0], head_dim))
    _require_shape("head_weights", head_weights, (rows, heads))
    with jax.default_matmul_precision("highest"):
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


def exact_topk(
    scores: jax.Array,
    valid_lengths: jax.Array,
    *,
    top_k: int,
) -> SelectedPositions:
    """Select exact descending scores with lowest-position tie order.

    Context positions outside each row's valid length are masked. If context
    is shorter than ``top_k``, the fixed-width tail is exactly ``-1`` and
    ``valid_counts`` reports the live prefix. ``lax.approx_max_k`` is banned.
    """

    if scores.ndim != 2:
        raise ValueError("DSA scores must have shape [rows, context]")
    rows, context = scores.shape
    _require_shape("valid_lengths", valid_lengths, (rows,))
    if not jnp.issubdtype(valid_lengths.dtype, jnp.integer):
        raise ValueError("valid lengths must have an integer dtype")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    positions = jnp.arange(context, dtype=jnp.int32)
    masked = jnp.where(
        positions[None, :] < valid_lengths.astype(jnp.int32)[:, None],
        scores.astype(jnp.float32),
        _NEGATIVE_INFINITY,
    )
    padded_width = max(context, top_k)
    if padded_width != context:
        masked = jnp.pad(
            masked,
            ((0, 0), (0, padded_width - context)),
            constant_values=_NEGATIVE_INFINITY,
        )
    _, selected = lax.top_k(masked, top_k)
    selected = selected.astype(jnp.int32)
    valid_counts = jnp.clip(
        valid_lengths.astype(jnp.int32),
        jnp.int32(0),
        jnp.int32(min(context, top_k)),
    )
    slots = lax.broadcasted_iota(jnp.int32, (rows, top_k), 1)
    selected = jnp.where(slots < valid_counts[:, None], selected, jnp.int32(-1))
    return SelectedPositions(selected, valid_counts)


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
    _require_shape("global_positions", global_positions, (local_context,))
    _require_shape("valid_lengths", valid_lengths, (rows,))
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
    valid = (
        (ordered_positions[None, :] >= 0)
        & (ordered_positions[None, :] < valid_lengths[:, None])
    )
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
        positions = jnp.take(
            ordered_positions, safe_indices, axis=0
        )
    positions = jnp.where(values == _NEGATIVE_INFINITY, jnp.int32(-1), positions)
    return values.astype(jnp.float32), positions.astype(jnp.int32)


def merge_topk_candidates(
    candidate_scores: jax.Array,
    candidate_positions: jax.Array,
    valid_lengths: jax.Array,
    *,
    top_k: int,
    global_context_size: int,
) -> SelectedPositions:
    """Merge stage-local candidates independent of collective concatenation order.

    Inputs are ``[local_group, rows, candidates]``. A stable ascending-global-
    position pre-sort makes the final ``lax.top_k`` tie break identical to a
    flat global score row even if an all-gather returns groups in another order.
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
    position_order = jnp.argsort(positions, axis=1, stable=True)
    sorted_scores = jnp.take_along_axis(scores, position_order, axis=1)
    sorted_positions = jnp.take_along_axis(positions, position_order, axis=1)
    _, selected_slots = lax.top_k(sorted_scores, top_k)
    selected = jnp.take_along_axis(sorted_positions, selected_slots, axis=1)
    valid_counts = jnp.clip(
        valid_lengths.astype(jnp.int32),
        jnp.int32(0),
        jnp.int32(min(top_k, global_context_size)),
    )
    slots = lax.broadcasted_iota(jnp.int32, (rows, top_k), 1)
    selected = jnp.where(slots < valid_counts[:, None], selected, jnp.int32(-1))
    return SelectedPositions(selected.astype(jnp.int32), valid_counts)


def distributed_exact_topk_reference(
    shard_scores: jax.Array,
    shard_global_positions: jax.Array,
    valid_lengths: jax.Array,
    *,
    top_k: int,
    global_context_size: int,
) -> SelectedPositions:
    """Reference local-candidate/all-gather/merge semantics for one stage group."""

    if shard_scores.ndim != 3:
        raise ValueError("shard_scores must have shape [local_group, rows, local_context]")
    groups, _, local_context = shard_scores.shape
    _require_shape(
        "shard_global_positions", shard_global_positions, (groups, local_context)
    )
    candidate_scores, candidate_positions = jax.vmap(
        lambda scores, positions: local_topk_candidates(
            scores, positions, valid_lengths, top_k=top_k
        )
    )(shard_scores, shard_global_positions)
    return merge_topk_candidates(
        candidate_scores,
        candidate_positions,
        valid_lengths,
        top_k=top_k,
        global_context_size=global_context_size,
    )
