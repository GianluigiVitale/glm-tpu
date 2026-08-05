"""Checkpoint-oriented GLM linear, dense-MLP, and output primitives."""

from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp


def linear(
    hidden_states: jax.Array,
    weight_out_in: jax.Array,
    bias: jax.Array | None = None,
    *,
    output_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """Apply ``y = x @ weight.T + bias`` with explicit output precision.

    Checkpoint weights always retain PyTorch ``[out_features, in_features]``
    orientation. The reference decoder normally receives already-dequantized
    BF16 weights and therefore returns BF16; FP32 router/indexer callers can
    request FP32 explicitly. No implicit repartition or concatenation occurs.
    """

    if hidden_states.ndim < 1:
        raise ValueError("linear input must have at least one dimension")
    if weight_out_in.ndim != 2:
        raise ValueError("linear weight must have shape [out_features, in_features]")
    if hidden_states.shape[-1] != weight_out_in.shape[1]:
        raise ValueError(
            "linear input width does not match checkpoint weight: "
            f"input={hidden_states.shape[-1]} weight={weight_out_in.shape}"
        )
    if bias is not None and bias.shape != (weight_out_in.shape[0],):
        raise ValueError(
            f"linear bias must have shape {(weight_out_in.shape[0],)}, got {bias.shape}"
        )
    if not jnp.issubdtype(hidden_states.dtype, jnp.inexact):
        raise ValueError("linear input must have an inexact dtype")
    if not jnp.issubdtype(weight_out_in.dtype, jnp.inexact):
        raise ValueError("linear weight must have an inexact dtype")

    dtype = jnp.dtype(output_dtype or hidden_states.dtype)
    if not jnp.issubdtype(dtype, jnp.inexact):
        raise ValueError("linear output dtype must be inexact")
    result = lax.dot_general(
        hidden_states.astype(dtype),
        weight_out_in.astype(dtype),
        dimension_numbers=(
            ((hidden_states.ndim - 1,), (1,)),
            ((), ()),
        ),
        preferred_element_type=dtype,
    )
    if bias is not None:
        result = result + bias.astype(dtype)
    return result.astype(dtype)


def silu(value: jax.Array) -> jax.Array:
    """GLM's activation with an explicit same-dtype result boundary."""

    return (value * jax.nn.sigmoid(value)).astype(value.dtype)


def dense_swiglu(
    hidden_states: jax.Array,
    gate_weight_out_in: jax.Array,
    up_weight_out_in: jax.Array,
    down_weight_out_in: jax.Array,
) -> jax.Array:
    """Reference dense GLM MLP used by transformer layers 0--2."""

    if gate_weight_out_in.shape != up_weight_out_in.shape:
        raise ValueError("dense gate and up projection shapes must match")
    intermediate = gate_weight_out_in.shape[0]
    if down_weight_out_in.shape != (hidden_states.shape[-1], intermediate):
        raise ValueError(
            "dense down projection must map intermediate width back to hidden width"
        )
    gate = linear(hidden_states, gate_weight_out_in)
    up = linear(hidden_states, up_weight_out_in)
    return linear(silu(gate) * up, down_weight_out_in)


def residual_add(residual: jax.Array, update: jax.Array) -> jax.Array:
    """Add one transformer residual without dtype promotion or broadcasting."""

    if residual.shape != update.shape:
        raise ValueError("residual and update shapes must match exactly")
    if residual.dtype != update.dtype:
        raise ValueError("residual and update dtypes must match exactly")
    return (residual + update).astype(residual.dtype)


def embedding_lookup(token_ids: jax.Array, embedding_table: jax.Array) -> jax.Array:
    """Gather logical token embeddings with an explicit integer-id contract."""

    if embedding_table.ndim != 2:
        raise ValueError("embedding table must have shape [vocab, hidden]")
    if not jnp.issubdtype(token_ids.dtype, jnp.integer):
        raise ValueError("token ids must have an integer dtype")
    return jnp.take(embedding_table, token_ids, axis=0)


def vocabulary_logits(
    hidden_states: jax.Array,
    lm_head_weight_out_in: jax.Array,
) -> jax.Array:
    """Compute non-upcast GLM vocabulary logits in activation precision."""

    return linear(hidden_states, lm_head_weight_out_in)
