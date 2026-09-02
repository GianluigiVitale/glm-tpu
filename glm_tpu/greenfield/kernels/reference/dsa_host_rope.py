"""DSA indexer key/query rotation from host-evaluated FP32 cos/sin rows.

Companion to :mod:`dsa`: identical LayerNorm and pairing semantics, but the
rotation consumes one FP32 ``cos|sin`` row per position (see
:mod:`rotary_table`) instead of evaluating ``cos``/``sin`` on the accelerator.
:mod:`dsa` itself is untouched so historical source authorities keep their
byte pins.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from .dsa import DsaNumericalContract, _affine_layer_norm, _require_shape
from .rotary import apply_rotary


def rotary_cos_sin_from_rows(
    rope_table_rows: jax.Array, positions: jax.Array, *, rotary_dim: int
) -> tuple[jax.Array, jax.Array]:
    """Split FP32 ``cos|sin`` rows shaped ``positions.shape + (rotary_dim,)``."""

    if not isinstance(rotary_dim, int) or isinstance(rotary_dim, bool):
        raise TypeError("rotary_dim must be an integer")
    if rotary_dim <= 0 or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even integer")
    if rope_table_rows.shape != (*positions.shape, rotary_dim):
        raise ValueError("DSA rotary rows must match positions and rotary_dim")
    if rope_table_rows.dtype != jnp.float32:
        raise ValueError("DSA rotary rows must be FP32")
    half = rotary_dim // 2
    return rope_table_rows[..., :half], rope_table_rows[..., half:]


def dsa_index_keys_from_projection_host_rope(
    projected: jax.Array,
    key_norm_weight: jax.Array,
    key_norm_bias: jax.Array,
    positions: jax.Array,
    rope_table_rows: jax.Array,
    *,
    contract: DsaNumericalContract | None = None,
    key_norm_mode: Literal["divide_sqrt", "multiply_rsqrt"] = "multiply_rsqrt",
) -> jax.Array:
    """Normalize and rotate an FP32 key projection with host cos/sin rows."""

    contract = DsaNumericalContract() if contract is None else contract
    tokens = projected.shape[0] if projected.ndim == 2 else -1
    _require_shape("projected", projected, (tokens, contract.head_dim))
    _require_shape("key_norm_weight", key_norm_weight, (contract.head_dim,))
    _require_shape("key_norm_bias", key_norm_bias, (contract.head_dim,))
    _require_shape("positions", positions, (tokens,))
    if projected.dtype != jnp.float32:
        raise ValueError("DSA key projection must remain FP32")
    if not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("DSA positions must have an integer dtype")
    keys = _affine_layer_norm(
        projected,
        key_norm_weight,
        key_norm_bias,
        epsilon=contract.key_layer_norm_epsilon,
        mode=key_norm_mode,
    )
    cos, sin = rotary_cos_sin_from_rows(
        rope_table_rows, positions, rotary_dim=contract.rotary_dim
    )
    rotated = apply_rotary(
        keys[..., : contract.rotary_dim],
        cos,
        sin,
        interleaved=contract.interleaved_rotary,
    )
    return jnp.concatenate((rotated, keys[..., contract.rotary_dim :]), axis=-1)


def rotate_dsa_query_host_rope(
    query: jax.Array,
    positions: jax.Array,
    rope_table_rows: jax.Array,
    *,
    contract: DsaNumericalContract | None = None,
) -> jax.Array:
    """Rotate FP32 indexer queries ``[rows, heads, head_dim]`` with host rows."""

    contract = DsaNumericalContract() if contract is None else contract
    rows = query.shape[0] if query.ndim == 3 else -1
    _require_shape("query", query, (rows, contract.num_heads, contract.head_dim))
    _require_shape("positions", positions, (rows,))
    if query.dtype != jnp.float32:
        raise ValueError("DSA query must remain FP32")
    cos, sin = rotary_cos_sin_from_rows(
        rope_table_rows, positions, rotary_dim=contract.rotary_dim
    )
    rotated = apply_rotary(
        query[..., : contract.rotary_dim],
        cos[:, None, :],
        sin[:, None, :],
        interleaved=contract.interleaved_rotary,
    )
    return jnp.concatenate((rotated, query[..., contract.rotary_dim :]), axis=-1)
