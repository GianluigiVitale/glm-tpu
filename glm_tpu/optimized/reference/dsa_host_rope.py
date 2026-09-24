"""DSA indexer key/query rotation from host-evaluated FP32 cos/sin rows.

Companion to :mod:`dsa`: identical LayerNorm and pairing semantics, but the
rotation consumes one FP32 ``cos|sin`` row per position (see
:mod:`rotary_table`) instead of evaluating ``cos``/``sin`` on the accelerator.
:mod:`dsa` itself is untouched so historical source authorities keep their
byte pins.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp


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
