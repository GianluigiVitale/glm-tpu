"""Commit bounded proposals as contiguous rows, with no all-layer slice window.

The caller retains the original all-owner healthy transaction and state donation.
Flatten only layer/physical-row addresses; never flatten feature or model owners.
This does not by itself prove TPU layout, aliasing or runtime memory savings.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def apply_prefill_flat_rows(cache_stack: Any, targets: Any, rows: Any) -> Any:
    """Scatter complete width rows; invalid targets drop past the entire stack."""
    if (cache_stack.ndim != 4 or min(cache_stack.shape) <= 0
            or cache_stack.dtype != jnp.bfloat16):
        raise ValueError("flat commit requires a BF16 stacked cache")
    layers, pages, local_rows, width = cache_stack.shape
    if (targets.ndim != 1 or not 1 <= targets.shape[0] <= 128
            or targets.dtype != jnp.int32 or rows.dtype != cache_stack.dtype
            or rows.shape != (layers, targets.shape[0], width)):
        raise ValueError("flat commit rows/targets disagree with cache stack")
    per_layer = pages * local_rows
    total_rows = layers * per_layer
    if total_rows >= 2147483647:
        raise ValueError("flat cache row addresses must fit positive int32")
    valid = (targets >= 0) & (targets < per_layer)
    # Clip BEFORE addition, so even malformed INT_MAX targets cannot overflow.
    safe = jnp.clip(targets, 0, per_layer - 1)
    address = jnp.arange(layers, dtype=jnp.int32)[:, None] * per_layer + safe[None, :]
    address = jnp.where(valid[None, :], address, total_rows).reshape(-1)
    flat = cache_stack.reshape(total_rows, width)
    return flat.at[address].set(rows.reshape(-1, width), mode="drop").reshape(cache_stack.shape)
