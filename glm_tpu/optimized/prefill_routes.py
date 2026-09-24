"""Lossless, device-resident expert grouping for opt-in multirow prefill."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp


class PrefillRoutes(NamedTuple):
    """Permutation metadata; ``valid`` MUST join the caller's serving health."""

    sorted_flat_ids: Any
    inverse_permutation: Any
    group_sizes: Any
    valid: Any


def group_prefill_routes(
    route_indices: Any, *, num_experts: int = 256
) -> PrefillRoutes:
    """Group by expert, preserving original token/slot order inside each group.

    Invalid dynamic IDs or duplicate experts produce ``valid=False``; they are
    not silently admitted. No host callback or per-token dispatch is used.
    """
    if route_indices.ndim != 2 or min(route_indices.shape) <= 0:
        raise ValueError("routes must be nonempty [tokens,top_k]")
    if route_indices.dtype != jnp.int32:
        raise ValueError("routes must be int32")
    if (
        not isinstance(num_experts, int)
        or isinstance(num_experts, bool)
        or num_experts <= 0
    ):
        raise ValueError("num_experts must be a positive integer")
    if route_indices.shape[1] > num_experts:
        raise ValueError("top_k exceeds expert count")
    flat = route_indices.reshape(-1)
    valid_ids = (flat >= 0) & (flat < num_experts)
    ordered_rows = jnp.sort(route_indices, axis=1)
    unique = jnp.all(ordered_rows[:, 1:] != ordered_rows[:, :-1])
    valid = jnp.all(valid_ids) & unique
    # The placeholder only keeps metadata construction bounded on invalid
    # input. The explicit health bit remains false and forbids serving.
    safe_flat = jnp.where(valid_ids, flat, 0)
    order = jnp.argsort(safe_flat, stable=True)
    inverse = (
        jnp.zeros_like(order).at[order].set(jnp.arange(flat.size, dtype=jnp.int32))
    )
    counts = jnp.bincount(safe_flat, length=num_experts).astype(jnp.int32)
    return PrefillRoutes(order, inverse, counts, valid)


def gather_prefill_route_rows(hidden: Any, routes: PrefillRoutes, *, top_k: int) -> Any:
    """Gather only local features; no hidden-state all-gather or dead capacity."""
    if (
        hidden.ndim != 2
        or top_k <= 0
        or routes.sorted_flat_ids.shape != (hidden.shape[0] * top_k,)
    ):
        raise ValueError("hidden and route geometry disagree")
    return hidden[routes.sorted_flat_ids // top_k]


def restore_prefill_route_rows(
    sorted_values: Any, routes: PrefillRoutes, *, top_k: int
) -> Any:
    """Restore [tokens,slot,features] before the original BF16 slot reduction."""
    if (
        sorted_values.ndim != 2
        or top_k <= 0
        or sorted_values.shape[0] != routes.inverse_permutation.size
        or sorted_values.shape[0] % top_k
    ):
        raise ValueError("sorted output and route geometry disagree")
    return sorted_values[routes.inverse_permutation].reshape(
        -1, top_k, sorted_values.shape[-1]
    )
