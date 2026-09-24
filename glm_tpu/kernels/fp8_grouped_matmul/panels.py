"""Device-resident, lossless expert-relative M32 panels for opt-in prefill.

Build once from validated global sorted-route counts; reuse for gate/up/down.
Only activations are packed. No checkpoint or full-expert weight copy is made.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp


class ExpertPanels(NamedTuple):
    """Internal plan from ``build_expert_panels``; callers must gate on valid."""

    expert_ids: Any
    row_starts: Any
    live_counts: Any
    active_panels: Any
    restore_indices: Any
    owned_rows: Any
    valid: Any


def build_expert_panels(group_sizes: Any, group_offset: Any, *, rows: int, local_groups: int) -> ExpertPanels:
    """Pack owned experts separately, with at most ceil(rows/32)+G-1 panels.

    Counts include all sorted rows, including unowned experts. Invalid dynamic
    metadata produces no active panels and false health, never truncated work.
    The static product bound makes count prefix sums safe in int32.
    """
    if (
        type(rows) is not int
        or type(local_groups) is not int
        or min(rows, local_groups) <= 0
        or group_sizes.ndim != 1
        or group_sizes.size < local_groups
        or group_sizes.dtype != jnp.int32
        or (rows + 31) * group_sizes.size > 2147483647
    ):
        raise ValueError("expert panels require bounded rows and int32 group counts")
    if group_offset.shape != () or group_offset.dtype != jnp.int32:
        raise ValueError("expert panels require an int32 scalar offset")
    valid = (
        jnp.all((group_sizes >= 0) & (group_sizes <= rows))
        & (jnp.sum(group_sizes) == rows)
        & (group_offset >= 0)
        & (group_offset <= group_sizes.size - local_groups)
    )
    counts = jnp.where(valid, group_sizes, 0)
    offset = jnp.clip(group_offset, 0, group_sizes.size - local_groups)
    ends = jnp.cumsum(counts)
    starts = ends - counts
    local_ids = offset + jnp.arange(local_groups, dtype=jnp.int32)
    local_counts = counts[local_ids]
    panel_counts = (local_counts + 31) // 32
    panel_ends = jnp.cumsum(panel_counts)
    panel_starts = panel_ends - panel_counts
    capacity = (rows + 31) // 32 + local_groups - 1
    panel = jnp.arange(capacity, dtype=jnp.int32)
    # right skips all empty groups, including repeated zero prefix sums.
    expert = jnp.searchsorted(panel_ends, panel, side="right")
    expert = jnp.clip(expert, 0, local_groups - 1)
    within = (panel - panel_starts[expert]) * 32
    active = panel < panel_ends[-1]
    row_starts = jnp.where(active, starts[local_ids[expert]] + within, 0)
    live = jnp.where(active, jnp.clip(local_counts[expert] - within, 0, 32), 0)
    row = jnp.arange(rows, dtype=jnp.int32)
    row_expert = jnp.clip(jnp.searchsorted(ends, row, side="right"), 0, group_sizes.size - 1)
    owned = valid & (row_expert >= offset) & (row_expert < offset + local_groups)
    local_row_expert = jnp.clip(row_expert - offset, 0, local_groups - 1)
    relative_row = row - starts[row_expert]
    restore = panel_starts[local_row_expert] * 32 + relative_row
    return ExpertPanels(
        jnp.where(active, expert, 0),
        row_starts,
        live,
        panel_ends[-1],
        jnp.where(owned, restore, 0),
        owned,
        valid,
    )


def pack_expert_panel_rows(values: Any, panels: ExpertPanels) -> Any:
    """Gather aligned [panel,32,feature] rows; every padding row is zero."""
    if values.ndim != 2 or values.shape[0] != panels.restore_indices.size:
        raise ValueError("panel packing row geometry disagrees")
    lane = jnp.arange(32, dtype=jnp.int32)
    source = panels.row_starts[:, None] + lane[None, :]
    mask = lane[None, :] < panels.live_counts[:, None]
    # A masked result does not make an out-of-range gather safe.
    gathered = values[jnp.clip(source, 0, values.shape[0] - 1)]
    return jnp.where(mask[..., None], gathered, jnp.zeros((), values.dtype))


def unpack_expert_panel_rows(values: Any, panels: ExpertPanels) -> Any:
    """Restore sorted rows with no scatter overlap and zeros for unowned rows."""
    if values.ndim != 3 or values.shape[:2] != (panels.expert_ids.size, 32):
        raise ValueError("panel unpacking geometry disagrees")
    flat = values.reshape(-1, values.shape[-1])
    restored = flat[jnp.clip(panels.restore_indices, 0, flat.shape[0] - 1)]
    return jnp.where(panels.owned_rows[:, None], restored, jnp.zeros((), values.dtype))
