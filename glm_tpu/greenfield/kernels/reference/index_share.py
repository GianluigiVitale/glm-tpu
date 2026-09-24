"""Static IndexShare schedule and compact selected-position carriage.

The device payload is only score-ordered ``int32[rows, top_k]`` positions.
Valid counts are derived from the exact ``-1`` suffix.  Producer identity is
compile-time schedule metadata and decode event identity is the position that
already travels with the residual, so neither duplicates bytes in the stage
transfer buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ....optimized.geometry import ModelGeometry
from ....optimized.reference.dsa import SelectedPositions


@dataclass(frozen=True, slots=True)
class IndexShareSchedule:
    """Exact full/shared source relation for every transformer layer."""

    modes: tuple[str, ...]
    group_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "modes", tuple(self.modes))
        if (
            not isinstance(self.group_size, int)
            or isinstance(self.group_size, bool)
            or self.group_size <= 0
        ):
            raise ValueError("IndexShare group_size must be a positive integer")
        if not self.modes or set(self.modes) - {"full", "shared"}:
            raise ValueError("IndexShare modes must be a non-empty full/shared tuple")
        source = None
        for layer, mode in enumerate(self.modes):
            if mode == "full":
                source = layer
            elif source is None:
                raise ValueError(
                    "a shared IndexShare layer has no preceding full producer"
                )
            elif layer - source >= self.group_size:
                raise ValueError(
                    "an IndexShare shared run exceeds the declared group size"
                )

    @classmethod
    def from_geometry(cls, geometry: ModelGeometry) -> "IndexShareSchedule":
        return cls(geometry.indexer_types, geometry.index_share_group_size)

    @property
    def source_layers(self) -> tuple[int, ...]:
        source = -1
        result = []
        for layer, mode in enumerate(self.modes):
            if mode == "full":
                source = layer
            result.append(source)
        return tuple(result)

    def source_for(self, layer_id: int) -> int:
        if (
            not isinstance(layer_id, int)
            or isinstance(layer_id, bool)
            or not 0 <= layer_id < len(self.modes)
        ):
            raise ValueError("layer_id is outside the IndexShare schedule")
        return self.source_layers[layer_id]

    def crossings(self, stage_starts: tuple[int, ...]) -> tuple[int, ...]:
        """Return stage starts that must receive compact IndexShare state."""

        starts = tuple(stage_starts)
        if (
            not starts
            or starts[0] != 0
            or tuple(sorted(set(starts))) != starts
            or starts[-1] >= len(self.modes)
        ):
            raise ValueError("stage starts must be sorted, unique, begin at zero")
        return tuple(layer for layer in starts[1:] if self.modes[layer] == "shared")


@dataclass(frozen=True, slots=True)
class IndexShareState:
    """Score-ordered selected positions plus static producer identity."""

    source_layer: int
    positions: jax.Array

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_layer, int)
            or isinstance(self.source_layer, bool)
            or self.source_layer < 0
        ):
            raise ValueError("IndexShare source_layer must be non-negative")
        if self.positions.ndim != 2 or self.positions.shape[1] == 0:
            raise ValueError("IndexShare positions must be [rows, positive_width]")
        if self.positions.dtype != jnp.int32:
            raise ValueError("IndexShare positions must have dtype int32")

    @property
    def selection(self) -> SelectedPositions:
        counts = jnp.sum(self.positions >= 0, axis=1, dtype=jnp.int32)
        return SelectedPositions(self.positions, counts)


def produce_index_share_state(
    selected: SelectedPositions, *, source_layer: int
) -> IndexShareState:
    """Stash a full indexer's exact score-ordered selection."""

    if selected.positions.ndim != 2 or selected.positions.shape[1] == 0:
        raise ValueError("selected positions must be [rows, positive_width]")
    if (
        selected.positions.dtype != jnp.int32
        or selected.valid_counts.dtype != jnp.int32
    ):
        raise ValueError("IndexShare selection positions/counts must be int32")
    if selected.valid_counts.shape != (selected.positions.shape[0],):
        raise ValueError("IndexShare valid_counts do not match selected rows")
    return IndexShareState(source_layer, selected.positions)


def pack_index_share_transfer(state: IndexShareState) -> jax.Array:
    """Return the only cross-stage DSA payload; no KV rows are materialized."""

    if state.positions.shape != (1, 2048):
        raise ValueError(
            "decode_batch1 IndexShare transfer must be exactly int32[1,2048]"
        )
    return state.positions


def restore_index_share_transfer(
    packed_positions: jax.Array, *, source_layer: int
) -> IndexShareState:
    """Restore state using the compile-time expected producer identity."""

    if packed_positions.shape != (1, 2048):
        raise ValueError(
            "decode_batch1 IndexShare transfer must be exactly int32[1,2048]"
        )
    return IndexShareState(source_layer, packed_positions)


def resolve_index_share_layer(
    schedule: IndexShareSchedule,
    layer_id: int,
    *,
    fresh_selection: SelectedPositions | None,
    shared_state: IndexShareState | None,
) -> tuple[SelectedPositions, IndexShareState]:
    """Produce on a full layer or reuse the exact state on a shared layer."""

    expected_source = schedule.source_for(layer_id)
    if schedule.modes[layer_id] == "full":
        if fresh_selection is None:
            raise ValueError("a full IndexShare layer requires a fresh selection")
        return fresh_selection, produce_index_share_state(
            fresh_selection, source_layer=layer_id
        )
    if fresh_selection is not None:
        raise ValueError("a shared IndexShare layer must not compute a fresh selection")
    if shared_state is None:
        raise ValueError("a shared IndexShare layer requires prior selected positions")
    if shared_state.source_layer != expected_source:
        raise ValueError(
            f"shared layer {layer_id} expected producer {expected_source}, "
            f"got {shared_state.source_layer}"
        )
    return shared_state.selection, shared_state


def validate_index_share_state_host(
    state: IndexShareState,
    *,
    expected_source_layer: int | None = None,
) -> None:
    """Fail-closed host/artifact validation; never call on the decode path."""

    if (
        expected_source_layer is not None
        and state.source_layer != expected_source_layer
    ):
        raise ValueError(
            "IndexShare producer identity does not match the expected event"
        )
    positions = np.asarray(state.positions, dtype=np.int32)
    for row_id, row in enumerate(positions):
        invalid = np.flatnonzero(row == -1)
        live_end = int(invalid[0]) if invalid.size else row.size
        if np.any(row[:live_end] < 0) or np.any(row[live_end:] != -1):
            raise ValueError(f"IndexShare row {row_id} does not have an exact -1 tail")
        if np.unique(row[:live_end]).size != live_end:
            raise ValueError(
                f"IndexShare row {row_id} contains duplicate live positions"
            )
