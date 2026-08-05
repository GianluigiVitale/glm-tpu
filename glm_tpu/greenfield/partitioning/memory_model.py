"""Explicit per-chip 256K runtime memory accounting for pipeline plans."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

from ..errors import PartitioningValidationError
from ..types import ModelGeometry


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _ceil_div(left: int, right: int) -> int:
    return (left + right - 1) // right


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    """Measured floors and explicit unknowns used by initial Gate-B planning.

    ``temporary_floor_bytes`` is the protected PP8 real-layer peak minus its
    post-timing resident HBM (5,639,681,536 - 4,860,038,656).  It is a
    measured one-layer floor, not a complete-decoder overlay proof.  The
    overlay remains zero and ``full_decoder_overlay_measured`` remains false
    until Gate D compiles and measures the complete executable.
    """

    target_context_length: int = 262_144
    hbm_limit_bytes: int = 33_014_413_312
    cache_element_bytes: int = 2
    cache_alignment: int = 128
    kv_rope_cache_width: int = 128
    selected_index_element_bytes: int = 4
    selected_index_buffers_per_stage: int = 2
    temporary_floor_bytes: int = 779_642_880
    reserved_overlay_bytes: int = 0
    full_decoder_overlay_measured: bool = False
    provenance: str = "protected-pp8-db417-one-layer-floor"

    def __post_init__(self) -> None:
        for field in (
            "target_context_length",
            "hbm_limit_bytes",
            "cache_element_bytes",
            "cache_alignment",
            "kv_rope_cache_width",
            "selected_index_element_bytes",
            "selected_index_buffers_per_stage",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PartitioningValidationError(f"{field} must be positive")
        for field in ("temporary_floor_bytes", "reserved_overlay_bytes"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PartitioningValidationError(
                    f"{field} must be non-negative"
                )
        if not self.provenance:
            raise PartitioningValidationError("memory policy provenance is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_alignment": self.cache_alignment,
            "cache_element_bytes": self.cache_element_bytes,
            "full_decoder_overlay_measured": self.full_decoder_overlay_measured,
            "hbm_limit_bytes": self.hbm_limit_bytes,
            "kv_rope_cache_width": self.kv_rope_cache_width,
            "provenance": self.provenance,
            "reserved_overlay_bytes": self.reserved_overlay_bytes,
            "selected_index_buffers_per_stage": self.selected_index_buffers_per_stage,
            "selected_index_element_bytes": self.selected_index_element_bytes,
            "target_context_length": self.target_context_length,
            "temporary_floor_bytes": self.temporary_floor_bytes,
        }

    @property
    def policy_sha256(self) -> str:
        return sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RuntimeMemory:
    """Per-chip runtime state for one contiguous pipeline stage."""

    kv_cache_bytes: int
    indexer_cache_bytes: int
    selected_index_bytes: int
    transport_buffer_bytes: int
    temporary_floor_bytes: int
    reserved_overlay_bytes: int

    @property
    def dsa_state_bytes(self) -> int:
        return (
            self.indexer_cache_bytes
            + self.selected_index_bytes
            + self.transport_buffer_bytes
        )

    @property
    def accounted_bytes(self) -> int:
        return (
            self.kv_cache_bytes
            + self.dsa_state_bytes
            + self.temporary_floor_bytes
            + self.reserved_overlay_bytes
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "accounted_bytes": self.accounted_bytes,
            "dsa_state_bytes": self.dsa_state_bytes,
            "indexer_cache_bytes": self.indexer_cache_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "reserved_overlay_bytes": self.reserved_overlay_bytes,
            "selected_index_bytes": self.selected_index_bytes,
            "temporary_floor_bytes": self.temporary_floor_bytes,
            "transport_buffer_bytes": self.transport_buffer_bytes,
        }


def layer_kv_cache_bytes(
    geometry: ModelGeometry,
    *,
    local_parallel_size: int,
    policy: MemoryPolicy,
) -> int:
    """Conservative per-chip MLA cache bytes for one attention layer."""

    if local_parallel_size <= 0:
        raise PartitioningValidationError("local_parallel_size must be positive")
    if policy.target_context_length > geometry.max_position_embeddings:
        raise PartitioningValidationError(
            "memory target exceeds model maximum position embeddings"
        )
    if policy.kv_rope_cache_width < geometry.qk_rope_head_dim:
        raise PartitioningValidationError(
            "KV rope cache width cannot truncate the model rope dimension"
        )
    latent_width = _align(geometry.kv_lora_rank, policy.cache_alignment)
    tokens_per_chip = _ceil_div(
        policy.target_context_length, local_parallel_size
    )
    return (
        tokens_per_chip
        * (latent_width + policy.kv_rope_cache_width)
        * policy.cache_element_bytes
    )


def layer_indexer_cache_bytes(
    geometry: ModelGeometry,
    layer_id: int,
    *,
    local_parallel_size: int,
    policy: MemoryPolicy,
) -> int:
    """Per-chip BF16 index-key cache; shared IndexShare layers own none."""

    if not 0 <= layer_id < geometry.num_layers:
        raise PartitioningValidationError("base layer_id is out of range")
    if geometry.indexer_types[layer_id] != "full":
        return 0
    tokens_per_chip = _ceil_div(
        policy.target_context_length, local_parallel_size
    )
    return (
        tokens_per_chip
        * geometry.dsa_indexer_head_dim
        * policy.cache_element_bytes
    )


def stage_runtime_memory(
    geometry: ModelGeometry,
    *,
    layer_start: int,
    layer_end_exclusive: int,
    local_parallel_size: int,
    policy: MemoryPolicy,
) -> RuntimeMemory:
    """Account all target-context state for a contiguous base-layer stage."""

    if not 0 <= layer_start < layer_end_exclusive <= geometry.num_layers:
        raise PartitioningValidationError("invalid stage layer range")
    layer_count = layer_end_exclusive - layer_start
    kv_bytes = layer_count * layer_kv_cache_bytes(
        geometry, local_parallel_size=local_parallel_size, policy=policy
    )
    indexer_bytes = sum(
        layer_indexer_cache_bytes(
            geometry,
            layer,
            local_parallel_size=local_parallel_size,
            policy=policy,
        )
        for layer in range(layer_start, layer_end_exclusive)
    )
    selected_bytes = (
        geometry.dsa_top_k
        * policy.selected_index_element_bytes
        * policy.selected_index_buffers_per_stage
    )
    residual_bytes = geometry.hidden_size * policy.cache_element_bytes
    transport_bytes = residual_bytes + geometry.dsa_top_k * 4
    return RuntimeMemory(
        kv_cache_bytes=kv_bytes,
        indexer_cache_bytes=indexer_bytes,
        selected_index_bytes=selected_bytes,
        transport_buffer_bytes=transport_bytes,
        temporary_floor_bytes=policy.temporary_floor_bytes,
        reserved_overlay_bytes=policy.reserved_overlay_bytes,
    )
