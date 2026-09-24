"""The decoder's cache configuration: model geometry, context capacity and the KV-cache, attention,
DSA-indexer and MoE contracts derived from them.
"""

from __future__ import annotations

from dataclasses import dataclass

from glm_tpu.config.model import ModelGeometry
from glm_tpu.exceptions import PlanValidationError
from glm_tpu.layers.contracts import (
    DsaNumericalContract,
    GlmMoeNumericalContract,
    MlaNumericalContract,
    StageLocalKvLayout,
)


@dataclass(frozen=True, slots=True)
class CacheConfig:
    geometry: ModelGeometry
    context_capacity: int
    logical_page_size: int = 512
    packed_cache_width: int = 640
    sparse_segment_block: int = 512
    rms_norm_epsilon: float = 1e-5
    exact_dsa: bool = False
    strategy_nd_dense: bool = False
    host_main_rope_table: bool = False

    def __post_init__(self) -> None:
        geometry = self.geometry
        if geometry.num_layers <= 0:
            raise PlanValidationError("WS32 decoder requires transformer layers")
        if (
            not isinstance(self.context_capacity, int)
            or isinstance(self.context_capacity, bool)
            or not (0 < self.context_capacity <= geometry.max_position_embeddings)
        ):
            raise PlanValidationError("WS32 decoder context capacity is invalid")
        if self.logical_page_size <= 0 or self.logical_page_size % 8:
            raise PlanValidationError("WS32 decoder page must divide over expert-8")
        if self.packed_cache_width != (geometry.kv_lora_rank + geometry.qk_rope_head_dim + 64):
            raise PlanValidationError("WS32 packed cache width contract drifted")
        if self.sparse_segment_block <= 0 or (geometry.dsa_top_k % self.sparse_segment_block):
            raise PlanValidationError("WS32 sparse segment does not divide DSA top-k")
        if self.sparse_segment_block % 128:
            raise PlanValidationError("WS32 compiled sparse segment must divide into 128")
        if (
            not isinstance(self.rms_norm_epsilon, (int, float))
            or isinstance(self.rms_norm_epsilon, bool)
            or self.rms_norm_epsilon <= 0
        ):
            raise PlanValidationError("WS32 RMS epsilon must be positive")
        if not isinstance(self.exact_dsa, bool):
            raise PlanValidationError("WS32 exact DSA flag must be boolean")
        if not isinstance(self.strategy_nd_dense, bool):
            raise PlanValidationError("WS32 StrategyND dense flag must be boolean")
        if not isinstance(self.host_main_rope_table, bool):
            raise PlanValidationError("WS32 host main-rotary table flag must be boolean")
        if self.strategy_nd_dense and (
            geometry.hidden_size != 6144
            or geometry.dense_intermediate_size != 12288
            or geometry.first_dense_layers != 3
            or geometry.fp8_block_shape != (128, 128)
        ):
            raise PlanValidationError("WS32 StrategyND dense path requires exact GLM-5.2 geometry")
        if not self.full_index_slots or self.full_index_slots[0] != 0:
            raise PlanValidationError("WS32 layer zero must seed IndexShare state")
        producer: int | None = None
        for layer_id, indexer_kind in enumerate(geometry.indexer_types):
            if indexer_kind == "full":
                producer = layer_id
            elif producer is None or layer_id - producer >= geometry.index_share_group_size:
                raise PlanValidationError("WS32 IndexShare schedule drifted")

    @property
    def main_rope_table_shape(self) -> tuple[int, int]:
        """Replicated host BF16 ``cos|sin`` table, one row per position (§23.8)."""
        return (self.context_capacity, self.geometry.qk_rope_head_dim)

    @property
    def page_count(self) -> int:
        return (self.context_capacity + self.logical_page_size - 1) // self.logical_page_size

    @property
    def local_rows_per_page(self) -> int:
        return self.logical_page_size // 8

    @property
    def full_index_slots(self) -> tuple[int, ...]:
        return tuple(layer_id for layer_id, kind in enumerate(self.geometry.indexer_types) if kind == "full")

    @property
    def full_index_slot_by_layer(self) -> tuple[int | None, ...]:
        slots: list[int | None] = []
        next_slot = 0
        for kind in self.geometry.indexer_types:
            if kind == "full":
                slots.append(next_slot)
                next_slot += 1
            else:
                slots.append(None)
        return tuple(slots)

    @property
    def kv_cache_shape(self) -> tuple[int, int, int, int]:
        return (
            self.geometry.num_layers,
            self.page_count,
            self.logical_page_size,
            self.packed_cache_width,
        )

    @property
    def index_cache_shape(self) -> tuple[int, int, int, int]:
        return (
            len(self.full_index_slots),
            self.page_count,
            self.logical_page_size,
            self.geometry.dsa_indexer_head_dim,
        )

    @property
    def dsa_contract(self) -> DsaNumericalContract:
        return DsaNumericalContract(
            hidden_size=self.geometry.hidden_size,
            q_lora_rank=self.geometry.q_lora_rank,
            num_heads=self.geometry.dsa_indexer_heads,
            head_dim=self.geometry.dsa_indexer_head_dim,
            rotary_dim=self.geometry.qk_rope_head_dim,
            top_k=self.geometry.dsa_top_k,
        )

    @property
    def attention_contract(self) -> MlaNumericalContract:
        return MlaNumericalContract(
            num_heads=self.geometry.attention_heads,
            kv_lora_rank=self.geometry.kv_lora_rank,
            qk_nope_head_dim=self.geometry.qk_nope_head_dim,
            qk_rope_head_dim=self.geometry.qk_rope_head_dim,
            qk_head_dim=(self.geometry.qk_nope_head_dim + self.geometry.qk_rope_head_dim),
            v_head_dim=self.geometry.v_head_dim,
            packed_cache_width=self.packed_cache_width,
            top_k=self.geometry.dsa_top_k,
        )

    @property
    def moe_contract(self) -> GlmMoeNumericalContract:
        return GlmMoeNumericalContract(
            hidden_size=self.geometry.hidden_size,
            intermediate_size=self.geometry.moe_intermediate_size,
            num_experts=self.geometry.num_routed_experts,
            top_k=self.geometry.routed_top_k,
            stage_size=8,
            fp8_block_shape=self.geometry.fp8_block_shape,
        )

    @property
    def cache_layout(self) -> StageLocalKvLayout:
        return StageLocalKvLayout(
            logical_page_size=self.logical_page_size,
            local_parallel_size=8,
            packed_cache_width=self.packed_cache_width,
        )
