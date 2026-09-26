"""The decoder's cache configuration: model geometry, context capacity and the KV-cache, attention,
DSA-indexer and MoE contracts derived from them.

The contracts are defined here (``glm_tpu.layers.contracts`` re-exports them for the layers). Importing this module
imports no JAX: ``StageLocalKvLayout.owner``, which the layers call inside their programs, imports ``jax.numpy``
when it runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from glm_tpu.config.model import ModelGeometry
from glm_tpu.exceptions import PlanValidationError

if TYPE_CHECKING:
    import jax


@dataclass(frozen=True, slots=True)
class MlaNumericalContract:
    """Compile-relevant GLM-5.2 absorbed-MLA arithmetic contract."""

    num_heads: int = 64
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256
    v_head_dim: int = 256
    packed_cache_width: int = 640
    top_k: int = 2048
    score_dtype: str = "float32"
    probability_rounding: str = "unnormalized_cache_dtype_before_pv"
    position_order: str = "ascending_global_position"

    def __post_init__(self) -> None:
        for field in (
            "num_heads",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "qk_head_dim",
            "v_head_dim",
            "packed_cache_width",
            "top_k",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.qk_nope_head_dim + self.qk_rope_head_dim != self.qk_head_dim:
            raise ValueError("qk_head_dim must equal nope plus RoPE dimensions")
        if self.kv_lora_rank + self.qk_rope_head_dim > self.packed_cache_width:
            raise ValueError("packed cache width truncates latent or RoPE state")
        if self.score_dtype != "float32":
            raise ValueError("sparse MLA scores and softmax must be FP32")
        if self.probability_rounding != "unnormalized_cache_dtype_before_pv":
            raise ValueError("sparse MLA probability-rounding contract drifted")
        if self.position_order != "ascending_global_position":
            raise ValueError("sparse MLA accumulation order must be position canonical")

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim**-0.5


@dataclass(frozen=True, slots=True)
class StageLocalKvLayout:
    """Context striping for one attention layer's topology-local cache."""

    logical_page_size: int = 512
    local_parallel_size: int = 4
    packed_cache_width: int = 640

    def __post_init__(self) -> None:
        for field in (
            "logical_page_size",
            "local_parallel_size",
            "packed_cache_width",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.logical_page_size % self.local_parallel_size:
            raise ValueError("logical page size must divide over the local stage group")

    @property
    def local_rows_per_page(self) -> int:
        return self.logical_page_size // self.local_parallel_size

    def owner(self, positions: jax.Array) -> jax.Array:
        """Return the local chip owning each non-negative absolute position."""
        import jax.numpy as jnp  # this module imports no JAX; the layers call owner inside their programs

        return (positions.astype(jnp.int32) % jnp.int32(self.logical_page_size)) // jnp.int32(self.local_rows_per_page)


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


@dataclass(frozen=True, slots=True)
class GlmMoeNumericalContract:
    """Compile-relevant arithmetic contract for one sparse GLM layer."""

    hidden_size: int = 6144
    intermediate_size: int = 2048
    num_experts: int = 256
    top_k: int = 8
    stage_size: int = 4
    routed_scaling_factor: float = 2.5
    fp8_block_shape: tuple[int, int] = (128, 128)
    activation_dtype: str = "bfloat16"
    router_dtype: str = "float32"
    fp8_scale_dtype: str = "float32"
    scoring_function: str = "sigmoid"
    routing_method: str = "noaux_tc"
    routing_tie_policy: str = "jax_lax_top_k_lowest_expert_id"
    reduction_association: str = "top_k_axis_then_stage_psum; routed_and_shared_stacked_not_mixed"

    def __post_init__(self) -> None:
        integer_fields = (
            "hidden_size",
            "intermediate_size",
            "num_experts",
            "top_k",
            "stage_size",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.num_experts % self.stage_size:
            raise ValueError("num_experts must divide evenly over the stage")
        if self.intermediate_size % self.stage_size:
            raise ValueError("shared intermediate size must divide over the stage")
        if self.routed_scaling_factor <= 0:
            raise ValueError("routed_scaling_factor must be positive")
        if len(self.fp8_block_shape) != 2 or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in self.fp8_block_shape
        ):
            raise ValueError("fp8_block_shape must contain two positive integers")
        if self.scoring_function != "sigmoid":
            raise ValueError("GLM-5.2 requires sigmoid MoE scoring")
        if self.routing_method != "noaux_tc":
            raise ValueError("GLM-5.2 requires noaux_tc routing")

    @property
    def local_experts(self) -> int:
        return self.num_experts // self.stage_size

    @property
    def local_shared_intermediate(self) -> int:
        return self.intermediate_size // self.stage_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "activation_dtype": self.activation_dtype,
            "fp8_block_shape": list(self.fp8_block_shape),
            "fp8_scale_dtype": self.fp8_scale_dtype,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "reduction_association": self.reduction_association,
            "routed_scaling_factor": self.routed_scaling_factor,
            "router_dtype": self.router_dtype,
            "routing_method": self.routing_method,
            "routing_tie_policy": self.routing_tie_policy,
            "scoring_function": self.scoring_function,
            "stage_size": self.stage_size,
            "top_k": self.top_k,
        }


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
