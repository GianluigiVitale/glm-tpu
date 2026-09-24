"""Isolated all-chip WS32 batch-one decoder composition.

The decoder is one ``shard_map`` body over ``expert=8 x feature=4``.  Every
layer executes on its stationary final-owner weights; the residual remains
feature-sharded, KV rows remain context-sharded over expert, and only compact
DSA selections are carried between IndexShare layers.

Moved verbatim at S2f from ``glm_tpu/greenfield/runtime/ws32_decoder.py`` (its production definitions; the
research remainder is archived at ``archive/research-20260922``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from .errors import PlanValidationError
from .geometry import ModelGeometry
from .reference.attention import MlaNumericalContract, StageLocalKvLayout
from .reference.dsa import DsaNumericalContract
from .reference.moe import GlmMoeNumericalContract
from .ws32_layer import (
    Ws32AttentionWeights,
    Ws32DenseWeights,
    Ws32DsaWeights,
    Ws32MoeWeights,
    Ws32QkvAWeights,
    Ws32StrategyNdDenseWeights,
)


class Ws32LayerWeights(NamedTuple):
    qkv_a: Ws32QkvAWeights
    attention: Ws32AttentionWeights
    dsa: Ws32DsaWeights | None
    post_attention_norm_weight_local: Any
    dense: Ws32DenseWeights | Ws32StrategyNdDenseWeights | None
    moe: Ws32MoeWeights | None


class Ws32DecoderWeights(NamedTuple):
    embedding_local: Any
    layers: tuple[Ws32LayerWeights, ...]
    final_norm_weight_local: Any
    lm_head_local: Any


class Ws32DecoderState(NamedTuple):
    kv_cache_local: Any
    index_cache_local: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    position: Any
    block_tables: Any
    context_lengths: Any
    contract_valid: Any


class Ws32DecodeStepResult(NamedTuple):
    state: Ws32DecoderState
    next_token: Any
    final_residual_local: Any


@dataclass(frozen=True, slots=True)
class Ws32DecoderConfig:
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
        if not isinstance(self.context_capacity, int) or isinstance(
            self.context_capacity, bool
        ) or not (0 < self.context_capacity <= geometry.max_position_embeddings):
            raise PlanValidationError("WS32 decoder context capacity is invalid")
        if self.logical_page_size <= 0 or self.logical_page_size % 8:
            raise PlanValidationError("WS32 decoder page must divide over expert-8")
        if self.packed_cache_width != (
            geometry.kv_lora_rank + geometry.qk_rope_head_dim + 64
        ):
            raise PlanValidationError("WS32 packed cache width contract drifted")
        if self.sparse_segment_block <= 0 or (
            geometry.dsa_top_k % self.sparse_segment_block
        ):
            raise PlanValidationError("WS32 sparse segment does not divide DSA top-k")
        if self.sparse_segment_block % 128:
            raise PlanValidationError("WS32 compiled sparse segment must divide into 128")
        if not isinstance(self.rms_norm_epsilon, (int, float)) or isinstance(
            self.rms_norm_epsilon, bool
        ) or self.rms_norm_epsilon <= 0:
            raise PlanValidationError("WS32 RMS epsilon must be positive")
        if not isinstance(self.exact_dsa, bool):
            raise PlanValidationError("WS32 exact DSA flag must be boolean")
        if not isinstance(self.strategy_nd_dense, bool):
            raise PlanValidationError(
                "WS32 StrategyND dense flag must be boolean"
            )
        if not isinstance(self.host_main_rope_table, bool):
            raise PlanValidationError(
                "WS32 host main-rotary table flag must be boolean"
            )
        if self.strategy_nd_dense and (
            geometry.hidden_size != 6144
            or geometry.dense_intermediate_size != 12288
            or geometry.first_dense_layers != 3
            or geometry.fp8_block_shape != (128, 128)
        ):
            raise PlanValidationError(
                "WS32 StrategyND dense path requires exact GLM-5.2 geometry"
            )
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
        return (
            self.context_capacity + self.logical_page_size - 1
        ) // self.logical_page_size

    @property
    def local_rows_per_page(self) -> int:
        return self.logical_page_size // 8

    @property
    def full_index_slots(self) -> tuple[int, ...]:
        return tuple(
            layer_id
            for layer_id, kind in enumerate(self.geometry.indexer_types)
            if kind == "full"
        )

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
            qk_head_dim=(
                self.geometry.qk_nope_head_dim
                + self.geometry.qk_rope_head_dim
            ),
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


def _qkv_specs() -> Ws32QkvAWeights:
    return Ws32QkvAWeights(
        P("feature"),
        P(None, "feature"),
        P(None, "feature"),
        P(),
        P(None, "feature"),
        P(None, "feature"),
        P(),
    )


def _attention_specs() -> Ws32AttentionWeights:
    return Ws32AttentionWeights(
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("feature", "expert"),
        P("feature", "expert"),
    )


def _dsa_specs() -> Ws32DsaWeights:
    return Ws32DsaWeights(
        P("expert", None),
        P("expert", None),
        P(None, "feature"),
        P(None, "feature"),
        P(),
        P(),
        P("expert", "feature"),
    )


def _dense_specs(
    *, strategy_nd: bool = False
) -> Ws32DenseWeights | Ws32StrategyNdDenseWeights:
    if strategy_nd:
        return Ws32StrategyNdDenseWeights(
            P("expert", None, None),
            P("expert", None, None),
            P("expert", None, "feature"),
            P("expert", None, "feature"),
        )
    return Ws32DenseWeights(
        P("expert", "feature"),
        P("expert", "feature"),
        P("expert", "feature"),
        P("expert", "feature"),
        P("feature", "expert"),
        P("feature", "expert"),
    )


def _moe_specs() -> Ws32MoeWeights:
    return Ws32MoeWeights(
        P("expert", "feature"),
        P("expert"),
        P("expert", None, "feature"),
        P("expert", None, "feature"),
        P("expert", None, "feature"),
        P("expert", None, "feature"),
        P("expert", "feature", None),
        P("expert", "feature", None),
        P(None, "feature"),
        P(None, "feature"),
        P(None, "feature"),
        P(None, "feature"),
        P("feature", None),
        P("feature", None),
    )


def ws32_decoder_weight_specs(config: Ws32DecoderConfig) -> Ws32DecoderWeights:
    """Return the exact pytree of global partition specifications."""

    layers = []
    for indexer_kind, mlp_kind in zip(
        config.geometry.indexer_types,
        config.geometry.mlp_layer_types,
        strict=True,
    ):
        layers.append(
            Ws32LayerWeights(
                _qkv_specs(),
                _attention_specs(),
                _dsa_specs() if indexer_kind == "full" else None,
                P("feature"),
                (
                    _dense_specs(strategy_nd=config.strategy_nd_dense)
                    if mlp_kind == "dense"
                    else None
                ),
                _moe_specs() if mlp_kind == "sparse" else None,
            )
        )
    return Ws32DecoderWeights(
        P("expert", "feature"),
        tuple(layers),
        P("feature"),
        P("expert", "feature"),
    )


def _fp8_names(prefix: str) -> tuple[str, str]:
    return f"{prefix}.weight_bits", f"{prefix}.scale_inv"


def ws32_decoder_weight_names(config: Ws32DecoderConfig) -> Ws32DecoderWeights:
    """Return the exact final-layout tensor name for every decoder input."""

    layers = []
    for layer_id, (indexer_kind, mlp_kind) in enumerate(
        zip(
            config.geometry.indexer_types,
            config.geometry.mlp_layer_types,
            strict=True,
        )
    ):
        prefix = f"model.layers.{layer_id}"
        attention_prefix = f"{prefix}.self_attn"
        q_a_bits, q_a_scale = _fp8_names(
            f"{attention_prefix}.q_a_proj"
        )
        kv_a_bits, kv_a_scale = _fp8_names(
            f"{attention_prefix}.kv_a_proj_with_mqa"
        )
        q_b_bits, q_b_scale = _fp8_names(
            f"{attention_prefix}.q_b_proj"
        )
        kv_b_bits, kv_b_scale = _fp8_names(
            f"{attention_prefix}.kv_b_proj"
        )
        o_bits, o_scale = _fp8_names(f"{attention_prefix}.o_proj")
        dsa = None
        if indexer_kind == "full":
            wq_b_bits, wq_b_scale = _fp8_names(
                f"{attention_prefix}.indexer.wq_b"
            )
            wk_bits, wk_scale = _fp8_names(
                f"{attention_prefix}.indexer.wk"
            )
            dsa = Ws32DsaWeights(
                wq_b_bits,
                wq_b_scale,
                wk_bits,
                wk_scale,
                f"{attention_prefix}.indexer.k_norm.weight",
                f"{attention_prefix}.indexer.k_norm.bias",
                f"{attention_prefix}.indexer.weights_proj.weight",
            )
        dense = None
        moe = None
        if mlp_kind == "dense":
            if config.strategy_nd_dense:
                strategy_prefix = f"{prefix}.mlp.strategy_nd"
                dense = Ws32StrategyNdDenseWeights(
                    f"{strategy_prefix}.merged_gate_up.weight_bits_in_out",
                    f"{strategy_prefix}.merged_gate_up.scale_inv_in_out",
                    f"{strategy_prefix}.down.weight_bits_in_out",
                    f"{strategy_prefix}.down.scale_inv_in_out",
                )
            else:
                gate_bits, gate_scale = _fp8_names(f"{prefix}.mlp.gate_proj")
                up_bits, up_scale = _fp8_names(f"{prefix}.mlp.up_proj")
                down_bits, down_scale = _fp8_names(f"{prefix}.mlp.down_proj")
                dense = Ws32DenseWeights(
                    gate_bits,
                    gate_scale,
                    up_bits,
                    up_scale,
                    down_bits,
                    down_scale,
                )
        else:
            expert_gate_bits, expert_gate_scale = _fp8_names(
                f"{prefix}.mlp.experts.gate_proj"
            )
            expert_up_bits, expert_up_scale = _fp8_names(
                f"{prefix}.mlp.experts.up_proj"
            )
            expert_down_bits, expert_down_scale = _fp8_names(
                f"{prefix}.mlp.experts.down_proj"
            )
            shared_gate_bits, shared_gate_scale = _fp8_names(
                f"{prefix}.mlp.shared_experts.gate_proj"
            )
            shared_up_bits, shared_up_scale = _fp8_names(
                f"{prefix}.mlp.shared_experts.up_proj"
            )
            shared_down_bits, shared_down_scale = _fp8_names(
                f"{prefix}.mlp.shared_experts.down_proj"
            )
            moe = Ws32MoeWeights(
                f"{prefix}.mlp.gate.weight",
                f"{prefix}.mlp.gate.e_score_correction_bias",
                expert_gate_bits,
                expert_gate_scale,
                expert_up_bits,
                expert_up_scale,
                expert_down_bits,
                expert_down_scale,
                shared_gate_bits,
                shared_gate_scale,
                shared_up_bits,
                shared_up_scale,
                shared_down_bits,
                shared_down_scale,
            )
        layers.append(
            Ws32LayerWeights(
                Ws32QkvAWeights(
                    f"{prefix}.input_layernorm.weight",
                    q_a_bits,
                    q_a_scale,
                    f"{attention_prefix}.q_a_layernorm.weight",
                    kv_a_bits,
                    kv_a_scale,
                    f"{attention_prefix}.kv_a_layernorm.weight",
                ),
                Ws32AttentionWeights(
                    q_b_bits,
                    q_b_scale,
                    kv_b_bits,
                    kv_b_scale,
                    o_bits,
                    o_scale,
                ),
                dsa,
                f"{prefix}.post_attention_layernorm.weight",
                dense,
                moe,
            )
        )
    return Ws32DecoderWeights(
        "model.embed_tokens.weight",
        tuple(layers),
        "model.norm.weight",
        "lm_head.weight",
    )


def _weight_name_leaves(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple):
        return tuple(
            name for item in value for name in _weight_name_leaves(item)
        )
    raise TypeError("WS32 weight-name tree contains a non-string leaf")


def _bind_weight_name_tree(value: object, arrays: Mapping[str, Any]) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        return arrays[value]
    if isinstance(value, tuple):
        children = tuple(_bind_weight_name_tree(item, arrays) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*children)
        return children
    raise TypeError("WS32 weight-name tree contains a non-string leaf")


def bind_ws32_decoder_weights(
    arrays: Mapping[str, Any],
    config: Ws32DecoderConfig,
) -> Ws32DecoderWeights:
    """Bind one exact manifest tensor map to the typed 78-layer decoder."""

    names = ws32_decoder_weight_names(config)
    leaves = _weight_name_leaves(names)
    expected = set(leaves)
    if len(expected) != len(leaves):
        raise ValueError("WS32 decoder weight names are not bijective")
    observed = set(arrays)
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        unexpected = sorted(observed - expected)[:5]
        raise ValueError(
            "WS32 decoder tensor set drifted: "
            f"missing={missing}, unexpected={unexpected}"
        )
    result = _bind_weight_name_tree(names, arrays)
    if not isinstance(result, Ws32DecoderWeights):
        raise AssertionError("WS32 decoder binding lost its typed root")
    return result


def ws32_decoder_state_specs() -> Ws32DecoderState:
    return Ws32DecoderState(
        P(None, None, "expert", None),
        P(None, None, "expert", None),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
        P(),
    )


def ws32_decode_result_specs() -> Ws32DecodeStepResult:
    return Ws32DecodeStepResult(
        ws32_decoder_state_specs(), P(), P(None, "feature")
    )


def _validate_local_state(
    state: Ws32DecoderState,
    config: Ws32DecoderConfig,
) -> None:
    geometry = config.geometry
    expected_kv = (
        geometry.num_layers,
        config.page_count,
        config.local_rows_per_page,
        config.packed_cache_width,
    )
    expected_index = (
        len(config.full_index_slots),
        config.page_count,
        config.local_rows_per_page,
        geometry.dsa_indexer_head_dim,
    )
    if state.kv_cache_local.shape != expected_kv or (
        state.kv_cache_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 local KV state geometry drifted")
    if state.index_cache_local.shape != expected_index or (
        state.index_cache_local.dtype != jnp.bfloat16
    ):
        raise ValueError("WS32 local index state geometry drifted")
    if state.selected_positions.shape != (1, geometry.dsa_top_k) or (
        state.selected_positions.dtype != jnp.int32
    ):
        raise ValueError("WS32 selected-position state geometry drifted")
    if state.selected_valid_counts.shape != (1,) or (
        state.selected_valid_counts.dtype != jnp.int32
    ):
        raise ValueError("WS32 selected-count state geometry drifted")
    if state.selected_scores.shape != (1, geometry.dsa_top_k) or (
        state.selected_scores.dtype != jnp.float32
    ):
        raise ValueError("WS32 selected-score state geometry drifted")
    if state.position.shape != (1,) or state.position.dtype != jnp.int32:
        raise ValueError("WS32 position state geometry drifted")
    if state.block_tables.shape != (1, config.page_count) or (
        state.block_tables.dtype != jnp.int32
    ):
        raise ValueError("WS32 block-table state geometry drifted")
    if state.context_lengths.shape != (1,) or (
        state.context_lengths.dtype != jnp.int32
    ):
        raise ValueError("WS32 context-length state geometry drifted")
    if state.contract_valid.shape != (1,) or (
        state.contract_valid.dtype != jnp.bool_
    ):
        raise ValueError("WS32 decoder health must be one boolean row")


WS32_MAIN_ROPE_THETA = 8_000_000.0
"""Accepted GLM main-attention rotary base (config `rope_parameters.rope_theta`).

It is the default of every WS32 main-attention rotary site, so the host table
(spec §23.8) must be built with the same value or the two forms diverge.
"""


def build_ws32_main_rope_table(config: Ws32DecoderConfig) -> Any:
    """Host BF16 ``cos|sin`` table for the main-attention rotary (spec §23.8).

    Built with the accepted GLM runtime's own construction
    (:func:`build_rotary_table_host`: positive FP32 powers, reciprocal, NumPy
    FP32 trigonometry, stored BF16), which DB531 proved reproduces the legacy
    64-wide rotary suffix bitwise when applied with FP32 products and one final
    BF16 round.
    """
    from .reference.rotary import build_rotary_table_host

    table = build_rotary_table_host(
        config.context_capacity,
        rotary_dim=config.geometry.qk_rope_head_dim,
        theta=WS32_MAIN_ROPE_THETA,
    )
    if table.shape != config.main_rope_table_shape:
        raise PlanValidationError("WS32 main rotary table geometry drifted")
    return table
