"""Isolated all-chip WS32 batch-one decoder composition.

The decoder is one ``shard_map`` body over ``expert=8 x feature=4``.  Every
layer executes on its stationary final-owner weights; the residual remains
feature-sharded, KV rows remain context-sharded over expert, and only compact
DSA selections are carried between IndexShare layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ..errors import PlanValidationError
from ..kernels.pallas import SparseMlaConfig
from ..kernels.reference.attention import (
    MlaNumericalContract,
    StageLocalKvLayout,
)
from ..kernels.reference.dsa import DsaNumericalContract
from ..kernels.reference.moe import GlmMoeNumericalContract
from ..kernels.ws32_io import (
    Ws32SplitGreedySampleResult,
    ws32_embedding_mapped,
    ws32_split_final_sample_mapped,
)
from ..kernels.ws32_layer import (
    Ws32AttentionWeights,
    Ws32DenseWeights,
    Ws32DsaWeights,
    Ws32MoeWeights,
    Ws32QkvAWeights,
    ws32_transformer_layer_mapped,
)
from ..types import ModelGeometry


class Ws32LayerWeights(NamedTuple):
    qkv_a: Ws32QkvAWeights
    attention: Ws32AttentionWeights
    dsa: Ws32DsaWeights | None
    post_attention_norm_weight_local: Any
    dense: Ws32DenseWeights | None
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


class Ws32DsaObservation(NamedTuple):
    """All full-indexer decisions from one complete decoder step."""

    producer_layer_ids: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any


class Ws32ObservedDecodeStepResult(NamedTuple):
    result: Ws32DecodeStepResult
    dsa: Ws32DsaObservation


class Ws32PrefillResult(NamedTuple):
    state: Ws32DecoderState
    next_token: Any


class Ws32CacheWriteProbe(NamedTuple):
    """Compact rows written at the most recently completed position."""

    position: Any
    kv_rows: Any
    index_rows: Any
    contract_valid: Any


@dataclass(frozen=True, slots=True)
class Ws32DecoderConfig:
    geometry: ModelGeometry
    context_capacity: int
    logical_page_size: int = 512
    packed_cache_width: int = 640
    sparse_segment_block: int = 512
    rms_norm_epsilon: float = 1e-5

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
        if not self.full_index_slots or self.full_index_slots[0] != 0:
            raise PlanValidationError("WS32 layer zero must seed IndexShare state")
        producer: int | None = None
        for layer_id, indexer_kind in enumerate(geometry.indexer_types):
            if indexer_kind == "full":
                producer = layer_id
            elif producer is None or layer_id - producer >= geometry.index_share_group_size:
                raise PlanValidationError("WS32 IndexShare schedule drifted")

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


@dataclass(frozen=True, slots=True)
class Ws32DecoderProgram:
    """One exact all-chip decode executable and its proof-only observers."""

    config: Ws32DecoderConfig
    mesh: Any
    execute: Any
    observe: Any
    probe_cache_write: Any


@dataclass(frozen=True, slots=True)
class Ws32TeacherForcedPrefillProgram:
    """One device-resident prompt scan with no per-token host dispatch."""

    config: Ws32DecoderConfig
    mesh: Any
    prompt_length: int
    execute: Any


def make_ws32_initial_state(
    mesh: Any,
    config: Ws32DecoderConfig,
) -> Ws32DecoderState:
    """Allocate the zero cache directly on its final owners.

    The callback returns one local shard at a time.  In particular it never
    materializes either complete cache in host memory.
    """

    import ml_dtypes
    import numpy as np
    from jax.sharding import NamedSharding

    if tuple(mesh.axis_names) != ("expert", "feature") or tuple(
        np.asarray(mesh.devices, dtype=object).shape
    ) != (8, 4):
        raise PlanValidationError(
            "WS32 initial state requires one exact expert8 x feature4 mesh"
        )

    def make(
        shape: tuple[int, ...],
        spec: P,
        dtype: Any,
        fill: int | float | bool,
    ) -> Any:
        sharding = NamedSharding(mesh, spec)
        local_shape = sharding.shard_shape(shape)

        def callback(_: tuple[slice, ...]) -> np.ndarray:
            return np.full(local_shape, fill, dtype=dtype)

        return jax.make_array_from_callback(shape, sharding, callback)

    blocks = np.arange(config.page_count, dtype=np.int32)[None, :]
    block_sharding = NamedSharding(mesh, P())
    block_tables = jax.make_array_from_callback(
        blocks.shape,
        block_sharding,
        lambda _: blocks.copy(),
    )
    geometry = config.geometry
    return Ws32DecoderState(
        make(
            config.kv_cache_shape,
            P(None, None, "expert", None),
            ml_dtypes.bfloat16,
            0,
        ),
        make(
            config.index_cache_shape,
            P(None, None, "expert", None),
            ml_dtypes.bfloat16,
            0,
        ),
        make((1, geometry.dsa_top_k), P(), np.int32, -1),
        make((1,), P(), np.int32, 0),
        make((1, geometry.dsa_top_k), P(), np.float32, -np.inf),
        make((1,), P(), np.int32, 0),
        block_tables,
        make((1,), P(), np.int32, 1),
        make((1,), P(), np.bool_, True),
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


def _dense_specs() -> Ws32DenseWeights:
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
                _dense_specs() if mlp_kind == "dense" else None,
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


def ws32_dsa_observation_specs() -> Ws32DsaObservation:
    return Ws32DsaObservation(P(), P(), P(), P())


def ws32_observed_decode_result_specs() -> Ws32ObservedDecodeStepResult:
    return Ws32ObservedDecodeStepResult(
        ws32_decode_result_specs(), ws32_dsa_observation_specs()
    )


def ws32_prefill_result_specs() -> Ws32PrefillResult:
    return Ws32PrefillResult(ws32_decoder_state_specs(), P())


def ws32_cache_write_probe_specs() -> Ws32CacheWriteProbe:
    return Ws32CacheWriteProbe(P(), P(), P(), P())


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


def _ws32_decode_impl(
    token_ids: Any,
    state: Ws32DecoderState,
    weights: Ws32DecoderWeights,
    *,
    config: Ws32DecoderConfig,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    observe_dsa: bool,
) -> tuple[Ws32DecodeStepResult, Ws32DsaObservation | None]:
    """Execute one complete batch-one step and optionally retain DSA events."""

    _validate_local_state(state, config)
    if token_ids.shape != (1,) or token_ids.dtype != jnp.int32:
        raise ValueError("WS32 decoder input must be one int32 token")
    if len(weights.layers) != config.geometry.num_layers:
        raise ValueError("WS32 decoder weight layer count drifted")
    embedded = ws32_embedding_mapped(
        token_ids,
        weights.embedding_local,
        vocab_size=config.geometry.vocab_size,
    )
    hidden_update = embedded.residual_local
    carried_residual = jnp.zeros_like(hidden_update)
    kv_cache = state.kv_cache_local
    index_cache = state.index_cache_local
    selected_positions = state.selected_positions
    selected_valid_counts = state.selected_valid_counts
    selected_scores = state.selected_scores
    health = state.contract_valid & embedded.contract_valid
    sparse_config = SparseMlaConfig(
        segment_block=config.sparse_segment_block
    )
    observed_positions = []
    observed_valid_counts = []
    observed_scores = []

    for layer_id, layer_weights in enumerate(weights.layers):
        indexer_kind = config.geometry.indexer_types[layer_id]
        mlp_kind = config.geometry.mlp_layer_types[layer_id]
        index_slot = config.full_index_slot_by_layer[layer_id]
        layer_index_cache = index_cache[0 if index_slot is None else index_slot]
        result = ws32_transformer_layer_mapped(
            hidden_update,
            carried_residual,
            kv_cache[layer_id],
            layer_index_cache,
            selected_positions,
            selected_valid_counts,
            selected_scores,
            state.position,
            state.block_tables,
            state.context_lengths,
            layer_weights.qkv_a,
            layer_weights.attention,
            layer_weights.dsa,
            layer_weights.post_attention_norm_weight_local,
            layer_weights.dense,
            layer_weights.moe,
            health,
            indexer_kind=indexer_kind,
            mlp_kind=mlp_kind,
            dsa_contract=config.dsa_contract,
            attention_contract=config.attention_contract,
            moe_contract=config.moe_contract,
            cache_layout=config.cache_layout,
            block_shape=config.geometry.fp8_block_shape,
            rms_norm_epsilon=config.rms_norm_epsilon,
            sparse_attention_config=sparse_config,
            sparse_attention_interpret=sparse_attention_interpret,
            linear_interpret=linear_interpret,
        )
        hidden_update = result.output_local
        carried_residual = result.carried_residual_local
        kv_cache = kv_cache.at[layer_id].set(result.cache_local)
        if index_slot is not None:
            index_cache = index_cache.at[index_slot].set(
                result.index_cache_local
            )
            if observe_dsa:
                observed_positions.append(result.selected_positions)
                observed_valid_counts.append(result.selected_valid_counts)
                observed_scores.append(result.selected_scores)
        selected_positions = result.selected_positions
        selected_valid_counts = result.selected_valid_counts
        selected_scores = result.selected_scores
        health = result.contract_valid

    sampled: Ws32SplitGreedySampleResult = ws32_split_final_sample_mapped(
        hidden_update,
        carried_residual,
        weights.final_norm_weight_local,
        weights.lm_head_local,
        hidden_size=config.geometry.hidden_size,
        vocab_size=config.geometry.vocab_size,
        rms_norm_epsilon=config.rms_norm_epsilon,
    )
    next_state = Ws32DecoderState(
        kv_cache,
        index_cache,
        selected_positions,
        selected_valid_counts,
        selected_scores,
        state.position + jnp.ones_like(state.position),
        state.block_tables,
        state.context_lengths + jnp.ones_like(state.context_lengths),
        health & sampled.contract_valid,
    )
    step = Ws32DecodeStepResult(
        next_state,
        sampled.token_id,
        sampled.final_residual_local,
    )
    if not observe_dsa:
        return step, None
    if len(observed_positions) != len(config.full_index_slots):
        raise AssertionError("WS32 DSA observation cardinality drifted")
    observation = Ws32DsaObservation(
        jnp.asarray(config.full_index_slots, dtype=jnp.int32),
        jnp.stack(tuple(observed_positions), axis=0),
        jnp.stack(tuple(observed_valid_counts), axis=0),
        jnp.stack(tuple(observed_scores), axis=0),
    )
    return step, observation


def ws32_decode_mapped(
    token_ids: Any,
    state: Ws32DecoderState,
    weights: Ws32DecoderWeights,
    *,
    config: Ws32DecoderConfig,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> Ws32DecodeStepResult:
    """Execute one complete batch-one target-model step on all 32 chips."""

    result, observation = _ws32_decode_impl(
        token_ids,
        state,
        weights,
        config=config,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
        observe_dsa=False,
    )
    if observation is not None:
        raise AssertionError("default WS32 decoder retained DSA observations")
    return result


def ws32_decode_observed_mapped(
    token_ids: Any,
    state: Ws32DecoderState,
    weights: Ws32DecoderWeights,
    *,
    config: Ws32DecoderConfig,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> Ws32ObservedDecodeStepResult:
    """Execute one proof-only step returning all 21 full-indexer decisions."""

    result, observation = _ws32_decode_impl(
        token_ids,
        state,
        weights,
        config=config,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
        observe_dsa=True,
    )
    if observation is None:
        raise AssertionError("observed WS32 decoder lost DSA observations")
    return Ws32ObservedDecodeStepResult(result, observation)


def ws32_cache_write_probe_mapped(
    state: Ws32DecoderState,
    *,
    config: Ws32DecoderConfig,
    expert_axis: str = "expert",
) -> Ws32CacheWriteProbe:
    """Gather only the last written cache rows, never a context-sized tensor."""

    _validate_local_state(state, config)
    position = state.position - jnp.ones_like(state.position)
    logical_page = position[0] // jnp.int32(config.logical_page_size)
    owner = (
        position[0] % jnp.int32(config.logical_page_size)
    ) // jnp.int32(config.local_rows_per_page)
    local_row = (
        position[0] % jnp.int32(config.local_rows_per_page)
    )
    safe_page = jnp.clip(logical_page, 0, state.block_tables.shape[1] - 1)
    physical_page = state.block_tables[0, safe_page]
    safe_physical_page = jnp.clip(
        physical_page, 0, state.kv_cache_local.shape[1] - 1
    )
    owns = jax.lax.axis_index(expert_axis) == owner
    kv_local = state.kv_cache_local[
        :, safe_physical_page, local_row, :
    ]
    index_local = state.index_cache_local[
        :, safe_physical_page, local_row, :
    ]
    kv_local = jnp.where(owns, kv_local, jnp.zeros_like(kv_local))
    index_local = jnp.where(owns, index_local, jnp.zeros_like(index_local))
    with jax.named_scope("greenfield_ws32_cache_probe/kv_owner_reduce"):
        kv_rows = jax.lax.psum(kv_local, axis_name=expert_axis)
    with jax.named_scope("greenfield_ws32_cache_probe/index_owner_reduce"):
        index_rows = jax.lax.psum(index_local, axis_name=expert_axis)
    metadata_valid = (
        (position[0] >= 0)
        & (position[0] < state.context_lengths[0])
        & (logical_page >= 0)
        & (logical_page < state.block_tables.shape[1])
        & (physical_page >= 0)
        & (physical_page < state.kv_cache_local.shape[1])
    )
    content_valid = (
        jnp.all(jnp.isfinite(kv_rows))
        & jnp.all(jnp.isfinite(index_rows))
        & jnp.all(jnp.any(kv_rows != jnp.bfloat16(0), axis=-1))
        & jnp.all(jnp.any(index_rows != jnp.bfloat16(0), axis=-1))
    )
    return Ws32CacheWriteProbe(
        position,
        kv_rows,
        index_rows,
        state.contract_valid & (metadata_valid & content_valid)[None],
    )


def build_ws32_decoder_program(
    mesh: Any,
    config: Ws32DecoderConfig,
    *,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> Ws32DecoderProgram:
    """Build the exact normal, observer, and cache-probe shard-map programs."""

    import numpy as np

    if tuple(mesh.axis_names) != ("expert", "feature") or tuple(
        np.asarray(mesh.devices, dtype=object).shape
    ) != (8, 4):
        raise PlanValidationError("WS32 decoder requires one exact expert8 x feature4 mesh")
    weight_specs = ws32_decoder_weight_specs(config)
    state_specs = ws32_decoder_state_specs()

    def execute_body(
        token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
    ) -> Ws32DecodeStepResult:
        with jax.named_scope("greenfield_ws32_complete_decoder"):
            return ws32_decode_mapped(
                token_ids,
                state,
                weights,
                config=config,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
            )

    def observe_body(
        token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
    ) -> Ws32ObservedDecodeStepResult:
        with jax.named_scope("greenfield_ws32_complete_decoder_dsa_observer"):
            return ws32_decode_observed_mapped(
                token_ids,
                state,
                weights,
                config=config,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
            )

    def probe_body(state: Ws32DecoderState) -> Ws32CacheWriteProbe:
        with jax.named_scope("greenfield_ws32_cache_probe"):
            return ws32_cache_write_probe_mapped(state, config=config)

    return Ws32DecoderProgram(
        config=config,
        mesh=mesh,
        execute=jax.shard_map(
            execute_body,
            mesh=mesh,
            in_specs=(P(), state_specs, weight_specs),
            out_specs=ws32_decode_result_specs(),
            check_vma=False,
        ),
        observe=jax.shard_map(
            observe_body,
            mesh=mesh,
            in_specs=(P(), state_specs, weight_specs),
            out_specs=ws32_observed_decode_result_specs(),
            check_vma=False,
        ),
        probe_cache_write=jax.shard_map(
            probe_body,
            mesh=mesh,
            in_specs=(state_specs,),
            out_specs=ws32_cache_write_probe_specs(),
            check_vma=False,
        ),
    )


def build_ws32_teacher_forced_prefill_program(
    mesh: Any,
    config: Ws32DecoderConfig,
    *,
    prompt_length: int,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> Ws32TeacherForcedPrefillProgram:
    """Build a complete teacher-forced prompt scan inside one shard map."""

    if not isinstance(prompt_length, int) or isinstance(prompt_length, bool) or (
        prompt_length <= 0 or prompt_length >= config.context_capacity
    ):
        raise PlanValidationError(
            "WS32 prefill prompt must be positive and leave decode capacity"
        )
    import numpy as np

    if tuple(mesh.axis_names) != ("expert", "feature") or tuple(
        np.asarray(mesh.devices, dtype=object).shape
    ) != (8, 4):
        raise PlanValidationError("WS32 prefill requires one exact expert8 x feature4 mesh")

    def body(
        prompt_token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
    ) -> Ws32PrefillResult:
        if prompt_token_ids.shape != (prompt_length,) or (
            prompt_token_ids.dtype != jnp.int32
        ):
            raise ValueError("WS32 teacher-forced prompt geometry drifted")
        initial_token = jnp.full((1,), -1, dtype=jnp.int32)

        def scan_step(
            carry: tuple[Ws32DecoderState, Any], token: Any
        ) -> tuple[tuple[Ws32DecoderState, Any], None]:
            result = ws32_decode_mapped(
                token[None],
                carry[0],
                weights,
                config=config,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
            )
            return (result.state, result.next_token), None

        with jax.named_scope("greenfield_ws32_teacher_forced_prefill"):
            final, _ = jax.lax.scan(
                scan_step,
                (state, initial_token),
                prompt_token_ids,
                unroll=1,
            )
        return Ws32PrefillResult(final[0], final[1])

    return Ws32TeacherForcedPrefillProgram(
        config=config,
        mesh=mesh,
        prompt_length=prompt_length,
        execute=jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(), ws32_decoder_state_specs(), ws32_decoder_weight_specs(config)),
            out_specs=ws32_prefill_result_specs(),
            check_vma=False,
        ),
    )
