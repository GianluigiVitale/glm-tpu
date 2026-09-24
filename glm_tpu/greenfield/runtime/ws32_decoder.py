"""Isolated all-chip WS32 batch-one decoder composition.

The decoder is one ``shard_map`` body over ``expert=8 x feature=4``.  Every
layer executes on its stationary final-owner weights; the residual remains
feature-sharded, KV rows remain context-sharded over expert, and only compact
DSA selections are carried between IndexShare layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from ...optimized.errors import PlanValidationError
from ..kernels.pallas.sparse_attention import SparseMlaConfig
from ...optimized.reference.attention import MlaNumericalContract, StageLocalKvLayout
from ...optimized.reference.dsa import DsaNumericalContract
from ...optimized.reference.fp8 import dequantize_fp8_bits_block_weight
from ...optimized.reference.moe import GlmMoeNumericalContract
from ...optimized.reference.prefill_index import (
    decode_stage_local_prefill_index_wk_bf16,
    promote_stage_local_prefill_index_wk,
    repair_stage_local_prompt_index_cache,
)
from ...optimized.ws32_io import (
    Ws32SplitGreedySampleResult,
    ws32_embedding_mapped,
    ws32_split_final_sample_mapped,
)
from ..kernels.ws32_layer import (
    Ws32AttentionWeights,
    Ws32DenseWeights,
    Ws32DsaWeights,
    Ws32ExactDsaWeights,
    Ws32MoeWeights,
    Ws32QkvAWeights,
    Ws32StrategyNdDenseWeights,
    ws32_transformer_layer_mapped,
)
from ...optimized.geometry import ModelGeometry
from glm_tpu.optimized.ws32_decoder import (
    WS32_MAIN_ROPE_THETA,
    Ws32DecodeStepResult,
    Ws32DecoderConfig,
    Ws32DecoderState,
    Ws32DecoderWeights,
    Ws32LayerWeights,
    _attention_specs,
    _bind_weight_name_tree,
    _dense_specs,
    _dsa_specs,
    _fp8_names,
    _moe_specs,
    _qkv_specs,
    _validate_local_state,
    _weight_name_leaves,
    bind_ws32_decoder_weights,
    build_ws32_main_rope_table,
    ws32_decode_result_specs,
    ws32_decoder_state_specs,
    ws32_decoder_weight_names,
    ws32_decoder_weight_specs,
)  # S2f: moved to production


class Ws32ExactDsaRawLayerWeights(NamedTuple):
    q_a_bits_local: Any
    q_a_scale_local: Any
    kv_a_bits_local: Any
    kv_a_scale_local: Any
    wq_b_bits_local: Any
    wq_b_scale_local: Any
    wk_bits_local: Any
    wk_scale_local: Any
    head_weight_local: Any


class Ws32DecodedExactDsaWeights(NamedTuple):
    qkv_a_bits: Any
    qkv_a_scale: Any
    wq_b_weight_local: Any
    wk_weight_bf16: Any
    head_weight_local: Any


class Ws32DsaObservation(NamedTuple):
    """All full-indexer decisions from one complete decoder step."""

    producer_layer_ids: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any


class Ws32ObservedDecodeStepResult(NamedTuple):
    result: Ws32DecodeStepResult
    dsa: Ws32DsaObservation


class Ws32PrefillStepResult(NamedTuple):
    result: Ws32DecodeStepResult
    normalized_inputs_local: Any


class Ws32PrefillResult(NamedTuple):
    state: Ws32DecoderState
    next_token: Any


class Ws32ChunkedPrefillResult(NamedTuple):
    """One exact prompt chunk: carried state plus the separate repaired index buffer.

    The carried ``state`` keeps the unrepaired on-device index rows every later
    prompt step must score (the sealed monolithic semantics); the exact M64
    repaired rows of this chunk's positions are written only into
    ``repaired_index_local``, which the runner installs as the index cache once
    the last chunk has been scanned.
    """

    state: Ws32DecoderState
    next_token: Any
    repaired_index_local: Any


class Ws32CacheWriteProbe(NamedTuple):
    """Compact rows written at the most recently completed position."""

    position: Any
    kv_rows: Any
    index_rows: Any
    contract_valid: Any


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


@dataclass(frozen=True, slots=True)
class Ws32ChunkedPrefillProgram:
    """One device-resident scan of a fixed-length prompt chunk (spec §23.2)."""

    config: Ws32DecoderConfig
    mesh: Any
    chunk_length: int
    repair_prompt_chunk: int
    execute: Any


@dataclass(frozen=True, slots=True)
class Ws32ExactDsaMaterializerProgram:
    """Two completed device boundaries for DB526/527 exact owners."""

    config: Ws32DecoderConfig
    mesh: Any
    decode: Any
    promote: Any


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


def select_ws32_exact_dsa_raw_weights(
    weights: Ws32DecoderWeights,
    config: Ws32DecoderConfig,
) -> tuple[Ws32ExactDsaRawLayerWeights, ...]:
    """Select raw full-indexer leaves without copying checkpoint arrays."""

    if not config.exact_dsa:
        raise ValueError("WS32 exact DSA raw selection requires its default-off flag")
    selected = []
    for layer_id in config.full_index_slots:
        layer = weights.layers[layer_id]
        if layer.dsa is None:
            raise ValueError("WS32 exact DSA source layer lost indexer weights")
        selected.append(
            Ws32ExactDsaRawLayerWeights(
                layer.qkv_a.q_a_bits_local,
                layer.qkv_a.q_a_scale_local,
                layer.qkv_a.kv_a_bits_local,
                layer.qkv_a.kv_a_scale_local,
                layer.dsa.wq_b_bits_local,
                layer.dsa.wq_b_scale_local,
                layer.dsa.wk_bits_local,
                layer.dsa.wk_scale_local,
                layer.dsa.head_weight_local,
            )
        )
    if len(selected) != len(config.full_index_slots):
        raise AssertionError("WS32 exact DSA source cardinality drifted")
    return tuple(selected)


def _exact_dsa_raw_specs(
    config: Ws32DecoderConfig,
) -> tuple[Ws32ExactDsaRawLayerWeights, ...]:
    return tuple(
        Ws32ExactDsaRawLayerWeights(
            P(None, "feature"),
            P(None, "feature"),
            P(None, "feature"),
            P(None, "feature"),
            P("expert", None),
            P("expert", None),
            P(None, "feature"),
            P(None, "feature"),
            P("expert", "feature"),
        )
        for _ in config.full_index_slots
    )


def ws32_decoded_exact_dsa_specs(
    config: Ws32DecoderConfig,
) -> tuple[Ws32DecodedExactDsaWeights, ...]:
    return tuple(
        Ws32DecodedExactDsaWeights(
            P(),
            P(),
            P("feature", None),
            P(),
            P("feature", None),
        )
        for _ in config.full_index_slots
    )


def ws32_exact_dsa_specs(
    config: Ws32DecoderConfig,
) -> tuple[Ws32ExactDsaWeights, ...]:
    return tuple(
        Ws32ExactDsaWeights(
            P(),
            P(),
            (P("feature", None),) * 4,
            P(),
            P("feature", None),
        )
        for _ in config.full_index_slots
    )


def _pack_ws32_fused_qkv_a(
    q_bits: Any,
    q_scale: Any,
    kv_bits: Any,
    kv_scale: Any,
    *,
    config: Ws32DecoderConfig,
) -> tuple[Any, Any]:
    geometry = config.geometry
    q_width = geometry.q_lora_rank
    kv_width = geometry.kv_lora_rank + geometry.qk_rope_head_dim
    virtual_shards = 32
    if q_width % virtual_shards or kv_width % virtual_shards:
        raise ValueError("WS32 exact qkv-a virtual shard geometry drifted")
    q_local = q_width // virtual_shards
    kv_local = kv_width // virtual_shards
    q_packed = jnp.transpose(
        q_bits.reshape(virtual_shards, q_local, geometry.hidden_size),
        (0, 2, 1),
    )
    kv_packed = jnp.transpose(
        kv_bits.reshape(virtual_shards, kv_local, geometry.hidden_size),
        (0, 2, 1),
    )
    q_expanded = jnp.repeat(q_scale, 128, axis=0)[:q_width]
    kv_expanded = jnp.repeat(kv_scale, 128, axis=0)[:kv_width]
    q_scale_packed = jnp.transpose(
        q_expanded.reshape(virtual_shards, q_local, q_scale.shape[1]),
        (0, 2, 1),
    )
    kv_scale_packed = jnp.transpose(
        kv_expanded.reshape(virtual_shards, kv_local, kv_scale.shape[1]),
        (0, 2, 1),
    )
    return (
        jnp.concatenate((q_packed, kv_packed), axis=-1),
        jnp.concatenate((q_scale_packed, kv_scale_packed), axis=-1),
    )


def build_ws32_exact_dsa_materializer_program(
    mesh: Any,
    config: Ws32DecoderConfig,
) -> Ws32ExactDsaMaterializerProgram:
    """Build the one-shot exact-owner decode and BF16-to-FP32 boundary."""

    import numpy as np

    if not config.exact_dsa:
        raise PlanValidationError("WS32 exact materializer requires exact_dsa")
    if tuple(mesh.axis_names) != ("expert", "feature") or tuple(
        np.asarray(mesh.devices, dtype=object).shape
    ) != (8, 4):
        raise PlanValidationError("WS32 exact materializer requires expert8 x feature4")
    contract = config.dsa_contract

    def gather_feature(value: Any, *, axis: int) -> Any:
        return lax.all_gather(
            value,
            axis_name="feature",
            axis=axis,
            tiled=True,
        )

    def gather_expert(value: Any, *, axis: int) -> Any:
        return lax.all_gather(
            value,
            axis_name="expert",
            axis=axis,
            tiled=True,
        )

    def decode_layer(raw: Ws32ExactDsaRawLayerWeights) -> Ws32DecodedExactDsaWeights:
        with jax.named_scope("greenfield_ws32_exact_dsa_materializer/qkv_a"):
            q_bits = gather_feature(raw.q_a_bits_local, axis=1)
            q_scale = gather_feature(raw.q_a_scale_local, axis=1)
            kv_bits = gather_feature(raw.kv_a_bits_local, axis=1)
            kv_scale = gather_feature(raw.kv_a_scale_local, axis=1)
            qkv_bits, qkv_scale = _pack_ws32_fused_qkv_a(
                q_bits,
                q_scale,
                kv_bits,
                kv_scale,
                config=config,
            )
        with jax.named_scope("greenfield_ws32_exact_dsa_materializer/tuple4_query"):
            full_wq_bits = gather_expert(raw.wq_b_bits_local, axis=0)
            full_wq_scale = gather_expert(raw.wq_b_scale_local, axis=0)
            owner_width = contract.num_heads * contract.head_dim // 4
            owner_scale_rows = owner_width // config.geometry.fp8_block_shape[0]
            owner = lax.axis_index("feature")
            wq_bits = lax.dynamic_slice_in_dim(
                full_wq_bits,
                owner * owner_width,
                owner_width,
                axis=0,
            )
            wq_scale = lax.dynamic_slice_in_dim(
                full_wq_scale,
                owner * owner_scale_rows,
                owner_scale_rows,
                axis=0,
            )
            wq_weight = dequantize_fp8_bits_block_weight(
                wq_bits,
                wq_scale,
                block_shape=config.geometry.fp8_block_shape,
                output_dtype=jnp.float32,
            )
        with jax.named_scope("greenfield_ws32_exact_dsa_materializer/wk_decode"):
            wk_bits = gather_feature(raw.wk_bits_local, axis=1)
            wk_scale = gather_feature(raw.wk_scale_local, axis=1)
            wk_bf16 = decode_stage_local_prefill_index_wk_bf16(
                wk_bits,
                wk_scale,
                contract=contract,
                fp8_block_shape=config.geometry.fp8_block_shape,
            )
        with jax.named_scope("greenfield_ws32_exact_dsa_materializer/head_owner"):
            full_head = gather_expert(
                gather_feature(raw.head_weight_local, axis=1),
                axis=0,
            )
            local_heads = contract.num_heads // 4
            head_weight = lax.dynamic_slice_in_dim(
                full_head,
                lax.axis_index("feature") * local_heads,
                local_heads,
                axis=0,
            )
        return Ws32DecodedExactDsaWeights(
            qkv_bits,
            qkv_scale,
            wq_weight,
            wk_bf16,
            head_weight,
        )

    def decode_body(
        raw_layers: tuple[Ws32ExactDsaRawLayerWeights, ...],
    ) -> tuple[Ws32DecodedExactDsaWeights, ...]:
        return tuple(decode_layer(raw) for raw in raw_layers)

    def promote_body(
        decoded_layers: tuple[Ws32DecodedExactDsaWeights, ...],
    ) -> tuple[Ws32ExactDsaWeights, ...]:
        values = []
        for decoded in decoded_layers:
            with jax.named_scope(
                "greenfield_ws32_exact_dsa_materializer/wk_promote"
            ):
                wk_weight = promote_stage_local_prefill_index_wk(
                    decoded.wk_weight_bf16,
                    contract=contract,
                )
            values.append(
                Ws32ExactDsaWeights(
                    decoded.qkv_a_bits,
                    decoded.qkv_a_scale,
                    (decoded.wq_b_weight_local,) * 4,
                    wk_weight,
                    decoded.head_weight_local,
                )
            )
        return tuple(values)

    decoded_specs = ws32_decoded_exact_dsa_specs(config)
    return Ws32ExactDsaMaterializerProgram(
        config=config,
        mesh=mesh,
        decode=jax.shard_map(
            decode_body,
            mesh=mesh,
            in_specs=(_exact_dsa_raw_specs(config),),
            out_specs=decoded_specs,
            check_vma=False,
        ),
        promote=jax.shard_map(
            promote_body,
            mesh=mesh,
            in_specs=(decoded_specs,),
            out_specs=ws32_exact_dsa_specs(config),
            check_vma=False,
        ),
    )


def ws32_dsa_observation_specs() -> Ws32DsaObservation:
    return Ws32DsaObservation(P(), P(), P(), P())


def ws32_observed_decode_result_specs() -> Ws32ObservedDecodeStepResult:
    return Ws32ObservedDecodeStepResult(
        ws32_decode_result_specs(), ws32_dsa_observation_specs()
    )


def ws32_prefill_result_specs() -> Ws32PrefillResult:
    return Ws32PrefillResult(ws32_decoder_state_specs(), P())


def ws32_chunked_prefill_result_specs() -> Ws32ChunkedPrefillResult:
    return Ws32ChunkedPrefillResult(
        ws32_decoder_state_specs(), P(), P(None, None, "expert", None)
    )


def ws32_cache_write_probe_specs() -> Ws32CacheWriteProbe:
    return Ws32CacheWriteProbe(P(), P(), P(), P())


def _ws32_decode_impl(
    token_ids: Any,
    state: Ws32DecoderState,
    weights: Ws32DecoderWeights,
    *,
    config: Ws32DecoderConfig,
    exact_dsa_weights: tuple[Ws32ExactDsaWeights, ...] | None = None,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    observe_dsa: bool,
    observe_prefill_inputs: bool = False,
    main_rope_table: Any | None = None,
    final_sample: Callable[..., Ws32SplitGreedySampleResult] | None = None,
) -> tuple[
    Ws32DecodeStepResult,
    Ws32DsaObservation | None,
    Any | None,
]:
    """Execute one complete batch-one step and optionally retain DSA events."""

    _validate_local_state(state, config)
    if config.host_main_rope_table != (main_rope_table is not None):
        raise ValueError("WS32 host main-rotary table flag/input presence drifted")
    main_rope_table_row = None
    if main_rope_table is not None:
        if main_rope_table.shape != config.main_rope_table_shape or (
            main_rope_table.dtype != jnp.bfloat16
        ):
            raise ValueError("WS32 main rotary table geometry drifted")
        with jax.named_scope("greenfield_ws32_main_rope_table_lookup"):
            # ``mode="clip"`` matches the PP16 sibling's basic indexing; JAX's
            # default fill mode would put NaN in the cache for a position at or
            # past the capacity.  The runner's capacity contract
            # (context_capacity > prompt + every generated step) makes the clamp
            # unreachable in a valid run.
            main_rope_table_row = jnp.take(
                main_rope_table, state.position, axis=0, mode="clip"
            )[0]
    if token_ids.shape != (1,) or token_ids.dtype != jnp.int32:
        raise ValueError("WS32 decoder input must be one int32 token")
    if len(weights.layers) != config.geometry.num_layers:
        raise ValueError("WS32 decoder weight layer count drifted")
    if config.exact_dsa != (exact_dsa_weights is not None):
        raise ValueError("WS32 exact DSA flag/input presence drifted")
    if exact_dsa_weights is not None and len(exact_dsa_weights) != len(
        config.full_index_slots
    ):
        raise ValueError("WS32 exact DSA owner cardinality drifted")
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
    prefill_inputs = []

    for layer_id, layer_weights in enumerate(weights.layers):
        indexer_kind = config.geometry.indexer_types[layer_id]
        mlp_kind = config.geometry.mlp_layer_types[layer_id]
        index_slot = config.full_index_slot_by_layer[layer_id]
        exact_layer_weights = (
            None if index_slot is None or exact_dsa_weights is None else exact_dsa_weights[index_slot]
        )
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
            exact_dsa_weights=exact_layer_weights,
            main_rope_table_row=main_rope_table_row,
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
            if observe_prefill_inputs:
                prefill_inputs.append(result.normalized_input_local[0])
        selected_positions = result.selected_positions
        selected_valid_counts = result.selected_valid_counts
        selected_scores = result.selected_scores
        health = result.contract_valid

    # The request builder may replace only this output boundary. The default
    # remains the original fused norm/logits/greedy head, with no added inputs.
    sample_head = ws32_split_final_sample_mapped if final_sample is None else final_sample
    sampled: Ws32SplitGreedySampleResult = sample_head(
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
        observation = None
    else:
        if len(observed_positions) != len(config.full_index_slots):
            raise AssertionError("WS32 DSA observation cardinality drifted")
        observation = Ws32DsaObservation(
            jnp.asarray(config.full_index_slots, dtype=jnp.int32),
            jnp.stack(tuple(observed_positions), axis=0),
            jnp.stack(tuple(observed_valid_counts), axis=0),
            jnp.stack(tuple(observed_scores), axis=0),
        )
    if not observe_prefill_inputs:
        prefill = None
    else:
        if len(prefill_inputs) != len(config.full_index_slots):
            raise AssertionError("WS32 prefill input cardinality drifted")
        prefill = jnp.stack(tuple(prefill_inputs), axis=0)
    return step, observation, prefill


def ws32_decode_mapped(
    token_ids: Any,
    state: Ws32DecoderState,
    weights: Ws32DecoderWeights,
    *,
    config: Ws32DecoderConfig,
    exact_dsa_weights: tuple[Ws32ExactDsaWeights, ...] | None = None,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    main_rope_table: Any | None = None,
) -> Ws32DecodeStepResult:
    """Execute one complete batch-one target-model step on all 32 chips."""

    result, observation, prefill = _ws32_decode_impl(
        token_ids,
        state,
        weights,
        config=config,
        exact_dsa_weights=exact_dsa_weights,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
        main_rope_table=main_rope_table,
        observe_dsa=False,
    )
    if observation is not None:
        raise AssertionError("default WS32 decoder retained DSA observations")
    if prefill is not None:
        raise AssertionError("default WS32 decoder retained prefill inputs")
    return result


def ws32_decode_observed_mapped(
    token_ids: Any,
    state: Ws32DecoderState,
    weights: Ws32DecoderWeights,
    *,
    config: Ws32DecoderConfig,
    exact_dsa_weights: tuple[Ws32ExactDsaWeights, ...] | None = None,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    main_rope_table: Any | None = None,
) -> Ws32ObservedDecodeStepResult:
    """Execute one proof-only step returning all 21 full-indexer decisions."""

    result, observation, prefill = _ws32_decode_impl(
        token_ids,
        state,
        weights,
        config=config,
        exact_dsa_weights=exact_dsa_weights,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
        main_rope_table=main_rope_table,
        observe_dsa=True,
    )
    if observation is None:
        raise AssertionError("observed WS32 decoder lost DSA observations")
    if prefill is not None:
        raise AssertionError("observed WS32 decoder retained prefill inputs")
    return Ws32ObservedDecodeStepResult(result, observation)


def ws32_prefill_step_mapped(
    token_ids: Any,
    state: Ws32DecoderState,
    weights: Ws32DecoderWeights,
    exact_dsa_weights: tuple[Ws32ExactDsaWeights, ...],
    *,
    config: Ws32DecoderConfig,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    main_rope_table: Any | None = None,
) -> Ws32PrefillStepResult:
    """Execute one exact step and retain only full-indexer norm inputs."""

    result, observation, prefill = _ws32_decode_impl(
        token_ids,
        state,
        weights,
        config=config,
        exact_dsa_weights=exact_dsa_weights,
        sparse_attention_interpret=sparse_attention_interpret,
        linear_interpret=linear_interpret,
        main_rope_table=main_rope_table,
        observe_dsa=False,
        observe_prefill_inputs=True,
    )
    if observation is not None or prefill is None:
        raise AssertionError("WS32 exact prefill observation contract drifted")
    return Ws32PrefillStepResult(result, prefill)


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
    # Spec §23.8: the replicated host BF16 main-rotary table is one extra
    # trailing input; when the flag is off the bodies keep their default None.
    table_specs = (P(),) if config.host_main_rope_table else ()

    def execute_body(
        token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
        main_rope_table: Any | None = None,
    ) -> Ws32DecodeStepResult:
        with jax.named_scope("greenfield_ws32_complete_decoder"):
            return ws32_decode_mapped(
                token_ids,
                state,
                weights,
                config=config,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
                main_rope_table=main_rope_table,
            )

    def observe_body(
        token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
        main_rope_table: Any | None = None,
    ) -> Ws32ObservedDecodeStepResult:
        with jax.named_scope("greenfield_ws32_complete_decoder_dsa_observer"):
            return ws32_decode_observed_mapped(
                token_ids,
                state,
                weights,
                config=config,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
                main_rope_table=main_rope_table,
            )

    def execute_exact_body(
        token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
        exact_dsa_weights: tuple[Ws32ExactDsaWeights, ...],
        main_rope_table: Any | None = None,
    ) -> Ws32DecodeStepResult:
        with jax.named_scope("greenfield_ws32_complete_decoder"):
            return ws32_decode_mapped(
                token_ids,
                state,
                weights,
                config=config,
                exact_dsa_weights=exact_dsa_weights,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
                main_rope_table=main_rope_table,
            )

    def observe_exact_body(
        token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
        exact_dsa_weights: tuple[Ws32ExactDsaWeights, ...],
        main_rope_table: Any | None = None,
    ) -> Ws32ObservedDecodeStepResult:
        with jax.named_scope("greenfield_ws32_complete_decoder_dsa_observer"):
            return ws32_decode_observed_mapped(
                token_ids,
                state,
                weights,
                config=config,
                exact_dsa_weights=exact_dsa_weights,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
                main_rope_table=main_rope_table,
            )

    def probe_body(state: Ws32DecoderState) -> Ws32CacheWriteProbe:
        with jax.named_scope("greenfield_ws32_cache_probe"):
            return ws32_cache_write_probe_mapped(state, config=config)

    if config.exact_dsa:
        exact_specs = ws32_exact_dsa_specs(config)
        execute = jax.shard_map(
            execute_exact_body,
            mesh=mesh,
            in_specs=(P(), state_specs, weight_specs, exact_specs, *table_specs),
            out_specs=ws32_decode_result_specs(),
            check_vma=False,
        )
        observe = jax.shard_map(
            observe_exact_body,
            mesh=mesh,
            in_specs=(P(), state_specs, weight_specs, exact_specs, *table_specs),
            out_specs=ws32_observed_decode_result_specs(),
            check_vma=False,
        )
    else:
        execute = jax.shard_map(
            execute_body,
            mesh=mesh,
            in_specs=(P(), state_specs, weight_specs, *table_specs),
            out_specs=ws32_decode_result_specs(),
            check_vma=False,
        )
        observe = jax.shard_map(
            observe_body,
            mesh=mesh,
            in_specs=(P(), state_specs, weight_specs, *table_specs),
            out_specs=ws32_observed_decode_result_specs(),
            check_vma=False,
        )
    return Ws32DecoderProgram(
        config=config,
        mesh=mesh,
        execute=execute,
        observe=observe,
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

    # Spec §23.8: replicated host BF16 main-rotary table as one trailing input.
    table_specs = (P(),) if config.host_main_rope_table else ()
    if tuple(mesh.axis_names) != ("expert", "feature") or tuple(
        np.asarray(mesh.devices, dtype=object).shape
    ) != (8, 4):
        raise PlanValidationError("WS32 prefill requires one exact expert8 x feature4 mesh")

    def body(
        prompt_token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
        main_rope_table: Any | None = None,
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
                main_rope_table=main_rope_table,
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

    def exact_body(
        prompt_token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
        exact_dsa_weights: tuple[Ws32ExactDsaWeights, ...],
        main_rope_table: Any | None = None,
    ) -> Ws32PrefillResult:
        if prompt_token_ids.shape != (prompt_length,) or (
            prompt_token_ids.dtype != jnp.int32
        ):
            raise ValueError("WS32 teacher-forced prompt geometry drifted")
        initial_token = jnp.full((1,), -1, dtype=jnp.int32)

        def scan_step(
            carry: tuple[Ws32DecoderState, Any], token: Any
        ) -> tuple[tuple[Ws32DecoderState, Any], Any]:
            step = ws32_prefill_step_mapped(
                token[None],
                carry[0],
                weights,
                exact_dsa_weights,
                config=config,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
                main_rope_table=main_rope_table,
            )
            return (
                (step.result.state, step.result.next_token),
                step.normalized_inputs_local,
            )

        with jax.named_scope("greenfield_ws32_teacher_forced_prefill"):
            final, prompt_inputs_local = jax.lax.scan(
                scan_step,
                (state, initial_token),
                prompt_token_ids,
                unroll=1,
            )
        repaired_index_cache = final[0].index_cache_local
        owner = lax.axis_index("expert")
        for full_slot, layer_id in enumerate(config.full_index_slots):
            dsa = weights.layers[layer_id].dsa
            if dsa is None:
                raise AssertionError("WS32 exact prefill lost DSA parameters")
            with jax.named_scope(
                f"greenfield_ws32_exact_dsa/prompt_m64_repair_layer_{layer_id}"
            ):
                prompt_inputs = lax.all_gather(
                    prompt_inputs_local[:, full_slot, :],
                    axis_name="feature",
                    axis=1,
                    tiled=True,
                )
                repaired = repair_stage_local_prompt_index_cache(
                    repaired_index_cache[full_slot],
                    prompt_inputs,
                    final[0].block_tables,
                    exact_dsa_weights[full_slot].wk_weight,
                    dsa.key_norm_weight,
                    dsa.key_norm_bias,
                    owner,
                    contract=config.dsa_contract,
                    logical_page_size=config.logical_page_size,
                    local_rows_per_page=config.local_rows_per_page,
                    prompt_chunk=2048,
                    physical_rows=64,
                    local_parallel_size=8,
                )
                repaired_index_cache = repaired_index_cache.at[
                    full_slot
                ].set(repaired)
        repaired_state = Ws32DecoderState(
            final[0].kv_cache_local,
            repaired_index_cache,
            final[0].selected_positions,
            final[0].selected_valid_counts,
            final[0].selected_scores,
            final[0].position,
            final[0].block_tables,
            final[0].context_lengths,
            final[0].contract_valid,
        )
        return Ws32PrefillResult(repaired_state, final[1])

    if config.exact_dsa:
        execute = jax.shard_map(
            exact_body,
            mesh=mesh,
            in_specs=(
                P(),
                ws32_decoder_state_specs(),
                ws32_decoder_weight_specs(config),
                ws32_exact_dsa_specs(config),
                *table_specs,
            ),
            out_specs=ws32_prefill_result_specs(),
            check_vma=False,
        )
    else:
        execute = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(
                P(),
                ws32_decoder_state_specs(),
                ws32_decoder_weight_specs(config),
                *table_specs,
            ),
            out_specs=ws32_prefill_result_specs(),
            check_vma=False,
        )

    return Ws32TeacherForcedPrefillProgram(
        config=config,
        mesh=mesh,
        prompt_length=prompt_length,
        execute=execute,
    )


def ws32_prefill_chunk_plan(prompt_length: int, chunk_length: int) -> tuple[int, int]:
    """Return ``(full_chunks, tail_length)`` for a chunked exact prefill.

    The tail program always exists (``1 <= tail_length <= chunk_length``) so the
    compiled graph set is identical for every workload and no padding token is
    ever teacher-forced: ``full_chunks * chunk_length + tail_length == prompt_length``.
    """
    for name, value in (("prompt_length", prompt_length), ("chunk_length", chunk_length)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PlanValidationError(f"WS32 prefill {name} must be a positive integer")
    full_chunks = (prompt_length - 1) // chunk_length
    tail_length = prompt_length - full_chunks * chunk_length
    if not 1 <= tail_length <= chunk_length or (
        full_chunks * chunk_length + tail_length != prompt_length
    ):
        raise AssertionError("WS32 prefill chunk plan drifted")
    return full_chunks, tail_length


def ws32_repair_prompt_chunk(chunk_length: int) -> int:
    """Repair tile width for one chunk: multiple of 64 rows, at most 2048."""
    if not isinstance(chunk_length, int) or isinstance(chunk_length, bool) or chunk_length <= 0:
        raise PlanValidationError("WS32 repair chunk requires a positive chunk length")
    return min(2048, (chunk_length + 63) // 64 * 64)


def build_ws32_chunked_prefill_program(
    mesh: Any,
    config: Ws32DecoderConfig,
    *,
    chunk_length: int,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
) -> Ws32ChunkedPrefillProgram:
    """Build the exact teacher-forced scan of one fixed-length prompt chunk.

    Semantics are those of :func:`build_ws32_teacher_forced_prefill_program`
    split at chunk boundaries without changing what any step observes: the
    scan carries the unrepaired index cache, and the exact M64 repair of this
    chunk's positions (``position_offset`` = the state position at entry, a
    traced int32) is written into the separate ``repaired_index_local`` buffer
    that the caller threads through every chunk and installs after the last one.
    Only the exact-DSA configuration has a repair; the default path returns the
    buffer unchanged so both paths share one program signature.
    """
    if not isinstance(chunk_length, int) or isinstance(chunk_length, bool) or (
        chunk_length <= 0 or chunk_length >= config.context_capacity
    ):
        raise PlanValidationError(
            "WS32 prefill chunk must be positive and leave decode capacity"
        )
    import numpy as np

    # Spec §23.8: replicated host BF16 main-rotary table as one trailing input.
    table_specs = (P(),) if config.host_main_rope_table else ()
    if tuple(mesh.axis_names) != ("expert", "feature") or tuple(
        np.asarray(mesh.devices, dtype=object).shape
    ) != (8, 4):
        raise PlanValidationError("WS32 prefill requires one exact expert8 x feature4 mesh")
    repair_prompt_chunk = ws32_repair_prompt_chunk(chunk_length)

    def _require_chunk(prompt_token_ids: Any) -> None:
        if prompt_token_ids.shape != (chunk_length,) or (
            prompt_token_ids.dtype != jnp.int32
        ):
            raise ValueError("WS32 chunked prompt geometry drifted")

    def _require_buffer(repaired_index_local: Any, state: Ws32DecoderState) -> None:
        if (
            repaired_index_local.shape != state.index_cache_local.shape
            or repaired_index_local.dtype != state.index_cache_local.dtype
        ):
            raise ValueError("WS32 repaired index buffer geometry drifted")

    def body(
        prompt_token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
        repaired_index_local: Any,
        main_rope_table: Any | None = None,
    ) -> Ws32ChunkedPrefillResult:
        _require_chunk(prompt_token_ids)
        _require_buffer(repaired_index_local, state)
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
                main_rope_table=main_rope_table,
            )
            return (result.state, result.next_token), None

        with jax.named_scope("greenfield_ws32_teacher_forced_prefill"):
            final, _ = jax.lax.scan(
                scan_step,
                (state, initial_token),
                prompt_token_ids,
                unroll=1,
            )
        return Ws32ChunkedPrefillResult(final[0], final[1], repaired_index_local)

    def exact_body(
        prompt_token_ids: Any,
        state: Ws32DecoderState,
        weights: Ws32DecoderWeights,
        exact_dsa_weights: tuple[Ws32ExactDsaWeights, ...],
        repaired_index_local: Any,
        main_rope_table: Any | None = None,
    ) -> Ws32ChunkedPrefillResult:
        _require_chunk(prompt_token_ids)
        _require_buffer(repaired_index_local, state)
        initial_token = jnp.full((1,), -1, dtype=jnp.int32)
        # The state position at entry is the number of prompt tokens already
        # scanned, i.e. this chunk's first absolute position.
        chunk_start = jnp.asarray(state.position[0], dtype=jnp.int32)

        def scan_step(
            carry: tuple[Ws32DecoderState, Any], token: Any
        ) -> tuple[tuple[Ws32DecoderState, Any], Any]:
            step = ws32_prefill_step_mapped(
                token[None],
                carry[0],
                weights,
                exact_dsa_weights,
                config=config,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
                main_rope_table=main_rope_table,
            )
            return (
                (step.result.state, step.result.next_token),
                step.normalized_inputs_local,
            )

        with jax.named_scope("greenfield_ws32_teacher_forced_prefill"):
            final, chunk_inputs_local = jax.lax.scan(
                scan_step,
                (state, initial_token),
                prompt_token_ids,
                unroll=1,
            )
        repaired = repaired_index_local
        owner = lax.axis_index("expert")
        for full_slot, layer_id in enumerate(config.full_index_slots):
            dsa = weights.layers[layer_id].dsa
            if dsa is None:
                raise AssertionError("WS32 exact prefill lost DSA parameters")
            with jax.named_scope(
                f"greenfield_ws32_exact_dsa/prompt_m64_repair_layer_{layer_id}"
            ):
                chunk_inputs = lax.all_gather(
                    chunk_inputs_local[:, full_slot, :],
                    axis_name="feature",
                    axis=1,
                    tiled=True,
                )
                repaired_slot = repair_stage_local_prompt_index_cache(
                    repaired[full_slot],
                    chunk_inputs,
                    final[0].block_tables,
                    exact_dsa_weights[full_slot].wk_weight,
                    dsa.key_norm_weight,
                    dsa.key_norm_bias,
                    owner,
                    contract=config.dsa_contract,
                    logical_page_size=config.logical_page_size,
                    local_rows_per_page=config.local_rows_per_page,
                    prompt_chunk=repair_prompt_chunk,
                    physical_rows=64,
                    local_parallel_size=8,
                    position_offset=chunk_start,
                    valid_rows=chunk_length,
                )
                repaired = repaired.at[full_slot].set(repaired_slot)
        return Ws32ChunkedPrefillResult(final[0], final[1], repaired)

    buffer_spec = P(None, None, "expert", None)
    if config.exact_dsa:
        execute = jax.shard_map(
            exact_body,
            mesh=mesh,
            in_specs=(
                P(),
                ws32_decoder_state_specs(),
                ws32_decoder_weight_specs(config),
                ws32_exact_dsa_specs(config),
                buffer_spec,
                *table_specs,
            ),
            out_specs=ws32_chunked_prefill_result_specs(),
            check_vma=False,
        )
    else:
        execute = jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(
                P(),
                ws32_decoder_state_specs(),
                ws32_decoder_weight_specs(config),
                buffer_spec,
                *table_specs,
            ),
            out_specs=ws32_chunked_prefill_result_specs(),
            check_vma=False,
        )
    return Ws32ChunkedPrefillProgram(
        config=config,
        mesh=mesh,
        chunk_length=chunk_length,
        repair_prompt_chunk=repair_prompt_chunk,
        execute=execute,
    )


def make_ws32_repaired_index_buffer(mesh: Any, config: Ws32DecoderConfig) -> Any:
    """Allocate the zero repaired-index buffer with the index cache's sharding."""
    import ml_dtypes
    import numpy as np
    from jax.sharding import NamedSharding

    sharding = NamedSharding(mesh, P(None, None, "expert", None))
    shape = config.index_cache_shape
    local_shape = sharding.shard_shape(shape)
    return jax.make_array_from_callback(
        shape,
        sharding,
        lambda _: np.zeros(local_shape, dtype=ml_dtypes.bfloat16),
    )
