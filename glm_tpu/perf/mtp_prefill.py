"""Target prompt hidden export for native MTP bootstrap, outside frozen source.

The block runtime mirrors the frozen layer/cache/repair/commit flow, adding
post-final-norm hidden rows and their live mask. An AST test constrains the
mirror diff. Invalid/padded rows export zeros; a bad live hidden row refuses
all cache writes on every owner. Additional outputs can change XLA realization:
CPU comparison is not trained TPU numerical or HBM admission.
"""
from __future__ import annotations
from typing import Any, Callable, NamedTuple
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P
from ..greenfield.errors import PlanValidationError
from ..greenfield.kernels.pallas import SparseMlaConfig
from ..greenfield.kernels.reference.attention import StageLocalKvLayout
from ..greenfield.kernels.prefill_pending_rows import (apply_prefill_pending_rows, capture_prefill_pending_rows, prefill_pending_addresses)
from ..greenfield.kernels.prefill_flat_rows import apply_prefill_flat_rows
from ..greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped
from ..greenfield.kernels.ws32_io import Ws32SplitGreedySampleResult, ws32_split_final_sample_mapped
from ..greenfield.kernels.ws32_prefill_layer import ws32_prefill_transformer_layer_mapped
from ..greenfield.kernels.ws32_prefill_window import ws32_prefill_layer_window_mapped
from ..greenfield.runtime.ws32_decoder import (Ws32DecoderConfig, Ws32DecoderState, Ws32DecoderWeights, _validate_local_state, ws32_decoder_weight_specs)
from ..greenfield.runtime.ws32_batched_prefill import (Ws32BatchedPrefillState, _require_config, _require_window_options, _all_owners_healthy, ws32_prefill_embedding_mapped)
from ..greenfield.runtime import ws32_batched_prefill as frozen
from .function_bindings import bind_dependencies


class MtpPrefillResult(NamedTuple):
    state: Ws32BatchedPrefillState
    next_token: Any
    normalized_hidden_local: Any
    hidden_valid: Any


def export_hidden_mapped(update, residual, norm_weight, *, hidden_size, epsilon):
    """Final target normalization, with the same BF16 rounding as the head.

    Rows stay independent. The final token head still uses its original one-row
    expression; this batched export has its own numerical comparison requirement.
    """
    normalized, final_residual = ws32_fused_add_rms_norm_mapped(
        update, residual, norm_weight, global_hidden_size=hidden_size, epsilon=epsilon,
    )
    valid = (jnp.all(jnp.isfinite(normalized), axis=1)
             & jnp.all(jnp.isfinite(final_residual), axis=1))
    return normalized, valid


def ws32_batched_prefill_mapped(
    token_ids: Any,
    valid_rows: Any,
    state: Ws32BatchedPrefillState,
    weights: Ws32DecoderWeights,
    materialized_wk: tuple[Any, ...],
    main_rope_table: Any,
    *,
    config: Ws32DecoderConfig,
    key_tile: int = 4096,
    sparse_attention_interpret: bool = False,
    linear_interpret: bool = False,
    mlp_window: bool = False,
    paired_position_sort: bool = False,
    rolled_prefix: bool = False,
    expert_panels: bool = False,
    sorted_local_merge: bool = False,
    canonical_dense: bool = False,
    pending_cache_rows: bool = False,
    flat_pending_rows: bool = False,
    capture_barrier: bool = False,
    final_sample: Callable[..., Ws32SplitGreedySampleResult] | None = None,
) -> MtpPrefillResult:
    """Propose one complete layer-major block; commit only all-owner success.

    The position in the carried state is the sole append offset. Intermediate
    states cannot be passed back after repair promotion: ``finished`` refuses
    further prefill. Failed proposals leave both cache trees/frontiers intact
    and latch false health on all owners. Prefix authenticity belongs to the
    fresh allocator or authenticated restore, not a guessed nonzero position.
    """
    _require_config(config)
    if type(pending_cache_rows) is not bool:
        raise PlanValidationError("pending cache rows must be a static bool")
    if type(flat_pending_rows) is not bool or (flat_pending_rows and not pending_cache_rows):
        raise PlanValidationError("flat pending rows require static bool and pending cache rows")
    if type(capture_barrier) is not bool or (capture_barrier and not flat_pending_rows):
        raise PlanValidationError("capture barrier requires static bool and flat pending rows")
    if type(paired_position_sort) is not bool:
        raise ValueError("paired position sort must be a static bool")
    _require_window_options(
        mlp_window, rolled_prefix, expert_panels, sorted_local_merge, canonical_dense
    )
    _validate_local_state(state.decoder, config)
    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("batched prefill requires expert8/feature4")
    if (
        token_ids.ndim != 1
        or not 1 <= token_ids.shape[0] <= (128 if mlp_window else 32)
        or token_ids.dtype != jnp.int32
    ):
        raise ValueError("batched prefill row count/dtype exceeds selected mode")
    if canonical_dense and token_ids.shape[0] not in (114, 128):
        raise ValueError("canonical dense requires physical B114/B128")
    for value, dtype in (
        (valid_rows, jnp.int32),
        (state.prompt_length, jnp.int32),
        (state.finished, jnp.bool_),
    ):
        if value.shape != () or value.dtype != dtype:
            raise ValueError("batched prefill scalar metadata geometry drifted")
    if (
        state.repaired_index_local.shape != state.decoder.index_cache_local.shape
        or state.repaired_index_local.dtype != jnp.bfloat16
    ):
        raise ValueError("batched repaired buffer geometry drifted")
    if (
        main_rope_table.shape != config.main_rope_table_shape
        or main_rope_table.dtype != jnp.bfloat16
    ):
        raise ValueError("batched main rotary table geometry drifted")
    if len(weights.layers) != config.geometry.num_layers or len(materialized_wk) != len(
        config.full_index_slots
    ):
        raise ValueError("batched layer/repair owner cardinality drifted")
    for layer_id, layer in enumerate(weights.layers):
        full = config.full_index_slot_by_layer[layer_id] is not None
        dense = config.geometry.mlp_layer_types[layer_id] == "dense"
        if (
            (layer.dsa is not None) != full
            or (layer.dense is not None) != dense
            or (layer.moe is not None) == dense
        ):
            raise ValueError("batched weights disagree with model layer schedule")
    for wk in materialized_wk:
        if (
            wk.shape
            != (config.geometry.dsa_indexer_head_dim, config.geometry.hidden_size)
            or wk.dtype != jnp.float32
        ):
            raise ValueError(
                "batched repair requires completed full FP32 wk per producer"
            )

    rows = token_ids.shape[0]
    decoder = state.decoder
    offset = decoder.position[0]
    count = jnp.clip(valid_rows, 0, rows)
    # Avoid signed overflow even for adversarial caller metadata. Clipping is
    # only for safe execution; the original values still determine refusal.
    start = jnp.clip(offset, 0, config.context_capacity - 1)
    end = start + jnp.minimum(count, config.context_capacity - start)
    span_valid = (
        (valid_rows > 0)
        & (valid_rows <= rows)
        & (offset >= 0)
        & (offset == start)
        & (state.prompt_length > 0)
        & (state.prompt_length < config.context_capacity)
        & (start <= state.prompt_length - count)
        & (decoder.context_lengths[0] == start + 1)
        & ~state.finished
        & jnp.all(decoder.contract_valid)
    )
    live = jnp.arange(rows, dtype=jnp.int32) < count
    positions = jnp.minimum(
        start + jnp.arange(rows, dtype=jnp.int32), config.context_capacity - 1
    )
    rope = jnp.take(main_rope_table, positions, axis=0, mode="clip")
    embedded = ws32_prefill_embedding_mapped(
        token_ids,
        weights.embedding_local,
        valid_rows,
        vocab_size=config.geometry.vocab_size,
    )
    update = embedded.residual_local
    residual = jnp.zeros_like(update)
    health = embedded.contract_valid & span_valid
    kv = decoder.kv_cache_local
    unrepaired = decoder.index_cache_local
    repaired = state.repaired_index_local
    pending_kv, pending_unrepaired, pending_repaired = [], [], []
    if pending_cache_rows:
        addresses = prefill_pending_addresses(
            decoder.block_tables,
            offset,
            valid_rows,
            lax.axis_index("expert"),
            physical_pages=kv.shape[1],
            window_rows=rows,
            layout=StageLocalKvLayout(local_parallel_size=8),
        )
        health = health & addresses.valid
    selected = jnp.full((rows, config.geometry.dsa_top_k), -1, jnp.int32)
    counts = jnp.zeros((rows,), jnp.int32)
    scores = jnp.full(selected.shape, -jnp.inf, jnp.float32)
    sparse = SparseMlaConfig(segment_block=config.sparse_segment_block)
    layer_program = (
        ws32_prefill_layer_window_mapped
        if mlp_window
        else ws32_prefill_transformer_layer_mapped
    )
    window_options = (
        dict(rolled_prefix=rolled_prefix, expert_panels=expert_panels)
        if mlp_window
        else {}
    )
    for layer_id, layer in enumerate(weights.layers):
        slot = config.full_index_slot_by_layer[layer_id]
        # The correction applies only to dense MLPs (layers0..2 in GLM).
        # Do not change the MoE call, host stride or causal prefix schedule.
        layer_options = window_options
        if canonical_dense and layer.dense is not None:
            layer_options = {**window_options, "canonical_dense": True}
        # Shared layers never use/write this placeholder index buffer. Their
        # selections come from the actual preceding producer; KV is always OWN.
        source_slot = 0 if slot is None else slot
        with jax.named_scope(f"greenfield_ws32_batched_prefill/layer_{layer_id}"):
            result = layer_program(
                update,
                residual,
                kv[layer_id],
                unrepaired[source_slot],
                repaired[source_slot],
                selected,
                counts,
                scores,
                offset,
                valid_rows,
                decoder.block_tables,
                layer.qkv_a,
                layer.attention,
                layer.dsa,
                None if slot is None else materialized_wk[slot],
                layer.post_attention_norm_weight_local,
                layer.dense,
                layer.moe,
                health,
                main_rope_table_rows=rope,
                dsa_contract=config.dsa_contract,
                attention_contract=config.attention_contract,
                moe_contract=config.moe_contract,
                rms_norm_epsilon=config.rms_norm_epsilon,
                key_tile=key_tile,
                sparse_attention_config=sparse,
                sparse_attention_interpret=sparse_attention_interpret,
                linear_interpret=linear_interpret,
                paired_position_sort=paired_position_sort,
                sorted_local_merge=sorted_local_merge,
                **layer_options,
            )
        continuation = (
            result.output_local,
            result.carried_residual_local,
            result.selected_positions,
            result.selected_valid_counts,
            result.selected_scores,
            result.contract_valid,
        )
        if pending_cache_rows:
            # No later layer reads this layer's KV or this producer's index
            # slot. Shared layers consume selected metadata, not these buffers.
            captured = (
                capture_prefill_pending_rows(result.cache_local, addresses.targets),
            )
            if slot is not None:
                if slot != len(pending_unrepaired):
                    raise PlanValidationError("pending producer slots must be ordered")
                captured += (
                    capture_prefill_pending_rows(
                        result.unrepaired_index_cache, addresses.targets
                    ),
                    capture_prefill_pending_rows(
                        result.repaired_index_cache, addresses.targets
                    ),
                )
            if capture_barrier:
                # DB614 sank row extraction into the final healthy branch and
                # retained all78 full-layer proposals until commit. Thread ALL
                # continuation operands through the same barrier as the compact
                # rows so next-layer work cannot bypass capture. Never put a
                # full cache/weight into this tuple. This is a compiler ordering
                # boundary, not a host dispatch or a change to model arithmetic.
                # Actual optimized lifetimes/allocations must still prove fit.
                continuation, captured = lax.optimization_barrier(
                    (continuation, captured)
                )
            pending_kv.append(captured[0])
            if slot is not None:
                pending_unrepaired.append(captured[1])
                pending_repaired.append(captured[2])
        else:
            kv = kv.at[layer_id].set(result.cache_local)
            if slot is not None:
                unrepaired = unrepaired.at[slot].set(result.unrepaired_index_cache)
                repaired = repaired.at[slot].set(result.repaired_index_cache)
        update, residual, selected, counts, scores, layer_health = continuation
        health = health & layer_health

    last = jnp.maximum(count - 1, 0)
    # Branch only on replicated scheduling metadata, NOT potentially differing
    # per-owner health: the head contains collectives and must execute uniformly.
    final = (end == state.prompt_length) & ~state.finished

    def sample(_: Any) -> tuple[Any, Any]:
        sample_head = ws32_split_final_sample_mapped if final_sample is None else final_sample
        head = sample_head(
            lax.dynamic_slice_in_dim(update, last, 1),
            lax.dynamic_slice_in_dim(residual, last, 1),
            weights.final_norm_weight_local,
            weights.lm_head_local,
            hidden_size=config.geometry.hidden_size,
            vocab_size=config.geometry.vocab_size,
            rms_norm_epsilon=config.rms_norm_epsilon,
        )
        return head.token_id, jnp.all(head.contract_valid)

    # final is replicated metadata; no head compute for intermediate blocks.
    token, head_health = lax.cond(
        final,
        sample,
        lambda _: (jnp.full((1,), -1, jnp.int32), jnp.bool_(True)),
        operand=None,
    )
    normalized_hidden, hidden_health = export_hidden_mapped(
        update, residual, weights.final_norm_weight_local,
        hidden_size=config.geometry.hidden_size, epsilon=config.rms_norm_epsilon,
    )
    healthy = _all_owners_healthy(
        span_valid & jnp.all(~live | health) & head_health
        & jnp.all(~live | hidden_health)
    )

    def commit(_: Any) -> Ws32BatchedPrefillState:
        next_kv, next_unrepaired, next_repaired = kv, unrepaired, repaired
        if pending_cache_rows:
            apply_rows = apply_prefill_flat_rows if flat_pending_rows else apply_prefill_pending_rows
            next_kv = apply_rows(
                kv, addresses.targets, jnp.stack(pending_kv)
            )
            next_unrepaired = apply_rows(
                unrepaired, addresses.targets, jnp.stack(pending_unrepaired)
            )
            next_repaired = apply_rows(
                repaired, addresses.targets, jnp.stack(pending_repaired)
            )
        active_index = lax.cond(
            final, lambda _: next_repaired, lambda _: next_unrepaired, operand=None
        )
        next_decoder = Ws32DecoderState(
            next_kv,
            active_index,
            lax.dynamic_slice_in_dim(selected, last, 1),
            lax.dynamic_slice_in_dim(counts, last, 1),
            lax.dynamic_slice_in_dim(scores, last, 1),
            end[None],
            decoder.block_tables,
            (end + 1)[None],
            healthy[None],
        )
        return Ws32BatchedPrefillState(
            next_decoder, next_repaired, state.prompt_length, final
        )

    def refuse(_: Any) -> Ws32BatchedPrefillState:
        return state._replace(
            decoder=decoder._replace(contract_valid=jnp.zeros((1,), jnp.bool_))
        )

    return MtpPrefillResult(
        lax.cond(healthy, commit, refuse, operand=None),
        jnp.where(healthy & final, token, -1),
        jnp.where((healthy & live)[:, None], normalized_hidden, jnp.bfloat16(0)),
        healthy & live,
    )


def _result_specs(state, token):
    return MtpPrefillResult(state, token, P(None, "feature"), P())


def build_ws32_batched_prefill_program(mesh, config, **options):
    """Reuse frozen builder validation/shardings with the extended result tree.

    The challenger binds its private runtime and resident weight specs into
    this function before calling it. No frozen module global is modified.
    """
    builder = bind_dependencies(
        frozen.build_ws32_batched_prefill_program,
        ws32_batched_prefill_mapped=ws32_batched_prefill_mapped,
        ws32_decoder_weight_specs=ws32_decoder_weight_specs,
        Ws32BatchedPrefillResult=_result_specs,
    )
    return builder(mesh, config, **options)
