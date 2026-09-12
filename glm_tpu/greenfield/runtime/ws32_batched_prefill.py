"""Default-off layer-major prompt execution; the promoted decoder is unchanged.

One invocation embeds a live block once and visits each layer once. There is no
token scan of the decoder. A distinct state type owns unrepaired/repaired index
lifetimes and a monotone append frontier. Only the final block runs the head and
promotes repaired keys. This is an unpromoted numerical path, not §21 evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import PartitionSpec as P

from ..errors import PlanValidationError
from ..kernels.pallas import SparseMlaConfig
from ..kernels.prefill_pending_rows import (
    apply_prefill_pending_rows,
    capture_prefill_pending_rows,
    prefill_pending_addresses,
)
from ..kernels.reference.attention import StageLocalKvLayout
from ..kernels.ws32_io import (
    Ws32EmbeddingResult,
    _require_vocabulary_geometry,
    ws32_split_final_sample_mapped,
)
from ..kernels.ws32_prefill_layer import ws32_prefill_transformer_layer_mapped
from ..kernels.ws32_prefill_window import ws32_prefill_layer_window_mapped
from .ws32_decoder import (
    Ws32DecoderConfig,
    Ws32DecoderState,
    Ws32DecoderWeights,
    _validate_local_state,
    make_ws32_initial_state,
    make_ws32_repaired_index_buffer,
    ws32_decoder_state_specs,
    ws32_decoder_weight_specs,
)


class Ws32BatchedPrefillState(NamedTuple):
    decoder: Ws32DecoderState
    repaired_index_local: Any
    prompt_length: Any
    finished: Any


class Ws32BatchedPrefillResult(NamedTuple):
    state: Ws32BatchedPrefillState
    # -1 until a healthy final block; never a padded row's token.
    next_token: Any


@dataclass(frozen=True, slots=True)
class Ws32BatchedPrefillProgram:
    config: Ws32DecoderConfig
    block_rows: int
    execute: Any
    mlp_window: bool = False
    paired_position_sort: bool = False
    rolled_prefix: bool = False
    expert_panels: bool = False
    sorted_local_merge: bool = False
    canonical_dense: bool = False
    pending_cache_rows: bool = False


def ws32_batched_prefill_state_specs() -> Ws32BatchedPrefillState:
    return Ws32BatchedPrefillState(
        ws32_decoder_state_specs(), P(None, None, "expert", None), P(), P()
    )


def make_ws32_batched_prefill_state(
    mesh: Any, config: Ws32DecoderConfig, *, prompt_length: int
) -> Ws32BatchedPrefillState:
    """Fresh zero prefix with one immutable prompt-length/page-table identity.

    Resume must restore this WHOLE state from authenticated state evidence, not
    manufacture a populated prefix by changing position on a zero cache.
    """
    import numpy as np
    from jax.sharding import NamedSharding

    _require_config(config)
    if (
        not isinstance(prompt_length, int)
        or isinstance(prompt_length, bool)
        or not 0 < prompt_length < config.context_capacity
    ):
        raise PlanValidationError(
            "batched prompt must be positive and leave decode room"
        )
    replicated = NamedSharding(mesh, P())
    return Ws32BatchedPrefillState(
        make_ws32_initial_state(mesh, config),
        make_ws32_repaired_index_buffer(mesh, config),
        jax.make_array_from_callback(
            (), replicated, lambda _: np.asarray(prompt_length, np.int32)
        ),
        jax.make_array_from_callback(
            (), replicated, lambda _: np.asarray(False, np.bool_)
        ),
    )


def _require_config(config: Ws32DecoderConfig) -> None:
    # Reuse raw final-layout weight views, never silently consume the promoted
    # convolution aliases or StrategyND overlay as if they were the raw kernels.
    if config.exact_dsa or config.strategy_nd_dense:
        raise PlanValidationError(
            "batched prefill requires raw weight config, no exact aliases/StrategyND"
        )
    if not config.host_main_rope_table or config.logical_page_size != 512:
        raise PlanValidationError("batched prefill requires host main RoPE and page512")


def ws32_prefill_embedding_mapped(
    token_ids: Any, embedding_local: Any, valid_rows: Any, *, vocab_size: int
) -> Ws32EmbeddingResult:
    """One expert8 reduction for all live rows, with no vocabulary replication."""
    local_vocab, _ = _require_vocabulary_geometry(
        embedding_local, vocab_size=vocab_size
    )
    if (
        token_ids.ndim != 1
        or not 1 <= token_ids.shape[0] <= 128
        or token_ids.dtype != jnp.int32
    ):
        raise ValueError("batched embedding requires1..128 int32 token IDs")
    if valid_rows.shape != () or valid_rows.dtype != jnp.int32:
        raise ValueError("batched embedding live count must be int32 scalar")
    live = jnp.arange(token_ids.shape[0]) < jnp.clip(valid_rows, 0, token_ids.shape[0])
    token_valid = (token_ids >= 0) & (token_ids < vocab_size)
    start = lax.axis_index("expert").astype(jnp.int32) * jnp.int32(local_vocab)
    owns = live & token_valid & (token_ids >= start) & (token_ids < start + local_vocab)
    safe_tokens = jnp.clip(token_ids, 0, vocab_size - 1)
    local_ids = jnp.clip(safe_tokens - start, 0, local_vocab - 1)
    selected = jnp.where(owns[:, None], embedding_local[local_ids], 0)
    with jax.named_scope("greenfield_ws32_prefill_embedding/expert_owner_reduce"):
        hidden = lax.psum(selected, "expert")
    health = ~live | (token_valid & jnp.all(jnp.isfinite(hidden), axis=1))
    return Ws32EmbeddingResult(hidden, health)


def _all_owners_healthy(local: Any) -> Any:
    """One scalar consensus per block, explicit feature4 then expert8 groups."""
    with jax.named_scope("greenfield_ws32_prefill_commit/health_consensus"):
        return lax.pmin(lax.pmin(local.astype(jnp.int32), "feature"), "expert") != 0


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
) -> Ws32BatchedPrefillResult:
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
        update, residual = result.output_local, result.carried_residual_local
        if pending_cache_rows:
            # No later layer reads this layer's KV or this producer's index
            # slot. Shared layers consume selected metadata, not these buffers.
            pending_kv.append(
                capture_prefill_pending_rows(result.cache_local, addresses.targets)
            )
            if slot is not None:
                if slot != len(pending_unrepaired):
                    raise PlanValidationError("pending producer slots must be ordered")
                pending_unrepaired.append(
                    capture_prefill_pending_rows(
                        result.unrepaired_index_cache, addresses.targets
                    )
                )
                pending_repaired.append(
                    capture_prefill_pending_rows(
                        result.repaired_index_cache, addresses.targets
                    )
                )
        else:
            kv = kv.at[layer_id].set(result.cache_local)
            if slot is not None:
                unrepaired = unrepaired.at[slot].set(result.unrepaired_index_cache)
                repaired = repaired.at[slot].set(result.repaired_index_cache)
        selected, counts, scores = (
            result.selected_positions,
            result.selected_valid_counts,
            result.selected_scores,
        )
        health = health & result.contract_valid

    last = jnp.maximum(count - 1, 0)
    # Branch only on replicated scheduling metadata, NOT potentially differing
    # per-owner health: the head contains collectives and must execute uniformly.
    final = (end == state.prompt_length) & ~state.finished

    def sample(_: Any) -> tuple[Any, Any]:
        head = ws32_split_final_sample_mapped(
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
    healthy = _all_owners_healthy(span_valid & jnp.all(~live | health) & head_health)

    def commit(_: Any) -> Ws32BatchedPrefillState:
        next_kv, next_unrepaired, next_repaired = kv, unrepaired, repaired
        if pending_cache_rows:
            next_kv = apply_prefill_pending_rows(
                kv, addresses.targets, jnp.stack(pending_kv)
            )
            next_unrepaired = apply_prefill_pending_rows(
                unrepaired, addresses.targets, jnp.stack(pending_unrepaired)
            )
            next_repaired = apply_prefill_pending_rows(
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

    return Ws32BatchedPrefillResult(
        lax.cond(healthy, commit, refuse, operand=None),
        jnp.where(healthy & final, token, -1),
    )


def finish_ws32_batched_prefill(
    result: Ws32BatchedPrefillResult,
) -> tuple[Ws32DecoderState, Any]:
    """One host boundary before serving: refuse incomplete/unhealthy prefill.

    Device-side all-owner consensus has already gated the atomic final commit.
    The returned decoder state owns repaired indices and retains one-row shapes;
    the next decode input is the returned greedy token, not the last prompt ID.
    """
    import numpy as np

    state = result.state
    if (
        not bool(np.asarray(state.finished))
        or not np.asarray(state.decoder.contract_valid).all()
        or not np.asarray(state.decoder.position == state.prompt_length).all()
        or not np.asarray(
            state.decoder.context_lengths == state.prompt_length + 1
        ).all()
        or not np.asarray(result.next_token >= 0).all()
    ):
        raise ValueError("batched prefill is not complete and healthy; decode refused")
    return state.decoder, result.next_token


def _require_window_options(
    mlp_window: bool,
    rolled_prefix: bool,
    expert_panels: bool,
    sorted_local_merge: bool,
    canonical_dense: bool = False,
) -> None:
    if any(
        type(v) is not bool
        for v in (
            mlp_window,
            rolled_prefix,
            expert_panels,
            sorted_local_merge,
            canonical_dense,
        )
    ):
        raise PlanValidationError("prefill window options must be static booleans")
    if not mlp_window and (rolled_prefix or expert_panels or sorted_local_merge):
        raise PlanValidationError("new window components require explicit mlp_window")
    if canonical_dense and not (mlp_window and rolled_prefix and expert_panels):
        raise PlanValidationError("canonical dense requires rolled panel MLP window")


def build_ws32_batched_prefill_program(
    mesh: Any,
    config: Ws32DecoderConfig,
    *,
    block_rows: int,
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
) -> Ws32BatchedPrefillProgram:
    """Build raw prefill; <=128 MLP rows require explicit window opt-in."""
    import numpy as np

    _require_config(config)
    if type(pending_cache_rows) is not bool:
        raise PlanValidationError("pending cache rows must be a static bool")
    _require_window_options(
        mlp_window, rolled_prefix, expert_panels, sorted_local_merge, canonical_dense
    )
    if type(paired_position_sort) is not bool:
        raise PlanValidationError("paired position sort must be a static bool")
    if (
        isinstance(block_rows, bool)
        or not isinstance(block_rows, int)
        or not isinstance(mlp_window, bool)
        or not 1 <= block_rows <= (128 if mlp_window else 32)
    ):
        raise PlanValidationError("batched prefill block rows exceed selected mode")
    if canonical_dense and block_rows not in (114, 128):
        raise PlanValidationError("canonical dense requires physical B114/B128")
    if tuple(mesh.axis_names) != ("expert", "feature") or np.asarray(
        mesh.devices
    ).shape != (8, 4):
        raise PlanValidationError(
            "batched prefill requires exact expert8/feature4 mesh"
        )

    def body(
        tokens: Any,
        count: Any,
        state: Ws32BatchedPrefillState,
        weights: Ws32DecoderWeights,
        wk: tuple[Any, ...],
        rope: Any,
    ) -> Ws32BatchedPrefillResult:
        if tokens.shape != (block_rows,):
            raise ValueError("batched prefill static row count drifted")
        return ws32_batched_prefill_mapped(
            tokens,
            count,
            state,
            weights,
            wk,
            rope,
            config=config,
            key_tile=key_tile,
            sparse_attention_interpret=sparse_attention_interpret,
            linear_interpret=linear_interpret,
            mlp_window=mlp_window,
            paired_position_sort=paired_position_sort,
            rolled_prefix=rolled_prefix,
            expert_panels=expert_panels,
            sorted_local_merge=sorted_local_merge,
            canonical_dense=canonical_dense,
            pending_cache_rows=pending_cache_rows,
        )

    specs = ws32_batched_prefill_state_specs()
    execute = jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(
                P(),
                P(),
                specs,
                ws32_decoder_weight_specs(config),
                tuple(P() for _ in config.full_index_slots),
                P(),
            ),
            out_specs=Ws32BatchedPrefillResult(specs, P()),
            check_vma=False,
        )
    )
    return Ws32BatchedPrefillProgram(
        config,
        block_rows,
        execute,
        mlp_window,
        paired_position_sort,
        rolled_prefix,
        expert_panels,
        sorted_local_merge,
        canonical_dense,
        pending_cache_rows,
    )
