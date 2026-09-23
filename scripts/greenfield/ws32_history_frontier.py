"""Unwired layers0..6 diagnostic using the existing batched layer kernels.

This is NOT a decoder or a hardware-admitted experiment. It retains separate
update/residual observations and full history, without a head or layers7..77.
Extra outputs can perturb compilation: original step0 event0..3 reproduction
is mandatory before attributing the saved 8K failure to these observations.
"""

from __future__ import annotations

from typing import Any, NamedTuple


LAYERS = tuple(range(7))
PRODUCERS = (0, 1, 2, 6)


class HistoryCaches(NamedTuple):
    kv: tuple[Any, ...]
    unrepaired: tuple[Any, ...]
    repaired: tuple[Any, ...]


class LayerBoundary(NamedTuple):
    update: Any
    residual: Any
    normalized_input: Any
    route_ids: Any
    route_weights: Any
    health: Any


class ProducerRows(NamedTuple):
    positions: Any
    counts: Any
    scores: Any


class HistoryFrontier(NamedTuple):
    caches: HistoryCaches
    boundaries: tuple[LayerBoundary, ...]
    producers: tuple[ProducerRows, ...]
    healthy: Any


def build_program(
    mesh: Any, config: Any, *, block_rows: int, canonical_dense: bool,
    interpret: bool = False,
) -> Any:
    """Build one B128/B114 prefix graph; no launch, load or execution authority.

Global caches are explicit [expert8,pages,64,width] owner arrays. Shared
indexers carry selected state but never acquire an index-cache or WK slot.
The host supplies the original live span; both histories must be completed
before repaired keys are handed to a separate exact-decode observer.
"""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.kernels.pallas.sparse_attention import SparseMlaConfig
    from glm_tpu.greenfield.kernels.ws32_prefill_window import ws32_prefill_layer_window_mapped
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import (
        _require_config, ws32_prefill_embedding_mapped,
    )
    from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_weight_specs

    _require_config(config)
    if (type(block_rows) is not int or block_rows not in (114, 128)
            or type(canonical_dense) is not bool or type(interpret) is not bool
            or config.geometry.num_layers < 7
            or config.geometry.indexer_types[:7] != (
                "full", "full", "full", "shared", "shared", "shared", "full")
            or config.geometry.mlp_layer_types[:7] != ("dense",) * 3 + ("sparse",) * 4):
        raise ValueError("history frontier requires original layers0..6 and B128/B114")
    if tuple(mesh.axis_names) != ("expert", "feature") or np.shape(mesh.devices) != (8, 4):
        raise ValueError("history frontier requires expert8 x feature4")
    weights = ws32_decoder_weight_specs(config)
    cache_spec = P("expert", None, None, None)
    cache_specs = HistoryCaches((cache_spec,) * 7, (cache_spec,) * 4, (cache_spec,) * 4)
    boundary_spec = LayerBoundary(
        P(None, "feature"), P(None, "feature"), P(None, "feature"),
        P(), P(), P("expert", "feature", None),
    )

    def execute(tokens: Any, count: Any, offset: Any, table: Any,
                caches: HistoryCaches, embedding: Any, layers: Any,
                wk: Any, rope: Any, healthy: Any) -> HistoryFrontier:
        if (tokens.shape != (block_rows,) or tokens.dtype != jnp.int32
                or count.shape != () or count.dtype != jnp.int32
                or offset.shape != () or offset.dtype != jnp.int32
                or healthy.shape != () or healthy.dtype != jnp.bool_
                or table.shape != (1, config.page_count) or table.dtype != jnp.int32
                or rope.shape != config.main_rope_table_shape or rope.dtype != jnp.bfloat16
                or len(layers) != 7 or len(wk) != 4
                or tuple(map(len, caches)) != (7, 4, 4)):
            raise ValueError("history frontier input geometry differs")
        for family, width in zip(caches, (config.packed_cache_width,
                config.geometry.dsa_indexer_head_dim,
                config.geometry.dsa_indexer_head_dim), strict=True):
            for value in family:
                if (value.shape != (1, config.page_count, 64, width)
                        or value.dtype != jnp.bfloat16):
                    raise ValueError("history frontier owner cache differs")
        for key in wk:
            if (key.shape != (config.geometry.dsa_indexer_head_dim, config.geometry.hidden_size)
                    or key.dtype != jnp.float32):
                raise ValueError("history frontier completed WK differs")
        for i, layer in enumerate(layers):
            if ((layer.dsa is not None) != (i in PRODUCERS)
                    or (layer.dense is not None) != (i < 3)
                    or (layer.moe is not None) != (i >= 3)):
                raise ValueError("history frontier weight branch differs")

        # Subtraction avoids accepting int32 offset+count overflow.
        span = ((offset >= 0) & (offset < config.context_capacity)
                & (count > 0) & (count <= block_rows)
                & (count <= config.context_capacity - jnp.clip(offset, 0, config.context_capacity)))
        start = jnp.clip(offset, 0, config.context_capacity - 1)
        positions = jnp.minimum(start + jnp.arange(block_rows, dtype=jnp.int32),
                                config.context_capacity - 1)
        rope_rows = jnp.take(rope, positions, axis=0, mode="clip")
        embedded = ws32_prefill_embedding_mapped(tokens, embedding, count,
                                                vocab_size=config.geometry.vocab_size)
        update, residual = embedded.residual_local, jnp.zeros_like(embedded.residual_local)
        health = embedded.contract_valid & span & healthy
        selected = jnp.full((block_rows, config.geometry.dsa_top_k), -1, jnp.int32)
        counts = jnp.zeros(block_rows, jnp.int32)
        scores = jnp.full(selected.shape, -jnp.inf, jnp.float32)
        kv, unrepaired, repaired = map(list, caches)
        boundaries, producers = [], []
        for i, layer in enumerate(layers):
            slot = config.full_index_slot_by_layer[i]
            source_slot = 0 if slot is None else slot
            with jax.named_scope(f"greenfield_ws32_batched_prefill/layer_{i}"):
                result = ws32_prefill_layer_window_mapped(
                    update, residual, kv[i][0], unrepaired[source_slot][0],
                    repaired[source_slot][0], selected, counts, scores,
                    offset, count, table, layer.qkv_a, layer.attention, layer.dsa,
                    None if slot is None else wk[slot],
                    layer.post_attention_norm_weight_local, layer.dense, layer.moe, health,
                    main_rope_table_rows=rope_rows, dsa_contract=config.dsa_contract,
                    attention_contract=config.attention_contract, moe_contract=config.moe_contract,
                    rms_norm_epsilon=config.rms_norm_epsilon, key_tile=512,
                    sparse_attention_config=SparseMlaConfig(segment_block=config.sparse_segment_block),
                    sparse_attention_interpret=interpret, linear_interpret=interpret,
                    paired_position_sort=True, sorted_local_merge=True,
                    rolled_prefix=True, expert_panels=True,
                    canonical_dense=canonical_dense and layer.dense is not None,
                )
            kv[i] = result.cache_local[None]
            if slot is not None:
                unrepaired[slot] = result.unrepaired_index_cache[None]
                repaired[slot] = result.repaired_index_cache[None]
                producers.append(ProducerRows(result.selected_positions,
                                               result.selected_valid_counts, result.selected_scores))
            update, residual = result.output_local, result.carried_residual_local
            selected, counts, scores = result.selected_positions, result.selected_valid_counts, result.selected_scores
            health = result.contract_valid
            boundaries.append(LayerBoundary(update, residual, result.normalized_input_local,
                result.route_indices, result.route_weights, health[None, None]))
        live = jnp.arange(block_rows, dtype=jnp.int32) < count
        accepted = jax.lax.pmin(jax.lax.pmin(
            (span & healthy & jnp.all(~live | health)).astype(jnp.int32), "feature"), "expert") == 1
        proposed = HistoryCaches(tuple(kv), tuple(unrepaired), tuple(repaired))
        committed = jax.tree.map(lambda new, old: jnp.where(accepted, new, old), proposed, caches)
        return HistoryFrontier(committed, tuple(boundaries), tuple(producers), accepted)

    return jax.jit(jax.shard_map(execute, mesh=mesh,
        in_specs=(P(), P(), P(), P(), cache_specs, weights.embedding_local,
                  weights.layers[:7], (P(),) * 4, P(), P()),
        out_specs=HistoryFrontier(cache_specs, (boundary_spec,) * 7,
                                  (ProducerRows(P(), P(), P()),) * 4, P()),
        check_vma=False))
