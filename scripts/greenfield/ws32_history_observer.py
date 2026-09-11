"""Unwired exact-decode prefix observer; no head, generation or serving path.

Calls the original decoder's embedding and transformer layer boundary, including
its exact-DSA operands and StrategyND dense overlay. Extra outputs/reduced depth
can change compiler realization; saved first-event reproduction is still required.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, NamedTuple

from scripts.greenfield.ws32_history_frontier import LayerBoundary, PRODUCERS


class FirstObservation(NamedTuple):
    boundaries: tuple[LayerBoundary, ...]
    dsa: Any
    healthy: Any


def observer_config(raw_config: Any, *, strategy_nd_dense: bool = True) -> Any:
    """Retain all numerical geometry but bound the original schedule to0..6.

    The original first-decode step ran exact DSA with the StrategyND dense
    overlay, so that is the default. Plain dense weights are admitted ONLY so
    loop parity against the production observer is provable on CPU: the
    StrategyND kernel accepts production shapes alone. A guarded executor must
    refuse ``strategy_nd_dense=False`` for the actual diagnostic.
    """
    if type(strategy_nd_dense) is not bool:
        raise ValueError("history observer dense representation must be boolean")
    if (raw_config.exact_dsa or raw_config.strategy_nd_dense
            or not raw_config.host_main_rope_table
            or raw_config.context_capacity != 8192
            or raw_config.geometry.num_layers < 7
            or raw_config.geometry.indexer_types[:7] != (
                "full", "full", "full", "shared", "shared", "shared", "full")
            or raw_config.geometry.mlp_layer_types[:7] != ("dense",) * 3 + ("sparse",) * 4):
        raise ValueError("history observer requires original raw8K preparation")
    if strategy_nd_dense and (raw_config.logical_page_size != 512
                              or raw_config.sparse_segment_block != 512
                              or raw_config.rms_norm_epsilon != 1e-5
                              or raw_config.packed_cache_width != 640):
        # The original run used the production defaults; a page, segment,
        # epsilon or cache-width change would observe a different model.
        raise ValueError("history observer requires the original8K numerical geometry")
    geometry = replace(raw_config.geometry, num_layers=7,
        mlp_layer_types=raw_config.geometry.mlp_layer_types[:7],
        indexer_types=raw_config.geometry.indexer_types[:7])
    return replace(raw_config, geometry=geometry, exact_dsa=True,
                   strategy_nd_dense=strategy_nd_dense)


def build_program(mesh: Any, config: Any, *, sparse_attention_interpret: bool = False,
                  linear_interpret: bool = False) -> Any:
    """Observe token220 at8155 using completed REPAIRED history on each branch.

    All input placements and completion certificates belong to the guarded host
    executor. A healthy call here does not prove that the host completed8155 rows.
    KV/index inputs are immutable and no resumed state or sampled token is returned.
    The interpret flags exist for CPU parity tests exactly as on the production
    builder; the StrategyND dense path refuses interpretation, so a CPU test
    runs plain dense weights. Hardware keeps both False.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.pallas import SparseMlaConfig
    from glm_tpu.greenfield.kernels.ws32_io import ws32_embedding_mapped
    from glm_tpu.greenfield.kernels.ws32_layer import (
        Ws32DenseWeights, Ws32StrategyNdDenseWeights, ws32_transformer_layer_mapped,
    )
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        Ws32DsaObservation, ws32_decoder_weight_specs, ws32_dsa_observation_specs,
        ws32_exact_dsa_specs,
    )

    if (type(sparse_attention_interpret) is not bool or type(linear_interpret) is not bool
            or not config.exact_dsa or type(config.strategy_nd_dense) is not bool
            or not config.host_main_rope_table or config.context_capacity != 8192
            or config.geometry.num_layers != 7 or config.full_index_slots != PRODUCERS
            or tuple(mesh.axis_names) != ("expert", "feature") or np.shape(mesh.devices) != (8, 4)):
        raise ValueError("history observer requires exact7-layer8K expert8/feature4")
    if not config.strategy_nd_dense and not (sparse_attention_interpret and linear_interpret):
        # Plain dense exists for interpreted CPU parity only; a compilable
        # plain-dense observer would observe a model the failed run never ran.
        raise ValueError("history observer admits plain dense weights only under interpretation")
    dense_kind = Ws32StrategyNdDenseWeights if config.strategy_nd_dense else Ws32DenseWeights
    specs = ws32_decoder_weight_specs(config)
    cache_spec = P("expert", None, None, None)
    boundary_spec = LayerBoundary(P(None, "feature"), P(None, "feature"),
        P(None, "feature"), P(), P(), P("expert", "feature", None))

    def observe(token: Any, position: Any, lengths: Any, table: Any,
                kv: Any, repaired: Any, embedding: Any, layers: Any,
                exact: Any, rope: Any, healthy: Any) -> FirstObservation:
        if (token.shape != (1,) or token.dtype != jnp.int32
                or position.shape != (1,) or position.dtype != jnp.int32
                or lengths.shape != (1,) or lengths.dtype != jnp.int32
                or table.shape != (1, config.page_count) or table.dtype != jnp.int32
                or healthy.shape != () or healthy.dtype != jnp.bool_
                or rope.shape != config.main_rope_table_shape or rope.dtype != jnp.bfloat16
                or len(kv) != 7 or len(repaired) != 4 or len(layers) != 7 or len(exact) != 4):
            raise ValueError("history observer input geometry differs")
        for family, width in ((kv, config.packed_cache_width),
                              (repaired, config.geometry.dsa_indexer_head_dim)):
            for value in family:
                if value.shape != (1, config.page_count, 64, width) or value.dtype != jnp.bfloat16:
                    raise ValueError("history observer owner cache differs")
        for i, layer in enumerate(layers):
            if ((layer.dsa is not None) != (i in PRODUCERS)
                    or (layer.moe is not None) != (i >= 3)
                    or (not isinstance(layer.dense, dense_kind) if i < 3
                        else layer.dense is not None)):
                raise ValueError("history observer lost original dense/indexer branch")

        # Keep the original observer's dynamic position lookup and one-row shape.
        rope_row = jnp.take(rope, position, axis=0, mode="clip")[0]
        embedded = ws32_embedding_mapped(token, embedding, vocab_size=config.geometry.vocab_size)
        update, residual = embedded.residual_local, jnp.zeros_like(embedded.residual_local)
        valid = healthy & jnp.all((token == 220) & (position == 8155) & (lengths == 8156))
        health = embedded.contract_valid & valid
        positions = jnp.full((1, config.geometry.dsa_top_k), -1, jnp.int32)
        counts = jnp.zeros((1,), jnp.int32)
        scores = jnp.full(positions.shape, -jnp.inf, jnp.float32)
        index_cache = list(repaired)
        boundaries, observed_positions, observed_counts, observed_scores = [], [], [], []
        with jax.named_scope("greenfield_ws32_complete_decoder_dsa_observer"):
            for i, layer in enumerate(layers):
                slot = config.full_index_slot_by_layer[i]
                result = ws32_transformer_layer_mapped(
                    update, residual, kv[i][0], index_cache[0 if slot is None else slot][0],
                    positions, counts, scores, position, table, lengths,
                    layer.qkv_a, layer.attention, layer.dsa,
                    layer.post_attention_norm_weight_local, layer.dense, layer.moe, health,
                    exact_dsa_weights=None if slot is None else exact[slot],
                    main_rope_table_row=rope_row,
                    indexer_kind=config.geometry.indexer_types[i],
                    mlp_kind=config.geometry.mlp_layer_types[i],
                    dsa_contract=config.dsa_contract, attention_contract=config.attention_contract,
                    moe_contract=config.moe_contract, cache_layout=config.cache_layout,
                    block_shape=config.geometry.fp8_block_shape,
                    rms_norm_epsilon=config.rms_norm_epsilon,
                    sparse_attention_config=SparseMlaConfig(segment_block=config.sparse_segment_block),
                    sparse_attention_interpret=sparse_attention_interpret,
                    linear_interpret=linear_interpret,
                )
                update, residual = result.output_local, result.carried_residual_local
                positions, counts, scores = result.selected_positions, result.selected_valid_counts, result.selected_scores
                health = result.contract_valid
                boundaries.append(LayerBoundary(update, residual, result.normalized_input_local,
                    result.route_indices, result.route_weights, health[None, None]))
                if slot is not None:
                    index_cache[slot] = result.index_cache_local[None]
                    observed_positions.append(positions)
                    observed_counts.append(counts)
                    observed_scores.append(scores)
        accepted = jax.lax.pmin(jax.lax.pmin(
            (valid & jnp.all(health)).astype(jnp.int32), "feature"), "expert") == 1

        def masked(value: Any, sentinel: Any) -> Any:
            # A refused step must not look like an observation: the kernels
            # still compute under a false health flag, so every output is
            # replaced by a sentinel unless the whole fleet accepted the step.
            return jnp.where(accepted, value, jnp.full_like(value, sentinel))

        boundaries = tuple(LayerBoundary(
            masked(b.update, jnp.nan), masked(b.residual, jnp.nan),
            masked(b.normalized_input, jnp.nan), masked(b.route_ids, -1),
            masked(b.route_weights, jnp.nan), b.health & accepted) for b in boundaries)
        dsa = Ws32DsaObservation(jnp.asarray(PRODUCERS, jnp.int32),
            masked(jnp.stack(observed_positions), -1), masked(jnp.stack(observed_counts), 0),
            masked(jnp.stack(observed_scores), -jnp.inf))
        return FirstObservation(boundaries, dsa, accepted)

    return jax.jit(jax.shard_map(observe, mesh=mesh,
        in_specs=(P(), P(), P(), P(), (cache_spec,) * 7, (cache_spec,) * 4,
                  specs.embedding_local, specs.layers, ws32_exact_dsa_specs(config), P(), P()),
        out_specs=FirstObservation((boundary_spec,) * 7, ws32_dsa_observation_specs(), P()),
        check_vma=False))
