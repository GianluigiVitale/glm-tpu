"""Diagnostic-only dense0/1 prefix; never a decoder or a new accepted baseline.

Reduced execution and extra outputs can change compiler realization. Every
layer0/1 cache byte must reproduce each branch's DB604 witness before causality.
No head, generation, repaired-cache promotion, atomic commit or serving claims.
"""

from __future__ import annotations

from typing import Any


def build_program(mesh: Any, config: Any, *, interpret: bool = False,
                  canonical_dense: bool = False) -> Any:
    """Same physical B128 for one128-live and four32-live diagnostic calls.

    Inputs are replicated tokens/count/offset/table/RoPE, six owner-sharded
    caches, original embedding and two original layer-weight trees plus two
    completed FP32 WKs. Cache shape is [8,pages,64,width], as in layer programs.
    All operations reuse current native kernels; no model-sized dummy head.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.pallas import SparseMlaConfig
    from glm_tpu.greenfield.kernels.ws32_prefill_window import ws32_prefill_layer_window_mapped
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import (
        _require_config, ws32_prefill_embedding_mapped,
    )
    from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_weight_specs

    _require_config(config)
    if (type(interpret) is not bool or type(canonical_dense) is not bool
            or config.geometry.num_layers < 2
            or config.geometry.indexer_types[:2] != ("full", "full")
            or config.geometry.mlp_layer_types[:2] != ("dense", "dense")):
        raise ValueError("dense frontier requires original dense/full-index layers0/1")
    specs = ws32_decoder_weight_specs(config)
    cache_spec = P("expert", None, None, None)
    output_specs = (
        P(None, "feature"), P(None, "feature"), cache_spec, cache_spec,
        cache_spec, P(), P(), P(), P(), P(), P("expert", "feature", None),
        P(None, "feature"),
    )

    def execute(tokens: Any, count: Any, offset: Any, table: Any, caches: Any,
                embedding: Any, layers: Any, wk: Any, rope: Any) -> tuple:
        if (tokens.shape != (128,) or tokens.dtype != jnp.int32
                or count.shape != () or count.dtype != jnp.int32
                or offset.shape != () or offset.dtype != jnp.int32
                or table.shape != (1, config.page_count) or table.dtype != jnp.int32
                or rope.shape != config.main_rope_table_shape or rope.dtype != jnp.bfloat16
                or len(caches) != 2 or len(layers) != 2 or len(wk) != 2):
            raise ValueError("dense frontier input geometry differs")
        for family in caches:
            if len(family) != 3:
                raise ValueError("dense frontier needs three independent cache families")
            for value, width in zip(family, (config.packed_cache_width,
                    config.geometry.dsa_indexer_head_dim,
                    config.geometry.dsa_indexer_head_dim), strict=True):
                if value.shape != (1, config.page_count, 64, width) or value.dtype != jnp.bfloat16:
                    raise ValueError("dense frontier owner-cache shape differs")
        for layer, key in zip(layers, wk, strict=True):
            if (layer.dsa is None or layer.dense is None or layer.moe is not None
                    or key.shape != (config.geometry.dsa_indexer_head_dim, config.geometry.hidden_size)
                    or key.dtype != jnp.float32):
                raise ValueError("dense frontier layer/WK branch differs")
        # Fixed diagnostic spans only. Invalid caller metadata latches health;
        # outputs remain proposals and must never be used as a resumed state.
        span = ((count == 128) & (offset == 0)) | (
            (count == 32) & ((offset == 0) | (offset == 32) | (offset == 64) | (offset == 96))
        )
        start = jnp.clip(offset, 0, config.context_capacity - 1)
        positions = jnp.minimum(start + jnp.arange(128, dtype=jnp.int32), config.context_capacity - 1)
        hrope = jnp.take(rope, positions, axis=0, mode="clip")
        embedded = ws32_prefill_embedding_mapped(tokens, embedding, count,
                                                vocab_size=config.geometry.vocab_size)
        update, residual = embedded.residual_local, jnp.zeros_like(embedded.residual_local)
        health = embedded.contract_valid & span
        selected = jnp.full((128, config.geometry.dsa_top_k), -1, jnp.int32)
        counts = jnp.zeros(128, jnp.int32)
        scores = jnp.full(selected.shape, -jnp.inf, jnp.float32)
        outputs = []
        for layer_id, (layer, cache, key) in enumerate(zip(layers, caches, wk, strict=True)):
            with jax.named_scope(f"greenfield_ws32_batched_prefill/layer_{layer_id}"):
                result = ws32_prefill_layer_window_mapped(
                    update, residual, *(c[0] for c in cache), selected, counts, scores,
                    offset, count, table, layer.qkv_a, layer.attention, layer.dsa, key,
                    layer.post_attention_norm_weight_local, layer.dense, None, health,
                    main_rope_table_rows=hrope, dsa_contract=config.dsa_contract,
                    attention_contract=config.attention_contract, moe_contract=config.moe_contract,
                    rms_norm_epsilon=config.rms_norm_epsilon, key_tile=512,
                    sparse_attention_config=SparseMlaConfig(segment_block=config.sparse_segment_block),
                    sparse_attention_interpret=interpret, linear_interpret=interpret,
                    paired_position_sort=True, sorted_local_merge=True,
                    rolled_prefix=True, expert_panels=True,
                    canonical_dense=canonical_dense,
                )
            outputs.append((result.output_local, result.carried_residual_local,
                result.cache_local[None], result.unrepaired_index_cache[None],
                result.repaired_index_cache[None], result.selected_positions,
                result.selected_valid_counts, result.selected_scores,
                result.route_indices, result.route_weights, result.contract_valid[None, None],
                result.normalized_input_local))
            update, residual = result.output_local, result.carried_residual_local
            selected, counts, scores = result.selected_positions, result.selected_valid_counts, result.selected_scores
            health = health & result.contract_valid
        return tuple(outputs)

    return jax.jit(jax.shard_map(execute, mesh=mesh,
        in_specs=(P(), P(), P(), P(), ((cache_spec,) * 3,) * 2,
                  specs.embedding_local, specs.layers[:2], (P(), P()), P()),
        out_specs=(output_specs, output_specs), check_vma=False))
