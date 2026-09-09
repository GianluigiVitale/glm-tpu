"""Diagnostic-only layer0 rolled observation adapter; no launch authority.

Reuse the actual transformer prefix and dense suffix, including their original
FP32/BF16 boundaries. This copies only the fixed four-tile scheduling adapter,
not model arithmetic. Extra device outputs may perturb compilation: all original
outputs and cache witnesses must reproduce before interpreting the packet.
"""

from __future__ import annotations

from typing import Any


def observed_window(*args: Any, **options: Any) -> tuple[Any, dict[str, Any]]:
    """Return original layer outputs plus owner-local post-norm observations.

    Only the fixed B128 diagnostic caller is supported. Trace-time dictionaries
    become scan auxiliary outputs, never host callbacks or cache history carries.
    Production window code and its observation-hook refusal remain unchanged.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from glm_tpu.greenfield.kernels.ws32_prefill_layer import (
        Ws32PrefillLayerResult,
        ws32_prefill_mlp_mapped,
        ws32_prefill_transformer_layer_mapped,
    )

    (
        update,
        residual,
        kv,
        index,
        repair,
        selected,
        counts,
        scores,
        offset,
        count,
        table,
        qkv,
        attention,
        dsa,
        wk,
        norm_weight,
        dense,
        moe,
        incoming,
    ) = args
    if (
        update.ndim != 2
        or update.shape[0] != 128
        or residual.shape != update.shape
        or incoming.shape != (128,)
        or offset.shape != ()
        or offset.dtype != jnp.int32
        or count.shape != ()
        or count.dtype != jnp.int32
        or dense is None
        or moe is not None
        or dsa is None
        or options.pop("rolled_prefix", None) is not True
        or options.pop("expert_panels", None) is not True
        or "_observe" in options
    ):
        raise ValueError(
            "norm capture requires fixed dense/full-index B128 rolled window"
        )
    rope = options.pop("main_rope_table_rows")
    capacity = kv.shape[0] * kv.shape[1] * 8
    start = jnp.clip(offset, 0, capacity - 1)
    live_count = jnp.clip(count, 0, 128)
    span_valid = (
        (offset == start)
        & (count >= 0)
        & (count <= 128)
        & (live_count <= capacity - start)
    )

    def tiles(value: Any) -> Any:
        return value.reshape((4, 32, *value.shape[1:]))

    inputs = (
        jnp.arange(4, dtype=jnp.int32),
        tiles(update),
        tiles(residual),
        tiles(selected),
        tiles(counts),
        tiles(scores),
        tiles(incoming),
        tiles(rope),
    )

    def body(caches: tuple[Any, ...], values: tuple[Any, ...]) -> tuple:
        tile, u, r, positions, valid_counts, selected_scores, health, hrope = values
        observations: dict[str, Any] = {}

        def observe(name: str, arrays: dict[str, Any]) -> None:
            if name == "post_norm":
                for field in (
                    "update",
                    "residual",
                    "summed",
                    "local_square_sum",
                    "square_sum",
                    "inverse",
                    "normalized",
                    "carried",
                ):
                    key = f"post_norm/{field}"
                    if key in observations:
                        raise ValueError("duplicate post-norm capture")
                    observations[key] = arrays[field]
            elif name == "attention_mlp_boundary":
                for field in ("normalized_mlp", "live"):
                    key = f"boundary/{field}"
                    if key in observations:
                        raise ValueError("duplicate MLP boundary capture")
                    observations[key] = arrays[field]

        first = tile * 32
        result = ws32_prefill_transformer_layer_mapped(
            u,
            r,
            *caches,
            positions,
            valid_counts,
            selected_scores,
            start + jnp.minimum(first, capacity - 1 - start),
            jnp.clip(live_count - first, 0, 32),
            table,
            qkv,
            attention,
            dsa,
            wk,
            norm_weight,
            dense,
            None,
            health & span_valid,
            main_rope_table_rows=hrope,
            prefix_only=True,
            _observe=observe,
            **options,
        )
        if len(observations) != 10:
            raise ValueError("required executing post-norm boundary not observed")
        return (
            result.cache_local,
            result.unrepaired_index_cache,
            result.repaired_index_cache,
        ), (
            result.normalized_mlp_local,
            result.carried_residual_local,
            result.selected_positions,
            result.selected_valid_counts,
            result.selected_scores,
            result.contract_valid,
            result.normalized_input_local,
            observations,
        )

    with jax.named_scope("greenfield_ws32_prefill_rolled_prefix"):
        caches, stacked = lax.scan(body, (kv, index, repair), inputs, unroll=1)
    flat = tuple(v.reshape((128, *v.shape[2:])) for v in stacked[:7])
    packet = {
        key: value.reshape((128, *value.shape[2:])) for key, value in stacked[7].items()
    }
    # The actual weight operand is loop invariant: return it once, not four copies.
    packet["post_norm/weight"] = norm_weight
    live = jnp.arange(128, dtype=jnp.int32) < live_count
    output, ids, weights, mlp_health = ws32_prefill_mlp_mapped(
        flat[0],
        live,
        dense,
        None,
        moe_contract=options["moe_contract"],
        linear_interpret=options.get("linear_interpret", False),
        expert_panels=True,
    )
    output = jnp.where(live[:, None], output, 0)
    health = flat[5] & mlp_health & (~live | jnp.all(jnp.isfinite(output), axis=1))
    result = Ws32PrefillLayerResult(
        output,
        flat[1],
        *caches,
        flat[2],
        flat[3],
        flat[4],
        ids,
        weights,
        health,
        flat[6],
    )
    return result, {key: value[None, None] for key, value in packet.items()}


def build_completed_dense_suffix(
    mesh: Any, config: Any, *, interpret: bool = False
) -> Any:
    """Replay captured BF16 normalized rows through the unchanged physicalB128 MLP.

    No normalization, row geometry change, checkpoint load or runtime initialization.
    Compare original outputs from their own inputs before an identical-input test.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_weight_specs
    from glm_tpu.greenfield.kernels.ws32_prefill_layer import ws32_prefill_mlp_mapped

    if type(interpret) is not bool:
        raise ValueError("interpret must be a static bool")

    def body(normalized: Any, live: Any, dense: Any) -> tuple[Any, Any]:
        if normalized.shape != (128, config.geometry.hidden_size // 4):
            raise ValueError("dense replay must retain physicalB128")
        output, _, _, health = ws32_prefill_mlp_mapped(
            normalized,
            live,
            dense,
            None,
            moe_contract=config.moe_contract,
            linear_interpret=interpret,
            expert_panels=True,
        )
        output = jnp.where(live[:, None], output, 0)
        health = health & (~live | jnp.all(jnp.isfinite(output), axis=1))
        return output, health[None, None]

    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(
                P(None, "feature"),
                P(),
                ws32_decoder_weight_specs(config).layers[0].dense,
            ),
            out_specs=(P(None, "feature"), P("expert", "feature", None)),
            check_vma=False,
        )
    )


def build_capture_program(mesh: Any, config: Any, *, interpret: bool = False) -> Any:
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
    from glm_tpu.greenfield.kernels.ws32_prefill_window import (
        ws32_prefill_layer_window_mapped,
    )
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import (
        _require_config,
        ws32_prefill_embedding_mapped,
    )
    from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_weight_specs

    capture_layer0 = True
    _require_config(config)
    if (
        type(interpret) is not bool
        or type(capture_layer0) is not bool
        or config.geometry.num_layers < 2
        or config.geometry.indexer_types[:2] != ("full", "full")
        or config.geometry.mlp_layer_types[:2] != ("dense", "dense")
    ):
        raise ValueError("dense frontier requires original dense/full-index layers0/1")
    specs = ws32_decoder_weight_specs(config)
    cache_spec = P("expert", None, None, None)
    output_specs = (
        P(None, "feature"),
        P(None, "feature"),
        cache_spec,
        cache_spec,
        cache_spec,
        P(),
        P(),
        P(),
        P(),
        P(),
        P("expert", "feature", None),
        P(None, "feature"),
    )

    def execute(
        tokens: Any,
        count: Any,
        offset: Any,
        table: Any,
        caches: Any,
        embedding: Any,
        layers: Any,
        wk: Any,
        rope: Any,
    ) -> tuple:
        if (
            tokens.shape != (128,)
            or tokens.dtype != jnp.int32
            or count.shape != ()
            or count.dtype != jnp.int32
            or offset.shape != ()
            or offset.dtype != jnp.int32
            or table.shape != (1, config.page_count)
            or table.dtype != jnp.int32
            or rope.shape != config.main_rope_table_shape
            or rope.dtype != jnp.bfloat16
            or len(caches) != 2
            or len(layers) != 2
            or len(wk) != 2
        ):
            raise ValueError("dense frontier input geometry differs")
        for family in caches:
            if len(family) != 3:
                raise ValueError(
                    "dense frontier needs three independent cache families"
                )
            for value, width in zip(
                family,
                (
                    config.packed_cache_width,
                    config.geometry.dsa_indexer_head_dim,
                    config.geometry.dsa_indexer_head_dim,
                ),
                strict=True,
            ):
                if (
                    value.shape != (1, config.page_count, 64, width)
                    or value.dtype != jnp.bfloat16
                ):
                    raise ValueError("dense frontier owner-cache shape differs")
        for layer, key in zip(layers, wk, strict=True):
            if (
                layer.dsa is None
                or layer.dense is None
                or layer.moe is not None
                or key.shape
                != (config.geometry.dsa_indexer_head_dim, config.geometry.hidden_size)
                or key.dtype != jnp.float32
            ):
                raise ValueError("dense frontier layer/WK branch differs")
        # Fixed diagnostic spans only. Invalid caller metadata latches health;
        # outputs remain proposals and must never be used as a resumed state.
        span = ((count == 128) & (offset == 0)) | (
            (count == 32)
            & ((offset == 0) | (offset == 32) | (offset == 64) | (offset == 96))
        )
        start = jnp.clip(offset, 0, config.context_capacity - 1)
        positions = jnp.minimum(
            start + jnp.arange(128, dtype=jnp.int32), config.context_capacity - 1
        )
        hrope = jnp.take(rope, positions, axis=0, mode="clip")
        embedded = ws32_prefill_embedding_mapped(
            tokens, embedding, count, vocab_size=config.geometry.vocab_size
        )
        update, residual = embedded.residual_local, jnp.zeros_like(
            embedded.residual_local
        )
        health = embedded.contract_valid & span
        selected = jnp.full((128, config.geometry.dsa_top_k), -1, jnp.int32)
        counts = jnp.zeros(128, jnp.int32)
        scores = jnp.full(selected.shape, -jnp.inf, jnp.float32)
        outputs, captures = [], {}
        for layer_id, (layer, cache, key) in enumerate(
            zip(layers, caches, wk, strict=True)
        ):
            window = ws32_prefill_layer_window_mapped
            if capture_layer0 and layer_id == 0:
                from scripts.greenfield.ws32_dense_norm_boundary import observed_window

                window = observed_window
            with jax.named_scope(f"greenfield_ws32_batched_prefill/layer_{layer_id}"):
                result = window(
                    update,
                    residual,
                    *(c[0] for c in cache),
                    selected,
                    counts,
                    scores,
                    offset,
                    count,
                    table,
                    layer.qkv_a,
                    layer.attention,
                    layer.dsa,
                    key,
                    layer.post_attention_norm_weight_local,
                    layer.dense,
                    None,
                    health,
                    main_rope_table_rows=hrope,
                    dsa_contract=config.dsa_contract,
                    attention_contract=config.attention_contract,
                    moe_contract=config.moe_contract,
                    rms_norm_epsilon=config.rms_norm_epsilon,
                    key_tile=512,
                    sparse_attention_config=SparseMlaConfig(
                        segment_block=config.sparse_segment_block
                    ),
                    sparse_attention_interpret=interpret,
                    linear_interpret=interpret,
                    paired_position_sort=True,
                    sorted_local_merge=True,
                    rolled_prefix=True,
                    expert_panels=True,
                )
            if capture_layer0 and layer_id == 0:
                result, captures = result
            outputs.append(
                (
                    result.output_local,
                    result.carried_residual_local,
                    result.cache_local[None],
                    result.unrepaired_index_cache[None],
                    result.repaired_index_cache[None],
                    result.selected_positions,
                    result.selected_valid_counts,
                    result.selected_scores,
                    result.route_indices,
                    result.route_weights,
                    result.contract_valid[None, None],
                    result.normalized_input_local,
                )
            )
            update, residual = result.output_local, result.carried_residual_local
            selected, counts, scores = (
                result.selected_positions,
                result.selected_valid_counts,
                result.selected_scores,
            )
            health = health & result.contract_valid
        return (tuple(outputs), captures) if capture_layer0 else tuple(outputs)

    return jax.jit(
        jax.shard_map(
            execute,
            mesh=mesh,
            in_specs=(
                P(),
                P(),
                P(),
                P(),
                ((cache_spec,) * 3,) * 2,
                specs.embedding_local,
                specs.layers[:2],
                (P(), P()),
                P(),
            ),
            out_specs=(
                ((output_specs, output_specs), P("expert", "feature"))
                if capture_layer0
                else (output_specs, output_specs)
            ),
            check_vma=False,
        )
    )


def build_owner_packet_suffix(mesh: Any, config: Any, *, interpret: bool = False) -> Any:
    """Owner-explicit physicalB128 replay of one captured normalized packet.

    Both input and output retain expert/feature axes, so this adapter neither
    introduces communication nor asserts unmeasured expert replication. Moving
    a wide32-row segment to the first32 rows is the fixed placement intervention.
    """
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.runtime.ws32_decoder import ws32_decoder_weight_specs
    from glm_tpu.greenfield.kernels.ws32_prefill_layer import ws32_prefill_mlp_mapped

    if type(interpret) is not bool:
        raise ValueError("interpret must be a static bool")

    def body(packet: Any, start: Any, count: Any, dense: Any) -> tuple:
        if (packet.shape != (1, 1, 128, config.geometry.hidden_size // 4)
                or packet.dtype != jnp.bfloat16
                or start.shape != () or start.dtype != jnp.int32
                or count.shape != () or count.dtype != jnp.int32):
            raise ValueError("owner suffix requires original B128 packet/int32 metadata")
        valid = (((count == 128) & (start == 0))
                 | ((count == 32) & ((start == 0) | (start == 32)
                                    | (start == 64) | (start == 96))))
        offset = jnp.clip(start, 0, 96)
        live = jnp.arange(128, dtype=jnp.int32) < jnp.clip(count, 0, 128)
        # Safe bounded selection, with no host restoration or global hidden gather.
        rows = jnp.minimum(offset + jnp.arange(128, dtype=jnp.int32), 127)
        normalized = jnp.take(packet[0, 0], rows, axis=0, mode="clip")
        normalized = jnp.where(live[:, None], normalized, 0)
        output, _, _, health = ws32_prefill_mlp_mapped(
            normalized, live, dense, None, moe_contract=config.moe_contract,
            linear_interpret=interpret, expert_panels=True,
        )
        output = jnp.where(live[:, None], output, 0)
        health = (health & valid & (~live | (jnp.all(jnp.isfinite(normalized), axis=1)
                                           & jnp.all(jnp.isfinite(output), axis=1))))
        return output[None, None], health[None, None]

    return jax.jit(jax.shard_map(
        body, mesh=mesh,
        in_specs=(P("expert", "feature"), P(), P(),
                  ws32_decoder_weight_specs(config).layers[0].dense),
        out_specs=(P("expert", "feature"), P("expert", "feature")), check_vma=False,
    ))
