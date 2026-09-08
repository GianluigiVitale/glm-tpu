"""Unwired, untimed routing-prefix diagnostic for the layer3 refusal.

Reuse existing attention/norm kernels, stop before the expert MLP, and expose
both sides of the router projection. Instrumentation may alter TPU lowering:
the original ordered-route swap MUST reproduce before claiming its cause.
This module neither initializes TPU nor grants a campaign/acceptance pass.
"""

from __future__ import annotations

from typing import Any


FIELDS = (
    "router_input",
    "partial_logits",
    "logits",
    "bias",
    "biased_scores",
    "routes",
    "route_weights",
    "attention_update",
    "combined_residual",
    "post_residual",
    "kv",
    "prefix_health",
)


def router_boundary_mapped(
    hidden: Any,
    router_weight: Any,
    correction_bias: Any,
    live: Any,
) -> tuple[Any, ...]:
    """Source arithmetic of both existing routers, with actual intermediates.

    Deliberately keep DEFAULT dot precision and the shared noaux_tc selector.
    A caller passing one row retains the scalar geometry; no vmap converts it
    into the batched projection. The same function supports same-input replays.
    """
    import jax
    from jax import lax
    import jax.numpy as jnp
    from glm_tpu.greenfield.kernels.reference.moe import route_glm_noaux_tc_logits

    if (
        hidden.ndim != 2
        or not 1 <= hidden.shape[0] <= 32
        or hidden.dtype != jnp.bfloat16
        or router_weight.shape != (32, hidden.shape[1])
        or router_weight.dtype != jnp.bfloat16
        or correction_bias.shape != (32,)
        or correction_bias.dtype != jnp.float32
        or live.shape != (hidden.shape[0],)
        or live.dtype != jnp.bool_
        or lax.axis_size("expert") != 8
        or lax.axis_size("feature") != 4
    ):
        raise ValueError("router diagnostic geometry/dtype differs")
    clean = jnp.where(live[:, None], hidden, 0)
    partial = lax.dot_general(
        clean.astype(jnp.float32),
        router_weight.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    with jax.named_scope("prefill_router_boundary/feature_reduce"):
        local_logits = lax.psum(partial, "feature")
    with jax.named_scope("prefill_router_boundary/expert_gather"):
        logits = lax.all_gather(local_logits, "expert", axis=1, tiled=True)
        bias = lax.all_gather(correction_bias, "expert", axis=0, tiled=True)
    ids, weights = route_glm_noaux_tc_logits(logits, bias, top_k=8)
    return (
        clean,
        partial[None, None],
        logits,
        bias,
        jax.nn.sigmoid(logits) + bias[None],
        ids,
        jnp.where(live[:, None], weights, 0),
    )


def scalar_prefix_inputs(
    values: tuple[Any, ...],
    row: int,
    previous: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    """Carry prefix output10 KV, never feed this tuple to old result indexing."""
    from scripts.greenfield.probe_ws32_prefill_layer import scalar_inputs

    one = list(scalar_inputs(values, row))
    if previous is not None:
        if (
            len(previous) != len(FIELDS)
            or previous[10].shape != values[2].shape
            or previous[10].dtype != values[2].dtype
        ):
            raise ValueError("scalar prefix cache result shape/dtype differs")
        one[2] = previous[10]
    return tuple(one)


def build_router_replay_program(mesh: Any) -> Any:
    """Compile separately at M17 and M1 for identical captured BF16 inputs."""
    import jax
    from jax.sharding import PartitionSpec as P

    return jax.shard_map(
        router_boundary_mapped,
        mesh=mesh,
        in_specs=(P(None, "feature"), P("expert", "feature"), P("expert"), P()),
        out_specs=(
            P(None, "feature"),
            P("expert", "feature", None, None),
            P(),
            P(),
            P(),
            P(),
            P(),
        ),
        check_vma=False,
    )


def build_router_prefix_program(
    mesh: Any,
    input_specs: tuple[Any, ...],
    *,
    batched: bool,
    dsa_contract: Any,
    attention_contract: Any,
    rms_norm_epsilon: float = 1e-5,
    linear_interpret: bool = False,
    sparse_attention_interpret: bool = False,
) -> Any:
    """Actual layer3 attention and post-norm prefix, no full DSA or MLP.

    Reuses the twenty-input admission tree but never consumes expert payloads.
    Scalar callers carry their own returned KV across rows. Both paths expose
    the unrounded-add normalization actually consumed by their router.
    Returned prefix health is incoming+attention only, not full-layer health;
    the diagnostic collector must independently reject nonfinite router arrays.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped
    from glm_tpu.greenfield.kernels.ws32_layer import ws32_attention_layer_mapped
    from glm_tpu.greenfield.kernels.ws32_prefill_attention import (
        ws32_prefill_prepare_attention_mapped,
        ws32_prefill_index_share_attention_mapped,
    )

    if type(batched) is not bool or len(input_specs) != 20:
        raise ValueError("router prefix requires the twenty-input static layer3 tree")

    def body(*values):
        (
            update,
            residual,
            cache,
            index,
            repair,
            selected,
            counts,
            scores,
            offset,
            count,
            table,
            qkv,
            attention_weights,
            dsa,
            wk,
            post_norm,
            dense,
            moe,
            health,
            rope,
        ) = values
        rows = update.shape[0]
        if (
            (not batched and rows != 1)
            or dsa is not None
            or wk is not None
            or dense is not None
            or moe is None
            or table.shape != (1, 2)
        ):
            raise ValueError("router prefix is only the bounded layer3 IndexShare case")
        live = jnp.arange(rows) < count
        norm, combined = ws32_fused_add_rms_norm_mapped(
            jnp.where(live[:, None], update, 0),
            jnp.where(live[:, None], residual, 0),
            qkv.input_norm_weight_local,
            global_hidden_size=dsa_contract.hidden_size,
            epsilon=rms_norm_epsilon,
        )
        if batched:
            prepared = ws32_prefill_prepare_attention_mapped(
                combined,
                qkv,
                precomputed_normalized_local=norm,
                rms_norm_epsilon=rms_norm_epsilon,
                linear_interpret=linear_interpret,
            )
            attention = ws32_prefill_index_share_attention_mapped(
                combined,
                prepared,
                cache[0],
                selected,
                counts,
                offset,
                count,
                table,
                attention_weights,
                main_rope_table_rows=rope,
                contract=attention_contract,
                linear_interpret=linear_interpret,
                sparse_attention_interpret=sparse_attention_interpret,
                add_residual=False,
            )
        else:
            attention = ws32_attention_layer_mapped(
                combined,
                cache[0],
                index[0],
                selected,
                counts,
                scores,
                offset[None],
                table,
                (offset + 1)[None],
                qkv,
                attention_weights,
                None,
                dsa_contract=dsa_contract,
                attention_contract=attention_contract,
                precomputed_normalized_local=norm,
                main_rope_table_row=rope[0],
                linear_interpret=linear_interpret,
                sparse_attention_interpret=sparse_attention_interpret,
                add_residual=False,
            )
        router_input, post_residual = ws32_fused_add_rms_norm_mapped(
            attention.output_local,
            combined,
            post_norm,
            global_hidden_size=dsa_contract.hidden_size,
            epsilon=rms_norm_epsilon,
        )
        router = router_boundary_mapped(
            router_input,
            moe.router_weight_local,
            moe.correction_bias_local,
            live,
        )
        valid = health[0, 0] & attention.contract_valid
        return (
            *router,
            attention.output_local,
            combined,
            post_residual,
            attention.cache_local[None],
            valid[None, None],
        )

    return jax.shard_map(
        body,
        mesh=mesh,
        in_specs=input_specs,
        out_specs=(
            P(None, "feature"),
            P("expert", "feature", None, None),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(None, "feature"),
            P(None, "feature"),
            P(None, "feature"),
            P("expert", None, None, None),
            P("expert", "feature", None),
        ),
        check_vma=False,
    )
