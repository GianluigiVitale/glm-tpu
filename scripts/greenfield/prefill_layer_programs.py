"""Device programs for distinct bounded full-layer prefill admission.

No TPU initialization or execution at import time. Production decoder is not
modified. The scalar loop is an external, untimed reference only; the candidate
always runs the complete batched layer. Normalized inputs are observable so
each path's M64 repair can be checked against its own actual inputs.
"""

from __future__ import annotations

from typing import Any


def build_layer_programs(
    mesh: Any,
    input_specs: tuple[Any, ...],
    *,
    full_indexer: bool,
    sparse_mlp: bool,
    key_tile: int = 128,
    candidate_window: bool = False,
    **numerical_options: Any,
) -> tuple[Any, Any]:
    """Build batched candidate and existing raw-layout scalar reference.

    Inputs: update,residual,KV,index,repaired-index,positions,counts,scores,
    offset,live-count,page-table,qkv-weights,attention-weights,DSA-weights,
    completed-wk,post-norm,dense-weights,MoE-weights,owner-health,host-RoPE.
    Unused static weight branches may be None. Cache inputs retain an explicit
    leading expert owner dimension; health retains both owner dimensions.
    Outputs follow Ws32PrefillLayerResult, with owner dimensions retained on
    cache/health arrays and normalized-input observability as the final leaf.
    """

    import jax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.ws32_layer import ws32_transformer_layer_mapped
    from glm_tpu.greenfield.kernels.ws32_prefill_layer import (
        ws32_prefill_transformer_layer_mapped,
    )
    from glm_tpu.greenfield.kernels.ws32_prefill_window import (
        ws32_prefill_layer_window_mapped,
    )

    if len(input_specs) != 20:
        raise ValueError("layer admission requires the exact20-field input tree")
    output_specs = (
        P(None, "feature"),
        P(None, "feature"),
        P("expert", None, None, None),
        P("expert", None, None, None),
        P("expert", None, None, None),
        P(),
        P(),
        P(),
        P(),
        P(),
        P("expert", "feature", None),
        P(None, "feature"),
    )

    def candidate(*values: Any) -> tuple[Any, ...]:
        (
            u,
            r,
            c,
            ic,
            rc,
            s,
            n,
            sc,
            o,
            count,
            table,
            q,
            a,
            d,
            wk,
            post,
            dense,
            moe,
            health,
            rope,
        ) = values
        layer_fn = (
            ws32_prefill_layer_window_mapped
            if candidate_window
            else ws32_prefill_transformer_layer_mapped
        )
        result = layer_fn(
            u,
            r,
            c[0],
            ic[0],
            rc[0],
            s,
            n,
            sc,
            o,
            count,
            table,
            q,
            a,
            d if full_indexer else None,
            wk if full_indexer else None,
            post,
            None if sparse_mlp else dense,
            moe if sparse_mlp else None,
            health[0, 0],
            main_rope_table_rows=rope,
            key_tile=key_tile,
            **numerical_options,
        )
        return (
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

    def reference(*values: Any) -> tuple[Any, ...]:
        (
            u,
            r,
            c,
            ic,
            rc,
            s,
            n,
            sc,
            o,
            count,
            table,
            q,
            a,
            d,
            wk,
            post,
            dense,
            moe,
            health,
            rope,
        ) = values
        result = ws32_transformer_layer_mapped(
            u,
            r,
            c[0],
            ic[0],
            s,
            n,
            sc,
            o[None],
            table,
            (o + 1)[None],
            q,
            a,
            d if full_indexer else None,
            post,
            None if sparse_mlp else dense,
            moe if sparse_mlp else None,
            health[0, 0],
            indexer_kind="full" if full_indexer else "shared",
            mlp_kind="sparse" if sparse_mlp else "dense",
            main_rope_table_row=rope[0],
            **numerical_options,
        )
        # Repair is deliberately a separate completed program. Returning the
        # existing destination here is not a claim that the scalar path repaired it.
        return (
            result.output_local,
            result.carried_residual_local,
            result.cache_local[None],
            result.index_cache_local[None],
            rc,
            result.selected_positions,
            result.selected_valid_counts,
            result.selected_scores,
            result.route_indices,
            result.route_weights,
            result.contract_valid[None, None],
            result.normalized_input_local,
        )

    def mapped(fn: Any) -> Any:
        return jax.jit(
            jax.shard_map(
                fn,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=output_specs,
                check_vma=False,
            )
        )

    return mapped(candidate), mapped(reference)


def build_repair_program(mesh: Any, *, contract: Any) -> Any:
    """Reconstruct M64 keys from actual observed per-path normalization.

    The caller must provide a completed FP32 wk derived via the existing
    BF16-decode then FP32-promotion executable boundaries. This program returns
    keys only: the independent host comparator resolves cache addresses.
    """

    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.reference.prefill_index import (
        physical_m64_prompt_index_key_chunk,
    )

    def repair(normalized, offset, count, wk, norm, bias):
        rows = normalized.shape[0]
        if not 1 <= rows <= 32:
            raise ValueError("repair admission requires1..32 rows")
        full = lax.all_gather(normalized, "feature", axis=1, tiled=True)
        repeat = jnp.minimum(jnp.arange(64), jnp.maximum(count - 1, 0))
        positions = offset + repeat
        return physical_m64_prompt_index_key_chunk(
            full[repeat],
            positions,
            wk,
            norm,
            bias,
            contract=contract,
        )[:rows].astype(jnp.bfloat16)

    return jax.jit(
        jax.shard_map(
            repair,
            mesh=mesh,
            in_specs=(P(None, "feature"), P(), P(), P(), P(), P()),
            out_specs=P(),
            check_vma=False,
        )
    )
