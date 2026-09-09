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
    capture_boundaries: bool = False,
    completed_prefix: bool = False,
    paired_position_sort: bool = False,
    rolled_prefix: bool = False,
    expert_panels: bool = False,
    sorted_local_merge: bool = False,
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
    Diagnostic capture returns (original_outputs, observations) for the candidate
    ONLY. Every observation retains explicit expert/feature owner dimensions;
    Python callbacks run while tracing, never on the device or the host runtime
    critical path. Additional outputs may perturb fusion: original-signature
    reproduction and independent graph/memory admission remain mandatory.
    completed_prefix instead returns the existing ten prefix fields, including
    its actual normalized MLP input; it is a distinct new execution contract.
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
    if completed_prefix and (candidate_window or capture_boundaries):
        raise ValueError("completed prefix is separate from fused window/capture modes")
    if any(
        type(v) is not bool for v in (rolled_prefix, expert_panels, sorted_local_merge)
    ):
        raise ValueError("candidate component options must be static booleans")
    if (rolled_prefix or expert_panels) and not candidate_window:
        raise ValueError("rolled prefix and panels require candidate_window")
    if rolled_prefix and capture_boundaries:
        raise ValueError("rolled prefix cannot expose unrolled boundary observations")
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
        observations: dict[str, Any] = {}

        def observe(name: str, arrays: dict[str, Any]) -> None:
            for field, value in arrays.items():
                key = f"{name}/{field}"
                if key in observations:
                    raise ValueError(f"duplicate prefill boundary observation: {key}")
                observations[key] = value[None, None]

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
            paired_position_sort=paired_position_sort,
            sorted_local_merge=sorted_local_merge,
            **(
                dict(rolled_prefix=rolled_prefix, expert_panels=expert_panels)
                if candidate_window
                else {}
            ),
            **({"_observe": observe} if capture_boundaries else {}),
            **({"prefix_only": True} if completed_prefix else {}),
            **numerical_options,
        )
        if completed_prefix:
            return (
                result.normalized_mlp_local,
                result.carried_residual_local,
                result.cache_local[None],
                result.unrepaired_index_cache[None],
                result.repaired_index_cache[None],
                result.selected_positions,
                result.selected_valid_counts,
                result.selected_scores,
                result.contract_valid[None, None],
                result.normalized_input_local,
            )
        outputs = (
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
        return (outputs, observations) if capture_boundaries else outputs

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

    def mapped(fn: Any, specs: Any = output_specs) -> Any:
        return jax.jit(
            jax.shard_map(
                fn,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=specs,
                check_vma=False,
            )
        )

    candidate_specs = (
        (output_specs, P("expert", "feature")) if capture_boundaries else output_specs
    )
    if completed_prefix:
        candidate_specs = output_specs[:8] + output_specs[10:]
    return mapped(candidate, candidate_specs), mapped(reference)


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
