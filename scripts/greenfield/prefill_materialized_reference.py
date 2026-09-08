"""Default-off scalar layer3 reference with a completed BF16 MLP input.

This is a NEW reference realization, not the v1 fused scalar comparator or a
production execution path. Both programs must complete as separate executables.
"""

from __future__ import annotations

from typing import Any


def build_materialized_reference(
    mesh: Any,
    input_specs: tuple[Any, ...],
    *,
    dsa_contract: Any,
    attention_contract: Any,
    moe_contract: Any,
    rms_norm_epsilon: float = 1e-5,
    linear_interpret: bool = False,
    sparse_attention_interpret: bool = False,
) -> tuple[Any, Any]:
    """Build separate JIT prefix and raw scalar MLP; never JIT their composition."""
    import jax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped
    from glm_tpu.greenfield.kernels.ws32_layer import (
        ws32_attention_layer_mapped,
        ws32_mlp_mapped,
    )

    if len(input_specs) != 20:
        raise ValueError("materialized reference needs20-field layer schema")

    def prefix(*values: Any) -> tuple[Any, ...]:
        (
            u,
            r,
            kv,
            ic,
            rc,
            s,
            n,
            sc,
            offset,
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
        if (
            u.shape[0] != 1
            or d is not None
            or wk is not None
            or dense is not None
            or moe is None
        ):
            raise ValueError("materialized reference is scalar IndexShare+MoE only")
        norm, combined = ws32_fused_add_rms_norm_mapped(
            u,
            r,
            q.input_norm_weight_local,
            global_hidden_size=dsa_contract.hidden_size,
            epsilon=rms_norm_epsilon,
        )
        attention = ws32_attention_layer_mapped(
            combined,
            kv[0],
            ic[0],
            s,
            n,
            sc,
            offset[None],
            table,
            (offset + 1)[None],
            q,
            a,
            None,
            dsa_contract=dsa_contract,
            attention_contract=attention_contract,
            precomputed_normalized_local=norm,
            main_rope_table_row=rope[0],
            linear_interpret=linear_interpret,
            sparse_attention_interpret=sparse_attention_interpret,
            add_residual=False,
        )
        mlp_input, carried = ws32_fused_add_rms_norm_mapped(
            attention.output_local,
            combined,
            post,
            global_hidden_size=moe_contract.hidden_size,
            epsilon=rms_norm_epsilon,
        )
        return (
            mlp_input,
            carried,
            attention.cache_local[None],
            norm,
            (health[0, 0] & attention.contract_valid)[None, None],
        )

    def suffix(mlp_input: Any, carried: Any, post: Any, moe: Any) -> tuple[Any, ...]:
        result = ws32_mlp_mapped(
            carried,
            post,
            None,
            moe,
            mlp_kind="sparse",
            contract=moe_contract,
            rms_norm_epsilon=rms_norm_epsilon,
            linear_interpret=linear_interpret,
            precomputed_normalized_local=mlp_input,
            add_residual=False,
        )
        return result.output_local, result.route_indices, result.route_weights

    return (
        jax.jit(
            jax.shard_map(
                prefix,
                mesh=mesh,
                in_specs=input_specs,
                out_specs=(
                    P(None, "feature"),
                    P(None, "feature"),
                    P("expert", None, None, None),
                    P(None, "feature"),
                    P("expert", "feature", None),
                ),
                check_vma=False,
            )
        ),
        jax.jit(
            jax.shard_map(
                suffix,
                mesh=mesh,
                in_specs=(
                    P(None, "feature"),
                    P(None, "feature"),
                    input_specs[15],
                    input_specs[17],
                ),
                out_specs=(P(None, "feature"), P(), P()),
                check_vma=False,
            )
        ),
    )


def assemble_reference_result(
    values: tuple[Any, ...], prefix: tuple[Any, ...], mlp: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Return old12-field schema while carrying only this reference's own state."""
    if len(values) != 20 or len(prefix) != 5 or len(mlp) != 3:
        raise ValueError("materialized reference result schema differs")
    if prefix[2].shape != values[2].shape or prefix[2].dtype != values[2].dtype:
        raise ValueError("materialized reference KV shape/dtype differs")
    return (
        mlp[0],
        prefix[1],
        prefix[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        mlp[1],
        mlp[2],
        prefix[4],
        prefix[3],
    )
