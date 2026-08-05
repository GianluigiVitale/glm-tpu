"""Topology-local PP8 executables for the real Gate C proof."""

from __future__ import annotations

from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from ..kernels.reference.attention import (
    MlaNumericalContract,
    SparseAttentionResult,
    StageLocalKvLayout,
    canonicalize_selected_positions,
    combine_stage_local_attention,
    gather_stage_local_selected_kv,
    sparse_mla_attention,
)
from ..kernels.reference.dsa import (
    DsaNumericalContract,
    SelectedPositions,
    dsa_index_keys,
    dsa_scores,
    local_topk_candidates,
    merge_topk_candidates,
)
from ..kernels.reference.linear import linear, residual_add, silu
from ..kernels.reference.rmsnorm import rms_norm
from ..kernels.reference.rotary import apply_rotary, rotary_cos_sin
from ..sharding.hlo_contract import parse_hlo_module


class GateCDenseResult(NamedTuple):
    normalized: jax.Array
    gate: jax.Array
    up: jax.Array
    activated: jax.Array
    update: jax.Array
    output: jax.Array


class GateCDsaResult(NamedTuple):
    normalized: jax.Array
    q_residual: jax.Array
    query: jax.Array
    head_weights: jax.Array
    index_keys_by_owner: jax.Array
    scores_by_owner: jax.Array
    selected_positions: jax.Array
    selected_scores: jax.Array
    valid_counts: jax.Array


class GateCIndexShareResult(NamedTuple):
    normalized: jax.Array
    q_residual: jax.Array
    q_nope: jax.Array
    q_rope: jax.Array
    q_absorbed: jax.Array
    current_cache_row: jax.Array
    cache_by_owner: jax.Array
    attention_positions: jax.Array
    attended_latent: jax.Array
    attention_lse: jax.Array
    value_states: jax.Array
    attention_update: jax.Array
    output: jax.Array
    contract_valid: jax.Array
    selected_positions: jax.Array


def _validate_mesh(mesh: Mesh, *, axis_name: str, stage_size: int) -> None:
    if mesh.axis_names != (axis_name,):
        raise ValueError(
            f"Gate C mesh axes must be exactly {(axis_name,)}, got {mesh.axis_names}"
        )
    if mesh.devices.size != stage_size:
        raise ValueError(
            f"Gate C stage requires {stage_size} devices, got {mesh.devices.size}"
        )


def stage_local_dense_gate_c(
    residual: jax.Array,
    norm_weight: jax.Array,
    gate_weight: jax.Array,
    up_weight: jax.Array,
    down_weight: jax.Array,
    *,
    mesh: Mesh,
    epsilon: float = 1e-5,
    axis_name: str = "stage",
) -> GateCDenseResult:
    """Execute the real dense SwiGLU with one four-chip output combine."""

    stage_size = int(mesh.devices.size)
    _validate_mesh(mesh, axis_name=axis_name, stage_size=stage_size)
    if gate_weight.shape != up_weight.shape:
        raise ValueError("Gate C dense gate/up shapes disagree")
    hidden = residual.shape[-1]
    intermediate = gate_weight.shape[0]
    if residual.shape != (1, hidden) or norm_weight.shape != (hidden,):
        raise ValueError("Gate C dense residual/norm shapes are invalid")
    if gate_weight.shape[1] != hidden or down_weight.shape != (
        hidden,
        intermediate,
    ):
        raise ValueError("Gate C dense checkpoint shapes are invalid")
    if intermediate % stage_size:
        raise ValueError("Gate C dense intermediate must divide over the stage")

    def mapped(
        local_residual: jax.Array,
        local_norm: jax.Array,
        local_gate_weight: jax.Array,
        local_up_weight: jax.Array,
        local_down_weight: jax.Array,
    ) -> GateCDenseResult:
        normalized = rms_norm(
            local_residual, local_norm, epsilon=epsilon
        )
        gate = linear(normalized, local_gate_weight)
        up = linear(normalized, local_up_weight)
        activated = (silu(gate) * up).astype(normalized.dtype)
        local_update = linear(activated, local_down_weight)
        update = lax.psum(local_update, axis_name=axis_name)
        output = residual_add(local_residual, update)
        return GateCDenseResult(
            normalized,
            gate,
            up,
            activated,
            update,
            output,
        )

    execute = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=(
            P(),
            P(),
            P(axis_name, None),
            P(axis_name, None),
            P(None, axis_name),
        ),
        out_specs=GateCDenseResult(
            P(),
            P(None, axis_name),
            P(None, axis_name),
            P(None, axis_name),
            P(),
            P(),
        ),
        check_vma=False,
    )
    return execute(
        residual,
        norm_weight,
        gate_weight,
        up_weight,
        down_weight,
    )


def _local_dsa_query(
    q_residual: jax.Array,
    normalized: jax.Array,
    query_weight: jax.Array,
    head_weight: jax.Array,
    position: jax.Array,
    *,
    contract: DsaNumericalContract,
) -> tuple[jax.Array, jax.Array]:
    local_heads = query_weight.shape[0] // contract.head_dim
    with jax.default_matmul_precision("highest"):
        query = linear(
            q_residual,
            query_weight,
            output_dtype=jnp.float32,
        ).reshape(1, local_heads, contract.head_dim)
        head_weights = linear(
            normalized,
            head_weight,
            output_dtype=jnp.float32,
        ) * jnp.float32(contract.num_heads**-0.5)
    cos, sin = rotary_cos_sin(
        position,
        rotary_dim=contract.rotary_dim,
        theta=contract.theta,
        dtype=jnp.float32,
    )
    rotated = apply_rotary(
        query[..., : contract.rotary_dim],
        cos[:, None, :],
        sin[:, None, :],
        interleaved=contract.interleaved_rotary,
    )
    return (
        jnp.concatenate(
            (rotated, query[..., contract.rotary_dim :]), axis=-1
        ).astype(jnp.float32),
        head_weights.astype(jnp.float32),
    )


def stage_local_dsa_gate_c(
    decode_residual: jax.Array,
    history_hidden_by_owner: jax.Array,
    history_positions_by_owner: jax.Array,
    decode_position: jax.Array,
    context_lengths: jax.Array,
    input_norm_weight: jax.Array,
    q_a_weight: jax.Array,
    q_a_norm_weight: jax.Array,
    wq_b_weight: jax.Array,
    wk_weight: jax.Array,
    key_norm_weight: jax.Array,
    key_norm_bias: jax.Array,
    head_weight: jax.Array,
    *,
    mesh: Mesh,
    contract: DsaNumericalContract = DsaNumericalContract(),
    rms_norm_epsilon: float = 1e-5,
    axis_name: str = "stage",
) -> GateCDsaResult:
    """Score striped context and merge an exact four-owner global top-k."""

    stage_size = int(mesh.devices.size)
    _validate_mesh(mesh, axis_name=axis_name, stage_size=stage_size)
    if contract.num_heads % stage_size:
        raise ValueError("DSA heads must divide over the local stage")
    if history_hidden_by_owner.ndim != 3 or history_positions_by_owner.shape != (
        history_hidden_by_owner.shape[0],
        history_hidden_by_owner.shape[1],
    ):
        raise ValueError("DSA owner-packed history shapes are invalid")
    if history_hidden_by_owner.shape[0] != stage_size:
        raise ValueError("DSA history does not cover every local owner")
    local_context = history_hidden_by_owner.shape[1]
    context_capacity = stage_size * local_context
    local_heads = contract.num_heads // stage_size

    def mapped(
        residual: jax.Array,
        local_history_container: jax.Array,
        local_positions_container: jax.Array,
        position: jax.Array,
        valid_lengths: jax.Array,
        norm_weight: jax.Array,
        qa_weight: jax.Array,
        qa_norm_weight: jax.Array,
        local_wq_weight: jax.Array,
        key_weight: jax.Array,
        kn_weight: jax.Array,
        kn_bias: jax.Array,
        local_head_weight: jax.Array,
    ) -> GateCDsaResult:
        local_history = local_history_container[0]
        local_positions = local_positions_container[0]
        normalized = rms_norm(
            residual, norm_weight, epsilon=rms_norm_epsilon
        )
        q_residual = rms_norm(
            linear(normalized, qa_weight),
            qa_norm_weight,
            epsilon=rms_norm_epsilon,
        )
        local_query, local_head_weights = _local_dsa_query(
            q_residual,
            normalized,
            local_wq_weight,
            local_head_weight,
            position,
            contract=contract,
        )
        query_width = local_heads * contract.head_dim
        packed_query = jnp.concatenate(
            (
                local_query.reshape(1, query_width),
                local_head_weights,
            ),
            axis=-1,
        )
        gathered_query = lax.all_gather(
            packed_query,
            axis_name=axis_name,
            axis=0,
            tiled=False,
        )
        query = jnp.transpose(
            gathered_query[..., :query_width], (1, 0, 2)
        ).reshape(1, contract.num_heads, contract.head_dim)
        gathered_head_weights = jnp.transpose(
            gathered_query[..., query_width:], (1, 0, 2)
        ).reshape(1, contract.num_heads)
        keys = dsa_index_keys(
            local_history,
            key_weight,
            kn_weight,
            kn_bias,
            local_positions,
            contract=contract,
        )
        local_scores = dsa_scores(query, keys, gathered_head_weights)
        candidate_scores, candidate_positions = local_topk_candidates(
            local_scores,
            local_positions,
            valid_lengths,
            top_k=contract.top_k,
        )
        gathered_scores = lax.all_gather(
            candidate_scores,
            axis_name=axis_name,
            axis=0,
            tiled=False,
        )
        gathered_positions = lax.all_gather(
            candidate_positions,
            axis_name=axis_name,
            axis=0,
            tiled=False,
        )
        selected = merge_topk_candidates(
            gathered_scores,
            gathered_positions,
            valid_lengths,
            top_k=contract.top_k,
            global_context_size=context_capacity,
        )
        flat_scores = jnp.transpose(gathered_scores, (1, 0, 2)).reshape(
            1, stage_size * contract.top_k
        )
        flat_positions = jnp.transpose(
            gathered_positions, (1, 0, 2)
        ).reshape(1, stage_size * contract.top_k)
        matches = (
            selected.positions[:, :, None] == flat_positions[:, None, :]
        )
        selected_scores = jnp.max(
            jnp.where(matches, flat_scores[:, None, :], -jnp.inf),
            axis=-1,
        )
        return GateCDsaResult(
            normalized,
            q_residual,
            query,
            gathered_head_weights,
            keys[None, ...],
            local_scores[None, ...],
            selected.positions,
            selected_scores.astype(jnp.float32),
            selected.valid_counts,
        )

    execute = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=(
            P(),
            P(axis_name, None, None),
            P(axis_name, None),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(axis_name, None),
            P(),
            P(),
            P(),
            P(axis_name, None),
        ),
        out_specs=GateCDsaResult(
            P(),
            P(),
            P(),
            P(),
            P(axis_name, None, None),
            P(axis_name, None, None),
            P(),
            P(),
            P(),
        ),
        check_vma=False,
    )
    return execute(
        decode_residual,
        history_hidden_by_owner,
        history_positions_by_owner,
        decode_position,
        context_lengths,
        input_norm_weight,
        q_a_weight,
        q_a_norm_weight,
        wq_b_weight,
        wk_weight,
        key_norm_weight,
        key_norm_bias,
        head_weight,
    )


def _reshape_gathered_heads(
    value: jax.Array,
    *,
    num_heads: int,
    head_width: int,
) -> jax.Array:
    return jnp.transpose(value, (1, 0, 2, 3)).reshape(
        1, num_heads, head_width
    )


def stage_local_index_share_gate_c(
    decode_residual: jax.Array,
    cache_by_owner: jax.Array,
    selected_positions: jax.Array,
    selected_valid_counts: jax.Array,
    decode_position: jax.Array,
    block_tables: jax.Array,
    context_lengths: jax.Array,
    input_norm_weight: jax.Array,
    q_a_weight: jax.Array,
    q_a_norm_weight: jax.Array,
    q_b_weight: jax.Array,
    kv_a_weight: jax.Array,
    kv_a_norm_weight: jax.Array,
    kv_b_weight: jax.Array,
    o_weight: jax.Array,
    *,
    mesh: Mesh,
    contract: MlaNumericalContract = MlaNumericalContract(),
    cache_layout: StageLocalKvLayout = StageLocalKvLayout(),
    rms_norm_epsilon: float = 1e-5,
    rope_theta: float = 8_000_000.0,
    axis_name: str = "stage",
) -> GateCIndexShareResult:
    """Reuse compact DSA state, write local KV, and attend over local owners."""

    stage_size = int(mesh.devices.size)
    _validate_mesh(mesh, axis_name=axis_name, stage_size=stage_size)
    if cache_layout.local_parallel_size != stage_size:
        raise ValueError("IndexShare cache layout and stage size disagree")
    if contract.num_heads % stage_size:
        raise ValueError("attention heads must divide over the local stage")
    if cache_by_owner.shape[0] != stage_size or cache_by_owner.ndim != 4:
        raise ValueError("IndexShare cache must expose one leading owner axis")
    local_heads = contract.num_heads // stage_size
    combined_width = contract.qk_nope_head_dim + contract.v_head_dim
    selected = SelectedPositions(selected_positions, selected_valid_counts)

    def mapped(
        residual: jax.Array,
        local_cache_container: jax.Array,
        selection_positions: jax.Array,
        selection_counts: jax.Array,
        position: jax.Array,
        tables: jax.Array,
        lengths: jax.Array,
        norm_weight: jax.Array,
        qa_weight: jax.Array,
        qa_norm_weight: jax.Array,
        local_qb_weight: jax.Array,
        kva_weight: jax.Array,
        kva_norm_weight: jax.Array,
        local_kvb_weight: jax.Array,
        local_o_weight: jax.Array,
    ) -> GateCIndexShareResult:
        owner = lax.axis_index(axis_name)
        local_cache = local_cache_container[0]
        local_selected = SelectedPositions(
            selection_positions, selection_counts
        )
        normalized = rms_norm(
            residual, norm_weight, epsilon=rms_norm_epsilon
        )
        q_residual = rms_norm(
            linear(normalized, qa_weight),
            qa_norm_weight,
            epsilon=rms_norm_epsilon,
        )
        q_states = linear(q_residual, local_qb_weight).reshape(
            1, local_heads, contract.qk_head_dim
        )
        q_nope = q_states[..., : contract.qk_nope_head_dim]
        q_rope_unrotated = q_states[..., contract.qk_nope_head_dim :]
        cos, sin = rotary_cos_sin(
            position,
            rotary_dim=contract.qk_rope_head_dim,
            theta=rope_theta,
            dtype=q_rope_unrotated.dtype,
        )
        q_rope = apply_rotary(
            q_rope_unrotated,
            cos[:, None, :],
            sin[:, None, :],
            interleaved=True,
        )

        current_kv = linear(normalized, kva_weight)
        current_latent = rms_norm(
            current_kv[..., : contract.kv_lora_rank],
            kva_norm_weight,
            epsilon=rms_norm_epsilon,
        )
        current_rope_unrotated = current_kv[
            ...,
            contract.kv_lora_rank : contract.kv_lora_rank
            + contract.qk_rope_head_dim,
        ]
        current_rope = apply_rotary(
            current_rope_unrotated[:, None, :],
            cos[:, None, :],
            sin[:, None, :],
            interleaved=True,
        )[:, 0, :]
        padding = contract.packed_cache_width - (
            contract.kv_lora_rank + contract.qk_rope_head_dim
        )
        current_cache_row = jnp.concatenate(
            (
                current_latent,
                current_rope,
                jnp.zeros((1, padding), dtype=normalized.dtype),
            ),
            axis=-1,
        )
        current = position[0].astype(jnp.int32)
        logical_page = current // jnp.int32(cache_layout.logical_page_size)
        within_page = current % jnp.int32(cache_layout.logical_page_size)
        target_owner = within_page // jnp.int32(
            cache_layout.local_rows_per_page
        )
        local_row = within_page % jnp.int32(cache_layout.local_rows_per_page)
        physical_page = tables[0, logical_page]

        def write_current(value: jax.Array) -> jax.Array:
            return value.at[physical_page, local_row].set(current_cache_row[0])

        local_cache = lax.cond(
            owner == target_owner,
            write_current,
            lambda value: value,
            local_cache,
        )

        local_kvb = local_kvb_weight.reshape(
            local_heads,
            combined_width,
            contract.kv_lora_rank,
        )
        local_weight_uk = local_kvb[
            :, : contract.qk_nope_head_dim, :
        ]
        local_weight_uv = jnp.transpose(
            local_kvb[:, contract.qk_nope_head_dim :, :], (0, 2, 1)
        )
        q_absorbed_local = jnp.einsum(
            "rhp,hpl->rhl",
            q_nope.astype(jnp.float32),
            local_weight_uk.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        packed_query = jnp.concatenate(
            (q_absorbed_local, q_rope), axis=-1
        )
        gathered_query = lax.all_gather(
            packed_query,
            axis_name=axis_name,
            axis=0,
            tiled=False,
        )
        full_query = _reshape_gathered_heads(
            gathered_query,
            num_heads=contract.num_heads,
            head_width=contract.kv_lora_rank
            + contract.qk_rope_head_dim,
        )
        q_absorbed = full_query[..., : contract.kv_lora_rank]
        full_q_rope = full_query[..., contract.kv_lora_rank :]
        segment = gather_stage_local_selected_kv(
            local_cache,
            tables,
            local_selected,
            lengths,
            layout=cache_layout,
            owner_index=owner,
        )
        partial = sparse_mla_attention(
            q_absorbed,
            full_q_rope,
            segment,
            contract=contract,
        )
        gathered_outputs = lax.all_gather(
            partial.output,
            axis_name=axis_name,
            axis=0,
            tiled=False,
        )
        gathered_lse = lax.all_gather(
            partial.logsumexp,
            axis_name=axis_name,
            axis=0,
            tiled=False,
        )
        gathered_validity = lax.all_gather(
            partial.contract_valid,
            axis_name=axis_name,
            axis=0,
            tiled=False,
        )
        combined: SparseAttentionResult = combine_stage_local_attention(
            gathered_outputs,
            gathered_lse,
            gathered_validity,
        )
        attended_local = lax.dynamic_slice_in_dim(
            combined.output,
            owner * local_heads,
            local_heads,
            axis=1,
        )
        value_states = jnp.einsum(
            "rhl,hlv->rhv",
            attended_local.astype(jnp.float32),
            local_weight_uv.astype(jnp.float32),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)
        local_update = linear(
            value_states.reshape(1, local_heads * contract.v_head_dim),
            local_o_weight,
        )
        update = lax.psum(local_update, axis_name=axis_name)
        output = residual_add(residual, update)
        attention_positions = canonicalize_selected_positions(
            local_selected
        ).selection.positions
        return GateCIndexShareResult(
            normalized,
            q_residual,
            q_nope,
            full_q_rope,
            q_absorbed,
            current_cache_row,
            local_cache[None, ...],
            attention_positions,
            combined.output,
            combined.logsumexp,
            value_states,
            update,
            output,
            combined.contract_valid,
            selection_positions,
        )

    execute = jax.shard_map(
        mapped,
        mesh=mesh,
        in_specs=(
            P(),
            P(axis_name, None, None, None),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(axis_name, None),
            P(),
            P(),
            P(axis_name, None),
            P(None, axis_name),
        ),
        out_specs=GateCIndexShareResult(
            P(),
            P(),
            P(None, axis_name, None),
            P(),
            P(),
            P(),
            P(axis_name, None, None, None),
            P(),
            P(),
            P(),
            P(None, axis_name, None),
            P(),
            P(),
            P(),
            P(),
        ),
        check_vma=False,
    )
    return execute(
        decode_residual,
        cache_by_owner,
        selected.positions,
        selected.valid_counts,
        decode_position,
        block_tables,
        context_lengths,
        input_norm_weight,
        q_a_weight,
        q_a_norm_weight,
        q_b_weight,
        kv_a_weight,
        kv_a_norm_weight,
        kv_b_weight,
        o_weight,
    )


def validate_gate_c_hlo(
    optimized_hlo: str,
    *,
    case: str,
    stage_size: int = 4,
    hidden_size: int = 6144,
) -> dict[str, object]:
    """Reject non-local, unexpected, or dead-row Gate C communication."""

    if case not in ("dense", "dsa", "index_share"):
        raise ValueError(f"unknown Gate C HLO case {case!r}")
    module = parse_hlo_module(optimized_hlo)
    expected_groups = (tuple(range(stage_size)),)
    violations = []
    collectives = module.collectives
    for collective in collectives:
        if collective.replica_groups != expected_groups:
            violations.append(
                f"{collective.name} escaped local group: "
                f"{collective.replica_groups}"
            )
        if collective.opcode not in ("all-gather", "all-reduce"):
            violations.append(
                f"unexpected Gate C collective {collective.opcode}"
            )
    all_gathers = [
        item for item in collectives if item.opcode == "all-gather"
    ]
    all_reduces = [
        item for item in collectives if item.opcode == "all-reduce"
    ]
    if case == "dense":
        if len(collectives) != 1 or len(all_reduces) != 1:
            violations.append(
                "dense Gate C requires exactly one local all-reduce"
            )
    elif case == "dsa":
        if all_reduces or not 2 <= len(all_gathers) <= 3:
            violations.append(
                "DSA Gate C requires two/three local all-gather instructions "
                "and no all-reduce"
            )
        gathered_domains = sum(
            max(1, len(item.operand_shapes)) for item in all_gathers
        )
        if gathered_domains < 3:
            violations.append(
                "DSA Gate C lost a query/score/position gather domain"
            )
    else:
        if len(all_reduces) != 1 or not 2 <= len(all_gathers) <= 4:
            violations.append(
                "IndexShare Gate C requires one local output all-reduce and "
                "two-to-four local all-gather instructions"
            )
        gathered_domains = sum(
            max(1, len(item.operand_shapes)) for item in all_gathers
        )
        if gathered_domains < 4:
            violations.append(
                "IndexShare Gate C lost a query/output/LSE/validity gather domain"
            )
    forbidden_shapes = []
    for instruction in module.instructions:
        for shape in instruction.operand_shapes + instruction.result_shapes:
            if shape.dimensions in ((32, hidden_size), (32, 1, hidden_size)):
                forbidden_shapes.append(
                    {"instruction": instruction.name, "shape": shape.to_dict()}
                )
    if forbidden_shapes:
        violations.append(
            f"Gate C contains forbidden dead-row/full-pod tensors: {forbidden_shapes}"
        )
    if module.num_partitions not in (None, stage_size):
        violations.append(
            f"Gate C expected {stage_size} partitions, found {module.num_partitions}"
        )
    return {
        "all_gather_count": len(all_gathers),
        "all_reduce_count": len(all_reduces),
        "case": case,
        "collective_count": len(collectives),
        "collectives": [item.to_dict() for item in collectives],
        "module_name": module.name,
        "num_partitions": module.num_partitions,
        "num_replicas": module.num_replicas,
        "passed": not violations,
        "violations": violations,
    }
