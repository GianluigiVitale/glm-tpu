"""BF16-resident non-routed weights for the WS32 decode step (opt-in challenger, D8).

Measured on the pod (docs/perf, phase 4): a one-row ``[1,1536] x [2048,1536]``
projection costs 43-60 us in every FP8-decoding form (Pallas or XLA) but
10 us from a resident BF16 table.  TPU v4 has no FP8 datapath, so the frozen
step spends ~40 ms of its 121 ms decoding e4m3 on the vector units.

The frozen kernels compute, per element, ``bf16(f32(bits) * scale_block)``
and feed that BF16 value to the MXU with FP32 accumulation.  Decoding the
same expression once at load time and keeping the BF16 table resident gives
bitwise-identical MXU operands, so this is an exact transformation of the
non-routed weights (attention q_a/kv_a/q_b/kv_b/o, DSA wq_b/wk, shared
experts, dense layers).  Routed experts (22 GB/chip) stay FP8 and go through
the route-grouped kernel.  Cost: about +1.8 GB HBM per chip.

Every function here mirrors one frozen body in ``kernels/ws32_layer.py`` /
``kernels/ws32.py`` with the FP8 Pallas projection replaced by ``dot_general``
at the same FP32-accumulate / BF16-round boundary.  The only numerical
difference is the MXU accumulation order inside one contraction, which the CPU
tests bound; the TPU pod run reports token/state agreement.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from ..greenfield.kernels.pallas import SparseMlaConfig, pregathered_sparse_mla_pallas
from ..greenfield.kernels.reference.attention import (
    MlaNumericalContract,
    StageLocalKvLayout,
    canonicalize_selected_positions,
    gather_stage_local_selected_kv_aligned,
)
from ..greenfield.kernels.reference.dsa import (
    DsaNumericalContract,
    SelectedPositions,
    dsa_index_keys_from_projection,
    dsa_scores,
    local_topk_candidates,
    merge_topk_candidates_with_scores,
)
from ..greenfield.kernels.reference.moe import GlmMoeNumericalContract
from ..greenfield.kernels.reference.rmsnorm import rms_norm
from ..greenfield.kernels.reference.rotary import (
    apply_rotary,
    apply_rotary_fp32_final_round,
    rotary_cos_sin,
)
from ..greenfield.kernels.stage_local import _require_decode_metadata
from ..greenfield.kernels.ws32 import ws32_fused_add_rms_norm_mapped, ws32_router_from_shards_mapped
from ..greenfield.kernels.ws32_layer import (
    Ws32AttentionLayerResult,
    Ws32AttentionResult,
    Ws32DsaResult,
    Ws32MlpResult,
    Ws32PreparedAttention,
    Ws32TransformerLayerResult,
)
from ..greenfield.runtime import ws32_decoder as decoder
from .fp8_routed_experts import RoutedProjectionConfig, ws32_moe_grouped_routes_mapped


# ----------------------------------------------------------------------------- weights
class Bf16QkvAWeights(NamedTuple):
    input_norm_weight_local: Any
    q_a_local: Any          # bf16 [q_lora, local_hidden]
    q_a_norm_weight: Any
    kv_a_local: Any         # bf16 [kv_lora + rope, local_hidden]
    kv_a_norm_weight: Any


class Bf16AttentionWeights(NamedTuple):
    q_b_local: Any          # bf16 [local_heads * qk_head, q_lora]
    kv_b_local: Any         # bf16 [local_heads * 448, 512]
    o_local: Any            # bf16 [local_hidden, local_heads * v_head]


class Bf16DsaWeights(NamedTuple):
    wq_b_local: Any         # bf16 [local_dsa_heads * 128, q_lora]
    wk_local: Any           # bf16 [128, local_hidden]
    key_norm_weight: Any
    key_norm_bias: Any
    head_weight_local: Any


class Bf16DenseWeights(NamedTuple):
    gate_local: Any
    up_local: Any
    down_local: Any


class Bf16MoeWeights(NamedTuple):
    router_weight_local: Any
    correction_bias_local: Any
    expert_gate_bits_local: Any
    expert_gate_scale_local: Any
    expert_up_bits_local: Any
    expert_up_scale_local: Any
    expert_down_bits_local: Any
    expert_down_scale_local: Any
    shared_gate_local: Any  # bf16
    shared_up_local: Any    # bf16
    shared_down_local: Any  # bf16


class Bf16LayerWeights(NamedTuple):
    qkv_a: Bf16QkvAWeights
    attention: Bf16AttentionWeights
    dsa: Bf16DsaWeights | None
    post_attention_norm_weight_local: Any
    dense: Bf16DenseWeights | None
    moe: Bf16MoeWeights | None


class Bf16DecoderWeights(NamedTuple):
    embedding_local: Any
    layers: tuple[Bf16LayerWeights, ...]
    final_norm_weight_local: Any
    lm_head_local: Any


def decode_fp8_table(bits: Any, scale: Any, *, block_shape: tuple[int, int] = (128, 128)) -> Any:
    """Exactly the frozen kernels' per-element decode: ``bf16(f32(bits) * scale_block)``."""

    if bits.ndim != 2 or scale.ndim != 2 or bits.dtype != jnp.uint8 or scale.dtype != jnp.float32:
        raise ValueError("decode_fp8_table takes uint8 [N,K] bits and float32 [N/bn,K/bk] scales")
    n, k = bits.shape
    bn, bk = block_shape
    if scale.shape != ((n + bn - 1) // bn, (k + bk - 1) // bk):
        raise ValueError("decode_fp8_table scale geometry disagrees with the bits")
    rows = jnp.arange(n) // bn
    columns = jnp.arange(k) // bk
    expanded = scale[rows[:, None], columns[None, :]]
    decoded = lax.bitcast_convert_type(bits, jnp.float8_e4m3fn).astype(jnp.float32)
    return (decoded * expanded).astype(jnp.bfloat16)


def bf16_weight_specs(config: decoder.Ws32DecoderConfig) -> Bf16DecoderWeights:
    frozen = decoder.ws32_decoder_weight_specs(config)
    layers = []
    for spec in frozen.layers:
        q = spec.qkv_a
        a = spec.attention
        layers.append(Bf16LayerWeights(
            Bf16QkvAWeights(q[0], q[1], q[3], q[4], q[6]),
            Bf16AttentionWeights(a[0], a[2], a[4]),
            None if spec.dsa is None else Bf16DsaWeights(spec.dsa[0], spec.dsa[2], spec.dsa[4], spec.dsa[5], spec.dsa[6]),
            spec.post_attention_norm_weight_local,
            None if spec.dense is None else Bf16DenseWeights(spec.dense[0], spec.dense[2], spec.dense[4]),
            None if spec.moe is None else Bf16MoeWeights(*spec.moe[:8], spec.moe[8], spec.moe[10], spec.moe[12]),
        ))
    return Bf16DecoderWeights(frozen.embedding_local, tuple(layers), frozen.final_norm_weight_local, frozen.lm_head_local)


_DECODERS: dict = {}


def _decode_program(mesh: Any, bits: Any, scale: Any, spec: Any, block: tuple[int, int]) -> Any:
    """One jitted per-table decode, cached by (shape, spec); each chip decodes its own shard."""

    key = (tuple(bits.shape), tuple(scale.shape), tuple(spec), block)
    if key not in _DECODERS:
        _DECODERS[key] = jax.jit(jax.shard_map(
            lambda b, s: decode_fp8_table(b, s, block_shape=block),
            mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False,
        ))
    return _DECODERS[key]


def bf16_resident_weights(mesh: Any, config: decoder.Ws32DecoderConfig, weights: decoder.Ws32DecoderWeights) -> Bf16DecoderWeights:
    """Decode every non-routed FP8 shard on its owning chip; routed experts pass through.

    Shards of every affected table start on 128-block boundaries in both
    dimensions, so decoding a local shard with its local scale shard equals
    decoding the global table.  One small jitted ``shard_map`` per table (a
    single program over all layers ran out of HBM at compile time on the pod:
    XLA kept every FP32 intermediate of 78 layers live at once).
    """

    frozen_specs = decoder.ws32_decoder_weight_specs(config)
    block = tuple(config.geometry.fp8_block_shape)

    def dec(bits: Any, scale: Any, spec: Any) -> Any:
        return _decode_program(mesh, bits, scale, spec, block)(bits, scale)

    layers = []
    for layer, spec in zip(weights.layers, frozen_specs.layers, strict=True):
        q, qs = layer.qkv_a, spec.qkv_a
        a, as_ = layer.attention, spec.attention
        qkv = Bf16QkvAWeights(
            q.input_norm_weight_local,
            dec(q.q_a_bits_local, q.q_a_scale_local, qs[1]),
            q.q_a_norm_weight,
            dec(q.kv_a_bits_local, q.kv_a_scale_local, qs[4]),
            q.kv_a_norm_weight,
        )
        att = Bf16AttentionWeights(
            dec(a.q_b_bits_local, a.q_b_scale_local, as_[0]),
            dec(a.kv_b_bits_local, a.kv_b_scale_local, as_[2]),
            dec(a.o_bits_local, a.o_scale_local, as_[4]),
        )
        dsa = None
        if layer.dsa is not None:
            d, ds = layer.dsa, spec.dsa
            dsa = Bf16DsaWeights(
                dec(d.wq_b_bits_local, d.wq_b_scale_local, ds[0]),
                dec(d.wk_bits_local, d.wk_scale_local, ds[2]),
                d.key_norm_weight, d.key_norm_bias, d.head_weight_local,
            )
        dense = None
        if layer.dense is not None:
            dn, dns = layer.dense, spec.dense
            dense = Bf16DenseWeights(
                dec(dn.gate_bits_local, dn.gate_scale_local, dns[0]),
                dec(dn.up_bits_local, dn.up_scale_local, dns[2]),
                dec(dn.down_bits_local, dn.down_scale_local, dns[4]),
            )
        moe = None
        if layer.moe is not None:
            m, ms = layer.moe, spec.moe
            moe = Bf16MoeWeights(
                *m[:8],
                dec(m.shared_gate_bits_local, m.shared_gate_scale_local, ms[8]),
                dec(m.shared_up_bits_local, m.shared_up_scale_local, ms[10]),
                dec(m.shared_down_bits_local, m.shared_down_scale_local, ms[12]),
            )
        layers.append(Bf16LayerWeights(qkv, att, dsa, layer.post_attention_norm_weight_local, dense, moe))
    return Bf16DecoderWeights(weights.embedding_local, tuple(layers), weights.final_norm_weight_local, weights.lm_head_local)


# ----------------------------------------------------------------------------- projections
def _dot_f32(x: Any, weight_out_in: Any) -> Any:
    """``x @ W.T`` with BF16 operands and an FP32 accumulator (the frozen MXU boundary)."""

    return lax.dot_general(x, weight_out_in, (((x.ndim - 1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


def _feature_linear(x: Any, weight_local: Any, feature_axis: str) -> Any:
    partial = _dot_f32(x, weight_local)
    with jax.named_scope("glm_perf_bf16_linear/feature_reduce"):
        return lax.psum(partial, axis_name=feature_axis).astype(jnp.bfloat16)


def _expert_linear(x: Any, weight_local: Any, expert_axis: str) -> Any:
    partial = _dot_f32(x, weight_local)
    with jax.named_scope("glm_perf_bf16_linear/expert_reduce"):
        return lax.psum(partial, axis_name=expert_axis).astype(jnp.bfloat16)


# ----------------------------------------------------------------------------- attention bodies
def _head_weight_partial(normalized: Any, weights: Bf16DsaWeights) -> Any:
    return lax.dot_general(
        normalized.astype(jnp.float32), weights.head_weight_local.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())), preferred_element_type=jnp.float32,
    )


def fused_input_projections(normalized: Any, weights: Bf16QkvAWeights,
                            dsa: Bf16DsaWeights | None, *, feature_axis: str) -> tuple[Any, ...]:
    """Separate dot products, one FP32 feature sum, unchanged rounding boundaries.

    Q/KV are rounded to BF16 by the caller; WK and head weights remain FP32.
    No weight concatenation or reassociation of the contractions is performed.
    """
    partials = [_dot_f32(normalized, weights.q_a_local), _dot_f32(normalized, weights.kv_a_local)]
    if dsa is not None:
        partials.extend((_dot_f32(normalized, dsa.wk_local), _head_weight_partial(normalized, dsa)))
    cuts = []
    width = 0
    for partial in partials[:-1]:
        width += partial.shape[-1]
        cuts.append(width)
    with jax.named_scope("glm_perf_bf16_linear/fused_input_feature_reduce"):
        reduced = lax.psum(jnp.concatenate(partials, axis=-1), axis_name=feature_axis)
    return tuple(jnp.split(reduced, cuts, axis=-1))


def prepare_attention_bf16(residual_local: Any, weights: Bf16QkvAWeights, *, normalized: Any,
                           feature_axis: str, lora_norm_epsilon: float = 1e-5,
                           projected: tuple[Any, Any] | None = None) -> Ws32PreparedAttention:
    """Mirror of ``ws32_prepare_attention_mapped`` (raw path) on BF16 tables."""

    kv_lora_rank = weights.kv_a_norm_weight.shape[0]
    q_a = (_feature_linear(normalized, weights.q_a_local, feature_axis) if projected is None
           else projected[0].astype(jnp.bfloat16))
    q_residual = rms_norm(q_a, weights.q_a_norm_weight, epsilon=lora_norm_epsilon)
    projected_kv = (_feature_linear(normalized, weights.kv_a_local, feature_axis) if projected is None
                    else projected[1].astype(jnp.bfloat16))
    current_kv = jnp.concatenate(
        (rms_norm(projected_kv[..., :kv_lora_rank], weights.kv_a_norm_weight, epsilon=lora_norm_epsilon),
         projected_kv[..., kv_lora_rank:]), axis=-1,
    ).astype(jnp.bfloat16)
    return Ws32PreparedAttention(normalized, normalized, q_residual, current_kv)


def dsa_bf16(prepared: Ws32PreparedAttention, index_cache_local: Any, position: Any, block_tables: Any,
             context_lengths: Any, weights: Bf16DsaWeights, *, expert_axis: str, feature_axis: str,
             contract: DsaNumericalContract, cache_layout: StageLocalKvLayout, two_stage: bool = False,
             projected: tuple[Any, Any] | None = None) -> Ws32DsaResult:
    """Mirror of ``ws32_dsa_mapped`` (raw path) with BF16 wq_b / wk tables."""

    normalized = prepared.normalized_local
    local_heads = contract.num_heads // cache_layout.local_parallel_size
    owner = lax.axis_index(expert_axis)
    physical_page, local_row, target_owner, _, metadata_valid = _require_decode_metadata(
        position, block_tables, context_lengths, owner, layout=cache_layout,
        physical_page_count=index_cache_local.shape[0],
    )
    projected_query = _dot_f32(prepared.q_residual, weights.wq_b_local)
    query = projected_query.reshape(1, local_heads, contract.head_dim)
    if projected is None:
        with jax.named_scope("glm_perf_bf16_dsa/head_weight_feature_reduce"):
            reduced_head = lax.psum(_head_weight_partial(normalized, weights), axis_name=feature_axis)
    else:
        reduced_head = projected[1]
    local_head_weights = reduced_head * jnp.float32(contract.num_heads**-0.5)
    cos, sin = rotary_cos_sin(position, rotary_dim=contract.rotary_dim, theta=contract.theta, dtype=jnp.float32)
    rotated = apply_rotary(query[..., : contract.rotary_dim], cos[:, None, :], sin[:, None, :],
                           interleaved=contract.interleaved_rotary)
    local_query = jnp.concatenate((rotated, query[..., contract.rotary_dim:]), axis=-1).astype(jnp.float32)
    packed_query = jnp.concatenate((local_query.reshape(1, -1), local_head_weights), axis=-1)
    local_query_width = local_heads * contract.head_dim
    with jax.named_scope("glm_perf_bf16_dsa/query_expert_gather"):
        gathered_query = lax.all_gather(packed_query, axis_name=expert_axis, axis=0, tiled=False)
    query = jnp.transpose(gathered_query[..., :local_query_width], (1, 0, 2)).reshape(1, contract.num_heads, contract.head_dim)
    head_weights = jnp.transpose(gathered_query[..., local_query_width:], (1, 0, 2)).reshape(1, contract.num_heads)
    if projected is None:
        key_partial = _dot_f32(normalized, weights.wk_local)
        with jax.named_scope("glm_perf_bf16_dsa/key_feature_reduce"):
            projected_key = lax.psum(key_partial, axis_name=feature_axis)
    else:
        projected_key = projected[0]
    current_key_f32 = dsa_index_keys_from_projection(
        projected_key, weights.key_norm_weight, weights.key_norm_bias, position, contract=contract,
    ).astype(jnp.float32)
    current_key = current_key_f32.astype(index_cache_local.dtype)

    def write_current(value: Any) -> Any:
        return value.at[physical_page, local_row].set(current_key[0])

    index_cache_local = lax.cond(metadata_valid[0] & (owner == target_owner), write_current, lambda v: v, index_cache_local)
    if two_stage:
        from .dsa_candidates import score_cache_pages, two_stage_topk_mapped

        local_scores, global_positions = score_cache_pages(
            query, index_cache_local, head_weights, block_tables, layout=cache_layout, owner=owner,
        )
        selected, _ = two_stage_topk_mapped(
            local_scores, global_positions, context_lengths, top_k=contract.top_k,
            global_context_size=block_tables.shape[1] * cache_layout.logical_page_size,
            positions_in_order=True, expert_axis=expert_axis,
        )
    else:
        page_ids = block_tables[0]
        page_ok = (page_ids >= 0) & (page_ids < index_cache_local.shape[0])
        safe_pages = jnp.clip(page_ids, 0, index_cache_local.shape[0] - 1)
        logical_cache = jnp.take(index_cache_local, safe_pages, axis=0)
        logical_pages = jnp.arange(block_tables.shape[1], dtype=jnp.int32)
        local_rows = jnp.arange(cache_layout.local_rows_per_page, dtype=jnp.int32)
        global_positions = (
            logical_pages[:, None] * jnp.int32(cache_layout.logical_page_size)
            + owner.astype(jnp.int32) * jnp.int32(cache_layout.local_rows_per_page)
            + local_rows[None, :]
        )
        global_positions = jnp.where(page_ok[:, None], global_positions, jnp.int32(-1)).reshape(-1)
        local_keys = logical_cache.reshape(-1, contract.head_dim)
        with jax.named_scope("glm_perf_bf16_dsa/highest_score"):
            local_scores = dsa_scores(query, local_keys, head_weights, precision="highest")
        candidate_scores, candidate_positions = local_topk_candidates(
            local_scores, global_positions, context_lengths, top_k=contract.top_k,
        )
        with jax.named_scope("glm_perf_bf16_dsa/candidate_expert_gather"):
            gathered_scores = lax.all_gather(candidate_scores, axis_name=expert_axis, axis=0, tiled=False)
            gathered_positions = lax.all_gather(candidate_positions, axis_name=expert_axis, axis=0, tiled=False)
        selected = merge_topk_candidates_with_scores(
            gathered_scores, gathered_positions, context_lengths, top_k=contract.top_k,
            global_context_size=block_tables.shape[1] * cache_layout.logical_page_size,
        )
    selection_valid = canonicalize_selected_positions(
        SelectedPositions(selected.positions, selected.valid_counts)
    ).contract_valid
    return Ws32DsaResult(index_cache_local, selected.positions, selected.valid_counts, selected.scores,
                         metadata_valid & selection_valid)


def index_share_attention_bf16(residual_local: Any, prepared: Ws32PreparedAttention, cache_local: Any,
                               selected_positions: Any, selected_valid_counts: Any, position: Any,
                               block_tables: Any, context_lengths: Any, weights: Bf16AttentionWeights, *,
                               expert_axis: str, contract: MlaNumericalContract, cache_layout: StageLocalKvLayout,
                               main_rope_table_row: Any, sparse_attention_config: SparseMlaConfig,
                               sparse_attention_interpret: bool, lse_attention: bool = False) -> Ws32AttentionResult:
    """Mirror of ``ws32_index_share_attention_mapped`` (host rotary table path) on BF16 tables."""

    if main_rope_table_row is None:
        raise ValueError("bf16 IndexShare attention mirrors the host main-rotary path only")
    local_heads = contract.num_heads // cache_layout.local_parallel_size
    owner = lax.axis_index(expert_axis)
    physical_page, local_row, target_owner, _, metadata_valid = _require_decode_metadata(
        position, block_tables, context_lengths, owner, layout=cache_layout, physical_page_count=cache_local.shape[0],
    )
    q_states = _dot_f32(prepared.q_residual, weights.q_b_local).astype(jnp.bfloat16).reshape(1, local_heads, contract.qk_head_dim)
    q_nope = q_states[..., : contract.qk_nope_head_dim]
    q_rope_unrotated = q_states[..., contract.qk_nope_head_dim:]
    half = contract.qk_rope_head_dim // 2
    current_latent = prepared.current_kv[..., : contract.kv_lora_rank]
    current_rope_input = prepared.current_kv[..., contract.kv_lora_rank: contract.kv_lora_rank + contract.qk_rope_head_dim][:, None, :]
    with jax.named_scope("greenfield_ws32_main_rope_table"):
        cos = main_rope_table_row[:half][None, :]
        sin = main_rope_table_row[half:][None, :]
        q_rope = apply_rotary_fp32_final_round(q_rope_unrotated, cos[:, None, :], sin[:, None, :], interleaved=True)
        current_rope = apply_rotary_fp32_final_round(current_rope_input, cos[:, None, :], sin[:, None, :], interleaved=True)[:, 0, :]
    padding = contract.packed_cache_width - (contract.kv_lora_rank + contract.qk_rope_head_dim)
    current_cache_row = jnp.concatenate(
        (current_latent, current_rope, jnp.zeros((1, padding), dtype=jnp.bfloat16)), axis=-1
    ).astype(jnp.bfloat16)

    def write_current(value: Any) -> Any:
        return value.at[physical_page, local_row].set(current_cache_row[0])

    cache_local = lax.cond(metadata_valid[0] & (owner == target_owner), write_current, lambda v: v, cache_local)
    selected = SelectedPositions(selected_positions, selected_valid_counts)
    # Structured kv_b: head h owns rows [h*448, h*448+448) = [192 key rows | 256 value rows] x 512 latents.
    combined = contract.qk_nope_head_dim + contract.v_head_dim
    kv_b = weights.kv_b_local.reshape(local_heads, combined, contract.kv_lora_rank)
    key_rows = kv_b[:, : contract.qk_nope_head_dim, :]        # [h, 192, 512]
    value_rows = kv_b[:, contract.qk_nope_head_dim:, :]       # [h, 256, 512]
    q_absorbed = jnp.einsum(
        "rhq,hqk->rhk", q_nope, key_rows, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    if lse_attention:
        from .lse_attention import lse_attention_mapped

        partial = lse_attention_mapped(
            q_absorbed, q_rope, cache_local, block_tables, selected, context_lengths,
            contract=contract, layout=cache_layout, config=sparse_attention_config,
            interpret=sparse_attention_interpret, expert_axis=expert_axis,
        )
        attended, attention_valid = partial.output, partial.contract_valid
    else:
        aligned = gather_stage_local_selected_kv_aligned(
            cache_local, block_tables, selected, context_lengths, layout=cache_layout, owner_index=owner,
        )
        with jax.named_scope("glm_perf_bf16_attention/selected_cache_expert_exchange"):
            selected_cache = lax.psum(aligned.values, axis_name=expert_axis)
        with jax.named_scope("glm_perf_bf16_attention/pregathered_sparse_mla"):
            attended = pregathered_sparse_mla_pallas(
                q_absorbed, q_rope, selected_cache, aligned.valid_counts,
                contract=replace(contract, num_heads=local_heads), config=sparse_attention_config,
                interpret=sparse_attention_interpret,
            )
        attention_valid = aligned.contract_valid
    value_states = jnp.einsum(
        "rhk,hvk->rhv", attended, value_rows, preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    output_input = value_states.reshape(1, local_heads * contract.v_head_dim)
    update = _expert_linear(output_input, weights.o_local, expert_axis)
    return Ws32AttentionResult(update, cache_local, metadata_valid & attention_valid)


def attention_layer_bf16(residual_local: Any, cache_local: Any, index_cache_local: Any, selected_positions: Any,
                         selected_valid_counts: Any, selected_scores: Any, position: Any, block_tables: Any,
                         context_lengths: Any, qkv_a: Bf16QkvAWeights, attention: Bf16AttentionWeights,
                         dsa: Bf16DsaWeights | None, *, normalized: Any, dsa_contract: DsaNumericalContract,
                         attention_contract: MlaNumericalContract, cache_layout: StageLocalKvLayout,
                         sparse_attention_config: SparseMlaConfig, sparse_attention_interpret: bool,
                         main_rope_table_row: Any, expert_axis: str = "expert",
                         feature_axis: str = "feature", lse_attention: bool = False, dsa_two_stage: bool = False,
                         fused_feature_reductions: bool = False) -> Ws32AttentionLayerResult:
    projected = (fused_input_projections(normalized, qkv_a, dsa, feature_axis=feature_axis)
                 if fused_feature_reductions else None)
    prepared = prepare_attention_bf16(residual_local, qkv_a, normalized=normalized,
                                     feature_axis=feature_axis,
                                     projected=None if projected is None else projected[:2])
    dsa_valid = jnp.ones((1,), dtype=jnp.bool_)
    if dsa is not None:
        result = dsa_bf16(prepared, index_cache_local, position, block_tables, context_lengths, dsa,
                          expert_axis=expert_axis, feature_axis=feature_axis, contract=dsa_contract,
                          cache_layout=cache_layout, two_stage=dsa_two_stage,
                          projected=None if projected is None else projected[2:])
        index_cache_local = result.index_cache_local
        selected_positions, selected_valid_counts, selected_scores = (
            result.selected_positions, result.selected_valid_counts, result.selected_scores)
        dsa_valid = result.contract_valid
    attended = index_share_attention_bf16(
        residual_local, prepared, cache_local, selected_positions, selected_valid_counts, position,
        block_tables, context_lengths, attention, expert_axis=expert_axis, contract=attention_contract,
        cache_layout=cache_layout, main_rope_table_row=main_rope_table_row,
        sparse_attention_config=sparse_attention_config, sparse_attention_interpret=sparse_attention_interpret,
        lse_attention=lse_attention,
    )
    return Ws32AttentionLayerResult(
        attended.output_local, attended.cache_local, index_cache_local, selected_positions,
        selected_valid_counts, selected_scores, dsa_valid & attended.contract_valid,
    )


# ----------------------------------------------------------------------------- MLP bodies
def dense_bf16(normalized: Any, weights: Bf16DenseWeights, *, expert_axis: str, feature_axis: str) -> Any:
    """Mirror of ``ws32_dense_pallas_mapped`` on BF16 tables."""

    gate_partial = _dot_f32(normalized, weights.gate_local)
    up_partial = _dot_f32(normalized, weights.up_local)
    with jax.named_scope("glm_perf_bf16_dense/feature_gate_up_reduce"):
        gate_up = lax.psum(jnp.stack((gate_partial, up_partial), axis=0), axis_name=feature_axis).astype(jnp.bfloat16)
    activated = (gate_up[0] * jax.nn.sigmoid(gate_up[0]) * gate_up[1]).astype(jnp.bfloat16)
    down_partial = _dot_f32(activated, weights.down_local)
    with jax.named_scope("glm_perf_bf16_dense/expert_down_reduce"):
        return lax.psum(down_partial, axis_name=expert_axis).astype(jnp.bfloat16)


def mlp_bf16(post_attention_residual_local: Any, normalized: Any, dense: Bf16DenseWeights | None,
             moe: Bf16MoeWeights | None, *, mlp_kind: str, contract: GlmMoeNumericalContract,
             routed_projection: RoutedProjectionConfig | None, interpret: bool,
             expert_axis: str = "expert", feature_axis: str = "feature") -> Ws32MlpResult:
    if mlp_kind == "dense":
        if dense is None:
            raise ValueError("bf16 dense layer needs dense weights")
        update = dense_bf16(normalized, dense, expert_axis=expert_axis, feature_axis=feature_axis)
        return Ws32MlpResult(update, jnp.full((1, contract.top_k), jnp.int32(-1), dtype=jnp.int32),
                             jnp.zeros((1, contract.top_k), dtype=jnp.float32))
    if moe is None:
        raise ValueError("bf16 sparse layer needs MoE weights")
    route_indices, route_weights = ws32_router_from_shards_mapped(
        normalized, moe.router_weight_local, moe.correction_bias_local, top_k=contract.top_k,
    )
    update = ws32_moe_grouped_routes_mapped(
        normalized, route_indices, route_weights, *moe[2:8],
        None, None, None, None, None, None,
        contract=contract, config=routed_projection, interpret=interpret,
        shared_bf16=(moe.shared_gate_local, moe.shared_up_local, moe.shared_down_local),
    )
    return Ws32MlpResult(update, route_indices, route_weights)


# ----------------------------------------------------------------------------- layer / step
def transformer_layer_bf16(hidden_update_local: Any, carried_residual_local: Any, cache_local: Any,
                           index_cache_local: Any, selected_positions: Any, selected_valid_counts: Any,
                           selected_scores: Any, position: Any, block_tables: Any, context_lengths: Any,
                           layer: Bf16LayerWeights, incoming_contract_valid: Any, *, indexer_kind: str,
                           mlp_kind: str, config: decoder.Ws32DecoderConfig,
                           routed_projection: RoutedProjectionConfig | None, sparse_attention_interpret: bool,
                           linear_interpret: bool, main_rope_table_row: Any, lse_attention: bool = False, dsa_two_stage: bool = False,
                           fused_feature_reductions: bool = False) -> Ws32TransformerLayerResult:
    if (layer.dsa is None) != (indexer_kind == "shared"):
        raise ValueError("bf16 layer: full indexer alone must carry DSA weights")
    if (layer.dense is None) != (mlp_kind == "sparse") or (layer.moe is None) != (mlp_kind == "dense"):
        raise ValueError("bf16 layer weight presence drifted")
    dsa_contract = config.dsa_contract
    normalized_input, combined_residual = ws32_fused_add_rms_norm_mapped(
        hidden_update_local, carried_residual_local, layer.qkv_a.input_norm_weight_local,
        global_hidden_size=dsa_contract.hidden_size, epsilon=config.rms_norm_epsilon,
    )
    attention = attention_layer_bf16(
        combined_residual, cache_local, index_cache_local, selected_positions, selected_valid_counts,
        selected_scores, position, block_tables, context_lengths, layer.qkv_a, layer.attention, layer.dsa,
        normalized=normalized_input, dsa_contract=dsa_contract, attention_contract=config.attention_contract,
        cache_layout=config.cache_layout,
        sparse_attention_config=SparseMlaConfig(segment_block=config.sparse_segment_block),
        sparse_attention_interpret=sparse_attention_interpret, main_rope_table_row=main_rope_table_row,
        lse_attention=lse_attention, dsa_two_stage=dsa_two_stage,
        fused_feature_reductions=fused_feature_reductions,
    )
    normalized_mlp, post_attention_residual = ws32_fused_add_rms_norm_mapped(
        attention.output_local, combined_residual, layer.post_attention_norm_weight_local,
        global_hidden_size=config.moe_contract.hidden_size, epsilon=config.rms_norm_epsilon,
    )
    mlp = mlp_bf16(post_attention_residual, normalized_mlp, layer.dense, layer.moe, mlp_kind=mlp_kind,
                   contract=config.moe_contract, routed_projection=routed_projection, interpret=linear_interpret)
    return Ws32TransformerLayerResult(
        mlp.output_local, post_attention_residual, normalized_input, attention.cache_local,
        attention.index_cache_local, attention.selected_positions, attention.selected_valid_counts,
        attention.selected_scores, mlp.route_indices, mlp.route_weights,
        incoming_contract_valid & attention.contract_valid,
    )
