"""Prefill DSA indexer of the production engine: resident BF16 projections, one-pass selection.

The bodies of ``greenfield/kernels/ws32_prefill_dsa.py`` with the query/key projections over the
resident BF16 tables (``prefill_bf16.resident_matmul_f32``) and the selection by the one-pass
two-stage selector ``dsa_candidates.prefill_dsa_one_pass_mapped`` at DEFAULT dot precision (decode
scores at HIGHEST) with 512 candidates per owner (S2d fold). Both index caches, the M64 repair and
the health rules are the frozen ones; the frozen module stays untouched as the numerical oracle
of the tests.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .prefill_cache import write_prefill_cache_block
from .reference.attention import StageLocalKvLayout
from .reference.dsa import DsaNumericalContract, dsa_index_keys_from_projection
from .reference.prefill_index import physical_m64_prompt_index_key_chunk
from .reference.rotary import apply_rotary, rotary_cos_sin
from .ws32_layer import Ws32PreparedAttention
from .bf16_resident import Bf16DsaWeights
from .dsa_candidates import prefill_dsa_one_pass_mapped
from .prefill_bf16 import resident_matmul_f32


class PrefillDsaInputs(NamedTuple):
    query: Any
    head_weights: Any
    keys: Any
    normalized_full: Any
    contract_valid: Any


class Ws32PrefillDsaResult(NamedTuple):
    unrepaired_index_cache: Any
    repaired_index_cache: Any
    selected_positions: Any
    selected_valid_counts: Any
    selected_scores: Any
    contract_valid: Any


def ws32_prefill_dsa_inputs_mapped(
    prepared: Ws32PreparedAttention,
    positions: Any,
    live: Any,
    weights: Bf16DsaWeights,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    linear_interpret: bool = False,
) -> PrefillDsaInputs:
    """Produce multirow queries/heads and unrepaired keys from final owners.

    Returned full normalization crosses feature4 only, for the existing M64 repair
    leaf. It is exactly the BF16 normalization used for the unrepaired producer.
    Padded rows are sanitized before arithmetic; live nonfinite operands fail health.
    """
    if lax.axis_size("expert") != 8 or lax.axis_size("feature") != 4:
        raise ValueError("prefill DSA requires WS32 expert8/feature4 mesh")
    normalized = prepared.normalized_local
    if (
        normalized.ndim != 2
        or not 1 <= normalized.shape[0] <= 32
        or (
            normalized.shape[1] * 4 != contract.hidden_size
            or normalized.dtype != jnp.bfloat16
            or contract.num_heads != 32
            or contract.head_dim != 128
        )
    ):
        raise ValueError("prefill DSA normalized/contract geometry drifted")
    rows, hidden = normalized.shape
    if (
        prepared.q_residual.shape != (rows, contract.q_lora_rank)
        or prepared.q_residual.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill DSA q residual geometry drifted")
    if (
        positions.shape != (rows,)
        or positions.dtype != jnp.int32
        or (live.shape != (rows,) or live.dtype != jnp.bool_)
    ):
        raise ValueError("prefill DSA requires int32 positions and boolean live rows")
    heads = contract.num_heads // 8
    if weights.wq_b_local.shape != (
        heads * contract.head_dim,
        contract.q_lora_rank,
    ) or (
        weights.wk_local.shape != (contract.head_dim, hidden)
        or weights.head_weight_local.shape != (heads, hidden)
        or weights.head_weight_local.dtype != jnp.bfloat16
    ):
        raise ValueError("prefill DSA final weight owner geometry drifted")
    if any(
        w.shape != (contract.head_dim,) or w.dtype != jnp.bfloat16
        for w in (
            weights.key_norm_weight,
            weights.key_norm_bias,
        )
    ):
        raise ValueError("prefill DSA affine norm parameters must be BF16 owners")
    normalized = jnp.where(live[:, None], normalized, 0)
    q_input = jnp.where(live[:, None], prepared.q_residual, 0)
    positions = jnp.where(live, positions, 0)
    projected_q = resident_matmul_f32(q_input, weights.wq_b_local, interpret=linear_interpret).reshape(rows, heads, contract.head_dim)
    head_partial = lax.dot_general(
        normalized.astype(jnp.float32),
        weights.head_weight_local.astype(jnp.float32),
        dimension_numbers=(((1,), (1,)), ((), ())),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    with jax.named_scope("greenfield_ws32_prefill_dsa/head_feature_reduce"):
        head = lax.psum(head_partial, "feature") * jnp.float32(contract.num_heads**-0.5)
    cos, sin = rotary_cos_sin(
        positions,
        rotary_dim=contract.rotary_dim,
        theta=contract.theta,
        dtype=jnp.float32,
    )
    rotated = apply_rotary(
        projected_q[..., : contract.rotary_dim],
        cos[:, None],
        sin[:, None],
        interleaved=True,
    )
    query = jnp.concatenate((rotated, projected_q[..., contract.rotary_dim :]), axis=-1)
    packed = jnp.concatenate((query.reshape(rows, -1), head), axis=-1)
    with jax.named_scope("greenfield_ws32_prefill_dsa/query_expert_gather"):
        gathered = lax.all_gather(packed, "expert", axis=0, tiled=False)
    query_width = heads * contract.head_dim
    query = (
        gathered[..., :query_width]
        .transpose(1, 0, 2)
        .reshape(rows, contract.num_heads, contract.head_dim)
    )
    head = (
        gathered[..., query_width:].transpose(1, 0, 2).reshape(rows, contract.num_heads)
    )
    key_partial = resident_matmul_f32(normalized, weights.wk_local, interpret=linear_interpret)
    with jax.named_scope("greenfield_ws32_prefill_dsa/key_feature_reduce"):
        projected_key = lax.psum(key_partial, "feature")
    keys = dsa_index_keys_from_projection(
        projected_key,
        weights.key_norm_weight,
        weights.key_norm_bias,
        positions,
        contract=contract,
        key_norm_mode="divide_sqrt",
    ).astype(jnp.bfloat16)
    with jax.named_scope(
        "greenfield_ws32_prefill_dsa/repair_normalized_feature_gather"
    ):
        normalized_full = lax.all_gather(normalized, "feature", axis=1, tiled=True)
    valid = ~live | (
        (positions >= 0)
        & jnp.all(jnp.isfinite(normalized_full), axis=1)
        & jnp.all(jnp.isfinite(q_input), axis=1)
        & jnp.all(jnp.isfinite(projected_key), axis=1)
        & jnp.all(jnp.isfinite(keys), axis=1)
        & jnp.all(jnp.isfinite(query), axis=(1, 2))
        & jnp.all(jnp.isfinite(head), axis=1)
    )
    return PrefillDsaInputs(query, head, keys, normalized_full, valid)


def ws32_prefill_dsa_mapped(
    prepared: Ws32PreparedAttention,
    unrepaired_index_cache: Any,
    repaired_index_cache: Any,
    position_offset: Any,
    valid_rows: Any,
    block_table: Any,
    weights: Bf16DsaWeights,
    materialized_wk: Any,
    *,
    contract: DsaNumericalContract = DsaNumericalContract(),
    linear_interpret: bool = False,
) -> Ws32PrefillDsaResult:
    """Append both key versions; score exclusively from UNREPAIRED storage.

    The FP32 repair weight must already be materialized/completed by the loader.
    Neither this function nor a successful health result promotes repaired keys.
    Caller binds offset to its committed append frontier, populated prefix and
    page-table identity, consumes all32 health flags and rejects all proposed state
    on failure. Distinct cache state/alias lifetime is a decoder-level obligation.
    """
    if unrepaired_index_cache.shape != repaired_index_cache.shape:
        raise ValueError("prefill dual index caches must share owner geometry")
    if block_table.ndim != 2 or block_table.shape[0] != 1 or block_table.shape[1] <= 0:
        raise ValueError("prefill DSA requires one nonempty shared page table")
    if any(
        v.shape != () or v.dtype != jnp.int32 for v in (position_offset, valid_rows)
    ):
        raise ValueError("prefill DSA offset/count must be int32 scalars")
    rows = prepared.normalized_local.shape[0]
    layout = StageLocalKvLayout(
        local_parallel_size=8, packed_cache_width=contract.head_dim
    )
    capacity = block_table.shape[1] * layout.logical_page_size
    if capacity >= 2147483647:
        raise ValueError("prefill DSA capacity must fit positive int32")
    live = jnp.arange(rows, dtype=jnp.int32) < jnp.clip(valid_rows, 0, rows)
    safe_start = jnp.clip(position_offset, 0, capacity - 1)
    positions = safe_start + jnp.minimum(
        jnp.arange(rows, dtype=jnp.int32), capacity - 1 - safe_start
    )
    inputs = ws32_prefill_dsa_inputs_mapped(
        prepared,
        positions,
        live,
        weights,
        contract=contract,
        linear_interpret=linear_interpret,
    )
    owner = lax.axis_index("expert")
    write = write_prefill_cache_block(
        unrepaired_index_cache,
        inputs.keys,
        block_table,
        position_offset,
        valid_rows,
        owner,
        layout=layout,
    )
    # Repeat last live row AND position, including final partial blocks. Zero
    # live uses finite sanitized input0/position0 and the writer remains a no-op.
    repeat = jnp.minimum(
        jnp.arange(64, dtype=jnp.int32),
        jnp.maximum(jnp.clip(valid_rows, 0, rows) - 1, 0),
    )
    repair_positions = jnp.where(jnp.any(live), positions[repeat], 0)
    with jax.named_scope("greenfield_ws32_prefill_dsa/m64_repair"):
        repair_keys = physical_m64_prompt_index_key_chunk(
            inputs.normalized_full[repeat],
            repair_positions,
            materialized_wk,
            weights.key_norm_weight,
            weights.key_norm_bias,
            contract=contract,
        )[:rows].astype(jnp.bfloat16)
    repair = write_prefill_cache_block(
        repaired_index_cache,
        repair_keys,
        block_table,
        position_offset,
        valid_rows,
        owner,
        layout=layout,
    )
    page_ids = block_table[0]
    safe_pages = jnp.clip(page_ids, 0, unrepaired_index_cache.shape[0] - 1)
    keys = write.cache[safe_pages].reshape(-1, contract.head_dim)
    logical_positions = (
        jnp.arange(block_table.shape[1], dtype=jnp.int32)[:, None]
        * layout.logical_page_size
        + owner * layout.local_rows_per_page
        + jnp.arange(layout.local_rows_per_page, dtype=jnp.int32)[None]
    ).reshape(-1)
    # Full live-prefix page range/uniqueness was checked by the block writer.
    # Invalid metadata supplies no keys/lengths, so selection cannot read an
    # unsafe mapping and health remains false regardless of empty results.
    logical_positions = jnp.where(write.valid, logical_positions, -1)
    selected, selector_ok = prefill_dsa_one_pass_mapped(
        inputs.query,
        keys,
        inputs.head_weights,
        logical_positions,
        write.causal_lengths,
        global_context_size=capacity,
        top_k=contract.top_k,
        precision="default",
        candidates_per_owner=512,
    )
    valid = write.valid & repair.valid & jnp.all(inputs.contract_valid) & selector_ok
    return Ws32PrefillDsaResult(
        write.cache,
        repair.cache,
        selected.positions,
        selected.valid_counts,
        selected.scores,
        valid,
    )
