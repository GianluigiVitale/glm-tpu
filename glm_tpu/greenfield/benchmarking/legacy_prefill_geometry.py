"""Legacy prefill-geometry primitives for prompt-row diagnostics.

These functions mirror, operation by operation, the arithmetic of one owner of
the accepted legacy prefill step (``jit_step_fun_impl.m64`` after-codegen HLO,
2,048-row chunks) so a bounded probe can execute the layer-0 block for prompt
rows at the legacy M geometry and compare the resulting layer-1 index keys with
the sealed legacy cache.  Every rounding point follows the fused computations
recorded in ``HANDOFF.md`` (input norm ``fused_computation.7276/14539/8215``,
FP8 dequant ``9540``, fused q-a/kv-a ``11384``, q-a norm ``10924/14538/10923``
and weight ``9309``, post-attention norm ``7900/14537/8056``, SwiGLU ``9619``,
layer-1 norm ``7897/8058``).

The module is pure JAX with no collectives: the 32 model-parallel owners are
virtual partitions executed on one device, exactly as the accepted greenfield
decode forms do.  It authorizes no TPU run, decoder change or Gate-D claim.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

HIDDEN = 6144
OWNERS = 32
FUSED_QKV_A_WIDTH = 82
Q_A_WIDTH_PER_OWNER = 64
KV_A_WIDTH_PER_OWNER = 18
Q_LORA_RANK = 2048
LEGACY_CHUNK_ROWS = 2048
RMS_EPSILON = 1e-5
# The legacy folds ``1/width`` into one FP32 constant before the add.
HIDDEN_INVERSE = 0.000162760422  # 1 / 6144 as the HLO literal
Q_LORA_INVERSE = 0.00048828125  # 1 / 2048
FP8_BLOCK = 128


def _require_bf16(name: str, value: Any) -> None:
    if value.dtype != jnp.bfloat16:
        raise ValueError(f"{name} must be BF16")


def legacy_dequantize_fp8(weight_bits: Any, scale: Any, *, block: int = FP8_BLOCK) -> Any:
    """``bf16(f32(fp8) * f32(scale))`` with block scales expanded to ``[K, N]``.

    ``weight_bits`` is raw ``uint8`` FP8 E4M3FN ``[K, N]``.  ``scale`` is FP32
    and either ``[ceil(K/block), ceil(N/block)]`` (checkpoint block scales in
    the ``[K, N]`` orientation) or ``[ceil(K/block), N]`` (per-column scales,
    the packed layout of the fused q-a/kv-a owners).
    """

    if weight_bits.ndim != 2 or weight_bits.dtype != jnp.uint8:
        raise ValueError("FP8 weight bits must be uint8 [K, N]")
    rows, cols = weight_bits.shape
    row_blocks_count = -(-rows // block)
    if scale.dtype != jnp.float32 or scale.ndim != 2 or scale.shape[0] != row_blocks_count:
        raise ValueError(f"FP8 scale must be f32 [{row_blocks_count}, N or ceil(N/{block})]")
    row_blocks = jnp.arange(rows) // block
    if scale.shape[1] == cols:
        expanded = scale[row_blocks, :]
    elif scale.shape[1] == -(-cols // block):
        col_blocks = jnp.arange(cols) // block
        expanded = scale[row_blocks[:, None], col_blocks[None, :]]
    else:
        raise ValueError("FP8 scale columns must be per column or per block")
    weight = lax.bitcast_convert_type(weight_bits, jnp.float8_e4m3fn).astype(jnp.float32)
    return (weight * expanded).astype(jnp.bfloat16)


def legacy_convolution_rows(lhs: Any, weight: Any) -> Any:
    """``conv(bf16[M,K], bf16[K,N]) -> f32 -> bf16`` at the caller's M."""

    _require_bf16("convolution lhs", lhs)
    _require_bf16("convolution weight", weight)
    if lhs.ndim != 2 or weight.ndim != 2 or lhs.shape[1] != weight.shape[0]:
        raise ValueError("convolution operands must be [M,K] and [K,N]")
    return lax.conv_general_dilated(
        lhs,
        weight,
        window_strides=(),
        padding=(),
        dimension_numbers=("NC", "IO", "NC"),
        preferred_element_type=jnp.float32,
    ).astype(jnp.bfloat16)


def legacy_rms_rsqrt(square_sum: Any, *, inverse_width: float, epsilon: float = RMS_EPSILON) -> Any:
    """``rsqrt(sum * (1/width) + eps)`` with the legacy FP32 constant order."""

    if square_sum.dtype != jnp.float32:
        raise ValueError("RMS square sum must be FP32")
    return lax.rsqrt(square_sum * jnp.float32(inverse_width) + jnp.float32(epsilon))


def legacy_input_norm(hidden: Any, weight: Any, valid: Any | None = None) -> Any:
    """Layer input RMSNorm over full 6144-wide BF16 rows (``7276/14539/8215``).

    Invalid rows (``valid`` false) are selected to NaN before squaring, as the
    legacy pad mask does; callers must drop them.
    """

    _require_bf16("input norm hidden", hidden)
    if hidden.ndim != 2 or hidden.shape[1] != HIDDEN:
        raise ValueError("input norm expects [M, 6144] rows")
    if weight.shape != (HIDDEN,) or weight.dtype != jnp.bfloat16:
        raise ValueError("input norm weight must be bf16[6144]")
    rows = hidden
    if valid is not None:
        if valid.shape != (hidden.shape[0],) or valid.dtype != jnp.bool_:
            raise ValueError("input norm mask must be bool[M]")
        rows = jnp.where(valid[:, None], hidden, jnp.bfloat16(jnp.nan))
    values = rows.astype(jnp.float32)
    square_sum = jnp.sum(values * values, axis=1)
    inverse = legacy_rms_rsqrt(square_sum, inverse_width=HIDDEN_INVERSE)
    scaled = (values * inverse[:, None]).astype(jnp.bfloat16).astype(jnp.float32)
    return (scaled * weight.astype(jnp.float32)[None, :]).astype(jnp.bfloat16)


def legacy_fused_qkv_a(normalized: Any, owner_weight_bits: Any, owner_scale: Any) -> Any:
    """Fused q-a/kv-a projection for all 32 virtual owners (``11384``).

    ``owner_weight_bits`` is ``u8[32, 6144, 82]`` and ``owner_scale`` is
    ``f32[32, 48, 82]`` (per-column scales for each 128-row block, the packed
    greenfield layout; owner ``o`` holds q-a columns ``[64o, 64o+64)`` followed
    by kv-a columns ``[18o, 18o+18)``); returns ``bf16[32, M, 82]``.
    """

    _require_bf16("fused q-a/kv-a normalized rows", normalized)
    if owner_weight_bits.shape != (OWNERS, HIDDEN, FUSED_QKV_A_WIDTH):
        raise ValueError("fused q-a/kv-a weights must be u8[32, 6144, 82]")
    if owner_scale.shape != (OWNERS, HIDDEN // FP8_BLOCK, FUSED_QKV_A_WIDTH) or (
        owner_scale.dtype != jnp.float32
    ):
        raise ValueError("fused q-a/kv-a scales must be f32[32, 48, 82]")

    def project(owner: tuple[Any, Any]) -> Any:
        bits, scale = owner
        return legacy_convolution_rows(normalized, legacy_dequantize_fp8(bits, scale))

    return lax.map(project, (owner_weight_bits, owner_scale))


def legacy_q_a_norm(projected: Any, q_a_norm_weight: Any) -> Any:
    """Sharded q-a RMSNorm then gathered weight multiply (``10924/14538/10923/9309``).

    ``projected`` is ``bf16[32, M, 82]``; the first 64 lanes of each owner are
    the q-a shard.  Per-owner FP32 sums are combined in owner order (the f32
    all-reduce), the scalar rsqrt is applied per owner, the 32 normalized
    shards are gathered to ``[M, 2048]`` and multiplied by the weight in FP32
    with one final BF16 rounding.
    """

    _require_bf16("q-a projection", projected)
    if projected.ndim != 3 or projected.shape[0] != OWNERS or projected.shape[2] != FUSED_QKV_A_WIDTH:
        raise ValueError("q-a projection must be bf16[32, M, 82]")
    if q_a_norm_weight.shape != (Q_LORA_RANK,) or q_a_norm_weight.dtype != jnp.bfloat16:
        raise ValueError("q-a norm weight must be bf16[2048]")
    q_shards = projected[:, :, :Q_A_WIDTH_PER_OWNER].astype(jnp.float32)  # [32, M, 64]
    owner_sums = jnp.sum(q_shards * q_shards, axis=2)  # [32, M]
    total = owner_sums[0]
    for owner in range(1, OWNERS):
        total = total + owner_sums[owner]
    inverse = legacy_rms_rsqrt(total, inverse_width=Q_LORA_INVERSE)  # [M]
    normalized_shards = (q_shards * inverse[None, :, None]).astype(jnp.bfloat16)
    gathered = jnp.transpose(normalized_shards, (1, 0, 2)).reshape(-1, Q_LORA_RANK)
    return (gathered.astype(jnp.float32) * q_a_norm_weight.astype(jnp.float32)[None, :]).astype(jnp.bfloat16)


def legacy_kv_a_lanes(projected: Any) -> Any:
    """Gather the 18 kv-a lanes of every owner into ``bf16[M, 576]``."""

    _require_bf16("kv-a projection", projected)
    lanes = projected[:, :, Q_A_WIDTH_PER_OWNER:FUSED_QKV_A_WIDTH]  # [32, M, 18]
    return jnp.transpose(lanes, (1, 0, 2)).reshape(projected.shape[1], OWNERS * KV_A_WIDTH_PER_OWNER)


def legacy_post_attention_norm(attention: Any, embedding: Any, weight: Any) -> tuple[Any, Any]:
    """Post-attention residual + RMSNorm (``7900/14537/8056``).

    Returns ``(carried, normalized)`` where ``carried = bf16(f32(attention) +
    f32(embedding))`` is the BF16 residual the legacy carries into the layer-1
    norm and ``normalized`` is the BF16 dense input.  The variance is taken on
    the unrounded FP32 sum.
    """

    for name, value in (("attention", attention), ("embedding", embedding)):
        _require_bf16(name, value)
        if value.ndim != 2 or value.shape[1] != HIDDEN:
            raise ValueError(f"{name} must be [M, 6144]")
    if weight.shape != (HIDDEN,) or weight.dtype != jnp.bfloat16:
        raise ValueError("post-attention norm weight must be bf16[6144]")
    total = attention.astype(jnp.float32) + embedding.astype(jnp.float32)
    square_sum = jnp.sum(total * total, axis=1)
    inverse = legacy_rms_rsqrt(square_sum, inverse_width=HIDDEN_INVERSE)
    scaled = (total * inverse[:, None]).astype(jnp.bfloat16).astype(jnp.float32)
    normalized = (scaled * weight.astype(jnp.float32)[None, :]).astype(jnp.bfloat16)
    return total.astype(jnp.bfloat16), normalized


def legacy_swiglu(gate_up: Any, *, half_width: int) -> Any:
    """``bf16(g * (1 / (1 + exp(-g))) * u)`` in FP32 from ``bf16[M, 2*half]`` (``9619``)."""

    _require_bf16("gate/up", gate_up)
    if gate_up.ndim != 2 or gate_up.shape[1] != 2 * half_width:
        raise ValueError("gate/up must be [M, 2 * half_width] with gate first")
    gate = gate_up[:, :half_width].astype(jnp.float32)
    up = gate_up[:, half_width:].astype(jnp.float32)
    one = jnp.float32(1.0)
    sigmoid = one / (jnp.exp(-gate) + one)
    return ((gate * sigmoid) * up).astype(jnp.bfloat16)


def legacy_layer1_input_norm(dense: Any, attention: Any, embedding: Any, weight: Any) -> tuple[Any, Any]:
    """Layer-1 input norm with the legacy residual association (``7897/8058``).

    ``D = bf16(f32(attention) + f32(embedding))``; the variance and the
    normalized value use the unrounded ``f32(dense) + f32(D)``.  Returns
    ``(carried, normalized)`` with ``carried = bf16(f32(dense) + f32(D))``.
    """

    for name, value in (("dense", dense), ("attention", attention), ("embedding", embedding)):
        _require_bf16(name, value)
        if value.ndim != 2 or value.shape[1] != HIDDEN:
            raise ValueError(f"{name} must be [M, 6144]")
    if weight.shape != (HIDDEN,) or weight.dtype != jnp.bfloat16:
        raise ValueError("layer-1 norm weight must be bf16[6144]")
    residual = (attention.astype(jnp.float32) + embedding.astype(jnp.float32)).astype(jnp.bfloat16)
    total = dense.astype(jnp.float32) + residual.astype(jnp.float32)
    square_sum = jnp.sum(total * total, axis=1)
    inverse = legacy_rms_rsqrt(square_sum, inverse_width=HIDDEN_INVERSE)
    scaled = (total * inverse[:, None]).astype(jnp.bfloat16).astype(jnp.float32)
    normalized = (scaled * weight.astype(jnp.float32)[None, :]).astype(jnp.bfloat16)
    return total.astype(jnp.bfloat16), normalized


def legacy_owner_partials_convolution(lhs_owner_slices: Any, owner_weight_bits: Any, owner_scale: Any) -> Any:
    """Per-owner ``bf16`` partials of a K-sharded projection (o_proj / down).

    ``lhs_owner_slices`` is ``bf16[32, M, K_owner]``, ``owner_weight_bits`` is
    ``u8[32, K_owner, N]``, ``owner_scale`` is ``f32[32, ceil(K_owner/128),
    ceil(N/128)]``; returns ``bf16[32, M, N]`` — the operands of the legacy
    BF16 32-way all-reduce.
    """

    _require_bf16("owner lhs slices", lhs_owner_slices)
    if lhs_owner_slices.ndim != 3 or lhs_owner_slices.shape[0] != OWNERS:
        raise ValueError("owner lhs slices must be bf16[32, M, K_owner]")
    if owner_weight_bits.shape[0] != OWNERS or owner_scale.shape[0] != OWNERS:
        raise ValueError("owner weights and scales must carry 32 owners")

    def project(owner: tuple[Any, Any, Any]) -> Any:
        lhs, bits, scale = owner
        return legacy_convolution_rows(lhs, legacy_dequantize_fp8(bits, scale))

    return lax.map(project, (lhs_owner_slices, owner_weight_bits, owner_scale))


def bf16_pairwise_tree_reduce(partials: Any) -> Any:
    """Balanced pairwise BF16 tree over 32 owner partials (diagnostic baseline).

    Not the physical association; the probe compares it with the DB533 row-0
    association so a mismatch attributable to the collective is visible.
    """

    _require_bf16("owner partials", partials)
    if partials.ndim != 3 or partials.shape[0] != OWNERS:
        raise ValueError("owner partials must be bf16[32, M, N]")
    values = partials
    while values.shape[0] > 1:
        half = values.shape[0] // 2
        values = (values[:half].astype(jnp.float32) + values[half:].astype(jnp.float32)).astype(jnp.bfloat16)
    return values[0]


def strategy_nd_row_association_reduce(partials: Any, row_reduce: Any) -> Any:
    """Apply a ``(32, 1, 6144) -> (1, 6144)`` DB533-style association per row."""

    _require_bf16("owner partials", partials)
    if partials.ndim != 3 or partials.shape[0] != OWNERS or partials.shape[2] != HIDDEN:
        raise ValueError("association reduce expects bf16[32, M, 6144]")
    rows = jnp.transpose(partials, (1, 0, 2))[:, :, None, :]  # [M, 32, 1, 6144]
    return jax.vmap(row_reduce)(rows)[:, 0, :]
