"""Host-side owner packing of GLM-5.2 layer weights into the legacy TP32 layout.

The legacy prefill shards every projection over 32 model-parallel owners.  For
a bounded single-device probe the owners are virtual: this module slices the
checkpoint tensors (``[out, in]`` FP8 bits with ``[out/128, in/128]`` FP32
block scales) into the exact per-owner ``[K, N]`` operands that the legacy
convolutions consume, so ``legacy_prefill_geometry`` can execute them.

Pure NumPy; no TPU, no model execution, no claim.
"""

from __future__ import annotations

from typing import Any

import numpy as np

HIDDEN = 6144
OWNERS = 32
Q_LORA_RANK = 2048
KV_A_WIDTH = 576
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 192
QK_HEAD_DIM = 256
V_HEAD_DIM = 256
HEADS = 64
HEADS_PER_OWNER = 2
DENSE_INTERMEDIATE = 12288
BLOCK = 128


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_fp8(name: str, bits: np.ndarray, scale: np.ndarray, shape: tuple[int, int]) -> None:
    _require(bits.shape == shape and bits.dtype == np.uint8, f"{name} bits must be uint8 {shape}")
    expected = tuple(-(-dim // BLOCK) for dim in shape)
    _require(scale.shape == expected and scale.dtype == np.float32, f"{name} scale must be f32 {expected}")


def transpose_to_k_n(bits: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn checkpoint ``[out, in]`` bits/scales into convolution ``[K=in, N=out]``."""

    return np.ascontiguousarray(bits.T), np.ascontiguousarray(scale.T)


def pack_fused_qkv_a_owners(
    q_a_bits: np.ndarray, q_a_scale: np.ndarray, kv_a_bits: np.ndarray, kv_a_scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Owner ``o``: q-a output columns ``[64o, 64o+64)`` then kv-a ``[18o, 18o+18)``.

    Returns ``bits u8[32, 6144, 82]`` and per-column scales ``f32[32, 48, 82]``.
    """

    _check_fp8("q_a_proj", q_a_bits, q_a_scale, (Q_LORA_RANK, HIDDEN))
    _check_fp8("kv_a_proj_with_mqa", kv_a_bits, kv_a_scale, (KV_A_WIDTH, HIDDEN))
    q_width = Q_LORA_RANK // OWNERS  # 64
    kv_width = KV_A_WIDTH // OWNERS  # 18
    k_blocks = np.arange(HIDDEN) // BLOCK  # [6144]
    bits = np.empty((OWNERS, HIDDEN, q_width + kv_width), dtype=np.uint8)
    scale = np.empty((OWNERS, HIDDEN // BLOCK, q_width + kv_width), dtype=np.float32)
    for owner in range(OWNERS):
        q_rows = np.arange(owner * q_width, (owner + 1) * q_width)
        kv_rows = np.arange(owner * kv_width, (owner + 1) * kv_width)
        bits[owner, :, :q_width] = q_a_bits[q_rows].T
        bits[owner, :, q_width:] = kv_a_bits[kv_rows].T
        # per-column scale for each 128-row K block: scale[out_block, in_block]
        scale[owner, :, :q_width] = q_a_scale[q_rows // BLOCK, :].T  # [48, 64]
        scale[owner, :, q_width:] = kv_a_scale[kv_rows // BLOCK, :].T  # [48, 18]
    del k_blocks
    return bits, scale


def pack_q_b_owners(q_b_bits: np.ndarray, q_b_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Owner ``o``: heads ``2o, 2o+1`` → ``bits u8[32, 2048, 512]``, ``scale f32[32, 16, 4]``."""

    _check_fp8("q_b_proj", q_b_bits, q_b_scale, (HEADS * QK_HEAD_DIM, Q_LORA_RANK))
    width = HEADS_PER_OWNER * QK_HEAD_DIM  # 512
    bits = np.empty((OWNERS, Q_LORA_RANK, width), dtype=np.uint8)
    scale = np.empty((OWNERS, Q_LORA_RANK // BLOCK, width // BLOCK), dtype=np.float32)
    for owner in range(OWNERS):
        rows = slice(owner * width, (owner + 1) * width)
        bits[owner] = q_b_bits[rows].T
        scale[owner] = q_b_scale[owner * (width // BLOCK) : (owner + 1) * (width // BLOCK), :].T
    return bits, scale


def dequantize_bf16_host(bits: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Load-time ``bf16(f32(fp8) * f32(scale))`` on the host (checkpoint ``[out, in]``)."""

    import ml_dtypes

    rows, cols = bits.shape
    fp8 = bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    expanded = scale[(np.arange(rows) // BLOCK)[:, None], (np.arange(cols) // BLOCK)[None, :]]
    return (fp8 * expanded).astype(ml_dtypes.bfloat16)


def absorbed_kv_b_owners(kv_b_bits: np.ndarray, kv_b_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Legacy load-time absorbed MLA weights per owner (BF16, identity post-scale).

    Returns ``w_uk_t bf16[32, 2, 192, 512]`` (``[H, P, L]`` per owner) and
    ``w_uv bf16[32, 2, 512, 256]`` (``[H, L, V]``).
    """

    combined = QK_NOPE_HEAD_DIM + V_HEAD_DIM  # 448
    _check_fp8("kv_b_proj", kv_b_bits, kv_b_scale, (HEADS * combined, KV_LORA_RANK))
    dequantized = dequantize_bf16_host(kv_b_bits, kv_b_scale).reshape(HEADS, combined, KV_LORA_RANK)
    w_uk_t = np.ascontiguousarray(dequantized[:, :QK_NOPE_HEAD_DIM, :])  # [H, 192, 512]
    w_uv = np.ascontiguousarray(np.transpose(dequantized[:, QK_NOPE_HEAD_DIM:, :], (0, 2, 1)))  # [H, 512, 256]
    return (
        w_uk_t.reshape(OWNERS, HEADS_PER_OWNER, QK_NOPE_HEAD_DIM, KV_LORA_RANK),
        w_uv.reshape(OWNERS, HEADS_PER_OWNER, KV_LORA_RANK, V_HEAD_DIM),
    )


def pack_o_proj_owners(o_bits: np.ndarray, o_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Owner ``o``: K rows ``[512o, 512o+512)`` → ``bits u8[32, 512, 6144]``, ``scale f32[32, 4, 48]``."""

    _check_fp8("o_proj", o_bits, o_scale, (HIDDEN, HEADS * V_HEAD_DIM))
    k_width = HEADS_PER_OWNER * V_HEAD_DIM  # 512
    bits_kn, scale_kn = transpose_to_k_n(o_bits, o_scale)  # [16384, 6144], [128, 48]
    bits = bits_kn.reshape(OWNERS, k_width, HIDDEN)
    scale = scale_kn.reshape(OWNERS, k_width // BLOCK, HIDDEN // BLOCK)
    return np.ascontiguousarray(bits), np.ascontiguousarray(scale)


def pack_gate_up_owners(
    gate_bits: np.ndarray, gate_scale: np.ndarray, up_bits: np.ndarray, up_scale: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Owner ``o``: ``[gate cols 384o.., up cols 384o..]`` → ``u8[32, 6144, 768]``, ``f32[32, 48, 6]``."""

    _check_fp8("gate_proj", gate_bits, gate_scale, (DENSE_INTERMEDIATE, HIDDEN))
    _check_fp8("up_proj", up_bits, up_scale, (DENSE_INTERMEDIATE, HIDDEN))
    half = DENSE_INTERMEDIATE // OWNERS  # 384
    bits = np.empty((OWNERS, HIDDEN, 2 * half), dtype=np.uint8)
    scale = np.empty((OWNERS, HIDDEN // BLOCK, 2 * (half // BLOCK)), dtype=np.float32)
    for owner in range(OWNERS):
        rows = slice(owner * half, (owner + 1) * half)
        blocks = slice(owner * (half // BLOCK), (owner + 1) * (half // BLOCK))
        bits[owner, :, :half] = gate_bits[rows].T
        bits[owner, :, half:] = up_bits[rows].T
        scale[owner, :, : half // BLOCK] = gate_scale[blocks, :].T
        scale[owner, :, half // BLOCK :] = up_scale[blocks, :].T
    return bits, scale


def pack_down_owners(down_bits: np.ndarray, down_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Owner ``o``: K rows ``[384o, 384o+384)`` → ``u8[32, 384, 6144]``, ``f32[32, 3, 48]``."""

    _check_fp8("down_proj", down_bits, down_scale, (HIDDEN, DENSE_INTERMEDIATE))
    half = DENSE_INTERMEDIATE // OWNERS
    bits_kn, scale_kn = transpose_to_k_n(down_bits, down_scale)  # [12288, 6144], [96, 48]
    return (
        np.ascontiguousarray(bits_kn.reshape(OWNERS, half, HIDDEN)),
        np.ascontiguousarray(scale_kn.reshape(OWNERS, half // BLOCK, HIDDEN // BLOCK)),
    )


def chunk_embeddings(
    prompt_token_ids: np.ndarray,
    unique_token_ids: np.ndarray,
    unique_embedding_bits: np.ndarray,
    *,
    start: int,
    rows: int,
) -> np.ndarray:
    """Gather BF16 embedding bit rows for prompt positions ``[start, start+rows)``."""

    tokens = prompt_token_ids[start : start + rows]
    _require(tokens.shape == (rows,), "prompt chunk exceeds the prompt")
    lookup = {int(token): index for index, token in enumerate(unique_token_ids.tolist())}
    indices = np.array([lookup[int(token)] for token in tokens], dtype=np.int64)
    return np.ascontiguousarray(unique_embedding_bits[indices])


def summarize_bits(name: str, value: np.ndarray) -> dict[str, Any]:
    from hashlib import sha256

    return {
        "name": name,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    }
