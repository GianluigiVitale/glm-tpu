"""CPU tests for the legacy TP32 owner packing of GLM-5.2 layer weights."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking import legacy_prefill_owner_packing as pk


def _fp8(rng, shape):
    return rng.integers(0, 120, size=shape, dtype=np.uint8)


def _scale(rng, shape):
    return rng.uniform(0.5, 1.5, size=shape).astype(np.float32)


def test_fused_qkv_a_owner_layout_and_scales():
    rng = np.random.default_rng(0)
    q_bits, q_scale = _fp8(rng, (2048, 6144)), _scale(rng, (16, 48))
    kv_bits, kv_scale = _fp8(rng, (576, 6144)), _scale(rng, (5, 48))
    bits, scale = pk.pack_fused_qkv_a_owners(q_bits, q_scale, kv_bits, kv_scale)
    assert bits.shape == (32, 6144, 82) and scale.shape == (32, 48, 82)
    owner = 7
    np.testing.assert_array_equal(bits[owner, :, :64], q_bits[64 * owner : 64 * owner + 64].T)
    np.testing.assert_array_equal(bits[owner, :, 64:], kv_bits[18 * owner : 18 * owner + 18].T)
    # column c of owner 7 is q_a output row 64*7+c whose scale block row is (448+c)//128
    assert scale[owner, 5, 3] == q_scale[(64 * owner + 3) // 128, 5]
    assert scale[owner, 40, 64 + 2] == kv_scale[(18 * owner + 2) // 128, 40]
    with pytest.raises(ValueError):
        pk.pack_fused_qkv_a_owners(q_bits[:1], q_scale, kv_bits, kv_scale)


def test_q_b_and_o_proj_owners_follow_head_pairs():
    rng = np.random.default_rng(1)
    qb_bits, qb_scale = _fp8(rng, (16384, 2048)), _scale(rng, (128, 16))
    bits, scale = pk.pack_q_b_owners(qb_bits, qb_scale)
    assert bits.shape == (32, 2048, 512) and scale.shape == (32, 16, 4)
    np.testing.assert_array_equal(bits[3], qb_bits[1536:2048].T)
    np.testing.assert_array_equal(scale[3], qb_scale[12:16, :].T)
    o_bits, o_scale = _fp8(rng, (6144, 16384)), _scale(rng, (48, 128))
    ob, osc = pk.pack_o_proj_owners(o_bits, o_scale)
    assert ob.shape == (32, 512, 6144) and osc.shape == (32, 4, 48)
    np.testing.assert_array_equal(ob[5], o_bits[:, 2560:3072].T)
    np.testing.assert_array_equal(osc[5], o_scale[:, 20:24].T)


def test_absorbed_kv_b_matches_load_time_dequant():
    rng = np.random.default_rng(2)
    kvb_bits, kvb_scale = _fp8(rng, (28672, 512)), _scale(rng, (224, 4))
    w_uk_t, w_uv = pk.absorbed_kv_b_owners(kvb_bits, kvb_scale)
    assert w_uk_t.shape == (32, 2, 192, 512) and w_uv.shape == (32, 2, 512, 256)
    assert w_uk_t.dtype == ml_dtypes.bfloat16
    full = pk.dequantize_bf16_host(kvb_bits, kvb_scale).reshape(64, 448, 512)
    head = 2 * 9 + 1
    np.testing.assert_array_equal(w_uk_t[9, 1].view(np.uint16), full[head, :192, :].view(np.uint16))
    np.testing.assert_array_equal(w_uv[9, 1].view(np.uint16), full[head, 192:, :].T.view(np.uint16))
    fp8 = kvb_bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    expanded = kvb_scale[(np.arange(28672) // 128)[:, None], (np.arange(512) // 128)[None, :]]
    np.testing.assert_array_equal(
        full.reshape(28672, 512).astype(np.float32), (fp8 * expanded).astype(ml_dtypes.bfloat16).astype(np.float32)
    )


def test_dense_owners_gate_up_then_down():
    rng = np.random.default_rng(3)
    g_bits, g_scale = _fp8(rng, (12288, 6144)), _scale(rng, (96, 48))
    u_bits, u_scale = _fp8(rng, (12288, 6144)), _scale(rng, (96, 48))
    bits, scale = pk.pack_gate_up_owners(g_bits, g_scale, u_bits, u_scale)
    assert bits.shape == (32, 6144, 768) and scale.shape == (32, 48, 6)
    np.testing.assert_array_equal(bits[2, :, :384], g_bits[768:1152].T)
    np.testing.assert_array_equal(bits[2, :, 384:], u_bits[768:1152].T)
    np.testing.assert_array_equal(scale[2, :, :3], g_scale[6:9, :].T)
    d_bits, d_scale = _fp8(rng, (6144, 12288)), _scale(rng, (48, 96))
    db, dsc = pk.pack_down_owners(d_bits, d_scale)
    assert db.shape == (32, 384, 6144) and dsc.shape == (32, 3, 48)
    np.testing.assert_array_equal(db[4], d_bits[:, 1536:1920].T)
    np.testing.assert_array_equal(dsc[4], d_scale[:, 12:15].T)


def test_chunk_embeddings_gather_by_unique_token():
    prompt = np.array([5, 9, 5, 2, 9], dtype=np.int32)
    unique = np.array([2, 5, 9], dtype=np.int32)
    table = np.arange(3 * 4, dtype=np.uint16).reshape(3, 4)
    rows = pk.chunk_embeddings(prompt, unique, table, start=1, rows=3)
    np.testing.assert_array_equal(rows, table[[2, 1, 0]])
    with pytest.raises(ValueError):
        pk.chunk_embeddings(prompt, unique, table, start=3, rows=3)
