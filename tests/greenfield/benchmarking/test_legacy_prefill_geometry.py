"""CPU unit coverage for the legacy prefill-geometry primitives."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from glm_tpu.greenfield.benchmarking import legacy_prefill_geometry as geo  # noqa: E402


def _bf16(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def test_dequantize_matches_legacy_formula_and_refuses_bad_scale():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 120, size=(256, 130), dtype=np.uint8)  # avoid NaN codes
    scale = rng.uniform(0.5, 2.0, size=(2, 2)).astype(np.float32)
    out = np.asarray(geo.legacy_dequantize_fp8(jnp.asarray(bits), jnp.asarray(scale))).astype(np.float32)
    fp8 = bits.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    expanded = scale[(np.arange(256) // 128)[:, None], (np.arange(130) // 128)[None, :]]
    np.testing.assert_array_equal(out, _bf16(fp8 * expanded))
    with pytest.raises(ValueError):
        geo.legacy_dequantize_fp8(jnp.asarray(bits), jnp.asarray(scale[:1]))
    # per-column scales: [Kb, N]
    column_scale = rng.uniform(0.5, 2.0, size=(2, 130)).astype(np.float32)
    out2 = np.asarray(geo.legacy_dequantize_fp8(jnp.asarray(bits), jnp.asarray(column_scale))).astype(np.float32)
    np.testing.assert_array_equal(out2, _bf16(fp8 * column_scale[np.arange(256) // 128, :]))


def test_input_norm_rounding_points_and_mask():
    rng = np.random.default_rng(2)
    x = _bf16(rng.normal(size=(3, geo.HIDDEN)))
    w = _bf16(rng.normal(size=(geo.HIDDEN,)))
    valid = np.array([True, True, False])
    out = np.asarray(
        geo.legacy_input_norm(
            jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.bfloat16), jnp.asarray(valid)
        )
    ).astype(np.float32)
    square = (x * x).sum(axis=1, dtype=np.float32)
    inv = 1.0 / np.sqrt(square * np.float32(geo.HIDDEN_INVERSE) + np.float32(1e-5))
    expected = _bf16(_bf16(x * inv[:, None]) * w[None, :])
    assert np.isnan(out[2]).all()
    # FP32 summation order differs between XLA CPU and NumPy; allow one BF16 ulp.
    np.testing.assert_allclose(out[:2], expected[:2], rtol=2**-7, atol=0)


def test_fused_qkv_a_and_q_a_norm_shapes_and_owner_order():
    rng = np.random.default_rng(3)
    normalized = jnp.asarray(_bf16(rng.normal(size=(4, geo.HIDDEN))), jnp.bfloat16)
    bits = jnp.asarray(rng.integers(0, 120, size=(32, geo.HIDDEN, 82), dtype=np.uint8))
    scale = jnp.asarray(rng.uniform(0.5, 1.5, size=(32, 48, 82)).astype(np.float32))
    projected = geo.legacy_fused_qkv_a(normalized, bits, scale)
    assert projected.shape == (32, 4, 82) and projected.dtype == jnp.bfloat16
    q = geo.legacy_q_a_norm(projected, jnp.asarray(_bf16(rng.normal(size=(2048,))), jnp.bfloat16))
    assert q.shape == (4, 2048) and q.dtype == jnp.bfloat16
    kv = geo.legacy_kv_a_lanes(projected)
    assert kv.shape == (4, 576)
    # owner o occupies lanes [64o, 64o+64) of q and [18o, 18o+18) of kv
    p = np.asarray(projected).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(kv).astype(np.float32)[:, 18 * 5 : 18 * 6], p[5, :, 64:82])
    with pytest.raises(ValueError):
        geo.legacy_fused_qkv_a(normalized, bits[:31], scale[:31])


def test_swiglu_is_fp32_sigmoid_then_bf16():
    rng = np.random.default_rng(4)
    gu = _bf16(rng.normal(size=(5, 768)))
    out = np.asarray(geo.legacy_swiglu(jnp.asarray(gu, jnp.bfloat16), half_width=384)).astype(np.float32)
    g, u = gu[:, :384], gu[:, 384:]
    expected = _bf16(g * (np.float32(1.0) / (np.exp(-g) + np.float32(1.0))) * u)
    np.testing.assert_allclose(out, expected, rtol=2**-7, atol=0)


def test_residual_associations_follow_the_legacy_order():
    rng = np.random.default_rng(5)
    attn = _bf16(rng.normal(size=(2, geo.HIDDEN)))
    emb = _bf16(rng.normal(size=(2, geo.HIDDEN)))
    dense = _bf16(rng.normal(size=(2, geo.HIDDEN)))
    w = _bf16(rng.normal(size=(geo.HIDDEN,)))
    carried, normalized = geo.legacy_post_attention_norm(
        jnp.asarray(attn, jnp.bfloat16), jnp.asarray(emb, jnp.bfloat16), jnp.asarray(w, jnp.bfloat16)
    )
    np.testing.assert_array_equal(np.asarray(carried).astype(np.float32), _bf16(attn + emb))
    assert normalized.shape == (2, geo.HIDDEN)
    carried1, normalized1 = geo.legacy_layer1_input_norm(
        jnp.asarray(dense, jnp.bfloat16),
        jnp.asarray(attn, jnp.bfloat16),
        jnp.asarray(emb, jnp.bfloat16),
        jnp.asarray(w, jnp.bfloat16),
    )
    np.testing.assert_array_equal(
        np.asarray(carried1).astype(np.float32), _bf16(dense + _bf16(attn + emb))
    )
    assert normalized1.dtype == jnp.bfloat16


def test_owner_partials_and_tree_reduce():
    rng = np.random.default_rng(6)
    lhs = jnp.asarray(_bf16(rng.normal(size=(32, 2, 128))), jnp.bfloat16)
    bits = jnp.asarray(rng.integers(0, 120, size=(32, 128, geo.HIDDEN), dtype=np.uint8))
    scale = jnp.asarray(rng.uniform(0.5, 1.5, size=(32, 1, 48)).astype(np.float32))
    partials = geo.legacy_owner_partials_convolution(lhs, bits, scale)
    assert partials.shape == (32, 2, geo.HIDDEN) and partials.dtype == jnp.bfloat16
    reduced = geo.bf16_pairwise_tree_reduce(partials)
    assert reduced.shape == (2, geo.HIDDEN)
    identity = lambda rows: rows[0]  # noqa: E731  (32,1,6144) -> (1,6144)
    picked = geo.strategy_nd_row_association_reduce(partials, identity)
    np.testing.assert_array_equal(np.asarray(picked), np.asarray(partials[0]))
