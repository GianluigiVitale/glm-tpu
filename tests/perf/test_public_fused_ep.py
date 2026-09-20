"""Lossless public-kernel layout adaptation; no TPU or trained-parity claim."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.perf.public_fused_ep import pack_weights, pad_rows, load_public_candidate


def test_weight_and_scale_layout_decodes_to_original_values():
    rng = np.random.default_rng(741)
    values = []
    originals = []
    for shape in ((3, 256, 512), (3, 256, 512), (3, 512, 256)):
        fp8 = jnp.asarray(rng.normal(size=shape), jnp.float8_e4m3fn)
        bits = jax.lax.bitcast_convert_type(fp8, jnp.uint8)
        scale = jnp.asarray(rng.uniform(.1, 1.9, (shape[0], shape[1]//128, shape[2]//128)), jnp.float32)
        values.extend((bits, scale))
        expanded = np.repeat(np.repeat(np.asarray(scale), 128, 1), 128, 2)
        originals.append(jnp.asarray(np.asarray(fp8).astype(np.float32)*expanded, jnp.bfloat16))
    result = jax.jit(pack_weights)(*values)
    assert result.w1.shape == (3, 512, 512)
    assert result.w1_scale.shape == (3, 4, 1, 512)
    assert result.w2.shape == (3, 256, 512)
    assert result.w2_scale.shape == (3, 2, 1, 512)
    decoded1 = (result.w1.astype(jnp.float32)*jnp.repeat(result.w1_scale[:, :, 0], 128, axis=1)).astype(jnp.bfloat16)
    decoded2 = (result.w2.astype(jnp.float32)*jnp.repeat(result.w2_scale[:, :, 0], 128, axis=1)).astype(jnp.bfloat16)
    expected1 = jnp.concatenate((originals[0].transpose(0, 2, 1), originals[1].transpose(0, 2, 1)), axis=2)
    np.testing.assert_array_equal(decoded1, expected1)
    np.testing.assert_array_equal(decoded2, originals[2].transpose(0, 2, 1))
    # Bit transport must also preserve every original FP8 bit, not just roundtrip values.
    np.testing.assert_array_equal(jax.lax.bitcast_convert_type(result.w1[:, :, :256], jnp.uint8), values[0].transpose(0, 2, 1))
    np.testing.assert_array_equal(jax.lax.bitcast_convert_type(result.w1[:, :, 256:], jnp.uint8), values[2].transpose(0, 2, 1))
    np.testing.assert_array_equal(jax.lax.bitcast_convert_type(result.w2, jnp.uint8), values[4].transpose(0, 2, 1))


@pytest.mark.parametrize('rows', [2, 3, 32, 128])
def test_live_rows_survive_padding_and_dummy_contributions_are_zero(rows):
    h = jnp.arange(rows*128).reshape(rows, 128).astype(jnp.bfloat16)
    routes = (jnp.arange(rows*8).reshape(rows, 8) % 256).astype(jnp.int32)
    weights = jnp.full((rows, 8), .125, jnp.float32)
    x, ids, w = jax.jit(pad_rows)(h, routes, weights)
    assert x.shape[0] == ((rows+31)//32)*32
    for actual, expected in ((x, h), (ids, routes), (w, weights)):
        np.testing.assert_array_equal(actual[:rows], expected)
    assert not np.asarray(x[rows:]).any() and not np.asarray(w[rows:]).any()
    for row in np.asarray(ids[rows:]):
        assert len(set(row.tolist())) == 8


def test_untrusted_public_source_is_refused_before_import(tmp_path):
    (tmp_path/'fused_moe_rs.py').write_text("raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match='digest differs'):
        load_public_candidate(tmp_path)


def test_bad_weight_layout_is_refused():
    b = jnp.zeros((1, 128, 128), jnp.uint8)
    s = jnp.ones((1, 1, 1), jnp.float32)
    with pytest.raises(ValueError, match='layout differs'):
        pack_weights(b, s.astype(jnp.bfloat16), b, s, b, s)
