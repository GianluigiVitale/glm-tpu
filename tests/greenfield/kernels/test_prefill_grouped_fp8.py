"""Interpreted ragged raw-FP8 admission before any TPU layer execution."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.pallas.mosaic import tpu_info

from glm_tpu.greenfield.kernels.pallas.prefill_grouped_fp8 import (
    prefill_grouped_fp8_matmul,
)
from glm_tpu.greenfield.kernels.pallas.fp8_matmul import fp8_block_matmul_f32


@pytest.fixture(autouse=True)
def cpu_tpu_interpretation(monkeypatch):
    assert jax.default_backend() == "cpu"
    monkeypatch.setitem(
        tpu_info.registry,
        "cpu",
        lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1),
    )
    tpu_info.get_tpu_info.cache_clear()
    yield
    tpu_info.get_tpu_info.cache_clear()


def inputs():
    rng = np.random.default_rng(243)
    lhs = jnp.asarray(rng.normal(0, 0.25, (19, 256)), jnp.bfloat16).at[7].set(0)
    weights = jnp.asarray(rng.normal(0, 0.15, (4, 128, 256)), jnp.float8_e4m3fn)
    bits = lax.bitcast_convert_type(weights, jnp.uint8)
    scales = (
        jnp.asarray(rng.uniform(0.25, 2.0, (4, 1, 2)), jnp.float32).at[1, 0, 1].set(0)
    )
    return lhs, bits, scales


@pytest.mark.parametrize("offset", [0, 4])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_grouped_matches_per_row_projection_with_shared_tail_tiles(offset, dtype):
    lhs, bits, scales = inputs()
    counts = np.asarray([1, 2, 0, 3, 4, 1, 0, 8], np.int32)
    call = jax.jit(
        lambda x, w, s, c, o: prefill_grouped_fp8_matmul(
            x, w, s, c, o, result_dtype=dtype, interpret=True
        )
    )
    actual, valid = call(lhs, bits, scales, jnp.asarray(counts), jnp.int32(offset))
    assert bool(valid)
    groups = np.repeat(np.arange(8), counts)
    project = jax.jit(lambda x, w, s: fp8_block_matmul_f32(x, w, s, interpret=True))
    expected = []
    for i, g in enumerate(groups):
        if offset <= g < offset + 4:
            expected.append(
                project(lhs[i : i + 1], bits[g - offset], scales[g - offset])
            )
        else:
            expected.append(jnp.zeros((1, 128), jnp.float32))
    reference = jnp.concatenate(expected).astype(dtype)
    view = np.uint32 if dtype == jnp.float32 else np.uint16
    np.testing.assert_array_equal(
        np.asarray(actual).view(view), np.asarray(reference).view(view)
    )
    assert np.isfinite(np.asarray(actual, dtype=np.float32)).all()


def test_empty_owner_and_invalid_counts_have_defined_zero_outputs():
    lhs, bits, scales = inputs()
    call = jax.jit(
        lambda c, o: prefill_grouped_fp8_matmul(lhs, bits, scales, c, o, interpret=True)
    )
    counts = jnp.asarray([19, 0, 0, 0, 0, 0, 0, 0], jnp.int32)
    result, valid = call(counts, jnp.int32(4))
    assert bool(valid)
    np.testing.assert_array_equal(result, 0)
    for c, o in (
        (counts.at[1].set(-1), 0),
        (counts.at[0].set(18), 0),
        (counts, 5),
        (counts, -1),
        (counts, 2147483647),
        (counts, 2147483644),
    ):
        result, valid = call(c, jnp.int32(o))
        assert not bool(valid)
        np.testing.assert_array_equal(result, 0)


def test_multiple_output_blocks_and_scale_slab_boundary():
    rng = np.random.default_rng(401)
    lhs = jnp.asarray(rng.normal(0, 0.2, (17, 1152)), jnp.bfloat16)
    bits = lax.bitcast_convert_type(
        jnp.asarray(rng.normal(0, 0.15, (2, 256, 1152)), jnp.float8_e4m3fn), jnp.uint8
    )
    scales = (
        jnp.asarray(rng.uniform(0.1, 2.0, (2, 2, 9)), jnp.float32).at[0, 1, 8].set(0)
    )
    counts = jnp.asarray([7, 10], jnp.int32)
    actual, valid = jax.jit(
        lambda x, w, s: prefill_grouped_fp8_matmul(
            x, w, s, counts, jnp.int32(0), interpret=True
        )
    )(lhs, bits, scales)
    assert bool(valid)
    project = jax.jit(lambda x, w, s: fp8_block_matmul_f32(x, w, s, interpret=True))
    expected = jnp.concatenate(
        [project(lhs[:7], bits[0], scales[0]), project(lhs[7:], bits[1], scales[1])]
    )
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32)
    )
