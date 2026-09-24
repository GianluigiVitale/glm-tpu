"""CPU mechanism admission; not TPU allocation, arithmetic or speed proof."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.pallas.mosaic import tpu_info

from glm_tpu.kernels.fp8_grouped_matmul.panels import (
    build_expert_panels,
    pack_expert_panel_rows,
    unpack_expert_panel_rows,
)
from glm_tpu.kernels.fp8_grouped_matmul.panel_kernel import prefill_panel_fp8_matmul


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


@pytest.mark.parametrize("offset", [0, 2, 4])
def test_exclusive_panels_restore_every_owned_row_once(offset):
    counts = np.array([1, 31, 32, 33, 0, 7], np.int32)
    m = int(counts.sum())
    x = jnp.arange(m * 3, dtype=jnp.float32).reshape(m, 3)
    plan = jax.jit(lambda c, o: build_expert_panels(c, o, rows=m, local_groups=2))(
        jnp.asarray(counts), jnp.int32(offset)
    )
    assert bool(plan.valid)
    assert int(plan.active_panels) == sum(
        (int(c) + 31) // 32 for c in counts[offset : offset + 2]
    )
    packed = pack_expert_panel_rows(x, plan)
    live = np.arange(32)[None, :] < np.asarray(plan.live_counts)[:, None]
    np.testing.assert_array_equal(np.asarray(packed)[~live], 0)
    owned = (np.repeat(np.arange(6), counts) >= offset) & (
        np.repeat(np.arange(6), counts) < offset + 2
    )
    indices = np.asarray(plan.restore_indices)[owned]
    assert len(np.unique(indices)) == int(counts[offset : offset + 2].sum())
    assert live.reshape(-1)[indices].all()
    poisoned = jnp.where(jnp.asarray(live)[..., None], packed, jnp.nan)
    expected = np.where(owned[:, None], np.asarray(x), 0)
    np.testing.assert_array_equal(unpack_expert_panel_rows(poisoned, plan), expected)


def test_invalid_metadata_and_empty_owner_skip_all_work():
    m = 19
    x = jnp.full((m, 128), jnp.nan, jnp.bfloat16)
    w = jnp.full((2, 256, 128), 127, jnp.uint8)  # FP8 NaNs
    scales = jnp.full((2, 2, 1), jnp.nan, jnp.float32)
    base = jnp.array([19, 0, 0, 0], jnp.int32)

    @jax.jit
    def run(c, o):
        p = build_expert_panels(c, o, rows=m, local_groups=2)
        return prefill_panel_fp8_matmul(x, w, scales, p, interpret=True)

    value, valid = run(base, jnp.int32(2))
    assert bool(valid)
    np.testing.assert_array_equal(value, 0)
    for c, o in (
        (base, -1),
        (base, 3),
        (base, 2147483647),
        (base.at[1].set(-1), 0),
        (base.at[0].set(18), 0),
        (base.at[0].set(2147483647), 0),
    ):
        value, valid = run(c, jnp.int32(o))
        assert not bool(valid)
        np.testing.assert_array_equal(value, 0)


def _dequantize_and_dot(x, bits, scales, counts, offset, dtype):
    """Pure-JAX reference: dequantize each local expert to BF16 with its full-K scale table
    (one scale per N128 x K128 block), contract in increasing K128 order with FP32 accumulation;
    rows outside the local experts ``[offset, offset + groups)`` are zero."""
    groups, n, k = bits.shape
    m = x.shape[0]
    local = np.repeat(np.arange(counts.size), counts) - offset
    owned = (local >= 0) & (local < groups)
    expanded = jnp.repeat(jnp.repeat(scales, 128, axis=1), 128, axis=2)
    weights = (
        lax.bitcast_convert_type(bits, jnp.float8_e4m3fn).astype(jnp.float32) * expanded
    ).astype(jnp.bfloat16)
    expert = np.clip(local, 0, groups - 1)
    acc = jnp.zeros((m, n), jnp.float32)
    for ki in range(k // 128):
        block = slice(ki * 128, (ki + 1) * 128)
        partial = jnp.einsum(
            "mk,gnk->gmn",
            x[:, block],
            weights[:, :, block],
            preferred_element_type=jnp.float32,
        )
        acc = acc + partial[expert, np.arange(m)]
    return jnp.where(owned[:, None], acc, 0).astype(dtype), owned


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_full_k_scales_and_n128_stripes_match_dequantize_and_dot(dtype):
    # Random operands; the ninth K block catches accidental reuse of a ki%8 scale slab, and the
    # zero and 3.25 scales sit in that block of two different N128 stripes and experts. Until S2f
    # this compared with the archived grouped FP8 kernel (archive/research-20260922); on jax
    # 0.10.1 CPU the kernel, that kernel and this reference are bitwise equal on these inputs.
    rng = np.random.default_rng(911)
    counts = np.array([1, 33, 7, 0], np.int32)
    m, k, n = 41, 1152, 512
    x = jnp.asarray(rng.normal(0, 0.2, (m, k)), jnp.bfloat16)
    bits = lax.bitcast_convert_type(
        jnp.asarray(rng.normal(0, 0.15, (2, n, k)), jnp.float8_e4m3fn), jnp.uint8
    )
    scales = jnp.asarray(rng.uniform(0.1, 2.0, (2, 4, 9)), jnp.float32)
    scales = scales.at[0, 1, 8].set(0).at[1, 3, 8].set(3.25)

    @jax.jit
    def run(x, bits, scales):
        plan = build_expert_panels(
            jnp.asarray(counts), jnp.int32(1), rows=m, local_groups=2
        )
        return prefill_panel_fp8_matmul(
            x, bits, scales, plan, result_dtype=dtype, interpret=True
        )

    value, ok = run(x, bits, scales)
    expected, owned = _dequantize_and_dot(x, bits, scales, counts, 1, dtype)
    assert bool(ok) and int(owned.sum()) == 40
    assert value.dtype == dtype
    actual = np.asarray(value).astype(np.float64)
    reference = np.asarray(expected).astype(np.float64)
    np.testing.assert_array_equal(actual[~owned], 0)
    # FP32 accumulation order inside one K128 dot may differ by platform: a few FP32 ulps, at
    # most one rounding step of the BF16 result. A wrong scale block or stripe moves an owned
    # value by ~0.1-1 (|value| <= ~7).
    if dtype == jnp.float32:
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
    else:
        np.testing.assert_allclose(actual, reference, rtol=2**-7, atol=2**-9)


def test_all_rows_one_expert_capacity_and_scale_zero():
    counts = jnp.array([0, 129, 0, 0], jnp.int32)
    plan = build_expert_panels(counts, jnp.int32(0), rows=129, local_groups=4)
    assert int(plan.active_panels) == 5
    np.testing.assert_array_equal(np.asarray(plan.live_counts)[:5], [32, 32, 32, 32, 1])
    x = jnp.ones((129, 128), jnp.bfloat16)
    w = jnp.full((4, 256, 128), 56, jnp.uint8)  # 1.0
    s = jnp.ones((4, 2, 1), jnp.float32).at[1, 1, 0].set(0)
    value, ok = jax.jit(
        lambda x: prefill_panel_fp8_matmul(x, w, s, plan, interpret=True)
    )(x)
    assert bool(ok)
    np.testing.assert_array_equal(np.asarray(value)[:, :128], 128)
    np.testing.assert_array_equal(np.asarray(value)[:, 128:], 0)


def test_bad_static_geometry_refuses():
    c = jnp.array([1, 0], jnp.int32)
    for offset in (jnp.array([0], jnp.int32), jnp.float32(0)):
        with pytest.raises(ValueError):
            build_expert_panels(c, offset, rows=1, local_groups=2)
    with pytest.raises(ValueError):
        build_expert_panels(c, jnp.int32(0), rows=True, local_groups=2)
