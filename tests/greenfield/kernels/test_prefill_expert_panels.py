"""CPU mechanism admission; not TPU allocation, arithmetic or speed proof."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info

from glm_tpu.optimized.prefill_expert_panels import (
    build_expert_panels,
    pack_expert_panel_rows,
    unpack_expert_panel_rows,
)
from glm_tpu.optimized.prefill_panel_fp8 import prefill_panel_fp8_matmul


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
