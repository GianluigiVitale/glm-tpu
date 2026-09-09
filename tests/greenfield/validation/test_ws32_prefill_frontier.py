"""Original-byte capsule tests, no TPU/model correctness claim."""

from copy import deepcopy

import ml_dtypes
import numpy as np
import pytest

from scripts.greenfield.ws32_prefill_frontier import (
    capture_cache, compare_cache, replay_cache,
)


def fixture(slot=0):
    table = np.asarray([[1, 0]], np.int32)
    cache = np.zeros((2, 2, 64, 128), ml_dtypes.bfloat16)
    if slot // 4 < 2:
        cache[:, 1] = np.asarray(0.125, ml_dtypes.bfloat16)
    return cache, dict(slot=slot, block_table=table, layer_ids=(0, 6))


@pytest.mark.parametrize("slot", range(32))
def test_all_slots_page_mapping_and_original_reconstruction(slot):
    value, kwargs = fixture(slot)
    rows, record = capture_cache(value, **kwargs)
    restored = replay_cache(rows, record)
    expected = list(range((slot // 4)*64, (slot // 4+1)*64)) if slot // 4 < 2 else []
    assert restored["owner_positions"] == expected
    assert record["physical_pages"] == [1]*len(expected)
    assert restored["untouched_bytes_zero"]
    assert compare_cache(rows, record, rows.copy(), deepcopy(record))["bytes_equal"]


def test_signed_zero_difference_is_not_hidden_by_float_equality():
    value, kwargs = fixture()
    value[1, 1, 4, 9] = 0
    first, a = capture_cache(value, **kwargs)
    value[1, 1, 4, 9] = np.asarray(-0.0, ml_dtypes.bfloat16)
    second, b = capture_cache(value, **kwargs)
    result = compare_cache(first, a, second, b)
    assert not result["bytes_equal"]
    assert result["earliest_differing_writer"] == 6
    writer = result["differing_writers"][0]
    assert writer["first_position"] == 4 and writer["first_component"] == 9
    assert writer["first_wide_bits"] == 0 and writer["first_narrow_bits"] == 32768
    assert writer["max_abs"] == 0
    assert not result["numerical_promotion"]


@pytest.mark.parametrize("bad", [1.0, -0.0, float("nan")])
def test_uncaptured_future_write_refuses_even_negative_zero(bad):
    value, kwargs = fixture()
    value[0, 0, 3, 2] = bad  # Physicalpage0 is logicalpage1, not written.
    with pytest.raises(ValueError, match="outside rows"):
        capture_cache(value, **kwargs)


def test_nonfinite_live_values_refuse():
    value, kwargs = fixture()
    value[0, 1, 0, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        capture_cache(value, **kwargs)


@pytest.mark.parametrize("field,value", [
    ("frontier", True), ("frontier", 256), ("slot", True),
    ("slot", 32), ("block_table", [[0, 0]]),
    ("block_table", [[True, 0]]), ("block_table", [[2**40, 0]]),
    ("layer_ids", [6, 0]), ("layer_ids", [0, 0]),
    ("shape", [2, 17, 64, 128]), ("shape", [2, 2, 65, 128]),
    ("positions", [False]*64), ("local_rows", [0]*64),
    ("dtype", "float16"), ("whole_cache_sha256", "0"*64),
])
def test_mutated_schema_and_metadata_refuse(field, value):
    cache, kwargs = fixture()
    rows, record = capture_cache(cache, **kwargs)
    record[field] = value
    with pytest.raises(ValueError):
        replay_cache(rows, record)


def test_mutated_original_bits_refuse():
    value, kwargs = fixture()
    rows, record = capture_cache(value, **kwargs)
    rows[0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="row bits"):
        replay_cache(rows, record)


def test_mixed_owner_comparison_refuses():
    a, ka = fixture(0)
    b, kb = fixture(4)
    ar, am = capture_cache(a, **ka)
    br, bm = capture_cache(b, **kb)
    with pytest.raises(ValueError, match="mixes owners"):
        compare_cache(ar, am, br, bm)


def test_production_kv_shape_with_last_layer_difference():
    value = np.zeros((78, 16, 64, 640), ml_dtypes.bfloat16)
    kwargs = dict(slot=4, block_table=np.arange(16,dtype=np.int32)[None],
                  layer_ids=tuple(range(78)))
    a, am = capture_cache(value, **kwargs)
    value[77, 0, 63, 511] = 0.25
    b, bm = capture_cache(value, **kwargs)
    changed = compare_cache(a, am, b, bm)
    assert changed["earliest_differing_writer"] == 77
    assert changed["differing_writers"][0]["first_position"] == 127
    assert b.nbytes == 78*64*640*2
