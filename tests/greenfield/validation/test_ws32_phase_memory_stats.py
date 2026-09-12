"""Historical reservation high-water mark versus current phase capacity."""
import pytest

from scripts.greenfield.seal_short_decoder_ws32 import _memory_valid


def sample():
    # Original E0 rank0 after-compile counters; no TPU evidence fabricated.
    return dict(bytes_in_use=29249556992, bytes_limit=33014398976,
                bytes_reservable_limit=3091634176, bytes_reserved=0,
                largest_alloc_size=3277946880, largest_free_block_bytes=3091634176,
                num_allocs=2504, peak_bytes_in_use=29929935360,
                peak_bytes_reserved=3110486016)


def test_long_phase_accepts_historical_peak_above_current_reservable_limit():
    assert _memory_valid(sample(), phase_residency=True)
    assert not _memory_valid(sample())  # Historical policy stays unchanged.


def test_preexisting_valid_record_stays_valid():
    value = sample()
    value['bytes_reservable_limit'] = value['bytes_limit']
    assert _memory_valid(value)
    assert _memory_valid(value, phase_residency=True)


@pytest.mark.parametrize('field,bad', [
    ('bytes_in_use', 29929935361),
    ('peak_bytes_in_use', 33014398977),
    ('peak_bytes_reserved', 33014398977),
    ('bytes_reserved', 3091634177),
    ('bytes_reserved', 3110486017),
    ('bytes_reservable_limit', 33014398977),
    ('bytes_limit', 33014398975),
    ('num_allocs', -1), ('num_allocs', True), ('num_allocs', 1.0),
    ('bytes_reserved', float('nan')),
])
def test_physical_and_current_capacity_limits_not_relaxed(field, bad):
    value = sample(); value[field] = bad
    assert not _memory_valid(value, phase_residency=True)


@pytest.mark.parametrize('bad', [None, {}, [], {'bytes_limit': 33014398976}])
def test_long_phase_requires_complete_actual_counters(bad):
    assert not _memory_valid(bad, phase_residency=True)


def test_current_reservation_must_not_exceed_its_historical_peak():
    value = sample(); value['bytes_reserved'] = 1; value['peak_bytes_reserved'] = 0
    assert not _memory_valid(value, phase_residency=True)
