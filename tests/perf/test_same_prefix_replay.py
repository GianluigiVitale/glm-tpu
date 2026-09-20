"""Fault injection for replay reset, prefix restoration and reference checks."""
from typing import NamedTuple

import numpy as np
import pytest

from glm_tpu.perf.speculative_diagnostics import compare_same_prefix


class State(NamedTuple):
    position: np.ndarray
    cache: np.ndarray
    contract_valid: np.ndarray


class Result(NamedTuple):
    state: State
    next_token: np.ndarray
    final_residual_local: np.ndarray


class Proposal(NamedTuple):
    cache: np.ndarray
    predictions: np.ndarray
    final_residual_local: np.ndarray
    contract_valid: np.ndarray


def run(*, rows=3, offsets=(0, 2, 5), drift=False, bad_commit=False,
        bad_prediction=False, bad_reference=False, unhealthy=False, measure_window=None):
    initial = State(np.array([3], np.int32), np.full(32, -7, np.int32), np.array([True]))
    roots = []
    expected = np.arange(10, 22, dtype=np.int32)

    def ordinary(token, state):
        cache = state.cache.copy()
        cache[state.position[0]] = token[0]
        return Result(state._replace(position=state.position+1, cache=cache),
                      token+1+int(bad_reference), token.reshape(1, 1))

    def verify(tokens, state):
        roots.append(state.cache.copy())
        cache = state.cache.copy()
        start = state.position[0]
        cache[start:start+rows] = tokens + int(drift)
        return Proposal(cache, tokens+1+int(bad_prediction), tokens[:, None],
                        np.array([not unhealthy]))

    def commit(state, proposal, count):
        cache = proposal.cache.copy()
        end = state.position[0]+int(count)
        if not bad_commit:
            cache[end:] = state.cache[end:]
        return state._replace(position=np.array([end], np.int32), cache=cache)

    def healthy(value, label):
        if not value.all():
            raise RuntimeError(label)

    report = compare_same_prefix(expected, initial, rows=rows, offsets=offsets,
        verify=verify, commit=commit, ordinary=ordinary, replicate=np.asarray,
        ready=lambda x: x, healthy=healthy,
        compare=lambda a, b: dict(bitwise_equal=bool(np.array_equal(a, b))),
        measure_window=measure_window)
    return report, roots, initial


@pytest.mark.parametrize('rows', [1, 2, 3, 4])
def test_all_prefixes_and_overlapping_windows(rows):
    report, roots, initial = run(rows=rows)
    assert report['ordinary_reference_equal']
    assert report['first_prediction_mismatch'] is None
    assert len(roots) == len(report['windows']) == 3
    for window in report['windows']:
        assert window['residual_comparison']['bitwise_equal']
        assert len(window['prefixes']) == rows+1
        assert all(c['bitwise_equal'] for p in window['prefixes']
                   for c in p['state_comparisons'].values())
    assert np.all(initial.cache == -7)
    assert not report['measured_speculative_throughput']


def test_candidate_cache_drift_cannot_contaminate_next_window():
    report, roots, _ = run(drift=True)
    np.testing.assert_array_equal(roots[1][3:5], [10, 11])
    np.testing.assert_array_equal(roots[2][3:8], np.arange(10, 15))
    assert all(not w['prefixes'][-1]['state_comparisons']['cache']['bitwise_equal']
               for w in report['windows'])
    assert all(w['prefixes'][0]['state_comparisons']['cache']['bitwise_equal']
               for w in report['windows'])


def test_unrestored_future_rows_detected_even_when_predictions_match():
    report, _, _ = run(bad_commit=True)
    assert report['first_prediction_mismatch'] is None
    for w in report['windows']:
        assert not w['prefixes'][0]['state_comparisons']['cache']['bitwise_equal']
        assert w['prefixes'][-1]['state_comparisons']['cache']['bitwise_equal']


def test_prediction_and_historical_reference_disagreement_are_separate():
    report, _, _ = run(bad_prediction=True)
    assert report['ordinary_reference_equal'] and report['first_prediction_mismatch'] == 1
    report, _, _ = run(bad_reference=True)
    assert not report['ordinary_reference_equal'] and report['first_prediction_mismatch'] == 1


def test_unhealthy_proposal_stops_before_commit():
    with pytest.raises(RuntimeError, match='verify_0'):
        run(unhealthy=True)


@pytest.mark.parametrize('offsets', [(2, 0), (0, 0), (-1,), (9,), ()])
def test_refuse_invalid_or_padded_windows(offsets):
    with pytest.raises(ValueError, match='ordered live windows'):
        run(offsets=offsets)


def test_optional_timing_receives_ordinary_root_and_separate_predictions():
    observed=[]

    def measure(start,tokens,root,baseline,predicted):
        assert root.position[0] == 3+start
        np.testing.assert_array_equal(root.cache[3:3+start],np.arange(10,10+start))
        np.testing.assert_array_equal(baseline,tokens+1)
        np.testing.assert_array_equal(predicted,tokens+2)
        observed.append(start)
        return dict(diagnostic_only=True)

    report,_,_=run(drift=True,bad_prediction=True,measure_window=measure)
    assert observed==[0,2,5]
    assert report['first_prediction_mismatch']==1
    assert all(w['timing']==dict(diagnostic_only=True) for w in report['windows'])
