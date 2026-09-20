from types import SimpleNamespace

import numpy as np
import pytest

from glm_tpu.perf.prefix_timing import measure_prefix_window


def run(*, fail=None):
    events = []
    root = SimpleNamespace(position=10, contract_valid=np.array([True]))
    tokens = np.array([20, 21, 22], np.int32)
    calls = dict(ordinary=0, verify=0)

    def ordinary(token, state):
        calls['ordinary'] += 1
        assert state.position == 10 + int(token[0]) - 20
        events.append('ordinary')
        return SimpleNamespace(next_token=token+1, state=SimpleNamespace(
            position=state.position+1, contract_valid=np.array([True])))

    def verify(inputs, state):
        calls['verify'] += 1
        assert state is root
        events.append('verify')
        return SimpleNamespace(predictions=inputs+(2 if fail=='prediction' else 1),
            contract_valid=np.array([fail!='health']))

    counter = iter(range(1000))

    def clock():
        events.append('clock')
        return next(counter)

    def require(condition, message):
        events.append('require')
        if not condition:
            raise ValueError(message)

    def phase(label, action):
        events.append(label)
        return action()

    def ready(value):
        events.append('ready')
        return value

    report = measure_prefix_window(tokens, root, tokens+1, tokens+1, iterations=5,
        ordinary=ordinary, verify=verify, put=lambda x:x.copy(), ready=ready,
        phase=phase, barrier=lambda name:events.append('barrier'), require=require,
        label='window', clock=clock)
    assert root.position == 10
    return report, events, calls


def test_paired_trials_reset_state_warm_both_paths_and_exclude_checks():
    report, events, calls = run()
    assert calls == dict(ordinary=21, verify=7)  # 2 warm pairs + 5 measured
    assert report['ordinary_seconds'] == report['verify_seconds'] == [1]*5
    assert not report['measured_speculative_throughput']
    assert [e for e in events if e.startswith('window_trial')] == [
        f'window_trial_{i}_{m}' for i in range(5)
        for m in (('ordinary','verify') if i%2==0 else ('verify','ordinary'))]
    for i, event in enumerate(events):
        if event == 'barrier':
            assert events[i+1] == 'clock'
            stop = events.index('clock', i+2)
            assert 'require' not in events[i+2:stop]
            assert events[stop-1] == 'ready'
            assert events[stop+1] == 'require'


@pytest.mark.parametrize('failure', ['prediction', 'health'])
def test_timed_execution_checks_predictions_and_health(failure):
    with pytest.raises(ValueError, match='timed'):
        run(fail=failure)
