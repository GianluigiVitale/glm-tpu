"""Paired, warmed target-window latency; deliberately not serving throughput."""
import math
import time

import numpy as np


SCHEMA = 'glm_perf_prefix_timing_v1'
SCOPE = ('Teacher-forced target calls with completion waits; excludes drafting, '
         'acceptance, commit, health votes, delivery and entry barrier.')


def trial_order(index):
    return ['ordinary', 'verify'] if index % 2 == 0 else ['verify', 'ordinary']


def measure_prefix_window(tokens, root, ordinary_predictions, verifier_predictions, *,
                          iterations, ordinary, verify, put, ready, phase, barrier,
                          require, label, clock=time.perf_counter):
    """Caller admits immutable, non-donating executables and owns fleet leases.

    Inputs are transferred before timing. Each trial starts from the same root;
    ordinary executes each row sequentially, completing it before the next call.
    Health and predictions are checked outside the timer, inside a fleet phase.
    Only small prediction/health arrays survive intermediate ordinary rows.
    """
    if type(iterations) is not int or iterations not in (5, 20):
        raise ValueError('prefix timing requires 5 or 20 paired trials')
    inputs = ready(put(tokens))
    row_inputs = [ready(put(tokens[i:i+1])) for i in range(len(tokens))]
    ready(root)

    def ordinary_call():
        current = root
        predictions, health = [], []
        for token in row_inputs:
            result = ready(ordinary(token, current))
            current = result.state
            predictions.append(result.next_token)
            health.append(result.state.contract_valid)
        return predictions, health

    def verify_call():
        result = ready(verify(inputs, root))
        return [result.predictions], [result.contract_valid]

    calls = dict(ordinary=ordinary_call, verify=verify_call)
    expected = dict(ordinary=ordinary_predictions, verify=verifier_predictions)
    output = dict(schema=SCHEMA, iterations=iterations, warmup_pairs=2,
                  scope=SCOPE, measured_speculative_throughput=False,
                  orders=[trial_order(i) for i in range(iterations)],
                  ordinary_seconds=[], verify_seconds=[])
    for warm, count in ((True, 2), (False, iterations)):
        for index in range(count):
            for mode in trial_order(index):
                name = f'{label}_{"warm" if warm else "trial"}_{index}_{mode}'

                def action():
                    barrier(name)
                    start = clock()
                    predictions, health = calls[mode]()
                    elapsed = clock() - start
                    require(math.isfinite(elapsed) and elapsed > 0, 'invalid prefix latency')
                    require(all(np.asarray(x).all() for x in health), 'timed target unhealthy')
                    actual = np.concatenate([np.asarray(x) for x in predictions])
                    require(np.array_equal(actual, expected[mode]), 'timed predictions changed')
                    return elapsed

                elapsed = phase(name, action)
                if not warm:
                    output[mode+'_seconds'].append(elapsed)
    return output


def validate_prefix_timing(value, iterations):
    if (type(value) is not dict or value.get('schema') != SCHEMA
            or type(value.get('iterations')) is not int or value['iterations'] != iterations
            or type(value.get('warmup_pairs')) is not int or value['warmup_pairs'] != 2
            or value.get('scope') != SCOPE
            or value.get('measured_speculative_throughput') is not False
            or value.get('orders') != [trial_order(i) for i in range(iterations)]):
        raise ValueError('missing or differently scoped prefix timings')
    for mode in ('ordinary', 'verify'):
        samples = value.get(mode+'_seconds')
        if (type(samples) is not list or len(samples) != iterations
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples)):
            raise ValueError('invalid prefix timing samples')


def summarize_prefix_timings(values, iterations):
    """A fleet trial is bounded by its slowest rank, not eight independent runs."""
    if len(values) != 8:
        raise ValueError('prefix timing requires all eight ranks')
    for value in values:
        validate_prefix_timing(value, iterations)
    return dict(schema=SCHEMA, scope=SCOPE, iterations=iterations, warmup_pairs=2,
        measured_speculative_throughput=False, aggregation='maximum rank duration per paired trial',
        **{mode+'_seconds': [max(v[mode+'_seconds'][i] for v in values)
                            for i in range(iterations)] for mode in ('ordinary', 'verify')})
