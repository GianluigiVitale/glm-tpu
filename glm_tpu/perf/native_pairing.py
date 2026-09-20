"""Order native comparison requests and pair results after all modes finish."""
from hashlib import sha256
import math
import re

import numpy as np


COMPARISON_PROTOCOL = 'fresh_prefill_warmed_v1'

# Short DB610 diagnostic only. Reverse both configuration and mode order on
# repeat two; each adjacent comparison changes just one synchronization choice.
NATIVE_CONTROLS = tuple(
    (f'{name}_repeat{repeat}', timing, acceptance)
    for repeat in (1, 2)
    for name, timing, acceptance in
        (('host_blocking', 'blocking', 'host'), ('host_none', 'none', 'host'),
         ('device_none', 'none', 'device'))[::1 if repeat == 1 else -1]
)


def comparison_order(label, policy):
    if policy not in ('ordinary_first', 'alternating'):
        raise ValueError('unknown native comparison order policy')
    forward = ('ordinary', 'r2', 'r3')
    if policy == 'ordinary_first' or label in ('db610', 'question'):
        return forward
    match = re.fullmatch(r'[a-z][a-z0-9_]{0,23}_repeat([1-3])', label)
    if match is None:
        raise ValueError('alternating native order requires an authenticated repeat label')
    return forward if int(match[1]) % 2 else forward[::-1]


def run_paired_modes(label, *, order_policy, ordinary, speculative, observe):
    """Callbacks create fresh request roots; output pairing is outside their timers.

    Ordinary may run last. Hold only its small private token trail and the two
    speculative trails, never a previous request's device state. Callback
    failures abort without retrying or running another mode.
    """
    reports, generated = {}, {}
    order = comparison_order(label, order_policy)
    for mode in order:
        tokens, report = ordinary() if mode == 'ordinary' else speculative(int(mode[1:]))
        if (not isinstance(tokens, np.ndarray) or tokens.ndim != 1 or tokens.dtype != np.int32
                or tokens.size < 2 or sha256(tokens.tobytes()).hexdigest() != report['token_sha256']
                or not math.isfinite(report['tokens_per_second']) or report['tokens_per_second'] <= 0):
            raise ValueError('native comparison callback returned invalid tokens or rate')
        # Preserve completed output even if a subsequent callback reuses a host buffer.
        generated[mode] = tokens.copy()
        reports[mode] = report
        observe(mode, report, tuple(reports))
    baseline, base = generated['ordinary'], reports['ordinary']
    for mode in ('r2', 'r3'):
        tokens, result = generated[mode], reports[mode]
        common = min(len(tokens), len(baseline))
        mismatch = np.flatnonzero(tokens[:common] != baseline[:common])
        result['ordinary_agreement'] = dict(
            all_equal=bool(np.array_equal(tokens, baseline)),
            first_mismatch=int(mismatch[0]) if len(mismatch) else (
                common if len(tokens) != len(baseline) else None),
            baseline_token_sha256=base['token_sha256'], multirow_numerical_boundary=True)
        result['paired_wall_speedup'] = result['tokens_per_second'] / base['tokens_per_second']
    return reports
