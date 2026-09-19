"""Fixed-reference request-loop measurement with private token arrays.

The caller owns model graph/memory admission, fleet collectives and workload
leases. This helper measures the existing host loops, including their delivery
votes, against one immutable completed prefill result. No transport or serving
admission is implied by its in-memory sink.
"""
from hashlib import sha256
import time

import numpy as np

from ..greenfield.runtime.ws32_request_session import Ws32RequestSession
from .request_loop import PackedRequestSession


def measure_request_trail(policy, prefill_result, expected, *, decode_step,
        packed, replicate_uniform, fleet_all, warm_steps=5, clock=time.perf_counter):
    """Return private token IDs, aggregate report, final state and last result."""
    if (type(packed) is not bool or type(warm_steps) is not int or warm_steps < 0
            or expected.dtype != np.int32 or expected.shape != (policy.max_new_tokens,)
            or policy.max_new_tokens <= warm_steps+1):
        raise ValueError('request-loop diagnostic reference/options disagree')
    events, votes, transfers, last = [], [], [], [None]
    def step(*args):
        result = decode_step(*args)
        last[0] = result.decoded if packed else result
        return result
    def vote(valid):
        started = clock()
        result = fleet_all(valid)
        votes.append(clock()-started)
        return result
    def replicate(value):
        transfers.append(True)
        return replicate_uniform(value)
    session = (PackedRequestSession if packed else Ws32RequestSession)(policy,
        decode_step=step,replicate_uniform=replicate,fleet_all=vote,deliver=events.append,
        delivery_boundary='research in-memory event append; no transport',
        request_started=clock(),clock=clock)
    session.accept_prefill(prefill_result)
    for _ in range(warm_steps):
        if session.finished:break
        session.step()
    votes.clear(); transfers.clear()
    measured = []
    warmed = len(session.decode_seconds)
    started = clock()
    while not session.finished:
        before = clock()
        session.step()
        measured.append(clock()-before)
    elapsed = clock()-started
    values = np.asarray([e.token_id for e in events],np.int32)
    mismatch = np.flatnonzero(values != expected[:len(values)])
    all_equal = len(values)==len(expected) and not mismatch.size
    first = (int(mismatch[0]) if mismatch.size else
             (None if all_equal else len(values)))
    report = dict(packed=packed,healthy=not session.failed,emitted=len(values),
        decode_steps=len(session.decode_seconds),warm_steps=warmed,samples=len(measured),
        finish_reason=events[-1].finish_reason,wall_seconds=elapsed,
        tokens_per_second=len(measured)/elapsed if measured and elapsed>0 else None,
        p50_ms=float(np.median(measured)*1e3) if measured else None,
        p99_ms=float(np.percentile(measured,99)*1e3) if measured else None,
        model_step_p50_ms=float(np.median(session.decode_seconds[warmed:])*1e3) if measured else None,
        timed_votes=len(votes),timed_uniform_transfers=len(transfers),vote_wall_seconds=sum(votes),
        token_comparison=dict(compared=len(expected),matches=int(np.sum(values==expected[:len(values)])),
            all_equal=all_equal,first_mismatch_index=first,observed_sha256=sha256(values.tobytes()).hexdigest()),
        delivery_boundary='research in-memory event append; no transport',
        excludes_prefill_and_compile=True)
    state = session._state
    session.release()
    return values,report,state,last[0]
