"""Fresh greedy question measurement through the guarded packed request loop.

Private input/output stays with the caller. No quality score or speculative
speedup is inferred from timing a single ordinary continuation.
"""
from hashlib import sha256
import json
import time

import numpy as np

from ..user_request import TOKENIZER_FILES, TEMPLATE_SHA, read_bounded
from ..greenfield.runtime.ws32_request_session import RequestPolicy
from .request_loop import PackedRequestSession


def load_question(path, digest, *, capacity, vocab_size, eos_ids):
    raw = read_bounded(path, 1 << 20)
    if sha256(raw).hexdigest() != digest:
        raise ValueError('question file digest differs')
    value = json.loads(raw)
    required = {'schema','request_id','prompt_ids','prompt_ids_sha256','max_new_tokens',
                'tokenizer_files','chat_template_sha256','thinking','decode_policy'}
    if (type(value) is not dict or set(value) != required
            or value['schema'] != 'glm_perf_question_v1'
            or value['tokenizer_files'] != TOKENIZER_FILES
            or value['chat_template_sha256'] != TEMPLATE_SHA
            or value['thinking'] != 'on/max' or value['decode_policy'] != 'greedy'
            or type(value['prompt_ids']) is not list or not value['prompt_ids']
            or any(type(x) is not int or not 0 <= x < vocab_size for x in value['prompt_ids'])):
        raise ValueError('question schema, tokenization identity or greedy policy differs')
    ids = np.asarray(value['prompt_ids'], np.int32)
    if sha256(ids.tobytes()).hexdigest() != value['prompt_ids_sha256']:
        raise ValueError('question token digest differs')
    policy = RequestPolicy(value['request_id'], 0, len(ids), value['max_new_tokens'],
                           capacity, vocab_size, tuple(eos_ids))
    return ids, policy


def question_blocks(ids):
    """Use the already admitted B128/B114 graphs, with explicit live counts."""
    if ids.ndim != 1 or ids.dtype != np.int32 or not ids.size:
        raise ValueError('question requires nonempty int32 prompt')
    result = []
    for start in range(0, len(ids), 128):
        block = ids[start:start+128]
        physical = 114 if len(block) <= 114 else 128
        result.append((np.pad(block, (0,physical-len(block)), constant_values=-1),len(block)))
    return result


def measure_question(policy, prefill_result, *, decode_step, replicate_uniform,
                     fleet_all, deliver, request_started, clock=time.perf_counter):
    events, timings, votes = [], [], []
    def sink(event):
        deliver(event)
        events.append(event)
    def vote(valid):
        start = clock()
        result = fleet_all(valid)
        votes.append(clock()-start)
        return result
    session = PackedRequestSession(policy, decode_step=decode_step,
        replicate_uniform=replicate_uniform, fleet_all=vote, deliver=sink,
        delivery_boundary='rank0 private JSONL token write+flush; no network transport',
        request_started=request_started, clock=clock)
    session.accept_prefill(prefill_result)
    first_delivered = clock()
    votes.clear()
    started = clock()
    while not session.finished:
        before = clock()
        session.step()
        timings.append(clock()-before)
    elapsed = clock()-started
    tokens = np.asarray([event.token_id for event in events], np.int32)
    windows = []
    for start in range(0,len(timings),256):
        chunk = timings[start:start+256]
        windows.append(dict(first_decode_step=start,count=len(chunk),wall_seconds=sum(chunk),
                            tokens_per_second=len(chunk)/sum(chunk)))
    report = dict(healthy=not session.failed,emitted=len(tokens),decode_steps=len(timings),
        finish_reason=events[-1].finish_reason,wall_seconds=elapsed,
        tokens_per_second=len(timings)/elapsed if timings and elapsed>0 else None,
        p50_ms=float(np.median(timings)*1e3) if timings else None,
        p99_ms=float(np.percentile(timings,99)*1e3) if timings else None,
        model_step_p50_ms=float(np.median(session.decode_seconds)*1e3) if timings else None,
        timed_votes=len(votes),vote_wall_seconds=sum(votes),warm_steps_excluded=0,
        ttft_seconds=first_delivered-request_started,
        request_wall_seconds=clock()-request_started,
        token_sha256=sha256(tokens.tobytes()).hexdigest(),windows=windows,
        sampling='greedy',speculative=False,answer_correctness='not yet assessed',
        delivery_boundary=session.delivery_boundary,
        excludes_cold_load_compile=True,decode_rate_excludes_prefill=True)
    state = session._state
    session.release()
    return tokens,report,state
