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


def summarize_question_rows(rows,controller):
    """Require complete, agreed ordinary question receipts; export no raw text."""
    if len(rows)!=8 or not all('question' in r and 'question_identity' in r and 'question_prefill' in r for r in rows):
        raise ValueError('fresh question requires all eight complete reports')
    identity=rows[0]['question_identity'];policy=identity['policy']
    policy= RequestPolicy(**dict(policy,eos_ids=tuple(policy['eos_ids'])))
    if (identity['file_sha256']!=controller.get('question_sha256')
            or any(r['question_identity']!=identity for r in rows)):
        raise ValueError('question input/policy differs across workers/controller')
    reference=rows[0]['question'];devices=[]
    for digest in (identity['file_sha256'],identity['prompt_ids_sha256'],reference['token_sha256']):
        if type(digest) is not str or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('question digest is malformed')
    for r in rows:
        q=r['question'];pre=r['question_prefill'];steps=q['decode_steps']
        if (any(q.get(k) is not True for k in ('healthy','all_host_token_agreement','final_cache_finite',
                                             'excludes_cold_load_compile','decode_rate_excludes_prefill'))
                or q.get('speculative') is not False or q.get('sampling')!='greedy'
                or q.get('delivery_boundary')!='rank0 private JSONL token write+flush; no network transport'
                or q.get('warm_steps_excluded')!=0 or type(steps) is not int or steps<0
                or q['emitted']!=steps+1 or not 1<=q['emitted']<=policy.max_new_tokens
                or q['timed_votes']!=2*steps or q['finish_reason'] not in ('eos','length')
                or (q['finish_reason']=='length' and q['emitted']!=policy.max_new_tokens)
                or any(q[k]!=reference[k] for k in ('emitted','decode_steps','finish_reason','token_sha256','delivery_boundary'))):
            raise ValueError('question generation scope, health or output agreement differs')
        expected={*(f'question_prefill_{i}' for i in range((policy.prompt_tokens+127)//128)),
                  *(f'question_prefill_health_{i}' for i in range((policy.prompt_tokens+127)//128)),
                  'question_reference_admission','compile_question_packed','graph_consensus_question_packed',
                  'hlo_question_packed','memory_question_packed','question_generation','question_token_agreement','question_cache_check'}
        if any(r['phases'].get(name,{}).get('passed') is not True for name in expected):
            raise ValueError('question execution/admission phase missing or failed')
        if (pre['prompt_tokens']!=policy.prompt_tokens or pre['wall_seconds']<=0
                or not np.isclose(pre['prompt_tokens_per_second'],policy.prompt_tokens/pre['wall_seconds'],rtol=1e-9)):
            raise ValueError('question prefill timing scope differs')
        values=[q[k] for k in ('wall_seconds','ttft_seconds','request_wall_seconds','vote_wall_seconds')]
        if (not all(np.isfinite(x) and x>=0 for x in values)
                or q['request_wall_seconds']+1e-6<q['ttft_seconds']+q['wall_seconds']
                or q['ttft_seconds']+1e-6<pre['wall_seconds'] or q['vote_wall_seconds']>q['wall_seconds']):
            raise ValueError('question timing boundary differs')
        if steps:
            if (q['wall_seconds']<=0 or not np.isclose(q['tokens_per_second'],steps/q['wall_seconds'],rtol=1e-9)
                    or not all(np.isfinite(q[k]) and q[k]>0 for k in ('p50_ms','p99_ms','model_step_p50_ms'))):
                raise ValueError('question wall rate/latency differs')
        elif q['tokens_per_second'] is not None:
            raise ValueError('first-token-only question has no decode rate')
        at=0;seconds=0.
        for w in q['windows']:
            if (w['first_decode_step']!=at or type(w['count']) is not int or not 1<=w['count']<=256
                    or w['wall_seconds']<=0 or not np.isfinite(w['wall_seconds'])
                    or not np.isclose(w['tokens_per_second'],w['count']/w['wall_seconds'],rtol=1e-9)):
                raise ValueError('question timing windows differ')
            at+=w['count'];seconds+=w['wall_seconds']
        if at!=steps or seconds>q['wall_seconds']+1e-6:raise ValueError('question timing coverage differs')
        memory=q['memory_after']
        if len(memory)!=4 or {d['device_id'] for d in memory}!={d['device_id'] for d in r['decode']['memory_after']}:
            raise ValueError('question physical memory coverage differs')
        if any(not 0<d['peak_bytes_in_use']<=d['bytes_limit'] for d in memory):
            raise ValueError('question peak memory exceeds its physical limit')
        devices.extend(d['device_id'] for d in memory)
    if len(devices)!=32 or set(devices)!=set(range(32)):raise ValueError('question memory does not cover all32 chips')
    def span(values):return dict(min=min(values),max=max(values))
    timing=('wall_seconds','ttft_seconds','request_wall_seconds','vote_wall_seconds','p50_ms','p99_ms','model_step_p50_ms','tokens_per_second')
    return dict(prompt_tokens=policy.prompt_tokens,max_new_tokens=policy.max_new_tokens,
        question_file_sha256=identity['file_sha256'],prompt_ids_sha256=identity['prompt_ids_sha256'],
        emitted=reference['emitted'],decode_steps=reference['decode_steps'],finish_reason=reference['finish_reason'],
        all_host_token_agreement=True,token_sha256=reference['token_sha256'],
        timings={k:span([r['question'][k] for r in rows]) if reference[k] is not None else None for k in timing},
        prefill={k:span([r['question_prefill'][k] for r in rows]) for k in ('wall_seconds','prompt_tokens_per_second')},
        rank0_windows=[{k:w[k] for k in ('first_decode_step','count','wall_seconds','tokens_per_second')} for w in reference['windows']],
        delivery_boundary=reference['delivery_boundary'],
        excludes_cold_load_compile=True,decode_rate_excludes_prefill=True,warm_steps_excluded=0,
        sampling='greedy',speculative=False,answer_correctness='requires separate answer assessment',
        maximum_peak_hbm_bytes=max(d['peak_bytes_in_use'] for r in rows for d in r['question']['memory_after']))
