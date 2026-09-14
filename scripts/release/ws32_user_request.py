"""One user request through the real admitted native runtime, without scoring.

Worker component only: the caller must own both leases and the original
source/checkpoint/topology/HLO/HBM admission and final evidence publication.
It is not a CLI launcher, benchmark substitution or deployment authorization.
"""
from __future__ import annotations

import gzip
from time import perf_counter
from typing import Any, Callable

import numpy as np

from glm_tpu import user_request
from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
from scripts.greenfield.ws32_native_benchmark_requests import TokenSink, _before_deadline


def execute_user_request(*, loaded: Any, request: dict, store: Any, tokenizer: Any,
                         observations: Any, deadline: float,
                         clock: Callable[[], float] = perf_counter) -> dict:
    """Keep the same fresh-cache, first-delivery and no-retry semantics.

No benchmark registry is loaded. The caller provides NativeObservability and
RequestStore from the protected worker; evidence never receives a quality score.
The terminal record distinguishes instrumented wall from pure decode samples.
"""
    runtime = loaded.runtime
    def validate():
        user_request.validate(request)
        if observations is None:
            raise ValueError("user delivery requires native DSA/cache/memory observability")
        if (runtime.raw_config.context_capacity != user_request.CAPACITY
                or runtime.raw_config.geometry.vocab_size != user_request.VOCAB):
            raise ValueError("loaded user runtime differs from frozen request profile")
        _before_deadline(deadline, clock)
    runtime._phase(validate)
    runtime._phase(lambda: store.begin(0, request))
    sink = runtime._phase(lambda: TokenSink(store, request["request_id"]))
    row = dict(schema="glm_ws32_user_result_v1", request_id=request["request_id"],
        request_sha256=request["request_sha256"], rank=store.rank,
        complete=False, benchmark=False, quality_score=None, protected_result_sealed=False,
        delivery_boundary="rank0_local_jsonl_write_flush" if store.rank == 0 else "nonoutput_rank")
    try:
        policy = runtime._phase(lambda: RequestPolicy(request_id=request["request_id"],
            seed=request["seed"], prompt_tokens=len(request["prompt_ids"]),
            max_new_tokens=request["max_new_tokens"], context_capacity=user_request.CAPACITY,
            vocab_size=user_request.VOCAB, eos_ids=user_request.EOS))
        session = runtime.start_request(np.asarray(request["prompt_ids"], np.int32), policy,
            deliver=sink, delivery_boundary=row["delivery_boundary"], request_started=clock())
        row["first_token_delivered_before_decode"] = sink.count == 1 and not session.decode_seconds
        runtime._phase(lambda: observations.begin(0, session))
        while not session.finished:
            runtime._phase(lambda: _before_deadline(deadline, clock))
            session.step()
        evidence = observations.finish(session)
        row.update(complete=True, finish_reason=session.events[-1].finish_reason,
            phases=dict(runtime.phase_seconds), generated_tokens=sink.count,
            token_ids_sha256=sink.digest.hexdigest(), observations=evidence,
            cold_load_compile_seconds=loaded.record["cold_load_compile_seconds"],
            ttft_seconds=session.ttft_seconds if store.rank == 0 else None,
            delivered_request_seconds=session.delivered_request_seconds if store.rank == 0 else None,
            decode_seconds=list(session.decode_seconds))
        def save():
            if store.memory_names != ["before_cache", "cache_ready", "prefill_done"]:
                raise ValueError("user request lacks original memory boundaries")
            if store.rank == 0:
                ids = sink.ids[:-1] if sink.ids and sink.ids[-1] in user_request.EOS else sink.ids
                text = tokenizer.decode(ids, skip_special_tokens=False).encode()
                row["answer"] = store.save("answer.txt.gz", gzip.compress(text, mtime=0), 16 << 20)
            store.save("result.json.gz", gzip.compress(user_request.canonical(row), mtime=0), 4 << 20)
        runtime._phase(sink.close)
        runtime._phase(save)
        runtime._phase(runtime.close_request)
        return row
    except Exception as exc:
        row.update(complete=False, failure_type=type(exc).__name__, generated_tokens=sink.count,
                   token_ids_sha256=sink.digest.hexdigest())
        # Keep already delivered bytes. Never restart or regenerate this request.
        try:
            store.save("failure.json", user_request.canonical(row), 4 << 20)
        except Exception:
            pass
        raise
    finally:
        runtime._phase(sink.close)
