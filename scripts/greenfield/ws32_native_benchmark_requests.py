"""Execute registered native requests; bounded answers, timings and resume.

The protected outer worker supplies a cold-loaded NativeBenchmarkRuntime and
both-lease ownership. This file does not authorize a launch. It writes actual
rank0 token delivery to an append-only local JSONL sink (not network TTFT),
keeps partial failures, and never treats CPU fixtures as benchmark evidence.
"""
from __future__ import annotations

from dataclasses import asdict
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
from scripts.greenfield import ws32_native_benchmark_protocol as protocol
from scripts.greenfield.ws32_history_preflight import _plain_path

RANK_CAP = 512 << 20
RESERVE = 1 << 30
MEMORY_RAW_CAP = 4 << 20
MEMORY_GZIP_CAP = 2 << 20
TOKEN_CAP = 32 << 20


class RequestStore:
    """One fresh rank directory, original-only writes and incremental byte cap."""
    def __init__(self, root: Path, rank: int):
        if type(rank) is not int or not 0 <= rank < 8:
            raise ValueError("native request store requires launch rank0..7")
        _plain_path(root)
        root.mkdir(exist_ok=False)
        self.root, self.rank, self.used, self.current = root, rank, 0, None
        self.memory_names: list[str] = []

    def _budget(self, size: int) -> None:
        if self.used + size > RANK_CAP or shutil.disk_usage(self.root).free < size + RESERVE:
            raise ValueError("native request evidence storage reserve/cap exhausted")

    def save(self, name: str, data: bytes, cap: int) -> dict:
        if self.current is None or len(data) > cap:
            raise ValueError("native original name/size outside active request")
        path = self.current / name
        _plain_path(path)
        if path.parent != self.current:
            raise ValueError("native original escaped its request")
        self._budget(len(data))
        with path.open("xb") as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        self.used += len(data)
        return dict(name=name, bytes=len(data), sha256=sha256(data).hexdigest())

    def begin(self, index: int, request: Mapping) -> None:
        if type(index) is not int or not 0 <= index < 228:
            raise ValueError("native request index outside registered set")
        self.current = self.root / f"item{index:03d}"
        self.current.mkdir(exist_ok=False)
        self.memory_names = []
        self.save("identity.json", protocol.canonical(dict(index=index,
            request_id=request["request_id"], prompt_ids_sha256=request["prompt_ids_sha256"],
            seed=request["seed"], rank=self.rank)), 4096)

    def preserve_memory(self, phase: str, value: Mapping) -> None:
        expected = ("before_cache", "cache_ready", "prefill_done")
        if len(self.memory_names) >= 3 or phase != expected[len(self.memory_names)]:
            raise ValueError("native request original memory phases differ")
        raw = protocol.canonical(value)
        if len(raw) > MEMORY_RAW_CAP:
            raise ValueError("native original memory record exceeds raw cap")
        self.save(phase + ".json.gz", gzip.compress(raw, mtime=0), MEMORY_GZIP_CAP)
        self.memory_names.append(phase)


class TokenSink:
    """Only rank0 writes raw tokens; all ranks hash the same ordered int32 IDs."""
    def __init__(self, store: RequestStore, request_id: str):
        self.store, self.request_id = store, request_id
        self.count, self.bytes, self.digest = 0, 0, sha256()
        self.ids: list[int] = []
        self.stream = (store.current / "tokens.jsonl").open("xb") if store.rank == 0 else None

    def __call__(self, event: Any) -> None:
        if event.request_id != self.request_id or event.index != self.count:
            raise ValueError("native output frontier/request changed")
        raw = protocol.canonical(asdict(event)) + b"\n"
        if self.bytes + len(raw) > TOKEN_CAP:
            raise ValueError("native raw token output exceeds cap")
        if self.stream is not None:
            self.store._budget(len(raw))
            self.stream.write(raw)
            self.stream.flush()  # actual local sink delivery, NOT just device readiness
            self.store.used += len(raw)
            if self.count == 0 or (self.count + 1) % 128 == 0:
                os.fsync(self.stream.fileno())
            self.ids.append(event.token_id)
        self.digest.update(np.asarray([event.token_id], dtype="<i4").tobytes())
        self.count += 1
        self.bytes += len(raw)

    def close(self) -> None:
        if self.stream is not None and not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


def score_answer(request: Mapping, ids: list[int], tokenizer: Any, registry: Any) -> dict:
    # Retain the raw EOS in tokens.jsonl, but do not append its literal special
    # token spelling to a final standalone MC letter. Keep </think> intact.
    answer_ids = ids[:-1] if ids and ids[-1] in protocol.EOS else ids
    text = tokenizer.decode(answer_ids, skip_special_tokens=False)
    if request["dataset"] == "aime_2026":
        return dict(text=text, extracted=None, correct=None, score_status="AWAITING_APPROVED_JUDGE")
    # The prompt ends INSIDE <think>. Never count a provisional boxed answer
    # in an unfinished reasoning trace as the final answer after truncation.
    final = text.rsplit("</think>", 1)[-1] if "</think>" in text else ""
    spec = registry.REGISTRY[request["dataset"]]
    extracted = spec.extract(final, SimpleNamespace(**request["item"]))
    return dict(text=text, extracted=extracted,
                correct=bool(spec.score(extracted, request["item"]["gold"])),
                score_status="SCORED_PINNED_GPQA_HARNESS")


def execute_requests(*, loaded: Any, payload: Mapping, plan: Mapping,
                     store: RequestStore, tokenizer: Any, deadline: float,
                     clock: Callable[[], float] = perf_counter,
                     after_request: Callable[[Mapping], None] | None = None) -> list[dict]:
    """Run the full predeclared set until completion/failure/operational deadline.

    A budget stop retains the current partials and is NOT a completed benchmark.
    Pause/resume proof retains the first live session between two host calls;
    no sleep, cache-copy, replay or extra model step is added for the proof.
    Outer completion separately requires source/HLO/HBM/DSA/cache/trace replay.
    """
    protocol.validate(payload, plan)
    registry = protocol.benchmark_registry()
    runtime = loaded.runtime
    rows = []
    for index, request in enumerate(payload["requests"]):
        runtime._phase(lambda: _before_deadline(deadline, clock))
        runtime._phase(lambda: store.begin(index, request))
        sink = runtime._phase(lambda: TokenSink(store, request["request_id"]))
        session = None
        row = dict(index=index, request_id=request["request_id"], rank=store.rank,
            complete=False, benchmark_score=None, performance_claim=False,
            delivery_boundary="rank0_local_jsonl_write_flush" if store.rank == 0 else "nonoutput_rank",
            protocol_sha256=sha256(protocol.canonical(plan)).hexdigest())
        try:
            policy = runtime._phase(lambda: RequestPolicy(request_id=request["request_id"], seed=request["seed"],
                prompt_tokens=len(request["prompt_ids"]), max_new_tokens=plan["max_new_tokens"],
                context_capacity=plan["context_capacity"], vocab_size=plan["vocab_size"],
                eos_ids=tuple(plan["eos_ids"])))
            session = runtime.start_request(np.asarray(request["prompt_ids"], np.int32), policy,
                deliver=sink, delivery_boundary=row["delivery_boundary"], request_started=clock())
            row["first_token_delivered_before_decode"] = sink.count == 1 and not session.decode_seconds
            # start_request returns with the live session paused. Resume that
            # SAME object below; no discarded prefill/draw or hidden warmup.
            row["live_session_resume"] = not session.finished
            while not session.finished:
                runtime._phase(lambda: _before_deadline(deadline, clock))
                session.step()
            row.update(complete=True, finish_reason=session.events[-1].finish_reason,
                phases=dict(runtime.phase_seconds), generated_tokens=sink.count,
                token_ids_sha256=sink.digest.hexdigest(),
                ttft_seconds=session.ttft_seconds if store.rank == 0 else None,
                delivered_request_seconds=session.delivered_request_seconds if store.rank == 0 else None,
                decode_seconds=list(session.decode_seconds))
            def finalize_answer():
                if store.memory_names != ["before_cache", "cache_ready", "prefill_done"]:
                    raise ValueError("completed request lacks original memory boundaries")
                if store.rank == 0:
                    score = score_answer(request, sink.ids, tokenizer, registry)
                    text = score.pop("text").encode()
                    row["answer"] = store.save("answer.txt.gz", gzip.compress(text, mtime=0), 16 << 20)
                    row.update(score)
                store.save("result.json.gz", gzip.compress(protocol.canonical(row), mtime=0), 4 << 20)
            runtime._phase(sink.close)
            runtime._phase(finalize_answer)
            if after_request is not None:
                runtime._phase(lambda: after_request(row))
            runtime._phase(runtime.close_request)
            rows.append({key: value for key, value in row.items() if key != "decode_seconds"})
        except Exception as exc:
            row.update(complete=False, failure=f"{type(exc).__name__}: {exc}",
                generated_tokens=sink.count, token_ids_sha256=sink.digest.hexdigest())
            # No retry of a donated/partially delivered request. Keep originals;
            # the outer owner handles process completion/publication/cleanup.
            try:
                store.save("failure.json", protocol.canonical(row), 4 << 20)
            except Exception:
                pass  # retain primary failure; already-written tokens survive
            raise
        finally:
            runtime._phase(sink.close)
    return rows


def _before_deadline(deadline: float, clock: Callable[[], float]) -> None:
    now = clock()
    if not math.isfinite(deadline) or not math.isfinite(now):
        raise ValueError("invalid registered operational clock")
    if now >= deadline:
        raise TimeoutError("registered operational tranche exhausted; campaign incomplete")
