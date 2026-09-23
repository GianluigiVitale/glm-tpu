"""G9: serving wire formats and request identity, characterized through the real code.

Nothing here reimplements a format: every byte is produced by the production functions, driven
with deterministic synthetic inputs and fakes for the parts that need a fleet, a tokenizer or a
device. Recorded:

* request bodies and ``request_sha256`` for every context profile, sequential and concurrent
  batches, refusals (including a non-ASCII request id, which the 181c013e profile rejects) and
  the two canonical-JSON contracts on non-ASCII message content;
* worker output files (``tokens.jsonl`` TokenEvent lines solo and with ``batch_round``,
  ``answer.txt``, directory layout) and the prefill block schedule (rows, count; a 114-token tail
  included) from the real ``run_queued`` over the real ``OrdinaryRuntime.generate`` and the real
  ``run_concurrent`` over the real ``generate_concurrent`` -> ``batched_runtime.generate_batch`` ->
  ``BatchedSession``, both with synthetic device results;
* the API's messages-size measure, found by bisection over ``glm_tpu.api.convert`` itself;
* the resident protocol (ready file bytes, command and stop bytes written to worker stdin,
  per-round records) from the real ``resident_loop`` and ``resident_controller``;
* record key sets: worker ``runner.rank{r}.json`` from the real worker ``main`` with the fleet
  faked (sequential and concurrent), together with the arguments ``main`` passes to
  ``OrdinaryRuntime`` (names and described values, e.g. ``context_capacity``, ``concurrent_size``),
  controller ``summarize()``, runtime request reports and phase names;
* HTTP through the real UI/API handler with a fake resident: status, headers (CSP included)
  and body bytes, SSE streams byte for byte, ids and timestamps normalized by regex.

Run as ``python -m tools.equivalence.wire`` (prints one JSON line: ``{"wire": ..., "http": ...}``).
"""

from __future__ import annotations

import ast
from contextlib import ExitStack, contextmanager
import http.client
import io
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterator
from unittest import mock

import numpy as np

from .common import REPO, emit, sha256_hex, source_record

NON_ASCII_MESSAGES = [
    {"role": "system", "content": "Sei un assistente preciso. 中文 ✓"},
    {"role": "user", "content": "Quanto fa 6×7? Risposta in una riga, caffè."},
]
PROFILES = (8192, 32768, 166912, 262144)


def _bytes_record(raw: bytes) -> dict[str, Any]:
    return dict(sha256=sha256_hex(raw), bytes=len(raw))


def _error(call: Any) -> str | None:
    try:
        call()
    except (ValueError, TypeError, KeyError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


# ----------------------------------------------------------------------------- requests
def requests_record() -> dict[str, Any]:
    from glm_tpu import user_request as legacy
    from glm_tpu.api import convert
    from glm_tpu.optimized import request

    out: dict[str, Any] = {}
    prompt = [(i * 7919 + 13) % 154880 for i in range(96)]
    bodies = {}
    for capacity in PROFILES:
        value = request.from_token_ids(prompt, request_id=f"golden-{capacity}", max_new_tokens=64,
                                       context_capacity=capacity)
        bodies[capacity] = value
        out[f"single_{capacity}"] = dict(_bytes_record(legacy.canonical(value)), request_sha256=value["request_sha256"],
                                         keys=sorted(value))
    sequential = request.batch([bodies[8192], request.from_token_ids(prompt[:40], request_id="golden-8192-b",
                                                                      max_new_tokens=8)])
    out["batch_sequential"] = dict(_bytes_record(legacy.canonical(sequential)),
                                   request_sha256=sequential["request_sha256"], keys=sorted(sequential))
    lanes = [request.from_token_ids(prompt[: 20 + 10 * i], request_id=f"lane-{i}", max_new_tokens=32,
                                    context_capacity=32768) for i in range(4)]
    concurrent = request.batch(lanes, concurrent=True)
    out["batch_concurrent"] = dict(_bytes_record(legacy.canonical(concurrent)),
                                   request_sha256=concurrent["request_sha256"], keys=sorted(concurrent))
    tampered = dict(bodies[32768], max_new_tokens=65)
    out["refusals"] = dict(
        non_ascii_request_id=_error(lambda: request.from_token_ids(prompt, request_id="richiesta-è",
                                                                   max_new_tokens=8)),
        tampered=_error(lambda: request.validate(tampered)),
        unsupported_capacity=_error(lambda: request.from_token_ids(prompt, request_id="x", max_new_tokens=8,
                                                                   context_capacity=4096)),
        over_budget=_error(lambda: request.from_token_ids(prompt, request_id="x", max_new_tokens=8150)),
        concurrent_capacity=_error(lambda: request.batch(
            [request.from_token_ids(prompt, request_id=f"c{i}", max_new_tokens=8) for i in range(2)],
            concurrent=True)),
        duplicate_ids=_error(lambda: request.batch([bodies[8192], bodies[8192]])),
    )
    out["stop_cause"] = dict(eos=request.stop_cause(bodies[8192], "eos"),
                             output_cap=request.stop_cause(bodies[8192], "length"),
                             context_exhausted=request.stop_cause(
                                 request.from_token_ids(prompt, request_id="full", max_new_tokens=8192 - len(prompt)),
                                 "length"))
    out["messages_non_ascii"] = dict(
        wire=_bytes_record(legacy.canonical(NON_ASCII_MESSAGES)),
        api_cap_measure=api_cap_measure(),
        converted=_bytes_record(legacy.canonical(convert(NON_ASCII_MESSAGES))),
    )
    out["constants"] = dict(schemas=[request.SCHEMA, request.BATCH_SCHEMA, request.CONCURRENT_SCHEMA],
                            capacities=list(request.CAPACITIES), prompt_limits={str(k): v for k, v in
                                                                                 request.PROMPT_LIMITS.items()},
                            concurrent_limit=request.CONCURRENT_LIMIT, payload_cap=legacy.PAYLOAD_CAP,
                            messages_cap=legacy.MESSAGES_CAP, max_new=legacy.MAX_NEW, vocab=legacy.VOCAB,
                            eos=list(legacy.EOS))
    return out


CAP_PROBES = (("ascii", "a"), ("latin", "\u00e8"), ("cjk", "\u4e2d"), ("astral", "\U0001f600"))


def api_cap_measure() -> dict[str, Any]:
    """The largest one-message content (in characters, per character class) that
    ``glm_tpu.api.convert`` accepts, found by bisection over ``convert`` itself -- so the record is
    the API's own size measure (at 181c013e ``len(json.dumps(messages).encode())``, i.e. ASCII
    escapes: 6 bytes per non-ASCII BMP character, 12 per astral one)."""
    from glm_tpu.api import ApiError, convert

    def accepts(char: str, count: int) -> bool:
        try:
            convert([{"role": "user", "content": char * count}])
        except ApiError as exc:
            if "size" not in str(exc):
                raise
            return False
        return True

    out: dict[str, Any] = {}
    for label, char in CAP_PROBES:
        low, high = 1, 1 << 21
        if not accepts(char, low) or accepts(char, high):
            raise RuntimeError("the API size limit is outside the probed range")
        while high - low > 1:
            middle = (low + high) // 2
            low, high = (middle, high) if accepts(char, middle) else (low, middle)
        out[label] = low
    return out


# ----------------------------------------------------------------------------- worker / runtime fakes
def _state(position: int, healthy: bool = True) -> Any:
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderState

    return Ws32DecoderState(np.zeros((1,)), np.zeros((1,)), np.zeros((1, 1), np.int32), np.ones((1,), np.int32),
                            np.zeros((1, 1), np.float32), np.array([position], np.int32),
                            np.zeros((1, 1), np.int32), np.array([position + 1], np.int32), np.array([healthy]))


def _runtime(outputs: list[int]) -> Any:
    """The real OrdinaryRuntime host logic over synthetic device results (no devices)."""
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillResult, Ws32BatchedPrefillState
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecodeStepResult
    from glm_tpu.optimized.request_loop import PackedDecodeResult
    from glm_tpu.optimized.runtime import OrdinaryRuntime

    runtime = object.__new__(OrdinaryRuntime)
    runtime.capacity, runtime.concurrent_size, runtime.active = 8192, 0, False
    runtime.put = lambda value: value
    runtime.weights = runtime.wk = runtime.rope = None
    runtime.record = dict(schema="glm_optimized_runtime_v1", programs={}, phases={}, requests=[])
    runtime.save = lambda record: None
    runtime.admit = lambda name: None
    runtime.stats = lambda: []
    runtime.vote = lambda value: value
    prompt: dict[str, int] = {}
    calls: list[int] = []

    def initialize(count: Any) -> Any:  # a fresh cache starts a fresh request
        prompt["length"] = int(np.asarray(count))
        calls.clear()
        return SimpleNamespace(decoder=_state(0))

    def execute_prefill(block: Any, count: Any, fresh: Any, *args: Any) -> Any:
        length = prompt["length"]
        runtime.prefill_calls.append([int(np.asarray(block).size), int(np.asarray(count))])
        return Ws32BatchedPrefillResult(Ws32BatchedPrefillState(_state(length), np.zeros((1,)),
                                                                np.array(length, np.int32), np.array(True)),
                                        np.array([outputs[0]], np.int32))

    def decode(token: Any, previous: Any, *args: Any) -> Any:
        calls.append(int(np.asarray(token)[0]))
        index = len(calls)
        position = prompt["length"] + index
        value = outputs[min(index, len(outputs) - 1)]
        step = Ws32DecodeStepResult(_state(position), np.array([value], np.int32), np.zeros((1, 1)))
        return PackedDecodeResult(step, np.array([value, 1, position, position + 1], np.int32))

    runtime.initialize = initialize
    runtime.prefill = {114: execute_prefill, 128: execute_prefill}
    runtime.prefill_calls = []
    runtime.decode = decode
    return runtime


def _batched_runtime(schedules: list[list[int]]) -> Any:
    """The real ``OrdinaryRuntime.generate_concurrent`` -> ``batched_runtime.generate_batch`` ->
    ``BatchedSession`` host logic over synthetic device results (no devices). ``schedules[lane]``
    lists a lane's tokens: the first from prefill, the next ones from successive decode rounds."""
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import Ws32BatchedPrefillResult, Ws32BatchedPrefillState
    from glm_tpu.optimized.runtime import OrdinaryRuntime

    runtime = object.__new__(OrdinaryRuntime)
    runtime.capacity, runtime.concurrent_size, runtime.active = 32768, len(schedules), False
    runtime.put = lambda value: value
    runtime.weights = runtime.wk = runtime.rope = None
    runtime.record = dict(schema="glm_optimized_runtime_v1", programs={}, phases={}, requests=[])
    runtime.save = lambda record: None
    runtime.admit = lambda name: None
    runtime.stats = lambda: []
    runtime.vote = lambda value: value
    lanes: dict[str, Any] = dict(index=-1, lengths=[])
    runtime.prefill_calls = []

    def initialize_batch(lengths: Any) -> Any:
        lanes.update(index=-1, lengths=[int(x) for x in np.asarray(lengths)])
        return dict(round=0)

    def initialize(length: Any) -> Any:
        lanes["index"] += 1
        return SimpleNamespace(decoder=_state(0))

    def execute_prefill(block: Any, count: Any, fresh: Any, *args: Any) -> Any:
        lane = lanes["index"]
        length = lanes["lengths"][lane]
        runtime.prefill_calls.append([lane, int(np.asarray(block).size), int(np.asarray(count))])
        return Ws32BatchedPrefillResult(Ws32BatchedPrefillState(_state(length), np.zeros((1,)),
                                                                np.array(length, np.int32), np.array(True)),
                                        np.array([schedules[lane][0]], np.int32))

    def insert_batch(state: Any, one: Any, index: Any) -> Any:
        return state

    def decode_batch(tokens: Any, state: Any, weights: Any, rope: Any, active: Any) -> Any:
        step = state["round"] + 1
        rows = []
        for lane, length in enumerate(lanes["lengths"]):
            token = schedules[lane][min(step, len(schedules[lane]) - 1)]
            rows.append([token, 1, length + step, length + step + 1])
        metadata = np.asarray(rows, np.int32)
        return SimpleNamespace(state=dict(round=step), next_token=metadata[:, :1], metadata=metadata)

    runtime.initialize_batch = initialize_batch
    runtime.initialize = initialize
    runtime.prefill = {114: execute_prefill, 128: execute_prefill}
    runtime.insert_batch = insert_batch
    runtime.decode_batch = decode_batch
    return runtime


@contextmanager
def _fleet_fakes() -> Iterator[None]:
    from jax.experimental import multihost_utils

    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "decoded:" + ",".join(map(str, tokens)))
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(multihost_utils, "process_allgather",
                                              lambda value: np.stack([value] * 8)))
        stack.enter_context(mock.patch("transformers.AutoTokenizer.from_pretrained",
                                       lambda *args, **kwargs: tokenizer))
        yield


_REPORT_VALUES = ("emitted", "finish_reason", "stop_cause", "output_directory", "output_budget_tokens",
                  "token_sha256", "request_sha256", "prompt_tokens", "timed_decode_tokens", "sampling",
                  "speculative", "request_id")


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) + ("/" if p.is_dir() else "") for p in root.rglob("*"))


def worker_record() -> dict[str, Any]:
    from glm_tpu.optimized import request
    from scripts.release import ws32_optimized_worker as worker

    out: dict[str, Any] = {}
    items = [request.from_token_ids([30, 31, 32], request_id="golden-q1", max_new_tokens=4),
             request.from_token_ids([40, 41, 42, 43, 44], request_id="golden-q2", max_new_tokens=3),
             request.from_token_ids([50 + i % 7 for i in range(114)], request_id="golden-q3", max_new_tokens=2),
             request.from_token_ids([60 + i % 5 for i in range(242)], request_id="golden-q4", max_new_tokens=2)]
    value = request.batch(items)
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch, _fleet_fakes():
        root = Path(scratch)
        runtime = _runtime([7, 9, 10, 154820])
        reports = worker.run_queued(runtime, request.requests(value), value, root, 0, time.perf_counter() + 600,
                                    save=lambda reports: None)
        out["run_queued"] = dict(
            layout=_tree(root),
            tokens=[(root / f"item{i:03d}" / "tokens.jsonl").read_text() for i in range(len(items))],
            answers=[(root / f"item{i:03d}" / "answer.txt").read_text() for i in range(len(items))],
            report_keys=sorted(reports[0]),
            reports=[{k: r[k] for k in _REPORT_VALUES if k in r} for r in reports],
            runtime_phases=sorted(runtime.record["phases"]),
            runtime_record_keys=sorted(runtime.record),
            prefill_calls=runtime.prefill_calls,
        )

    with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch, _fleet_fakes():
        root = Path(scratch)
        lengths, budgets = (130, 114, 20, 242), (3, 2, 4, 3)
        pending = [request.from_token_ids([7 + lane] * length, request_id=f"lane{lane}", max_new_tokens=budget,
                                          context_capacity=32768)
                   for lane, (length, budget) in enumerate(zip(lengths, budgets, strict=True))]
        # lane 1 stops on EOS in round 1; the others run to their output budget (length)
        schedules = [[10, 11, 12, 13], [20, 154820, 22], [30, 31, 32, 33, 34], [40, 41, 42, 43]]
        runtime = _batched_runtime(schedules)
        reports, aggregate = worker.run_concurrent(runtime, pending, root, 0, time.perf_counter() + 600)
        out["run_concurrent"] = dict(
            layout=_tree(root),
            tokens=[(root / f"item{i:03d}" / "tokens.jsonl").read_text() for i in range(len(pending))],
            answers=[(root / f"item{i:03d}" / "answer.txt").read_text() for i in range(len(pending))],
            report_keys=sorted(reports[0]), aggregate_keys=sorted(aggregate),
            reports=[{k: r[k] for k in (*_REPORT_VALUES, "batch_size", "batch_rounds", "context_capacity") if k in r}
                     for r in reports],
            aggregate=dict(batch_size=aggregate["batch_size"], decode_rounds=aggregate["decode_rounds"]),
            runtime_phases=sorted(runtime.record["phases"]), runtime_record_keys=sorted(runtime.record),
            batches=len(runtime.record.get("batches", [])), prefill_calls=runtime.prefill_calls)

    # resident_loop: ready file, per-round record, stop; the model run itself is faked.
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch:
        root = Path(scratch)
        single = request.from_token_ids([7], request_id="resident", max_new_tokens=2, context_capacity=32768)
        base = dict(schema="glm_optimized_worker_v1", rank=0, complete=True)
        runtime = SimpleNamespace(capacity=32768, phase=lambda name, action: action())
        seen = []

        def fake_run(actual: Any, pending: Any, value: Any, job: Path, rank: int, deadline: Any, **kwargs: Any) -> Any:
            seen.append(dict(job=job.name, warmup=kwargs.get("warmup")))
            return [dict(emitted=2, token_sha256="c" * 64)]

        commands = json.dumps(dict(sequence=1, request=single)) + "\n" + json.dumps(dict(stop=True)) + "\n"
        with mock.patch.object(worker, "run_queued", fake_run):
            worker.resident_loop(runtime, base, root, 0, 3600, stream=io.StringIO(commands))
        record = json.loads((root / "resident-0001" / "runner.rank0.json").read_text())
        out["resident_loop"] = dict(layout=_tree(root), ready=(root / "resident-ready.json").read_text(),
                                    round_record_keys=sorted(record), calls=seen,
                                    persist_format=_bytes_record((root / "resident-ready.json").read_bytes()))
    out["main_record"] = worker_main_record()
    return out


def worker_main_record() -> dict[str, Any]:
    """``runner.rank{r}.json`` from the real worker ``main`` (fleet, devices and model faked), for a
    sequential request and a concurrent batch, and the arguments ``main`` passes to
    ``OrdinaryRuntime`` (names, and values described: numbers and strings as they are, objects by
    where they came from, the ``save`` callback by the file it writes)."""
    from glm_tpu.optimized import request

    sequential = request.from_token_ids([30, 31, 32], request_id="golden-main", max_new_tokens=3)
    concurrent = request.batch([request.from_token_ids([30 + lane] * (4 + lane), request_id=f"golden-lane{lane}",
                                                       max_new_tokens=3, context_capacity=32768)
                                for lane in range(3)], concurrent=True)
    return dict(_worker_main_run(sequential), concurrent=_worker_main_run(concurrent))


def _worker_main_run(value: dict[str, Any]) -> dict[str, Any]:
    from scripts.release import ws32_optimized_worker as worker

    import glm_tpu.optimized.runtime as runtime_module
    import scripts.greenfield.run_short_decoder_ws32 as original

    constructed: list[dict[str, Any]] = []

    class FakeRuntime:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)
            self.record = dict(cold_load_compile_seconds=1.0, programs={"decode": {}}, physical_identity={})

    def fake_preflight(args: Any) -> Any:
        args.process_id = 0
        args.context_capacity = value["context_capacity"]  # as the real preflight binds it
        return value, dict(sha256="b" * 64)

    def fake_run(runtime: Any, pending: Any, value: Any, root: Any, rank: int, deadline: Any, **kwargs: Any) -> Any:
        return [dict(emitted=3, token_sha256="c" * 64, finish_reason="length") for _ in pending]

    def fake_concurrent(runtime: Any, pending: Any, root: Any, rank: int, deadline: Any) -> Any:
        return fake_run(runtime, pending, None, root, rank, deadline), dict(batch_size=len(pending))

    fleet = dict(jax=SimpleNamespace(process_index=lambda: 0), mesh=SimpleNamespace(), physical=SimpleNamespace(),
                 topology=SimpleNamespace())
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch:
        root = Path(scratch)
        argv = ["--output", str(root), "--code-hash", "a" * 40, "--source-manifest-sha256", "d" * 64,
                "--request-file-sha256", "e" * 64, "--topology-rebinding-sha256", "f" * 64,
                "--coordinator-address", "203.0.113.10:8476", "--wall-seconds", "60"]
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(worker, "preflight", fake_preflight))
            stack.enter_context(mock.patch.object(worker, "run_queued", fake_run))
            stack.enter_context(mock.patch.object(worker, "run_concurrent", fake_concurrent))
            stack.enter_context(mock.patch.object(original, "_initialize_runtime", lambda args: (
                fleet["jax"], fleet["mesh"], fleet["physical"], fleet["topology"], "f" * 64)))
            stack.enter_context(mock.patch.object(runtime_module, "OrdinaryRuntime", FakeRuntime))
            stack.enter_context(mock.patch("socket.gethostname", lambda: "example-w-0"))
            code = worker.main(argv)
        record = json.loads((root / "runner.rank0.json").read_text())
        layout = _tree(root)  # what main wrote (before the save callback is probed below)
        if len(constructed) != 1:
            raise RuntimeError(f"worker main constructed {len(constructed)} runtimes")
        kwargs = constructed[0]
        known = {"<mesh from _initialize_runtime>": fleet["mesh"],
                 "<physical from _initialize_runtime>": fleet["physical"],
                 "<topology from _initialize_runtime>": fleet["topology"]}
        described: dict[str, Any] = {}
        for name, item in sorted(kwargs.items()):
            label = next((k for k, v in known.items() if item is v), None)
            if label is not None:
                described[name] = label
            elif name == "args":
                described[name] = sorted(vars(item))
            elif name == "save":
                before = set(_tree(root))
                item({"probe": True})
                described[name] = "writes " + ",".join(sorted(set(_tree(root)) - before))
            elif callable(item):
                described[name] = f"{getattr(item, '__module__', '?')}:{getattr(item, '__qualname__', '?')}"
            elif isinstance(item, Path):
                described[name] = ("<source root>" if item.resolve() == REPO.resolve()
                                   else "<output>/" + str(item.relative_to(root)))
            elif item is None or isinstance(item, (bool, int, float, str)):
                described[name] = item
            else:
                described[name] = f"<{type(item).__name__}>"
        return dict(exit_code=code, keys=sorted(record), request_keys=sorted(record["request"]), layout=layout,
                    runtime_arguments=described)


def controller_record() -> dict[str, Any]:
    from glm_tpu.optimized import request
    from scripts.release import launch_ws32_optimized_request as launch
    from scripts.release import ws32_optimized_worker as worker

    rows = [dict(rank=i, hostname=f"example-w-{i}", complete=True, code_hash="a" * 40, request_sha256="b" * 64,
                 request=dict(token_sha256="c" * 64, emitted=3),
                 programs=dict(decode=dict(stablehlo_sha256="d" * 64, optimized_hlo_sha256="e" * 64)))
            for i in range(8)]
    summary = launch.summarize(rows, "a" * 40, "b" * 64)
    resident = launch.summarize(rows, "a" * 40, "b" * 64, idle_after=False)
    out: dict[str, Any] = dict(summary=dict(keys=sorted(summary), bytes=_bytes_record(
        json.dumps(summary, sort_keys=True).encode())), resident_summary_keys=sorted(resident))
    # resident_controller: collect round 0, admit inbox 0001, then stop (clock and SSH faked).
    first = request.from_token_ids([7], request_id="golden-r0", max_new_tokens=2)
    second = request.from_token_ids([8, 9], request_id="golden-r1", max_new_tokens=2)
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch:
        root = Path(scratch)
        fleet = [dict(row, request_sha256=first["request_sha256"]) for row in rows]
        worker.persist(root / "resident-ready.json", dict(sequence=0))
        stage = [0]

        def collect(commands: Any, command: str, run_root: Path, label: str) -> None:
            for rank, row in enumerate(fleet):
                worker.persist(run_root / f"{label}.rank{rank}.log", row)

        def wait(seconds: float) -> None:
            stage[0] += 1
            if stage[0] == 1:
                path = root / "inbox" / "0001.json"
                path.write_bytes(json.dumps(second).encode())
                path.chmod(0o600)
            elif stage[0] == 2:
                job = root / "resident-0001"
                job.mkdir()
                fleet[:] = [dict(row, request_sha256=second["request_sha256"]) for row in fleet]
                worker.persist(root / "resident-ready.json", dict(sequence=1))
            elif stage[0] == 3:
                path = root / "inbox" / "stop.json"
                path.write_text('{"stop":true}')
                path.chmod(0o600)

        processes = [SimpleNamespace(poll=lambda: None, stdin=io.BytesIO()) for _ in range(8)]
        printed = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(launch, "remote_all", collect))
            stack.enter_context(mock.patch.object(launch.time, "sleep", wait))
            stack.enter_context(mock.patch("sys.stdout", printed))
            launch.resident_controller([], processes, root, "a" * 40, first, 3600, False)
        stdin = processes[0].stdin.getvalue()
        measurement = json.loads((root / "resident-measurement.json").read_text())
        out["resident_controller"] = dict(
            worker_stdin=_bytes_record(stdin), stdin_lines=len(stdin.splitlines()),
            stop_line=stdin.splitlines()[-1].decode(),
            all_workers_equal=len({p.stdin.getvalue() for p in processes}) == 1,
            command_prefix=stdin.splitlines()[0][:24].decode(),
            measurement_keys=sorted(measurement), layout=_tree(root),
            stdout_markers=sorted({line.split(" ", 1)[0] for line in printed.getvalue().splitlines() if line}))
    return out


def runtime_record_keys() -> dict[str, Any]:
    """Keys of the ``glm_optimized_runtime_v1`` record ``OrdinaryRuntime.__init__`` builds (static:
    the constructor needs devices). From S4 this locator follows the renamed module."""
    source = (REPO / "glm_tpu" / "optimized" / "runtime.py").read_text()
    for node in ast.walk(ast.parse(source)):
        targets = [t for t in getattr(node, "targets", []) if isinstance(t, ast.Attribute) and t.attr == "record"]
        if (isinstance(node, ast.Assign) and targets and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "dict"):
            keys = sorted(k.arg for k in node.value.keywords if k.arg)
            identity = next((k.value for k in node.value.keywords if k.arg == "physical_identity"), None)
            sub = sorted(k.arg for k in identity.keywords) if isinstance(identity, ast.Call) else []
            return dict(keys=keys, physical_identity_keys=sub)
    raise ValueError("runtime record constructor not found")


# ----------------------------------------------------------------------------- HTTP
_NORMALIZE = [
    (re.compile(rb"chatcmpl-[0-9a-f]{32}"), b"chatcmpl-X"),
    (re.compile(rb"call_[0-9a-f]{24}"), b"call_X"),
    (re.compile(rb'"created": [0-9]+(?:\.[0-9]+)?'), b'"created": 0'),
    (re.compile(rb'"(id|chat|job)": "[0-9a-f]{32}"'), rb'"\1": "X"'),
]


def normalize_body(raw: bytes) -> bytes:
    for pattern, replacement in _NORMALIZE:
        raw = pattern.sub(replacement, raw)
    return raw


class FakeResident:
    """The resident backend contract the UI/API use, without tokenizer, controller or model."""

    def __init__(self) -> None:
        self.answer = "42"
        self.thinking = "weighing it up"
        self.sequence = 770

    def check(self) -> None:
        pass

    def prepare(self, messages: Any, key: str) -> dict[str, Any]:
        return dict(prompt_ids=[1, 2] + [3] * len(messages), max_new_tokens=32000, request_id="ui-" + key,
                    context_capacity=32768)

    def prepare_api(self, messages: Any, key: str, *, tools: Any, effort: Any, max_new_tokens: Any,
                    context_capacity: int) -> dict[str, Any]:
        ids = [1] * (len(json.dumps(messages)) // 10 + 1)
        budget = context_capacity - len(ids) if max_new_tokens is None else max_new_tokens
        return dict(prompt_ids=ids, max_new_tokens=budget, request_id="api-" + key, context_capacity=context_capacity)

    def next_sequence(self) -> int:
        return self.sequence + 1

    def publish(self, job: Any, root: Any) -> None:
        pass

    def observe(self, job: Any) -> dict[str, Any]:
        self.sequence = job["sequence"]
        return dict(status="complete", answer=self.answer, thinking=self.thinking, output_tokens=7,
                    decode_tps=13.5, prefill_seconds=1.0, stop_cause="eos")


def http_record() -> dict[str, Any]:
    from glm_tpu.api import Api
    from glm_tpu.ui import Chats, ThreadingHTTPServer, handler

    token = "golden-local-key"
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-http-") as scratch:
        backend = FakeResident()
        store = Chats(Path(scratch) / "state", backend)
        service = Api(store, backend, capacity=32768, wait_seconds=60)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler(store, service, token))
        port = server.server_port
        stop = threading.Event()

        def pump() -> None:
            while not stop.is_set():
                store.step()
                time.sleep(0.01)

        threads = [threading.Thread(target=server.serve_forever, daemon=True),
                   threading.Thread(target=pump, daemon=True)]
        for thread in threads:
            thread.start()
        host = f"127.0.0.1:{port}"
        cases: dict[str, Any] = {}

        def call(name: str, method: str, path: str, *, body: Any = None, headers: dict[str, str] | None = None,
                 auth: bool = False, ui: bool = False, host_header: str | None = None) -> bytes:
            data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
            sent = {"Host": host_header or host}
            if auth:
                sent["Authorization"] = "Bearer " + token
            if ui:
                sent["X-GLM-UI"] = "1"
            if data is not None:
                sent["Content-Type"] = "application/json"
            sent.update(headers or {})
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            try:
                connection.request(method, path, body=data, headers=sent)
                response = connection.getresponse()
                raw = response.read()
                status, header_list = response.status, response.getheaders()
            finally:
                connection.close()
            normalized = normalize_body(raw)
            kept = []
            for key, value in header_list:
                if key.lower() == "date":
                    continue
                if key.lower() == "server":
                    value = re.sub(r"Python/[0-9.]+", "Python/X", value)
                if key.lower() == "content-length" and normalized != raw:
                    value = "X"
                kept.append([key, value])
            record: dict[str, Any] = dict(status=status, headers=kept)
            if len(normalized) > 4096:
                record["body"] = _bytes_record(normalized)
            else:
                record["body_text"] = normalized.decode("utf-8")
            cases[name] = record
            return raw

        def wait_for(predicate: Any) -> None:
            deadline = time.monotonic() + 30
            while not predicate():
                if time.monotonic() > deadline:
                    raise TimeoutError("fake resident job did not complete")
                time.sleep(0.02)

        try:
            call("get_index", "GET", "/")
            call("get_app_js", "GET", "/app.js")
            call("get_style_css", "GET", "/style.css")
            call("get_state_empty", "GET", "/api/state")
            created = json.loads(call("post_chat_create", "POST", "/api/chat", body=dict(action="create"), ui=True))
            call("post_chat_send", "POST", "/api/chat", ui=True,
                 body=dict(action="send", chat=created["id"], id="a" * 32, text=NON_ASCII_MESSAGES[1]["content"]))
            wait_for(lambda: all(j["status"] == "complete" for j in store.snapshot()["jobs"]))
            call("get_state_after_answer", "GET", "/api/state")
            call("get_models", "GET", "/v1/models", auth=True)
            call("post_completion", "POST", "/v1/chat/completions", auth=True,
                 body=dict(model="glm-5.3", messages=NON_ASCII_MESSAGES))
            call("post_completion_stream", "POST", "/v1/chat/completions", auth=True,
                 body=dict(model="glm-5.3-low", messages=NON_ASCII_MESSAGES, stream=True))
            backend.answer = ("I will look.<tool_call>read_file<arg_key>path</arg_key><arg_value>a.txt</arg_value>"
                              "<arg_key>lines</arg_key><arg_value>12</arg_value></tool_call>")
            tools = [dict(type="function", function=dict(name="read_file", description="Read a file",
                                                        parameters=dict(type="object",
                                                                        properties=dict(path=dict(type="string")))))]
            call("post_completion_tool_call", "POST", "/v1/chat/completions", auth=True,
                 body=dict(model="glm-5.3", messages=[dict(role="user", content="read a.txt")], tools=tools))
            call("post_completion_tool_call_stream", "POST", "/v1/chat/completions", auth=True,
                 body=dict(model="glm-5.3-high", messages=[dict(role="user", content="read a.txt")], tools=tools,
                           stream=True))
            backend.answer = "42"
            call("error_401_models", "GET", "/v1/models")
            call("error_401_completion", "POST", "/v1/chat/completions", body=dict(messages=[]))
            call("error_403_host", "GET", "/", host_header="evil.example:80")
            call("error_403_ui_header", "POST", "/api/chat", body=dict(action="create"))
            call("error_404_asset", "GET", "/nope")
            call("error_404_v1", "GET", "/v1/nope", auth=True)
            call("error_404_ui", "POST", "/api/nope", ui=True, body=dict(action="create"))
            call("error_404_model", "POST", "/v1/chat/completions", auth=True,
                 body=dict(model="gpt-4", messages=[dict(role="user", content="hi")]))
            call("error_400_json", "POST", "/v1/chat/completions", auth=True, body=b"{not json")
            call("error_400_effort", "POST", "/v1/chat/completions", auth=True,
                 body=dict(messages=[dict(role="user", content="hi")], reasoning_effort="medium"))
            call("error_400_chat", "POST", "/api/chat", ui=True, body=dict(action="send", chat="0" * 32))
        finally:
            stop.set()
            server.shutdown()
            server.server_close()
    return cases


def record() -> dict[str, Any]:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    wire = dict(requests=requests_record(), worker=worker_record(), controller=controller_record(),
                runtime_record=runtime_record_keys())
    return dict(wire=wire, http=http_record(), source=source_record())


def main() -> int:
    emit(record())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
