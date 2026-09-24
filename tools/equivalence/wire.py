"""G9: serving wire formats and request identity, characterized through the real code.

Nothing here reimplements a format: every byte is produced by the production functions, driven
with deterministic synthetic inputs and fakes for the parts that need a fleet, a tokenizer or a
device. Recorded:

* request bodies and ``request_sha256`` for every context profile, sequential and concurrent
  batches, refusals (including a non-ASCII request id, which the 181c013e profile rejects) and
  the two canonical-JSON contracts on non-ASCII message content;
* worker output files (``tokens.jsonl`` TokenEvent lines solo and with ``batch_round``,
  ``answer.txt``, directory layout) and the prefill block schedule (rows, count; a 114-token tail
  included) from the real ``run_queued`` over the real ``OrdinaryRuntime.generate`` -- at 8,192 and
  at the donated long-context capacities 32,768 and 166,912 -- and the real ``run_concurrent`` over
  the real ``generate_concurrent`` -> ``batched_runtime.generate_batch`` -> ``BatchedSession``, both
  with synthetic device results;
* the fleet votes' refusals (``fleet_refusals``): a host with a different output digest at the
  output consensus and a prefill with an invalid contract, sequential and batched;
* the API's messages-size measure, found by bisection over ``glm_tpu.entrypoints.openai.chat_utils.convert`` itself;
* the resident protocol (ready file bytes, command and stop bytes written to worker stdin,
  per-round records) from the real ``resident_loop`` and ``resident_controller``;
* record key sets: worker ``runner.rank{r}.json`` from the real worker ``main`` (sequential and
  concurrent) with its real ``preflight`` against a synthetic staged run directory and its real
  ``_initialize_runtime`` over synthetic 2x4x4 topology captures and a synthetic staged
  ``site.json`` (faked: the environment marker, the hostname, the checkpoint pins ``site_args``
  binds from the site and the template check -- both read private assets --, ``jax.distributed``,
  the device queries and ``Mesh``); recorded: the arguments
  ``preflight`` binds, the topology binding it authenticates, the ``jax.distributed`` arguments,
  the mesh axis names and device order (digest), the arguments ``main`` passes to
  ``OrdinaryRuntime`` (names and described values, e.g. ``context_capacity``,
  ``concurrent_size``), and the refusals ``preflight`` and ``_initialize_runtime`` must produce on
  tampered inputs (deployed-source digest, existing namespace, coordinator port, owner-only
  modes, request, binding and site digests, staged coordinator, host mapping), and the jax
  configuration and JAX/XLA/libtpu
  environment the worker process has when it constructs the runtime (changes against a reference
  taken before any production import); controller ``summarize()``, runtime request reports and
  phase names;
* the controller's launch path: the real launcher ``main`` against a synthetic host
  (``controller.launcher_record``: locks, staged bundle, remote command lines, preflight, the
  worker environment and ``execv``, collection, the failure path);
* HTTP through the real UI/API handler with a fake resident: status, headers (CSP included)
  and body bytes, SSE streams byte for byte, ids and timestamps normalized by regex.

Run as ``python -m tools.equivalence.wire`` (prints one JSON line: ``{"wire": ..., "http": ...}``).
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
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
from typing import Any
from unittest import mock

import numpy as np

from .common import REPO, digest_json, emit, sha256_hex, source_record

NON_ASCII_MESSAGES = [
    {"role": "system", "content": "Sei un assistente preciso. 中文 ✓"},
    {"role": "user", "content": "Quanto fa 6×7? Risposta in una riga, caffè."},  # noqa: RUF001 (non-ASCII chat content on purpose)
]
PROFILES = (8192, 32768, 166912, 262144)
SYNTHETIC_COORDINATOR = "203.0.113.10:8476"  # documentation address (RFC 5737), production port
SYNTHETIC_CODE_HASH = "a" * 40


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
    from glm_tpu.utils.json_utils import canonical
    from glm_tpu.entrypoints.openai.chat_utils import convert
    from glm_tpu.engine import request

    out: dict[str, Any] = {}
    prompt = [(i * 7919 + 13) % 154880 for i in range(96)]
    bodies = {}
    for capacity in PROFILES:
        value = request.from_token_ids(
            prompt, request_id=f"golden-{capacity}", max_new_tokens=64, context_capacity=capacity
        )
        bodies[capacity] = value
        out[f"single_{capacity}"] = dict(
            _bytes_record(canonical(value)), request_sha256=value["request_sha256"], keys=sorted(value)
        )
    sequential = request.batch(
        [bodies[8192], request.from_token_ids(prompt[:40], request_id="golden-8192-b", max_new_tokens=8)]
    )
    out["batch_sequential"] = dict(
        _bytes_record(canonical(sequential)), request_sha256=sequential["request_sha256"], keys=sorted(sequential)
    )
    lanes = [
        request.from_token_ids(prompt[: 20 + 10 * i], request_id=f"lane-{i}", max_new_tokens=32, context_capacity=32768)
        for i in range(4)
    ]
    concurrent = request.batch(lanes, concurrent=True)
    out["batch_concurrent"] = dict(
        _bytes_record(canonical(concurrent)), request_sha256=concurrent["request_sha256"], keys=sorted(concurrent)
    )
    tampered = dict(bodies[32768], max_new_tokens=65)
    out["refusals"] = dict(
        non_ascii_request_id=_error(lambda: request.from_token_ids(prompt, request_id="richiesta-è", max_new_tokens=8)),
        tampered=_error(lambda: request.validate(tampered)),
        unsupported_capacity=_error(
            lambda: request.from_token_ids(prompt, request_id="x", max_new_tokens=8, context_capacity=4096)
        ),
        over_budget=_error(lambda: request.from_token_ids(prompt, request_id="x", max_new_tokens=8150)),
        concurrent_capacity=_error(
            lambda: request.batch(
                [request.from_token_ids(prompt, request_id=f"c{i}", max_new_tokens=8) for i in range(2)],
                concurrent=True,
            )
        ),
        duplicate_ids=_error(lambda: request.batch([bodies[8192], bodies[8192]])),
    )
    out["stop_cause"] = dict(
        eos=request.stop_cause(bodies[8192], "eos"),
        output_cap=request.stop_cause(bodies[8192], "length"),
        context_exhausted=request.stop_cause(
            request.from_token_ids(prompt, request_id="full", max_new_tokens=8192 - len(prompt)), "length"
        ),
    )
    out["messages_non_ascii"] = dict(
        wire=_bytes_record(canonical(NON_ASCII_MESSAGES)),
        api_cap_measure=api_cap_measure(),
        converted=_bytes_record(canonical(convert(NON_ASCII_MESSAGES))),
    )
    out["constants"] = dict(
        schemas=[request.SCHEMA, request.BATCH_SCHEMA, request.CONCURRENT_SCHEMA],
        capacities=list(request.CAPACITIES),
        prompt_limits={str(k): v for k, v in request.PROMPT_LIMITS.items()},
        concurrent_limit=request.CONCURRENT_LIMIT,
        payload_cap=request.PAYLOAD_CAP,
        messages_cap=request.MESSAGES_CAP,
        max_new=request.MAX_NEW,
        vocab=request.VOCAB,
        eos=list(request.EOS),
    )
    return out


CAP_PROBES = (("ascii", "a"), ("latin", "\u00e8"), ("cjk", "\u4e2d"), ("astral", "\U0001f600"))


def api_cap_measure() -> dict[str, Any]:
    """The largest one-message content (in characters, per character class) that
    ``glm_tpu.entrypoints.openai.chat_utils.convert`` accepts, found by bisection over ``convert``
    itself -- so the record is
    the API's own size measure (at 181c013e ``len(json.dumps(messages).encode())``, i.e. ASCII
    escapes: 6 bytes per non-ASCII BMP character, 12 per astral one)."""
    from glm_tpu.entrypoints.openai.protocol import ApiError
    from glm_tpu.entrypoints.openai.chat_utils import convert

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
    from glm_tpu.models.glm_moe_dsa.state import DecoderState

    return DecoderState(
        np.zeros((1,)),
        np.zeros((1,)),
        np.zeros((1, 1), np.int32),
        np.ones((1,), np.int32),
        np.zeros((1, 1), np.float32),
        np.array([position], np.int32),
        np.zeros((1, 1), np.int32),
        np.array([position + 1], np.int32),
        np.array([healthy]),
    )


def _runtime(outputs: list[int], *, capacity: int = 8192, healthy: bool = True) -> Any:
    """The real OrdinaryRuntime host logic over synthetic device results (no devices). ``capacity``
    is the loaded context (a long-context runtime donates its state; its host logic runs here);
    ``healthy=False`` makes every prefill report an invalid contract (a failed prefill)."""
    from glm_tpu.models.glm_moe_dsa.state import BatchedPrefillResult, BatchedPrefillState
    from glm_tpu.models.glm_moe_dsa.state import DecodeStepResult
    from glm_tpu.models.glm_moe_dsa.model import PackedDecodeResult
    from glm_tpu.runner.tpu_runner import OrdinaryRuntime

    runtime = object.__new__(OrdinaryRuntime)
    runtime.capacity, runtime.concurrent_size, runtime.active = capacity, 0, False
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
        return BatchedPrefillResult(
            BatchedPrefillState(_state(length, healthy), np.zeros((1,)), np.array(length, np.int32), np.array(True)),
            np.array([outputs[0]], np.int32),
        )

    def decode(token: Any, previous: Any, *args: Any) -> Any:
        calls.append(int(np.asarray(token)[0]))
        index = len(calls)
        position = prompt["length"] + index
        value = outputs[min(index, len(outputs) - 1)]
        step = DecodeStepResult(_state(position), np.array([value], np.int32), np.zeros((1, 1)))
        return PackedDecodeResult(step, np.array([value, 1, position, position + 1], np.int32))

    runtime.initialize = initialize
    runtime.prefill = {114: execute_prefill, 128: execute_prefill}
    runtime.prefill_calls = []
    runtime.decode = decode
    return runtime


def _batched_runtime(schedules: list[list[int]], *, healthy: bool = True) -> Any:
    """The real ``OrdinaryRuntime.generate_concurrent`` -> ``batched_runtime.generate_batch`` ->
    ``BatchedSession`` host logic over synthetic device results (no devices). ``schedules[lane]``
    lists a lane's tokens: the first from prefill, the next ones from successive decode rounds;
    ``healthy=False`` makes every prefill report an invalid contract."""
    from glm_tpu.models.glm_moe_dsa.state import BatchedPrefillResult, BatchedPrefillState
    from glm_tpu.runner.tpu_runner import OrdinaryRuntime

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
        return BatchedPrefillResult(
            BatchedPrefillState(_state(length, healthy), np.zeros((1,)), np.array(length, np.int32), np.array(True)),
            np.array([schedules[lane][0]], np.int32),
        )

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


def _allgather(value: Any, *, divergent: bool = False) -> Any:
    """``process_allgather`` of eight hosts; ``divergent``: host 3 reports a different value."""
    rows = np.stack([np.asarray(value)] * 8)
    if divergent:
        rows = rows.copy()
        rows[3] = rows[3] ^ np.asarray(1, rows.dtype)
    return rows


@contextmanager
def _installed_site(site: Any) -> Iterator[Any]:
    """``site`` as the process's current site for the block (the worker entry installs its own)."""
    from glm_tpu.config.site import set_current_site

    previous = set_current_site(site)
    try:
        yield site
    finally:
        set_current_site(previous)


@contextmanager
def _fleet_fakes(*, divergent: bool = False) -> Iterator[None]:
    from jax.experimental import multihost_utils

    from .site_fixture import site as synthetic_site

    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "decoded:" + ",".join(map(str, tokens)))
    with ExitStack() as stack:
        # the answer writer finds the tokenizer through the current site (patched loader below)
        stack.enter_context(_installed_site(synthetic_site(Path("/site"))))
        stack.enter_context(
            mock.patch.object(
                multihost_utils, "process_allgather", lambda value: _allgather(value, divergent=divergent)
            )
        )
        stack.enter_context(mock.patch("transformers.AutoTokenizer.from_pretrained", lambda *args, **kwargs: tokenizer))
        yield


_REPORT_VALUES = (
    "emitted",
    "finish_reason",
    "stop_cause",
    "output_directory",
    "output_budget_tokens",
    "token_sha256",
    "request_sha256",
    "prompt_tokens",
    "timed_decode_tokens",
    "sampling",
    "speculative",
    "request_id",
)


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) + ("/" if p.is_dir() else "") for p in root.rglob("*"))


LONG_CONTEXTS = (32768, 166912)  # donated runtimes (capacity > 8,192); the 32K and 128K profiles


def _run_queued(capacity: int) -> dict[str, Any]:
    """The real ``run_queued`` over the real ``generate`` of a runtime loaded at ``capacity``: four
    queued requests (3, 5, 114 and 242 = 128 + 114 prompt tokens), warm-up included."""
    from glm_tpu.engine import request
    from glm_tpu.worker import tpu_worker as worker

    items = [
        request.from_token_ids([30, 31, 32], request_id="golden-q1", max_new_tokens=4, context_capacity=capacity),
        request.from_token_ids(
            [40, 41, 42, 43, 44], request_id="golden-q2", max_new_tokens=3, context_capacity=capacity
        ),
        request.from_token_ids(
            [50 + i % 7 for i in range(114)], request_id="golden-q3", max_new_tokens=2, context_capacity=capacity
        ),
        request.from_token_ids(
            [60 + i % 5 for i in range(242)], request_id="golden-q4", max_new_tokens=2, context_capacity=capacity
        ),
    ]
    value = request.batch(items)
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch, _fleet_fakes():
        root = Path(scratch)
        runtime = _runtime([7, 9, 10, 154820], capacity=capacity)
        reports = worker.run_queued(
            runtime, request.requests(value), value, root, 0, time.perf_counter() + 600, save=lambda reports: None
        )
        return dict(
            layout=_tree(root),
            tokens=[(root / f"item{i:03d}" / "tokens.jsonl").read_text() for i in range(len(items))],
            answers=[(root / f"item{i:03d}" / "answer.txt").read_text() for i in range(len(items))],
            report_keys=sorted(reports[0]),
            reports=[{k: r[k] for k in _REPORT_VALUES if k in r} for r in reports],
            runtime_phases=sorted(runtime.record["phases"]),
            runtime_record_keys=sorted(runtime.record),
            prefill_calls=runtime.prefill_calls,
        )


def _attempt(call: Any) -> str:
    try:
        call()
    except Exception as exc:  # the expected refusal
        return f"{type(exc).__name__}: {exc}"
    return "accepted"


def fleet_refusals() -> dict[str, str]:
    """The fleet votes of ``generate`` and ``generate_batch`` refuse: a host whose output digest
    differs at the output consensus, and a prefill whose contract is invalid (the local health vote
    is false), sequential and batched. ``accepted`` means the check is gone."""
    from glm_tpu.engine import request

    single = request.from_token_ids([30, 31, 32], request_id="golden-vote", max_new_tokens=3)
    lanes = [
        request.from_token_ids(
            [7 + lane] * (3 + lane), request_id=f"vote-lane{lane}", max_new_tokens=2, context_capacity=32768
        )
        for lane in range(2)
    ]
    schedules = [[10, 11, 12], [20, 21, 22]]
    out: dict[str, str] = {}

    def sequential(*, healthy: bool = True) -> None:
        _runtime([7, 9, 10], healthy=healthy).generate(
            single, deliver=lambda event: None, deadline=time.perf_counter() + 600
        )

    def batched(*, healthy: bool = True) -> None:
        _batched_runtime(schedules, healthy=healthy).generate_concurrent(
            lanes, deliver=lambda *args: None, deadline=time.perf_counter() + 600
        )

    with _fleet_fakes(divergent=True):
        out["output_consensus_divergent_host"] = _attempt(sequential)
        out["batch_output_consensus_divergent_host"] = _attempt(batched)
    with _fleet_fakes():
        out["prefill_health_vote_false"] = _attempt(lambda: sequential(healthy=False))
        out["batch_prefill_health_vote_false"] = _attempt(lambda: batched(healthy=False))
    return out


def worker_record() -> dict[str, Any]:
    from glm_tpu.engine import request
    from glm_tpu.worker import tpu_worker as worker

    out: dict[str, Any] = {}
    out["run_queued"] = _run_queued(8192)
    out["run_queued_long_context"] = {str(capacity): _run_queued(capacity) for capacity in LONG_CONTEXTS}
    out["fleet_refusals"] = fleet_refusals()

    with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch, _fleet_fakes():
        root = Path(scratch)
        lengths, budgets = (130, 114, 20, 242), (3, 2, 4, 3)
        pending = [
            request.from_token_ids(
                [7 + lane] * length, request_id=f"lane{lane}", max_new_tokens=budget, context_capacity=32768
            )
            for lane, (length, budget) in enumerate(zip(lengths, budgets, strict=True))
        ]
        # lane 1 stops on EOS in round 1; the others run to their output budget (length)
        schedules = [[10, 11, 12, 13], [20, 154820, 22], [30, 31, 32, 33, 34], [40, 41, 42, 43]]
        runtime = _batched_runtime(schedules)
        reports, aggregate = worker.run_concurrent(runtime, pending, root, 0, time.perf_counter() + 600)
        out["run_concurrent"] = dict(
            layout=_tree(root),
            tokens=[(root / f"item{i:03d}" / "tokens.jsonl").read_text() for i in range(len(pending))],
            answers=[(root / f"item{i:03d}" / "answer.txt").read_text() for i in range(len(pending))],
            report_keys=sorted(reports[0]),
            aggregate_keys=sorted(aggregate),
            reports=[
                {k: r[k] for k in (*_REPORT_VALUES, "batch_size", "batch_rounds", "context_capacity") if k in r}
                for r in reports
            ],
            aggregate=dict(batch_size=aggregate["batch_size"], decode_rounds=aggregate["decode_rounds"]),
            runtime_phases=sorted(runtime.record["phases"]),
            runtime_record_keys=sorted(runtime.record),
            batches=len(runtime.record.get("batches", [])),
            prefill_calls=runtime.prefill_calls,
        )

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
        out["resident_loop"] = dict(
            layout=_tree(root),
            ready=(root / "resident-ready.json").read_text(),
            round_record_keys=sorted(record),
            calls=seen,
            persist_format=_bytes_record((root / "resident-ready.json").read_bytes()),
        )
    out["main_record"] = worker_main_record()
    return out


# ----------------------------------------------------------------------------- worker process (preflight, init)
def _run_name() -> str:
    """A run-directory name of the form the worker requires; synthetic timestamp (the epoch)."""
    return "optimized_request_" + time.strftime("%Y%m%dT%H%M%S", time.gmtime(0)) + "000000Z"


def _canonical_sha(value: Any) -> str:
    return sha256_hex(json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True))


def synthetic_fleet() -> dict[str, Any]:
    """Synthetic 2x4x4 topology (the G4 one), its eight launch captures and an authenticated
    topology rebinding: the inputs of the worker's topology binding and ``_initialize_runtime``."""
    from glm_tpu.distributed.mesh import build_physical_mesh

    from .identities import _synthetic_topology

    topology = _synthetic_topology()
    physical = build_physical_mesh(topology)
    contract = dict(topology=topology.to_dict(), code_hash=SYNTHETIC_CODE_HASH)
    contract_hash = _canonical_sha(contract)
    order = [[process * 4 + offset for offset in range(4)] for process in range(8)]
    captures = [
        dict(
            captured_utc="1970-01-01T00:00:00Z",
            contract=contract,
            contract_hash=contract_hash,
            fleet_contract_hashes=[contract_hash] * 8,
            fleet_local_device_ids_in_runtime_order=order,
            hostname=f"example-w-{i}",
            jax_device_count=32,
            jax_local_device_count=4,
            jax_process_count=8,
            jax_process_index=i,
            jax_version="synthetic",
            launch_process_id=i,
            local_device_ids=order[i],
            schema_version=1,
        )
        for i in range(8)
    ]
    fleet = _canonical_sha(
        dict(
            fleet_local_device_ids_in_runtime_order=order,
            records=[
                {
                    k: c[k]
                    for k in ("contract_hash", "hostname", "jax_process_index", "launch_process_id", "local_device_ids")
                }
                for c in captures
            ],
        )
    )
    payloads = {f"topology.rank{i}.json": json.dumps(c, sort_keys=True).encode() for i, c in enumerate(captures)}
    binding = dict(
        schema="glm_perf_topology_rebinding_v1",
        physical_devices_identical=True,
        all_hosts_idle_after=True,
        original_topology_sha256=topology.topology_hash,
        mesh_sha256=physical.mesh_hash,
        original_fleet_sha256=fleet,
        fleet_sha256=fleet,
        capture_sha256={name: sha256_hex(raw) for name, raw in payloads.items()},
        host_to_slots={
            str(i): [s for s, d in enumerate(physical.flattened_device_ids) if d in order[i]] for i in range(8)
        },
        code_hash=SYNTHETIC_CODE_HASH,
    )
    return dict(
        topology=topology,
        physical=physical,
        fleet=fleet,
        payloads=payloads,
        binding=json.dumps(binding, sort_keys=True).encode(),
    )


def _write_private(path: Path, raw: bytes, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
    os.chmod(path, mode)


def stage_run(runs: Path, value: dict[str, Any], fleet: dict[str, Any], *, tamper: str | None = None) -> list[str]:
    """A staged run directory as the controller leaves it for the worker (owner-only request,
    source manifest of real repository files, the controller-resolved synthetic ``site.json``,
    topology rebinding with its captures); returns the worker argv. ``tamper`` breaks exactly one
    input (refusal probes)."""
    from glm_tpu.utils import json_utils
    from glm_tpu.worker import tpu_worker as worker

    from .site_fixture import site as synthetic_site

    root = runs / _run_name()
    root.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    raw_request = json_utils.canonical(value) + b"\n"
    names = [
        Path(worker.__file__).resolve().relative_to(REPO.resolve()).as_posix(),
        Path(importlib_file("glm_tpu.runner.tpu_runner")).resolve().relative_to(REPO.resolve()).as_posix(),
    ]
    manifest = {name: sha256_hex((REPO / name).read_bytes()) for name in names}
    if tamper == "deployed_source":
        manifest[names[0]] = "0" * 64
    raw_manifest = json.dumps(manifest, sort_keys=True).encode()
    _write_private(root / "request.json", raw_request, 0o640 if tamper == "owner_only" else 0o600)
    _write_private(root / "source_manifest.json", raw_manifest)
    site = synthetic_site(
        Path("/site"),
        fleet=dict(coordinator_address=SYNTHETIC_COORDINATOR),
        paths=dict(run_root=str(runs), model_path="/site/model", hlo_dump_root="/site/hlo"),
    )
    raw_site = site.resolved_json()
    _write_private(root / "site.json", raw_site)
    _write_private(root / "topology_rebinding.json", fleet["binding"])
    (root / "topology_capture").mkdir(mode=0o700)
    for name, raw in fleet["payloads"].items():
        _write_private(root / "topology_capture" / name, raw)
    if tamper == "existing_namespace":
        (root / "native.rank0").mkdir(mode=0o700)
    return [
        "--output",
        str(root),
        "--code-hash",
        SYNTHETIC_CODE_HASH,
        "--source-manifest-sha256",
        sha256_hex(raw_manifest),
        "--request-file-sha256",
        "f" * 64 if tamper == "request_digest" else sha256_hex(raw_request),
        "--site-sha256",
        "d" * 64 if tamper == "site_digest" else sha256_hex(raw_site),
        "--topology-rebinding-sha256",
        "e" * 64 if tamper == "binding_pin" else sha256_hex(fleet["binding"]),
        "--coordinator-address",
        SYNTHETIC_COORDINATOR.replace(":8476", ":8477")
        if tamper == "port"
        else SYNTHETIC_COORDINATOR.replace(".10:", ".11:")
        if tamper == "site_coordinator"
        else SYNTHETIC_COORDINATOR,
        "--wall-seconds",
        "60",
    ]


def importlib_file(module: str) -> str:
    import importlib

    return str(importlib.import_module(module).__file__)


@dataclass(frozen=True)
class FakeDevice:
    """A TPU device as ``_initialize_runtime`` reads it (from the synthetic topology)."""

    id: int
    process_index: int
    local_hardware_id: int
    coords: tuple[int, ...]
    core_on_chip: int
    platform: str
    device_kind: str


class RecordedMesh:
    """``jax.sharding.Mesh`` stand-in: the device grid and axis names ``_initialize_runtime`` built."""

    def __init__(self, devices: Any, axis_names: Any) -> None:
        self.devices = np.asarray(devices, dtype=object)
        self.axis_names = tuple(axis_names)


@contextmanager
def worker_host(fleet: dict[str, Any], runs: Path, *, hostname: str = "example-w-0") -> Iterator[dict[str, Any]]:
    """The host a worker runs on, faked: environment marker, hostname, the checkpoint pins
    ``site_args`` binds from the staged site (it reads the private source-completion receipt) and
    the template check (private tokenizer), ``jax.distributed`` and the device queries (the
    synthetic topology's rank-0 view), ``Mesh``. The staged ``site.json`` itself is read, hashed and
    validated by the real preflight, which installs it as the current site (restored here)."""
    import jax
    import jax.sharding

    from glm_tpu.config.site import get_current_site
    from glm_tpu.config import model
    from glm_tpu.engine import resident_protocol as protocol
    from glm_tpu.worker import tpu_worker as worker

    calls: dict[str, Any] = dict(template=[], distributed=[], meshes=[])
    topology, physical = fleet["topology"], fleet["physical"]
    devices = [
        FakeDevice(
            d.device_id,
            d.process_index,
            d.local_device_id,
            tuple(d.coordinates),
            d.core_on_chip,
            d.platform,
            d.device_kind,
        )
        for d in topology.devices
    ]
    process = int(hostname.rsplit("-w-", 1)[1])

    def site_binding(args: Any, site: Any) -> Any:
        for name, value in dict(
            model_id=model.MODEL_ID,
            model_revision=model.REVISION,
            source_inventory_sha256="1" * 64,
            checkpoint_manifest_sha256="2" * 64,
            checkpoint_success_sha256="3" * 64,
            source_complete_sha256="4" * 64,
            checkpoint_root=Path("/site/checkpoint"),
            source_inventory=Path("/site/inventory"),
            checkpoint_transport="shm",
            topology_capture_root=Path("/site/captures"),
            topology_sha256=topology.topology_hash,
            topology_fleet_sha256=fleet["fleet"],
            mesh_sha256=physical.mesh_hash,
            slice_name=topology.slice_name,
            num_processes=8,
            hlo_dump_root=site.paths.hlo_dump_root,
        ).items():
            setattr(args, name, value)
        return args

    def verified_template(repo: Any, tokenizer_root: Any) -> str:
        calls["template"].append(
            dict(
                repo_is_source=Path(repo).resolve() == REPO.resolve(),
                tokenizer_is_site_root=Path(tokenizer_root) == get_current_site().paths.model_path,
            )
        )
        return "<template>"

    def distributed(**kwargs: Any) -> None:
        calls["distributed"].append(kwargs)

    def mesh(devices: Any, axis_names: Any) -> RecordedMesh:
        calls["meshes"].append(RecordedMesh(devices, axis_names))
        return calls["meshes"][-1]

    with ExitStack() as stack:
        stack.enter_context(_installed_site(None))  # restores the current site the preflight installs
        stack.enter_context(mock.patch.dict(os.environ, {protocol.WORKER_ENV_FLAG: "1"}))
        stack.enter_context(mock.patch.object(worker, "site_args", site_binding))
        stack.enter_context(mock.patch.object(model, "verified_template", verified_template))
        stack.enter_context(mock.patch("socket.gethostname", lambda: hostname))
        stack.enter_context(mock.patch.object(jax.distributed, "initialize", distributed))
        stack.enter_context(mock.patch.object(jax, "default_backend", lambda: "tpu"))
        stack.enter_context(mock.patch.object(jax, "device_count", lambda *args: 32))
        stack.enter_context(mock.patch.object(jax, "process_count", lambda *args: 8))
        stack.enter_context(mock.patch.object(jax, "process_index", lambda *args: process))
        stack.enter_context(mock.patch.object(jax, "devices", lambda *args: list(devices)))
        stack.enter_context(
            mock.patch.object(
                jax, "local_devices", lambda *args, **kwargs: [d for d in devices if d.process_index == process]
            )
        )
        stack.enter_context(mock.patch.object(jax.sharding, "Mesh", mesh))
        yield calls


def _describe_arg(value: Any, root: Path, labels: dict[str, str] | None = None) -> Any:
    """A bound worker argument: numbers and strings as they are (synthetic), paths relative to the
    staged run directory, values with a label (e.g. the source-manifest digest, which follows the
    bytes of the repository files it lists) by the label, anything else by type."""
    if isinstance(value, str) and labels and value in labels:
        return labels[value]
    if isinstance(value, Path):
        text = str(value)
        if text == str(root) or text.startswith(str(root) + "/"):
            return "<run>" + text[len(str(root)) :]
        return text if text.startswith("/site/") else f"<path:{value.name}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return f"<{type(value).__name__}>"


def _mesh_record(mesh: Any, physical: Any) -> dict[str, Any]:
    order = [int(device.id) for device in np.asarray(mesh.devices).flat]
    return dict(
        axis_names=list(mesh.axis_names),
        shape=list(np.asarray(mesh.devices).shape),
        device_order_digest=digest_json(order),
        device_order_is_physical=order == list(physical.flattened_device_ids),
    )


def worker_main_record() -> dict[str, Any]:
    """``runner.rank{r}.json`` from the real worker ``main`` (real ``preflight`` and
    ``_initialize_runtime`` on a synthetic staged run; devices, fleet and model faked), for a
    sequential request and a concurrent batch, the arguments ``main`` passes to
    ``OrdinaryRuntime`` (names, and values described: numbers and strings as they are, objects by
    where they came from, the ``save`` callback by the file it writes), what ``preflight`` bound
    and ``_initialize_runtime`` built, and the refusals both must produce."""
    from glm_tpu.engine import request

    sequential = request.from_token_ids([30, 31, 32], request_id="golden-main", max_new_tokens=3)
    concurrent = request.batch(
        [
            request.from_token_ids(
                [30 + lane] * (4 + lane), request_id=f"golden-lane{lane}", max_new_tokens=3, context_capacity=32768
            )
            for lane in range(3)
        ],
        concurrent=True,
    )
    record = dict(_worker_main_run(sequential), concurrent=_worker_main_run(concurrent))
    record["refusals"] = worker_refusals(sequential)
    return record


def _worker_main_run(value: dict[str, Any]) -> dict[str, Any]:
    from glm_tpu.worker import tpu_worker as worker

    import glm_tpu.distributed.parallel_state as parallel_state
    import glm_tpu.runner.tpu_runner as runtime_module

    constructed: list[dict[str, Any]] = []
    seen: dict[str, Any] = {}

    class FakeRuntime:
        def __init__(self, **kwargs: Any) -> None:
            constructed.append(kwargs)
            seen["construction"] = process_changes()  # what the worker process set before building it
            self.record = dict(cold_load_compile_seconds=1.0, programs={"decode": {}}, physical_identity={})

    real_preflight, real_initialize = worker.preflight, parallel_state._initialize_runtime

    def preflight(args: Any) -> Any:  # pass-through: the real preflight runs
        result = real_preflight(args)
        seen["preflight"] = (args, result)
        return result

    def initialize(args: Any) -> Any:  # pass-through: the real _initialize_runtime runs
        seen["initialize"] = real_initialize(args)
        return seen["initialize"]

    def fake_run(runtime: Any, pending: Any, value: Any, root: Any, rank: int, deadline: Any, **kwargs: Any) -> Any:
        return [dict(emitted=3, token_sha256="c" * 64, finish_reason="length") for _ in pending]

    def fake_concurrent(runtime: Any, pending: Any, root: Any, rank: int, deadline: Any) -> Any:
        return fake_run(runtime, pending, None, root, rank, deadline), dict(batch_size=len(pending))

    fleet = synthetic_fleet()
    umask = os.umask(0o077)
    os.umask(umask)
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch:
        runs = Path(scratch) / "runs"
        runs.mkdir(mode=0o700)
        argv = stage_run(runs, value, fleet)
        root = Path(argv[1])
        with ExitStack() as stack:
            calls = stack.enter_context(worker_host(fleet, runs))
            stack.enter_context(mock.patch.object(worker, "preflight", preflight))
            stack.enter_context(mock.patch.object(worker, "run_queued", fake_run))
            stack.enter_context(mock.patch.object(worker, "run_concurrent", fake_concurrent))
            stack.enter_context(mock.patch.object(parallel_state, "_initialize_runtime", initialize))
            stack.enter_context(mock.patch.object(runtime_module, "OrdinaryRuntime", FakeRuntime))
            try:
                code = worker.main(argv)
            finally:
                os.umask(umask)
        if code != 0:
            raise RuntimeError("worker main failed: " + (root / "failure.rank0.log").read_text()[-2000:])
        record = json.loads((root / "runner.rank0.json").read_text())
        layout = _tree(root)  # what main wrote (before the save callback is probed below)
        if len(constructed) != 1:
            raise RuntimeError(f"worker main constructed {len(constructed)} runtimes")
        bound, (checked, binding) = seen["preflight"]
        jax_module, mesh, physical, topology, fleet_sha = seen["initialize"]
        kwargs = constructed[0]
        known = {
            "<mesh from _initialize_runtime>": mesh,
            "<physical from _initialize_runtime>": physical,
            "<topology from _initialize_runtime>": topology,
        }
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
                described[name] = (
                    "<source root>" if item.resolve() == REPO.resolve() else "<output>/" + str(item.relative_to(root))
                )
            elif item is None or isinstance(item, (bool, int, float, str)):
                described[name] = item
            else:
                described[name] = f"<{type(item).__name__}>"
        labels = {
            argv[argv.index("--source-manifest-sha256") + 1]: "<staged source manifest sha256>",
            argv[argv.index("--site-sha256") + 1]: "<staged site sha256>",
        }
        preflight_record = dict(
            bound={name: _describe_arg(item, root, labels) for name, item in sorted(vars(bound).items())},
            request_sha256_equal=checked["request_sha256"] == value["request_sha256"],
            binding={k: binding[k] for k in sorted(binding)},
            template_checks=calls["template"],
        )
        initialize_record = dict(
            distributed=[{k: _describe_arg(v, root) for k, v in sorted(call.items())} for call in calls["distributed"]],
            mesh=_mesh_record(mesh, fleet["physical"]),
            meshes_built=len(calls["meshes"]),
            returns_jax=jax_module.__name__ == "jax",
            physical_mesh_hash_bound=physical.mesh_hash == bound.mesh_sha256,
            topology_hash_bound=topology.topology_hash == bound.topology_sha256,
            fleet_sha256_bound=fleet_sha == bound.topology_fleet_sha256,
        )
        construction = {
            kind: {
                k: _describe_arg(v, root) if isinstance(v, str) and v.startswith("/") else v for k, v in changes.items()
            }
            for kind, changes in seen["construction"].items()
        }
        return dict(
            exit_code=code,
            keys=sorted(record),
            request_keys=sorted(record["request"]),
            layout=layout,
            runtime_arguments=described,
            preflight=preflight_record,
            initialize=initialize_record,
            runtime_construction=construction,
        )


REFUSALS = (
    "deployed_source",
    "existing_namespace",
    "port",
    "owner_only",
    "request_digest",
    "binding_pin",
    "site_digest",
    "site_coordinator",
)


def worker_refusals(value: dict[str, Any]) -> dict[str, Any]:
    """The real ``preflight`` on staged runs with exactly one input broken, and the real
    ``_initialize_runtime`` on a host whose name does not match its launch capture: each must
    refuse (the message is recorded; ``accepted`` means the check is gone)."""
    from glm_tpu.worker import tpu_worker as worker

    import glm_tpu.distributed.parallel_state as parallel_state

    fleet = synthetic_fleet()
    out: dict[str, Any] = {}

    attempt = _attempt

    def namespace(argv: list[str]) -> Any:
        pairs = dict(zip(argv[::2], argv[1::2], strict=True))
        return argparse.Namespace(
            **{k[2:].replace("-", "_"): v for k, v in pairs.items()}
            | dict(
                output=Path(pairs["--output"]),
                wall_seconds=int(pairs["--wall-seconds"]),
                preflight_only=False,
                keep_loaded=False,
            )
        )

    for case in REFUSALS:
        with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch:
            runs = Path(scratch) / "runs"
            runs.mkdir(mode=0o700)
            args = namespace(stage_run(runs, value, fleet, tamper=case))
            with worker_host(fleet, runs):
                out[f"preflight_{case}"] = attempt(lambda args=args: worker.preflight(args))
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch:
        runs = Path(scratch) / "runs"
        runs.mkdir(mode=0o700)
        args = namespace(stage_run(runs, value, fleet))
        with worker_host(fleet, runs):
            worker.preflight(args)
        with worker_host(fleet, runs, hostname="example-w-1"):
            args.process_id = 0  # launched as rank 0 on the host captured as rank 1
            out["initialize_host_mapping"] = attempt(lambda: parallel_state._initialize_runtime(args))
    return out


def controller_record() -> dict[str, Any]:
    from glm_tpu.engine import request
    from glm_tpu.executor import multihost_executor as launch
    from glm_tpu.utils import io_utils

    rows = [
        dict(
            rank=i,
            hostname=f"example-w-{i}",
            complete=True,
            code_hash="a" * 40,
            request_sha256="b" * 64,
            request=dict(token_sha256="c" * 64, emitted=3),
            programs=dict(decode=dict(stablehlo_sha256="d" * 64, optimized_hlo_sha256="e" * 64)),
        )
        for i in range(8)
    ]
    summary = launch.summarize(rows, "a" * 40, "b" * 64)
    resident = launch.summarize(rows, "a" * 40, "b" * 64, idle_after=False)
    out: dict[str, Any] = dict(
        summary=dict(keys=sorted(summary), bytes=_bytes_record(json.dumps(summary, sort_keys=True).encode())),
        resident_summary_keys=sorted(resident),
    )
    from .controller import launcher_record

    out["launcher_main"] = launcher_record()
    # resident_controller: collect round 0, admit inbox 0001, then stop (clock and SSH faked).
    import base64

    from glm_tpu.executor.fleet import HelperTexts

    from .site_fixture import site as synthetic_site

    first = request.from_token_ids([7], request_id="golden-r0", max_new_tokens=2)
    second = request.from_token_ids([8, 9], request_id="golden-r1", max_new_tokens=2)
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-wire-") as scratch:
        root = Path(scratch)
        fleet = [dict(row, request_sha256=first["request_sha256"]) for row in rows]
        io_utils.persist(root / "resident-ready.json", dict(sequence=0))
        stage = [0]

        def collect(commands: Any, command: str, run_root: Path, label: str) -> None:
            # What the fetch helper prints for runner.rank{rank}.json: base64 of the bytes the
            # worker's own persist wrote on that host.
            with tempfile.TemporaryDirectory(prefix="glm-equivalence-host-") as host:
                for rank, row in enumerate(fleet):
                    record = Path(host) / f"runner.rank{rank}.json"
                    io_utils.persist(record, row)
                    line = json.dumps({record.name: base64.b64encode(record.read_bytes()).decode()})
                    (run_root / f"{label}.rank{rank}.log").write_text(line + "\n")

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
                io_utils.persist(root / "resident-ready.json", dict(sequence=1))
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
            launch.resident_controller(
                [],
                processes,
                root,
                "a" * 40,
                first,
                3600,
                False,
                hosts=[row["hostname"] for row in rows],
                fleet=synthetic_site(root).fleet,
                helpers=HelperTexts.from_package(),
            )
        stdin = processes[0].stdin.getvalue()
        measurement = json.loads((root / "resident-measurement.json").read_text())
        out["resident_controller"] = dict(
            worker_stdin=_bytes_record(stdin),
            stdin_lines=len(stdin.splitlines()),
            stop_line=stdin.splitlines()[-1].decode(),
            all_workers_equal=len({p.stdin.getvalue() for p in processes}) == 1,
            command_prefix=stdin.splitlines()[0][:24].decode(),
            measurement_keys=sorted(measurement),
            layout=_tree(root),
            stdout_markers=sorted({line.split(" ", 1)[0] for line in printed.getvalue().splitlines() if line}),
        )
    return out


def runtime_record_keys() -> dict[str, Any]:
    """Keys of the ``glm_optimized_runtime_v1`` record ``OrdinaryRuntime.__init__`` builds (static:
    the constructor needs devices). From S4 this locator follows the renamed module."""
    source = (REPO / "glm_tpu" / "runner" / "tpu_runner.py").read_text()
    for node in ast.walk(ast.parse(source)):
        targets = [t for t in getattr(node, "targets", []) if isinstance(t, ast.Attribute) and t.attr == "record"]
        if (
            isinstance(node, ast.Assign)
            and targets
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "dict"
        ):
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
        return dict(
            prompt_ids=[1, 2] + [3] * len(messages),
            max_new_tokens=32000,
            request_id="ui-" + key,
            context_capacity=32768,
        )

    def prepare_api(
        self, messages: Any, key: str, *, tools: Any, effort: Any, max_new_tokens: Any, context_capacity: int
    ) -> dict[str, Any]:
        ids = [1] * (len(json.dumps(messages)) // 10 + 1)
        budget = context_capacity - len(ids) if max_new_tokens is None else max_new_tokens
        return dict(prompt_ids=ids, max_new_tokens=budget, request_id="api-" + key, context_capacity=context_capacity)

    def next_sequence(self) -> int:
        return self.sequence + 1

    def publish(self, job: Any, root: Any) -> None:
        pass

    def observe(self, job: Any) -> dict[str, Any]:
        self.sequence = job["sequence"]
        return dict(
            status="complete",
            answer=self.answer,
            thinking=self.thinking,
            output_tokens=7,
            decode_tps=13.5,
            prefill_seconds=1.0,
            stop_cause="eos",
        )


def http_record() -> dict[str, Any]:
    from glm_tpu.entrypoints.openai.serving_chat import Api
    from glm_tpu.entrypoints.serve.job_queue import Chats
    from glm_tpu.entrypoints.serve.server import ThreadingHTTPServer
    from glm_tpu.entrypoints.serve.http_handler import handler

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

        threads = [
            threading.Thread(target=server.serve_forever, daemon=True),
            threading.Thread(target=pump, daemon=True),
        ]
        for thread in threads:
            thread.start()
        host = f"127.0.0.1:{port}"
        cases: dict[str, Any] = {}

        def call(
            name: str,
            method: str,
            path: str,
            *,
            body: Any = None,
            headers: dict[str, str] | None = None,
            auth: bool = False,
            ui: bool = False,
            host_header: str | None = None,
        ) -> bytes:
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
            call(
                "post_chat_send",
                "POST",
                "/api/chat",
                ui=True,
                body=dict(action="send", chat=created["id"], id="a" * 32, text=NON_ASCII_MESSAGES[1]["content"]),
            )
            wait_for(lambda: all(j["status"] == "complete" for j in store.snapshot()["jobs"]))
            call("get_state_after_answer", "GET", "/api/state")
            call("get_models", "GET", "/v1/models", auth=True)
            call(
                "post_completion",
                "POST",
                "/v1/chat/completions",
                auth=True,
                body=dict(model="glm-5.3", messages=NON_ASCII_MESSAGES),
            )
            call(
                "post_completion_stream",
                "POST",
                "/v1/chat/completions",
                auth=True,
                body=dict(model="glm-5.3-low", messages=NON_ASCII_MESSAGES, stream=True),
            )
            backend.answer = (
                "I will look.<tool_call>read_file<arg_key>path</arg_key><arg_value>a.txt</arg_value>"
                "<arg_key>lines</arg_key><arg_value>12</arg_value></tool_call>"
            )
            tools = [
                dict(
                    type="function",
                    function=dict(
                        name="read_file",
                        description="Read a file",
                        parameters=dict(type="object", properties=dict(path=dict(type="string"))),
                    ),
                )
            ]
            call(
                "post_completion_tool_call",
                "POST",
                "/v1/chat/completions",
                auth=True,
                body=dict(model="glm-5.3", messages=[dict(role="user", content="read a.txt")], tools=tools),
            )
            call(
                "post_completion_tool_call_stream",
                "POST",
                "/v1/chat/completions",
                auth=True,
                body=dict(
                    model="glm-5.3-high", messages=[dict(role="user", content="read a.txt")], tools=tools, stream=True
                ),
            )
            backend.answer = "42"
            call("error_401_models", "GET", "/v1/models")
            call("error_401_completion", "POST", "/v1/chat/completions", body=dict(messages=[]))
            call("error_403_host", "GET", "/", host_header="evil.example:80")
            call("error_403_ui_header", "POST", "/api/chat", body=dict(action="create"))
            call("error_404_asset", "GET", "/nope")
            call("error_404_v1", "GET", "/v1/nope", auth=True)
            call("error_404_ui", "POST", "/api/nope", ui=True, body=dict(action="create"))
            call(
                "error_404_model",
                "POST",
                "/v1/chat/completions",
                auth=True,
                body=dict(model="gpt-4", messages=[dict(role="user", content="hi")]),
            )
            call("error_400_json", "POST", "/v1/chat/completions", auth=True, body=b"{not json")
            call(
                "error_400_effort",
                "POST",
                "/v1/chat/completions",
                auth=True,
                body=dict(messages=[dict(role="user", content="hi")], reasoning_effort="medium"),
            )
            call("error_400_chat", "POST", "/api/chat", ui=True, body=dict(action="send", chat="0" * 32))
        finally:
            stop.set()
            server.shutdown()
            server.server_close()
    return cases


_PROCESS: dict[str, Any] = {}
_UNSET = "<unset>"
# Environment variables that configure JAX, XLA, libtpu or the worker (third-party imports set
# unrelated ones, e.g. a torch cache directory).
COMPILE_ENVIRONMENT = re.compile(r"(XLA_|LIBTPU|TPU_|JAX_|PJRT_|GLM_)")


def snapshot_process() -> None:
    """The jax configuration (with the Pallas/Mosaic options registered) and environment before
    any production module is imported (in the G9 child: the start of ``record``): the reference
    for ``process_changes``."""
    import jax
    import jax.experimental.pallas  # registers the Pallas options
    import jax.experimental.pallas.tpu  # registers the Mosaic options

    _PROCESS.update(config=dict(jax.config.values), environ=dict(os.environ))


def _config_value(value: Any) -> Any:
    return value if value is None or isinstance(value, (bool, int, str)) else repr(value)


def process_changes() -> dict[str, dict[str, Any]]:
    """Every jax configuration option and JAX/XLA/libtpu/worker environment variable whose value
    differs from the ``snapshot_process`` reference: lowering-relevant settings such as
    ``jax_default_matmul_precision``, ``XLA_FLAGS`` or ``LIBTPU_INIT_ARGS`` made by the worker
    process or at import time by a module it loaded. Options registered after the snapshot (by a
    jax module imported later) have no reference and are left out."""
    import jax

    if not _PROCESS:
        raise RuntimeError("process_changes needs snapshot_process first")
    config, environ = _PROCESS["config"], _PROCESS["environ"]
    now = dict(jax.config.values)
    names = {k for k in set(environ) | set(os.environ) if COMPILE_ENVIRONMENT.match(k)}
    return dict(
        jax_config={k: _config_value(now.get(k, _UNSET)) for k in sorted(config) if config[k] != now.get(k, _UNSET)},
        environment={
            k: os.environ.get(k, _UNSET) for k in sorted(names) if environ.get(k, _UNSET) != os.environ.get(k, _UNSET)
        },
    )


def record() -> dict[str, Any]:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    snapshot_process()
    wire = dict(
        requests=requests_record(),
        worker=worker_record(),
        controller=controller_record(),
        runtime_record=runtime_record_keys(),
    )
    return dict(wire=wire, http=http_record(), source=source_record())


def main() -> int:
    emit(record())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
