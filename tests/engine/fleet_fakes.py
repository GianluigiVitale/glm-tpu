"""Fleet and device fakes for the worker-level vote-schedule tests (``tests/engine/test_vote_schedule.py``).

Every fleet vote of the ordinary request path is an eight-host ``process_allgather``
(``glm_tpu.distributed.parallel_state._batched_fleet_all``), so the order of the votes is fleet protocol:
a host that leaves it strands its peers in their next collective. The tests drive the worker's entry points
(``run_queued``, ``run_concurrent``, ``resident_loop`` of ``glm_tpu.worker.tpu_worker``) on ranks 0 and 1
of such a fleet, in two threads and in lock step:

* :class:`Fleet` replaces ``jax.experimental.multihost_utils.process_allgather`` by an exchange of the two
  ranks' values; hosts 2-7 contribute rank 1's value (like rank 1, they write no tokens). The runtime's
  ``vote`` is the real ``_batched_fleet_all`` over that exchange. A rank waiting in a collective that its
  peer never enters gets :class:`StrandedCollective` (where a real fleet would hang). Each rank logs every
  collective as the fleet sees it, with the phases, device calls and deliveries around it.
* :class:`Device` is one rank's synthetic device side: the programs the runtime has compiled (cache
  initializer, prefill, decode; the batch programs of a concurrent runtime), a clock that advances one tick
  per device call (so a deadline expires at a known call), and the scripted results (:class:`Script`).
* The ADAPTER (the section below it) is the only code here that knows how the worker's runtime object is
  built and what it calls its programs, phases, sessions and poison flag. A work unit that restructures the
  runtime (WU-E1 ``TPUModelRunner`` + ``LLMEngine``, WU-E2 one ``RequestSession``) changes only the adapter;
  ``test_vote_schedule.py`` and its literal expectations stay byte-identical.

The runtime is built by its real constructor with the worker's keywords; only the checkpoint load and
compilation (``_load``), the HLO dump-space check and the device memory statistics are synthetic, so the real
memory admission (``admit`` -> ``project_memory`` -> its phase vote) runs. Nothing here opens a device:
NumPy values stand for device arrays and ``put`` is the identity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import functools
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from glm_tpu.models.glm_moe_dsa.model import BatchedDecodeResult, PackedDecodeResult
from glm_tpu.models.glm_moe_dsa.state import BatchedPrefillResult, BatchedPrefillState, DecodeStepResult, DecoderState

REPO = Path(__file__).resolve().parents[2]
RANKS = (0, 1)
HOSTS = 8
TIMEOUT = 60.0  # seconds a rank waits for its peer in one collective (a backstop: a departed peer is seen at once)


class StrandedCollective(RuntimeError):
    """A rank waits in a collective that its peer never enters."""


@dataclass
class Script:
    """The scripted device results of one test (both ranks unless ``rank_tokens`` says otherwise).

    ``tokens[prompt length]``: the token the prefill samples, then one per decode step (the last repeats);
    a request is identified by its prompt length. ``bad_metadata = (prompt length, step)``: that decode step
    reports a wrong position (sequential) or an unhealthy lane (concurrent). ``fail_delivery = (request id,
    token index)``: rank 0's token-file write of that event raises ``OSError``.
    """

    tokens: dict[int, list[int]]
    prefill_healthy: bool = True
    prefill_finished: bool = True
    bad_metadata: tuple[int, int] | None = None
    fail_delivery: tuple[str, int] | None = None
    rank_tokens: dict[tuple[int, int, int], int] = field(default_factory=dict)  # (rank, prompt length, step)


@dataclass
class Outcome:
    value: Any = None
    error: BaseException | None = None


def decoder_state(position: int, healthy: bool = True) -> DecoderState:
    """A one-row decoder state whose next position is ``position``."""
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


class Fleet:
    """Ranks 0 and 1 of an eight-host fleet; ``start`` builds their runtimes, ``run`` runs an entry point."""

    def __init__(self, monkeypatch: Any, root: Path) -> None:
        from jax.experimental import multihost_utils

        self.root = root
        root.mkdir()
        self.events: dict[int, list[tuple]] = {rank: [] for rank in RANKS}
        self.sessions: dict[int, list[Any]] = {rank: [] for rank in RANKS}
        self.nodes: dict[int, SimpleNamespace] = {}
        self._cond = threading.Condition()
        self._values: dict[tuple[int, int], np.ndarray] = {}
        self._entered = dict.fromkeys(RANKS, 0)
        self._departed: set[int] = set()
        self._local = threading.local()
        monkeypatch.setattr(multihost_utils, "process_allgather", self.allgather)
        install(monkeypatch, self)

    @property
    def rank(self) -> int:
        return self._local.rank

    def start(self, script: Script, *, capacity: int = 8192, concurrent_size: int = 0) -> None:
        for rank in RANKS:
            root = self.root / f"rank{rank}"
            root.mkdir()
            device = Device(self, rank, script)
            runtime = build_runtime(self, device, root, capacity=capacity, concurrent_size=concurrent_size)
            self.nodes[rank] = SimpleNamespace(rank=rank, root=root, device=device, runtime=runtime)

    def run(self, body: Callable[[SimpleNamespace], Any]) -> dict[int, Outcome]:
        """``body(node)`` on both ranks at once; the outcome (value or exception) per rank."""
        outcomes: dict[int, Outcome] = {}
        self._departed = set()

        def main(rank: int) -> None:
            self._local.rank = rank
            try:
                outcomes[rank] = Outcome(value=body(self.nodes[rank]))
            except BaseException as exc:  # the outcome is the result
                outcomes[rank] = Outcome(error=exc)
            finally:
                with self._cond:
                    self._departed.add(rank)
                    self._cond.notify_all()

        threads = [threading.Thread(target=main, args=(rank,), name=f"rank{rank}") for rank in RANKS]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(2 * TIMEOUT)
        assert not any(thread.is_alive() for thread in threads), "a rank did not finish"
        return outcomes

    def allgather(self, value: Any) -> np.ndarray:
        rank = self.rank
        peer = 1 - rank
        local = np.asarray(value)
        with self._cond:
            index = self._entered[rank]
            self._entered[rank] += 1
            self._values[index, rank] = local
            self._cond.notify_all()
            self._cond.wait_for(lambda: (index, peer) in self._values or peer in self._departed, timeout=TIMEOUT)
            if (index, peer) not in self._values:
                raise StrandedCollective(f"rank {rank} waits in collective {index}; rank {peer} never enters it")
            first, second = self._values[index, 0], self._values[index, 1]
        if first.shape != second.shape or first.dtype != second.dtype:
            raise StrandedCollective(
                f"collective {index}: rank 0 sends {first.dtype}{first.shape}, rank 1 {second.dtype}{second.shape}"
            )
        gathered = np.stack([first] + [second] * (HOSTS - 1))
        if first.shape == ():
            self.events[rank].append(("vote", int(first), int(second), bool(gathered.all())))
        else:
            self.events[rank].append(("digest", bool((gathered == gathered[0]).all())))
        return gathered


class Device:
    """One rank's synthetic device programs and clock."""

    def __init__(self, fleet: Fleet, rank: int, script: Script) -> None:
        self.fleet, self.rank, self.script = fleet, rank, script
        self.tick = 0.0
        self.prompt = self.done = self.step = 0

    def clock(self) -> float:
        return self.tick

    def put(self, value: Any) -> Any:
        return value

    def _call(self, *event: Any) -> None:
        self.fleet.events[self.rank].append(("device", *event))
        self.tick += 1.0

    def token(self, length: int, step: int) -> int:
        override = self.script.rank_tokens.get((self.rank, length, step))
        tokens = self.script.tokens[length]
        return tokens[min(step, len(tokens) - 1)] if override is None else override

    def initialize(self, count: Any) -> Any:
        self.prompt, self.done, self.step = int(np.asarray(count)), 0, 0
        self._call("initialize", self.prompt)
        return SimpleNamespace(decoder=decoder_state(0))

    def prefill(self, block: Any, count: Any, fresh: Any, *bound: Any) -> BatchedPrefillResult:
        rows, count = int(np.asarray(block).size), int(np.asarray(count))
        self._call("prefill", rows, count)
        self.done += count
        final = self.done == self.prompt
        state = BatchedPrefillState(
            decoder_state(self.done, self.script.prefill_healthy),
            np.zeros((1,)),
            np.array(self.prompt, np.int32),
            np.array(final and self.script.prefill_finished),
        )
        return BatchedPrefillResult(state, np.array([self.token(self.prompt, 0) if final else -1], np.int32))

    def decode(self, token: Any, state: Any, *bound: Any) -> PackedDecodeResult:
        self._call("decode")
        self.step += 1
        position = self.prompt + self.step
        value = self.token(self.prompt, self.step)
        reported = position + 1 if self.script.bad_metadata == (self.prompt, self.step) else position
        result = DecodeStepResult(decoder_state(position), np.array([value], np.int32), np.zeros((1, 1)))
        return PackedDecodeResult(result, np.array([value, 1, reported, reported + 1], np.int32))

    def initialize_batch(self, lengths: Any) -> dict[str, Any]:
        lengths = tuple(int(x) for x in np.asarray(lengths))
        self._call("initialize_batch", *lengths)
        return dict(lengths=lengths, round=0)

    def insert(self, bank: Any, one: Any, lane: Any) -> Any:
        self._call("insert", int(np.asarray(lane)))
        return bank

    def decode_batch(self, tokens: Any, bank: Any, weights: Any, rope: Any, active: Any) -> BatchedDecodeResult:
        self._call("decode_batch", *(int(x) for x in np.asarray(active)))
        bank = dict(bank, round=bank["round"] + 1)
        step, rows = bank["round"], []
        for length in bank["lengths"]:
            health = 0 if self.script.bad_metadata == (length, step) else 1
            rows.append([self.token(length, step), health, length + step, length + step + 1])
        metadata = np.asarray(rows, np.int32)
        return BatchedDecodeResult(bank, metadata[:, :1], metadata)


class DeliveryStream:
    """Rank 0's token file (the value of the ``open_tokens`` phase): every written event is logged as a
    delivery; ``Script.fail_delivery`` makes that write raise (rank 0 only: other ranks deliver nothing)."""

    def __init__(self, stream: Any, fleet: Fleet, device: Device) -> None:
        self.stream, self.fleet, self.device = stream, fleet, device

    def write(self, text: str) -> int:
        event = json.loads(text)
        key = (event["request_id"], event["index"])
        failed = self.device.script.fail_delivery == key
        self.fleet.events[self.device.rank].append(
            ("deliver", *key, event["token_id"], event["finish_reason"], event.get("batch_round"), failed)
        )
        if failed:
            raise OSError("injected delivery failure")
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()

    def __enter__(self) -> DeliveryStream:
        self.stream.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self.stream.__exit__(*exc)


# ============================================================================== ADAPTER
# The runtime's construction and names. A work unit that restructures the runtime changes this section only.
SEQUENTIAL_PROGRAMS = ("cache_init", "prefill_128", "prefill_114", "decode")
BATCH_PROGRAMS = ("batch_cache_init", "cache_init", "prefill_128", "prefill_114", "batch_insert", "batch_decode")
COMPILED_MEMORY = dict(
    output_size_in_bytes=1 << 20, temp_size_in_bytes=1 << 20, generated_code_size_in_bytes=0, alias_size_in_bytes=0
)
CHIPS = tuple(
    dict(device_id=i, bytes_in_use=1 << 30, bytes_limit=32 << 30, peak_bytes_in_use=1 << 30) for i in range(4)
)


def install(monkeypatch: Any, fleet: Fleet) -> None:
    """Patches for every runtime of one test: the loader, the HLO dump-space check, recorded sessions."""
    import shutil

    from glm_tpu.engine import llm_engine
    from glm_tpu.runner import tpu_runner

    monkeypatch.setattr(tpu_runner.TPUModelRunner, "_load", _load)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: SimpleNamespace(total=1 << 40, used=0, free=1 << 40))
    monkeypatch.setattr(llm_engine, "RequestSession", _recorded(llm_engine.RequestSession, fleet))
    monkeypatch.setattr(llm_engine, "BatchedSession", _recorded(llm_engine.BatchedSession, fleet))


def _recorded(cls: type, fleet: Fleet) -> type:
    class Recorded(cls):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            fleet.sessions[fleet.rank].append(self)

    Recorded.__name__ = Recorded.__qualname__ = cls.__name__
    return Recorded


def _load(runtime: Any, repo: Any, physical: Any) -> None:
    """``TPUModelRunner._load`` of a synthetic device: the compiled programs and their memory rows."""
    device = runtime.args.device
    runtime.stats = lambda: [dict(chip) for chip in CHIPS]
    runtime.put = device.put
    runtime.weights = runtime.wk = runtime.rope = None
    for name in BATCH_PROGRAMS if runtime.concurrent_size else SEQUENTIAL_PROGRAMS:
        runtime.record["programs"][name] = dict(compiled_memory=dict(COMPILED_MEMORY))
    runtime.initialize = device.initialize
    runtime.prefill = {114: device.prefill, 128: device.prefill}
    if runtime.concurrent_size:
        runtime.initialize_batch, runtime.insert_batch = device.initialize_batch, device.insert
        runtime.decode_batch = device.decode_batch
    else:
        runtime.decode = device.decode


def build_runtime(fleet: Fleet, device: Device, root: Path, *, capacity: int, concurrent_size: int) -> Any:
    """The object the worker's entry points receive, built as ``tpu_worker.main`` builds it: the engine around
    the runner (``TPUModelRunner``, then ``LLMEngine``)."""
    from glm_tpu.distributed import parallel_state
    from glm_tpu.engine.llm_engine import LLMEngine
    from glm_tpu.runner.tpu_runner import TPUModelRunner

    native = root / f"native.rank{device.rank}"
    native.mkdir()
    runner = TPUModelRunner(
        args=SimpleNamespace(hlo_dump_root=str(root / "hlo"), device=device),
        repo=REPO,
        root=native,
        mesh=None,
        physical=SimpleNamespace(mesh_hash="m" * 64, flattened_device_ids=tuple(range(32))),
        topology=SimpleNamespace(topology_hash="t" * 64),
        fleet_sha="f" * 64,
        vote=parallel_state._batched_fleet_all,
        save=lambda record: None,
        context_capacity=capacity,
        **(dict(concurrent_size=concurrent_size) if concurrent_size else {}),
    )
    phase = type(runner).phase

    def logged_phase(name: str, action: Callable[[], Any]) -> Any:
        fleet.events[device.rank].append(("phase", name))
        value = phase(runner, name, action)
        return DeliveryStream(value, fleet, device) if name == "open_tokens" and value is not None else value

    runner.phase = logged_phase
    engine = LLMEngine(runner)
    engine.generate = functools.partial(type(engine).generate, engine, clock=device.clock)
    engine.generate_concurrent = functools.partial(type(engine).generate_concurrent, engine, clock=device.clock)
    return engine


def poisoned(node: SimpleNamespace) -> bool:
    """The runtime's poison flag (the engine's): an active or failed request."""
    return node.runtime.active


def session_states(fleet: Fleet, rank: int) -> list[tuple[bool, list[tuple[str, int, int, str | None]]]]:
    """``(failed, committed events)`` of every request session the rank created, in order; a batched
    session's events lane by lane."""
    out = []
    for session in fleet.sessions[rank]:
        lanes = session.events if isinstance(session.events, list) else [session.events]
        events = [(e.request_id, e.index, e.token_id, e.finish_reason) for lane in lanes for e in lane]
        out.append((bool(session.failed), events))
    return out
