"""The fleet-vote schedule and the failure paths of the ordinary request path, through the worker's entry points.

Every vote is an eight-host collective, so its order is fleet protocol. Each test runs one worker entry point
(``glm_tpu.worker.tpu_worker.run_queued``, ``run_concurrent`` or ``resident_loop``) on ranks 0 and 1 of a fake
fleet in lock step (``tests/engine/fleet_fakes.py``) and asserts:

* rank 0's log, rendered, equal to a literal: the phases (``phase <name>: <inner events>, vote ...``), the
  device calls, every other collective (``vote <rank 0> <rank 1> -> <agreed>``: the deadline and prefill
  votes, the request session's prefill-health, validity and delivery votes, the batched session's round
  votes; ``digest same|differs``: the output consensus) and rank 0's deliveries;
* rank 1's log equal to rank 0's without the deliveries (rank 1 writes no tokens): the same collectives in the
  same order on every rank, including every failure path;
* each rank's exception (type, message, cause chain), the runtime's poison flag, and each request session's
  ``failed`` flag and committed events (committed before delivery: a failed delivery's event is committed).

The expected logs are the behaviour at the S5 A2 commit, reviewed line by line against the code. A work unit
that restructures the runtime or the sessions (WU-E1, WU-E2) changes only the adapter section of
``fleet_fakes.py``; this module stays byte-identical.
"""

from __future__ import annotations

import io
import json
import re
from textwrap import dedent
from types import SimpleNamespace
from typing import Any

import pytest

from glm_tpu.engine import request
from glm_tpu.worker.tpu_worker import resident_loop, run_concurrent, run_queued
from tests.engine.fleet_fakes import RANKS, Fleet, Script, poisoned, session_states
from tests.fixtures.site import example_site, installed_site

EOS = request.EOS[0]
DEADLINE = 10_000.0  # device ticks: never reached unless a test says so
FLEET_REFUSED = "request fleet rejected state/delivery"
BATCH_REFUSED = "concurrent batch rejected state, delivery or deadline"


@pytest.fixture
def fleet(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(decode=lambda tokens, **kw: " ".join(map(str, tokens))),
    )
    with installed_site(example_site(tmp_path / "site")):  # the answer writer's tokenizer location
        yield Fleet(monkeypatch, tmp_path / "fleet")


# ------------------------------------------------------------------------------ rendering and checks
def describe(event: tuple) -> str:
    kind, *rest = event
    if kind == "vote":
        first, second, agreed = rest
        return f"vote {first} {second} -> {int(agreed)}"
    if kind == "digest":
        return "digest " + ("same" if rest[0] else "differs")
    if kind == "device":
        return " ".join(str(part) for part in rest)
    request_id, index, token, reason, round_index, failed = rest
    text = f"deliver {request_id}[{index}] {token} {reason or '-'}"
    return text + ("" if round_index is None else f" round {round_index}") + (" (write fails)" if failed else "")


def render(events: list[tuple]) -> str:
    """One line per event; a phase and the events up to its vote on one line."""
    lines: list[str] = []
    phase: list[str] | None = None
    for event in events:
        if event[0] == "phase":
            if phase is not None:
                lines.append(" ".join(phase) + " (no vote)")
            phase = [f"phase {event[1]}:"]
            continue
        if phase is None:
            lines.append(describe(event))
            continue
        phase.append(describe(event) if event[0] == "vote" else describe(event) + ",")
        if event[0] == "vote":
            lines.append(" ".join(phase))
            phase = None
    if phase is not None:
        lines.append(" ".join(phase) + " (no vote)")
    return "\n".join(lines) + "\n"


def chain(error: BaseException | None) -> list[str]:
    """``Type: message`` of the exception and of each ``__cause__``."""
    out = []
    while error is not None:
        out.append(f"{type(error).__name__}: {error}")
        error = error.__cause__
    return out


def expected_log(text: str) -> str:
    """A literal log without its review annotations (``  # ...``: the call site of each vote)."""
    return "".join(re.sub(r"  # .*$", "", line) + "\n" for line in dedent(text).strip("\n").splitlines())


def check_log(fleet: Fleet, expected: str, since: dict[int, int] | None = None) -> None:
    """Rank 0's log (from ``since``) is ``expected``; rank 1's is rank 0's without the deliveries."""
    since = since or dict.fromkeys(RANKS, 0)
    first, second = fleet.events[0][since[0] :], fleet.events[1][since[1] :]
    assert render(first) == expected_log(expected)
    assert second == [event for event in first if event[0] != "deliver"]


def marks(fleet: Fleet) -> dict[int, int]:
    return {rank: len(fleet.events[rank]) for rank in RANKS}


def check_errors(outcomes: dict, *expected: list[str]) -> None:
    """Every rank raised: rank 0 ``expected[0]``, rank 1 ``expected[-1]`` (one list: the same on both)."""
    for rank in RANKS:
        assert chain(outcomes[rank].error) == expected[min(rank, len(expected) - 1)], rank


def delivered(node: SimpleNamespace, path: str = "tokens.jsonl") -> list[tuple]:
    """A token file as ``(request id, index, token, finish reason)`` rows (plus the batch round, if any)."""
    rows = []
    for line in (node.root / path).read_text().splitlines():
        row = json.loads(line)
        rows.append((row["request_id"], row["index"], row["token_id"], row["finish_reason"]))
        if "batch_round" in row:
            rows[-1] += (row["batch_round"],)
    return rows


def check_success(fleet: Fleet, outcomes: dict, stops: list[str], files: dict[str, list[tuple]]) -> None:
    """Every rank returned equal reports and stays usable; only rank 0 wrote the token files."""
    reports = {}
    for rank in RANKS:
        assert outcomes[rank].error is None, chain(outcomes[rank].error)
        assert not poisoned(fleet.nodes[rank])
        assert not any(failed for failed, _ in session_states(fleet, rank))
        value = outcomes[rank].value
        reports[rank] = value if isinstance(value, list) else value[0]
        assert [report["stop_cause"] for report in reports[rank]] == stops
    assert [r["token_sha256"] for r in reports[0]] == [r["token_sha256"] for r in reports[1]]
    for path, rows in files.items():
        assert delivered(fleet.nodes[0], path) == rows
        assert not (fleet.nodes[1].root / path).exists()


# ------------------------------------------------------------------------------ entry points
def queued(fleet: Fleet, value: dict, *, warmup: bool = True, deadlines: tuple[float, float] = (DEADLINE, DEADLINE)):
    """``run_queued`` as the worker's ``main`` (warmup) or ``resident_loop`` (no warmup) calls it."""
    return fleet.run(
        lambda node: run_queued(
            node.runtime,
            request.requests(value),
            value,
            node.root,
            node.rank,
            deadlines[node.rank],
            save=lambda reports: None,
            warmup=warmup,
        )
    )


def concurrent(fleet: Fleet, values: list[dict], *, deadlines: tuple[float, float] = (DEADLINE, DEADLINE)):
    """``run_concurrent`` as the worker's ``main`` calls it (it always warms up first)."""
    return fleet.run(lambda node: run_concurrent(node.runtime, values, node.root, node.rank, deadlines[node.rank]))


def resident(fleet: Fleet, *commands: dict):
    """``resident_loop`` reading ``commands`` from the controller's stdin."""
    text = "".join(json.dumps(command) + "\n" for command in commands)
    record = dict(schema="glm_optimized_worker_v1", complete=True)
    return fleet.run(
        lambda node: resident_loop(
            node.runtime, dict(record, rank=node.rank), node.root, node.rank, 3600, stream=io.StringIO(text)
        )
    )


def one(prompt: int, *, request_id: str = "q0", max_new_tokens: int = 4, capacity: int = request.CAPACITY) -> dict:
    return request.from_token_ids(
        [5] * prompt, request_id=request_id, max_new_tokens=max_new_tokens, context_capacity=capacity
    )


def lanes(*shapes: tuple[int, int]) -> list[dict]:
    """Concurrent requests ``c0, c1, ...`` of (prompt length, max new tokens)."""
    return [
        one(prompt, request_id=f"c{lane}", max_new_tokens=cap, capacity=request.CONCURRENT_CAPACITY)
        for lane, (prompt, cap) in enumerate(shapes)
    ]


# ------------------------------------------------------------------------------ sequential requests
def test_sequential_request_stops_at_eos(fleet):
    """The worker's cold path (``main``): a disposable warmup request (two tokens, delivered nowhere), then the
    request. Per request: four memory-admission phases, two deadline votes, the prefill block's health vote,
    the first token's prefill-finished, validity and delivery votes (the session folds the deadline into each
    of its votes), two votes per decoded token, and the output-consensus phase (the digest, then its vote);
    around the request, the token-file and answer phases."""
    fleet.start(Script(tokens={3: [7, 9, EOS]}))
    outcomes = queued(fleet, one(3))
    check_log(
        fleet,
        """
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        vote 1 1 -> 1  # token delivered + deadline
        phase output_consensus: digest same, vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        deliver q0[0] 7 -
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[1] 9 -
        vote 1 1 -> 1  # token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[2] 154820 eos
        vote 1 1 -> 1  # token delivered + deadline
        phase output_consensus: digest same, vote 1 1 -> 1
        phase write_answer: vote 1 1 -> 1
        """,
    )
    check_success(
        fleet, outcomes, ["eos"], {"tokens.jsonl": [("q0", 0, 7, None), ("q0", 1, 9, None), ("q0", 2, EOS, "eos")]}
    )
    assert (fleet.nodes[0].root / "answer.txt").read_text() == f"7 9 {EOS}"
    assert not (fleet.nodes[1].root / "answer.txt").exists()


def test_sequential_request_stops_at_length_after_two_prefill_blocks(fleet):
    """A 130-token prompt: two prefill blocks (128 rows, then 2 tokens in the 114-row program), each after a
    deadline vote and followed by its health vote."""
    fleet.start(Script(tokens={130: [7, 9, 10]}))
    outcomes = queued(fleet, one(130, max_new_tokens=3))
    check_log(
        fleet,
        """
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 130
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 128 128
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # deadline
        prefill 114 2
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        vote 1 1 -> 1  # token delivered + deadline
        phase output_consensus: digest same, vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 130
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 128 128
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # deadline
        prefill 114 2
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        deliver q0[0] 7 -
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[1] 9 -
        vote 1 1 -> 1  # token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[2] 10 length
        vote 1 1 -> 1  # token delivered + deadline
        phase output_consensus: digest same, vote 1 1 -> 1
        phase write_answer: vote 1 1 -> 1
        """,
    )
    check_success(
        fleet,
        outcomes,
        ["output_cap"],
        {"tokens.jsonl": [("q0", 0, 7, None), ("q0", 1, 9, None), ("q0", 2, 10, "length")]},
    )


@pytest.mark.parametrize(
    ("tokens", "max_new_tokens", "stop"), [([EOS], 4, "eos"), ([7], 1, "output_cap")], ids=["eos", "length"]
)
def test_the_first_token_can_finish_the_request_without_decode(fleet, tokens, max_new_tokens, stop):
    """The prefill's token ends the request (EOS, or a one-token cap): its three votes, no decode call."""
    fleet.start(Script(tokens={3: tokens}))
    outcomes = queued(fleet, one(3, max_new_tokens=max_new_tokens), warmup=False)
    reason = "eos" if stop == "eos" else "length"
    check_log(fleet, FIRST_TOKEN_LOG.replace("<token>", str(tokens[0])).replace("<reason>", reason))
    check_success(fleet, outcomes, [stop], {"tokens.jsonl": [("q0", 0, tokens[0], reason)]})


FIRST_TOKEN_LOG = """
phase open_tokens: vote 1 1 -> 1
phase memory_cache_init: vote 1 1 -> 1
phase memory_prefill_128: vote 1 1 -> 1
phase memory_prefill_114: vote 1 1 -> 1
phase memory_decode: vote 1 1 -> 1
initialize 3
vote 1 1 -> 1  # deadline
vote 1 1 -> 1  # deadline
prefill 114 3
vote 1 1 -> 1  # prefill block healthy
vote 1 1 -> 1  # prefill finished + deadline
vote 1 1 -> 1  # first token valid + deadline
deliver q0[0] <token> <reason>
vote 1 1 -> 1  # first token delivered + deadline
phase output_consensus: digest same, vote 1 1 -> 1
phase write_answer: vote 1 1 -> 1
"""


def test_queued_requests_run_one_at_a_time_in_their_own_directories(fleet):
    """A sequential batch: the warmup uses the first request; then, per request, its directory, token-file,
    generation and answer phases, one request after the other."""
    fleet.start(Script(tokens={3: [7, EOS], 4: [8, 9]}))
    outcomes = queued(fleet, request.batch([one(3), one(4, request_id="q1", max_new_tokens=2)]))
    check_log(
        fleet,
        """
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        vote 1 1 -> 1  # token delivered + deadline
        phase output_consensus: digest same, vote 1 1 -> 1
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        deliver q0[0] 7 -
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[1] 154820 eos
        vote 1 1 -> 1  # token delivered + deadline
        phase output_consensus: digest same, vote 1 1 -> 1
        phase write_answer: vote 1 1 -> 1
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 4
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        deliver q1[0] 8 -
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q1[1] 9 length
        vote 1 1 -> 1  # token delivered + deadline
        phase output_consensus: digest same, vote 1 1 -> 1
        phase write_answer: vote 1 1 -> 1
        """,
    )
    check_success(
        fleet,
        outcomes,
        ["eos", "output_cap"],
        {
            "item000/tokens.jsonl": [("q0", 0, 7, None), ("q0", 1, EOS, "eos")],
            "item001/tokens.jsonl": [("q1", 0, 8, None), ("q1", 1, 9, "length")],
        },
    )


# ------------------------------------------------------------------------------ concurrent batches
def test_two_concurrent_conversations_share_each_decode_round(fleet):
    """``run_concurrent``: a disposable warmup batch, then the directories and token files, then the batch.
    Per batch: a deadline vote and the bank's admission; per lane, a deadline vote, the lane's admissions, its
    prefill blocks (deadline and health votes each) and the prefill-finish phase; the decode admission and a
    deadline vote; the first round's validity and delivery votes; per decode round, a start vote, one decode
    call for all live lanes, validity and delivery votes (the batched session folds the deadline into each);
    then one output digest per lane (no phase, no vote) and the answer phases."""
    fleet.start(Script(tokens={3: [7, EOS], 5: [8, 9, 10]}), capacity=request.CONCURRENT_CAPACITY, concurrent_size=2)
    outcomes = concurrent(fleet, lanes((3, 4), (5, 3)))
    check_log(
        fleet,
        """
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 5
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 5
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 5
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        digest same
        digest same
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 5
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 5
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 5
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[0] 7 - round 0
        deliver c1[0] 8 - round 0
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[1] 154820 eos round 1
        deliver c1[1] 9 - round 1
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 0 1
        vote 1 1 -> 1  # round valid + deadline
        deliver c1[2] 10 length round 2
        vote 1 1 -> 1  # round delivered + deadline
        digest same
        digest same
        phase write_answer: vote 1 1 -> 1
        phase write_answer: vote 1 1 -> 1
        """,
    )
    check_success(
        fleet,
        outcomes,
        ["eos", "output_cap"],
        {
            "item000/tokens.jsonl": [("c0", 0, 7, None, 0), ("c0", 1, EOS, "eos", 1)],
            "item001/tokens.jsonl": [("c1", 0, 8, None, 0), ("c1", 1, 9, None, 1), ("c1", 2, 10, "length", 2)],
        },
    )


def test_four_concurrent_conversations_stop_independently(fleet):
    """Four lanes: a lane that stops leaves the active mask of the next decode round (``decode_batch``'s mask);
    the last round serves one lane."""
    script = Script(tokens={2: [EOS], 3: [7, EOS], 4: [8, 9, 10], 5: [6, 6, 6, EOS]})
    fleet.start(script, capacity=request.CONCURRENT_CAPACITY, concurrent_size=4)
    outcomes = concurrent(fleet, lanes((2, 3), (3, 3), (4, 3), (5, 4)))
    check_log(
        fleet,
        """
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 2 3 4 5
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 2
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 2
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 4
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 2
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 5
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 5
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 3
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 0 1 1 1
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        digest same
        digest same
        digest same
        digest same
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 2 3 4 5
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 2
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 2
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 4
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 2
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 5
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 5
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 3
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[0] 154820 eos round 0
        deliver c1[0] 7 - round 0
        deliver c2[0] 8 - round 0
        deliver c3[0] 6 - round 0
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 0 1 1 1
        vote 1 1 -> 1  # round valid + deadline
        deliver c1[1] 154820 eos round 1
        deliver c2[1] 9 - round 1
        deliver c3[1] 6 - round 1
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 0 0 1 1
        vote 1 1 -> 1  # round valid + deadline
        deliver c2[2] 10 length round 2
        deliver c3[2] 6 - round 2
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 0 0 0 1
        vote 1 1 -> 1  # round valid + deadline
        deliver c3[3] 154820 eos round 3
        vote 1 1 -> 1  # round delivered + deadline
        digest same
        digest same
        digest same
        digest same
        phase write_answer: vote 1 1 -> 1
        phase write_answer: vote 1 1 -> 1
        phase write_answer: vote 1 1 -> 1
        phase write_answer: vote 1 1 -> 1
        """,
    )
    check_success(
        fleet,
        outcomes,
        ["eos", "eos", "output_cap", "eos"],
        {
            "item000/tokens.jsonl": [("c0", 0, EOS, "eos", 0)],
            "item001/tokens.jsonl": [("c1", 0, 7, None, 0), ("c1", 1, EOS, "eos", 1)],
            "item002/tokens.jsonl": [("c2", 0, 8, None, 0), ("c2", 1, 9, None, 1), ("c2", 2, 10, "length", 2)],
            "item003/tokens.jsonl": [
                ("c3", 0, 6, None, 0),
                ("c3", 1, 6, None, 1),
                ("c3", 2, 6, None, 2),
                ("c3", 3, EOS, "eos", 3),
            ],
        },
    )


# ------------------------------------------------------------------------------ resident loop
def test_resident_loop_votes_each_round_before_and_after_the_request(fleet):
    """``resident_loop``: per round the ready, command and directory phases, then the request without warmup;
    the stop command ends the loop after its own ready and command phases."""
    fleet.start(Script(tokens={3: [7, 9, EOS]}))
    outcomes = resident(fleet, dict(sequence=1, request=one(3)), dict(stop=True))
    check_log(
        fleet,
        """
        phase resident_ready: vote 1 1 -> 1
        phase resident_command: vote 1 1 -> 1
        phase resident_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        deliver q0[0] 7 -
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[1] 9 -
        vote 1 1 -> 1  # token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[2] 154820 eos
        vote 1 1 -> 1  # token delivered + deadline
        phase output_consensus: digest same, vote 1 1 -> 1
        phase write_answer: vote 1 1 -> 1
        phase resident_ready: vote 1 1 -> 1
        phase resident_command: vote 1 1 -> 1
        """,
    )
    for rank in RANKS:
        assert outcomes[rank].error is None and outcomes[rank].value is None
        result = json.loads((fleet.nodes[rank].root / f"resident-0001/runner.rank{rank}.json").read_text())
        assert result["resident_sequence"] == 1 and result["request"]["stop_cause"] == "eos"
    tokens = [("q0", 0, 7, None), ("q0", 1, 9, None), ("q0", 2, EOS, "eos")]
    assert delivered(fleet.nodes[0], "resident-0001/tokens.jsonl") == tokens
    assert not (fleet.nodes[1].root / "resident-0001/tokens.jsonl").exists()
    assert json.loads((fleet.nodes[0].root / "resident-ready.json").read_text()) == {"sequence": 1}
    assert not (fleet.nodes[1].root / "resident-ready.json").exists()


def test_resident_command_out_of_sequence_fails_its_phase_on_every_rank(fleet):
    """Every rank reads the same command; the command phase votes its refusal and re-raises it."""
    fleet.start(Script(tokens={3: [7, 9, EOS]}))
    outcomes = resident(fleet, dict(sequence=2, request=one(3)))
    check_log(
        fleet,
        """
        phase resident_ready: vote 1 1 -> 1
        phase resident_command: vote 0 0 -> 0
        """,
    )
    check_errors(outcomes, ["ValueError: resident sequence differs"])
    assert not any((node.root / "resident-0001").exists() for node in fleet.nodes.values())


# ------------------------------------------------------------------------------ sequential failure paths
def test_a_delivery_failure_on_rank_0_stops_every_rank_at_its_delivery_vote(fleet):
    """Rank 0's token write fails: rank 0 votes no, the fleet refuses, every rank raises at the same vote
    (only rank 0 keeps the cause). The token was committed before its delivery on every rank; the runtime is
    poisoned and refuses the next request after that request's token-file phase, before any device call."""
    fleet.start(Script(tokens={3: [7, 9, 10, 11]}, fail_delivery=("q0", 1)))
    outcomes = queued(fleet, one(3), warmup=False)
    check_log(
        fleet,
        """
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        deliver q0[0] 7 -
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[1] 9 - (write fails)
        vote 0 1 -> 0  # token delivered + deadline
        """,
    )
    check_errors(
        outcomes,
        [f"RuntimeError: {FLEET_REFUSED}", "OSError: injected delivery failure"],
        [f"RuntimeError: {FLEET_REFUSED}"],
    )
    for rank in RANKS:
        assert poisoned(fleet.nodes[rank])
        assert session_states(fleet, rank) == [(True, [("q0", 0, 7, None), ("q0", 1, 9, None)])]
    assert delivered(fleet.nodes[0]) == [("q0", 0, 7, None)]
    since = marks(fleet)
    for node in fleet.nodes.values():
        (node.root / "next").mkdir()
    value = one(4, request_id="q1")
    outcomes = fleet.run(
        lambda node: run_queued(
            node.runtime, [value], value, node.root / "next", node.rank, DEADLINE, save=lambda r: None, warmup=False
        )
    )
    check_log(fleet, "phase open_tokens: vote 1 1 -> 1", since)
    check_errors(outcomes, ["RuntimeError: optimized runtime has an active or failed request"])


def test_a_first_token_delivery_failure_on_rank_0_stops_every_rank(fleet):
    """The same for the prefill's token: committed before its delivery on every rank, then refused by the
    first token's delivery vote."""
    fleet.start(Script(tokens={3: [7, 9, 10, 11]}, fail_delivery=("q0", 0)))
    outcomes = queued(fleet, one(3), warmup=False)
    check_log(
        fleet,
        """
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        deliver q0[0] 7 - (write fails)
        vote 0 1 -> 0  # first token delivered + deadline
        """,
    )
    check_errors(
        outcomes,
        [f"RuntimeError: {FLEET_REFUSED}", "OSError: injected delivery failure"],
        [f"RuntimeError: {FLEET_REFUSED}"],
    )
    for rank in RANKS:
        assert poisoned(fleet.nodes[rank])
        assert session_states(fleet, rank) == [(True, [("q0", 0, 7, None)])]
    assert delivered(fleet.nodes[0]) == []


def test_invalid_decode_metadata_is_refused_before_delivery(fleet):
    """The first decode step reports a wrong position: its validity vote fails on every rank; the token is
    neither committed nor delivered."""
    fleet.start(Script(tokens={3: [7, 9, 10, 11]}, bad_metadata=(3, 1)))
    outcomes = queued(fleet, one(3), warmup=False)
    check_log(
        fleet,
        """
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        deliver q0[0] 7 -
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 0 0 -> 0  # token valid + deadline
        """,
    )
    check_errors(outcomes, [f"RuntimeError: {FLEET_REFUSED}"])
    for rank in RANKS:
        assert poisoned(fleet.nodes[rank])
        assert session_states(fleet, rank) == [(True, [("q0", 0, 7, None)])]
    assert delivered(fleet.nodes[0]) == [("q0", 0, 7, None)]


def test_a_deadline_expiring_on_rank_1_mid_decode_stops_every_rank(fleet):
    """Rank 1's deadline passes during the second decode step: rank 1 votes no in the validity vote, every
    rank raises there; the second decoded token is not committed."""
    fleet.start(Script(tokens={3: [7, 9, 10, 11]}))
    outcomes = queued(fleet, one(3), warmup=False, deadlines=(DEADLINE, 3.5))
    check_log(
        fleet,
        """
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        deliver q0[0] 7 -
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[1] 9 -
        vote 1 1 -> 1  # token delivered + deadline
        decode
        vote 1 0 -> 0  # token valid + deadline
        """,
    )
    check_errors(outcomes, [f"RuntimeError: {FLEET_REFUSED}"])
    for rank in RANKS:
        assert poisoned(fleet.nodes[rank])
        assert session_states(fleet, rank) == [(True, [("q0", 0, 7, None), ("q0", 1, 9, None)])]


def test_an_unhealthy_prefill_block_is_refused_before_any_session(fleet):
    """The prefill block's health vote fails: no request session is created, nothing is delivered."""
    fleet.start(Script(tokens={3: [7, 9]}, prefill_healthy=False))
    outcomes = queued(fleet, one(3), warmup=False)
    check_log(
        fleet,
        """
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 0 0 -> 0  # prefill block healthy
        """,
    )
    check_errors(outcomes, ["RuntimeError: optimized prefill failed"])
    for rank in RANKS:
        assert poisoned(fleet.nodes[rank]) and session_states(fleet, rank) == []
    assert delivered(fleet.nodes[0]) == []


def test_an_unfinished_prefill_fails_the_first_token_prefill_vote(fleet):
    """A healthy but unfinished prefill: the session's prefill-finished vote fails on every rank, with the
    refusal of ``finish_batched_prefill`` as the cause on every rank."""
    fleet.start(Script(tokens={3: [7, 9]}, prefill_finished=False))
    outcomes = queued(fleet, one(3), warmup=False)
    check_log(
        fleet,
        """
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 0 0 -> 0  # prefill finished + deadline
        """,
    )
    check_errors(
        outcomes,
        [f"RuntimeError: {FLEET_REFUSED}", "ValueError: batched prefill is not complete and healthy; decode refused"],
    )
    for rank in RANKS:
        assert poisoned(fleet.nodes[rank]) and session_states(fleet, rank) == [(True, [])]
    assert delivered(fleet.nodes[0]) == []


def test_an_output_mismatch_fails_the_consensus_phase_after_delivery(fleet):
    """Rank 1 decodes a different third token: every token is delivered, then the output-consensus digest
    differs and its phase fails on every rank. The sessions finished (not failed); the runtime is poisoned."""
    fleet.start(Script(tokens={3: [7, 9, 10]}, rank_tokens={(1, 3, 2): 42}))
    outcomes = queued(fleet, one(3, max_new_tokens=3), warmup=False)
    check_log(
        fleet,
        """
        phase open_tokens: vote 1 1 -> 1
        phase memory_cache_init: vote 1 1 -> 1
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_decode: vote 1 1 -> 1
        initialize 3
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        vote 1 1 -> 1  # prefill finished + deadline
        vote 1 1 -> 1  # first token valid + deadline
        deliver q0[0] 7 -
        vote 1 1 -> 1  # first token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[1] 9 -
        vote 1 1 -> 1  # token delivered + deadline
        decode
        vote 1 1 -> 1  # token valid + deadline
        deliver q0[2] 10 length
        vote 1 1 -> 1  # token delivered + deadline
        phase output_consensus: digest differs, vote 0 0 -> 0
        """,
    )
    check_errors(outcomes, ["RuntimeError: optimized output differs across hosts"])
    for rank in RANKS:
        assert poisoned(fleet.nodes[rank])
        third = 10 if rank == 0 else 42
        assert session_states(fleet, rank) == [
            (False, [("q0", 0, 7, None), ("q0", 1, 9, None), ("q0", 2, third, "length")])
        ]
    assert delivered(fleet.nodes[0]) == [("q0", 0, 7, None), ("q0", 1, 9, None), ("q0", 2, 10, "length")]


# ------------------------------------------------------------------------------ concurrent failure paths
BATCH_WARMUP = [("c0", 0, 7, None), ("c0", 1, 9, "length"), ("c1", 0, 8, None), ("c1", 1, 9, "length")]


def concurrent_failure(fleet: Fleet, script: Script, **kwargs: Any) -> dict:
    fleet.start(script, capacity=request.CONCURRENT_CAPACITY, concurrent_size=2)
    return concurrent(fleet, lanes((3, 3), (4, 3)), **kwargs)


def test_a_concurrent_delivery_failure_on_rank_0_stops_every_rank(fleet):
    """Rank 0's write of lane 0's second token fails: every rank raises at that round's delivery vote. The
    batched session keeps no cause, and the round is committed lane by lane up to the failed delivery: rank 0
    committed lane 0's token only, rank 1 (no-op deliveries) both lanes'."""
    outcomes = concurrent_failure(fleet, Script(tokens={3: [7, 9, 10], 4: [8, 9, 10]}, fail_delivery=("c0", 1)))
    check_log(
        fleet,
        """
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 4
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 4
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        digest same
        digest same
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 4
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 4
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[0] 7 - round 0
        deliver c1[0] 8 - round 0
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[1] 9 - round 1 (write fails)
        vote 0 1 -> 0  # round delivered + deadline
        """,
    )
    check_errors(outcomes, [f"RuntimeError: {BATCH_REFUSED}"])
    measured = [("c0", 0, 7, None), ("c0", 1, 9, None), ("c1", 0, 8, None)]
    assert session_states(fleet, 0) == [(False, BATCH_WARMUP), (True, measured)]
    assert session_states(fleet, 1) == [(False, BATCH_WARMUP), (True, [*measured, ("c1", 1, 9, None)])]
    assert all(poisoned(node) for node in fleet.nodes.values())
    assert delivered(fleet.nodes[0], "item000/tokens.jsonl") == [("c0", 0, 7, None, 0)]
    assert delivered(fleet.nodes[0], "item001/tokens.jsonl") == [("c1", 0, 8, None, 0)]


def test_invalid_concurrent_metadata_is_refused_before_the_round_is_committed(fleet):
    """Lane 1 reports an unhealthy state in the second decode round of the batch: the round's validity vote
    fails on every rank before any of its tokens is committed."""
    outcomes = concurrent_failure(fleet, Script(tokens={3: [7, 9, 10], 4: [8, 9, 10]}, bad_metadata=(4, 2)))
    check_log(
        fleet,
        """
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 4
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 4
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        digest same
        digest same
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 4
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 4
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[0] 7 - round 0
        deliver c1[0] 8 - round 0
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[1] 9 - round 1
        deliver c1[1] 9 - round 1
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 0 0 -> 0  # round valid + deadline
        """,
    )
    check_errors(outcomes, [f"RuntimeError: {BATCH_REFUSED}"])
    measured = [("c0", 0, 7, None), ("c0", 1, 9, None), ("c1", 0, 8, None), ("c1", 1, 9, None)]
    for rank in RANKS:
        assert session_states(fleet, rank) == [(False, BATCH_WARMUP), (True, measured)]
        assert poisoned(fleet.nodes[rank])


def test_a_concurrent_deadline_expiring_on_rank_1_stops_every_rank(fleet):
    """Rank 1's deadline passes during the first decode round of the batch: its validity vote fails."""
    outcomes = concurrent_failure(fleet, Script(tokens={3: [7, 9, 10], 4: [8, 9, 10]}), deadlines=(DEADLINE, 15.5))
    check_log(
        fleet,
        """
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 4
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 4
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        digest same
        digest same
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 4
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 4
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[0] 7 - round 0
        deliver c1[0] 8 - round 0
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 0 -> 0  # round valid + deadline
        """,
    )
    check_errors(outcomes, [f"RuntimeError: {BATCH_REFUSED}"])
    for rank in RANKS:
        assert session_states(fleet, rank) == [(False, BATCH_WARMUP), (True, [("c0", 0, 7, None), ("c1", 0, 8, None)])]
        assert poisoned(fleet.nodes[rank])


def test_an_unhealthy_concurrent_prefill_is_refused_in_the_warmup(fleet):
    """The first lane's prefill block health vote fails (in the warmup batch): no session, no directory."""
    outcomes = concurrent_failure(fleet, Script(tokens={3: [7, 9, 10], 4: [8, 9, 10]}, prefill_healthy=False))
    check_log(
        fleet,
        """
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 4
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 0 0 -> 0  # prefill block healthy
        """,
    )
    check_errors(outcomes, ["RuntimeError: concurrent prefill failed"])
    for rank in RANKS:
        assert session_states(fleet, rank) == [] and poisoned(fleet.nodes[rank])
        assert not (fleet.nodes[rank].root / "item000").exists()


def test_a_concurrent_output_mismatch_fails_after_delivery(fleet):
    """Rank 1 decodes a different third token for lane 0: the batch completes and delivers, then lane 0's
    digest differs and every rank raises before lane 1's digest (no phase, no vote)."""
    outcomes = concurrent_failure(fleet, Script(tokens={3: [7, 9, 10], 4: [8, 9, 10]}, rank_tokens={(1, 3, 2): 42}))
    check_log(
        fleet,
        """
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 4
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 4
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 1 -> 1  # round valid + deadline
        vote 1 1 -> 1  # round delivered + deadline
        digest same
        digest same
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        phase request_directory: vote 1 1 -> 1
        phase open_tokens: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        phase memory_batch_cache_init: vote 1 1 -> 1
        initialize_batch 3 4
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 3
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 3
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 0
        vote 1 1 -> 1  # deadline
        phase memory_cache_init: vote 1 1 -> 1
        initialize 4
        phase memory_prefill_128: vote 1 1 -> 1
        phase memory_prefill_114: vote 1 1 -> 1
        phase memory_batch_insert: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        prefill 114 4
        vote 1 1 -> 1  # prefill block healthy
        phase batch_prefill_finish: vote 1 1 -> 1
        insert 1
        phase memory_batch_decode: vote 1 1 -> 1
        vote 1 1 -> 1  # deadline
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[0] 7 - round 0
        deliver c1[0] 8 - round 0
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[1] 9 - round 1
        deliver c1[1] 9 - round 1
        vote 1 1 -> 1  # round delivered + deadline
        vote 1 1 -> 1  # round start + deadline
        decode_batch 1 1
        vote 1 1 -> 1  # round valid + deadline
        deliver c0[2] 10 length round 2
        deliver c1[2] 10 length round 2
        vote 1 1 -> 1  # round delivered + deadline
        digest differs
        """,
    )
    check_errors(outcomes, ["RuntimeError: concurrent output differs across hosts"])
    for rank in RANKS:
        third = 10 if rank == 0 else 42
        lanes_ = [("c0", 0, 7, None), ("c0", 1, 9, None), ("c0", 2, third, "length")]
        lanes_ += [("c1", 0, 8, None), ("c1", 1, 9, None), ("c1", 2, 10, "length")]
        assert session_states(fleet, rank) == [(False, BATCH_WARMUP), (False, lanes_)]
        assert poisoned(fleet.nodes[rank])
