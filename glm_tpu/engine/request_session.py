"""Request sessions: token delivery for one request, or for a batch of requests, on programs the
runtime has already compiled and admitted.

The worker owns checkpoint loading, compilation, memory admission, leases and
fleet cleanup. A session owns the generated-token frontier. It does not create
compute, load weights, call another engine or copy caches to host.
Pause/resume here means retaining this SAME live session/cache in memory; it
is not a claim of process-crash or durable KV-checkpoint recovery.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
import math
from time import perf_counter
from typing import Any
import time

import jax
import numpy as np

from glm_tpu.models.glm_moe_dsa.state import BatchedPrefillResult, finish_batched_prefill, DecodeStepResult
from glm_tpu.engine.outputs import TokenEvent
from glm_tpu.models.glm_moe_dsa.model import PackedDecodeResult


def request_uniform(*, seed: int, request_id: str, token_index: int) -> float:
    """Stateless replayable SHA256/24-bit uniform in [0,1), exactly FP32.

    Persist this algorithm identity, seed, request id and next token index with
    the whole request state. Resume must not reset the counter. One draw per
    delivered/generated token, including the first token after prefill. This
    protocol is explicit and does not claim the model card's unspecified RNG.
    """
    if any(type(x) is not int or not 0 <= x < 2**64 for x in (seed, token_index)):
        raise ValueError("seed and token index must be uint64 integers")
    if type(request_id) is not str or not request_id:
        raise ValueError("a nonempty request id is required")
    identity = sha256(request_id.encode("utf-8")).digest()
    digest = sha256(
        b"glm-ws32-request-uniform-v1\0" + seed.to_bytes(8, "big") + identity + token_index.to_bytes(8, "big")
    ).digest()
    return int.from_bytes(digest[:3], "big") / 2**24


@dataclass(frozen=True, slots=True)
class SampledRequestPolicy:
    request_id: str
    seed: int
    prompt_tokens: int
    max_new_tokens: int
    context_capacity: int
    vocab_size: int
    eos_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        request_uniform(seed=self.seed, request_id=self.request_id, token_index=0)
        if any(
            type(x) is not int or x <= 0
            for x in (self.prompt_tokens, self.max_new_tokens, self.context_capacity, self.vocab_size)
        ):
            raise ValueError("positive integer request dimensions required")
        if self.prompt_tokens + self.max_new_tokens > self.context_capacity:
            raise ValueError("full registered generation cap must fit; no silent truncation")
        if (
            type(self.eos_ids) is not tuple
            or not self.eos_ids
            or len(set(self.eos_ids)) != len(self.eos_ids)
            or any(type(x) is not int or not 0 <= x < self.vocab_size for x in self.eos_ids)
        ):
            raise ValueError("unique in-vocabulary EOS ids required")


class Ws32RequestSession:
    """Consume one prefill result, then decode/emit with a resumable RNG index.

    All hosts follow the same schedule. ``fleet_all`` must implement the
    authenticated worker's all-host boolean vote (identity in CPU-only tests).
    ``deliver`` runs at the actual output boundary (e.g. rank0 write+flush);
    it must be bounded. Its completion time is recorded, never prefill-ready
    time mislabeled as delivery. Non-output ranks use a no-op delivery callback
    and their timestamps MUST NOT be published as client delivery latency.

    ``decode_step`` is the already-compiled sampled decoder with weights/RoPE
    bound; ``replicate_uniform`` supplies its one replicated FP32 scalar. No
    per-token compile, transformer dispatch or RNG reset lives in this class.
    An exception poisons the session: no retry with possibly consumed buffers,
    and no duplicate token emission after an ambiguous delivery failure.
    """

    def __init__(
        self,
        policy: SampledRequestPolicy,
        *,
        decode_step: Callable[[Any, Any, Any], DecodeStepResult],
        replicate_uniform: Callable[[np.ndarray], Any],
        fleet_all: Callable[[bool], bool],
        deliver: Callable[[TokenEvent], None],
        delivery_boundary: str,
        request_started: float,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        now = clock()
        if (
            not math.isfinite(request_started)
            or not math.isfinite(now)
            or request_started > now
            or not delivery_boundary.strip()
        ):
            raise ValueError("request clock and named delivery boundary required")
        self.policy = policy
        self._decode = decode_step
        self._replicate = replicate_uniform
        self._fleet_all = fleet_all
        self._deliver = deliver
        self._clock = clock
        self.delivery_boundary = delivery_boundary
        self.request_started = request_started
        self._state = None
        self._pending_token = None
        self._events: list[TokenEvent] = []
        self._decode_seconds: list[float] = []
        self._delivered_at: list[float] = []
        self._failed = False
        self._busy = False

    @property
    def events(self) -> tuple[TokenEvent, ...]:
        return tuple(self._events)

    @property
    def finished(self) -> bool:
        return bool(self._events and self._events[-1].finish_reason is not None)

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def ttft_seconds(self) -> float | None:
        return None if not self._delivered_at else self._delivered_at[0] - self.request_started

    @property
    def decode_seconds(self) -> tuple[float, ...]:
        """Completed device-step wall only; excludes delivery/host health vote."""
        return tuple(self._decode_seconds)

    @property
    def delivered_request_seconds(self) -> float | None:
        return None if not self._delivered_at else self._delivered_at[-1] - self.request_started

    def next_uniform(self) -> Any:
        """Same draw for every nonfinal prompt block; first generated index=0."""
        if self._failed or self.finished:
            raise RuntimeError("terminal request cannot sample")
        value = request_uniform(seed=self.policy.seed, request_id=self.policy.request_id, token_index=len(self._events))
        return self._replicate(np.asarray(value, np.float32))

    def release(self) -> None:
        """Drop terminal cache roots, retaining the compact output/timing log.

        An unfinished live request cannot be released as if it were complete.
        The worker's all-live census still detects any external cache aliases.
        No weights or shared buffers are explicitly deleted here.
        """
        if self._busy or not (self.finished or self._failed):
            raise RuntimeError("only a terminal request may release its cache")
        self._state = None
        self._pending_token = None

    def _begin(self) -> None:
        if self._failed or self.finished or self._busy:
            raise RuntimeError("terminal or reentrant request operation")
        self._busy = True

    def _vote(self, valid: bool, cause: Exception | None = None) -> None:
        agreed = self._fleet_all(bool(valid))
        if type(agreed) is not bool or not agreed or not valid:
            raise RuntimeError("request fleet rejected state/delivery") from cause

    def _accept(self, state: Any, token_array: Any) -> TokenEvent:
        expected = self.policy.prompt_tokens + len(self._events)
        error = None
        try:
            token = np.asarray(token_array)
            valid = (
                token.shape == (1,)
                and token.dtype == np.int32
                and 0 <= int(token[0]) < self.policy.vocab_size
                and bool(np.asarray(state.contract_valid).all())
                and np.array_equal(np.asarray(state.position), [expected])
                and np.array_equal(np.asarray(state.context_lengths), [expected + 1])
            )
        except Exception as exc:
            valid, error = False, exc
        self._vote(valid, error)
        if error is not None:  # Defensive: a bad vote callback cannot mask a local failure.
            raise RuntimeError("invalid request metadata") from error
        token_id = int(token[0])
        reason = (
            "eos"
            if token_id in self.policy.eos_ids
            else "length"
            if len(self._events) + 1 == self.policy.max_new_tokens
            else None
        )
        event = TokenEvent(self.policy.request_id, len(self._events), token_id, reason)
        # Commit the state/frontier BEFORE invoking user delivery code. A sink
        # failure may already have emitted bytes; never regenerate or retry it.
        self._state, self._pending_token = state, token_array
        self._events.append(event)
        delivered = None
        try:
            self._deliver(event)
            delivered = self._clock()
            floor = self._delivered_at[-1] if self._delivered_at else self.request_started
            if not math.isfinite(delivered) or delivered < floor:
                raise ValueError("delivery clock moved backwards")
        except Exception as exc:
            error = exc
        self._vote(error is None, error)
        if error is not None:
            raise RuntimeError("delivery failed; request cannot be retried") from error
        self._delivered_at.append(delivered)
        return event

    def accept_prefill(self, result: BatchedPrefillResult) -> TokenEvent:
        """Deliver the sampled FIRST token now, before any decode call."""
        self._begin()
        try:
            if self._events or self._state is not None:
                raise RuntimeError("prefill was already consumed")
            jax.block_until_ready(result)
            error = None
            try:
                state, token = finish_batched_prefill(result)
            except Exception as exc:
                error = exc
            self._vote(error is None, error)
            if error is not None:
                raise RuntimeError("invalid prefill") from error
            return self._accept(state, token)
        except Exception:
            self._failed = True
            raise
        finally:
            self._busy = False

    def step(self) -> TokenEvent:
        """Resume this live cache; one sampled decode, no hidden extra token."""
        self._begin()
        try:
            if not self._events:
                raise RuntimeError("complete prefill before decoding")
            draw = self.next_uniform()
            started = self._clock()
            result = self._decode(self._pending_token, self._state, draw)
            jax.block_until_ready(result)
            elapsed = self._clock() - started
            self._vote(math.isfinite(elapsed) and elapsed >= 0)
            self._decode_seconds.append(elapsed)
            return self._accept(result.state, result.next_token)
        except Exception:
            self._failed = True
            raise
        finally:
            self._busy = False


class BatchedSession:
    def __init__(self, requests, *, decode, put, vote, deliver, deadline, clock=time.perf_counter):
        self.requests = requests
        self.decode, self.put, self.vote, self.deliver = decode, put, vote, deliver
        self.deadline, self.clock = deadline, clock
        self.events = [[] for _ in requests]
        self.active = np.ones(len(requests), bool)
        self.finished_at = [None] * len(requests)
        self.failed = False
        self.round = 0

    def require(self, valid):
        agreed = self.vote(bool(valid) and self.clock() < self.deadline)
        if not valid or agreed is not True:
            raise RuntimeError("concurrent batch rejected state, delivery or deadline")

    def accept(self, metadata):
        """Admit the whole round before committing/delivering any of its tokens."""
        status = np.asarray(metadata)
        valid = status.shape == (len(self.requests), 4) and status.dtype == np.int32
        if valid:
            for lane, item in enumerate(self.requests):
                if not self.active[lane]:
                    continue
                token, health, position, length = map(int, status[lane])
                expected = len(item["prompt_ids"]) + len(self.events[lane])
                valid &= (
                    0 <= token < item["vocab_size"] and health == 1 and position == expected and length == expected + 1
                )
        self.require(valid)
        error = None
        try:
            for lane, item in enumerate(self.requests):
                if not self.active[lane]:
                    continue
                token = int(status[lane, 0])
                reason = (
                    "eos"
                    if token in item["eos_ids"]
                    else "length"
                    if len(self.events[lane]) + 1 == item["max_new_tokens"]
                    else None
                )
                event = TokenEvent(item["request_id"], len(self.events[lane]), token, reason)
                self.events[lane].append(event)
                if reason:
                    self.active[lane] = False
                self.deliver(lane, event, self.round)
                if reason:
                    self.finished_at[lane] = self.clock()
        except Exception as exc:
            error = exc
        self.require(error is None)
        if error is not None:
            raise RuntimeError("concurrent delivery failed") from error

    def run(self, state, tokens, first_metadata):
        if self.failed or self.round or any(self.events):
            raise RuntimeError("concurrent session cannot be reused")
        try:
            self.accept(first_metadata)
            self.decode_started = self.clock()
            while self.active.any():
                self.require(True)
                out = jax.block_until_ready(self.decode(tokens, state, self.put(self.active)))
                # Transfer exclusive ownership before delivery: an ambiguous
                # failure poisons this session and cannot replay consumed state.
                state, tokens = out.state, out.next_token
                self.round += 1
                self.accept(out.metadata)
                del out
            self.decode_seconds = self.clock() - self.decode_started
        except Exception:
            self.failed = True
            raise
        finally:
            # No live cache survives the completed or failed batch.
            del state, tokens


@dataclass(frozen=True, slots=True)
class RequestPolicy:
    """One greedy request's identity and generation limits.

    The checks, their order and their exceptions are those of the frozen sampled
    ``ws32_request_session.RequestPolicy`` at seed 0 (the release's value): the request id must be a
    nonempty, UTF-8-encodable string (the frozen policy hashed it for its uniform draws).
    """

    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    context_capacity: int
    vocab_size: int
    eos_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("a nonempty request id is required")
        self.request_id.encode("utf-8")
        if any(
            type(x) is not int or x <= 0
            for x in (self.prompt_tokens, self.max_new_tokens, self.context_capacity, self.vocab_size)
        ):
            raise ValueError("positive integer request dimensions required")
        if self.prompt_tokens + self.max_new_tokens > self.context_capacity:
            raise ValueError("full registered generation cap must fit; no silent truncation")
        if (
            type(self.eos_ids) is not tuple
            or not self.eos_ids
            or len(set(self.eos_ids)) != len(self.eos_ids)
            or any(type(x) is not int or not 0 <= x < self.vocab_size for x in self.eos_ids)
        ):
            raise ValueError("unique in-vocabulary EOS ids required")


class PackedRequestSession(Ws32RequestSession):
    """Frozen prefill/delivery policy, compact host admission for greedy decode.

    ``decode_step(token,state)`` must return the PackedDecodeResult produced by
    the builder above, with weights and rotary data bound. The fleet callback
    still performs one real all-host vote per invocation.
    """

    def __init__(
        self,
        policy: RequestPolicy,
        *,
        decode_step: Callable[[Any, Any], PackedDecodeResult],
        fleet_all: Callable[[bool], bool],
        deliver: Callable[[TokenEvent], None],
        delivery_boundary: str,
        request_started: float,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not isinstance(policy, RequestPolicy):
            raise ValueError("an explicit greedy RequestPolicy is required")
        # Greedy decoding draws no uniforms: nothing to replicate (next_uniform refuses).
        super().__init__(
            policy,
            decode_step=decode_step,
            replicate_uniform=None,
            fleet_all=fleet_all,
            deliver=deliver,
            delivery_boundary=delivery_boundary,
            request_started=request_started,
            clock=clock,
        )

    def next_uniform(self) -> Any:
        raise RuntimeError("greedy decoding draws no uniforms")

    def step(self) -> TokenEvent:
        self._begin()
        try:
            if not self._events:
                raise RuntimeError("complete prefill before decoding")
            started = self._clock()
            packed = self._decode(self._pending_token, self._state)
            jax.block_until_ready(packed)
            elapsed = self._clock() - started
            error = None
            try:
                status = np.asarray(packed.metadata)  # One device-to-host read.
                expected = self.policy.prompt_tokens + len(self._events)
                valid = (
                    math.isfinite(elapsed)
                    and elapsed >= 0
                    and status.shape == (4,)
                    and status.dtype == np.int32
                    and 0 <= int(status[0]) < self.policy.vocab_size
                    and int(status[1]) == 1
                    and int(status[2]) == expected
                    and int(status[3]) == expected + 1
                )
            except Exception as exc:
                valid, error = False, exc
            self._vote(valid, error)
            if error is not None:
                raise RuntimeError("invalid packed request metadata") from error
            self._decode_seconds.append(elapsed)
            token_id = int(status[0])
            reason = (
                "eos"
                if token_id in self.policy.eos_ids
                else "length"
                if len(self._events) + 1 == self.policy.max_new_tokens
                else None
            )
            event = TokenEvent(self.policy.request_id, len(self._events), token_id, reason)
            result = packed.decoded
            # Commit before invoking the sink: an ambiguous failure cannot retry.
            self._state, self._pending_token = result.state, result.next_token
            self._events.append(event)
            delivered = None
            try:
                self._deliver(event)
                delivered = self._clock()
                floor = self._delivered_at[-1] if self._delivered_at else self.request_started
                if not math.isfinite(delivered) or delivered < floor:
                    raise ValueError("delivery clock moved backwards")
            except Exception as exc:
                error = exc
            self._vote(error is None, error)
            if error is not None:
                raise RuntimeError("delivery failed; request cannot be retried") from error
            self._delivered_at.append(delivered)
            return event
        except Exception:
            self._failed = True
            raise
        finally:
            self._busy = False
