"""Sequential request delivery on already-admitted native sampled programs.

The protected worker owns checkpoint loading, compilation, memory admission,
leases and fleet cleanup. This class owns the generated-token frontier. It does
not create compute, load weights, call a legacy engine or copy caches to host.
Pause/resume here means retaining this SAME live session/cache in memory; it
is not a claim of process-crash or durable KV-checkpoint recovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Callable

import jax
import numpy as np

from ..kernels.ws32_sampling import request_uniform
from .ws32_batched_prefill import Ws32BatchedPrefillResult, finish_ws32_batched_prefill
from .ws32_decoder import Ws32DecodeStepResult


@dataclass(frozen=True, slots=True)
class RequestPolicy:
    request_id: str
    seed: int
    prompt_tokens: int
    max_new_tokens: int
    context_capacity: int
    vocab_size: int
    eos_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        request_uniform(seed=self.seed, request_id=self.request_id, token_index=0)
        if any(type(x) is not int or x <= 0 for x in (
            self.prompt_tokens, self.max_new_tokens, self.context_capacity, self.vocab_size
        )):
            raise ValueError("positive integer request dimensions required")
        if self.prompt_tokens + self.max_new_tokens > self.context_capacity:
            raise ValueError("full registered generation cap must fit; no silent truncation")
        if (type(self.eos_ids) is not tuple or not self.eos_ids
                or len(set(self.eos_ids)) != len(self.eos_ids)
                or any(type(x) is not int or not 0 <= x < self.vocab_size for x in self.eos_ids)):
            raise ValueError("unique in-vocabulary EOS ids required")


@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: str
    index: int
    token_id: int
    finish_reason: str | None


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
        self, policy: RequestPolicy, *,
        decode_step: Callable[[Any, Any, Any], Ws32DecodeStepResult],
        replicate_uniform: Callable[[np.ndarray], Any],
        fleet_all: Callable[[bool], bool],
        deliver: Callable[[TokenEvent], None],
        delivery_boundary: str,
        request_started: float,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        now = clock()
        if (not math.isfinite(request_started) or not math.isfinite(now)
                or request_started > now or not delivery_boundary.strip()):
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
        value = request_uniform(seed=self.policy.seed, request_id=self.policy.request_id,
                                token_index=len(self._events))
        return self._replicate(np.asarray(value, np.float32))

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
            valid = (token.shape == (1,) and token.dtype == np.int32
                     and 0 <= int(token[0]) < self.policy.vocab_size
                     and bool(np.asarray(state.contract_valid).all())
                     and np.array_equal(np.asarray(state.position), [expected])
                     and np.array_equal(np.asarray(state.context_lengths), [expected + 1]))
        except Exception as exc:
            valid, error = False, exc
        self._vote(valid, error)
        if error is not None:  # Defensive: a bad vote callback cannot mask a local failure.
            raise RuntimeError("invalid request metadata") from error
        token_id = int(token[0])
        reason = ("eos" if token_id in self.policy.eos_ids else
                  "length" if len(self._events) + 1 == self.policy.max_new_tokens else None)
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

    def accept_prefill(self, result: Ws32BatchedPrefillResult) -> TokenEvent:
        """Deliver the sampled FIRST token now, before any decode call."""
        self._begin()
        try:
            if self._events or self._state is not None:
                raise RuntimeError("prefill was already consumed")
            jax.block_until_ready(result)
            error = None
            try:
                state, token = finish_ws32_batched_prefill(result)
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
