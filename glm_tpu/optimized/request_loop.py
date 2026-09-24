"""The packed greedy decode program and its request session: one metadata read, two votes per token.

``build_packed_decoder_program`` wraps the decode step so each token returns an int32
``[token, all-owner health, next position, next context length]`` vector (one device-to-host read).
``PackedRequestSession`` admits health/frontier/timing together before delivery, followed by the
delivery-success vote; exceptions still poison the live session. The release decodes greedily:
the request policy carries no seed and no uniform draw is ever made (S2d).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import PartitionSpec as P

from .request_session import TokenEvent, Ws32RequestSession
from .ws32_decoder_challenger import build_ws32_challenger_decoder_program


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


class PackedDecodeResult(NamedTuple):
    decoded: Any
    # int32 [token, all-owner health, next position, next context length]
    metadata: Any


@dataclass(frozen=True, slots=True)
class PackedDecoderProgram:
    execute: Any


def pack_decode_metadata_mapped(token, health, position, lengths, draw_valid):
    if (token.shape != (1,) or token.dtype != jnp.int32 or health.shape != (1,)
            or health.dtype != jnp.bool_ or position.shape != (1,) or position.dtype != jnp.int32
            or lengths.shape != (1,) or lengths.dtype != jnp.int32
            or draw_valid.shape != () or draw_valid.dtype != jnp.bool_):
        raise ValueError('packed decode metadata geometry/dtype drifted')
    valid = lax.pmin((health[0] & draw_valid).astype(jnp.int32),('expert','feature'))
    return jnp.stack((token[0],valid,position[0],lengths[0]))


def build_packed_decoder_program(mesh, config, *, sparse_attention_interpret=False, linear_interpret=False):
    """The decode step's arguments ``(token, state, weights[, main_rope_table])``; no state donation
    or speculative extra model step is introduced (the runtime applies donation above 8,192 slots)."""
    base = build_ws32_challenger_decoder_program(mesh,config,
        sparse_attention_interpret=sparse_attention_interpret,linear_interpret=linear_interpret)
    pack = jax.shard_map(pack_decode_metadata_mapped,mesh=mesh,
                        in_specs=(P(),)*5,out_specs=P(),check_vma=False)
    def execute(token,state,weights,*extra):
        expected = int(config.host_main_rope_table)
        if len(extra) != expected:
            raise ValueError('packed decoder arguments disagree with rotary/sampling flags')
        rope = extra[:1] if config.host_main_rope_table else ()
        # The greedy step has no draw to admit; the metadata still ANDs this constant into the
        # all-owner health (part of the compiled program).
        valid = jnp.bool_(True)
        result = base.execute(token,state,weights,*rope)
        metadata = pack(result.next_token,result.state.contract_valid,
                        result.state.position,result.state.context_lengths,valid)
        return PackedDecodeResult(result,metadata)
    return PackedDecoderProgram(jax.jit(execute))


class PackedRequestSession(Ws32RequestSession):
    """Frozen prefill/delivery policy, compact host admission for greedy decode.

    ``decode_step(token,state)`` must return the PackedDecodeResult produced by
    the builder above, with weights and rotary data bound. The fleet callback
    still performs one real all-host vote per invocation.
    """

    def __init__(
        self, policy: RequestPolicy, *,
        decode_step: Callable[[Any, Any], PackedDecodeResult],
        fleet_all: Callable[[bool], bool],
        deliver: Callable[[TokenEvent], None],
        delivery_boundary: str,
        request_started: float,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not isinstance(policy, RequestPolicy):
            raise ValueError('an explicit greedy RequestPolicy is required')
        # Greedy decoding draws no uniforms: nothing to replicate (next_uniform refuses).
        super().__init__(policy, decode_step=decode_step, replicate_uniform=None, fleet_all=fleet_all,
                         deliver=deliver, delivery_boundary=delivery_boundary,
                         request_started=request_started, clock=clock)

    def next_uniform(self) -> Any:
        raise RuntimeError('greedy decoding draws no uniforms')

    def step(self) -> TokenEvent:
        self._begin()
        try:
            if not self._events:
                raise RuntimeError('complete prefill before decoding')
            started = self._clock()
            packed = self._decode(self._pending_token,self._state)
            jax.block_until_ready(packed)
            elapsed = self._clock() - started
            error = None
            try:
                status = np.asarray(packed.metadata)  # One device-to-host read.
                expected = self.policy.prompt_tokens + len(self._events)
                valid = (math.isfinite(elapsed) and elapsed >= 0
                         and status.shape == (4,) and status.dtype == np.int32
                         and 0 <= int(status[0]) < self.policy.vocab_size
                         and int(status[1]) == 1 and int(status[2]) == expected
                         and int(status[3]) == expected+1)
            except Exception as exc:
                valid,error = False,exc
            self._vote(valid,error)
            if error is not None:
                raise RuntimeError('invalid packed request metadata') from error
            self._decode_seconds.append(elapsed)
            token_id = int(status[0])
            reason = ('eos' if token_id in self.policy.eos_ids else
                      'length' if len(self._events)+1 == self.policy.max_new_tokens else None)
            event = TokenEvent(self.policy.request_id,len(self._events),token_id,reason)
            result = packed.decoded
            # Commit before invoking the sink: an ambiguous failure cannot retry.
            self._state,self._pending_token = result.state,result.next_token
            self._events.append(event)
            delivered = None
            try:
                self._deliver(event)
                delivered = self._clock()
                floor = self._delivered_at[-1] if self._delivered_at else self.request_started
                if not math.isfinite(delivered) or delivered < floor:
                    raise ValueError('delivery clock moved backwards')
            except Exception as exc:
                error = exc
            self._vote(error is None,error)
            if error is not None:
                raise RuntimeError('delivery failed; request cannot be retried') from error
            self._delivered_at.append(delivered)
            return event
        except Exception:
            self._failed = True
            raise
        finally:
            self._busy = False
