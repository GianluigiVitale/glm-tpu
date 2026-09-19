"""Opt-in D4 request loop: one metadata read and two votes per decode token.

The model step is unchanged. A request's original hash-based FP32 uniforms are
placed once; the device selects the draw from the committed token frontier.
Health/frontier/timing are admitted together before delivery, followed by the
original delivery-success vote. Exceptions still poison the live session.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from ..greenfield.kernels.ws32_sampling import request_uniform
from ..greenfield.runtime.ws32_request_session import RequestPolicy, TokenEvent, Ws32RequestSession
from .ws32_decoder_challenger import build_ws32_challenger_decoder_program, Ws32PerfOptions


class PackedDecodeResult(NamedTuple):
    decoded: Any
    # int32 [token, all-owner health, next position, next context length]
    metadata: Any


@dataclass(frozen=True, slots=True)
class PackedDecoderProgram:
    execute: Any
    sampled: bool


def request_uniform_values(policy: RequestPolicy) -> np.ndarray:
    """The original draw at every generated-token index, including prefill index0."""
    if not isinstance(policy, RequestPolicy):
        raise ValueError('an explicit RequestPolicy is required')
    return np.asarray([request_uniform(seed=policy.seed,request_id=policy.request_id,token_index=i)
                       for i in range(policy.max_new_tokens)],np.float32)


def make_request_uniform_bank(mesh: Any, policy: RequestPolicy) -> Any:
    return jax.device_put(request_uniform_values(policy),NamedSharding(mesh,P()))


def pack_decode_metadata_mapped(token, health, position, lengths, draw_valid):
    if (token.shape != (1,) or token.dtype != jnp.int32 or health.shape != (1,)
            or health.dtype != jnp.bool_ or position.shape != (1,) or position.dtype != jnp.int32
            or lengths.shape != (1,) or lengths.dtype != jnp.int32
            or draw_valid.shape != () or draw_valid.dtype != jnp.bool_):
        raise ValueError('packed decode metadata geometry/dtype drifted')
    valid = lax.pmin((health[0] & draw_valid).astype(jnp.int32),('expert','feature'))
    return jnp.stack((token[0],valid,position[0],lengths[0]))


def build_packed_decoder_program(mesh, config, *, options=Ws32PerfOptions(), sampling=None,
                                 sparse_attention_interpret=False, linear_interpret=False):
    """Model arguments followed by ``uniform_bank, prompt_length`` for sampled mode.

    Greedy mode keeps the existing argument list. Sampled callers bind the bank
    and scalar prompt length once; there is no per-token device_put or RNG reset.
    No state donation or speculative extra model step is introduced.
    """
    base = build_ws32_challenger_decoder_program(mesh,config,options=options,sampling=sampling,
        sparse_attention_interpret=sparse_attention_interpret,linear_interpret=linear_interpret)
    pack = jax.shard_map(pack_decode_metadata_mapped,mesh=mesh,
                        in_specs=(P(),)*5,out_specs=P(),check_vma=False)
    def execute(token,state,weights,*extra):
        expected = int(config.host_main_rope_table) + (2 if base.takes_uniform else 0)
        if len(extra) != expected:
            raise ValueError('packed decoder arguments disagree with rotary/sampling flags')
        rope = extra[:1] if config.host_main_rope_table else ()
        valid = jnp.bool_(True)
        draw = ()
        if base.takes_uniform:
            bank,prompt_length = extra[-2:]
            if (bank.ndim != 1 or bank.shape[0] < 1 or bank.dtype != jnp.float32
                    or prompt_length.shape != () or prompt_length.dtype != jnp.int32):
                raise ValueError('sampled packed decoder requires FP32 bank and int32 prompt length')
            index = state.position[0] - prompt_length + jnp.int32(1)
            value = jnp.take(bank,index,mode='clip')
            valid = (index >= 1) & (index < bank.shape[0]) & jnp.isfinite(value) & (value >= 0) & (value < 1)
            draw = (value,)
        result = base.execute(token,state,weights,*rope,*draw)
        metadata = pack(result.next_token,result.state.contract_valid,
                        result.state.position,result.state.context_lengths,valid)
        return PackedDecodeResult(result,metadata)
    return PackedDecoderProgram(jax.jit(execute),base.takes_uniform)


class PackedRequestSession(Ws32RequestSession):
    """Frozen prefill/delivery policy, compact host admission for decode only.

    ``decode_step(token,state)`` must return the PackedDecodeResult produced by
    the builder above, with weights, rotary data, bank and prompt length bound.
    ``replicate_uniform`` is retained for the original prefill interface only.
    The fleet callback still performs one real all-host vote per invocation.
    """

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
