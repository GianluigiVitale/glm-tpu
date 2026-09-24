"""Native multi-request execution on already-admitted resident programs.

This is a worker component, NOT a launcher or standalone authorization. Reuses
the frozen B128/B114 batched path, existing input packing/fresh-cache allocator,
sampled decoder and request session. No legacy execution or checkpoint copies.
The protected caller must admit ALL resident raw/decode/exact weight roots and
programs simultaneously; old one-way long-phase memory receipts do not do that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Mapping

import jax
import numpy as np

from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig
from glm_tpu.optimized.request_session import RequestPolicy, TokenEvent, Ws32RequestSession
from scripts.greenfield.ws32_batched_prefill_runner import graph_inputs, replicated


def prompt_blocks(length: int) -> tuple[tuple[int, int, int], ...]:
    """(offset, live rows, physical rows), using only frozen B128/B114 shapes.

    A remainder<=114 uses B114; 115..128 uses the existing B128 final branch.
    No padding token is part of the request and no new prompt-specific compile
    shape is generated. DB603's2034 tokens retain fifteen B128 then one B114.
    """
    if type(length) is not int or length <= 0:
        raise ValueError("positive integer prompt length required")
    result = []
    for offset in range(0, length, 128):
        live = min(128, length - offset)
        result.append((offset, live, 114 if live <= 114 else 128))
    return tuple(result)


@dataclass
class NativeBenchmarkRuntime:
    mesh: Any
    raw_config: Ws32DecoderConfig
    decode_config: Ws32DecoderConfig
    raw_weights: Any
    decode_weights: Any
    wk: tuple[Any, ...]
    exact_weights: tuple[Any, ...] | None
    rope: Any
    # Values MUST be compiled sampled programs; no implicit JIT in this loop.
    prefill_compiled: Mapping[int, Any]
    decode_compiled: Any
    cache_initializer: Any
    # A protected all-resident source/HLO/HBM check, not a boolean attestation.
    # stage=before_cache must include the planned cache in its allocation budget.
    # stage=cache_ready/prefill_done binds actual all-live arrays/peak counters.
    authorize: Callable[[str, Mapping[str, Any], Any], None]
    fleet_all: Callable[[bool], bool]
    clock: Callable[[], float] = perf_counter
    active: Ws32RequestSession | None = field(default=None, init=False)
    phase_seconds: dict[str, float] = field(default_factory=dict, init=False)
    failed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if (self.raw_config.exact_dsa or self.raw_config.strategy_nd_dense
                or not self.raw_config.host_main_rope_table
                or not self.decode_config.host_main_rope_table
                or self.raw_config.geometry != self.decode_config.geometry
                or self.raw_config.context_capacity != self.decode_config.context_capacity
                or set(self.prefill_compiled) != {114, 128}
                or self.decode_config.exact_dsa != (self.exact_weights is not None)
                or len(self.wk) != len(self.raw_config.full_index_slots)):
            raise ValueError("native benchmark program/config/weight view drifted")
        if any(not hasattr(fn, "memory_analysis") for fn in (
            *self.prefill_compiled.values(), self.decode_compiled, self.cache_initializer
        )):
            raise TypeError("native benchmark needs explicit compiled executables")

    def resident_roots(self) -> dict[str, Any]:
        """Same buffers may appear in both views: census physical aliases once."""
        return dict(raw_weights=self.raw_weights, decode_weights=self.decode_weights,
                    wk=self.wk, exact_weights=self.exact_weights, rope=self.rope)

    def _phase(self, operation: Callable[[], Any]) -> Any:
        error = None
        value = None
        try:
            value = operation()
        except Exception as exc:
            error = exc
        agreed = self.fleet_all(error is None)
        if type(agreed) is not bool or not agreed or error is not None:
            raise RuntimeError("native benchmark fleet phase refused") from error
        return value

    def _decode(self, token: Any, state: Any, uniform: Any) -> Any:
        extra = (self.exact_weights,) if self.decode_config.exact_dsa else ()
        return self.decode_compiled(token, state, self.decode_weights,
                                    *extra, self.rope, uniform)

    def start_request(
        self, token_ids: np.ndarray, policy: RequestPolicy, *,
        deliver: Callable[[TokenEvent], None], delivery_boundary: str,
        request_started: float,
    ) -> Ws32RequestSession:
        """Fresh isolated prefill and immediate first-token delivery, then pause.

        The caller can step the returned live session and retain it across
        pauses. ``close_request`` is required before starting the next item.
        A failed start is terminal for this runtime; outer recovery owns cleanup.
        """
        try:
            def validate():
                if self.failed or self.active is not None:
                    raise RuntimeError("runtime is failed or still owns a prior request")
                if (not isinstance(token_ids, np.ndarray) or token_ids.dtype != np.int32
                        or token_ids.shape != (policy.prompt_tokens,)
                        or not np.all((token_ids >= 0) & (token_ids < policy.vocab_size))
                        or policy.context_capacity != self.raw_config.context_capacity
                        or policy.vocab_size != self.raw_config.geometry.vocab_size):
                    raise ValueError("request ids/capacity differ from registered policy")
                self.authorize("before_cache", self.resident_roots(), None)
            self._phase(validate)
            self.phase_seconds = dict(cache_initialization=0.0, input_transfer=0.0,
                                      prefill_device=0.0, prefill_request=0.0)
            session = self._phase(lambda:Ws32RequestSession(
                policy, decode_step=self._decode,
                replicate_uniform=lambda value:replicated(self.mesh, value),
                fleet_all=self.fleet_all, deliver=deliver,
                delivery_boundary=delivery_boundary, request_started=request_started,
                clock=self.clock,
            ))
            self.active = session
            started = self.clock()
            current = self._phase(lambda:self.cache_initializer(
                replicated(self.mesh, np.asarray(policy.prompt_tokens, np.int32))))
            jax.block_until_ready(current)
            self.phase_seconds["cache_initialization"] = self.clock() - started
            self._phase(lambda:self.authorize("cache_ready", self.resident_roots(), current))
            prefill_started = self.clock()
            uniform = session.next_uniform()  # index0; same draw throughout prompt
            for offset, live, physical in prompt_blocks(policy.prompt_tokens):
                transferred = self.clock()
                inputs = self._phase(lambda:graph_inputs(
                    self.mesh, token_ids[offset:offset+live], current, self.raw_weights,
                    self.wk, self.rope, mlp_window=True, physical_rows=physical))
                jax.block_until_ready((inputs[0], inputs[1], uniform))
                self.phase_seconds["input_transfer"] += self.clock() - transferred
                dispatched = self.clock()
                result = self.prefill_compiled[physical](*inputs, uniform)
                jax.block_until_ready(result)
                self.phase_seconds["prefill_device"] += self.clock() - dispatched
                # Input state may be consumed/donated. Retain ONLY result.state;
                # never retry the pre-dispatch state even after a refusal.
                current = result.state
                del inputs
                end = offset + live
                final = end == policy.prompt_tokens
                def check_frontier():
                    if (not np.asarray(current.decoder.contract_valid).all()
                            or not np.array_equal(np.asarray(current.decoder.position), [end])
                            or not np.array_equal(np.asarray(current.decoder.context_lengths), [end + 1])
                            or int(np.asarray(current.prompt_length)) != policy.prompt_tokens
                            or bool(np.asarray(current.finished)) != final
                            or (not final and not np.array_equal(np.asarray(result.next_token), [-1]))):
                        raise ValueError("native prefill health/frontier drifted")
                self._phase(check_frontier)
            self.phase_seconds["prefill_request"] = self.clock() - prefill_started
            self._phase(lambda:self.authorize("prefill_done", self.resident_roots(), current))
            # No decode/observer/warmup may precede this actual first delivery.
            session.accept_prefill(result)
            return session
        except Exception:
            self.failed = True
            raise

    def close_request(self) -> None:
        """Release a terminal request's cache before the next fresh allocation."""
        if self.active is None:
            raise RuntimeError("runtime owns no request")
        if self.active.failed:
            self.failed = True
        self.active.release()
        self.active = None


def bind_admitted_runtime(*, repo: Any, mesh: Any, raw_config: Ws32DecoderConfig,
        decode_config: Ws32DecoderConfig, raw_weights: Any, decode_weights: Any,
        wk: tuple[Any, ...], exact_weights: Any, rope: Any,
        compiled: Mapping[str, Any], reports: Mapping[str, Any],
        local_slots: Mapping[int, int], process_index: int,
        preserve_memory: Callable[[str, Mapping[str, Any]], None],
        fleet_all: Callable[[bool], bool]) -> NativeBenchmarkRuntime:
    """Bind real HLO-checked handles and memory admission to the request loop.

    The protected loader must have finished/released preparation executables.
    No hidden memory bypass or automatic JIT path. The caller still holds both
    leases, owns the code/checkpoint/topology identity and preserves evidence.
    """
    from hashlib import sha256
    from scripts.greenfield import ws32_native_benchmark_memory as memory
    from scripts.greenfield import ws32_native_benchmark_programs as programs
    programs.require_source(repo)
    if set(compiled) != set(memory.ROLES) or set(reports) != set(memory.ROLES):
        raise ValueError('native runtime needs all six actual inspected resident programs')
    for role in memory.ROLES:
        report = reports[role]
        if (report.get('passed') is not True or report.get('graph') != role
                or report.get('profile') != 'ws32_native_sampled_request_v1'
                or report.get('raw_stablehlo_sha256') != programs.RAW[role][1]
                or report.get('raw_optimized_hlo_sha256') != sha256(compiled[role].as_text().encode()).hexdigest()):
            raise ValueError('native runtime compiled object differs from inspected graph')
    authorize = memory.RequestMemoryAdmission(compiled=compiled, devices=tuple(jax.local_devices()),
        local_slots=local_slots, process_index=process_index, preserve=preserve_memory)
    return NativeBenchmarkRuntime(mesh=mesh, raw_config=raw_config, decode_config=decode_config,
        raw_weights=raw_weights, decode_weights=decode_weights, wk=wk,
        exact_weights=exact_weights, rope=rope,
        prefill_compiled={128:compiled['prefill_chunk'],114:compiled['prefill_tail']},
        decode_compiled=compiled['decode'], cache_initializer=compiled['cache_init'],
        authorize=authorize, fleet_all=fleet_all)
