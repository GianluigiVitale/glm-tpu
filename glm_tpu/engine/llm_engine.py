"""The engine: generation over a loaded model runner (``glm_tpu.runner.tpu_runner.TPUModelRunner``).

``LLMEngine.generate`` serves one fresh request at a time; ``LLMEngine.generate_concurrent`` serves one concurrent
batch of conversations sharing the weights. The runner loads, compiles and admits the programs; the engine drives
them, votes with the fleet through the runner and owns the poison flag of the loaded model.
"""

import gc
from hashlib import sha256
import time

import jax
import numpy as np

from glm_tpu.engine.request import batch, validate
from glm_tpu.engine.request_session import BatchedSession, RequestPolicy, RequestSession
from glm_tpu.models.glm_moe_dsa import state as pre


class LLMEngine:
    """Generation over one loaded ``TPUModelRunner`` (``runner``): its compiled programs, fleet votes, phases and
    runtime record. ``active`` is the poison flag: set when a request starts and cleared only when it completes, so
    an ambiguous delivery or a fleet failure leaves the engine refusing every later request."""

    def __init__(self, runner):
        self.runner = runner
        self.active = False

    def generate(self, request, *, deliver, deadline, clock=time.perf_counter):
        """One fresh request; ambiguous delivery or fleet failure poisons this runtime."""
        validate(request)
        if self.runner.concurrent_size:
            raise RuntimeError("use generate_concurrent on a batched runtime")
        self.runner.require(
            request["context_capacity"] == self.runner.capacity, "request capacity differs from loaded model"
        )
        if self.active:
            raise RuntimeError("optimized runtime has an active or failed request")
        self.active = True
        ids = np.asarray(request["prompt_ids"], np.int32)
        policy = RequestPolicy(
            request["request_id"],
            len(ids),
            request["max_new_tokens"],
            self.runner.capacity,
            request["vocab_size"],
            tuple(request["eos_ids"]),
        )

        def budget():
            self.runner.require(self.runner.vote(clock() < deadline) is True, "optimized request deadline expired")

        for name in ("cache_init", "prefill_128", "prefill_114", "decode"):
            self.runner.admit(name)
        fresh = jax.block_until_ready(self.runner.initialize(self.runner.put(np.int32(len(ids)))))
        staged = []
        for start in range(0, len(ids), 128):
            block = ids[start : start + 128]
            rows = 114 if len(block) <= 114 else 128
            staged.append(
                (
                    self.runner.put(np.pad(block, (0, rows - len(block)), constant_values=-1)),
                    self.runner.put(np.int32(len(block))),
                )
            )
        budget()
        started = clock()
        for _i, (block, count) in enumerate(staged):
            budget()
            result = jax.block_until_ready(
                self.runner.prefill[block.size](
                    block, count, fresh, self.runner.weights, self.runner.wk, self.runner.rope
                )
            )
            fresh = result.state
            self.runner.require(
                self.runner.vote(bool(np.asarray(fresh.decoder.contract_valid).all())) is True,
                "optimized prefill failed",
            )
        prefill_seconds = clock() - started
        session = RequestSession(
            policy,
            decode_step=lambda t, s: self.runner.decode(t, s, self.runner.weights, self.runner.rope),
            fleet_all=lambda valid: self.runner.vote(valid and clock() < deadline),
            deliver=deliver,
            request_started=started,
            delivery_boundary="rank0 private JSONL token write+flush; no network transport",
            clock=clock,
        )
        session.accept_prefill(result)
        # Release prefill references before entering sustained decode.
        del fresh, result
        decode_started = clock()
        while not session.finished:
            session.step()
        elapsed = clock() - decode_started
        tokens = np.asarray([event.token_id for event in session.events], np.int32)
        # Deadline admission participates in the existing fleet votes. A local
        # pre-dispatch exception would strand peers in the next collective.
        from jax.experimental import multihost_utils

        token_digest = sha256(tokens.tobytes()).hexdigest()

        def agree_tokens():
            hashes = np.asarray(multihost_utils.process_allgather(np.frombuffer(bytes.fromhex(token_digest), np.uint8)))
            self.runner.require(bool((hashes == hashes[0]).all()), "optimized output differs across hosts")

        self.runner.phase("output_consensus", agree_tokens)
        report = dict(
            request_sha256=request["request_sha256"],
            prompt_tokens=len(ids),
            emitted=len(tokens),
            timed_decode_tokens=len(tokens) - 1,
            finish_reason=session.events[-1].finish_reason,
            prefill_seconds=prefill_seconds,
            prefill_tokens_per_second=len(ids) / prefill_seconds,
            decode_wall_seconds=elapsed,
            decode_tokens_per_second=(len(tokens) - 1) / elapsed if len(tokens) > 1 else None,
            ttft_seconds=session.ttft_seconds,
            token_sha256=token_digest,
            peak_memory=self.runner.stats(),
            sampling="greedy",
            speculative=False,
        )
        session.release()
        self.active = False
        self.runner.record["requests"].append(report)
        self.runner.save(self.runner.record)
        return tokens, report

    def generate_concurrent(self, requests, *, deliver, deadline, clock=time.perf_counter):
        payload = batch(requests, concurrent=True)
        self.runner.require(
            len(requests) == self.runner.concurrent_size and payload["context_capacity"] == self.runner.capacity,
            "batch differs from compiled conversation count/capacity",
        )
        if self.active:
            raise RuntimeError("optimized runtime has an active or failed request")
        self.active = True
        started = clock()

        def budget():
            self.runner.require(self.runner.vote(clock() < deadline) is True, "concurrent deadline expired")

        budget()
        self.runner.admit("batch_cache_init")
        lengths = np.asarray([len(v["prompt_ids"]) for v in requests], np.int32)
        state = jax.block_until_ready(self.runner.initialize_batch(self.runner.put(lengths)))
        first_tokens = []
        metadata = []
        prefill_times = []
        for lane, item in enumerate(requests):
            budget()
            self.runner.admit("cache_init")
            fresh = jax.block_until_ready(self.runner.initialize(self.runner.put(lengths[lane])))
            # Admit with the batch bank AND this prefill cache resident.
            for name in ("prefill_128", "prefill_114", "batch_insert"):
                self.runner.admit(name)
            ids = np.asarray(item["prompt_ids"], np.int32)
            prefill_started = clock()
            for start in range(0, len(ids), 128):
                budget()
                part = ids[start : start + 128]
                rows = 114 if len(part) <= 114 else 128
                out = jax.block_until_ready(
                    self.runner.prefill[rows](
                        self.runner.put(np.pad(part, (0, rows - len(part)), constant_values=-1)),
                        self.runner.put(np.int32(len(part))),
                        fresh,
                        self.runner.weights,
                        self.runner.wk,
                        self.runner.rope,
                    )
                )
                fresh = out.state
                self.runner.require(
                    self.runner.vote(bool(np.asarray(fresh.decoder.contract_valid).all())) is True,
                    "concurrent prefill failed",
                )
            one, token = self.runner.phase("batch_prefill_finish", lambda: pre.finish_batched_prefill(out))  # noqa: B023, F821 (phase() calls this closure in this iteration, before the del below; pyflakes checks it after the del)
            prefill_times.append(clock() - prefill_started)
            first_tokens.append(int(np.asarray(token)[0]))
            metadata.append([first_tokens[-1], 1, int(lengths[lane]), int(lengths[lane]) + 1])
            state = jax.block_until_ready(self.runner.insert_batch(state, one, self.runner.put(np.int32(lane))))
            # Only one disposable prefill state, never eight individual cache copies.
            del fresh, out, one, token
            gc.collect()
        self.runner.admit("batch_decode")
        budget()
        session = BatchedSession(
            requests,
            decode=lambda t, s, a: self.runner.decode_batch(t, s, self.runner.weights, self.runner.rope, a),
            put=self.runner.put,
            vote=self.runner.vote,
            deliver=deliver,
            deadline=deadline,
            clock=clock,
        )
        session.run(state, self.runner.put(np.asarray(first_tokens, np.int32)[:, None]), np.asarray(metadata, np.int32))
        del state
        gc.collect()
        from jax.experimental import multihost_utils

        results = []
        for lane, item in enumerate(requests):
            tokens = np.asarray([e.token_id for e in session.events[lane]], np.int32)
            digest = sha256(tokens.tobytes()).hexdigest()
            hashes = np.asarray(multihost_utils.process_allgather(np.frombuffer(bytes.fromhex(digest), np.uint8)))
            self.runner.require(bool((hashes == hashes[0]).all()), "concurrent output differs across hosts")
            elapsed = max(0.0, session.finished_at[lane] - session.decode_started)
            report = dict(
                request_sha256=item["request_sha256"],
                prompt_tokens=int(lengths[lane]),
                emitted=len(tokens),
                timed_decode_tokens=len(tokens) - 1,
                finish_reason=session.events[lane][-1].finish_reason,
                prefill_seconds=prefill_times[lane],
                prefill_tokens_per_second=lengths[lane] / prefill_times[lane],
                decode_wall_seconds=elapsed,
                decode_tokens_per_second=(len(tokens) - 1) / elapsed if elapsed > 0 else None,
                token_sha256=digest,
                peak_memory=self.runner.stats(),
                sampling="greedy",
                speculative=False,
                batch_size=len(requests),
                batch_rounds=session.round,
                context_capacity=self.runner.capacity,
            )
            results.append((tokens, report))
        aggregate = dict(
            batch_size=len(requests),
            decode_rounds=session.round,
            prefill_seconds=sum(prefill_times),
            decode_wall_seconds=session.decode_seconds,
            aggregate_decode_tokens_per_second=sum(len(t) - 1 for t, _ in results) / session.decode_seconds
            if session.decode_seconds > 0
            else None,
            wall_seconds=clock() - started,
        )
        self.runner.record.setdefault("batches", []).append(aggregate)
        self.runner.record["requests"].extend(report for _, report in results)
        self.runner.save(self.runner.record)
        self.active = False
        return results, aggregate
