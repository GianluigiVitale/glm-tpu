"""G3: CPU32 execution goldens of the production composition (one subprocess).

Frozen fixture v1 on a 32-device CPU mesh, Pallas kernels in interpret mode, TPU-v4 chip info.
Everything production computes on device is executed with the production builders and options
(plus the two interpret flags) and recorded as positional leaf digests
``sha256(dtype | shape | raw bytes)`` in pytree-flatten order; key paths are labels only.

Groups: fixture checkpoint leaves; WK decode/promote outputs; resident BF16 weights; host RoPE
tables (fixture capacity, 8,192, 32,768, 166,912); ``cache_init(157)``; prompt A (157 tokens =
B128 + B114 tail) state, token and health after each block; prompt B (114 tokens, one block);
prompt C (a refused prefill on a finished state); three packed decode steps; an 8-token
``PackedRequestSession`` host loop with identity votes (tokens + TokenEvent JSON lines);
``batch_cache_init`` and ``batch_insert`` through the real ``compile_batch``; kernel components.
``batch_decode`` is fingerprint-only (CPU cannot execute its vmapped BF16xBF16->F32 dot).

Run as ``python -m tools.equivalence.golden_run`` (prints one JSON line).
"""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from .common import emit, environment, require_cpu, sha256_hex, source_record, tree_record

PROMPT_A = [(i * 37 + 11) % 256 for i in range(157)]
PROMPT_B = [(i * 53 + 5) % 256 for i in range(114)]
EOS_IDS = (0,)
SESSION_TOKENS = 8
DECODE_STEPS = 3
ROPE_CAPACITIES = (8192, 32768, 166912)


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.tolist()


def composition(mesh: Any) -> dict[str, Any]:
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from glm_tpu.greenfield.runtime import ws32_decoder as dec
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import finish_ws32_batched_prefill
    from glm_tpu.greenfield.runtime.ws32_request_session import RequestPolicy
    from glm_tpu.optimized.bf16_resident import bf16_resident_weights
    from glm_tpu.optimized.prefill_challenger import build_ws32_prefill_challenger_program
    from glm_tpu.optimized.request_loop import PackedRequestSession, build_packed_decoder_program
    from scripts.greenfield.ws32_compile_originals import build_wk_programs
    from scripts.greenfield.ws32_native_benchmark_programs import build_cache_initializer

    from . import fixture
    from .programs import PREFILL_OPTIONS

    interpret = dict(sparse_attention_interpret=True, linear_interpret=True)
    groups: dict[str, Any] = {}
    timings: dict[str, float] = {}

    def timed(name: str) -> None:
        timings[name] = round(time.perf_counter() - started[0], 1)
        started[0] = time.perf_counter()

    started = [time.perf_counter()]

    def put(value: Any) -> Any:
        return jax.device_put(value, NamedSharding(mesh, P()))

    frozen = fixture.fixture_v1(panel_geometry=True)
    config = frozen.config
    groups["fixture_checkpoint"] = tree_record(frozen.arrays, labels=True)
    raw = fixture.bind(mesh, frozen)
    timed("fixture")

    # --- load: WK decode/promote, resident BF16, RoPE, cache initializer (runtime.py _load)
    decode, promote = build_wk_programs(mesh, P(None, "feature"), P(None, "feature"), contract=config.dsa_contract)
    decoded, wk = [], []
    for layer_id in config.full_index_slots:
        d = raw.layers[layer_id].dsa
        completed = jax.block_until_ready(decode(d.wk_bits_local, d.wk_scale_local))
        decoded.append(completed)
        wk.append(jax.block_until_ready(promote(completed)))
    wk = tuple(wk)
    groups["wk_decode"] = tree_record(decoded)
    groups["wk_promote"] = tree_record(wk)
    weights = jax.block_until_ready(bf16_resident_weights(mesh, config, raw))
    groups["resident_bf16"] = tree_record(weights)
    rope_host = np.asarray(dec.build_ws32_main_rope_table(config))
    rope = put(rope_host)
    tables = {str(config.context_capacity): rope_host}
    for capacity in ROPE_CAPACITIES:
        tables[str(capacity)] = np.asarray(dec.build_ws32_main_rope_table(
            fixture.decoder_config(panel_geometry=True, capacity=capacity)))
    groups["rope_tables"] = tree_record(tables, labels=True)
    initialize = build_cache_initializer(mesh, config)
    groups["cache_init_157"] = tree_record(jax.block_until_ready(initialize(put(np.int32(157)))))
    timed("load")

    prefill = {rows: build_ws32_prefill_challenger_program(mesh, config, block_rows=rows, **PREFILL_OPTIONS,
                                                           **interpret).execute for rows in (128, 114)}

    def run_prompt(name: str, ids: list[int]) -> Any:
        """runtime.generate's prefill loop: 128-row blocks, a <=114-row tail on the B114 graph."""
        ids_array = np.asarray(ids, np.int32)
        fresh = jax.block_until_ready(initialize(put(np.int32(len(ids_array)))))
        blocks = []
        result = None
        for start in range(0, len(ids_array), 128):
            block = ids_array[start:start + 128]
            rows = 114 if len(block) <= 114 else 128
            result = jax.block_until_ready(prefill[rows](
                put(np.pad(block, (0, rows - len(block)), constant_values=-1)), put(np.int32(len(block))),
                fresh, weights, wk, rope))
            fresh = result.state
            blocks.append(dict(rows=rows, state=tree_record(result.state), next_token=_scalar(result.next_token),
                               health=_scalar(result.state.decoder.contract_valid),
                               finished=_scalar(result.state.finished)))
        groups[name] = dict(blocks=blocks)
        return result

    result_a = run_prompt("prompt_a", PROMPT_A)
    timed("prompt_a")
    result_b = run_prompt("prompt_b", PROMPT_B)
    timed("prompt_b")

    # Prompt C: one more block on B's finished state must be refused without side effects.
    block = np.asarray(PROMPT_A[:114], np.int32)
    refused = jax.block_until_ready(prefill[114](put(block), put(np.int32(114)), result_b.state, weights, wk, rope))
    before, after = result_b.state, refused.state
    groups["prompt_c_refused"] = dict(
        state=tree_record(after), next_token=_scalar(refused.next_token),
        health=_scalar(after.decoder.contract_valid),
        caches_unchanged=bool(np.array_equal(np.asarray(before.decoder.kv_cache_local),
                                             np.asarray(after.decoder.kv_cache_local))
                              and np.array_equal(np.asarray(before.decoder.index_cache_local),
                                                 np.asarray(after.decoder.index_cache_local))),
        frontier_unchanged=bool(np.array_equal(np.asarray(before.decoder.position), np.asarray(after.decoder.position))
                                and np.array_equal(np.asarray(before.decoder.context_lengths),
                                                   np.asarray(after.decoder.context_lengths))))
    timed("prompt_c")

    # Three packed decode steps from prompt A (what PackedRequestSession.step dispatches).
    decode_fn = build_packed_decoder_program(mesh, config, **interpret).execute
    state, token = finish_ws32_batched_prefill(result_a)
    steps = []
    for _ in range(DECODE_STEPS):
        packed = jax.block_until_ready(decode_fn(token, state, weights, rope))
        steps.append(dict(metadata=_scalar(packed.metadata), state=tree_record(packed.decoded.state),
                          next_token=_scalar(packed.decoded.next_token)))
        state, token = packed.decoded.state, packed.decoded.next_token
    groups["decode_steps"] = dict(first_token=_scalar(result_a.next_token), steps=steps)
    timed("decode_steps")

    # The host request loop with identity votes (runtime.generate's session part).
    policy = RequestPolicy("golden-a", 0, len(PROMPT_A), SESSION_TOKENS, config.context_capacity,
                           config.geometry.vocab_size, EOS_IDS)
    events = []
    ticks = [0.0]

    def clock() -> float:
        ticks[0] += 1.0
        return ticks[0]

    session = PackedRequestSession(policy, decode_step=lambda t, s: decode_fn(t, s, weights, rope),
                                   replicate_uniform=put, fleet_all=lambda valid: bool(valid), deliver=events.append,
                                   request_started=0.0, delivery_boundary="golden list append", clock=clock)
    session.accept_prefill(result_a)
    while not session.finished:
        session.step()
    lines = [json.dumps(asdict(event), sort_keys=True) for event in session.events]
    tokens = [event.token_id for event in session.events]
    groups["request_session"] = dict(tokens=tokens, token_sha256=sha256_hex(np.asarray(tokens, np.int32).tobytes()),
                                     jsonl_sha256=sha256_hex("".join(line + "\n" for line in lines)),
                                     finish_reason=session.events[-1].finish_reason, events=len(lines))
    session.release()
    timed("request_session")

    # Batch bank initializer and donated insertion through the real compile_batch.
    from glm_tpu.optimized.batched_runtime import compile_batch

    runtime = SimpleNamespace(concurrent_size=4, mesh=mesh, config=config, put=put, weights=weights, rope=rope,
                              compile=lambda name, fn, values, **kwargs: fn)
    compile_batch(runtime, jax.block_until_ready(initialize(put(np.int32(157)))))
    bank = jax.block_until_ready(runtime.initialize_batch(put(np.array([157, 114, 3, 1], np.int32))))
    groups["batch_cache_init"] = tree_record(bank)
    lanes = [finish_ws32_batched_prefill(result_a)[0], finish_ws32_batched_prefill(result_b)[0]]
    for index, lane_state in enumerate(lanes):
        bank = jax.block_until_ready(runtime.insert_batch(bank, lane_state, put(np.int32(index * 2 + 1))))
    groups["batch_insert"] = tree_record(bank)
    timed("batch")
    return dict(groups=groups, timings=timings)


def components(mesh: Any) -> dict[str, Any]:
    """Direct kernel/component calls on deterministic random inputs."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P

    from glm_tpu.optimized.bf16_resident import decode_fp8_table

    rng = np.random.default_rng(2026)
    out: dict[str, Any] = {}
    bits = rng.integers(0, 256, size=(256, 384), dtype=np.uint8)
    scale = rng.uniform(0.25, 2.0, size=(2, 3)).astype(np.float32)
    out["decode_fp8_table"] = tree_record(jax.jit(decode_fp8_table)(jnp.asarray(bits), jnp.asarray(scale)))

    # two-stage DSA top-k: ties, skew and a forced full-width fallback (8 owners).
    from glm_tpu.greenfield.kernels.reference.dsa import ScoredSelectedPositions
    from glm_tpu.optimized.dsa_candidates import two_stage_topk_mapped

    sub = Mesh(np.asarray(jax.devices()[:8], object), ("expert",))

    def body(scores: Any, positions: Any, lengths: Any) -> Any:
        return two_stage_topk_mapped(scores[0], positions[0], lengths, top_k=16, global_context_size=256,
                                     candidates_per_owner=4)

    fn = jax.jit(jax.shard_map(body, mesh=sub, in_specs=(P("expert"), P("expert"), P()),
                               out_specs=(ScoredSelectedPositions(P(), P(), P()), P()), check_vma=False))
    # The draw sequence of tests/release/test_optimized_dsa_candidates.py (seed 334, 60 trials).
    topk_rng = np.random.default_rng(334)
    positions = np.arange(256, dtype=np.int32).reshape(8, 32)
    digests, fallbacks = [], []
    for trial in range(60):
        perm = np.stack([topk_rng.permutation(32) for _ in range(8)])
        p = np.take_along_axis(positions, perm, axis=1)
        scores = topk_rng.integers(-8, 9, size=(8, 3, 32)).astype(np.float32)
        if trial % 3 == 0:
            scores[0] += 100
        if trial % 3 == 1:
            scores.fill(0)
        lengths = np.array([0, 7 if trial % 2 else 64, 256], np.int32)
        if trial % 3 == 2:
            lengths = np.array([0, 256, 256], np.int32)
        if trial == 59:  # only the omitted +0 on owner 7 forces the full-width fallback
            p = np.arange(256, dtype=np.int32).reshape(32, 8).T
            scores.fill(-0.0)
            scores[7, :, :20] = 0.0
            lengths[:] = 256
        selected, fallback = fn(jnp.asarray(scores), jnp.asarray(p), jnp.asarray(lengths))
        digests.append(tree_record(selected)["digest"])
        fallbacks.append(bool(fallback))
    cases = dict(trials=60, digest=sha256_hex("\n".join(digests)), fallbacks=fallbacks)
    out["two_stage_topk"] = cases
    del mesh
    return out


def main() -> int:
    from . import fixture, lowering

    require_cpu()
    mesh = fixture.cpu_mesh()
    started = time.perf_counter()
    with lowering.tpu_v4_info():
        result = composition(mesh)
        result["components"] = components(mesh)
    result.update(environment=environment(), source=source_record(), seconds=round(time.perf_counter() - started, 1))
    result["digest"] = sha256_hex(json.dumps(dict(groups=result["groups"], components=result["components"]),
                                             sort_keys=True, separators=(",", ":")))
    emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
