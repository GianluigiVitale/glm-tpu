"""G3: CPU32 execution goldens of the production composition (one subprocess).

Frozen fixture v1 on a 32-device CPU mesh, Pallas kernels in interpret mode, TPU-v4 chip info.
Everything is executed by production's own runtime code: the real ``OrdinaryRuntime.__init__``
and ``_load`` build and run the load (``driver.build_runtime``: checkpoint I/O, admission and TPU
compilation faked; the prefill and decode builders get the two interpret flags), and prompts A
and B run through the real ``OrdinaryRuntime.generate`` (votes identity, ``process_allgather``
faked, request validation relaxed for the fixture's 1,536-slot, 256-token profile), with the
prefill and decode programs ``_load`` compiled wrapped to record every block and step. Results
are positional leaf digests ``sha256(dtype | shape | raw bytes)`` in pytree-flatten order; key
paths are labels only.

Groups: fixture checkpoint leaves; WK decode (the compiled ``wk_decode`` over every full-index
layer) and promote (``runtime.wk``); resident BF16 weights (``runtime.weights``); host RoPE tables
(``runtime.rope`` at the fixture capacity, and 8,192 / 32,768 / 166,912); ``cache_init(157)``;
prompt A (157 tokens = B128 + B114 tail) state, token and health after each block; prompt B
(114 tokens, one block); prompt C (a refused prefill on B's finished state); the first three packed
decode steps of prompt A's session; the 8-token session (tokens + TokenEvent JSON lines);
``batch_cache_init`` and ``batch_insert`` of a real concurrent (n=4) runtime; kernel components.
``batch_decode`` is fingerprint-only (CPU cannot execute its vmapped BF16xBF16->F32 dot).

Run as ``python -m tools.equivalence.golden_run`` (prints one JSON line).
"""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from typing import Any

import numpy as np

from .common import canonical_json, emit, environment, require_cpu, sha256_hex, source_record, tree_record

PROMPT_A = [(i * 37 + 11) % 256 for i in range(157)]
PROMPT_B = [(i * 53 + 5) % 256 for i in range(114)]
EOS_IDS = (0,)
SESSION_TOKENS = 8
DECODE_STEPS = 3
ROPE_CAPACITIES = (8192, 32768, 166912)


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    return array.tolist()


def fixture_request(request_id: str, ids: list[int], max_new_tokens: int, config: Any) -> dict[str, Any]:
    """The request fields ``generate`` reads, on the fixture's profile (1,536 slots, vocabulary 256)."""
    value = dict(request_id=request_id, prompt_ids=list(ids), max_new_tokens=max_new_tokens,
                 context_capacity=config.context_capacity, vocab_size=config.geometry.vocab_size,
                 eos_ids=list(EOS_IDS))
    return dict(value, request_sha256=sha256_hex(canonical_json(value)))


def composition(mesh: Any) -> dict[str, Any]:
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    from glm_tpu.greenfield.runtime import ws32_decoder as dec
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import finish_ws32_batched_prefill

    from . import driver, fixture
    from .programs import fixture_arrays

    groups: dict[str, Any] = {}
    timings: dict[str, float] = {}

    def timed(name: str) -> None:
        timings[name] = round(time.perf_counter() - started[0], 1)
        started[0] = time.perf_counter()

    started = [time.perf_counter()]

    def put(value: Any) -> Any:
        return jax.device_put(value, NamedSharding(mesh, P()))

    frozen = fixture.fixture_v1(panel_geometry=True)
    groups["fixture_checkpoint"] = tree_record(frozen.arrays, labels=True)
    arrays = fixture_arrays(mesh, frozen, concrete=True)
    plans = driver.fixture_plans(arrays)
    timed("fixture")

    # --- load: the real OrdinaryRuntime.__init__ / _load
    built = driver.build_runtime(mesh, tier="fixture", capacity=fixture.CAPACITY, concurrent_size=0, arrays=arrays,
                                 plans=plans, concrete=True, fixture_geometry=frozen.config.geometry, interpret=True)
    runtime = built.runtime
    config = runtime.config
    raw = dec.bind_ws32_decoder_weights(arrays, config)  # what _load bound from the same loaded arrays
    wk_decode = built.recorder.program("wk_decode").fn
    decoded = [jax.block_until_ready(wk_decode(raw.layers[i].dsa.wk_bits_local, raw.layers[i].dsa.wk_scale_local))
               for i in config.full_index_slots]
    groups["wk_decode"] = tree_record(decoded)
    groups["wk_promote"] = tree_record(runtime.wk)
    groups["resident_bf16"] = tree_record(runtime.weights)
    tables = {str(config.context_capacity): np.asarray(runtime.rope)}
    for capacity in ROPE_CAPACITIES:
        tables[str(capacity)] = np.asarray(dec.build_ws32_main_rope_table(
            fixture.decoder_config(panel_geometry=True, capacity=capacity)))
    groups["rope_tables"] = tree_record(tables, labels=True)
    groups["cache_init_157"] = tree_record(jax.block_until_ready(runtime.initialize(runtime.put(np.int32(157)))))
    del raw, decoded
    timed("load")

    # --- generate: the real host loop over the programs _load compiled (recorded per call)
    prefill_programs = dict(runtime.prefill)
    calls: dict[str, list[Any]] = dict(prefill=[], decode=[])

    def recording(kind: str, fn: Any, rows: int | None = None) -> Any:
        def call(*args: Any) -> Any:
            result = fn(*args)
            calls[kind].append((rows, result))
            return result
        return call

    runtime.prefill = {rows: recording("prefill", fn, rows) for rows, fn in prefill_programs.items()}
    runtime.decode = recording("decode", runtime.decode)
    ticks = [0.0]

    def clock() -> float:
        ticks[0] += 1.0
        return ticks[0]

    def generate(name: str, ids: list[int], max_new_tokens: int) -> tuple[Any, ...]:
        for kind in calls:
            calls[kind].clear()
        events: list[Any] = []
        with driver.serving_fakes(relaxed_validation=True):
            tokens, _report = runtime.generate(fixture_request(name, ids, max_new_tokens, config),
                                               deliver=events.append, deadline=float("inf"), clock=clock)
        blocks = [dict(rows=rows, state=tree_record(result.state), next_token=_scalar(result.next_token),
                       health=_scalar(result.state.decoder.contract_valid), finished=_scalar(result.state.finished))
                  for rows, result in calls["prefill"]]
        return tokens, events, list(calls["prefill"]), list(calls["decode"]), blocks

    tokens, events, prefill_a, decode_a, blocks = generate("golden-a", PROMPT_A, SESSION_TOKENS)
    groups["prompt_a"] = dict(blocks=blocks)
    result_a = prefill_a[-1][1]
    groups["decode_steps"] = dict(first_token=_scalar(result_a.next_token), steps=[
        dict(metadata=_scalar(packed.metadata), state=tree_record(packed.decoded.state),
             next_token=_scalar(packed.decoded.next_token)) for _, packed in decode_a[:DECODE_STEPS]])
    lines = [json.dumps(asdict(event), sort_keys=True) for event in events]
    token_list = [int(t) for t in np.asarray(tokens).tolist()]
    groups["request_session"] = dict(tokens=token_list,
                                     token_sha256=sha256_hex(np.asarray(token_list, np.int32).tobytes()),
                                     jsonl_sha256=sha256_hex("".join(line + "\n" for line in lines)),
                                     finish_reason=events[-1].finish_reason, events=len(lines))
    timed("prompt_a")
    _, _, prefill_b, _, blocks = generate("golden-b", PROMPT_B, 1)
    groups["prompt_b"] = dict(blocks=blocks)
    result_b = prefill_b[-1][1]
    timed("prompt_b")

    # Prompt C: one more block on B's finished state must be refused without side effects.
    block = np.asarray(PROMPT_A[:114], np.int32)
    refused = jax.block_until_ready(prefill_programs[114](put(block), put(np.int32(114)), result_b.state,
                                                          runtime.weights, runtime.wk, runtime.rope))
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

    # Batch bank initializer and donated insertion of a real concurrent runtime (compile_batch in _load).
    batched = driver.build_runtime(mesh, tier="fixture", capacity=fixture.CAPACITY, concurrent_size=4, arrays=arrays,
                                   plans=plans, concrete=True, fixture_geometry=frozen.config.geometry,
                                   interpret=True).runtime
    bank = jax.block_until_ready(batched.initialize_batch(put(np.array([157, 114, 3, 1], np.int32))))
    groups["batch_cache_init"] = tree_record(bank)
    lanes = [finish_ws32_batched_prefill(result_a)[0], finish_ws32_batched_prefill(result_b)[0]]
    for index, lane_state in enumerate(lanes):
        bank = jax.block_until_ready(batched.insert_batch(bank, lane_state, put(np.int32(index * 2 + 1))))
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
