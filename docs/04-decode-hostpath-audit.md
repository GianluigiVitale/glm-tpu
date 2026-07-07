# 04 — Per-decode-step HOST-path audit (multi-host Ray serving, TPU v4)

**Scope.** Static audit (read-only, no TPU) of everything that happens on the HOST per decode
step for the GLM-5.2 Stage-1 stack: vLLM V1 engine (`~/vllm-build`, a30addc75) + tpu-inference
fork branch `glm-5.2-v4` (761ea755), 8 hosts × 4 v4 chips, TP=32 + EP, `TPU_MULTIHOST_BACKEND=ray`,
offline `vllm.LLM` (bench/engine.py). The FLOP/roofline side is a separate doc; this one owns the
question *"how many milliseconds per step does the host path cost, where, and what fixes it."*

Config anchor (the smoke-16 recipe actually run): `max_num_seqs=8`, `max_model_len` 4096–8192,
`max_num_batched_tokens=512` bucket floor via `TPU_MIN_TOKEN_BUCKET=512` (launcher), block_size 512,
greedy sampling, no spec decode, no structured output, no KV connector, dp_size=1 (pure TP — no DP
attention). Measured baseline: **~0.95 tok/s single-stream ≈ 1,053 ms/step** (RESEARCH_LOG
2026-07-07), vs DSV4's **61.2 ms/step** on the same pod.

---

## 0. Two facts that shape everything below

1. **The compiled Ray DAG is FORCED ON, channel type `shm`.** Not configurable:
   `tpu_inference/executors/ray_distributed_executor.py:96-107` sets
   `os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"`, `use_ray_compiled_dag = True`,
   `use_ray_spmd_worker = True`. There is **no per-step `worker.execute_method.remote("execute_model")`
   fan-out** on the forward path — the DAG channels replace it. The DAG is built lazily on the first
   step (`_execute_dag`, ray_distributed_executor.py:484 / vllm ray_executor.py:449) — the first
   decode step pays a one-time multi-second graph compile + actor channel setup.

2. **Async scheduling defaults ON for this stack.** `engine.py` doesn't set it; vLLM resolves
   `async_scheduling=None` → enabled because `RayDistributedExecutor.supports_async_scheduling()`
   returns True (ray_distributed_executor.py:89-94; vllm/config/vllm.py:954-994, log line
   "Asynchronous scheduling is enabled"). This selects `EngineCore.step_with_batch_queue`
   (batch-queue depth = `max_concurrent_batches` = **2**, vllm/config/vllm.py:499-508) and the
   TPU-specific async result protocol described in hops 5–6. Everything in this audit assumes
   async-on unless marked SYNC.

---

## 1. The step timeline — one decode step, hop by hop

Steady-state decode, N requests, bucket 512, greedy. "Exposed" = NOT hidden under TPU compute by
the depth-2 batch queue. Latency figures are order-of-magnitude estimates for this pod (same-host
zmq/shm ~0.05–0.2 ms; cross-host Ray object push / actor RPC ~0.5–3 ms) — the §5 trace confirms them.

| # | Hop | What crosses / runs | Size @ N=8 / 64 (measured) | Est. cost | Exposed? |
|---|-----|--------------------|---------------------------|-----------|----------|
| 0 | LLM proc ↔ EngineCoreProc (zmq, same host) | `EngineCoreOutputs` out, msgspec | ~0.6 / 3 KB | ~0.1 ms | No — front-end detokenizes in parallel with the next engine step (SyncMPClient, core busy loop) |
| 1 | `EngineCore.step_with_batch_queue` (driver proc, w-0) | `scheduler.schedule()` + `update_from_output()`, O(N) Python | — | 0.2–1 ms | **Yes** |
| 2 | `_execute_dag` → `forward_dag.execute((SchedulerOutput, GrammarOutput))` | pickle → shm channel (w-0 worker) + mutable-object push to 7 remote nodes (Ray 2.55 `shared_memory_channel`: one extra local reader relays object changes to each remote node) | **0.9 / 2.2 KB** pickled (measured; 6.7 KB @ 256) | 0.05 ms local; 0.5–2 ms remote, parallel | No — driver returns immediately (non_block) |
| 3 | Worker: `RayWorkerWrapper.execute_model_ray` (DAG bg thread) → `tpu_runner.execute_model` | deserialize + `update_states` + `_prepare_inputs` + jit dispatches (details §2) | H2D ≈ 5–10 KB total | **3–8 ms** Python+dispatch | No — overlaps step N−1's device compute (async dispatch, no sync point) |
| 4 | TPU compute (model_fn + compute_logits + sample) | — | — | 1,053 ms today (GLM, unoptimized); 61 ms DSV4 | (FLOP doc) |
| 5 | DAG output: each worker returns **`result_id: int`** (NOT the output) through its output channel; driver `ray.get(refs)` | 8 ints, remote pushes back to w-0 | bytes | 0.5–2 ms | **Yes** |
| 6 | `AsyncResultFuture.result()` fires a **second, classic actor-RPC round**: `execute_method("get_execute_model_output", id)` to ALL 8 workers (ray_distributed_executor.py:58-69); worker runs `AsyncTPUModelRunnerOutput.get_output()` → **`jax.device_get(next_tokens)` — THE per-step blocking sync** (tpu_runner.py:164-207, runner/utils.py:1021) → pickled `ModelRunnerOutput` back; driver `ray.get(ret_refs[0])` | **0.6 / 3 KB** (measured; 11 KB @ 256) | 1–3 ms RPC round-trip + residual compute wait | **Yes** |
| 7 | Driver: `update_from_output` → hop 0 | — | — | 0.2–1 ms | **Yes** |

**Exposed host floor per step (async mode, multi-host): ≈ 2–7 ms** (hops 1+5+6+7), provided hop 3
fully hides under compute. In SYNC mode (async scheduling off — e.g. required by continue_decode)
the whole worker prep + the device wait + full-output DAG return serialize: **≈ 8–20 ms/step** of
host time plus zero compute overlap.

Notes on the return path (hop 5–6):
- The full logits **never** leave the worker. What returns per step is: `result_id` (int) via the
  DAG, then `ModelRunnerOutput` (req-id strings + 1 sampled token per req + empty dicts) via the
  second RPC — ~78 B/req. Logprobs, if requested, are materialized worker-side
  (`_jax_logprobs_materialize`) and returned as lists — at max_logprobs=k that adds O(N·k·12 B).
- `next_tokens` is `jax.copy_to_host_async`'d at dispatch time (tpu_runner.py:1820), so the
  hop-6 `device_get` should complete ~immediately after the compute does — it is a *wait*, not a
  transfer.
- The `get_execute_model_output` RPC goes to all 8 workers every step (each pops its
  `_execute_model_outputs[result_id]` — required for cleanup); only rank 0's reply is awaited
  (no KV connector ⇒ `aggregator=None`).

---

## 2. Worker-side per-step host work in detail (hop 3)

`tpu_runner._execute_model` (tpu_runner.py:1227) → `_prepare_inputs` (:2349). All per-step,
scales with bucket/batch as noted:

**Python/numpy (O(N) + O(bucket)):**
- `persistent_batch_manager.update_states` — O(N) dict work.
- `_prepare_input_metadata` (:2122) — O(N) loops (dp_size=1 ⇒ `logits_indices_selector=None`).
- input_ids/positions gather: `np.repeat` + `np.take` over the 512-token bucket (:2427-2471) —
  tens of µs at bucket 512.
- `_prepare_async_token_substitution_indices` (:2197) — O(N); feeds the on-device placeholder
  substitution (async scheduling replaces last step's placeholder tokens ON TPU, not on host —
  `_substitute_placeholder_token`, donated jit :253).
- `_modify_prev_results` (:1121) — commits step N−1's tokens into `input_batch` host arrays;
  `device_get` on an already-async-copied array ⇒ cheap.

**Host→device uploads per step (the "tensor upload" inventory, bucket 512, N=8, greedy):**

| Upload | Contents | Bytes | Call |
|---|---|---|---|
| metadata blob (ONE `device_put`) | input_ids[512] + query_start_loc[9] + seq_lens[8] + logits_indices[8] + block_tables[8×16] | ~2.7 KB (int32) | `DeviceBuffer.build()` + `device_array` (:2624-2656) — deliberately packed to avoid N device_puts |
| positions | int32[512] | 2 KB | :2586 |
| request_distribution | int32[3] | 12 B | same call as blob |
| sampling metadata | greedy: `_cache_collision_dummy` int32[2] only; sampled: + temperature/top_p/top_k ×float32[padded_reqs] | 8 B / ~100 B | sampling_metadata.py:from_input_batch |
| async substitution indices | 3 int32 arrays ≤ bucket len | ~4 KB | `_apply_async_token_substitution` (:2288) |

Total ≈ **5–9 `device_put` dispatches, ~10 KB**. Bytes are irrelevant; the per-call dispatch
latency (~0.1–0.3 ms each, plus the `TPU_MULTIHOST_BACKEND=ray` path routing every one through
`jax.make_array_from_callback` per local shard — layers/common/utils.py:128-170) is what costs:
**~0.5–2 ms/step**. Block tables only get big at large max_model_len × max_num_seqs (e.g. 128 reqs
× 1024 blocks = 512 KB) — still one packed upload.

**Jit dispatch (per-step cache-lookup + arg-processing):**
- `model_fn(state_leaves, kv_caches, …)` + `compute_logits_fn(state_leaves, …)`: for the GLM path
  (`MODEL_IMPL_TYPE=vllm`) `state_leaves` **is the raw params dict**
  (model_loader.py:583-591 — "for the vllm-impl path … `state_leaves` is just the dict"), so
  every call re-flattens an O(10³)-leaf pytree and per-leaf checks shardings. Estimate
  **0.5–3 ms per call × 2 calls**; the flax_nnx path pre-flattens exactly to avoid this
  (model_loader.py:522, tpu_runner.py:897-901). Measure before fixing (§5).
- `unpack_arrays`' `jnp.split` (+ `_select_from_array_fn`, `_to_float32_fn`, `_rng_split_fn`,
  sample) — ~5–8 more small dispatches, all pre-compiled by `CompilationManager.capture_model`
  (compilation_manager.py:221-316, incl. `_precompile_input_unpack`, `_precompile_substitute_placeholder_token`).

**Blocking syncs on the worker (complete inventory for this config):**
- `jax.device_get(next_tokens)` in `host_extract_sampled_tokens` — async mode: inside
  `get_output()` at hop 6; SYNC mode: inline in `_sample_from_logits` (:1865), serializing the step.
- That's it, in steady decode. (`expert_indices` device_get only with
  `enable_return_routed_experts`; logprobs materialize only with logprobs>0; prompt-logprobs
  `_host` path only when requested — note its documented live-compile hazard, tpu_runner.py:165-168.)

---

## 3. Per-step recompile / cache-lookup hazards

- **Shape-keyed executable switching**: token-bucket flips (512↔1024 when a prefill joins) and
  `padded_num_reqs` changes select different precompiled programs — a cache *hit* if
  `capture_model` covered the combination, else a **serving live compile** (minutes at GLM scale).
  `VLLM_XLA_CHECK_RECOMPILATION` arms `ForbidCompile` (tpu_runner.py:489); `DSV4_OBSERVE_COMPILES=1`
  logs any HLO→executable compile per half-step *including* `get_output` (:1012-1037, the PR-10
  observability) — this is the tool that caught the worker-3 halt (an HLO→executable **layout**
  miss that ForbidCompile cannot see, ~/moe-tpu/docs/15).
- **Multi-host stagger**: any live compile staggers across the 8 hosts and can drop the launch
  group. Mitigation available in the launcher: `JAX_SHARE_BINARY_BETWEEN_HOSTS=1` (default off).
- `TPUSupportedSamplingMetadata._cache_collision_dummy` exists purely to keep logprobs/non-logprobs
  variants from colliding in the persistent cache — evidence this class of hazard is live.
- First step: one-time Ray DAG graph build + channel allocation (seconds) — exclude from timing.
- **Empty scheduler cycles**: a step with `total_num_scheduled_tokens == 0` still traverses hops
  1→7 end-to-end just to return `EMPTY_MODEL_RUNNER_OUTPUT` (tpu_runner.py:1234-1249 logs
  "Should not schedule a request that does nothing!"). DSV4 recorded **5,189** such cycles in one
  max_num_seqs=1 GSM8K run (~/moe-tpu RESEARCH_LOG 2026-06-16) — thousands of wasted multi-host
  round-trips. dp_size=1 on GLM should mostly avoid this, but count the warning in every run.

---

## 4. The DSV4 anchor (what the same pod already measured)

`~/moe-tpu/bench/dsv4_decode_throughput.py` + RESEARCH_LOG 2026-06-16/17:

- Steady-state 32-chip single-seq decode: **61.2 ms/step (16.3 tok/s), FLAT across context
  32→1280**, and Pallas attention kernel ON = OFF at **1.000×** — decode was "entirely comm/MoE-
  bound": MoE all-reduce + per-step overheads, with attention a negligible slice. Gen-bench
  steady state ~32 tok/s; small runs are XLA-compile-dominated (3.4 tok/s).
- Batched decode was **not** a quick win: max_num_seqs=16 → 13.9 tok/s aggregate vs 31.5
  single-seq (per-step latency + prefill serialized through the tiny mbt=32 chunk).
- Interpretation under this audit: of the 61 ms floor, the async-mode exposed host path (§1,
  2–7 ms) was ~3–11%; the rest is device time (MoE all-reduce + recompute). The flatness vs
  context is consistent with a device-side floor, not a host floor.
- GLM today (1,053 ms/step): host path is <1% — the current bottleneck is device-side (FLOP doc).
  **But** every Stage-2 device win rediscovers the host floor: at a DSV4-like 60–100 ms/step the
  2–7 ms exposed path is 3–10%, and in SYNC mode (8–20 ms) it is 10–30%.

---

## 5. What a 20-step profiler trace must confirm

Run `jax.profiler` on w-0 + one remote worker (PYTHON_TRACER_LEVEL≥1), 20 steady decode steps,
plus `DSV4_OBSERVE_COMPILES=1`:

1. **The bubble**: gap on the TPU timeline between step-N's last device op and step-N+1's first —
   this IS the exposed host floor. Expect 2–7 ms (async). If ≥10 ms, itemize against hops 1/5/6/7.
2. **0 backend compiles** after warmup, in all three observed regions (`execute_model`,
   `sample_tokens`, `get_output`) on ALL hosts — and all 20 steps on the same bucket (512) and
   padded_num_reqs (no executable flapping).
3. **Hop-6 wait**: `device_get(next_tokens)` inside `get_execute_model_output` should return
   ~instantly after compute end (copy_to_host_async worked). A ms-scale D2H there means the async
   copy isn't being scheduled.
4. **Hop-3 overlap**: worker-side `_prepare_inputs` + dispatches (expect 3–8 ms) must sit fully
   inside step N−1's device window. Count H2D transfers (~5–9 small) — anything large or numerous
   is a regression.
5. **Jit arg-processing**: time from `execute_model_ray` entry to the model_fn's first device op;
   if the params-dict flatten shows ≥3–5 ms/call, the pre-flatten fix (below) is justified.
6. **Ray hops**: driver-side timestamps around `forward_dag.execute`, `ray.get(refs)` and the
   `get_execute_model_output` round (expect ≤2 ms, 0.5–2 ms, 1–3 ms) — watch for a straggler among
   w-1..w-7. Zero "does nothing" warnings.

---

## 6. Top-3 host-side fixes (ranked by expected effect ÷ risk)

1. **Batched multi-step decode — `additional_config={"enable_continue_decode": true, "max_decode_steps": 10}`.**
   Already in the fork (tpu_runner.py:1416 `_execute_continue_decode` + decode_loop.py: an
   on-device `while_loop` with donated KV, EOS early-exit on device, ONE `device_get` per burst;
   scheduler monkeypatch in core/sched/utils.py handles multi-token accounting via
   num_lookahead_tokens). Amortizes the ENTIRE per-step host path (both Ray rounds + prep +
   dispatch) over up to 10 steps ⇒ effective host cost ~1–2 ms/step even in its mandatory SYNC
   mode. Constraints: requires `async_scheduling=False` (tpu_platform.py:380-382), decode-only
   batches (`is_decode_only`, :1253), no PP/pooling; logprobs/spec-decode paths bypass it. This is
   the right mode for the single-stream/small-batch benchmark runs (GSM8K, GPQA generation);
   precompile exists (`_precompile_continue_decode`).
2. **Protect the async pipeline you already have.** Confirm "Asynchronous scheduling is enabled"
   in every engine log (engine.py doesn't pin it; adding a spec config or an executor change can
   silently disable it — vllm/config/vllm.py:954-992 — turning 2–7 ms exposed into 8–20 ms), and
   drive the batch queue: with only one schedulable batch the depth-2 queue degenerates to sync
   timing. Grep every run for "Should not schedule a request that does nothing!" and treat a
   nonzero count as a bug (DSV4: 5,189 wasted multi-host rounds).
3. **Shave the exposed result path + per-step dispatch (upstreamable micro-fixes, measure first).**
   (a) The async protocol pays TWO sequential Ray rounds per step (DAG ints, then the
   `get_execute_model_output` actor-RPC round) — piggyback step-N's resolved output onto step-N+1's
   DAG output (or resolve `get_output()` inside `execute_model_ray` before returning, waiting
   worker-side instead of RPC-fetching) to remove 1–3 ms/step. (b) Pre-flatten the params dict once
   for the vllm-impl path exactly as flax_nnx does (model_loader.py:522 vs :591) to cut the
   per-call pytree walk from both `model_fn` and `compute_logits_fn`. (c) Fold `positions` into
   the DeviceBuffer blob and skip the greedy-mode sampling-dummy upload — 2 fewer device_puts.

---

*Method note: payload sizes measured by pickling representative decode-step `SchedulerOutput` /
`ModelRunnerOutput` objects with this tree's vllm on CPU (scratchpad `size_probe.py`); latencies
are order-of-magnitude estimates pending the §5 trace. No TPU code executed.*
