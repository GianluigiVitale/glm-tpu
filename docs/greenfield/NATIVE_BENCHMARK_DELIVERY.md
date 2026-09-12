# Native benchmark/request delivery — implementation and remaining boundary

Status: 2026-09-12. Long workloads COMPLETE DB616–620; do not repeat them.
No official-card score, TPU sampled-request result or delivered TPU TTFT yet.

Controller preflight read both exact pinned dataset payloads successfully:
GPQA198 (1,373,492B CSV), AIME30 (10,065B parquet), all card-mode items built
without importing a legacy engine. Only hashes/counts were retained, not raw
questions/answers or model outputs. Receipt:
`../artifacts/native-benchmark-dataset-access-20260912.json`. Ordered-item hash
uses the pure builder BEFORE `load_items` adds dataset metadata. This is access
and item-identity evidence, not completed campaign registration or a score.

## Implemented

- `glm_tpu/greenfield/runtime/ws32_sampled_request.py` builds sampled batched
  prefill and one-row decode/observer programs from the EXISTING model bodies.
  Exact-DSA and host rotary inputs follow the original config. The final head
  receives the original split residual, not a re-normalized rounded output.
  Original option validation and public greedy builders/defaults remain.
- Append one replicated FP32 uniform to each original program's arguments.
  It is runtime data: request seeds/token positions do not trigger compilation.
  Nonfinal prompt blocks do not execute a sampling head or consume an RNG draw.
- `glm_tpu/greenfield/runtime/ws32_request_session.py` controls the generated
  frontier. `next_uniform()` supplies draw0 for final prefill, then draw1 for
  the first decode. `accept_prefill()` delivers token0 immediately; `step()`
  feeds the previous generated token to one completed native decode invocation.
  EOS wins over a simultaneous length limit; all generated IDs, including EOS,
  are retained. The full registered cap must fit capacity, never silently cut.
- A live session retains the SAME cache object across pauses. This proves an
  in-memory continuation API only, NOT durable/process-crash KV restoration.
  A failed dispatch/delivery makes the session terminal: it never retries
  possibly consumed buffers or a token that might already have reached a sink.
- The worker supplies its all-host vote and actual bounded delivery callback.
  Health/frontier checks precede emission; delivery completion is acknowledged.
  TTFT is sink completion minus caller's request-arrival timestamp, not device
  prefill-ready time. Only the output rank's named boundary is a delivery metric.
  Device-step wall excludes host vote/output; total delivered-request wall does
  not. Cold load/compile and input/cache/prefill phase timing remain worker work.

## Tests and limits

Commands from the repository root, always `JAX_PLATFORMS=cpu`:

```bash
JAX_PLATFORMS=cpu /home/gianl/vllm-env/bin/python -m pytest -q \
  tests/greenfield/runtime/test_ws32_sampled_request.py \
  tests/greenfield/runtime/test_ws32_request_session.py \
  tests/greenfield/kernels/test_ws32_sampling.py \
  tests/greenfield/runtime/test_ws32_decoder.py
```

Recorded runs (overlapping): initial sampled chain/kernel/decoder35PASS163.62s;
session plus real model integration10PASS149.52s; final session controls and
four optional-input ABI combinations12PASS9.61s. Final self-review tightened
fleet vote types and retained original failure causes; the last batch tests
the resulting control path. The real chain uses eight nonzero-weight CPU
layers with a uniform output head for independently predictable token choices.
It tests real cache/IndexShare/prefill/decode computation and confirms that
changing only the draw changes tokens without changing same-input state.
Compiled CPU collective groups are local4/8. Sampler tests independently check
probabilities/ties/CDF and the original final norm/logits boundary.

Exact-DSA optional input routing has a fake-math ABI test; it is not exact-alias
TPU validation. CPU timings are NOT model throughput, quality or physical HBM.
The original real TPU results remain DB616–620, at their own recorded pins.

## Next: one protected native benchmark worker, not another engine

1. Reuse the current final-layout loader, raw prefill/overlay decode views,
   phase residency and both-lease outer ownership/cleanup workflow. Bind the
   sampled builders/session; precompile before warm TTFT. Do not load/repack a
   second checkpoint or call `bench/engine.py`/`run_bench.make_generate`.
2. Both new output seams change source bytes. Historical source recipes and
   HLO hashes deliberately remain strict. Register the sampled path explicitly
   and inspect its actual HLO/all-live memory before running model requests;
   do not widen old hashes, claim old sampled evidence, or reacquire long runs.
   Long-capacity ownership/donation and phase loading must stay explicit.
3. Reuse pure `bench/benchmarks.py`, `extract.py`, `provenance.py` and retained
   tokenizer/template/EOS metadata. Do not import the legacy execution module
   indirectly through `bench/run_bench.py`. GPQA198 and AIME30 full item sets,
   revisions, prompts, samples, seeds, stopping and budget must be registered
   before outputs. Model card pin/targets: `HF_GLM52_CARD_PIN.md`.
4. Card generation: temperature1.0/top_p0.95/max163840. GPQA formatting is
   unspecified; disclose the existing harness choice. AIME requires its exact
   card system prompt and GPT-5.5 medium judge; paid approval remains unresolved.
   Do not quietly substitute deterministic extraction or claim matched parity.
   Worst-case228×163840 tokens at6.15tok/s is about70days before prefill; real
   average length is unmeasured. Register a realistic campaign stop/resume budget
   and report incomplete coverage rather than dropping slow/missed items.
5. Connect primary DB and append-only raw outputs, actual output delivery,
   request isolation/live resume and explicitly scoped durable recovery, then
   report card gaps/uncertainty/caveats, archive regionally and prove8/8clean.

Fresh post-DB620 observation: all8workers absent/no libtpu holders. Local free
space2.257GB is below the existing6GiB launch floor; before any protected run,
use only verified recoverable local-copy eviction and recheck storage. No
cloud objects, model payloads or infrastructure were changed by this work.
