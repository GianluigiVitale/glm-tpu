# Goal — Test public TPU optimizations and measure GLM-5.2 ordinary versus MTP speed

Owner objective (2026-09-20): test the useful implementations found in the two
Kaggle model folders and original `vllm-project/tpu-inference`, adapt them to
GLM-5.2-FP8 on our existing 32 TPU v4 chips, and measure whether they improve
prefill and accepted decode throughput. Resolve the current speculative-output
mismatch and expensive verifier. Finish with real-weight ordinary versus native
MTP/speculative comparisons, answer checks, and an explicit keep/reject decision.

Native MTP is the drafter inside speculative decoding; they are not two
independent accelerators to stack. Keep GLM-5.2 and the existing hardware. The
`vikasclawd/GLM-5.3-Int4-Int8Mix` and `Tech2wild/GLM-5.3-Int4-Int8Mix`
repositories remain excluded. Do not resume cancelled benchmark or old queues.

This file defines the execution goal to submit or resume. Rewriting it does not
itself launch a TPU workload. When activated, proceed through implementation and
real measurements autonomously; do not stop at another literature review.

## Execution checklist

- [ ] Reconcile the recorded replay with its controller, receipts and workload
  lease before starting anything. Continue an existing run; never duplicate it.
- [ ] Isolate the ordinary/verifier mismatch on identical real-weight prefixes,
  then verify acceptance, rejection and cache restoration independently.
- [ ] Test the applicable Kaggle Qwen/GLM and original vLLM TPU candidates below.
  Record source and local code evidence for ideas already present or inapplicable.
- [ ] Measure each admitted change on TPU before combining winners; check trained
  correctness and memory again for the combined implementation.
- [ ] Compare ordinary, optimized ordinary, one-draft MTP and two-draft MTP on
  matched completed answers and sustained generation; tune wider drafts only
  after admission. Measure prefill separately.
- [ ] Publish the paired speed/correctness table, candidate decisions and receipts;
  clean all eight hosts and merge eligible work after the publication gates.

## Read first

- `AGENTS.md`, `HANDOFF.md`, `docs/release/STATUS.md`.
- `docs/perf/KAGGLE_MTP_REVIEW_20260920.md` — actual reference call paths,
  rollback differences, missing long-target parity and measured verifier cost.
- `docs/perf/UPSTREAM_MTP_REUSE_20260920.md` and its JSON source receipt —
  pinned public code, merged/open status and concrete adaptation boundaries.
- `docs/perf/MTP_STATE_CONTRACT_20260919.md`,
  `docs/perf/MTP_PROGRESS_20260919.md`, and
  `docs/perf/MTP_COMPLETION_AUDIT_20260920.md` — existing implementation,
  completed trials and preserved failures. Earlier completed-goal status describes
  those trials; this goal defines the new work when activated.

## Starting state and measured baseline

Worktree `/home/gianl/glm-tpu-perf-ref`, branch
`perf/reference-lowhanging-fruit-20260919`. Reference-review checkpoint:
`9c8b86cb`. Frozen `MODEL_SOURCE=edecdd94` remains unchanged. Native layer-78
acquisition, packing, loading, guarded drafting/acceptance and real TPU trials
are already complete; reuse verified assets and tests rather than repeating them.
Implementation checkpoint `f097649a`; representative execution `dc047933`.
Private main `c142d284` published evidence only, not the experimental engine.

Qualified ordinary research path: fixed D1 grouped experts, D8 BF16-resident
non-routed weights, D10 DSA, D4 packed loop, and D8/P1/P2 prefill. At 2,034
prompt tokens it measured **138.85 prompt tok/s and 14.04 wall decode tok/s**,
with DB610 29/29 agreement. Receipt:
`docs/perf/tpu-real-request-loop-20260919T192804Z.json`.

Completed representative suite, two repeats per case:

| Request | Ordinary wall tok/s | One native draft (R2) | Two native drafts (R3) |
|---|---:|---:|---:|
| Prose | 14.29–14.44 | 13.29–13.30 | 13.45–13.49 |
| Code/reasoning | 14.27–14.30 | 13.59–13.60 | 14.92–14.93 |
| Structured | 14.33–14.37 | 14.12–14.15 | 15.88–15.90 |

Receipt: `docs/perf/tpu-real-native-suite-20260920T031727Z.json`.
R2/R3 mean two/three target verification rows, including the pending token.
Long outputs diverge from ordinary: code at index 5, prose at 6, structured
at 816/823 for R2/R3. DB610 passes but does not establish long-output correctness.
Prose needs technical corrections; code exhausted 7,168 tokens without a final
answer; structured values are correct but unwanted Markdown fences fail format.
These measurements are implementation results, not an MTP hardware ceiling.

First structured R3: ~114.13–114.19 s verification out of 128.689 s decode wall,
2.924 accepted output tokens/round out of a maximum three, and ~4 s draft+refresh.
Target verification is the priority. Prior peak HBM was 28,228,678,144 bytes/chip
at capacity 8,192; every changed graph needs fresh admission. Five-row verification
previously failed its CPU numerical envelope. Decode D5 failed trained parity and
stays disabled. Bug-affected synthetic 72.1/64.3 ms results are not baselines.
All eight hosts were authenticated idle at the previous completion; recheck live
state before running. The recorded execution checkpoint follows; reconcile it
with current receipts before resuming. The first same-prefix replay
worker is implemented: R1/R2/R3 windows reset to ordinary state; every commit count
is compared, including restored future rows, residuals and per-layer cache
identity. Targeted CPU checks passed (55 diagnostic tests; 24 replay/input tests,
with overlap). The leased short code/prose TPU replay launched as
`perf_real_prefix_replay_20260920T090655Z`, immutable worker `c430276b`, after
all eight hosts passed authenticated idle checks. Controller/worker results are
under `/home/gianl/glm-run/` with that run tag; its last recorded status is pending,
not a completed result or a fresh live-status check. No new speed claim or
resolved trained mismatch is established. Per-layer activation/logit-margin
localization follows the initial cache/prediction replay if needed.
The standalone Kaggle global-max attention adaptation now passes CPU checks for
1/2/3/4/32 rows; maximum observed difference from frozen online softmax is
0.001953125 on the small synthetic fixture. Receipt:
`docs/perf/global-max-attention-cpu-20260920.json` and the separate R2 receipt.
A paired real-geometry TPU
primitive benchmark is prepared, not launched; trained admission and integration
remain pending. The existing selected-KV model path is unchanged.
Optional layer/head observations are implemented and CPU-tested for follow-up:
normalized inputs, hidden updates, carried residuals, selections and top-two
logit margins. They compare instrumented results against the original executables
before attributing a mismatch. The first running worker remains `c430276b`.
At the 09:49 UTC authenticated observation, code R1/R2/R3 and prose R1 were
complete on rank0; the run remained live. Code R2/R3 reproduce the first token
disagreement at index 5 from identical ordinary roots; R1 predictions agree.
These are provisional observations, with fleet aggregation and layer localization
still pending. Resume the same controller, not a duplicate workload.
An opt-in `--native-component-timing none` ablation is now CPU-checked: it removes
per-component profiling waits while retaining request synchronization, fleet
agreement and delivery. Its TPU speed comparison remains pending.

## Work, in order

### 1. Reproduce and isolate correctness on real weights

After targeted CPU checks and fleet admission, make the first TPU workload a
short same-prefix replay around the known code/prose mismatch. Start ordinary
and verifier from identical committed state, teacher-force identical tokens,
and compare R1/R2/R3 target predictions, top-logit margins, hidden states,
DSA selections and newly written cache spans. Locate the first differing layer.
Avoid another multi-thousand-token run before this diagnostic.

Use the Kaggle random/oracle-draft parity test structure: force full acceptance,
first rejection and rejection at each later position; check correction/bonus,
EOS, tail budgets, accepted frontier and rejected physical KV/index writes.
Check native draft hidden/logit behavior separately against the pinned trained
GLM-5.2 reference where executable. If reference execution is unavailable,
record that gap rather than claiming independent native parity.

Distinguish target arithmetic/batching differences from shifted inputs,
IndexShare state, acceptance and rollback bugs. A wrong draft should be rejected,
not change the target's greedy result. A numerical tolerance alone cannot qualify
a different greedy token as exact. Preserve diagnostics and resolve the mismatch
before promoting speculative execution; rejected variants may still have labelled
microbenchmarks. Share ordinary/verifier layer bodies where practical, following
the public shared-target-forward pattern, then repeat the same-prefix comparison.

### 2. Test each public reuse candidate against the existing implementation

Maintain a candidate table in the progress report with source pin, local change,
CPU result, TPU admission, measured latency, correctness and keep/reject reason.
Every row below needs a disposition. Adapt applicable ideas and measure them;
for an already implemented or incompatible item, show the code/test evidence.
Do not blindly cherry-pick runtime-specific patches or run irrelevant models.

| Candidate | Experiment and comparison |
|---|---|
| Kaggle GLM local attention | Adapt global score maximum plus FP32 numerator/denominator reduction and final normalization to our absorbed queries, BF16 cache and WS32 head layout. Retain multiple verification queries. Compare with selected-KV exchange and preserve the rejected D5 implementation as historical evidence. Cover empty owners, causal masks, selected-index ties, RoPE and reduction rounding. |
| TPU #3332 multi-query verification | Reuse explicit small decode-window sizing and page-crossing tests. Inspect our lowering for unnecessary padding or work proportional to prefill capacity. Our path does not use upstream RPA classification, so test the equivalent behavior rather than claiming a direct patch applies. |
| TPU #2533/#2535/#2610 and device rejection sampler | Compile the draft chain/input updates together, keep acceptance metadata on device where possible, and eliminate avoidable host materialization. Skip inactive padded work if introduced. Compare an unsplit request path with existing component-blocking timing; retain fleet agreement, failure handling and delivery guarantees. |
| TPU #3040/#3388 fused EP MoE | Test overlap of expert compute and output communication against current M8 packing. First resolve feature4×expert8 weight layout, scale format, v4 FP8 conversion/VMEM and token partitioning: the public `num_tokens // ep_size` scheme cannot directly serve 2/3 rows across eight ranks. Measure padding/adaptation cost. Evaluate admitted prefill tiles as well as small verifier windows; large-M published gains are not our decode gains. |
| TPU #3219/#3476 grouping and route indexing | Check whether occupancy-based work and division-based route indexing improve our small expert planner or prefill route path. Avoid duplicating optimizations already present; require a relevant microbenchmark before integration. |
| TPU #2324/#2248 v4 and token alignment | Audit actual FP8 conversion, tile-memory, FP32-score, causal-position and aligned-token reduction invariants. Our original-slot restoration already addresses the reported alignment pattern; add targeted regression coverage only where missing. These open GLM5.1 patches are references, not a validated replacement runtime. |
| Existing MTP fixes and sparse primitives | Verify selected-last-query IndexShare, single post-norm reuse and compact shard argmax against the reviewed vLLM fixes. Treat Qwen GDN rollback as an invariant/test reference only: GLM5.2 has no GDN. Our own TPU #3480 supplies primitives, not independent engine validation. Evaluate sparse-MLA reuse only with an explicit GLM cache/geometry adapter; DeepSeek-v4 packing is different. |

For each changed numerical path: meaningful CPU proof or documented error boundary,
then bounded TPU correctness/latency measurement at real geometry, then trained
same-prefix validation before integration. Preserve source attribution/licenses.
Measure individual changes before combining winners, including effects on ordinary
decode; a gain common to both paths is not a speculative-only gain. Use HLO and a
bounded trace to identify residual gathers, collectives, padding and host gaps.

### 3. Tune the admitted verifier and measure prefill separately

Compare one/two/three target rows first. Try three native drafts/four target rows
only after its own CPU, trained-prefix and memory gates. Revisit five rows only
if the earlier numerical failure is explained and fixed. Choose draft length
from measured accepted tokens per wall second, not acceptance rate alone.
Oracle/perfect-acceptance timings are diagnostic upper bounds, never served speed.
A cheap n-gram drafter may provide a control; do not acquire another large model.

For attention/MoE changes applicable to prefill, measure the existing ~2K DB610
case and one longer prompt that fits an admitted capacity (start with 8K; include
cache and output space in admission). Extend length only when the measured result
justifies it. Report prompt tok/s, warm TTFT, startup and MTP bootstrap separately.
Speculative decode gains alone do not establish a prefill improvement. Do not
restart the old 128K/256K campaigns or DB616–621 for this work.

### 4. Run paired real-answer speed comparisons

After short correctness gates, compare the preserved ordinary baseline, optimized
ordinary, optimized one-draft R2 and two-draft R3, plus any admitted winning draft
length. Reuse the private prose/code/structured suite and retained prompt hashes.
Keep prompt, capacity, token budget, greedy policy, delivery and warmup identical
within each pair; reset state and run at least two repeats with alternating order.
Report per-case results, not independent-chip trials or only the best prompt.

Use matched long-generation budgets to measure sustained speed. Also include a
bounded reasoning/code case with a checkable completed answer: tests for code,
exact values and parsing for structured output, and explicit reasoning checks.
If increasing a cap or changing a prompt to obtain completion, document a new
case and rerun all paired modes; preserve the original capped results. Token
agreement and answer correctness are separate checks. Unfinished reasoning is
not a correct final answer, and high acceptance is not a correctness proof.

Primary rate: accepted delivered output tokens / decode wall time, including
draft, verification, rejected work, cache refresh/commit, required host votes and
rank0 write/flush. Exclude and separately report cold load/compile, prefill and
network transport. State first-token treatment consistently. Record model-only
and component times separately, with profiling disabled for primary throughput.
Report acceptance by draft position, tokens/round, verifier latency, rejected work,
peak HBM, output agreement and completed-answer checks alongside wall tok/s.

### 5. Publish measured decisions and finish cleanly

Working success criterion: at least 25% higher wall decode throughput on the
representative suite, with per-case regressions disclosed and correctness gates
satisfied. 30+ tok/s remains an aspiration, not a promised result. Report prefill
improvements independently. If candidates lose or cannot meet correctness/memory
requirements, retain their evidence and keep the qualified ordinary path.

Update this State, `docs/perf/MTP_PROGRESS_20260919.md`, compact receipts and the
README comparison table as results land. Mark candidates tested, already present,
rejected with evidence, or unresolved; a source survey alone is not completion.
Finish with baseline/optimized ordinary/MTP speed and correctness, exact source
pins, limitations, authenticated eight-host cleanup and actual publication state.

The final comparison must have one row per prompt, mode and repeat, with:
prompt/output token counts, prefill tok/s, warm TTFT, accepted decode wall tok/s,
speedup against the paired ordinary baseline, draft acceptance, verifier latency,
peak HBM, ordinary-token agreement, EOS/budget termination, completed-answer
correctness, and the receipt/source pin. Keep failed or divergent trials visible
and separate from qualified winners. Include a short explanation of which
bottleneck changed and whether the measured gain justifies keeping each change.

## Authority and operating rules

When activated, TPU work is authorized on all 32 v4 chips of existing pod
`db-v4-64-od` (8 hosts × 4 chips), zone `us-central2-b`, through
`gcloud compute tpus tpu-vm ssh db-v4-64-od --worker=all`.
One workload at a time under `~/.glm-tpu-workload.lock` and the existing pod lease;
authenticate fleet idle first, respect sync/cron locks, and leave all eight hosts
clean. Disable automatic workload retries. Never create/delete/resize TPU, VM or
queued resources. Follow existing monitoring intervals except for diagnosis.

Work autonomously within scope. Keep experiments outside frozen source until
CPU, TPU and source/HLO/memory evidence supports promotion. All pytest uses
`JAX_PLATFORMS=cpu`; TPU experiments use explicit scripts. No environment upgrades
or full-model safety copies merely to try a candidate. Keep weights, credentials,
private prompts and raw DBs out of Git. Only `gs://driftbench-dsv4-uc`,
US-CENTRAL2, within storage bounds. Preserve originals, research branches,
rejected experiments, history and DB616–621 evidence.

Commit and push useful milestones to the private perf branch without asking;
no force-push, history rewriting or Co-Authored-By lines. At a useful reviewed
milestone, clean up and merge eligible work into private main after required
release checks, material review findings are resolved, and regional backup is
verified. Self-review is not independent review. Preserve rejected/unfinished
experiments on research branches. Say whether main receives implementation or
only evidence; publishing documentation does not deploy an engine.
