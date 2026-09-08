# B128 layer6 refusal — original evidence and smallest next discriminator

Status: 2026-09-08. Numerical admission FAILED, not a speed result. Historical
Gate D/DB567 and batched2K/DB588 remain sealed. Larger-window prefill, its own
competitive8K and efficient L7/L8 remain open. This note grants no TPU launch.

## Exact failed run

- Tag: `greenfield_fp8_ws32_prefill_layer_window_numerical_l6_20260908T132240925858058Z`.
- Executed pin: `0f994e373dade76ede59da179da6856d17234e5d`.
- All4 actual graph profiles passed; completed WK decode/promote, one candidate
  B128 and four B32 controls. Seven executable calls total; five model calls.
- Failed at `boundary/comparison`: `window/control ordered DSA selection differs`.
  No competitive or tail case ran. No timing/performance campaign ran; no DB row
  or SUCCESS exists for this failed numerical trial.
- Wrapper terminated exit1 at13:25:07Z. Its normal census and root device/libtpu
  census each confirm eight distinct clean hosts. No model retry was launched.
- Originals: `gs://driftbench-dsv4-uc/results/<tag>/workers/rank0/` through
  `rank7/`,16 files per rank plus each `worker_receipts.json`.

The read-only `scripts/greenfield/analyze_prefill_window_failure.py` downloads
the exact recorded generations, verifies sizes/CRC32C/SHA256, original fixture,
four graph identities across hosts, selected checkpoint tensor hashes, eight
unique processes and actual32-device/slot mapping. It replays original arrays
without changing the admission comparator. No checkpoint payload is copied.

Receipt: `docs/artifacts/prefill-window-boundary-refusal-v2-20260908.json`.
The earlier `prefill-window-boundary-refusal-20260908.json` remains disclosed:
it had not revalidated physical device-to-slot assignment. Independent review
caught that omission; v2 supplies the missing mapping proof and reproduces the
same numerical findings. Neither receipt promotes the failed trial.

## What the arrays establish

All32 owners agree on the replicated observations; same-feature expert replicas
are byte-identical. Four feature shards remain distinct where appropriate.

| Observation | Original B128 versus fourB32 |
|---|---|
| DSA order changes | rows2,52,69,82,101,121 |
| DSA selected-set changes | none; this boundary has only506–633 causal keys, below2048 |
| Own selected-score order/ties/count/padding | pass on both paths; not a competitive full-score-row proof |
| Ordered router changes | 15rows |
| Router selected-set changes | rows2,10,32,62,113 |
| Pre-attention normalized input | bit-identical |
| KV | written differences at slots0–3; within unchanged bounds |
| Unrepaired index | written differences at slots0–7; within unchanged bounds |
| Repaired index / untouched cache bytes | bit-identical on all32 owners |
| Output/residual | unchanged aggregate and every-row bounds pass |
| Maximum output/residual absolute difference | 0.001956939697265625 / 0.000244140625 |
| Route weights, raw slotwise | fail; maximum0.009262464940547943 |
| Route weights aligned by expert ID on123 matching-set rows | bounds pass, maximum0.00008444488048553467 |

The last comparison is diagnostic ONLY: it excludes the five changed-set rows
and cannot turn the exact ordered-router failure into a pass. Set changes:
row2 expert144 versus208; row10 206 versus200; row32 39 versus234;
row62 220 versus239; row113 250 versus200 (candidate versus control).

## What is not established

This is NOT proved to be harmless rounding, a faulty router, a faulty selector,
or hardware bypass of a BF16 conversion. Matching pre-attention normalization
is not matching router input. Matching selected sets below2048 is weak evidence:
all valid keys fit, while their changed order can affect attention accumulation.
Most changed-route rows do not coincide with changed DSA-order rows; that alone
does not locate the cause because attention/query and normalization arithmetic
can also change without a selected-set change.

Maximum written KV difference is3.725290298461914e-9; unrepaired index difference
is0.001953125. These changes are upstream of routing and MUST be included in
localization. The initial rank0-only summary had incorrectly generalized exact
caches to all32 owners; independent review of v2 caught that interpretation error.
The fixed graph/source identities and bounded cache checks rule out some wiring
failures, not floating-point arithmetic differences inside the prefix.
Do not retry unchanged, insert speculative barriers, emulate a legacy reduction
tree, widen tolerances, or replace the reference on the strength of these arrays.

Additional rank4 original-generation replay locates its sole changed KV element
at prompt-relative row38, and its two changed index elements at rows42/98 (slots0/4).
This is later than the first DSA order change at row2. In a correct causal path,
those later cache differences cannot explain that first changed selection.
The earliest DSA discriminator must therefore include query/head/scorer operands,
not only the downstream router; no specific arithmetic mechanism is proved yet.

## Smallest next discriminator

Use the same retained layer6 subset and boundary fixture only. Reuse existing
layer/DSA/router primitives and protected selected-layer worker/publication;
do not build another engine or repeat full-model/cleared layer0/3 tests.

1. Capture candidate and control's actual DSA query/head-weight/current-key
   operands at the first differing selection, plus attention update, combined residual
   BEFORE postnorm, completed BF16 normalized MLP input, router partial/global
   logits, correction bias, biased scores, ordered routes and route weights.
   Preserve original selection/cache/output observations needed to compare to
   this failure. The original `normalized` field is pre-attention, not MLP input.
2. Observation changes may perturb compilation. First require reproduction of
   the original failure's route IDs and DSA order/scores, and quantify whether
   original output/cache/residual fingerprints reproduce. Do not attribute the
   original failure from a diagnostic whose relevant boundary has changed.
   Preserve such a perturbation as a separate result; no automatic retry series.
3. On completed captured BF16 router inputs, replay the existing router at B128
   and fourB32. Compare against FP64 CPU dot/selection using the same captured
   input and checkpoint-bound weights/bias. Use each path's own input when they
   differ; a reference made from one path's inputs cannot adjudicate the other.
4. Decision: identical input with divergent routing localizes router lowering.
   Different input with own-input-correct routing redirects to attention/postnorm.
   Wrong own-score selection localizes selector/metadata, not a tolerance issue.
   DSA row2 needs its own-input score replay with identical causal keys, since
   later changed cache rows cannot explain it. Expand attention projection
   captures only if the first differing boundary justifies them.

This is a bounded **diagnostic**, not a replacement numerical-admission protocol.
Its new graphs, output schema, memory budget, provenance and original-signature
reproduction need focused CPU composition and independent review before launch.
No new model/weights, full decoder, competitive/tail rerun or performance sweep
is needed to answer this first question. Stop expanding the diagnostic once
the first differing completed boundary is identified; fix or adjudicate that
specific difference under the existing contract, then resume larger-window work.
