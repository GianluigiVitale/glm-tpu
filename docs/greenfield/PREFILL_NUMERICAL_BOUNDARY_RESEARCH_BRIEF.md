# Deep research request: localize the B128 prefill numerical boundary

Prepared 2026-09-08. Evidence baseline: `ed0a3e05` on
`rewrite/topology-first-decode`. This is a self-contained research request,
not a new acceptance protocol or permission to run an experiment.

## Request to the researcher

Find the smallest defensible way to distinguish an implementation defect from
legitimate arithmetic differences in the specific failure below, and recommend
an efficient fix or a valid adjudication path. Research primary papers, pinned
JAX/OpenXLA source, official TPU documentation and relevant upstream issue
discussions. Give direct URLs and source/version dates. Separate demonstrated
facts from hypotheses and unsupported explanations.

Do not produce another general MoE throughput survey. Our two prior reports
already cover expert-aligned weight panels, exact sorted top-k merging and
rolled prefix loops. The missing answer is numerical localization. We will
continue local capture work while this research runs; research is not a gate
or a reason to wait. You cannot inspect our private arrays from this document:
explicitly identify every conclusion that needs additional local observations.

## TL;DR

An existing small-window batched full decoder passes its protected 2K test.
We enlarged only the layer/MLP window to 128 rows, retaining four causal
attention/indexer tiles of at most32 rows. A real-weight layer6 test against
four complete B32 layer calls failed on DSA ordering. Original arrays also show
five actual expert-set changes. Layer output errors remain within existing
bounds, but this does NOT authorize ignoring routing differences.

The pre-attention normalized BF16 inputs match. Actual DSA query/head operands
and completed normalized MLP/router inputs have not yet been captured from
this failed execution. Therefore we do not know the first arithmetic cause.
Instrumentation previously changed a reference's arithmetic: reproducing the
original failure under observation is essential before assigning causality.

## 1. Fixed setting and current architecture

- GLM-5.2-FP8: 78 layers, width6144, 256 routed experts, top8, one shared
  expert, intermediate2048; full indexer32 heads of width128, DSA top2048.
- Native JAX engine, not a legacy execution import. Existing8 hosts/32 TPUv4
  chips; weight-stationary mesh expert8 × feature4. Repeated collectives use
  those subgroups, never full-pod hidden reconstruction.
- Installed JAX0.10.1 and libtpu0.0.41. Do not assume current online API
  signatures match this installation. For example, installed `lax.top_k`
  rejects `is_stable=True`, although current online docs expose it.
- FP8 checkpoint data/scales feed tiled kernels; BF16 activations and FP32
  accumulation/reduction boundaries matter. No full BF16 weight expansion.
- Candidate: four32-row attention/DSA/repair prefixes within ONE layer graph,
  carry all three caches, concatenate normalized MLP rows, then one B128
  router/grouped-MoE suffix. Control: four full B32 layer calls, carrying the
  same caches. These are different compilation contexts even when prefix
  source functions are shared. No serial full-decoder-per-token fallback.
- Prompt scoring consumes UNREPAIRED index keys. M64-repaired keys live in a
  separate buffer and become visible only after the entire prompt. Each layer
  owns its own KV; IndexShare does not share another layer's KV values.
- Current failed fixture: real checkpoint layer6, synthetic input/cache state,
  capacity4096, prefix505, 128 live rows, key tile512. It is not a natural
  full-model activation sample. Planned competitive prefix2553 and tail cases
  were never executed after the boundary refusal.

Historical Gate D is closed (DB567); this is the new larger-window prefill
admission, not evidence that the old gate reopened. DB588's B17/B11 full78-layer
2K result has exact20/20 tokens, protected DSA/cache,102.203s request-prefill
and129.171ms decode p50. It does not certify this B128 path or efficient256K.
DB589's supplied-route real-MoE baseline improved equal128-row work by3.329×
for distributed routes and1.280× for concentrated routes; neither is a model
speedup or an XPlane-backed prefill attribution. There is a reason to retain
batching, but no numerical exemption follows from that gain.

## 2. Original failure and what is actually known

Tag: `greenfield_fp8_ws32_prefill_layer_window_numerical_l6_20260908T132240925858058Z`.
Execution pin: `0f994e373dade76ede59da179da6856d17234e5d`.
All four acquired graph profiles and allocation checks passed. Two WK helper
calls, one B128 and four B32 model calls completed; then the comparator raised
`window/control ordered DSA selection differs`. No timing campaign, SUCCESS
or DB row exists for this failed trial. Authenticated normal and root/libtpu
censuses confirmed8/8 clean at13:25:07Z. No unchanged retry was launched.

Read-only analysis authenticated all128 original partial files using exact
generation/size/CRC32C/SHA256, fixed input fixture, graph identities, selected
tensor hashes, eight unique processes and all32 physical device/slot bindings.

| Observation | B128 versus four B32, original arrays |
|---|---|
| DSA ordered-position differences | rows2,52,69,82,101,121 |
| DSA selected-set differences | zero, but only506–633 causal keys exist; all fit below2048 |
| Own selected-score order/ties/counts/padding | pass on both; not an independent competitive full-score-row proof |
| Ordered router differences | 15 rows |
| Actual router set differences | rows2,10,32,62,113 |
| Pre-attention normalized input | bit-identical on all32 owners |
| Output / residual maximum absolute difference | 0.001956939697265625 / 0.000244140625; every-row and aggregate bounds pass |
| Raw slotwise route-weight maximum difference | 0.009262464940547943; fails bounds |
| Expert-ID-aligned weights,123 matching-set rows only | max0.00008444488048553467; bounds pass, diagnostic ONLY |
| Written KV | differs slots0–3, max3.725290298461914e-9; fixed written-row bounds pass |
| Written unrepaired index | differs slots0–7, max0.001953125; fixed written-row bounds pass |
| Repaired index and untouched cache bytes | exact on all32 owners |

Expert replacements (candidate versus control): row2 144/208; row10 206/200;
row32 39/234; row62 220/239; row113 250/200. Aligning weights excludes these
five rows and is NOT a way to pass the failed test.

The first DSA discrepancy at row2 orders positions142/396 versus396/142.
Do not assume a same-score tie: the paths' own scores differ. Later cache
changes are localized to row38 for KV and rows42/98 for unrepaired index in
the original rank4 data (physical slots0/4). They cannot explain row2 under
correct causal visibility. A future-key masking defect remains a hypothesis
to test, not something this temporal observation alone disproves.

Important correction retained: a preliminary rank0-only reading suggested
exact caches. Other physical owners disproved that fleet-wide claim. Use all
owners, not one launcher rank. Matching the pre-attention normalized input is
NOT evidence that the post-attention normalized router input matches.

## 3. Arithmetic and boundaries that the research must respect

The actual layer uses split residuals. Input normalization produces a
normalized attention input and a combined residual. Attention produces an
update. A second fused add/RMSNorm combines that update with the residual,
producing the normalized MLP input and post-attention carried residual.
The normal result currently exposes PRE-attention normalized input, not the
completed normalized MLP input. A rounded residual output is not automatically
the FP32 pre-normalization operand; references must use the actual operands,
epsilon and cast/rounding convention.

Indexer preparation includes FP8 query projection, FP32 head-weight projection
with `HIGHEST` precision and feature4 reduction, on-device interleaved rotary,
and BF16 cached keys after projection/norm/rotary. Main-attention rotary uses
the separately validated host table; indexer rotary deliberately stays on
device. Do not propose a blanket host-indexer-table replacement: that variant
was previously refuted as a legacy-faithfulness fix.

The current DSA score operation, with `precision="default"`, is:

```text
per_head[r,h,s] = relu(dot_f32(query[r,h,:], key[s,:]) / sqrt(128))
score[r,s] = sum_h(head_weight[r,h] * per_head[r,h,s])
```

Head weights are signed. ReLU precedes their multiplication. Input casts and
`preferred_element_type=float32` are explicit, but the backend arithmetic of
DEFAULT must be established rather than assumed to be IEEE scalar FP32.
Selection uses exact descending score, then lowest absolute position on ties,
with causal validity, counts and padding. Invalid values must not enter the
live set. Tiled local candidates are merged across expert8 ownership.

Router: completed BF16 normalized MLP features and BF16 checkpoint router
weights are cast to FP32 for local `dot_general`, preferred FP32 result,
DEFAULT precision; feature4 sum then expert8 gather forms all256 logits.
Selection scores are `sigmoid(logits) + correction_bias`. Select top8 with
the existing tie semantics. Weights use the UNBIASED sigmoid values of selected
experts, normalized in FP32; correction bias affects selection only. Routed
scaling is applied later to the reduced routed output. Own-logit selector
correctness and cross-program route agreement are different questions.

## 4. Previous expensive lesson: avoid another numerical archaeology loop

Earlier layer3 diagnostics are sealed as DB585–587. Capturing a completed BF16
router input and replaying the same input through batched/scalar routers with
an own-input FP64 reference separated a fused-prefix discrepancy from the
standalone router. However, simplifying the reference prefix changed20,963
normalized BF16 values versus the original capture. Visible BF16 conversions
in HLO did NOT prove a specific hardware rounding-bypass mechanism.

Reproducing the exact original12-field prefix and then capturing the actual
additional PREnorm boundary enabled a bounded layer3 admission under unchanged
contracts. Original failed trials stayed failed. This is a useful method,
not proof that the present layer6 failure has the same cause or deserves the
same reference change. Do not restart legacy reduction-tree emulation, propose
blind precision/barrier sweeps, or treat a standalone replay as the fused graph.

## 5. Research questions, ranked

### A. Minimal observation that still represents the original failure

What supported JAX/OpenXLA/TPUv4 mechanisms can expose the following actual
executing boundaries with the least change to fusion/layout/reduction?

1. Row2 DSA query, signed head weights, causal key operands and scores.
2. Attention update and combined FP32 pre-postnorm operand, including both
   split-residual inputs and the normalization statistics/convention needed
   to reconstruct its math.
3. Completed BF16 normalized MLP input, local router partials, global logits,
   correction bias, selection scores, ordered IDs and route weights.

For each proposal, explain what it guarantees and what it cannot guarantee.
An extra output, compilation barrier or separately jitted prefix can change
the very computation being diagnosed. Require original route/DSA signatures
and original output/cache/residual fingerprints to be checked first. If they
do not reproduce, what is the smallest next discriminator, with an explicit
stop condition, instead of a series of increasingly different captures?

### B. Distinguish source inputs, matmul precision and selection errors

Design a compact same-input B128 versus four B32 router replay and a row2 DSA
replay. Use captured operands from EACH original path when their inputs differ.
Specify FP64 CPU reference arithmetic, exact checkpoint/scale/bias binding,
nonlinear operations, expected rounding points and exact tie rules.

What evidence distinguishes DEFAULT TPU matmul precision, fused cast changes,
shape-dependent reduction, upstream norm differences, or incorrect masking/
metadata? Which of these can differ when BF16 inputs agree, and which require
FP32 operand differences? Consult pinned source or explicit backend guarantees;
do not assume `preferred_element_type`, `HIGHEST`, or
`shape_invariant_numerics` establishes bitwise equivalence of whole programs.
Give hypothesis → smallest test → predicted result → ruled-out alternatives.

### C. DSA order versus attention and routing sensitivity

Here every valid key fits in top2048, so set agreement is nearly vacuous while
scores and ordering differ. Can a permutation of the SAME selected keys change
this sparse attention accumulation enough to affect routing? How should a
controlled permutation replay isolate order-dependent accumulation from changed
query/attention projections or pre-normalization values? Keep it diagnostic;
do not change production selection order as a workaround.

Prioritize row2, where later changed cache writes cannot be a causal explanation.
Include a bounded future-key perturbation/control to distinguish actual causal
masking from assumptions based on the intended source program.

### D. Error bounds and near-boundary router decisions

How can operand-derived forward-error bounds and actual top8/9 selection margins
distinguish legitimate rounding from systematic arithmetic error? Account for
sigmoid, correction bias, cancellation, feature reductions and normalization.
Do not fit a tolerance to the five observed swaps or import a scorer-only bound
into an upstream-normalization problem. Return a proposed diagnostic analysis,
not a retrospective PASS rule.

The binding contract allows bounded internal tensors, exact own-score DSA/ties,
reviewed high-precision cross-oracle adjudication, and exact protected short
tokens. It does not demand universal legacy bit identity. Conversely, this
specific failed admission remains failed: an independently justified future
adjudication needs explicit review/preregistration, not silent comparator edits.
Explain what each level of evidence proves and which end-to-end obligations
still remain after a local discrepancy is explained.

### E. Fix the identified boundary without throwing away batching

For each evidence-supported cause, give the narrowest plausible fix and its
cost/risk: materializing only a necessary boundary, a precision change at one
projection, a correctly rounded normalization, or correcting causal/index
metadata. Do not recommend changes before the discriminator identifies a cause.
Estimate extra bytes/dispatches/collectives symbolically and describe how to
measure the true full-layer wall impact. Do not force the entire decoder back
to one prompt token at a time or insert a host roundtrip per prefix tile.

## 6. Requested deliverable

- Start with a one-page answer: most plausible explanations, evidence against
  each, and the single next local experiment with highest information value.
- A decision tree that terminates in a specific fix, a justified adjudication
  proposal, or a precisely identified missing capture; no open-ended survey.
- A capture/replay table: field, dtype/shape/owner, reason, estimated bytes and
  whether observation can perturb its producer. Prefer selected-layer and
  selected-row captures; explain when full rows/causal keys are necessary.
- Pinned primary-source citations for compiler/precision claims. Clearly mark
  version uncertainty, analogous results and unverified issue hypotheses.
- Minimal pseudocode/tests where helpful, including failure/perturbation cases.
  No API that requires an unapproved environment upgrade.
- A short list of genuinely missing local evidence. If the report cannot
  decide root cause without it, say so; do not fill that gap with confidence.

## 7. Constraints and optional evidence attachments

No new hardware or infrastructure management. Only existing retained weights;
no checkpoint repack, full-size safety copy or new model variant. Approved
bucket is US-CENTRAL2 with live ceiling2.5e12 bytes. Small evidence/captures only.
CPU/reference first; TPU workflows serialized with authenticated ownership,
memory/HLO admission and cleanup. Do not recommend full128K/256K runs to answer
a one-layer numerical question. No weakened health/cache/DSA/tensor checks.

This brief contains the facts needed to research the method. If the chat can
accept additional files, these are the useful attachments, not the whole repo:

1. `PREFILL_WINDOW_BOUNDARY_DIAGNOSIS.md` — original failure interpretation.
2. `../artifacts/prefill-window-boundary-refusal-v2-20260908.json` — all-owner
   analysis; SHA256 `2b5e556dfeea92071859650b70c7ca51d882ab1d5a26d54845beee6d0e273fff`.
3. `PREFILL_RESEARCH2_ADJUDICATION.md` — what we already accepted/deferred from
   the second performance report, to avoid duplicate research.
4. Small relevant excerpts of `kernels/ws32_prefill_window.py`,
   `ws32_prefill_layer.py`, `ws32_prefill_dsa.py`, `reference/dsa.py`,
   `reference/moe.py` and the fused add/RMSNorm implementation, all under
   `glm_tpu/greenfield/`, pinned to the executing revision.

Originals remain generation-bound under
`gs://driftbench-dsv4-uc/results/<tag>/workers/rankN/`; that is an internal
evidence location, not a public citation or a request to download weights.
The forthcoming local boundary capture may refine this brief. Clearly separate
predictions made now from conclusions drawn after those observations arrive.
