# 17 — The Engine Lottery: post-mortem of the streamer/dequant finite-corruption hunt

**Window covered:** 2026-07-17 03:15 UTC → 2026-07-24 01:15 UTC (with pre-history back to gate2's death,
2026-07-12/13, which this bug caused).
**Authoritative sources:** `docs/RESEARCH_LOG.md` (2026-07-17 04:15 → 2026-07-23 11:30 — every claim here
traces to a dated entry), `CLAUDE.md` §Progress, `HANDOFF.md`, `docs/suggestions.md` (the method doctrine),
`docs/upstream/pageloop-v4-sublane-drop-REPORT.md` (the sibling defect class).
**Status at writing (updated 07-24):** root cause named, measured, and ground-truth-adjudicated;
manifest refusal landed and proven live; the corruption window narrowed to post-dequant/pre-t2j CPU
storage (Phase J); the PWAL-time self-healing repair (`GLM_WK_OOB_DIR`, fork `dc0443a43`) **validated
on metal** — 4-draw arm 07-23/24: 5 zero-fill strikes repaired across 3 draws, every serving engine
manifest-VERIFIED 8/8, 12/12 needles, the one NaN-flavor strike refused fail-closed. GATE4 launches
on this stack.

A note on the log itself: the RESEARCH_LOG's 2026-07-18 timestamps are internally inconsistent (see
Appendix C, item 1). This document uses **file order** as the authoritative sequence, per the log's own
"the record over the narrative" rule.

---

## 1. Executive summary

**The bug.** The weight-load path (runai GCS streamer + the fused fp8-dequant load of the DSA indexer
`wk`) silently delivers corrupt bytes for a few random tensors on some fraction of engine launches; the
corruption comes in two flavors — a NaN/Inf flavor that our init scans caught and refused, and a
**finite** flavor (measured end-state: the dequantized-wk half of the fused layer-10 indexer tensor
delivered as all-zero bytes) that was invisible to every NaN scan and to the H2D checksum, because the
checksum verifies CPU→device fidelity of bytes that were already wrong. A corrupt indexer weight degrades
that layer's sparse selections per-instance-permanently, which presented for six days as an
"engine-instance lottery": most engines healthy, some engines missing long-context needles with clean
instruments everywhere we looked.

**The cost.** Two n=77 sparse-128K gates killed by this one bug: gate2 (2026-07-12/13, dead at d=0.95
0/11 after 22/22, db run 165) and gate3 (2026-07-18, dead at 33/35, d=1.0 0/2 miss-abort, db runs 199).
The hunt ran 2026-07-17 03:15 → 2026-07-23 11:30 (~6.3 days wall; the log's own 07-23 entry counts "the
5-day hunt" from the lottery's first capture on 07-17 08:20). On the order of **100 engine
launches/inits across ~25 arms and loops** (Appendix B), ~230 GB-class dump campaigns, two fix builds
that validated null, and a gate schedule slip of ~5 days (128K gate from ~07-19 to ~07-24).

**The fix.** (1) A categorical detector that closes the gate path: `GLM_STATE_HASH` per-leaf uint32
sums of the **final** model state verified at init against a **golden reference manifest**
(`GLM_STATE_HASH_REF`, fail-closed refusal; manifest built by majority-of-3 boot engines plus offline
safetensors ground truth for frequent-victim tensors). (2) A **self-healing repair at the corruption
window's closing edge** (Phase J): the zeroing was localized to the fused leaf's CPU storage *after*
dequant and *before* t2j, so a PWAL-time guard (`GLM_WK_OOB_DIR`) detects the impossible all-zero half
and rewrites it bitwise from a gcsfuse checkpoint mirror — validated on metal 07-23/24 (§5.5). The
originally planned load-path replacement (gcsfuse Plan A as the *load* path) measured ~10× too slow
for init and was demoted to the repair source. Defense in depth: repair → refuse → relaunch. A real
second bug — the t2j zero-copy alias race — was found and fixed en route (fork `629c20e84`) and is
independently upstreamable, but it was **not** this bug.

---

## 2. The bug, in full detail

### 2.1 Where it lives

The serving load path is: runai streamer (ranged GCS reads) → CPU torch tensors → for the DSA indexer
`wk`, a **fused load + fp8→bf16 dequant** (`_try_load_fp8_indexer_wk`, vllm `deepseek_v2.py:746-791`:
dequant of fp8 `wk` `[128, 6144]` with block scales, concatenated with the raw bf16 `weights_proj`
`[32, 6144]` into one `[160, 6144]` bf16 leaf) → t2j conversion → async H2D → device weights → derived
tensors (the `glm_dsa_adapted_*` indexer params, the absorbed `w_uk_t`/`w_uv`) computed at init.

On some launches, for a small number of tensors (~2-3 corrupt leaves per launch even on "healthy"
engines — measured 07-23 08:10), the bytes that reach the device are wrong. The final measured specimen
is deterministic: **the fp8-wk byte range arrives (or dequants to) all zeros**. Whether the zeros came
from the GCS **read** or from the **dequant compute** was the discrimination still open at root-cause
time; Phase J closed it with a third answer — **neither**. A dequant-time zero-check observed GOOD
values on launches whose device state was later corrupt, so the zeroing happens **after the dequant
completes, in the fused leaf's CPU storage, before/at t2j conversion** (§5.4). The leading mechanism
candidate is the load path's CPU-storage free machinery (`_free_cpu_storage`-class `resize_(0)`)
releasing the buffered fused param's storage out of order — the same held-reference-across-an-async-
boundary family as the t2j alias race (§2.7).

### 2.2 The two flavors

- **NaN/Inf flavor** ("the load class"): corrupt fp8 bytes decode to non-finite values (fp8 e4m3fn has
  NaN codes; no Inf — the observed Inf lives in the ≥16-bit stage). First caught live 07-17 18:47 by
  `GLM_PWAL_NAN_CHECK` at engine init: `w-0 layer=0 wk NaN:640 Inf:640; layer=1 wk NaN:1331 Inf:205`
  (+2 more hosts). The counts are exact multiples of 128 (1280 = 10 and 1536 = 12 blocks of 128) —
  128-element-aligned (256-byte) granule corruption in a ≥16-bit stage. This flavor is loud once you
  scan for it; the armed stack refused these engines at init from 07-18 on.
- **Finite flavor** ("the state class"): corrupt bytes decode to ordinary finite values — including the
  exact endgame specimen, a zero-fill. Invisible to every non-finite scan **and** to `GLM_LOAD_CHECKSUM`
  (CPU-vs-device byte verify: corrupt-in ⇒ corrupt-out passes, by design). This flavor is what killed
  gate2 and gate3 and consumed the residual hunt.

### 2.3 The exact sums (ground truth, 07-23 11:30)

For `layers.10.self_attn.indexer.wk_weights_proj.weight` (the fused `[160, 6144]` bf16 leaf), uint32
wraparound byte-sums:

| Quantity | Sum |
|---|---|
| True fused-leaf sum (from the GCS safetensors, offline fp8-dequant replicated **bit-exactly** against vllm's own `scaled_dequantize`) | **239851472** |
| … of which the dequantized-wk half `[128, 6144]` | 191463636 |
| … of which the raw bf16 `weights_proj` half `[32, 6144]` | 48387836 |
| The corrupt value observed on sick launches | **48387836** |

Corrupt sum == the weights_proj half alone ⇒ **the dequantized-wk half is delivered as all-zero
bytes** — a deterministic zero-fill of the fp8-wk range in the fused-load/dequant path. The same corrupt
sum was sighted three times: statepair draw 3 (8/8 hosts), statepair draw 2 (1/8 hosts), and the
manifest bootstrap engine itself (rank 0). The derived tensor `glm_dsa_adapted_wk` on the corrupt
engines sums to 0 — the adaptation of a zeroed source is zeroed.

### 2.4 Per-host independence and replica-severity arithmetic

The model is replicated per host (8 model replicas on the v4-64 pod); each host runs its own streamer
reads. Corruption is drawn **per host, per launch**:

- NaN-flavor host sets across the discriminator specimens: p1 {w4}, p2 {w3,w4,w7}, p5 {} (clean slots —
  a finite-flavor miss), p7 {w4}, p8 {w1}.
- Finite-flavor layer-10 sightings: 8/8 hosts (statepair d3), 1/8 hosts (statepair d2).

Severity is arithmetic over replicas: each host's attention contribution enters the `o_proj` psum, so
**1 poisoned host out of 8 is usually survivable** (7 clean replicas dominate — probe p1 retrieved
correctly with w4 fully poisoned), while **3 poisoned hosts produce fluent filler** (p2). The observed
output gradient tracks poisoned-host count and victim severity: crisp answer → coherent haystack filler
→ mangled-digit near-miss (`'7657'` vs `'797567'`; `'665060'` vs `'648060'`) → semi-degraded filler →
heavy babble. Note the caveat in Appendix C item 3: with a *deterministic* corrupt value, independent
per-host events produce byte-identical corruption when they hit, so "identical on all 8 hosts" (07-23
08:10's broadcast inference) does not by itself prove a single upstream read.

### 2.5 Frequent victims

Some tensors are **frequent victims** — their byte ranges are systematically vulnerable in the streamer
read pattern. The layer-10 indexer wk drew the identical corrupt value in 3 independent launches;
indexer-region tensors (layers 0/1 wk especially) surfaced all week in the NaN flavor too. The earlier
guess ("first/largest early tensors in stream order", 07-19 00:55) was superseded by the frequent-victim
observation (07-23 10:40). Consequence: a naive majority-vote manifest can enshrine a corrupt value —
hence the ground-truth override in the golden-manifest protocol (§5).

### 2.6 Why it presented as an "engine lottery"

- The victim set is drawn once per launch and is then **per-instance-fixed** (weights are loaded once):
  each engine lands in a discrete behavioral state — exactly the "per-engine-DISCRETE score states"
  measured 07-21 11:45 (draw1≡draw2 bit-identical selections AND scores, 0/2310 keys differing; draw3,
  also healthy, a different state at 1925/2310; sick draw6 a third state).
- Severity depends on *which* tensors were hit and how many hosts: most launches draw benign victims
  (every engine carries ~2-3 corrupt leaves; "healthy" engines were also corrupt, just harmlessly).
- Expression is **length/budget-dependent**: at 5K a query selects ~41% of positions, at 32K ~6% — a
  degraded scorer survives at 5K and fails in the sparse regime. This made the 5K health probe blind
  and made sickness look intermittent per-request.
- It is **position-gated at 2048**: chunk-0 tokens take the ctx≤topk dense fallback, which ignores
  selections entirely — so a corrupt indexer weight leaves the first chunk bit-identical to healthy,
  which repeatedly mimicked "clean early layers / clean early events" and (earlier, spuriously) "clean
  page 0".
- The victim layer varies per draw (lscan's victim ≥L15; statepair's at L10) — every fixed cache-dump
  window missed it.

### 2.7 Relation to the sibling defect class

`docs/upstream/pageloop-v4-sublane-drop-REPORT.md` documents a genuinely different defect (XLA:TPU
lowering silently dropping sublane row-stripes on a donated shard_map-resident paged cache) with the
**same presentation family**: silent, no NaN, per-instance lottery via inherited state,
accuracy-check-invisible, "run A striped, run B clean". That report's existence was a major prior
during the hunt — it made "v4 lowering fault on the donated indexer cache" the best-ranked hypothesis
for four days, and it is why two plausible fixes (onehot gather, cache un-donation) targeted the read
path. The prior was reasonable; it was also wrong here. Same presentation ≠ same mechanism.

The DSV4 "flaky dequant crash" (moe-tpu, 2026-06-19: ~60% intermittent silent crash per engine build
during the fp8 dequant phase of the runai load, never root-caused, retry-mitigated, zero post-load
verification) is the same family as this bug — the loop the 07-23 11:30 entry closes. DSV4 had no
verification; its silent-corruption form would have sailed through undetected.

---

## 3. The hunt, chronologically

Every hypothesis, its verdict, its cost, and what would have been needed to skip it. Draw counts and
outcomes are itemized in Appendix B.

### Phase A — The discriminator: the lottery is real (07-17 03:15 → 16:50, ~13.5 h)

**Hypothesis:** gate2's 0/11 death was either a real per-engine lottery or the w-4 disk-at-0 confound.
**Instrument:** `probe_lottery.sh` — 20 planned scrambled draws, gate2-verbatim config, fixed seed
(gate2's first d=0.95 needle), scrambler interleave (a review BLOCKER fix: back-to-back identical
engines inherit stale≈fresh HBM and mask the never-written class), INFRA-vs-verdict taint ordering.
**Result:** stopped early at 8 draws — 6 valid: 2 CORRECT / 4 MISS (67%; Wilson 95% ≈ 30-90%), 2 INFRA
(orphaned Ray firewall rule after pod recreation; a git index.lock race from my own forensics agent).
2 of 8 scramblers also missed at 32K. **The lottery is real** — far above gate2's ~1/7 estimate.
**Byte-diff forensics** (p1 CORRECT vs p2 MISS, 22 slots × 8 hosts × 4 shards): per-HOST quiet-NaN
(0x7fc0) poisoning of the layer-1 indexer k-cache — p1 {w4} 92%, p2 {w3,w4,w7} 91-92%; the poison unit
is a whole host; the lottery is which hosts.
**Eliminated:** the disk-pressure explanation of gate2 (clean disks, still missing); then (08:55, via
per-host libtpu log fingerprints) the per-host-binary/compile-split hypothesis — **w4 computed
byte-identical NaN poison while provably executing the same executables as clean hosts**, so the
discriminator is per-host runtime state.
**The unrecognized clue:** probe p5 MISSED with **all 22 dumped slots clean** and coherent filler — the
finite flavor's first sighting, correctly flagged as "either a genuine selection-quality miss or
corruption in an undumped buffer", not yet attributable.
**Cost:** 16 engine inits (8 probes + 8 scramblers), 2×3.7 GiB dump archives, ~13.5 h.
**To skip:** nothing — this was the right first instrument, and its redesign-by-review (scrambler,
taint ordering, N=20 power) is a keeper.

### Phase B — From runtime state to the weight load (07-17 17:40 → 19:10)

**Hypotheses:** (a) garbage input (params born bad), (b) runtime clobber (sparse-branch arena temp over
the freshly written k-cache). The PWAL-copy hypothesis (a per-host precomputed indexer-param copy used
only by sparse chunks) was **refuted at code level** in ~1.5 h — `compute_indexer_keys` is straight-line
code upstream of the dense/sparse cond, same stored params every chunk.
**Instrument:** `GLM_PWAL_NAN_CHECK` — init-time NaN scan of orig + precomputed indexer params.
**Result (07-17 19:10):** on the first armed engine that missed, the check had already flagged **at
init**: the loaded wk weights themselves arrive NaN/Inf, per host, per instance, before any serving
step. The "engine lottery" is the weight-load path.
**Eliminated:** runtime clobber; PWAL copies; born-at-serving-time anything.
**Cost:** the timeline run (3 scrambler+probe pairs, 6 engines; pair 3 delivered the flag) plus the
earlier code map. The per-step timeline specimen then proved (07-18 "07:30" entry) **static
load-corrupt wk, full stop** — 8060 fully-NaN page-instances, 0 partial, 0 clobbered; NaN front ==
write front — and corrected the "page-0-clean" narrative (an off-by-one: physical page 0 is vLLM's
null block; there never was a clean chunk).
**The prophetic paragraph:** the same entry states the **finite-corruption implication** verbatim:
"corrupt fp8 bytes only SOMETIMES decode to NaN — most garbage decodes to random FINITE values,
invisible to any non-finite scan… p5's clean-engine coherent-filler miss is exactly this signature ⇒
NaN-refusal is necessary but NOT sufficient; the loader must be actually FIXED before the gate." This
is the root cause, named on day 2. It was under-weighted for the next four days (§4).

### Phase C — The t2j alias race: a real bug, the wrong primary (07-18 ~04:20 → 09:25)

**Owner course-correction (04:20):** re-read the corpus in full before building more instruments — the
"flaky dequant crash (DSV4 hit it too)" breadcrumb had sat in CLAUDE.md while six instruments were
built; and we had measured rates without ever byte-comparing one corrupt tensor to its truth (the
dump1090 move).
**Found by reading (05:30):** `utils.py` t2j bit-cast branch takes a **zero-copy numpy view** of the
torch storage; JAX's PJRT host staging requires the buffer to outlive the async H2D DMA;
`_free_cpu_storage`/`resize_(0)` frees it. Lose the race ⇒ DMA reads freed/reused heap — per-host,
per-launch, contiguous-granule garbage. The prior-art sweep (06:10) recovered the DSV4 crash story and
unified it: page unmapped ⇒ SIGSEGV (DSV4's crash), page reused ⇒ silent garbage (our corruption);
rates match (~60% / 56%).
**Concurrency A/B (06:35):** RUNAI_STREAMER_CONCURRENCY 32-vs-8, 16 init draws: 3/8 vs 6/8 corrupt
(overall 9/16 = 56%) — **streamer concurrency exonerated** (and the inverse trend mildly fit the race
story).
**The fix (08:05, `629c20e84`):** eager copy in both t2j branches + 4 import-bypass reroutes + the
alias-free invariant test — **3 boundary-alias failures on pristine → 10/10 with the fix** (the
must-fail-first proof); 262 regression tests unchanged.
**Validation whiplash (08:50 → 09:00 → 09:15):** draw 3/3 corrupt ⇒ "fix did not collapse the rate"
(the pre-registered must-be-zero rule); then the fuller pool said 1/9 post-fix vs 9/16 pre-fix, Fisher
p≈0.036 ⇒ "collapsed MOST, ~10% residual, GATE PATH OPEN"; then adversarial review 2/3 **refuted the
overclaim**: the pools were instrument-mismatched (matched-only 1/3 vs 9/16, one-sided p=0.46, n.s.);
the residual CI was Wilson [2.0%, 43.5%]; and — decisive — **every gate protection was NaN-class**,
so at finite-taint rate q per draw, P(≥1 of 7 gate engines tainted) = 30%/52%/73% at q=5%/10%/17%,
making a 0-miss gate un-attributable.
**Verdict on the t2j race:** a REAL bug (proof test + review-verified alias severing + torchax's own
TODO circling the same function), correct hygiene, upstreamable — but **refuted as the primary
mechanism** of this hunt's corruption. How much of the 56%→~10% NaN-rate drop it truly caused was
never cleanly resolved (the pooling confound; and the finite flavor was invisible to every instrument
in that comparison).
**Cost:** ~1 day incl. 16 A/B inits + 9 validation/dissection draws + 3 adversarial reviews.
**To skip the wrong-primary attribution:** instrument-matched arms by design, and the categorical
instrument first (which is exactly what review 2/3 forced).

### Phase D — The checksum era and gate3's death (07-18 09:55 → 20:30)

**Instrument:** `GLM_LOAD_CHECKSUM` (`a225d16b4`) — uint32 wraparound byte-sum of every t2j-staged
tensor on CPU (post-copy) compared against the same sum computed on device; raise on mismatch. Catches
NaN and finite **H2D** corruption categorically. Liveness proven: 8×8 host SUMMARY, verified=1882
tensors/host, 312 benign 0-d-scalar skips; validation 4/4 clean draws, zero false positives across
~60K tensor-checksums.
**The gate (11:05 → 20:30):** gate3 launched at n=77 with quadruple protection (health probe + PWAL +
LOAD NaN + checksum). d=0.0, 0.05, 0.95 all 11/11 — **33/33, including gate2's 0/11 killer cell**,
retroactively attributing gate2's death to the lottery, not the kernel. En route (14:35) the health
probe caught a sick engine in ~2 min (d=0.05 try 1) and the relaunch design worked — but that engine
was **THE RESIDUAL SPECIMEN** (db run 193): byte-verified clean on every surface (checksum, PWAL, LOAD,
0 NaN in 17.8M dumped cache elements), yet fluent-filler-missed a 5K d=0.5 needle. The entry correctly
narrowed candidates to "(a) CPU-side finite corruption BEFORE t2j (unexcluded…) or (b) engine-instance
state". Then d=1.0 went 0/2 (both pred=None, garbled-start + fluent filler) on an engine that had
passed its 5K health probe 30 min earlier — **gate3 dead at 33/35** by the pre-committed 2-miss abort.
**The pivotal reasoning error (20:30):** the death entry declares "weights categorically exonerated"
— an overreach beyond the instruments' documented coverage (the checksum verifies the H2D leg only;
candidate (a) from six hours earlier was silently dropped). The hunt now had the wrong search space.
**Cost:** the gate itself (~9.4 h, 35 needles, 5 engine draws, 2 depths burned) + 4 validation draws.
**To skip:** a reference-manifest check (the eventual fix) before gating. The checksum was the right
instrument at the wrong boundary — it verified our transfer, not the truth.

### Phase E — The write-path arms (07-18 20:50 → 07-19 12:45, ~16 h)

**Corpus re-read under the engine-state lens (owner rule):** verdict — H1 = pageloop-family READ-side
lowering fault (the sibling report is near-isomorphic: per-executable, silent, byte-clean, NaN-clean,
accuracy-check-invisible); H2 = actor-order mesh mismatch (Guard 1 the free tripwire); H3 = true
selection-quality miss (near-refuted — siblings 11/11 at the same depths).
**Eliminated cheaply:** process-index permutation (identical across sick and healthy engines — dumps'
npz metadata); CPU boundary math at the exact miss geometry (prompt_tok=127363, d=1.0, partial final
block 387/2048 — every falsification attempt passed; the payload gather is structurally unguarded but
the selection invariant held on CPU).
**The Guard-2 trip (07-19 06:55):** hunt v4 draw 1 raised `DCPCacheStaleStripeError` — stripe 1 of the
layer-0 indexer k_cache "512 freshly-written rows unchanged" during a 128K cell. The state class
appeared caught: DCP stripe write-loss on the DSA owner-scatter (impl=flat) at gate geometry — the
pageloop report's own law ("validation is per-buffer AND per-geometry") apparently biting its
validated case. Localization (07:15) named the site; the flat arm went 2/2 sick.
**The retraction (10:40):** stripe forensics across 106 captured steps × 8 hosts found **no drop and
no misroute anywhere** — stripe-1 rows are valid RoPE-structured keys, all 8 replicas bit-identical
over the full timeline, and the exact guard signature reproduces **benignly**: the ladder's
deterministic cells recycle physical pages whose stale bytes already equal the freshly computed keys ⇒
the write is a byte-level no-op ⇒ "unchanged" ⇒ **Guard-2 replay/page-reuse FALSE POSITIVE**. The
06:55 conviction was retracted the same morning.
**The A/B confirmation (12:45):** the barrier arm (donation broken before the write) was **still
sick** with zero guard trips ⇒ write path double-exonerated (different write formulation, same
sickness). Also retracted en route: the "sick engines are 2.6× slower" signature — the hunt-era
slowness was Guard-2 snapshot cost paid by all draws (gate3's sick needles ran ~626 s, same as
healthy).
**Cost:** ~16 h; hunt draws v1-v4 (2 LOAD_REFUSED — one of which, 07-19 00:55, delivered the genuine
**CPU-side pre-t2j attribution of the NaN load class** via `GLM_CPU_LOAD_NAN_CHECK`: the streamer/CPU
stage is guilty, H2D is faithful) + 2 sick flat draws + 1 barrier draw + an ENOSPC draw (22-slot 128K
dumps; the v4 trim to slots {0,1,2,4} traded away 17 indexer layers — a trade that later mattered).
**Eliminated:** H2 (Guard 1 clean), the write path, the slowness signature.
**To skip:** Guard-2's page-reuse blindness documented as a known-false-positive mode (its "only
whole-stripe-stale is visible" limitation was documented; the reuse false-positive was not), and an
expected-value compare instead of a changed-bytes compare.

### Phase F — The read-path arms: three init deaths and a control (07-19 13:20 → 22:30, ~9 h)

Three kernel-bypass discriminators were built and armed, and all three died identically at engine init
(~2-2.3 h each, worker SYSTEM_ERROR/actor death):

- `GLM_DSA_DECODE_INTERPRET=1` — Pallas interpreter for the decode kernel: the interpreter unrolls
  dsa_sparse_decode into an enormous graph at top-2048×640 production shape. NOT VIABLE.
- `GLM_DSA_ATTEND_GATHER_BARRIER=1` — read-side optimization_barrier: forces a full cache-slice copy
  (~29 G/chip) ⇒ device OOM. NOT VIABLE.
- `GLM_DSA_DECODE_ATTEND=xla` (the pre-built XLA attend oracle as serving path) — also died at init.

**The confound split (22:30):** a control draw at the same PIN with all probe envs unset initialized
healthy ⇒ the probe **commits** were exonerated; each armed arm died on its own variant's 128K-shape
compile — engines precompile at max_len regardless of request length. Lesson banked: an A/B arm must
match its **engine geometry** to its question; the fallback is a genuine 32K-geometry engine
(max_len 33280), not 32K requests on a 128K engine.
**Cost:** 3 burned arms (~6.5 h) + control + two audits landed en route (the workflow audit's M5
jaxpr-liveness test later proved the xla dispatcher genuinely strips every pallas_call — which made
the next phase's negative result trustworthy).
**Eliminated:** directly, nothing (all three inconclusive-by-infra) — the cost of not knowing the
max_len precompile behavior in advance.

### Phase G — The selection A/Bs at 32K geometry (07-19 23:00 → 07-20 17:40, ~19 h)

**X32 (07-20 00:40):** with a pre-registered interpretation rule (clean draws alone would be ambiguous
— the 32K-geometry base rate was unmeasured), draw 1 came back **SICK with the XLA attend** (0/6, 3
miss): the **Mosaic decode kernel is exonerated** (sick without it); the fault is upstream
(selection/gather); and 32K-geometry engines can be sick — a 25-min repro instead of a 14-h lottery
(amplifier unidentified: Guard-2 snapshots, per-step dumps, or mixed-length churn; identification
deferred, exploitation first — a deliberate, flagged confound, see §4).
**v1-vs-v2 selection (03:15 → 14:10):** v1sel draw 1 went 3/3 before a benign page-reuse sanity trip
(Guard 2 disarmed for A/B arms thereafter); v2-control without Guard 2 = 1 SICK / 3 draws (including
the first **mangled-needle specimen** `'7657'` vs `'797567'` — structure-preserving near-miss ⇒
selection degradation, not content corruption); v1 = 1 SICK / 2 draws (second mangled specimen
`'665060'` vs `'648060'`) ⇒ **the three efficiency-campaign v2 transforms exonerated as sole cause**.
The full pre-campaign revert arm was launched, then retired mid-run (draw 1 LOAD_REFUSED) when the
dump-diff superseded rate arms.
**Fingerprints (17:40, offline, free):** sick and healthy draws ran **identical executables** (unique
fingerprint sets 193/192/192, pairwise diffs ≤1, sick-only = 0) ⇒ per-launch compile variance refuted.
Same program + same logical inputs + different behavior ⇒ runtime state. The owner's tie-break
hypothesis (fp8-grid score ties ⇒ instance-divergent selection) was folded into the next measurement.
**Cost:** ~19 h, 9 serving draws + 2 retired.
**Eliminated:** Mosaic decode kernel, the v2 transforms as sole cause, Guard-2-as-sole-amplifier,
compile variance.
**To skip:** dump the selection **scores** first (the phase's own conclusion). The rate A/Bs each
answered one binary question per day-fraction; the score dump answered five at once.

### Phase H — The measurement: topkdump → mechanism → wrong conviction (07-20 17:40 → 07-21 21:20)

**Topkdump arm:** fixed-seed selection+score dumps, identically armed on every draw (F3-consistent).
Draws 1-4 healthy, draw 5 LOAD_REFUSED, draw 6 SICK with dumps armed (observer effect refuted). Now in
hand: 4 healthy + 1 sick engines' dumps on identical inputs and proven-identical executables.
**The mechanism, measured (07-21 11:45):** engines occupy **discrete per-instance-fixed score states**
— draw1≡draw2 bit-identical (0/2310 keys), draw3 (healthy) differs on 1925/2310, sick draw6 a distinct
state. The sick diff is pure **score divergence** (1562/1562 differing decode keys; |Δscore| up to
~107-122 on a −115..+82 range; k-th boundary swings like −91.2→+5.7; **0 tie flips — the tie-break
hypothesis refuted**). The clean smoking gun: cell C (32K d=0.5) healthy selects 41/42 needle-block
positions per decode query, sick selects 3.2/42 (122/285 queries select zero).
**Localization (12:30 → 14:50):** the scorer chain is provably layout-independent on CPU
(falsification suite, max|Δscore|=0); offline write-vs-read discriminator on the banked dumps:
**290,530 key-vectors byte-identical across healthy and sick** (including the pageloop-predicted
sublane rows {0,1,8,9}: 72,704 vectors, 0 diffs), block tables identical — yet scores diverge on this
identical input, beginning **exactly when the read spans >1 physical page**. Verdict: WRITE exonerated,
READ convicted — the scorer's multi-page paged gather, engine-fixed via the buffer-address
neighborhood drawn at init (the pageloop class, read flavor).
**Why this was a reasonable conviction:** identical caches + identical tables + divergent scores +
a boundary correlation + a documented sibling defect on the same buffer class. **Why it was wrong:**
the dumped caches covered only layers 0/1/2/4 — all upstream of the (per-draw) victim layer, so
"identical input" was true only for the early window; and ">1 physical page" is numerically the same
boundary as "pos ≥ 2048", i.e. the dense-fallback/sparse split — the data supported two readings and
the sibling-defect prior picked the wrong one.
**Fix 1 — onehot (16:10, `473904510`):** `GLM_DSA_SCORER_GATHER=onehot` replaces `k_cache[page_ids]`
with a one-hot matmul page fetch — bit-exact by construction, zero gathers in the fixed path (the
suspect op class removed, not patched). Validation criterion pre-registered and categorical: two
engines' full selection+score dumps must be bit-identical. **HONEST NULL (21:20):** 1930/2310 keys
divergent vs baseline 1925 — the gather exonerated.
**Fix 2 — un-donate (23:55, `d7ad7963b`):** `GLM_DSA_IDX_CACHE_NO_DONATE=1` excludes indexer caches
from step-fn donation (~700 MB/shard double-buffer; the report's own enabling condition "donated"
targeted). **SECOND NULL (07-22 09:50):** 2030/2310 divergent — unchanged.
**Cost:** ~2 days end-to-end for the two fix cycles (build + land + review + overnight validation each),
6 topkdump draws + 4 fixval + ~4 fixval2 draws. **Eliminated (for good):** tie-breaks, the scorer
gather, cache donation, and — critically — the entire read-formulation class.
**To skip:** rule (d) of §6 — when a failure is per-instance with clean data, hash the full state
including derived tensors **before** building fixes. The state-hash that ended the hunt cost ~15 min
per draw once built.

### Phase I — The entry-layer bisection and the root cause (07-22 09:50 → 07-23 08:10)

**The per-event histogram breakthrough (07-22 09:50, zero new data):** re-analyzing the *same* topkdump
artifacts per event instead of pooled: prefill keys **identical for evt00-03, divergent from evt04-20**
(decode divergent everywhere — feedback). Hidden states are instance-identical through ~L13-16 and
divergent from ~L17 ⇒ the fault enters in the **transformer layer compute at a fixed depth** — the
indexer/selection machinery was only the instrument that measured it, and both nulled fixes had
targeted the instrument.
**The layerscan (21:15):** `GLM_DCP_CACHE_DUMP_LAYERS=13-21` — k-caches L13/L15 identical (34/34
steps), L17/L19/L21 divergent (32/34; the 2 identical are kv=2048 first-chunk steps). Three clues:
entry bracketed L16-17; **position-gated at 2048** (dense-path tokens stay clean through divergent
layers); small onset (0.16) amplifying (5.2) through depth. Intersection ⇒ new prime suspect: the
**derived on-device state** (absorbed weights / per-layer prepped buffers, computed at init after the
checksum's coverage).
**The state hash — and its leaky first design, caught (23:25):** `GLM_STATE_HASH` (`8448b738c`) logs a
uint32 sum per final model-state leaf. Two init-only draws: 19,640/19,640 (host,leaf) sums identical —
derived state "exonerated"… except the design was leaky and the log caught it post-hoc: engines can
coincide in the same lottery state (the draw1≡draw2 precedent, ~1/3-1/2 odds), and init-only draws
never reveal which state they were in. **The rigorous rerun** (statepair): 3 *serving* draws pairing
each engine's leaf fingerprints with its measured score-state.
**THE ROOT CAUSE (07-23 08:10):** sick d3 differs from healthy d1 on exactly **16 (host,leaf) entries
= 2 tensors × 8 hosts**, both at `layers.10.self_attn.indexer`: `wk_weights_proj.weight` (loaded bf16,
sum 239851472 healthy vs 48387836 corrupt, identical on all 8 hosts) and its derived
`glm_dsa_adapted_wk` (sum 0). The healthy pair d1-vs-d2 differ on 3 leaves too — every engine carries
a few corrupt-loaded tensors; location/severity decides sickness. **The load class and the state class
were one bug**: the streamer delivers corrupt-but-finite bytes; NaN flavor ⇒ caught and refused;
finite flavor ⇒ invisible to NaN scans and to the H2D checksum. Every observation of the hunt
reconciles (per-instance-fixed discrete states, position gating, small-onset amplification, per-draw
entry layer, budget-dependent expression, health-probe blindness, gate2/gate3 deaths, mangled digits).
**Ground truth (09:40 → 11:30):** manifest refusal landed (`696adb9ca`); the first bootstrap engine
was **itself corrupt** at layer-10 (same sum, third sighting) ⇒ frequent victims ⇒ the majority-of-3 +
safetensors-ground-truth protocol; `ground_truth_sum.py` then confirmed the healthy sum bit-exactly
from the checkpoint and identified the corruption as **the dequantized-wk half zeroed** (§2.3).
**Cost:** ~26 h from histogram to adjudicated ground truth — the cheapest phase of the hunt, run
almost entirely on already-banked artifacts plus five short draws.

### Phase J — The loader window hunt and the self-healing load (07-23 11:30 → 07-24 01:15)

Root cause in hand, the owner rule ("no workaround gating — root cause properly fixed + validated
before any re-gate") forbade re-gating behind the refusal guard alone: refusal makes bad engines
visible, not rare (the pre-guard oobval arm refused 3/4 draws — a gate would starve). The load path
itself had to be fixed. Four candidate mechanisms fell in sequence:

**gcsfuse as the load path (deferred, not falsified):** Plan A — replacing streamer reads with a
gcsfuse mount — would have discriminated read-vs-dequant, but a FUSE-backed full load measured ~10×
too slow for engine init on this pod. Kept as the *repair* source, not the load path (below).

**The buffering-clone hypothesis (disproven):** `_try_load_fp8_indexer_wk` buffers the wk/scale
tensors across loader-iterator yields; if the streamer recycled those buffers, the fused leaf would
read freed memory. A `.clone()` at buffering time was applied on all 8 hosts — and changed nothing:
the runai iterator already yields `tensor.clone()` (`weight_utils.py:1076`). Banked as a null; the
patch (`patches/vllm-fused-indexer-wk-clone.patch`) is kept only as upstream-report context.

**The dequant-time window (excluded by instrument):** an OOB verify at the end of the fused
load/dequant (re-read + re-dequant if the wk half sums to zero) **never fired** on strike launches —
at that point the values are still good. This is the decisive narrowing: the zeroing lands in the
window **post-dequant / pre-t2j**, while the fused leaf sits in CPU storage awaiting conversion.
Leading candidate: `_free_cpu_storage`-class `resize_(0)` ordering (audit material for the upstream
report; the family matches the t2j alias race — a reference held across an async boundary).

**The fix that landed — repair at the window's closing edge:** since the corruption strikes while
the leaf is parked in CPU storage, the last CPU touch is where a repair is total:
`precompute_indexer_params` (PWAL), where the adapted indexer tensors are derived. The wk-oob guard
(`glm_dsa_indexer.py`, env `GLM_WK_OOB_DIR`, default-off) checks both halves of every fused indexer
leaf for the impossible all-zero signature and repairs bitwise from the gcsfuse checkpoint mirror
(offline dequant proven bit-exact vs vllm's `scaled_dequantize`), failing loud if unrepairable.

Two implementation traps worth recording. (1) The PWAL hook runs under BOTH of torchax's
interception modes (`XLAFunctionMode` and `XLADispatchMode`), and each alone breaks the safetensors
read's tensor construction: `DisableTorchFunctionSubclass()` failed on metal, then
`DisableTorchFunction()` failed on metal too — the dispatch mode still intercepts aten calls (one
wasted draw each). The working escape is torchax's own internal idiom, the PAIR
`mode_utils.no_dispatch(), torch._C.DisableTorchFunction()`. The deeper lesson: the CPU suite
passed 12/12 throughout because it never ran the guard under torchax's modes — once a test runs the
repair under `torchax.default_env()`, the old escape reproduces the *exact* metal error on CPU
(mutation-verified). Env-sensitive code must be unit-tested under the production interception
stack, not bare CPU. (2) The first validation arm burned 4 draws in ~75 s each — the
code-fingerprint guard (`GLM_EXPECT_CODE_HASH`) refusing init because worker 6's checkout was one
commit stale: its sync had failed on a stale `.git/index.lock` and the failure was eyeball-checked
past. The 07-09 stale-worker night took 8 hours to notice; the guard caught the recurrence in 75
seconds (`sync_workers.sh` now machine-enforces 8-host HEAD==origin, exit 2 on drift).

**Validation (gval_20260723T233335Z, PIN `dc0443a43`, REF + OOB armed, 4 draws):**

| draw | strike | outcome |
|---|---|---|
| 1 | zero-fill ×1 (layers.10, w3) | repaired → VERIFIED=8 → 4/4 needles |
| 2 | none | VERIFIED=8 → 4/4 needles |
| 3 | zero-fill ×3 (layers.1 ×1 + layers.10 ×2) | all repaired → VERIFIED=8 → 4/4 needles |
| 4 | zero-fill ×1 (layers.10, w4) + NaN ×1 (layers.1, w5) | repair fired; NaN half → `LoadNanCheckError` fail-closed refusal |

Five zero-fill strikes repaired across three draws (layers.10 — the frequent victim — struck four
times on four different hosts; per-host independence now observed directly), every serving engine
manifest-VERIFIED 8/8 byte-exact, 12/12 needles, zero unverified serves. The NaN-flavor draw is the
refuse path working as designed — the guard repairs the deterministic all-zero signature only; the
manifest and NaN scans keep everything else fail-closed.
**Cost:** ~14 h from adjudicated ground truth to a validated self-healing load.

---

## 4. Why it took 6 days — the honest analysis

### 4.1 The structural cause: the bug violated an implicit trust boundary

After 07-18 the stack had, at load time: a full-weight on-device non-finite scan (`GLM_LOAD_NAN_CHECK`),
an indexer-param scan (`GLM_PWAL_NAN_CHECK`), a CPU-side pre-t2j scan (`GLM_CPU_LOAD_NAN_CHECK`), and a
categorical CPU-vs-device byte checksum (`GLM_LOAD_CHECKSUM`). Each instrument was individually correct
and did exactly what it claimed. Jointly they created an implicit contract nobody had written down:
**"an engine that passes all four has verified weights."** False. The NaN scans verify non-finiteness
only; the checksum verifies the H2D transfer of bytes that may already be wrong. The finite flavor
slipped exactly between them — corrupt before t2j, finite everywhere. The gap was not entirely
undocumented: the 07-18 "08:50" entry states the scan's honest limit ("finite corruption is invisible"),
the "07:30" entry names the finite-corruption implication in full, and the 14:35 residual-specimen entry
lists "CPU-side finite corruption before t2j" as candidate (a), explicitly "unexcluded". What was never
written was the **joint coverage map** — and six hours after candidate (a) was written down, the gate3
death entry declared "weights categorically exonerated", an exoneration broader than any instrument's
documented coverage. That single sentence set the search space for the next four days.

### 4.2 Why the wrong branch looked right

- **The sibling defect was a near-isomorphic prior.** The pageloop-v4 report (a real, byte-diff-proven
  v4 lowering defect on the same buffer family, found by this same team in this same stack) matches the
  residual on almost every axis: silent, NaN-clean, byte-clean, per-instance via inherited state,
  accuracy-check-invisible. The corpus re-read — done by the rules, under the owner's own doctrine —
  ranked it the direct hit. The base rate of "we have hit this defect class before, here" legitimately
  dominated "the load path corrupts finitely", which had no specimen with a byte-level fingerprint yet.
- **The instrument measured the symptom far downstream.** The topk/score dumps sit at the end of the
  scorer chain. A corrupt weight at L10-L17 and a mislowered gather in the scorer produce the *same*
  observable there: divergent scores on (apparently) identical inputs.
- **Every cache-dump window missed the per-draw victim layer.** The ENOSPC-driven trim to slots
  {0,1,2,4} (07-19 04:15) meant the "identical caches" evidence covered only layers upstream of any
  victim; the later 13-21 window caught *that draw's* divergence but the victim moves per draw
  (statepair's was at L10). "Caches byte-identical" was always true and always incomplete.
- **The boundary evidence was ambiguous and the wrong reading was chosen.** "Divergence begins exactly
  when the read spans >1 physical page" and "divergence begins at pos ≥ 2048 (the dense-fallback/sparse
  split)" are the same boundary in this geometry. The first reading indicts the paged gather; the
  second indicts anything selection-dependent — including a corrupt indexer weight. The gather reading
  fit the prior; two fixes were built against it; both nulled.
- **The amplified repro changed variables.** The hunt ladder reproduced at ~5/5 serving draws vs the
  gate's ~1/5, via an unidentified amplifier (Guard-2 per-step snapshots, per-step dumps, or
  mixed-length churn — "identification deferred, exploitation first"). The choice to exploit was right
  on expected value, but it meant every arm ran with extra armed machinery and at 32K geometry, and two
  observations later had to be retracted as instrument artifacts (the 2.7× slowness = Guard-2 snapshot
  cost; the Guard-2 stripe trip = page-reuse false positive).

### 4.3 The reasoning errors, named

1. **Categorical exoneration beyond instrument coverage** (07-18 20:30, "weights categorically
   exonerated") — the pivotal error. The instruments licensed "H2D-verified and non-finite-clean",
   nothing more.
2. **Dropping a live branch without killing it.** Candidate (a) — CPU-side finite corruption — was
   written down at 14:35 and never refuted; it simply fell out of the narrative because its
   discriminator (a reference manifest against GCS truth) looked expensive to build, while the
   engine-state branch had a cheap-looking next experiment. The manifest-class instrument, when finally
   built (GLM_STATE_HASH), took hours to land and ~15 min per draw to run.
3. **Fixing downstream of the last measurement instead of bisecting upstream first.** Two fix cycles
   (onehot, un-donate) were built against sites downstream of where the divergence *entered*; the
   per-event histogram that revealed the entry point used zero new data and could have run on 07-21
   morning, before either fix.
4. **A false positive believed for four hours** (Guard-2 stripe trip) because it matched the expected
   shape of the leading hypothesis — conviction announced 06:55, retracted 10:40 by its own follow-up
   forensics. The system worked; the lesson is that a guard's false-positive modes belong in its
   documentation before its trips are believed.
5. **The early over- and under-claims around the t2j fix** (08:50 "did not work" → 09:00 "gate path
   open" → 09:15 review refutation) — pooling instrument-mismatched arms. Corrected within hours by
   the adversarial-review discipline, at the cost of whiplash.

### 4.4 The disciplines that worked (and demonstrably shortened the hunt)

- **Corpus-first (owner-enforced).** Found the t2j alias race by reading, unified it with DSV4's
  never-root-caused crash, and later produced the H1/H2/H3 ranking with explicit corpus gaps. The
  rule's one failure mode — a strong sibling prior can outweigh a weakly-instrumented true branch — is
  exactly §6(e).
- **One-variable A/Bs with pre-registered interpretation rules.** The X32 rule ("clean draws alone are
  ambiguous") was written before the data; the barrier arm was explicitly re-purposed as falsification;
  the concurrency A/B, v1/v2 arms, and write-barrier arm each eliminated exactly one thing.
- **Categorical-over-rate criteria.** The fix-validation criterion (two engines bit-identical dumps,
  "no rate statistics") is what made both nulls cheap and unambiguous — a rate-based validation would
  have cost 20 draws per fix and could have "passed" by luck.
- **Honest nulls, banked immediately.** onehot and un-donate were declared null the moment the numbers
  landed (1930 and 2030 vs baseline 1925 divergent keys), with the refuted mechanism recorded. No fix
  was defended.
- **Adversarial review as part of the loop.** Review 2/3 killed the "gate path open" overclaim and
  forced the checksum build; the discriminator's own review found the C-vs-D blind spot (scrambler
  interleave) and the N=14 power hole; the workflow audit's M5 liveness test made the X32 negative
  trustworthy.
- **Banking everything to GCS.** The root cause was found almost entirely on *banked* artifacts —
  the per-event histogram, the fingerprint comparison, the write-vs-read discriminator, and the
  statepair diff all ran offline on dumps archived draws or days earlier.
- **Pre-committed abort/protocol rules.** Gate3's 2-miss abort, the INFRA-vs-verdict taint ordering,
  and the health-probe relaunch all executed as designed; no verdict was polluted by infrastructure.

### 4.5 What a 20-minute check would have saved

Rule (d) of §6, applied at gate3's death (07-18 20:30): "the failure is per-instance; the data is
clean; hash the full final state — loaded *and* derived — across a sick/healthy pair." GLM_STATE_HASH
is ~40 lines on top of the already-existing leaf walker, ~15 min per instrumented draw. Run then, it
would have produced the layer-10 diff on the first sick/healthy pair — i.e., 07-19 — instead of
07-23. Estimated saving: ~4 days, two fix builds, and roughly 40 engine draws.

---

## 5. The fix and the protection

### 5.1 The categorical detector: manifest refusal (landed, `696adb9ca`, synced 8×)

`GLM_STATE_HASH` computes a uint32 wraparound sum per final model-state leaf (covering loaded weights
AND derived tensors — the absorbed `w_uk_t`/`w_uv` and adapted indexer params are provably in the
walk). `GLM_STATE_HASH_REF` verifies every (host, leaf) sum at load tail against a golden manifest and
**refuses to serve** on any mismatch (`StateHashMismatchError`; state-only/manifest-only leaves are
mismatches too). `GLM_STATE_HASH_WRITE` is the atomic per-rank bootstrap mode. 14 tests green. This
closes the gate path regardless of which load path is used: a refused load costs a relaunch retry,
never a bad gate depth.

### 5.2 The golden-manifest protocol (the 07-23 10:40 entry is authoritative)

Because frequent victims exist (the bootstrap engine itself carried the corrupt layer-10 value), naive
majority-vote can enshrine corruption:

1. Three WRITE-mode engines produce three candidate manifests.
2. Per-leaf majority across the three.
3. Any leaf that disagrees across the three, **or matches a known-corrupt sum**, gets its ground-truth
   sum computed offline from the GCS safetensors via ranged reads (`ground_truth_sum.py`, banked, with
   the fused/stacked name-mapping rules; replicated leaves sum directly, sharded leaves need the shard
   transform — done only for disputed leaves).
4. Golden = majority + ground-truth overrides → GCS.
5. A REF-mode validation draw must verify before the manifest is promoted.

The offline consensus scraped from statepair logs (859/~2455 leaves; log-line regex lossy) is banked as
PRELIMINARY ONLY.

### 5.3 The load-path elimination (attempted; demoted to repair source)

The plan was to eliminate GCS streaming from the serving load: gcsfuse Plan A at $0 first, per-host
local-disk copy as the fallback (both pre-authorized in CLAUDE.md §COST). Outcome (Phase J): a
FUSE-backed full load measured ~10× too slow for engine init, so gcsfuse serves instead as the
**out-of-band repair source** for §5.5; the streamer remains the load path, wrapped in the
repair+refuse stack. The local-disk fallback remains available if the strike rate ever outgrows the
stack.

### 5.4 The read-vs-dequant discrimination — resolved (Phase J)

**Neither.** A dequant-time zero-check saw good values on strike launches; the zeroing lands
post-dequant / pre-t2j in the fused leaf's CPU storage. The gcsfuse load-path discriminator became
moot for the gate (too slow to serve as the load path, and the window is now localized on the
consumer side). Still open for the upstream report: the exact free/ordering mechanism
(`_free_cpu_storage`-class `resize_(0)` audit), the NaN-flavor's 128-granule pattern, and how much
of the pre-fix 56% NaN rate the t2j fix truly removed (the pooling confound was never resolved;
moot for the gate path but material to the upstream t2j PR's claims).

### 5.5 The self-healing load: PWAL-time verify+repair (landed `dc0443a43`, synced 8×; validated)

At the last CPU touch of the fused indexer leaves (`precompute_indexer_params`), with
`GLM_WK_OOB_DIR` set, each leaf's halves are checked for the impossible all-zero signature and
repaired bitwise from the gcsfuse checkpoint mirror; unrepairable ⇒ raise (fail-closed, never
serve-and-hope). Byte-identical behavior when unset. Validated on metal (Phase J table): 5/5
zero-fill strikes repaired, every repaired engine then manifest-VERIFIED 8/8 and needle-perfect.
Defense in depth is: **repair (5.5) → refuse (5.1) → relaunch (orchestrator retry)**, in that
order. Known optional extension: also repairing non-finite (NaN-flavor) halves would convert those
refusals into serves; deliberately not done before GATE4 (no new code between validation and gate).

---

## 6. Catching this class in the future

The owner's key ask. These are generalizable rules, each grounded in a specific failure of this hunt.

**(a) End-to-end reference verification of EVERY derived artifact at trust boundaries.** At every
boundary where bytes are transformed or moved — load (storage→CPU), conversion (CPU→device), derivation
(weights→absorbed/adapted tensors) — verify against **ground truth**, not against a property (finiteness)
or against the previous stage (which may already be wrong). If bytes cross a boundary, the check is
"does the result equal what the checkpoint implies", computable as a cheap reference sum. Our NaN scans
checked a property; our checksum checked stage-N-vs-stage-N+1; nothing checked stage-N-vs-truth until
07-23.

**(b) Fingerprint-everything instruments as STANDING equipment.** A uint32 wraparound sum per tensor is
reduction-order-independent, costs seconds per host, needs no reference to be useful (cross-host,
cross-instance, cross-time diffs), and becomes categorical the moment a reference exists.
GLM_LOAD_CHECKSUM and GLM_STATE_HASH are ~small additions to an existing leaf walker. They should exist
from day 1 of any model-loading stack and be armed in every long campaign — the marginal cost measured
here was ~load-pass time (~23 min draws including everything). Addendum (07-23): the equipment pays
rent beyond its target bug — the code-fingerprint guard turned a recurrence of the 07-09
stale-worker night (a silent one-host sync failure) from an 8-hour data-poisoning event into a
75-second refusal. Standing guards catch the failure classes you did NOT predict; that is the
argument for leaving them armed, not merely buildable.

**(c) Cross-instance determinism as a first-class health check.** Two independently launched engines
given identical inputs must agree **bitwise** on selections, scores, and state sums. This is the single
sharpest instrument this hunt produced: it converts "quality lottery" into "byte diff". Run a
determinism pair **before any long campaign** — it would have flagged this stack before gate2 launched
(the 07-21 measurement found even two "healthy" engines differing on 1925/2310 score keys; that number
should have been 0 and is, itself, a refusal criterion). Caveat learned: instances can coincide by
luck (draw1≡draw2, ~1/3-1/2 odds here) — pair the determinism check with state fingerprints so
coincidence is detectable.

**(d) When a failure is per-instance with clean data, hash the FULL state FIRST.** Per-instance-fixed
behavior means some per-instance-fixed state differs. Before building any fix or any downstream
instrument, enumerate and fingerprint everything that is fixed at init: loaded weights, derived
tensors, caches, buffer metadata. It is a ~20-minute check per instance once the walker exists. This
hunt did it last; done first (at gate3's death) it saves ~5 days. The corollary: "the data I dumped is
clean" only exonerates the state you actually dumped.

**(e) Keep an explicit instrument-coverage map.** For every guard/check, a standing document line:
what it verifies, what it provably does NOT cover, and its known false-positive/false-negative modes.
Our four load-time instruments were each individually correct and jointly blind to
finite-CPU-side corruption; the blindness was noted in scattered log entries but never assembled where
a "weights are exonerated" claim would have to confront it. A claim of exoneration must cite the
coverage map, and any claim that exceeds it is an overclaim by definition. (Guard-2's page-reuse
false positive belongs on the same map — a trip is evidence only when its false-positive modes are
excluded.)

**(f) Suspect the loader before the kernel.** Loaders touch every byte exactly once, at init, off the
tested path, with no reference check by default; kernels are exercised millions of times and are
covered by parity suites. This hunt's priors ran the other way — reasonably, because a documented
sibling kernel-class defect existed — but the asymmetry stands: *silent per-instance weight-dependent
behavior should rank "the bytes are wrong" above "the compute is wrong" until the bytes are verified
against truth.* DSV4's flaky dequant crash was the same loader family, retry-mitigated and never
verified; the GLM port inherited both the bug and the blind spot.

**(g) Unit-test env-sensitive code under the PRODUCTION interception stack.** The wk-oob guard's CPU
suite passed 12/12 while the guard crashed on metal twice, because the tests ran bare-CPU and the
metal path runs under torchax's function AND dispatch modes, which rewrite tensor construction
inside third-party libraries (safetensors). Once one test entered `torchax.default_env()`, the bug
reproduced on CPU *exactly* (mutation-verified against both broken escapes). The rule: if code runs
under an interception/override regime in production — torch function/dispatch modes, custom device
contexts, import hooks — at least one unit test must run the full path inside that regime. "Passes
on CPU" is an overclaim when production CPU is not bare CPU.

---

## 7. Byproducts

The hunt paid for itself in equipment and in two upstreamable findings.

### 7.1 Instruments built (all env-gated, default-off, byte-identical when off)

| Instrument | Env / script | What it does |
|---|---|---|
| Indexer-param init scan | `GLM_PWAL_NAN_CHECK` | init-time NaN/Inf scan of orig+precomputed indexer params; raises on UPSTREAM non-finite |
| Full-weight init scan | `GLM_LOAD_NAN_CHECK` | on-device non-finite scan of all loaded weights at load_model tail (~5-15 s/host); reject dumps on refusal |
| CPU stage splitter | `GLM_CPU_LOAD_NAN_CHECK` | pre-t2j torch-tensor scan; splits streamer/CPU-stage vs H2D attribution |
| H2D byte verify | `GLM_LOAD_CHECKSUM` | uint32 CPU-vs-device sum per t2j-staged tensor; raises on mismatch (catches finite H2D corruption) |
| State fingerprints | `GLM_STATE_HASH` / `_REF` / `_WRITE` | uint32 sum per final model-state leaf (incl. derived tensors); manifest verify + fail-closed refusal |
| Selection/score dumps | `GLM_DSA_DUMP_TOPK` | traced-in callback dumping selected indices AND scores per event |
| Cache content dumps | `GLM_DCP_CACHE_DUMP_LAYERS` | per-step cache dumps, layer-selectable |
| Write-probe sentinel | `GLM_WRITE_PROBE` | startup sentinel through the real owner-scatters; refuse-to-serve on holes (both row widths) |
| Decode interpret probe | `GLM_DSA_DECODE_INTERPRET` | Pallas interpreter for the decode kernel (not viable at production shape — documented) |
| Read-barrier probe | `GLM_DSA_ATTEND_GATHER_BARRIER` | optimization_barrier before the payload take (device-OOMs at 128K — documented) |
| XLA decode attend | `GLM_DSA_DECODE_ATTEND=xla` | kernel-bypass serving path; CI-proven live (jaxpr pallas_call-free); also a mitigation candidate |
| One-hot scorer fetch | `GLM_DSA_SCORER_GATHER=onehot` | gather-free scorer page fetch, bit-exact; kept (harmless, gather-class-free) though null as a fix |
| Indexer un-donation | `GLM_DSA_IDX_CACHE_NO_DONATE` | excludes indexer caches from step-fn donation (~700 MB/shard); kept, null as a fix |
| Engine health probe | (gate scripts) | 2-chunk mini-needle + NaN scan at init; detect-and-relaunch (blind to the finite class — documented) |
| Orchestrators/ops | `probe_lottery.sh`, `hunt_residual.sh`, `loader_ab.sh`, `dissect_load.sh`, `gate_sparse128k.sh`, `disk_watchdog.sh`, `dump_archiver.sh` | scrambled discriminator, ladder hunt with INFRA taxonomy, init-only A/B loops, dissection pattern, gated gate driver, disk guard, quota'd GCS archival |
| Ground truth tool | `scratchpad/ground_truth/ground_truth_sum.py` | offline leaf-sum from GCS safetensors incl. bit-exact fp8-dequant replication + name-mapping rules |

### 7.2 The t2j fix (real, distinct, upstreamable)

Fork `629c20e84`: eager copy in both t2j branches + 4 direct-import bypass reroutes + the alias-free
handoff invariant test (3-fail→10/10 must-fail-first proof). Adversarially reviewed: the GLM-5.2 vLLM
load chain genuinely severed; 4+ sibling alias sites identified off the GLM path (parked on
`~/wt-sibling-alias`). The workflow audit's cut plan: squash + cherry-pick onto current main, cite
torchax's own TODO(gxd3). Its on-metal rate-collapse claim needs a clean, instrument-matched
measurement before the PR asserts one (§5.4).

### 7.3 The filable upstream bugs

1. **The streamer/dequant zero-fill** (this bug): deterministic corrupt bytes — the fp8-wk half of a
   fused-loaded tensor delivered as zeros, with a bit-exact ground-truth reference and a
   known-frequent-victim tensor. Filable with a strong repro once the gcsfuse discriminator assigns
   the component (runai-model-streamer vs the fused-dequant load path).
2. **The t2j alias race** (fix PR + report; §7.2).

(The pageloop-v4 sublane-drop report, staged before this window, is the sibling defect class — already
packaged in `docs/upstream/`.)

### 7.4 The elimination-methodology record

The refuted-hypotheses ledger, with the instrument that killed each: disk pressure (clean-disk
discriminator), per-host binaries (executable fingerprints), PWAL copies (code map), runtime clobber
(per-step timeline), scale-tensors-as-primary (geometry correction), streamer concurrency (16-init
A/B), t2j-as-primary (post-fix corrupt draw + matched-pool stats), process-index permutation (npz
metadata), mesh/actor order (Guard 1 clean), CPU boundary math (miss-geometry falsification suite),
write path (stripe forensics + barrier A/B), Mosaic decode kernel (X32 sick-with-xla), the v2
transforms as sole cause (v1 sick), compile variance (fingerprint sets), tie-breaks (0 tie flips),
the scorer gather (onehot null), cache donation (un-donate null), derived-state corruption
(state-hash… provisionally — then *un*-refuted in its loaded-weight form by statepair). Eighteen
eliminations, every one banked with its evidence in the log.

---

## Appendix A — Timeline table

File order (authoritative); timestamps as written in the log. † marks the 07-18 timestamp anomaly
(Appendix C, item 1).

| Date/time (UTC) | Event | Verdict / consequence |
|---|---|---|
| 07-16 23:55 | VM lost + full recovery (pod recreated, 8 disks wiped) | everything committed survived; 14-probe experiment must restart |
| 07-17 04:15 | Safety/ops-debt commit landed (845f4ffeb); discriminator redesigned by its own review | scrambler interleave, N=20, taint ordering |
| 07-17 03:15 | Draw-1 INFRA: orphaned Ray firewall rule | fixed; classifier worked; standing recreation-checklist rule |
| 07-17 08:20 | Lottery caught on camera: per-host NaN of layer-1 indexer k-cache | p1 {w4} correct, p2 {w3,w4,w7} miss; replica-severity arithmetic |
| 07-17 08:55 | Log-fingerprint forensics | per-host binary split REFUTED; per-host runtime state |
| 07-17 09:50 | Draw-3 INFRA: git index.lock race | agents must not run git on worker checkouts |
| 07-17 16:50 | Discriminator stopped: 6 valid, 4 MISS (67%) | p5 = clean-slot miss (finite flavor, unrecognized); w4-poison outcome not deterministic |
| 07-17 17:40 | PWAL-copy hypothesis refuted at code level | surviving: garbage input vs runtime clobber |
| 07-17 19:10 | PWAL flags at INIT: loaded wk arrives NaN/Inf | ROOT-CAUSE CLASS = the weight load |
| 07-18 06:40† | Loader recon: streamer has zero integrity machinery; counts are 128-multiples | detect→refuse is the only integrity layer |
| 07-18 07:30† | Per-step timeline verdict: static load-corrupt wk | page-0-clean was an off-by-one; FINITE-CORRUPTION IMPLICATION named |
| 07-18 08:50† | GLM_LOAD_NAN_CHECK landed (c68794241); concurrency A/B launched | scale-as-primary corrected; honest limit noted (finite invisible) |
| 07-18 04:20 | Owner course-correction: corpus-first, dissect-don't-rate | plan change: the dissection run |
| 07-18 05:30 | t2j alias race found by reading | fix class: sever the alias |
| 07-18 06:10 | Prior-art sweep: DSV4 flaky-dequant crash recovered | one mechanism, two manifestations (~60% rate match) |
| 07-18 06:35 | Concurrency A/B complete: 9/16 corrupt, both arms | streamer concurrency exonerated |
| 07-18 07:05 | KICKOFF rewrite | force-push violation logged (honest) |
| 07-18 08:05 | t2j fix landed (629c20e84) | must-fail-first proof 3-fail→10/10 |
| 07-18 08:50 | Validation draw 3/3 corrupt | "fix did not collapse rate" (pre-registered rule) |
| 07-18 09:00 | Fuller pool: 1/9 vs 9/16, p≈0.036 | "~10% residual; gate path open" (overclaim) |
| 07-18 09:05 | Review 1/3 | t2j fix safe on GLM path; class-closure claim refuted (4+ sibling sites) |
| 07-18 09:15 | Review 2/3 | "gate path open" REFUTED; finite-garbage un-attributability; build GLM_LOAD_CHECKSUM |
| 07-18 09:25 | Review 3/3 | CPU splitter safe; M1 coverage hole; checksum subsumes |
| 07-18 09:40 | Xprof 128K needle | no ≥40% dominator; gate proceeds |
| 07-18 09:55 | GLM_LOAD_CHECKSUM landed (a225d16b4), draw 1 clean | verified=1882/host, 312 benign skips |
| 07-18 11:05 | Validation 4/4 clean | GATE3 LAUNCHES (n=77) |
| 07-18 14:35 | Gate d=0.05 try-1 sick: THE RESIDUAL SPECIMEN (db 193) | byte-verified clean everywhere; candidates (a) CPU-finite / (b) engine state |
| 07-18 19:15 | d=0.95 cleared 11/11 (gate2's killer cell) | 33/33; gate2 retroactively attributed to the lottery |
| 07-18 20:15 | Gate miss #1 (d=1.0 t=0) | state-class signature on a probed engine |
| 07-18 20:30 | GATE3 DEAD 33/35 (d=1.0 0/2) | "weights categorically exonerated" — the pivotal overreach |
| 07-18 20:50 | Miss-dump forensics | process-index permutation refuted |
| 07-18 21:05 | Hypothesis: unwritten-slot read × uninitialized HBM | CPU test planned at miss geometry |
| 07-18 21:35 | Corpus verdict: H1 pageloop-read, H2 mesh order, H3 selection | hunt orchestrator built |
| 07-18 22:15 | CPU audit at miss geometry: all falsifications pass | boundary math exonerated; hunt v1 draw 1 LOAD_REFUSED |
| 07-19 00:55 | Dissect verdict: CPU-flagged pre-t2j | NaN load class = streamer/CPU stage; GCS-elimination pre-authorized fix |
| 07-19 04:15 | Hunt draws 2-3 ENOSPC | dump trim to slots {0,1,2,4} (later consequential) |
| 07-19 06:55 | Guard-2 trip: "DCP stripe write-loss" (layer-0 indexer k_cache) | state class "caught and named" — premature |
| 07-19 07:15 | Localization: DSA owner-scatter (flat) at gate geometry | fix plan: barrier probe |
| 07-19 09:45 | Flat arm 2/2 sick; slowness signature retracted | barrier arm launched |
| 07-19 10:40 | CORRECTION: caches CLEAN; Guard-2 trip = page-reuse false positive | write conviction retracted; shift to decode READ |
| 07-19 12:45 | Barrier arm still sick, zero trips | write path exonerated by A/B |
| 07-19 13:20 | Read probes landed (interpret + gather-barrier) | interpret arm running |
| 07-19 15:30 | Interpret arm init-dies (OOM/SIGSEGV) | not viable at production shape |
| 07-19 16:50 | GLM_DSA_DECODE_ATTEND=xla landed | the kernel-bypass discriminator |
| 07-19 17:50 | Gather-barrier arm init-dies (~29G/chip copy) | xla-attend arm is the discriminator |
| 07-19 19:45 | Three arms, three identical init deaths | control draw to split the confound |
| 07-19 20:30 / 21:10 | PR-playbook + workflow audits (48/51 confirmed) | M5 liveness test etc. land as a10d2a426 |
| 07-19 22:30 | Control healthy at probe PIN | deaths were variant 128K-shape compiles; max_len precompile lesson |
| 07-19 23:00 | X32 running; interpretation rule pre-registered | clean-alone = ambiguous, declared in advance |
| 07-20 00:40 | X32 draw 1 SICK with xla attend | Mosaic decode kernel EXONERATED; amplified repro noted |
| 07-20 03:15 | v1sel 3/3 then benign trip | Guard 2 disarmed for A/B arms; amplifier confound sequenced |
| 07-20 09:15 | v2-control (no Guard 2): 1 sick/3; mangled-needle specimen | repro survives Guard-2 removal |
| 07-20 14:10 | v1 sick too (2nd mangled specimen) | v2 transforms exonerated as sole cause; precamp arm launched |
| 07-20 17:40 | Executable fingerprints identical sick-vs-healthy | compile variance refuted; TOPKDUMP arm launched |
| 07-21 09:25 | Diff pair complete (draw 6 sick with dumps armed) | observer effect refuted |
| 07-21 11:45 | THE MECHANISM MEASURED: discrete per-engine score states | tie-break refuted (0 flips); needle 3.2/42 vs 41/42 |
| 07-21 12:30 | CPU scorer logic exonerated by falsification | ranked metal sites; write-vs-read offline |
| 07-21 14:50 | WRITE exonerated / READ convicted (290,530 vectors, 0 diffs) | onehot fix designed; categorical criterion |
| 07-21 16:10 | onehot landed (473904510) | zero-gather scorer fetch |
| 07-21 17:20 | Fix review: safe-for-validation | 2 MAJOR caveats, neither in gate regime |
| 07-21 21:20 | HONEST NULL: onehot (1930 vs 1925 divergent) | gather exonerated; donation the surviving suspect |
| 07-21 23:55 | Un-donate fix landed (d7ad7963b) | overnight categorical validation |
| 07-22 00:20 | Ops: fixval2/fixval collision (sequencing error) | no data lost; standing pgrep rule |
| 07-22 09:50 | SECOND NULL (2030/2310); PER-EVENT HISTOGRAM: identical evt00-03, divergent evt04+ | ENTRY LAYER in transformer compute; indexer was only the instrument |
| 07-22 21:15 | lscan3: entry bracketed L15→L17; position-gated ≥2048; onset 0.16→5.2 | prime suspect: derived on-device state |
| 07-22 23:25 | State-hash init pair: 19,640/19,640 identical | leaky design CAUGHT (coincidence odds); statepair rerun |
| 07-23 08:10 | ROOT CAUSE: statepair — 16 (host,leaf) diffs, layers.10 indexer wk 239851472 vs 48387836; adapted_wk = 0 | load class + state class = ONE BUG (streamer finite corruption) |
| 07-23 09:40 | Manifest refusal landed (696adb9ca) | bootstrap = WRITE-mode draw |
| 07-23 10:40 | Bootstrap engine itself corrupt (48387836, 3rd sighting) | frequent victims ⇒ majority-of-3 + safetensors ground truth |
| 07-23 11:30 | GROUND TRUTH: true sum 239851472 = 191463636 + 48387836 | corruption = dequantized-wk half ZEROED; read-vs-dequant open |

## Appendix B — Arm/draw ledger

Engine launches (init or serving) in the window, by arm. "Sick" = real misses; INFRA/LOAD_REFUSED as
classified by the orchestrators. Durations approximate where the log gives them.

| Arm / loop | When | Draws (outcomes) | Cost | What it bought |
|---|---|---|---|---|
| probe_lottery discriminator | 07-17 | 8 probes + 8 scramblers: 2 CORRECT, 4 MISS, 2 INFRA; 2 scramblers missed at 32K | ~13.5 h; 2×3.7 GiB archives | lottery real at ~67%; per-host NaN census; p5 finite-flavor specimen |
| probe_timeline (PWAL armed) | 07-17 | 3 pairs (6 engines): pairs 1-2 correct, pair-3 probe MISS with init flag | ~4 h | load class discovered at init |
| loader concurrency A/B | 07-18† | 16 init-only draws: 9 corrupt (c=32: 3/8; c=8: 6/8) | ~4 h | streamer concurrency exonerated; 56% baseline |
| t2j validation + dissection | 07-18 | 3 + 6 draws: 1 corrupt total | ~half day | rate drop measured (pooling-confounded); CPU tensors clean on instrumented draws |
| checksum validation | 07-18 | 4 init draws, all clean | ~23 min/draw | zero false positives in ~60K tensor checksums |
| xprof needle | 07-18 | 1 (correct) | ~40 min | no ≥40% dominator; gate cleared to launch |
| GATE3 | 07-18 11:05→20:30 | 5 engine draws, 35 needles: 33 correct, 2 miss, 1 sick engine caught by probe | ~9.4 h; 2 depths burned | the residual specimen (db 193); gate2 cell cleared 11/11; gate dead 33/35 |
| hunt v1/v3 | 07-18/19 | 3 draws: 2 LOAD_REFUSED (one = the CPU-side dissect verdict, 6563 s), 1 ENOSPC at 4/6 | overnight | NaN load class attributed to the streamer/CPU stage |
| hunt v4 (flat arm) | 07-19 | 2 draws, both SICK with Guard-2 trips | ~2 h/draw + forensics | the false-positive conviction, then (via stripe forensics) caches proven CLEAN |
| barrier (write) arm | 07-19 | 1 draw: SICK, zero trips (disk-guard kill late) | ~2 h | write path exonerated by A/B |
| interpret arm | 07-19 | 1 draw: INFRA (init death ~2 h) | ~2 h | none (inconclusive-by-infra) |
| gather-barrier (read) arm | 07-19 | 1 draw: INFRA (init death, 104 compiles, 2.3 h) | ~2.3 h | none (device-OOM class identified) |
| xattend (128K) arm | 07-19 | 1 draw: INFRA (init death ~2 h) | ~2 h | none directly; triggered the control |
| control draw | 07-19 | 1: HEALTHY through 32K | ~2 h | probe commits exonerated; max_len precompile lesson |
| X32 (xla attend, 32K geometry) | 07-20 | 1 draw: SICK 0/6 | ~25 min repro | Mosaic decode kernel exonerated; amplified repro |
| v1sel (Guard 2 on) | 07-20 | 1 draw: 3/3 then benign sanity trip | ~1 h | v1 teaser; Guard-2 disarm decision |
| v2-control (no Guard 2) | 07-20 | 3 draws: 1 SICK, 2 healthy | ~30 min/draw | repro survives Guard-2 removal; mangled specimen #1 |
| v1b (no Guard 2) | 07-20 | 2 draws: 1 healthy, 1 SICK | ~30 min/draw | v2 transforms exonerated; mangled specimen #2 |
| precamp revert arm | 07-20 | 2 draws: 1 LOAD_REFUSED, 1 unfinished (retired) | ~2 h | superseded by the dump-diff |
| topkdump arm | 07-20/21 | 6 draws: 4 healthy, 1 LOAD_REFUSED, 1 SICK | overnight; 15 GB dumps | THE MEASUREMENT: discrete score states; tie-break refuted; write-vs-read decided offline |
| fixval (onehot) | 07-21/22 | 4 draws: 2 healthy, 1 LOAD_REFUSED, 1 INFRA (collision) | ~9 h cycle | honest null #1 |
| fixval2 (no-donate + onehot) | 07-22 | ~4 draws (verdict from the pair) | ~10 h cycle | honest null #2; per-event histogram ran on its artifacts |
| lscan (layers 13-21) | 07-22 | 2-3 draws (verdict "lscan3") | ~day incl. analysis | entry bracketed L15→L17; position gating; onset amplification |
| state-hash init pair | 07-22 | 2 init draws (~15 min each) | ~1 h | leaky null (caught) |
| statepair (serving + hashes) | 07-22/23 | 3 draws: 2 healthy, 1 SICK | overnight | THE ROOT CAUSE (16-leaf diff at layers.10) |
| manifest bootstrap | 07-23 | 3 WRITE-mode draws (1 done — itself corrupt; 2 launched) | in progress | frequent-victim discovery; golden protocol |
| oobval (REF armed, pre-guard) | 07-23 | 4 draws: 1 clean-VERIFIED (4/4), 3 refused | ~1.5 h | strike rate ~75% that night; dequant-time check never fired = the window-narrowing datum |
| gval false-start #1 (Subclass escape) | 07-23 | 1 draw: PWAL crash (UntypedStorage) | ~20 min | DisableTorchFunctionSubclass insufficient |
| gval false-start #2 (stale w6) | 07-23 22:28 | 4 draws: 4× CodeFingerprintMismatch in ~75 s | ~5 min | the fingerprint guard catching the index.lock sync failure |
| gval false-start #3 (function-only escape) | 07-23 23:11 | 1 draw: guard DETECTED strike, OOB read crashed (dispatch mode) | ~17 min | DisableTorchFunction alone insufficient; detection proven on metal |
| **gval (the validation)** | 07-23 23:33 → 07-24 01:15 | 4 draws: 3 serving (5 zero-fill repairs, VERIFIED=8, 12/12 needles), 1 NaN-refused | ~1.7 h | **the self-healing load validated; per-host independence observed directly** |

Total: ~115 engine launches, ~166 h wall clock 07-17 03:15 → 07-24 01:15, plus two killed n=77 gates
(gate2 pre-window, gate3 in-window) attributable to the same bug.

## Appendix C — Record discrepancies and open items (flagged, not smoothed over)

1. **The 07-18 timestamp anomaly.** The entries stamped 07-18 06:40, 07:30, and 08:50 ("GLM_LOAD_NAN_CHECK
   landed … the A/B fires") appear in the file BEFORE the 04:20 owner-course-correction entry, and
   causality agrees with file order (the 08:50 entry *launches* the concurrency A/B whose results the
   04:20 entry cites). Most consistent reading: the trio was written late 07-17/overnight and misdated
   07-18. File order is treated as authoritative throughout this document.
2. **"6-day" vs "5-day".** This post-mortem's window (07-17 03:15 → 07-23 11:30) is ~6.3 days; the log's
   07-23 08:10 entry says "the 5-day hunt" (counting from the lottery's capture at 07-17 08:20). Both
   are stated here; neither is adjusted.
3. **8/8-host replication vs per-host independence.** 07-23 08:10 infers "single upstream read (or
   broadcast)" from the corruption being identical on all 8 hosts — but 10:40 records the same corrupt
   value on 1/8 hosts (statepair d2), and the NaN-flavor host sets were clearly independent. Since the
   corrupt value is deterministic (zero-fill), independent per-host events would also produce identical
   bytes wherever they hit. The broadcast inference is therefore weaker than the entry implies.
   **Resolved 07-24:** the gval validation arm observed strikes on single distinct hosts per draw
   (w3; w4+w5; two hosts in draw 3) with the identical deterministic value — per-host independent
   events, not a broadcast; the 07-23 08:10 8/8 case was all hosts drawing the same deterministic
   corruption in one launch.
4. **The t2j fix's true contribution.** 56% (9/16) pre-fix vs ~10-15% post-fix NaN-refusal rates were
   never instrument-matched (review 2/3: matched-only p=0.46). With the streamer zero-fill now known to
   exist independently, how much of the drop the t2j fix caused is unresolved. Relevant to the upstream
   PR's claims only.
5. **The 07-17 2.6× slowness.** The hunt-era "sick engines ~2.7× slower" was retracted as Guard-2
   snapshot cost (07-19 09:45), but the original 07-17 08:20 observation (probe2 3690 s vs probe1
   1394 s/needle) predates Guard-2 arming and is left unexplained (plausibly NaN-arithmetic slow paths;
   never adjudicated).
6. **Benign-victim leaves vs score-state identity.** 07-23 08:10: statepair's healthy d1-vs-d2 differ on
   3 (host,leaf) sums yet both served correctly; the log does not record whether their *score states*
   coincided, nor which tensors the 3 benign leaves were. "Benign victims don't touch the measured
   paths" is the implied but unverified reconciliation.
7. **The evt→layer mapping vs the statepair victim.** lscan inferred indexers on odd layers (evt_j =
   layer 2j+9); the statepair victim is `layers.10` (even). The mapping was flagged "robust under both
   mappings" for its anchor but was never reconciled with the layer-10 sighting. Per-draw victim
   location makes this benign, but the mapping detail is unresolved.
8. **The amplifier's identity.** The hunt-config repro rate (~5/5 with Guard 2 + dumps; ~1/3 without
   Guard 2; gate config ~1/5) was exploited but never attributed (07-20 00:40: "identification
   deferred"). Under the final mechanism, per-launch victim draws should be config-independent —
   the apparent rate differences are small-n and/or expression-sensitivity (mixed-length ladders probe
   more budget regimes), but this was never closed out.
9. **Warm-cache behavior was inconsistent across the week** (07-17: a pure persistent-cache-hit engine;
   07-20: "cache hit rate 0.0%", recompiles every draw, "cause unknown, noted not chased"). Never
   resolved; fingerprint determinism made it moot for attribution.
10. **HANDOFF.md and CLAUDE.md §Progress are stale** relative to this arc (frozen at 07-18 19:45
    mid-gate3 and at the checksum era respectively); the RESEARCH_LOG is the only complete record of
    07-19 → 07-23. This document is written from the log.

