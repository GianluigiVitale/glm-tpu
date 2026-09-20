# GLM-5.2 native MTP / speculative decoding

Owner activated the current-model experiment on 2026-09-19. GLM-5.3 remains out
of scope. Baseline is the trained D1/D8/D10/D4 candidate: 138.85 prompt tok/s and
14.04 wall decode tok/s, with the short DB610 parity and timing boundaries in
[the paired receipt](tpu-real-request-loop-20260919T192804Z.json).

## Source feasibility established

The retained authenticated inventory has **1,569 layer-78 tensors totaling
10,032,632,960 bytes**, in canonical shards 136, 137 and 138 of 141. Current
GCS generation/size/CRC metadata matches the sealed source record; the three
headers were reread at those exact generations and compared with the recorded
header hashes, names, shapes, dtypes and offsets. The inventory fingerprint,
config and model index also match the sealed runtime provenance.
[Audit receipt](mtp-source-audit-20260919.json).

Layer 78's transformer tensor schema equals full-index target layer 74's,
plus BF16 `eh_proj`, `enorm`, `hnorm` and `shared_head.norm`. It has 256 complete
routed experts and their FP8 scales. Embedding and output projection use the
target's shared tables. The existing packed WS32 checkpoint contains no layer
78 tensors. This establishes retained source availability; it is not a fresh
weight-payload hash, a loaded draft model or an HBM-admitted implementation.
No new tensor payload has been acquired in this experiment.

## Forward-path reference

Inspected upstream vLLM at `36fa72d2d0d2f86c7c83e1e99c9012b7bd26463b`, including
the model registry, speculative-config mapping and generic/NVIDIA DSA MTP paths.
`glm_moe_dsa` selects the DSA MTP architecture. The draft combines normalized
next-token embedding with normalized previous target/draft hidden state through
`eh_proj`, executes its own transformer/cache, and uses its own final norm with
the shared vocabulary head. Recurrent drafting recycles **post-final-norm**
hidden state; applying that norm twice or recycling the pre-norm residual is
incorrect. The first draft iteration computes its own DSA shortlist; subsequent
iterations can reuse it under `index_share_for_mtp_iteration`.
[Generic MTP implementation](https://github.com/vllm-project/vllm/blob/36fa72d2d0d2f86c7c83e1e99c9012b7bd26463b/vllm/model_executor/models/deepseek_mtp.py),
[DSA MTP implementation](https://github.com/vllm-project/vllm/blob/36fa72d2d0d2f86c7c83e1e99c9012b7bd26463b/vllm/models/deepseek_v32/nvidia/mtp.py).

The current prefill API returns only its last token and state. A correct MTP
prefill must additionally obtain the target hidden states needed to populate
the draft cache, with the correct token/position shift. Starting a draft with
an invented empty prompt cache would not establish the intended MTP behavior.

## Verifier and acceptance work in progress

`glm_tpu/perf/speculative_verify.py` proposes several target input rows in one
layer-major pass. Attention currently retains one-row D8/D10 arithmetic and
causal cache writes inside each layer; the MLP pools rows into existing FP8
expert panels with the ordinary decoder's BF16 route accumulation. This avoids
scanning the entire decoder for each proposed token. It is not yet TPU-proven
or timed and may need further batching to make verification economical.

The proposal includes each row's target prediction, residual, normalized hidden
state and selection metadata. A separate prefix commit restores rejected rows
from the original physical caches, then advances only the accepted input span.
It refuses invalid counts or failed all-owner health atomically. This proposal
API is internal; arbitrary caller-supplied proposals are not authenticated.

First CPU32 comparison on the eight-layer fixture matched all three target
predictions but **failed bitwise residual equality**: 2,228 of 3,072 elements
differed, maximum absolute error 0.0546875. Isolated MLP comparison also finds
matrix-shape differences: dense and sparse maximum errors 0.001953125 on the
chosen synthetic inputs. These failures are preserved as diagnostic evidence;
the verifier is not numerically admitted. Do not cite a bitwise model proof or
speedup. The first differing cache layer is layer 1, after the first dense MLP;
layer 0 cache writes match bitwise.

The optimized CPU HLO exposes one concrete cause: the one-row down projection
fuses the activation's final multiply in FP32, removing a BF16 round which the
batched dot retains. `xla_allow_excess_precision=false` makes the isolated sparse
MLP bitwise equal and leaves one dense output rounding difference, but the full
eight-layer model still differs. Keeping dense calls separate also reduces but
does not eliminate propagation. No compiler flag or frozen numerical body was
changed. XLA permits removal of precision-changing conversion pairs; this
diagnosis does not establish TPU numerical behavior.
[XLA precision option](https://github.com/openxla/xla/blob/main/xla/xla.proto).

The CPU model comparison now explicitly characterizes this boundary: all three
target predictions agree; residual relative L2 error is 0.007373, active KV error
is at most 0.006264, and active index-cache error is at most 0.004523. Two DSA
selection positions exchange order on the second row; the selected key set is
unchanged on this short prefix. These are measurements on one synthetic fixture,
not general bounds. Test tolerances are regression envelopes, not exactness
admission. Trained-weight output agreement and longer-context cuts remain open.
[CPU receipt](mtp-cpu-verifier-boundary-20260919.json).

Changing later draft tokens leaves the first row's prediction, residual and
KV/index writes bitwise unchanged in that graph. Invalid draft IDs and an
out-of-capacity span refuse without committing. An independent physical-cache
oracle passes for every prefix of five proposed rows across owner and page
boundaries, with permuted physical pages, EOS/budget truncation, invalid counts
and a single-owner health failure. These cache-control results are bitwise proofs
within their scope; they do not make the model arithmetic bitwise equal.

`speculative_accept.py` implements greedy prefix acceptance plus a target
correction/bonus token, truncated at EOS or the remaining output budget. Its
two CPU tests pass across every rejection/EOS/budget combination for 1/2/3/5
verification rows and invalid-input refusal (2.20 s). This is acceptance logic
only, not a complete host session, draft-quality result or performance result.

`tools/perf_speculative_verify.py` prepares a full 78-layer synthetic comparison
of ordinary D1/D8/D10 and 2/3/5-row verification, including fresh graph identity,
research HLO checks, per-chip memory admission, prefix commit, numerical reports
and optional all-host device traces. Dispatch/completion timing excludes drafts,
fleet votes and delivery; any reported rate is a perfect-acceptance estimate.
The synthetic prefix is explicitly 64 zero-cache positions. The first run is
now complete; its negative result is recorded below.

Validation before the first acquisition: six focused CPU tests pass (acceptance,
physical commit, numerical characterization and benchmark comparison guards).
`tools/check_release.py` passes, including 524 tests with one skip; the frozen
source pin and isolated package check are unchanged. The curation ledger has no
unresolved rows. These are research readiness checks, not serving promotion.

The first acquisition uses the documented CPU numerical boundary. Trained-weight
target agreement is required before promotion. Draft-cache implementation and
paired speculative request comparisons remain outstanding.

## Wider CPU comparison and draft-state contract

The one- and two-row CPU cases also match target predictions and stay inside
the existing numerical regression envelope. The five-row case **fails that
envelope**: residual max absolute error 0.171875, relative L2 0.019374; final
active KV max error 0.152344 and index max error 0.099365. All five predictions
still match, and causal independence/rollback/refusal checks pass. The threshold
has not been widened. The five-row test records an expected qualification failure;
five-row timing is diagnostic only until its numerical behavior is resolved.
[Five-row negative receipt](mtp-cpu-five-row-rejection-20260919.json).

The upstream proposer shifts token IDs while retaining target positions and
refreshes accepted draft-cache rows using target hidden states. This rules out
reusing recurrent draft cache entries solely because their IDs were accepted.
The explicit position/refresh contract and preliminary 388 MB/chip weight estimate
are in [MTP state contract](MTP_STATE_CONTRACT_20260919.md). No MTP weight payload
has been loaded or native drafter executed.

## First full-target TPU economics: rejected

Run `perf_mtp_verifier_20260919T214702Z`, immutable source `844d1475`, completed
on all 32 chips with fresh graph consensus, research HLO checks and memory
admission. All eight hosts were authenticated idle afterward. Direct SSH ran
each workload exactly once; gcloud's automatic workload retry wrapper was not
used. The initial controller attempt exited before any SSH because the cron
sync lease was held; the successful controller waited for that lease.
[Eight-rank receipt](tpu-mtp-verifier-20260919T214702Z.json).

| Synthetic target work | Perfect-acceptance estimate | Relative to ordinary model calls |
|---|---:|---:|
| 2 verifier rows + commit | 14.9451–14.9456 tok/s | 0.984x |
| 3 verifier rows + commit | 16.9301–16.9889 tok/s | 1.115–1.118x |
| 5 verifier rows + commit | 19.2859–19.2868 tok/s | 1.270x |

Ordinary synthetic decode takes 65.837–65.841 ms. These estimates exclude draft
generation/refresh, rejected work, host votes and delivery; they are **not
measured speculative throughput**. Every tested size disagrees with sequential
target predictions and selected positions. All outputs are finite, which does
not establish correctness. No variant is admitted and the trained 14.04 wall
tok/s baseline is unchanged.

An initial rank-0 trace of two-row verification attributes about 62.1 ms to
collectives and 25.1 ms to routed FP8 panels. The largest collective signature
is the routed-expert reduction (31.4 ms, including owner imbalance/wait), followed
by selected-KV exchange inside the per-row attention loop (14.2 ms). This is
one host's instrumented trace, not the unprofiled wall timing or a fleet average.
The subsequent complete fleet trace analysis is recorded below.

Next candidate: keep the ordinary decoder's routed-plus-shared feature reduction
and per-row expert-reduction geometry while pooling routed weight panels, and
replace serial attention work with a causally masked batched construction.
These changes require their own CPU and TPU comparisons. MTP drafting cannot
repair target-verifier disagreement or make a losing perfect-acceptance bound
profitable, so it remains deferred until the verifier is usable.

## Fleet trace and second verifier candidate

All 32 host/variant trace summaries were collected successfully. Each of four
variants has eight hosts, 64 core planes and two calls per core. Mean device
step times under profiling are 67.85 ms ordinary and 138.96 / 178.42 / 257.74 ms
for 2/3/5 rows. Mean collective self-time is 36.16 / 60.78 / 74.82 / 109.78 ms;
FP8/custom-call time is 12.14 / 27.08 / 39.54 / 63.55 ms. Collective times include
owner waits. These traces explain the negative economics; they are not output
wall-throughput measurements. [Fleet trace receipt](tpu-trace-mtp-verifier-20260919T214702Z.json).

The opt-in `canonical_mlp` keeps the ordinary D1 routed-plus-shared feature sum
and per-row expert sum while pooling the routed weight panels. Its isolated
CPU32 comparison passes bitwise on all shards at 2/3/5 rows, both concentrated
and spread/reordered routes; malformed routes and weights refuse. However,
the complete five-row verifier still fails the existing numerical envelope:
residual max error 0.171875, relative L2 0.020100. All five predictions match
in this small fixture; that does not qualify it. Mapping RMSNorm separately per
row produced the same errors and was not adopted.
[Canonical-MLP rejection](mtp-canonical-mlp-cpu-rejection-20260919.json).

The separate opt-in `batched_attention` prepares and writes every proposed KV
and index row into one tentative cache before query evaluation. Each query uses
its own causal length, and prefix commit still restores rejected physical rows.
Cache preparation and head projections retain one-row operations; the selected-KV
exchange, sparse attention and DSA selection operate across query rows. This
keeps decode D5 disabled. Ordinary builders do not enable either new option.

The attention-only CPU32 comparison passes across starts 61, 127 and 510 with
permuted physical pages, testing owner/page crossings and the DSA top-k cutoff.
KV/index caches, outputs, selected IDs/counts and health are bitwise equal to
sequential one-row calls on every shard. Changing later proposed rows leaves
the first output/selection bitwise unchanged. Full-index DSA scores have a small
FP32 rounding boundary: maximum observed absolute difference 8.94e-8; shared
scores are exact. The test's 2e-6 absolute/relative guard is an empirical fixture
bound, not proof of arbitrary near-tie selection.
[Attention CPU receipt](mtp-batched-attention-cpu-20260919.json).

The combined candidate passes the two- and three-row complete CPU fixture
comparisons: predictions agree, residual max error is 0.046875 and relative L2
is 0.006535 / 0.006559. Five rows still exceed the unchanged envelope (max
0.171875, relative L2 0.020100) and remain an expected qualification failure.
All checked causal/rollback/refusal invariants pass. Canonical-MLP-only three
rows also pass. These are small synthetic-model boundaries, not trained-token
admission. The ordinary D1/D8/D10 CPU regression against the frozen decoder
passes, as do five acceptance/cache/comparison guard tests.
[Combined CPU receipt](mtp-verifier-candidate-cpu-20260919.json).
The next hardware comparison is limited to two and three rows; no native MTP
payload or drafter execution has occurred.

## Combined-candidate acquisition: also rejected

`perf_mtp_verifier_batched_20260919T223024Z`, immutable source `80bfc8ba`,
started after acquiring both workload leases and the sync lease and authenticating
all eight hosts idle. It completed on all eight hosts at 22:44:52–55 UTC, with
fresh graph/memory admission and authenticated final idle. Controller session
31423 returned zero; no automatic workload retries were used.

| Synthetic candidate | Perfect-acceptance estimate | Within-run speedup |
|---|---:|---:|
| 2 rows + commit | 14.192–14.194 tok/s | 0.954x |
| 3 rows + commit | 16.119–16.120 tok/s | 1.083–1.084x |

Ordinary model calls took 67.216–67.226 ms. Drafting, health votes and delivery
are excluded. Both sizes still disagree with sequential target predictions and
selected positions on every host. These estimates do not establish speculative
throughput, and neither size meets the working improvement criterion even at
perfect acceptance on this synthetic workload. The previous candidate and this
one remain rejected. [Completed receipt](tpu-mtp-verifier-batched-20260919T223024Z.json).

## Trained target diagnostic preparation

`--diagnose-speculative-verifier` is an opt-in to the real-weight DB610 worker.
It first requires all 29 ordinary DB610 tokens to match, retains the initial
prefill state, and compiles/adopts fresh two-/three-row verifier and committer
graphs under the existing fleet identity, HLO and live-memory checks. Every
reference input is then consumed in blocks, with zero-padded final blocks
committing only their live prefix. Predictions, final cache/frontier state and
per-block model-call costs are compared with ordinary decode. It uses reference
tokens as proposals; it does not execute MTP or measure accepted throughput.

The fleet summarizer requires both variants on all eight hosts, every graph and
health phase, complete block counts, matching reference/graph identities and
32-chip memory coverage. It preserves numerical mismatches as negative results
and publishes only named aggregate fields. The worker refuses experimental D5
or noncanonical prefill in this diagnostic mode.

Control/receipt regressions pass (78 tests, excluding two separate CPU32 tests).
A real JAX CPU32 multi-round test passes with the trained decoder's routed-kernel
tile sizes (85.15 s): two and three rows each reproduce five ordinary successors,
preserve the final frontier, and restore padded/future cache rows. This remains
an eight-layer synthetic CPU test, not a trained-weight or broad quality result.
The release check also passed (524 tests, one skip; source pin and package checks
passed), and ledger consistency has zero unresolved entries.

## Trained target acquisition: tokens match, economics insufficient

`perf_real_mtp_verifier_20260919T225845Z` launched from immutable source
`49447af5` after both workload leases, the sync lease and authenticated idle on
all eight hosts. The controller uses one SSH execution per host, exclusive start
markers and no automatic workload retries. It completed the ordinary trained
DB610 trail and both reference-token diagnostics. Controller session 7887
returned zero, all eight final idle checks passed, and the strict fleet summary
accepted graph, memory, source/checkpoint identity and execution evidence.
No native MTP payload was loaded or executed.

The ordinary path matches all 29 reference tokens; **both verifier sizes match
all 28 successors on every host**. This is token agreement on one teacher-forced
trail, not bitwise state agreement or broad answer-quality validation.

| Target rows | Perfect-acceptance estimate, including padded tail | Paired model-call speedup |
|---|---:|---:|
| 2 | 14.548–14.602 tok/s | 0.962–0.966x |
| 3 | 16.026–16.091 tok/s | 1.077–1.082x |

These estimates include verifier and prefix-commit calls but exclude drafting,
health votes, comparison work and delivery. Three-row full blocks alone estimate
17.113–17.190 tok/s; the complete 28-successor trail also pays for the padded
last block. Neither establishes a 25% improvement, even before drafting costs.

Frontiers, page tables, health and selected counts match bitwise. Final KV/index
caches and selected-position ordering do not: across hosts, maximum KV absolute
errors are 2.141602 / 1.427734 and maximum whole-cache relative L2 errors are
0.017423 / 0.015777 for two/three rows. Unchanged prompt rows contribute to those
whole-cache denominators. All values are finite; matching tokens do not erase
these numerical differences. This candidate is not serving-admitted.

Ordinary prefill in this acquisition measured 140.438–140.440 prompt tok/s at
2,034 tokens. Peak measured HBM remained 28,228,678,144 bytes/chip. The previous
paired request-loop result remains the 14.04 wall-decode baseline; model-call
diagnostics have a different timing boundary.
[Complete trained diagnostic receipt](tpu-real-mtp-verifier-20260919T225845Z.json).

The second synthetic acquisition's complete 24 host/variant trace summaries
are now retained in [the fleet trace receipt](tpu-trace-mtp-verifier-batched-20260919T223024Z.json).
Profiled ordinary / two-row / three-row device times average
68.47 / 139.94 / 184.49 ms. Collective self-times are
36.27 / 63.20 / 82.76 ms; FP8/custom-call self-times are
12.14 / 26.14 / 38.67 ms. Batched gather/scatter work rises from 2.55 ms to
13.42 / 17.24 ms, and sort/top-k from 1.84 ms to 9.60 / 12.13 ms.
These measurements identify candidate costs, not accepted output throughput;
collectives include owner waits, and both candidates failed token agreement.

## Smaller expert tiles and per-row DSA candidate

Prepared while the first trained acquisition used archived `49447af5` source,
a separate opt-in candidate addresses two costs identified in the completed
synthetic trace. It was not part of that completed trained experiment.

`small_expert_tiles` groups all occurrences of an owned expert into the ordinary
decoder's M8 tile, retaining ascending K128 contractions and fused gate/up.
At most eight verification rows imply at most eight occurrences of any expert,
because each token's routes are distinct. This avoids general prefill M32
packing and reuses each decoded expert across its proposed tokens. Invalid
route IDs/duplicates/ownership refuse, and empty owners explicitly write zeros.

`rowwise_dsa` preserves the scalar two-stage selection fallback inside each
token's DSA call, while the selected-KV attention exchange remains batched.
The previous `vmap` executes both conditional branches: its traced full-width
16,384-candidate fallback top-k costs 4.22 / 6.08 ms for two/three rows on host 0,
in addition to the shortlist path. Avoiding that work is a hypothesis until TPU
measurement; this option also retains ordinary per-row FP32 score arithmetic.

CPU evidence: 13 projection tests pass bitwise against independent ordinary
route projections for 2/3/5/8 rows, both FP32/BF16 results, changing route orders
and empty owners. Complete MoE comparisons pass bitwise on all 32 CPU shards
for 2/3/5 rows. The attention fixture passes every field bitwise, now including
DSA scores, across populated/permuted pages and owner/page crossings. Full
eight-layer model predictions agree for all tested sizes, but rounding remains:
two/three rows stay inside the unchanged envelope (max residual 0.046875;
relative L2 0.006535 / 0.006559), and five rows still fail (max 0.171875;
relative L2 0.020100). Five rows remain unqualified.

The two-/three-row multi-round diagnostic also passes on CPU32, including the
short final block and restoration of padded/future cache rows (87.01 s).
Receipt/control regressions pass 82 tests. The trained worker supports explicit
flags for these options, and its summarizer requires identical options across
all hosts and both variants. Ordinary execution defaults remain unchanged.
[CPU candidate receipt](mtp-m8-rowwise-cpu-20260919.json).

## Preserve per-row arithmetic expression boundaries

Further CPU diagnosis traced a one-row discrepancy to the first normalization
after a sparse layer, despite matching stored layer outputs. Moving the final
expert scaling/addition outside its mapped loop removed that one-row difference.
For two rows, keeping both normalization and the cheap final expert reductions
as unrolled one-row expressions reproduced every checked layer activation and
the final residual. The expensive expert projections still pool rows. Added
diagnostic outputs did not change either final residual in these fixtures.
[Layer-boundary evidence and failed strict assertions](mtp-cpu-output-boundaries-20260919.json).

This does **not** establish bitwise equality of arbitrary multi-row state.
Stricter checks found one-row stored DSA-score differences of a few FP32 units,
a two-row full-prefix KV difference, and residual differences for three/five
rows. One-row residual/state checks therefore require bitwise agreement on all
32 CPU replicas except the explicitly bounded scores; two-row residuals require
bitwise agreement, while other multi-row floating state retains the documented
empirical envelope. Selected IDs, frontiers, physical rollback and causal
independence retain their separate checks. No tolerance was widened.

The new three-row residual boundary is max 0.0625, relative L2 0.003549; five
rows still fail the unchanged envelope (max 0.1640625, relative L2 0.011153).
Complete MoE and multi-round control checks pass (three tests, 143.87 s).
The final two-row test also passes all causal/refusal checks: residuals are
bitwise on all 32 CPU replicas, index-cache comparisons are exact, and the
checked KV prefix differs by one element (max 0.000244140625, relative L2
0.0000026195). This is a measured fixture boundary, not a universal bound.
[Final CPU qualification receipt](mtp-unrolled-boundaries-cpu-20260919.json).
Only two/three rows are included in the next trained TPU diagnostic, using M8
expert reuse, per-row DSA and these expression-boundary changes. Any token match
must be reported alongside cache differences; no serving promotion is implied.

The acquisition launched as `perf_real_mtp_verifier_m8_20260919T235646Z` from
immutable `a7b1ca1b`, after authenticating all eight hosts idle under the workload
and pod leases. The controller stages a Git archive and uses one SSH workload
invocation per trusted host, with exclusive worker markers and no automatic
retries. At the 23:57 UTC observation the worker had launched; results and final
cleanup are pending. No other TPU workload is queued.

The separate documentation checkpoint has been reconciled with the current
scope: it retains the validated ordinary-path receipt, source audit and
historical cancellation. Stale no-active-goal/current-idle claims and the
out-of-scope GLM-5.3 assessment are excluded from the proposed main update;
their originals remain on preserved research refs. Main merge and regional
backup verification remain pending; no faster engine deployment is claimed.

At 00:08 UTC on 2026-09-20, the live M8 acquisition had completed trained-weight
verification/loading and the first prefill graph's eight-host hash consensus.
It remained active without a reported failure. No speed/correctness result was
available at that observation.

## Written-cache diagnostic for later acquisitions

Whole-cache relative L2 includes unchanged prompt rows and can understate the
error in a short generated suffix. Future trained diagnostics separately compare
the newly committed span, following the physical page table and every local
expert shard/feature replica. Hosts without rows in that span report zero sampled
elements. The strict fleet summary checks all eight hosts' scope, the exact
2034..2062 span and four-feature coverage, and refuses missing or inconsistent
measurements. Whole-cache comparisons remain present to catch changes outside
the expected writes. These host comparisons stay outside model timing.

CPU checks cover physical-page permutations, page/owner crossings, unowned spans,
nonfirst-replica errors, nonfinite values and malformed fleet reports. This
change passes 107 focused diagnostic/acquisition tests (2.37 s); the unchanged
CPU32 model tests are excluded from this host-only check. Release checks also
pass: 524 tests passed, one skipped, frozen-source and isolated-package checks
passed. This
diagnostic was added after the current immutable `a7b1ca1b` acquisition launched;
it cannot recover per-row metrics from that run's aggregate receipt. It changes
no numerical kernel or acceptance threshold.

The native MTP input-projection prototype now has a CPU32 check: separate
embedding/previous-hidden normalization, position-zero embedding mask and the
correct feature concatenation/projection order. H256 synthetic inputs at
1/3/114/128 rows match an independent unsharded expression numerically (observed
maximum error zero), including all-owner invalid-input refusal. This remains
separate from the active verifier acquisition. The MTP transformer, prompt
hidden export/cache bootstrap and accepted-history refresh are still unfinished;
no trained drafter or accepted throughput is claimed.
[Scoped projection receipt](mtp-projection-cpu-20260920.json).

## Completed trained M8 verifier — 2026-09-20

`perf_real_mtp_verifier_m8_20260919T235646Z` completed from `a7b1ca1b` with
all eight hosts authenticated idle. The ordinary path matches all 29 DB610
tokens, and both two-/three-row candidates match all 28 successors on every
host. Graph consensus, scoped HLO/memory admission, health and source/input
identity checks pass. No native drafter executed in this acquisition.

| Target rows | Perfect-acceptance estimate, padded tail included | Paired model-call speedup |
|---|---:|---:|
| 2 | 16.39–16.45 tok/s | 1.084–1.089x |
| 3 | 17.44–17.50 tok/s | 1.151–1.155x |

Three-row full blocks alone estimate 18.62–18.69 tok/s, or 1.229–1.233x paired
model-call speed. The one-row final tail still incurs a three-row verification.
These figures exclude drafting/refresh, host votes, comparisons and delivery;
they are not accepted speculative throughput or a 25% wall-speedup result.
The previous three-row implementation estimated only 16.03–16.09 tok/s on the
same trail, but its acquisition is a separate run. Prefer the paired ratios
within each run rather than treating cross-run differences as controlled proof.

Numerical state is still different: maximum KV errors are 1.298828125 / 1.359375;
maximum whole-cache relative L2 is 0.015602 / 0.015951. Index, selected-position
arrays and scores differ; frontiers, block tables, counts and health agree
bitwise. All compared values are finite. Whole-cache error includes unchanged
prompt rows; this immutable worker predates the new written-span metric.
Short-trail token agreement does not establish broad answer quality or state
equivalence.

Ordinary prefill measured 139.771–139.776 prompt tok/s at 2,034 tokens; peak HBM
remained 28,228,678,144 bytes/chip. The separate 14.04 wall-decode baseline
remains the relevant request-loop measurement. The result leaves little room
for drafting overhead or rejected proposals; end-to-end acceptance measurement
is still required. No implementation is promoted by this diagnostic.
[Complete eight-host receipt](tpu-real-mtp-verifier-m8-20260919T235646Z.json).

## Private main checkpoint published — 2026-09-20

The reconciled documentation/evidence checkpoint is now on private main at
`5e9ce605b50ca379a9c541e1142686b346f7705e` (620 tracked files, four added compact
documents/receipts). The experimental engine and unfinished MTP work remain on
the perf branch. Original staged checkpoint `132dc75b` and all research history
are preserved. No model-source pin or supported numerical body changed.

The publication held workload, pod, sync and cron leases; authenticated all
eight hosts idle before/after; and verified the release-tree/shared-Git-store
mirrors against checksums and exact object generations before and after the
normal fast-forward push. Final verification covers 1,605 release-tree files
and 2,174 shared-Git-store files. The approved bucket is US-CENTRAL2 with
soft-delete disabled; the checked existing mirror inputs total about 472 MB,
with no model-weight copy. The first preflight refused a stale local main ref;
an ordinary non-force fetch advanced it to the already-published baseline before
publication resumed. Both attempts' originals are preserved privately.

The authoritative final record is
`gs://driftbench-dsv4-uc/results/perf_checkpoint_20260919/20260920T003310Z/final-promotion.json`,
generation `1789864667007100`, SHA256
`d4adfbf3ca89a0c3959a3e3ac4b98718bfa0266bf6b199ff5b2cabde56195fd1`.
[Compact publication receipt](perf-checkpoint-promotion-20260920.json).
This completes the useful-result checkpoint, not native drafting or the active
accepted-throughput/representative-correctness goal.

## Prompt hidden export and fresh question — 2026-09-20

The target prompt hidden export is implemented outside frozen source. Its
canonical B128/B114 CPU fixture preserves ordinary prefill state and next token
bitwise on all 32 owners, exports post-final-norm rows, and refuses cache writes
when only the final physical replica has an invalid live norm. Padded/refused
rows return zero hidden states and false validity. An AST comparison constrains
the control-flow mirror to the export/health additions. This is a synthetic
two-layer proof, not trained TPU admission; additional outputs can change XLA
realization and memory. [CPU receipt](mtp-prefill-export-cpu-20260920.json).

The owner requested a fresh complicated question and an ordinary-versus-MTP
comparison. The research worker can now run a bounded, authenticated private
question after reproducing all 29 DB610 tokens. It reuses the canonical
B128/B114 prefill graphs and packed ordinary decoder, includes all host votes
and rank0 token write/flush in decode wall timing, stops at EOS or the exact
output cap, and compares complete output hashes across eight hosts. The prepared
338-token optimization question has a 6,144-token generation cap, a pinned local
thinking template, and a separately computed exact oracle. Raw question, oracle
and future response stay outside Git. Native MTP is not used by this baseline;
its combined speculative comparison remains pending.

## Native transformer and source placement — 2026-09-20

`mtp_draft.py` now computes the input projection, native one-layer transformer
and normalized shared-head output as a separate, uncommitted draft proposal.
Full-index rows support bootstrap/target-hidden refresh; single recurrent rows
reuse the preceding shortlist and skip the entire DSA indexer while writing MLA
KV. The existing prefix committer rolls back rejected physical rows. A CPU32
synthetic composition check agrees bitwise with projected inputs passed through
the separately tested one-layer verifier. Poisoned DSA tables leave recurrence
unchanged; causality, fresh-index refresh, rejection rollback and invalid-hidden
refusal pass. This is not an independent trained upstream-model comparison.
[Eleven-test component receipt](mtp-native-components-cpu-20260920.json).

`mtp_checkpoint.py` maps authenticated layer-78 source intervals into the
separate draft layer zero without changing original source names/offsets. It
uses the native head norm and shares the target embedding/head arrays. Metadata
reconciliation covers all 1,569 source tensors, 7,200 placements, 39 destination
tables per slot, exact coverage and no overlapping writes. The raw pack would
occupy 363,837,792 bytes/chip (before BF16 conversion), 11,642,809,344 bytes total;
replicated tables account for the difference from the 10,032,632,960 source
bytes. No tensor payload was read or acquired by this metadata check.
[Placement receipt](mtp-native-placement-20260920.json).

The fresh ordinary question acquisition started as
`perf_real_long_question_20260920T005757Z`, immutable source `248ef059`. Its
controller acquired workload/pod/cron/sync leases, authenticated all eight
hosts idle, and launched each worker once with automatic retries disabled.
At 00:59:31 UTC the live controller and rank0 source-inventory phase were
confirmed; no question answer or timing result existed yet. Native MTP is not
part of that run. Acquisition, prompt bootstrap/history refresh orchestration
and the accepted-token host loop remain necessary before its paired comparison.

At 01:10:20 UTC the fresh-question controller was re-polled and remained live.
All eight workers had reached `memory_prefill_128`, with no reported errors;
rank0 had compiled/admitted the B128 graph and retained WK graphs. Other graphs
were still compiling. No response or fresh-question timing was available.
