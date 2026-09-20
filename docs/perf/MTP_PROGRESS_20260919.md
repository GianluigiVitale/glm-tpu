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

## Native packer/loader prepared — 2026-09-20

The research packer writes only native MTP tables into exact final-owner binary
files, with complete source-range, tensor and file hashes. Its loader checks the
manifest, physical mesh, local owner set and each tensor immediately before
placing it on its chip. The generation-bound reader checks the approved bucket
region, audited object generation/size/CRC and source header, then permits only
native tensor ranges. No whole source-object SHA256 recomputation is claimed.
The base embedding/head are absent from the pack. Nineteen CPU checks cover
exact tiny payloads/all32 owners, restricted reads, source failures and altered
identity/schema/files. The real inventory yields the expected 39 raw tables and
363,837,792 bytes/chip. [CPU receipt](mtp-pack-cpu-20260920.json).

The acquisition controller is prepared but has not run. It requires idle fleet
and workload/pod/cron/sync leases, uses the prior authenticated physical owner
map, binds immutable source and disables workload/transport retries. Cleanup
checks both TPU holders and the exact native worker PID/start-time because the
packer itself is CPU-only. It will reconcile all source tensor hashes between
owners before publishing the private pack index. Do not overlap acquisition
with the active long-question run.

At 01:20:45 UTC that ordinary-question controller was confirmed live. All eight
workers had reached the final question-prefill health check; rank0's reference
trail matched 29/29 DB610 tokens. The output contained546 delivered token records.
No completed response, wall-speed receipt or correctness judgement existed yet.

## Completed fresh long-question baseline — 2026-09-20

`perf_real_long_question_20260920T005757Z` completed from immutable `248ef059`,
with authenticated idle on all eight hosts. The DB610 gate matched 29/29 tokens
and every host agreed on the fresh output hash. The run generated 6,144 tokens,
including the first prefill token, and timed 6,143 decode steps. Rank0 measured
**14.4141 wall tok/s**, 426.1795 s of decode, p50/p99 69.19/73.84 ms. Every host
reports essentially the same wall rate. The 256-token windows range 14.35–14.50
tok/s. All host votes and rank0 private token JSONL write/flush are included;
there is no network transport and no excluded warm decode prefix. The 338-token
prompt took 2.9555 s (114.36 prompt tok/s); warmed TTFT 3.1954 s. Cold model load
and compilation are separate. Peak HBM remained 28,228,678,144 bytes/chip.

The model was still reasoning when it reached the registered 6,144-token cap.
There was no closing thinking marker or completed final answer, so the exact
TSP oracle cannot establish answer correctness for this truncated continuation.
Preserve it as the sustained ordinary speed baseline; it is not a successful
quality result or accepted speculative throughput. Raw input/output/oracle
remain private. The strict summary checks all eight sources/graphs/phases,
output/timing/window/memory agreement and the final token-file trail; 80 focused
CPU summary/host tests pass.
[Complete receipt](tpu-real-long-question-20260920T005757Z.json).

## Native acquisition started

After the baseline completed and the existing backup released its cron lease,
`perf_native_mtp_acquire_20260920T013227Z` launched from immutable `50a8bbc5`.
All eight hosts were authenticated idle. The controller and CPU-only native
workers hold workload/pod/cron leases, verify generation-bound source ranges,
and use the preserved physical owner map. No TPU model is initialized by this
pack. At 01:35:10 UTC the controller was confirmed live; no completed fleet pack
index had been reported. The first preflight at 01:31:16 refused the busy cron
lease before any remote launch and is preserved separately. No automatic retry
was queued; the successful attempt followed an explicit free-lease check.


## Native acquisition and complete session integration (2026-09-20)

Native-only acquisition `perf_native_mtp_acquire_20260920T013227Z` completed from
`50a8bbc5`. All 32 owner payloads cover 39 raw tables each (363,837,792 bytes/chip).
The 1,569 source tensor hashes agree across readers of the same generation-bound
source ranges. All eight hosts were authenticated idle after worker completion;
this acquisition initialized no TPU backend. [Receipt](mtp-native-acquisition-20260920T013227Z.json).

`mtp_state.py` now owns the committed native cache, cached first draft and
normalized native hidden. Recurrent IndexShare supplies only a disposable second
guess. Full refresh starts from the committed native root and uses the accepted
target's shifted predictions and normalized hidden; rejected physical rows are
rolled back. `speculative_request.py` checks the entire token plan across hosts,
commits target/native roots together before delivery, and poisons the request
on any ambiguous delivery failure. EOS, budget tails and first/partial rejection
are covered in CPU tests. The synthetic device/session integration also compares
every refreshed native cache against a teacher-forced replay of accepted target
history. This is not independent trained native-model validation.

The opt-in real worker now supports a pinned native pack index. It measures
ordinary/R2/R3 continuations with fresh prompt caches, native bootstrap, target
hidden-export DB610 parity and per-graph HLO/memory admission. R2 means one native
draft; R3 adds one recurrent draft. Both use target-verified greedy acceptance.
Wall time includes all speculation/refresh/commit/vote/delivery work; separate
component timings are synchronized inside that wall boundary. No real native
speedup is established until this worker completes and its receipts are checked.

CPU integration receipt: [native session](mtp-native-session-cpu-20260920.json).
66 checks passed in 69.98 s; the expanded device proof including eight-row
bootstrap and owner-crossing partial refresh passed in 68.99 s.


The real comparison controller `perf_real_native_mtp_20260920T015817Z` started
from immutable `bcec7ddd` at 01:58:17 UTC, under the existing exclusive leases.
No native result is claimed before completion and all-host receipt validation.
The new native summary checker rejects incomplete graph/pack identity, token
counts, acceptance accounting, timings, memory evidence or output agreement.
Its first focused suite passed 69 tests (including ordinary summary regressions).

At the 02:08:44 UTC live observation, all eight hosts had completed target
checkpoint verification/loading and BF16 preparation, without recorded errors.
Baseline graphs are compiling; no native MTP timing is available yet.


## Representative continuation inputs prepared (2026-09-20)

Private prose/code/structured cases are ready for the required broader comparison:
204/347/696 prompt tokens and 2,048/4,096/2,048 output-token budgets. The code
case's interval-scheduling optimum was checked by all 4,096 subsets and an
independent dynamic program; the structured JSON numeric totals have a separate
record-wise cross-check. Prose still requires manual explanation review. Raw
prompts, oracles and token IDs remain outside Git. The suite declares two paired
repeats per case and is prepared, **not launched**; it cannot overlap the active
native comparison. [Preparation receipt](mtp-representative-inputs-20260920.json).

The optional native-suite loader authenticates every case, rejects path escapes,
symlinks, duplicate/reserved labels and invalid repeat counts, and assigns
separate measurement labels to identical repeated requests. The real worker and
strict summary now accept these registered cases and compile their exact native
bootstrap tails. Existing single-question behavior remains available. The focused
suite/summary/ordinary validation checks passed 80 tests in 2.80 s on CPU.


At 02:20:08 UTC the same controller remained live. All eight logs show successful
`native_reference_admission` and `native_bind`: the ordinary DB610 gate passed
and trained native weights were loaded/prepared. Rank0's private receipt records
29/29 reference tokens matching. Native comparison graph compilation is underway;
there is no accepted native throughput result or end-to-end native correctness
claim yet. No workload was restarted.


Final-answer checking is prepared in `answer_assessment.py` without executing
model-produced Python. It distinguishes unfinished reasoning, manually reviewed
prose, exact standalone structured JSON, and an optimal final schedule/tour.
Checks reject duplicate JSON keys, nonfinite/incorrect numeric types, overlapping
or duplicate jobs, invalid tours and wrong optimal values. A correct final data
object is explicitly narrower than correct generated code or explanatory proof.
The answer/summary tests passed 21 checks in 1.20 s on CPU; paired ordinary
receipts also now enforce EOS/output-budget terminal accounting.


At 02:31:30 UTC the original controller was again confirmed live. Every host
had reached `memory_native_inputs_1`, without recorded errors. The two native
hidden-export prefill graphs, packed ordinary decoder, native refresh R1/R2/R3/R8
and R1 input preparation were compiled and admitted. Target verifier/commit
compilation remains; the cases dictionary is still empty and no accepted native
throughput is claimed. Next manual observation is conservatively after 02:42:30.


At 02:42:58 UTC the same controller remained live. All 16 comparison programs
were compiled/admitted, and all eight hosts had completed hidden-export DB610
prefill through `db610_prefill_health_15`, without recorded errors. Request
measurement follows; no native accepted-token rate was present at this poll.
The requirement-by-requirement [completion audit](MTP_COMPLETION_AUDIT_20260920.md)
keeps the original scope and remaining measurement/publication gates explicit.


## First native accepted-token measurements (live, 2026-09-20)

At 02:53:36 UTC, controller `perf_real_native_mtp_20260920T015817Z`
(source `bcec7ddd`) remained live; all eight logs reached
`native.question.r2_prefill_health_2` without recorded failures. Rank0's live
private receipt records the following short DB610 comparison:

| Mode | Wall tok/s | Paired ratio | Draft acceptance | Output agreement |
|---|---:|---:|---|---|
| Ordinary | 13.0775 | 1.0000x | — | 29/29 reference |
| One-draft MTP speculation (R2) | 12.6385 | 0.9664x | 14/14 first drafts | 29/29 ordinary |
| Two-draft MTP speculation (R3) | 14.3078 | 1.0941x | 10/10 first, 8/9 second drafts | 29/29 ordinary |

R2/R3 emitted 2.0/2.8 tokens per round. Their cumulative synchronized target
verification calls cost 1.7126/1.5820 seconds; native refresh cost 0.0446/0.0383
seconds. These costs are included in wall throughput with drafting, commits,
host checks and delivery. The short check shows functioning trained native
acceptance, not a sustained or representative gain. Multi-row numerical cache
boundaries still apply despite matching tokens.

The same long question's paired ordinary baseline completed at 14.2902 wall
tok/s; its R2/R3 measurements are still pending. The live run is incomplete,
final all-rank validation and authenticated cleanup are outstanding, and no
second workload has launched. Next manual observation is after 03:04:00 UTC.
The retained upstream proposer was cross-checked again: shifted tokens retain
target positions, recurrent hidden state is post-final-norm, and later draft
iterations reuse the first native pass's index selections. No alignment change
was indicated by that source inspection.


The fleet summary now preserves ordinary request TTFT, request/decode wall time,
host-vote time and per-token p50/p99/model-call latency alongside speculative
round latency. It validates each host's timing containment rather than dropping
the already-recorded baseline fields. Terminal reason and timed decode count
are also retained. This is post-processing only; immutable TPU worker
`bcec7ddd` is unchanged. Native-summary, ordinary-question and real-validation
checks passed 90 CPU tests in 2.92 s, including invalid ordinary TTFT, request
wall, vote time and latency refusal.


## Long one-draft result (live, 2026-09-20)

At 03:04:39 UTC the same controller remained live, with all eight logs at
`native.question.r3_prefill_health_2` and no recorded failures. Rank0's completed
R2 question measured **13.0927 wall tok/s versus ordinary 14.2902 (0.9162x)**.
It accepted 2,788/3,354 first drafts (83.12%), averaging 1.831 emitted tokens
per round. Synchronized target verification consumed 406.070 s; native refresh
10.662 s, prefix commits 4.352 s and draft preparation 1.665 s, all included
in wall time. Warm TTFT was 4.847 s versus ordinary 4.370 s.

The R2 token trail differs from ordinary at zero-based index 6. The multi-row
verifier has a documented floating-point boundary, but this observation alone
does not isolate the cause of the first divergence. Do not claim token-exact
speculation or promote it as an ordinary replacement. The short DB610 token
match did not establish long-request equivalence.

Pinned local-tokenizer decoding of both completed 6,144-token outputs found
no closing reasoning marker or final answer. The exact TSP oracle cannot grade
a finished tour because none was produced. Correctness remains unestablished.
R3 is pending; full-fleet aggregation and authenticated cleanup are outstanding.
The private observation snapshot is
`observation.rank0.20260920T030400Z.json` in the run root, SHA256
`ad6692ec41a6a37a5a5e7385d98910abe66adf9ef053a352e911c1fab5c421e8`.
Raw decoded responses and the interim answer assessment remain private.
Next manual observation is after 03:15:00 UTC; no second workload has launched.


The subsequent representative suite remains unlaunched. Its initial 2,048/4,096/
2,048 output caps have been superseded by a common 7,168-token cap, because both
completed long variants exhausted 6,144 tokens without a finished answer. All
prompt IDs and other request fields are unchanged, all oracles are byte-identical,
and EOS still stops generation. Each prompt plus output budget fits the existing
8,192-token capacity. The original private input set and preparation receipt are
preserved; this changes no completed measurement or current TPU execution.
The authenticated loader accepted all six revised policies (two repeats per case).
[Revised preparation receipt](mtp-representative-inputs-extended-20260920.json).


## Completed first native fleet comparison (2026-09-20)

Controller `perf_real_native_mtp_20260920T015817Z` exited successfully. All
original eight-host receipts pass strict source/input/native-pack/graph/admission,
output-accounting and cleanup checks. The
[completed receipt](tpu-real-native-mtp-20260920T015817Z.json) supersedes the
provisional observations above without changing their originals.

| 6,144-token long request | Wall tok/s (eight-host range) | Paired ratio | Tokens/round |
|---|---:|---:|---:|
| Ordinary | 14.290246–14.290249 | 1.0000x | 1 |
| One native draft (R2) | 13.092679–13.092688 | 0.9162x | 1.8310 |
| Two native drafts (R3) | 12.768179–12.768182 | 0.8935x | 2.3077 |

R3 accepted 2,071/2,662 first and 1,410/2,662 second drafts. Its synchronized
verification cost 426.03–426.42 seconds, native draft continuation 4.78–4.85,
refresh 10.41–10.44 and commit 3.56–3.65 seconds. Total decode wall was
481.118 seconds, compared with ordinary 429.874 and R2 469.193. All overhead
is included; cold loading/compilation, prefill/bootstrap and network delivery
are excluded from decode throughput. The long prompt's native bootstrap adds
0.681–0.731 seconds. Peak HBM remained 28,228,678,144 bytes/chip.

Both speculative outputs differ from ordinary starting at token index 6.
All three private decoded responses exhaust the output cap during reasoning,
so no finished-answer correctness claim is available. Keep ordinary as the
qualified path; the native experiment remains research. DB610's all-mode
29-token match and short R3 gain do not override the long negative result.
The prepared representative suite and repeats remain necessary to report
prompt-dependent behavior and assess completed answers. No second workload
has launched at this evidence checkpoint.


Representative controller `perf_real_native_suite_20260920T031727Z` started
from immutable `dc047933` at 03:17:27 UTC. Initial observation at 03:17:32
confirmed all workload/sync leases and authenticated eight-host idle before
staging. It compares ordinary/R2/R3 for all three cases with two repetitions
and the revised matched cap. No representative timing is available yet;
next manual observation is after 03:28:00 UTC. No automatic workload retry.

An evidence-only checkpoint is prepared and pushed at `deea6dd1` on
`release/mtp-evidence-20260920`, in a new worktree based on private main
`5e9ce605`. It retains a byte-identical copy of the completed first native
receipt and a connected comparison/recommendation document. The changed links
resolve, content audit is clean and its 622-file ledger has zero errors. No
runtime/test/configuration files changed. This is a draft: representative
results, final release/self-review and regional backup/idle checks remain
required before main publication. The separate publication script is prepared
but has not acquired leases, written cloud artifacts or merged anything.


At 03:28:32 UTC the representative controller was confirmed live by its
original session handle. All eight run logs reached `bf16_prepare` without
recorded failures. Native graphs/cases are not yet present in rank0's receipt;
no representative performance result is claimed. Next manual observation
is after 03:39:00 UTC. No restart or overlapping workload occurred.

The completed long receipt also bounds the benefit of better acceptance at its
measured average round cost. R2 took 469.1935/3,355 = 139.85 ms/round; R3 took
481.1179/2,662 = 180.74 ms/round. Holding these costs fixed, perfect acceptance
would estimate 14.30/16.60 tok/s. R3 needs 2.583 tokens/round to break even with
ordinary 14.29, versus measured 2.308; a 25% gain would require 3.228, beyond
the three-row limit. These are explicit fixed-cost estimates, not measured
perfect acceptance or hardware-wide bounds. Changed trajectories/expert routes
can change cost. At the measured cost, improved draft accuracy alone cannot
reach the working 25% target; the target-verification path would also need work.


At 03:39:07 UTC the same representative controller remained live. All eight
logs reached `native_bind` without recorded failures; rank0 records passed
`native_reference_admission`, `native_load` and `native_bind`. Native graph and
case dictionaries remain empty while compilation starts. Next manual observation
is after 03:50:00 UTC. No representative timing is available yet.


At 03:50:30 UTC the original representative session handle was confirmed live.
All eight logs reached `memory_native_refresh_2` without recorded failures.
Five native programs are compiled/admitted: prefill B114/B128, ordinary and
refresh R1/R2. Rank0 still has no measured cases. Next manual observation is
after 04:01:00 UTC. This interval is a verified wait; no numerical source,
workload or publication state changed.


At 04:01:12 UTC the original representative controller remained live. All eight
logs reached `memory_native_inputs_3` without recorded failures. Fifteen of
seventeen native graphs are compiled/admitted, including refresh R4; the R3
verifier and commit graphs remain. No request case is measured yet. Next manual
observation is after 04:12:00 UTC. This is a verified wait on the same workload.


## First representative request completed (live, 2026-09-20)

At 04:12:28 UTC the same controller remained live, with all eight hosts at
`native.prose_repeat1.r2_prefill_health_1` and no recorded failures. All 17
graphs are compiled/admitted. DB610 again matches all 29 reference tokens in
ordinary/R2/R3. Rank0's first ordinary prose request completed at EOS with
2,655 generated tokens at **14.4424 wall tok/s**; its speculative comparisons
remain pending. The private immutable observation SHA256 is
`d97f77c3c5abffaca08428cd665e9379418bddf4b0969384cf9b2551f4961964`.
These are provisional rank0 observations, not a completed suite/fleet receipt.

Pinned-tokenizer decoding confirms a completed final prose answer. A scoped
assistant self-review found corrections needed in the cache-complexity claim
and the acceptance/correction and rejected-cache explanations; arithmetic and
the TTFT/throughput distinction are sound. This is not independent review or
a model-wide quality score. The original manual review was revised after
checking primary sources: independent parallel MTP heads are a valid generic
design, so the answer is not penalized merely for differing from this GLM's
recurrent predictor. [MTP architecture paper](https://arxiv.org/abs/2404.19737),
[speculative sampling algorithm](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).
Raw prompt/output and both review revisions remain private. Current manual
review SHA256: `7c34f668cb37d5dbcf92683941586675c4f858c52e75042a7bd687e579ac5907`.

The completed-result publisher can now attach a bounded manual prose assessment
only when its case/mode, response and token hashes match. It publishes check
booleans and provenance, not private review text, and distinguishes self-review
from independent assessment. It has not summarized the active suite as complete.
Next manual workload observation is after 04:23:00 UTC.

## First paired prose comparison (live, 2026-09-20)

At 04:23:35 UTC the original controller remained live; all eight logs reached
`native.prose_repeat2.r2_prefill_health_1` without recorded failures. Rank0's
first prose comparison completed at EOS in all modes:

| Mode | Wall tok/s | Generated tokens | Paired rate |
|---|---:|---:|---:|
| Ordinary | 14.4424 | 2,655 | 1.0000x |
| One draft (R2) | 13.2894 | 6,134 | 0.9202x |
| Two drafts (R3) | 13.4892 | 3,745 | 0.9340x |

R2 accepted 2,801/3,332 drafts and emitted 1.8406 tokens/round. R3 accepted
1,246/1,545 first and 954/1,545 second drafts, emitting 2.4233 tokens/round.
Both speculative trails differ from ordinary at token index 6. Different
generated lengths mean this table compares token throughput, not equal-answer
completion time. The second ordinary request completed at 14.2908 tok/s with
identical output tokens and decoded response. Its speculative comparisons
remain pending. Private observation SHA256:
`25258a8b08bbc68ddcb4637c3effc56d206b45f539b97c361e4927762bab68f4`.
These remain provisional rank0 observations, not the completed fleet summary.

Scoped self-reviews also found technical corrections in the two speculative
final answers. R2 incorrectly claims MTP eliminates the draft/target distribution
mismatch and counts a correction token as an accepted draft. R3 describes an
invalid support-only sampling acceptance rule and averages per-round rates
instead of dividing expected token count by expected elapsed time. Their
concrete cache rollback examples pass this review. Generic parallel MTP heads
are not penalized. The repeated ordinary response reuses its review only after
exact response/token hash agreement. None of these is an independent review or
a model-wide quality score; raw text and detailed assessments remain private.

Review SHA256: R2 `f51cc2dcd20517bf55fc4034a050b489203a697f08f853cd751307f8c9d21f29`;
R3 `2fc115576bb8246db05c75be9c6dbd6d4f0ecae83cc564c1e4461a96b52ed7cd`.
The publisher's manual-review attachment passed one valid bound case and six
refusal checks (response/token identity, case, mode, independence and inconsistent
correctness). This validates post-processing boundaries only. Next manual
workload observation is after 04:34:00 UTC; no workload or numerical source change.

At 04:34:21 UTC the same controller remained live, with all eight logs at
`native.prose_repeat2.r3_prefill_health_1` and no recorded failures. The second
R2 prose request completed at EOS with 6,134 tokens at **13.3001 wall tok/s**,
0.9307x its paired ordinary rate. Its token array is exactly equal to the first
R2 run, allowing the same scoped answer review with freshly bound case identity.
The two-draft repeat remains pending. Snapshot SHA256:
`628a033397e7df025bc7e8a3fd82e3d952409b4d564fe3ef8a37a515d7ef7e0b`.
Next manual observation is after 04:45:00 UTC. No restart or overlapping workload.

At 04:45:19 UTC the same controller remained live; all eight logs reached
`native.code_repeat1.r2_prefill_health_2` without recorded failures. Repeated
R3 prose completed at **13.4504 wall tok/s**, 0.9412x its paired ordinary rate.
Its 3,745-token EOS output is exactly equal to the first R3 run. All six prose
responses now have completed, hash-bound scoped reviews. Across the two repeats,
ordinary measured 14.2908–14.4424, R2 13.2894–13.3001 and R3 13.4504–13.4892
wall tok/s on rank0. These are descriptive repeat ranges, not fleet aggregation
or confidence intervals; both speculative modes lost in both repeats.

The first ordinary code request reached the matched 7,168-token cap at
**14.2740 wall tok/s**. Pinned-tokenizer decoding confirms no closed reasoning
or finished answer; its correctness remains unestablished. No intermediate code
is treated as a delivered solution. The speculative code comparisons and later
cases remain pending. Private observation SHA256:
`13602c29b4ecd8ff31186dfd57e6845533cff24eef93c56e67c18833ab60e724`.
Next manual observation is after 04:56:00 UTC.

A private generated-function checker is prepared for any completed code answer,
separate from the existing final-schedule oracle. It checks 227 small inputs by
exhaustive subset search, returned schedule feasibility/value, and same-input
determinism across 454 calls. A reference implementation passes and three broken
fixtures fail. This establishes checker readiness only: no generated function
has yet been executed or declared correct. Private checker SHA256:
`24616c6cace20ac2fbd4ec1bd2ad27d596791a0fb599cf2384a13f6edd3c3deb`.

At 04:56:21 UTC the same controller remained live; all eight logs reached
`native.code_repeat1.r3_prefill_health_2` without recorded failures. Code R2
completed 7,168 generated tokens at **13.6045 wall tok/s**, versus ordinary
14.2740 (0.9531x). It accepted 3,385/3,782 drafts, averaging 1.8950 emitted
tokens/round, but target verification consumed 456.99 of 526.81 decode seconds.
Its output first differs from ordinary at token index 5. Pinned-tokenizer and
token-file checks confirm this response also exhausted its budget during
reasoning, with no completed answer. No final schedule or generated function
can be marked correct. R3 and the remaining requests are still pending.

Private immutable observation SHA256:
`de2b5acb7283dcec0890650dfb0d97f7cfdbdead31b9d536ac721541b0364813`.
Private decoded response SHA256:
`8da00714a2f5f18e266abc6daaa9c25a1fdfcad3027aef75bb8fb344814636bd`.
These remain provisional rank0 results, not a completed fleet summary.
Next manual observation is after 05:07:00 UTC. No restart or source change.

At 05:07:17 UTC the same controller remained live; all eight logs reached
`code_repeat2_prefill_health_2` without recorded failures. First code R3
completed at **14.9319 wall tok/s**, versus ordinary 14.2740: **1.0461x**.
It accepted 2,425/2,634 first and 2,108/2,634 second drafts, averaging 2.7210
emitted tokens/round. Verification consumed 425.11 of 479.98 decode seconds.
This is a small prompt-dependent throughput gain, below the working 25% target;
its repeat is still pending. It is not a token-exact replacement: the first
mismatch with ordinary is index 5. All three code modes used the 7,168-token
cap without a completed final answer, so answer correctness is unestablished.

Private observation SHA256:
`117ec2d6a0925fe480be4a489284ed357d655f7e20ec51a1a97278ad4d69acad`.
Pinned-decoded R3 response SHA256:
`2299f55b1080b8483b5b950ced00690e243401b061dacf8ae3455390241f7c62`.
The stored token stream, terminal reason and private oracle identities were
checked; intermediate reasoning is not graded as a delivered code solution.
Repeated code and structured comparisons remain active/pending. These are still
rank0 observations, not the completed fleet summary. Next observation is after
05:18:00 UTC; no restart, source change or overlapping workload occurred.

At 05:18:17 UTC the original controller remained live. All eight logs reached
`native.code_repeat2.r2_prefill_health_2` without recorded failures. Repeated
ordinary code completed at **14.3027 wall tok/s**, compared with 14.2740 in the
first run. All 7,168 token IDs and the decoded response are identical; it again
ends at the cap during reasoning with no final answer. The speculative repeats
and structured cases remain pending. Private observation SHA256:
`4113eb67a0aa7cecebc0429570d34b8fef66be5bed5ab157c9121317357438cd`.
This remains provisional rank0 evidence. Next observation is after 05:29:00 UTC.

At 05:29:22 UTC the same controller remained live, with all eight logs at
`structured_repeat1_prefill_health_5` and no recorded failures. Both speculative
code repeats completed. Rank0's two-repeat ranges are:

| Code mode | Wall tok/s | Paired ratio range |
|---|---:|---:|
| Ordinary | 14.2740–14.3027 | 1.0000x |
| One draft (R2) | 13.5887–13.6045 | 0.9501–0.9531x |
| Two drafts (R3) | 14.9183–14.9319 | 1.0430–1.0461x |

Each mode's repeated token array is exactly equal to its first run. R2 again
accepted 3,385/3,782 drafts; R3 accepted 2,425/2,634 first and 2,108/2,634
second drafts. All six code responses exhausted 7,168 tokens during reasoning,
so none delivered a final schedule or function to validate. The repeated R3
gain is 4.3–4.6%, below the 25% working criterion, and both speculative modes
still differ from ordinary at token index 5. These are descriptive rank0 repeat
ranges, not a confidence interval or the completed fleet summary.

Private immutable observation SHA256:
`d23cdd1546b0386fc0457d8b89124da3bb9152ee1d68ffadb1d6a3dfabc9c773`.
Pinned decoding, private-oracle identity, token streams and terminal reasons
were checked for both new completed requests. The structured comparisons,
final fleet aggregation and cleanup remain pending. Next observation is after
05:40:00 UTC; no restart or overlapping workload occurred.

At 05:40:23 UTC the same controller remained live; all eight logs reached
`native.structured_repeat2.r2_prefill_health_5` without recorded failures.
The first structured comparison completed at EOS:

| Mode | Wall tok/s | Generated tokens | Paired ratio |
|---|---:|---:|---:|
| Ordinary | 14.3264 | 2,629 | 1.0000x |
| One draft (R2) | 14.1185 | 2,390 | 0.9855x |
| Two drafts (R3) | 15.8832 | 2,045 | 1.1087x |

R2 accepted 1,189/1,201 drafts and emitted 1.9892 tokens/round. R3 accepted
687/699 first and 659/699 second drafts, emitting 2.9242 tokens/round. Their
outputs first differ from ordinary at token indices 816/823. R3's 10.9% gain
is below the working 25% criterion; its repeat remains pending. Different output
lengths also affect request completion time independently of token throughput.

All three completed answers contain **exact correct JSON values**, including
types and ordering, but enclose the object in Markdown code fences. This fails
the requested standalone JSON format, so the strict oracle reports
`values_correct_format_failed`, not incorrect arithmetic. Removing only those
fences yields the expected object in all three cases. The repeated ordinary
request finished at 14.3701 tok/s with all 2,629 tokens identical and the same
formatting limitation. Four private responses were checked against authenticated
oracle/tokenizer identities and their original token streams. No raw answers
or input records are committed.

Private immutable observation SHA256:
`f4a908ee06d8544e79a6cea6a6f2fe2056f0a469ea74dd755815816d89abfcd5`.
The final two speculative repeats, completed fleet summary and authenticated
cleanup remain pending. Next observation is after 05:51:00 UTC.


## Completed representative suite and cleanup (2026-09-20)

The original controller exited successfully, confirmed at 05:51:11 UTC. All eight
rank receipts are complete and native-complete; every original cleanup receipt
confirms its host idle. The strict publisher authenticated archived source,
controller/native-pack/input identities, all admissions, NPZ/JSONL token trails,
answer checks and fleet agreement, then aggregated the two fresh request repeats.
[Completed eight-host receipt](tpu-real-native-suite-20260920T031727Z.json).
Receipt SHA256: `c65eb6b3125193de83404bafa01323cf62ad8f5b6f477c2f5d44c3602e3897e7`.

| Request | Ordinary wall tok/s | One native draft | Two native drafts | Two-draft paired change |
|---|---:|---:|---:|---:|
| Prose | 14.29–14.44 | 13.29–13.30 | 13.45–13.49 | −6.6% to −5.9% |
| Code/reasoning | 14.27–14.30 | 13.59–13.60 | 14.92–14.93 | +4.3% to +4.6% |
| Structured | 14.33–14.37 | 14.12–14.15 | 15.88–15.90 | +10.7% to +10.9% |

Ranges cover two fresh requests and synchronized host reports, not independent
host trials or confidence intervals. Every mode reproduced its own token trail
on repetition. Speculative trails differ from ordinary at index 6 for prose,
5 for code, and 816/823 for structured R2/R3. No mode meets the 25% working
criterion or qualifies as a token-exact replacement. Keep ordinary as the
DB610-qualified research baseline; do not deploy this speculative experiment.

Prose completed but needs technical corrections under scoped assistant self-review;
all six code responses exhausted 7,168 tokens during reasoning without a final
answer; all six structured answers have exact correct values but Markdown fences
violate the requested standalone JSON format. These are bounded answer checks,
not independent review or a model-wide quality score. Target verification
remains the dominant speculative cost. Maximum recorded native peak HBM is
28,228,678,144 bytes/chip. No prefill speedup is claimed from decode measurements.

The repeated structured R2/R3 rates are 14.1466/15.9028 wall tok/s on rank0;
all output tokens repeat exactly and the final values/format boundary is unchanged.
The receipt includes per-mode prefill, TTFT, request/decode wall, ordinary p50/p99,
speculative round p50, acceptance by draft position, all component/vote costs and
native HBM. Both publisher and repeat-aggregation source files are preserved with
the private originals. Only hashes, aggregate counts and scoped assessments enter
Git. No model-produced Python was executed because no code response delivered a
finished function. The original long-question incomplete result stays preserved.

The experiment is complete as a measured mixed/negative outcome. Final eligible
main publication and its release/review/regional backup checks remain outstanding.
No further hardware workload is active or queued.


## Final publication and objective closure (2026-09-20)

The completed comparison is published on private main at `c142d284`, a fast-forward
from `5e9ce605`. Nine changed files contain only documentation, two compact
receipts and the curation ledger. No runtime is deployed. Implementation and
rejected variants remain preserved at research checkpoint `f097649a`.

Final CPU release checks passed: 524 tests passed, one skipped; doctor, content
inventory, frozen-source verification, compileall and isolated wheel checks passed.
The candidate was self-reviewed, not independently reviewed. Pre-main regional
mirror verification passed before publication. Post-main verification initially
stopped on a shared Git `FETCH_HEAD` checksum mismatch; the original failure is
preserved. An explicit recovery reran the same regional mirror and full path,
checksum, local-stability and cloud-generation checks without repeating the merge.
No repair was needed after that mirror; no files were excluded or checks relaxed.
The final release tree (1,608 files) and shared Git tree (2,614 files) both passed.
Fresh authenticated checks again found all eight hosts idle. No job is queued.

[Publication receipt](mtp-evidence-promotion-20260920.json) binds the exact main
pin and generation-verified regional artifacts. Its backup covers publication
time, before this research-only status follow-up. The final requirement audit
is satisfied as a measured mixed/negative result: ordinary remains the qualified
research baseline; the 25% gain criterion is unmet and output equivalence remains
unresolved. All answer-quality limits in the completed comparison still apply.


## Owner-requested Kaggle implementation review (2026-09-20)

[Direct source comparison](KAGGLE_MTP_REVIEW_20260920.md) inspected both pinned
model folders, Qwen rollback/configuration, GLM speculative/cache/test paths and
our current implementation. The earlier general survey is not proof that this
comparison had already been completed. Long target parity remains unresolved;
the mixed speed result does not establish a hardware ceiling. No new runtime
change, test execution or TPU workload occurred. The next diagnostic is a short
same-prefix comparison at the first mismatch, before another long benchmark.


## Public verifier reuse follow-up (2026-09-20)

[Upstream implementation and patch comparison](UPSTREAM_MTP_REUSE_20260920.md)
now records actual target-forward call paths, current merged/open PR status,
source identities and concrete reuse decisions. The most useful new attention
candidate is Kaggle's global-max plus FP32-numerator reduction, differing from
our rejected BF16-partial LSE merge. This is a source-derived candidate, not a
proved fix or measured gain. Shared target arithmetic and compact device
orchestration are the other priorities. No code was deployed or TPU job launched.


## Original TPU Inference follow-up (2026-09-20)

Expanded the [reuse report](UPSTREAM_MTP_REUSE_20260920.md) with the merged fused
EP MoE kernel, its two/three-row token-partition obstacle, and open GLM5.1 v4
correctness patches. Our route-slot restoration already represents the reported
token-alignment invariant. The closest sparse-DSA PR is our own contribution;
it supplies primitives rather than independent full-verifier validation. Ten
additional pinned source files and six PR identities are recorded in the receipt.
Source review only; no runtime changes or new TPU workload.


## Activated public-reuse goal: same-prefix diagnostic implementation

The owner activated the rewritten goal on 2026-09-20. Added
`compare_same_prefix` and the private `prefix_replay` worker path. Every R1/R2/R3
window starts from ordinary teacher-forced state, preventing accumulated verifier
cache drift from confounding the comparison. Every prefix including zero compares
full state, with separate written-span cache statistics and differing cache layer
IDs. Private historical code/prose inputs are hash-bound and restricted to short
windows. Actual trained replay remains pending at this implementation checkpoint;
layer cache differences do not establish the first differing arithmetic operation.

CPU validation: `test_same_prefix_replay.py` plus `test_speculative_diagnostics.py`
55 passed; replay plus authenticated-input tests 24 passed (overlapping suites).
Fault injection detects wrong predictions, stale historical references, future-row
rollback failures and unhealthy proposals; a drifted proposal cannot contaminate
the next replay root. No model body or frozen source changed. The next workload
uses the existing controller leases/admission/cleanup and ordinary DB610 gate.


### First short TPU replay launched; strict summary prepared

Run `perf_real_prefix_replay_20260920T090655Z` launched immutable worker `c430276b`
on the existing fleet after authenticated eight-host idle checks, holding the
workload/pod leases and disabling SSH workload retries. Controller:
`/tmp/run_perf_prefix_replay.py`; originals under `/home/gianl/glm-run/` plus the
run tag. It authenticates historical ordinary token hashes and private prompt
files from `dc047933` before staging 16-token code/prose references. R1/R2/R3
windows begin at offsets 0 through 6. This is a correctness run, not speed evidence.

A separate summary change validates every window, all commit counts, required
phases, cache comparisons, written spans and four memory records per host. The
existing DB610 summary also admits exactly the expected replay graphs and checks
fleet/source/checkpoint identity. It publishes aggregate differences only. CPU
summary plus real-validation suites: 69 passed; after adding slot-map refusal,
all 16 dedicated summary tests passed. These summarizer changes are not part of
the immutable running worker. No trained result or cleanup outcome is claimed yet.

| Public reuse candidate | Current experimental disposition |
|---|---|
| Shared target arithmetic / Kaggle parity structure | Same-prefix trained replay completed, eight-host summary/cleanup passed. Code R2/R3 and prose R3 diverge from identical roots before acceptance; layer/head localization prepared. |
| Kaggle global-max / FP32 numerator local attention | Standalone adaptation and 1/3/4/32-row CPU checks complete; paired TPU primitive benchmark prepared, hardware/trained tests pending. |
| TPU #3332 verification sizing / page boundaries | R1/R2 TPU lowering has one-query attention calls, no prefill-sized query interface; existing CPU owner/page-crossing and future-draft independence tests pass. No direct RPA classification patch applies; global-max multi-query attention remains the experiment. |
| TPU proposal JIT / device rejection / inactive rows | Supported R2/R3 input construction is already one compiled program; component-profiling-off ablation is implemented/CPU-checked, TPU comparison pending. Device acceptance remains separate work; exact row shapes introduce no padded requests. |
| Fused EP MoE #3040/#3388 | Small-row/layout obstacles recorded; adaptation and measured admission pending. |
| Grouping / indexing #3219/#3476 | #3476 division-based gather already present; route/IO/M8 CPU checks passed. #3219 larger buckets remain a tuning reference; current M8 uses active-group bounds. |
| v4 / token alignment #2324/#2248 | Source invariants compared; regression coverage audit pending. |
| Existing MTP fixes / sparse primitives | Represented mechanisms and own PR identified; targeted validation/disposition pending. |


### Global-max attention CPU candidate; first replay still compiling

Added `global_max_attention_mapped`, adapted from the pinned Kaggle formulation
with its MIT notice retained. It uses our absorbed NoPE/RoPE geometry and scale,
canonical owner-local paged selection, query exchange, global maxima and FP32
numerator/denominator reduce-scatter. It returns zero for empty selections and
propagates invalid live operands across owners. It does not replace a model body.

[CPU receipt](global-max-attention-cpu-20260920.json): four cases (1/3/4/32 rows),
eight CPU owners, small H8/latent128/K128 geometry. Maximum observed absolute
error versus frozen online-softmax interpreter is 0.001953125; no bitwise or
trained-parity claim. Tests include empty owners, nontrivial physical pages,
causal boundaries, duplicate IDs, live and unselected NaNs. The first 32-row test
fixture included future positions and correctly failed health; corrected fixture
positions, without relaxing the causal guard or tolerance.

`tools/perf_tpu_microbench.py --which global_max_attention` now prepares paired
frozen/global-max primitives at real H64/latent512/K2048 geometry for 1/3/4/32
rows and prefix/balanced/concentrated selections, with graph consensus, HLO and
memory admission, numerical differences and synchronized timing. This workload
has not launched while the trained replay owns the fleet.

At the 09:19 UTC observation of `perf_real_prefix_replay_20260920T090655Z`,
the controller and authenticated rank0 worker process were live. Checkpoint load,
BF16 preparation and prefill128 compile/graph/HLO/memory checks had completed;
no failure was recorded. The current execution remains pending. Next routine
observation should be at least ten minutes later unless diagnosing a failure.


### Layer/head follow-up diagnostic prepared

`verifier_trace.py` observes the existing ordinary/verifier layer bodies rather
than introducing another target implementation. Optional callbacks export each
layer's normalized input, hidden update, carried residual and selections. The
head probe consumes the live pre-final-norm pair: recomputing from the rounded
`final_residual_local` would change the arithmetic. Compact top-two candidates
are exchanged; raw IDs and activations do not enter the public summary.

Extra outputs can change compilation, so the follow-up compares instrumented
and original predictions, residuals and proposal/state fields. The summary marks
instrumentation drift explicitly; its first differing layer cannot automatically
be attributed to the original graph. The traced replay is opt-in behind a pinned
input bundle and retains graph/HLO/memory admission. Controller
`/tmp/run_perf_prefix_trace.py` is prepared, not launched or queued.

CPU checks: the eight-layer/CPU32 fixture passed ordinary and three-row verifier
trace geometry, original prediction agreement, head prediction agreement and
exact lowest-ID ordering for a fully tied vocabulary (1 test, 89.22 s).
Helper/summary/replay/real-validation checks passed 90 tests; after adding an
additional consistency check and CLI refusal, 35 summary/input tests passed.
No trained layer trace has run yet.

Source applicability checks also passed 26 existing route, WS32 I/O and M8 tests.
`gather_prefill_route_rows` already uses `sorted_flat_ids // top_k`, the exact
index construction proposed by #3476. M8 projection limits its grid to active
expert groups, restores original route slots, and tests independent bitwise
projection plus empty/invalid owners. Compact sampling tests verify tied token
selection and absence of a full-vocabulary all-gather. These are represented
mechanisms; no new speedup is attributed to them.

At 09:29 UTC the authenticated rank0 process remained live, with no recorded
failure. DB610 had passed and code prefill reproduced the historical first token;
the first verifier graph was compiling. Next routine observation is >=09:39 UTC.

At 09:39 UTC the controller and authenticated rank0 worker were live. The code
R1/R2 windows were complete locally: ordinary reproduced the retained history;
R1 predictions agreed, while R2 first disagreed at output index 5 (offset 3,
row 1, and offset 4, row 0). Residuals and selected metadata differ even for R1.
Every zero-prefix commit compared equal on rank0. These are provisional local
observations, not the eight-host summary or evidence that all cache owners agree.
The follow-up layer/head trace is justified by a mismatch from identical roots,
without draft acceptance or accumulated verifier state. Next routine observation
is >=09:49 UTC. The final combined replay/input/summary/trace/real-validation CPU
check passed 104 tests in 89.95 s, including the CPU32 layer/head fixture.

### Verification-window lowering and boundary audit

[Static TPU graph receipt](verifier-window-lowering-20260920.json) authenticates
the rank0 R1/R2 StableHLO and optimized HLO against the immutable replay record.
Both expose 78 static sparse-attention calls with a single-query input/output;
R2 retains that interface within mapped control flow. The selected-KV all-reduce
has 2,621,440 output bytes at R1 and 5,242,880 at R2 per layer. These are shapes
and static occurrences, not runtime collective counts or measured traffic.
They support testing local attention; they do not show a prefill-sized query
padding bug. Internal MXU padding and capacity-dependent cache/DSA work remain.

The existing rowwise attention CPU32 test passes exact output/cache/score
comparison for full/shared index layers at starts 61/127/510, five rows,
64-row owner shards, 512-row pages and physical page permutation [2,0,1].
Changing future drafts leaves the first query's output/selection unchanged.
This covers the relevant #3332 crossing invariants in our different runtime.

Added the missing R2 shape to the standalone global-max CPU test and prepared
TPU primitive matrix (now 1/2/3/4/32 rows). The R2 test passed with maximum
absolute difference 0.001953125 from frozen online softmax;
[separate receipt](global-max-attention-cpu-r2-20260920.json) preserves the earlier
four-shape receipt. R2 plus the boundary test: 2 passed in 26.88 s.
The single-launch controller `/tmp/run_perf_globalmax_attention.py` is prepared
with pinned-source verification, leases, authenticated idle/cleanup and no SSH
workload retries. It is not launched or queued; the current replay still owns
the fleet and trained mismatch localization comes first.

At 09:49 UTC, the same controller and rank0 worker were authenticated live,
without a recorded failure. Code R1/R2/R3 and prose R1 completed all seven
local windows and reproduced their historical ordinary references. Code R2/R3
both first disagree at output index 5; code/prose R1 predictions agree. All
observed zero-prefix commits compare equal on rank0. This is still provisional
local evidence; the owning cache ranks and completed fleet summary are pending.
Next routine observation: >=09:59 UTC. The follow-up controller remains prepared
only; no second workload was launched.

### Component-profiling ablation prepared

`--native-component-timing none` removes the extra per-component waits in
`native_comparison.py`. It retains the same compiled draft/verify/commit/refresh
executables, the session's proposal and commit/refresh readiness checks, fleet
votes/plan agreement, failure handling and rank0 write/flush. This isolates
profiling barriers; it is not whole-round JIT or device acceptance. R2/R3 input
construction is already one compiled program in `mtp_state.py::inputs_mapped`;
R3 has one additional recurrent native call after its cached first proposal.

Disabled component measurements are `null`, never asynchronous dispatch times
or invented zero device costs. The summary authenticates the timing mode against
the controller and every rank, rejects mixed modes/fabricated components, and
still checks request-phase wall accounting. The legacy blocking mode remains
the default. The future ablation controller must explicitly record
`native_component_timing: "none"` to match the worker flag.

Native summary/request/base-validation/input checks: 106 passed in 4.90 s.
The strict updated summarizer also accepted all eight original representative
suite records at `perf_real_native_suite_20260920T031727Z`; preserved public
receipts were not rewritten. This option has not run on TPU and makes no speed
claim. It is ready for the paired orchestration experiment after correctness
localization; the original replay remains immutable.

The follow-up traced replay now records hashes of each original verifier and
ordinary prediction window. The strict trace summary checks the ordinary hash
against the pinned teacher-forced reference and distinguishes equal mismatch
patterns from actual cross-host prediction identity. A synthetic last-host
disagreement is preserved as a negative result. The first untraced worker's
older schema remains supported with its explicit weaker identity scope.
Replay/summary/base-validation checks: 93 passed in 3.08 s.

### First same-prefix TPU replay completed

[Fleet receipt](tpu-real-prefix-replay-20260920T090655Z.json), immutable worker
`c430276b`: the controller exited successfully, all eight rank receipts completed,
DB610 passed, and the strict summary plus authenticated eight-host cleanup passed.
The 09:59 UTC observation confirmed the controller and worker had terminated.

All ranks agree on the reported mismatch patterns: code R2 first differs at
output index 5 in offsets 3/4; code R3 at index 5 in offsets 2/3; prose R3 at
index 6 in offset 4. R1 and prose R2 predictions agree across these seven reset
windows. Prose R2's earlier long divergence remains unresolved because this run
does not accumulate verifier state between windows. R1 is not bitwise equivalent:
residual, later-layer cache and selection differences remain despite token matches.
For R2/R3, cache differences reach layer 0; this is not yet an arithmetic trace.

Every zero-prefix commit is bitwise equal across all eight hosts. Across 1,344
rank/window/prefix/cache comparisons, the full-cache numerical difference count
equals the accepted written-span count. Thus no counted numerical differences
occur outside accepted spans; this count alone is not a signed-zero bit proof.
Positions, lengths, counts and contract-health fields agree. Peak HBM remains
28,228,678,144 bytes/chip. No drafter or served-throughput claim follows.

The next diagnostic uses code offsets 0/3 and prose offsets 0/4, with layer/head
observations and original prediction hashes. Code offset 3 reproduces both R2
and R3 disagreement; using only offset 4 would miss R3. Its instrumentation must
be checked against original outputs before attributing the first differing layer.

### Trained layer/head trace launched

`perf_real_prefix_trace_20260920T100227Z` launched at immutable worker
`d8bf78eb6d7f92f964d8c4884a3f04dd878d5f07` after fresh eight-host authenticated
idle checks. It owns the workload/pod leases, uses one-shot SSH and includes
code offsets [0,3], prose [0,4], and R1/R2/R3. The controller is
`/tmp/run_perf_prefix_trace.py`, PID 1179243; originals are under
`/home/gianl/glm-run/` plus the run tag. At 10:03 UTC both controller and
authenticated rank0 worker were live. Next routine observation >=10:13 UTC.
No result, arithmetic root cause or cleanup outcome is claimed yet. The
global-max primitive and profiling-off comparisons remain prepared, not queued.

### Four-row CPU gate and repeated compilation

The current M8/rowwise-DSA four-row verifier failed its unchanged CPU numerical
envelope. The first attempt stopped at residual max error 0.1640625 (>0.0625).
A complete diagnostic then retained that failure and checked the remaining
invariants: fourth-prefix KV error also exceeded the bound (0.0718994140625).
Token agreement, future-draft independence, selection IDs, zero/rejected physical
rows and invalid-token/span refusal passed on this eight-layer synthetic fixture.
[Receipt](mtp-r4-rowwise-cpu-20260920.json). The test records one expected failure
(76.98 s), not a qualification pass; thresholds are unchanged and no four-row
native serving path was enabled. Fix target arithmetic before revisiting it.

The first replay's code/prose verifier graphs have identical StableHLO and
optimized HLO hashes for each of R1/R2/R3. Nevertheless, rank0 spent a further
543.13 seconds compiling the prose verifier copies (144.30/195.45/203.38 s).
`prefix_replay.py` now retains the same jitted program objects across cases,
including trace and commit programs, and clears them after all cases. Tokens,
weights and caches remain dynamic arguments; every case still runs graph
consensus, HLO inspection and live memory admission. A local JAX check confirmed
executable reuse with fresh same-shape inputs, and 51 replay/input/summary tests
passed in 2.68 s. Model-level compile savings and retained-code memory cost are
not yet measured. The immutable running trace is unchanged by this improvement.

[Exploratory R4 CPU layer trace](mtp-r4-cpu-trace-20260920.json) uses the same
fixture and observes both target bodies. Instrumented verifier fields and every
ordinary result/state compare bitwise equal to their original executables.
Hidden updates and carried residuals first differ at zero-based layer 1, among
the initial dense layers before routed experts; normalized inputs first differ
at layer 2. DSA score bits differ at layer 0, but selected positions/counts match
throughout. This narrows the CPU investigation without identifying a particular
attention/normalization operation or claiming trained equivalence. Score error
magnitudes are omitted because subtracting matching infinite padding produced
an unusable exploratory metric; bit comparisons and finite activation metrics
remain valid. The private script and original report hashes are retained.

At 10:13:54 UTC, the `perf_real_prefix_trace_20260920T100227Z` controller and
authenticated rank0 worker were live. Its last completed phase was BF16 weight
preparation, with no recorded failure; the prefill graph was compiling. No
trained layer observation exists yet. Next routine observation >=10:24 UTC.

### R4 CPU attention preparation localization

The sequential-attention ablation kept M8 expert pooling and reproduced every
reported error metric of the batched/rowwise baseline, including residual
max error 0.1640625 and fourth-prefix KV max error 0.0718994140625.
[Receipt](mtp-r4-sequential-attention-cpu-20260920.json). The sequential path
computes DSA inside each row; its `rowwise_dsa` option is false because that
flag applies only to batched attention. Matching error metrics do not establish
bitwise equality between the two verifier outputs.

A [normalization-boundary probe](mtp-r4-norm-trace-cpu-20260920.json) narrows the
first differing hidden update to attention output in dense layer 1, rows 2/3
(zero-based). Inputs to that layer's attention agree; its output differs by up
to 0.0078125 before post-attention normalization. A separate
[preparation probe](mtp-r4-prepare-trace-cpu-20260920.json) finds an earlier
single-element current-KV difference at layer 1, row 1: 0.000244140625. Prepared
queries still agree through that layer. Both probes reproduce the original
verifier and four ordinary results/states bitwise when compared as global arrays;
they do not separately assert each replicated CPU shard.

Replacing only the preparation `lax.map` with Python-unrolled one-row calls
removes the prefix-2 KV discrepancy. It does not qualify R4: residual max error
remains 0.1640625, fourth-prefix KV max error is 0.0675048828125, and the unchanged
envelope still fails. [Diagnostic receipt](mtp-r4-unrolled-prepare-cpu-20260920.json)
includes the executed injection and base-source identities. No runtime default
changed. These synthetic CPU probes guide the next ablation; they do not identify
the trained TPU cause or establish a speed gain.

At 10:24:30 UTC, the existing `perf_real_prefix_trace_20260920T100227Z` controller
and rank0 worker were authenticated live. The last completed phase advanced to
`replay_code_first_token`; the receipt is incomplete with no recorded error.
Continue this same leased run. Next routine observation >=10:34:31 UTC.

### Opt-in unrolled attention passes the R1–R4 CPU gate

Fully unrolling one-row attention calls inside each layer removes all reported
R4 errors in the exploratory fixture, while preserving pooled M8 expert work.
This is now an opt-in `unrolled_attention=True` verifier option, mutually
exclusive with batched attention. It retains the layer-major verifier and
ordinary attention body; it does not scan whole decoder steps. Default batched
verification and native serving selection are unchanged.

The strengthened R1/R2/R3/R4 CPU checks all pass: **4 passed in 303.32 s**.
Residuals and every field of every nonempty committed state, including DSA
scores, agree bitwise on all addressable CPU replicas. Existing token, causal,
zero/rejected-write and refusal checks also pass. The two trace tests pass in
132.49 s, including an unrolled R3 original-versus-instrumented bit comparison
and tied-head ordering. [Receipt](mtp-unrolled-attention-cpu-20260920.json).
This establishes the bounded synthetic CPU result, not trained target parity.

`--prefix-replay-unrolled-attention` selects this candidate for the short trained
diagnostic only, with matching trace support. The flag requires a pinned replay
before runtime initialization, and strict summaries require controller plus all
eight rank mode records to agree. The 94 replay/input/summary/validation checks
pass in 4.25 s. The release command also passes (524 tests passed, one skipped),
including unchanged frozen source and isolated package checks. Attention
exchanges remain per-row: TPU compilation, HBM and latency must be measured.
`/tmp/run_perf_unrolled_replay.py` is prepared, not launched or queued; it must
wait for the active trace and authenticated cleanup.

At 10:34:46 UTC the original `perf_real_prefix_trace_20260920T100227Z` controller
and rank0 worker were authenticated live, with `replay_code_r1_historical_reference`
complete. Rank0's code R1 offsets 0/3 retain prediction agreement, but traced
verifier residuals/selection metadata differ from the original executable.
The ordinary instrumented results remain stable on that rank. Therefore its
traced first activation difference at layer 4 is **not** an attribution of the
original mismatch. These are provisional rank0 observations; the other windows
and strict eight-host summary remain pending. Original and instrumented results
remain separately recorded. Next routine observation >=10:44:47 UTC.
