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
No trained-target verifier acquisition or native MTP payload load has run yet.
