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
The synthetic prefix is explicitly 64 zero-cache positions. No TPU run has
started for this work yet.

Validation before the first acquisition: six focused CPU tests pass (acceptance,
physical commit, numerical characterization and benchmark comparison guards).
`tools/check_release.py` passes, including 524 tests with one skip; the frozen
source pin and isolated package check are unchanged. The curation ledger has no
unresolved rows. These are research readiness checks, not serving promotion.

Next: measure these verifier economics under the documented CPU numerical
boundary before loading the MTP extension. Reject any nonfinite path and require
trained-weight target agreement before promotion. Then implement the draft cache
and request loop and perform the goal's paired real-weight comparisons.
