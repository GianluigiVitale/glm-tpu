# MTP positions, refresh and rollback

This is a source-derived implementation contract for the current GLM-5.2 MTP
experiment, not a completed drafter or an acceptance-rate claim. Target-verifier
economics are being acquired first. GLM-5.3 remains out of scope.

## Authoritative reference

The inspected vLLM revision is `36fa72d2d0d2f86c7c83e1e99c9012b7bd26463b`.
The GLM DSA architecture selects the DeepSeek DSA MTP implementation. Relevant
primary sources:

- [Target model](https://github.com/vllm-project/vllm/blob/36fa72d2d0d2f86c7c83e1e99c9012b7bd26463b/vllm/model_executor/models/deepseek_v2.py): final normalized target hidden states.
- [Generic MTP layer](https://github.com/vllm-project/vllm/blob/36fa72d2d0d2f86c7c83e1e99c9012b7bd26463b/vllm/model_executor/models/deepseek_mtp.py): normalized embedding/previous hidden, projection, transformer, final norm and shared head.
- [Proposer](https://github.com/vllm-project/vllm/blob/36fa72d2d0d2f86c7c83e1e99c9012b7bd26463b/vllm/v1/spec_decode/llm_base_proposer.py): `set_inputs_first_pass`, shifted token IDs with unchanged positions, target-hidden refresh and recurrent drafting.
- [MLA wrapper](https://github.com/vllm-project/vllm/blob/36fa72d2d0d2f86c7c83e1e99c9012b7bd26463b/vllm/model_executor/layers/mla.py): `skip_topk` suppresses the indexer call, while MLA cache updates still execute.

Retained metadata/source copies are under `/dev/shm/glm-mtp-reference`; no
upstream Python is executed. The proposer SHA256 is
`260ac95740b07fca10b5deb8b1e9f24e89577f35dcbc65db979ae5e5b26e2a77`;
the MLA wrapper SHA256 is
`594f71cb59ca7e9d7e26d76914272ce1f5b7ca280107756ca9dd03b1403cf1a2`.

## Position convention

Let `x[t]` be a target input token and `h[t]` its post-final-norm target hidden
state. The MTP input at **cache/rotary position t** is:

`eh_proj(concat(enorm(embed(x[t+1])), hnorm(h[t])))`.

The embedding is zeroed at position 0 before its norm, as in the reference.
The output at position t predicts `x[t+2]`. The token IDs shift by one; positions
do not shift. Subsequent recurrent draft steps increment the position and use
the previous draft's **post-final-norm** hidden state. The vocabulary projection
must not apply that final norm a second time.

For a prompt of length N, the target has consumed positions `0..N-1` and emitted
pending token `x[N]`. Draft prefill must consume target hidden states `h[0..N-1]`
with token IDs `[x[1], ..., x[N]]` at positions `0..N-1`. Its last output is the
first proposed token `x[N+1]`. Starting MTP at target position N, or starting with
an empty draft prompt cache, would implement a different draft computation.

The frozen target prefill API returns only its final token and cache state.
The research `export_mtp_hidden=True` option now additionally returns normalized
hidden rows and a live-valid mask through `glm_tpu/perf/mtp_prefill.py`;
see [the scoped CPU proof](mtp-prefill-export-cpu-20260920.json). Preserve the existing B128/B114 arithmetic, repair, health consensus and
atomic commit. Returning extra hidden rows can change compilation; compare the
target output and memory before treating that API as equivalent. MTP prompt
cache construction contributes to TTFT and must be measured separately.

## Each verification round

Suppose target position is P and its already emitted pending token is `x[P]`.
The verifier consumes `[x[P], draft[0], ..., draft[K-1]]`, at positions
`P..P+K`. It returns a target prediction and normalized hidden state for every
row. Greedy acceptance emits the matching prefix plus a correction/bonus, capped
at EOS or the remaining output budget. If c tokens are emitted:

- Exactly c target input rows are consumed. Commit target cache positions
  `P..P+c-1`; the last emitted token remains pending at position `P+c`.
- Restore rejected target KV/index rows and retain only the final accepted
  selection/frontier metadata. Refuse the round if any owner reports failure.
- Restore speculative draft writes beyond the retained draft prefix. Refresh
  draft positions `P..P+c-1` using **target** hidden rows and shifted next-token
  IDs. The last refresh row uses the target correction/bonus ID, not a rejected
  input proposal. The last refresh output supplies the first next-round draft.
- Do not retain recurrent draft hidden states in place of target-hidden refresh
  merely because their token IDs were accepted. The reference first pass uses
  the target hidden rows again; the cached draft computation must reflect them.

Draft refresh runs its full indexer for the consumed rows. Additional recurrent
steps reuse the last refresh row's shortlist with `skip_topk`; they still update
their tentative MLA KV cache. The reference skips the entire indexer call on
these steps. Their index-cache entries need not be invented: the next committed
refresh recomputes the accepted rows using target hidden states. Target and
draft caches, positions, selected keys and health must remain separate.

This refresh is real steady-state work. Include it, recursive drafting, target
verification, rejection rollback, host votes and delivery in measured accepted
output tok/s. Verifier-only perfect-acceptance estimates omit these costs.

## Input projection prototype

`glm_tpu/perf/mtp_projection.py` implements the input norms and `eh_proj` outside
the frozen model. It masks the position-zero embedding, preserves that row's
previous hidden state, gathers the two normalized feature sets separately, then
concatenates all embedding features before all hidden features. Projection
weights are BF16 `[H/4, 2H]`, replicated over expert8; the dot accumulates FP32.
Health is reduced over all 32 owners. It does not manage tokens or cache state.

The pinned [vLLM IR RMSNorm](https://github.com/vllm-project/vllm/blob/36fa72d2d0d2f86c7c83e1e99c9012b7bd26463b/vllm/ir/ops/layernorm.py)
rounds the normalized input to the weight dtype before multiplication. The
prototype uses the existing WS32 norm with zero carried residual to preserve
that BF16 boundary. The downloaded reference is retained without execution at
`/dev/shm/glm-mtp-reference/ir_layernorm.py`, SHA256
`b7621f663a97caf6908507b6cbe03adf23d0001a441d44af2b24c1ad0fc8bf57`.

The CPU32 synthetic H256 fixture agrees numerically with an independent
unsharded expression at 1/3/114/128 rows (observed max absolute error zero).
Tests check position-zero behavior, row independence, malformed weights,
negative positions and refusal when only the last physical replica is poisoned.
This is a projection-only proof, not an executed native drafter or GPU/TPU
parity. No trained MTP payload has been loaded.
[CPU receipt](mtp-projection-cpu-20260920.json).

## Required comparisons before use

Test the shift with a nonconstant prompt and with rejection at every draft
position; test the correction ID replacing the rejected input in the last
refresh row. Compare refreshed draft state against a teacher-forced MTP replay
on accepted target history, including page/owner crossings. Check IndexShare
first-pass versus recurrent behavior, EOS/length truncation, health refusal and
resume after rejection. Prove these with CPU fixtures before real-weight MTP
timing. No such complete drafter proof is claimed by this document.

## Static storage estimate

Using the authenticated packed layer-74 schema as the matching full-index body,
the planned resident MTP weights add 387,719,808 bytes/chip: 302,063,616 routed
FP8/scale bytes, 47,898,240 non-routed resident bytes, a 37,748,736-byte output-
sharded `eh_proj`, and 9,216 bytes of MTP norms. Existing embedding/head arrays
are shared. At capacity 8,192, one MTP KV/index-cache pair adds 1,572,864
bytes/chip; retaining 2,034 target hidden rows adds 6,248,448 bytes/chip.
[Static payload receipt](mtp-static-memory-projection-20260919.json).

This is a placement estimate, not measured allocation or HBM admission. It
excludes dequantization/repair workspaces, executables, allocation padding and
simultaneously live proposals. The planned `eh_proj` gathers the normalized
embedding and previous-hidden feature shards, then projects to the local output
feature shard. Packing and runtime placement still require implementation and
identity verification.

## Native transformer prototype

`mtp_draft.py` implements the separate single full-index sparse layer using
`MtpWeights`: native projection, one transformer, native shared-head norm, and
the target's existing embedding/head arrays. Full-index proposals accept up to
eight rows; recurrent IndexShare accepts one row and requires a populated
shortlist. The shared path omits DSA preparation, key writes and selection while
still updating MLA KV. It returns a `VerificationProposal`; the caller commits
to draft state only and retains the previous committed root for target-hidden
refresh. Host orchestration must enforce the shifted token/hidden identity.

The [CPU component proof](mtp-native-components-cpu-20260920.json) covers
composition, causality, shortlist reuse, skipped DSA tables, full-index refresh
and atomic refusal/rollback. The [metadata placement proof](mtp-native-placement-20260920.json)
checks every retained native tensor's owner intervals and shares base I/O rather
than copying it. Neither proof acquires trained payload, admits TPU memory/HLO,
executes the complete host protocol or establishes acceptance/throughput.
