# Public implementations to reuse for MTP verification — 2026-09-20

The owner asked for implementation reuse, not another explanation of why our
current experiment is slow. This review follows the verifier call paths and
merged fixes. The result is a concrete reuse order, with hardware/model boundaries.
No new TPU measurement or runtime change is claimed.

## Pinned sources

- Kaggle clone remains at `1aa1f083ae253470d9f355fe9c3eb12003300e91`, also the
  upstream main returned during this review. Both model folders were inspected.
- Public `vllm-project/tpu-inference` main:
  `9cab26a702c448c40710f504360d0d9f78e227a7` (2026-09-20).
- Our starting research source: `84120fe9`.
- [Source/status receipt](upstream-mtp-reuse-20260920.json) authenticates downloaded
  TPU source against the pinned Git tree's blob hashes and records PR merge/head
  identities. Original API responses/diffs/code are retained privately under
  `/home/gianl/glm-run/public_mtp_source_review_20260920`.

## How verification actually runs

| Implementation | Target verification | Cache and host handling | What to reuse |
|---|---|---|---|
| Qwen Kaggle / upstream TPU inference | The normal target `model_fn` receives the flattened pending+draft rows, causal positions and attention metadata. Target logits go through device rejection sampling. The MTP head proposes; it is not the verifier. | JIT the entire draft chain and input preparation; update rejection-adjusted lengths/positions on device. The Kaggle patch also restores GDN state by accepted checkpoint selection. | Shared target forward path and compact device metadata, rather than separate scalar/verifier implementations and repeated host materialization. |
| Kaggle GLM Flash | `resident.py::_run_verify` invokes the same `_prog_group`/`model.decoder_layer` used by ordinary execution, with T=k+1 and per-token histories. | `spec_decode` retains the matching prefix plus correction; restores small recurrent/pool states. Positional cache rows beyond the frontier are overwritten later. | Shared layer body and their random/oracle-draft parity tests. Preserve our stronger physical-row restoration until an equivalent visibility contract is proved. |
| Our GLM5.2 | `speculative_verify.py::verify_mapped` is a separate layer-major implementation. It pools expert rows but deliberately retains per-row dense projections, preparation, DSA and several reductions. | `speculative_request.py` separately synchronizes proposal, agreement, commit/refresh and delivery. The benchmark additionally blocks after each timed draft/verify/commit/refresh component. | Unify scalar and verifier arithmetic around shared layer functions; compare identical prefixes. Sharing code alone does not guarantee batch-invariant floating-point results. |

Qwen's dense hybrid model does not exercise our expert routing or DSA. Kaggle's
GLM serving scheduler uses ordinary `decode_rows`; its 64 tok/s is not an MTP
benchmark. Its small verification MoE path uses a kernel per token/expert slot;
it is not evidence that replacing our M8 expert reuse with that path is faster.

## Concrete public fixes and local disposition

| Public work, status checked now | Actual implementation | Local decision |
|---|---|---|
| [TPU #3332](https://github.com/vllm-project/tpu-inference/pull/3332), merged | Treat verification as multi-query decode, pass `decode_query_size`, avoid prefill padding; split new-KV DMA writes across page boundaries. | Reuse the small-window attention design and boundary tests. Our JAX path does not use this RPA classifier, so cherry-picking it cannot directly fix our verifier. Its Gemma/v6e throughput gain is not a WS32 prediction. |
| [TPU #2533](https://github.com/vllm-project/tpu-inference/pull/2533) and [#2535](https://github.com/vllm-project/tpu-inference/pull/2535), merged | Compile the full proposal chain and preparation, including recurrent input/position updates and selection. | Our proposal IDs are already device-generated, but timed component boundaries and host acceptance still split execution. Reuse the fused orchestration approach after correctness isolation; drafter-only savings cannot remove our dominant verifier cost. |
| [TPU #2610](https://github.com/vllm-project/tpu-inference/pull/2610), merged | Give padded draft requests zero sequence length so attention skips them. | Useful guard for future batching. Our experiment has one request and exact 1/2/3-row shapes, so this is not the observed central bottleneck. |
| [TPU #3178](https://github.com/vllm-project/tpu-inference/pull/3178), still open; [kernel #3189](https://github.com/vllm-project/tpu-inference/pull/3189), merged | Per-position GDN state checkpoints plus physical-slot accepted offsets; no full recurrent-state copying. | Essential for Qwen's recurrent target, already ported by Kaggle. Our GLM5.2 has no GDN state; reuse the rollback invariants/tests, not this model-specific patch. |
| [vLLM #44420](https://github.com/vllm-project/vllm/pull/44420), [#45895](https://github.com/vllm-project/vllm/pull/45895), [#47238](https://github.com/vllm-project/vllm/pull/47238), [#47448](https://github.com/vllm-project/vllm/pull/47448), all merged | Reuse MTP shortlist; retain the selected last query's shortlist; recycle post-final-norm hidden and apply that norm only once. | These mechanisms already appear in `mtp_state.py`, `mtp_draft.py` and `commit_prefix_mapped`: selected last committed row, disposable IndexShare continuation, normalized hidden and one norm. Re-audit against trained reference, but do not claim these as newly missing optimizations. |
| [vLLM #46448](https://github.com/vllm-project/vllm/pull/46448), merged | Select draft argmax with compact per-shard winners instead of exchanging vocabulary logits. | Already represented by our `ws32_greedy_sample_mapped` score/index exchange. |
| [TPU #3219](https://github.com/vllm-project/tpu-inference/pull/3219), merged | Size grouped matmul computation to actual group occupancy, with several tile buckets. | Keep as a future tuning reference. Our small verifier already packs each expert's at-most-eight rows into M8; the public larger bucket scheme is not automatically better. |

The current upstream sparse-MLA kernel is also real public code:
`tpu_inference/kernels/experimental/deepseek_v4/core_attention/sparse_mla.py`.
It accepts multiple query rows and per-query sparse indices, but uses DeepSeek-v4
FP8 cache packing, sliding-window partials, attention sinks and a different cache
contract. It cannot replace our GLM5.2 BF16 640-wide cache without an adapter,
geometry/dtype validation and v4 memory admission. No ready-to-run WS32 GLM5.2
replacement was established by this review.

## A concrete attention difference worth adapting first

Kaggle [GLM `_attend`](https://github.com/ARahim3/kaggle-tpu-lab/blob/1aa1f083ae253470d9f355fe9c3eb12003300e91/glm53-flash/engine/glm53/model.py#L594)
keeps selected keys local, gathers queries, takes a global score maximum,
computes local exponential sums and FP32 weighted-value numerators, then
reduce-scatters numerator and denominator. Only the final divided output is
cast to the key dtype. Its exp weights are still cast to the key dtype before
the value product: this is not an all-FP32 attention claim.

Our active verifier exchanges the aligned selected KV payload with `psum` in
`bf16_resident.py::index_share_attention_bf16`, for each batched query. Our rejected
D5 alternative instead merges already locally normalized BF16 outputs using LSE
weights (`lse_attention.py::merge_attention_scatter` and the partial kernels).
Casting each normalized local output before global combination introduces a
rounding boundary absent from Kaggle's formulation. This is a concrete code
difference, not proof that it caused D5's trained token mismatch. Both reduction
order and exponent quantization also differ from the frozen online-softmax path.

**Reuse candidate:** adapt Kaggle's global-max/FP32-numerator formulation to our
absorbed NoPE+RoPE queries, causal selections, empty-owner health and WS32 head
sharding, retaining multiple verification queries throughout. This targets the
large KV exchange and a known numerical boundary together. It must pass short
same-prefix comparisons before another long comparison; the previously rejected
D5 implementation remains disabled. Preserve attribution/license if code is ported.

## Priority and validation, grounded in our measured costs

1. Use the public shared-target-forward pattern and parity test structure to
   isolate the first long-output mismatch with identical state and forced inputs.
   Compare row predictions/logit margins, selections and accepted caches; distinguish
   arithmetic divergence from rollback bugs. vLLM's
   [batch-invariance implementation](https://github.com/vllm-project/vllm/blob/main/docs/features/batch_invariance.md)
   provides another concrete reference for fixed reductions, but its documented
   GPU/XPU support is not a TPU switch we can enable.
2. Adapt and measure the multi-query local-attention formulation above. Use #3332
   for explicit decode-window sizing and crossing-page tests. Share the resulting
   body with ordinary decode where possible instead of adding another independent
   target implementation.
3. Fuse the device proposal/verification/acceptance metadata path following the
   TPU proposer and rejection-sampling implementation, preserving required fleet
   agreement and delivery-failure handling. Keep detailed timing as a profiling
   mode and measure an unsplit end-to-end path separately.
4. Only then evaluate three drafts/four target rows and longer windows. Qwen's
   default three drafts is a tuning point, not permission to skip our wider-row
   correctness checks. No prediction of 30–60 tok/s follows from the public results.

Our first structured R3 run spent ~114.13–114.19 of 128.689 wall seconds in
verification, accepted 2.924 of at most three tokens/round, and spent ~4.0 seconds
in draft+refresh. Thus attention/verifier work is a stronger target than a new
or more elaborate drafter. This is inference from the completed receipt, not a
new per-kernel attribution. The previous result remains valid for the measured
implementation and does not exhaust these public implementation options.
