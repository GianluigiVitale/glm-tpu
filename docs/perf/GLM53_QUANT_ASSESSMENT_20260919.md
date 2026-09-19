# GLM-5.3 mixed INT4/INT8 on TPU v4 — assessment, 2026-09-19

Either checkpoint is a plausible starting point for reducing routed-expert weight
traffic and FP8 conversion overhead. Neither is a drop-in native-INT8 engine,
nor evidence that our 32 v4 chips would immediately reach thousands of prefill
tokens/s or 30–60 decode tokens/s. No weights were acquired and no GLM-5.3 run
was launched for this assessment. The owner paused the previous campaign.

## What was inspected

Public model cards, companion implementation/benchmark reports, Hugging Face
file metadata and pinned configs. [Metadata receipt](glm53-quant-metadata-20260919.json)
records revisions, config digests, file sizes and the local-config comparison.
Tensor payloads and quantization accuracy were not independently verified.

| Property | vikasclawd | Tech2wild |
|---|---|---|
| HF revision | `be70d304ef14dd0603e284ceb2549db67d5c8545` | `206507bbb047d8223964a0414cd83230c59428f9` |
| Safetensors files | 282 | 282 |
| Weight-file bytes | 405,241,861,872 | 405,241,870,672 |
| Main routed experts | W4A16, symmetric, groups of 128 | Same intended recipe |
| Other selected linears | W8A16, symmetric, groups of 128 | Same intended recipe |
| MTP selected linears | W8A16, per output channel | Same intended recipe |

Both use `compressed-tensors` / `pack-quantized`. Attention indexers, routing and
other sensitive tables are protected; this is not uniform INT4. Activations
remain 16-bit. The files total about **405 GB / 377 GiB**, not 378 decimal GB.
All 282 container hashes differ, despite almost equal sizes; headers and packing
can change file hashes, so this does not prove different numerical quality.
Sources: [vikas config](https://huggingface.co/vikasclawd/GLM-5.3-Int4-Int8Mix/blob/be70d304ef14dd0603e284ceb2549db67d5c8545/config.json),
[Tech2wild config](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix/blob/206507bbb047d8223964a0414cd83230c59428f9/config.json).

Their non-quantization configuration matches our GLM-5.2 configuration except
for the recorded Transformers version: 78 layers, hidden width 6144, 256 routed
experts with eight selected, expert intermediate width 2048, and the same
attention/indexer geometry. GLM-5.3 is a new weight set, but this does not reduce
our operation count. These are the full model, not GLM-5.3-Flash.

## What native INT8 on v4 does and does not buy

Google specifies **275 trillion operations/s per chip for BF16 or INT8** on v4,
with 32 GiB HBM and 1,200 GB/s bandwidth. It also identifies 8-bit weight loading
as useful for low-batch inference. Thus native INT8 is relevant, but there is
no advertised 2x peak arithmetic advantage over BF16 on this generation.
[Google TPU v4 specifications](https://docs.cloud.google.com/tpu/docs/v4).

A weight-only file does not automatically select integer matrix multiplication.
Our loader and kernels expect FP8 bytes with block scales; these files require
packed-integer decoding, different scale interpretation, sharding and kernels.
One option retains BF16 activations and fuses INT4-to-BF16 conversion. Another
quantizes activations and performs integer dot products, with groupwise scales
and accumulation. The second adds an accuracy boundary beyond these published
W4A16 checkpoints. Google's quantized-dot example explicitly quantizes both
operands for an INT8 dot. [AQT implementation](https://github.com/google/aqt).

Inference from the geometry: 4-bit expert weights plus one BF16 scale per 128
weights occupy about 4.125 bits/weight, roughly half our FP8 expert storage.
That can improve memory headroom and bandwidth. Expanding them to INT8 in HBM
would sacrifice much of that storage advantage; expanding all experts to BF16
would exceed our aggregate HBM. Packed storage also carries unpack/scale work,
so a smaller checkpoint alone is not a throughput result.

## What the publisher benchmarks establish

The vikas card reports approximately **26.3 tok/s for one prose request** on four
DGX Sparks, using CUDA/Marlin and MTP speculation. Its 89.2 tok/s figure is
aggregate throughput at concurrency eight. A separate long-prompt test reports
about 550 prefill tok/s at roughly 35K tokens. These differ from our hardware,
context, runtime and timing boundaries; they are not TPU predictions.
[Vikas model card](https://huggingface.co/vikasclawd/GLM-5.3-Int4-Int8Mix).

The Tech2wild companion report is newer/more specific than its card: **53.32
e2e tok/s** with DFlash2 on counting to 100, versus **18.70** on single-request
technical prose; MTP-4 gives **19.62** on that prose workload. Draft acceptance
is 95.6% for counting and 22.7% for prose. These rates include prefill/request
overheads. They demonstrate workload-dependent speculation, not a general
53 tok/s benefit from INT4. The report explicitly has not established BF16
quality parity. [Tech2wild detailed results](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/blob/main/bench/RESULTS-dflash2.md).

The two checkpoint formats give no reason to expect inherently different TPU
speed. Their published rates use different software/settings and cannot select
a numerical-quality winner. Both quantizations need quality evaluation; small
smoke tests and weight reconstruction error are not model-quality parity.
GLM-5.3 may improve base-model capability, but the name alone does not establish
that its 4-bit variant improves every task over our GLM-5.2-FP8 baseline.

## Implication for this repository

The current trained-weight research result is **138.85 prompt tok/s** at 2,034
tokens and **14.04 decode wall tok/s**, including host checks and in-memory
delivery. Both request-loop variants match all 29 DB610 reference tokens on
all eight hosts and have bitwise-equal final state/residual. That is a short
trail, not broad quality validation or a network service rate.
[Paired real-weight receipt](tpu-real-request-loop-20260919T192804Z.json).

Decode has a real opportunity: replacing our costly FP8 software conversion
and reducing expert reads. But attention, routing, synchronization and the
busiest expert owner remain. There is no measured basis for promising a 2x–4x
whole-model gain from these files.

Prefill needs more than quantization. In the latest synthetic eight-layer
profile, routed FP8 panels account for about 46.4 ms of a 201 ms block; collective
and gather costs are also large. Holding every other cost fixed, even removing
those panel kernels entirely would improve that particular block only about
1.30x. This is an illustrative arithmetic bound on one profile, not a universal
ceiling: scheduling and reduced collective waits can change the rest too.
[Trace receipt](tpu-trace-prefill-sparse-slice-20260919T200836Z.json).

Recommendation: retain either as a future TPU-port candidate; do not switch
now or choose by GPU headline rates. A renewed effort should first establish
packed-W4 versus resident-W8 kernel cost and accuracy on actual v4 geometry,
then a layer and full-model result. MTP/speculation is a separate possible
speed gain. No such work is authorized by this assessment or has begun.
