# Project summary

glm-tpu runs GLM-5.3 FP8 inference on a Cloud TPU v4-64 slice (32 TPU v4 chips
on 8 hosts), in native JAX with custom Pallas kernels.

## Problem

GLM-5.3 (Z.AI) is a 78-layer mixture-of-experts language model with
multi-head latent attention and DeepSeek Sparse Attention (DSA). Its official
FP8 checkpoint holds 755,632,050,320 bytes in 141 shards. Running it on TPU v4
poses three problems. The weights must be split over 32 chips on eight hosts,
because no single host's four chips can hold them. TPU v4 has no FP8
arithmetic, so the FP8 weights must be decoded on the chip. And the eight hosts
must execute identical programs in lockstep, with conversations that share the
weights but not their state. The work also needs evidence that answers and
speed come from a known model, checkpoint and commit.

## Approach

- **Mesh.** The 32 chips form one `expert=8 x feature=4` device mesh, bound to
  the slice's physical 2x4x4 topology by a recorded topology binding. The
  hidden state stays sharded over `feature`; routed experts are distributed
  over `expert`. Every compiled program is admitted only if its collectives are
  all-reduce, all-gather or reduce-scatter over one mesh axis or the full slice,
  at most 128 MiB per exchange and at most 4 KiB across the full slice.
- **Weights.** The 256 routed experts of each MoE layer (top 8 per token) stay
  in FP8 E4M3 with 128x128 block scales and are decoded tile by tile inside
  Pallas kernels. The non-routed weights are converted once, at load, to
  resident BF16 tensors.
- **Attention.** The KV cache holds one latent row per position, striped by
  position over the chips. A DSA indexer selects the top 2,048 cached positions
  per query; a fused Pallas kernel copies only the aligned tiles around the
  selected rows into on-chip memory and attends to them. Prefill (processing the prompt) runs
  layer-major blocks of 128 prompt rows and merges partial attentions by
  log-sum-exp; decode (generating one token per step) is one compiled
  `shard_map` program over all 78 layers.
- **Serving.** A resident session keeps the loaded weights and compiled
  programs and serves sequential requests, each with a fresh cache. A batch mode
  shares the weights across four independent 32K caches, prefilling the prompts
  one after another and decoding them together. A loopback chat UI and an
  OpenAI-compatible `/v1` API are clients of the resident session.
- **Integrity.** The checkpoint is re-packed into 32 per-slot files and sealed
  only after every file and tensor checksum is verified. A launch runs only a
  clean checkout of a pushed commit. All eight hosts vote on every phase and
  every token, and each run records per-host token, program and memory
  receipts.

[Architecture and code map](ARCHITECTURE.md).

## Contributions

1. Multi-host FP8 mixture-of-experts inference with DeepSeek Sparse Attention
   on 32 TPU v4 chips, which have no FP8 arithmetic.
2. Pallas TPU kernels for route-grouped FP8 expert projections and for fused
   sparse attention over the selected cache rows, in decode and in prefill.
3. Resident and four-chat batched serving with independent conversation state,
   exposed through a command line, a chat UI and an OpenAI-compatible API.
4. A graph-equivalence method that lowers every production TPU program on a CPU
   host and compares it with a recorded baseline. It showed that restructuring
   the code base for release left the device programs unchanged, and a TPU
   comparison confirmed identical tokens.

## Evaluation

Setup: 32 TPU v4 chips on 8 hosts; `zai-org/GLM-5.3` at revision
`aca966e4e02791568aa6a4ced368624b3d897f42`; JAX and jaxlib 0.10.1, libtpu
0.0.41; a 32,768-slot context; greedy decoding with thinking on at maximum
effort. Timings are those of the slowest host. Output counts include thinking.
Decode rates exclude prompt prefill, queue overhead and startup.

| Measurement | Result |
|---|---:|
| Single-chat decode, one completed answer | 13.57 tokens/s |
| Decode across 770 sequential GSM8K questions | 13.50 tokens/s |
| Prompt prefill, 85 tokens | 0.965 s |
| Cold start: verification, loading and compilation (once per session) | 1,219.89 s |
| Peak HBM per chip | 28.23 GB |
| Four simultaneous chats, per active chat | 4.89–5.12 tokens/s |

GSM8K: evaluated on the first 770 of the 1,319 test questions (rows 0–769, in
order): 740 correct (96.1%). The 30 unsuccessful cases are 27 incorrect final
answers and 3 answers that exhausted the context, which count as incorrect.
This is a partial evaluation, not a full test-set score. The separate four-chat
test answered four questions correctly at normal end of sequence.

Equivalence: on the same fleet and the same prepared requests, the released tree
and the pre-restructure baseline produced identical token streams for 17 of 17
requests (10 sequential, 4 concurrent, 3 resident). Decode speed was 0.994–1.024
of the baseline, and the programs the TPUs compiled equal the ones the CPU
harness lowers (9/9 programs, 1,062/1,062 Pallas kernels).

[Receipts, scoring and timing boundaries](STATUS.md).

## Limitations

- The GSM8K result covers an ordered prefix of a public benchmark. The stopping
  point was chosen during the run, after progress had been observed. It is neither a full test-set
  score nor evidence of uncontaminated held-out performance.
- Scoring compares final numeric answers, not every reasoning step.
- The prompts were short; they do not establish quality at full 32K input.
- Resident mode serves requests one at a time through a private file inbox. The
  chat UI and the API use that one queue: there is no parallel decoding, online
  batch admission or durable KV recovery. Model processes stay loaded until an
  explicit stop or a failure.
- The CPU equivalence proof lowers the TPU programs but does not run XLA's TPU
  compiler; a compiler or libtpu change needs a new TPU comparison.

## Reuse and provenance

Z.AI supplies the trained model, the architecture and the tokenizer. Hugging
Face reference material and JAX, Pallas, XLA and libtpu keep their
attribution. This project implements the inference system, not a new
foundation model or compiler. [Notices](../../THIRD_PARTY_NOTICES.md). The
GLM-5.2 history and its licenses remain preserved.

The software was built with AI coding agents under the author's direction;
changes are verified by the automated equivalence checks and test suite, not by
independent human review.

## Reproducibility and license

The [README](../../README.md#reproducibility) gives the offline CPU checks and
the steps from the Hugging Face weights to an answer. Model, template,
checkpoint, code and result hashes bind each execution. A reviewer package is a
`git archive` of one commit
([reviewer guide](REVIEWER_GUIDE.md#source-package)). Hardware reproduction
needs a TPU v4-64 slice and the site's assets.

The project's own work is under the [Apache License 2.0](../../LICENSE);
third-party material keeps its own license.
