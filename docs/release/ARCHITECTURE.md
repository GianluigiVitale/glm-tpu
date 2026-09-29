# Architecture and code map

`glm_tpu` runs GLM-5.3 FP8 inference in native JAX with Pallas kernels on one
TPU v4 slice of eight hosts and 32 chips. Its layout follows vLLM and
tpu-inference: entry points, an executor that owns the fleet, a per-host worker,
a model runner that builds and admits the device programs, an engine that drives
them, and the model, layers and kernels underneath. This page describes the tree
as it is; the measured results are in [STATUS](STATUS.md).

## Module map

| Package | Modules | Role |
|---|---|---|
| `entrypoints/cli` | `main`, `ask`, `prepare`, `collect_env`, `checkpoint`, `topology`, `types` | the `glm-tpu` command line |
| `entrypoints/serve`, `entrypoints/openai`, `entrypoints/ui` | `server`, `http_handler`, `job_queue`, `security`; `serving_chat`, `serving_models`, `chat_utils`, `protocol`, `tool_parser`; `conversations` and `static/` | the loopback server, the OpenAI-compatible `/v1` API and the browser chat UI, attached to a resident session |
| `executor` | `multihost_executor`, `launch_policy`, `staging`, `fleet`, `remote/`; `jobs`, `topology_job`, `pack_job` | the rank-0 controller: launch policy, staging by `git archive`, the exact SSH command strings, and the standard-library helper programs it sends to the hosts; the model-free fleet jobs on the same machinery (the topology capture, the checkpoint pack) |
| `worker` | `tpu_worker` | the per-host worker process |
| `runner` | `tpu_runner`, `programs`, `compilation_manager`, `kv_cache_manager`, `admission`, `hlo_utils` | the model runner: checkpoint loading, the program set, compilation with preserved graph originals, the fresh cache, memory and collective admission |
| `engine` | `llm_engine`, `request`, `request_session`, `outputs`, `resident_protocol`, `resident_client` | generation over a loaded runner, request schemas and preparation, the per-request host loop, the token event stream, the run-directory protocol and its inbox client |
| `models/glm_moe_dsa` | `model`, `prefill`, `decoder_layer`, `state`, `weights`, `hf_config/` | GLM-5.3: the decode and prefill programs, the layer bodies, the device state, the weight trees, the pinned Hugging Face configuration |
| `layers`, `layers/attention`, `layers/moe` | `norm`, `rope`, `linear`, `mlp`, `fp8`, `embed`, `sampler`, `contracts`; `mla`, `dsa_indexer`, `kv_cache`; `router`, `routed_experts` | per-shard layer bodies, run inside `shard_map` on the device mesh |
| `kernels` | `sparse_mla/{kernel,partial_kernel}`, `fp8_grouped_matmul/{kernel,panel_kernel,panels}`, `names` | the Pallas TPU kernels and their `pallas_call` names |
| `model_loader`, `model_loader/sharded_state` | `source_inventory`, `source_marker`, `placement`, `pack_worker`; `format`, `writer`, `manifest`, `seal`, `verify`, `loader` | the checkpoint pipeline: source inventory and completion marker, per-device placement, packing, the packed format, the seal, verification and loading |
| `distributed` | `mesh`, `topology`, `topology_capture`, `parallel_state` | the logical and physical mesh, the topology capture and binding, JAX distributed initialization and the all-host vote |
| `config` | `model`, `cache`, `site`, `parallel` | the pinned model geometry and identity, the cache configuration and numerical contracts, the site file, the mesh axis names |
| top level | `envs`, `exceptions`, `utils/{io_utils,json_utils}` | environment variables, fail-closed errors, create-once writes readable only by you (mode 600), canonical JSON |

The configuration, executor and utility packages import no JAX, so the controller
and the command line run without initializing a device. `tests/` mirrors this
layout.

## The request path

1. **Preparation** (`glm-tpu ask` or `prepare-request`, CPU only). The chat is
   rendered with the pinned GLM-5.3 template and tokenized with the pinned
   tokenizer (`engine/request.py`); the prepared request is canonical JSON with its
   own SHA-256, written outside the checkout and readable only by you (mode 600). The whole input is
   tokenized; nothing is truncated. A request that does not fit its capacity is
   refused.
2. **Launch** (`executor/multihost_executor.py`, rank 0). The launch policy picks
   the commit; the controller takes the site's locks, checks the eight hosts idle,
   stages `git archive` of the commit with the request and the resolved site
   configuration to every host, runs a CPU preflight there and starts the workers
   ([OPERATIONS](OPERATIONS.md#what-a-run-does)).
3. **Load and compile** (`worker/tpu_worker.py`, `runner/tpu_runner.py`). Each worker
   verifies its staged source, the site configuration, the topology binding and
   its four checkpoint files, initializes JAX distributed with the fleet, loads its
   slot files onto its four chips, builds the program set (`runner/programs.py`),
   compiles each program and admits it: the compiled graph's collectives are
   parsed and checked (`runner/admission.py`, `runner/hlo_utils.py`) and the live
   memory is projected against every chip's limit. Every phase ends with an
   all-host vote.
4. **Generate** (`engine/llm_engine.py`, `engine/request_session.py`). The engine
   runs the prefill program over the prompt, then one decode step per token.
   Every host follows the same schedule and votes on each phase and on each
   token's validity and delivery. Rank 0 writes and flushes each token event to
   its local JSONL file, then decodes the final text. A deadline, peer or
   delivery failure poisons the loaded engine: it refuses every later request.
5. **Records**. Each worker writes its runtime record; the controller collects
   them, checks that the eight hosts agree on tokens, programs and memory, and
   verifies cleanup.

## Parallelism

The 32 chips form one `expert=8 x feature=4` mesh (`distributed/mesh.py`,
`config/parallel.py`) bound to the slice's physical 2x4x4 topology by an
authenticated topology binding (`distributed/topology.py`): device order is read
from every process's live devices and the captured fleet mapping, never inferred
from device ids. The decode row's residual is sharded over `feature` and
replicated over `expert`. Gate/up projections shard their contracting hidden
dimension over `feature` and their output or expert identity over `expert`; down
projections reverse those roles. No layout materializes a `[32, hidden]`
activation.

Every compiled program must pass the collective check before it runs: only
all-reduce, all-gather and reduce-scatter, each over the four-chip `feature`
groups, the eight-chip `expert` groups or the full slice, with global device ids,
at most 128 MiB per exchange, and at most 4 KiB when the group is the whole slice
(the fleet-consensus values). The request loop's fleet votes are host-side
all-gathers between device calls (`distributed/parallel_state.py`), not per-layer
dispatch.

## The model

GLM-5.3 (`GlmMoeDsaForCausalLM`, geometry pinned in `config/model.py`): 78 decoder
layers, hidden size 6,144, vocabulary 154,880.

- **Attention**: multi-head latent attention with 64 heads, a 2,048-rank query
  projection, a 512-rank KV latent and a 64-wide rotary part, absorbed so the
  cache holds one latent row per position (`layers/attention/mla.py`).
- **Sparse attention (DSA)**: an indexer with 32 heads of 128 scores the cached
  keys and selects the top 2,048 positions per query (`layers/attention/dsa_indexer.py`).
  The three dense layers and every fourth layer from layer 6 (21 in all) run their
  own indexer; the other 57 reuse the selection of the preceding one (IndexShare,
  groups of four). The KV cache is striped by position over the chips that own it
  (`layers/attention/kv_cache.py`); a chip gathers only the selected rows it owns,
  and the owners combine their parts over the `expert` axis, prefill through
  log-sum-exp merges of partial attentions (`layers/attention/mla.py`,
  `kernels/sparse_mla/`).
- **Feed-forward**: the first three layers are dense (intermediate size 12,288);
  the other 75 are mixtures of 256 routed experts (top 8, intermediate size
  2,048) and one shared expert (`layers/moe/`, `kernels/fp8_grouped_matmul/`).
- **Weights**: FP8 E4M3 with 128x128 block scales, placed per device slot by
  `model_loader/placement.py`. The routed experts run from their FP8 bytes in
  Pallas kernels; the non-routed weights are converted once at load into
  resident BF16 tensors.
- **Sampling**: greedy (`layers/sampler.py`); a prepared request sets thinking
  on at maximum effort.

**Prefill** (`models/glm_moe_dsa/prefill.py`) runs layer-major blocks of prompt
rows (128-row blocks, with a 114-row tail block): one call embeds a block, visits every
layer once and commits the caches only when every owner is healthy.
**Decode** (`models/glm_moe_dsa/model.py`) is one compiled `shard_map` program for
one token over all layers; concurrent decoding adds a leading conversation
dimension over the same weights with separate caches.

## Resident and concurrent modes

A resident session (`ask --keep-loaded`) keeps the loaded runner and its compiled
programs and serves prepared requests from the run directory's inbox one at a
time, each with a fresh cache (`engine/resident_protocol.py`). The chat UI and the
`/v1` API are clients of that inbox (`engine/resident_client.py`); they never load
weights or start workers.

Concurrent mode (`ask --concurrent`) serves a fixed group of up to four
conversations of 32,768 slots each: prompts are prefilled one after another into a
bank of per-conversation caches, then one decode program advances all of them;
each conversation keeps its own delivery and stopping (`BatchedSession`). There
is no online admission into a running group. [CONCURRENT](CONCURRENT.md).

Capacities are fixed at load: 8,192, 32,768, 166,912 (the `128k` profile: 128K
prompt / 163K total) or 262,144 combined slots; generation is capped at 163,840
tokens.

## Numerical conventions

The arithmetic boundaries the code keeps (`config/cache.py` holds the numerical
contracts the layers share):

- Activations are BF16. FP8 weights are decoded on the device from the exact E4M3FN
  bit table and scaled per 128x128 block (`layers/fp8.py`).
- RMSNorm computes its statistics in FP32, rounds the normalized value to the
  activation dtype and then multiplies the weight in that dtype (`layers/norm.py`).
- The fused residual add sums the two BF16 inputs in FP32; the normalization
  consumes that unrounded sum, while the carried residual is the same sum rounded
  to BF16.
- Rotary embeddings use dimension 64 and theta 8,000,000 with FP32 angles and
  interleaved pairs; the main attention rotates in FP32 and rounds once to BF16
  (`layers/rope.py`).
- The router applies an FP32 sigmoid; the correction bias takes part only in
  choosing the top 8 experts; the weights come from the unbiased scores,
  normalized in FP32, and the routed scale is applied to the reduced routed
  output (`layers/moe/router.py`).
- Linear weights keep the checkpoint orientation `[out_features, in_features]`;
  the BF16 projections accumulate in FP32 (`layers/linear.py`).

The unsharded reference model in `tests/reference/` restates these rules
independently of production; [VALIDATION](../../tests/reference/VALIDATION.md)
records how it was validated.

## Evidence that a change kept the programs

`tools/equivalence` lowers every production program for the TPU platform on the
CPU host and compares normalized digests, CPU execution goldens, checkpoint
identities, import closures, executed functions and wire formats with the
recorded baseline ([README](../../tools/equivalence/README.md),
[TESTING](TESTING.md)). A change under `glm_tpu/` that is not a declared numerical
change keeps all of them identical.
