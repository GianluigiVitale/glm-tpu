# DRAFT — upstream bug report (owner files; do not submit as-is without owner review)

> Suggested target: `vllm-project/vllm` (runai-streamer load path + fused fp8 load in
> `deepseek_v2.py`), cross-referenced to `vllm-project/tpu-inference` (the CPU-storage
> free machinery named in "Suspected mechanism" lives in the TPU backend).

---

**Title:** [Bug] runai_streamer load path: probabilistic, silent, often-FINITE weight
corruption of a fused-loaded (fp8-dequant) tensor on multi-host TPU serving — CPU-side,
post-dequant, pre-device-transfer

## Summary

On a multi-host TPU deployment loading a large fp8 checkpoint with
`load_format="runai_streamer"`, the load path silently delivers corrupt bytes for a small
number of tensors (typically 0–3 per host) on some fraction of engine launches. The
corruption is drawn independently per host, per launch, and comes in three measured
flavors: (1) a deterministic **all-zero fill of exactly one half of a fused-loaded
tensor** (the fp8-dequanted half of a fused dequant+concat load), (2) scattered NaN/Inf
in 128-element granules, and (3) **finite non-zero garbage**. Flavors (1) and (3) are
finite, so they pass every non-finite scan, and a CPU-vs-device byte checksum also passes
(it faithfully transfers bytes that were already wrong on CPU) — the result is a served
engine with silently degraded output quality and no crash. Victim selection shows strong
concentration on one tensor loaded through a fused load+dequant path that buffers tensors
across loader-iterator yields; plain (directly assigned) leaves were rare victims. We
localized the corruption window by instrument to **after the fp8 dequant completes and
before the CPU→device conversion**, i.e. the fused leaf's CPU storage is clobbered while
parked awaiting conversion. This bug cost us two multi-day evaluation campaigns before we
root-caused it; we report it with a ground-truth byte-sum fingerprint of the corruption
and an honest statement that we do not yet have a minimal repro.

## Environment

- vLLM: commit `a30addc75` (our last-known-good build; editable install, reported version
  `0.1.dev1+ga30addc75.tpu`)
- Backend: `tpu-inference` (JAX/torchax TPU backend; our tree carries model-enablement
  patches for the workload below — the load-path code discussed here is the upstream path)
- Hardware: TPU **v4-64 pod, 8 hosts × 4 chips**; model replicated per host (8 model
  replicas), each host runs its **own independent** streamer read of the checkpoint
- Model: GLM-5.2-FP8 (753B, ~744 GB safetensors), streamed from a **same-region** GCS
  bucket (us-central2 bucket, us-central2 pod)
- Load: `load_format="runai_streamer"`; `runai-model-streamer` 0.15.4 (+`-gcs` 0.15.4)
- Stack: torch 2.10.0+cpu, torchax 0.0.11, jax/jaxlib 0.10.1, libtpu 0.0.41,
  safetensors 0.7.0, Python via uv venv
- The fused load path involved: `_try_load_fp8_indexer_wk`
  (`vllm/model_executor/models/deepseek_v2.py`, ~line 746): the DSA indexer `wk` is
  stored fp8 with a separate `weight_scale_inv`; the loader **buffers** the weight and
  scale across loader-iterator yields until both arrive, dequantizes fp8→bf16
  (`scaled_dequantize`), and writes the result into shard 0 of the fused
  `wk_weights_proj.weight` parameter (`[160, 6144]` bf16 = dequanted wk `[128, 6144]`
  concat raw-bf16 `weights_proj` `[32, 6144]`).

All our observations are from configurations using this fused load+dequant path (we did not run a no-fused-path control);
victim concentration is overwhelmingly on the fused `wk_weights_proj` leaves (and the fp8
`wk` source region for the NaN flavor). Plain leaves were occasional, benign-severity
victims (~2–3 corrupt leaves per launch even on behaviorally healthy engines).

## The three measured flavors

All measurements are uint32 wraparound byte-sums per tensor, verified against ground
truth computed **offline from the GCS safetensors**, with the fp8→bf16 dequant replicated
bit-exactly against vLLM's own `scaled_dequantize`.

**Flavor 1 — deterministic zero-fill of the dequanted half** (the canonical specimen),
`layers.10.self_attn.indexer.wk_weights_proj.weight`:

| Quantity | uint32 byte-sum |
|---|---|
| True fused-leaf sum (offline ground truth from checkpoint) | **239851472** |
| … of which the dequantized-wk half `[128, 6144]` | 191463636 |
| … of which the raw bf16 `weights_proj` half `[32, 6144]` | 48387836 |
| **Corrupt value observed on struck hosts** | **48387836** |

Corrupt sum == the `weights_proj` half alone ⇒ **the dequantized-wk half arrives on
device as all-zero bytes**. This exact value was sighted **9+ times** across independent
hosts and launches (8/8 hosts in one launch; 1/8 in another; a manifest-bootstrap engine;
4 strikes on 4 different hosts across one validation night; 5 consecutive launches one
night later — that night the strike was near-deterministic for this one tensor). A
derived tensor computed from the zeroed half sums to 0, confirming the zeros are really
in the source param, not a measurement artifact.

**Flavor 2 — NaN/Inf granules:** e.g. `layer=0 wk NaN:640 Inf:640; layer=1 wk NaN:1331
Inf:205` — per-tensor non-finite counts arrive in **exact multiples of 128 elements**
(1280 and 1536 = 10 and 12 granules of 128), i.e. 256-byte-aligned granules in a ≥16-bit
stage. Loud once you scan for it; our init-time NaN scan refuses these.

**Flavor 3 — finite non-zero garbage:** `layers.1.self_attn.indexer.wk_weights_proj.weight`
expected sum 239780972, actual **48776186**, and its derived tensor summed **738963 ≠ 0**
— proving the wk half was corrupt-but-NOT-zero (the derivation of zeros is zero). Only a
reference-manifest check catches this flavor; NaN scans and H2D checksums are blind.

## Window localization (what we measured, in order)

1. **Not H2D:** a CPU-vs-device byte checksum of every staged tensor passes on corrupt
   launches — the transfer faithfully moves already-corrupt bytes.
2. **Not the buffering across iterator yields (clone disproven):** we suspected the
   streamer recycling the buffered wk/scale tensors held across yields, and added a
   `.clone()` at buffering time — no change. The runai iterator **already** yields
   `tensor.clone()` (`vllm/model_executor/model_loader/weight_utils.py:1076-1077`), so
   loader-side buffer recycling into the buffered references is not the mechanism.
3. **Not the read and not the dequant:** a verify instrumented at **dequant completion**
   (re-check the wk half for all-zero right after `scaled_dequantize`) **never fired on
   launches whose device state was later corrupt** — the values are still good at that
   point.
4. Therefore the zeroing strikes **the fused param's CPU storage after dequant completes
   and before the CPU→device conversion**, while the leaf is parked awaiting conversion.
   A guard that re-verifies (and repairs out-of-band) at the very last CPU touch of these
   params, immediately before conversion, catches and fixes every zero-fill strike —
   confirming the window from the closing edge as well.

## Suspected mechanism (labeled as suspected — not confirmed)

The family that fits is an **eager CPU-storage free racing a still-referenced buffer
across an async boundary**. The TPU backend's load path frees each param's CPU storage
eagerly after staging it for device transfer, via `untyped_storage().resize_(0)`
(`tpu_inference/layers/vllm/quantization/unquantized.py` — `_free_cpu_storage()`, called
from `process_weights_after_loading`; also
`tpu_inference/layers/vllm/process_weights/cleanup_sharding.py:120`). We separately found
and fixed a confirmed bug of exactly this family in the same path: the torch→JAX
conversion's bit-cast branch handed JAX a **zero-copy numpy view** of the torch storage
while the async H2D DMA was still reading it, so a subsequent `resize_(0)` freed memory
the DMA was consuming (that fix is independently reproducible via an alias-invariant
test and will be reported/PR'd separately). The residual corruption reported here
survives that fix, but the measured window — a parked CPU buffer partially zeroed
between two pipeline stages, at page-like granularity, per host, per launch,
probabilistically — matches the same free/ordering family. **We have not identified the
exact free or reuse that hits the fused leaf; we name `_free_cpu_storage`/`resize_(0)`
ordering as the audit candidate, not as a finding.**

## Reproduction (honestly stated)

We could not build a minimal repro yet. What we know about triggering it:

- **Probabilistic per host per launch.** Nightly per-host strike rates in our campaign
  ranged roughly **20%–75%**, and one night the layers.10 zero-fill was near-deterministic
  (5/5 consecutive launches, plus 2 hosts of a concurrent launch). Rates vary night to
  night with no config change.
- Needs (in our observations): multi-host serving, `load_format=runai_streamer` from GCS,
  and the fused fp8-dequant load path that holds tensors across loader-iterator yields
  and dequantizes on CPU before assignment. Streamer concurrency is exonerated (a
  16-launch A/B at concurrency 32 vs 8 showed corruption in both arms).
- Single-host and CUDA paths: untested by us (see "Not claiming").

**In lieu of a repro, our detection tooling** (we are happy to share/port it so
maintainers can instrument any environment):

- **Per-leaf uint32 wraparound byte-sum fingerprints** of the final loaded model state
  (loaded weights AND derived tensors), verified at init against a **golden reference
  manifest** with fail-closed refusal. The manifest is built by majority-of-3 boot
  engines plus offline ground-truth sums computed directly from the checkpoint
  safetensors (the offline fp8-dequant replicated bit-exactly against
  `scaled_dequantize`) — necessary because frequent-victim tensors can poison a naive
  majority vote (our first bootstrap engine was itself corrupt).
- **Last-CPU-touch detection**: checking the fused leaf's halves for the impossible
  all-zero signature at the final CPU read before device conversion detects every
  zero-fill strike (and, paired with an out-of-band checkpoint mirror, repairs it
  bitwise — validated on metal: 5/5 strikes repaired, all engines then byte-verified).
- Cheap dequant-completion and pre-conversion NaN scans for flavor 2.

The fingerprint approach costs seconds per host and needs no reference to be useful
(cross-host/cross-launch diffs alone expose strikes).

## Impact

- **Silently degraded serving quality.** Two of the three flavors are finite: no crash,
  no NaN anywhere, all init scans and transfer checksums pass, and the engine serves. A
  corrupt projection weight degrades that layer's behavior per-instance-permanently; in
  our workload this presented as an "engine lottery" (most launches healthy, some
  launches quietly wrong on long-context retrieval) that took ~6 days and ~115 engine
  launches to root-cause.
- It killed **two multi-day n=77 evaluation campaigns** before we understood it — both
  gate failures were retroactively attributed to load-time corruption, not the model
  code under test.
- Because victim choice and severity are drawn per launch, most strikes are behaviorally
  benign — which means installations running this path may be affected without knowing.

## What we are NOT claiming

- **No minimal repro yet.** Everything above is from a production-scale deployment with
  heavy instrumentation, not from an isolated test.
- **The exact mechanism is unconfirmed.** The window (post-dequant, pre-conversion CPU
  storage) is measured; the specific free/ordering that clobbers it is not. We name the
  eager `resize_(0)` CPU-storage-free machinery as the audit candidate only.
- **Scope untested elsewhere:** single-host, CUDA/`is_cuda_alike` streamer path
  (`device="cuda:*"` streaming), other checkpoints, and current vLLM main are all
  untested by us; our data is from the commit pinned above on TPU multi-host.
- We are not claiming the streamer library itself is at fault — the clone at yield and
  the good-values-at-dequant measurement point away from the read stage; the consumer
  side of the load path is where the evidence points.

## References (ours, available on request)

- Full post-mortem with the elimination ledger (18 refuted hypotheses, each with the
  instrument that killed it), the arm/draw ledger (~115 launches), and the ground-truth
  adjudication protocol.
- The separate (confirmed, fixed, independently upstreamable) t2j zero-copy alias race
  and its must-fail-first alias-invariant test.
- `ground_truth_sum.py` — offline per-leaf sums from checkpoint safetensors, including
  the fused/stacked name-mapping and bit-exact fp8-dequant replication.

---

This report was prepared with AI assistance (Claude); all measurements and claims were
verified by the project maintainer.
