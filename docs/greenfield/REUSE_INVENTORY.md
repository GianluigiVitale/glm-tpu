# Greenfield reuse inventory

This is the pre-implementation audit requested on 2026-08-07: use the substantial work already on
disk before creating another kernel, harness, loader, protection, or architecture prototype. The
machine-readable authority is `configs/greenfield-reuse-inventory.json`; this document explains the
decisions.

The boundary is deliberate: reuse does not mean importing the legacy model. Non-model utilities in
`glm-tpu` are called directly; portable interfaces and semantics are adapted independently; legacy
and vLLM model execution stays oracle-only. A unit test scans every Python module below
`glm_tpu/greenfield` and rejects imports rooted at `tpu_inference` or `vllm`.

## What is already reused

| Area | Existing source | Greenfield use |
|---|---|---|
| Provenance | `bench/provenance.py`, `bench/results.db` | Every protected run links append-only DB rows and a DB snapshot. |
| Wall/trace truth | `parse_xplane.py`, `extract_steady_decode.py` | Fresh fleet XPlanes, exact step selection, source/shape attribution, separate profiler-free wall. |
| Fleet safety | E0/resume ownership guards | Lease, exact pin, authenticated census, archive-before-success and clean failure exits in greenfield wrappers. |
| Quality/long context | `moe-tpu` DSV4 harness and GLM benchmark suite | GLM-specific generation passkey ladder, prompt-length correction, raw output and per-trial provenance. |
| Model truth | local HF config/modeling and vLLM GLM class | Geometry, names, dtypes and numerical semantics only; no class import into execution. |
| Checkpoint protection | legacy checksum/NaN/state-hash/write-probe failure classes | Independent final-owner checksums, finite scans, manifests, device round trips and cache-health refusal. |
| DSA validation | legacy `dsa_topk_dump.py`/`dsa_topk_diff.py` | Portable sealed event artifacts and exact set/tie/order/IndexShare comparisons. |
| Distributed q-a norm | legacy FP8 linear/sharding source, vLLM RMSNorm source and accepted E0 XPlane | Bounded independent TP32 diagnostic with one FP32 variance all-reduce and one BF16 rank-3 all-gather; never a production architecture. |
| Exact 8K local scorer | legacy XLA scorer source, DB485 run config and accepted XPlane | Bounded independent `R=32`, `P=512`, three-owned-page DCP scorer association; diagnostic only, never a production dead-row path. |
| Checkpoint layout | existing greenfield plan/pack/load chain | Gate B is already complete; the 834 GB runtime derivative is the only full PP8 decoder input. |
| Kernels | legacy DSA/GMM/quantized matmul plus DSV4 paged-attention research | Arithmetic/tiling reference; greenfield implementations remain independent and protected. |
| Parity | `moe-tpu/parity` and existing GLM parity harnesses | Random-checkpoint transplant, real-layer differentials and cache/chunk tests, adapted to GLM. |

This explains why the branch already contains mature topology discovery, HLO parsing, protected
launchers, direct loaders, exact reference layers, FP8 Pallas kernels, DSA/top-k/sparse-attention
kernels, oracles and a complete short decoder. Those are assets to extend, not replace.

## Pinned candidates not yet integrated

### Gate D/F: exact DSA and base latency

- `bb3e4c7d6` (`scorer-walk-unroll`) is an exact map-to-scan8 fallback. It is useful only if the
  already protected one-row Pallas scorer loses end-to-end; do not pre-emptively port it.
- `13bfbca3a` and `979f818e0` prove legacy row narrowing semantics, but greenfield already has a
  static live row count of one. Their batch reconstruction is intentionally not reused.
- `83915fe74` pins routed/shared combine association. Greenfield has already adapted that association
  into a single topology-local tuple combine.
- The legacy `indexer_kernel.py` and `sparse_mla_kernel.py` remain line-level arithmetic and edge-case
  references. Importing them would pull in the old execution stack.

### Gate G: WS32_2D

Do not design WS32 from a blank page. Start with these existing pins:

- `fce8d6c41` (`patemotter/2dtp-merge`) for reciprocal expert/feature `edf`/`efd` layouts;
- `dab2db7b3` / `6baf66e2a` for FP32 partial preservation across 2D reductions;
- `57adb4b99` and `2baf3f0a0` for the existing quantized 2D matmul draft/reference/benchmark.

They are research inputs, not drop-ins: they target the DeepSeek native-JAX layout and do not prove
GLM DSA, FP8 block-scale ownership, persistent sharded residuals or the protected workload.

### Gate H: speculation

Do not rebuild GLM MTP plumbing. Pins `6beacb5d2` and `89e1d5b5a` already prove dense-MTP shared
weight mapping, draft-only checkpoint filtering, exact failure behavior and IndexShare seed/emit
carriage. `parity/glm_mtp_parity.py` is already in this repository. Integration stays deferred until
the base decoder passes Gates D-G, as required by the specification.

## Preserved negative evidence

- Full-pod MoE all-gather (`aa608543b`) did not solve the synchronization floor.
- MoE compute-row narrowing (`b3c25df47`) produced a real but insufficient ~3.395 wall tok/s.
- Legacy DSA live-row variants are structurally superseded by true batch-one code.
- The returned full-model residual observer (`78a5fce88`) perturbed arithmetic and is closed.
- Old collective/XPlane/provenance worktrees are superseded by the broader greenfield versions; their
  tests and failure modes remain evidence, not a second implementation track.

## Worktree and repository map

The audit covered:

- `/home/gianl/glm-tpu` and all eleven registered worktrees;
- `/home/gianl/tpu-inference` plus the GLM optimization, protection, residual and scorer worktrees;
- `/home/gianl/moe-tpu`, especially its DSV4 parity, paged-attention and long-context work;
- `/home/gianl/vllm-build` as a source-level GLM/HF reference only;
- `/home/gianl/glm-run`, including accepted and rejected greenfield/legacy artifacts through DB 488;
- the local HF config/reference snapshots and the user-provided resume attachment.

One historical worktree, `/home/gianl/tpu-inference-moe-live-allgather`, contains an owner change in
`mla_attention.py`; it is explicitly read-only and was not touched.

## Rule before new implementation

Before adding a new greenfield component:

1. Search this registry and the pinned source paths.
2. Reuse a same-repo non-model utility directly when its contract fits.
3. Otherwise adapt the smallest isolated semantic/interface surface with independent tests.
4. Keep legacy/vLLM model execution oracle-only.
5. Record the source pin, disposition and greenfield consumer here and in the JSON registry.
6. Preserve negative evidence instead of rerunning a rejected mechanism.

DB489 has now rejected the distributed q-a norm as sufficient: it improves the sealed order error
but still misses 1,501 XLA and 1,249 Pallas slots. The run's internal state deltas do not constitute
captured query/key proof because the portable artifact seals selected positions/scores only.

The next adaptation therefore targets a concrete scorer-shape mismatch. DB485 provenance pins
`GLM_DSA_SCORER=xla`, `max_model_len=8704`, DCP8, 512 local keys per 4,096-token global page,
`GLM_DSA_BT_WIDTH=owned`, and 24 cache blocks. The accepted local body walks exactly three pages at
`R=32`; the older diagnostic nested eight shards in one 84-page program. Greenfield independently
reproduces the local cache/block-table/length operands and exact XLA einsum/reduction association,
then stitches eight stripes only outside the compiled scorer. This remains bounded diagnostic code:
production `decode_batch1` is still one row and may never inherit the legacy dead-row bucket.
