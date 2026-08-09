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
| Prompt index-cache oracle | legacy `dcp_cache_dump.py`, DB505--510, and existing association helpers | Greenfield validates 32 physical replicas and reconstructs logical BF16 layer-0 keys. DB506 rejects current M1; DB507 reaches 45 rotary-half mismatches; DB508/509 reject external-chunk and BF16-weight explanations; DB510 proves the gather-coupled input-RMS lowering restores DB507 exactly. The next discriminator adapts the accepted flat BF16 paged-cache write consumer before changing RoPE math. |
| DSA internal observer | oracle-only `83ff4a357` two-commit child of accepted `b3c25df47` | Default-off any-full-producer/one-position callback, independent all-21 greenfield observer, and hash-pinned five-state comparator; zero-copy torchax/JAX boundary; no legacy execution import. |
| DSA query association | protected capture recovery plus DB499 at `b41c3ab` | Exact local FP32 `wq_b` owner boundary for M=1; Pallas/streamed alternatives are retained as negative evidence. |
| Production q-a association | accepted layer-0 capture, DB491 distributed q-a artifact, and the all-21 greenfield observer at `380659a` | Reuse the accepted shard-major fused `N=82` packing and norm association as a one-row, stage-local virtualized discriminator; never import the TP32 execution path or its full-pod collectives. |
| Distributed q-a norm | legacy FP8 linear/sharding source, vLLM RMSNorm source and accepted E0 XPlane | Bounded independent TP32 diagnostic with one FP32 variance all-reduce and one BF16 rank-3 all-gather; never a production architecture. |
| Fused wk precision | accepted OOB repair/adapter source and layer-0 state hash | Rejected as an order discriminator: BF16-origin and direct-FP32-origin `wk` produce identical stored prompt keys. Preserve BF16-origin state identity in production. |
| Layer-0 input/state | vLLM embedding/model source, TPU OOT embedding and accepted state hashes | Raw BF16 embedding row, model-epsilon input RMSNorm and adapted DSA leaves are pinned; no hidden embedding multiplier or TPU transform exists. |
| Exact 8K local scorer | legacy XLA scorer source, DB485 run config and accepted XPlane | Bounded independent `R=32`, `P=512`, three-owned-page DCP scorer association; diagnostic only, never a production dead-row path. |
| Checkpoint layout | existing greenfield plan/pack/load chain | Gate B is already complete; the 834 GB runtime derivative is the only full PP8 decoder input. |
| Prefill HLO control-flow proof | existing greenfield HLO parser and preserved old/fused 8K executables | Classify one outer prompt scan, exact fused-qkv compiler internals and every unknown physical loop; never exempt a total count. |
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
- The BF16-origin `wk` discriminator (`948f981`) is closed: its upstream adapted bytes differ, but
  its prompt keys are elementwise identical after the complete stored-key path.
- The fused 8K attempt at `3058dc8` is a preserved linter refusal, not a decoder failure or result.
  Its HLO supplies the reusable positive discriminator: 78 exact fused-qkv internal loops under
  one outer prompt scan. Commit `1f11013` adapts the existing parser to pin those identities while
  rejecting any unknown loop.
- The loop-corrected 8K attempt at `e4079ac` is a preserved DSA refusal, not performance evidence.
  It proves full prefill and exact first-token execution but rejects event-0 order and event-1 set
  drift before timing. Do not repeat another full decoder until a bounded prompt-cache comparison
  identifies whether the stored layer-0 key state differs.
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

The next adaptation targeted a concrete scorer-shape mismatch. DB485 provenance pins
`GLM_DSA_SCORER=xla`, `max_model_len=8704`, DCP8, 512 local keys per 4,096-token global page,
`GLM_DSA_BT_WIDTH=owned`, and 24 cache blocks. The accepted local body walks exactly three pages at
`R=32`; the older diagnostic nested eight shards in one 84-page program. Greenfield independently
reproduces the local cache/block-table/length operands and exact XLA einsum/reduction association,
then stitches eight stripes only outside the compiled scorer. This remains bounded diagnostic code:
production `decode_batch1` is still one row and may never inherit the legacy dead-row bucket.

DB490 proves that exact local scorer is elementwise identical to the older reconstruction and
rejects scorer geometry. The same registry then exposes the next reusable source truth: accepted
vLLM constructs q_a/kv_a RMSNorm from the pinned model config, whose SHA is
`22e49334...65ff` and whose epsilon is `1e-5`. The earlier `1e-6` greenfield conclusion is
superseded. The existing distributed association runner is therefore reused once with explicit
config provenance; no new scorer, loader, model path, or protection harness is being invented.

DB491 proves that correction reduces exact-local-XLA mean score error about 17x, from `0.0228811`
to `0.00134283`, while retaining the exact selected set. It still misses 1,408 order positions, so
epsilon alone is rejected as a sufficient fix. A source-faithful logical-width GSPMD `mean` is
bitwise identical to the existing manual distributed norm on forced-32 exact BF16 inputs; its
full-geometry HLO retains the same one all-reduce/all-gather association. It is therefore rejected
without a redundant TPU launch.

The BF16-origin `wk` launch
`greenfield_layer0_dsa_association_20260807T235427432046987Z` at `948f981` reused DB491's immutable
q artifact and omitted the closed 32-chip phase. It stopped at its intended novelty guard because
the candidate and baseline prompt keys were elementwise identical after projection, key
LayerNorm, RoPE and BF16 cache storage. There is no scorer matrix, DB row, `SUCCESS`, decoder or
performance claim. The authenticated failure-exit census contains eight unique `CENSUS_OK` hosts
and hashes to `892250e0...099d`; partial evidence is archived under the approved diagnostic prefix.
The candidate is rejected and must not be rerun.

The associated input/state audit is also now durable. Accepted layer-0 state hashes pin raw BF16
embedding `[154880,6144]` sum `1668496656`, input-norm `[6144]` sum `1006936`, adapted
`weights_proj` FP32 sum `48158645`, adapted `wk` FP32 sum `193298069`, adapted `wq_b` FP32 sum
`3765880530`, fused `wk_weights_proj` BF16 sum `241456714`, and q-a norm BF16 sum `305844`.
Accepted vLLM returns `embed_tokens(input_ids)` without scaling; its vocabulary sharding masks all
nonowners and all-reduces the one nonzero row, while the TPU OOT class delegates unchanged.
Layer 0 clones that raw BF16 row as residual and applies the model-epsilon input RMSNorm. The
greenfield input builder already selects the exact raw checkpoint rows, so embedding/input
construction is rejected as a remaining unexplained DSA-order hypothesis.

The next diagnostic also reuses rather than replaces the accepted machinery. Oracle-only legacy
pin `83ff4a357` adds one default-off callback at the already-proven DSA scorer boundary and captures
only the live layer-0 position-8155 normalized hidden, q-a state, query, head weights and current
pre-cache key. Its follow-up fixes the production torchax-tracer boundary by reusing the scorer's
existing zero-copy `as_jax` bridge; 19 focused forced-device/torchax tests pass. Greenfield pin
`6af9092` extends the existing protected 8K DSA launcher: it syncs a
detached observer worktree while preserving the accepted checkout, requires exact raw output and
bitwise-identical DB485 DSA event tensors, validates eight replicated process artifacts, compares
the five states on one local TPU host against the immutable input/DB491 q-a artifacts, then retains
the existing DB snapshot, approved archive and authenticated zero-work exit. The 423-test CPU-only
suite passes with one expected skip and the two pre-existing SWIG warnings. This is
implementation/readiness evidence only; no observer TPU capture or numerical conclusion exists yet.

The corrected observer subsequently produced source DB493, and authenticated recovery artifact
`greenfield_legacy_layer0_dsa_internals_recovery_20260808T030853085141329Z` sealed one
topology-owned live row. Its independent comparison is bitwise exact for normalized hidden and q-a
state and identifies `query` as the first divergent field. DB495 proves the production Pallas MXU
query is exactly the observed M=32 drift (`4,096` mismatches, max `0.009170532`). DB496--DB498
reduce the error with streamed/unrolled raw-FP8 association but retain 2,840/1,208 mismatches.

DB499 / `greenfield_layer0_dsa_query_association_20260808T034636385240375Z` at `b41c3ab` closes the
source search: both preadapted and raw-FP8 materialized PP8-owner candidates are bitwise exact. The
accepted production boundary is only local `f32[1024,2048]` (8 MiB) before a true-row dot; HLO
`ea5e5c56...6c89` contains no global `f32[4096,2048]` table. Production adapts this boundary only
for DSA `wq_b`; q-a, wk, attention, dense, and MoE remain on their existing selected backends.

The corrected 8K refusal at `f129e63` supplied the next concrete reuse boundary. Its exact first
token and exact event-0 set did not prove the full layer-0 producer: DB499 was conditional on the
accepted q-a input. The completed all-21 observer at `380659a` instead proves production layer-0
normalized hidden exact while q-a already differs in 494/2,048 BF16 values (max `0.015625`), with
all 4,096 query values then different. Layer 1 is downstream: normalized hidden differs in
3,974/6,144 values and all later observed fields differ.

The accepted source mechanism is already preserved by DB491's independent diagnostic: raw FP8
q-a and kv-a weights are packed into 32 physical output shards, each `N=82` (`64 q-a + 18 kv-a`),
then q-a normalization reduces the 32 local 64-wide square sums. The new bounded discriminator
reuses that exact pack and arithmetic while virtualizing the 32 shards on one stage-local device.
It accepts only `[1,6144]`, rejects dead `[32,...]` token shapes and all collectives/callbacks, and
tests projection (`lax.map`/`vmap`/unrolled) against four explicit norm associations. The existing
DB499 one-host protected wrapper is parameterized for this q-a target rather than duplicated. CPU
tests prove shape/failure/HLO plumbing only; one serialized bounded TPU run must choose an exact
candidate before any production integration or full 8K retry.

DB501 rejects that first virtual family: all 12 M1 dot/mapping/norm candidates collapse to one
376-mismatch BF16 output despite different HLO hashes. The reusable positive clue is the already
preserved DB491 HLO: its exact physical local body is `convolution ... bf_io->bf`, whereas DB501's
M1 `dot_general` becomes multiply/reduce. The next adaptation must call the JAX zero-spatial
convolution primitive directly with one external row and enforce its optimized HLO. Padding the
public decode input back to 32 rows is rejected by the greenfield contract.

The v2 protected matrix implements only that new primitive and retains the four norm associations.
Its extra HLO check requires `f32[1,82] convolution` with `bf_io->bf`; this prevents a source-level
convolution label from hiding another multiply/reduce lowering. DB501's old 12 variants remain
archived negative evidence and are deliberately excluded from the new run.

DB502 accepts the discriminator: all four convolution/norm variants match the accepted q-a state
bitwise with one live row and local, collective-free HLO. The production adaptation is therefore
the direct convolution plus its fused kv-a companion. The diagnostic pack function itself is not
the runtime design: the plan-aware offline packer must emit N82 weights and expanded scales in the
final device layout so decode never repacks q-a/kv-a state.

Production pin `0082bac` now consumes this evidence in
`kernels/reference/qkv_a.py`, `kernels/layer.py`, the runtime decoder, and the
feature-runtime pack/load chain. The adaptation preserves only the proven
arithmetic and packed-state definition; it does not import the diagnostic or
legacy execution path. DB503 at `f715039` now proves that actual integrated
production helper is bitwise exact for q-a and kv-a with one physical one-row N82
convolution and no collective/dead row. The protected fused pack and DB504 at
`c16b37f` now extend the existing plan-aware pack/loader rather than introducing a
new checkpoint path. Every one of the 10,880 final tensors is directly loaded and
device-round-tripped, with zero reshard/concat/dequantization, while the full body
retains exactly 78 one-row convolutions. Gate B is re-closed; reuse this same
artifact and loader for the protected 8K retry.

The loop-corrected retry at `e4079ac` passed load, all decoder/observer/prefill HLO contracts,
teacher-forced prefill, and exact first token `101252`, then failed the first device-resident DSA
observer before timing. Event 0 keeps the exact 2,048-member set but differs first at score-order
offset 8; event 1 has six set swaps. This rules out another blind decoder retry.

The next diagnostic reuses the accepted legacy `dcp_cache_dump.py` rather than adding another
model observer. Greenfield pin `e5a6991` arms it only for slot 0 on the sealed 8K oracle, requires
four post-forward dumps per host and exact eight-process/four-model-replica coverage, maps the live
block table back to 8,155 logical BF16 keys, and seals source/checkpoint/token hashes. After the
legacy runtime stops, an independent one-host program runs the actual production raw-FP8 `wk`,
input RMSNorm, key LayerNorm/RoPE, and BF16 write inside a one-row `lax.scan`. Its HLO must contain
one raw-FP8 key kernel and one outer scan, with no collective, callback, decoded weight overlay,
full-prompt hidden materialization, or dead `[32,6144]` row. This is readiness only until one
serialized protected capture completes; it imports no legacy/vLLM execution into greenfield.

The first protected capture completed the accepted DB505/item1788 oracle and wrote all 32 cache
snapshots, but the old parser refused the real `model=32,dcp=1` fully replicated layout before the
greenfield comparison. The corrected contract requires four local and 32 physical bitwise-equal
replicas, physical shape `[24,16,32,128]`, and 512-token logical pages. The resume wrapper binds
the source row, capture/oracle/census hashes, all local cache-file hashes, and byte-identical
approved-bucket copies of the eight final snapshots. It then runs only the independent one-host
production comparison; the 753B accepted model load is deliberately not repeated.

DB510 at `6286a06` proves the compiled embedding gather feeding input RMSNorm is causal: it
reproduces DB507's 45-mismatch output byte-for-byte while retaining a true M2048 chunk, one
BF16-RHS convolution and no loop/communication/dead row. The remaining 45 values affect one member
of an interleaved rotary pair, but that does not distinguish RoPE math from sub-BF16 pre-RoPE
rounding. The already-audited accepted source provides the narrower next interface:
`compute_indexer_keys` feeds FP32 rows to the flat BF16 paged-cache write. Greenfield will adapt
only that consumer in the existing protected harness using the sealed cache geometry and block
table; the legacy function itself remains oracle-only.
