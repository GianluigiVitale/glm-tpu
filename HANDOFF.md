# HANDOFF — topology-first greenfield rewrite

**Updated:** 2026-08-06 03:23 UTC

## Authority and isolation

- Branch/worktree: `rewrite/topology-first-decode` at
  `/home/gianl/glm-tpu-topology-rewrite`.
- Starting harness pin: `a4a17ac4e90b15f1994bd8b26917ef62daa52660`.
- Legacy oracle pin: `b3c25df47ac98783912dc658878181ec0a8ae16d`.
- Collective result pins: `fcd8426735...` and `b12af9633...`.
- Transport latency pin: `577aa4bf706976f3d25f552526d0b6eaa5d4920e`; trace pin:
  `8aee3351a61f89141762dda237f582b8deed3a1c`.
- Read `AGENTS.md`, `goal.md`, and `docs/glm-tpu-revolution.md` before this status file.

Those tracked branch-local files are authoritative. Main checkout, old worktree, campaign, and
legacy `AGENTS.md`/`CLAUDE.md`/`HANDOFF.md` files have no authority here. The incremental TP32 plan
and pipeline-parallelism ban are superseded. Never edit/delete the owner's untracked main files.

## Implemented and verified

- Frozen, hashed geometry/topology/plan types; runtime physical discovery; deterministic PP8/PP16
  groups/rings; exact optimized-HLO parser with executable-partition-to-physical-device mapping.
- Complete protected 75-operation collective floor over physical 2/4/8/32-chip groups.
- Device-resident PP8/PP16 transport: closed physical lanes, exact point-to-point HLO, one compiled
  global program, deterministic checksums, warmed distributions, and fresh fleet XPlanes.
- Independent batch-one MoE reference: FP8 128x128 block dequant, FP32 sigmoid/noaux_tc routing,
  correction bias for selection only, exact lowest-id ties, normalized top-8, post-reduction 2.5
  scale, 64 complete experts/chip, and intermediate-sharded shared expert. Routed/shared partials
  share one four-chip combine without mixing value domains. Forced four-device distributed and
  all-top-4-on-one-chip cases pass bounded tensor equivalence; routing is elementwise exact.
- One-layer format/packer validates every source leaf and writes four final PP8 identity-owned
  files with complete expert ownership, shared-intermediate shards, replicated router state,
  payload/file/manifest checksums, finite checks, and append-only refusal. Tiny roundtrip/corruption
  tests pass. Real artifact `greenfield_one_layer_pack_20260805T151828912346032Z` packed 1,544
  layer-3 leaves / 9,706,940,416 unique source bytes from only shards 38–40 into four independently
  hashed 2,429,096,640-byte files. The 9,716,380,672-byte packed payload reconciles exactly,
  including intentional router replication. Manifest `68ef8201...f938`, approved-bucket upload,
  and local/remote `SUCCESS` pass at code `1969d925...f3d4`; this is layout evidence, not Gate B.
- Independent raw-source PyTorch oracle capture never imports the greenfield JAX kernels, packed
  checkpoint, model class, or legacy execution. Real artifact
  `greenfield_one_layer_oracle_20260805T162210370718434Z` pins 104 exact source tensors and accepted
  legacy/vLLM source hashes. Normal routes `[161,217,206,240,186,180,37,81]` span all four PP8
  slots; the adversarial routes are all experts 128–135 on slot 2. Manifest `c63ffa19...ebff`,
  274,944-byte safetensor SHA `4aa7910b...784b`, and local/remote `SUCCESS` pass at `27ebdec`.
- Direct PP8 loader validates every packed identity/hash, maps captured physical stage slots to the
  isolated runtime subcube, transfers 56 already-owned shards, and performs all 24 FP8 lookup/scale
  conversions on device. It performs zero host FP8 dequantizations and zero host global concats.
- Protected PP8 layer-3 metal proof DB 417 / `greenfield_real_layer_pp8_20260805T165737737514245Z`
  passed both oracle cases at code `db19893`. Normal/concentrated p50 wall is `0.696215/1.134090 ms`
  over 1,000 profiler-free samples after 200 warmups. Routes are elementwise exact; both output
  comparisons have max `0.03125`, p99 `0.01171875`, and mean below `0.00236` BF16 absolute error.
  Optimized HLO SHA `950b5eb2...977d` contains exactly one
  `bf16[2,1,6144]` all-reduce over `{{0,1,2,3}}` and no other collective. HBM after timing is
  `4,860,038,656` bytes/chip with `5,639,681,536` measured peak against `33,014,413,312` available.
  A fresh 20-step/8-core XPlane observes exactly one physical `psum` per step. All hashes, DB
  integrity, approved archive, remote `SUCCESS`, and authenticated 8/8 post-census pass.
- Final-layout PP16 artifact `greenfield_one_layer_pack_pp16_20260805T172003732526347Z` writes
  two independently hashed 4,855,045,080-byte files at pack code `51d1df9`: experts
  `0:128/128:256`, shared intermediate `0:1024/1024:2048`, manifest
  `38573723...e454`, the same immutable source revision, exact source/payload reconciliation,
  and approved-bucket `SUCCESS`.
- Protected PP16 DB 418 / `greenfield_real_layer_pp16_20260805T172807177182695Z` passed at
  `6751a93`. Normal/concentrated p50 is `0.900610/1.075795 ms`; routes are exact and
  output max/p99/mean errors are at most `0.03125/0.0078125/0.002141`. HLO SHA
  `f9a97fbb...d051` contains exactly one `bf16[2,1,6144]` all-reduce over
  `{{0,1}}`. Peak HBM is `11,276,493,312` bytes/chip; the fresh 4-core/20-step XPlane
  has exactly one physical all-reduce on all 80 core-steps. DB, archive, hashes, and 8/8 pre/post
  census pass.
- Complete inventory/plan/layout maps all 118,629 source leaves / 755,617,140,416 bytes to 32 PP8
  base owners plus four MTP owners. Full pack `greenfield_full_pack_pp8_20260805T182222755355852Z`
  contains 760,215,571,712 payload bytes; packed manifest is `08694931...78f1` and the approved
  checkpoint prefix has `SUCCESS`.
- Protected DB 420 / `greenfield_full_checkpoint_load_pp8_20260805T201317772639405Z` directly
  loaded all 32 base owners / 750,122,559,744 bytes on all 32 chips at `91a8c47`. All 122,640 final
  shards pass identity, finite, physical ownership, and device-roundtrip checks; host FP8 dequant,
  global concat, and runtime reshard counts are zero. Maximum weights-only HBM is 24,840,958,464
  bytes/chip with at least 8,173,454,848 free. Ledger, DB integrity, three 8/8 censuses, approved
  archive, and an independent byte-for-byte download of all 47 files pass. Gate B is complete.
- Exact reference RMSNorm/final norm, dense/SwiGLU/residual/embedding/logits, accepted RoPE, FP32
  DSA scorer, exact lowest-global-position ties, distributed exact top-k, stage-local striped KV
  lookup, sparse MLA/LSE merge, and compact IndexShare carriage are implemented at `10e5097`.
  Full-width FP32 and BF16 comparisons against the pinned legacy sparse-attention oracle are
  elementwise exact, including the four-owner BF16 merge. These reference tests prove semantics
  only; the protected real TPU result is recorded below. No decoder, serving, decoder-HBM, or
  genuine token-throughput result exists.
- Independent raw-source Gate C oracle
  `greenfield_gate_c_oracle_20260805T212801776974822Z` passed at `602d42f`. It consumes 31 exact
  layer-2/3 tensors from source shards 20/38/40, never imports JAX/model/legacy execution, and
  captures real dense, full-DSA, and IndexShare-reuse cases. The 2,304-position full scorer selects
  exactly 2,048 unique positions including current position 2,303; layer 3 reuses the identical
  score-ordered `int32[1,2048]` state (8,192 bytes), privately sorts only for attention, and proves
  write-before-attend. The 36,060,088-byte safetensor is `a8927db...0364`; manifest is
  `54262529...4a9f`; local/remote `SUCCESS` and the local evidence ledger pass. This is an
  independent correctness artifact, not TPU or performance proof.
- Bounded final-owner Gate C checkpoint
  `greenfield_gate_c_pack_20260805T214609093206269Z` passed at `8a50d6a`. It is an exact
  content-addressed subset of protected full layout `aca0eb6d...6a0` and oracle
  `54262529...4a9f`: 31/31 raw leaves from shards 20/38/40 are hashed while streaming once, with
  413,810,816 unique source bytes becoming four equal 125,611,520-byte PP8 stage-0 payloads.
  Packed payload/file totals are 502,446,080/502,461,536 bytes; layout is `cdbea04f...c678` and
  packed manifest is `3c5c48da...2a8a`. All four file hashes, the sealed local ledger, remote
  size/generation/CRC32C, approved-bucket archive, and local/remote `SUCCESS` pass. This removes
  the need to load ~92 GB of complete stage-0 owner files for Gate C but is not a TPU result.
- Protected DB 421 / `greenfield_gate_c_pp8_20260805T224645828157364Z` passed real dense,
  full-DSA, and IndexShare TPU execution at exact code `dc20b3f` on physical PP8 stage 0
  (worker 2). It directly loads 502,446,080 packed bytes with 124 final-owner transfers, 44
  device FP8 dequantizations, and zero host dequant/global concat/runtime reshard. Dense output
  max error is `0.00390625`; DSA score max/mean is `0.003605/0.000965`; IndexShare output max/mean
  is `0.0078125/0.000167`. DSA selection and lowest-position tie order are elementwise exact for
  the actual TPU FP32 score row, the 8,192-byte score-ordered state is fed directly to IndexShare,
  and cache/state integrity is exact. Optimized HLO is strictly four-chip local: dense `0AG/1AR`,
  DSA `3AG/0AR`, and TPU-rewritten IndexShare `2AG/3AR`, all over `{{0,1,2,3}}`. A fresh trace has
  20 invocations/case on eight cores and matches the HLO-derived physical collective counts.
  Peak HBM is 280,745,984 bytes/chip for this bounded layer proof. DB integrity, sealed ledger,
  approved archive/SUCCESS, and 8/8 pre/post census pass. This is correctness/mechanism evidence,
  not latency or tok/s evidence.
- The independent PyTorch CPU oracle and TPU scorer are not bitwise-identical: bounded FP32 score
  drift changes two members of the 2,048-of-2,304 cutoff set (2,046 overlap) and therefore many
  score-order positions. This is preserved as an explicit non-relaxed diagnostic. The runtime does
  not use CPU positions: it selects exactly from actual TPU scores and carries that exact state.
  Do not claim raw cross-framework position identity from DB 421.
- The executable-ready PP8 runtime derivative is complete at
  `greenfield_runtime_pack_pp8_20260806T002756318310857Z`: 32 files / 11,648 tensor records /
  834,178,632,448 file bytes, runtime manifest `fdedaae3...ec`, layout
  `841a18f6...ac`, and exactly 26,068,042,432 weight bytes/chip. It consumes DB420 ownership
  exactly, keeps FP8 as U8, streams with bounded host memory, and has approved-bucket `SUCCESS`.
- Rejected diagnostic `greenfield_short_decoder_compile_pp8_20260806T004625161993456Z` at
  `20f44e6` loaded ~104.713 GB/host and compiled the real 78-layer 2K body successfully in
  380.003 seconds. Optimized HLO has ~192,401 instructions and only local groups, but the backend
  emitted 2,707,043 program bundles and 580 overlays. Program/argument memory is 1.17/23.41 GB;
  observed HBM is ~25.46 GiB/chip, so capacity is not the immediate failure. Fleet sequencing
  showed stages advancing one at a time at 100% duty; one body invocation projected ~25–30
  minutes because the readable U8 lookup/dequant and eight conditional expert branches exploded
  into the executable. The run was stopped and is never performance evidence.
- The same HLO contains 219 all-gathers, 294 physical all-reduces, and 16 permutes. The previous
  312-AR contract was counting logical results as launches: all 312 results exist, while TPU XLA
  fuses 78 padded-u32 results into 43 singles, 16 pairs, and one triple, saving 18 physical
  launches. The contract now pins both 294 physical instructions and 312 logical components plus
  exact result shapes, so neither tuple fusion nor missing work is obscured.
- Protected DB 422 / `greenfield_fp8_matmul_20260806T015232890679994Z` proves the first independent
  Pallas raw-FP8 kernel on v4 at `df44475`. Production shape is `M8xK6144 @ N2048xK6144`; U8 is
  bitcast E4M3FN, only 128x128 weight tiles are dequantized in VMEM, BF16 feeds the MXU, and FP32
  accumulates. It is elementwise exact against full JAX dequant+dot for the protected input. One
  compact TPU custom call compiles in 0.538 s, contains no full BF16/F32 weight overlay, uses 69,632
  scoped VMEM bytes, and has profiler-free p50/p90/p95/p99
  `0.520605/0.530900/0.534043/0.544112 ms` over 1,000 samples after 200 warmups. Peak process HBM
  is 315,956,736 bytes. DB/archive/hashes and 8/8 pre/post census pass. This is standalone kernel
  wall, not layer latency or tok/s. Three preceding Mosaic-layout diagnostics failed closed before
  timing, were archived, and each ended 8/8 clean.
- Protected DB 423 / `greenfield_fp8_up_gate_20260806T020551714072561Z` proves paired gate and up
  projections in one v4 custom call at `7654338`. Both outputs are elementwise exact against their
  independent full-dequant/FP32-dot fallbacks. Compile is `0.540 s`; profiler-free p50/p90/p95/p99
  is `0.815435/0.827151/0.833372/0.851290 ms` over 1,000 samples after 200 warmups. The HLO has two
  raw FP8 matrices and two bounded scale tables, no full BF16/F32 weight overlay, and 184,320 bytes
  scoped VMEM. Peak process HBM is 695,120,896 bytes; DB/archive/hashes and 8/8 cleanup pass. This
  is still a dense same-weight mechanism kernel: real routed integration must select potentially
  different expert matrices for the eight routes without recreating the rejected branch graph.
- Protected DB 424--427 implement the distinct-selected-expert gate/up mechanism and keep raw FP8
  weights in HBM. All four accepted runs preserve route order, select eight distinct local expert
  matrices, match complete dequantization/FP32-dot within max BF16 error `0.0078125`, contain one
  selected-expert Pallas kernel and no full BF16/F32 overlay, have approved archives/DB linkage,
  and end 8/8 clean. Their p50s are `18.744/18.009/20.305/23.197 ms`; widening K tiles helped only
  4%, while software pipelining and vector-scale staging regressed. They are correctness/mechanism
  proofs but are performance-rejected and must not enter the decoder. Three intervening scale-layout
  diagnostics failed comparison before timing and were preserved without DB claims; they exposed
  and fixed a BlockSpec element-offset-versus-block-index error.
- DB 427 receives each full local raw table as `[G,N,K]` and transposes it to the Pallas `[G,K,N]`
  access order inside the JIT, so final-layout packing remains the next controlled discriminator.
  The original inference that its two auxiliary custom calls were transpose/bitcast fusions was
  wrong: preserved follow-up HLO proves they are bounded `AssumeGatherIndicesInBound` markers for
  the two compact scale gathers. This correction is explicit; custom-call count alone did not prove
  the transpose cost.
- Diagnostics `...T024137692707617Z` at `5b77934` and `...T024407985780210Z` at `911ca88` supplied
  final `[G,K,N]` tables and failed closed on the over-strict one-total-call contract before
  correctness/timing; both ended 8/8 clean. The second preserved full HLO: the selected kernel's
  operands are raw `u8[64,6144,2048]`, plus exactly two bounded metadata markers and no weight
  transform call. The corrected contract allows only those markers, forbids every other auxiliary
  call/full F8 table view, and retains tile-local U8-to-F8 bitcast inside Pallas.
- Protected DB 428 / `greenfield_fp8_selected_up_gate_20260806T024602582280149Z` passes that final
  layout at `e5a70be`. Eight distinct all-local routes match full dequant/FP32 dot with max/p99/mean
  BF16 error `0.0078125/0/5.77e-7` (up exact). Compile is `0.905 s`; p50/p90/p95/p99 is
  `4.492525/4.504463/4.509004/4.517450 ms`, a `5.16x` improvement over DB 427's `23.196868 ms`.
  HLO has one raw-U8 selected Pallas kernel, exactly two allowed scale-gather markers, no unexpected
  auxiliary call/full F8 or BF16/F32 table, and 946,176 scoped VMEM bytes. Peak HBM is
  2,395,303,936 bytes; DB/archive/evidence hashes/remote SUCCESS and 8/8 cleanup pass. This proves
  final layout was a major cost, but `4.49 ms` gate/up alone remains performance-rejected.
- Protected paired DB 429/430 at `b900cea` compact owned routes on device and use their dynamic count
  as the Pallas pipeline bound; outputs are gathered back to original top-8 order and non-owner rows
  are exact zeros. Normal two-owned routes p50/p90/p95/p99 is
  `1.341385/1.354894/1.360536/1.371749 ms`; concentrated eight is
  `4.496970/4.509263/4.514649/4.524864 ms`. Both comparisons pass (normal max error
  `0.00012207`, concentrated `0.0078125`; p99 0), one raw-U8 TPU kernel plus four bounded metadata
  markers/no overlay or unexpected call passes, and peak HBM is 2.307/2.395 GB. Both DB records,
  archives, hashes, remote SUCCESS markers, and 8/8 cleanup pass. Compaction is a `3.35x` normal
  win and adds only ~`0.004 ms` to the all-eight ceiling, but `1.34 ms` gate/up is not yet promoted.
- Protected DB 431/432 at `ea8a61a` tested one persistent `[G,K,gate_then_up]` stream against that
  split-stream baseline. Exactness, the raw-U8/no-overlay HLO contract, DB/archive/hashes, and 8/8
  cleanup pass. Normal-two p50 improved only `1.341385 -> 1.327685 ms` (`1.02%`), concentrated-eight
  regressed `4.496970 -> 4.509895 ms` (`0.29%`), and measured peak allocation rose from
  `2.307/2.395` to `3.919/4.007 GB`. This is an honest null and is not promoted. The accepted
  split-stream API/layout was restored after the measurement; DB 429/430 remain the baseline.
- Compact-scale diagnostic `...T031705611686395Z` at `364867e` failed closed before timing because
  Mosaic requires the dynamic K-block HBM offset to align to its 128-element TPU tile; it has no DB
  or performance claim and ended 8/8 clean. Protected DB 433 at `8be32e0` then tested an aligned
  `[G,Nblock,128]` scale table with masked four-block extraction inside Pallas. It is exact and all
  HLO/DB/archive/cleanup gates pass, but normal-two p50 regressed `1.341385 -> 3.416799 ms` (`2.55x`)
  while scoped VMEM grew `946,176 -> 6,596,608` bytes. No concentrated run was needed to reject it.
  The DB 429/430 scale staging and API were restored after the proof.

## Protected evidence

Topology DB 405 / `greenfield_topology_20260805T125842425591441Z` proved 32 v4 chips in `2x4x4`,
actual local ordering, topology hash `294e777...559`, PP8 `d5943ab8...c14`, PP16
`6383e57c...f21`, remote `SUCCESS`, and 8/8 clean census.

Collective DB 406–414 all have approved archives and clean census. Dominant `bf16[2,6144]` p50 for
75 all-reduces is g2/g4/g8/g32 `0.836/1.083/1.471/3.941 ms`; 75 full-ring permutes are `0.791 ms`.
The matrix covers required bf16 shapes, `f32[1,6144]`, and `int32[1,2048]` metadata. Small decode
reduce-scatter is not falsely timed: TPU XLA rewrites it to all-reduce, and three diagnostics fail
closed with saved HLO.

Transport DB 415 / `greenfield_transport_20260805T142953361259007Z` passed 16 cases at `577aa4b`:

- One invocation carries every rank-dependent payload through every stage and back to its exact
  lane origin. HLO contains exactly 8 PP8 or 16 PP16 neighbor `collective-permute` operations with
  exact bf16/int32 shapes and pairs. It contains no all-reduce/gather/to-all/reduce-scatter, host
  staging, Ray/Python stage dispatch, or model-equivalent compute.
- `bf16[1,6144]` fleet-max profiler-free p50: PP8 `0.331145 ms` vs `0.300895` control
  (`0.030250 ms` net); PP16 `0.407385` vs `0.331230` (`0.076155 ms` net). Other payloads are
  approximately `0.024–0.077 ms` net. This is synthetic mechanism latency, not token speed.
- Approved archive `SUCCESS`, DB linkage, full distributions, provenance, and 8/8 cleanup pass.

Trace DB 416 / `greenfield_transport_trace_20260805T143832942547470Z` passed at `8aee335`:

- Both plans have 8 fresh XPlanes, 64 TPU cores, and exactly 20 selected steps/core.
- Every PP8 core reports exactly 8 permute starts + 8 dones/step; PP16 reports 16 + 16;
  forbidden collective count is zero. Trace-contaminated timing is excluded from latency claims.
- Archive, hashes, DB snapshot, authenticated pre/post zero census, and remote `SUCCESS` pass.

Real layer DB 417 / `greenfield_real_layer_pp8_20260805T165737737514245Z` passed at `db19893`:

- The loader consumes the bounded final-ownership PP8 pack directly; no host dequant/global concat.
- Normal p50/p90/p95/p99 is `0.696215/0.716489/0.722115/0.742864 ms`; concentrated is
  `1.134090/1.155466/1.163958/1.195100 ms`. These are profiler-free one-layer wall distributions,
  not token throughput. The concentrated distribution has one retained `40.141 ms` host outlier.
- Exact routes, bounded tensor comparison, one local four-rank HLO collective, per-chip HBM, a fresh
  post-timing XPlane, append-only DB/archive, checksums, and 8/8 clean pre/post census all pass.

Real layer DB 418 / `greenfield_real_layer_pp16_20260805T172807177182695Z` passed at
`6751a93`:

- The two-chip stage is captured physical stage 10, global ids `[4,5]`, coordinates
  `[(0,2,0),(1,2,0)]`. Final-owner load performs 28 direct transfers, 12 device
  dequantizations, zero host dequant/global concat, and owns 128 complete experts plus shared width
  1024 per chip.
- Normal p50/p90/p95/p99 is `0.900610/0.919235/0.925838/0.949264 ms`; concentrated is
  `1.075795/1.093593/1.101284/1.156754 ms`. Exactly one local two-rank combine is present.
- HBM after timing is `9,710,615,552` bytes/chip; measured peak is `11,276,493,312` of
  `33,014,413,312`. Correctness, HLO, wall, fresh trace, DB/archive, and cleanup all pass.

Full load DB 420 / `greenfield_full_checkpoint_load_pp8_20260805T201317772639405Z` passed at
`91a8c47`:

- All eight standalone stages resolved from captured topology and loaded all 32 final base owners.
  Stage payloads are 89.393–99.323 GB; per-chip peak is 22.357–24.841 GB.
- Raw FP8 U8 plus local FP32 scales remain in final ownership. File/tensor SHA, finite checks,
  byte totals, device round trips, state manifests, DB integrity, archive, and three 8/8 censuses
  pass. The independently downloaded 47-file archive is byte-identical to local evidence.
- This is load integrity, not decode. `promotion_memory_proven=false`: the 8.173 GB minimum free is
  before KV, DSA state, executables/overlays, and decoder temporaries. It has no tok/s claim.

Gate C DB 421 / `greenfield_gate_c_pp8_20260805T224645828157364Z` passed at `dc20b3f`:

- The direct bounded load binds packed manifest `3c5c48da...2a8a`, layout
  `cdbea04f...c678`, and independent oracle `54262529...4a9f`; all load fast-path counters pass.
- Dense/full-DSA/IndexShare outputs are bounded against the raw oracle. Exact device-score top-k,
  tie order, 8,192-byte IndexShare carriage, write-before-attend, selected KV, live cache, and
  state reuse pass. Raw CPU-oracle selection has 2,046/2,048 set overlap because the two
  FP32 implementations differ near the cutoff; it is diagnostic only and is not silently relaxed.
- HLO and fresh XPlane prove only local four-rank collectives with exact per-case counts. Maximum
  bounded-proof peak HBM is 280,745,984 bytes/chip. DB/archive/ledger and 8/8 cleanup pass.
- This is Gate C layer correctness/mechanism evidence, not a full decoder or token-speed result.

Rejected DB 419 / `greenfield_full_checkpoint_load_pp8_20260805T194526167625636Z` is preserved but
never promotable: its first harness appended to `orchestrator.log` after hashing it, so its local
ledger failed. DB 420 is a fresh run using the corrected seal-before-success harness.

Rejected diagnostic `greenfield_real_layer_pp16_20260805T172345348528631Z` proved that
libtpu cannot initialize local devices `0,1` as a standalone `2x1x1` slice: the driver
reports duplicate coordinate assignment. Its driver logs are archived and it has no DB/performance
claim. A protected subset diagnostic then proved the correct runtime: initialize the known-good
`2x2` host subcube while placing arrays/executable only on adjacent devices `0,1`; DB
418 uses that exact mechanism.

Rejected diagnostic `...T143608529930365Z` has no DB/status claim: valid distinct rank traces were
downloaded into worker-0's original local trace tree, causing a duplicate-host parser refusal. The
fixed proof uses an isolated canonical fleet directory; it did not reinterpret the failed result.

## Interpretation

Legacy attributes `106.495 ms/token` to 75 full-pod MoE combines (`1.420 ms/layer`), while exact
75-op full-pod all-reduce needs `3.941 ms` and an entire PP8 residual ring costs only `0.331 ms`.
The old loss is therefore arrival skew, reshard/layout work, barrier waiting, and surrounding
decomposition—not raw 12 KiB ICI or stage-transfer bandwidth. Primitive replacement alone is not
enough; stage-local model layout remains the structural requirement.

## Exact next sequence

1. DB 429/430 remain the route-proportional `1.341/4.497 ms` normal-two/concentrated-eight baseline;
   DB 431/432 reject a merged stream and DB 433 rejects in-kernel compact-scale extraction. The
   owned-route grid is mathematically independent but currently declared `arbitrary`; test a
   `parallel` route dimension (and only then route/output grid order) under exact HLO/correctness and
   both distributions. Only after a competitive protected kernel should activation, down, and
   routed/shared combine be composed in binding Pallas order and the real one-layer gate repeated.
   Never substitute same-weight M8 or a slow correctness-only GMM for production proof.
2. Recompile only after the short-model kernel/overlay gate is acceptable. Bind the optimized
   layer to the complete runtime artifact, add embedding/final norm/distributed logits/token
   control, and prove complete 2K/8K Gate D with raw tokens, exact DSA/cache state, local-only HLO,
   measured HBM, fresh trace, and profiler-free steady wall. Do not attempt 256K before it passes.
3. Continue Gates E–H in binding order. The one-layer comparison provisionally favors PP8 for
   normal routing (`0.696` vs `0.901 ms`) while PP16 wins the concentrated adversary
   (`1.076` vs `1.134 ms`); only complete protected decoder evidence may choose the
   final plan.

The pod ended the latest proof with all eight hosts `CENSUS_OK`.
