# HANDOFF — topology-first greenfield rewrite

**Updated:** 2026-08-07 00:15 UTC

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
- Final-layout Pallas derivative `greenfield_one_layer_pallas_pack_20260806T041854316280053Z`
  transposes only routed raw-FP8 tables offline, preserves shared/router bytes, and writes four
  independently hashed 2,429,096,824-byte files. Manifest `3da63bd9...e427`, layout
  `3d9f3b85...545e`, 9,716,380,672-byte payload, exact source-transform verification, approved
  archive, and remote `SUCCESS` pass. Its loader performs 56 final-owner raw transfers and zero
  host/device dequantization, host concat, or runtime weight transpose.
- Protected raw-FP8 Pallas layer-3 proof DB 438 /
  `greenfield_real_layer_pp8_pallas_20260806T050514347248323Z` passed at `5fed847`. Normal and
  concentrated p50/p90/p95/p99 are `3.164060/3.185520/3.193292/3.229554 ms` and
  `7.171980/7.194523/7.204370/7.263152 ms`. Routes are exact; output max/p99/mean error is at most
  `0.03125/0.01171875/0.002444`, and route-weight max error is below `9e-8`. HLO SHA
  `0c8878cc...b20a` has four exact raw-U8 Pallas calls and one local
  `bf16[2,1,6144]` all-reduce over `{{0,1,2,3}}`, with no decoded weight overlay or other
  collective. Compile is `1.674 s`; measured peak HBM is 2,431,646,720 bytes/chip. Fresh XPlane,
  DB/archive/hashes/remote `SUCCESS`, and 8/8 cleanup pass. This is the first correct no-overlay
  real MoE layer, but its latency is performance-rejected pending kernel-boundary fusion.
- Protected fused-routed proof DB 439 /
  `greenfield_real_layer_pp8_pallas_20260806T052141734174170Z` passed at `fb04875`. One Pallas call
  now performs selected gate/up, exact BF16 SwiGLU, and down with gate/up retained only in VMEM.
  Normal/concentrated p50 is `3.121940/7.063344 ms`, only `1.3%/1.5%` faster than DB 438. Exact
  routes, bounded outputs, peak HBM 2,431,378,432 bytes/chip, archive, and 8/8 cleanup pass. HLO
  `918bbabd...826f` drops from four to three raw-U8 calls, seven to five bounded gathers, and two to
  zero bitpacked gather/scatters while retaining one exact local all-reduce and no overlay. Fresh
  XPlane still spends `2.788/4.690 ms` per alternating step in the physical psum (`59.4%` of busy
  time). The routed boundary was real but not dominant.
- Protected DB 440 / `greenfield_real_layer_pp8_pallas_20260806T052955364574577Z` fused the
  remaining shared gate/up+SwiGLU+down boundary into a second all-in-one kernel and passed every
  correctness/HLO/HBM/archive/cleanup gate at `cfd5bab`, but regressed DB 439 normal/concentrated
  p50 to `3.169569/7.122444 ms` (`+1.53%/+0.84%`). Custom-call busy time rose
  `1.724 -> 1.863 ms` while psum stayed `2.787 ms`; reject the composition. The tested kernel is
  retained default-off, while the active stage and exact HLO guard are restored to DB 439's
  three-call composition. Do not spend more time on launch-only fusion: next characterize and
  reduce route-imbalance/collective arrival skew before integrating the short decoder. No decoder
  or tok/s result exists yet.
- Expert-feature derivative `greenfield_one_layer_pallas_feature_pack_20260806T054520020812918Z`
  redistributes every routed expert across all four intermediate-feature owners without changing
  per-chip persistent bytes. Manifest `a8b91435...5cc6`, layout `e613d9ef...c431`, exact
  9,716,380,672-byte payload/source reconciliation, approved remote `SUCCESS`, and direct-loader
  tests pass. Runtime weight transpose/concat/dequant counts are zero.
- Protected DB 441 / `greenfield_real_layer_pp8_pallas_feature_20260806T055854589778101Z` at
  `65ded2c` is the first structural route-balance win. Every chip owns all 256 expert identities and
  one 512-wide intermediate slice, so normal and all-eight-concentrated routes execute the same
  local work. Normal/concentrated p50 is `2.308015/2.318155 ms`, improving DB 439 by
  `26.07%/67.18%`. Routes are exact and output max/p99/mean error is at most
  `0.03125/0.01171875/0.002507`. HLO `3bbd527f...383f` has three raw-U8 calls, one exact local
  four-chip all-reduce, no decoded overlay, and the selected kernel remains `1.577 ms`. Fresh
  XPlane shows the physical psum collapse from `2.787760` to `0.028511 ms`; peak HBM is
  2,430,860,800 bytes/chip. DB/archive/hashes/remote `SUCCESS` and 8/8 cleanup pass. Select this
  routed layout for PP8 full-runtime integration. This is a real layer, not decoder or tok/s proof.
- Complete feature-runtime derivative
  `greenfield_runtime_feature_pack_pp8_20260806T064010287072141Z` passed at pack code `d9a883b`.
  It writes 32 files / 834,178,632,960 file bytes and 834,177,357,824 payload bytes, with exactly
  26,068,042,432 runtime weight bytes/chip. Manifest `54e2f89b...d9917`, layout
  `ba21c4ec...c9e`, plan `f46f91c3...826a`, and schedule `b407fcf5...1773` bind every source
  tensor hash, feature slice, offline transpose, destination tensor/file hash, GCS generation and
  CRC32C, and protected source runtime `fdedaae3...e31dec`. Mounted verification, checkpoint/result
  `SUCCESS`, approved archive, and 8/8 post-census pass. This is a complete executable checkpoint
  artifact, not a decoder or tok/s result.
- Commits `74a2952`, `33aa420`, and `aff0f42` bind the default-off Pallas feature-MoE backend to
  the 78-layer decoder, require the exact feature checkpoint layout, require 75 each of the three
  production raw-U8 kernels, forbid decoded full-expert overlays, and abort before first execution
  on HLO drift. Focused runtime/feature/decoder coverage is 14/14; protected full-body metal proof
  is still required.
- Raw-FP8 stage-linear integration at `a86ff8b` replaces q_a, q_b, kv_a, attention output, and
  dense SwiGLU with 315 exact Pallas calls across the 78-layer body, while retaining the accepted
  225-call feature-MoE path. Standalone fused RMSNorm/linear DB 451 is elementwise exact at the
  production M1/K6144/N2048 shape and p50 `0.526695 ms`. Protected body DB 452 passes strict HLO,
  ownership, fleet, archive, and cleanup contracts at p50 `4,876.099672 ms`, a `12.06x` reduction
  from DB 442's `58,804.002894 ms`. This is transformer-body wall, not raw-token throughput.
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
- Protected DB 434/435 at `a21ad09` declared the independent compact-route grid axis `parallel`.
  Normal/concentrated p50 became `1.344635/4.497115 ms` versus DB 429/430's
  `1.341385/4.496970 ms`; exactness, HLO/no-overlay, DB/archive/hashes, and 8/8 cleanup pass. The
  annotation is a performance null and was restored to `arbitrary`. After final layout, device
  compaction, merged-stream, scale-staging, and scheduling experiments, DB 429/430 are the selected
  gate/up basis for composing activation and down; this is a kernel-stage choice, not a decoder or
  token-speed promotion.
- Protected DB 436/437 at `f496eb1` prove exact BF16 SwiGLU plus selected raw-FP8 down projection.
  Normal-two/concentrated-eight p50 is `0.773045/2.421714 ms`; p90/p95/p99 is
  `0.781022/0.786141/0.794347` and `2.432291/2.435744/2.445903 ms`. Max/p99/mean BF16
  error is `0.015625/0.0078125/0.000193` and `0.03125/0.03125/0.001447`. The identical HLO
  has one raw `u8[64,2048,6144]` Pallas call, exactly two bounded compaction scatters plus
  selected-scale/final-order gathers, no unexpected call or decoded overlay, and 491,520 bytes
  scoped VMEM. Peak allocation is 1.489/1.502 GB. DB/archive/hashes/remote SUCCESS and 8/8 cleanup
  pass. A prior `22e46ab` diagnostic compiled the kernel but failed the over-strict metadata
  classifier before correctness/timing; it has no DB claim and ended 8/8 clean. Activation and down
  are fused without an activated-intermediate HBM write, but the separately called gate/up kernel
  still writes its two BF16 outputs; this is not yet a complete expert, layer, decoder, or tok/s.

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

1. Bind the sealed run-139/item-628 oracle to teacher-forced prefill and recurrent decode, then
   prove complete 2K Gate D raw tokens, state/cache integrity, local-only HLO, measured HBM, fresh
   trace, profiler-free steady wall, DB/archive, and cleanup.
2. Add default-off full-decoder DSA event observability and obtain an independent exact event-order
   oracle. The sealed raw-token artifact does not by itself prove all 21 producer events.
3. Repeat Gate D at 8K, then optimize the measured complete decoder below 200 ms. Only afterward
   run 128K/256K and identical-condition PP16/WS32 adjudication in the binding order. No body-only
   or synthetic-state result is an answer tok/s claim.

The pod ended the latest proof with all eight hosts `CENSUS_OK`.

## Protected Pallas DSA scorer

DB 443 / `greenfield_dsa_score_20260806T095455945075126Z` at `d068a9f` passes Section 7.2 item 5
at the production one-row 256K/LP4 shape: query `f32[1,32,128]`, local BF16 keys
`[65,536,128]`, and scores `f32[1,65,536]`. One Pallas call performs both highest-precision dots,
per-head ReLU, signed head weighting, and the FP32 head reduction without a per-head score overlay.
HLO `5e2b7185...b295` contains no collective, batch-32 dead rows, unexpected custom call, or
`[32,context]` HBM tensor.

Against the TPU JAX reference, score max/mean/p99 error is
`2.861e-6/2.417e-7/1.386e-6`; all 2,048 selected positions and their order are elementwise exact.
After 200 warmups, 1,000 profiler-free samples give p50/p90/p95/p99
`0.326595/0.335482/0.341040/0.350320 ms`. Compile is `0.394 s`; peak HBM is `20,491,776` bytes.
Runner/summary SHAs are `992bc991...fe12` / `37789e92...0af`; DB snapshot, approved archive/remote
`SUCCESS`, and 8/8 cleanup pass. This is a standalone scorer proof, not integrated layer/token wall.

## Protected exact Pallas DSA top-k

DB 445 / `greenfield_dsa_topk_20260806T102752126905724Z` at `3870c2f` closes Section 7.2 item 6.
TPU v4 has no SparseCore sort, so six exact TensorCore bitonic calls reduce one runtime-shaped
`f32[1,65,536]`/`s32[65,536]` owner row to 2,048 candidates; two more calls merge the permuted
four-owner `f32/s32[4,1,2,048]` union. Scores, positions, valid counts, sentinel tails, high-score
ties, and lowest-global-position order are elementwise exact against both the TPU JAX reference
and an independent host lexicographic oracle.

After 200 warmups, 1,000 profiler-free samples give local p50/p90/p95/p99
`1.364405/1.376845/1.379481/1.388090 ms` and merge
`0.337671/0.349403/0.354289/0.362475 ms`. Optimized local/merge HLO SHAs are
`e6b8e209...e7e90e` / `d990a754...0b674`; they contain exactly `6/2` named Pallas calls and no
XLA sort/top-k, collective, unexpected call, or batch-32 row. Compile is `8.381/6.176 s`; peak HBM
is `17,794,560` bytes. Runner/summary SHAs are `8d41a09c...bbc7` / `a0bd5514...5924`;
DB/archive/remote `SUCCESS` and 8/8 cleanup pass. This is standalone selector evidence, not layer
or token wall.

DB 444 at `bacfbdf` is the exact but performance-rejected reduction predecessor: local/merge p50
was `59.979532/4.495320 ms`. Three earlier bitonic/lowering diagnostics failed closed before
timing on TPU-v4 layout, scalar-bool, and Mosaic legalization limits; all preserved diagnostics
ended 8/8 clean. The accepted bitonic network is `43.96x/13.31x` faster than DB 444.

## Protected fused selected-KV sparse attention

DB 446 / `greenfield_sparse_attention_20260806T112854965924060Z` at `20527b9` closes Section 7.2
item 7. Exact owner filtering/order feeds a single local Pallas attention call. Each selected row
uses the minimum legal aligned eight-row TPU-v4 HBM-to-VMEM DMA tile, then compact external
`[2048,1,8]` BF16 lane metadata selects the requested row before online FP32 softmax. There is no
selected-KV HBM tensor; only output/LSE escape the kernel. Empty/tail/skew/invalid-metadata health
and default-off reference fallback pass.

After 200 warmups and 1,000 profiler-free samples, balanced 512-of-2,048-owner p50/p90/p95/p99 is
`0.302346/0.316633/0.323973/0.379626 ms`; worst concentrated 2,048-of-2,048 is
`0.398000/0.413380/0.421573/0.475643 ms`. Output max error is `0.00390625`; LSE max error is
`3.815e-6`. HLO `63aac56e...9809` has exactly one ordering and one sparse-attention Pallas call,
one compact bounded gather, no selected `[2048,640]` tensor, collective, dead batch, or sort/top-k
fallback. Peak HBM is 87,524,864 bytes. DB/archive/remote `SUCCESS` and 8/8 cleanup pass. The
minimum eight-row DMA causes finite-cache overfetch; no OOB occurs, but arbitrary NaNs in the seven
unselected physical lanes are outside the finite-cache contract because TensorCore `0*NaN` can
propagate. This is standalone kernel latency, not token speed.

DB 447 / `greenfield_gate_c_pp8_sparse_attention_20260806T114113674126789Z` at `6221e87` proves
the default-off Pallas path integrated into the real PP8 dense + full-DSA + IndexShare Gate C.
All bounded comparisons pass: selected/attention positions are exact, attended-latent max/mean
error is `0.0029297/0.0006252`, LSE `0.0009532/0.0002057`, and layer output
`0.0078125/0.0001641`. IndexShare feeds the producer-selected 8,192-byte state directly and
preserves order. HLO `ca8017ea...dfe7` has exactly the two expected Pallas calls, no dead row, and
only the four-chip stage group: three local gathers (query, LSE, partial output) plus validity and
output reductions. Peak HBM is 281,821,696 bytes/chip. Fresh XPlane, DB/archive hashes, remote
`SUCCESS`, and 8/8 cleanup pass. This remains a correctness/mechanism proof, not performance or
token-speed evidence. The preceding `...T113534819091053Z` run compiled correctly but failed an
over-strict HLO variant classifier before any DB claim; its diagnostic is preserved and clean.

## Protected asynchronous stage remote copy

DB 448 / `greenfield_transport_pallas_pp8_20260806T115640677420842Z` at `29b040d` and DB 449 /
`greenfield_transport_pallas_paired_pp8_20260806T120235225511627Z` at `bfaec9d` close Section 7.2
item 8 with an evidence-backed rejection. The generic Pallas path transfers one
`bf16[1,6144]` payload; the production path starts asynchronous remote DMAs for both the residual
and `s32[1,2052]` compact metadata before either wait. Both use logical device addressing, exact
physical stage targets, deterministic cross-backend checksums, default-off dispatch with the
accepted `ppermute` fallback, 200 warmups, 1,000 profiler-free samples, and exact HLO guards.

The generic control / `ppermute` / Pallas p50 is `0.301991/0.330680/0.407570 ms`; control-subtracted
transport is `0.028690/0.105580 ms`, so Pallas is `3.68x` slower. The production paired control /
two-`ppermute` / paired-Pallas p50 is `0.380135/0.411980/0.488110 ms`; net is
`0.031845/0.107975 ms`, so paired Pallas is `3.39x` slower and `0.076130 ms` worse raw. HLO
`94725ae7...6084` has exactly eight generic communicating Pallas calls; HLO
`12789dff...fb3` has exactly eight paired calls, each with two enqueue DMAs before its waits.
Neither has a top-level collective or dead row. Correctness, DB/archive/remote `SUCCESS`, and 8/8
cleanup pass. Evidence therefore retains `ppermute`; the Pallas mechanism remains default-off.

Trace DB 450 / `greenfield_transport_trace_pallas_paired_pp8_20260806T120629031314193Z` at
`01ec9ed` supplies the fresh physical proof: eight XPlanes, 64 cores, 20 steps/core, exactly eight
paired remote-copy calls/step, zero collective-permute/all-reduce/all-gather calls, and a
`0.557786 ms` trace-contaminated step cycle excluded from performance claims. Summary SHA is
`ea2aa27f...286e`; all XPlane/host-record hashes, DB snapshot, approved archive/remote `SUCCESS`,
and authenticated 8/8 cleanup pass. Two earlier compile diagnostics failed closed before timing:
one exposed physical-versus-logical device addressing and one rejected an invalid `collective_id`
without a custom barrier. Both were preserved without DB claims and ended clean.

## Protected raw-FP8 stage linears and first body integration

DB 451 / `greenfield_fp8_rmsnorm_linear_20260806T122035458333587Z` at `ef709da` proves the
production M1/K6144/N2048 fused RMSNorm/raw-FP8 linear. FP32 square/mean/rsqrt remains outside the
call; BF16 rounding, norm-weight multiplication, tiled raw-FP8 decode, and matmul occur inside one
Pallas call without a normalized-activation or decoded-weight HBM overlay. The TPU output is
elementwise exact against the accepted reference. After 200 warmups, 1,000 profiler-free samples
give p50/p90/p95/p99 `0.526695/0.537813/0.541743/0.550947 ms`; compile is `0.238409 s`, peak HBM
is 316,252,672 bytes, and HLO `c1d87788...e61` contains the one required custom call. Interpreter
tail/dtype/validation tests, DB/archive/remote `SUCCESS`, and 8/8 cleanup pass.

Commit `a86ff8b` then binds a default-off `pallas_feature_linear` body backend: 78 q_a, 78 q_b,
78 kv_a, 78 attention-output, and three dense fused-SwiGLU replacements, plus the already accepted
75 each feature-MoE calls. Protected DB 452 /
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_trace0_20260806T122817958087998Z`
passes at the exact commit. Fleet p50/p99 is `4,876.099672/4,876.667969 ms`, down from DB 442's
`58,804.002894 ms` by `12.06x` (`91.71%`). All 315 stage-linear and 225 feature-MoE custom-call
counts are exact; no forbidden decoded weight/expert overlay exists. HLO `7e797b34...39dc` retains
the exact `219AG/294AR/16CP` structure. Compile max is `156.878 s`; maximum peak HBM is
26,135,755,776 bytes/chip. All eight hosts agree, strict HLO/metadata checks pass, DB 452 and the
approved archive/remote `SUCCESS` are sealed, and post-run census is 8/8 clean. This is a decisive
body-only result, not complete decoder latency or tok/s. Structured kv_b and DSA wq_b/wk remain on
the reference dequant path and are the next item-9 targets.

Fresh trace DB 453 /
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_trace2_20260806T124021528304196Z`
at `0a934c2` reproduces p50 `4,876.357135 ms` (within 0.006% of DB 452) and captures eight
XPlanes / 64 cores / two steps per core. The six remaining `reference/fp8.py:69-70` signatures
total `573.504900 ms` per average core: DSA-shaped dequants contribute `96.134925 ms`, while
structured kv_b contributes `477.369975 ms`. Eight serial stages predict `4,588.039200 ms`, or
`94.09%` of body wall. The apparent `3,917.040445 ms` collective time is therefore stage-idle
backpressure at the 16 compact permutes, not payload transfer latency. Device mean/max step is
`4,700.132040/4,874.983940 ms`; busy time is `4,525.764602 ms`. Summary/XPlane SHAs are
`a2f25c74...d935` / `04df4b66...09a7`; DB/archive/remote `SUCCESS` and 8/8 cleanup pass. This
evidence makes structured kv_b the first remaining item-9 target, followed by DSA wq_b/wk.

DB 454 / `greenfield_fp8_structured_kv_b_20260806T133007099354989Z` at `faca3e4` proves both
production raw-FP8 structured kv_b contractions without a runtime weight transpose or decoded
`[7168,512]` overlay. Query absorption and value projection are elementwise exact against the
independent host oracle; p50/p90/p95/p99 is `0.266695/0.276818/0.283333/0.295438 ms` and
`0.283515/0.295668/0.300083/0.319414 ms`, with paired-sum p50 `0.550236 ms`. HLO SHAs are
`5c805fbe...9d60` / `cc36fddb...f122`, each with exactly one named raw-weight call. Compile max is
`0.646 s`; peak HBM is 4,388,352 bytes. CPU aligned/64-row-offset interpreter tests, DB/archive/
remote `SUCCESS`, and 8/8 cleanup pass. The preceding `16abcd9` diagnostic failed before timing
on a TPU metadata BlockSpec alignment requirement, was fixed with compact 8x128 scale staging, and
ended 8/8 clean.

Protected body DB 455 /
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_trace0_20260806T134131684536168Z`
at `51a5e75` integrates 78 query-absorption and 78 value calls. Fleet p50/p99 is
`1,063.484099/1,064.161217 ms`, improving DB 452 by `4.585x` (`78.19%`) and DB 442 by `55.294x`
(`98.19%`). All 756 required Pallas calls are exact, the decoded kv_b overlay is absent, HLO
`2bae55e0...6c0f` retains exact `219AG/294AR/16CP`, compile max is `178.329 s`, and peak HBM is
26,138,384,384 bytes/chip. Fleet agreement, DB/archive/remote `SUCCESS`, and 8/8 cleanup pass.
This is still body-only, not complete token latency/tok/s. The immediately preceding exact-code
run produced the same ~1.064-second wall but was rejected because the outer harness duplicated the
old kernel-count dictionary; it has no DB claim and ended clean. DSA wq_b/wk is now the sole
remaining whole-matrix FP8 dequant path in item 9.

## Protected raw-FP8 DSA and 293 ms body

Commit `652b30c` adds an FP32-output raw-FP8 block matmul and binds DSA wq_b/wk without complete
BF16/F32 decoded weights. Standalone protected DB 456/457 prove production wq_b/wk p50
`0.236741/0.216940 ms`, max error below `1.67e-6`, exactly one named call each, no decoded overlay,
DB/archive/remote `SUCCESS`, and 8/8 cleanup. Commit `8ef9304` then preserves those DSA tensors as
U8 in the bounded loader and integrates them into Gate C. Protected DB 458 passes dense, exact
device-score DSA set/order, IndexShare state/cache integrity, strict two-call/local-only HLO, fresh
trace, DB/archive, and cleanup. Loader dequants fall from 44 to 36 with zero host dequant/reshard.

The first body attempt at `8ef9304` failed closed before timing because its harness expected 78
DSA projection pairs. The exact architecture has only 21 full IndexShare producer layers; the
other 57 reuse compact state. Commit `b9f9ed8` binds that count. Protected DB 459 then passes at
p50/p99 `293.232988/293.715581 ms`, a `3.627x`/`72.43%` improvement over DB 455 and
`200.537x`/`99.501%` over DB 442. Compile max is `176.973 s`, peak HBM `26,137,713,664` bytes/chip,
HLO `4f59da5a...e90` contains exactly 738 raw-FP8 calls and `219AG/294 physical AR/16CP`, no
decoded weight overlay or full-pod layer group. DB/archive/remote `SUCCESS` and 8/8 cleanup pass.
This remains transformer body only; `3.410 body steps/s` is not answer speed or a tok/s claim.

Fresh trace DB 460 reproduces `293.011417 ms` at the same code/HLO with eight XPlanes, 64 cores,
and two steps/core. Device mean/max step is `281.534/292.641 ms`. Actual local all-reduce work is
only `0.207 ms` per average core; the `234.422 ms` collective category is stage backpressure at
the two compact permute chains. Pallas custom calls total `26.671 ms/core` (`213.367 ms` serial
stage estimate); feature MoE alone is `14.782 ms/core` (`118.259 ms` serial). Crucially, complete
U8-to-F8 input formatting fusions consume `5.745 ms/core`, approximately `45.959 ms` serialized.
XPlane/summary SHAs are `7f1082ba...aca` / `5d055bd3...a08`; DB/archive/remote `SUCCESS` and 8/8
cleanup pass. This measurement, not guesswork, selects whole-table FP8 formatting as exact next.

Commit `b4bff66` moves every complete common/structured/shared U8-to-FP8 reinterpretation inside
the resident Pallas VMEM tile and makes the HLO contracts reject complete F8 operands. Focused
CPU/interpreter/HLO coverage is 44/44. Protected DB 461 proves M8/K6144/N2048 elementwise exact,
one direct `u8[2048,6144]` call/no formatting overlay, and p50 `0.406895 ms` versus DB 422's
`0.520605` (`-21.84%`). DB 462 proves structured q/value exact at p50
`0.201080/0.224360 ms`, paired `0.425840` versus DB 454's `0.550236` (`-22.61%`). DB 463 proves
shared up/gate exact with two direct U8 operands at p50 `0.566560 ms` versus DB 423's `0.815435`
(`-30.52%`). All three have strict HLO, DB/archive/remote `SUCCESS`, and 8/8 cleanup.

Protected Gate C DB 464 passes dense/full-DSA/IndexShare correctness, exact selected state/cache,
local HLO, fresh trace, and cleanup at the same pin. Protected body DB 465 then reaches p50/p99
`244.431673/245.138700 ms`, improving DB 459 by `16.64%` (`1.200x`). Compile max is `173.169 s`,
peak HBM `26,130,830,336` bytes/chip, and HLO `a9072597...e86` retains 738 exact Pallas calls and
`219AG/294 physical AR/16CP` while all eight complete F8 weight shapes are absent. This body-only
rate is `4.091 steps/s`; it is not complete answer speed and remains above the `<=200 ms` gate.

Fresh trace DB 466 reproduces `244.812285 ms` with eight XPlanes/64 cores/two steps. Device
mean/max falls `281.534/292.641 -> 234.391/243.724 ms`; busy time falls `270.397 -> 225.161 ms`.
The old complete-weight bitcast category falls `5.745 -> 0 ms/core`, while formatting removal
reduces step cycle by `45.278 ms` and body wall by `48.801 ms`. Feature-MoE is unchanged at
`14.782 ms/core` (`118.256 ms` serialized) and is now the dominant real compute; attention output
is `4.385 ms/core` (`35.084 ms` serialized). Actual psum is only `0.198 ms/core`; compact permute
time is backpressure. XPlane/summary SHAs are `734f5f04...e3f` / `32bd4c1d...bc2`; DB/archive/
remote `SUCCESS` and 8/8 cleanup pass.

## Protected feature-MoE output-tile promotion

Commits `7223e16` and `54d97fe` add and strictly validate the default-off routed output-tile-256
challenger while retaining the exact tile-128 fallback. Same-commit protected one-layer DB 467/468
at `54d97fe` pass exact routes, bounded normal/concentrated tensors, raw-U8/no-overlay HLO, fresh
trace, DB/archive, and 8/8 cleanup. Tile 256 improves normal p50
`2.2184095 -> 2.1684095 ms` (`2.254%`) and concentrated p50
`2.219930 -> 2.1687195 ms` (`2.307%`); the selected call falls
`1.577027 -> 1.539488 ms`.

Commit `4d20f94` binds that choice through the complete decoder and exact HLO/provenance guards.
Identical-condition protected body baseline DB 469 /
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_trace0_20260806T155923343193548Z`
records p50/p99 `244.679475/244.897979 ms` at tile 128. Candidate DB 470 /
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_trace2_20260806T160854944395218Z`
records `241.782586/242.768365 ms` at tile 256: `-2.896889 ms`, `-1.184%`, `1.01198x`.
Both retain exact `219AG/294 physical AR/16CP`, 738 raw-FP8 calls, identical plan/checkpoint/layout,
and ~26.131 GB peak HBM/chip. DB 470's fresh eight-host/64-core/two-step XPlane shows the intended
feature call at `14.429044 ms/core`, down from DB 466's `14.782 ms/core`; actual psum remains only
`0.198293 ms/core`, and attention output remains `4.385456 ms/core`. Summary/XPlane-summary SHAs
are `4a07e963...f328` / `dad56da3...d07a`; archive/remote `SUCCESS` and both 8/8 post-censuses pass.
Commit `897621c` promotes `pallas_feature_linear` plus tile 256 as the production decoder defaults;
reference mode still resolves to tile 128, and both choices remain explicit. Affected tests pass
15/15. This is a protected transformer-body win, not raw-token latency or answer tok/s.

## Rejected fused routed weighting/sum

Commits `89f2c64` and `5dda806` add a default-off candidate that moves exact BF16 route weighting
and the route-axis sum into the selected-expert Pallas call. The optimized selected result shrinks
from `bf16[8,8,6144]` to `bf16[8,6144]`, the old routed HBM result is strictly rejected, and the
fallback retains its separate two-prefetch signature. CPU/interpreter/HLO coverage passes 26/26.
The first protected diagnostic at `89f2c64` failed closed before timing because TPU scalar memory
forbids vector loads; its evidence is preserved and its failure-exit census is 8/8 clean. Commit
`5dda806` replaces that load with eight scalar loads and explicit BF16 association.

Same-commit protected candidate DB 472 /
`greenfield_real_layer_pp8_pallas_feature_ot256_wsum_20260806T164727316423011Z` and baseline DB 473 /
`greenfield_real_layer_pp8_pallas_feature_ot256_20260806T164837422574502Z` both pass exact routes,
bounded normal/concentrated output, strict local HLO, fresh trace, DB/archive/remote `SUCCESS`, and
8/8 cleanup. Candidate versus baseline normal p50 is `2.168395/2.164785 ms` (`+0.1668%`); concentrated
is `2.167525/2.1694595 ms` (`-0.0892%`). The candidate selected call is `1.541705 ms/step` versus
`1.539492` (`+0.1437%`), while maximum peak HBM is `2,430,519,808` versus `2,430,589,440` bytes.
Candidate/baseline HLO SHAs are `853ec297...b638` / `36fe8fb4...314d`; summary SHAs are
`6c98c90a...e6ab` / `7a1fcd5d...1762`. The structural elimination is real but performance-neutral;
reject promotion, retain default-off, and do not spend full-body compiles on it. The next measured
body bottleneck is attention output (`4.385456 ms/core`, approximately `35.084 ms` serialized).

## Rejected wide attention-output tile

Protected DB 470 attributes `4.385456 ms/core` (`35.084 ms` serialized) to 78 raw-FP8
attention-output projections, exact production shape `M8xK4096xN6144`. Commit `98b32a4` adds a
bounded standalone tile-256 challenger with distinct per-128-row FP8 scale application, an exact
`_ot256` fingerprint, odd/tail interpreter tests, and a protected harness; tile 128 remains the
runtime default. Local kernel/HLO coverage passes 31/31.

Same-commit protected baseline DB 474 /
`greenfield_fp8_attention_output_20260806T170743299244891Z` and candidate DB 475 /
`greenfield_fp8_attention_output_ot256_20260806T170834500879132Z` are both elementwise exact,
contain one direct-U8 custom call and no decoded weight overlay, have approved archives/remote
`SUCCESS`, and end with 8/8 cleanup. Candidate versus baseline p50 is `0.636600/0.6358845 ms`
(`+0.1125%`); p99 is `0.660573/0.663005 ms`. Candidate/baseline HLO SHAs are
`6518417e...2bad` / `19ec31f9...4718`; summary SHAs are `8adf20ed...2f53` /
`4d2457b2...7631`. Reject decoder integration: doubling the output tile is neutral. The next
structural candidate is one Pallas call for structured `kv_b` value projection plus attention
output projection, keeping the intermediate value states in VMEM.

## Rejected structured-value/attention-output fusion

Commit `7421e57` implements that structural candidate as one default-off Pallas call. It performs
the exact 16-head structured raw-FP8 value projection, repacks the BF16 value state in VMEM, and
feeds the production `M8xK4096xN6144` raw-FP8 output projection without an HBM value-state result.
The interpreter comparison, validation failures, and affected HLO/kernel suite pass 32/32.

Protected same-process A/B DB 476 /
`greenfield_fp8_fused_attention_output_20260806T171842700453846Z` is elementwise exact and finite.
Candidate HLO `855d9b29...d453` has one exact direct-U8 call with `[7168,512]` and `[6144,4096]`
raw operands, one `[8,6144]` result, no old kernel boundary, no value-state HBM result, and no
decoded weight overlay. Baseline HLO `76840cda...4e24` has the expected two calls. Candidate versus
baseline p50 is `1.1736855/0.6975100 ms` (`+68.27%`); p99 is `1.198501/0.720133 ms`. Compile is
`0.510/0.562 s`; peak HBM is 29,378,560 bytes. Runner/summary SHAs are
`b1ef75c1...247a` / `593557d3...697`. DB/archive/remote `SUCCESS` and authenticated 8/8 cleanup
pass. Reject decoder integration: eliminating this HBM boundary loses substantially more to the
serialized nested pipeline than it saves. Retain the exact fallback and default-off mechanism.

## Protected feature-body attribution

DB 442 / `greenfield_short_decoder_compile_pp8_pallas_feature_trace2_20260806T092025101122999Z`
at `0cd5209` is the successful, sealed attribution run. One profiler-free sample records fleet-max
body p50 `58,804.002894 ms`; this remains transformer body only, not token latency or tok/s. The
fresh trace contains eight XPlanes / 64 cores / two steps per core. Mean device step is
`56,722.255839 ms`, with `54,643.549475 ms` busy per step.

XPlane assigns `47,294.061096 ms` (`86.55%` busy) to the 16 compact stage
`collective-permute` start/done regions and `7,318.308852 ms` (`13.39%`) to gather/scatter. This is
pipeline backpressure, not a 47-second transfer: only the active stage computes while the other
stages wait at the permutes, and all dequant gather signatures total `7,317.697973 ms` per average
core. Eight serial PP8 stages therefore predict `58,541.584 ms`, within 0.45% of body wall. The
largest callers are attention output (`3,353.665 ms/core`), shared q_a (`1,616.121`), q_b
(`1,077.740`), kv_b (`477.370`), and kv_a (`413.463`). Feature-MoE is only `14.784 ms/core` in the
trace. Thus whole-matrix reference FP8 dequantization, not ICI bandwidth or MoE, is the immediate
2K critical path.

HLO `7ef2b071...f59a` retains exact `219AG/294AR/16CP`, 75 of each feature-MoE Pallas kernel, local
four-chip layer groups, and no decoded expert overlay. Compile max is `169.123 s`; peak HBM is
`26,144,010,752` bytes/chip. Summary SHA is `0361d44e...64e1`, XPlane-summary SHA
`91a424fc...d17`, approved archive/remote `SUCCESS`, DB integrity linkage, and 8/8 cleanup pass.

## Rejected complete feature-body diagnostic

`greenfield_short_decoder_compile_pp8_pallas_feature_20260806T084346269707216Z` at `a8194cd`
produced eight complete host records after exact load, compile, HLO validation, metadata validation,
and 13 executions. Fleet-max body p50/p99 is `58,804.040/58,804.323 ms`; this is transformer body
only, not a token or tok/s result. HLO `64df6dc7...2ea0` has 79,861 instructions, 1,195,999 bundles,
389 overlays, exact `219AG/294AR/16CP`, 75 of each required feature-Pallas MoE kernel, and no decoded
expert overlay. Every gather/reduce uses only the eight four-chip PP8 groups. Peak HBM is
`26,144,010,752` bytes/chip; all metadata contracts pass. Final DB/SUCCESS sealing failed only
because the outer validator expected a device-dequant counter absent from the runtime loader schema;
the diagnostic remains rejected, archived without a DB row, and ended 8/8 clean. The loader now
emits that counter and a profiler-after-wall two-step fleet-XPlane mode is ready for exact attribution.

## Protected complete-token mechanism

Commit `61c93b4` adds the default-off recurrent PP8 token step: stage-0 sharded embedding, the full
78-layer body, exact final norm, four local LM-head shards, deterministic global top-1, and one tiny
rank-lane token return. Position/context increment on device and every recurrent output can feed the
next compiled call. Forced 32-device CPU execution twice recursively passes. Commit `0819b67` pins
the TPU lowering discovered by the first fail-closed compile: XLA converts the two four-element
top-1 gathers into exact local one-hot sums. The production contract requires one `bf16[4]` and one
`s32[4]` all-reduce over all eight PP8 groups, one `s32[1]` token permute over the exact stage ring,
and forbids full-vocabulary collectives. Commit `f7934d5` fixes the outer state validator to compare
DSA state with the just-attended position rather than the incremented next-context slot; runtime/HLO
coverage passes 26/26.

Protected DB 477 /
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_token_trace0_20260806T193421580006958Z`
passes at exact code `f7934d5`. Fleet p50/p99 complete-step wall is
`242.411736/242.906796 ms` over ten profiler-free recurrent samples after two warmups. This is
`4.125 synthetic-state steps/s`, not answer tok/s. All hosts generate the same in-vocabulary token;
the captured timed window is `[504,364,2934,1010,474,314,62045,438,10549,11]`, and recurrent
position/context/DSA state closes exactly at `13/14/13`. HLO `70fc29a6...ed3a` has exactly
`219AG/297 physical AR/17CP`, 315 logical reduction components, 738 raw-FP8 calls, only four-chip
layer/top-1 groups, no dead full-pod row, no decoded/formatted weight overlay, and no full-vocabulary
logits materialization. Compile max is `167.178 s`; maximum measured peak HBM is
`26,131,539,968` bytes/chip against `33,014,398,976`. Evidence hashes, DB snapshot/integrity,
approved archive/remote `SUCCESS`, and authenticated 8/8 post-census pass.

This closes only the complete-token execution mechanism. `raw_token_claim=false`, the initial state
is synthetic, and no XPlane was requested, so Gate D remains open. Exact next: create a provenance-
pinned real 2K prompt/prefill or captured final-layout recurrent state plus independent raw-token and
DSA oracle; run the complete decoder with exact tokens, DSA set/tie order, state/cache integrity,
fresh fleet XPlanes, wall/HBM, DB/archive, and cleanup; repeat at 8K. Only then optimize the measured
complete decoder below 200 ms and proceed to 128K/256K.

## Device-resident prefill and sealed 2K raw-token oracle

Commit `684be03` adds a correctness-first teacher-forced prefill executable around the complete
token step. One outer `lax.scan` carries residual, KV/index caches, position/context, and compact
metadata entirely on device across the whole prompt; there is no per-token host or per-stage
dispatch. Forced 32-device CPU execution passes exact cache/index/metadata writes. Its optimized
HLO contains one device loop and one static copy of the complete decoder collectives
(`58AG/17AR/17CP`), all local. This is a mechanism proof, not a production 128K prefill latency
claim; the 2,034-token reference prefill is intentionally sequential.

Commits `d9923a3` and `0047c9c` add, validate, and archive an independent append-only 2K token
oracle. Artifact `greenfield_short_context_oracle_20260806T202544155912103Z` pins accepted legacy
DB run 139 / item row 628 (`passkey_L2040_d0.25`, seed 283835, gold `110391`), source harness
`b8e891e`, source fork `f0c63c302`, current legacy repository pin `b3c25df...16d`, exact tokenizer
file hashes, 2,034 prompt IDs, and 20 raw output IDs. Prompt/output token-ID SHAs are
`ec693ddf...56c` / `37761c49...3f6f`; tensor SHA is `cc5bc455...ab9a`; manifest is
`f580c149...fe19`. Local inspection, file ledger `985bc6f9...959f`, remote-object ledger
`1672b527...3dc2`, local/remote `SUCCESS` SHA `07700db5...6eec`, and approved-bucket archive pass.
The first attempt `...T202503456286091Z` failed closed during remote verification because the local
gcloud schema calls the field `crc32c_hash`; it has no `SUCCESS` and no claim. Relevant CPU/runtime/
HLO regressions pass 38/38 when explicitly pinned to the CPU backend. Gate D remains open until the
real prompt produces exact raw tokens and all DSA/cache/trace/wall/HBM protections pass.

## All-event DSA observer and fresh-oracle capture

Commits `564145f` and `dce2588` extend the greenfield decoder observer and independent legacy-oracle
sealer from one event to every recurrent DSA producer event. The production decoder remains
observer-off. The compact oracle contract pins 14 recurrent steps × 21 producer layers, exact causal
selected sets, score tensors, lowest-position tie order, sentinel tails, source dump checksums, token
oracle, DB row, source fork/harness, and append-only archive provenance. Commit `6856f8e` adds a hard
eight-host disk-reserve preflight so a full capture cannot begin below 10 GiB free per host.

Protected capture `greenfield_short_context_dsa_oracle_20260806T212500Z` loaded the complete legacy
model but failed closed before engine construction/decode. The load checksum reported
`verified=1882 mismatches=0 skipped=312`; the independent state manifest then caught the known
canonical-streamer finite zero-fill on worker 3 at
`layers.10.self_attn.indexer.glm_dsa_adapted_wk` (expected sum `191463636`, observed `0`) and the
fused leaf (expected `239851472`, observed `48387836`). Six other hosts reached exact
`leaves=2455 combined=371110325`; worker 0 did not complete final verification after the peer
refusal. No DSA artifact or result was accepted. Failure diagnostics were preserved and stop/census
both prove 8/8 zero work.

The failure was a capture-wrapper omission, not a new decoder/kernel fault: it omitted the already
metal-validated PWAL OOB self-healer. Commit `3c45b47` fixes the methodology. Capture now requires
the approved `driftbench-dsv4-uc` read-only gcsfuse model mirror on every host, arms exact
`GLM_WK_OOB_DIR=/home/gianl/gcs-models/models/GLM-5.2-FP8` and
`GLM_WK_OOB_GOLDEN=/tmp/golden.json`, binds both values into the DB/sealed-oracle contract, and
accepts only eight independent worker-log checksum plus exact final-manifest proofs with no refusal.
Repair messages are retained when a strike occurs but are not required on a clean draw. Shellcheck,
syntax, affected CPU tests, and the matcher against the preserved failed draw pass; the matcher
correctly reports six exact hosts and refuses workers 0/3.

The self-healing retry `greenfield_short_context_dsa_oracle_20260806T221300Z` at source code
`f5e047a` completed the real legacy execution. Workers 0, 1, and 4 repaired the same layer-10 WK
zero-fill/fused-leaf corruption from the pinned read-only OOB mirror; all eight hosts then
independently passed `verified=1882 mismatches=0 skipped=312` and exact
`leaves=2455 combined=371110325`. DB run 480 / item row 1763 consumed the 2,034-token prompt,
generated 20 tokens, and returned exact passkey `110391` (`correct=True`). The capture emitted all
420 callback dumps (20 generated steps x 21 events) on the canonical callback replica. The strict
eight-host capture/cleanup gates passed, but the original sealer refused because legacy padded
scheduler rows contain stale selections. This was a sealer defect: the legacy verdict and explicit
`valid` mask define only row 0 as live; rows 1--31 are informational padding.

Commit `02f836e` fixes that defect without relaxing live-row correctness: format v2 excludes padded
rows by the pinned valid mask, records their non-sentinel counts for provenance, separately binds
the corrected sealer and immutable source-capture code hashes, adds realistic regression coverage,
and provides a fail-closed recovery wrapper. Validation passes 16/16. Recovery artifact
`greenfield_short_context_dsa_oracle_recovery_20260806T231905802593249Z` seals the 14 recurrent
steps at positions 2034--2047 x all 21 producers x 2,048 selected positions: 294 exact source dumps,
lowest-position tie order, sentinel tails, counts, scores, producer IDs, token-oracle linkage, DB
snapshot, and source evidence. Manifest is
`71224832652ce61024786d39d43dcbfdc6cde76bf2eff0350531b272f38f4f57`; evidence ledger is
`9c87cada...a966`; remote ledger `99f39557...e72cf` verifies 443 objects by byte count, CRC32C,
and generation under the approved bucket. Local/remote `SUCCESS`, clean pre/post 8/8 census, and
source failure-exit 8/8 census all pass. Capture and archival phases are complete.

Commit `2dc300c` completes the observer-only harness integration. It loads the exact manifest pin,
compiles a separate no-donation observer, replays all 14 recurrent steps from the preserved prefill
state before production, reconstructs every stage/full-indexer slot, and requires four-lane
replication plus exact 14 x 21 position order/count/producer/sentinel-tail equality. Observer tokens
must equal token-oracle IDs 1--14 while prefill supplies ID 0. Its HLO must match production's exact
local collective contract and contain no callback or input/output alias; production timing and trace
remain observer-off. The protected wrapper pins both oracles, requires the exact 2+10+2 window, and
can declare Gate D only after fleet agreement, cache/state, fresh XPlane, wall/HBM, DB/archive, and
8/8 cleanup pass. Refusal, runtime, syntax, ShellCheck, and HLO coverage passes 20/20.

First protected Gate D attempt
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_token_oracle_dsa_trace2_20260806T234842911424415Z`
at `d303186` loaded the fleet and compiled production, observer, and prefill executables, then failed
closed before first prefill execution. Production and observer HLO pass exact local
`219AG/297AR/17CP`; observer isolation passes. Prefill has the same counts, exact token ring/shapes,
one device loop, and no host transfer, but the validator accepted only the direct token-return
metadata path and rejected the scan-wrapped path
`jit(execute)/while/body/closed_call/shard_map/ppermute`. This is contract metadata, not a physical
lowering, token, DSA, or throughput failure. All HLO/log evidence is archived under the failed tag,
and authenticated failure-exit census is 8/8 clean.

Commit `ff582bd` narrowly allows the two observed exact source paths (direct recurrent and
scan-wrapped prefill) while retaining exact opcode, singleton shape, 32 ring pairs, local groups,
one-hot reductions, and collective counts. An unrelated source path still fails. The preserved real
TPU prefill HLO now passes and the decoder/harness regression passes 15/15.

Exact next: retry protected real-prompt 2K Gate D from clean `ff582bd` with complete-token,
token-oracle, DSA-oracle, and two-step trace modes enabled. If exact DSA or token comparison fails,
preserve and localize the first step/event/selected-offset mismatch; do not relax it. If it passes,
seal the real answer-token rate and optimize the measured recurrent path below 200 ms.
