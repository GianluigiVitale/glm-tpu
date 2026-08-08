# HANDOFF — topology-first greenfield rewrite

**Updated:** 2026-08-08 13:52 UTC

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

Second protected attempt
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_token_oracle_dsa_trace2_20260807T001553313414003Z`
at `34b7611` passed production/observer/prefill lowering, loaded the real checkpoint, executed the
2,034-token device prefill, and reached the first real observer step. It then failed closed because
the old harness required total selected-position order identity against the independent legacy TPU
program: event 0 offset 39 expected `970` and observed `1670`, with 40,075 order mismatches across
21 events. Counts, producer IDs, lane replication, padded slots, and next position all passed. No
production timing or answer tok/s was reached or claimed; failure-exit census is 8/8 clean.

The subsequent arithmetic/methodology audit found two distinct issues. First, total rank identity
between independent TPU score programs contradicted the binding device-score contract: exactness is
the canonical global top-k/set, lowest-position ties, and tails of the executing FP32 score row;
cross-program score tensors use bounded comparison and raw legacy total order is diagnostic only.
Second, the audit then classified q_a/kv_a LoRA RMSNorm epsilon `1e-5` as a defect and changed it
to `1e-6`. Later pinned-config/source audit (recorded at the end of this handoff) proves that
classification was wrong: GLM-5.2 requires `1e-5` for these RMSNorms. This paragraph preserves the
historical decision without treating it as the current numerical contract.

Commit `b406e3a` changed the LoRA epsilon throughout stage, layer, and Gate C paths and upgraded the
observer without changing the default-off production output surface. The separate callback-free
observer exports bit-exact executing FP32 selected scores and gates exact selected set/count,
`-1`/`-inf` tails, unique causal positions, and canonical executing-score order/lowest-position
ties. Production/observer output equality and production HLO identity pass.

Third protected attempt
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_token_oracle_dsa_trace2_20260807T011313172353535Z`
at `9281341` again passed load/compile/prefill and reached position 2,034. All 21 events passed exact
set/tails, executing-score order/ties, producer, replication, padding, and position. The added
position-aligned legacy score bound failed (`max/p99/mean 69.367/28.666/3.301`) before token/timing.
The raw tensor was written by JAX process 0 on physical worker 2, retrieved as 989,858-byte
`step_00_position_2034.npz` (SHA `3cfaa3b3...a53803`), and preserved. Failure-exit census is 8/8
clean; there is still no answer tok/s claim.

Per-event analysis proves correct layer mapping and depth accumulation, not a weight/layer swap.
Layer 0 matches the same-input bound (`max/p99/mean 0.028/0.025/0.010`, correlation ~1.0); later
hidden states diverge progressively under 78-layer topology/reduction reassociation, reaching the
global diagnostic above. The Gate C bounds apply only when both programs consume the same captured
hidden input. Applying them to different full-network hidden states was another harness scope bug;
the binding contract gates executing-device selection and exact raw tokens.

Commit `715870e` fixes that scope, adds an immediate exact prefill first-token gate, retains all
legacy score errors/order as diagnostics, validates and archives all 14 raw observer tensors from
the physical JAX-process-0 worker, bumps schema to 5, and fixes NumPy index JSON serialization.
Exact device set/tail/tie gates and all 15 token IDs remain mandatory. Relevant verification is 59
tests plus Python compile, Bash syntax, ShellCheck, and diff checks; the commit is pushed.

Fourth protected attempt
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_token_oracle_dsa_trace2_20260807T014648797562615Z`
at `0a42c93` passed the exact prefill first token (`220`) and every one of 14 x 21 executing-device
DSA event contracts at positions 2,034--2,047. Observer tokens matched the independent oracle for
the first ten recurrent steps, then diverged at observer offset 10: expected `16345`, observed
`12877`. The two later differing tokens are downstream of that changed autoregressive input. All
14 raw NPZs are preserved locally and remotely; the failure stopped before production warmup,
timing, or trace and ended with authenticated 8/8 zero work. This proves DSA routing is not the
remaining first-divergence cause; Gate D and answer tok/s remain unproven.

Commit `ffa3db7` adds a compact, separate, no-donation logit observer for that exact boundary. It
records the canonical global top 16 token IDs plus bit-exact FP32-cast logit payloads, exact
lowest-ID ties, expected-token rank, top-1/top-2 and top-1/expected margins, lane/sentinel health,
and independent hashes in each NPZ. It widens only the observer's existing score/id exchanges and
adds no collective; the HLO isolation gate subtracts only those two declared shapes and requires
every non-token collective shape/count to remain identical. The static observer flag leaves the
production sampling source path/output/HLO surface unchanged. Schema 6 and the fleet artifact
validator require the new tensor. Full local verification is 343 passed / 1 skipped plus Python
compile, Bash syntax, ShellCheck, and diff checks; the commit is pushed.

Fifth protected attempt
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_token_oracle_dsa_trace2_20260807T024019917849358Z`
at `4a3802e` completed the top-16 diagnostic. It again passed the exact prefill token and all
14 x 21 executing-device DSA events. At the first divergence, decode position 2,044, the expected
token `16345` is runner-up rather than missing: greenfield scores `12877=20.875` and
`16345=20.75`, an inverted margin of `0.125` (one BF16 ULP at this magnitude). The other first-ten
tokens remain exact; later mismatches are downstream. The run failed closed before production
timing and ended with authenticated 8/8 zero work.

Commits `627d0e5` and `8482243` add a separate legacy-oracle-only top-16 logprob capture with sealed
prompt/output identity, DB provenance, per-host checksum/state verification, append-only archive,
remote verification, and clean-fleet enforcement. They do not import the legacy execution path
into greenfield production. Full greenfield verification is 345 passed / 1 skipped plus static
checks; both commits are pushed.

Protected legacy capture
`greenfield_short_context_legacy_logprobs_20260807T032528485799884Z` succeeded at `8482243`, DB run
481 / item 1764. It reproduced all 15 expected output tokens. At position 2,044 legacy instead ranks
`16345` first at logprob `-0.808534` and `12877` second at `-1.058534`, a correct-token margin of
exactly `0.25`. The 16-way tensor/manifest is sealed under
`gs://driftbench-dsv4-uc/oracles/greenfield/glm52/short_context_logprobs/2k/greenfield_short_context_legacy_logprobs_20260807T032528485799884Z`
with manifest SHA `3e6b5864...34a62`; every host passed `1882` checksum leaves, the common 2,455-leaf
state hash `371110325`, and final zero-work census. This proves the first failure is a real
greenfield arithmetic/model-execution inversion, not sampling, tie-breaking, DSA selection,
corrupt weights, or a missing candidate. Gate D and answer tok/s remain unproven.

The leading concrete cause is now the feature-sharded MoE reduction boundary. Each chip computes
only a 512/2,048 down-projection slice, casts that partial to BF16, then local-psums; legacy owns a
complete expert and rounds only after the full 2,048 contraction. Exact next: add a default-off
diagnostic that retains each routed down partial in FP32 through the local-four reconstruction,
casts only the reconstructed complete expert to BF16, then applies deterministic owner masking and
the existing BF16 routed/shared combine. Prove its isolated kernel/reference/HLO contracts first,
then run one protected 2K challenger. If it restores raw tokens, replace the diagnostic overhead
with a final-layout complete-expert Pallas pack; if not, capture compact per-layer residuals on the
exact teacher-forced trajectory to locate the earliest divergence. Never relax the raw-token gate.

## FP32 routed-down reconstruction is now a protected real-TPU mechanism

Commits `2d60723`, `794ba2d`, and `aa8105c` implement the default-off diagnostic, its numerical and
four-device semantics, strict one-layer/decoder HLO contracts, and an opaque on-device
`greenfield_fp32_to_bf16_r8_h6144` Pallas conversion boundary. Full verification is 353 passed / 1
skipped. Two protected attempts failed closed before timing:
`...downf32_20260807T044842501378323Z` exposed both an incorrect one-layer singleton-shape
expectation and TPU XLA commuting the BF16 cast into the reduction; after correcting the shape,
`...downf32_20260807T045959921629248Z` proved `lax.optimization_barrier` still allowed the same
BF16-result reduction. Both artifacts are preserved and both failure-exit censuses are 8/8 clean.

DB 482 / `greenfield_real_layer_pp8_pallas_feature_ot256_downf32_20260807T051217371806033Z` at
`aa8105c` passes the protected PP8 real-MoE-layer gate. Optimized HLO
`2a01bb2d...9a26` contains exactly one local `f32[8,6144]` four-chip all-reduce feeding the exact
FP32-input/BF16-output Pallas boundary, plus the existing local `bf16[2,1,6144]` routed/shared
combine. It has four intended Pallas calls, no decoded weight overlay, no global collective, and
the contract passes. Normal/concentrated routes are exact; output max/p99/mean errors are at most
`0.03125/0.01171875/0.002444`. Profiler-free normal/concentrated p50 is
`2.197155/2.195815 ms`, only about 1.50%/1.21% slower than DB 473's same-OT256 BF16 path; p99 is
`2.258372/2.245028 ms`. Fresh XPlane records 8 cores, 20 steps/core, `2.246920 ms` cycle,
`1.802875 ms` busy, and two physical psums totaling `0.043176 ms/step`. Compile is `1.720 s`; peak
HBM is `2,430,771,200` bytes/chip. Summary SHA is `1aa45e40...3a8e`; DB snapshot, evidence ledger,
approved archive/remote `SUCCESS`, and authenticated 8/8 cleanup pass.

This proves only the numerical mechanism and local lowering, not Gate D or answer tok/s. Exact next:
run the protected complete 2K decoder/oracle/DSA challenger with FP32 reconstruction enabled. If all
15 raw tokens restore, replace the diagnostic boundary with a final-layout complete-expert Pallas
pack and remeasure. If token 2,044 still inverts, keep teacher forcing on the sealed trajectory and
capture compact per-layer residuals to locate the first arithmetic divergence. Never relax the raw
token, DSA, state/cache, HLO, wall/HBM, trace, DB/archive, or cleanup gates.

The first full challenger
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_downf32_token_oracle_dsa_trace2_20260807T051528775553030Z`
at `8198f95` loaded and compiled for about 17 minutes, then failed closed at the pre-execution HLO
gate. All other collective counts/shapes passed, but the contract predicted the 75 new results as
`f32[8,1,6144]`; the real complete TPU HLO lowers them as `f32[8,6144]`, identical to the sealed
one-layer lowering. No prefill, token, DSA, timing, or tok/s claim was reached. Diagnostics are
preserved and failure-exit cleanup is 8/8 clean. The exact next action is the same protected 2K
challenger with only that observed exact shape pinned; no arithmetic or acceptance gate is relaxed.

Commit `70e5792` pins that observed complete-decoder result shape. The second protected challenger
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_downf32_token_oracle_dsa_trace2_20260807T053429062047282Z`
then passed complete production/observer/prefill HLO, executed the real 2,034-token device prefill,
returned exact first token `220`, and passed all `14 x 21` executing-device DSA contracts. Its first
ten recurrent tokens are exact. At the first boundary, position 2,044, the FP32 reconstruction
changed the previous wrong margin (`12877=20.875`, `16345=20.75`) to an exact BF16-logit tie
(`12877=20.75`, `16345=20.75`); exact lowest-token-id tie order therefore still selects `12877`
instead of oracle token `16345`. The complete expected/observed recurrent sequences are
`[104550,101294,16,13,3155,537,10662,432,13,576,16345,374,6176,13]` and
`[104550,101294,16,13,3155,537,10662,432,13,576,12877,374,6303,13]`; the position-2,046 mismatch
is downstream of the changed autoregressive input. Legacy's sealed position-2,044 logprob margin
remains `+0.25` for `16345`, so the diagnostic removed one BF16 ULP of inversion but did not recover
the full arithmetic difference. The run failed closed before production warmup/timing/trace, makes
no tok/s claim, preserved all observer artifacts, and ended with authenticated 8/8 zero work.

Exact next: keep the sealed token trajectory teacher-forced and capture compact residuals at every
layer boundary in independent legacy-oracle and greenfield observer executables. Locate the first
divergent boundary before changing more arithmetic. The observer must remain default-off and
callback-free on device, production HLO/timing must stay isolated, artifacts must bind layer,
position, trajectory, code/model/checkpoint hashes, and no token/tie gate may be relaxed.

## Callback-free layer-boundary observer is ready for protected capture

Commit `42df9cf` implements the default-off greenfield half of that diagnostic. A separate
no-donation observer returns BF16 residuals for all 79 boundaries (input to layer 0 through output
of layer 77), with only the owning four-chip stage writing each boundary. Stage-transition
boundaries have two independent writers and must agree bitwise; four-lane replication and exact
zero nonwriters are also mandatory. The host canonicalizer fails closed on shape/dtype, missing or
duplicate layers/ranks, lane drift, cross-stage writer drift, or nonwriter data and binds the
canonical byte hash.

The 14-step observer replay remains on the sealed trajectory by replacing only the next token with
the independent oracle token after every diagnostic step. Actual output tokens and top-16 logits
are still recorded, so the known position-2,044 raw-token mismatch remains a hard failure and no
production timing can be claimed. The residual tensor is captured specifically for the computation
at position 2,044. Its device executable has no host callback; production remains observer-off.
The protected wrapper uploads and retrieves the residual bundle even on this expected fail-closed
exit. Verification is 358 passed / 1 skipped for the full greenfield suite, 25/25 focused after the
final validation hardening, plus Python compile, Bash syntax, ShellCheck, and diff checks.

Exact next command, only after a clean eight-host census, is the serialized protected challenger:

```bash
GLM_GREENFIELD_FEATURE_RECONSTRUCT_DOWN_FP32=1 \
GLM_GREENFIELD_COMPLETE_TOKEN_PATH=1 \
GLM_GREENFIELD_SHORT_CONTEXT_ORACLE=1 \
GLM_GREENFIELD_SHORT_CONTEXT_DSA_ORACLE=1 \
GLM_GREENFIELD_SHORT_DECODER_TRACE_STEPS=2 \
GLM_GREENFIELD_LAYER_RESIDUAL_OBSERVER=1 \
GLM_GREENFIELD_LAYER_RESIDUAL_POSITION=2044 \
bash scripts/greenfield/run_short_decoder_compile_pp8.sh
```

Accept this run only as a diagnostic: require the existing first ten exact recurrent tokens, the
same position-2,044 tie/mismatch, all 14 x 21 DSA contracts, exact residual writer/replication/hash
contract, preserved HLO/logs/artifacts, and authenticated 8/8 zero-work cleanup. Then add an
independent observer-only legacy capture at the pinned oracle source, compare corresponding
boundaries, and isolate the earliest divergent layer/operation before changing arithmetic.

## Greenfield position-2,044 layer boundaries captured on real TPU

Protected diagnostic
`greenfield_short_decoder_compile_pp8_pallas_feature_linear_ot256_downf32_token_oracle_dsa_residual_p2044_trace2_20260807T063203702756771Z`
at code `312f043` completed the intended capture and failed closed only at the unchanged raw-token
gate. Production, observer, and prefill HLO contracts pass; the observer isolation contract has no
callback/alias and exactly matches production collective counts. Prefill token `220`, all 14 x 21
DSA contracts, and the first ten recurrent tokens pass. Position 2,044 remains expected `16345`,
observed `12877`; with teacher forcing, positions 2,045--2,047 return exact `374,6176,13`, proving
the former later mismatch was downstream.

The artifact contains all 79 `[6144]` BF16 boundary bit rows at position 2,044. Four-lane stage
replication, all seven cross-stage duplicate writers, zero nonwriters, shape, in-memory BF16 dtype,
and finiteness pass. Canonical byte SHA is
`d14366c997dea49dfdd71740ba2d64690201fe794b3bd401145231d0f5656b57`; no mismatch lists are
populated. The three residual files, 14 DSA NPZs, production/observer/prefill HLO, and all eight
logs exist locally and at the approved result prefix; the failure-exit census is authenticated 8/8
zero work. No production timing, DB row, Gate D, or tok/s claim exists.

Serialization caveat discovered by independent readback: NumPy preserves the exact two bytes per
BF16 element and the canonical SHA, but writes the custom `ml_dtypes.bfloat16` NPZ field descriptor
as `void16`. The values remain losslessly readable as `uint16` BF16 bits, but the file is not
self-describing. Exact next: change future residual artifacts to an explicit little-endian
`uint16` BF16-bit field and add readback validation; do not rerun this expensive capture solely for
metadata because the protected raw bytes are exact and immutable. Then implement the independent,
observer-only legacy boundary capture at the pinned oracle source and compare the two trajectories.

## Independent legacy boundary observer is armed

Greenfield commit `e7e0dbd` fixes all future residual files to use the explicit portable
`residual_bfloat16_bits` little-endian uint16 field; the first protected artifact remains exactly
recoverable from its immutable `void16` bytes and is accepted by the compatibility reader only.

The isolated legacy-oracle branch `greenfield/legacy-residual-observer` is pushed at
`cf066ab3e29151153d3f930a4c5ecc20ba716a9f`, exactly one commit above sealed oracle
`b3c25df47ac98783912dc658878181ec0a8ae16d`. It registers default-off pre-hooks on all 78 decoder
layers and the final norm, returns a separate callback-free `[79,32,6144]` BF16 decode output, and
writes only addressable target-row shards as explicit little-endian BF16 bits. The accepted oracle
checkout is unchanged. Eight observer tests include the real TorchAX `functional_call`/JAX JIT
seam and StableHLO callback absence; two existing MTP output-pytree tests and 18 unaffected runner
tests pass. Three pre-existing `continue_decode` unit failures caused by a `None` mocked flight-
recorder field are unrelated and were not changed.

Greenfield commit `ee22501` adds fail-closed eight-process reconstruction, bitwise duplicate/coverage
checks, compatibility readback of the sealed greenfield artifact, per-boundary BF16 comparison,
first-divergence reporting, and the protected wrapper. Verification is 362 passed / 1 skipped with
`JAX_PLATFORMS=cpu`, plus Python compile, Bash syntax, ShellCheck, and diff checks. An earlier test
invocation accidentally inherited the local TPU backend; it was terminated, produced no evidence,
and left no libtpu holder before the complete CPU-only rerun.

Exact next command, only from a clean branch/fleet, is:

```bash
bash scripts/greenfield/run_capture_legacy_layer_residuals.sh
```

The wrapper materializes a pin-specific detached observer worktree on every host, proves raylet
environment/import/code fingerprints, reproduces all 15 sealed legacy tokens and position-2,044
top-16 margin, validates checksum/state on all hosts, retrieves exactly eight process files,
reconstructs every one of the 79 x 6,144 BF16 values, compares them to greenfield SHA
`d14366c9...f5656b57`, archives append-only evidence, and ends with authenticated 8/8 zero work.
It is diagnostic correctness evidence only: no timing, throughput, Gate D, or performance claim.

## All-boundary legacy draw rejected; isolated selected observers are pinned

Protected run
`greenfield_legacy_layer_residual_p2044_20260807T074001755663269Z` compiled and
executed the first legacy observer, but failed closed because the generated tokens no longer
matched the sealed prefix. It has no DB row, `SUCCESS`, accepted residual, Gate D, or performance
claim; its failure-exit census proves 8/8 zero work. The eight retrieved boundary files and their
comparison are retained only under explicit `failed_draw` names. They report boundary 0 equal and
boundaries 1--78 unequal, but cannot localize production arithmetic because making all 79 values
live changed the decode executable itself. A read-only CPU sensitivity projection through the real
final norm and decisive LM-head rows confirms the rejected final residual favors wrong token
`12877` over `16345` by `0.125`; sealed legacy instead favors `16345` by `0.25`.

Observer pin `15f9606000c4dfd50b52873a35c5458b1f9339ad` replaces that methodology on
the isolated `greenfield/legacy-residual-observer` branch, exactly two commits above sealed oracle
`b3c25df47ac98783912dc658878181ec0a8ae16d`. Production again has its historical output tree and
donated cache path. Three separate observer executables capture boundaries `1,77,78` before the
production focus step from the same pre-step cache; each uses production compiler options, never
donates, returns only the selected already-live hidden/residual components, and finishes before
production begins. Logical BF16 residual addition occurs only on the host after both programs.
Every boundary is accepted only if the observer final hidden row equals production bit-for-bit.
Each optimized observer HLO also writes an append-only SHA contract and rejects callbacks or a
top-level input/output alias.

Greenfield harness commit `cd98b07e94a2bac4497e0ed74302adf119d076bc` pins that observer,
requires 24 fleet-agreeing HLO contracts, validates selected-boundary coverage/replication, and
labels the first unequal selected boundary as coarse localization rather than the earliest model
boundary. Failed logprob draws now persist the exact expected/observed sequences and first mismatch
before raising. Observer verification is 10/10 plus 40/40 related regressions; the complete
greenfield suite is 365 passed / 1 skipped. Python compile, Bash syntax, ShellCheck, and diff checks
pass.

Exact next: after the documentation commit is pushed and a clean eight-host census passes, run
`bash scripts/greenfield/run_capture_legacy_layer_residuals.sh`. Accept only exact 15-token
production reproduction, sealed position-2,044 top-16 margin, 8/8 checksum/state, 24 callback-free
and alias-free fleet HLO contracts, bitwise observer/production final-hidden equality for all three
boundaries, exact selected-boundary reconstruction, approved append-only archive, DB linkage, and
8/8 zero-work cleanup. If 77->78 contains the dominant accepted error jump, instrument layer 77
sublayers next; otherwise refine the selected interval. Never infer the earliest divergence from a
sparse boundary set and never relax the raw-token gate.

## First selected-observer attempt rejected; deferred-cache fix pinned

Protected run
`greenfield_legacy_layer_residual_p2044_20260807T093610754508593Z` loaded and verified the full
model on all eight hosts, compiled production boundary-32 plus selected boundary `1,77,78`
executables, and compiled every production bucket through 2,048. It then failed closed before
generation during the deferred warmup pass: the observer task retained the initially queued
`bf16[8,16,32,128]` KV-cache object after the preceding production warmup donated that buffer.
JAX correctly refused it as deleted. There is no runner result, residual, DB row, `SUCCESS`, Gate D,
or arithmetic claim. The wrapper stopped the owned runtime and the failure census is 8/8 clean.

Observer pin `4284e8798d49168536927630274a985643debeb6`, three commits above the unchanged sealed oracle,
substitutes `runner.kv_caches` only when a deferred observer warmup dispatches. That value is the
valid output of the immediately preceding same-bucket production warmup; every other exact input
is retained, observer donation remains disabled, and observer cache outputs remain discarded. A
CPU regression donates and proves deletion of the originally queued JAX buffer, then proves the
observer succeeds through the refreshed cache. Observer/HLO tests pass 16/16; the broader runner
invocation still exposes the previously documented unrelated `padded_num_reqs=None` mock failure.

The protected wrapper now pins the new observer, detached runtime path, and exact oracle distance.
Exact next remains one serialized selected-boundary retry under the unchanged acceptance contract.

## Second selected-observer attempt rejected; warmup isolation is now fail-fast

Protected run
`greenfield_legacy_layer_residual_p2044_20260807T111359503822439Z` reached generation and reproduced
all 15 sealed legacy tokens. At position 2,044 it reproduced token `16345` first, `12877` second,
and the sealed `+0.25` margin. All eight hosts passed `1,882/0/312` checkpoint checks and the exact
`2,455`-leaf / `371110325` state manifest. The source runner wrote DB run 483 / item 1767, but the
outer protected validator correctly rejected the capture: every selected observer reported
`production_output_equal=[False, False, False]`, with about 24,000 differing BF16 hidden elements
and maximum error `0.2578125--0.260009765625` per host. No accepted residual, comparison, `SUCCESS`,
Gate D, or performance claim exists. Failure-exit cleanup is authenticated 8/8 `CENSUS_OK`.

The run also exposed an HLO-recorder bug: the compilation name contains launcher metadata after
`boundary <n>`, while the recorder regex required the number at end-of-string, so all expected HLO
files were absent and fleet integrity independently refused every host. Rejected-draw-only CPU
reconstruction shows boundary-77 mean error `0.265951`, boundary-78 mean error `0.312380`, and a
`+0.046429` jump across layer 77. This resembles the earlier rejected all-boundary hint but is not
accepted localization because the observer changed production arithmetic.

Observer pin `6239d0e80d0888ad7177384c03404d494fc544a1`, four commits above the unchanged sealed
oracle, makes isolation a precondition rather than a post-generation discovery. Before production
warmup donates the exact 32-row cache, it takes a blocked device copy of that cache and preserves the
production hidden output. Each selected non-donating observer warmup consumes that exact snapshot and
must reproduce the production BF16 hidden output bit-for-bit; any drift refuses during warmup before
larger buckets or generation. Tapped hidden/residual outputs retain their natural sharding while only
the final hidden output remains explicitly constrained. The HLO regex now accepts the real launcher
suffix. Focused observer tests pass 13/13, including real deletion of the donated original buffer and
warmup-drift refusal; the greenfield residual validator passes 4/4.

The greenfield wrapper pins this observer/runtime, requires the exact warmup-isolation success line,
requires `production_output_equal=[True, True, True]`, scans stdout and stderr for refusals, and still
requires one NPZ plus three callback-free/alias-free HLO contracts per host. Bash syntax, ShellCheck,
and diff checks pass. Exact next: commit/push this wrapper and evidence, prove a clean fleet census,
then run one serialized protected retry. If warmup isolation fails again, do not repeat the expensive
draw blindly; use the now-recorded HLO and pursue an opaque device tap or same-input layer-77 sublayer
equivalence without relaxing production-output equality.

## Third selected-observer attempt proves returned boundaries perturb arithmetic

Protected retry
`greenfield_legacy_layer_residual_p2044_20260807T123637962992026Z` ran greenfield pin
`d573c920fc02c86616fb469acbc2c8bd77d7e620`, observer pin
`6239d0e80d0888ad7177384c03404d494fc544a1`, and unchanged sealed oracle
`b3c25df47ac98783912dc658878181ec0a8ae16d`. All eight model streams passed the exact checkpoint
scan (`1,882` verified, zero mismatches, `312` skipped) and state manifest (`2,455` leaves,
combined `371110325`). All eight hosts also emitted fleet-identical callback-free, alias-free HLO
contracts for boundaries 1, 77, and 78. Their hashes are respectively
`6ef7a835...a0f2`, `68fe79cd...d4cf`, and `32f3eddf...2627`.

The first observer warmup then failed the newly exact production-output comparison with
`181,592` BF16 mismatches and maximum absolute error `0.140625`. Therefore merely returning one
naturally sharded selected boundary changes the compiled legacy arithmetic; no selected residual
is admissible, no NPZ was written, and this design must not be rerun. There is no DB result,
`SUCCESS`, Gate D, timing, or tok/s claim. Failure diagnostics are archived append-only under the
approved result prefix. The exit trap stopped all eight workers but its immediate census caught
two lingering Ray processes; the preserved recovery census
`census_recovery_20260807T_after_failure.txt` subsequently proves eight unique `CENSUS_OK` hosts
and is also archived to the same approved prefix.

Source audit found a separate honesty defect: `_precompile_backbone_text_only()` queued every
bucket through 2,048 before `_flush_compilations()` ran the advertised 32-row isolation check.
Observer commit `78a5fce88fc222feb69357e0bb411e6c9a9b944c` now flushes immediately when the selected observer
bucket is queued, before any larger token bucket. A regression pins the order
`queue:16, queue:32, flush, queue:64`; all 14 focused observer tests pass on CPU. This improves only
diagnostic failure latency and does not rehabilitate the numerically rejected observer.

Exact next: do not update the protected wrapper or launch another full checkpoint draw. Build the
smallest same-input layer-77 sublayer equivalence proof (or a genuinely opaque device-side tap)
whose production output remains bitwise unchanged. Compare attention residual, routed/shared MoE
update, and boundary-78 reconstruction under one common input/cache/DSA state, preserve reduction
association in the contract, and only then apply the smallest numerical correction before the
protected 2K Gate-D retry.

## Paired 8K oracles sealed; first PP8 8K draw rejected only by a linter collision

The current path supersedes the old residual-observer exact-next above. The accepted 2K Gate-D run
is DB run 484 / item 1768 at greenfield pin `095d7a1...`: exact tokens and all 294 DSA events,
topology-local `219` all-gathers / `372` all-reduces / `17` collective-permutes, peak HBM
`26,245,004,800` bytes, p50/p99 `244.091151/244.247375 ms`, and `4.096830 tok/s`. It proves Gate D
at 2K but fails the Gate-E `<=200 ms` and `>=4.5 wall tok/s` thresholds.

The paired current-runtime 8K token oracle is
`greenfield_short_context_oracle_8k_20260807T172307269147351Z`, manifest
`e4fbcbdbf0fc8b1969e2f82ee457ab1563db4a8b37d2dea2bc4d1e828a13acf2`, with 8,155 prompt tokens
and 20 generated tokens. The independently recovered protected 8K DSA oracle is
`greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z`, manifest
`f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da`. It contains exact positions
8,155--8,168 and 14 x 21 = 294 events, has an object-by-object verified local/remote ledger, exact
DB 485 / item 1769 provenance, five independent 8/8 stop/census sets, and immutable sealed logs.

Greenfield pin `b5591dd` added the isolated protected PP8 8K profile. Its first draw,
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_trace2_20260807T175648292491307Z`, loaded and compiled the production executable on all eight
hosts, then failed closed before any model execution because the HLO linter reported only
`decoder contains dead-row/full-pod live tensors`. It has no token/DSA/prefill execution, timing,
DB row, `SUCCESS`, Gate-D result, or performance claim. The failure census proves authenticated
8/8 zero work. Host logs and the exact optimized HLO are preserved locally and under the approved
result prefix.

The rejected HLO contains exactly 21 full-DSA scorer bodies. Each body has the pinned
head-query/key contraction, scale/clamp, head-weight multiply, and head reduction. Their legitimate
local score tensor is `f32[32,2048]`: 32 DSA indexer heads by the 8K/LP4 2,048-position local context
shard. This collided with both the 32-device dead-row sentinel and DSA selected width 2,048. The
accepted 2K HLO has the identical 21-body / 210-occurrence opcode fingerprint at `f32[32,512]`.

Greenfield commit `b7853e20ddf28eadcd323b2d8e4dce0e7fcfa0aa` is pushed. It models the 32 DSA heads explicitly
and admits the colliding shape only inside that exact six-operation scorer fingerprint with the
pinned contraction, head-weight broadcast/multiply, and final reduction. Unrelated f32 or BF16
`[32,2048]`, full-pod hidden rows, dead scorer-query rows, opcode/name drift, body-count drift, and
wrong dtypes remain fail-closed. Offline revalidation passes the accepted 2K decoder, 2K device-loop
prefill, isolated 2K DSA observer, and rejected 8K decoder; the 8K record reports 21/21 valid scorer
bodies, 210 admitted colliding occurrences, zero forbidden shapes, and the unchanged
`219/372/17` local collective contract. Focused runtime/HLO tests pass 53/53; the complete
CPU-only greenfield suite passes 380 with one skip.

Exact next: from a clean branch containing `b7853e2` and an authenticated idle fleet, run exactly
one serialized `bash scripts/greenfield/run_short_decoder_compile_pp8_8k.sh`. Accept only exact
paired token and 294-event DSA oracles, state/load/cache integrity, 8K HLO including 21/21 pinned
score bodies, fresh
eight-host XPlanes, profiler-free wall, per-chip HBM, DB/archive linkage, and authenticated 8/8
zero-work cleanup. Do not claim the first rejected draw and do not continue to 128K if 8K Gate D
fails.

## The 8K retry exposes real DSA drift; bounded layer-0 association probe is pinned

The one authorized retry,
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_trace2_20260807T184154192144771Z`,
passed production/observer/prefill HLO including all 21 exact scorer bodies and the unchanged
`219AG/372AR/17CP` local-only contract. It loaded and executed the real prompt and returned the exact
prefill first token `220`. It then failed the unrelaxed DSA oracle at position 8,155. Event 0/layer
0 has the exact selected set but a different order; event 1 already swaps eight positions, and the
later producer mismatches grow as high as 558. Total legacy-order mismatches are 41,550. The event-0
aligned score error is max/mean/p99 `0.029307/0.021944/0.026596`. No production timing, trace, DB
row, `SUCCESS`, Gate D, or token-rate claim exists. All eight logs are byte-identical at
`4c58289d...d5af`; authenticated failure cleanup is 8/8 `CENSUS_OK`.

Source-level comparison now identifies three real association deltas hidden by the vacuous 2K
selection (`2,035 < top_k=2,048`): sealed legacy precomputes DSA `wq_b/wk` as FP32 while greenfield
rounds their decoded tiles to BF16; legacy key LayerNorm divides by `sqrt` while greenfield
multiplies by `rsqrt`; and legacy forms prompt keys in M=2,048 chunks then scores M=32 rows over
512-key/DCP pages, whereas production is true one-row. The batch-32 geometry is permitted only in a
separate diagnostic oracle executable; the production challenger remains one row and its HLO gate
rejects dead rows.

Builder pin `c30b64d` produced append-only input
`greenfield_layer0_dsa_input_20260807T195410522988362Z`: only 37 unique real embedding rows and the
11 exact layer-0 tensors, 22,679,052 bytes, file SHA `8cc95cf9...daf7`, manifest
`577ec8a1...0619`. Probe pin `1ae70e2ea1882829328b18e60457c878b2f2d7db` independently implements
FP32/BF16 dequant and divide/rsqrt variants, exact M2,048/M32/page512/DCP8 geometry, and one-row XLA
and existing-Pallas scorers. Every output is compared against the sealed 2,048-position set/order
and aligned scores. Full CPU verification is 392 passed / 1 skipped; Python, Bash, ShellCheck, HLO
unit contracts, artifact readback, and diff checks pass.

Exact next: from a clean branch containing pin `1ae70e2`, acquire the global lease and run exactly
one serialized `bash scripts/greenfield/run_layer0_dsa_association_probe.sh`. It is bounded
diagnostic evidence, not decoder performance. Use its exact set/order matrix to choose the smallest
one-row correction; only after isolated CPU/HLO/TPU proof may one protected 8K Gate-D retry run. Do
not repeat the full 753B draw blindly and do not proceed to 128K while 8K DSA exactness fails.

## First bounded layer-0 probe stopped at an exact physical-layout HLO discrepancy

Bounded diagnostic `greenfield_layer0_dsa_association_20260807T201408349129630Z` at pin
`ae8eda19b35b67ff66fae73d78d71e9bc09e4745` acquired the serialized lease, passed the fresh
eight-host pre-census, compiled both real layer-0 state variants, and compiled the legacy scorer on
one local four-chip TPU host. It stopped before all comparisons because the initial scorer contract
required the logical intermediate `f32[32,32,512]`; optimized TPU HLO preserves the exact physical
transpose `f32[32,512,32]`. The entry/output shapes and both source contractions remain exact, and
the HLO contains no collective or host callback. Its SHA is
`1b1a28cb01842b6484b4a451d2a0c4276ce07b9ffaadb81f14e5c2668849a8bf`.

No comparison matrix, DB row, `SUCCESS`, timing, Gate-D, or arithmetic conclusion exists. The
append-only diagnostic is preserved locally and under the approved bucket failure prefix; its
failure-exit census is authenticated 8/8 clean. The narrow validator correction accepts only the
logical or this observed physical score-tile permutation and additionally requires the exact
einsum source markers. It does not wildcard dimensions, admit collectives, or relax the production
one-row dead-row refusal. Thirteen focused CPU tests pass, and read-only validation of the immutable
TPU HLO now records only `f32[32,512,32]` and passes.

Exact next: commit and push the narrow HLO correction, prove the branch/fleet clean, and run one
serialized bounded probe retry. If a later phase exposes another physical layout, preserve that
HLO and change the contract only for the exact observed lowering. Use only a completed exact
set/order matrix to choose a production correction; do not launch the full model first.

## First bounded matrix completes; legacy fused qkv-a association is the next isolated variable

Protected bounded run `greenfield_layer0_dsa_association_20260807T201857533092232Z` at
`7296d00873276192ced8a122f4b3b17d59a63f64` is accepted as DB run 486 and archived at the approved
result prefix with local/remote `SUCCESS`, exact evidence checksums, and authenticated 8/8 pre/post
zero-work censuses. All state/scorer HLO contracts pass; the one-row XLA scorer has no diagnostic
M=32 rows or collectives and is elementwise identical to the reconstructed M32/page512/DCP8 scorer.
The existing Pallas scorer differs from that reconstruction in all 8,156 scores, with max/mean
absolute error `0.011721/0.004481`.

None of the first six variants restores sealed order. All preserve the exact 2,048-position set.
FP32-versus-BF16 `wq_b/wk` and divide-versus-rsqrt key normalization have indistinguishable
selection metrics: 1,640 order mismatches and mean signed score error about `+0.0260275`. The
one-row XLA path has the same result, while Pallas reduces order mismatches to 1,505 and mean signed
error to `+0.0222018` but remains inexact. This rules out those associations as sufficient fixes;
it does not authorize a decoder change or performance claim.

Source audit found the missing upstream association: sealed legacy obtains `q_c` from one live
BF16 fused `q_a_proj + kv_a_proj_with_mqa` matmul of output width 2,624, then slices the leading
2,048 values. The first probe and greenfield compute `q_a` alone at width 2,048. Builder pin
`c9d0382` created append-only v2 real input
`greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z`, adding only the exact 576-row
companion FP8 tensor and scales. Its tensor file is 26,219,180 bytes, SHA
`be643e33...d7f9`, with internal manifest `574f3553...73141`; the v2 inspector also retains exact
read compatibility with the sealed v1 artifact.

Exact next: finish the bounded fused-width variant, requiring the full `bf16[32,2624]` result and
its 576-value companion to remain live in HLO, then run one serialized bounded probe. Compare its
M32, one-row XLA, and one-row Pallas set/order/scores. Do not alter or reload the full decoder unless
this isolated association supplies exact evidence.

## Predecoded fused width is rejected; raw-FP8 TP32-local projection is isolated next

Protected bounded run
`greenfield_layer0_dsa_association_20260807T203120669202672Z` at
`07d89f003acdd236f665f270d30656c3bce0d6be` completed as DB run 487, has local/remote `SUCCESS`,
exact evidence checksums, the approved-bucket archive, and authenticated 8/8 pre/post zero-work
censuses. Its fused state HLO keeps the full logical `bf16[32,2624]` result and live
`bf16[32,576]` companion, contains no collective/callback, and hashes to
`3753d5465f59abf61e559e75c3f0bb8cdd78972e6b94a28db4194e2a4bee60a7`.

The predecoded BF16 fused-width hypothesis is rejected. It changes all 4,096 query entries but
worsens the reconstructed legacy scorer to 1,703 order mismatches and one selected-set swap; its
one-row XLA result is identical to that M32/pagewise reconstruction. The fused-state Pallas scorer
keeps the set but still misses 1,523 order slots. This is bounded diagnostic evidence only: no
decoder, Gate-D, timing, or token-rate claim.

The next source audit found the exact association the prior fused probe still omitted. The sealed
oracle sets `DISABLE_WEIGHT_REQUANTIZATION=1`; its FP8 linear method retains
`fused_qkv_a_proj.weight` as raw `float8_e4m3fn[6144,2624]` with separate
`f32[48,2624]` scales (also recorded directly by the accepted legacy state-hash log). Although the
vLLM layer declares `disable_tp=True`, the TPU linear adapter classifies the object as a
`MergedColumnParallelLinear`, assigns output sharding over the 32-way `ATTN_HEAD=model` mesh, and
reorders each fused part independently. The actual shard-map body therefore consumes 64 q-a plus
18 kv-a columns and executes a local width-82 dot; the prior probe passed a predecoded logical
width-2,624 BF16 matrix to one device.

The bounded challenger now independently reconstructs both exact runtime layouts from the sealed
raw bits/scales, keeps raw FP8 and FP32 scale operands live, executes global-N2624 and mapped
TP32-local-N82 variants, and compares their M32, one-row XLA, and one-row Pallas results. The HLO
contracts require exact source/packed shapes, named associations, live companion output, and zero
collectives/dead one-row batches. The immutable real artifact reconstructs the exact sealed state
byte sums `2448103424` (FP8 weight) and `53100864` (FP32 scale), with local shapes
`[32,6144,82]` and `[32,48,82]`; the protected runner fails closed on any identity drift. Focused
CPU tests and offline full-geometry CPU HLO validation pass. Exact next: commit/push this
bounded-only change, then run one serialized probe. Only an exact set and order may authorize a
production correction and protected 8K retry.

## Raw-FP8 fused-QKV association rejected; pinned reuse audit now precedes implementation

Bounded protected run `greenfield_layer0_dsa_association_20260807T205946537394555Z` at
`05d991506ad7717f5e7c68354937a2327887dc5b` completed as DB 488 with checksum-valid local/remote
`SUCCESS`, approved-bucket archive, and authenticated 8/8 clean pre/post censuses. The runtime pack
keeps raw FP8, FP32 scales, and the exact TP32-local `bf16[32,32,82]` result live in HLO with no
collective or callback. It reconstructs the sealed state byte sums exactly.

The result rejects raw-FP8 fused projection association as the missing 8K correction. TP32-local
XLA has 1,703 order mismatches, one selected-set swap, and max/mean score error
`0.0340824/0.0259581`; global XLA has 1,650 mismatches, one set swap, and mean error `0.0261870`.
Pallas preserves the set but misses 1,523 order slots with mean error `0.0221107`. No decoder was
changed or loaded, and there is no Gate-D, timing, or throughput claim.

Before writing another component, the repository/worktree sweep is now a durable input:
`docs/greenfield/REUSE_INVENTORY.md` plus `configs/greenfield-reuse-inventory.json` classify the
existing GLM-TPU, legacy, moe-tpu, vLLM-reference, artifact, protection, parity, kernel, WS32 and
MTP work as direct reuse, adapted, oracle-only, rejected, or research candidates. `AGENTS.md` and
the README require consultation, and a unit test rejects any `tpu_inference` or `vllm` import below
`glm_tpu/greenfield`. The legacy model execution boundary remains unchanged.

Exact next: reuse the pinned legacy linear-sharding and RMSNorm semantics to isolate the actual
distributed q-a norm association. The fused runtime projects q-a as 32 physical 64-column shards,
then normalizes the logical 2,048 values; the prior bounded probes reassembled q-a and applied an
ordinary full-width norm. Prove that boundary under the immutable real input with a one-row
challenger and exact HLO before one serialized bounded TPU probe. Do not run the complete 753B
decoder until an exact set/order result authorizes a correction.

## Distributed q-a RMSNorm challenger is implemented and CPU/HLO sealed

The pinned reuse audit has been converted into an independent bounded implementation without
importing legacy or vLLM execution. The diagnostic reproduces the real local-N82 raw-FP8 fused
projection, keeps its 18-column kv-a companion sharded, reduces the q-a FP32 square sum over the
exact physical 32-rank group, and all-gathers the normalized BF16 `32 x 64` shards before the
replicated 2,048-wide norm weight. The score phase replaces only q-a query state; sealed prompt
keys, head weights, FP32 wq-b and the already-proven companion remain unchanged.

The HLO contract admits this full-pod group only in the named diagnostic phase. It requires exactly
one global-id `f32[32]` all-reduce and one rank-3 `bf16[32,64,32]` all-gather over ranks 0--31 and
still forbids collective-permute, reduce-scatter, all-to-all, callbacks and outside compilation.
TPU BF16 is strict; the separately flagged CPU path may admit XLA's BF16-to-FP32 gather promotion.
The protected wrapper serializes both phases under the existing global lease, syncs the exact pin
to eight hosts, seals per-rank HLO/output identity and a checksum-bound q-residual artifact, then
runs the existing one-host exact set/order matrix. It retains DB/archive/SUCCESS and three
authenticated 8/8 census gates.

Focused forced-32 coverage passes 32/32. The complete CPU-only greenfield suite passes
412 with one skip and two pre-existing SWIG warnings in 332.82 seconds; Python compilation, Bash
syntax, ShellCheck, line-length and diff checks also pass. This is implementation evidence only:
there is no TPU result, arithmetic conclusion, decoder change, Gate-D retry, timing or throughput
claim yet.

Exact next: commit and push the clean bounded implementation, prove the fleet idle, then run exactly
one serialized `bash scripts/greenfield/run_layer0_dsa_association_probe.sh`. Only an exact sealed
2,048-position set and order may authorize the smallest one-row production correction and one
protected 8K Gate-D retry. A failed association is recorded and closed; it must not trigger a blind
full-model run.

## First distributed-norm launch rejects a worker/JAX identity assumption

Protected bounded launch
`greenfield_layer0_dsa_association_20260807T220018198481723Z` at `8f56ac6` passed the clean branch,
global lease, eight-host pre-census, exact-pin sync and approved-bucket input staging. All eight
processes initialized the 8 x 4 TPU runtime and then failed before compilation because the new
script required TPU-VM launch worker id to equal `jax.process_index()`. Existing accepted topology
code explicitly proves JAX topology-orders these identities independently. Therefore this is a
harness refusal, not an arithmetic or HLO result. There is no q residual, comparison matrix, DB row,
`SUCCESS`, decoder execution, Gate-D, timing or throughput claim.

Failure capture SHA is `e42fe05b...6155`; the authenticated failure-exit census SHA is
`9ce2e75a...de4f` and contains eight unique `CENSUS_OK` hosts. Partial evidence is preserved under
the approved diagnostic prefix. The correction now preserves both launch and JAX process identity,
requires each to form an independent 0--7 fleet bijection, and retains exact global device coverage
0--31. The focused 32-test suite and Python/Bash/ShellCheck/diff checks pass.

Exact next: commit/push this narrow identity correction and rerun the serialized bounded probe once.
Do not interpret the rejected launch as numerical evidence and do not load the full decoder first.

## Distributed arithmetic passes; artifact publication owner fails

The corrected protected launch
`greenfield_layer0_dsa_association_20260807T220248890303736Z` at `78283a5` executed the complete
32-chip distributed projection/norm phase successfully on all eight hosts. Launch-to-JAX ordering
is `0:1, 1:6, 2:0, 3:7, 4:2, 5:4, 6:3, 7:5`; local device sets cover exactly 0--31. Every host
reports fleet-identical HLO SHA `40c98625...4467` and q-residual SHA `59e65063...b60b`. The TPU HLO
has exactly one global-id `f32[32]` all-reduce and one `bf16[32,64,32]` all-gather over ranks 0--31,
with no contract violation. Runtime raw-FP8/FP32-scale byte sums remain exactly
`2448103424/53100864`.

The wrapper then failed before the one-host score matrix because the script wrote the artifact on
topology-ordered JAX process 0 (launch worker 2), while the shell attempted to upload the artifact
only from launch worker 0. Seven hosts exited successfully and launch worker 0 failed on the absent
local directory. Thus the distributed arithmetic is useful partial diagnostic evidence, but the
overall run has no comparison matrix, DB row, final archive `SUCCESS`, arithmetic conclusion,
decoder execution, Gate-D, timing or throughput claim. Failure capture SHA is
`97356689...0f5e`; authenticated 8/8 cleanup SHA is `acba8d07...0ca2` and partial evidence is
archived in the approved bucket.

The narrow correction binds the single artifact/HLO writer to launch worker 0, records both its
launch and JAX identities in the manifest, and makes the validator compare that producer record to
the fleet map. Focused tests remain 32/32 with Python/Bash/ShellCheck/diff checks green. Exact next:
commit/push and run one final bounded retry; it should reuse the already-proven arithmetic and reach
the score matrix. Do not run the full decoder first.

## DB489 rejects distributed q-a RMSNorm and moves the audit to scorer association

Protected bounded run `greenfield_layer0_dsa_association_20260807T220615983460791Z` at
`54edbf7412f278dcec84baeca61771b4f4fa605a` is accepted as DB run 489. Local and approved-bucket
`SUCCESS`, checksum-sealed evidence, the DB snapshot, and all three authenticated eight-host
zero-work censuses pass. The distributed phase covers physical devices 0--31 and retains exactly one
global-id `f32[32]` all-reduce plus one `bf16[32,64,32]` all-gather. Its HLO SHA is
`314956e1...0202`, q-residual SHA is `59e65063...b60b`, artifact-manifest SHA is
`046b4f0e...50e2`, summary SHA is `4d5baaac...1454`, and DB-snapshot SHA is
`02956625...739`.

The hypothesis improves but does not restore the sealed 8K order. Distributed-norm XLA keeps the
exact selected set but misses 1,501 order positions, with max/mean/p99 score error
`0.0304594/0.0228811/0.0287610`; one-row Pallas keeps the set but misses 1,249 order positions, with
`0.0268021/0.0191863/0.0232533`. No set swap remains. The Pallas score delta relative to the
pagewise XLA reconstruction is max/mean/p99 `0.00945234/0.00335265/0.00679396` across all 8,156
scores. This is diagnostic correctness evidence only; it authorizes no decoder correction,
Gate-D retry, latency, or throughput claim.

Within the reconstructed matrix, `legacy_bf16_divsqrt` matches the FP32/divsqrt baseline's query,
keys, and head weights elementwise, yet the baseline scores still miss 1,640 sealed order slots.
This rules out those isolated precision variants, but it is not a captured-internal-state proof:
the immutable artifact contains sealed selected positions/scores, not legacy query/key tensors.
The strongest concrete remaining mismatch is scorer geometry. The accepted 8K oracle uses the XLA
DCP scorer with a three-page local walk; the bounded reconstruction nests eight shards inside one
84-page program. Exact next: reproduce the accepted local XLA scorer shapes, BF16-cache boundary,
page width, and reduction association before revisiting upstream state. Do not launch the full 753B
decoder first.

## Exact local DCP XLA scorer is implementation-ready

The DB485 source/run audit is now an independent bounded greenfield implementation. It packs each
of the eight DCP stripes into the accepted local cache shape `bf16[24,512,128]`, uses the exact
`s32[32,3]` block table and per-shard `s32[32]` prefix length, and compiles the unchanged XLA body
from `f32[32,32,128]` query plus `f32[32,32]` head weights to `f32[32,1536]`. Each shard runs
through one compiled executable and only row zero is stitched back to global positions offline.
The 32-row bucket is diagnostic-only and cannot enter production `decode_batch1`.

The protected wrapper reuses the checksum-bound DB489 distributed-q-a artifact (manifest identity
`046b4f0e...50e2`, source code `54edbf7`) instead of repeating its already-closed 32-chip phase.
It still holds the global lease, requires authenticated eight-host pre/post zero-work censuses,
syncs the exact clean pin, runs the scorer on one four-chip host, and retains DB/archive/SUCCESS
linkage. The new DB revision is `bounded-real-layer0-v4-exact-local-xla-dcp-score`; correctness is
the exact sealed set-and-order result, never timing.

Focused CPU coverage passes 36/36. The complete CPU-only greenfield suite passes 416 with one skip
and two pre-existing SWIG warnings in 333.26 seconds. The exact CPU-compiled scorer HLO passes its
contract at 20,188 bytes with the required cache, metadata, result and score-tile shapes and no
collective/callback. Python compilation, Bash syntax, ShellCheck, JSON, line-length and diff checks
pass. An initial unpinned pytest invocation briefly acquired the local libtpu lock as PID 436988;
that exact process was terminated before model use and a subsequent authenticated census returned
eight unique `CENSUS_OK` hosts. This is implementation evidence only: there is no TPU arithmetic
result, decoder correction, Gate-D retry, latency or throughput claim yet.

Exact next: commit and push this clean bounded checkpoint, then run exactly one serialized
`bash scripts/greenfield/run_layer0_dsa_association_probe.sh`. If and only if an exact set and order
is restored, translate the association to the smallest true-one-row production implementation and
retry protected 8K Gate D once. Otherwise record the rejection and resume the source/state audit;
do not launch the complete decoder blindly.

## DB490 rejects scorer geometry; pinned model epsilon is the next discriminator

Protected bounded run `greenfield_layer0_dsa_association_20260807T224711202903102Z` at
`01ba8cd143b855b3e0c2bb917f61a9d75b4410f3` completed as DB 490 with local/remote `SUCCESS`, exact
evidence hashes, approved archive, and authenticated 8/8 pre/post cleanup. The exact local XLA
scorer HLO SHA is `e4e4d6cd...65b4e`; it has the required query/cache/weight/table/length/output
shapes, physical score tile `f32[32,512,32]`, and no collective or callback.

The local scorer is elementwise identical to the prior nested pagewise reconstruction across all
8,156 scores: zero mismatches and zero max/mean/p99 delta. It therefore does not restore the sealed
order. The baseline remains exact-set with 1,640 order mismatches; the DB489 distributed state
remains exact-set with 1,501 order mismatches. Every selected-score delta has positive signed mean
equal to mean absolute error, pointing to a systematic state scale rather than scorer association.
Runner/summary/evidence/DB-snapshot SHAs are `81499059...d737`, `d1f69178...62af`,
`67f60373...d662`, and `ce0447bf...d0f`. This is diagnostic-only and carries no decoder, Gate-D,
latency, or throughput claim.

The next source audit found a concrete numerical-contract error. Both accepted config copies
(`reference/hf-repo/config.json` and the model-streamer config) hash to
`22e49334...65ff` and pin `rms_norm_eps=1e-5`. Accepted vLLM pin `a30addc...d1c` passes that value
to both q_a and kv_a RMSNorm. Greenfield production and every prior bounded q-a probe instead used
`1e-6`; DB489 therefore did not test the accepted model norm. Key affine LayerNorm remains a
separate `1e-6` operation. The bounded diagnostic now pins the config SHA, executes a fresh exact
32-chip q-a norm with `1e-5`, and feeds it to the already-proven exact local scorer. Production is
unchanged until that matrix passes exact set and order.

The focused diagnostic suite passes 33/33. The complete CPU-only greenfield suite passes 417 with
one skip and the same two pre-existing SWIG warnings in 333.75 seconds; Bash syntax, ShellCheck,
Python compilation, JSON and diff checks pass. Exact next: commit/push the diagnostic-only
correction, then run one serialized protected layer-0 matrix. If it restores exact order, change production q_a/kv_a
LoRA epsilon to `1e-5`, correct the affected reference contracts, and run one protected 8K Gate-D
retry. Do not load the full decoder for the diagnostic itself.

## DB491 proves the model epsilon correction is large but not sufficient

Protected bounded run `greenfield_layer0_dsa_association_20260807T231449677046310Z` at exact pin
`ea879a24d196f61e238a22ee5bb393d3b6fa938d` completed as DB run 491 / item 1775. Local and
approved-bucket `SUCCESS`, DB integrity, exact evidence checksums, and authenticated pre,
distributed-post, and post eight-host zero-work censuses pass. The run pins model-config SHA
`22e49334...65ff`, q-a/input RMSNorm epsilon `1e-5`, and the separate key LayerNorm epsilon
`1e-6`. Its diagnostic HLO has exactly one global-ID `f32[32]` all-reduce and one
`bf16[32,64,32]` all-gather over devices 0--31, with no other collective or callback. HLO,
q-residual, internal artifact-manifest, runner, summary, evidence-list, and DB-snapshot SHAs are
`11804add...6d2`, `20f07a17...29d`, `7518e7ef...c16`, `cbc90643...7cf`,
`6b25c686...887`, `df2d0c7b...06a`, and `ecd963e6...8d7`.

The corrected epsilon is directionally decisive but does not restore exact order. The exact local
DCP XLA scorer preserves the sealed 2,048-position set with zero swaps, but has 1,408 order
mismatches. Score max/mean/signed/p99 error is
`0.00506306/0.00134283/+0.000276074/0.00403500`, correlation `0.999995592`. Relative to DB489's
wrong-epsilon distributed result, mean score error improves `0.0228811 -> 0.00134283` (about 17x)
and order mismatches improve `1,501 -> 1,408`. The pagewise and exact local DCP XLA results remain
elementwise identical. One-row Pallas preserves the set with 1,161 order mismatches and
max/mean/signed/p99 `0.00962067/0.00436344/-0.00436344/0.00780971`.

This is bounded diagnostic evidence only. It authorizes neither a Gate-D retry nor latency/token-
rate claims, and production q_a/kv_a defaults remain unchanged while the residual association is
unresolved. Exact next: audit the accepted RMSNorm reduction lowering and already-preserved HLO
before adding code. In particular, discriminate `local sum -> psum -> /2048`,
`local mean -> psum -> /32`, and `local sum/2048 -> psum`, plus the exact BF16 weight-multiply
boundary. Use an existing artifact or a bounded candidate matrix first; do not rerun the complete
8K decoder blindly.

## Source-level GSPMD mean is statically identical; repaired-wk precision is next

The accepted vLLM/Torchax source-level q-a RMSNorm expression has now been reproduced as one
logical-width JAX `mean` under explicit global `NamedSharding`, without importing either execution
path. On 32 forced CPU devices, the GSPMD result is bitwise identical to the already-measured
manual `local sum -> psum -> /2048` diagnostic for exact BF16 inputs, including the live fused
qkv-a companion. The full production geometry compiles with the same one all-reduce / one
all-gather contract and the optimized HLO retains the `1/2048` scaling after the reduction. This
rejects source spelling or automatic partitioning as a new arithmetic discriminator. No TPU run,
DB row, decoder change, Gate-D result, timing, or throughput claim follows.

The HLO parser now also expands current JAX replica syntax such as
`mesh['stage'=2,'local'=4] {'local'}` into exact physical rank groups, so the diagnostic remains
fail-closed under the newer printer. Focused forced-32/HLO coverage passes 14/14.

A source/state audit exposes the next concrete dtype boundary. The sealed layer-0 legacy leaf
`wk_weights_proj.weight` is BF16 `[160,6144]` with byte sum `241456714`; the accepted run explicitly
repairs both halves from the mirror. Its `wk` repair dequantizes raw FP8 with `out_dtype=w.dtype`
(BF16), and `_linear_weight_f32` then casts that already-rounded fused leaf to FP32. Greenfield's
current distributed replacement instead takes prompt keys from a state whose `wk` was dequantized
directly to FP32. Exact next: reuse immutable DB491 q-residual manifest
`7518e7ef...d8c16`, replace only its key state with the existing BF16-origin `wk` variant, and run
the exact local-DCP/one-row scorer matrix on one TPU host. Do not repeat the closed 32-chip q-a
phase and do not load the complete decoder unless this isolated state restores exact set/order.

## BF16-origin wk discriminator is implementation-ready without a repeated full-pod phase

The bounded matrix now creates a second DB491 replacement state from the same checksum-bound q-a
residual, FP32 `wq_b`, head weights, and fused companion, changing only `index_keys` produced from
BF16-origin `wk`. It refuses if those keys are elementwise unchanged. The state reuses the exact
same compiled replacement executable, and the existing exact local-DCP XLA, one-row XLA, and
one-row Pallas scorers compare it against the sealed 2,048-position set/order. The runner records
the accepted fused/adapted state shapes, dtypes, byte sums, source association, and candidate key
delta as explicit provenance.

The protected wrapper now pins DB491 artifact manifest
`7518e7eff0487f0dc02cd4b0ff1c3d0fc3ef9ca7c43dcded7d809120e30d8c16` and source
`ea879a24d196f61e238a22ee5bb393d3b6fa938d`, copies and revalidates its checksummed payload, and
runs only the one-host scorer matrix under the global lease. It retains exact eight-host code
sync, pre/post zero-work censuses, DB snapshot, approved archive, and terminal SUCCESS. Focused
script/state tests pass 5/5; Python compilation, Bash syntax, ShellCheck, and diff checks pass.

Exact next: commit/push this clean bounded candidate, prove an authenticated idle fleet, and run
exactly one serialized `bash scripts/greenfield/run_layer0_dsa_association_probe.sh`. Only exact
set and lowest-position order from the BF16-origin-wk local-DCP candidate may authorize the
production epsilon/wk correction and one protected 8K Gate-D retry. Otherwise record the rejection
and continue the source/state audit; never run the full decoder blindly.

## BF16-origin wk is closed; capture actual accepted DSA internals next

Protected launch `greenfield_layer0_dsa_association_20260807T235427432046987Z` at exact pin
`948f98152ba91da6246caee20eea642d177450e3` reused the immutable DB491 q-a artifact and did not
repeat the closed 32-chip phase. It reached the one-host state matrix and stopped at its intended
novelty guard: BF16-origin and direct-FP32-origin `wk` produced elementwise-identical prompt keys
after projection, key LayerNorm, RoPE and BF16 cache storage. The candidate cannot change scores
or order and is rejected. It has no runner summary, score matrix, DB row, `SUCCESS`, decoder,
Gate-D, latency or throughput claim. Partial evidence is archived under the approved diagnostic
prefix; the authenticated eight-host failure-exit census SHA is
`892250e022e819539c51ee39a2c281b4a32cb16cf13dd787932c459db6ab099d`.

Accepted state identities are now pinned rather than inferred. Layer-0 shape/dtype/byte sums are:
embedding BF16 `1668496656`, input norm BF16 `1006936`, adapted `weights_proj` FP32 `48158645`,
adapted `wk` FP32 `193298069`, adapted `wq_b` FP32 `3765880530`, fused `wk_weights_proj` BF16
`241456714`, and q-a norm BF16 `305844`. The embedding/input audit finds no missing multiplier or
TPU transformation: accepted vLLM returns the raw selected BF16 row, vocabulary sharding reduces
one nonzero owner, the TPU OOT layer delegates unchanged, and layer 0 clones the row then applies
the config `1e-5` input RMSNorm. Greenfield already mirrors those operations.

Exact next: inspect and reuse the already-proven `GLM_DSA_DUMP_TOPK` callback path to capture the
smallest actual accepted layer-0 event-0 query, head weights and key state without returning tensors
through the model output. Implement it default-off in a dedicated oracle-only legacy worktree,
require unchanged protected raw tokens and exact DSA output, then run one serialized capture. Do
not rerun the BF16-wk probe or the complete greenfield decoder before the captured state identifies
a source-backed production correction.

## Exact DSA internal observer and protected comparison path are sealed

Oracle-only legacy commit `83ff4a3576602ca844ea090550139a2ff00b0bb1` is a two-commit child of
accepted `b3c25df47ac98783912dc658878181ec0a8ae16d`; both branches are clean and pushed. Its
default-off callback observes only `model.layers.0.self_attn.attn` at position 8155 and writes the
live normalized hidden, q-a state, query, head weights and current post-RoPE FP32 key. Default-off
jaxpr identity and a paired two-device armed/unarmed wrapper passed in the legacy repository.

Greenfield commit `6af9092a733055bc9dc659bd10d0b88ab61a05e0` is clean and pushed. It reuses the
existing protected 8K DSA capture stack rather than creating a second launcher: exact detached
observer sync, accepted-checkout preservation, global lease, eight-host pin/env/state/armed/file
checks, exact token oracle, bitwise DB485 event-tensor comparison, DB snapshot, approved archive
and authenticated post-census remain mandatory. After the legacy runtime stops, one local TPU host
independently reconstructs the same five tensors from input manifest `574f3553...73141` and DB491
q-a manifest `7518e7ef...d8c16`; the comparison records the first divergent field and exact deltas.
Legacy execution is never imported. The real sealed artifacts reconstruct successfully on CPU;
the full CPU-only greenfield suite passes 423 with one expected skip and two existing SWIG
warnings. This is readiness evidence only: no TPU capture, DB row, Gate-D result or performance
claim exists yet.

The first protected attempt,
`greenfield_legacy_layer0_dsa_internals_20260808T005359078558816Z`, failed closed during warmup
tracing before serving. `jnp.asarray(normalized_hidden)` attempted NumPy conversion of a
`torch.bfloat16` torchax tensor backed by an outer-JIT tracer. No observer file, token, DB row,
comparison, final `SUCCESS`, or performance result exists. The wrapper stopped its owned runtime
and authenticated eight `CENSUS_OK` hosts. Eleven diagnostic objects / 2,123,660 bytes are remotely
verified under the approved bucket's `dsa_internals/8k/failed/` prefix.

Commit `83ff4a357` replaces all observer-operand conversions with the scorer's existing zero-copy
`as_jax` bridge and adds the missing outer-JIT/torchax regression. Nineteen focused forced-device
tests pass in 972.44 seconds, including the real DCP wrapper integration; Python compile and diff
checks pass. The greenfield wrapper is pinned to this two-commit observer and a new detached runtime
path.

Exact next: after committing the wrapper/docs and proving the fleet idle, run exactly one serialized
retry of `bash scripts/greenfield/run_capture_legacy_layer0_dsa_internals.sh`. Use the first
divergent field to derive a source-backed production correction. Do not rerun the rejected BF16-wk
matrix or the complete greenfield decoder until the captured accepted state identifies it.

## DB499 closes the layer-0 DSA query association; production seal is in progress

The corrected protected observer completed as source DB493. Authenticated recovery artifact
`greenfield_legacy_layer0_dsa_internals_recovery_20260808T030853085141329Z` preserves exact sealed
tokens/events and localizes the first independently reconstructed state divergence to `query`:
normalized hidden and q-a state are bitwise exact. The accepted query SHA is
`1ff2c2ec...12a`; adapted `wq_b` remains the exact FP32 `[4096,2048]` state with byte sum
`3765880530`, so the remaining issue is projection association rather than checkpoint identity.

Protected DB499 / `greenfield_layer0_dsa_query_association_20260808T034636385240375Z` at
`b41c3abceccde90c8ab2419f8b3f8f58b0b140b8` proves the raw-FP8 materialized PP8 candidate
bitwise: zero of 4,096 query values differ. Its HLO `ea5e5c56...6c89` contains only the complete
local `f32[1024,2048]` owner shard (8 MiB), never the global `f32[4096,2048]` table. DB495's
production Pallas MXU path has 4,096 mismatches/max `0.009170532`; DB496--DB498 streamed/unrolled
paths improve to 2,840/1,208 mismatches but are still rejected. DB499 has DB/archive/SUCCESS and
authenticated 8/8 cleanup; it is bounded correctness evidence, not a decoder or performance claim.

Production now keeps the fast feature-MoE and stage-linear Pallas runtime but routes only DSA
`wq_b` through that exact local FP32 boundary. q-a/kv-a model epsilon is restored to `1e-5`; the
separate key affine LayerNorm stays `1e-6`. The HLO contract requires at least one local owner
boundary per full-indexer layer, rejects any global query table, and removes exactly the 21 old
Pallas `wq_b` calls while preserving all other Pallas calls. Exact next: finish the clean test/doc
pin, then run one serialized protected 8K Gate-D decoder with tokens, DSA, HLO, HBM, traces, DB,
archive, and clean-fleet gates. Do not use the catastrophic all-reference feature runtime.

## Corrected 8K run localizes the next boundary; internal observer is ready

Protected 8K attempt
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_trace2_20260808T041407656729112Z`
at `f129e636499d31c1c82c47727523c6527b3ee979` ran from 04:14 to 06:28 UTC and failed closed in
the isolated DSA observer before timing. The first generated token is exactly `101252` with the
accepted top-1 rank and a 6.125 logit margin. Layer-0 event 0 retains the exact selected set, so
DB499's query correction is effective, but its score row still has max/mean/p99 error
`0.00597572/0.001217406/0.003442` and non-exact order. Layer-1 event 1 is the first sharp change:
2,041 positions remain common, seven swap, and the aligned common scores have
max/mean/signed error `0.27013397/0.18290268/-0.18290268`. Event 2 has nine swaps and later
events grow. The near-uniform negative event-1 shift identifies the next causal boundary as the
layer-0 output entering layer-1 normalization/q-a/key state, not the already-exact layer-0 query.

This run has no timing, DB row, final `SUCCESS`, Gate-D, or performance claim. Eight host logs are
byte-identical (`8066a555...4d63`); the DSA payload is `3e54254c...b053`, failure ledgers are
`18501db9...6051` and `ef7688d...c715`, and all eight authenticated failure censuses report
`CENSUS_OK`.

The next diagnostic is now isolated and default-off. The existing oracle-only zero-copy callback
targets any full-indexer producer layer, while the greenfield executable returns the already-live
normalized hidden, q-a state, query, head weights and current FP32 key for all 21 producer events.
It requires exact reproduction of the sealed failed-run DSA observation and the accepted layer-0
query before writing an append-only artifact. A new hash-pinned offline comparator aligns any
accepted producer capture with the corresponding greenfield event and reports the first divergent
field. Targeted runtime/validation coverage passes 46/46, affected layer-kernel coverage 2/2, and
the complete CPU-only greenfield suite passes 437 with one expected skip and two existing SWIG
warnings in 340.85 seconds. Bash syntax, ShellCheck, Python compilation, JSON and diff checks pass.
These are readiness facts only. Exact next: commit/push the clean pin, capture accepted layer 1
once, run the greenfield all-event observer once, compare the five states, and change only the
first source-backed divergent boundary before retrying protected 8K.

## All-event observer corrects the causal boundary; one-row q-a probe is ready

The serialized PP8 8K internal-observer run
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_dsa_internal_trace2_20260808T093721804151742Z`
at `380659a` completed its intended fail-closed diagnostic from 09:37--11:50 UTC. Its baseline DSA
payload is bitwise identical to the prior `f129e63` refusal, the first token remains exact, eight
host logs are byte-identical (`dabe2c49...7806`), and the authenticated failure census is 8/8
clean (`3b176396...dfbc`). There is no timing, DB row, `SUCCESS`, Gate-D, or performance claim.

The decisive layer-0 comparison corrects the earlier DB499 interpretation. Production normalized
hidden is bitwise exact, but q-a already differs in 494/2,048 BF16 values (max `0.015625`); all
4,096 query values then differ. DB499 therefore proves only the `wq_b` boundary when fed the
accepted q-a state. Layer 1 is downstream: normalized hidden differs in 3,974/6,144 values, q-a in
1,156/2,048, and query/head/key all differ. Comparison SHA is `1bc43a8e...9ad5`, tensor SHA
`79b813da...9054`; its seal `283e5e88...0d5` is verified under the approved parent-run
`diagnostic_comparison/layer1/` prefix.

The next bounded candidate reuses DB491's accepted source mechanism: 32 shard-major fused
q-a/kv-a output shards of `N=82` (`64+18`) followed by explicit q-a norm association. It
virtualizes those shards on one stage-local TPU with a true `[1,6144]` row, no collective, no host
callback, and no legacy `[32,...]` token bucket. The existing DB499 one-host protected wrapper now
accepts `GLM_GREENFIELD_DSA_ASSOCIATION_TARGET=q_a` and tests 12 projection/norm associations.
Focused CPU tests pass 23/23 including the independent forced-32 test; the complete greenfield
CPU suite passes 441 with one expected skip and two pre-existing SWIG warnings. Python, Bash,
ShellCheck, JSON and diff checks pass. These are readiness facts only. Exact next: commit/push,
prove the fleet idle, and run exactly one serialized bounded q-a matrix. Only a bitwise candidate
with passing one-row/local HLO may be integrated into production before another 8K attempt.

## DB501 rejects M1 dot/norm virtualization and exposes the next HLO discriminator

Protected bounded run
`greenfield_layer0_q_a_association_20260808T123220826430916Z` at exact pin `b4488076` completed as
DB run 501 / item 1784. Local and approved-bucket `SUCCESS`, DB snapshot, byte-for-byte critical
object verification, and authenticated eight-host pre/post zero-work censuses pass. Runner,
candidate tensor, evidence-list, remote-object, and SUCCESS SHAs are `70745455...52de`,
`f755bcb1...568b`, `86520324...aae4`, `69f3d576...6a86`, and `a5f1c67c...14f7`.

None of the 12 true-row/local candidates is exact. All three projection mappings and all four norm
associations collapse to the same BF16 result SHA `439a4d54...d553`: 376/2,048 mismatches, max
`0.0078125`, mean `0.0000967367`, signed mean `+0.0000043714`, p99 `0.001953125`. Every optimized
HLO has the required `[1,6144]` input, `[1,2048]` output, shard-major FP8 `[32,6144,82]` weights,
FP32 `[32,48,82]` scales, and no collective/callback or dead token-row shape. This closes ordinary
M1 dot mapping and norm reassociation; it does not authorize a production change or 8K retry.

The accepted DB491 physical local body lowers its `bf16[32,6144] x bf16[6144,82]` projection to an
XLA `convolution` with `dim_labels=bf_io->bf`. DB501's M1 dot is instead optimized to an explicit
FP32 multiply/reduce. The next bounded discriminator is therefore a directly expressed zero-spatial
convolution whose public input/output remain exactly `[1,6144] -> [1,82]`. It must retain that
convolution in optimized TPU HLO and the same no-collective/no-dead-row contract before one
serialized comparison. Do not emulate the accepted kernel by restoring 31 dead decode rows.

That discriminator is now implementation-ready. The independent reference dequantizes each raw
FP8/FP32-scale N82 shard to BF16 exactly as before, but calls `lax.conv_general_dilated` with zero
spatial dimensions and `NC x IO -> NC` labels on the one live row. Only the four new convolution
plus norm variants enter the v2 protected matrix; DB501's 12 rejected dot variants are not rerun.
The HLO gate additionally requires a physical `f32[1,82] convolution` with `bf_io->bf` labels.
Focused CPU, Python, Bash, ShellCheck, line-length and diff checks pass. Exact next: commit/push and
run one serialized v2 matrix; no full decoder is authorized first.

## DB502 proves the exact one-row q-a convolution; packed production integration is next

Protected bounded run
`greenfield_layer0_q_a_association_20260808T124434046623046Z` at exact pin `c230c11` completed as
DB run 502. All four direct-convolution candidates match the accepted layer-0 q-a BF16 state
bitwise: 0/2,048 mismatches and SHA `c9fbac05...c70c`. Each optimized TPU HLO contains the required
physical `f32[1,82] convolution` with `bf_io->bf`, accepts one external `[1,6144]` live row, and
contains shard-major FP8 `[32,6144,82]` weights plus FP32 `[32,48,82]` scales without a collective,
callback, or dead token-row shape. The fused 576-wide kv-a companion is identical for all four
norm associations (`cf288bc2...e790`). This is the first source-backed production correction.

Local and approved-bucket `SUCCESS`, DB snapshot, six byte-for-byte critical remote objects, and
authenticated 8/8 pre/post zero-work censuses pass. Runner, summary, candidate NPZ, evidence,
remote-object and SUCCESS SHAs are `2a77d75d...75c4`, `75de66b6...040b`, `d9b14bdd...f76e`,
`815cc6a3...8f59`, `6a1d78e8...9257`, and `de2e080d...dab`. DB502 is bounded arithmetic evidence:
it has no decoder, token, latency, XPlane, Gate-D, or Gate-E claim.

Exact next: integrate the direct one-row fused q-a/kv-a convolution behind a default-off greenfield
backend, consume its kv-a companion rather than projecting kv-a twice, and extend the final-layout
checkpoint manifest/packer/direct loader to store the N82 weight and expanded scale tensors.
Gate B is reopened for this derived final layout. Do not pack N82 inside every decode step and do
not rerun the full 8K decoder until the packed layout, production HLO, and exact layer path pass.

## Fused qkv-a production path and final-layout derivative are implementation-ready

Commit `0082bac0f74fa4cac631c8a3085576d3bd10e6ef` integrates DB502 behind the default-off
`fused_n82_convolution` backend. The decoder now loads only final-layout
`u8[32,6144,82]` weights and `f32[32,48,82]` expanded scales, emits one live-row zero-spatial
convolution per layer, reuses the fused 576-wide kv-a companion, and requires zero legacy q-a/kv-a
Pallas calls. Backend/layout mismatch, separate source state, dead `[32,6144]` rows, old weight or
scale shapes, and convolution-count drift all fail before execution.

The feature-expert and fused-attention transforms share one streaming pass over the already-sealed
base runtime artifact; no intermediate 834 GB checkpoint is created. Reconstructing the real
78-layer manifests preserves the historical separate layout hash `ba21c4ec...c9e` exactly. The
fused derivative has layout hash `523afb1d...cb4`, layout-manifest hash `8bd08068...6f9`, 32 files,
`834,369,271,808` payload bytes, and `26,074,039,744` runtime bytes/chip--only `5,997,312`
bytes/chip above the accepted feature layout. The additional bytes reconcile as expanded live
scale state plus declared padding; source ownership remains unchanged.

Focused arithmetic, layout, full artifact verifier, runtime, HLO, and production-shaped stage
coverage passes 57/57 in 135.25 seconds with forced CPU. Python compilation, Bash syntax,
ShellCheck, JSON and diff checks pass. This is implementation/layout reconstruction evidence only:
no fused production TPU layer, packed artifact, Gate-B reclosure, decoder token, HBM, latency, or
performance claim exists yet. Exact next is a bounded production-helper layer-0 TPU comparison
against DB502 with exact q-a and kv-a plus physical HLO; only then pack and verify the full fused
artifact before retrying protected 8K.
