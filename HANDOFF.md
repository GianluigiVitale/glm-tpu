# HANDOFF — topology-first greenfield rewrite

**Updated:** 2026-08-10 11:31 UTC

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

## DB503 seals the fused qkv-a production helper; protected packing is authorized

Protected DB 503 / item 1786,
`greenfield_layer0_qkv_a_production_20260808T140131250842069Z`, ran the actual integrated
`_project_attention_qkv_a` production selector at exact code
`f715039399957bbc15f9366a6d947f33861d3c47`. Its one-row final-layout inputs are
`u8[32,6144,82]` and `f32[32,48,82]`. The normalized q-a output is bitwise exact to the accepted
capture (0/2,048 mismatches, SHA `c9fbac05...c70c`), and the fused 576-wide kv-a companion is
bitwise exact to sealed DB502 (SHA `cf288bc2...e790`).

Optimized TPU HLO SHA `1eec1393...509c` contains exactly one physical
`f32[1,82] convolution ... bf_io->bf`, the required packed state and one live input row, with no
collective, callback, forbidden shape, or contract violation. SQLite integrity, DB linkage,
runner/NPZ/summary/HLO/evidence/DB-snapshot hashes, local/remote byte equality, approved archive,
and authenticated 8/8 pre/post zero-work censuses pass. Runner, NPZ, summary, evidence,
remote-object, DB-snapshot, and SUCCESS SHAs are `e7cd9fbb...d5e`, `5dff6bb9...2fcb`,
`be9192f0...91e`, `7d1396d9...2d61`, `a412b251...5272`, `4292997f...6a`, and
`5d458cb8...03e`.

DB503 is bounded arithmetic/HLO evidence, not checkpoint, decoder, HBM, latency, XPlane, Gate-D,
or Gate-E evidence. It closes the prerequisite for creating the protected fused 32-file runtime
derivative. Exact next: pack and verify that append-only artifact, prove direct final-layout load
and the full 78-convolution production HLO, then retry protected 8K once. Gate B remains reopened
until the fused artifact/direct-load proof passes.

## Fused final-layout artifact and DB504 reclose Gate B

Protected pack
`greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z` completed at exact pack pin
`7d5dfb9ded22f1f7d66801a9ea63eabaaf36b4b7`. Its 32 final-owner files contain
`834,369,271,808` payload bytes / 10,880 tensors and reconcile to source payload
`750,122,559,744` plus declared transforms/padding. Runtime manifest identity is
`12339490...699a`, layout `523afb1d...cb4`, semantic layout manifest `8bd08068...6f9`, and the
mounted verifier records `verified=true`. Checkpoint/result SUCCESS files match locally and
remotely at SHA `368ef308...5b24`; authenticated pre/post pack censuses are 8/8 clean.

Commit `c16b37f38afed80a25fa2b234e3a8a4129353227` exposes the existing loader's per-tensor device
round trip through the protected decoder and fails fleet finalization unless verified bytes equal
the complete loaded payload. Focused tests pass 38 with one expected skip; Python compilation,
Bash syntax, ShellCheck and diff checks pass.

Protected DB 504,
`greenfield_short_decoder_compile_pp8_2k_pallas_feature_linear_ot256_downf32_splitres_qkva_roundtrip_hlo_20260808T150900Z`,
then directly loaded the fused artifact on all eight hosts. Every host verifies exactly
`104,296,158,976` loaded and device-round-tripped bytes / 1,360 tensors, giving the exact complete
artifact totals. Runtime checkpoint reshards, host global concatenations, and host/device FP8
dequantizations are all zero. All 32 chips retain final ownership; measured maximum peak HBM is
`26,143,616,000` bytes/chip, leaving `6,870,797,312` bytes against the observed
`33,014,413,312` capacity.

Optimized HLO SHA `710942ec...d69c` passes every contract and contains exactly 78 physical
one-row `f32[1,82]` fused qkv-a convolutions, zero old q-a/kv-a linear calls, the exact selected
feature-MoE/stage-linear kernel counts, no forbidden shape/overlay, and only explicit four-chip
stage-local repeated groups plus the 16 intended transport permutes. Summary/HLO-contract/HLO/
DB-snapshot/evidence/SUCCESS SHAs are `ed342ece...ca2`, `1c1a8dc9...9375`,
`266b1ed9...499e`, `2820fd82...4681`, `63c6b494...6420`, and `3b2dec6d...2f5f`; critical
approved-bucket bytes match, SQLite integrity is `ok`, and authenticated pre/post censuses are
8/8 clean. This re-closes Gate B for the fused runtime.

DB504 is a body-only mechanism proof with one measured sample after one warmup and no token,
oracle, DSA, XPlane, Gate-D, or performance claim; its `774.192188 ms` sample must not be used as
latency evidence. The preceding `...T152000Z` launch used invalid `warmup=0`, so all hosts refused
before JAX initialization; it has no DB/final SUCCESS and ended with an 8/8 clean failure census.

Exact next: commit and push this evidence seal, then run one protected 8K Gate-D retry against
runtime manifest `12339490...699a` with warmup 2, iterations 10, trace 2, exact raw-token and DSA
oracles, FP32 routed-down reconstruction and split residual state. Device round-trip is disabled
for that performance run because DB504 already closed it. Stop on any token/DSA/HLO/HBM/wall or
cleanup failure; only a complete accepted run may change Gate D/E status.

## Fused 8K prefill refusal is source-explained; exact loop classifier is pinned

The authorized retry,
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_qkva_trace2_20260808T153500Z`,
ran at exact pin `3058dc8` from 15:32--15:51 UTC and stopped before prefill execution. The old
prefill linter counted every physical `while` and rejected `79` rather than `1`. No token, DSA
event, timed iteration, XPlane, DB row, final `SUCCESS`, Gate-D result, or performance result
exists. All eight logs are byte-identical (`93cb2c80...4725`) and the authenticated pre/failure
censuses each contain eight unique `CENSUS_OK` hosts.

The archived optimized HLO (`9f8c2a7d...964e`) makes the cause exact: 78 instructions have
metadata ending in `one_row_fused_qkv_a_n82_convolution/while` inside the outer prefill body, and
exactly one instruction has `op_name="jit(execute)/while"`. There are no other loops. The decoder,
DSA-observer, fused-convolution, local-collective, transport and dead-row contracts all pass; only
the unclassified total-count assumption failed. Direct approved-bucket hashes match locally for
the decoder contract/HLO (`a406a211...8cc0` / `bc4320f7...4602`), prefill contract/HLO
(`9b43281f...3c67` / `9f8c2a7d...964e`), censuses and log. SQLite remains `ok` and the run tag is
absent from every DB evidence table, as required for this refusal.

Commit `1f110133bc4411d6a3bcc1d2c69a8334f915a8fb` replaces the count exemption with a fail-closed
identity classifier. The fused backend requires exactly one outer loop, exactly 78 qkv-a internal
loops under that outer body and zero unclassified loops; the separate backend still requires
exactly one total loop. The preserved old and fused HLOs pass respectively as `1+0` and `1+78`.
Focused tests pass 39/39, the forced-32 complete prefill regression passes in 72.70 seconds, and
Python compilation, Bash syntax, ShellCheck and diff checks pass. These are linter-readiness facts,
not decoder or performance evidence.

Exact next: seal this diagnostic pin, prove the fleet idle, and run exactly one serialized
protected 8K retry with the same fused runtime, paired token/DSA oracles, warmup 2, iterations 10
and trace 2. Any execution-time DSA/token/HBM/HLO/wall failure remains a stop condition.

## Loop-corrected 8K executes exactly, then refuses DSA before timing

The like-for-like retry
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_qkva_loopfix_trace2_20260808T161834055154820Z`
ran from 16:18--18:24 UTC at exact code
`e4079ac410f976476812e0a86ac9352e1407dc9d`. It passed the complete fused checkpoint load,
decoder/observer/prefill HLO contracts, teacher-forced 8K prefill, and the exact first token
`101252` at rank 1 with score `22.5` and top-two margin `6.75`. The device-resident observer then
failed closed before warmup, timing, trace, DB insertion, or final `SUCCESS`.

Event 0 has the exact 2,048-member set but its score order first differs at offset 8 (expected
position 8,089; observed 8,150). Event 1 already has six set swaps: expected-only
`[1052,2024,3853,6256,6787,7473]`, observed-only
`[825,3889,4899,5536,6113,6951]`. Downstream differences grow, so another blind full-decoder
retry is forbidden. The DSA NPZ SHA is `4647de15...6ac`; token observation SHA is
`e5e35f3b...03c`; all eight logs are identical at `63a29050...566e`. Pre/failure census SHAs
`c1e5f109...7d0` / `a7b172cb...ede` each contain eight unique `CENSUS_OK` hosts. Approved-bucket
critical bytes match, SQLite integrity is `ok`, and the tag occurs zero times in the DB.

## Accepted prompt-cache reuse and bounded production comparator are pinned

Commit `e5a699177cd0164d62aa202729011631c324e447` reuses the accepted runtime's existing default-off
`dcp_cache_dump.py` boundary instead of adding a new legacy execution path. The capture requires
the exact final step-4/chunk-2011 slot-0 dumps from all eight processes, reconstructs the 8,155
logical BF16 layer-0 index keys from the live block table, and proves exact four-way model-replica
identity for each DCP owner before sealing source/checkpoint/token hashes.

After the accepted runtime stops, an independent one-host greenfield probe runs the actual
production raw-FP8 `wk`, input RMSNorm, index-key LayerNorm/RoPE, and BF16 cache write in a one-row
`lax.scan`. Its fail-closed HLO contract requires exactly one
`greenfield_fp8_block_matmul_f32_m8_k6144_n128`, one outer scan, no collective/callback/transport,
no decoded weight overlay, no full-prompt hidden materialization, and no dead `[32,6144]` row.
Focused prompt-cache/oracle tests pass 27/27; Python compilation, Bash syntax, ShellCheck and diff
checks pass. Two broader CPU suites were manually stopped after unrelated existing forced-JAX
tests stopped advancing, so they are explicitly not suite-pass claims.

Exact next: commit this evidence seal, verify an idle fleet, and run exactly one serialized
`scripts/greenfield/run_capture_legacy_prompt_index_cache.sh`. Preserve and diagnose any capture,
replica, logical-page, production-HLO, or bit-comparison refusal; do not weaken the contract or
retry the full decoder until this bounded discriminator identifies the first cache-state cause.

## Accepted cache capture is complete; corrected protected resume is ready

The serialized accepted capture
`greenfield_legacy_layer0_prompt_index_cache_20260808T193700214955997Z` completed the real 8K
oracle as DB505/item1788: exact passkey, 8,155 prompt tokens, 20 generated tokens, sealed 294-event
DSA oracle, all 32 cache snapshots, owned-runtime stop, and 8/8 clean failure census. It then
failed closed before the independent greenfield comparison because the initial parser expected
the wrong mesh/page geometry. There is no final `SUCCESS` or greenfield comparison/DB claim.

Inspection of all eight final files proves the accepted cache is fully replicated: exact mesh
`model=32,dcp=1`, four addressable copies/process, 32 bitwise-identical physical copies, physical
shape `[24,16,32,128]`, and 512-token logical pages. Global-cache, full-block-table and logical
8,155-key SHAs are respectively `c65552a6...dad9`, `eedb3f92...b8a84`, and
`3808d502...859d1`. Commit `52c69df` corrects only this source contract and focused coverage passes
31/31 with two existing SWIG warnings.

`scripts/greenfield/run_prompt_index_cache_comparison.sh` now resumes from that immutable source
without reloading the accepted 753B model. It pins DB505/item1788, source/oracle/fleet/census
hashes, all 32 local cache hashes, and byte-identical approved-bucket copies of the eight final
snapshots. It re-seals the compact cache at the current code pin, runs only the four-chip
production one-row scan, accepts either exact or nonexact comparison as a diagnostic outcome,
links a new DB row, archives all derived evidence, and requires 8/8 zero work. Offline real-source
reconstruction and Python/Bash/ShellCheck/JSON/diff checks pass.

Exact next: commit/push this runner from a clean branch and run exactly one serialized
`scripts/greenfield/run_prompt_index_cache_comparison.sh`. Use its first mismatch (or exact result)
to choose the next source-backed arithmetic boundary. Do not repeat the accepted model load or
the full greenfield 8K decoder first.

## DB506/507 isolate prompt-cache production drift to the chunk-input boundary

Protected DB506/item1789,
`greenfield_layer0_prompt_index_cache_comparison_20260808T210743348875130Z` at `cee8bda`, resumes
DB505 without reloading the accepted model. It verifies all 32 source snapshots and eight archived
final snapshots byte-for-byte, reconstructs the exact accepted 8,155x128 BF16 cache SHA
`3808d502...859d1`, and executes the independent production M1 raw-FP8 scan. That scan is nonexact:
4,058 element mismatches over 1,071 positions, first at position 4, max/mean
`0.015625/1.02996e-5`. Every occurrence of token 374 differs in dimensions 70/79/86, placing the
cause upstream of scoring/top-k. Its HLO `e7e66d4e...849e` has one raw-FP8 kernel, one outer scan,
one live row, and no collective/callback/full prompt hidden/dead row. DB/archive/SUCCESS and 8/8
cleanup pass; this is diagnostic only.

Protected DB507/items1790--1792,
`greenfield_layer0_prompt_index_cache_association_20260808T214925579370178Z` at `31a23b8`, reuses
DB506 and tests only three new associations. Pallas M1 plus divide/sqrt remains far away at 4,050
mismatches. Accepted M2048 XLA projection plus multiply/rsqrt leaves 55 mismatches. M2048 XLA plus
divide/sqrt is the decisive near-exact path: only 45 values/45 positions differ, first at 113,
mean `3.15396e-8`; its SHA is `52bf55ed...cd8a`. All 45 mismatches are confined to rotary
dimensions 0--63; dimensions 64--127 are bitwise exact. HLO `93596359...36fd` has exactly one
`f32[2048,128] convolution ... bf_oi->bf`, one chunk map, physical sqrt/divide, no collective,
callback, full-prompt hidden tensor, or dead row. Association manifest `7216756c...7cae`, SUCCESS
`6ce52989...c227`, DB/archive and 8/8 pre/post cleanup pass. No candidate is yet exact, so no
production correction or full-decoder retry is authorized.

The first wrapper attempt `...association_20260808T214555292086140Z` expanded a local census
variable under `set -u` and refused before census/TPU/DB work. Its append-only local log is
preserved; `31a23b8` fixes that failure class and the successful DB507 supersedes it.

Exact next: compile one source-faithful `prefill_chunk` whose public inputs are the already-live
BF16 `[2048,6144]` hidden chunk and `[2048]` absolute positions. Run the same executable over the
four host-prepared chunks only in a bounded diagnostic. This removes the artificial unique-row
gather from the compiled map while retaining one physical M2048 convolution, divide/sqrt, no
loop/collective/callback and no full 8K hidden tensor. If it matches the accepted cache bitwise,
integrate that chunk-local prefill key path while keeping decode M1; otherwise inspect the exact
source RoPE/input-RMS association. Do not rerun DB507's rejected candidates or the full decoder.

## DB508 rejects the external-chunk interface and exposes a BF16 weight lowering

Protected DB508/item1793,
`greenfield_layer0_prompt_index_cache_association_20260808T221200429613435Z` at exact pin
`7027cf6b655d76751f7c166b38e0780428918a9e`, executes the required external live
`bf16[2048,6144]` hidden chunk plus `s32[2048]` absolute positions. The result is nonexact and
regresses to the production neighborhood: 4,045 element mismatches over 1,058 positions, first at
position 4, max/mean `0.015625/1.0294302e-5`, SHA `db2f77d...a7a1`. It differs from DB506's
production baseline by only 22 values/22 positions, max/mean `0.001953125/6.36644e-9`.

The optimized HLO `fde460cb...52a` passes the declared interface contract: one physical
`f32[2048,128] convolution ... bf_oi->bf`, zero loop/collective/callback/full-prompt-hidden/dead-row
shapes. It also provides the decisive physical delta. DB507 accepts the same FP32 `wk` public
parameter but inserts one loop-invariant `f32[128,6144] -> bf16[128,6144]` conversion before the
mapped convolution. DB508 keeps the convolution weight FP32. Therefore the 45-mismatch DB507
result was not caused by the artificial embedding gather or external-chunk boundary; its
near-exactness is associated with BF16 projection weights selected by the mapped lowering.

Association manifest `8539a81d...6d07`, compressed-HLO file `b2e98616...f274`, tensor
`10eeec43...c726`, SUCCESS `6a38369a...12e7`, evidence `48776445...214c`, remote objects
`11aa387a...34e4`, DB snapshot `81bff928...a7be`, approved archive and authenticated 8/8 pre/post
zero-work censuses all pass. This remains bounded diagnostic correctness evidence with no decoder,
Gate-D, latency or throughput claim.

Exact next: preserve the real external chunk interface and compile one candidate that explicitly
converts only the accepted adapted `wk` to BF16 before the M2048 convolution. Its HLO must pin a
BF16 convolution RHS, one convolution, zero loop/communication/callback/full-prompt/dead-row
shapes, and the same divide/sqrt norm. If that reproduces DB507's 45 rotary-half mismatches, isolate
the source-faithful RoPE association next. Do not retry the full decoder before bitwise cache
equality.

That candidate is now implementation-ready. `layer0_prompt_index_key_chunk` retains the public
accepted FP32 `wk` state and exposes a diagnostic-only `adapted_bf16` projection boundary. The
association validator requires exactly one explicit `f32[128,6144] -> bf16[128,6144]` conversion,
one M2048 convolution and no loop or forbidden operation/shape. The protected wrapper pins and
revalidates the complete DB506--DB508 local/DB/remote lineage before launch. Focused kernel and
cache-validation tests pass 32/32; Python compilation, Bash syntax, ShellCheck, JSON and diff
checks pass. Exact next: commit/push from a clean pin, prove the fleet idle, then run only
`GLM_GREENFIELD_PROMPT_CACHE_ASSOCIATION_PROFILE=chunk_bf16_weight
scripts/greenfield/run_prompt_index_cache_association_probe.sh`.

The first protected compile at `d4884bd`, tag
`greenfield_layer0_prompt_index_cache_association_20260808T224854448693338Z`, failed closed before
arithmetic because the HLO linter tied the required conversion to the source symbol
`%wk_weight`. Optimized TPU HLO already proved exactly one M2048 convolution with a physical BF16
RHS, zero loops and no forbidden operation/shape, but fusion/copy renaming produced zero matches
for that source-name regex. There is no DB row, SUCCESS, cache comparison or performance claim.
The remote diagnostic matches local orchestrator SHA `2b125a9f...7df`; authenticated failure-exit
census SHA `71bf2d2d...127` is 8/8 clean. The correction requires one shape-specific
`bf16[128,6144] convert(...)` under any optimized symbol plus the existing public FP32 parameter
and physical BF16 convolution operand. The probe now persists optimized HLO before validation.
Re-run only the same bounded profile after focused/static checks and a clean pushed pin.

Protected DB509/item1794,
`greenfield_layer0_prompt_index_cache_association_20260808T225610150435734Z` at `8f2545c`, rejects
that BF16-weight hypothesis. The public `wk` remains FP32, HLO performs exactly one explicit
shape-specific conversion, and the sole M2048 convolution physically consumes BF16. Nevertheless,
the 8,155x128 output is byte-identical to DB508: SHA `db2f77d...a7a1`, 4,045 mismatches over 1,058
positions, first position 4, max/mean `0.015625/1.0294302e-5`; it remains only 22 values from the
DB506 production baseline. Thus DB507's BF16 operand was correlated, not causal.

DB509 HLO SHA is `0638f148...9868` (compressed `6e269756...2160`); association manifest is
`df0b901e...21c1`. SUCCESS/evidence/remote-object/DB-snapshot SHAs are
`0bea72ba...9232`, `652da666...1f99`, `6378c961...7047`, and `3820af70...88f`.
Direct approved-bucket reads match association, tensor, HLO, DB, census and SUCCESS bytes; SQLite
is `ok`; authenticated pre/post censuses are 8/8 clean. This is diagnostic only. HLO/tensor delta
now points back to DB507's gather-coupled input-RMS reduction layout; isolate that physical
association before RoPE. Do not integrate BF16 `wk` or retry the decoder.

The gather-coupled discriminator is implementation-ready in the existing harness. It feeds only
the 37 immutable BF16 source embeddings, one `s32[2048]` row-index chunk and one `s32[2048]`
position chunk; the compiled device program gathers the live `[2048,6144]` rows, applies the same
input RMSNorm, explicitly retains the already-rejected BF16 weight boundary, and emits one key
chunk. HLO must contain exactly one physical embedding gather consumed by the input-RMS reduction,
one BF16-RHS convolution, zero loops and no communication/callback/full-prompt/dead-row shape.
The wrapper revalidates DB506--DB509 before launch. Focused tests pass 33/33 and Python/Bash/
ShellCheck/JSON/diff checks pass. Exact next: clean commit/push, then run only profile
`chunk_gather_bf16_weight`; this tests gather coupling without DB507's outer map.

## Gather-coupled compile is valid; exact weight-slice classifier is ready

The first protected gather-coupled attempt,
`greenfield_layer0_prompt_index_cache_association_20260808T232010491953822Z` at exact pin
`3ad7b79c77c20b4e9de44557bc9f277b0b1c3f9e`, compiled the intended discriminator but stopped
before execution. Its optimized HLO proves one physical embedding gather feeding the FP32 input
RMS reduction, one M2048 convolution with a BF16 `wk` producer, zero loops and no communication,
callback or full-prompt tensor. The input-RMS TPU window matches DB507's gather-coupled lowering:
iteration bounds `[16,1]`, kernel window `[16,48]`, estimated 88,032 cycles.

The sole refusal was the generic substring guard finding `f32[32,6144]`. Preserved HLO proves all
eight occurrences are compiler transfer instructions for the public `wk_weight.1`
`f32[128,6144]`: four `slice-start` operations cover output-feature intervals `0:32`, `32:64`,
`64:96`, `96:128`; four matching `slice-done` values are reassembled by `ConcatBitcast` before
projection. There is no `[32,6144]` parameter or hidden/token-row producer. No arithmetic, cache
comparison, DB row, SUCCESS, decoder or performance claim exists.

Compressed HLO, contract-failure, orchestrator and failure-census SHAs are
`c019cc08...52f9`, `3ad54503...c45`, `c2e5203e...7cff`, and `0fe9712a...895`; direct
approved-bucket reads match local bytes and the last census is authenticated 8/8 clean. The
validator now permits only that exact complete four-slice provenance and still rejects any extra
or unrelated `f32[32,6144]` line. The saved attempt and every protected DB507--DB509 HLO pass
offline; a dead-row mutation fails; focused CPU tests pass 34/34. Exact next: finish static checks,
commit/push a clean pin, and rerun only `chunk_gather_bf16_weight` once.

## DB510 restores the gather-coupled 45-value regime; cache-write consumer is next

Protected DB510/item1795,
`greenfield_layer0_prompt_index_cache_association_20260808T234459065479710Z` at exact pin
`6286a06341cb1f798c7059490e565e966ba85dc0`, passes the repaired HLO contract and executes the
gather-coupled candidate. Its 8,155x128 output is byte-identical to DB507's best candidate:
SHA `52bf55ed...cd8a`, 45 mismatches over 45 positions, first at 113, max/mean
`0.015625/3.1539646e-8`. All 45 mismatches are in rotary dimensions 0--63; every mismatch affects
one member of an interleaved pair while the partner and all dimensions 64--127 remain exact.

This proves the gather-coupled input-RMS lowering is causal for eliminating DB508/509's 4,000-value
drift. HLO `6dba4fbb...9de0` has the one physical gather feeding the matching `[16,1]`/`[16,48]`
FP32 reduction, one BF16-RHS M2048 convolution, the exact four `wk` feature slices, zero loops and
no collective/callback/full-prompt/dead-row state. It does not prove RoPE itself is wrong: tiny
pre-RoPE FP32 differences can be invisible in the directly stored half yet cross BF16 boundaries
after rotation.

Association manifest `3e29aadc...b0fb`; association/tensor/compressed-HLO/SUCCESS/evidence/
remote-object/DB-snapshot SHAs are `7d9c12be...e224`, `d896b172...227d`,
`cd293af3...7290`, `a7660e3e...e165`, `9ed5bdff...5b4f`, `d4663158...5f9e`, and
`50758ad2...e53`. Direct approved-bucket reads match all critical bytes, SQLite integrity is
`ok`, and authenticated pre/post censuses are 8/8 clean. This is diagnostic correctness evidence
only, with no decoder or performance standing.

The remaining source-backed physical delta is the accepted consumer: `compute_indexer_keys`
returns FP32 keys into the existing flat BF16 paged-cache scatter, whereas this discriminator
returns a compact BF16 key tensor directly. Exact next: adapt that already-audited cache-write
interface inside the same bounded harness, using the sealed `[24,16,32,128]` cache and 16-entry
live block table, require one physical scatter plus all DB510 HLO invariants, and compare once.
Do not guess at RoPE variants or retry the full decoder first.

## DB511 rejects the accepted flat BF16 cache-write consumer

Protected DB511/item1796,
`greenfield_layer0_prompt_index_cache_association_20260809T003039938922204Z` at exact pin
`31ab626547edd7b622bb23d7d994a889cd311eeb`, executes the gather-coupled M2048 candidate through
the real flat paged-cache addressing and BF16 scatter. Its reconstructed 8,155x128 logical cache is
byte-identical to DB510/DB507: SHA `52bf55ed...cd8a`, 45 mismatches over 45 positions, first at
113, max/mean `0.015625/3.1539646e-8`. The accepted cache-write consumer is therefore rejected as
causal; do not integrate it as a correction or retry the full decoder from this result.

Optimized HLO `d396040f...3938` has exactly one physical BF16 scatter on flat
`[12288,128]` cache state with a BF16 `[2048,128]` update, the exact `[24,16,32,128]` aliased cache
input/output and `s32[16]` live block table, one embedding gather feeding the DB510-matched input
RMS reduction, one BF16-RHS M2048 convolution, exact four-slice `wk` provenance, and zero loops,
collectives, callbacks or forbidden dead-row/full-prompt state. Compressed HLO SHA is
`e55cecef...c10` and manifest is `6cbe954b...bb6`.

Association/tensor/summary/SUCCESS/evidence/remote-object/DB-snapshot SHAs are
`addefc7e...8ebf8`, `64e6df3b...130f`, `8d420541...e1d2`, `ec6a4370...a7`,
`8978b333...cb25`, `5d7bb1f6...083f`, and `d71012bb...6eae`. Direct approved-bucket reads of
those objects, HLO, both censuses and terminal SUCCESS equal local bytes; SQLite integrity is
`ok`; authenticated pre/post censuses are 8/8 clean. This is diagnostic correctness evidence only.

Exact next: audit preserved accepted prompt artifacts for a physical pre-/post-RoPE boundary or
optimized source HLO. If no such evidence exists, isolate the literal accepted RoPE source spelling
inside the same gather/RMS/projection/scatter harness. If that is byte-identical to DB511, capture
the accepted pre-RoPE FP32 row at the first mismatching position rather than guessing more variants.

## Accepted RoPE evidence audit is closed; one literal discriminator is ready

The preserved accepted artifacts contain no optimized prompt RoPE HLO and no prompt pre-RoPE
state. DB493's only internal key boundary is post-RoPE FP32 at decode position 8,155; none of
DB511's 45 mismatches is at that position (the last prompt position, 8,154, is exact), so it cannot
discriminate the prompt drift. The accepted pin `b3c25df47ac98783912dc658878181ec0a8ae16d`
spells RoPE in `tpu_inference/layers/vllm/custom_ops/glm_dsa_indexer.py` as FP32
`theta ** (-arange / rope_dim)`, direct cosine/sine, and interleaved pair arithmetic. A CPU audit
over positions 0--8,154 found its cosine/sine arrays bitwise identical to the existing greenfield
helper; their StableHLO contains the same physical arithmetic, so this is a deliberately
single-shot lowering discriminator rather than a new numerical hypothesis.

The default-off `chunk_gather_cache_write_source_rope` profile now adapts that literal spelling
inside the already-proven DB511 gather/RMS/M2048-BF16-projection/flat-cache-scatter path. Its HLO
contract requires exactly one FP32 `[32]` power, one each FP32 `[2048,32]` cosine and sine, theta
`8e6`, exponent `0.015625`, and every DB511 invariant. It also revalidates DB511's local, DB and
remote bytes before launch and records whether the optimized HLO and the physical RoPE contract
equal the parent. Focused tests pass 35/35; Python compilation, Bash syntax, ShellCheck, JSON,
saved-DB511 HLO validation and diff checks pass. These are readiness facts only.

Exact next: commit/push this clean default-off discriminator and run exactly one protected
`GLM_GREENFIELD_PROMPT_CACHE_ASSOCIATION_PROFILE=chunk_gather_cache_write_source_rope
scripts/greenfield/run_prompt_index_cache_association_probe.sh`. If it leaves the same 45 values,
do not try more RoPE variants: capture the accepted pre-RoPE FP32 key at prompt position 113.

## DB512 rejects literal accepted-source RoPE spelling

Protected DB512/item1797,
`greenfield_layer0_prompt_index_cache_association_20260809T012205182369900Z` at exact pin
`da7027dfc1e76e2d64c8d4c4ee2ecbfa53858995`, produces the exact DB511/DB510 bytes: logical
8,155x128 BF16 SHA `52bf55ed...cd8a`, 45 values at 45 positions, first 113, max/mean
`0.015625/3.1539646e-8`. Literal accepted RoPE source spelling is therefore rejected as a
sufficient cause; do not add more formula variants or retry the full decoder.

Optimized HLO `c96ecd28...a931` differs bytewise from DB511 but has the identical physical RoPE
contract: one FP32 `[32]` power, one each FP32 `[2048,32]` cosine/sine, theta `8e6`, exponent
`0.015625`. It also retains one gather-coupled RMS producer, one BF16-RHS M2048 convolution, one
flat BF16 cache scatter and zero loop/communication/callback/forbidden state. Manifest
`9f037699...2c5`; tensor file `20398ae9...9413`; compressed HLO `eb2df033...e9b6`; summary
`890ec592...098`; SUCCESS `01882c1f...3406`; evidence `34d23f22...1c62`; remote objects
`9d0e5bbf...5488`; DB snapshot `5ba79f36...6e1`. SQLite is `ok`, direct approved-bucket bytes
match all critical local files, and authenticated pre/post censuses are 8/8 clean. This is bounded
diagnostic correctness only and leaves DB484/Gate E unchanged.

Exact next: reuse the existing default-off accepted observer machinery to capture only the
pre-RoPE FP32 layer-0 key row at prompt position 113 (and the adjacent post-key-LayerNorm boundary
only if needed to make the producer self-identifying). Bind it to the immutable DB505 prompt,
checkpoint and cache evidence, prove the observer does not perturb accepted output/cache state,
then compare against the already-sealed greenfield input. No wider tensor or formula matrix is
authorized.

## Accepted prompt-key producer capture is implementation-ready

Oracle-only observer pin `9c1d6b3b950d5c5dd45bdf885058202517097eba`, three commits above the
unchanged accepted pin, captures only layer-0 prompt position 113 at the actual key producer. It
records FP32 projection, post-key-LayerNorm and post-RoPE rows through the existing zero-copy,
default-off DSA callback. The prior scorer observer remains pinned separately at `83ff4a357`.

The greenfield comparator returns the same three already-computed DB512 boundaries from one
bounded M2048 state executable, carries its BF16 cache through the remaining three chunks, and
requires final SHA `52bf55ed...cd8a`. The accepted run simultaneously captures the full prompt
cache and must retain SHA `3808d502...859d1`, exact tokens, all 294 DSA events, checkpoint/state
integrity, DB/archive linkage and authenticated 8/8 cleanup. Both accepted and greenfield
post-RoPE FP32 casts must reproduce cache row 113 before the comparison can classify projection,
key LayerNorm or RoPE as the first divergent field. HLO requires the DB512 gather/RMS,
BF16-RHS convolution, literal RoPE and flat-scatter identities with no loop, communication,
callback, full-prompt hidden tensor or dead row.

Focused coverage passes 48/48 and the complete CPU-only greenfield suite passes 478 with one
expected skip and two existing SWIG warnings. Bash syntax, ShellCheck, Python compilation, JSON,
line-length and diff checks pass. This is readiness evidence only: no protected capture,
arithmetic conclusion, decoder retry, Gate-D change or performance claim exists yet.

Exact next: commit/push the clean pin, prove an authenticated idle fleet, and run exactly one
serialized `bash scripts/greenfield/run_capture_legacy_prompt_key_internals.sh`. Use its first
divergent field to implement only the smallest source-backed correction before one protected 8K
Gate-D retry. A capture/cache/HLO refusal must be preserved and diagnosed rather than relaxed.

## DB513 isolates projection association; FP32-RHS discriminator is ready

Protected DB513/item1798,
`greenfield_legacy_layer0_prompt_key_internals_p113_20260809T024848198817596Z`, completed at
greenfield pin `00404a0cb6ce18c71816cc47e16ddbaa5f1a5fa6`, observer pin
`9c1d6b3b950d5c5dd45bdf885058202517097eba`, and unchanged accepted oracle
`b3c25df47ac98783912dc658878181ec0a8ae16d`. The full accepted run retained exact passkey/raw
tokens, all 294 DSA events, 1,882/0 checkpoint checksum results, 2,455 state leaves, accepted cache
SHA `3808d502...859d1`, DB/archive linkage and authenticated 8/8 cleanup. Its final manifest is
`76c8577d...1d9c`; direct approved-bucket `SUCCESS` matches locally.

The one-row producer comparison classifies `pre_layer_norm_key` as the first divergent field.
At position 113 the projection differs in 79/128 FP32 values, max/mean
`3.5762787e-7/4.2949978e-8`; post-key-LayerNorm and post-RoPE differ only downstream. Both
post-RoPE FP32 casts reproduce their own BF16 cache row exactly. This rejects key LayerNorm, RoPE,
and cache scatter as the first cause. Comparison manifest `605eeac2...a04c`; accepted capture
manifest/tensor `dd361437...591f` / `db26efc4...bb64`. This is diagnostic correctness evidence,
not decoder or performance proof.

The first launch attempt stopped before TPU/model load because worker 0 had under 10 GB free. Its
failure evidence is archived and has no DB/SUCCESS/performance standing. Four local one-layer
feature-pack payloads (9,716,387,360 bytes total) were then removed only after their SHA/size,
approved GCS generation/CRC32C and remote `SUCCESS` were verified; they are exactly recoverable
from the plan-aware checkpoint prefix. The accepted retry then passed. Worker 0 currently has about
14 GB free because DB513 retains 3.3 GB of fully archived source dumps; preserve the compact
capture/cache/comparison evidence and verify remote-object coverage before any bounded reclamation.

The accepted source casts hidden and adapted `wk` to FP32 before `h @ wk.T`. The DB513 greenfield
reproduction instead explicitly rounds adapted `wk` to BF16 at the M2048 convolution. The new
default-off `adapted_fp32` profile changes only that operand inside the same gather-coupled input
RMS, divide/sqrt key norm, literal source RoPE and flat BF16 cache scatter. Its HLO linter requires
one physical FP32-RHS convolution, zero FP32-to-BF16 `wk` conversions, one gather, one scatter and
no loop/communication/callback/dead/full-prompt state. The wrapper reuses the sealed DB513 capture
artifact directly and pins its DB row, manifests, local/remote bytes and 8/8 censuses. Focused CPU
coverage passes 38/38; no TPU conclusion exists yet.

Exact next: commit/push the clean discriminator, then run exactly one serialized
`bash scripts/greenfield/run_prompt_key_projection_association_probe.sh`. If both the three FP32
producer states and the complete 8,155-row BF16 cache are exact, integrate only this projection
boundary into production and run one protected 8K Gate-D retry. If not, capture/compare the
normalized hidden row feeding projection before adding another arithmetic variant.

## DB514 rejects FP32 projection-weight precision; input capture is ready

Protected DB514/item1799,
`greenfield_layer0_prompt_key_projection_association_20260809T042103930425146Z`, completed at
greenfield pin `c5912db39738bdc802c6728110be036f8c93c176`. The physical M2048 projection consumes
an FP32 `wk` RHS and has zero FP32-to-BF16 weight conversions, yet all three producer tensors and
the complete prompt cache are byte-identical to DB513's BF16-RHS reproduction. Projection,
post-key-LayerNorm and post-RoPE observed SHAs remain `963269f9...5154`, `8f6184af...094e` and
`230dfb0b...dd6d`; cache SHA remains `52bf55ed...cd8a` with the same 45 mismatches. FP32 versus
BF16 adapted-`wk` precision is therefore conclusively rejected as the cause.

Comparison manifest is `2c41e2b2...3f7e`; cache/states optimized-HLO SHAs are
`b6009101...900b` / `1f4dae4e...6bdc`; SUCCESS/evidence/remote-object/DB-snapshot SHAs are
`b2806ee7...69e0`, `50e5daf6...6396`, `c39e25a7...5204` and `12dc47ca...09ac`. Direct approved-
bucket bytes, SQLite integrity and authenticated 8/8 pre/post cleanup pass. This is bounded
diagnostic correctness evidence only, with no decoder, Gate-D, latency or throughput claim. The
preceding `...T041940...` wrapper attempt failed before TPU use because a valid nine-character
stored fork abbreviation was compared to a ten-character slice; it has no candidate, DB row or
SUCCESS and preserves 8/8 clean failure evidence. The wrapper now accepts any unambiguous stored
abbreviation of at least seven characters.

DB513's 516 raw source-dump files (3,434,645,148 payload bytes) were removed locally only after the
local and remote object ledgers, terminal remote SUCCESS, exact paths/sizes, generations and
CRC32C values were verified for all 516 objects. The compact accepted capture, cache, comparison,
DB and manifests remain local; the raw files are exactly recoverable from the approved DB513
prefix. Worker 0 has about 17 GiB free.

Oracle-only observer pin `89fc453b6116ac3df71e666db6f4659775b313c3`, six commits above the
unchanged accepted parent, adds a separate default-off `prompt_key_input` mode. It captures the
actual FP32 `h = as_jax_f32(hidden_TD)` 6,144-wide row entering `h @ wk.T`, in addition to the
unchanged three key boundaries. Existing scorer and `prompt_key` modes remain unchanged; focused
legacy tests pass. Greenfield now seals either capture mode, compiles the independent gathered
M2048 input-RMS producer with no projection/loop/collective/callback/dead/full-prompt state, and
classifies either `projection_input_association` or `projection_lowering_association`. Focused
greenfield capture/kernel tests pass 40/40 on explicit CPU; Python, Bash, ShellCheck and diff checks
pass. The complete explicit-CPU greenfield suite passes 480 with one expected skip and the same two
pre-existing SWIG warnings in 363.45 seconds.

Exact next: commit/push the greenfield capture integration, prove the fleet idle, and run exactly
one serialized `bash scripts/greenfield/run_capture_legacy_prompt_projection_input.sh`. If the
6,144-wide input differs, correct only that upstream input-norm association. If it is bitwise
exact, preserve that exclusion and audit the accepted projection lowering/physical association.
Do not retry the complete 8K decoder before this discriminator produces an exact correction.

## DB515 proves the normalized projection input is exact; projection lowering remains

Protected DB515/item1800,
`greenfield_legacy_layer0_prompt_projection_input_p113_20260809T050055956585082Z`, completed at
greenfield pin `b8e30ed41e46816461c7e956e4dd7bc85e96d998`, observer pin
`89fc453b6116ac3df71e666db6f4659775b313c3`, and unchanged accepted parent
`b3c25df47ac98783912dc658878181ec0a8ae16d`. The full accepted 8K run retained exact passkey/raw
tokens, all 294 DSA events, checkpoint/state/cache integrity, DB/archive linkage and authenticated
8/8 cleanup. Accepted cache SHA remains `3808d502...859d1`; the bounded greenfield reproduction
remains `52bf55ed...cd8a` with the same 45 BF16 mismatches.

The actual accepted FP32 normalized 6,144-wide projection input at layer 0 / prompt position 113
is bitwise identical to the independently gathered greenfield input: SHA
`d0edbfa0...59566`, 0/6,144 mismatches and zero absolute error. The following projection is still
the first divergent field: 79/128 FP32 values differ, max/mean
`3.5762787e-7/4.2949978e-8`. This excludes embedding selection, input RMSNorm arithmetic and its
physical gather-coupled lowering at the captured row. Together with DB514, it also excludes
adapted-`wk` FP32 versus BF16 operand precision. The remaining classification is specifically
`projection_lowering_association`; no upstream input correction is authorized.

Comparison/capture manifests are `df048dd7...f258` / `64320e97...2ef9`; comparison, accepted
tensor, projection-input HLO, DB snapshot, evidence, remote-object and SUCCESS SHAs are
`bf7b49ab...a0e3`, `753e63d9...8f2d`, `6236cb8f...d2d2`, `43fdf121...01c0`,
`637ebad0...156d`, `06c0b778...5b6d` and `048528e3...d79`. Direct approved-bucket SUCCESS and
remote-object bytes match locally. The 516 raw source files / 3,434,670,354 bytes were reclaimed
locally only after exact path/size plus nonempty generation/CRC32C verification for all 516 ledger
entries and remote SUCCESS equality; compact capture/cache/comparison/HLO/DB evidence remains and
the raw files are exactly recoverable from the approved DB515 prefix.

A preserved 256K prefill XPlane at legacy pin `4647a8fbcd49` was also audited before scheduling new
TPU work. Its unchanged `h @ wk.T` source lowers at M2048 to a convolution fusion with tuple shape
`(f32[2048]{0:T(1024)S(3)}, f32[2048,128]{0,1:T(8,128)S(3)})`, the same visible output layout as
DB515's bounded candidate. It is useful negative historical evidence but cannot substitute for the
current accepted `b3c25df47` compiler association: it is an older code/config pin, one host was
preserved locally, and XPlane does not expose the convolution emitter or operand windows.

Exact next: capture one protected current-pin 8K prefill step with eight-host XPlanes and a
module-filtered final optimized HLO dump for `jit_step_fun_impl`. Require exact tokens, all DSA
events, state/load integrity, DB/archive linkage and 8/8 cleanup. Extract only source
`glm_dsa_indexer.py:1122` at M2048, including physical output/input layouts, convolution emitter,
window/megacore config and invocation count. Do not try an arithmetic matrix or rerun the full
greenfield 8K decoder until this source-backed comparison yields a bounded bitwise-exact projection
correction.

## Current-pin accepted projection-lowering capture is ready

The default-off protected capture required after DB515 is implementation-ready. It runs the plain
accepted `b3c25df47` 8K oracle, enables the existing phase profiler for exactly one prefill step,
and uses a module-filtered XLA dump for scheduled `jit_step_fun_impl`. The wrapper verifies the
profile/HLO environment on all eight raylets before the request, retains exact tokens/all DSA
events/load/state protections, and requires eight XPlanes plus at least one scheduled-HLO owner.

The independent sealer directly reuses the existing XPlane parser and requires eight unique hosts,
64 TPU cores, one prefill module/core, exactly 21 line-1122 M2048 projection fusions/core, and a
uniform physical lowering across all 21 full-indexer layers. It seals input/result/fusion layouts,
convolution emitter, megacore and window configs. XPlanes/trace JSONs are checksum-verified hard
links within the append-only run directory, avoiding a second local profile footprint while
preserving evidence after verified raw-source reclamation. Bash syntax, ShellCheck, Python compile
and 46 relevant unit/validation tests pass. The standard explicit-CPU suite passes 486 with one
expected skip and the two existing SWIG warnings in 363.72 seconds. No TPU or arithmetic
conclusion exists yet.

Exact next: commit/push the clean readiness pin, prove the protected fleet idle, and run exactly
one serialized `bash scripts/greenfield/run_capture_accepted_prompt_projection_lowering.sh`.
Compare its current accepted physical association to DB515 before implementing one bounded
projection correction; do not launch an arithmetic matrix or the full greenfield 8K decoder first.

## DB516 seals physical M64; one M64 projection discriminator is ready

Protected DB516/item1801 completed the accepted 8K oracle at capture pin `643d092`: exact passkey
`881446`/raw tokens, 483 DSA dumps, 1,882/0 load checks, 2,455 state leaves and eight one-step
XPlanes all passed. The original wrapper was interrupted only while gathering an overbroad XLA
dump and made no terminal claim. Recovery artifact
`greenfield_accepted_prompt_projection_lowering_recovery_20260809T105858202006975Z` at sealer
pin `4786e26` terminally seals the completed run without reloading the model.

All 64 TPU cores observe exactly 21 source-line-1122 projection fusions. Final physical HLO is not
M2048: each of 32 partitions owns BF16 lhs `[64,6144]`, FP32 rhs `[128,6144]` and FP32 result
`[64,128]`, using `EmitAllBatchInSublanes`; the logical 2,048 rows are 32 physical M64 shards.
The lowering manifest is `d9b492ee...fba6`, DSA event tensors are exact, the exact target HLO is
archived from all eight hosts, and local/remote CRC32C, DB snapshot, terminal SUCCESS and
authenticated pre/post 8/8 censuses pass. After sealing, 27,195 interrupted raw HLO files /
10,512,245,600 path-bytes were inventoried and reclaimed locally; the selected HLO remains in
eight remote source objects and the compressed sealed artifact. The exact remote `/tmp` source tag
was then removed on 8/8 hosts.

The next default-off discriminator reuses DB515's bitwise-exact normalized input and accepted
producer/cache plus DB516's physical contract. It maps one 2,048-row prefill chunk as 32 explicit
64-row projections with `lax.map`, changes no decode path, and requires one physical M64
convolution, one bounded projection-map loop, exact projection input, exact three producer states
and exact 8,155-row BF16 cache. Local semantic, HLO-linter, Bash, ShellCheck and compilation checks
pass. Exact next: finish the explicit-CPU batch, obtain one diff-only Fable commit-readiness
verdict, commit/push, prove the fleet idle, then run exactly one serialized
`bash scripts/greenfield/run_prompt_key_projection_association_probe.sh`. Do not retry the full 8K
decoder unless this bounded result is bitwise exact.

## First M64 attempt fails closed on RHS precision; scoped retry is ready

Protected attempt
`greenfield_layer0_prompt_key_projection_m64_20260809T113734128804822Z` at `50f619d` compiled the
intended 32×M64 map but stopped at the HLO contract before execution/comparison. Both executables
contain exactly one map loop and one `bf16[64,6144]` by `[128,6144]` convolution producing
`f32[64,128]` with `EmitAllBatchInSublanes`. The standalone map nevertheless inserted one
FP32-to-BF16 conversion for `wk`, so the convolution RHS was BF16 rather than DB516's accepted
FP32 RHS. The attempt has no arithmetic result, DB row, SUCCESS, decoder, Gate-D or performance
standing. Pre/failure censuses are authenticated 8/8 clean; the contract and all three HLOs are
preserved under the approved diagnostic prefix.

The correction requests per-operand `[DEFAULT, HIGHEST]` precision only for the opt-in physical-M64
dot: the source-derived lhs may return to physical BF16 while `wk` must remain FP32. The logical
M2048 default and every other projection call retain their prior precision. The HLO linter now
requires the exact BF16 M64 lhs as well as the FP32 RHS. Forced-CPU StableHLO pins the mixed
precision request and absence of HIGHEST from the logical path; the focused kernel/cache/lowering
suite passes 52/52. Exact next: obtain the narrow final Fable confirmation, commit/push, then run
one clean serialized retry. Do not re-review the already cleared `50f619d` base and do not run the
full 8K decoder unless the retry is bitwise exact.

## DB517 makes projection exact and moves the boundary to physical key LayerNorm

Protected DB517/item1802,
`greenfield_layer0_prompt_key_projection_m64_20260809T115707855267296Z` at `5926b05`, passes the
DB515/DB516 pins, one-host four-chip execution, DB/archive linkage and authenticated 8/8 pre/post
cleanup. Both state/cache HLOs contain one loop and one accepted-shape BF16 `[64,6144]` × FP32
`[128,6144]` convolution with no `wk` downcast or forbidden operation/shape. The position-113
projection is now bitwise exact: 0/128 mismatches. M64 placement plus mixed operand precision is
therefore the exact projection correction.

The first difference moves to `pre_rope_key`, so only key LayerNorm remains at that producer
boundary: 29/128 FP32 values differ, max `1.1920929e-7`. The complete BF16 cache improves from 45
to 22 mismatches, first at position 114, max `0.001953125`. Comparison manifest is
`f3587bd4...fecc`; terminal SUCCESS is `79f0ea68...03c0`; local/remote object CRC32C, SQLite
integrity and DB517 linkage pass. This is diagnostic correctness evidence, not decoder/Gate-D or
performance proof.

DB516's preserved physical HLO normalizes each `[64,128]` projection inside its partition: mean,
variance and sqrt are `[64]`, followed by a physical `[64,128]` affine. DB517 instead normalizes
the mapped output as grouped `[32,64,128]` with `[32,64]` reductions. The next default-off mode
keeps projection and key LayerNorm in one M64 `lax.map`; its linter requires one loop, exact mixed
convolution, physical `[64]` sqrt and `[64,128]` affine, and rejects grouped `[32,64]` key-norm
sqrt. DB517 is now a pinned prerequisite. Exact next: finish local checks, one diff-only Fable
commit-readiness review, commit/push, then one serialized protected key-LayerNorm discriminator.
Do not retry the full decoder unless producer states and the entire 8,155-row cache are exact.

## DB518 is exact; default-off full-prefill repair is locally frozen

Protected DB518/item1803,
`greenfield_layer0_prompt_key_norm_m64_20260809T122010714691723Z`, closes the bounded producer
search at greenfield `86243115452920fe4244bb77a9bbf4c44110aeab`. The DB515 normalized input,
projection, physical key LayerNorm, post-RoPE state and complete 8,155-row BF16 cache are all
elementwise exact. Accepted/candidate cache SHA is
`3808d502f3ea1829bf12ab7585d66f15dd83bf640657a17c35daabf5ab1859d1`; comparison manifest is
`1d80d088561181a63e734a91cd0124c4011cfc4151198488c052740050d66fe5`; terminal SUCCESS is
`a8d370166257622875feafd4d1da3f8d666204a8609baaffef2573b659f6bfee`. DB/archive/direct remote
bytes and authenticated 8/8 cleanup pass. This authorizes production integration but is not itself
decoder, Gate-D or performance evidence.

The current uncommitted batch adds a separate default-off prefill decoder. During the existing
device-resident teacher-forced scan it retains only each stage's exact BF16 normalized projection
inputs for full-indexer layers. After the scan, each LP4 lane recomputes its stage's prompt keys in
four M2048 chunks using raw-FP8 BF16-origin adapted `wk`, mixed
`[DEFAULT,HIGHEST]` M64 projection, physical affine key LayerNorm and RoPE, then overwrites only
its owned page rows. The production recurrent decoder remains the unchanged true one-row program.
At 8K the history budget is 501,043,200 bytes/device.

The prefill HLO gate requires 84 repair loops/projections, exact BF16 `[64,6144]` by FP32
`[128,6144]` operands on TPU, 168 physical `[64]` sqrt records, at least 84 physical affines/cache
writes, no grouped `[32,64]` norm, repair collective, host callback or full-pod history, and
positive measured 32-chip HBM headroom. The wrapper pins DB518 local hashes, live DB518/item1803,
and direct approved-bucket SUCCESS before launch. A forced-32 CPU program compiles and executes all
eight repair branches with the public eight-output prefill interface unchanged. Fable's one-time
diff audit found that the first draft recorded the rounded BF16 split-residual boundary even though
accepted fused RMSNorm consumes its unrounded FP32 sum; layer-0 DB518 cannot expose that difference.
Independent reproduction confirmed 1,019/6,144 differing values on a nontrivial pair and zero with
a zero residual. The repair now records `dsa_internals.normalized_hidden` directly, and the
forced-32 regression proves it equals the split DSA observer while differing from the rounded
boundary on nontrivial layers. The corrected affected suite passes 56/56 in 119.34 seconds; the
complete explicit-CPU greenfield suite passes 499 with one expected skip and two existing SWIG
warnings in 414.04 seconds. No TPU run or performance claim exists for this batch yet.

Fable's narrow xhigh follow-up verified the corrected recording, repair arithmetic, independent
split observer regression, unchanged 501,043,200-byte budget, LP4 ownership, default-off behavior,
and recurrent decoder, then returned `APPROVE COMMIT`. Exact next: commit/push this frozen batch.
After an authenticated idle-fleet census, launch exactly one serialized protected 8K PP8 run with
the accepted split/token/DSA/trace profile and
`GLM_GREENFIELD_PREFILL_INDEX_REPAIR=1`. Preserve and diagnose any HLO, HBM, token or DSA refusal;
only a fully protected pass may advance Gate D/E.

## First integrated repair run fails only on linter scope

The serialized protected 8K run
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_oracle_dsa_trace2_20260809T142231841175528Z`
at pushed pin `75e4e8f` passed its authenticated 8/8 pre-census and compiled the complete prefill,
then failed closed before execution because the decoder-wide safety linter classified intentional
repair tensors as recurrent state: 560 `f32[32,6144]` physical slice records and the local
`bf16[2048,6144]` prompt chunk. No tokens, DSA comparison, timing, DB row, terminal SUCCESS or
Gate-D/performance claim exists. The failure-exit census is authenticated 8/8 clean.

The independent repair contract itself passes every physical gate: 21 full-indexer layers, four
chunks/layer, 84 exact BF16-M64/FP32-weight projections, 168 physical square roots, 84 physical
affine operations, 189 cache writes, zero grouped square roots, zero collectives, zero forbidden
markers, no full-pod history and 501,043,200 estimated history bytes/device. Thus the protected
result supports a linter-scoping defect, not an arithmetic, HBM, token, DSA or latency conclusion.
Contract/HLO/failure-census SHAs are `c394c81b...9776d`, `c0517774...515c4` and
`bb0a5592...8335`.

The repair exception is now rooted only in exact repair operation metadata and explicit HLO
callee edges, covering unnamed TPU SPMD slice/fusion scaffolding without a module-wide shape
exemption. The preserved physical HLO is accepted for all 560 `f32[32,6144]`, 1,624
`bf16[2048,6144]` and 3,170 `f32[128,6144]` repair-associated occurrences. Synthetic negative
tests prove the same shapes in an unrelated computation and the default-off decoder remain
rejected. Focused runtime/compiler coverage passes 56/56 on explicitly selected CPU in 116.28s;
the complete explicit-CPU greenfield suite passes 501 with one expected skip and the same two
pre-existing SWIG warnings in 410.89s.

An earlier local test command omitted `JAX_PLATFORMS=cpu`, initialized the local TPU and was
interrupted as invalid evidence; the owned process terminated and released the TPU lock. A second
CPU command incorrectly forced 32 devices globally and exposed an unrelated fixture assumption;
it is also excluded. Only the explicit-platform, ordinary-device-count pass above is evidence.

Exact next: finish static checks, obtain one Fable xhigh audit of only this new linter-fix diff,
independently resolve any blocker, commit/push, prove the fleet idle, then run one serialized
protected 8K retry. Do not re-review the already-cleared arithmetic integration and do not claim
Gate D before execution, exact tokens/DSA/cache, wall/trace/HBM, DB/archive and cleanup all pass.

Fable's one-time xhigh review returned `APPROVE COMMIT`. Its independent replay found 1,809 of
15,736 computations in the repair-rooted set (1,452 exact-name seeds plus 357 explicit callees),
matched all 5,354 admitted sensitive-shape occurrences, proved ENTRY and all 116K main-scan
operations remain outside the scope, and confirmed flag-off forbids all 5,354. It also reran the
affected decoder/prefill tests 24/24 on explicit CPU. Three optional notes—document that optimized
HLO currently seeds via the exact top-level branch prefix, guard hypothetical future ENTRY
hoisting, and add a branch-prefix-only synthetic fixture—are below blocker and require no repeated
review. Exact next is final diff verification, commit/push, authenticated idle-fleet proof and one
protected retry.

## Integrated repair executes but exact DSA refuses before timing

The protected retry
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_oracle_dsa_trace2_20260809T154619026858354Z`
at pushed pin `ff5072e` compiled in the established full-prefill envelope, passed decoder/observer/
prefill HLO contracts and executed the teacher-forced prefill. The repair contract proves 21 full
indexers, 84 exact BF16-M64 by FP32-`wk` projections, 168 physical `[64]` square roots, 84+
physical affines, 189 cache writes, zero grouped square roots, repair collectives, callbacks or
full-pod history, and 501,043,200 history bytes/device. This closes the preceding linter issue.

The first generated token remains exactly `101252`, but the step-0 device DSA observer refused
before warmup/timing. All 21 producer events preserve valid score order, ties, lane replication,
counts and producer identities, yet none preserves the exact selected set: event 0 has four
expected-only/four observed-only positions, event 1 has nine swaps, and the maximum grows to 571.
There is no timed distribution, XPlane, DB row, terminal `SUCCESS`, Gate-D or performance standing.
All eight rank logs are byte-identical; the failure exit is authenticated `CENSUS_OK` on 8/8 hosts.

The failure is narrower than the previous decoder result: repair HLO and execution are proven, and
the post-scan cache change is causal. Against the preceding qkv-a loop-fix observation, every common
layer-0 selected score has different FP32 bits even though the recurrent layer-0 query path is
unchanged. HLO tracing exposes one unproven boundary: exact DB518 receives adapted
`f32[128,6144] wk` as an executable parameter after a completed raw-FP8 -> BF16 -> FP32
materialization, whereas the integrated executable fuses raw-FP8 dequantization, BF16 rounding and
FP32 promotion internally before carrying the result into the same-shaped M64 loop. Exact next:
reuse the DB518 one-host harness for one bounded internal-versus-materialized `wk`/LP4-scatter
discriminator. Do not run another full 8K compile until the actual production repair cache is
bitwise equal to DB518.

The discriminator implementation reuses the DB518 comparator/wrapper and adds only an explicit
weight-source choice plus a fail-closed entry-parameter/BF16-round HLO contract. Fable's one-time
diff review caught that the first matcher omitted optimized-HLO layout annotations and would have
falsely refused the raw-FP8 arm. The matcher is now anchored to layout-tolerant
`bf16[128,6144]` conversion rather than unrelated cache casts; focused tests pass and Fable's
narrow blocker-only follow-up returned `APPROVE COMMIT`. Do not re-review this frozen batch.

## First internal-weight discriminator stops only on flattened BF16-round HLO

Bounded protected attempt
`greenfield_layer0_prompt_key_weight_source_internal_20260809T181800Z` at pushed pin `8dee6b8`
passed DB518 source/lowering identity, the authenticated 8/8 pre-census, and one-host four-chip
compilation of the projection-input, cache and producer-state programs. Every existing M64
projection, physical key-LayerNorm, RoPE, cache-scatter and no-communication contract passes. The
new raw-weight contract also sees exactly one entry `u8[128,6144]` parameter and zero entry
`f32[128,6144]` parameters.

It stopped before arithmetic because optimized TPU HLO flattens the explicit adapted-weight round:
`f32[786432] multiply -> bf16[786432] convert -> f32[786432] convert -> reshape[128,6144]`.
The initial linter admitted only the logically shaped `bf16[128,6144]` form and therefore reported
zero rounds. No cache/producer comparison, DB row, terminal `SUCCESS`, decoder, Gate-D, latency or
performance conclusion exists. The exact HLO/contract diagnostics are archived under the approved
`diagnostic_local` prefix and the failure-exit census is authenticated 8/8 `CENSUS_OK`.

The correction admits only the two equivalent weight shapes, `[128,6144]` or flat `[786432]`, on a
single-line BF16 convert with the existing convert metadata; raw-entry and materialized-entry
discriminators remain unchanged. Replay against both preserved TPU HLOs finds exactly one round and
passes, while cache casts have different shapes. Focused tests pass 23/23, diff checks pass, and
Fable's one-time narrow review independently traced the flat value to the raw `wk` dequantization
chain and returned `APPROVE COMMIT`. Exact next: commit/push this two-file correction, prove the
fleet idle, and rerun the bounded raw-internal probe with a fresh append-only tag. This comparator
tests internal materialization arithmetic; it is not yet a production LP4-scatter proof. Do not
launch another full 8K decoder first.

## DB519 rejects in-executable weight materialization; LP4 fix is local

Protected DB519/item1804,
`greenfield_layer0_prompt_key_weight_source_internal_20260809T182400Z`, ran the corrected raw-FP8
arm at pushed pin `5e1cbb5`. All source, physical-M64 projection/key-norm, cache-scatter,
no-communication, DB/archive and authenticated 8/8 cleanup contracts pass. The entry contains one
raw-U8 `wk`, zero external FP32 `wk`, and one explicit BF16 adaptation round, yet all 8,155 cache
positions drift: 298,532 BF16 values, first position 0, max `0.03125`, candidate SHA
`8fd4a8c2...d5df08` versus accepted `3808d502...859d1`. All 128 position-113 projection values
differ. Comparison manifest is `b9669799...48d42`; SUCCESS is `027d68ef...7220`. This proves the
internal materialization boundary is causal and rejected; it is diagnostic, not Gate-D or
performance evidence.

The current batch moves only the five padded full-indexer `wk` slots/device into a separate,
completed stage-local raw-FP8 -> BF16 -> FP32 executable after direct final-layout load. Prefill
repair receives those FP32 arrays as independent parameters; recurrent decode and the packed
checkpoint remain unchanged. The full-fleet contract requires five raw/scale parameters, five
BF16 rounds/promotions, zero collective/callback, zero repair-side weight round, exact local shard
ownership and 15,728,640 derived bytes/device. A reused DB518 one-host harness now has a bounded
LP4 arm that runs the production materializer and repair over four lanes, uses sentinel caches to
reject any non-owner write, and requires all 8,155 assembled rows to equal DB518 bitwise. Focused
CPU/static coverage passes 71/71. Exact next: finish the affected suite, obtain one Fable review of
only this new diff, commit/push, then run the bounded LP4 arm. Do not run the full 8K decoder until
that protected LP4 result is exact.

## First LP4 materializer attempt exposes entry-parameter linter scope

Bounded protected attempt
`greenfield_layer0_prompt_key_materialized_lp4_20260809T205000Z` at pushed pin `99ce5ea` passed
the DB515/516/517 source contracts in the DB518-derived harness and authenticated 8/8 pre-census,
then compiled the
four-chip stage-local materializer and stopped before executing it. TPU HLO proves one BF16 round,
one FP32 promotion and zero collective/callback, but the linter counted parameters repeated inside
nested fusion computations as additional executable inputs: three raw-U8 and two scale parameters
instead of the single pair in `ENTRY`. There is no repair/cache comparison, DB row, terminal
`SUCCESS`, decoder, latency or Gate-D standing. Failure cleanup is authenticated 8/8 clean and the
diagnostics are archived under the approved bucket. Materializer-HLO gzip SHA is
`183f82a8...9a64a`; pre/failure-census SHAs are `d881b92a...a5276` and
`25a1d3a1...3c06`.

The narrow correction counts only parameters whose parsed computation begins `ENTRY ` while still
searching the complete module for BF16/FP32 conversions, collectives and callbacks. Replay of the
preserved TPU HLO now reports exactly one raw input, one scale input, one BF16 round, one FP32
promotion and passes. A synthetic nested-fusion regression preserves the same duplicate-parameter
shape and the focused prefill tests pass 13/13. Exact next: one blocker-only Fable audit of this new
two-file correction plus evidence notes, commit/push, then one fresh bounded LP4 retry. Do not
re-review `99ce5ea` and do not launch the full 8K decoder first.

## Second LP4 attempt proves materialization and exposes bounded root identity

Fresh bounded attempt
`greenfield_layer0_prompt_key_materialized_lp4_20260809T192658503682737Z` at pushed pin `2113b0d`
passes the corrected materializer HLO gate, executes the completed materializer, and compiles the
four-lane cache repair. It stops before repair execution because this direct harness names its JAX
root `mapped_repair`; optimized HLO consequently uses `jit(mapped_repair)/shard_map/...` instead of
the production decoder's repair branch metadata. The existing strict repair linter therefore sees
zero scoped operations even though the preserved module contains exactly four BF16-M64/FP32-wk
convolutions, eight physical `[64]` square roots, four physical affines, owner-cache scatters, zero
grouped square roots, zero collectives and zero repair-side weight rounds. A metadata-only replay
with the bounded root carrying `repair_stage_local_prompt_index_cache` passes the unchanged linter
with counts `4/4/8/4/8` for projection/exact operands/sqrt/affine/cache writes.

No repair arithmetic, cache comparison, DB row, terminal `SUCCESS`, decoder, latency or Gate-D
standing exists. Repair-HLO gzip SHA is `5576305c...05de`; pre/failure census SHAs are
`b7b38902...5dcf` and `6586bc71...9ebe`, with authenticated 8/8 clean failure exit. The minimal
correction renames only the bounded wrapper so JAX preserves the semantic repair identity; it does
not change arithmetic, sharding, the validator API, or the production decoder's strict scope.
Exact next: focused tests and one new-diff-only Fable audit, commit/push, then a fresh bounded LP4
retry. Do not run the full 8K decoder until the assembled LP4 cache is bitwise exact.

## Third LP4 attempt executes repair and exposes materializer-value drift

Fresh bounded attempt
`greenfield_layer0_prompt_key_norm_m64_20260809T193931471545587Z` at pushed pin `9cf4119` passes
both corrected HLO gates, executes the four-chip materializer and executes the owner-local cache
repair. Owner isolation passes, but the harness then refuses because a materialized FP32 `wk`
shard does not have accepted-adapter SHA `d680f7b1...83469`. The old message said the lanes differed,
but it stopped on the first lane and therefore does not yet distinguish bad raw placement,
lane-to-lane drift, or a common combined-materializer arithmetic drift. There is no cache exactness,
DB row, SUCCESS, decoder, latency or Gate-D result.

Materializer/repair HLO gzip SHAs are `764fe24f...6423b` and `397f04b8...0f085`; pre/failure
census SHAs are `9fe2527d...d2902` and `a3993b10...33130`. The approved diagnostic archive exists
and cleanup is authenticated 8/8. The next narrow batch records raw bits/scales and all four output
comparisons before refusing; it changes no arithmetic. After focused tests and one Fable audit of
only that new diff, run one protected diagnostic. Do not rerun the full decoder first.

The approved diagnostic retry
`greenfield_layer0_prompt_key_norm_m64_20260809T195128353553953Z` at pushed `e53d1fd` resolves
that ambiguity. All four lanes receive bitwise-exact raw bits SHA `8b18cfb9...c8b5e` and scales SHA
`463f1b22...8b9ee`; all four then produce the identical wrong FP32 SHA `b6429bf2...6e975` rather
than accepted `d680f7b1...83469`. Each lane differs in 698,727/786,432 values, first `[0,0]`, max
`0.0033482313`, mean `0.00006827695`, p99 `0.0004185438`. This rejects placement and lane drift and
isolates the combined materializer executable as the numerical boundary. Diagnostic JSON SHA is
`80e20cc8...9a18`; pre/failure census SHAs `22a6f95c...c12d` / `bd7a265b...bb0e4` prove 8/8 clean,
and the approved archive contains the diagnostic.

Exact next is a bounded two-completion proof: a stage-local raw-FP8-to-BF16 executable must finish,
then a separate BF16-to-FP32 executable must finish before unchanged repair compilation/execution.
Each phase has a strict entry/dtype/round/no-communication HLO contract. No production decoder retry
is authorized until that bounded cache is bitwise exact.

DB520/item1805, tag `greenfield_layer0_prompt_key_norm_m64_20260809T200559393031635Z` at
`0977022`, completes that proof. Both separate TPU executables pass: raw/scale -> BF16 HLO
`08b6c59f...ab2f9`, then BF16 -> FP32 HLO `1c107d68...b1f8`, with zero collectives/callbacks.
All four materialized shards equal accepted SHA `d680f7b1...83469`; owner writes are exactly
`[2048,2048,2048,2011]`; all 8,155 cache rows and producer states are bitwise exact, cache SHA
`3808d502...859d1`. Comparison manifest is `1e942555...08a59`, SUCCESS `643f80eb...083ca`, DB
snapshot `d466adc9...79581`, remote ledger `599ba9f1...affd`, approved archive and authenticated
8/8 pre/post cleanup pass. This is the required bounded arithmetic proof, not performance evidence.

The production successor applies those same two completed programs to the five local full-indexer
slots before unchanged repair. Exact next: focused tests, one Fable audit of only that production
diff, commit/push, then one protected full 8K retry through exact DSA and timing gates.

## Split-boundary 8K retry closes the repair regression; selected-runtime guard is next

Protected run
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_oracle_dsa_trace2_20260809T201759614361573Z`
ran from 20:18--22:33 UTC at pushed pin `5a41bb2`. It loaded the complete checkpoint, compiled the
78-layer prefill/decoder/observer, executed all 8,155 teacher-forced prompt tokens, and passed the
split materializer HLO contract: five stage-local slots, separate BF16 decode and FP32 promotion,
zero materializer collectives/callbacks, and zero repair-side weight round.

The first token remains exact at `101252`. Event 0 again has the exact 2,048-member selected set,
so the bad four-swap result from the combined materializer is removed. The observer still refuses
before warmup/timing: event 1 is the earliest set failure at seven swaps, identical in membership
to the pre-repair separate-qkv baseline, and later events diverge. There is no XPlane, wall
distribution, DB row, terminal `SUCCESS`, Gate-D or performance claim. Observation payload SHA is
`4fb4b087...2fc7c`; NPZ SHA is `427329b0...a6f`; eight logs are byte-identical at
`c0296290...b83e`; pre/failure censuses `2140efa1...4580` / `56f8622d...b2a8` authenticate 8/8
clean hosts.

The run used runtime manifest `54e2f89b...d9917`, the older separate q-a/kv-a artifact. It did not
exercise the already-proven fused N82 checkpoint (`12339490...699a`) together with the repaired
prompt-cache path. The fused-only full run had improved event 1 from seven to six swaps; DB502--504
prove its q-a/kv-a arithmetic, final layout, direct load and 78 physical convolutions, while
DB520 proves the repair's split materialization and layer-0 LP4 cache exactly. Their composition is
the next evidence-backed integration candidate, not a new arithmetic hypothesis.

The current narrow launch-safety batch makes the fused Gate-B artifact the selected linear-runtime
default, refuses protected repair with a separate-qkv runtime, and pins local/DB/remote DB520
evidence before fleet work. Exact next: focused/static tests, one new-diff-only Fable approval,
commit/push, idle-fleet proof, then one combined fused-qkv plus split-repair protected 8K run.

## Combined fused-runtime plus repair run isolates the remaining layer boundary

Protected run
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_qkva_oracle_dsa_trace2_20260809T225800140051438Z`
ran from 22:58--01:03 UTC at pushed pin `6b554c4`. All eight hosts read about 104.812 GB, compiled
the complete decoder/observer/prefill, and executed all 8,155 teacher-forced prompt tokens. The
fused runtime manifest is `12339490...699a`; decoder, observer-isolation, prefill and split
materializer contracts all pass. The repeated decoder contract remains exactly
`219AG/372AR/17CP` with eight local residual transfers and no forbidden full-pod reconstruction.

Token `101252` and its compact logit observation are exact. The DSA observer refuses before
warmup/timing: event 0 has exact membership but differs from legacy order first at offset 8; event
1 is the first set failure, with six expected/observed swaps, and later events diverge. Thus there
is no XPlane, wall distribution, DB row, terminal `SUCCESS`, Gate-D or performance result. NPZ,
token JSON, identical-rank-log and pre/failure-census SHAs are `34f4fe30...bbf`,
`e5e35f3b...03c`, `fb04f32a...cb0`, and `bedaf1bb...ccea` / `55f32a98...6fc`; cleanup is
authenticated 8/8 and the diagnostic prefix is in the approved bucket.

A direct event-by-event comparison to the earlier fused-only observation at `e4079ac` finds the
same selected-set membership at all 21 events and the same token observation. The repaired prefill
changes only small score/order bits and is not the source or solution of the remaining set drift.
Do not run another blind 8K candidate.

Reuse the existing non-donating all-event DSA-internal observer and the already-sealed accepted
layer-1 artifact instead of recapturing legacy state. Accepted layer-1 internals/comparison/seal
SHAs are `79b813da...9054`, `1bc43a8e...9ad5`, and `283e5e88...10d5`. The current batch makes
the observer baseline hash-selectable and admits it with repaired prefill while preserving the
residual-observer ban. Exact next: focused tests, one Fable audit of only this diff, commit/push,
then one serialized current-state internal observation. Compare event 1 to the sealed layer-1
capture and correct only the first divergent field before any Gate-D retry.

Fable's one-time read-only review returned `APPROVE COMMIT`. It independently confirmed that the
new combination only adds fail-closed observer conditions, every selectable artifact remains
SHA-checked locally/on all hosts/in Python, both features remain default-off, and the residual
observer ban is unchanged. No repeat review of this batch is authorized.

## Repaired-state observation moves the first divergence to the physical query dot

Protected run
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_oracle_dsa_dsa_internal_trace2_20260810T013247766447206Z`
at pushed pin `888cfcb` ran from 01:33--03:38 UTC. It reproduced the pinned fused-plus-repair DSA
observation byte-for-byte (NPZ `34f4fe30...bbf`, semantic SHA `267ffe90...a1e9f`) and refused before
timing. Layer-0 normalized hidden and q-a are now bitwise exact at SHAs `239e10d5...1765` and
`c9fbac05...c70c`. Query is the first divergence: 4,096/4,096 FP32 values differ, max
`0.0101393461`, actual `6fe17a94...355`, accepted `1ff2c2ec...12a`. Head weights and current key are
downstream. The internal NPZ/contract/log SHAs are `e1366c58...b50`, `b087aa92...bb2`, and
`03dc8b35...301`; the failure census is authenticated 8/8. There is no timing, trace, DB row,
`SUCCESS`, Gate-D/E or performance result.

Source/HLO inspection corrects the scope of DB499. Its process exposed four TPU devices but every
candidate used an ordinary unsharded `jax.jit`; the so-called LP4 candidate sliced four virtual
owners from entry `u8[4096,2048]` inside one device program. Its exact dot fusion consumes all four
completed `f32[1024,2048]` siblings together. The real decoder instead lowers one physical owner's
`f32[1024,2048]` and one 1,024-wide reduction independently per chip. DB499 therefore proves a
virtual grouped association, not the physical LP4 projection now required.

The existing DB499 harness is extended with `query_lp4`, a true four-device `shard_map`. It pins
the current observer and accepted q-a, shards raw or predecoded `wq_b` by physical owner, and
compares the current owner dot against an eight-head unrolled association. HLO rejects
communication, callbacks, global query weights, and dead rows. Exact next: focused/static tests,
one new-diff-only Fable approval, commit/push, then this bounded physical matrix. Do not run another
full decoder until one physical candidate is exact and its production integration is separately
proved.

Focused validation passes 15/15 with Python, Bash, ShellCheck, JSON and diff checks. Fable's
one-time read-only review independently compiled the real-geometry mapping on four forced devices,
verified head order/pins/HLO/DB/archive gates, and returned `APPROVE COMMIT`. Its only actionable
note was the corrected target-list error wording; no repeat review of this batch is authorized.

The first protected launch at `752d36e`, tag
`greenfield_layer0_physical_lp4_dsa_query_association_20260810T041439079476028Z`, passes both 8/8
idle censuses and compiles the physical raw owner-dot candidate. Its HLO SHA `0a8ba57c...8472`
proves local entry `u8[1024,2048]`/`f32[8,16]`, local completed `f32[1024,2048]`, output
`f32[8,128]`, four partitions and no communication/global table. The matrix then refuses before
the head-unrolled arm executes or any candidate result is persisted because the shared HLO linter
incorrectly requires a 1,024-wide intermediate for that arm, whose intended physical intermediate
is 128-wide. No candidate result artifact, DB row, `SUCCESS` or correctness/performance result
exists. Pre/failure census SHAs are
`fd4ed1ac...5710` / `33c967bb...9cad`; the partial diagnostic is in the approved bucket.

The narrow fail-closed correction makes the required projection width candidate-specific
(1,024 for owner dot, 128 for head-unrolled) and writes each optimized HLO before validation so a
future refusal preserves its exact cause. Arithmetic, sharding, tensors and evidence semantics are
unchanged. Run focused checks and one review of only this correction before a fresh-tag retry; the
already-approved parent batch is not reviewed again.

The correction retains 15/15 focused passes and all static checks. Fable's one-time review of only
this post-`752d36e` diff returned `APPROVE COMMIT`; its suggested logical-or-flattened 128-width
hardening and evidence wording clarifications are included. Do not re-review this batch.

## DB521 rejects physical dot variants and exposes an elided q-a BF16 boundary

Protected DB521,
`greenfield_layer0_physical_lp4_dsa_query_association_20260810T042459221979151Z` at pushed
`88350e3`, completed all four real four-chip candidates in six seconds. Raw versus predecoded
owner state and one 1,024-wide versus eight 128-wide reductions all produce the same query SHA
`eee61d94...bb`; none is exact and none reproduces current production. Against accepted, each has
2,728 FP32 mismatches/max `9.5367432e-7`; against current production it has 4,096 mismatches/max
`0.0101392269`. Tensor/evidence/SUCCESS SHAs are `7c2cf4c8...feb`, `86423acb...782`, and
`b2a43f13...852`; DB run 521, approved archive, direct object ledger and authenticated 8/8
pre/post cleanup pass. This is bounded correctness evidence only, with no decoder/Gate-D or
performance standing.

A direct selected-tensor audit of fused runtime manifest `12339490...699a` then rules out packed
state corruption: the four stage-0 slot-0 `wq_b` shards concatenate exactly to source SHA
`12f9ca94...e0`, scales to `0541bd9a...48d`, and head weights to `4dabc09e...624`; every shard
matches exactly one same-index source quarter and all mismatch counts are zero. The loader binds
files by physical device ID and refuses mesh/layout disagreement, so slot permutation is also
rejected.

The preserved full decoder HLO identifies the production-only boundary. Nominal q-a output is
correct BF16, but the query reduction consumes the FP32 q-a affine product before its BF16 convert:
the same fusion returns both `f32[1,2048]` to the query and `bf16[1,2048]` to the observer. DB521
starts from the latter sealed BF16 tensor. A new bounded `query_lp4_q_a_boundary` target composes
the proven fused-N82 helper with the physical owner query and tests unrounded default, explicit
BF16-barrier default, and explicit BF16-barrier HIGHEST associations. It requires exact q-a in
every arm, explicit StableHLO barrier counts, local-only optimized HLO and no callback/collective.
The real-shape four-forced-CPU program compiles/executes all arms; affected tests pass 74/74 plus
Python/Bash/ShellCheck/diff checks. Exact next: one Fable audit of only this new diff, commit/push,
idle-fleet census and one protected bounded target. Do not change production or retry 8K until the
unrounded arm reproduces current and a rounded arm is measured against accepted on TPU.

## DB522 proves the q-a round is causal; legacy physical head width remains

Protected DB522,
`greenfield_layer0_physical_lp4_dsa_q_a_boundary_20260810T051607194401685Z`, ran at pushed
`13123b8` and completed its three-arm four-chip matrix in 20 seconds. All three q-a tensors are
bitwise exact at SHA `c9fbac05...70c`. The unrounded/default arm reproduces the integrated current
query exactly at SHA `6fe17a94...355`, proving the bounded composition matches production. Both
explicit-BF16 arms produce DB521 SHA `eee61d94...bb`, reducing the accepted delta to 2,728 values
and max `9.5367432e-7`; neither is accepted-exact. Default and HIGHEST StableHLO differ as requested,
but TPU optimizes them to the same HLO SHA `a5b6742c...c76`. Thus the elided BF16 boundary causes
the large error, while precision flags do not close the residual.

SUCCESS/evidence/tensor SHAs are `99acefc0...157`, `4acf8eb3...231`, and `6d09ded4...1f2`.
DB522, approved archive, direct-object ledger and authenticated 8/8 pre/post cleanup pass. This is
correctness evidence only; DB484 remains the performance point and production is unchanged.

The remaining bounded hypothesis follows the actual architecture difference: accepted legacy TP32
computed one 128-wide query head per chip, whereas PP8 computes eight heads per chip. The existing
protected wrapper now has only two successors: an eight-execution single-head physical sweep to
prove the legacy-local N128 association, and one device-resident eight-iteration N128 `while` that
is production-compatible. The forced-four-CPU real geometry compiles both, retaining exactly zero
versus one loop and no communication/dead rows. Exact next: finish affected tests and evidence
notes, obtain one new-diff-only Fable approval, commit/push, then run this bounded head-geometry
target. The affected suite passes 75/75 in 131.86 seconds and all Python/Bash/ShellCheck/JSON/diff
checks pass. Do not wire the near-exact barrier or retry the full 8K decoder first.

## DB523 rejects physical N128 scheduling; explicit global-logical GSPMD is next

Protected DB523,
`greenfield_layer0_physical_lp4_dsa_head_geometry_20260810T053345314403882Z`, ran at pushed
`a3bd353` and completed in six seconds. The eight-execution physical single-head sweep and the
single device-resident eight-step loop are elementwise identical at SHA `bb4930ff...d3e`.
Both remain nonexact: 2,840 accepted mismatches/max `1.4305115e-6`, and all 4,096 values differ
from current production/max `0.0101393461`. Their HLO SHAs are `11edd83c...0ba` and
`6d458b78...583`; only the device-resident arm contains one loop. Neither communicates, carries a
dead row, or materializes a global query table.

SUCCESS/evidence/tensor/runner SHAs are `b34dfc6f...874`, `d7c4292b...301`,
`f73f773d...de8`, and `e4f4c3bb...a35`. DB523, approved archive, direct-object ledger and
authenticated 8/8 cleanup pass. This is correctness evidence only; DB484 remains the performance
point and production is unchanged. The result also equals DB499's virtual single-head/lax-map
candidate, so local N128 width or loop schedule is rejected.

DB499's accepted-exact candidates instead retained either the global logical M1/N4096 dot or a
fully unrolled global output. The remaining bounded discriminator therefore uses ordinary
`jax.jit` with explicit four-way `NamedSharding`: StableHLO must expose logical
`f32[4096,2048] -> f32[1,32,128]`, while each optimized physical partition must accept only
`f32[1024,2048]`, emit eight local heads, and contain no collective or global materialization.
Forced-four-CPU compilation already satisfies that structural contract. Exact next: finish this
coherent batch, run the affected suite and one new-diff-only Fable audit, commit/push, then execute
one protected bounded matrix. Do not retry the full decoder unless this candidate is exact.

## DB524 rejects four-way GSPMD partitioning; tuple-fused local reduction is next

Protected DB524,
`greenfield_layer0_physical_lp4_dsa_head_geometry_20260810T054953335423347Z`, ran at pushed
`70f549e` and completed in six seconds. Its explicitly four-way-sharded global-logical candidate
produces DB521 SHA `eee61d94...0bb`: 2,728 accepted mismatches/max `9.5367432e-7`. Thus ordinary
GSPMD preserves local ownership and zero communication but still lowers arithmetic as one physical
N1024 reduction; it does not preserve DB499's exact unpartitioned association. Optimized HLO SHA
`0ec1e68b...303` contains only `f32[1024,2048] -> f32[1,8,128]`, four partitions and no
collective/global table. StableHLO SHA `5a912167...f1c` proves the intended logical sharding.

SUCCESS/evidence/tensor/runner/results-DB SHAs are `e9e6de5f...df0`, `d6b6f4e9...8ae`,
`2f185e99...85c`, `aaa7e996...4e0`, and `6e22ccfd...9df`; DB524, approved archive,
direct-object ledger and authenticated 8/8 cleanup pass. This is correctness evidence only;
production and DB484 performance standing remain unchanged.

The causal HLO difference is now concrete. DB524's one-owner fusion requests 4,096 megacore
reduction bytes; DB499's exact four-owner tuple fusion requests 16,384 and uses a different TPU
reduction layout. The bounded successor passes the same local owner buffer through four explicit
aliases, retains all four N1024 reductions behind one optimization barrier, and returns only the
first local result. It carries no other owner's state, no collective and no global physical table;
the forced-four-CPU program retains all four dots and the barrier. Exact next: finish tests/docs,
one audit of only this new diff, commit/push, then one seconds-long protected result. No full model
retry is authorized first.

## DB525 is exact; production composition is the only pre-8K gate

Protected DB525/item1810,
`greenfield_layer0_physical_lp4_dsa_head_geometry_20260810T060457076587721Z`, ran at pushed
`a749ff0` and makes `physical_owner_tuple4_barrier_m1_n1024` bitwise equal to accepted query SHA
`1ff2c2ec...12a` with 0/4,096 mismatches. The same local FP32 owner is passed as four top-level
aliases; TPU retains one four-result N1024 fusion with `megacore_allreduce_bytes=16384`, then only
the first result remains live. StableHLO/optimized HLO SHAs are `3b10b7e5...9f7c8` /
`40d9ef25...4b7d`; there is no collective, callback, dead row, global physical query table or
other-owner state.

SUCCESS/evidence/tensor/runner/summary/results-DB/remote-ledger SHAs are `5b547f24...a6d582`,
`35f888c7...17624`, `c55a6638...790ce`, `da7acf8b...1dfc`, `5bba9e4e...1932`,
`41a43045...ef86`, and `e06bc413...3358`. DB/archive/direct-object linkage and authenticated 8/8
pre/post cleanup pass. This is bounded correctness evidence only; DB484 remains the only accepted
decoder performance point and Gate D/E remain open.

The current default-off production batch completes the causal composition: the proven fused-N82
q-a producer ends at an explicit BF16 barrier; a separate device-only materializer converts the
five stage-local raw-FP8 `wq_b` owners to FP32 (40 MiB/chip); four aliases of that one physical
state feed the DB525 helper for all 21 full-indexer layers. The HLO gate counts only tuple-valued
four-result 16-KiB fusions, not unrelated TPU fusions, and rejects communication/global tables.
Default execution is unchanged. Decoder plus prefill semantics and affected validation pass 75/75
on explicit CPU; Python/Bash/ShellCheck/diff checks are green before the final static batch.

Exact next: finish evidence/static checks, obtain one Fable xhigh review of only the diff from
cleared `a749ff0`, independently resolve any blocker, then commit/push. From an authenticated idle
fleet run only `query_lp4_production_exact`: first complete local raw-FP8-to-FP32 materialization,
then compile/execute the real fused q-a plus production tuple4 helper and require bitwise q-a/query
exactness with one local 16-KiB tuple fusion. Pin that protected result before any full 8K retry.

## First production-composition launch exposes a validator-only refusal

The production integration passed its one-time Fable review and was pushed as `7f636ed`. Protected
tag `greenfield_layer0_physical_lp4_dsa_query_production_exact_20260810T073812618515497Z` then
refused before materializer execution or query compilation. Its optimized materializer HLO is a
four-partition local `u8[1024,2048]`/`f32[8,16] -> f32[1024,2048]` program with no collective,
host callback or global table. The only custom-call targets are TPU's compiler-internal bounded
gather markers `AssumeGatherIndicesInBound` and `GatherScatterIndicesBitpacked`; the new validator
had incorrectly rejected every `custom-call` opcode.

Optimized/StableHLO SHAs are `103cd550...a9f71` / `f99669ea...6002d`; pre/failure census SHAs are
`ce3f1293...e3e31` / `08b777ac...2f7e`. The diagnostic is in the approved bucket and authenticated
8/8 cleanup passes. There is no materialized value, query result, DB row, `SUCCESS`, latency or
Gate-D/E evidence.

The narrow correction allowlists only those two metadata targets while continuing to reject every
collective, outfeed, host callback, unknown/Pallas custom call, global owner shape and partition
drift. It also persists the complete materializer HLO contract before refusal. The exact preserved
TPU HLO now passes that corrected contract; a synthetic `tpu_custom_call` remains rejected. Exact
next: affected/static tests and one Fable review of only this post-`7f636ed` correction, then
commit/push and retry only the same bounded target. Do not launch the full 8K decoder first.

## DB526 closes production composition and authorizes the protected 8K retry

Protected DB526/item1811,
`greenfield_layer0_physical_lp4_dsa_query_production_exact_20260810T080508327295662Z`, ran at
pushed `3aa9f9c` and completed the actual production composition in 19 seconds. The fused N82 q-a
output is bitwise exact at SHA `c9fbac05...cc70c`; the completed local materializer plus four-alias
tuple helper produces accepted query SHA `1ff2c2ec...cb12a` with zero of 4,096 mismatches.

Production StableHLO/optimized HLO SHAs are `4e7f3dc3...190f` / `78c29674...069c`. The optimized
program has one scoped tuple-valued four-reduction fusion, four partitions, one live row and no
collective/global query table. The local materializer completed 8 MiB/chip; its optimized HLO SHA
is `2cc9283a...44b3`, uses four partitions, and has only the two approved bounded-gather metadata
calls with no host marker, collective or global owner state.

SUCCESS/evidence/runner/summary/tensor/results-DB/remote-ledger SHAs are
`b3cff36b...3b11`, `82219191...2b7d`, `497dd606...96f2`, `890a9d8e...212e`,
`b371ad77...d247`, `caa8ff3f...03dd`, and `980f4f78...e4e3`. The same-region archive, direct
remote `SUCCESS`, DB integrity and authenticated 8/8 pre/post cleanup pass. This is bounded
correctness/HLO evidence only, not decoder latency or Gate D/E.

The current batch pins both DB525 mechanism evidence and DB526 production-composition evidence in
the full launcher, including local/DB/remote hashes and materializer/query contracts. Exact next:
focused/static tests and one new-diff-only Fable approval, commit/push, authenticated idle fleet,
then one protected combined 8K run with fused-qkv, split prefill repair, exact query association,
token/DSA oracle and trace. That full run is expected to take about two hours; do not call its
load/compile/prefill elapsed time token latency.

## First exact-query 8K compile passes arithmetic gates and finds one metadata-name omission

Protected tag
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_oracle_dsa_trace2_20260810T082522620015034Z`
ran at pushed `1824cf8` from 08:25--08:34 UTC. All eight hosts loaded the complete runtime and
compiled the 78-layer exact-query decoder. The decoder HLO, five-slot query materializer and split
prefill-weight materializers were preserved before execution. The run refused before prefill,
tokens, timing or tracing on exactly one linter violation:
`complete-token return source operation drifted`.

The return is the intended single `s32[1]` permute over the exact 32 stage-lane pairs, with the
same one local BF16 score exchange and one local S32 id exchange. Its source metadata is
`jit(mapped_token_exact_query)/shard_map/ppermute`; the linter admitted only the default direct
name `jit(mapped_token)/...` and the shared prefill-loop name. Every other decoder contract section
passes, including the exact query association. This is a feature-name omission, not a topology or
arithmetic drift.

Decoder-HLO-gzip/contract/query-materializer-contract/prefill-materializer-contract SHAs are
`86030c36...55cf`, `5a55efd4...da32`, `1ba4d7f7...a37b`, and `9c345ff4...e3df`. All eight rank
logs are identical at `bc65b0e7...b27e`; pre/failure census SHAs `5ec2756d...df37` /
`05762594...d7fa` authenticate 8/8 cleanup, and the approved diagnostic archive exists. There is
no DB row, `SUCCESS`, raw-token, latency, XPlane or Gate-D/E evidence.

The narrow correction makes the accepted direct token-return source name conditional on the
already-pinned `dsa_query_exact_association` flag. Exact mode accepts only
`mapped_token_exact_query`; default mode accepts only `mapped_token`; both retain the one shared
prefill-loop name. Replaying the exact preserved TPU HLO now passes the complete-token sub-contract,
and cross-mode names fail. Exact next: focused/static tests, one Fable review of only this
post-`1824cf8` correction, commit/push, then retry the same protected 8K profile.

## The internal observer isolates head/key; DB527 proves the exact correction

The protected run at pushed `fb1dea9`, tag
`greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_oracle_dsa_dsa_internal_trace2_20260810T093638877848825Z`,
completed full-fleet load/compile, the 8,155-token prefill and one observed recurrent step in 49
minutes. Token `101252` is exact. The separate device DSA observer then stopped before warmup,
timing or trace because all-event sets are nonexact. Its layer-0 internal breakpoint proves
normalized hidden, q-a and query bitwise exact, while head weights differ 32/32 (max
`2.4797022e-4`) and current key differs 97/128 (max `7.1525574e-7`). Internal NPZ/contract SHAs are
`a889b664...ed07` / `fb470de5...fb18`; all eight logs share SHA `ddca2f9a...eb2`, and pre/failure
census SHAs `eacd64d3...1018` / `78e9aba0...90f2` prove clean shutdown. There is no DB row,
`SUCCESS`, accepted latency or Gate-D/E result.

Protected DB527/item1812 at pushed `cd15aaf`, tag
`greenfield_layer0_physical_lp4_dsa_head_key_boundary_20260810T104647319991568Z`, replays that
exact observer input across five physical LP4 arms in six seconds. Exactly
`physical_normalized_barrier_materialized_divide_sqrt` matches accepted head SHA
`ec66b475...725e` and key SHA `9f1fb991...dbbc5` with zero mismatches. It uses one BF16 normalized
boundary, the already-materialized local FP32 wk owner and divide-by-sqrt key LayerNorm. The tuple4
key anchor is rejected. Its HLO has four partitions, two StableHLO dots, one barrier and no
collective/global owner table. SUCCESS/evidence/runner/tensor/summary SHAs are
`0c961dd1...5fb0`, `7c7563e7...7678`, `f4d2518c...1310`, `3ea5813f...b5b52`, and
`e76e799b...c1ea`; DB/archive/direct-object and authenticated 8/8 cleanup pass.

The current default-off production batch reuses the existing five-slot prefill wk materializer;
it adds no checkpoint state or new observer. It applies DB527's normalized barrier and divide/sqrt
only to the 21 full-indexer layers, pins the recurrent HLO counts, and keeps the existing internal
observer enabled for the first deployment. The full affected suite passes 82/82. Pre-deployment
review found and corrected two wiring blockers: prefill now forwards the exact head/key HLO flag,
and observer/warmup/timing/trace calls reuse the same nested query/WK PyTree used at compile time.
Focused regressions cover both and the correction-only Fable follow-up returned `APPROVE COMMIT`.
Exact next: commit/push, authenticate the idle fleet, then run the combined protected 8K profile.
Best case is roughly three hours from clean launch; one evidence-led correction/retry makes the
realistic Gate-D window four to eight hours.
