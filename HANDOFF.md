# HANDOFF — topology-first greenfield rewrite

**Updated:** 2026-08-05 17:38 UTC

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
- Greenfield tests last passed 86/86. Both PP8 and PP16 exact sparse-layer prerequisites are proven.
  Representative dense/DSA/IndexShare layers, full Gate-B checkpoint/loader, decoder, serving, and
  genuine token throughput remain unimplemented.

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

1. Begin Gate B for PP8: implement the complete byte-balanced stage assignment, memory budget, and
   plan-aware final-layout manifest before writing checkpoint payloads. Preserve IndexShare groups
   where feasible and account for weights/scales/KV/overlays/temporaries per chip.
2. Extend the bounded pack/direct-loader contracts to the complete checkpoint only after the
   manifest reconciles every source leaf and all 32 final owners. Fail closed on corruption,
   incompleteness, non-finite values, stale plan/code, or unmeasured HBM margin.
3. Continue Gates C–H in binding order. The one-layer comparison provisionally favors PP8 for
   normal routing (`0.696` vs `0.901 ms`) while PP16 wins the concentrated adversary
   (`1.076` vs `1.134 ms`); only complete protected decoder evidence may choose the
   final plan.

The pod ended the latest proof with all eight hosts `CENSUS_OK`.
