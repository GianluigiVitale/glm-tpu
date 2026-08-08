# Greenfield evidence and reusable protection map

The cross-repository reuse decisions and pinned implementation candidates are indexed in
[`REUSE_INVENTORY.md`](REUSE_INVENTORY.md) and
`configs/greenfield-reuse-inventory.json`. They define what is directly reused, independently
adapted, oracle-only, rejected, or reserved for a later gate; they do not weaken the evidence rules
below.

## Accepted protected greenfield evidence

Every accepted run has a local directory under `/home/gianl/glm-run`, a same-tag archive under
`gs://driftbench-dsv4-uc/results/`, append-only `bench/results.db` linkage, local/remote `SUCCESS`,
fleet agreement, and eight-host clean pre/post census.

| DB | tag | evidence |
|---:|---|---|
| 405 | `greenfield_topology_20260805T125842425591441Z` | physical `2x4x4` topology and PP8/PP16 rings |
| 406 | `greenfield_collectives_20260805T133905344573798Z` | `bf16[2,6144]` control/all-reduce, g2/4/8/32 |
| 407 | `greenfield_collectives_20260805T134254891049866Z` | dominant-payload all-gather matrix |
| 408 | `greenfield_collectives_20260805T134731771265066Z` | asynchronous collective-permute HLO validation |
| 409 | `greenfield_collectives_20260805T135545030179024Z` | fused tuple all-reduce HLO validation |
| 410 | `greenfield_collectives_20260805T135644529157649Z` | dominant-payload ppermute/all-to-all/fused tuple matrix |
| 411 | `greenfield_collectives_20260805T135850389312854Z` | supported `bf16[1,6144]` six-operation matrix |
| 412 | `greenfield_collectives_20260805T140125151247631Z` | supported `bf16[1,2048]` six-operation matrix |
| 413 | `greenfield_collectives_20260805T141114991474088Z` | supported `f32[1,6144]` six-operation matrix |
| 414 | `greenfield_collectives_20260805T141333601664455Z` | `int32[1,2048]` routing-metadata five-operation matrix |
| 415 | `greenfield_transport_20260805T142953361259007Z` | PP8/PP16 four-payload transport distributions and exact HLO |
| 416 | `greenfield_transport_trace_20260805T143832942547470Z` | fresh PP8/PP16 8-file/64-core/20-step XPlane proof |
| 417 | `greenfield_real_layer_pp8_20260805T165737737514245Z` | exact real layer-3 PP8 correctness/HLO/HBM/wall/XPlane proof |
| 418 | `greenfield_real_layer_pp16_20260805T172807177182695Z` | exact real layer-3 PP16 correctness/HLO/HBM/wall/XPlane proof |
| 420 | `greenfield_full_checkpoint_load_pp8_20260805T201317772639405Z` | complete PP8 base-decoder final-owner direct-load/integrity/HBM proof |
| 421 | `greenfield_gate_c_pp8_20260805T224645828157364Z` | real PP8 dense/full-DSA/IndexShare correctness/HLO/HBM/XPlane proof; no performance claim |
| 422 | `greenfield_fp8_matmul_20260806T015232890679994Z` | raw-FP8 block matmul exactness/HLO/HBM/wall proof |
| 423 | `greenfield_fp8_up_gate_20260806T020551714072561Z` | paired raw-FP8 gate/up exactness/HLO/HBM/wall proof |
| 429–430 | `greenfield_fp8_selected_up_gate_*_20260806T025*` | selected gate/up normal/concentrated route-proportional proof |
| 436–437 | `greenfield_fp8_selected_swiglu_down_*_20260806T034*` | selected SwiGLU/down normal/concentrated proof |
| 438 | `greenfield_real_layer_pp8_pallas_20260806T050514347248323Z` | exact final-layout raw-FP8 PP8 MoE layer; correctness/locality pass, latency rejected |
| 439 | `greenfield_real_layer_pp8_pallas_20260806T052141734174170Z` | fused routed raw-FP8 PP8 layer; exactness/locality pass, 1.3–1.5% wall win, performance rejected |
| 440 | `greenfield_real_layer_pp8_pallas_20260806T052955364574577Z` | fully fused routed/shared raw-FP8 layer; all protection gates pass, 0.8–1.5% regression vs DB 439, rejected |
| 441 | `greenfield_real_layer_pp8_pallas_feature_20260806T055854589778101Z` | feature-sharded raw-FP8 PP8 layer; exactness/locality pass, route-skew removed, selected for decoder integration |

Topology code is `75c8bb14...`. Collective matrix pins are `fcd8426735...` and
`b12af9633c8b14648db8d2a2ccd9a3c577a04817`. Topology hash is `294e777...559`, PP8 group hash
`d5943ab8...c14`, and PP16 group hash `6383e57c...f21`.

Superseded or failed diagnostics are preserved but not promotion evidence. DB 404 used a first
non-neighbor PP8 ordering. Reduce-scatter diagnostics `...T134618415607642Z`,
`...T135045281384327Z`, and `...T135140118884391Z` prove XLA rewrite-to-all-reduce and contain no
accepted latency. Earlier ppermute/tuple diagnostics led to the async-HLO and dtype fixes.
Transport trace `...T143608529930365Z` failed validation because worker-0's original XPlane and its
canonical downloaded copy occupied the same parser tree; it has no DB row or accepted status. DB
416 uses an isolated fleet directory and proves eight distinct embedded hostnames per plan.

## Durable checkpoint-layout artifact

`greenfield_one_layer_pack_20260805T151828912346032Z` is the first real bounded layer-3 artifact.
At code `1969d925...f3d4`, it validated 1,544 leaves / 9,706,940,416 unique bytes from only source
shards 38–40 and wrote four final PP8 files of 2,429,096,640 bytes each. Packed payload
9,716,380,672 reconciles the intentional router replication. Manifest SHA-256 is
`68ef82011892456409a194f6fa31697dd1e31d96fe1a3f0069228288f613f938`; local and approved-bucket
`SUCCESS` exist. This is checkpoint-layout mechanism evidence, not a DB-linked TPU performance run
and not Gate B for the complete model.

`greenfield_one_layer_oracle_20260805T162210370718434Z` is the independent correctness artifact at
code `27ebdec`. It reads 104 exact raw-source tensors at the same immutable source revision as the
pack, never constructs a model, and records accepted legacy/vLLM file hashes. Normal routes
`[161,217,206,240,186,180,37,81]` span all four PP8 slots; the adversarial case selects only experts
128–135 on slot 2. Manifest is `c63ffa19820d5c2c39865ac8611fb313ffc6ebcd2f893c3507745a356bfdebff`;
the 274,944-byte safetensor SHA-256 is `4aa7910b...784b`; local/remote `SUCCESS` exist. This is not
TPU performance evidence.

`greenfield_one_layer_pallas_pack_20260806T041854316280053Z` is the exact final-access-layout
derivative at code `9f42e23`. It transforms only the three routed FP8 table orientations, binds
every source/destination tensor hash, and preserves shared/router bytes. Its four files are each
2,429,096,824 bytes; payload is 9,716,380,672 bytes, layout is `3d9f3b85...545e`, and manifest is
`3da63bd9c2332dd67fc29a1d158e5468e0fdf1db1a9a0e8b7977277b0812e427`. Local and approved-bucket
`SUCCESS` pass. DB 438 is the protected execution bound to this artifact.

`greenfield_one_layer_pallas_feature_pack_20260806T054520020812918Z` is the selected structural
PP8 derivative. It maps every routed expert identity to every stage chip while slicing the 2,048
intermediate dimension four ways; reciprocal down ownership preserves the single local combine and
the per-chip persistent payload. Manifest is `a8b914350ea7b8fd281e425d6eb49eefad2082b48f16ec919c26ccbc7bbb5cc6`,
layout is `e613d9ef655f7add11f8b9fd9e0fd630343c09b714882f967c405986cc6cc431`, payload is
9,716,380,672 bytes, and source Pallas manifest is `3da63bd9...e427`. Exact transformation,
file/payload hashes, local/remote `SUCCESS`, and direct loading with zero runtime dequantization,
concat, or transpose pass. Protected DB 441 is bound to this artifact.

`greenfield_runtime_feature_pack_pp8_20260806T064010287072141Z` extends that selected ownership to
the complete PP8 base decoder. Pack code `d9a883b` writes 32 files / 834,178,632,960 file bytes,
834,177,357,824 payload bytes, 84,054,798,080 padding bytes, and 26,068,042,432 runtime weight
bytes/chip. Runtime manifest `54e2f89b...d9917`, layout `ba21c4ec...c9e`, plan
`f46f91c3...826a`, and schedule `b407fcf5...1773` bind 14,640 source tensor uses to 11,648 final
tensor records and protected source runtime `fdedaae3...e31dec`. Mounted verification, GCS
generation/CRC32C checks, checkpoint/result `SUCCESS`, approved archive, and 8/8 cleanup pass. This
is the executable final-layout checkpoint selected for the PP8 feature decoder; it is not TPU
execution, decoder latency, or token-throughput evidence.

`greenfield_gate_c_oracle_20260805T212801776974822Z` is the independent layer-2/3 Gate C
correctness artifact at code `602d42f`. It reads 31 raw tensors from source shards 20/38/40 at the
protected full-pack source revision and constructs no JAX/model/legacy execution. Its real
2,304-position full DSA scorer selects 2,048 unique positions including current position 2,303;
the layer-3 consumer carries the identical score-ordered `int32[1,2048]` payload (8,192 bytes),
sorts a private attention copy, and captures write-before-attend sparse MLA plus a real dense
layer-2 case. Manifest SHA-256 is `54262529bd561c57f0833a993e0ac6cdbec20b9e270a0f6f1726d0049d9c4a9f`;
the 36,060,088-byte safetensor SHA-256 is
`a8927db679ed74b9adc38d516fab4c4985acc5dc7489d6379e28b2ba52270364`; local/remote `SUCCESS` and
the local evidence ledger pass. This is correctness input, not TPU or performance evidence.

`greenfield_gate_c_pack_20260805T214609093206269Z` is the bounded exact-final-owner derivative at
code `8a50d6a`. Its 31 placements are copied from protected full PP8 layout `aca0eb6d...6a0`, and
their raw bytes must match independent oracle `54262529...4a9f` while streaming. Unique source,
packed payload, and packed file totals are 413,810,816, 502,446,080, and 502,461,536 bytes. Each of
the four stage-0 owners has 31 leaves and a 125,611,520-byte payload. Subset layout is
`cdbea04f...c678`; packed manifest is `3c5c48da...2a8a`. The local evidence ledger and all payload
hashes pass; approved-bucket size/generation/CRC32C and local/remote `SUCCESS` pass. It is bounded
checkpoint-layout evidence only, not a protected TPU/DB/performance result.

## Protected Gate C dense, DSA, and IndexShare proof

DB 421 / `greenfield_gate_c_pp8_20260805T224645828157364Z` runs on captured physical PP8 stage 0
at exact code `dc20b3fea3089dd3fbedd2a6cc19e86b0416f60f`. The direct loader binds packed manifest
`3c5c48da...2a8a`, layout `cdbea04f...c678`, and independent oracle `54262529...4a9f`; it transfers
502,446,080 final-owner bytes, performs 44 device dequantizations, and records zero host FP8
dequantizations, host global concatenations, or runtime checkpoint reshards.

Dense, full 2,304-position DSA, and IndexShare all pass their documented tensor contracts. Dense
output max error is `0.00390625`; DSA score max/mean error is `0.003605/0.000965`; IndexShare
output max/mean error is `0.0078125/0.000167`. The distributed top-k and lowest-position tie order
are elementwise exact for the actual TPU FP32 score row. The resulting score-ordered
`int32[1,2048]` state is fed directly into IndexShare and remains bitwise unchanged; private
attention order, selected cache, write-before-attend, live cache, and health state all pass.

The raw PyTorch CPU scorer differs slightly from the TPU FP32 scorer and changes two cutoff
members (2,046/2,048 set overlap). This cross-framework difference is retained as a failing
elementwise diagnostic, never used as runtime state, and never hidden by a tolerance. Internal
scores and final outputs remain inside the recorded bounded-error contract. Thus DB 421 proves
exact selection semantics for actual device scores, not raw cross-backend position identity.

Optimized HLO SHA-256 values are dense `643efea4...fc7c`, DSA `a6cf14bb...0af5`, and IndexShare
`302a6ece...ab70`. Their local collective counts are respectively `0AG/1AR`, `3AG/0AR`, and
TPU-rewritten `2AG/3AR`, always over `{{0,1,2,3}}`; no full-pod group or dead batch-32 tensor is
present. The fresh XPlane has 20 invocations/case on all eight TPU cores and matches the exact
HLO-derived physical counts. Peak bounded-proof HBM is 280,745,984 bytes/chip. The sealed hash
ledger, SQLite integrity, same-tag approved archive/remote `SUCCESS`, and 8/8 pre/post census pass.
This artifact deliberately contains no latency or token-throughput claim.

## Protected exact PP8 real layer

DB 417 / `greenfield_real_layer_pp8_20260805T165737737514245Z` is the first real checkpoint-backed
greenfield TPU layer result. At exact code `db19893aa8241cc3559f1f174400f9094b69fc78`, the direct
loader verifies pack/oracle/source identities, transfers final-owner shards, and dequantizes on
device with no host FP8 dequantization or global tensor concatenation. Normal and adversarial
concentrated routing are exact. Output max/p99 error is `0.03125/0.01171875` in both cases.

Profiler-free p50 over 1,000 samples is `0.696215 ms` normal and `1.134090 ms` concentrated. HLO SHA
`950b5eb2eaccf64341cb2d90e749365795c3f5cba2aee10cae929c1a1866977d` has exactly one local
`bf16[2,1,6144]` four-rank all-reduce and no other collective. Peak HBM is `5,639,681,536` bytes on
every participating chip. A fresh post-timing XPlane has 20 steps on all eight TPU cores and one
physical `psum` per step. Evidence hashes, DB snapshot/integrity, same-tag approved archive,
local/remote `SUCCESS`, and eight-host pre/post clean census pass. This proves the PP8 real sparse
layer prerequisite only; it is not a full Gate-C pass and is not model token throughput.

## Final-layout PP16 layer artifact and protected proof

`greenfield_one_layer_pack_pp16_20260805T172003732526347Z` is the two-chip final-ownership
artifact at pack code `51d1df96...e60b`. It uses the same immutable source revision and writes
two 4,855,045,080-byte files: 128 complete experts and shared-intermediate width 1024 per owner.
Source payload `9,706,940,416` and packed payload `9,710,087,168` bytes reconcile,
including declared router replication. Manifest SHA-256 is
`385737230d593d6b2daa46911d1ac31973d9b79a7d43c3353cedf6d9779fe454`; both file hashes and
local/remote `SUCCESS` pass. This remains bounded layout evidence, not full Gate B.

DB 418 / `greenfield_real_layer_pp16_20260805T172807177182695Z` is the mandatory protected
two-chip challenger at exact code `6751a935...f820`. It binds captured stage 10, global ids
`[4,5]`, and physical coordinates `[(0,2,0),(1,2,0)]`. Normal/concentrated p50 over
1,000 profiler-free samples is `0.900610/1.075795 ms`. Route ids are exact; normal output
max/p99/mean error is `0.015625/0.0078125/0.002121`, and concentrated is
`0.03125/0.0078125/0.002140`.

Optimized HLO SHA `f9a97fbb56fafc6cf17446da23bd2550e7d41c59f2d8633b40e2c0ea83f8d051`
contains exactly one `bf16[2,1,6144]` two-rank all-reduce and no other collective. Peak HBM
is `11,276,493,312` bytes/chip. A fresh post-timing 4-core/20-step XPlane has exactly one
physical all-reduce on every core-step. Evidence hashes, DB integrity/snapshot, approved archive,
local/remote `SUCCESS`, and eight-host clean pre/post census all pass.

Rejected diagnostic `greenfield_real_layer_pp16_20260805T172345348528631Z` has no DB/status
claim. libtpu rejected a standalone local `2x1x1` slice for visible devices `0,1` with
duplicate coordinate assignment. Archived driver logs preserve the cause. The accepted run
initializes the proven four-chip host subcube but places the PP16 arrays/executable only on adjacent
devices `0,1`; its HLO and XPlane prove only the two-chip stage executes.

## Complete PP8 checkpoint and protected direct load

The complete inventory/plan/layout chain covers 141 source files, 118,629 leaves, and
755,617,140,416 source payload bytes. Inventory, plan, execution, and layout hashes are
`a388627c...2fc4`, `c5bfacc3...c4ed`, `23fd23f8...5c89`, and `aca0eb6d...6a0`.
The real packed artifact `greenfield_full_pack_pp8_20260805T182222755355852Z` contains 32 base and
four MTP final-owner files, 760,215,571,712 payload bytes, and packed manifest
`0869493164a3a63797ea61d88c575f35bea8aa50790c46aa21ce6f0f7c4c78f1`. Its approved checkpoint
prefix has `SUCCESS`.

DB 420 / `greenfield_full_checkpoint_load_pp8_20260805T201317772639405Z` at exact code
`91a8c4704c4b435e58c654e48d112e5b2a5b4885` directly loads the 32 base owners on eight hosts / 32
chips. All 122,640 final tensor shards and 750,122,559,744 bytes pass identities, finite checks,
device round trips, stage/physical-owner checks, and zero host FP8 dequant/global concat/runtime
reshard assertions. Maximum weights-only peak HBM is 24,840,958,464 bytes/chip; minimum reported
free space is 8,173,454,848 bytes. The checksum ledger passes after completion, SQLite integrity is
`ok`, three censuses contain exactly eight authenticated hosts, and a fresh download of all 47
remote files is byte-identical to local evidence.

This passes Gate B's checkpoint criteria, not decoder-memory or performance gates. KV/cache,
executable, overlay, and decoder-temporary HBM are not included; `promotion_memory_proven=false`.
No token/s or model-latency claim is attached to DB 420.

DB 419 / `greenfield_full_checkpoint_load_pp8_20260805T194526167625636Z` is explicitly rejected as
promotion evidence. Its mechanical load completed, but the first harness appended a success line to
`orchestrator.log` after hashing it, so the local `evidence.sha256` ledger did not seal. Historical
files and DB row remain untouched; the corrected fresh proof is DB 420.

## Reusable tools, not execution dependencies

Gate-D code state `a2638a2` is deliberately not listed in the protected table. In addition to the
all-layer schedule/state and raw-FP8 kernels, it provides an executable-ready layout that exactly
reconciles DB420's 32 files / 122,640 leaves / 750,122,559,744 bytes, a bounded-memory derivative
streamer, a fused full layer, and an all-stage decoder-step map. Forced-CPU differential and
32-device HLO/state evidence pass, but no production runtime pack or complete decoder has run on
TPU and no DB, measured decoder HBM, latency, or tok/s claim attaches to it.

- Optimized-HLO contract: `glm_tpu/greenfield/sharding/hlo_contract.py` and
  `scripts/greenfield/inspect_hlo_contract.py`.
- Dependent collective mechanism: `glm_tpu/greenfield/benchmarking/collective_chain.py` and
  `scripts/greenfield/microbench_collectives.py`.
- Protected launcher: `scripts/greenfield/run_collective_chain.sh` (exact pin, lease, census,
  fleet HLO/checksum agreement, DB, archive, cleanup).
- Transport implementation/proof: `benchmarking/transport_chain.py`,
  `microbench_pipeline_transport.py`, `run_pipeline_transport.sh`, and the separate profiler-
  contaminated trace script/launcher.
- XPlane/wall truth: `scripts/analysis/parse_xplane.py` and `extract_steady_decode.py`.
- Append-only provenance: `bench/provenance.py`; greenfield hashes reside in `env_json` pending a
  dedicated schema field.
- Oracle-only integrity interfaces: `load_state_hash.py`, `write_probe.py`, `dsa_topk_dump.py`, and
  `dsa_topk_diff.py` in the pinned legacy checkout. Adapt interfaces without importing execution.

## Legacy measurement oracle

Accepted legacy parent: `287.666063 ms/device token`, `3.476` device tok/s, about `3.3` wall tok/s.
Its 75 sequential physical 32-chip MoE combine regions cost `106.495 ms/token`. The greenfield
collective floor and both real-layer plans show this is not a raw 12 KiB ICI or sparse-layer compute
floor, but no greenfield full-decoder token-speed result exists yet.

## Rejected complete feature-body evidence

`greenfield_short_decoder_compile_pp8_pallas_feature_20260806T084346269707216Z` at `a8194cd`
preserves eight host records, optimized HLO `64df6dc7...2ea0`, exact local collective/kernel
contracts, correct decoder metadata, measured HBM, and authenticated clean failure census. Body
p50/p99 is `58,804.040/58,804.323 ms`. The outer finalizer failed on a missing explicit-zero loader
metric, so there is no DB row or remote `SUCCESS`; this is rejected diagnostic evidence only.

## Protected complete feature-body attribution

DB 442 / `greenfield_short_decoder_compile_pp8_pallas_feature_trace2_20260806T092025101122999Z`
at `0cd5209` seals one profiler-free 78-layer/2K body sample (`58,804.002894 ms`) followed by eight
fresh XPlanes / 64 cores / two steps per core. XPlane mean device step is `56,722.255839 ms` and
busy time is `54,643.549475 ms`. Compact stage-permute regions account for `47,294.061096 ms` of
waiting while whole-matrix FP8 dequant gathers account for `7,317.697973 ms` per average core.
Eight serial stages predict `58,541.584 ms`, identifying reference weight dequantization as the
critical path rather than transfer bandwidth. Attention o/q_a/q_b are the top three callers.
Exact local HLO, feature kernel counts, HBM, hashes, approved archive/remote `SUCCESS`, DB linkage,
and clean 8/8 census pass. This is attribution evidence only, not Gate D or token-speed evidence.

## Protected production Pallas DSA scorer

DB 443 / `greenfield_dsa_score_20260806T095455945075126Z` at `d068a9f` proves a one-row
`f32[1,32,128] x bf16[65,536,128] -> f32[1,65,536]` Pallas scorer for one 256K/LP4 shard. HLO
`5e2b7185...b295` has exactly one kernel, no `[32,context]` overlay, batch-32 row, collective, or
unexpected custom call. TPU/reference score max error is `2.861e-6`; all 2,048 selected positions
and order are exact. Profiler-free p50/p99 is `0.326595/0.350320 ms` over 1,000 samples after 200
warmups. HBM, DB snapshot, hashes, approved archive/remote `SUCCESS`, and 8/8 cleanup pass. This
closes the standalone Section 7.2 scorer rung, not its layer integration or token performance.

## Protected exact Pallas DSA top-k

DB 445 / `greenfield_dsa_topk_20260806T102752126905724Z` at `3870c2f` proves the production
one-row local `65,536 -> 2,048` selector and permuted four-owner `4x2,048 -> 2,048` merge on TPU
v4. Exact high-score ties, lowest-global-position order, scores, positions, valid counts, and
sentinels match both TPU JAX and independent host oracles elementwise. Local/merge p50/p99 is
`1.364405/1.388090` and `0.337671/0.362475 ms` over 1,000 samples after 200 warmups. HLO SHAs
`e6b8e209...e7e90e` / `d990a754...0b674` contain exactly six/two TensorCore Pallas calls and no
XLA sort/top-k, collective, dead row, or unexpected call. Peak HBM, DB snapshot, hashes, approved
archive/remote `SUCCESS`, and 8/8 cleanup pass. DB 444 is the exact but rejected reduction
baseline at `59.979532/4.495320 ms` p50. These are standalone selector results, not layer/token
throughput.

## Rejected layer-0 DSA internal capture attempt

`greenfield_legacy_layer0_dsa_internals_20260808T005359078558816Z` failed during warmup tracing
before any request because the observer attempted `jnp.asarray` on an outer-JIT torchax tracer.
It has no observer payload, token, DB row, comparison, final `SUCCESS`, or performance claim. The
exact runtime was stopped, the failure-exit census is eight-host clean, and 11 diagnostic objects
(2,123,660 bytes) are verified under the approved bucket's `dsa_internals/8k/failed/` prefix.
Oracle-only correction `83ff4a357` reuses the scorer's zero-copy `as_jax` boundary and passes
19 focused forced-device/torchax tests. This is failure/fix evidence only; the corrected protected
capture is still required.

## DSA query association

- Accepted live-state source: source DB493 plus authenticated recovery artifact
  `greenfield_legacy_layer0_dsa_internals_recovery_20260808T030853085141329Z`; normalized hidden
  and q-a state exact, first divergence `query`.
- Exact bounded correction: DB499 / item linked in `bench/results.db`, run
  `greenfield_layer0_dsa_query_association_20260808T034636385240375Z`, code `b41c3ab`.
- Numerical proof: raw-materialized PP8 local owner has zero/4,096 mismatches; HLO
  `ea5e5c56...6c89` contains `f32[1024,2048]` and no global `f32[4096,2048]` table.
- Negative proof: DB495--DB498 reject production MXU and streamed/raw-lookup associations with
  4,096--1,208 mismatches.
- Scope: exact layer-0 query arithmetic only. Protected 8K tokens/DSA/state/cache/HBM/wall/trace
  evidence is still required before Gate D or performance promotion.

## Corrected 8K refusal and next-boundary observer

- Rejected run:
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_trace2_20260808T041407656729112Z`
  at `f129e636499d31c1c82c47727523c6527b3ee979`.
- Positive evidence: exact first token `101252`; event-0 selected set exact; DB499 layer-0 query
  correction remains active; eight byte-identical host logs; authenticated 8/8 clean failure exit.
- Refusal evidence: event 1 has 2,041 common positions / seven swaps and aligned score
  max/mean/signed error `0.27013397/0.18290268/-0.18290268`; no timing, DB row, `SUCCESS`, Gate D,
  or performance claim.
- Artifact identities: DSA NPZ `3e54254c...b053`; failure ledgers `18501db9...6051` and
  `ef7688d...c715`; host-log SHA `8066a555...4d63`.
- Diagnostic implementation: generic accepted full-producer callback, independent all-21-event
  greenfield five-state observer, and append-only hash-pinned accepted/greenfield comparator.
  Every path is default-off and does not enter the measured executable.
- Readiness evidence: 46/46 targeted runtime/validation tests, 2/2 affected layer-kernel tests,
  and 437 passed / 1 expected skip across the complete CPU-only greenfield suite; Python
  compilation, Bash syntax, ShellCheck, JSON and diff checks pass. CPU/HLO readiness is not TPU
  arithmetic proof.
- Exact next evidence: one protected accepted layer-1 capture and one greenfield observer capture,
  followed by direct field alignment before another 8K decoder retry.

## Completed all-event observer and corrected layer-0 boundary

- Failed protected run: `...dsa_dsa_internal_trace2_20260808T093721804151742Z` at `380659a`; no
  timing/DB/`SUCCESS`/Gate-D claim, eight byte-identical logs, authenticated 8/8 clean exit.
- Observer identity: contract `e9d6b9c4...a19f`, all-21 tensor `0c8b021d...0f4`; sealed baseline
  DSA observation matches the `f129e63` payload bitwise.
- Earliest production divergence: layer-0 normalized hidden exact; q-a 494/2,048 BF16 mismatches,
  max `0.015625`; query 4,096/4,096 mismatches. DB499 is conditional on accepted q-a input.
- Downstream layer-1 comparison: normalized hidden 3,974/6,144 mismatches, q-a 1,156/2,048, and all
  query/head/key values differ. Comparison/tensor/seal SHAs are `1bc43a8e...9ad5`,
  `79b813da...9054`, and `283e5e88...0d5`; approved archive is under the failed parent result's
  `diagnostic_comparison/layer1/` prefix.
- Next evidence: one bounded one-host q-a association matrix reusing DB491's fused `N=82`
  shard-major semantics, with one live row, no collective/callback, exact BF16 comparison, DB,
  archive, and authenticated cleanup. No full decoder retry is authorized before it passes.

## DB501 one-row q-a association rejection

- Protected DB 501 / item 1784: `greenfield_layer0_q_a_association_20260808T123220826430916Z` at
  `b4488076`; local/remote `SUCCESS`, DB snapshot, critical remote SHA equality and 8/8 clean
  pre/post censuses pass.
- Result: no exact candidate. All 12 mapping/norm variants have candidate SHA `439a4d54...d553`,
  376/2,048 BF16 mismatches, max/mean `0.0078125/0.0000967367`.
- Structure: all HLOs retain one live input/output row and shard-major N82 input state, with no
  collective/callback or dead `[32,...]` token tensor.
- Seals: runner `70745455...52de`, NPZ `f755bcb1...568b`, evidence `86520324...aae4`, remote objects
  `69f3d576...6a86`, SUCCESS `a5f1c67c...14f7`.
- Next evidence: direct one-row zero-spatial convolution, because accepted DB491 uses
  `convolution ... bf_io->bf` while DB501's M1 dot lowers to multiply/reduce. No 8K rerun first.
- Implementation readiness: the v2 matrix runs only four new convolution/norm candidates and
  requires an optimized physical `f32[1,82] convolution`; DB501's rejected 12 are not repeated.

## DB502 exact one-row q-a convolution

- Protected DB 502: `greenfield_layer0_q_a_association_20260808T124434046623046Z` at `c230c11`;
  local/remote `SUCCESS`, DB snapshot, six critical remote SHA equalities, and 8/8 clean pre/post
  censuses pass.
- Result: all four direct-convolution/norm candidates are exact, 0/2,048 BF16 mismatches, accepted
  SHA `c9fbac05...c70c`; fused kv-a companion SHA `cf288bc2...e790` is common to all four.
- Structure: each HLO contains `f32[1,82] convolution ... bf_io->bf`, one live row, packed N82
  state, and no collective/callback/dead token-row shape.
- Seals: runner `2a77d75d...75c4`, summary `75de66b6...040b`, NPZ `d9b14bdd...f76e`, evidence
  `815cc6a3...8f59`, remote objects `6a1d78e8...9257`, SUCCESS `de2e080d...dab`.
- Scope: arithmetic only, not a decoder or performance result. Next evidence is the default-off
  production path plus plan-aware packed N82 checkpoint and complete-layer/8K validation.

## Fused qkv-a production integration readiness

- Code pin: `0082bac0f74fa4cac631c8a3085576d3bd10e6ef`.
- Local evidence: 57 focused tests pass, including byte-exact N82 weight/scale transforms, a full
  small 32-file artifact pack/verifier round trip, production arithmetic equivalence, backend/layout
  refusals, zero obsolete Pallas-call counts, and fused HLO shape/count linting.
- Real-manifest reconstruction: separate layout hash remains `ba21c4ec...c9e`; fused layout
  `523afb1d...cb4`, semantic manifest `8bd08068...6f9`, payload `834,369,271,808`, and
  `26,074,039,744` runtime bytes/chip.
- Scope: readiness only. No protected fused artifact, production TPU helper/layer, Gate-B
  reclosure, Gate-D retry, or performance result exists.
