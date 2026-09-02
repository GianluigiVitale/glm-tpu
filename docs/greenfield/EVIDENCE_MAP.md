# Greenfield evidence and reusable protection map

The cross-repository reuse decisions and pinned implementation candidates are indexed in
[`REUSE_INVENTORY.md`](REUSE_INVENTORY.md) and
`configs/greenfield-reuse-inventory.json`. They define what is directly reused, independently
adapted, oracle-only, rejected, or reserved for a later gate; they do not weaken the evidence rules
below.

## Gate-D frozen execution pin and append-only evidence ref (prepared)

Recovery-v4 succeeded at `dc9ec46`, but persisting its required evidence advanced only worker 0's
linked execution branch to `edb12a5`; workers 1--7 remain clean complete standalone repositories at
the sealed target. The PP16 wrapper correctly refuses this mismatch before HLO work. Recovery v5
binds `gate-d-worker-repository-recovered-prestate-inventory.json` to the SHA-pinned v4 terminal and
accepts only that exact worker-1--7 state.

To avoid advancing executable `HEAD` after the next recovery, executable source freezes on
`rewrite/topology-first-decode` and install/recovery/HLO evidence is committed to the separate
`evidence/gate-d-topology-first` branch. `validate_gate_d_evidence_ref.py` sanitizes Git execution,
binds the canonical linked worktree and exact origin, authenticates actual remote heads, validates
all prior commits even in staged mode, and permits only new regular
`docs/artifacts/gate-d-*.json` blobs. It rejects merges, rewrites, sparse/index hiding, incomplete
or dirty trees and non-fast-forward history. The evidence mirror uses an immutable export verified
path-for-path, mode-for-mode and blob-for-blob against the authenticated commit, then writes and
rereads an exclusive canonical authority receipt. It publishes the verified archive and receipt to
a commit-keyed same-region prefix only with generation match zero and content MD5 from retained
no-follow descriptors. Generation-qualified byte replay and the exact two-object remote catalogue
must pass; live, noncurrent and soft-deleted generations are checked, and an existing
partial/different prefix is preserved and refused without overwrite. This is a prepared authority
only: no v5 install/recovery or PP16 HLO has run.

## Accepted protected greenfield evidence

Every accepted run has a local directory under `/home/gianl/glm-run`, a same-tag archive under
`gs://driftbench-dsv4-uc/results/`, append-only `bench/results.db` linkage, local/remote `SUCCESS`,
fleet agreement, and eight-host clean pre/post census.

| DB | tag | evidence |
|---:|---|---|
| 561 | `greenfield_pp16_lp2_exact_query_20260827T161033467085863Z` | real DB554 layer-0 LP2 exact query/head bitwise proof; two adjacent chips, local 2,048-row owner split into two runtime 1,024-row tuple4 groups, no global owner/host callback; bounded diagnostic only |
| 557 | `greenfield_full_checkpoint_load_pp16_20260827T070407924804237Z` | complete PP16 16-stage/32-owner raw direct-load, byte round-trip, state/HBM/archive/cleanup proof; no performance claim |
| 555 | `greenfield_topology_20260826T194116460015528Z` | replacement-pod physical `2x4x4` topology, fresh PP8/PP16 groups and fleet binding |
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

Replacement topology code is `bfe064d`; its fleet hash is `4a0c9a33...c301`. Collective matrix
pins are `fcd8426735...` and `b12af9633c8b14648db8d2a2ccd9a3c577a04817`. Topology hash remains
`294e777...559`, PP8 group hash `d5943ab8...c14`, and PP16 group hash `6383e57c...f21`.

Protected failed PP16 compile tag
`greenfield_short_decoder_compile_pp16_acquisition_20260827T122306083796688Z` at `9034764` is
nonterminal diagnostic evidence. All eight hosts loaded and compiled the complete 78-layer 2K
graph, then refused at the linter before execution. The graph has 219 AG / 301 physical AR / 315
logical AR components / 33 CP, exact LP2 groups only, 75 production MoE feature triples and no
forbidden hidden/dead-row state. The failures are bounded LP4-assumption collisions: LP2 LSE is
`f32[128]`, validity tuple packing produces arities `288/12/1`, and 21 ordinary LSE producers share
an ambiguous 8-KiB signature without external `wk` owners or divide-sqrt norms. HLO/StableHLO
gzip SHAs are `ff3ae81b...a841` / `243279aa...1a7`; 8/8 failure cleanup passes. There is no host
record, post-execute HBM, token, DB, `SUCCESS`, numerical, Gate-D or performance claim.

Protected failed PP16 execution tag
`greenfield_short_decoder_compile_pp16_acquisition_20260827T125930818874961Z` at `a670fdf` is the
nonterminal successor. The corrected HLO gate passes fleet-wide and the complete graph executes
once. All eight records agree on HLO/StableHLO `14cdc95d...64b0` / `c7b71689...7603`, synthetic
token 3592, active LP2 ranks `[0,1]`, health 1, producer 74, selected state `(0,1,2)`, aligned
position/context 3/4, visit mask 65,535 and peak HBM 27,252,078,592 bytes/chip. The only refusal is
the host validator's PP8 literal 255; 65,535 is exactly the all-16-stage mask. Offline replay passes
the plan-derived correction. Record/contract/census SHAs are `1a57c70c...f099` /
`2b7783f5...7e81` / `e35a3117...e40`; failure cleanup is 8/8. The one post-warmup 101.5-second sample
has no performance standing. There is no oracle/DSA comparison, trace, DB, terminal `SUCCESS`,
Gate-D or performance claim.

Protected nonterminal PP16 successor
`greenfield_short_decoder_compile_pp16_acquisition_20260827T132707782908362Z` at `309ee8b` passes
the corrected in-program HLO/token/metadata/HBM contracts after one warmup and one measured step.
All eight records match DB555's launcher-to-JAX mapping `[3,5,1,2,0,6,7,4]`; only the outer sealer
wrongly expects identity order. Offline corrected-validator replay passes all records, with fleet
HLO/StableHLO `72c7a09d...382f` / `c7b71689...7603`, peak HBM 27,252,078,592 and minimum largest
free block 4,322,943,488 bytes. Post census is 8/8 clean. The reproduced 101.5-second post-warmup
sample is a diagnostic stop requiring attribution, not accepted performance. There is no terminal
`SUCCESS`, DB, oracle/DSA/trace, Gate-D or performance claim; recover metadata-only, never rerun the
753B acquisition merely to correct outer host bookkeeping.

Accepted nonperformance feature-runtime tag
`greenfield_runtime_feature_pack_pp16_20260827T095428043535926Z` at `a973425` contains all 16
stages / 32 production feature-Pallas owners. Manifest `0f1bb271...52b6f1` binds base runtime
`b0f62466...4d2e5`, feature layout `77647844...399c7`, 6,944 tensor records and
869,545,347,072 payload bytes; complete mounted inspection passes. The exact 115-object archive has
ledger/SUCCESS self SHAs `9af3e535...28d3c` / `a6b36a46...1d7cf` and four clean fleet censuses.
One metadata-only recovery copied worker-6 records after its payload had already sealed; no payload
reran. This is the complete input for PP16 decoder compile, not DB, HBM, correctness or performance.

Accepted nonperformance runtime checkpoint tag
`greenfield_runtime_pack_pp16_20260827T091323450875229Z` at `af0e226` contains all 16 stages / 32
executable owners. Runtime manifest `b0f62466...4d2e5` binds layout `2ad20708...3d59b`, schedule
`65c31dfc...46987e`, 6,944 tensor records, 869,545,347,072 payload bytes and 122,448,155,520
explicit padding bytes. The independent mounted inspector passes every owner. Its 107-object
same-region result set has ledger SHA `e06f3e1f...0befd` and SUCCESS self SHA
`d1812196...fc83`; three authenticated fleet censuses pass. This is the complete base input for the
PP16 feature-runtime derivative, not a DB, HBM, decoder-correctness or performance result.

Complete nonperformance checkpoint tag `greenfield_full_pack_pp16_20260827T032310295108546Z` at
`3685ee4` contains 32 base plus two optional-MTP final owners: 757,149,950,848 payload bytes and
757,165,710,960 file bytes. Packed manifest SHA is `13ad2e92...fedb5`; independent inspection
passes the exact 72-object checkpoint set and every generation/CRC/sidecar/metadata/plan identity.
The results ledger/SUCCESS self SHAs are `c0097d9f...2af61` / `47c7745c...f9352`. Complete PP16
device round-trip/HBM subsequently passes at DB557 below; the pack itself has no performance
standing.

Accepted complete PP16 loader evidence is DB557 / tag
`greenfield_full_checkpoint_load_pp16_20260827T070407924804237Z` at `f7353eb`. All 16 stages and
physical ids 0--31 load 747,097,191,552 base bytes / 118,920 tensors directly, with exact device
round trip and zero FP8 dequantization, global concat, or runtime reshard. Maximum peak HBM is
24,748,712,448 bytes/chip and minimum largest-free-block is 8,265,700,864 bytes. Its 78-object
ledger self/file SHAs are `b70d5101...e1729` / `6d4907b9...5d16`; terminal SUCCESS self/file SHAs
are `d920e7bc...90aa` / `f348a192...8ea`; the final same-region set has exactly 80 objects and all
three fleet censuses are 8/8 clean. This closes Gate B with DB420 and the corruption-refusal suite,
but makes no decoder or performance claim.

Accepted bounded checkpoint-loader evidence is DB556 / tag
`greenfield_checkpoint_probe_load_pp16_20260827T031331899773974Z` at `46b8a4f`. The production
final-layout loader places both stage-0 PP16 owners on physical ids `[0,1]` and round-trips
50,347,904 raw bytes / 16 tensors with zero FP8 dequantization, global concat or runtime reshard.
Peak HBM is 25,201,152 bytes/chip and minimum largest-free-block is 32,989,212,160 bytes. State SHA
is `3957f3ed...3b535`; the 12-object ledger SHA is `a485bd19...990d4`; terminal SUCCESS self SHA
is `9da2910f...2a88`; fleet censuses and DB integrity pass. This authorizes a complete PP16 pack but
does not itself prove the complete checkpoint or decoder performance; DB557 supplies the complete
PP16 load/HBM proof.

Accepted nonperformance checkpoint-planning evidence is PP16 tag
`greenfield_checkpoint_plan_pp16_20260827T022537742669498Z` at `6dc7304`. It covers the complete
141-file / 118,629-leaf inventory with 16 exact two-chip stages, 32 base plus two MTP owner files,
and 757,149,950,848 planned packed bytes. Execution/plan/layout hashes are
`079cefe6...794c3`, `3c3ea07b...ed16`, and `f97de2d8...b15f9`; minimum modeled free bytes are
6,590,074,560/chip. The exact nine-object `US-CENTRAL2` terminal archive is generation/CRC bound.
This proves plan feasibility only: `promotion_memory_proven=false`, and there is no DB, TPU,
direct-load, measured-HBM, Gate-B, or performance claim.

Bounded real-byte pack tag `greenfield_checkpoint_probe_pp16_20260827T024902158913509Z` at
`42015a1` derives ten leaves from that exact layout and covers every layout/value class, both shard
axes, and both expert-owner slots. It reconciles 50,345,920 source bytes to two 25,173,952-byte
payloads in one grouped core invocation; owner SHAs are `87ac52df...7ea6` / `705ed0ff...0ac3` and
manifest SHA is `14c36aeb...c4e3`. Its exact nine-object / 50,375,010-byte same-region archive is
generation/CRC terminal-sealed. This closes the bounded pack discriminator only; direct-loader
compatibility subsequently passes in DB556, the complete pack is sealed, and DB557 supplies
full-checkpoint HBM/Gate-B evidence. Complete-decoder performance evidence remains missing.

The compile-only tag
`greenfield_ws32_short_decoder_8k_acquire_20260826T195124476893681Z` is preserved diagnostic
evidence, not an accepted DB result. It contains eight complete load/compile/HBM prevalidations and
six identical real HLO identities per rank. Recovery commits `bad88a6` / `8067bac` replay the
structural contracts and derive only HLO_ACQUIRED envelopes; no token graph is executed. A first
seal attempt used a mistyped checkpoint semantic pin and therefore produced no DB row, archive
SUCCESS or performance claim; authenticated cleanup is 8/8. Corrected local validation passes with
summary `5cafab04...60214`. Recovery commit `92cb72c` is pushed/owner-mirrored and the evidence-only
retry archived 137 generation-bound objects with fresh 8/8 recovery-pre/post zero-work censuses.
Source ledger is `b8f76d44...dde6`; acquisition remains deliberately outside DB/SUCCESS and makes
no performance claim. Its six StableHLO/optimized-HLO pairs are pinned in `HANDOFF.md` and authorize
the protected exact-DSA 8K numerical run.

The local WS32 combine association is now protected TPU evidence. Acquisition tag
`greenfield_ws32_strategy_nd_acquire_20260826T225219986470504Z` pins StableHLO/optimized-HLO
`1ef939fd...a2b` / `f01fd650...5e5` without execution. Numerical tag
`greenfield_ws32_strategy_nd_numerical_20260826T225424354570711Z` at code `676dea0` reproduces
DB550's accepted row `efde8532...8f8e` with zero mismatches on all eight hosts. HLO contains exactly
one BF16 group-8 all-gather over `[1,4,1,1536]`; pre/post censuses and append-only remote SUCCESS
are clean. This closes combine association only, with no DB/performance claim or checkpoint load.

The bounded real generation boundary is now protected TPU evidence. Identical-code acquisition and
numerical tags `greenfield_ws32_strategy_nd_layer0_*_20260826T23*` at `351d9f6` directly load the
sealed bounded owner checkpoint `ec6ef9cf...b3d3`. Fleet graphs are `3422d6a1...50f34` /
`a8832ab8...1e3bf` and contain exactly one feature-4 plus one expert-8 BF16 all-gather. All 32 real
partials match `9d9f65dd...16e35`; the final row matches `efde8532...7b4fc`, zero mismatches on all
hosts. Numerical summary/SUCCESS file SHAs are `bf7a837c...965` / `63f70dff...2cde`; raw tensors,
46 preterminal CRC/generation-ledger objects, HBM and 8/8 cleanup pass. This is exactness/locality
evidence with no DB/latency claim; it authorizes production integration but does not close Gate D.

Complete dense-overlay tag
`greenfield_ws32_strategy_nd_dense_overlay_pack_20260827T002508229552699Z` at `7844f2e` extends
that proven ownership to layers 0--2: 96 exact final owners, 2,102,200,128 bytes, manifest
`a8dc8791...4b6a` (`c17194b6...5c8c` file), SUCCESS file `166566b9...32a6`, and preterminal
ledger self/file `8f6f9ccf...a2d6` / `694e9af5...f10d`. The terminal approved-bucket set has 99
objects and direct verifier replay passes. Pushed production pins `7191206` / `1c1cb1f` consume
this as a default-off typed overlay and require exactly three feature-4 plus three expert-8 dense
gathers. This is checkpoint/composition evidence; full-decoder HLO and numerical proof are pending.

The complete production composition now has protected compile-only HLO evidence. Tag
`greenfield_ws32_short_decoder_8k_acquire_20260827T003758068665390Z` at `39e0ab8` loaded the full
checkpoint plus overlay on all eight hosts and produced six fleet-identical graph pairs. Exact
materialize/promote/prefill/observer/decode/cache-probe StableHLO pins are respectively
`1d925d96...f36e`, `e38eb7a4...ffff`, `9ef02642...b820`, `4e2496d9...89bf`,
`e23a9e77...2429`, and `664c331a...b14`; optimized pins are `09d22ef7...4d77`,
`ffc4a502...1dcb`, `68820860...891a`, `d6badd91...233d`, `8f964f9e...ce6`, and
`6ead75f3...b5a`. The three model graphs each contain exactly three feature-4 plus three expert-8
gathers, with no forbidden full-hidden values or group above eight; cache probe contains none.
Peak compiled HBM is 26,375,554,560 bytes/chip with at least 6,646,212,608 bytes largest-free-block
margin. This is HLO/HBM/cleanup evidence only: no arithmetic, DB, terminal SUCCESS or performance
claim exists. It authorizes one separately pinned numerical run.

That authorized numerical run is protected failed evidence. Tag
`greenfield_ws32_short_decoder_8k_numerical_20260827T011711674195301Z` at `04d059b` reproduces all
six graph pairs and all eight hosts agree on exact 20/20 tokens, valid state/cache and exact
event-0 DSA. Event 1/layer 1 still swaps seven selected positions and later events cascade. The
overlay changes the later DSA arrays relative to the prior failure but does not make them exact.
Diagnostic-only fleet p50 is `129.228901--129.2916055 ms/token`; peak HBM is
26,375,554,560 bytes/chip with at least 6,381,496,320 bytes largest-free-block margin. Rank-0
JSON/NPZ SHAs are `2b79c484...990b` / `2be686ff...eeb1`, all logs share `03629037...eac`, and
cleanup is 8/8. There is no summary, DB, terminal SUCCESS or performance claim. This rejects the
current WS32 plan on exactness; another full WS32 run is forbidden without a bounded layer-1 proof.

Bounded checkpoint tag
`greenfield_ws32_strategy_nd_layer0_pack_20260826T232037240198985Z` at pushed/mirrored code
`fbaaae3` maps layer-0 dense ranks into 32 exact WS32 expert-8/feature-4 owners. Payload is
`700,728,992` bytes; manifest self/file SHAs are `ec6ef9cf...b3d3` / `0f475ded...4512` and
SUCCESS self/file SHAs are `f853e187...3c5a` / `c1304592...6840`. The 33 pre-SUCCESS objects pass
size/CRC32C/generation equality; ledger self/file SHAs are `77c90337...0d14` /
`52bcad82...deb8`, and the terminal set is exactly 35 objects with `SUCCESS` last. This is bounded
checkpoint integrity/ownership evidence only, with no TPU, DB, model correctness or performance
claim.

The authorized numerical tag
`greenfield_ws32_short_decoder_8k_numerical_20260826T213125786075567Z` is protected failed
evidence, not an accepted result. It executed all six pinned real graphs, produced exact 20/20 raw
tokens, valid state/cache/HBM and exact event-0 DSA scores/positions, then refused at event 1/layer
1 with seven selected-position swaps. Maximum peak HBM is `26,331,170,304` bytes/chip and the
diagnostic p50 is `127.35638 ms/token`; neither is a promotion claim. Runner/NPZ hashes are
`b46fd1be...713aa` / `d2b3c9ab...4e65`; 24 worker artifacts are durable and pre/failure cleanup is
8/8. It created no DB row, terminal `SUCCESS`, or accepted performance record.

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

`greenfield_ws32_one_layer_pack_20260815T070628458699950Z` is the reviewed WS32 derivative of that
sealed PP8 layer, at code `4c749a668235892ad90d8c8399ff264d3ebc4727`. Its 32 final-owner files
carry 9,971,249,152 payload bytes, 311,601,536 per chip, with exact expert8/feature4 source slices,
FP8-scale ownership, tensor/file hashes and physical mesh `de5f59cb...0a88`. Manifest SHA-256 is
`4bf8679de10ebbba055e9d0be991495080388c6355fa449f393c28f4751e1f40`; slots 0 and 31 pass the
independent direct loader. The approved checkpoint prefix has exactly 40 objects: all 39
nonterminal objects passed local/remote CRC32C, size and generation equality before `SUCCESS` was
uploaded last and directly rehashed (`Nd1G9w==`, generation `1786777785681765`). Temporary payloads
were removed. This is durable checkpoint/direct-load evidence only; it has no DB row, JAX/TPU HLO,
HBM, correctness, latency, token or Gate-D claim.

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
- Scope at `0082bac`: readiness only. DB503 below supersedes the missing production-helper proof;
  no protected fused artifact, Gate-B reclosure, Gate-D retry, or performance result exists.

## DB503 fused qkv-a production helper

- Protected DB 503 / item 1786:
  `greenfield_layer0_qkv_a_production_20260808T140131250842069Z` at exact code `f715039`.
- Arithmetic: production normalized q-a matches the accepted 2,048 BF16 values bitwise, SHA
  `c9fbac05...c70c`; its fused 576-wide kv-a companion matches sealed DB502 bitwise, SHA
  `cf288bc2...e790`.
- Structure: final-layout `u8[32,6144,82]` / `f32[32,48,82]` inputs; optimized HLO SHA
  `1eec1393...509c`; exactly one physical `f32[1,82] convolution ... bf_io->bf`; no collective,
  callback, dead row, forbidden shape, or violation.
- Protection: SQLite integrity and DB snapshot, approved archive, critical local/remote SHA
  equality, terminal SUCCESS `5d458cb8...03e`, and authenticated 8/8 pre/post cleanup pass.
- Scope: bounded production arithmetic/HLO only. The protected fused checkpoint/direct load,
  complete decoder, HBM, XPlane, latency, Gate D, and Gate E remain unproved.

## Fused PP8 final layout and DB504 Gate-B reclosure

- Packed artifact: `greenfield_runtime_feature_qkv_pack_pp8_20260808T141032190315066Z` at pack
  pin `7d5dfb9`; 32 files, 10,880 tensors, `834,369,271,808` payload bytes, runtime manifest
  `12339490...699a`, layout `523afb1d...cb4`, semantic manifest `8bd08068...6f9`.
- Pack protection: mounted full verifier `verified=true`; checkpoint/result SUCCESS
  `368ef308...5b24`; approved bucket and authenticated 8/8 pre/post cleanup pass.
- Protected direct load: DB504 at `c16b37f`, run
  `greenfield_short_decoder_compile_pp8_2k_pallas_feature_linear_ot256_downf32_splitres_qkva_roundtrip_hlo_20260808T150900Z`.
- Load proof: eight hosts each load and device-round-trip `104,296,158,976` bytes / 1,360 tensors;
  zero runtime reshards, host concats, host FP8 dequantizations or device FP8 dequantizations.
- Device/HLO proof: maximum peak HBM `26,143,616,000` bytes/chip; optimized HLO
  `710942ec...d69c`; 78 exact one-row N82 convolutions; zero obsolete q-a/kv-a calls; every
  local-collective/transport/feature/stage-linear contract passes with no forbidden shape/overlay.
- Seals: summary `ed342ece...ca2`, HLO contract `1c1a8dc9...9375`, compressed HLO
  `266b1ed9...499e`, DB snapshot `2820fd82...4681`, evidence `63c6b494...6420`, SUCCESS
  `3b2dec6d...2f5f`; critical remote bytes, DB integrity, and 8/8 pre/post census pass.
- Gate decision: Gate B is re-closed. This body-only one-sample run has no token/DSA/trace or
  performance claim; protected 8K Gate D/E remains next.

## Failed fused 8K prefill and exact loop-classification correction

- Refused run: exact pin `3058dc8`, tag
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_qkva_trace2_20260808T153500Z`;
  stopped before prefill execution with no token/DSA/timing/trace/DB/final `SUCCESS` or gate claim.
- Preserved structure: prefill optimized HLO `9f8c2a7d...964e`; 79 physical loops resolve to 78
  exact fused-qkv internal identities under the outer body plus one exact outer prompt scan, with
  zero unclassified loops. Decoder contract `a406a211...8cc0` and HLO `bc4320f7...4602` pass their
  fused-qkv/local-group/transport/dead-row checks.
- Failure protection: eight identical logs `93cb2c80...4725`; pre-census `eb996b06...5a84` and
  failure census `d6d0c090...21b1` each prove eight unique clean hosts; critical GCS bytes match;
  SQLite integrity is `ok` and no evidence table contains the tag.
- Correction: `1f110133bc4411d6a3bcc1d2c69a8334f915a8fb` classifies outer, fused internal and unknown loops
  independently. Preserved old/new HLO pass as `1+0` and `1+78`; 39 focused tests plus the
  forced-32 complete prefill regression pass. This authorizes one retry but proves no Gate-D/E
  result.

## Loop-corrected fused 8K DSA refusal

- Refused run: exact code `e4079ac410f976476812e0a86ac9352e1407dc9d`, tag
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_oracle_dsa_qkva_loopfix_trace2_20260808T161834055154820Z`.
- Passed scope: complete fused final-layout load, decoder/observer/isolation/prefill HLO contracts,
  full teacher-forced prefill, exact first token `101252`, rank 1, score `22.5`, margin `6.75`.
- Refusal: event 0 exact set but first order difference at offset 8; event 1 expected-only
  `[1052,2024,3853,6256,6787,7473]`, observed-only
  `[825,3889,4899,5536,6113,6951]`; refused before warmup/timing/trace/DB/final `SUCCESS`.
- Seals: DSA NPZ `4647de15...6ac`, token observation `e5e35f3b...03c`, eight identical logs
  `63a29050...566e`, pre/failure census `c1e5f109...7d0` / `a7b172cb...ede`; critical remote
  bytes match, SQLite is `ok`, and the tag has no DB row.
- Decision: preserved correctness failure only. It cannot change Gate D/E and forbids another blind
  full-decoder retry.

## Prompt index-cache discriminator readiness

- Implementation pin: `e5a699177cd0164d62aa202729011631c324e447`.
- Oracle boundary: accepted default-off `dcp_cache_dump.py`; final step-4/chunk-2011 slot 0 only;
  exact eight DCP owners and four identical model replicas per owner; live block-table recovery of
  all 8,155 logical BF16 keys.
- Independent comparison: production raw-FP8 `wk`, input RMSNorm, key LayerNorm/RoPE and BF16
  writes execute in a one-row greenfield scan after the legacy runtime stops.
- HLO refusal contract: one raw-FP8 key kernel and one outer scan; no collective, callback,
  transport, decoded overlay, full-prompt hidden materialization, or `[32,6144]` dead row.
- Readiness: focused tests pass 27/27. No protected cache artifact/comparison exists yet, so this is
  implementation evidence only and proves no arithmetic cause, decoder result, or performance.

## DB505--512 accepted prompt-cache isolation

- Accepted source: DB505/item1788 captures all 32 final slot-0 snapshots. Corrected parsing proves
  `model=32,dcp=1`, four local/32 physical bitwise replicas, `[24,16,32,128]`, 512-token pages and
  logical 8,155x128 BF16 SHA `3808d502...859d1`.
- DB506/item1789: `...comparison_20260808T210743348875130Z` at `cee8bda`; production M1 raw-FP8
  scan differs in 4,058 elements/1,071 positions. HLO `e7e66d4e...849e` passes one-kernel,
  one-scan, one-row and no-communication/overlay/dead-row contracts. SUCCESS `f3b1f327...264c`,
  approved archive, DB and 8/8 cleanup pass.
- DB507/items1790--1792: `...association_20260808T214925579370178Z` at `31a23b8`. Pallas M1
  divide/sqrt has 4,050 mismatches; XLA M2048 multiply/rsqrt has 55; XLA M2048 divide/sqrt has 45
  over 45 positions with mean `3.15396e-8` and SHA `52bf55ed...cd8a`. All 45 are in rotary
  dimensions 0--63; dimensions 64--127 are exact.
- DB507 structure/protection: best HLO `93596359...36fd` has one physical M2048 convolution, one
  map, sqrt/divide and no forbidden communication/state. Association manifest `7216756c...7cae`,
  SUCCESS `6ce52989...c227`, evidence `affe8424...bcaf`, remote objects `7787dccd...bc0`, DB
  snapshot `b3fb207b...b47`, approved archive and authenticated 8/8 cleanup pass.
- DB508/item1793: `...association_20260808T221200429613435Z` at `7027cf6b`; the external live
  M2048 chunk has 4,045 mismatches/1,058 positions and is only 22 values from DB506 production.
  HLO `fde460cb...52a` has one convolution and zero loops/forbidden operations or shapes. Unlike
  DB507, it keeps `wk` FP32; DB507 explicitly converts the public FP32 parameter to BF16 once
  outside its map and uses that BF16 RHS in the convolution.
- DB508 protection: association manifest `8539a81d...6d07`, compressed HLO `b2e98616...f274`,
  SUCCESS `6a38369a...12e7`, evidence `48776445...214c`, remote objects `11aa387a...34e4`, DB
  snapshot `81bff928...a7be`, approved archive and authenticated 8/8 cleanup pass.
- DB509/item1794: `...association_20260808T225610150435734Z` at `8f2545c` explicitly rounds the
  public FP32 `wk`, and HLO `0638f148...9868` proves one physical BF16-RHS convolution plus zero
  loops/forbidden operations or shapes. Its output is nevertheless byte-identical to DB508:
  `db2f77d...a7a1`, 4,045 mismatches over 1,058 positions. BF16 `wk` is rejected as causal.
- DB509 protection: manifest `df0b901e...21c1`, compressed HLO `6e269756...2160`, SUCCESS
  `0bea72ba...9232`, evidence `652da666...1f99`, remote objects `6378c961...7047`, DB snapshot
  `3820af70...88f`, direct approved-bucket byte equality and authenticated 8/8 cleanup pass.
- DB510/item1795: `...association_20260808T234459065479710Z` at `6286a06` restores the physical
  device gather as the direct producer of input RMSNorm. It reproduces DB507 exactly:
  `52bf55ed...cd8a`, 45 values/45 positions, all in rotary dimensions 0--63. Every mismatch affects
  one member of a pair while its partner and dimensions 64--127 are exact.
- DB510 structure/protection: HLO `6dba4fbb...9de0` has the coupled gather/reduction, one
  BF16-RHS M2048 convolution, exact `wk` feature-slice provenance, zero loops and no forbidden
  operation/state. Manifest `3e29aadc...b0fb`, compressed HLO `cd293af3...7290`, SUCCESS
  `a7660e3e...e165`, evidence `9ed5bdff...5b4f`, remote objects `d4663158...5f9`, DB snapshot
  `50758ad2...e53`, direct remote equality and authenticated 8/8 cleanup pass.
- DB511/item1796: `...association_20260809T003039938922204Z` at `31ab626` adds the accepted flat
  BF16 paged-cache scatter while preserving every DB510 producer. Its logical output is unchanged:
  `52bf55ed...cd8a`, 45 values/45 positions, first 113. This rejects cache address/cast/scatter
  association as causal.
- DB511 structure/protection: optimized HLO `d396040f...3938` has exactly one BF16 physical scatter
  on flat `[12288,128]` cache state, exact aliased `[24,16,32,128]` cache input/output and
  `s32[16]` table, one gather-coupled RMS producer, one BF16-RHS M2048 convolution, and no
  loop/communication/callback/forbidden state. Manifest `6cbe954b...bb6`, compressed HLO
  `e55cecef...c10`, tensor `64e6df3b...130f`, SUCCESS `ec6a4370...a7`, evidence
  `8978b333...cb25`, remote objects `5d7bb1f6...083f`, DB snapshot `d71012bb...6eae`, direct
  remote equality and authenticated 8/8 cleanup pass.
- DB512/item1797: `...association_20260809T012205182369900Z` at `da7027d` spells the accepted
  `rope_cos_sin/apply_rope` source literally inside the exact DB511 producer/consumer path. It is
  byte-identical to DB511: `52bf55ed...cd8a`, 45 values/45 positions, first 113. Literal-source
  RoPE association is rejected as sufficient.
- DB512 structure/protection: optimized HLO `c96ecd28...a931` differs bytewise from DB511 but its
  physical RoPE contract is identical: one FP32 power/cosine/sine and the accepted constants. The
  gather/RMS, BF16 convolution, flat scatter and no-forbidden-state contracts pass. Manifest
  `9f037699...2c5`, tensor `20398ae9...9413`, compressed HLO `eb2df033...e9b6`, SUCCESS
  `01882c1f...3406`, evidence `34d23f22...1c62`, remote objects `9d0e5bbf...5488`, DB snapshot
  `5ba79f36...6e1`, direct remote equality and authenticated 8/8 cleanup pass.
- Decision: diagnostic correctness only. No candidate is exact and no decoder/performance/gate
  status changes. Gather-coupled input RMS is proven causal for the near-exact regime; the accepted
  cache consumer and literal-source RoPE spelling are rejected. The next required evidence is a
  bounded accepted pre-RoPE FP32 capture at position 113. Neither a rejected candidate nor the full
  decoder should be rerun.

## Prompt-key producer capture readiness — no protected result yet

- Oracle-only observer pin `9c1d6b3b9` captures projection, post-key-LayerNorm and post-RoPE FP32
  row 113 at the live accepted producer; the accepted checkout remains unchanged.
- The greenfield comparison path requires exact 8K tokens, 294 DSA events, accepted cache SHA
  `3808d502...859d1`, DB512 candidate SHA `52bf55ed...cd8a`, bitwise producer replicas, exact
  producer-to-cache casts, two strict HLO contracts, archive/DB linkage and 8/8 cleanup.
- Full CPU coverage is 478 passed / 1 skipped. This proves readiness only and does not alter Gate D,
  Gate E, latency or throughput status. The protected serialized capture is the next evidence item.

## DB518 exact prompt cache and integrated-repair refusal

- DB518/item1803 at `8624311` is the accepted bounded arithmetic result: the captured normalized
  input, all three FP32 producer states and all 8,155 BF16 cache rows are elementwise exact. Cache
  SHA is `3808d502...859d1`; comparison, DB/archive, direct remote bytes and authenticated 8/8
  cleanup pass. It is correctness evidence only.
- Integrated retry
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_oracle_dsa_trace2_20260809T154619026858354Z`
  at `ff5072e` passes all HLO contracts and executes prefill. Repair contract SHA is
  `a076b017...bee0`; prefill HLO gzip is `af72b2d5...d8fa4`; the device DSA observation is
  `25114ae8...ff0` and failure census is `74637617...2009`.
- The first token is exact, but all 21 DSA selected sets fail before timing. Event 0 swaps four
  positions, event 1 swaps nine, and later events reach 571 swaps. There is no DB row, terminal
  `SUCCESS`, trace or performance standing. Eight rank logs are byte-identical and cleanup is 8/8.
- HLO provenance differs at one still-unproven boundary: DB518's M64 loop consumes an entry
  `f32[128,6144] wk` parameter after completed adaptation, while the integrated loop consumes an
  internally dequantized/rounded/promoted value. One bounded materialization plus LP4-scatter
  discriminator is required before another full decoder retry.
- Bounded attempt `greenfield_layer0_prompt_key_weight_source_internal_20260809T181800Z` at
  `8dee6b8` compiled the raw-entry arm and passed all existing arithmetic/communication HLO gates,
  but stopped before execution because TPU flattened the explicit BF16 weight round to
  `bf16[786432]`; the initial linter admitted only `bf16[128,6144]`. It has no DB/SUCCESS or
  arithmetic/performance standing. Optimized HLO and the refusal are archived under the approved
  diagnostic prefix; pre/failure censuses are 8/8 clean.
- The narrow correction accepts exactly the logical or equivalent flat weight-round shape, retains
  one raw-U8/zero FP32 entry-weight requirements, and replays as one exact round on both preserved
  TPU HLOs. Focused tests pass 23/23 and the one-time Fable review returned `APPROVE COMMIT`.
  One fresh bounded retry is required before choosing an externally materialized repair or moving
  to the separate LP4-scatter discriminator; no full 8K retry is authorized yet.

## DB519 internal-materialization rejection and LP4 readiness

- DB519/item1804, tag
  `greenfield_layer0_prompt_key_weight_source_internal_20260809T182400Z`, code `5e1cbb5`, proves
  the internal raw-FP8 -> BF16 -> FP32 arm executes with its exact HLO boundary but produces
  298,532 BF16 mismatches across every one of 8,155 positions. Candidate SHA is
  `8fd4a8c2...d5df08`; accepted SHA remains `3808d502...859d1`.
- Comparison manifest `b9669799...48d42`, SUCCESS `027d68ef...7220`, evidence
  `d40b57e8...42425`, remote ledger `e08d2c85...bc0b`, SQLite integrity, direct remote bytes and
  authenticated 8/8 cleanup pass. This rejects internal materialization and has no performance or
  Gate-D standing.
- The local successor uses one separate stage-local materializer and passes its completed FP32
  outputs to repair. The fleet linter requires five raw/scale inputs, BF16 rounds/promotions, zero
  communication/callback, zero repair-side rematerialization, final-owner shard identities and
  15,728,640 bytes/device.
- A bounded four-chip extension of the existing DB518 harness runs these production functions,
  verifies each lane writes only its 128-row page ownership, and requires the assembled 8,155-row
  cache to match DB518 bitwise. Local focused coverage is 71/71; protected LP4 evidence does not
  exist yet, so another full decoder retry remains forbidden.

## LP4 materializer entry-scope refusal — no arithmetic result

- Attempt `greenfield_layer0_prompt_key_materialized_lp4_20260809T205000Z` at `99ce5ea` compiles
  the four-chip materializer, proves one BF16 round/FP32 promotion and no communication/callback,
  then refuses before execution because nested fusion parameters were mistaken for extra entry
  inputs. HLO gzip SHA is `183f82a8...9a64a`; pre/failure censuses are authenticated 8/8 clean.
- The exact correction restricts raw/scale parameter counts to parsed `ENTRY ` computations while
  keeping conversion and forbidden-operation inspection module-wide. Preserved-HLO replay and a
  nested-fusion regression pass. This is linter evidence only: no DB/SUCCESS, cache comparison,
  decoder, latency, throughput or Gate-D claim exists.

## LP4 bounded repair root-identity refusal — materializer proven, cache still unexecuted

- Attempt `greenfield_layer0_prompt_key_materialized_lp4_20260809T192658503682737Z` at `2113b0d`
  passes and executes the external materializer, then compiles but does not execute cache repair.
  Its generic bounded root emits `jit(mapped_repair)/shard_map/...`, outside the production-specific
  repair identity recognized by the unchanged strict linter.
- Preserved TPU HLO contains four exact projections, eight physical square roots, four affines,
  owner-cache writes, zero grouped square roots, zero collectives and zero repair weight rounds.
  A semantic root-name-only replay passes the existing contract with projection/exact/sqrt/affine/
  cache-write counts `4/4/8/4/8`.
- Repair-HLO gzip SHA is `5576305c...05de`; pre/failure-census SHAs are `b7b38902...5dcf` and
  `6586bc71...9ebe`; cleanup is authenticated 8/8. There is no cache comparison, DB/SUCCESS,
  decoder, performance or Gate-D result. The correction is only a bounded-wrapper rename; a fresh
  protected LP4 exactness result remains required before another full 8K run.

## LP4 repair execution — materialized output identity refusal

- Attempt `greenfield_layer0_prompt_key_norm_m64_20260809T193931471545587Z` at `9cf4119` passes
  both HLO gates, executes the four-chip materializer and cache repair, and passes sentinel owner
  isolation. Its first materialized FP32 shard differs from accepted adapter SHA
  `d680f7b1...83469`; because the initial verifier stopped immediately, this does not yet prove
  whether other lanes agree or whether raw input placement is exact.
- Materializer/repair HLO gzip SHAs are `764fe24f...6423b` and `397f04b8...0f085`. Pre/failure
  census SHAs are `9fe2527d...d2902` and `a3993b10...33130`, the diagnostic is archived in the
  approved bucket, and fleet cleanup is 8/8. There is no cache equality, DB/SUCCESS, decoder,
  performance or Gate-D standing.

## LP4 combined materializer rejected — split-boundary proof next

- Diagnostic retry `greenfield_layer0_prompt_key_norm_m64_20260809T195128353553953Z` at
  `e53d1fd` proves all four raw bits/scales shards are bitwise exact and all four output shards are
  identical. Their common SHA `b6429bf2...6e975` differs from accepted `d680f7b1...83469` in
  698,727/786,432 FP32 values per lane. Thus placement and lane drift are rejected; the combined
  adapter executable is causal.
- Compact diagnostic SHA is `80e20cc8...9a18`; pre/failure census SHAs `22a6f95c...c12d` and
  `bd7a265b...bb0e4` authenticate 8/8 cleanup, and the approved archive contains it. This is a
  rejection with no DB/SUCCESS/cache/Gate-D or performance standing.
- Next evidence is one bounded two-completion adapter: raw FP8 -> BF16 must complete before a
  separate BF16 -> FP32 promotion, then the unchanged LP4 repair must match the accepted cache
  bitwise. Separate HLO contracts reject phase fusion, communication and host callbacks.

## DB520 exact split-boundary LP4 repair

- DB520/item1805 at `0977022`, tag
  `greenfield_layer0_prompt_key_norm_m64_20260809T200559393031635Z`, passes separate BF16-decode and
  FP32-promotion HLO contracts (optimized SHAs `08b6c59f...ab2f9` / `1c107d68...b1f8`) with zero
  communication/callbacks, then executes unchanged owner-local repair.
- All four FP32 shards equal accepted `d680f7b1...83469`; writes are
  `[2048,2048,2048,2011]`; all 8,155 cache rows and producer states are bitwise exact at
  `3808d502...859d1`. Manifest `1e942555...08a59`, SUCCESS `643f80eb...083ca`, DB snapshot
  `d466adc9...79581`, remote ledger `599ba9f1...affd`, approved archive and 8/8 cleanup pass.
- This is bounded arithmetic evidence, not Gate D or performance proof. It authorizes production
  split-boundary wiring and one protected full 8K retry after tests and a one-time diff audit.

## Split-boundary full 8K refusal and missing fused composition

- Refused run: `5a41bb2`, tag
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_oracle_dsa_trace2_20260809T201759614361573Z`,
  20:18--22:33 UTC. Complete load/compile and 8,155-token prefill pass; five-slot split
  materialization and repair HLO contracts pass with zero communication/callback.
- Correctness: token `101252` and event-0 selected set are exact. Event 1 is the first failure at
  seven swaps, identical to the separate-qkv/no-repair baseline; later events diverge. The run
  refuses before warmup/timing/trace/DB/final `SUCCESS` and has no Gate-D/E standing.
- Seals: observation `4fb4b087...2fc7c`, NPZ `427329b0...a6f`, token JSON
  `e5e35f3b...03c`, eight identical logs `c0296290...b83e`, pre/failure censuses
  `2140efa1...4580` / `56f8622d...b2a8`; failure cleanup is authenticated 8/8.
- Configuration finding: runtime `54e2f89b...d9917` was the old separate-qkv artifact. It does not
  test the fused N82 runtime `12339490...699a` together with DB520 repair. DB502--504 and DB520
  independently prove those two boundaries, so their protected composition is the exact next
  integration result after the launcher-safety diff passes tests and one review.

## Combined fused-runtime and split-repair 8K refusal

- Refused run: pushed `6b554c4`, tag
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_qkva_oracle_dsa_trace2_20260809T225800140051438Z`,
  22:58--01:03 UTC. Full eight-host load/compile and 8,155-token prefill pass with runtime
  `12339490...699a`; decoder/observer/prefill/materializer HLO contracts all pass.
- Token `101252` is exact. Event 0 membership is exact; event 1 is the first set failure at six
  swaps. No warmup, timing, XPlane, DB row, terminal `SUCCESS`, Gate D or Gate E result exists.
- NPZ/token/log/pre/failure seals are `34f4fe30...bbf`, `e5e35f3b...03c`,
  `fb04f32a...cb0`, `bedaf1bb...ccea` and `55f32a98...6fc`; the same-region diagnostic archive
  exists and authenticated cleanup is 8/8.
- Direct comparison with the earlier fused-only refusal finds identical selected-set membership
  for all 21 events and identical compact token observation. This rejects the fused-plus-repair
  composition as a further correctness improvement: the remaining recurrent divergence is
  unchanged.
- Reuse checkpoint: accepted layer-1 internals/comparison/seal already exist at SHAs
  `79b813da...9054`, `1bc43a8e...9ad5`, and `283e5e88...10d5`. The next discriminator is the
  existing separate DSA-internal observer against those bytes, not another legacy capture or blind
  full decoder retry.

## Current-state DSA internal refusal and physical-query proof gap

- Refused run: pushed `888cfcb`, tag
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_oracle_dsa_dsa_internal_trace2_20260810T013247766447206Z`,
  01:33--03:38 UTC. Its observation is bitwise equal to the pinned fused-plus-repair baseline.
- Layer-0 normalized hidden and q-a are exact. Query is the first divergent field with 4,096
  mismatches/max `0.0101393461`; actual/accepted SHAs are `6fe17a94...355` / `1ff2c2ec...12a`.
  Internal NPZ/contract/log SHAs are `e1366c58...b50`, `b087aa92...bb2`, `03dc8b35...301`.
  Cleanup is authenticated 8/8; no timing, trace, DB, SUCCESS, Gate-D/E or performance claim exists.
- DB499's exact `raw_materialized_lp4` candidate is now scoped correctly: four TPU devices were
  visible, but the candidate was an unsharded single-device program over a global weight with four
  virtual owner slices. Its HLO does not reproduce the one-owner-per-chip production reduction.
- Readiness only: `query_lp4` extends the same bounded harness with a real four-chip `shard_map` and
  four candidates spanning raw/predecoded owner state and 1,024-wide/eight-by-128 reductions. It
  pins the accepted/current tensors and rejects communication, global tables, callbacks and dead
  rows. No physical candidate or production correction is accepted until protected TPU evidence.

### First physical-query HLO refusal

- Attempt `greenfield_layer0_physical_lp4_dsa_query_association_20260810T041439079476028Z` at
  `752d36e` passes 8/8 idle checks and executes the raw owner-dot arm transiently. It also compiles
  the head-unrolled arm before its old linter refuses, but that HLO was not yet persisted. Owner HLO SHA
  `0a8ba57c...8472` proves four partitions and exact local raw/scale/materialized/output shapes with
  no forbidden operation or global table.
- The next head-unrolled arm refuses before arithmetic because the linter asks every candidate for
  the owner-dot's 1,024-wide intermediate instead of its intentional 128-wide head intermediate.
  No DB/SUCCESS/query output/exactness/performance evidence exists. Pre/failure census SHAs are
  `fd4ed1ac...5710` / `33c967bb...9cad`; the approved partial archive exists.
- Candidate-specific width validation and pre-validation HLO persistence are readiness corrections
  only. A fresh protected matrix remains required.

## DB521 physical query matrix and q-a execution boundary

- DB521 at pushed `88350e3`, tag
  `greenfield_layer0_physical_lp4_dsa_query_association_20260810T042459221979151Z`, executes all
  four true four-device candidates in six seconds. Raw/predecoded owners and owner-wide/eight-head
  reductions are elementwise identical at SHA `eee61d94...bb`; they differ from accepted in 2,728
  FP32 values (max `9.5367432e-7`) and from production in all 4,096 values (max `0.0101392269`).
- Evidence/tensor/SUCCESS SHAs are `86423acb...782`, `7c2cf4c...feb`, and `b2a43f13...852`.
  The DB snapshot, same-region archive and authenticated 8/8 cleanup pass. This rejects raw versus
  predecoded ownership and owner-wide versus head-unrolled association; it is not performance
  evidence.
- Direct reads of the four stage-0 runtime shards rule out packing or physical-slot corruption.
  Concatenated `wq_b`, scale and head-weight bytes match the sealed source at SHAs
  `12f9ca94...e0`, `0541bd9a...48d`, and `4dabc09e...624`; all 4x4 slice-equality matrices are
  identity matrices with zero global mismatches.
- Preserved production HLO exposes the remaining discriminator. The fused q-a affine exists as
  FP32; query consumes that FP32 value before its separate BF16 observer conversion. Thus the
  recorded bitwise-exact q-a does not prove the query's physical input was rounded. The bounded
  `query_lp4_q_a_boundary` target composes the actual fused N82 helper with the real four-chip
  owner query and compares unrounded, BF16-barrier, and BF16-barrier-plus-HIGHEST arms. Production
  remains unchanged until a protected result selects an exact arm.

## DB522 fused q-a boundary result

- DB522 at `13123b8`, tag
  `greenfield_layer0_physical_lp4_dsa_q_a_boundary_20260810T051607194401685Z`, passes all protected
  contracts in 20 seconds. All q-a outputs are bitwise exact. The unrounded arm reproduces current
  query SHA `6fe17a94...355`; both explicit-BF16 arms reproduce DB521 SHA `eee61d94...bb` and stay
  2,728 values/max `9.5367432e-7` from accepted.
- The two rounded StableHLO programs differ by the requested HIGHEST precision attribute but both
  optimize to HLO SHA `a5b6742c...c76`. The BF16 barrier is causal for the large drift; HIGHEST is
  rejected as an exactness discriminator. No arm is promoted.
- SUCCESS/evidence/tensor SHAs are `99acefc0...157`, `4acf8eb3...231`, and `6d09ded4...1f2`.
  DB522, same-region archive, object ledger and authenticated 8/8 cleanup pass. No performance or
  Gate-D standing exists.
- The next target changes only physical head schedule: legacy-local one-chip/N128 sweep versus a
  PP8 owner-local device loop containing eight N128 reductions. It reuses accepted q-a and decoded
  owner bytes; it cannot authorize production until a protected arm is accepted-exact.

## DB523 physical head-geometry result and GSPMD successor

- DB523 at pushed `a3bd353`, tag
  `greenfield_layer0_physical_lp4_dsa_head_geometry_20260810T053345314403882Z`, completes in six
  seconds. Its physical one-head sweep and single device-resident eight-step loop are identical at
  SHA `bb4930ff...d3e`; both differ from accepted in 2,840 values/max `1.4305115e-6` and from
  current production in all 4,096 values/max `0.0101393461`.
- Sweep/serial HLO SHAs are `11edd83c...0ba` / `6d458b78...583`. The first has zero loops and eight
  host diagnostic executions; the second has one device execution and exactly one eight-step loop.
  Both remain local, one-row and communication-free. This rejects local N128 width/scheduling.
- SUCCESS/evidence/tensor/runner SHAs are `b34dfc6f...874`, `d7c4292b...301`,
  `f73f773d...de8`, and `e4f4c3bb...a35`; the DB snapshot, same-region archive, object ledger and
  authenticated 8/8 cleanup pass. No performance or Gate-D standing exists.
- Readiness only: the successor retains global logical M1/N4096 semantics under explicit four-way
  GSPMD sharding. Its StableHLO must contain global logical weight/output shapes and explicit
  sharding; its optimized per-partition HLO must contain only local N1024 weight/eight-head output,
  four partitions, no communication and no global materialization. CPU compilation proves the
  structure only; protected TPU arithmetic remains required.

## DB524 global-logical GSPMD result and tuple-fusion successor

- DB524 at pushed `70f549e`, tag
  `greenfield_layer0_physical_lp4_dsa_head_geometry_20260810T054953335423347Z`, completes in six
  seconds. The new GSPMD candidate equals DB521 query SHA `eee61d94...0bb`, with 2,728 accepted
  mismatches/max `9.5367432e-7`; it is not promoted.
- StableHLO SHA `5a912167...f1c` proves four partitions, logical `f32[4096,2048]` weight and
  explicitly sharded `[1,32,128]` output. Optimized HLO SHA `0ec1e68b...303` proves only local
  `f32[1024,2048]` weight/eight-head output and no collective/global materialization. Structural
  sharding passes; physical arithmetic remains the same DB521 association.
- SUCCESS/evidence/tensor/runner/results-DB SHAs are `e9e6de5f...df0`, `d6b6f4e9...8ae`,
  `2f185e99...85c`, `aaa7e996...4e0`, and `6e22ccfd...9df`; same-region archive, object ledger and
  authenticated 8/8 cleanup pass. No performance or Gate-D standing exists.
- Readiness only: exact DB499 HLO groups four N1024 reductions (`megacore_allreduce_bytes=16384`),
  versus DB524's one (`4096`). The successor passes the same owner buffer as four aliased local
  inputs and retains four dots through one StableHLO barrier before returning one result. CPU HLO
  proves mechanism only; protected TPU exactness remains required.

## DB525 exact physical tuple fusion and production-composition gate

- DB525/item1810 at pushed `a749ff0`, tag
  `greenfield_layer0_physical_lp4_dsa_head_geometry_20260810T060457076587721Z`, accepts
  `physical_owner_tuple4_barrier_m1_n1024`: zero of 4,096 mismatches and exact accepted query SHA
  `1ff2c2ec...12a`.
- StableHLO SHA `3b10b7e5...9f7c8` preserves four top-level aliases, four local N1024 dots and one
  barrier. Optimized HLO SHA `40d9ef25...4b7d` contains one tuple-valued 16-KiB megacore fusion,
  four local results, four partitions and no communication/global physical table.
- SUCCESS/evidence/tensor/runner/summary/DB/remote-ledger SHAs are
  `5b547f24...a6d582`, `35f888c7...17624`, `c55a6638...790ce`, `da7acf8b...1dfc`,
  `5bba9e4e...1932`, `41a43045...ef86`, and `e06bc413...3358`; same-region archive, DB linkage,
  direct remote bytes and authenticated 8/8 cleanup pass.
- This proves the physical arithmetic mechanism only. Production promotion additionally requires
  one bounded `query_lp4_production_exact` result that completes local raw-FP8-to-FP32 state as a
  separate executable and composes the actual fused q-a producer, BF16 barrier and exact helper.
  It must retain one tuple-valued 16-KiB fusion and match both sealed q-a and query bitwise before
  a full 8K Gate-D retry is authorized.

## First production-composition attempt — validator diagnostic only

- Pushed pin `7f636ed`, tag
  `greenfield_layer0_physical_lp4_dsa_query_production_exact_20260810T073812618515497Z`, stopped
  before materializer execution. Optimized/StableHLO SHAs are `103cd550...a9f71` /
  `f99669ea...6002d`.
- The HLO is four-partition and owner-local, with no collective/global table/host callback. Its
  only custom targets are the known bounded-gather metadata markers
  `AssumeGatherIndicesInBound` and `GatherScatterIndicesBitpacked`; a validator false positive
  rejected their generic `custom-call` opcode.
- The approved partial archive exists and pre/failure census SHAs `ce3f1293...e3e31` /
  `08b777ac...2f7e` authenticate 8/8 cleanup. There is no tensor result, DB row, `SUCCESS` or
  performance standing. A corrected contract and fresh bounded retry remain required.

## DB526 exact production composition

- DB526/item1811 at pushed `3aa9f9c`, tag
  `greenfield_layer0_physical_lp4_dsa_query_production_exact_20260810T080508327295662Z`, passes the
  actual fused-q-a/materializer/tuple4 production composition. Q-a SHA `c9fbac05...cc70c` and query
  SHA `1ff2c2ec...cb12a` are bitwise exact with zero mismatches.
- Production StableHLO/optimized HLO SHAs are `4e7f3dc3...190f` / `78c29674...069c`; the program
  contains one scoped 16-KiB tuple4 fusion, four partitions, one row and no collective/global query
  table. Materializer HLO `2cc9283a...44b3` completes 8 MiB/chip with no forbidden target, host
  marker, collective or global state.
- SUCCESS/evidence/runner/summary/tensor/results-DB/remote-ledger SHAs are
  `b3cff36b...3b11`, `82219191...2b7d`, `497dd606...96f2`, `890a9d8e...212e`,
  `b371ad77...d247`, `caa8ff3f...03dd`, and `980f4f78...e4e3`; approved archive, direct remote
  bytes and authenticated 8/8 cleanup pass.
- This closes the bounded pre-8K production-composition gate only. A complete protected 8K run
  must still pass exact tokens/DSA, state/cache/HBM/HLO, fresh trace, wall, DB/archive and cleanup.

## First exact-query full 8K compile — metadata refusal only

- Pushed pin `1824cf8`, tag
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_oracle_dsa_trace2_20260810T082522620015034Z`,
  compiles the complete exact-query decoder/materializers and refuses before execution only because
  the token-return source name is `mapped_token_exact_query` rather than the linter's default
  `mapped_token` spelling.
- The return remains one `s32[1]` permute over the exact 32 lane pairs; local score/id exchanges and
  all other decoder/materializer HLO gates pass. Decoder HLO gzip/contract SHAs are
  `86030c36...55cf` / `5a55efd4...da32`.
- Identical-rank-log SHA is `bc65b0e7...b27e`; pre/failure census SHAs are
  `5ec2756d...df37` / `05762594...d7fa`. The approved diagnostic archive and authenticated 8/8
  cleanup pass. No execution, token, DSA, DB, `SUCCESS`, trace, wall or Gate-D/E result exists.

## Full 8K internal breakpoint and DB527 recurrent head/key association

- Pushed `fb1dea9` full run
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_oracle_dsa_dsa_internal_trace2_20260810T093638877848825Z`
  proves exact first token plus exact layer-0 normalized/q-a/query, then exposes 32/32 head and
  97/128 current-key mismatches. NPZ/contract SHAs `a889b664...ed07` / `fb470de5...fb18`; clean
  failure census. No DB, timing, trace, `SUCCESS`, Gate-D or Gate-E claim.
- DB527/item1812 at `cd15aaf`, tag
  `greenfield_layer0_physical_lp4_dsa_head_key_boundary_20260810T104647319991568Z`, accepts only
  `physical_normalized_barrier_materialized_divide_sqrt`. Head/key are bitwise exact at SHAs
  `ec66b475...725e` / `9f1fb991...dbbc5`; HLO/StableHLO SHAs `4dc28823...03a1` /
  `d9fd33b4...dbcb` prove two local dots, one barrier, four partitions and no communication/global
  owner state.
- SUCCESS/evidence/runner/tensor/results-DB/remote-ledger SHAs are
  `0c961dd1...5fb0`, `7c7563e7...7678`, `f4d2518c...1310`, `3ea5813f...b5b52`,
  `57a88dc9...76c3`, and `a4bc6797...08f0`; archive and authenticated 8/8 cleanup pass. This
  authorizes the default-off full-decoder integration, not a performance claim.

## Corrected stacked layer-0 residual discriminator — rejected association

- Pushed pin `9b53ce2`, tag
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_oracle_dsa_layer0_residual_variants_trace2_20260810T202128342266137Z`,
  completes full load/compile/prefill, exact first token and exact layer-0 selection before its
  intended diagnostic refusal. The four-row NPZ/contract SHAs are `d134609e...cd16` /
  `7aeeb00a...96e1`.
- The old HLO contract passed locality and FP32-boundary checks, but optimized HLO SHA
  `19fdc1b3...e1e2` proves cross-arm fusion: tuple-valued BF16 and FP32 dense reductions,
  two-result RMS reductions and one four-result normalized-output fusion. The BF16 control differs
  from current production in 615/6,144 values, max `0.0009765625`; all four numerical arms are
  therefore inadmissible.
- HLO-gzip/old-contract/rank-log/pre-census/failure-census SHAs are
  `0d694ce3...a7f`, `b87f5a19...8cfe`, `b2fe159c...7e5`, `9576da95...f5e8`, and
  `9bd7e5c7...ff7`. Authenticated cleanup is 8/8. There is no DB row, terminal `SUCCESS`, timing,
  trace, Gate-D or performance result.
- The successor must compile and validate four single-arm executables independently, reuse the
  exact same post-prefill arrays for each non-donating execution, and stack only host-side after
  all arms complete. A full 8K decoder retry is not authorized first.

## Independent layer-0 arms and virtual-TP32 discriminator readiness

- Pushed pin `12315aa`, protected tag
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_oracle_dsa_layer0_residual_variants_trace2_20260810T221121969164909Z`,
  makes the four precision arms admissible: independent HLOs, exact BF16 production control and
  exact layer-0 DSA selection. All four remain accepted-target nonexact at
  `3960/4008/3998/4034` mismatches.
- Contract/suite/NPZ/pre-census/failure-census SHAs are `1d3d270f...b0e`, `31ed0093...75e`,
  `f7323ea2...45a`, `615dc21d...db9`, and `73d1e1dc...369d`; direct approved-bucket contract bytes
  agree and authenticated cleanup is 8/8. There is no DB/SUCCESS/timing evidence.
- Accepted oracle source pin `b3c25df47ac98783912dc658878181ec0a8ae16d` and captured
  after-codegen HLO gzip SHA `51d014de...47f0` show separately BF16-rounded local projection dots
  feeding a physical 32-way BF16 `RotatedPincerEmitter/StrategyND` reduction. This is the premise
  for reconstructing eight virtual contraction shards inside each PP8 owner.
- The default-off successor uses 8 attention K512 plus 8 dense I384 existing Pallas calls, explicit
  seven-add BF16 local trees and only LP4 physical reductions. Its affected forced-CPU suite passes
  67/67 in 198.84 seconds with compileall/Bash/ShellCheck/diff checks green. Protected TPU HLO and
  numerical output remain missing until the reviewed batch is committed and launched.

## Protected virtual-TP32 uniform-tree rejection

- Pushed `e19833a` protected tag
  `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_oracle_dsa_layer0_subshard_variants_trace2_20260811T001535466381463Z`
  completes four isolated arithmetic programs. Every optimized-HLO contract and the four-module
  suite pass, with exact 8xK512/8xI384 kernel counts, declared LP4 shapes, exact layer-0 selection,
  one row and no escaped collective.
- All arms are nonexact versus the sealed layer-1 target: `4200/4262/4366/4317` mismatches. A
  direct three-band scan is uniformly nonexact for all four, rejecting a single uniform row tree
  as an emulation of the accepted three-color StrategyND reduction.
- Contract/suite/NPZ/rank-log/pre/failure census SHAs are `5bbc7f29...07c5`,
  `f9a10221...690`, `2387e412...1eca`, `f1a40474...dd23`, `d54f1431...280d` and
  `0131ab15...a1e7`. The contract and artifacts are retrieved from the approved bucket; all eight
  logs agree and authenticated cleanup is 8/8. No DB/SUCCESS/timing claim exists.
- Next evidence must reuse the Gate-A protected collective harness to fingerprint the real 32-way
  BF16 StrategyND association and replay it offline. Another full-checkpoint tree guess is
  evidence-rejected. If the captured virtual partials cannot reach the accepted boundary, the
  existing layer-0 observer must isolate the accepted monolithic-attention versus greenfield LP4
  LSE-merge input before any projection-side integration.

## DB533 protected StrategyND association fingerprint

- DB533/item1818 at tag `greenfield_collective_association_20260811T213152133863450Z` seals one
  repeated exact-M32 fingerprint. Analysis/summary/SUCCESS SHAs are `e7e34828...4108`,
  `3ca82073...36b7`, and `e3b0c442...b855`; exact HLO/backend/layout, raw arrays, local/remote
  hashes, DB linkage and clean 8/8 censuses pass. No model/checkpoint or performance claim exists.
- Every physical row has a unique matching candidate on all 6,144 columns. Row zero uses the
  accepted model-to-device permutation and physical `y -> x -> z` phases: y changes tree only for
  hidden 2,048--4,095, x is a two-way add, and z alternates the two proven four-way pincer trees
  every 256 columns. Rows 0--7 share output SHA `7239b23e...1dc`.
- This directly authorizes one default-off layer-0 discriminator applying the recovered row-zero
  schedule to both attention-output and dense-down virtual partials through exactly two LP4
  gathers. It does not authorize production integration, Gate D, or a performance claim until the
  table-on control and exact layer-1 boundary pass on TPU.
- The revised pre-launch contract executes the same reducer as a device canary against all 32
  sealed trials/lanes, pins 82 BF16-round barriers and one exact LP4 gather, and globally refuses
  any third matching gather in the decoder. Local replay is exact at `0/196,608` mismatches and
  the affected suite passes `79/79`; protected TPU boundary evidence is still pending.
- The first protected attempt at `742eacd` proves the canary itself exact on TPU (`0` mismatches,
  32 trials x 32 lanes, 82 barriers, one exact LP4 gather). It stopped only because optimized HLO
  retained a leading shard-map singleton on the gather operand. Eight-rank cleanup is clean; no
  layer boundary, token, timing or Gate-D claim follows. The guard correction accepts only that
  preserved real encoding or the logical rank-3 encoding.

### Pre-launch provenance disposition

- Six independent Max-effort consultations completed; direct artifact inspection confirms accepted
  HLO SHA `51d014de...47f0` is the 2,048-row `PREFILL_ONLY` executable, not the 32-row decode
  executable containing position 8,155.
- The prototype's `bf16[1,6144]` physical-ring collective therefore cannot establish the decode
  association from backend-string equality. The accepted prefill psum is `bf16[2048,6144]` with
  iota group `{{0,...,31}}`; chunking, implicit member mapping and per-hop rounding remain open.
- Status: retained default-off as a non-gating diagnostic; protected launch rejected before use.
  No DB, trace, timing, model-correctness or Gate-D standing exists.
- Next admissible evidence is one isolated layer-0 ingredients capture at unsaturated boundaries:
  main latent-cache rows, pre-projection attention output, accepted/greenfield 32 BF16 partials and
  post-reduction row, then MLP boundaries. This separates upstream cache/attention divergence from
  projection values and association.

## DB530 layer-0 main-cache discriminator

- DB530/item1815,
  `greenfield_legacy_layer0_main_cache_20260811T052303163478417Z`, seals the accepted 8K cache
  snapshots and classifies `prefill_main_cache`. Selected/current comparisons report
  `30,544/18` BF16 mismatches; padding is exact and current mismatch dimensions are all within the
  64-wide main-RoPE suffix.
- Comparison manifest/tensor/SUCCESS SHAs are `fb47b2e3...69c9`, `a7121337...0924`, and
  `7a46ae65...cd10`. Local ledger, remote object generations/CRC32C, DB run/item, archive and two
  independent authenticated 8/8 clean censuses pass.
- DB503's exact current pre-RoPE companion plus accepted table row SHA `67b01e3c...a1d` reproduces
  legacy suffix SHA `e7c217ec...3281` when products/adds are FP32 and only the completed row is
  rounded to BF16. This is offline causal readiness, not yet protected TPU or decoder evidence.

## DB531 protected main-RoPE mechanism

- DB531/item1816, `greenfield_layer0_main_rope_20260811T072231959104598Z`, runs greenfield pin
  `ed7c74f...4f51` on TPU and changes the sealed current suffix comparison from the captured
  `18/64` mismatches to exact `0/64`; expected and candidate SHA are `e7c217ec...3281`.
- HLO SHA `c611c74d...f623` proves four FP32 products, two FP32 combines and the TPU-equivalent split
  final round `[32,32]`, with no BF16 rotary arithmetic, dynamic trig, callback or collective.
- Runner/tensor/SUCCESS SHAs are `d32d0357...7370`, `52d2a36e...4145`, and
  `8f9763e4...b6d7`. DB/archive/object-integrity and clean pre/post 8-host censuses pass.
- Evidence scope is one real current main-RoPE row only. It authorizes the default-off device-table
  and main-MLA integration batch, not Gate-D acceptance, performance, full-cache exactness or a DSA
  change.

## DB534 and DB536 accepted layer-0 attention operands

- DB534/item1819 seals the exact accepted post-`W_UV`/pre-`o_proj` row at position 8,155. Its SHA
  `79a6e290...2e9d` differs from the table-on PP8 operand `0103e22c...82ab` in 5,117/16,384 BF16
  values. Exact tokens, all-event DSA, state/load, DB/archive and 8/8 cleanup pass; it is diagnostic,
  not decoder performance evidence.
- DB536/item1820 seals both the accepted attended latent and the same post-WUV row. The latent SHA
  `923e9bfe...d2a` differs from table-on PP8 SHA `f98193a5...558` in 4,344/32,768 values, with
  `6.103515625e-05` maximum and `1.399234975e-06` mean absolute error. Every head differs. The
  protected classification is `attention_arithmetic_before_w_uv`.
- DB536 local/remote hashes match for capture JSON `f517b408...96a`, tensor NPZ
  `3a619a09...a30`, comparison JSON `6d43a176...a4c`, DB snapshot `52eee52f...b69` and terminal
  `SUCCESS` `88691576...47c`; exact DSA and authenticated 8/8 pre/post cleanup pass. This closes
  W_UV/o_proj as root causes and authorizes only the bounded full-segment/block/head-geometry probe.
- DB530 comparison NPZ `a7121337...0924` is also the accepted-cache input authority for that probe.
  Its accepted score-order rows have SHA `73298b7d...71be2`; stable position ordering produces SHA
  `8b59adca...87c6` and the same position SHA `ef78b044...fea08` as the table-on owner union. The
  current table-on segment differs at exactly position 8,145 / latent column 367, so both inputs run
  through each compiled arm and only the accepted-cache result can identify exact arithmetic.

## DB537 exact block-512 attention arithmetic

- DB537/item1821, `greenfield_layer0_attention_arithmetic_20260812T114701365714147Z`, runs the
  isolated five-arm four-chip probe at pin `83577222b4d48cb4b2fec544c06e3a71fbc821f5` and
  classifies `exact_arithmetic_arm_identified`. The exact accepted-cache arms are
  `pregathered_h16_b512`, `pregathered_attention_h2_b512`, and `pregathered_full_h2_b512`; the same
  three arms are exact with the table-on greenfield cache control.
- Every exact arm reproduces accepted attended-latent SHA
  `923e9bfeb65864868cef359cf98ebad67f2756794ad142ac87e66b71877a2d2a` with zero of 32,768 BF16
  mismatches. Both B128 controls retain 216 mismatches and maximum error `3.0517578125e-05`.
  Therefore the causal requirement is the complete 2,048-row selected segment with B512 recurrence;
  accepted two-head projection scheduling is unnecessary because the local 16-head arm is exact.
- Runner/tensor/summary/SUCCESS SHAs are `7961622c...a4ec`, `7d5ebe15...1f61`,
  `9793f89a...8540`, and `ecc2b873...3153`. All five real-TPU HLO contracts pass without
  communication, the checkpoint manifest/evidence/tensor chain is pinned, DB linkage is exact,
  remote CRC32C/content checks pass, and pre/post censuses authenticate eight idle hosts.
- This is bounded arithmetic evidence, not decoder correctness or performance. It authorizes one
  default-off PP8 production integration: each lane places only its owned selected rows in the
  canonical 2,048-row segment, one LP4 BF16 sum reconstructs that segment, and the existing
  pre-gathered H16 B512 kernel computes only that lane's 16 heads. The protected 8K run must still
  prove exact tokens/DSA, cache/state, local HLO, trace, wall, HBM, DB/archive and cleanup.
- The reviewed production HLO contract does not accept counts/shapes as consumption proof. Every
  exact-name B512 call must be inside the named attention scope, and cache operand 2 must trace
  through only cache-preserving shape transforms to one scoped LP4 exchange. The 78 exchange/call
  links must be bijective; logical/folded bypass, suffix and out-of-scope mutations are regression
  tested. This is readiness evidence only until the protected complete-decoder HLO passes.
- The first real integrated compile at `a2d3905` preserves a complete optimized HLO before its stale
  aggregate count guard refuses. It directly proves the selected path has local `ag/ar/cp =
  63/312/17`, all 78 cache-exchange/kernel links pass, and the old LSE/validity TPU rewrites
  (`f32[256]` and `u32[1,1,128]`, 78 each) are absent with their source gathers. Replaying that exact
  HLO against the corrected conditional count/shape contract passes with zero violations. This is
  HLO readiness and authenticated failure-cleanup evidence, not decoder correctness/performance.
- The `9a90c3c` retry additionally preserves passing production-decoder and DSA-observer HLOs plus
  the real prefill HLO. Prefill initially refused only because its wrapper omitted the immutable
  selected-path flag when calling the common validator. With that propagation restored, both the
  observer and prefill artifacts replay at `63/312/17`, 78 bijective links and zero violations.
  Execution still did not begin, so this remains HLO/readiness and clean-failure evidence.

## DB539 exact attention-output projection association

- DB539/item1823, tag
  `greenfield_legacy_layer0_attention_update_p8155_20260812T172809039093068Z`, seals the accepted
  layer-0 post-`o_proj` row at position 8,155. Its SHA is `68afed86...de7`. The DB538 StrategyND
  candidate is bitwise exact at 0/6,144 mismatches; the local LP4 candidate has 3,652 mismatches.
  Classification is `strategy_nd_attention_projection_exact` with only `strategy_nd` exact.
- Capture JSON/tensor/comparison/SUCCESS/results-DB/remote-ledger SHAs are `f5b502cf...a91e`,
  `02c78d13...cec6`, `780cf3c2...435c`, `f52d3e88...f508`, `c33f23bf...ea2`, and
  `3f3910fc...e045`. Legacy observer `23ab8780...761`, oracle `b3c25df4...16d`, DB538 linkage,
  live DB row, direct remote content and authenticated 8/8 pre/post censuses pass.
- This authorizes only the default-off production integration of DB533's row-zero StrategyND tree
  over eight K512 attention partials per LP4 owner. It does not close Gate D or establish latency.
  The complete protected 8K run must still pass exact tokens/DSA, state/cache, HBM, local HLO,
  fresh trace, profiler-free wall, DB/archive and cleanup.
- Production admission is a two-level compiler proof. Optimized HLO pins exact physical calls,
  local collectives, counts, groups, bijection and root liveness. Pre-fusion StableHLO separately
  pins the ordered contiguous K512 input/weight/scale slices and every DB533 barrier-rounded
  `y -> x -> z` add. Decode, DSA observer and prefill must archive both representations with
  fleet-identical hashes; either contract failing blocks execution.

## WS32 protected real-layer numerical baseline

- Sealed input: layer-3 WS32 manifest `4bf8679d...1f40`, mesh `de5f59cb...0a88`, and independent
  one-layer oracle manifest `c63ffa19...bff`.
- Protected compile tag `greenfield_ws32_real_layer_hlo_20260815T081128291986777Z` binds the exact
  launch-host/JAX permutation and all 32 direct device slots. Eight identical StableHLO/optimized
  HLO files are pinned at `dea1384e...c01f` / `6a6acf94...e157`.
- The scheduled graph proves 15/15 live inputs, 10/10 live F32-operand reductions, nine feature-4
  and one expert-8 group, no group larger than eight and no physical 32-row hidden reconstruction.
  Per-chip measured post-load peak is 311,859,200 bytes with at least 32,702,539,776 bytes in the
  largest free block.
- Protected numerical tag `greenfield_ws32_real_layer_numerical_20260815T090957477700205Z` reused
  those exact graphs and passed independent terminal shard/oracle recomputation. Normal and
  concentrated maximum absolute errors are `0.015625` / `0.03125`; maximum peak HBM is
  335,805,440 bytes/chip and the smallest largest-free block is 32,648,534,016 bytes/chip.
- Diagnostic-only p50 is `516.5821635` / `1160.885836` ms. This rejects the readable
  whole-matrix-dequant body as an execution candidate but is not token-throughput evidence. The
  51-object archive has terminal `SUCCESS`, CRC/generation linkage and authenticated 8/8 cleanup;
  there is deliberately no DB/performance row.
- Protected Pallas acquisition `greenfield_ws32_pallas_real_layer_hlo_20260815T094203579842098Z`
  pins StableHLO/optimized-HLO `fa11961d...6118` / `17ee208a...ff5d` and live closure
  `0a03b130...6a51`. Protected numerical tag
  `greenfield_ws32_pallas_real_layer_numerical_20260815T163210555760825Z` reproduces those graphs
  with 15/15 live inputs, 27 live raw-FP8 Pallas calls, nine feature-4 plus one expert-8 reduction,
  maximum error `0.03125` in both oracle cases and peak HBM `314,218,496` bytes/chip. Diagnostic
  p50 is `1.256975` / `2.303685` ms, selecting this body over the readable baseline without making
  a token-throughput claim. Its 51-object SUCCESS-last archive and 8/8 cleanup pass.
- Reviewed/pushed code `a908c36` composes the selected body into a complete default-off 78-layer
  WS32 short decoder with separate prefill/observer/decode/cache-probe executables, eight
  fail-closed HLO pins, exact sealed 2K/8K token and DSA oracles, one-row state, raw cache/DSA
  evidence, sequential executable lifetime, HBM/wall/XPlane collection, DB rollback and exact
  archive sealing. The complete WS32 local suite passes 70/70 and immutable correction audit
  `1b9038af...b668` returned `APPROVE COMMIT`. This is readiness only: the full 32-owner checkpoint
  pack, HLO acquisition and protected 2K/8K numerical executions remain missing.

## Section 18 completion audit

This table is the fail-closed project completion view. A bounded layer, CPU test, HLO-only replay
or one short-context result cannot satisfy a broader row.

| Section 18 requirement | Current direct evidence | Status |
|---|---|---|
| Serve GLM-5.2-FP8 at 256K on the existing v4-64 | Legacy oracle only; no greenfield 256K execution | Missing |
| Independent greenfield execution path | Isolated native-JAX PP8/WS32 code; legacy used only for sealed oracles | Implemented; final runtime proof pending |
| Plan-aware final-layout checkpoint | Complete PP8 pack/load DB420; reviewed WS32 full pack code, no sealed full WS32 pack | PP8 pass; WS32 pending |
| Exact DSA sets and tie order | Gate C DB421; PP8 DB484/DB563 use `top_k=context=2048`, disable oracle total-order/score bounds as gates and check ties only against executing scores | Gate C bounded pass; strict complete-decoder proof missing because the 2K selected set is cutoff-vacuous and oracle-relative tie equivalence is unproven |
| Raw tokens and quality | PP8 2K DB484 exact | Pass at 2K only |
| State/load/cache protection | PP8 DB420/DB484 | Pass at PP8 2K scope; selected-plan 8K/long-context pending |
| Repeated collectives topology-local | PP8/PP16 real layers and WS32 Pallas layer | Full selected decoder proof pending |
| No full-pod hidden reconstruction inside transformer | Bounded HLOs pass; sealed WS32 8K acquisition has only feature-4/expert-8 groups and no forbidden full-hidden value | Pass for acquired WS32 graph; selected final decoder still pending |
| Protected PP8 and PP16 measurements | PP8 full 2K DB484; PP16 transport/layer only | PP16 full-decoder measurement missing |
| WS32 protected measurement or evidence-backed rejection | Protected WS32 8K numerical run has exact tokens/state/cache/event 0 but seven event-1 swaps; diagnostic p50 about 129.23 ms/token, no DB/SUCCESS/performance claim | Current WS32 plan rejected on exactness; Gate-G final adjudication still pending |
| Device and profiler-free wall agree | PP8 2K and bounded runs only | Selected plan/long-context pending |
| Four-depth 128K smoke | None | Missing |
| Protected 256K E0 | None | Missing |
| Every accepted artifact linked to results DB | Existing accepted DB rows link; future required rows absent | Pending future gates |
| Same-region durable archive | Existing accepted runs pass; future required archives absent | Pending future gates |
| Authenticated eight-host cleanup | Existing accepted runs pass; future required runs absent | Pending future gates |
| Base vs speculative throughput reported separately | No speculative promotion has begun | Pending after base decoder |

## PP16 full-width rejection recovery and next discriminator

Commit `8cf6d07` recovered the already-completed protected PP16 arithmetic under tag
`greenfield_pp16_feature2_full_width_recovery_20260829T055100992443538Z` without a model rerun.
The exact 18-object same-region archive ends in `NUMERICAL_REJECTED` generation
`1787982898735781`, CRC `qGjEoQ==`, SHA `67d08d19...62fb`; evidence ledger is
`58e1403e...7f23`. Both fresh censuses are 8/8 clean, all 28 source objects were read at immutable
generations, original terminal generation `1787980539108624` is unchanged, DB integrity passes
with zero source/recovery rows, and no exact-success marker exists.

CPU authentication proves the full-width and half-width captures bitwise equal across 11 common
arrays. Reconstructed layer-0 prompt cache `35350ca0...8d7a` differs from DB518
`3808d502...59d1` at exactly 71 coordinates; the earliest is position 113 / hidden 35,
`47091` versus `47092` BF16 bits. The immediate critical path is one observation-only PP16 layer-0
discriminator at that position, with every prior output required bitwise unchanged. Only after that
boundary is localized may a separately reviewed model retry occur. Gate D remains open and must not
be represented as Section 18 completion.

The p113 observer and compile-only wrapper remain pre-numerical evidence. Observer
mode has 24 exact result roots while the default graph remains unchanged; its classifier requires
all 15 ordinary captured arrays and both owner observations byte-identical before interpreting the
accepted normalized/key boundaries. The corrected contracts require exact observer JAXpr identity,
exact rejection/oracle NPZ hashes and the complete p113 selected-prefix/tail invariant. Ordered
source results and exact optimized-root geometry are explicitly non-causal acquisition hints;
binding maps remain empty until the acquired canonical graph is separately inspected and pinned.

Commit `2a20740e` made one compile-only attempt under tag
`greenfield_pp16_feature2_position113_acquire_20260829T070550803626247Z`. It refused before main
lowering because two otherwise-identical JAXpr mesh descriptions render as CPU and TPU-v4 runtime
labels. Only three already-known materializer HLOs exist; there is no observer main HLO, tensor, DB
row, terminal success or performance claim. The 15-object diagnostic ledger is
`7948231f...6185`, cleanup is 8/8, and tracked diagnosis
`docs/artifacts/pp16-feature2-position113-jaxpr-runtime-diagnosis.json` proves that changing exactly
the two mesh renderings reproduces observed TPU SHA `4e7f821d...d5d`. The correction pins both raw
forms plus canonical complete-JAXpr SHA `c8b59417...d1cd` and refuses mixed/unknown meshes. Review,
commit/push/mirror and one separately approved compile-only retry remain required.

The approved retry is now protected HLO evidence under tag
`greenfield_pp16_feature2_position113_acquire_20260829T072101902362381Z`. It compiled without main
execution and archived 20 byte-verified same-region objects with clean 8/8 pre/post censuses.
Stable/optimized/canonical identities are `bc2fcc77...6035`, `f1cd8286...1bfe`, and
`5b5dfacf...a10e` (6,662,190 bytes; 14,781 stripped debug references). Compact provenance is
`docs/artifacts/pp16-feature2-position113-hlo-acquisition.json`.

The DB518 layer-0 successor compile-only acquisition is sealed at
`docs/artifacts/pp16-feature2-layer0-db518-hlo-acquisition.json`. It binds the exact acquired
StableHLO/raw/canonical identities and proves four field-29 prompt-key chunks reach the layer-0
cache update and scorer through only LP2 collectives. It is causal HLO evidence only: main
execution, numerical correctness, Gate D, DB linkage and performance remain unclaimed.

Its separate default-off numerical entrypoint is now locally prepared. It bytewise authenticates all
20 acquisition objects and both causal topology hashes, retains zero warmups/exactly one invocation,
and hard-pins the acquired identity even through direct exact-CLI use. Two independent classifiers
require the accepted full-width event-1/layer-1/dense boundary and exact DB518 p113/key plus all
8,155 reconstructed logical cache rows. The historical p113 classifier still requires all ordinary
outputs unchanged; only the new DB518 classifier permits correction effects. An invalid non-CPU-
forced suite attempt opened local accelerators, was terminated/excluded and ended with authenticated
8/8 cleanup (`5fe2caf7...1a8e4`). The valid CPU-forced suite passes 199/199 in 175.93 s. Same-Sol
review approved diff `447117e2...06de` for commit/push/mirror and exactly one zero-warmup protected
DB518 numerical invocation through the wrapper. There is not yet numerical, Gate-D, DB or
performance evidence.

This acquisition is not numerical Gate-D evidence. Its local, unreviewed successor adds the missing
causal certificate: nine ENTRY observer roots bind to fields 7--15 of the final while; four scans
have exact 2048/2048/2048/2012 limits and 27 same-index handoffs; each field update reaches both its
prior field and the exact runtime-position-113 predicate; the count is an add recurrence; first-scan
observer storage is zero-sanitized. Complete StableHLO/canonical-HLO identity remains mandatory.
The first Sol pass found that unordered ancestry still admitted a current-key half-padding swap,
an inverted selected-field predicate and an inverted count predicate. The correction pins all 32
ordered branch subgraphs, requires the exact direct/fused predicate wrappers, and requires count
conversion from the unmodified predicate bitcast. A follow-up review found that a pinned current
SSA name could still be redefined as prior. The successor also binds every caller source definition
and complete transitive executable ancestry, including referenced computations, in aggregate SHA
`05322ce7...b02b`; the exact name-decoy is the tenth rebased-identity hostile case. Complete
PP16 feature2 validation passes 168/168. Numerical execution stays frozen until this classifier
passes the same independent Sol review.

That first structural review blocked ancestry-only binding: a current arm could be replaced with
its prior value and the count increment with the valid-count select. The local correction resolves
fused callee parameters to exact caller predicate/true-current/false-prior sources for all eight
selected fields in all four scans, and field 15 must be exactly
`prior_count + convert(position == 113)`. Both attacks now refuse under deliberately rebased
canonical identities. Full validation passes; this correction still needs Sol re-review.

The completed structural batch is commit `17785f9`; the same Sol reviewer returned `APPROVE
COMMIT`, origin matches and all nine changed files were byte-verified in the locked same-region
mirror. This closes only graph causality, not numerical Gate D.

The current uncommitted successor is a default-off, zero-warmup, exactly-once p113 numerical
discriminator. It authenticates the compile-only 20-object acquisition and current causal
certificate before the call, then requires the sealed 15 ordinary outputs and both observer owners
bitwise unchanged. Its only terminal is diagnostic `POSITION113_CAPTURE_CLASSIFIED`; there is no
DB, Gate-D, token, exactness or performance claim. Oracle/rejection bytes are hashed on both sides
of the TPU interval and each NPZ is parsed from the exact byte buffer that produced its hash. The
first Sol numerical review blocked the prior hash-then-reopen implementation; hostile replacement
tests now cover both sources. It needs a correction-only execution verdict before commit/push or
TPU use.

Protected tag `greenfield_pp16_feature2_position113_numerical_20260829T085708974224501Z` at
`94d1ea5` closes that discriminator. The one zero-warmup LP2 invocation retained all 15 ordinary
arrays and all nine cross-owner observer equalities. Normalized BF16 is exact; current key differs
from accepted at 74 FP32 values and one BF16 value, hidden 35 (`47091/47092`). Candidate BF16 SHA
`ac1cf9e...59bc` exactly equals the sealed greenfield row, while accepted is
`a4d52dc2...db87`. The 28-object same-region archive, 8/8 pre/post cleanup and terminal generation/
CRC pass. Compact artifact `docs/artifacts/pp16-feature2-position113-numerical.json` has SHA
`7c551346...9a7e`. This is diagnostic only. It selects integration of the already-proven DB518
physical-M64 key association before layer-0 causal consumption; it does not close Gate D.

## Local DB518 layer-0 integration preparation

- The uncommitted successor is default-off and leaves all three prior executable identities
  unchanged. It batch-reconstructs the exact scalar embedding/input-RMS boundary, consumes the
  authenticated runtime positions, and injects DB518 physical-M64 FP32 keys before layer-0 causal
  cache write/scoring.
- Forced-LP2 tests prove batch/scalar BF16 equality, one local exchange/reduction/gather per chunk,
  exact injected cache/internal-key bits and true bypass of the rejected one-row key projection.
  The complete affected suite passes 202/202.
- New semantic identities are raw CPU `f2c8b067...a7aef`, raw TPU-v4
  `2a81016a...b866b` and canonical JAXpr `f87c0f16...9447f`. There is deliberately no TPU HLO pin
  yet. Exact mode can only archive a compile acquisition and reports non-causal root hints.
- This is implementation/readiness evidence only: no model arithmetic, tensor, DB row, exactness,
  Gate-D or performance claim exists. Sol review, commit/push/mirror and one separately authorized
  compile-only acquisition remain required.

## Protected DB518 numerical rejection and FP32-boundary insufficiency

Reviewed/pushed/mirrored commit `dafe2ee` executed the DB518-enabled PP16 graph once with zero
warmups under tag
`greenfield_pp16_feature2_layer0_db518_numerical_20260829T115022665987633Z`. Position-113
normalized/key checks and all 8,155 logical prompt-cache rows are bitwise exact at cache SHA
`3808d502...859d1`; all ordinary arrays equal the prior rejection. The terminal remains
`NUMERICAL_REJECTED`: layer-1 normalized differs once per equal owner at hidden 2795
(`48422/48423`) and downstream q-a/query/head/event-1 DSA differ. Capture/comparison/summary SHAs
are `534bacc5...62f0`, `06ee82b9...2a4d`, and `b01a5ac1...0d2c`; cleanup is authenticated 8/8.
There is no DB, token, trace/wall, performance or Gate-D claim.

Offline capsule `docs/artifacts/pp16-feature2-layer1-straddler-classification.json`, SHA
`eebe1c5d...7b36`, fail-closes on all sealed inputs, DB518 runner `fd51aacb...7a80` and optimized
HLO `634cf81a...07ca`. Both runtime norm-weight receipts match DB550 SHA `10e34f4f...b6c87`. Literal
double-round BF16 arithmetic cannot produce protected bit `48422` for norm-weight bit `15762`,
while the optimized graph has 12 scoped BF16 float-type-correction markers. A deterministic NumPy
FP32 single-round model constructs both observed and accepted output witnesses from different FP32
preimages that round to the same current and accepted BF16 carried rows. This proves only that the
retained BF16 evidence cannot identify the accepted FP32 boundary. At that historical point, the
next planned action was an accepted FP32 layer-1 RMS-input acquisition through a separately
reviewed non-perturbing oracle path. Unchanged
BF16/full-8K reruns and bounded-error promotion remain forbidden; Gate D is open.

## Historical layer-1 FP32 boundary preparation — observer rejected

This preparation record is historical. The protected acquisition later proved that the callback
perturbs the executable, so none of the preparation below authorizes reuse or another acquisition.

Oracle-only commit `8dc7d20fedca5a98c27bfd1774827305973fa4c1` is now consumed only through a
default-off protected mode. The observer hooks the exact real layer-1 input RMSNorm and sends the
identical hidden-update/residual BF16 objects to a non-returning callback; FP32 addition exists only
in the host artifact path. The greenfield sealer pins vLLM `a30addc7548...`, independently
reconstructs and bitwise-checks the FP32 sum, and compares both operands plus rounded/unrounded
results to exact DB550 leaves. It revalidates the tracked straddler capsule, raw source file,
self-hashed manifests and exact source-code hashes at terminal publication.

Focused synthetic, exact archive-runtime and real protected-source terminal tests pass 10/10;
adjacent coverage passes 44/44 with one protected-source skip. The explicitly CPU-forced complete
validation suite has 334 passes, 40 skips and only two unrelated WS32 baseline failures already
reproduced at clean `HEAD`. At this historical point no protected acquisition, accepted FP32
tensor, Gate-D, DB or performance result existed. That planned review/acquisition sequence was
completed and rejected by the protected result recorded below; it is not a current next action.

The first execution-only review then blocked before launch on recreated-pod state: workers 1--7
lack the base Git checkout assumed by observer worktree creation and all hosts lack
`/tmp/golden.json`. No TPU/model/Ray action occurred. The pending correction uses a self-contained
bundle of exact observer pin `8dc7d20f` (947 tracked entries, 12 accepted-parent commits), validates
copy/clone/ancestry/cleanup on all hosts, and restores the 321,146-byte golden manifest only from
the SHA-bound `driftbench-dsv4-uc` rank source. Terminal replay seals both receipt sets. Corrected
focused coverage passes 11/11 and all adjacent shared-wrapper suites pass 165 with two skips. This
batch's full forced-CPU validation has 335 passes and 40 skips with only the same two unrelated WS32
baseline failures already reproduced at clean `HEAD`; no WS32 file changed. The first correction
audit caught caller-CWD-dependent bundle verification and an unset default-mode terminal branch;
the final diff supplies repository-explicit verification, a safe default argument and independent
clone/verify/checkout refusal, then reproduces those validation results. This is still offline
preparation. A second audit caught real `gcloud ssh` banners contaminating exact receipt files;
stdout receipts and seven archived SSH-status files are now separate, and a real-banner-shaped
terminal regression proves mixed receipts refuse. The same validation results pass on this final
diff. Sol approved correction commit/push/mirror at staged SHA `95f6a0ec...1d20317` and authorized
no execution. Gate D remains open.

The correction was subsequently persisted as `89e0e3c`. Fable approved one new literal tag, but
the invocation failed closed before run initialization because the comparison oracle still named
the deleted controller-local `/home/gianl/glm-run` copy. Artifact
`docs/artifacts/layer1-rms-input-recreated-controller-prerequisite-failure.json` records status 2,
the burned tag, vacant local/remote namespace, free lease and fresh 8/8 clean census. The reviewed
successor uses the canonical same-region gcsfuse oracle, pins manifest
`f8154c5f...b26da` plus SUCCESS `0b798974...df1b9`, and fully inspects its files and 294-event
contract before the workload lease in all four downstream-comparison modes. Focused 13/13,
shared-wrapper 167/2 and complete forced-CPU 1,315/53 validation pass apart from the same two
unrelated WS32 baseline failures. Fable approved commit/push/mirror only; it contains no accepted
FP32 operand or Gate-D result.

Fresh tag `...174920080399586Z` subsequently reached Ray bootstrap but no model/TPU work. It exposed
a recreated-pod firewall rule still targeted at the deleted pod. Artifact
`docs/artifacts/layer1-rms-input-recreated-pod-ray-firewall-failure.json` binds the burned tag,
before/post rule hashes, archived diagnostics and 8/8 cleanup. After changing only `targetTags`, a
lease-held no-model smoke proves seven TCP/6379 receipts, seven joins, exactly 8 Ray nodes/32 TPU
resources and 8/8 cleanup. This is infrastructure evidence only, not Gate D.

Protected tag `...192233297523063Z` later completed the model/item but exactly reproduced DB551:
557,434 selected-position and 573,438 selected-score mismatches beginning at event 1, with event 0
and structural arrays exact. Artifact
`docs/artifacts/layer1-rms-input-observer-perturbation-rejection.json` binds DB565/item1869, raw
array/operand SHAs, 8/8 cleanup and the generation-bound cost cleanup. Classification is
`REJECTED_OBSERVER_PERTURBATION`; neither its BF16 operands nor host FP32 reconstruction is an
accepted oracle. All layer-1 RMS-input callback acquisition/sealing routes must fail before
cloud/JAX/model work.
No documented TPU backend tracepoint currently extracts the value while promising unchanged
executable identity. Without such a supported mechanism, the FP32 boundary is unobservable and
hidden-value capture remains closed; Gate D remains open.

Isolated observer tombstone commit `c7973435aa2fc948da9185ef99938f886613ce2f` is pushed. Sol's
correction-only audit approved commit/push/mirror with no P0--P2 blockers and no execution. Final
coverage is 10/10 focused greenfield, 175 passed/2 skipped affected shared and 4/4 observer tests.

Offline certificate `docs/artifacts/callback-executable-class-certificate.json`, SHA
`6e58bca961c0629799683ccfb75efda195206885e36975e497630ec379076e48`, supersedes the proposed
no-TPU HLO diff. It byte-authenticates DB485's manifest and terminal SUCCESS seal, including the
full `b3c25df47ac98783912dc658878181ec0a8ae16d` pin and DB485/item1769 identity. It authenticates
log SHAs `c680eb58...2f51`, `29911df0...f12` and `2285a893...f549` through run date, EngineCore
PID, every anchored Git marker and tag/profile anchors. Each run has exactly the ordered
`[32,64,128,256,512,1024,2048]` backbone buckets and seven ordered executable,
executable-including-data and host-transfer fingerprint triples. DB551 and DB565 match; DB485 is
disjoint in all three identity dimensions. Exact 509/507/45-file inventory snapshots contain no
HLO/StableHLO/XLA/compile/executable/fingerprint-named object, and every anchored engine config
recorded `debug_dump_path=None`. Object payloads were not exhaustively scanned for embedded
compiler IR; this is an inventory-name and sealed-recovery-path result, not content-absence proof.

Classification is `CALLBACK_EXECUTABLE_CLASS_REJECTED;`
`ACCEPTED_CLASS_HAS_NO_SEALED_HLO_RECOVERY_PATH`. Proven class identity does not localize
materialization, fusion, scheduling, collective order or any other optimized operation; the source
diffs contain other diagnostic support changes and compiler-fingerprint identity is an explicit
assumption. The surviving-cache check remains only a dated host observation. No CPU-derived IR,
callback retry, unsupported value tracepoint, Gate-D, DB or performance claim follows. Exact next
after offline review/persistence is separate design review of at most one accepted-`b3c25df`
compile-only dump. Its seven fingerprint triples must equal DB485 in all three identity dimensions
before any HLO is admissible; it executes no decode and remains mechanism-only. The already-rejected
DB518 downstream DSA path does not receive a relaxed boundary or a full-8K retry.

The first and second accepted-tree acquisition snapshots were blocked by Sol before commit or TPU use.
The corrected offline batch binds exact runtime kwargs/pins, unique workers 0--7, raw/decompressed
sizes and one 32-partition scheduled module per M32--M2048 bucket using the established 156
row-parallel-reduction signature. It uses a bounded task-owned driver, retry-until-census cleanup,
read-only same-region mounts and generation-zero manifest replay with `SUCCESS` alone last. Focused
callback/acquisition coverage is 45/45 after the second audit exposed two guaranteed late failures:
the tracked-only vLLM transport omitted generated `_version.py` and an absent nonowner HLO root
failed under `pipefail` before its receipt. The transport now injects a byte-pinned generated file,
imports exact `0.1.dev1+ga30addc75` on every host before model work and seals that identity; the
compactor explicitly distinguishes absent/empty nonowners from invalid roots. Fake storage tests
also execute generation-zero and terminal-last publication, including the exact ledger generation
bound by `SUCCESS`. A third audit then refused dangling-link vacancy, symlink/special-file archive
admission and an incomplete/tampered terminal contract. Both archive walkers now lstat the complete
tree, local/fleet vacancy treats links as occupied, folder markers are unexpected, and terminal
publication requires exact canonical 17-key bytes derived from the manifest, ledger and remote
prefix. Hostile tests cover each class; focused coverage is 45/45 and the corrected adjacent suite
passes 415 tests with 10 expected skips. No TPU, Ray, model load, DB or bucket write occurred. This
remains unreviewed preparation, not evidence; exact next is correction-only Sol review, then
persistence/mirror before any separately approved single compile-only invocation.

The subsequently authorized invocation at pin `6274f760`, tag
`accepted_db485_compile_only_hlo_20260830T003358774368259Z`, completed the exact accepted runtime
with `generate_calls=0`, all seven DB485 fingerprint triples and 8-host load/state integrity. It is
not a successful HLO artifact: every host returned `HLO_OWNER ... count=7`, contradicting the
one-owner/seven-nonowner evidence assumption, so validation failed before copy/seal/publication.
Cleanup is 8/8 authenticated and the same-region failure prefix contains diagnostics only; no
`manifest.json` or `SUCCESS` exists. This is direct evidence of replicated per-host dump
materialization, not proof of independent compilation or identical HLO bytes. The tag is burned and
compact hashes are recorded in
`docs/artifacts/accepted-db485-compile-only-hlo-replica-contract-failure.json`. Gate D remains open.
The reviewed-next correction must gather all eight audit manifests, compare
raw bucket identities, validate worker 0 as canonical only after equality, and retain divergent
payloads before destructive cleanup. It now quiesces before a globally bounded eight-host copy,
round-trip validates every gzip, binds metadata hashes into each host audit, and deletes remote dumps
on the normal path only after eight audits agree and worker 0's canonical payload validates. If
failure preservation starts, deletion instead requires all eight copied sets to validate; otherwise
it refuses every tag-dump deletion and requires eight explicit retention receipts.
Focused hostile tests pass 45/45 at the final affected diff. Before the final hostname/mkdir-only
correction, the complete explicitly CPU-forced adjacent benchmarking suite passed 425 tests with 10
expected skips in 234.68 seconds; its unaffected remainder is reused. A fresh read-only fleet census
is 8/8 clean. Fable remains usage-blocked; the correction-only Sol follow-up found no P0--P2 issue
and approved tracked diff `b72383df...74646` plus failure artifact `b099ca24...56bb` for
commit/push/locked same-region mirror only. The correction remains offline evidence preparation,
and no retry is currently authorized. It is persisted as signed-off commit `4d419506...97c3`, with
local/origin equality and all ten changed files byte-identical in the exact `US-CENTRAL2`
repository mirror. Exact next is a separate execution-only review.

The fresh accepted retry `accepted_db485_compile_only_hlo_20260830T025924791267740Z` supersedes
the failed ownership assumption. Exact eight-host raw identities permitted worker 0 to seal the
canonical seven-bucket archive: 57 objects/162,332,478 bytes, manifest `8bf5abf3...391a`, remote
ledger `c87adfe5...e5bf`, SUCCESS file `d6049f4d...86cf`, generation `1788062144194143` and
8/8 zero-work cleanup. Compact evidence is
`docs/artifacts/accepted-db485-compile-only-hlo-success.json`. It executed no request and makes no
numerical/DSA/Gate-D/DB/performance claim.

`docs/artifacts/db485-layer1-rms-hlo-causality.json` is reproducible with
`scripts/greenfield/classify_db485_layer1_rms_hlo.py`. It proves accepted and DB518 share the same
logical BF16 dense-plus-rounded-carried association. It does not prove that TPU correction metadata
materialized BF16 at the DB518 FP32 tuple/copy boundary, so the missing-unrounded-state physical
cause remains unresolved. The v3 certificate also SHA-pins accepted `fused_computation.16511` and
its BF16-only input bitcast. The qkv-a consumer receives the weighted RMS output as
`bf16[32,6144]`, decodes its FP8 weight to BF16 and accumulates the convolution in FP32; no
pre-round FP32 RMS operand is exposed across that HLO boundary. This is a semantic dataflow
certificate, not proof that any BF16 value physically materialized in HBM.
Sealed WS32 supplies boundary-local pre-dense and next-layer-RMS
evidence only; no complete dense value path is claimed. It binds exact accepted/DB518 source/copy
roles plus SHA-pinned reviewed computations, WS32 local roles, inverse, weight-owner indices,
correction geometry, physical groups
and live roots, with in-scope hostile mutation refusal. Another inspected HLO distinction is
reduction plus weighted output: accepted squares `32x6144` locally
and uses `T(8,128)(2,1)`/window `2x48`/split 0; DB518 squares `1x3072`, reduces over feature-2 and
weights with `T(2,128)(2,1)`/window `1x12`/split 1; WS32 squares `1x1536`, reduces over feature-4
and weights with `T(2,128)(2,1)`/window `1x6`/split 1. Both one-row plans gather afterward. Historical one-row variants already
reject direct-layout, Pallas, source-fused and ownership reformulations. No TPU successor is
authorized; Gate D remains open.

## Gate-D coherent-state inventory and blocker merge

Fable 5 Max was usage-blocked before reviewing evidence. The goal-authorized Sol fallback and an
independent artifact audit both return `NO TPU SUCCESSOR`. No preserved artifact contains the
accepted unperturbed FP32 layer-1 RMS input: accepted `79b813da...9054`, DB550
`f194d757...4298`, DB518 `534bacc5...62f0` and WS32 `2be686ff...eeb1` retain BF16 hidden/cache
boundaries or downstream FP32 values, while callback file `f92d742c...08ebe` contains only a host
reconstruction and belongs to the rejected DB551/DB565 executable class.

An event-1 discriminator requires one candidate's own full 8,156-key history plus its query,
head weights, current key and scorer provenance. DB518 is coherent and already fails; older
one-row candidates do not contain that history, so replay against borrowed accepted/DB518 cache is
invalid. Existing split-state, direct/layout/Pallas/source-fused/scalar/output-owned,
gather-before-weight/double-round, callback and unchanged-run mechanisms are closed. M32/dead-row,
CPU-substitute HLO, tolerance relaxation and mixed-cache routes are invalid. Require a complete
candidate-coherent metadata/SHA capsule before any future JAX or TPU work; otherwise stop. Gate D
remains open and no TPU successor is authorized.

## PP16 feature2 K-half N82 CPU admission

- Capsule: `docs/artifacts/pp16-feature2-qkv-khalf-cpu-admission.json`, SHA
  `33e8dd0a1b0fb8e9c46c1acb3ce05357beff8969fb694464f32f6a071493619c`.
- Sources: PP16 final manifest `b385458f...6bab`, N82 weights/scales
  `6e8b4efd...855d` / `3ca2712f...11cc5`, accepted layer-1 internals file
  `79b813da...9054`, and DB518 result `534bacc5...62f0`.
- Structure: the helper retains the prior gathered/replicated hidden row and full packed weights;
  it selects one disjoint K=3,072 weight half and 24 scale rows per LP2 rank, then performs exactly
  one `{0,1}` FP32 `[32,1,82]` reduction. BF16 appears only after that reduction; there is no new
  gather, host callback or dead row inside the helper.
- CPU arithmetic: accepted full-K and FP32-partial q-a are both exact at `0/2,048`; BF16 partials
  reject at `656/2,048`. Current full-K and FP32-partial are identical on CPU, so CPU cannot decide
  whether the physical TPU boundary changes the result.
- Historical admission only: DB518's layer-1 index cache was reusable for this narrow challenger
  because the candidate changed qkv-a after normalized-history/`wk` cache construction. The
  subsequent coherent event-1 CPU rejection below supersedes the former metal-successor
  requirement. No TPU successor is authorized and no decoder, numerical success, performance, DB
  or Gate-D claim exists.

## PP16 feature2 K-half event-1 CPU rejection

- Capsule: `docs/artifacts/pp16-feature2-qkv-khalf-event1-cpu-rejection.json`, SHA
  `b7f5e43284fcbc2160cefc94d7aa95a7ce9ecbb19bbd60d3fa8d9178a5a1f9ec`.
- The replay authenticates both runtime owners and 22 selective tensor ranges, DB518, accepted
  layer-1 internals/event 1 and the prior admission capsule. It verifies DB518's saved current cache
  row, removes only owner 1/page 15/row 219 and proves all historical rows remain unchanged.
- The captured-q CPU control fails to reproduce DB518 TPU event 1 at 1,723 positions/all scores;
  the intended split-K association fails accepted event 1 at 1,892 positions/all scores/18 set
  members. CPU cannot provide an exact control and the arm is independently downstream of the
  first wrong normalized value/head path.
- Classification is `CPU_EVENT1_ADMISSION_REJECTED;NO_TPU_SUCCESSOR`. No TPU execution is
  authorized; tombstone the K-half arm and return to a new upstream normalized-state mechanism.
  Complete arm values, toolchain, executed sources and all checked input identities are frozen.
  DB518's sealed comparison proves its layer-0 prompt cache exact; no accepted layer-1 cache oracle
  exists, so the sensitivity arm makes no accepted-history claim.

## Gate-D immutable watchpoint frontier

- Contract `configs/greenfield-gate-d-observability.json`, SHA
  `6fcf46066e7e500202baeaffab5a731ff17a40b67f49db711f0ebc51e37914fc`, binds the sealed accepted
  layer-1 internals (`79b813da...9054`) and coherent DB518 result (`534bacc5...62f0`) to their
  code/plan/executable and sibling-evidence identities.
- The stdlib-only `glm_tpu/greenfield/observability.py` snapshots and hashes bounded regular files
  before parsing, validates typed NPY/NPZ storage and selected-array SHAs, refuses mixed candidate
  coherence, and compares ordered watchpoints at the raw-bit level. The CLI runs under `python -S`;
  report creation is canonical, append-only and symlink-safe.
- Canonical report SHA `366898b88ffdd2205ecf94838e5af3d0042c4ce3f735cec7f8ee86bd4487150b`
  classifies `OBSERVABILITY_GAP`: neither authority stores the unperturbed FP32 row entering
  layer-1 RMSNorm. The first downstream divergence is BF16 normalized hidden 2795 (`27bd` accepted,
  `26bd` DB518), then 47 q-a, 4,096 query and 32 head mismatches per owner. Cross-authority
  comparisons remain incomplete: candidate current-key and accepted cache/event-1 are absent.
- Compact capsule `docs/artifacts/gate-d-observability-frontier.json`, SHA
  `4fbf4583826f3db39eb30e285cd97a4f69f5ff5dad13eb3511b7b727d6cb0200`, records reproduction, review,
  source and validation identities. Focused hostile coverage is 19/19; explicitly CPU-forced
  adjacent validation is 334 passed/40 skipped with the same two unrelated WS32 fixture failures.
  Fable returned only its usage-limit refusal; the same Sol reviewer approved the corrections with
  no P0--P2 blocker. This proves the evidence frontier, not the numerical root cause, correctness,
  performance, Gate D or a TPU successor. Future candidates must move upstream and supply complete
  coherent state; no JAX/TPU action follows from this report.

## Gate-D mechanism admission frontier

- Contract `configs/greenfield-gate-d-mechanism-admission.json`, SHA
  `f96d144ecc1bdca7382a0a220b41aa30f3f77b69a7b42dff73fc551d5869a7d3`, authenticates nine sealed
  mechanism families, their structured primitive fingerprints, 15 named candidates, the immutable
  watchpoint frontier and current upstream
  refs. Core/CLI SHAs are `1cacd246...33ba` / `ea18b497...70c7c` and the canonical report SHA is
  `2af563ed...7abb`.
- Family closure artifact `docs/artifacts/gate-d-mechanism-family-closure.json`, SHA
  `545740afbb2f79ecf6783871673c5aa6038b34550b3a120d399b8c3881d4ccf0`, records duplicate closures
  only. Upstream snapshot `docs/artifacts/upstream-glm52-obsolescence-snapshot.json`, SHA
  `76badaf418f3d7bc721eb98802389660a41f16456a40543f86802f6c062e0b1d`, finds new experimental
  GLM5 DSA/indexer/cache/attention paths but no new upstream layer-1 RMS ownership mechanism.
- Compact capsule `docs/artifacts/gate-d-mechanism-admission-frontier.json`, SHA
  `9dafd4c6642686d446c344f9ea3d9a4f5c7d29511b8ecb5e8d02031dcd776315`, classifies
  `NO_ADMISSIBLE_MECHANISM;GATE_D_OPEN;NO_TPU_SUCCESSOR`. All 13 historical/illegal candidates are
  duplicate or contract rejected. The otherwise structurally legal plan-local persistent-FP32
  shadow lacks a candidate-coherent capsule; current upstream DSA work is also downstream of the
  causal frontier. No JAX, TPU, cloud or model work occurred.
- This is finite-catalogue admission evidence, not proof that no solution exists. A successor must
  be genuinely new, true-one-row and topology-local, move or expose `layer1.rms_input_fp32`, and
  pass the existing stdlib observability auditor over typed candidate RMS-input/normalized/cache/
  query/head/current-key/scorer arrays, common identity and layout/owner evidence. Component-wise
  no-symlink path traversal and append-only publication fail closed. Offline admission still never
  authorizes TPU execution.
- Fable session `d6512b94-52af-4a03-85de-d7d5b43ea5cc` was usage-blocked and supplied no opinion.
  The Sol fallback rejected initial staged SHA `5063f72a...7b62` for the opaque-capsule,
  renamed-family and intermediate-symlink bypasses. Corrected SHA `5d23db63...9982` closes all
  three with permanent hostile tests; correction review returned `APPROVE PERSISTENCE`, no P0--P2.

## Direct unrounded persistent-FP32 shadow source/HLO rejection

- Artifact `docs/artifacts/plan-local-persistent-fp32-shadow-source-rejection.json`, SHA
  `ec78266731f1e0730fe108c37e3455ad921c724801693bda5f9245c680e6dbd6`, pins accepted vLLM commit
  `a30addc...d1c`, exact file/function AST hashes, current greenfield sources and accepted
  logical-HLO certificate `a8c9577d...ba21`.
- Source and sealed HLO establish the accepted logical BF16 recurrence, not physical BF16
  materialization. Directly consuming an unrounded shadow at the next boundary removes that round
  and is nonexact; a no-dependency shadow is non-causal; origin-only use duplicates the transient
  fused sum.
- Rounded/widened, algebraically compensated or auxiliary device-only consumers are not closed by
  this audit. They require separate fingerprints and causal HLO/coherence adjudication. The
  artifact binds candidate id plus prior contract/report/compact hashes and supersedes only the
  three narrow interpretations. No JAX/TPU successor is authorized.

## Rounded, compensated and auxiliary-shadow normal forms

- Contract `configs/greenfield-gate-d-shadow-variant-adjudication.json`, SHA
  `28ab4766a3a279624be7625d565eda3ec3169257926e46e855569e75a2b58f44`, binds the accepted
  transient-FP32 normalization/BF16 returned recurrence, seven watchpoints, one-row/local execution
  rules and six proposal forms to authenticated evidence.
- Core/CLI SHAs are `33d60081bb2c0904036c2a55f3fda4aebf095dc57457a278112e340517924463` /
  `6b041474acb925532ef1ec96495bc9e45533e66dd8de320bddfc79ce33c09039`. Canonical report
  `docs/artifacts/gate-d-shadow-variant-adjudication.json`, SHA
  `a9b392712dd685488ad9e07a70274d4f5b61eec1e14c8327e1c28e43c8baa622`, classifies
  `NO_OFFLINE_COMPLETE_SHADOW_VARIANT;AUXILIARY_DEVICE_VARIANTS_REMAIN_UNADJUDICATED;GATE_D_OPEN;NO_TPU_SUCCESSOR`.
- Rounded/widened primary state is duplicate-closed; restoring unrounded state is semantic drift;
  cancelled compensation without a consumer is non-causal; host/full-pod/dead-row observation is
  illegal. Compensated and tuple-carried device variants remain separate unresolved identities
  under one search umbrella; their physical equivalence is unproved. Each still needs source/AST
  binding, causal StableHLO, offline coherent state and admission before compile-only review. Exact
  optimized TPU HLO is an output required before later numerical review, not a compile prerequisite.
- Fresh source snapshot `docs/artifacts/upstream-glm52-shadow-search-20260830.json`, SHA
  `263d4e7a1d846854fbba3d8a39097241996e93f6517c592fb57ebd6597fb5aa8`, pins vLLM main
  `5e71a11e...deb5`, TPU Inference main `c8593a4c...6005` and PR-2324 head `c2822bd7...4a17`.
  No upstream residual-shadow mechanism was found. This is source-search evidence only.
- Hostile validation passes 17/17: renamed/deleted/added/class-drift forms, semantic mislabeling,
  every locality prohibition, boolean schema/unhashable members,
  recurrence/watchpoint weakening, forged certificate fields, evidence drift, hostile JSON,
  final/intermediate/evidence symlinks, occupied outputs, byte stability and `python -S` without
  JAX. The versioned contract classifies declarations only; it does not bind a future implementation
  to those declarations. This is not Gate-D closure.
- Compact capsule `docs/artifacts/gate-d-shadow-variant-adjudication-frontier.json`, SHA
  `6baec8732175021d72c4663adeab50e1bf2c3a7c11fe85a2bbb030372a16dec0`, binds reproduction,
  source snapshot, validation and the three exact staged-review identities. Sol blocked the first
  two drafts for workflow/canonicalization/provenance/security/claim-scope issues and one remaining
  documentation contradiction, then approved corrected staged SHA `e0da721b...922b` for persistence
  with no P0--P2. The approval explicitly authorizes no JAX, compilation or TPU work.

## Gate-D capsule constructability and precompile sequencing

- `glm_tpu/greenfield/capsule_constructability.py` plus
  `scripts/greenfield/audit_capsule_constructability.py` authenticate the old candidate NPZ,
  inventory the exact seven watchpoints and inspect admission authority from SHA-bound source AST
  under `python -S`.
- Contract `configs/greenfield-gate-d-capsule-constructability.json` SHA
  `632c30643032b787bf82ddebabfbee29211cd450f0e8b42aac6f8a24f91cd40a`; canonical report
  `docs/artifacts/gate-d-capsule-constructability.json` SHA
  `77fa743228fb322a1ca0ac1190f73f27a59e12cd56b4a6637e5980c38bb1a11f`.
- DB518 lacks candidate position-8155 `layer1.rms_input_fp32` and `layer1.current_key`; every
  present state array remains old DB518 code/HLO authority and is unusable as either new variant's
  candidate authority. The observer FP32 bytes are rejected perturbation evidence, not an oracle.
- Observability v1 refuses both explicit `stablehlo` identity and an uncompiled candidate;
  admission v1 binds a non-null executable-identity SHA but no explicit source/AST authority in its
  capsule tuple. Do not overload its generic fingerprint kind. Preserve v1. The exact next is an
  append-only precompile admission v2, then source-bound StableHLO and a coherent offline capsule.
  This result authorizes no JAX, compile or TPU action and does not close Gate D.

## Gate-D precompile admission v2

- Append-only v2 is implemented separately from historical admission v1 in
  `glm_tpu/greenfield/gate_d_precompile_admission.py`, with contract
  `configs/greenfield-gate-d-precompile-admission-v2.json`, CLI
  `scripts/greenfield/admit_gate_d_precompile.py` and canonical report
  `docs/artifacts/gate-d-precompile-admission-v2.json`.
- V2 authenticates the exact inherited v1 contract/core/frontier and closed fingerprints, the
  sealed two-variant normal-form catalogue, committed Git blob/AST identity for concrete or
  non-admissible declarative source, a content-derived plan, parsed StableHLO structure, explicit
  plan-local topology and the exact layer-1/position-8155 typed eight-watchpoint NPZ schema. The
  parent auditor is stdlib-only; structural HLO validation uses a SHA-bound offline jaxlib-MLIR
  parser running as a bound non-root UID in an exact content/symlink/safe-mode-tree-SHA-bound
  root-owned CPython runtime with `-I -S`, `cwd=/` and no inherited import environment. Provisioning
  is trusted manual administration outside parser authority: only a fixed root-owned mode-0555
  installed tool is run as `sudo -n /usr/bin/python3 -I -S /opt/glm-tpu/bin/`
  `provision_gate_d_python_runtime.py ...`; that exact pre-start command or equivalent absolute-path
  shebang is the security boundary, while the runtime check only detects accidental misinvocation.
  Admission binds it byte-for-byte to reviewed source,
  and it nofollow-validates exact root:root `/opt` parents. Atomic `RENAME_NOREPLACE` refuses every
  pre-existing target, including dangling symlinks and publish races. Parser Python sources load from exact sealed memfds; parser
  native libraries must map from those same sealed inodes, and every other file mapping must be a
  recorded root-owned immutable dependency. It imports neither `jax` nor a backend. It rejects host effects, implicit,
  ring, nonlocal or greater-than-four groups, changed accepted-primary backward slices, mixed
  authority, hostile archives and symlinked or occupied evidence paths. Auxiliary causality and its
  candidate-specific slice are checked separately from the unchanged primary. The tuple survivor
  binds its exact concrete weighted/double-rounded RMS operators and real false-default device
  caller; abstract fixtures remain non-admissible. Both BF16 RMS operands
  are required, and their independently derived 6,144-value FP32 sum must byte-match the declared
  `layer1.rms_input_fp32` array.
- Current classification is
  `NO_PRECOMPILE_CANDIDATE_ADMITTED;GATE_D_OPEN;NO_JAX_OR_TPU_SUCCESSOR`.
  `auxiliary_device_tuple_dependency` has concrete committed source/callsite authority at
  `c8b2200` and exact sealed `PP16_LP2` plan/watchpoint authority. It remains refused only for
  causal StableHLO and a coherent candidate capsule. `compensated_auxiliary_dependency` retains
  all four gaps. No JAX, compile, model, cloud or TPU work occurred.
- Canonical core/contract/report/locality SHAs are respectively `1d434b0f...cee23`,
  `72673fbc...fb48`, `e1fcf314...aff7` and `49cf6bb1...25eb`. Validator SHA is
  `2f4e73b0...c37a7`; installed/source provisioner and runtime-tree SHAs are
  `2b9c8c2b...0594` and `308748a9...d616`. Source certificate
  `gate-d-tuple-auxiliary-source-authority.json` is `c95c8aa1...f0e1`; PP16 plan authority is
  `98b4fa21...7880` with content plan SHA `d824c19c...5833`. Focused hostile admission passes
  72/72; source-only plus adjacent offline suites pass 71/71.
  These are admission-tool tests, not candidate or Gate-D evidence.
- Fable session `d6512b94-52af-4a03-85de-d7d5b43ea5cc` was usage-blocked and supplied no opinion.
  The Sol fallback approved exact staged SHA `5003b44e...55bf` for persistence with no P0--P2 and
  explicitly no JAX/backend/model/cloud/compile/TPU authorization or Gate-D closure.
- The same Fable session remained usage-blocked for the plan batch. Sol withheld the first draft
  after the local audit found a movable sealed-stage owner group, then approved corrected staged
  SHA `775b3644...c8fa` with no P0--P2 after exact stage-zero binding and a fully rebound migration
  attack. The verdict remains persistence-only with no execution or Gate-D authorization.
- A complete synthetic fixture now proves the immutable parser path and remains refused for two
  explicit reasons: `MISSING_EXECUTABLE_SOURCE_AUTHORITY` and
  `MISSING_PINNED_COHERENT_CAPSULE_PRODUCER`. The concrete tuple candidate still has no causal
  StableHLO artifact at all, so it remains refused for `MISSING_CAUSAL_STABLEHLO_AUTHORITY` plus
  `MISSING_CANDIDATE_COHERENT_CAPSULE` and cannot request compile-only review. Structural fixtures
  remain preparation, not candidate lowering, optimized TPU HLO or numerical truth.
- Sol blocked the first sealed-parser draft because the child runtime and early stdlib imports were
  still same-UID writable, CWD shadowing remained possible, maps parsing was lossy for paths with
  spaces/renames, and the parent did not post-seal rehash. None of that draft was persisted or used
  to authorize work. The corrected hostile suite covers all four defects; ephemeral memfd inode
  values remain exact internal comparisons but are emitted only as deterministic counts.
- Sol blocked the first root-runtime correction too: tree hashing omitted modes, provisioning could
  preserve setuid/setgid bits or xattrs, mutable repo code was passed directly to sudo, the mapping
  scan preceded parsing, and raw runtime/dependency inodes leaked into authority hashes. The final
  correction binds the root directory and all canonical safe modes, strips/rejects privilege bits
  and xattrs, refuses root validator execution, demotes provisioning to a fixed root-owned
  trusted-admin tool outside parser authority, nofollow-validates exact safe staging parents,
  performs the authority scan only after both parses and all validation walks, and normalizes raw
  identities and numeric UID only after parent corroboration.
- The canonical admission artifact itself performed no JAX/backend/compile work. Separately, one
  forced-CPU lowering-only diagnostic inspected the committed source and found that its real HLO
  includes the `[6144]` weight, an explicit broadcast, `chlo.square`, and caller epsilon `1e-5`.
  It produced no authority artifact; the synthetic fixture must not be relabeled as real lowering.
- Source-only producer `scripts/greenfield/produce_gate_d_tuple_auxiliary_stablehlo.py` is the
  proposed replacement for that diagnostic. It binds the exact committed RMS source/callsite and
  PP16 authority, clean producer pin/bytes, sealed CPython and an explicit root-owned JAX dependency
  capsule that excludes libtpu/plugins and records all loaded Python/native files. Its stdlib-only
  corrected builder produced source-tree SHA `55233c63...a0df` without importing JAX. Sol approved
  pre-install staged SHA `645e36e5...b515`; the sealed provisioner installed the exact tree at
  `/opt/glm-tpu/gate-d-jax-site-55233c63939e` root-owned with its manifest mode 0444. Exact producer
  SHA `2513a305...2182` was installed root:root mode 0555 at its fixed path via an immutable
  provisioned one-file source and no-replace hard link. The producer lowers only abstract real-shape accepted/candidate RMS
  functions and writes raw plus reversibly annotated StableHLO through inode-pinned append-only
  dirfds. Static/dynamic builder/producer tests pass 5/5 and registry-plus-source passes 9/9. It has not run
  and is not StableHLO, executable-compilation, numerical or Gate-D evidence yet. It has not been
  invoked; commit/push/mirror and a separate invocation review remain mandatory. Real `chlo.square`
  or signature refusal by the current synthetic-derived parser is a valid diagnostic outcome, not
  permission to rewrite the raw HLO.
  Sol blocked the superseded tree `561118be...d151`: its manifest would become root-only after
  provisioning and the generic producer shebang did not seal the interpreter before startup. The
  corrected builder requires future unprivileged file/directory access and the producer shebang
  names the exact sealed interpreter with `-I -S`; the blocked tree must never be installed.

- The one Sol-approved post-persistence producer start at clean pin `d8f5830` failed before JAX
  import, output-directory creation or lowering: sealed CPython does not export the optional
  `fcntl.F_ADD_SEALS` name. The burned output path remains absent and no retry is authorized.
  `docs/artifacts/gate-d-stablehlo-producer-pre-jax-failure.json` (`9a5c88a1...4bc7`) binds the literal command, old
  producer/site identities and no-output classification. Local Linux UAPI headers give numeric
  commands 1033/1034 and seal mask 15; corrected source `7292477b...0b6e` uses them while retaining
  an exact post-seal mask check. A sealed-runtime numeric memfd probe and source regression pass.
  Sol approved exact staged correction `d0bea1d5...2620` for install only. One-file tree
  `4bf3edb4...a5d0a` was provisioned root-owned and atomically linked into the fixed path; fixed and
  new source share exact SHA/inode, mode 0555 and link count two, while the old immutable source
  remains at link count one. Expanded hostile coverage passes 82/82. The correction is not yet
  persisted or invoked and supplies no StableHLO authority.

## Corrected scalar-frontier research adjudication

- `docs/greenfield/GATE_D_SCALAR_FRONTIER_V2_ADJUDICATION.md` accepts only the corrected report's
  T1 derivation: finite BF16 operands determine one correctly rounded FP32 sum. DB550 seals the
  accepted operands; every future candidate must seal its own two operands under the candidate's
  complete authority tuple.
- DB518 cannot be repaired retrospectively because it lacks its candidate RMS operands/input.
  Admission v2 turns the derivation into a capsule invariant rather than accepting a label or
  rejected callback observation.
- T2--T4 are not local mechanism authority. Their written inversion omits the per-element norm
  weight at the retained weighted, double-rounded qkv-a input, and sealed replay already reports no
  global-scalar solution for the tested one-row formulas. No compile or TPU successor follows.

## Real tuple-auxiliary StableHLO authority; coherent capsule remains missing

- One forced-CPU lowering-only acquisition at clean producer pin `6a95811d...bead` produced the
  exact committed accepted and tuple-candidate RMS graphs at real shapes and epsilon `1e-5` under
  JAX/JAXLIB `0.10.1`. It compiled no executable, executed no array/model, loaded no TPU plugin or
  libtpu, and performed no cloud/TPU workflow.
- Local acquisition
  `/home/gianl/gate-d-runs/greenfield_gate_d_tuple_auxiliary_stablehlo_20260830T224500Z` is archived
  byte-identically in `US-CENTRAL2` at
  `gs://driftbench-dsv4-uc/results/greenfield/glm52/gate_d_stablehlo/`
  `greenfield_gate_d_tuple_auxiliary_stablehlo_20260830T224500Z/`. Accepted raw, candidate raw,
  reversibly annotated candidate, producer receipt and `SUCCESS` SHAs are respectively
  `649cdc9e...cc82`, `3645d16e...3e43`, `87e7fb6b...9791`, `8042acf1...5eb4` and
  `0839d98f...8979`.
- Admission v2 now binds canonical receipt/SUCCESS bytes, exact forced-CPU environment, producer
  Git/blob/source identity, source authority, plan authority and the exact 533-Python/46-native
  loaded-dependency manifests by count and canonical-list digest. Noncanonical/dot-dot paths,
  truncation, allowed-root rebinding, path escape, backend/source/annotation/producer/SUCCESS drift
  attacks fail closed. The exact parser allowlist adds only `chlo.square` and pins its three CHLO
  implementation files byte-for-byte.
- Immutable exact parsing proves accepted and candidate primary backward slices identical at SHA
  `5037b5a7...5610`; carried and weighted primary slices are `b4108427...5fad` and
  `41097082...54bb`. The candidate's rooted FP32 RMS-sum auxiliary slice is
  `02eee1d7...2f7c`, has no post-source operations and neither graph contains a collective. Parser
  authority remains isolated, sealed-memfd and no-`jax`.
- Exact module/function metadata is also causal authority: accepted and candidate require one
  partition/replica, exact module/function symbols, public visibility, function types, result names
  and no argument/result sharding or alias extras. Replica, partition, symbol, visibility, sharding
  and alias attacks fail closed rather than preserving a misleading zero-collective slice result.
- Certificate `docs/artifacts/gate-d-tuple-auxiliary-stablehlo-authority.json` is
  `8cc45b81...338b`; canonical admission report
  `docs/artifacts/gate-d-precompile-admission-v2-stablehlo.json` is `454090e7...f25f`; compact
  adjudication `docs/artifacts/gate-d-tuple-auxiliary-stablehlo-parser-adjudication.json` is
  `0e3c6259...5c3c`. StableHLO authority hash is `19f43f39...6d88` and contract/core/parser SHAs are
  `9006a42b...67d6`, `2325cbd6...ef23` and `4338229a...6b6d`.
- At this historical StableHLO milestone, focused plus hostile admission passed 89/89. The tuple
  reported one remaining capsule reason and the compensated declaration retained all four gaps.
  Later coherent tuple replay closed that historical next step with a numerical rejection; current
  compensated source authority independently removes only its source gap.
  Classification remains
  `NO_PRECOMPILE_CANDIDATE_ADMITTED;GATE_D_OPEN;NO_JAX_OR_TPU_SUCCESSOR`.
- The historical exact next was one pinned replayable PP16 capsule. That action is complete and
  rejected; do not rerun it. At this historical StableHLO milestone, the next step was compensated
  candidate-specific plan/watchpoint authority; that later batch is recorded below. No compile-only
  optimized-HLO acquisition or TPU action was authorized here.

## Tuple-auxiliary coherent-capsule source and trust-boundary hardening

- Default-off replay `glm_tpu/greenfield/benchmarking/gate_d_tuple_capsule.py` has SHA
  `9b9d11d0...03b4`; source-only producer
  `scripts/greenfield/produce_gate_d_tuple_auxiliary_capsule.py` has SHA `2cbf8ed9...8268`.
  Exactly two CPU owners execute the real stage-local layer-1/event-1 path and return explicit
  owner-axis evidence for every replicated critical value. Nothing in this batch was installed or
  invoked.
- The producer takes same-fd snapshots of all eight upstream authorities before execution, reads
  safetensors through retained descriptors, emits 22 exact tensor receipts and revalidates
  path/inode/size/slice/source/dependency identities before provisional `SUCCESS`-last local
  publication. Admission independently reconstructs runtime-backed inputs and DB518 cache,
  derives the FP32 RMS input from its two sealed BF16 operands, fixes the accepted event/RMS hashes,
  requires two agreeing device owners and rejects nonfinite BF16/FP32 values.
- The first same-scope Sol review withheld because the outer execution authority could select a
  different producer/replay closure. The correction independently pins producer path/blob/SHA,
  replay path/blob/SHA and every replay-visible committed `glm_tpu` blob except the separately
  contract-bound admission verifier. The reproduced 139-blob manifest SHA is
  `f3683029...c0b0`; hostile producer-path/blob, replay-blob and imported-kernel substitutions fail
  closed.
- Core/contract/test SHAs are `250f41c2...3568`, `9d9ede40...f539` and
  `99f877d5...7e26`. Full admission coverage is 119/119 and adjacent constructability/admission/
  observability/shadow coverage is 69/69. Fable-max was usage-blocked; corrected Sol verdict is
  `APPROVE PERSISTENCE` with no P0--P2 and explicitly no installation, invocation, JAX/compile,
  cloud/TPU, admission or Gate-D authorization.
- Gate D remains open. Exact next after durable persistence is a separate review of trusted manual
  installation and exactly one bounded protected replay; its output must then be independently
  archived/sealed and pass offline admission before any compile-only successor is considered.

## First coherent-capsule start: pre-execution SUCCESS-format failure

- Persisted pin `e82fe83` was pushed and mirrored in `US-CENTRAL2`; reviewed producer
  `2cbf8ed9...8268` was installed root:root mode 0555. Exactly one serialized process start used a
  clean environment and the now-burned tag
  `greenfield_gate_d_tuple_auxiliary_capsule_20260831T020400Z`.
- The process imported the sealed JAX stack, then failed before `_execute`, numerical arrays,
  output creation, model, cloud or TPU work. Runtime `SUCCESS` is an exact 88-byte sha256sum-format
  line with SHA `dbef7e36...ee7e`; the producer wrongly treated it as JSON. The output path remains
  absent. Failure artifact `gate-d-tuple-capsule-runtime-success-format-failure.json` is
  `382ff46a...31e0` and the one-shot has no retry authority.
- Corrected producer `d3082c2e...f869b` checks the exact line bytes and manifest self SHA. Hostile
  JSON, spacing, tab, hash and suffix variants are rejected. Admission repins exact producer Git
  object `6f0d31ee...14f0`; core/contract/test SHAs are `49d0d868...f6713`,
  `ae00f0ad...47f5d` and `e6d63e08...3ebd7`.
- Admission coverage passes 124/124 and adjacent Gate-D coverage passes 69/69. Fable-max was
  usage-blocked; same-scope Sol approved persistence only with no P0--P2. No reinstall/retry,
  archive/admission, compile, cloud/TPU or Gate-D successor is authorized by this correction.

## Second coherent-capsule start: pre-execution runtime-data-root failure

- Persisted pin `3a60012` and corrected installed producer `d3082c2e...f869b` were independently
  reviewed for one fresh-tag process start. The exact tag was
  `greenfield_gate_d_tuple_auxiliary_capsule_20260831T022000Z`.
- The process imported sealed JAX, then failed before `_execute`, numerical arrays, output creation,
  model, cloud or TPU work. The local authority root has the exact manifest and SUCCESS but no
  payload tree; its destination filenames resolve under the exact read-only same-region gcsfuse
  root instead. Both stage-zero payload files are present there at 27,176,812,872 bytes each. The
  burned output remains absent. Failure artifact
  `gate-d-tuple-capsule-runtime-data-root-failure.json` is `651cc1ba...c65cee`.
- The source correction independently binds the local manifest/SUCCESS authority and mounted
  payload root. Producer and admission require exactly one `/home/gianl/gcs-models` mount with
  `fuse.gcsfuse`, source `driftbench-dsv4-uc`, root `/`, and `ro,nosuid,nodev`; the exact mount
  record must survive payload verification unchanged. The longest covering mount for the data root
  and both owner payloads must be that authority; every opened descriptor must match its mount ID
  and device major/minor. Writable, wrong-source, wrong-path, duplicate and nested mount records
  fail closed.
- Corrected producer/blob are `a81fd59e...1794a`/`d82cfff6...e5b1`; admission core/contract/test
  SHAs are `a7818911...d4ce`, `228bbfe6...e4ea` and `41c6d1df...4069`. Full admission passes
  131/131 plus 69/69 adjacent Gate-D tests. A live descriptor-only preflight bound both
  27,176,812,872-byte payloads to mount ID 73 and device 0:43 without reading tensor payloads.
  No install, retry, archive/admission, compile,
  cloud/TPU or Gate-D authorization follows from this source correction.

## Successful corrected tuple-capsule replay; outer authority still pending

- At pushed and byte-mirrored pin `17a3518`, the corrected producer was provisioned from sealed
  one-file tree `4b8a9180...d5613` using the fixed no-replace runtime provisioner. The new current
  and versioned paths are the same root-owned 0555 inode with SHA `a81fd59e...1794a`; spent
  `d3082c2e...f869b` remains preserved by its versioned inode.
- One separately reviewed start at fresh tag
  `greenfield_gate_d_tuple_auxiliary_capsule_20260831T031000Z` passed in 12.5 seconds. It is a
  forced-two-CPU bounded replay only: no JAX plugin, libtpu, model, decoder or TPU work. Read-only
  same-region gcsfuse payload access occurred; no cloud workflow or infrastructure mutation did.
  SUCCESS-last publication contains eight watchpoints and 22 exact tensor receipts. State/input/
  device/receipt SHAs are `68ee47b1...f236`, `dd5f1cbb...8b1b`, `ca7c2b9c...bd68` and
  `afbc4a94...21a1`; capsule/SUCCESS SHAs are `aa7425fd...7ef5`/`c8c5fb84...3b24`.
- Read-only postflight rehashed all six files, verified the SUCCESS tuple, checked all three NPZs,
  confirmed the process gone and left Git clean. This is candidate-coherent local evidence, not
  admission or numerical acceptance.
- New no-JAX outer-authority builder/test SHAs are `fc0de9a2...ecbe2` and
  `26148da4...8b1c`; hostile receipt, shape and canonical-JSON coverage passes 4/4. It remains
  source-only pending adversarial review, persistence and separate execution authorization. The
  admission contract still contains null capsule bindings; no archive, admission, compile, TPU or
  Gate-D successor is authorized.

## First outer-authority build: pre-argument Python-version failure

- At clean pushed/mirrored pin `85c187e`, one separately reviewed no-JAX builder start used
  `/usr/bin/python3 -I -S`. That launcher is Python 3.10.12, while importing the greenfield package
  requires `enum.StrEnum` (Python >=3.11). It failed during import before arguments, capsule reads,
  output creation, admission, archive, cloud workflow, infrastructure mutation or TPU work.
- The intended output remains absent and the one-shot is burned. Canonical negative artifact
  `gate-d-tuple-authority-builder-python-version-failure.json` (`e27e2721...87bc0`) records the
  exact invocation, launcher hashes and scope. Fable-max was usage-blocked and supplied no opinion.
- No code correction is implied: the builder's tests run under Python 3.12.13. The launcher-only
  successor is the existing sealed Python 3.12.13 executable SHA `02104489...acec7`, but it has no
  retry authority until the negative evidence is reviewed, committed, pushed and mirrored and a
  separate review approves one new process start. The contract remains null-bound.

## Tuple-capsule execution authority: coherent offline rejection

- At clean pushed/mirrored pin `73b51c0`, one separately reviewed start with sealed Python 3.12.13
  built `gate-d-tuple-auxiliary-capsule-execution-authority.json` (`ab0840bf...915a0`, 142,157
  bytes). The no-JAX builder process exited successfully and is gone.
- The authority preserves matching accepted RMS operand hashes and event-1 valid count 2048, but
  candidate event positions `bb199543...aeeec` differ from accepted `e55e66c6...8ad7`, and
  candidate scores `6e666fe9...e05c` differ from accepted `a61587a9...b0e7`.
- This is immutable coherent negative evidence. The independent accepted-output equality check in
  `gate_d_precompile_admission.py` must reject it. Do not bind the null contract or run a
  redundant admission. The tuple-auxiliary mechanism has no compile/TPU successor and does not
  close Gate D; retain its capsule/producer only as observability infrastructure and a tombstone.

## Compensated-auxiliary concrete source authority; source only

- Committed source pin `e16d74f` adds a distinct false-default compensated auxiliary without
  changing the accepted primary RMS arithmetic. The rooted auxiliary graph widens the BF16
  carried residual, subtracts it from the original FP32 transient behind a barrier, then adds the
  correction behind a second barrier. This is a graph identity hypothesis only; exact numerical
  cancellation under lowering/execution is explicitly unproved.
- `gate-d-compensated-auxiliary-source-authority.json` (`237095c7...6fb0`) binds both committed
  files, five exact symbols, real caller imports/guards/selection/return and source semantic SHA
  `33d98c89...fd5`. Host effects, primary drift, import rebinding, restoration/operator/barrier,
  result and caller-control mutations fail closed.
- Admission core/contract/current-report SHAs are `fde1847f...b252`, `0ce578d5...ddeb` and
  `9e2f3b5e...594c`. The report remains
  `NO_PRECOMPILE_CANDIDATE_ADMITTED;GATE_D_OPEN;NO_JAX_OR_TPU_SUCCESSOR`: compensated source is
  authenticated, while plan, causal StableHLO, pinned producer and candidate-coherent capsule
  authority remain absent. Historical tuple source/capsule tests now read their original Git
  commits rather than silently rebasing sealed authority onto current files. Full admission
  coverage passes 140/140.
- Fable 5 Max returned a hard usage-limit refusal and supplied no opinion. The same-scope Sol
  fallback first withheld persistence on misleading tuple-tombstone wording and two stale
  references. After correction and complete re-verification, Sol returned `APPROVE PERSISTENCE`
  with no P0--P2; authorized scope is source-only persistence.
- No JAX lowering, compilation, numerical execution, model, cloud workflow or TPU action occurred.
  Exact next after persistence is a separate source-exact plan-authority batch, reusing the sealed
  PP16 physical topology only after proving that reuse is semantically applicable; causal
  StableHLO remains a later separately reviewed batch.

## Compensated-auxiliary PP16 plan/watchpoint authority; offline only

- `gate-d-compensated-auxiliary-pp16-plan-authority.json` has file SHA
  `7d0a5615...dbdf` and content-derived plan SHA `eb2c050b...b6dc`. It reuses the complete sealed
  PP16 physical group order/topology and exact stage-zero owners, but unlike the historical tuple
  plan it additionally binds candidate id, mechanism fingerprint, verified compensated source
  authority SHA `c38492a0...b4c3` and source pin `e16d74f`.
- The candidate-specific frontier map is exact: mapped source result
  `restored_input_rms_fp32[2,1,6144]`, source owner axis 0 with ids `[0,1]`, required bitwise owner
  agreement and selection prefix `[0,0]` map to capsule
  `layer1.rms_input_fp32.value[6144]`, storage `<f4`. It binds ordered operands `hidden_update`,
  `residual` and the later independent derivation `binary32.rne.add_after_exact_bf16_widen`. This
  names what a future coherent capsule must prove; it does not assert that the current source graph
  numerically cancels after lowering.
- The verifier rejects fully rebound candidate id, claim scope, source authority/code pin,
  mechanism fingerprint, source-result, operand-order and derivation mutations. It also rejects
  migration of the compensated binding into the tuple plan, PP8 rebinding, extra binding keys and
  bool/float substitutions for exact integer and schema-version fields. Admission passes 158/158; adjacent
  Gate-D suites pass 69/69; the append-only report reproduces byte-identically.
- Core/contract/test/report SHAs are `206fd203...eb97`, `6559939a...0a25`,
  `d82ebb74...680f` and `3281afec...8540`. Classification remains
  `NO_PRECOMPILE_CANDIDATE_ADMITTED;GATE_D_OPEN;NO_JAX_OR_TPU_SUCCESSOR`. The compensated reasons
  are now exactly `MISSING_CAUSAL_STABLEHLO_AUTHORITY`,
  `MISSING_PINNED_COHERENT_CAPSULE_PRODUCER` and `MISSING_CANDIDATE_COHERENT_CAPSULE`.
- No JAX import/lowering, numerical execution, model, cloud workflow or TPU action occurred. Exact
  next after review/persistence is a separate compensated causal-StableHLO source batch. Plan
  persistence alone cannot authorize its process start.
- Fable 5 Max returned a hard usage-limit refusal and supplied no opinion. Same-scope Sol initially
  withheld persistence on PP8 rebinding, permissive frontier types, ambiguous mapped-owner
  selection, stale prose and permissive JSON schema-version equality. After all findings were
  corrected and the complete suites rerun, Sol returned `APPROVE PERSISTENCE` for staged diff
  `b4f6d3cf...aced1` with no remaining P0--P2. Authorization is commit/push/locked same-region
  mirror only; no lowering, JAX, model, cloud/TPU, admission or Gate-D successor is authorized.

## Compensated causal-StableHLO producer source and parser contract; offline only

- `scripts/greenfield/produce_gate_d_compensated_auxiliary_stablehlo.py` is a distinct default-off
  producer, SHA `a9431e8e...eed9`. It binds compensated source pin `e16d74f`, certificate
  `237095c7...6fb0`, PP16 plan file `7d0a5615...dbdf`, exact committed source blobs/ASTs, sealed
  interpreter/site identities, fixed future installed path and forced-CPU abstract lowering only.
  It names module `jit_gate_d_compensated_auxiliary_rms` and third result
  `result.restored_rms_input_fp32`; it cannot silently reuse the tuple producer identity.
- The immutable parser request now includes exact candidate id. Tuple and compensated candidates
  have separate allowed module/result metadata, producer source/installed paths and Python
  dependency-manifest digests. Hostile compensated module-symbol/result-name changes fail closed;
  the historical real tuple StableHLO still validates exactly.
- StableHLO producer receipt, SUCCESS and causal-certificate schema versions now pass strict
  integer validation. JSON `1.0`/`2.0` and bool aliases fail closed instead of comparing equal to
  integers in Python.
- Validator/core/contract/admission-test/new-producer-test SHAs are `17fadedf...0c19`,
  `5403069b...b6e7`, `3f1c817c...55f4`, `da6e29bf...24a2` and `286cd38d...8259`. Admission passes
  166/166; adjacent plus both producer-source and tuple-source suites pass 83/83. Canonical report
  `gate-d-precompile-admission-v2-compensated-stablehlo-source.json` is `f495209a...02f1` and still
  reports no admitted candidate, Gate D open and no JAX/TPU successor.
- No install, JAX import, lowering, compilation, numerical/model execution, cloud workflow or TPU
  action occurred. After review/persistence, immutable provisioning and one fresh-tag forced-CPU
  abstract lowering require a separate explicit review. This batch supplies no process-start
  authority.
- Fable 5 Max was attempted once with the exact batch context but returned a hard usage-limit
  refusal and supplied no opinion. The existing independent Sol reviewer found no code-side
  P0--P2 and initially withheld only on ambiguous current/historical validator wording in the
  observability playbook. After the metadata-only correction, Sol returned `APPROVE PERSISTENCE`
  with no remaining P0--P2. Approval is limited to commit, push and locked same-region mirror; it
  grants no installation, process-start, admission or Gate-D-successor authority.

## Compensated causal-StableHLO producer immutable installation; not invoked

- At clean local/origin and byte-mirrored pin `361a573668e1414c384d7d9cdcca6619cb6d3798`, Sol
  approved only the exact immutable-provisioning procedure. Fable 5 Max remained hard
  usage-limited and supplied no opinion.
- Sealed provisioner `2b9c8c2b...0594` installed the one-file canonical-safe tree
  `06c2a601...5f60f` at versioned path
  `/opt/glm-tpu/gate-d-compensated-stablehlo-producer-a9431e8e`. The producer was hard-linked
  without replacement into the fixed bin path; both entries are the same root:root mode-0555,
  xattr-free inode with link count two and exact SHA `a9431e8e...eed9`.
- Canonical local-install evidence is
  `docs/artifacts/gate-d-compensated-stablehlo-producer-install.json`, SHA `531c7876...51cd`.
  No producer invocation,
  JAX import, output, lowering, compilation, model/cloud/TPU work, admission or Gate-D closure
  occurred. One fresh-tag forced-CPU abstract lowering requires a separate review after this
  install evidence is persisted.

## Compensated causal-StableHLO acquisition/archive and pre-parser launcher failure

- One reviewed forced-CPU abstract-lowering start at clean pin `8333222` succeeded in 4.35 seconds:
  CPU/one device, 533 Python and 46 native dependencies, no executable/array/model/libtpu/plugin/
  cloud/TPU work. Accepted/candidate-raw/candidate-annotated/receipt/SUCCESS SHAs are
  `649cdc9e...cc82`, `c93f0e2d...79ef`, `baaf8572...eea6`, `a68b9ec9...5f03` and
  `fa4b12c3...7432`.
- Same-region archive publication was corrected before mutation: four generation-zero
  nonterminals were size/CRC32C/generation/SHA verified, then `SUCCESS` was published alone and
  last as generation `1788156932667954`. The final inventory is exactly five objects.
- Certificate `gate-d-compensated-auxiliary-stablehlo-authority.json` is `af109e0f...eb90`; bound
  contract draft is `f694a535...7c81`. Neither claims parser success.
- The first approved offline parser parent used `/usr/bin/python3` 3.10.12 and failed during package
  import on missing `enum.StrEnum`, before argparse, output creation, parser child or JAX. Failure
  artifact `gate-d-compensated-stablehlo-admission-launcher-failure.json`, SHA `8cbd49fb...57fd`,
  records no output/no retry.
  The fix is launcher-only: use the existing sealed Python 3.12.13 after separate persistence and
  review. Fable session `d525bf0a-7f03-4b4d-9ed3-07eb3cadf7d3` was hard usage-limited/no opinion.
- The sealed-Python retry reached the immutable parser, but the parent fail-closed with
  `INVALID_CAUSAL_STABLEHLO_AUTHORITY`. Artifact
  `gate-d-compensated-stablehlo-parser-result-drift.json` (`7fbc5c76...40d0`) records no
  JAX/backend/device successor. The stale expected hash `d55e5f45...d9a43` exactly matches a
  synthetic `restored - correction` fixture, not the committed two-optimization-barrier restored
  result. Pure canonical reconstruction predicts `b0afe414...18ce6`; only a later immutable-parser
  run can promote that hypothesis.
- Source-only correction aligns the fixture to committed graph structure, pins all six auxiliary
  operations and emits compact expected/observed mismatch fields. Core/test/contract SHAs are
  `fa939326...aa837`, `a11cda4b...077f` and `aeb2f45a...60071`; focused authority/attack checks pass
  8/8 and the compensated subset passes 32/32. No real-parser retry is authorized before separate
  review and persistence.

## Compensated causal-StableHLO admission; capsule frontier remains

- At clean, pushed and byte-mirrored commit `cb71b39e5a1f2c5c36fd74be5c2e62e516b9fb09`, one
  separately reviewed sealed-Python 3.12.13 `-I -S` admission run consumed exact contract
  `aeb2f45a...0071`. It created fresh append-only report
  `docs/artifacts/gate-d-precompile-admission-v2-compensated-stablehlo.json`, SHA
  `973c6f95...b8db`.
- Immutable-parser authority `e1b2e410...0185` accepts exact compensated candidate-only slice
  `b0afe414...18ce6`, with operations `optimization_barrier`, `add`, `convert`, `convert`,
  `optimization_barrier`, `subtract`. Accepted/candidate primary slices are identical at
  `5037b5a7...5610`; the candidate slice contains no collectives.
- Classification remains
  `NO_PRECOMPILE_CANDIDATE_ADMITTED;GATE_D_OPEN;NO_JAX_OR_TPU_SUCCESSOR`. Compensated reasons are
  now exactly `MISSING_PINNED_COHERENT_CAPSULE_PRODUCER` and
  `MISSING_CANDIDATE_COHERENT_CAPSULE`. StableHLO graph identity is proven; numerical cancellation
  and candidate-coherent eight-watchpoint replay are not.
- The parent admission run imported no JAX and initialized no backend/device; no compilation,
  numerical/model/cloud or TPU work occurred. This result does not authorize a producer run or
  successor. Exact next is a separately reviewed source-only coherent-capsule producer design.
- Fable remained at its recorded hard usage limit/no opinion. The goal-authorized existing Sol
  reviewer returned `APPROVE PERSISTENCE` for staged evidence diff `005590a5...6bf8`, with no
  P0--P2 and no successor authorization.

## Compensated coherent-capsule producer source; no process start

- `glm_tpu/greenfield/benchmarking/gate_d_compensated_capsule.py`, SHA
  `0f1930c0...313d`, is a distinct two-owner replay. It calls only the compensated RMS boundary,
  exposes `restored_rms_input_fp32` as the rooted capsule value and keeps both QKV and DSA primary
  inputs on accepted `rms.output`.
- `scripts/greenfield/produce_gate_d_compensated_auxiliary_capsule.py`, SHA
  `039997b9...d19f`, binds candidate `compensated_auxiliary_dependency`, source pin `e16d74f`,
  source/plan/StableHLO authorities `c38492a0...b4c3`, `eb2c050b...b6dc` and
  `e1b2e410...0185`, plus a distinct future root-owned installed path. Its execution/security
  machinery is normalized-AST identical to the previously reviewed tuple producer while its
  candidate and replay identities cannot select that tombstoned path.
- New hostile source tests pass 8/8. No installation, producer invocation, JAX import, replay,
  compilation, numerical/model/cloud/TPU work or admission occurred. Separate review/persistence
  precedes candidate-specific authority-builder/admission support; no process start is authorized.
- Fable remains at its recorded 100% limit/no opinion. Existing Sol returned
  `APPROVE SOURCE PERSISTENCE` for staged diff `4566728c...ef72`, with no P0--P2 and no
  installation/invocation/successor authority.

## Compensated capsule execution-authority builder and admission bindings

- `scripts/greenfield/build_gate_d_compensated_capsule_execution_authority.py`, SHA
  `dce8084d...51d2`, is stdlib-only and derives a canonical outer authority from a bound compensated
  capsule/receipt/SUCCESS tuple. It exposes only candidate-observed event-1/RMS operand hashes,
  pending independent accepted-output comparison, plus exact
  producer environment/source/dependency provenance; it does not import JAX or accept extra keys.
- Admission core `46fa2382...010c` replaces the tuple-only implementation and upstream-input
  singletons with exact two-candidate catalogues. Compensated producer/replay Git blobs are
  `15f05ad3...349d` and
  `ad38b591...00ca`; their complete replay-visible 140-file source manifest is
  `286cac1c...7295`. Installed paths, producer/replay sources, manifests and plan/source/StableHLO
  upstream authorities are candidate-specific; unknown or cross-candidate bindings fail closed.
- Contract `81ceb19b...43bc` binds the new core but deliberately retains null capsule fields.
  Focused builder/catalogue/cross-binding and hostile schema checks passed 41/41 in 120.52 seconds.
  A direct-file check then caught and corrected three copied compensated authority SHA literals;
  the post-correction changed-path slice passed 4/4 in 14.21 seconds and now asserts every
  candidate-specific authority file hash. The checks also caught and closed float/boolean schema
  acceptance in the capsule publication chain. No
  current admission report was regenerated
  and no installation, invocation, JAX/parser, compilation, model/cloud/TPU or successor work ran.
  Full offline validation and any immutable install remain separately reviewed steps.
- Fable remains at its recorded 100% limit/no opinion. Existing Sol approved source persistence
  for corrected staged diff `3133f3b2...d4f4`, with no remaining P0--P2 and explicitly no
  process-start or successor authorization.
- At clean commit `ff062cb3fbc39f77f4b5314f55a80d461b931f86`, Sol separately authorized
  only the complete offline test command. The admission-core and compensated-builder files passed
  187/187 in 393.54 seconds. It produced no tracked change or persistent report and performed no
  producer installation/invocation, JAX successor, model/cloud/TPU work or Gate-D closure.

## Compensated capsule admitted for compile-only review; TPU numerical proof remains

- The exact compensated forced-CPU capsule is now cross-bound to the accepted TPU event-1/tie
  oracle, RMS operands, cache/query/head/key/scorer watchpoints, producer receipt/SUCCESS,
  execution environment and replay-visible source catalogue. CPU replay is explicitly classified
  as logical non-interference only; protected TPU equality remains deferred and mandatory.
- Strengthening the immutable StableHLO report added exactly four independently checked fields:
  candidate primary slice, rooted auxiliary result, forbidden-operation catalogue and rooted
  device-result status. This changed derived parser authority from historical
  `e1b2e410...0185` to current `2f13a564...b8b1` without changing accepted/candidate StableHLO,
  certificate, source, plan, producer or runtime identities. Parentless transition authority
  `gate-d-compensated-stablehlo-authority-transition.json`, SHA `017e05f5...19a84`, permits the
  old hash only for capsule `5b7ad71f...c1a4` and execution authority `8b8c9cc7...a660`; copied,
  chained, cross-candidate, parser/toolchain-drifted and common-field-drifted bridges fail closed.
- Current core/config/validator/test SHAs are `30ff4ce5...ade53`, `6a2d913f...b01a5`,
  `762dcb6c...4c4a` and `3f026066...2073`. Proportional coverage spans the current 235-node
  admission file: a 210-pass prefix, the corrected fail-closed outer-authority node, the complete
  remaining tail, the transition 23/23 slice, and the new strict `2.0`/`true` schema attacks.
  The exact current-contract integration passes; no TPU/JAX/backend work occurs.
- One reviewed sealed-Python `-I -S -B` append-only run wrote
  `gate-d-precompile-admission-v2-compensated-capsule.json`, SHA
  `7cd7e569...6b37d`. It admits exactly `compensated_auxiliary_dependency` with classification
  `PRECOMPILE_LOGICAL_POLICY_PASSED;COMPILE_ONLY_REVIEW_ELIGIBLE;TPU_NUMERICAL_UNPROVEN;`
  `NO_TPU_EXECUTION_AUTHORIZED;GATE_D_OPEN`. `gate_d_closed`, JAX/TPU-work and TPU-successor flags
  are false. Fable remained usage-limited; the existing same-scope Sol reviewer found no P0--P2
  and approved source persistence plus this one report generation.
- Exact next after commit/push/locked same-region persistence is a separately reviewed optimized
  PP16 TPU-HLO acquisition and locality/numerical-policy inspection. It is not yet authorized.
  Only after that HLO passes may a separately reviewed protected layer-1 exact replay run; a full
  exact 8K decoder remains later. Gate D is open.

## Compensated PP16 publisher storage site installed; no HLO start

- Sol approved only the exact absolute `/usr/bin/sudo` immutable-provisioning command after first
  withholding the bare-`sudo` draft. At clean pushed and byte-mirrored pin `bbe6714`, the reviewed
  root-owned provisioner `2b9c8c2b...0594` installed the no-replacement target
  `/opt/glm-tpu/gate-d-storage-site-d94fd4c3e0ff`.
- Install evidence `gate-d-compensated-pp16-storage-site-install.json` binds 1,593 files /
  44,929,742 bytes, manifest `67b485a5...6e3a`, complete sealed tree `d94fd4c3...1b510`, three
  matching source builds and the exact target ownership/mode. The production publisher's own
  read-only post-install validator accepted the sealed Python runtime, 47 dependency roots,
  manifest and tree.
- The install performed no JAX/backend/cloud/TPU work and authorizes no HLO launch, numerical
  execution, performance claim or Gate-D closure. A fresh, separate process-start review remains
  mandatory.

## First compensated PP16 HLO start failed before census/JAX; tag burned

- The separately approved tag `gate_d_compensated_pp16_hlo_20260831T191040789252844Z` failed in
  `strict_census pre` because Bash nounset expanded `${label}` before a combined `local` command
  assigned it. The exit trap repeated the same failure. This is a wrapper P1, not a model/HLO
  result; no census command, JAX import, backend, compiler, executable or TPU work started.
- The partial directory is preserved with five hashed files and an empty `hlo/`; the remote prefix
  remains vacant and has no terminal. Both leases were free and a separate post-failure census was
  8/8 clean. Exact evidence is
  `gate-d-compensated-pp16-hlo-pre-census-failure.json` (`fbc6d04a...5293f2`); this tag must never
  be retried.
- The narrow correction splits `label` and dependent `member` declarations. New tests bind exact
  ordering, exercise the pattern under `/usr/bin/bash -u`, and scan every wrapper `local` line for
  a same-command dependency. A fresh tag still requires reviewed persistence and separate start
  approval.

## Bounded PP16 compensated-HLO acquisition source prepared; no process start

- A new TPU-only builder is normalized-AST identical to the admitted sealed CPU replay except for
  the exact two-device platform check. The sealed CPU module remains byte-identical at
  `0f1930c0...313d`; the new module is `a75b6eeb...e54fc`.
- Driver `526aad55...748e4` binds the exact admitted report, capsule/input authority and DB555 PP16
  stage-zero group. It uses 16 abstract input specifications, including one live residual row and
  two local owners, and records but never invokes the single compiled executable. All project
  imports come from a sealed memfd archive built from exact committed Git blobs; the running driver
  is itself committed-blob authenticated and isolated from mutable worktree import paths.
- Publisher `2d83ab19...59d874` uses component-wise no-follow and direct exclusive local writes,
  generation-zero remote creation, generation/size/CRC32C/SHA ledgers, generation-qualified replay,
  exact preterminal inventory and a final-operation terminal upload. Wrapper `013eb69d...9fc35d`
  remains default-off, serialized and 8/8-census protected; diagnostics use a separate fresh
  generation-zero subprefix and never publish a success terminal.
- Test `8d7fa965...4232c5` passes 45/45 (67/67 with adjacent compensated/reuse suites), including
  hostile authority/shape/hash mutations, AST
  equivalence, exactly one compile and no executable call, local final races, intermediate/final
  symlinks, mutable-worktree replacement after source sealing, concurrent remote insertion,
  diagnostic-ledger ordering and terminal-last ordering. No JAX/backend/cloud/TPU work occurred.
  The first Sol review's three P1s and the second review's four P1s are corrected in source. The
  latter cover the exact private live root, redirect-free wrapper evidence, raw vacancy retention,
  generation-qualified remote bytes plus terminal receipts, and distinct exact publisher/compiler
  paths with exact compiler environment/loaded
  dependency provenance. Descriptor-bound root opens, exact local inventory and terminal-phase
  diagnostic refusal close adjacent races. Correction review of `f40eb8ae...099ac9` then found and
  the current source fixes mismatched `numpy/`/`numpy.libs/` ordering, authority hash/read TOCTOU,
  unguarded post-terminal append, between-process run-directory substitution and inherited wrapper
  control state. The fix uses one bytewise path order, single-snapshot authority parsing, a shared
  preterminal writer/log guard, an inherited locked dirfd across all subprocesses and an
  exact fail-fast wrapper environment.
- A third Sol review found that self-reexec occurred after Bash could consume caller `BASH_ENV`,
  diagnostic terminal publication still allowed local writes, and remote proof imported mutable
  cloud-client code. The corrected pre-start contract requires a separately reviewed absolute
  `env -i` plus `/usr/bin/bash --noprofile --norc` command and has a hostile-`BASH_ENV` regression.
  Both terminal classes now reject later wrapper writes/appends. New stdlib-only builder
  `b7f4f869...4c5989` reproduced the same 1,593-file / 44,929,742-byte storage-client tree twice;
  tree/manifest SHAs are `d94fd4c3...1b510` / `67b485a5...6e3a`. Publication requires it as an
  exact root-owned read-only `/opt` tree under the previously sealed Python 3.12 runtime, and embeds
  the complete runtime/tree/manifest identity in the payload. Build record
  `gate-d-compensated-pp16-storage-site-build.json` (`f6fe8b59...852657`) proves no install,
  network, JAX/backend or TPU work. Same-scope correction review and durable persistence remain
  mandatory before a separately reviewed immutable install or process start; no numerical or
  Gate-D successor exists.

## Root-owned libtpu compiler supplement installed; no HLO retry

- The failed protected HLO start used mutable `vllm-env/bin/python`, resolving a user-writable UV
  Python/stdlib. Review withheld the first root-owned-interpreter correction because JAX and
  `libtpu` still came from the mutable venv site; no lowering or compilation had occurred.
- Minimal stdlib-only builder `eb0d34b0...8e37` copied exactly `libtpu` and its pinned distribution
  metadata. Two independent unprivileged builds produced the same 13-file / 719,846,262-byte tree
  `db7598c8...8eca` and manifest `d34064f4...7efa`. Hostile tests cover escaping symlinks, special
  files, hardlinks, source drift and no-clobber publication.
- At clean pushed and byte-mirrored pin `e922f96`, the separately approved absolute root-owned
  provisioner installed `/opt/glm-tpu/gate-d-libtpu-site-db7598c867f3` without replacement. A
  second invocation verified the existing complete tree, ownership and safe modes. Exact evidence
  is `gate-d-libtpu-site-install.json`.
- Build/install imported no JAX, started no backend, used no cloud/network and touched no TPU. This
  closes only the immutable `libtpu` storage prerequisite. The withheld compiler correction must
  still bind sealed Python, JAX and `libtpu` before import and after compile, reject unauthorized
  native mappings and strip mutable PATH entries before any fresh HLO start is reviewed. Gate D is
  open.

## Sealed PP16 compiler dependency correction prepared; no process start

- Current driver `3842d06d...1f88` now validates the exact root-owned Python runtime and complete
  sealed JAX/`libtpu` trees plus manifests before any JAX import and again after compilation. Its
  exact import path contains only the sealed Git-memfd project archive, the two sealed dependency
  sites and root stdlib; compiler PATH is `/usr/bin:/bin`.
- Driver and publisher `f481c610...4c15` both reject Python/native dependencies outside the sealed
  runtime/sites or `/usr`/`/lib*`; the only worktree exception is the separately Git-blob-verified
  driver file. Publisher binds the compiler runtime and both dependency-site identities before any
  remote success publication. Publisher records must use exact canonical realpaths and are reopened
  `O_NOFOLLOW` to verify regular/single-link device, inode, size, stable metadata and exact SHA
  before remote mutation. Wrapper/test SHAs are `7a7abfae...2a14` and `bf6556a7...e5c7`.
- Focused HLO-acquisition plus capsule suites pass 64/64. A root-Python `-I -S -B` no-import probe
  revalidated both complete trees/manifests and resolved `jax`, `jaxlib` and `libtpu` exclusively
  from `/opt/glm-tpu` sealed sites. Hostile checks cover mutable interpreter/site/path rebinding,
  unauthorized native mappings, tree drift and publisher refusal before remote mutation.
- No JAX import, backend initialization, cloud operation, HLO lowering/compile or TPU work occurred.
  This source batch requires review, commit/push and same-region mirror before a distinct fresh HLO
  start can be considered. Gate D remains open.

## Gate-D recovery v3 canonical-catalogue failure and strict v4 correction

- Burned tag `gate_d_repo_recovery_20260901T030059438894459Z` failed closed on worker 1
  `promisor_sha` before receive/prepare/exchange or repository/JAX/HLO/model/TPU mutation. Exact
  failure evidence is
  `gate-d-worker-repository-recovery-v3-promisor-path-preflight-failure.json` (SHA
  `7370ccfb...6ff7f`): eight unique quiescence markers, unchanged pins, no tag paths, four free
  leases and exactly 12 nonterminal diagnostic objects. It authorizes no retry.
- The old inventory and verifier used different catalogue encodings. The same worker-1 sidecars
  produce basename-newline SHA `2633dfb9...f124f` and object-relative-newline SHA
  `95035d91...a9211`; unchanged missing-object hashes and pre-attempt sidecar mtimes reject the
  drift hypothesis. Historical evidence remains append-only.
- `gate-d-worker-repository-promisor-path-inventory.json` (SHA `03be218c...fd70b`) defines canonical
  sorted `relative-path NUL size NUL` bytes and binds all worker-specific values. The verifier
  opens each entry `O_NOFOLLOW` and requires regular, single-link, zero-byte exact
  `pack/pack-<40 lowercase hex>.promisor` identity. Invalid paths/types/links/sizes and ambiguous
  encodings fail. One serialized live replay returned 7/7 target-worker `REPO_PREFLIGHT_OK` without
  mutation or tag paths; focused recovery/acquisition tests pass 82/82.
- The default-off corrected controller targets a fresh v4 immutable path. Persistence, installation
  and invocation remain three separately reviewed boundaries. PP16 HLO stays unauthorized until a
  clean, persisted, exact-pin, unique-host 8/8 recovery terminal exists.
- Source persistence is commit `39b4059cacc88ba16a7a493e83afd042460517a5`. After Sol withheld a
  first draft using two wrong workload-lock names, no install ran; the no-op probe's newly created
  root-owned zero-byte wrong lock was removed while flock-held and is lstat-vacant. The corrected
  install-only command replayed all four canonical lease identities and installed exactly one
  root-owned mode-0555 controller at `/opt/glm-tpu/gate-d-worker-recovery-v4`. Controller/tree SHAs
  are `fa637dd1...c353` / `bd80291e...98d0`; Git bytes, membership, ownership, modes, links, xattrs,
  staging absence, lock release and process absence pass. Install artifact
  `gate-d-worker-repository-recovery-v4-install.json` has SHA `eaf702da...159db`. No controller
  invocation, recovery/repository mutation, cloud/network, JAX/HLO, model or TPU work occurred.
- Sol then approved one invocation only under fresh tag
  `gate_d_repo_recovery_20260901T033629027256215Z`. Seven workers passed every preflight, prepare,
  swap and final record; worker 0 remained the protected linked worktree. Exact final verifier,
  pre/pre-swap/post census and quiescence evidence each contains eight unique hosts at pin
  `dc9ec468ad1529b09c4dd06005849ee70e0fa7c7`. Seven retained old paths replay their sealed old pins;
  new/bundle paths are absent and all per-worker/global leases are free. Independent replay matched
  all 19 remote objects / 9,152,508 bytes to local evidence and proves terminal generation
  `1788234096347915` last with terminal SHA `491fe58f...7067`.
- The external local-vacancy probe named historical root `gate-d-runs`, not controller root
  `/home/gianl/gate-d-repo-recovery`. The controller's O_EXCL tag-directory creation nevertheless
  proved actual vacancy before recovery work; no collision occurred. Complete artifact
  `gate-d-worker-repository-recovery-v4-complete.json` has SHA `5f9e7ffc...150d2`. This closes only
  repository recovery. No JAX/HLO/model/TPU work ran; any PP16 HLO start remains separately reviewed.

## Gate-D recovery v2 promisor-preflight failure and strict v3 correction

- Burned tag `gate_d_repo_recovery_20260901T022519340652102Z` failed closed at worker 1
  `config_not_allowlisted` before clone/prepare/exchange or repository/JAX/HLO/model/TPU mutation.
  `gate-d-worker-repository-recovery-v2-promisor-preflight-failure.json` binds unchanged 8-host
  pins/paths, 8/8 quiescence, the nonterminal 12-object prefix and local evidence hashes. It is
  failure evidence, not permission to rerun v2.
- `gate-d-worker-repository-promisor-inventory.json` (SHA `67bd8b76...b7f4f`) supersedes only the
  inaccurate full-local-closure interpretation of the earlier census. Workers 1/3/4/5/7 are exact
  format-1 `blob:none` promisor prestates; workers 2/6 are complete format-0 prestates; worker 0 is
  the current linked worktree. Successful `fsck` on a partial clone does not prove every promised
  object is locally present.
- The v3 source keeps two independent contracts. Historical old repositories may use an exact
  worker-specific standalone/promisor contract bound to pin, format, missing-object count/hash and
  promisor-sidecar count/hash. Every prepared/current target remains strict format-0 standalone,
  zero-missing and zero-promisor. A worker already at the target pin is checked against the target
  contract, so fresh-tag partial resume cannot be rejected by its obsolete old-state contract.
- Current controller/worker/wrapper hashes are `2a992618...b13e`, `a0bf5ca9...c572` and
  `efb94572...9701`; complete recovery/acquisition tests pass 82/82, including a real local
  missing-object `blob:none` clone. Sol approved exact persistence SHA `3c4fd23f...ac93`; commit
  `9a7b451` is pushed and same-region mirrored. A separate install-only review approved the exact
  no-replace command. `gate-d-worker-repository-recovery-v3-install.json` (SHA
  `e4bef8d5...e206`) proves the one-file root-owned 0555 install, zero xattrs, tree
  `fb01c9f1...66df` and exact Git bytes. No invocation ran. Exact next is one separately reviewed
  fresh recovery tag. PP16 HLO remains unauthorized before persisted exact-pin unique-host 8/8
  closure.

## Gate-D repository fleet heterogeneity and targeted recovery correction

- `docs/artifacts/gate-d-worker-repository-heterogeneous-inventory.json` records the corrected
  read-only 8-host census. Worker 0 is the clean current linked worktree; workers 1--7 are clean,
  full-object-closure standalone repositories split across two stale pins. This directly rejects
  both "no recovery needed" and the unchanged all-eight exchange.
- Burned diagnostic tag `gate_d_repo_recovery_20260901T014300000000001Z` performed no clone,
  rename, upload, JAX/HLO, model or TPU work. Its first failure was `git_boundary`: the old exact
  verifier incorrectly required `.git` to be a directory and therefore rejected worker 0's valid
  linked-worktree pointer.
- The correction binds worker 0 to the exact linked common/admin/pointer/backpointer/commondir/
  config/object/index structure and workers 1--7 to the strict standalone path. It hash-binds the
  eight-record inventory and per-worker observed stale pins, rejecting layout drift, duplicate IDs
  and any third pin. Recovery mutates only the still-stale subset so partial success can resume,
  refuses a no-op, pre-verifies untouched worker 0, and requires unique-host exact-pin 8/8
  post-verification before terminal publication. Complete inventory, recovery and HLO-acquisition
  validation passes 84/84.
- Sol approved persistence only; staged SHA `b965001a...925a` is commit `652f16e`, pushed and
  same-region mirrored. A separate install-only review approved the no-replace provisioner command.
  `docs/artifacts/gate-d-worker-repository-recovery-v2-install.json` proves the resulting exact
  root-owned 0555 controller (`f1f16f60...65c8`) and one-entry tree (`862f8943...b80f`) match Git
  with no xattrs. No recovery invocation or worker mutation ran. Recovery still requires a separate
  review of one fresh tag and exact command; historical controllers remain inert. Gate D and HLO
  acquisition remain open.

## Fresh PP16 HLO start refused during concurrent Git transport; sync removed from run

- The separately approved fresh tag `gate_d_compensated_pp16_hlo_20260831T222752715835766Z` passed
  vacancy, mirror and authenticated 8/8 pre-census, then failed before JAX when five of eight
  concurrent GitHub SSH fetches reported MAC/pack corruption. Concurrency causality is unproven.
  Workers 0/2/6 reached clean pin `508aaa3`;
  workers 1/3/4/5/7 remained clean at `4a44f58`. The HLO directory is empty, no runner log exists,
  the failure-exit census is 8/8 clean and both leases are free.
- Append-only diagnostics contain exactly ten remote objects and no `HLO_ACQUIRED`; exact evidence
  is `gate-d-compensated-pp16-hlo-concurrent-git-sync-failure.json`. The tag is burned. This is
  repository transport evidence only, not a compiler, HLO, numerical or Gate-D result.
- Corrected wrapper `f093aea4...a1b3d3` performs no fetch, clone, checkout or mutation in its fleet
  verification block. Under an isolated no-lock/no-lazy-fetch Git environment, it sequentially
  requires a direct canonical worktree/repository, strict passive config, clean exact HEAD without
  special index/sparse state, non-shallow complete closure, and no alternates, promisor state,
  grafts, replacement refs or object symlinks. It fails on the first nonzero worker, then requires
  eight unique `SYNC_OK` markers. Test `32c24967...84a335` proves the exact verifier does not invoke
  hostile fsmonitor/filter/diff commands or change repository bytes/metadata; redirected worktrees,
  hidden dirty files and missing objects reject. Recovery to the final
  corrected pin must occur separately under both leases through a reviewed SHA-bound immutable
  transfer and preserved-old/atomic publication; only then may a new HLO tag be reviewed.

## Gate-D worker repository recovery corrected after two review blocks; no fleet mutation

- Fable session `9152fb51-b797-4887-afc2-8250989a3c05` returned its hard usage limit without an
  opinion. Sol withheld the first shell draft on six P1/P2 classes: remote work could outlive local
  leases; canonical/preserved identities and rollback were weak; SCP could overwrite a raced path;
  local locks/evidence were pathname-raceable; the exit trap wrote after terminal start; and two
  happy/static tests overstated guarantees. That draft was never committed, mirrored or run.
- Sol also withheld replacement snapshot `35741224...`: lock names remained replaceable,
  quiescence trusted mutable/fail-open tools, evidence reopened names, terminal preceded exact
  archive completeness, and tests/docs overstated this. It too was never installed or run.
- Corrected controller `fd8950de...b2225` must be root-owned mode 0555 beneath an immutable parent
  and byte-match its committed blob. Controller and HLO wrapper share immutable workload/rsync
  locks; each worker uses a distinct root-owned `gate_d_repo_recovery.lock`. Every lock is
  nofollow/identity-corroborated under `/opt/glm-tpu/locks`; legacy leases remain held. Worker
  `47d65f03...9beb0` retains the O_EXCL stream, unique carrier and hard timeout.
- The worker records canonical/prepared device+inode and exact verifier pins, moves the verified
  prepared clone into its vacant final preservation slot with `RENAME_NOREPLACE`, exchanges that
  slot atomically with the canonical repository, verifies both resulting repositories and rolls
  back on any post-exchange failure. A sanitized privileged `/proc` audit fails closed on
  process/lock read or parse uncertainty; TERM/INT cannot release local leases before 8/8 proof.
  Retained evidence FDs reject name substitution. Exact generation/size/SHA records for the
  16-object preterminal set are checked and embedded before the 17th terminal object; later work is
  read-only replay/exact-set verification.
- Test `16cd0dcd...5b91` passes 14/14, including the absolute `/usr/bin/sudo` hostile-PATH
  regression; adjacent wrapper/inventory tests make 80/80. Ruff/format,
  compilation, Bash syntax and diff checks pass. Sol withheld `9c260108...` for the PATH-resolved
  privileged token/test gap and ambiguous lock wording, then approved persistence-only corrected
  `257ac5ce...`. Commit `6066bca31e6d4bebfe7038eb52dda9a3fc851915` is pushed and its exact
  source hashes replay from the locked `US-CENTRAL2` mirror. Source is reviewed/persisted but
  uninstalled. Root-lock/controller provisioning and cron adoption are not performed; no recovery,
  JAX/backend, HLO, model or TPU work is authorized.

Sol withheld first prerequisite snapshot `8697ef1e...` before persistence or execution because its
running source, legacy handoff, privileged install allowlist and evidence parent/name were not
strong enough. Sol later approved corrected two-file v1 bundle `457e7933...3aad` for persistence
and, separately, installation only. Commit `fdca1b5...80c9` is pushed/exact-mirrored and v1 is
installed root-owned at its no-replace path. Its approved invocation under burned tag
`gate_d_repo_prerequisite_20260901T005642867190721Z` failed closed before mutation because v1
applied a 64-hex SHA-256 regex to the valid 40-hex Git commit returned by `ls-remote`. Exact
postflight leaves legacy cron, repositories, lock/controller/evidence absence and zero TPU/JAX/HLO
work unchanged. The terminal record is
`docs/artifacts/gate-d-repository-prerequisite-v1-git-oid-parser-failure.json`.

The reviewed v2 bundle `70bd0197...9d32` targets a fresh immutable v2 root. Controller
`532c2dec...3298`, unchanged helper `13dc4f85...f68f` and tests `7202aab6...1841` separate Git OID
and SHA-256 types, accept only exact `<40hex><TAB><exact-ref><LF>` bytes, reject missing LF, CRLF,
CR, VT, FF and malformed records, and add bounded malformed-response diagnostics. Focused coverage
passes 23/23. One adjacent run stopped 97/98 on a pre-existing
hostile-object test's earlier `tracked_state` result. Test `94e30c10...d384` now requires the exact
allowed fail-closed reason set; five consecutive focused replays and the complete adjacent boundary
pass 103/103. Commit `2e52847...98da` is pushed/exact-mirrored and v2 is installed exact. Its
approved invocation under burned tag `gate_d_repo_prerequisite_20260901T011459146872151Z`
materialized the reviewed local/8-host locks, exact recovery controller and new cron, then failed
closed before receipt because live repository `docs/artifacts` mode 0775 violates the writer's
non-group-writable parent contract. Exact live reconstruction is recorded in
`docs/artifacts/gate-d-repository-prerequisite-v2-evidence-root-failure.json`; no repository,
JAX/HLO/model or TPU work ran.

Sol approved v3 persistence and install separately. Commit `844ff36...3c23` is pushed and its six
changed files exact-hash replay from `US-CENTRAL2`; immutable v3 tree `67187acf...ec2` is installed
root-owned with exact 0755 two-file membership and zero xattrs. The separately approved invocation
under fresh tag `gate_d_repo_prerequisite_20260901T013157641110677Z` returned
`PREREQUISITES_PROVISIONED` in 24.29534 s. Exact receipt
`docs/artifacts/gate-d-repository-prerequisite-v3-provisioned.json` is 6,027 bytes and SHA
`b95eaec1...30da`; it binds the 0700 private parent, 0600 O_EXCL receipt, installed FDs, exact
controller/source hashes, four held leases, unchanged reviewed cron (`existing`), and eight unique
worker recovery locks. Independent live stat replay matches all eight lock identities; the local
repository remains clean at origin. No repository mutation/recovery, JAX/HLO, model, TPU or bucket
write ran. Prerequisites are now materialized and receipted; repository recovery requires its own
review, and HLO remains unauthorized.


## 2026-09-01 compensated PP16 rejection recovered; exact conversion-placement explanation

The one completed protected tag
`gate_d_compensated_pp16_numerical_20260901T094622505067868Z` invoked its compiled TPU
executable exactly once and returned `NUMERICAL_REJECTED`. Its historical publisher then failed
only because one exact Git-authenticated HLO helper lay outside the old dependency-root policy.
No TPU retry occurred. CPU-only recovery commit `5c4297c6cbfca0f6363276249aa116db25a7e724`
passed 158 focused tests and final Sol review. A fresh validate-only replay made zero remote writes.
Fresh seal tag `gate_d_compensated_pp16_recovery_20260901T102415964684137Z` then published 12
exact objects terminal-last in `US-CENTRAL2`; terminal generation is `1788258353080631`,
CRC32C `ZmPYCw==`, SHA-256 `a3a4a0ec...aad959`. The source diagnostic remains untouched.

The recovered arrays close the prior operand ambiguity. Both BF16 operand rows independently widen
and add to the accepted and both TPU-owner FP32 RMS inputs byte-for-byte. Offline certificate
`gate-d-scalar-frontier-conversion-placement.json` (SHA-256 `8ee468eb...c170ce`) compares the
accepted strict model (normalized-BF16 round, widened per-element weight multiply, final BF16
round) with the retained-FP32-through-weight-multiply model:

- strict double rounding reproduces all 6,144 accepted output bits;
- one-round `bf16((x*scale)*weight)` reproduces all 6,144 protected TPU output bits;
- their 1,622 differing indices exactly equal accepted-versus-TPU mismatches;
- accepted strict-scale preimage is the 44-f32-value interval
  `0x43329838..0x43329863`, containing reference scale `0x4332984c`;
- TPU output bit `0x3c18` at index 0 is skipped by strict double rounding for every positive
  finite FP32 scalar, so no uniform RMS scalar can explain the protected row.

This proves retained FP32 through the weight multiply is an exact explanation and rules out the
FP32 RMS input or any uniform strict-double-round scalar. It does not yet prove the physical cause;
conversion placement/excess precision remains the leading hypothesis pending causal A/B evidence.
The compensated auxiliary mechanism itself is tombstoned, but alternative physical causes are not.
Gate D remains open. Exact next is default-off source design and CPU-only validation/persistence of
a source-exact causal challenger that forces the existing post-normalization BF16 round before
weight multiplication (`reduce_precision(e8m7)` first; excess-precision-disabled comparator).
Compile/HLO acquisition and any bounded PP16 numerical start remain unauthorized; each requires
its own fresh review, tag and exact execution authority. Full 8K remains forbidden.

## 2026-09-01 forced normalized-BF16 boundary source design; CPU-only

The scalar-frontier certificate `8ee468eb...c170ce` motivated one default-off causal challenger,
`fused_add_rms_norm_with_forced_bf16_boundary`. It leaves the FP32 residual sum, variance and
normalization unchanged, applies explicit `reduce_precision(e8m7)` to the normalized FP32 value,
widens the exact BF16 weight to FP32 for multiplication, and performs the final BF16 conversion.
The normal decoder has no caller for this function.

CPU-only certificate `gate-d-forced-normalized-bf16-source-design.json` (SHA-256
`54bb2750...d3100`) binds the sealed capsule, protected output and scalar-frontier evidence. On the
real 6,144-element layer-1 row, the candidate and carried residual match accepted bits exactly; its
1,622 differences from the protected TPU row have the exact prior mismatch-index hash. Three
additional BF16 row families match the accepted reference bitwise. Fourteen exact signed-zero,
subnormal, normal, tie, max-finite and overflow cases prove e8m7 equals explicit BF16 cast/widen
before weighting. A `jax.make_jaxpr` trace—not HLO lowering—proves one e8m7 edge solely feeds a
multiply whose other operand is the sole BF16-weight-to-FP32 lineage, then one final BF16 output.

This proves source and CPU exactness only. Physical causality, optimized-HLO preservation,
performance and Gate-D closure remain unproved. No TPU compile, HLO acquisition, cloud write,
numerical run or full 8K occurred. Persistence is the only authority in this batch. A future
compile/HLO acquisition and any bounded PP16 numerical start each require a separate fresh review,
tag and exact authority; full 8K remains forbidden.

## 2026-09-01 forced-round PP16 HLO integration source; no HLO/TPU

The default-off builder `build_gate_d_forced_round_pp16_hlo_replay` composes the forced
normalized-BF16 edge into the sealed one-row QKV/DSA form. It now refuses any runtime pair except
the topology-authorized PP16 stage zero: ordered device ids `[0,1]`, TPU v4, process 0,
coordinates `[(0,0,0),(1,0,0)]`, core 0, and axis `feature`. The exact local group is passed to
DSA; all 16 input specs, nine rooted output specs, cache layout, aliases and the single
`jit(shard_map(...))` wrapper are structurally bound.

Fable returned only its hard usage-limit refusal. Sol initially withheld persistence because
dependency bytes, arbitrary source effects and exact PP16 sharding/topology were not closed. The
corrected analyzer binds predecessor `54bb2750...d3100`, the forced-function module/function
hashes, topology artifact `49cf6bb1...25eb`, base commit/tree, the complete eight-path delta and
seven direct runtime dependency hashes. The auditor has an exact normalized whole-module AST lock
plus targeted lineage/sharding checks, so added imports/calls, indirect execution, group/spec drift
and non-stage-zero devices fail hostile tests. Certificate generation bypasses the broad validation
initializer and proves zero loaded `jax`, `jaxlib` or `jax_plugins` modules.

Certificate `gate-d-forced-round-pp16-hlo-source.json` has SHA-256
`518be87b650729d365dd09aba20b5f5a03d4bcddccc81978357cdbb02cbe02c6`. Focused hostile coverage
passes 25/25 and the bounded adjacent CPU/source suite passes 156/156; Ruff, Python compilation,
sanitized exact regeneration, JSON and diff checks pass. CPU tests may import JAX, but no valid
builder call, lowering, HLO acquisition, TPU compile/execution, cloud write, numerical or
performance run occurred. This remains persistence-only source evidence: physical causality and
Gate D are open; full 8K is forbidden. Sol's corrected-scope re-review found no P0--P2 and returned
`APPROVE PERSISTENCE`. After commit/push/mirror, exact next is a separately reviewed bounded HLO
acquisition proving the e8m7 edge survives optimization before any numerical A/B.

## 2026-09-01 forced-round PP16 HLO acquisition source; no HLO/TPU

A new isolated driver, `acquire_gate_d_forced_round_pp16_hlo.py`, copies the already hardened
compile-only boundary without changing the historically pinned compensated driver. It binds the
exact source certificate `518be87b...e02c6`, compensated capsule-input admission
`7cd7e569...6b37d`, topology `49cf6bb1...25eb`, exact stage zero and all eight committed
builder/runtime dependency blobs before opening the inherited run directory or importing JAX.
Its abstract contract has 16 inputs, one live row and nine rooted two-owner outputs. The only
compiled object is never invoked; compilation caching stays disabled.

The source auditor compares every inherited helper AST with base driver
`a4599e0d...b5dd15`, permits only two new validation helpers, and requires exactly one lower and
one compile call with zero executable calls. The older source certificate now regenerates from its
exact commit `2a050c1`, preserving it after later repository changes. The
persistence-only certificate `gate-d-forced-round-pp16-hlo-acquisition-source.json` has SHA-256
`efe04d98267fc5265952b28ee386a4894d028b39f2d62d0a1554f24591e73196`; focused and adjacent
source/security coverage passes 154/154. This batch authorizes
persistence only: no wrapper/start, JAX import, lowering, HLO, TPU compile/execution, cloud write,
numerical/performance work or 8K occurred. Gate D remains open. Exact next after validation and
review is commit/push/mirror, then a separate execution-authority review for one fresh compile-only
tag.

Fable remained at its recorded 100% usage limit. The goal-authorized Sol fallback reviewed exact
staged diff `7d22104a...6457`, found no P0--P2 and returned `APPROVE PERSISTENCE`; it authorized no
HLO acquisition or TPU work.

## 2026-09-01 forced-round PP16 HLO orchestration source; no HLO/TPU

An isolated default-off wrapper and generation-bound publisher target only the forced-round PP16
compile-only driver. Sol withheld the first source snapshot on two P1s: `cd /` exported `OLDPWD`
before a mutable-path self-reexec, making the second allowlist fail deterministically, and reopening
that user-writable wrapper left a same-UID substitution window. The corrected source removes the
self-reexec. A dedicated root-owned launcher authenticates exact committed bytes, snapshots the
wrapper into a fully sealed memfd retained as FD 10, inherits already-held root-owned lock FDs
11/12, and binds driver/publisher/mirror-verifier execution to a root-owned immutable capsule. The
wrapper verifies those descriptors and launcher before `cd /`, then closes FD 10. An executed
hostile regression replaces the named wrapper after snapshot and proves Bash executes only the
sealed original; a separate launcher-to-Bash regression proves `OLDPWD` is absent before the sole
`cd`. Authenticated eight-host census and terminal-last generation binding remain unchanged. The
compiled executable remains uninvoked and compilation caching remains off.

The source-only certificate
`gate-d-forced-round-pp16-hlo-orchestration-source.json` authorizes persistence only. No wrapper
start, JAX import, lowering, HLO, TPU compile/execution, bucket write, numerical/performance work or
8K occurred; Gate D remains open. Exact next after hostile validation, adversarial review,
commit/push and locked same-region mirror replay is a separately reviewed literal fresh-tag command
for exactly one compile-only acquisition, followed by offline HLO adjudication before any numerical
A/B.

Corrected certificate SHA-256 is
`f4fcba8497599dd19f94ec8ded2566ad3d27a412a04adad584fc22417e18847c`; focused launcher and
orchestration coverage passes 54/54 and the adjacent preserved-source/security suite passes
208/208. Corrected adversarial persistence review is still required; no launcher/capsule install or
execution is authorized by this source batch.

## 2026-09-01 forced-round PP16 HLO immutable install source

- Source: `scripts/greenfield/install_gate_d_forced_round_pp16_hlo_runtime.py`.
- Auditor: `scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_install_source.py`.
- Certificate: `docs/artifacts/gate-d-forced-round-pp16-hlo-install-source.json`.
- Tests: `tests/greenfield/validation/test_install_gate_d_forced_round_pp16_hlo_runtime.py`.
- Authority: persistence only. No privileged install, launcher invocation, JAX/HLO/TPU/cloud,
  numerical/performance or Gate-D claim.
- Boundary: exact root-owned source membership and payload hashes; capsule first, launcher last;
  atomic `RENAME_NOREPLACE`; byte-exact idempotency; hostile existing targets never replaced.
- Predecessors: commit `a012b93fdbd7c6fe1f84db2260708ba55b38e8f6`, orchestration certificate
  `f4fcba8497599dd19f94ec8ded2566ad3d27a412a04adad584fc22417e18847c`.
- Historical reproducibility: the orchestration auditor reads all audited inputs and its exact delta
  from Git objects at `a012b93`; its existing certificate remains byte-identical.
- Review corrections: cleanup is created-inode/publish-state bound and preserves hostile
  PID-collision staging objects; Git disables replacement/lazy fetches and rejects `refs/replace`;
  the prior no-op path loop is removed. Corrected staged SHA `826627f4...0f294f1` received Sol
  `APPROVE PERSISTENCE` and is commit `102a75f`.
- Exact next after validation/review/persistence: separately review the literal install-only
  commands; only then prepare a separately reviewed fresh compile-only tag.

## 2026-09-01 forced-round PP16 HLO runtime install

- Artifact: `docs/artifacts/gate-d-forced-round-pp16-hlo-runtime-install.json`.
- Source authority: commit `102a75f`, reviewed diff `826627f4...0f294f1`, fixed provisioner
  `2b9c8c2b...0594`, source tree `ab902d62...c919c`.
- Installed: root-owned/read-only exact source capsule, three-file runtime capsule, launcher last;
  exact membership/hashes, nlink 1 files and empty xattrs.
- State: `launcher_invoked=false`; both locks free; no launcher process.
- Claim: installation only. No JAX/HLO/TPU/cloud, numerical/performance or Gate-D evidence.
- Exact next: review/persist this evidence, then separately review one fresh-tag compile-only start.

## 2026-09-01 forced-round PP16 HLO pre-census contention failure

- V2 source/runtime persistence chain ends at commit `878f9689d1d939257d31097e1cc86d224ddcab4d`.
- Burned tag: `gate_d_forced_round_pp16_hlo_20260901T134523411363994Z`.
- Artifact: `docs/artifacts/gate-d-forced-round-pp16-hlo-pre-census-busy-failure.json`, SHA-256
  `a6b2aedfe43cf9c97d0288686192736a12d0917d0e98766ff23cc09bb24d22aa`.
- Failure boundary: worker 0 `CENSUS_BUSY`, workers 1--7 `CENSUS_OK`; the protected workflow's
  driver/JAX/lowering/compile/HLO/executable/numerical/8K never started.
- Directly observed holder: PID 304691, pre-launch environment/device probe, `libtpu` lock and four
  accelerators open. Its origin is unknown. It later exited without intervention.
- Recovery evidence: PID absent, exact 8/8 clean census, four leases free, pod READY/HEALTHY, bucket
  `US-CENTRAL2`; no cleanup mutation was required.
- Archive: ten generation-qualified objects verified byte-for-byte; diagnostic ledger generation
  `1788270441162145` is terminal-last.
- Exact next: after persistence review, use unchanged source and a separately reviewed fresh tag for
  one compile-only HLO acquisition. Gate D remains open; numerical execution and 8K are forbidden.

## 2026-09-01 forced-round PP16 root-launcher lock failure

- Burned tag: `gate_d_forced_round_pp16_hlo_20260901T140358476702832Z`; source pin `0d1d2f8`.
- Artifact: `docs/artifacts/gate-d-forced-round-pp16-hlo-launcher-lock-failure.json`, SHA-256
  `60ae6ba504f8c6ad06f1ed8b30fa04550f8cd123586c38a7b4f0638f093f2c66`.
- Boundary: root launcher `_open_locked_fds` returned `EAGAIN` before wrapper/run directory/JAX/HLO/
  TPU/cloud work. The launcher did not expose which root lock failed, so its identity is unknown.
- A five-minute sync cron logged work from 14:05:10--14:05:22, overlapping the 14:05:12 attempt;
  causality is not claimed.
- Post-audit: both local tag paths absent, live/versioned/soft-deleted remote histories vacant, four
  locks free and zero launcher/acquirer processes.
- Exact next after persistence: one new reviewed fresh tag outside the cron window. Numerical and 8K
  remain forbidden; Gate D is open.

## 2026-09-01 forced-round PP16 causal HLO adjudication

- Successful compile-only tag:
  `gate_d_forced_round_pp16_hlo_20260901T141137500138602Z`; source pin `b8bdeb1`.
- Immutable graph identities: optimized TPU HLO
  `a0b87e2b43ba81bfe1549fbcc13434d2fbdca91b11c8b2f9a317ff9a9dbd9d45` (263,868 bytes),
  StableHLO `45eae705b60783bf8b65a1d3d209c79e8d43685cec0aa912b3141caf36fb19e1`
  (76,529 bytes).
- Archive closure: 15 byte/hash-verified payloads, generation-qualified remote ledger and terminal
  marker; terminal generation `1788272143585315` is last. Independent read-only all-version
  catalogue plus generation-qualified downloads prove exactly 17 unique live objects, zero
  soft-deleted objects and exact size/SHA replay. The soft-deleted query is explicitly exhaustive
  and binds its exact no-object exit/output semantics. Replay artifact
  `gate-d-forced-round-pp16-hlo-remote-replay.json` has SHA-256
  `85bcd02c9b112ef3e21d065ebfae04f0d06e4a922f1d15c0f971fb669992e1d8`. Executable invocation
  count is zero; pre/post census is 8/8 clean.
- Validator: `glm_tpu/greenfield/validation/gate_d_forced_round_hlo.py`; CLI:
  `scripts/greenfield/adjudicate_gate_d_forced_round_pp16_hlo.py`; hostile/exact tests:
  `tests/greenfield/validation/test_gate_d_forced_round_hlo_adjudication.py` (32/32).
- Accepted causal chain: two BF16 RMS inputs -> FP32 add/RMS inverse -> sole e8m7 edge -> BF16
  weight multiply -> BF16 boundary; that boundary is QKV carrier index 4, roots output 0 and is the
  sole source of both DSA projection consumers. No competing unrounded primary edge exists.
- Locality: exactly three all-gathers, channels 2/3/4, each physical group `[0,1]`; no forbidden
  host/send/recv/global operation or unexpected custom-call target.
- Report: `docs/artifacts/gate-d-forced-round-pp16-hlo-causal-adjudication.json`, SHA-256
  `4db0ea2bfd8f34ec631213b6724d220038c08b23da788164c62e75efacbbab74`.
- Verification: focused hostile/exact coverage is 32/32; the CPU-pinned adjacent admitted suite is
  198/198. One immutable historical installer-certificate regeneration assertion is deliberately
  deselected because it correctly rejects these four successor paths; its allowlist remains sealed.
- Safety incident: an earlier adjacent command omitted `JAX_PLATFORMS=cpu`, reached real JAX
  arithmetic, opened worker-0 TPU devices and stalled. Whether any TPU numerical operation completed
  is unproven; no accepted output exists and it is excluded from evidence. The owned pytest was
  terminated; no protected launcher, model or cloud action ran. Fresh census is 8/8 `CENSUS_OK`
  and local holders are empty.
- Classification: `HLO_CAUSAL_STRUCTURE_ACCEPTED;PP16_LOCALITY_ACCEPTED;TPU_NUMERICAL_UNPROVEN;
  GATE_D_OPEN`. Persistence only; numerical execution, performance, 8K and closure remain false.
- Exact next after review/persistence: separately authorize the smallest one-row protected
  numerical A/B. Do not start the complete decoder until that result is accepted.

## 2026-09-01 forced-round PP16 numerical source

- Driver: `scripts/greenfield/run_gate_d_forced_round_pp16_numerical.py`, SHA-256
  `5e1c1f704c12f1a65947a1265e80d64e6404a0bdc473d883cc095e082e7f2187`.
- Certificate: `docs/artifacts/gate-d-forced-round-pp16-numerical-source.json`, 3,807 bytes,
  SHA-256 `17fe23a0641ddb93ed3b46eb63d13fa0d875054262fbe148ee41f4d83c616c2d`.
- Scope: one logical row, PP16 stage zero `[0,1]`, exact sealed input/DSA authorities, one compiled
  invocation and one bounded output-tree host transfer. Both acceptance and rejection keep Gate D
  open and make no decoder/performance claim.
- HLO boundary: exact acquired StableHLO; optimized HLO is the accepted `a0b87e2b...d9d45`
  preimage after only three exact path/callsite metadata replacements. Expected numerical HLO is
  263,876 bytes, SHA-256 `6ec8989d...4f1dfa`; graph-source blobs remain bound to the acquisition.
- The corrected runtime identity validator distinguishes the acquired optimized preimage from the
  source-location-derived optimized executable. It rejects either substitution/drift before the
  sole invocation; the prior direct dictionary comparison would have rejected every valid runtime.
- Numerical acceptance additionally requires exact equality at all eleven non-scorer causal
  watchpoint records. The CPU scorer records remain diagnostic because the independent accepted
  TPU event hashes are authoritative. Git reads are isolated from user/system configuration,
  replacement objects and network transports.
- Verification: 23/23 focused and 221/221 adjacent CPU-only tests with one sealed-history
  deselection; Ruff 0.16.5 and Python compilation pass. No install, JAX
  backend initialization, compile, TPU/cloud work or numerical evidence occurred.
- Authority: persistence only. Adversarial review precedes commit/push/mirror; immutable
  install/orchestration and any fresh-tag TPU execution require separate review.

## 2026-09-01 forced-round PP16 numerical orchestration/install source

- Certificate: `docs/artifacts/gate-d-forced-round-pp16-numerical-orchestration-install-source.json`,
  2,873 bytes, SHA-256 `fe53ef872a34b3a08dec26d50678beb0b228df1c27eef7becf4a1d7fcef21ce5`.
- Sources: isolated publisher `bca06fbb...5460d8`, wrapper `d3a687c3...05521c`, root-owned
  descriptor launcher `44ef4118...bf8709`, and atomic no-replacement installer
  `46de88be...cf89d6`; the corrected driver remains `5e1c1f70...7f2187`.
- Boundary: default-off, both global locks, pushed-pin and same-region mirror replay,
  authenticated 8/8 zero-work censuses, one row/two chips, one invocation/transfer, exact HLO and
  causal/event output validation, append-only evidence and terminal-last archive. Live,
  all-version and exhaustive soft-deleted vacancy is proven before local run creation and repeated
  immediately before either upload path. Installer
  publishes the 0555 capsule before the 0555 launcher and never invokes it.
- Verification: 52/52 focused and 273/273 adjacent CPU-only hostile tests with one sealed-history
  deselection; Ruff 0.16.5 and Python/Bash compilation pass. No install, launcher, backend, compile, TPU
  or cloud action occurred. Persistence only; literal install and execution need separate reviews.

## 2026-09-01 forced-round PP16 numerical immutable runtime install

- Artifact: `docs/artifacts/gate-d-forced-round-pp16-numerical-runtime-install.json`.
- Persisted authority: reviewed diff `487d3646...94fd6d`, commit `d26e83f...5e45b`, origin exact;
  locked `US-CENTRAL2` mirror connectivity/fsck and checkout archive `f763b787...1b9b6b` pass.
- Install: separately reviewed absolute-`/usr/bin/sudo` commands provisioned exact source tree
  `038355c9...a575f`, then published root-owned/read-only runtime capsule
  `954ff083...9a869d` before launcher `44ef4118...bf8709`; all files have nlink 1 and no xattrs.
- State: `launcher_invoked=false`; both locks free; no launcher process. No JAX/backend/HLO/TPU,
  cloud, numerical, performance or Gate-D claim. One fresh-tag protected execution requires its own
  literal-command review.

## 2026-09-01 forced-round PP16 numerical v1 failure and v2 source

- Failed protected tag: `gate_d_forced_round_pp16_numerical_20260901T161156923004972Z`; stopped at
  runtime HLO identity before executable invocation. StableHLO remained `45eae705...19e1`; actual
  optimized HLO was `31f2a46d...b5aa`.
- Root cause: source metadata derivation selected enclosing `ast.If` line 1272 instead of exact
  nested `main()` `ast.Call` line 1273. The optimized-HLO graph body did not differ.
- Failure artifact: `docs/artifacts/gate-d-forced-round-pp16-numerical-v1-hlo-metadata-failure.json`,
  SHA-256 `6bc2f475328f77dbe89e3ee8d5636a0a3182638cd2567e1244fa118561ef7c78`.
  Generation-bound diagnostic archive and 8/8 pre/failure cleanup are bound; no numerical or
  performance claim exists and the tag is burned.
- V2 driver derives the exact call node and uses immutable
  `/usr/local/libexec/glm-tpu/gate-d-forced-round-pp16-numerical-v2`; expected optimized HLO is
  `ccd6ffb4909b1bc4dca5a36f106cde4a84304b230161afb484afb5667bb7206c` (263,876 bytes).
- V2 source certificate: `docs/artifacts/gate-d-forced-round-pp16-numerical-source-v2.json`, SHA-256
  `4caef2af207f21edf10597881d29d48303b7de3fd61ecc961b1f4e396db84932`.
- V2 orchestration/install certificate:
  `docs/artifacts/gate-d-forced-round-pp16-numerical-orchestration-install-source-v2.json`, SHA-256
  `a8790efd54bd82370e2ac27dc53cb8d1995ab9630b6d3a92cbb2206796e0e609`.
- Verification: 76/76 focused CPU-only tests, Ruff 0.16.5, Python compilation and Bash syntax.
  Historical v1 adjacent counts are referenced separately; two invalid broad attempts are excluded.
  The accidental default-backend attempt was terminated and followed by an authenticated four-lock
  eight-host census with eight unique `CENSUS_OK` results.
- Authority: source persistence only. No v2 install/launcher/cloud/protected numerical/8K authority.
  Sol returned `APPROVE` with no P0/P1/P2 findings and explicitly granted no install/execution
  authority. Exact next is commit/push/mirror and a separate install-only review.

## 2026-09-01 forced-round PP16 numerical v2 immutable runtime install

- Artifact: `docs/artifacts/gate-d-forced-round-pp16-numerical-runtime-install-v2.json`.
- Persisted source pin: local/origin commit `ed40021a54d8ff420997a3089eff5f52f1b0d7a7`;
  locked `US-CENTRAL2` mirror checkout SHA
  `a9436b24746146a93ef4b77d142694c778682ced5290403014a1313367fcd5f6`.
- A separate Sol review approved only the two literal absolute-`/usr/bin/sudo` install commands
  with no P0--P2. The fixed provisioner installed exact source tree
  `1c71f032105a2666ee3ef2f9acc2a233c4c43b0a4c3cc52dac4901a159970a09`; the second-stage
  installer published exact capsule tree `5dff4dbb19d4e1af24363d83a842454dc7a19a298a6ecf58e5a38b32bc6e5851`
  before launcher `ecff3088...5a8d` and reported `launcher_invoked=false`.
- Independent postconditions: exact membership/hashes, root ownership, modes, nlink 1 and empty
  xattrs; four root/user workload and rsync locks free; no v2 process; no local libtpu holder.
- Authority: installation only. No launcher/JAX/backend/HLO/TPU/cloud/network, numerical,
  performance or 8K claim. Gate D remains open. Exact next after review/persistence is a separate
  execution-only review for one fresh append-only protected tag.

## 2026-09-01 forced-round PP16 numerical diagnostic adjudication

- Burned tag: `gate_d_forced_round_pp16_numerical_20260901T164835192240185Z`; exactly one compiled
  executable invocation and one bounded nine-array host transfer completed. The numerical result is
  `NUMERICAL_REJECTED`; success publication alone failed on an overly narrow caller dependency rule.
- Adjudicator: `scripts/greenfield/adjudicate_gate_d_forced_round_pp16_numerical_diagnostic.py`,
  committed pin `b13a4eeddb17eb5757535297778c1c7610841056`, source SHA-256
  `cbb47b6262c1a7251558f061744bad4482701732dc69403b16bf316c76d496db`. Origin and locked
  `US-CENTRAL2` mirror are exact; checkout archive SHA-256 is
  `f55f63dbc22816bc040f907f219892c40f602e291a75d4a15ebc4b1e9ca833e6`.
- Artifact: `docs/artifacts/gate-d-forced-round-pp16-numerical-diagnostic-adjudication.json`, 5,977
  bytes, SHA-256 `edb358db9671a979e7d50b8b82aa8e5d834072c927cd82b63be923e4284d4e29`.
  It replays 18 exact generation-qualified objects, requires terminal generation
  `1788281509956605` after every payload, proves exhaustive soft-deleted count zero, validates both
  8-host clean censuses, exact StableHLO/optimized HLO, runtime/mirror/dependencies and independently
  reclassifies all nine output arrays against the sealed accepted capsule.
- Scientific result: RMS operands/input and both normalized BF16 owners match exactly; both query
  owners and current key differ. Forced normalized-BF16 materialization is therefore rejected as
  sufficient. The first observed divergence is the query projection output; the exact dot
  algorithm/precision/layout cause remains unproven.
- Verification: Sol first withheld two P1s (non-exhaustive soft-delete query and missing dual-pin
  helper binding); both were fixed and the re-review returned `APPROVE` with no P0--P2. Final suites
  pass 73/73 focused and 435/435 adjacent CPU-only tests with one immutable historical certificate
  regeneration assertion intentionally deselected; Ruff, Python compilation and diff-check pass.
- Authority: bounded numerical rejection and authenticated diagnostic only. Gate D, decoder, 8K,
  performance and causality remain open. Exact next is the smallest CPU-only explicit
  projection-product/rounding/accumulation/association analyzer; do not rerun this tag or mechanism.

## 2026-09-01 projection-arithmetic frontier analysis

- Analyzer `scripts/greenfield/analyze_gate_d_projection_arithmetic_frontier.py` is committed at
  `2e8528c023aee9354610651fbf28b742760e5b5c`, source SHA-256
  `02a82800eee81e9822f91c9c962568d85242f1fdce56e3e5e6275a2ba672ed4f`; origin is exact and the
  locked `US-CENTRAL2` mirror checkout archive is
  `32982b1b8633bd16ccf3fcefa27b7e09ceeb29abe98ab54b986871141c156d6b`.
- Canonical artifact `docs/artifacts/gate-d-projection-arithmetic-frontier-analysis.json` is 24,001
  bytes, SHA-256 `4a6be0f33f221df46f384cd2047da1aa664ae4c45763e68f0f1df05f8644e20c`.
  Its exact sealed CPU runtime binds 532 loaded module files and 11/49 before/after native mapped
  files to fixed byte manifests, strictly parses the 22-entry JAX and two-entry libtpu capsule
  allowlists, uses one CPU device and opens no accelerator file descriptor.
- The CPU FP32 JAX `dot_general` control followed by the identical one-row LayerNorm/RoPE suffix
  reproduces the accepted 128-float current key bit-for-bit (0/128 mismatches; SHA-256
  `5006ad4f...329b`). All 27 explicit reduction probes are rejected; the best direct 48x128
  tile/lane association still misses 100/128 values. This is one captured position-8155 witness,
  not a general TPU arithmetic identity or root-cause proof.
- Sol returned `APPROVE` with no P0--P2 after the loaded-byte/capsule hardening. Current validation
  passes 23/23 focused and 99/99 causal-boundary CPU-only tests, Ruff 0.16.5, formatting, Python
  compilation and canonical rerun comparison.
- Classification: `CPU_F32_DOT_CONTROL_EXACT_ACCEPTED_KEY_CAPTURED_INPUT;
  ENUMERATED_REDUCTION_PROBES_REJECTED;TPU_CAUSALITY_UNPROVEN;GATE_D_OPEN`. No TPU compile/run,
  performance, mechanism, full DSA or 8K authority exists. Exact next after artifact persistence is
  a separately reviewed default-off source/compile-only design for the smallest one-row TPU F32
  projection-contraction discriminator; do not rerun the full DSA path or 8K decoder.

## 2026-09-01 PP16 projection-contraction source certificate

- Source pin `e3af2ca776ba5a789c6b2c0cc7bbd42258bbdc32` is exact on origin; its locked
  `US-CENTRAL2` mirror replay passed strict connectivity and full checkout comparison with archive
  SHA-256 `fdb39dabd574eff131c85204916b08cfdbbce2325df28ae08cc060d0fe5859a2`.
- Builder `glm_tpu/greenfield/benchmarking/gate_d_projection_contraction_pp16.py` is default-off
  and source-only. It accepts only physical PP16 stage zero `[0,1]`, carries one local BF16 row,
  performs one FP32 projection under highest matmul precision, preserves the divide-sqrt suffix,
  roots normalized/projection/current-key owners, and contains no collective or execution action.
- Certificate `docs/artifacts/gate-d-projection-contraction-pp16-source.json` is 4,176 bytes,
  SHA-256 `5744eee0ef2cf35a4566cc0de165be1338160daaae3f4dd133554b2aa8280e9f`.
  Its complete builder AST is `58d6f008...c00704`; exact direct dependency hashes and the accepted
  projection/topology predecessors are bound.
- A forced two-CPU-device abstract replay proves owner-preserving output shapes
  `[2,1,6144]`, `[2,1,128]`, `[2,1,128]`. Hostile tests reject AST aliases/getattr/top-level
  effects, dependency drift, path-shadow packages and spoofed preloaded validators. Sol returned
  final `APPROVE` with no P0--P2; focused validation is 26/26 and the adjacent causal boundary is
  74/74 after exact artifact binding.
- Classification: `PROJECTION_ONLY_PP16_SOURCE_ACCEPTED;COMPILE_UNPROVEN;
  TPU_CAUSALITY_UNPROVEN;GATE_D_OPEN`. It authorizes persistence only, not HLO acquisition, TPU
  compile/execution, performance, full DSA/8K or Gate-D closure. Exact next is a separately reviewed
  compile-only HLO acquisition boundary for this callable; no executable invocation is authorized.

## 2026-09-01 projection-contraction HLO acquirer source certificate

- Commit `c686e6491387ae46ebfc468f401f79c7177ca0f5` is exact on origin. The locked
  `US-CENTRAL2` mirror passed strict connectivity and full-checkout equality with archive SHA-256
  `6730f25458d88c343864488be28bd46d5e47cefe572370221e39a1f574966b23`.
- The default-off acquirer has four abstract PP16 stage-zero inputs, three owner-preserving rooted
  outputs, one `eval_shape`, one lowering and one compilation call. Its complete source audit finds
  zero executable invocations and grants no numerical-execution authority.
- The first committed analyzer at `afd54dc` failed before validator execution because an ordinary
  package import reached `validation/__init__.py` and unavailable `ml_dtypes` under `python -I -S`.
  No JAX/backend/TPU/cloud action occurred. The successor executes only the already snapshotted,
  commit-matched validator bytes in a private namespace under exact canonical import finders.
- Canonical certificate
  `docs/artifacts/gate-d-projection-contraction-hlo-acquisition-source.json` is 2,217 bytes,
  SHA-256 `a94395ef13f5cfa7de7bc221fdc33d4498637a02ffc55d3cba985834dd99b93a`.
  Sol approved the import-boundary correction with no P0--P2.
- Classification: `PROJECTION_HLO_ACQUIRER_SOURCE_ACCEPTED;ORCHESTRATION_UNPROVEN;
  TPU_COMPILE_UNAUTHORIZED;TPU_EXECUTION_UNAUTHORIZED;GATE_D_OPEN`. Persistence only is
  authorized. Exact next is a separately reviewed append-only publisher/wrapper and immutable
  installer/launcher; no compile, execution, full DSA/8K or Gate-D closure exists.

## 2026-09-01 projection-contraction HLO orchestration/install source certificate

- Reviewed source correction `d4288831edbd4f08ea61ef8faf6edbe65eabfb4d` is exact on origin.
  The locked `US-CENTRAL2` mirror passed connectivity, fsck and full-checkout equality; checkout
  archive SHA-256 is `77ac64a219f8819c9c569bcdd31c729b44fcf546e4cefb14729f36cc1acc5323`.
- The default-off publisher, wrapper, descriptor launcher and no-replacement installer bind the
  exact compile-only acquirer. The launcher snapshots the committed wrapper into sealed FD 10;
  the wrapper inherits already-held root workload/rsync locks as FDs 11/12. The installer publishes
  the exact read-only capsule before the launcher and has zero launcher invocations.
- Canonical certificate
  `docs/artifacts/gate-d-projection-contraction-pp16-hlo-orchestration-install-source.json` is
  3,026 bytes, SHA-256
  `f6d1736105e67dbbe9b336b727ca015a3bf7506231aa8ae7dcd407a81cbe666e`. It binds all seven
  source hashes, the three predecessor certificates and an empty loaded-JAX module set.
- Exact-source replay and the complete projection-HLO source/security set pass 108/108 in an
  isolated CPU-only environment. Sol approved staged correction
  `4d4077dc0519ce1e821d87cee0e6fd047b07eed8934d83f6f23c200ea9ccbe1d` with no P0--P2;
  it independently reviewed the exact certificate/test/docs batch and returned
  `APPROVE PERSISTENCE` with no P0--P2.
- Classification: `PROJECTION_HLO_ORCHESTRATION_INSTALL_SOURCE_ACCEPTED;
  INSTALL_UNAUTHORIZED;TPU_COMPILE_UNAUTHORIZED;TPU_EXECUTION_UNAUTHORIZED;GATE_D_OPEN`.
  No privileged install, launcher, JAX/backend, lowering, compile, bucket write, numerical work,
  full DSA/8K or performance claim occurred. Exact next after review/persistence is a separate
  literal immutable install-only review; compilation and executable invocation remain unauthorized.

## 2026-09-01 projection-contraction HLO immutable runtime installed

- Sol separately returned `APPROVE INSTALL ONLY` for exactly two literal absolute-`/usr/bin/sudo`
  commands. The fixed provisioner copied exact five-file tree
  `9c157b99e7e67cc01ea8b1209155a6203269b0bd11438463a4fda0bc48c679cc` to the root-owned
  0755 source capsule. The install-only program then published the root-owned 0555 three-file
  runtime capsule before launcher `082b44f9...023e`; it reported `launcher_invoked=false`.
- Artifact `docs/artifacts/gate-d-projection-contraction-pp16-hlo-runtime-install.json` is 3,761
  canonical bytes, SHA-256
  `99a134c400629c936b761bf738873f2a81a41512f8c1823d26d5232b0648abc6`. Every file has exact
  reviewed bytes, uid/gid 0, mode 0555, nlink 1 and empty xattrs. A later read-only observation at
  `2026-09-01T20:06:27.167677Z` simultaneously acquired and identity-bound all four canonical
  leases: root/user workload and root/user rsync. No launcher/acquirer process was present.
- Classification: `IMMUTABLE_RUNTIME_INSTALLED_NOT_INVOKED;TPU_COMPILE_UNAUTHORIZED;
  TPU_EXECUTION_UNAUTHORIZED;GATE_D_OPEN`. No launcher, JAX/backend, HLO acquisition, bucket write,
  TPU/numerical/performance/full-DSA/8K work occurred. Exact next after adversarial review and
  persistence is a separately reviewed literal fresh-tag compile-only launcher command; executable
  invocation remains forbidden.

## 2026-09-01 projection-contraction HLO v1 pre-JAX failure; v2 source correction

- Approved tag `gate_d_projection_contraction_pp16_hlo_20260901T201007969634980Z` failed closed in
  0.2 seconds inside the embedded Git authority verifier: `git config --includes=false` returned
  129 because `--includes` takes no value. No run directory, JAX import, backend/compiler, remote
  host command, cloud mutation or TPU work began; executable invocation count is zero and the tag is
  burned. Canonical failure artifact
  `gate-d-projection-contraction-pp16-hlo-v1-git-config-failure.json` is 2,240 bytes, SHA-256
  `14de62b52fbdde659c6d222442d2e737a417b3d84893cefdd3d0476cbcfc4086`. Later read-only
  checks prove local/live/all-version/exhaustive-soft-deleted vacancy and simultaneously acquire all
  four identity-bound canonical leases.
- V2 source uses supported `--no-includes` and adds a complete success-path regression against a
  clean pushed production-origin clone; the prior tests had exercised only the helper and a
  replacement-ref failure before this argument. Immutable v1 targets remain untouched. New source,
  capsule and launcher targets are versioned v2. Wrapper, launcher and installer SHA-256 values are
  `058338a1...992`, `b3b6ff7c...9a23` and `21a14f20...31be`; focused source/security coverage passes
  80/80 and the complete projection-HLO source/security slice passes 111/111, with Bash syntax and
  Ruff checks. Sol independently reviewed staged diff `a0eddcbb...9740` and returned
  `APPROVE PERSISTENCE` with no P0--P2.
- Authority remains source persistence only. No v2 install/launcher/HLO/TPU/cloud/numerical/full
  DSA/8K action is authorized. Exact next is commit/push/mirror, regenerated v2 source certificate,
  then separate install-only and fresh-tag compile-only reviews.

The regenerated v2 source certificate
`gate-d-projection-contraction-pp16-hlo-orchestration-install-source-v2.json` is 3,026 canonical
bytes, SHA-256 `b0e58ea7c5a5e3bb0158f936442a67336d279b9f991c2247adfe9f2629759c79`.
It binds commit `efe99ba87c1e1a7163f436fb7cc55bcd46e395ae`, all seven exact sources, the v2
immutable targets, the supported Git flag and the unchanged predecessor authorities. It authorizes
persistence only; review and persistence precede any v2 installation.

## 2026-09-01 projection-contraction HLO v2 immutable runtime installed

- Certificate commit `f9fa964b61ce51a8e2e0452a3603b7e9c9d30093` is exact on origin; locked
  `US-CENTRAL2` mirror replay passed with checkout archive SHA-256
  `cd10f198240a75aec827919afb739fa664341ef1cb3302929109d3412a768057`.
- Sol returned `APPROVE INSTALL ONLY` for exactly two literal absolute-`/usr/bin/sudo` commands.
  The fixed provisioner copied exact five-file tree
  `49409ba79045fa402ff9957c25ed6a658b6fb15b803dccda7f61d7d941930145` to the root-owned
  0755 v2 source capsule. The installer published the root-owned 0555 three-file runtime capsule
  before launcher `b3b6ff7c...9a23` and reported `launcher_invoked=false`.
- Artifact `gate-d-projection-contraction-pp16-hlo-runtime-install-v2.json` is 3,699 canonical
  bytes, SHA-256 `6a10743f80b924e5fe025f97b51f62359193653f73b83b65809cfbd8c42a951c`.
  Every installed file is root-owned, exact, mode 0555, nlink 1 and xattr-free. One read-only
  observation at `2026-09-01T20:28:18.971859772Z` simultaneously acquired all four identity-bound
  canonical leases and found no launcher/acquirer process. Sol independently verified staged diff
  `ccfe3c149b22f4c5af35376cbd1d32f31cd61d6a65fbd22ccab717dc9571c100` against live objects and
  returned `APPROVE PERSISTENCE` with no P0--P2.
- Classification: `IMMUTABLE_RUNTIME_V2_INSTALLED_NOT_INVOKED;TPU_COMPILE_UNAUTHORIZED;
  TPU_EXECUTION_UNAUTHORIZED;GATE_D_OPEN`. No launcher, JAX/backend, HLO acquisition, bucket write,
  TPU/numerical/performance/full-DSA/8K work occurred. Exact next after review and persistence is a
  separately reviewed fresh-tag compile-only launcher command; numerical execution remains
  forbidden.

## 2026-09-01 projection-contraction HLO v2 pre-wrapper root-lock failure

- Approved fresh tag `gate_d_projection_contraction_pp16_hlo_20260901T203408107762365Z` failed
  closed at the launcher's nonblocking canonical root-lock loop before protected-wrapper execution.
  The cron sync session ran from `20:35:01Z` through `20:35:23Z` with the exact root/user rsync-lock
  chain and completed all four repository mirrors during the overlap.
- The shared traceback line cannot prove which root-lock iteration failed. Sol blocked the initial
  root-rsync-specific interpretation as P1. Canonical artifact
  `gate-d-projection-contraction-pp16-hlo-v2-root-lock-failure.json` therefore classifies
  non-exclusive `PRE_WRAPPER_CANONICAL_ROOT_LOCK_CONTENTION` plus observed cron root-rsync overlap.
  It is 3,471 bytes, SHA-256
  `b1bafcf9ee2608cb1de75c114781b2ec3c2c5d6232c12f4303b616f1fcfa3159`.
- Local run state and live/all-version/soft-deleted remote histories remain vacant. No wrapper,
  cloud, JAX/backend/compiler, remote-host or TPU work occurred. A later observation simultaneously
  acquired all four canonical leases and found no launcher/acquirer process. The tag is burned.
- Classification: `PRE_WRAPPER_CANONICAL_ROOT_LOCK_CONTENTION;CRON_ROOT_RSYNC_OVERLAP_OBSERVED;
  NO_RUN_DIRECTORY;NO_CLOUD_MUTATION;PROTECTED_WORKFLOW_NO_JAX_OR_TPU_WORK;TAG_BURNED;
  GATE_D_OPEN`. Exact next after review/persistence is a separately reviewed new tag outside the
  cron window; numerical execution and full 8K remain forbidden. Sol verified corrected staged diff
  `d1515c3bbf73aeba716124737520aed449b0132267c65144f316ba3cacbc93d0` and returned
  `APPROVE PERSISTENCE` with no P0--P2.

## 2026-09-01 projection-contraction HLO v2 compile and publication failure

- Fresh tag `gate_d_projection_contraction_pp16_hlo_20260901T204123954066730Z` completed one
  abstract-input PP16 TPU-v4 compile in 0.424247954 seconds. The executable invocation count is
  zero, numerical execution is false, and both fleet censuses are 8/8 clean. The optimized HLO is
  31,857 bytes, SHA-256 `7f678b10...84f4d`; StableHLO is 7,420 bytes, SHA-256
  `4b3fa252...e2e3`.
- Success publication failed closed because the compiler dependency manifest contained required
  flag `GLM_GATE_D_PROJECTION_CONTRACTION_HLO=1` while the publisher's expected environment omitted
  it. The bounded diagnostic ledger is generation `1788295391457443`, SHA-256
  `16ce098c...40bc`; no `HLO_ACQUIRED` terminal exists and the tag is burned. Canonical failure
  artifact `gate-d-projection-contraction-pp16-hlo-v2-publication-failure.json` is 3,141 bytes,
  SHA-256 `d38c43af...10aa`.
- Independent read-only replay downloaded all 15 exact generations, proved the diagnostic ledger
  last, found 15 unique live paths and zero exhaustive soft-deleted generations. Replay artifact
  `gate-d-projection-contraction-pp16-hlo-diagnostic-remote-replay.json` is 4,819 bytes, SHA-256
  `5d6583be...8b61`. The validator cross-checks every diagnostic-ledger member and every locally
  consumed byte string, including runner and HLO, against that generation-qualified catalogue.
- Offline fail-closed adjudication binds the remote replay, runner, absent success terminal, exact
  owner shapes and complete contraction-to-both-rooted-outputs lineage. It proves one live row per
  each of two owners, BF16-hidden-to-FP32 conversion, an FP32 `128x6144` multiply reduced over width
  6144, and zero collectives or host effects. An isolated `python -I -S -B` CLI authenticates and
  directly loads only the leaf validator/parser/error sources through synthetic packages; hostile
  package initializers, forbidden modules and accelerator FDs are rejected. Exact report
  `gate-d-projection-contraction-pp16-diagnostic-hlo-adjudication.json` is 2,318 bytes, SHA-256
  `c36af80c...8150`. This is diagnostic structural evidence only: numerical and Gate-D closure
  remain unproven.
- V3 adds the missing publisher flag and a non-circular acquirer/publisher/wrapper environment
  regression, while preserving immutable v1/v2 targets. V3 publisher/wrapper/launcher/installer
  SHA-256 values are `f3f20a01...bc97`, `4dd06dcb...1c70`, `e6fceb3a...6772` and
  `7125172b...ab03`. Exact next is adversarial source review, commit/push/locked same-region mirror,
  a regenerated V3 source certificate, separate install-only review, and a fresh compile-only tag.

## 2026-09-01 projection-contraction HLO V3 source certificate; install unauthorized

- Sol verified staged diff `dab4e9c8...a8145e`, including exact parameter-to-root lineage inside
  all five current-key fusion bodies and hostile internal-bypass tests, and returned
  `APPROVE PERSISTENCE` with no P0--P2. Commit `9863782` is exact on origin and the locked
  `US-CENTRAL2` repository mirror completed successfully.
- Offline artifact
  `gate-d-projection-contraction-pp16-hlo-orchestration-install-source-v3.json` is 3,026 canonical
  bytes, SHA-256 `1983aa15...2fff`. It regenerates from committed code pin `9863782`, binds all
  seven exact V3 sources and predecessor authorities, loads no JAX, and retains the default-off
  install/compile/execution boundary.
- Classification: `PROJECTION_HLO_ORCHESTRATION_INSTALL_SOURCE_ACCEPTED;
  INSTALL_UNAUTHORIZED;TPU_COMPILE_UNAUTHORIZED;TPU_EXECUTION_UNAUTHORIZED;GATE_D_OPEN`.
  Exact next after adversarial review and persistence is a separately reviewed literal install-only
  command. No launcher, compile, TPU, numerical, full DSA/8K or performance authority exists.

## 2026-09-01 projection-contraction HLO V3 immutable runtime installed

- Sol independently verified the two literal privileged commands, exact five-member staging tree
  `286c3e3e...e9d6`, fixed provisioner `2b9c8c2b...0594`, absent targets and no-launch boundary,
  then returned `APPROVE INSTALL ONLY` with no P0--P2.
- The root-owned source tree, three-member runtime capsule and launcher are versioned V3, mode
  0755/0555, xattr-free and exact to their certificate hashes. The installer reported
  `launcher_invoked=false`; no launcher/acquirer process or local TPU owner remained. At
  `2026-09-01T21:32:40.839444364Z`, one read-only auditor simultaneously held all four canonical
  leases.
- Artifact `gate-d-projection-contraction-pp16-hlo-runtime-install-v3.json` is 3,699 canonical
  bytes, SHA-256 `964329c0...c1c8`. Origin and the locked `US-CENTRAL2` mirror are exact at authority
  commit `4a786ec`; replayed checkout archive SHA-256 is `feab9bf7...b2de`.
- Classification: `IMMUTABLE_RUNTIME_V3_INSTALLED_NOT_INVOKED;TPU_COMPILE_UNAUTHORIZED;
  TPU_EXECUTION_UNAUTHORIZED;GATE_D_OPEN`. Exact next after review/persistence is a separate fresh
  tag and literal compile-only launcher review. Numerical execution and full DSA/8K remain
  forbidden.

## 2026-09-01 projection-contraction HLO V3 successful compile adjudicated

- Fresh tag `gate_d_projection_contraction_pp16_hlo_20260901T213605719107105Z` completed one
  abstract-input compile on adjacent PP16 stage-zero TPU-v4 devices `[0,1]`. The compiled
  executable invocation count is zero; numerical execution, performance and Gate-D claims are
  false. Both fleet censuses are clean.
- The optimized HLO is 31,857 bytes, SHA-256 `817ba2ed...98ea`; StableHLO is 7,420 bytes,
  SHA-256 `4b3fa252...e2e3`. Exact causal adjudication proves one BF16 live row per each of two
  owners, BF16-to-FP32 conversion, the FP32 width-6144 projection reduction, both complete
  current-key fusion branches, rooted projection/current-key outputs, and zero collectives or host
  effects.
- Independent read-only replay downloaded all 17 exact generation-qualified objects from the
  `US-CENTRAL2` bucket, found zero exhaustive soft-deleted generations, and proved
  `HLO_ACQUIRED` generation `1788298702037876` is last. Replay artifact
  `gate-d-projection-contraction-pp16-hlo-success-remote-replay.json` is 5,219 bytes, SHA-256
  `9c809a93...7085`.
- The isolated `python -I -S -B` adjudicator loads only hash-pinned parser/error/validator bytes,
  executes no package initializer, imports no JAX and opens no accelerator descriptor. It binds
  every local remote member, the 15-member ledger, self-bound success marker and terminal receipt.
  Exact report `gate-d-projection-contraction-pp16-success-hlo-adjudication.json` is 2,301 bytes,
  SHA-256 `54eb6105...ad7`; 42 focused CPU-only tests pass, including hostile terminal, replay,
  fusion-lineage and import-boundary attacks.
- Sol first blocked persistence on ignored authenticated claim fields and duplicate evidence paths.
  The corrected validator requires exact runner/summary/evidence schemas, all closure/numerical/
  performance claims false, and exact unique evidence records. Sol verified staged diff
  `d056bb92...0250` and returned `APPROVE PERSISTENCE` with no remaining P0--P2.
- Classification: `HLO_CAUSAL_STRUCTURE_ACCEPTED;PP16_OWNER_LOCALITY_ACCEPTED;
  TPU_NUMERICAL_UNPROVEN;HLO_ACQUIRED_TERMINAL_VERIFIED;GATE_D_OPEN`. This authorizes persistence
  only. Exact next after commit/push and locked same-region mirror is a separately reviewed bounded
  numerical discriminator; full DSA/8K and performance remain unauthorized.
+
## 2026-09-01 immediate-stop numerical-orchestration checkpoint

- Persisted authority is commit `f762a28899b812b43f321c4e56bfc2f94f0a9811`, exact on origin
  and the locked `US-CENTRAL2` mirror; it contains the reviewed numerical publisher. No numerical
  launcher/install/TPU execution exists.
- The orchestration/install source batch is staged but uncommitted. Sol blocked obsolete local
  terminal filenames, then blocked mutable post-publisher local-terminal substitution. Both are P1;
  no prior review/hash authorizes persistence or execution.
- Unreviewed WIP now has the immutable publisher emit status plus marker/remote-terminal identity
  directly on stdout and makes the local terminal/receipt verifier supplementary. Publisher SHA is
  `a53f051b...d962`; wrapper SHA is `ea06e4d5...d58`. Downstream launcher/installer/analyzer pins
  are stale, and the last correction has not run tests or review.
- Exact next: substitution attack test; full hash cascade; Bash/Ruff/focused tests; one exact staged
  Sol delta review; commit/push/mirror; certificate; separate install-only review; install without
  launch; separate fresh-tag review; one bounded execution and adjudication.
- Classification: `NUMERICAL_ORCHESTRATION_WIP_UNREVIEWED;INSTALL_UNAUTHORIZED;
  TPU_EXECUTION_UNAUTHORIZED;FULL_DSA_8K_UNAUTHORIZED;GATE_D_OPEN`.
- Recovery-only WIP patch: `gs://driftbench-dsv4-uc/repos/wip/glm-tpu-gate-d-pp16-numerical/paused-20260901T225431Z.patch`; never treat it as an approved source pin.

## 2026-09-01 numerical-orchestration correction awaiting delta review

- Publisher `success` emits one anchored `NUMERICAL_RESULT status=… marker_sha256=…
  terminal_generation=… terminal_sha256=…` authority line after terminal replay/receipt; the wrapper
  dispatches only from it and treats the reopened local terminal/receipt as a supplementary check
  that can deny but never flip. Executed substitution regressions (opposite-status replacement both
  directions, receipt-generation replacement, seven non-authority publisher outputs, publisher
  failure) and publisher unit tests pass; focused suite 74/74; Ruff and `bash -n` clean.
- Hash chain: publisher `6ead9e13…e0ff`, wrapper `16e348e3…6250`, launcher `755c60fb…1cc6`,
  installer `72c1bb1f…d538`, analyzer `508b95c9…fd84`. Source-batch staged diff
  `59b8552b…7d5e`.
- Classification unchanged: `NUMERICAL_ORCHESTRATION_CORRECTED;REVIEW_PENDING;INSTALL_UNAUTHORIZED;
  TPU_EXECUTION_UNAUTHORIZED;FULL_DSA_8K_UNAUTHORIZED;GATE_D_OPEN`.

## 2026-09-01 numerical orchestration persisted at 7902b4c9; certificate generated

- Sol: P2 (stale goal checkpoint) corrected, then `APPROVE PERSISTENCE` for staged diff
  `4f4486d4…1315` = commit `7902b4c9517d55959151088ca43429d57bb486c5`, on origin and replayed from
  the locked `US-CENTRAL2` mirror (fsck true, checkout archive `d5c3483f…add8`).
- Certificate `docs/artifacts/gate-d-projection-contraction-pp16-numerical-orchestration-install-source.json`
  SHA-256 `1c75e761be61bf9e9043f29098ddf15f962d37657409d05f77a10b2aee358e38`, code hash
  `7902b4c9…86c5`; persistence only.
- Classification: `NUMERICAL_ORCHESTRATION_PERSISTED;CERTIFICATE_GENERATED;INSTALL_UNAUTHORIZED;
  TPU_EXECUTION_UNAUTHORIZED;FULL_DSA_8K_UNAUTHORIZED;GATE_D_OPEN`. Next: separate install-only
  command review; never launch from that review.

## 2026-09-01 numerical V1 runtime installed, not invoked; batched review requested

- Sol `APPROVE INSTALL ONLY`; provisioner + installer ran as two literal `/usr/bin/sudo -n` commands
  from staging tree `9488e9c3…1c8d`; capsule/launcher installed root 0555 with exact hashes;
  `launcher_invoked=false`; leases free; no process. Artifact
  `gate-d-projection-contraction-pp16-numerical-runtime-install.json` SHA-256 `9be45200eacd8076e4a25a78da2a97599ee80432c1ab1aab6111cd82e6b1a042`.
- Owner instruction: batch Sol reviews. Next single review = persistence of this batch + literal
  fresh-tag command for `gate_d_projection_contraction_pp16_numerical_20260901T233855937688834Z`.
- Classification: `NUMERICAL_RUNTIME_V1_INSTALLED_NOT_INVOKED;TPU_EXECUTION_UNAUTHORIZED;
  FULL_DSA_8K_UNAUTHORIZED;GATE_D_OPEN`.

## 2026-09-01 numerical V1 fail-closed (HLO metadata); V2 bridge batch

- Tag `…233855937688834Z` failed closed before executable invocation: optimized-HLO debug metadata
  (driver path + 2 call-site lines) differed; StableHLO identical. Failure artifact SHA `c07dd4f7fdb558304b4819fe1faedf8a27d39041df71d5c19c6f3f8cc0cc7980`.
- V2: bridge `c2732f71…6bea` derives numerical HLO `70485b06…0564` from accepted `817ba2ed…`;
  driver/publisher compare against derived bytes; V2 install targets; chain repinned; 79/79 tests.
- Classification: `NUMERICAL_V1_TAG_BURNED_HLO_METADATA;V2_BRIDGE_SOURCE_READY;REVIEW_PENDING;
  TPU_EXECUTION_UNAUTHORIZED;GATE_D_OPEN`.

## 2026-09-02 V2 discriminator NUMERICAL_REJECTED; rotary root cause candidate

- Run `…235818944668679Z` executed once on chips {0,1}; HLO bridged byte-exact; result REJECTED
  (normalized exact; projected/current key not). Publisher failed closed on the `sync.txt`
  wrapper/publisher mismatch; 18-object diagnostic archived (terminal generation `1788307624494629`).
- Adjudication artifact `gate-d-projection-numerical-v2-rejection-adjudication.json` SHA
  `25908caf…1ef7`: projection f32-accurate (≤4.8e-7), key LayerNorm bit-exact, all large errors in
  rotary dims; TPU implied cos error up to 9.9e-3 vs accepted 7.1e-5 → on-device TPU cos/sin at
  large angles. Classification: `V2_NUMERICAL_REJECTED_ADJUDICATED;ROOT_CAUSE_CANDIDATE_TPU_ON_DEVICE_ROTARY_COS_SIN;TPU_FIX_UNPROVEN;GATE_D_OPEN`.

## 2026-09-02 V3 host-rope batch ready for one batched review

- New `rotary_table.py`/`dsa_host_rope.py`/host-rope builder/validation; V3 driver+publisher use a
  structural HLO contract (no transcendentals, no collectives) and tolerance faithfulness; archived V2
  bytes rejected for rotary only, CPU-rotated TPU projection accepted. Chain driver `f519ce40…`,
  publisher `092175d8…`, wrapper `8f1522e5…`, launcher `6d3288dc…`, installer `d9ef0757…`.
- Classification: `V3_HOST_ROPE_SOURCE_READY;REVIEW_PENDING;TPU_EXECUTION_UNAUTHORIZED;GATE_D_OPEN`.
- Sol P1 corrections (publisher binds all metrics + source/row identities; CPU-only adjudicator;
  goal no longer jumps to 8K). Chain: publisher `5a8d30cf…`, wrapper `92bcb109…`, launcher
  `61a45b6d…`, installer `22dc77e7…`, analyzer `f1010452…`; 131/131 tests.

## 2026-09-02 V3 host-rope replay NUMERICAL_ACCEPTED (bounded rotary-fix proof)

- Tag `…004306002075694Z`, pin `ecf723d7…`: one invocation, HLO 0 collectives/0 transcendentals,
  key within 4.99e-7 of F64 reference (rotary 1.25e-7 vs 2.29e-3 in V2), implied cos/sin ≤6.5e-7,
  terminal generation `1788310923525951`. Artifact SHA `c4fca0a667ca5928ec10f04c741e331bae5ce0bf86ff24c4b5a412476ed40461`.
- Classification: `BOUNDED_TPU_HOST_ROPE_KEY_FAITHFUL_TERMINAL_VERIFIED;ROTARY_ROOT_CAUSE_FIX_BOUNDED_PROOF;DECODER_UNPROVEN;GATE_D_OPEN`.

## 2026-09-02 DSA host rotary rows in the decoder (default off) + selection witness

- Kernel/layer/prefill/decoder/prefill-runtime/compile-script/runner plumbing for
  `dsa_rope_table_enabled`; HLO contract pins table param + lookup and zero transcendentals with
  both tables; forced-CPU equivalence and zero-transcendental compile pass; 126/126 regression.
- Witness artifact `gate-d-dsa-selection-witness-layer1-position8155.json` SHA `514d6e5f6e5ba8f6c7c4382a262bc5cc85547ba0e1217872a5942987ec9fc4b1`: V3 key
  reproduces the accepted event exactly; V2 key perturbs scores 4.2e-3 without flipping it.
- Classification: `DSA_HOST_ROPE_DECODER_INTEGRATED_DEFAULT_OFF;SELECTION_WITNESS_V3_EXACT;
  MERGE_TO_REWRITE_PENDING;PROTECTED_8K_UNAUTHORIZED;GATE_D_OPEN`.

## 2026-09-02 DSA-table protected 8K run refused; host-rotary-table hypothesis refuted

- Run `…mainrope_dr_…_trace2_20260902T021346091708582Z`, pin `ba7d1e72`, fail-closed at the exact-DSA
  contract (event 0 order offset 49, event 1 six swaps); censuses 8/8 (`f5fbcd2f…`/`494a67ed…`);
  HLO DSA-table contract passed; first token exact; no SUCCESS/DB/trace/performance claim.
- Adjudication `gate-d-dsa-rope-table-8k-refusal-adjudication.json` SHA `0f7c36d28d748dc5754ef5dcc829b88961c5713bcf8a6db4ff4667e8960c27e0`
  (CPU, archived bytes): event 0 bit-exact without table, perturbed 2.6e-3 with it (position 0 too);
  event 1 unchanged (mean |Δ| 0.0148 both). Legacy indexer uses on-device `jnp.cos/sin`
  (`glm_dsa_indexer.py:1078`, `b3c25df47`).
- Classification: `HOST_ROTARY_TABLE_HYPOTHESIS_REFUTED_AT_ENGINE_LEVEL;PROTECTED_8K_REFUSED_TWICE_AT_EVENT_1;
  DSA_ROPE_TABLE_DEFAULT_OFF_DO_NOT_RELAUNCH;GATE_D_OPEN`.
- Sol corrections applied: adjudicator binds run pins/HLO contracts/logs/legacy source and enforces
  observer schema; artifact SHA `2a7c8fb5df041cad1dc57bbd312694d932b46f0ad6cb76d99ffd51e0869f5c97`;
  classification `HOST_ROTARY_TABLE_REFUTED_AS_LEGACY_FAITHFULNESS_FIX;EVENT_0_DELTA_ATTRIBUTION_TO_TABLE_ALONE_NOT_PROVEN;
  EVENT_1_MECHANISM_UNDETERMINED;PROTECTED_8K_REFUSED_TWICE_AT_EVENT_1;GATE_D_OPEN`.

## 2026-09-02 layer-1 scale frontier certificate (CPU, sealed bytes)

- `gate-d-layer1-scale-frontier-certificate.json` SHA `e8abfb9b55496ba7c2e58c8b8ac1bec2f7dc58c74b5096db101db5961b740982`:
  DB548 row `9b52a04e…` and accepted row `9936ee1e…` are both exact functions of the same FP32 RMS
  input (`dense + attention_update + combined_residual`, unrounded), weight `10e34f4f…`, eps 1e-5 and a
  single output rounding; scale windows s0−4…s0−1 (DB548) vs s0…s0+14 (accepted), s0=`0x433295db`.
- Classification: `LAYER1_SCALE_FRONTIER_CERTIFIED;SAME_FP32_RMS_INPUT;FRONTIER_IS_ONE_FP32_SCALAR;
  REDUCTION_STRUCTURE_NOT_IDENTIFIED;GATE_D_OPEN`.
- Legacy m32 module: 313 RMS reduces over `f32[32,6144]{T(8,128)}` dims={1} → `f32[32]`; greenfield:
  `f32[1,1,6144]{T(1,128)}` → scalar. Captured-RMS probe `accepted_split` arm redesigned as FP32-carry
  accepted-schedule arm (default-off, unexecuted); matcher `fp32_carry_schedule`; tests 5/5 (+2 skips),
  16/16 adjacent. Classification: `RMS_SCHEDULE_ARM_READY;REVIEW_PENDING;TPU_UNAUTHORIZED;GATE_D_OPEN`.
- Certificate hardened per Sol (fail-closed invariants, third association, global windows by
  monotonicity, generator provenance, effective-scale wording): artifact SHA
  `980bbb3933866ebc0228882d9c2f76d0b212e6347f268821a71f5bcab219db3f`, script `d9499821…`.
  Classification `LAYER1_SCALE_FRONTIER_CERTIFIED;…;EFFECTIVE_SCALE_IS_THE_ONLY_FREE_QUANTITY;
  REDUCTION_STRUCTURE_NOT_IDENTIFIED;NO_PHYSICAL_CAUSALITY_CLAIM;GATE_D_OPEN`. Probe execution blocked
  pending canonical protections in the launcher.
- Captured-RMS launcher hardened (literal tag, sanitized Git authority, mirror replay, three-scope
  vacancy, four leases; runs from the tooling worktree); probe takes `--expected-worktree`. Reserved
  tag `greenfield_layer0_captured_rms_replay_20260902T045006596197989Z` (vacant). Classification:
  `RMS_SCHEDULE_PROBE_LAUNCHER_CANONICAL;EXECUTE_REVIEW_PENDING;TPU_UNAUTHORIZED;GATE_D_OPEN`.
- Layer-1 RMS schedule chain V1 (commit `5207c406`): driver/publisher/wrapper/launcher/installer/
  analyzer + package modules; tests 70/2 skipped; certificate
  `gate-d-layer1-rms-schedule-orchestration-certificate.json`. Classification
  `…INSTALL_SOURCE_ACCEPTED;INSTALL_UNAUTHORIZED;TPU_NUMERICAL_EXECUTION_UNAUTHORIZED;GATE_D_OPEN`.

## 2026-09-02 bounded TPU: accepted RMS schedule arm exact (diagnostic archive)

- Tag `gate_d_layer1_rms_schedule_20260902T061305905714981Z`, pin `839c8fb7`: `runner.json` `aa1970dcf0607ee988fed26705b2ba3c20526633adb5fb19985c894764cda281`, `outputs.npz` `a8ef84f16bf80f5bddc6ba1002bbe6b02ed4db831e90eaa2f08d663a58a0f59b`;
  control == DB548 `9b52a04e…`, schedule arm == accepted `9936ee1e…` (0/6,144), schedule vs DB548 one
  mismatch at 2795; both HLO contracts passed; censuses 8/8; remote `diagnostic/` terminal only.
- Classification: `BOUNDED_TPU_LAYER1_RMS_SCHEDULE_ARM_EXACT;DIAGNOSTIC_ARCHIVE_ONLY;
  PUBLICATION_REFUSED_BY_STALE_PUBLISHER_CONSTANT;DECODER_UNPROVEN;GATE_D_OPEN`.
- Decoder flag `rms_accepted_schedule` (default off) staged with HLO contract, CLI/runner plumbing
  (`_ras`), unit/forced-CPU/static tests (rmsnorm 5/5, decoder 1/1, compile 67/67, kernel suites
  32/32). Classification: `RMS_ACCEPTED_SCHEDULE_INTEGRATED_DEFAULT_OFF;REVIEW_PENDING;
  PROTECTED_8K_UNAUTHORIZED;GATE_D_OPEN`.

## 2026-09-02 RMS schedule contract lineage-bound; diagnostic archive adjudicated; chain v3

- `docs/artifacts/gate-d-layer1-rms-schedule-diagnostic-adjudication.json` SHA `e504b795b31690bda4ff2084a7d4316f0d5bd87c53c308f294415eec027e5767`
  (offline replay of the bounded run's diagnostic archive: 18 objects + ledger, local SHAs, remote
  generation/CRC32C, three-scope vacancy of claim objects). Classification
  `BOUNDED_TPU_LAYER1_RMS_SCHEDULE_ARM_EXACT;DIAGNOSTIC_ARCHIVE_ADJUDICATED;GENERATION_BOUND_LEDGER_REPLAYED;
  NO_SUCCESS_NO_DB_NO_PERFORMANCE_NO_GATE_D_CLAIM;DECODER_UNPROVEN;GATE_D_OPEN`.
- Decoder contract `_validate_rms_accepted_schedule_hlo` / `_validate_rms_accepted_schedule_stablehlo`
  bind the full lineage (materialized carry → square → `f32[32,W]->f32[32]` dims={1} add-reduce,
  T(8,128) → 1/W → +1e-05 → rsqrt; StableHLO from the FP32 barrier) and classify the DSA key
  LayerNorm separately; `rms_accepted_schedule` threaded to every reachable RMS call. Tests on the
  archived TPU bytes (17 hostile cases) and forced-CPU decoder (33/33 conforming). Classification:
  `RMS_ACCEPTED_SCHEDULE_CONTRACT_LINEAGE_BOUND;FLAG_COVERS_ALL_RMS;REVIEW_PENDING;
  PROTECTED_8K_UNAUTHORIZED;GATE_D_OPEN`.
- Chain re-pinned v3 (publisher `a53f3fc2…`, driver `5f5fa802…`, wrapper `f52e5334…`, launcher
  `33c0a696…`, installer `83f72f20…`); not installed; no rerun planned.
- Sol BLOCK on 1ad426ed (LayerNorm exemption unbound; merge overwrote rewrite goal.md; 8K runner
  boundary). Fixed: exact LayerNorm binding (`index_key_width`, axis, per-opcode scale, add,
  1e-06, rows) + 21 synthetic hostile tests; merge redone with ba7d1e72 goal.md byte-identical;
  runner rejects untracked files, three-scope vacancy, detached pin worktree under `env -i`.
  Classification unchanged: `REVIEW_PENDING;PROTECTED_8K_UNAUTHORIZED;GATE_D_OPEN`.
- Protected 8K tag `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T092557993413054Z` (pin 7498e31b, `_ras`): failed closed at worker process start
  (`wrong greenfield worktree` from the new detached-source boundary); no TPU init; censuses 8/8;
  diagnostics under its `diagnostic_local/`. Classification:
  `FAILED_CLOSED_BEFORE_TPU;WORKTREE_BINDING_REFUSED_DETACHED_SOURCE;TAG_BURNED;NO_CLAIM;GATE_D_OPEN`.
  Fix: `_worktree_binding` in `compile_short_decoder.py` + worktree/prologue tests.
- Protected 8K tag `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T094725673772375Z` (pin a49e3ba1, `_ras`): 78-layer decoder compiled on TPU; HLO contracts
  refused before execution (78 unbound fused-qkv-a q-a norms; 1,570 accepted-schedule carries refused
  as dead rows); censuses 8/8; TPU HLO/StableHLO/contract archived under the tag (`hlo/`,
  `diagnostic_local/`). Classification: `FAILED_CLOSED_AT_HLO_CONTRACT_BEFORE_EXECUTION;
  TPU_HLO_ARCHIVED;QA_NORM_UNSCHEDULED;CARRIES_REFUSED_AS_DEAD_ROWS;TAG_BURNED;NO_CLAIM;GATE_D_OPEN`.
  Fixes: fused q-a norm on the accepted schedule; lineage-bound carry allowances in the live-tensor
  and fused qkv-a contracts, proven on the archived bytes (1,570 → 0).
- Protected 8K tag `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T113001516858996Z` (pin 978dc3a9, `_ras`): decoder compiled on TPU; RMS contract 313/313
  conforming with 313 barriers (first full-decoder proof that every norm carries the accepted
  schedule); fused qkv-a passed; live-tensor classifier refused 156 `bf16[32,2048]` pad/convert
  records inside the q-a carry producers; censuses 8/8; archived under the tag. Classification:
  `FAILED_CLOSED_AT_HLO_CONTRACT_BEFORE_EXECUTION;ALL_313_RMS_SCHEDULED_ON_TPU;QA_CARRY_BF16_PAD_REFUSED;
  TAG_BURNED;NO_CLAIM;GATE_D_OPEN`. Fix: BF16 admission inside the producer slice only.
- Protected 8K tag `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T121803168061440Z` (pin 66d32efa, `_ras`, sanitized launch): **decode-step and DSA-observer
  programs passed all contracts on TPU** (RMS 313/313, 313 barriers, live-tensor and fused qkv-a clean
  with lineage-bound carries); refused at the prefill contract (flag not forwarded); censuses 8/8;
  archived. Classification: `DECODE_STEP_CONTRACTS_PASSED_ON_TPU;PREFILL_CONTRACT_FLAG_NOT_FORWARDED;
  FAILED_CLOSED_BEFORE_EXECUTION;TAG_BURNED;NO_CLAIM;GATE_D_OPEN`. Fix: forward the flag in
  `validate_teacher_forced_prefill_hlo`.
- Protected 8K tag `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T130949040481337Z` (pin fdddeadf, `_ras`, sanitized launch): **first full execution with the
  accepted schedule** — all three programs' contracts passed on TPU; prefill token exact; decode token
  at 8155 exact (101252, margin 6.5); exact-DSA refused: event 0 one swap (4879→2540), event 1: 9,
  event 2: 12, … (21 events, 41,502 order mismatches); censuses 8/8; observer npz `f32ae99a…`.
  Classification: `EXECUTED_ON_TPU;TOKEN_EXACT_AT_8155;DSA_EVENT0_REGRESSED_ONE_SWAP;
  QA_NORM_SCHEDULE_INFERENCE_REFUTED;TAG_BURNED;NO_CLAIM;GATE_D_OPEN`. Decision: fused q-a norm back to
  the sharded legacy form (own contract kind); hidden-width and kv-a norms keep the accepted schedule.
- Protected 8K tag `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T154017247157875Z` (pin 906f3810): failed closed at the eight-host sync — transient GitHub
  fetch SSH fault on worker 6 (7/8 SYNC_OK); no TPU; censuses 8/8. Classification:
  `FAILED_CLOSED_AT_SYNC;TRANSIENT_ORIGIN_FETCH_FAULT_WORKER6;TAG_BURNED;NO_CLAIM;GATE_D_OPEN`.
- Protected 8K tag `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T155131089617216Z` (pin 906f3810, `_ras`, sharded q-a restored): executed; prefill + decode
  token exact; **event 0 exact (max_abs 0.0 over 2,048 positions)**; event 1: 7 set mismatches
  (baseline 6, all-norm run 9); censuses 8/8; observer `eae3ca24…`. Classification:
  `EXECUTED_ON_TPU;TOKEN_EXACT_AT_8155;DSA_EVENT0_EXACT_RESTORED;EVENT1_7_MISMATCHES;
  KV_A_SCHEDULE_INFERENCE_SUSPECT;TAG_BURNED;NO_CLAIM;GATE_D_OPEN`.
- Contract census now three kinds (157 accepted hidden-width / 78 sharded q-a / 78 kv-a row norms),
  mandatory when enabled, proven on three archived TPU modules (baseline default, first `_ras`, second
  `_ras`) and synthetic attacks. Flag scope: hidden-width norms only. Classification:
  `RMS_SCHEDULE_SCOPE_HIDDEN_WIDTH_ONLY;CENSUS_BOUND_THREE_KINDS;REVIEW_PENDING;GATE_D_OPEN`.
- Protected 8K tag `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T211600121108599Z` (pin 086d459a): failed closed at the eight-host sync — recurring GitHub fetch
  SSH fault on four or more workers; no TPU; censuses 8/8. Classification:
  `FAILED_CLOSED_AT_SYNC;TRANSIENT_ORIGIN_FETCH_FAULT_RECURRING;TAG_BURNED;NO_CLAIM;GATE_D_OPEN`.
- Protected 8K tag `greenfield_short_decoder_compile_pp8_8k_pallas_feature_linear_ot256_downf32_token_splitres_prefill_keyfix_queryexact_headkeyexact_scoredefault_mainrope_ras_pregatheredb512_strategynd_o_densefinalconv_oracle_dsa_metaparent_trace2_20260902T213510823966642Z` (pin 086d459a, `_ras`, hidden-width schedule only): executed; prefill +
  decode token exact; event 0 exact; **event 1 = 7 mismatches, identical set to the hidden+kv-a run**;
  censuses 8/8; observer `bf3ff47e…`. Classification: `EXECUTED_ON_TPU;TOKEN_EXACT_AT_8155;
  DSA_EVENT0_EXACT;EVENT1_7_MISMATCHES_SAME_SET;KV_A_SCHEDULE_IRRELEVANT;
  HIDDEN_WIDTH_SCHEDULE_DOES_NOT_CLOSE_EVENT1;TAG_BURNED;NO_CLAIM;GATE_D_OPEN`.
