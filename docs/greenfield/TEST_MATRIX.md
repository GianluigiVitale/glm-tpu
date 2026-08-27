# Greenfield test and gate matrix

| Level/gate | Current evidence | Status |
|---|---|---|
| L0 immutable geometry/plan | Canonical hashes/round trips/refusals | Pass |
| L0 synthetic topology/groups | Coordinate/order adversaries; PP8/PP16 physical rings | Pass |
| L1 forced CPU collectives/runtime | Seven operation/control variants; sizes 2/4/8/32; deterministic checksums | Pass (mechanism) |
| L2 optimized-HLO contract | Exact physical groups/counts/pairs/shapes, async op handling, tuple fusion, nonidentity device assignment | Pass |
| Gate A physical inventory/groups | Protected topology DB 405 and approved archive | Pass |
| Gate A dependent collective floor | DB 406–414: dominant/bf16/f32/metadata payloads; exact chain75 HLO and full distributions | Pass (mechanism) |
| Reduce-scatter support | Three protected diagnostics prove TPU-v4 XLA rewrites required small decode RS to AR | Evidence-rejected for this benchmark shape |
| Gate A PP8 transport | DB 415 exact 8-hop HLO/distributions; DB 416 8-file/64-core trace | Pass (mechanism) |
| Gate A PP16 transport | DB 415 exact 16-hop HLO/distributions; DB 416 8-file/64-core trace | Pass (mechanism) |
| Gate A inactive-stage/no-host-dispatch proof | One global device program; no model compute; HLO/trace only p2p permutes | Pass (skeleton) |
| Gate A overall | Topology, collective floor, PP8/PP16 transport, trace, provenance, cleanup | Pass |
| L0 MoE numerical contract | Block dequant, sigmoid/noaux_tc bias semantics, ties, normalization | Pass |
| L1 four-device PP8 MoE reference | Distributed routes + all-top-4-on-one-chip; bounded output equivalence | Pass (synthetic) |
| L2 MoE combine HLO | DB 417/418: one `bf16[2,1,6144]` exact four-/two-rank AR; no other collective or dead `[32,6144]` tensor | Pass on TPU (PP8/PP16) |
| One-layer pack format | Exact source leaf set, ownership, byte/hash round trip, corruption refusals | Pass on tiny fixture and real artifact |
| Real layer-3 pack | 1,544 leaves / 9,706,940,416 unique bytes / only shards 38–40; four final PP8 files; manifest `68ef8201...f938`; remote `SUCCESS` | Pass (layout mechanism; not Gate B) |
| Real WS32 layer-3 pack/load | 32 exact final owners / 9,971,249,152 payload bytes / 311,601,536 per chip; manifest `4bf8679d...1f40`; direct slots 0/31; 40-object CRC/generation-sealed archive | Pass (layout/direct-load mechanism; no TPU/HLO/Gate D) |
| Real WS32 layer-3 TPU numerical/HBM | Reference tag `greenfield_ws32_real_layer_numerical_20260815T090957477700205Z` passes correctness/topology but is execution-rejected at 516.582/1160.886 ms p50. Selected Pallas tag `greenfield_ws32_pallas_real_layer_numerical_20260815T163210555760825Z` has 15/15 live inputs, 27 Pallas calls, nine feature-4 plus one expert-8 reduction, bounded max error 0.03125/0.03125, 1.256975/2.303685 ms p50, peak 314,218,496 bytes/chip, 51-object SUCCESS-last archive and 8/8 cleanup. | Pallas body selected for full decoder; bounded layer only, no Gate-D/token-performance claim |
| Real layer-3 oracle | Raw-source PyTorch; 104 pinned tensors; normal all-slot routes; all-eight-on-slot-2 adversary; manifest `c63ffa19...ebff`; remote `SUCCESS` | Pass (correctness artifact) |
| Exact real PP8 local MoE layer | DB 417: exact routes, bounded normal/concentrated tensors, direct load, HBM, wall, XPlane, DB/archive/cleanup | Pass |
| Real layer-3 PP16 pack | Two final-owner files; 128 experts + shared width 1024/chip; manifest `38573723...e454`; remote `SUCCESS` | Pass (layout mechanism; not Gate B) |
| Exact real PP16 local MoE layer | DB 418: exact routes, bounded tensors, two-rank HLO, HBM, wall, XPlane, DB/archive/cleanup | Pass |
| Complete PP8 plan/layout/pack | 118,629 source leaves; 32 base + 4 MTP owners; exact byte/hash reconciliation; manifest `08694931...78f1` | Pass |
| Complete PP16 plan/layout | 118,629 leaves; 16 exact physical stages; 32 base + 2 MTP owners; 757,149,950,848 planned bytes; execution/plan/layout `079cefe6...794c3` / `3c3ea07b...ed16` / `f97de2d8...b15f9`; nine-object SUCCESS-last same-region archive | Plan pass; full pack/direct load/HBM and Gate B missing |
| Complete PP8 direct load | DB 420: all 32 chips / 8 stages, 750,122,559,744 bytes, exact identities, device round trip, peak weights-only HBM, archive/cleanup | Pass |
| Gate B base layout | Complete plan-aware checkpoint manifest/packer/direct loader and fail-closed corruption handling | Pass |
| Gate B fused qkv-a derivative | Pack `7d5dfb9` plus DB504: 32 final-owner files / 834,369,271,808 bytes; exact device round trip; 78 fused convolutions; zero runtime reshards/concat/dequantization | Pass; Gate B re-closed |
| Gate C reference kernels | Dense/norm/RoPE, FP32 DSA/top-k, compact IndexShare, stage-local KV, sparse MLA and LSE merge; full-width FP32/BF16 legacy comparisons exact | Pass on CPU/reference |
| Real Gate C oracle | 31 raw layer-2/3 tensors; 2,304-token full scorer; 2,048 exact positions; 8,192-byte IndexShare; dense/sparse-attention outputs; manifest `54262529...4a9f` | Pass (correctness artifact) |
| Bounded Gate C final-owner pack | Exact protected-layout subset; 31 raw hashes; 413,810,816 source bytes; four equal PP8 owners; manifest `3c5c48da...2a8a`; local/remote integrity | Pass (layout mechanism) |
| Real Gate C PP8 layers | DB 421: direct load; dense/full-DSA/IndexShare bounded outputs; exact device-score top-k/ties and state/cache; exact local HLO/XPlane/HBM; DB/archive/cleanup | Pass (correctness/mechanism) |
| Gate C cross-framework diagnostic | PyTorch CPU vs TPU FP32 score drift swaps 2/2,048 cutoff members; retained explicitly, not runtime state or tolerance-relaxed | Raw position identity does not pass |
| Gate C overall | MoE DB 417/418 plus dense/full-DSA/IndexShare DB 421 match the documented device arithmetic contract; no decoder/tok-s claim | Pass under documented device-score contract |
| Gate D runtime artifact | Complete 32-file / 834,178,632,448-byte PP8 runtime derivative; exact DB420 identity and 26,068,042,432 weight bytes/chip | Pass (layout/load mechanism) |
| Gate D reference-body diagnostic | Real 78-layer 2K load and compile; 25.46 GiB/chip; 219AG/294AR/16CP; 312 logical AR results; 2.7M bundles/580 overlays; staged execution projected 25–30 minutes/body | Evidence-rejected; reference graph cannot be the production engine |
| Pallas FP8 block matmul | DB 422: production M8/K6144/N2048, raw-U8/128x128 VMEM dequant, BF16 MXU/FP32 accumulation, exact fallback, one custom call/no overlay, wall/HBM/DB/archive/cleanup | Pass on TPU v4 |
| Pallas FP8 paired gate/up | DB 423: two raw matrices/two exact outputs, one custom call/no overlay, 0.815435 ms p50, wall/HBM/DB/archive/cleanup; same-weight M8 mechanism only | Pass on TPU v4; selected-expert GMM pending |
| Pallas FP8 selected gate/up | DB 429/430: device dynamic owned-route compaction; exact normal-two/concentrated-eight outputs/order/zeros; one raw-U8 TPU kernel + bounded metadata/no overlay; 1.341385/4.496970 ms p50. DB 431--435 merged-stream, aligned compact-scale, and route-parallel challengers are exact but performance-rejected/null. | Correctness/route-proportional mechanism pass; DB 429/430 selected for activation/down composition |
| Pallas FP8 selected SwiGLU/down | DB 436/437: exact BF16 activation and distinct raw-U8 selected down matrices; normal-two/concentrated-eight order/zeros; one Pallas kernel + exact compaction/scale/restore metadata/no overlay; 0.773045/2.421714 ms p50 | Correctness/route-proportional mechanism pass |
| Final-layout Pallas one-layer pack/load | Manifest `3da63bd9...e427`; exact source transforms; four final-owner raw files; 56 direct transfers; zero dequant/concat/runtime transpose | Pass |
| Exact raw-FP8 Pallas PP8 MoE layer | DB 438 baseline plus DB 439 fused-routed: exact routes, bounded normal/concentrated outputs, three raw-U8 kernels, one local combine, no overlay, HBM/wall/XPlane/DB/archive/cleanup; 3.121940/7.063344 ms p50. DB 440 fully fused shared too but regressed 0.8–1.5%. | Correctness/layout/locality pass; DB 439 selected, latency still rejected |
| Gate D feature-body diagnostic | Eight-host real 78-layer/2K execution at `a8194cd`; exact feature HLO/local groups/metadata/direct load; 58,804.040 ms p50, 26.144 GB peak HBM/chip; outer finalizer schema failure means no DB/SUCCESS | Evidence-rejected; non-MoE reference projection/dequant path requires XPlane attribution and Pallas replacement |
| Gate D feature-body XPlane | DB 442: one clean wall sample then 8 files/64 cores/2 trace steps; 58,804.003 ms wall; dequant gather 7,317.698 ms/core x 8 serial stages = 58,541.584 ms; exact HLO/HBM/DB/archive/cleanup | Attribution pass; performance rejected; compact permute duration is pipeline wait, not bandwidth |
| Pallas DSA scorer | DB 443: production one-row 256K/LP4 shard; one kernel/no per-head overlay/dead rows/collectives; score max error 2.861e-6; exact 2,048 positions/order; 0.326595/0.350320 ms p50/p99 | Standalone Section 7.2 item 5 pass; layer integration pending |
| Pallas exact DSA top-k | DB 445: exact local 65,536→2,048 plus permuted four-owner merge; TPU/host scores, positions, counts, ties, sentinels exact; 6/2 calls and no XLA sort/top-k/dead rows/collectives; local/merge p50 1.364405/0.337671 ms. DB 444 reduction path exact but rejected at 59.979532/4.495320 ms. | Standalone Section 7.2 item 6 pass; layer integration pending |
| Gate D fused qkv-a implementation | `0082bac`/DB503 plus DB504: exact isolated q-a/kv-a; protected full final-layout load; 78 physical one-row convolutions; no old calls, forbidden shape, reshard, concat or dequantization | Pass (mechanism) |
| Gate D protected 2K decoder | DB 484: exact token/DSA/state/cache/local HLO, peak HBM 26,245,004,800 bytes/chip, 244.091151 ms p50 and 4.096830 tok/s | Pass at 2K; below Gate E |
| Gate D protected 8K decoder | Protected `0312cf5` integration localizes the first recurrent mismatch to layer-0 output entering layer 1. Precision, uniform virtual-TP32 output association and monolithic-attention schedule families are protected negative evidence. DB532 seals exact M32 lowering and the sole live row; DB533 uniquely recovers all 6,144 live-row association columns plus the model-to-physical permutation. | Missing; finish/test/audit the exact two-arm row-zero discriminator, then run it once before complete 8K |
| WS32 full runtime pack | Reviewed pack/finalize/direct-load implementation covers 117,060 source tensors, 2,310 final tensors/slot and 24,567,890,256 payload bytes/slot with protected SUCCESS binding. | Local implementation/audit pass; sealed 32-slot artifact missing |
| WS32 protected short decoder | Reviewed/pushed `a908c36`; four executables/eight pins, exact sealed 2K/8K token+DSA inputs, raw cache/DSA evidence, one-row state, HBM/wall/XPlane, DB/archive/cleanup; 70/70 local tests and immutable Sol approval. | Readiness pass only; checkpoint, HLO acquisition and protected 2K/8K executions missing |
| Gate D prompt index-cache/scorer discriminator | DB505--518 make the complete 8,155-row cache exact; DB521--527 make current query/head/key exact. DB528 rejects page geometry. DB529's same-shape TPU default-precision arm matches the accepted logical and selected scores/set/order/ties exactly, with pinned HLO/DB/archive/cleanup. | Pass; do not repeat bounded scorer variants |
| Gates E–H | Await Gate D | Missing |

Last complete CPU-only suite before the production integration: 441 passed / 1 expected skip
(2026-08-08). Latest fused-qkv arithmetic/checkpoint/runtime/HLO/stage suite: 57 passed in
135.25 seconds at `0082bac`; Python/Bash/ShellCheck/JSON/diff checks pass.
Latest prompt-cache/association/reuse focused suite: 34 passed at `dcb7096`, plus 9 wrapper-regression tests at `31a23b8`. Two broader CPU suite attempts
were manually stopped after unrelated existing forced-JAX tests stopped advancing and are not
suite-pass claims.
Latest exact scorer-precision focused suite: 39 passed at the DB528-successor batch;
Python/Bash/ShellCheck/diff checks pass. This is readiness only until protected TPU evidence.
Latest default-off production scorer-precision integration suite: 65 passed in 177.11 seconds with
`JAX_PLATFORMS=cpu`; Python/Bash/diff checks pass. Full protected 8K evidence remains pending.
Latest isolated residual-discriminator affected suite: 68 passed in 213.61 seconds. It traces all
four separately compiled real stage-0 arms under normal and exact query/head-key signatures,
validates per-arm single-row HLO/kernel/reduction contracts, and rejects tuple-contaminated arms.
Python compileall, Bash, ShellCheck, JSON and diff checks pass; the protected TPU discriminator
at `12315aa` completed and rejected all four precision arms. Latest virtual-TP32 successor affected
suite: 67 passed in 198.84 seconds with explicit CPU backend. It validates the two BF16 eight-way
trees, four isolated program names, subshard kernel/collective HLO contracts and all existing
stage-local/decoder behavior. Python compileall, Bash, ShellCheck and diff checks pass; its protected
TPU discriminator completed at `e19833a` and rejected all four uniform trees under valid contracts.
The old one-row StrategyND prototype remains rejected. DB532 authorized the corrected repeated-M32
capture, and protected DB533 now passes it: every physical row has unique 6,144/6,144 coverage,
row zero is explicitly recovered, and analysis/summary/SUCCESS plus DB/archive/8-host cleanup are
sealed. This is association evidence only. The current default-off integration applies row zero to
both layer-0 projections through two LP4 gathers. Its affected CPU/forced-device suite passes
`79/79`; direct current-code DB533 replay has `0/196,608` mismatches. Python, Bash, ShellCheck,
JSON, all 15 embedded-Python blocks and diff checks pass, and the read-only exact-mode wrapper
preflight reaches the intended dirty-worktree refusal after validating DB533. The corrected HLO
contract globally pins exactly two separately scoped gathers, while a same-reducer device canary
pins all 32 sealed trials/lanes and 82 BF16-round barriers. The first protected attempt proves that
canary exact on TPU but refuses its real `bf16[1,8,1,6144]` folded operand shape before model
execution; cleanup is 8/8. The bounded guard fix passes focused tests and the preserved real HLO.
No TPU boundary result exists until its new-diff-only Sol audit and protected retry.
DB 417/418 prove decoded-overlay sparse-layer oracles; DB 420 proves complete checkpoint
integrity/direct loading; DB 421 proves real PP8 dense/full-DSA/IndexShare layers; DB 439 proves the
fused-routed deployable raw-FP8 PP8 MoE layer but misses its latency budget. None proves full-model correctness,
decoder HBM, token latency, or wall throughput.
