# Long-prefill phase residency — 2026-09-12

Status: CPU-tested, opt-in host components; NOT a protected long-model run or
measured TPU memory fit. Base pin `e602465c73cceb410ff1f7134bce8357bfbc41b7`.
Authority remains §25/§26: fix capacity and deliver; no throughput/math search.

## Concrete defect and retained evidence

The current `run_short_decoder_ws32.py` loads the StrategyND dense overlay and
materializes the entire exact-DSA decode tree before batched prefill. Prefill
instead uses the raw base view plus only the completed WK repair matrices.
The original worker retains `weights`, `loaded_dense_overlay`, and
`exact_dsa_weights` alongside those inputs. Its original loader and materializer
are not corrupt; their overlapping lifetimes are unnecessary for this phase.

DB610's original all-live census records 1,127,473,664 B/chip outside prefill
arguments. Those old records contain allocation sizes/groups, not semantic leaf
names: the apparent size correspondence is NOT an exact owner attribution.
No old census is retroactively relabelled. A new actual all-live census must
decide whether phase separation removes the observed excess.

DB615 is the actual E0 compiler anchor: scratch 3,372,240,896 B/chip, alias
3,630,978,560 B, code 101,054,464 B. Single graph plus 1 GiB reserve leaves
148,082,688 B before additional live allocations. Its 78 late captures are fixed.
Do not repeat its unchanged compilation or infer runtime fit from compiler fit.

## Implementation and exact phase boundary

`scripts/greenfield/ws32_phase_weights.py` introduces an explicit `PhaseWeights`
owner; it does not load, authorize, launch, or certify checkpoint bytes.

| Phase | Retained weights | Not created / released |
|---|---|---|
| Base load | Original verified raw checkpoint, same array objects | Dense overlay and exact decode tree not loaded/materialized |
| Repair preparation | Raw checkpoint + each producer's completed FP32 WK | Each temporary BF16 WK released after its separate promotion |
| Batched prefill | Raw checkpoint + 21 completed WK | No packed exact QKV/query aliases/head materialization or dense overlay |
| Decode handoff | Shared raw leaves + original loader-verified dense overlay | Old raw dense roots and prefill WK released |
| Decode preparation | Original full exact materializer outputs | Prefill executables/arguments must already be released |

WK preparation reuses `probe_ws32_prefill_layer.build_wk_programs`, including its
feature4 gather, completed BF16 decode and separate FP32 promotion. Compile two
programs once; dispatch each producer's own inputs through the existing admitted,
voted, memory-budgeted call path (21 × 2 = 42 calls). Local post-call checks and
root updates also use the existing voted phase, before any successor collective.
Do not add a new dispatcher.
The same primitives and selected full-vs-standalone materializer evidence exist
in DB612; no new numerical archaeology or full exact-tree capture is necessary.

`begin_decode` rebinds the existing decoder schema without tensor copies. For
the production geometry, exactly 18 raw dense leaves are replaced by 12 overlay
leaves; all other selected base leaves retain object identity. Missing/extra
overlay names refuse before root mutation. The overlay MUST come from the
unchanged checksum-verifying loader. This binder is not a replacement verifier.

The parent must drop its loader dictionaries, old argument tuples and borrowed
roots. This object cannot collect another caller's references. Its named roots
are inputs to, not a substitute for, the all-live census. Failed WK preparation
is terminal, not a partial retry. `begin_decode` is only authorized by the caller
after healthy, complete prefill. It alone does not prove request completion.

## Consumed-state memory adapter

`scripts/greenfield/ws32_owned_prefill_memory.py` is a distinct contract. The
existing host block loop accepts it only through the explicit
`state_ownership_contract`; the default remains the old non-donating budget.

- Read actual compiled argument metadata: exactly all leaves of argument 2 are
  donated, never IDs/count/weights/WK/RoPE; state shapes/dtypes match live inputs.
- Reuse the original physical-pointer/all-live census and allocator counters.
  State allocations cannot overlap nonstate inputs; unknown pointer ownership
  refuses this profile rather than assuming donation is safe.
- Subtract only the active executable's actual alias bytes from its output
  allocation. Preserve live baseline, scratch, all resident code, prior peak and
  the full required reserve. Alias bytes cannot exceed accounted state/output.
- E0 roles may share one compiled object; count its code once only for actual
  Python object identity, never from equal HLO, sizes or cached-wrapper guesses.
  Two distinct compiled objects are conservatively counted separately.
- Replay the budget arithmetic from the record; a declared fit is not a verdict.
  The protected sealer still must bind actual graph/profile/owner evidence.

This interface admits no companion executable inventory. Clear WK/prefill code
at the correct boundary; unlisted code cannot be omitted to fit. The unchanged
old budget still rejects all nonzero alias reports. Neither budget substitutes
for measured post-execution peak HBM on all 32 chips.

## Tests / limitations

Final focused run: **106 passed in 7.71 s**, `JAX_PLATFORMS=cpu`, including:

- Actual production 2,310-leaf name/tree ownership; weak references prove raw
  dense and repair arrays disappear after transition while shared weights live.
  Tiny placeholder base values test ownership, NOT checkpoint contents.
- One real reduced-geometry 32-device CPU WK computation equals the original
  full exact materializer bit-for-bit; production 21-producer shape/sharding
  preparation is abstract and allocates no production weights.
- Actual CPU compiled donation metadata, shared vs distinct executable objects,
  nonstate alias refusal, and the real host block loop consuming cache handles.
  CPU allocator counters are explicit fixtures, NOT measured TPU HBM.
- Prior-peak/reserve/tampered-budget failures and historical adapter/memory tests.

Initial fixture failed before computation because `first_dense_layers=3` exceeded
the reduced one-layer geometry; corrected to 1 without changing model code or
test bounds. Self-review caught unvoted local checks after completed WK calls;
existing fleet phases now gate them. Peer-only refusal tests verify no successor
decode/promote dispatch. Current-chat self-review only, not independent review.
Receipt: `../artifacts/prefill-phase-residency-local-20260912.json`.

## Next — finish integration, not another research/compile campaign

1. Connect this owner to the protected long worker: raw-only verified load,
   two WK jobs/42 protected calls, clear WK code, prefill, clear prefill borrowers
   and code, load verified overlay, bind decode, original exact materializer.
   Include phase preparation and decode preparation in honest request timings.
2. Use the distinct owned budget for E0 and bind DB615's actual graph evidence.
   Reuse original 128K non-donating graphs where their all-live budget fits;
   do not reacquire unchanged 128K or imply it needs E0's cache variant.
3. Update long worker and sealer/profile together, bind new phase/HLO provenance,
   preserve originals before refusal, then fresh launch-space/fleet/memory guards
   and the real four 128K depths followed by full 256K E0. No full model is
   authorized solely by these host tests. No fabricated complete-model fit.

No TPU workflow, weight copy, cloud eviction, policy change or infrastructure
action in this step. Local free space remains below the 6 GiB launch floor.
Current performance remains DB610 short2K: 62.761 prefill tok/s, 7.660 decode
wall tok/s; all long tests, HF-card benchmarks, request/resume/TTFT remain open.
