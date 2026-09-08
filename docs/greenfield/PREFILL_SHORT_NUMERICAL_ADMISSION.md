# First protected batched-prefill numerical admission

2026-09-08, starting pin `631eb5bc2affe1623bd09244629bb2f164594050`.
Worker/launcher/sealer integration is now implemented for the fixed2K profile.
Current-diff review, clean persistence and fresh preflights remain launch conditions.
No new model numerical execution, runtime peak, speedup or long-context proof.

## Fixed first workload

`validation/ws32_prefill_admission.py` registers only
`ws32_b17_b11_2k_cap8192_v1`: the sealed 2034-token prompt, main17/tail11,
capacity8192. All seven original graph pairs are bound to the SHA-verified
`../artifacts/prefill-batched-seven-graph-acquisition-20260908.json`, acquisition
pin `133fe71fff18c6514ab76ca89d432a90c03b01dd`. Runtime/kernel/sharding/model
source must remain unchanged from that pin. Worker/enforcement changes need
their own reviewed clean published pin. No unchanged acquisition is needed.

The first numerical experiment has a fixed300s request-prefill ceiling, NOT a
performance target. Historical serial2K is approximately237s; this ceiling
bounds diagnostic cost, not acceptable final latency. Cold load/compile and
validation are outside that ceiling and must be disclosed separately.
Every chip must retain1GiB both in the conservative predispatch budget and
actual observed peak. This reserve exceeds twice the historical approximately
0.4GiB compiled-versus-observed discrepancy. It does not prove long-capacity fit.
Final128K/256K prefill/TTFT targets and larger-row weight reuse remain open.

## Exact graph identity without recompiling for source line numbers

StableHLO remains byte-exact. Optimized HLO contains two worker FileLocation
entries (`main` and `<module>`), which move when reviewed worker wiring moves.
`validation/ws32_hlo_worker_locations.py` normalizes ONLY their eight integer
line/end-line/column/end-column values. Every other byte remains in a fixed,
per-graph equivalence hash: model source coordinates, stack-frame IDs/parents,
function/file tables, instructions, layouts and backend configuration included.
Wrong frame structure or any non-coordinate drift refuses. Original and actual
raw optimized hashes and actual worker coordinates are retained, not stripped.

Worker graph writers first rederive the existing structural inspection from
actual text. Only the fixed identity plus a report whose sole refusal is
UNREGISTERED can receive bounded profile authorization. The helper is not an
untrusted-JSON verifier. The sealer must independently repeat this operation.
Graph authorization explicitly does not establish numerical or memory success.

## Worker execution and evidence

Before the staged adapter call, worker clears `state`, `repaired_buffer` and
`batched_state` compile aliases and collects unreachable objects. The adapter
creates fresh state and accounts for actual all-JAX-live physical allocations,
both prefill compiled programs, active outputs/scratch and the fixed reserve.
Materialize/promote programs are already released; observer/decode compile later.
After prefill, raw-only weights/WK and prefill programs are released before
observer compilation. No second checkpoint or full host tensor copy is created.

Receipt/profile/model-source/compiled-memory preflight failures become local
error values, are published and fleet-voted BEFORE adapter dispatch. A peer-only
refusal must stop every host. The memory-admission record is atomically published
before its fleet vote and before the first model dispatch. Completion validates
the adapter record, actual first token and four owner-keyed TPU memory counters.
Failure publication preserves the original exception, even if publishing fails.

Append-only `batched_prefill_{preflight,memory,complete,failure}.rankN.json`
records carry code, launcher/JAX rank, host, profile/plan and prompt-ID hash.
The existing failure uploader preserves them under the same-region run's
`diagnostic_local/<tag>/` namespace. Partial records cannot seal success.

## Evidence and remaining integration

- 153 CPU tests passed in32.21s: worker preflight/peer refusal, memory publication
 failure, actual adapter accounting, phase uploads, fixed-profile refusals,
 debug-coordinate identity and mutation tests. No skips.
- Both original full B17/B11 integrated graph replays passed in121.37s;
 22 duplicate narrow tests deselected. All seven saved graph identity tests pass.
- Independent current-diff reviewer found no remaining P0-P2 for CPU persistence;
 pending replay condition is now satisfied. Deployment is still blocked below.

Before the single bounded2K numerical run:

1. Add the same fixed profile/source/pin checks BEFORE costly load/compile;
   propagate profile/reserve/budget through launcher and sealer.
2. Integrate mode-specific schema, actual raw-HLO replay and execution-record
   validation in the sealer, without coercing batched timing to serial fields.
3. Bind census, prefill and final execution peaks to authenticated captures and
   exactly32 unique physical owners; match actual compiled analyses and reserve.
4. Preserve pre-load/compile failures; require final frontier2034 and actual first
   generated token, own unchanged§21 tokens/DSA/state/cache, trace and wall.
5. Review current delta, persist and run fresh ownership/cost preflights under
   both leases. No TPU lifecycle action, serial long run or cleared layer rerun.

After short correctness: measured phase costs/targets and row-reuse scaling,
efficient four-depth128K and full256K proofs, DB/archive/authenticated cleanup.

## Integration update after379fd155 — 2026-09-08

The following parts of the preceding checklist are now implemented, still
behind the hard-disabled numerical entry:

- The worker checks the actual sealed prompt length, fixed profile/flags/graph
  pins and unchanged model source BEFORE runtime initialization and weight load.
- Keyed final execution counters are captured in addition to postprefill counters.
  `ws32_prefill_fleet_memory.py` joins all three boundaries to authenticated
  capture process/device IDs and exact physical mesh slots, with32 unique owners.
  Every budget is recomputed; analyses match both acquired and worker reports.
  Reserve is fixed1GiB and device limit33,014,398,976B. Current usage may fall as
  buffers are freed, but lifetime peak cannot decrease across the three boundaries.
  The resulting peak includes load/compile/trace history, not isolated prefill.
- Sealer batched-only schema and memory consumption are wired. Serial schemas,
  anonymous historical telemetry and default serial accounting remain unchanged.
- Sealer graph replay independently uses the actual raw texts and original pins.
  Each unique graph/SHA pair is parsed once; ALL ranks' raw hashes and complete
  reports still must match. All7 original graphs pass the actual sealer path
  (7PASS185.35s,22 duplicate narrow tests deselected, no skips).
- Own-mode execution accounting reuses the actual adapter validator, requires
  fixed2034frontier/300s ceiling and binds first token to observed continuation.
  Request-prefill wall is not coerced to serial timing or labelled deliveredTTFT.

Still required before launch: propagate fixed profile/reserve/budget and original
acquisition identity through the wrapper/common runner+summary/DB basis; exercise
the composed sealer path; preserve early load/compile failure diagnostics; remove
entry guards ONLY after this integration is reviewed and tested. Then clean pin,
cost/fleet ownership preflights, one own2K numerical run and complete protected
evidence. These integration updates are not a numerical pass or speedup.

## Launch integration after19efa7e0 — 2026-09-08

The fixed-profile worker, wrapper and sealer entries are now wired. No broader
batched/long workload is enabled. `scripts/greenfield/ws32_batched_launch.py`
derives the numerical environment from the existing acquisition recipe, adding
the fixed profile and all14 original HLO pins. Its read-only CPU preflight runs
before cloud, leases or checkpoint reads; the worker repeats actual prompt/source
checks before runtime/load. All controller evidence code still runs from its own
immutable sealing checkout.

`short_numerical_identity()` supplies worker/common-summary/DB fields: mode,
profile, plan, acquisition pin/receipt SHA,1GiB reserve and300s diagnostic ceiling.
DB publication and rollback share that identity and a distinct batched item ID;
historical serial fields remain absent. Summary retains actual/original graph
source-location identity and explicitly states no prefill speedup/deliveredTTFT.
Its performance flag continues to describe protected DECODE measurements only.

The fsynced numerical journal begins before runtime initialization, records
checkpoint verification/load boundaries and all7 compile hooks. Original failures
remain in runner logs; journal/phase records upload through the existing bounded
diagnostic namespace. Fresh batched materialization accepts that same-tag channel
but still requires every primary NPZ, trace, runner and graph. Generation/CRC/SHA
binding and256MiB diagnostic cap are unchanged; partial evidence cannot seal.

Final focused regression217PASS6.33s,35unchanged historical adjudication tests
deselected, no skips. Includes actual shell flag expansion/preflight, early journal
and uploader, fake-GCS diagnostic completeness/limits, batched+serial DB roundtrip,
and composed8-rank sealer mode/schema/raw-file/accounting/32-owner-memory path.
The composed fixture substitutes expensive HLO/math and external provenance/trace
boundaries; it is controller wiring proof, NOT new hardware/numerical evidence.
All7 actual graph replays remain the previously completed evidence, not rerun.

Next: final independent review and clean published/mirrored pin; fresh region,
live-storage+6GB reserve and normal/authenticated-root fleet checks under both
leases; one protected2K numerical discriminator. Worker hard limit45min includes
cold load/compile contingency and up to300s request-prefill; upload/cleanup are
separately observed. No checkpoint copy/repack. A failure preserves original
evidence and leads to local diagnosis, not an unchanged retry. Actual execution
HBM, own§21 tokens/DSA/cache and all efficient long-context proofs remain open.

Final independent current-diff review: no remainingP0-P2; conditional approval
for clean persistence and that one fixed2K test after fresh preflights. Updated
reuse registry4PASS1.93s. Cost anchor is the completed acquisition03:47:40–04:23:07
(~35.5min including preflights/upload/cleanup); worker launch03:49:55–04:22:58
(~33min). New numerical prefill adds at most300s before cost refusal; planning
roughly35–45min plus separately observed sealing/archive, not a completion ETA.

First launch0b717a5b/08:58:14 tag refused at09:00:28 before sync/load/compile:
local3.17GBfree below unchanged4GiB evidence floor, while bucket1.999TB and8/8idle
passed. Root-observed unused Sep5 pytest fixtures30/43 were reviewed/reclaimed;
free space now5.41GB. Original remote refusal/censuses retained (HANDOFF lists
generations/CRC). Same implementation is eligible for one corrected-state launch
after persistence and fresh preflights, not a numerical retry or relaxed floor.

## First numerical outcome — 2026-09-08 09:46Z, seal recovery required

Run9364bc3e/tag`…c17_hrope_bp1_20260908T090441883274696Z` completed on all8workers:
20/20tokens,14DSA steps, cache checks and healthy2034frontier. Request-prefill
fleetmax102.202932s; no deliveredTTFT measurement. Independent sealer refused
report equality: atomic_commit_proof.rollback_inputs keys are integers in the
replay but strings in archived JSON. Actual main replay JSON-roundtrips exactly
to original; this is not HLO drift or model failure.8/8 authenticated clean.
Narrow report-producer string keys, actual main/tail serialized-fleet regression
and independent review precede recovery-sealing the SAME run. No model retry,
no acceptance-bound change and no numerical/long-context promotion before seal.

## Recovery sealed — DB588, 2026-09-08 10:02Z

Same original run, corrected report-only sealing pin6f5bda69, terminalexit0.
46tests include actual main/tail strict serialized8rank equality and value/type
mutation refusal; independent review noP0-P2. Full sealer passes and DB588/archive/
8XPlane64core/32memory-owner/8hostcleanup are verified. Exact receipt:
`../artifacts/prefill-batched-own2k-sealed-20260908.json`; full identity HANDOFF.
Own2K gate is closed, not the efficient long-context project.20/20tokens;
own-score DSA order/ties and cross-oracle sets/tails exact over14x21events.
At observer positions2034..2047 no competitive top2048 cutoff is exercised.
Prefill102.202932s, decode129.1713125ms, lifetimepeak26.397GB/headroom6.618GB.
No deliveredTTFT or prefill speedup promotion. Next quantitative phase budgets
and B128MoE reuse with smaller attention/DSA tiles, not another B17 admission.
