# Section 18 completion audit — working evidence index

Status: **NOT COMPLETE**, 2026-09-07. This is an evidence checklist, not a new acceptance
contract or final sign-off. Authority remains [the specification](../glm-tpu-revolution.md),
including its accepted §21–§23 amendments. The current run/next action is in
[HANDOFF](../../HANDOFF.md). Do not rerun an accepted gate merely to populate this table.

## Requirement-by-requirement standing

| §18 requirement | Existing evidence and scope | Remaining proof |
|---|---|---|
| GLM serves at 256K on the existing 32 chips | DB572: 8K prompt at 262656 capacity; not full 256K prompt evidence | Fresh 513-page acquisition, then full 262144-token L8 E0; §23.7 qualifies serving as steady-state decode with hours-long prefill |
| Independent native execution | Greenfield WS32 model/runtime; protected consumers DB553/567/570/573/574; legacy reference and extraction utilities are not model execution | Bind final L8 source and dependency identities |
| Plan-aware final-layout checkpoint | WS32 32-owner runtime, manifest and SUCCESS below; direct loader checks each host's four payload hashes before loading | Bind the final consuming run to the same checkpoint and dense overlay; no new full pack required |
| DSA selected sets and tie order | DB567 under §21.6: own-score canonical selection exact; event 0 legacy-exact, first divergent event independently adjudicated; later events recorded, not adjudicated | Remaining long runs must pass within-engine exactness under §23.5; no cross-oracle DSA claim at long context |
| Raw tokens and quality | DB567: 20-token exact oracle prefix; B′ DB570; L7 DB573/574 correct passkeys | Correct keys at depths 0.05/0.95. L8 has NO_CORRECTNESS_ORACLE; diagnostic IDs are not a quality result |
| State/load/cache protections | Gate C contracts, direct checkpoint validation, short-context and capacity runs; L7 sealed summaries | Full L7/L8 state/cache/load records. Deep per-layer tensor comparisons were inherited from Gate C, not performed in DB567 |
| Local repeated collectives; no full-pod hidden reconstruction | Protected WS32 HLO and physical trace evidence; Gate D §21.6 | Final acquired/executed HLO hashes and physical groups/counts agree; fresh L8 trace |
| PP8 protected measurement; PP16 measurement or rejection | DB563 PP8 2K; §22.3 PP16 evidence-backed rejection | Closed under §22. Do not confuse historical PP16 synthetic full-decoder acquisition with protected numerical evidence (§22.5) |
| WS32 measurement and fastest correct plan | DB553 versus DB563 at protected 2K; WS32 promoted §22 | Long-context gates must not expose a reason to reopen the documented plan decision |
| Device and profiler-free wall agree | DB574 diagnostic comparison below; coverage alone is not numerical agreement | Compare final L8 trace durations and profiler-free distribution, disclose different windows/statistics, investigate material unexplained disagreement |
| Four-depth 128K smoke | DB573 depth1.0 and DB574 depth0.0 sealed | Depth0.05 and depth0.95, each independently sealed |
| 256K E0 | Input capsule exists; capacity measured only | Full prompt, 256 profiler-free timed steps, p50/p99, generated IDs, HBM and eight-file/64-core XPlane; legacy comparison per §23.5 |
| DB linkage and approved archive | Existing numerical rows have protected local/remote evidence | Final dependency map: run → inputs/checkpoint/overlay/acquisition → source/config/ledgers → DB/SUCCESS generation |
| Authenticated eight-host zero-work cleanup | Existing sealed rows; DB574 original collection includes supplemental device-holder checks | Terminal final-run census; a free lease, dead controller or missing lockfile is not enough |
| Base and effective throughput separate | Published figures are base decode; Gate H untouched | Report speculative/effective throughput as unmeasured; do not claim the unmet strong Gate F target or interactive prefill |

## Checkpoint and overlay anchors

Runtime tag: `greenfield_ws32_runtime_pack_20260815T214050854386790Z`.
Current worker-local load root:
`/dev/shm/glm-ws32-runtime/greenfield_ws32_runtime_pack_20260815T214050854386790Z`.
This is a volatile serving copy, not the durable archive location.

- Manifest identity: `c04f800edf15651198ab2c5183fff8a8ae9a427609b59cc61ccabdab32f5ee08`.
- Manifest file SHA: `88df414301e0d303506163c078484ec14cdb15ff03972789df58960b091a7cbf`.
- SUCCESS identity: `1bfea5bd2dd8b096a3e551d7f96697496d8277bdaaa328ca1c35144feb8f1760`.
- Payload: 32 owner files, 786172488192 bytes; preserve the pinned reproduction/source inventory.
- Dense overlay tag: `greenfield_ws32_strategy_nd_dense_overlay_pack_20260827T002508229552699Z`.
- Overlay manifest identity: `a8dc8791f034906d163d399b2393b89947e7aff25303e2c94b7dafbdf2694b6a`.
- Overlay manifest file SHA: `c17194b6dbf2a41c5ae869c6184d6c88f04eb5adfe2105b94965e8434c2b5c8c`.
- Overlay SUCCESS file SHA: `166566b9b066aa4803dacf769c51a890c5e79690770a9582e9dbdccb085332a6`.

The original d0.0 argv and input paths remain in
[`configs/greenfield-ws32-l7-d0-recovery.json`](../../configs/greenfield-ws32-l7-d0-recovery.json).
That capsule describes an already sealed run; it is NOT a command to rerun/recover DB574.
Historical missing-pack notes must not trigger creation of another near-terabyte layout.

## Sealed L7 anchors

All archives below use `gs://driftbench-dsv4-uc/results/<tag>/`.

| DB | Depth | Tag suffix after `greenfield_ws32_short_decoder_` | SUCCESS identity | Remote SUCCESS generation |
|---|---|---|---|---|
| 573 | 1.0 | `128k_d1_0_numerical_cap131072_hrope_20260907T012156230341652Z` | `795245b420a9a52d049678b00d044a521da11776056eb82e6fee511952c9cdfd` | `1788763301245387` |
| 574 | 0.0 | `128k_d0_0_numerical_cap131072_hrope_20260907T064941550123130Z` | `f123beba8caab916e1307607ef28b294606c95338f3a7d431ce374a210900667` | `1788784016426688` |

Full summary/ledger/DB identities and the original recovery chain are in HANDOFF and
[PERFORMANCE_LOG](PERFORMANCE_LOG.md). The compact supplemental collection record is
[`gate-l-ws32-d0-original-collection-20260907.json`](../artifacts/gate-l-ws32-d0-original-collection-20260907.json).
Acquisition records are dependencies, never additional successful numerical runs.

## Device/wall comparison to finish offline

DB574 summary reports trace `device_step_ms = 140.52092173200782` (two steps/core),
`step_cycle_ms = 147.94815226975`, versus profiler-free fleet wall p50
`143.496482` ms (ten timed samples), p99 `146.1185879` ms. The device-step mean is
about 2.1% below wall p50, but the windows and statistics differ. This is a diagnostic
consistency comparison, not same-window equality, and does not establish final L8 agreement.
Use the same explicit definitions for the final run, including any observed host/idle gap.

## Finalization checklist

1. Finish the fresh 128K acquisition and bind its actual graph hashes; never borrow hashes
   from a different Python pin. Then seal depths0.05/0.95 in order, stopping on first failure.
2. Acquire the full E0 graph set at capacity262656 and check compiled memory before the long run.
3. Seal L8 with its 256-step window; compare legacy DB402 prefill/decode against WS32 with
   both warm-up protocols disclosed. No correctness oracle is available for this synthetic E0.
4. Populate final DB/source/config/input/ledger/SUCCESS-generation dependencies and verify
   the surviving working set, same-region archive and terminal 8/8 cleanup.
5. Obtain the independent final requirement-by-requirement audit. Only then declare §18
   complete, retaining the explicit slow-prefill, later-DSA, tensor-inheritance and
   unmeasured-speculation qualifications. No acceptance threshold is changed here.

Readiness basis: independent Astra review of current source and evidence, 2026-09-07,
performed while the fresh acquisition ran. That review identified the remaining checks
above; it was not a final approval and did not require any additional TPU workload beyond L7/L8.
