# Frozen DB603 implementation — own8K continuation

## Latest result — 2026-09-09 10:10Z

Own8K ran at53a9d193 and FAILED: first raw-token mismatch at index11,
expected2619/observed576 on all8hosts.8155prompt IDs/64calls finished healthy,
fleetmax request-prefill123.065669s; all7HLO inspections and own-score/tie/cache
structure checks pass. FirstDSA event1 passes allsix existing FP64 bounds, but
this cannot excuse the token failure. No accepted8K performance or correctness.
Original24JSON/NPZ/log objects generation/CRC/SHA verified; normal/root8/8clean.
See ../artifacts/prefill-frozen-own8k-token-refusal-20260909.json and HANDOFF.
Next is one reviewed same-graph live32 grouping diagnostic, not a blind retry,
new kernel, speed campaign or slower final-path substitution. It is not yet wired.
The implementation/readiness narrative below records the pre-run state.

Status2026-09-09: CPU integration and independent current-diff review passed.
No own8K TPU execution or new speed/correctness result yet. Authority: §25.

## Minimal adaptation, not optimization

Runtime/kernel/sharding/model source remains byte-identical to
`7456bf6433e1dce966670deb252f4c64bbc5f432` (DB603). Reuse the same B128 main
and B114 tail programs, all five selected options and the five companion graphs.
The8K prompt has8155 IDs:63×128+91. The tail receives91 live IDs plus23 legal
token0 padding IDs, with the existing dynamic `valid_rows=91`. Padding is not
prompt content, does not advance the frontier and is not teacher-forced decode.

The host plan now distinguishes live tail91 from physical graph114 explicitly.
Its identity binds `tail_graph_rows=114`, `padding_token_id=0` and masked padding.
Default None leaves every prior plan/input identity unchanged. Only the new
`ws32_b128_b114_8k_cap8192_rolled_panels_merge_live91_v1` profile enables this
completion workflow; it binds DB603's receipt by SHA and disclaims numerical
inheritance. Old2K registration/targets/receipts are untouched historical evidence.

This avoids literalB91's new three-iteration compiler geometry. ExistingB114
executes prefix counts32,32,27,0; last live row90 supplies the head and selected
metadata. At8K the final block starts8064 and ends8155. Prompt repair promotion,
cache ownership, global health/rollback, topology and numerical contracts remain.

## Decisive checks performed

- CPU32, eight-layer panel-capable fixture at the cache end: actualB114/live91
  equals three B32 controls (32,32,27live) over the entire result tree. Poisoned
  inactive token IDs and inactive rotary rows leave token, health, frontier,
  selected metadata and both cache states identical. Invalid last liveID90
  refuses and atomically restores initial state. Synthetic, not real weights.
- Actual host adapter accounts8155 IDs in64calls, with91live in its final114
  physical input; the independently called sealer accepts the JSON-roundtripped
  accounting and rejects physical rows masquerading as live prompt rows.
-119CPU tests pass5.37s: host adapter, new profile, real worker/sealer parser
  startup, shell prefix, old rolled registration, current worker memory/refusals.
  The two existing expensive whole-HLO replay cases were deselected: unchanged
  saved structural proofs are reused, not reacquired. A pre-existing assertion
  matching old error text `1..32` was corrected to the unchanged production
  refusal text `explicit window scope`; no production guard was weakened.
- Production78-layer abstract lowering using the new8K plan reproduces BOTH
  registered raw StableHLO texts byte-for-byte:1test89.71s, no weights/TPU.
- `bash -n` and `git diff --check` pass. Independent gpt-6-astra reviewer
  `/root/observer_identity_review` finds no remainingP0-P2 on the current delta.

The CPU tail test passed in the earlier combined invocation:60passes/one
pre-existing regex assertion failure,91.68s. The corrected host assertion then
passes in the119test suite above. Do not report that earlier invocation all-green.

## Execution recipe and bounds

Use `scripts.greenfield.ws32_batched_launch.numerical_environment` with
`FROZEN_8K_PROFILE`, then the existing protected `run_short_decoder_ws32.sh`.
No direct worker bypass, new launcher, checkpoint or TPU resource operation.
Both leases, published clean pin, regional storage, fresh authenticated fleet
censuses, HLO, actual32-owner memory, source freeze and archive protections apply.

Prospective8K prefill diagnostic ceiling1200s =4×old2K300s. This is not a speed
target or a prediction: DB603 rate would project~128s but truncatingDSA has not
been measured on this prompt. Worker ceiling3600s covers observed cold work plus
that budget; old2K2700s unchanged. Direct SSH wait has no shorter overall timeout
and embeds the same worker bound; keepalives and original recovery remain.

First attempt binds NO adjudication or later-event acknowledgement. If original
observations first diverge, preserve them, then reuse
`adjudicate_ws32_first_divergent_event.py` for this run's own§21 analysis and
preregistration. Existing event/reference code is reusable; serial8K adjudication
records are not. A different unsupported event is a concrete diagnostic blocker,
not permission for preemptive scalar-reference archaeology or relaxed bounds.

Launch headroom restored: independently reviewed DB568/569 LOCAL tracecopies
ranks1..7,4192545740logicalB, removed underbothleases after fresh exact remote
generation/CRC/SHA/localinode/timestamp/rootFD/maps checks. Free2.861→7.054GB.
Receipts `../artifacts/db568-569-local-trace-{eviction-review,evicted}-20260909.json`.
All rank0/cloud originals, weights and compact evidence retained/recoverable.
Fresh wrapper checks remain required; this is not a fleet-idleness assertion.
After own8K closure: changed-capacity HLO/HBM, allfour batched128K depths,256KE0,
serving/resume/actual first-token delivery and remaining§18 protections.
