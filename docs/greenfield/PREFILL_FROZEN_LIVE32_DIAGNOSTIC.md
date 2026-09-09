# Frozen own8K failure — same-graph live-window discriminator

Status2026-09-09: CPU integration and independent review passed; no TPU run. §25 correctness diagnosis,
not optimization or a replacement completion baseline.

## Evidence and question

`../artifacts/prefill-frozen-own8k-token-refusal-20260909.json` binds all8 original
JSON/NPZ/log objects, matching token mismatch at index11 (expected2619/observed576)
and normal/root8cleanup. FirstDSA event1 passes allsix existing FP64 bounds,
own-score/ties/cache structure pass. None excuses the hard raw-token failure.
Observerstep10 emits the divergent token from an oracle-exact input. Subsequent
steps consume divergent IDs and cannot localize the original difference.
No logits, residuals or whole-prefill caches survived in the original NPZ.

Question: does grouping128 live prompt rows, versus32 on the SAME static graphs,
change the failing continuation? This isolates live-window numerical dependence
without inventing another kernel or trying arbitrary precision/row-count variants.
It cannot by itself distinguish a semantic defect from benign-looking rounding.

## Exact controlled protocol

- Model/runtime/kernel/sharding source stays at DB603/7456bf64; all five selected
  options, key512, checkpoint, main rotary and decode path remain unchanged.
- Same literal B128 main and B114 tail graphs, same seven registered rawgraph
  pairs, actual optimized-HLO inspection and32owner memory checks.
- Same8155 original prompt IDs.254 calls consume32 live IDs each in physical128
  slots; finalB114 consumes27 live. Pad0 is masked and never prompt content.
  Lastmain starts8096, ends8128; tail starts8128, ends8155. Physical padding
  reaches beyond8192 and must be harmless; test this before TPU deployment.
- New profile `ws32_b128_b114_8k_cap8192_live32_diagnostic_v1`, distinct tag suffix
  `_live32`, failed-source receipt SHA and diagnostic-only identity.
- Same1200s diagnostic prefill ceiling,3600s worker ceiling,1GiB measured-memory
  reserve, locks/censuses/ownership and original-byte regional publication.
  Failedrun123s × approximately4 gives a ~492s planning estimate, not a speed
  guarantee. Coldstartup remains separate. No >100GB artifact or checkpoint.
- Use existing numerical_environment(profile=FROZEN_LIVE32_PROFILE) and wrapper.
  Worker retains raw tokens, DSA, cache, graph/memory and trace evidence. Sealer
  expressly refuses diagnostic promotion even if worker correctness passes.
  No inherited adjudication, modified token oracle or changed tolerance.

## Decision rule fixed before hardware

Compare full protected token prefix and first mismatch with the retained failed
run and unchanged oracle. If it matches the oracle, live grouping matters and
the correctness investigation is narrowed; this is NOT permission to adopt a
slower final engine or claim own8K completion. If mismatch remains, record exact
signature and stop row-count trials. The next missing evidence is a bounded
predivergence state/logit capture, not another arbitrary model variant. If the
probe exposes an execution/health/memory defect, stop and fix that proven blocker.

## Local checks

81host/profile tests passed4.81s, including actual255call adapter and independently
called sealer accounting, invalidstride/refusal/oldprofile regression. Added tests
exercise real worker startup and promotion refusal. Expanded CPU32 eight-layer
test covers two physicalB128/live32 calls and physicalB114/live27 nearcapacity,
same-state B32 control, padding/rotary poison and final-live rollback. Production
78layer rawgraph identity and those CPU checks pass:33tests257.95s, including
old actual startup cases. Both rawproduction texts reproduce the registered
B128/B114 hashes byte-for-byte; no new model/kernel/sharding arithmetic.
Rolled registration/integration/reuse34tests6.21s pass, with two unchanged
expensive whole-HLO replay cases deliberately deselected (saved proofs reused).
Independent gpt-6-astra reviewer finds noP0-P2 and separately ran13newtests3.00s.
Shellsyntax/diff checks pass. Persist, mirror and apply existing fresh fleet/
storage preflights before ONE diagnostic. Expected diagnostic refusal at sealer
never becomes SUCCESS even if rawtokens pass; archive originals and cleanup.
