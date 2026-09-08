# B128 layer window — implementation and admission

Status: 2026-09-08. CPU integration PASS, default-off; not TPU admission.
Starting pin: `51dc69404b06efa5f24052d86a597e74b89bfa89`.

## Layer6 discriminator preparation — 2026-09-08

At base806a455d, the selected-only host loader verified35layer6 tensors/chip
on controller slots9/13/25/29:326,079,840B/chip,140observed leaves and
1,304,319,360B read/hashed/finite-checked in43.2735s. Exact pins/ledger digest
are in `../artifacts/prefill-window-layer6-host-admission-20260908.json`.
This checks only those four owners and that layer, not the fleet or full checkpoint.
No TPU backend initialization, new checkpoint, safety copy or bucket mutation.

`scripts/greenfield/prefill_window_protocol.py` now fixes cap4096, local keytile512,
page table(7,2,5,0,6,1,4,3), boundary(offset505,count128), competitive(2553,128)
and tail(2553,33). It reuses actual20-input/12-output packing; the explicit
candidate_window flag defaultsFalse. Control calls use the existing B32 complete
layer, carrying outputs2/3/4; no scalar reference. The helper accepts only fixed
authenticated fixtures, not general adversarial metadata or a serving request.

Own selected-score order/ties/causality/padding and ordered control selections
are exact. Output/residual/written-cache errors use existing fixed per-row AND
aggregate bounds; route IDs exact and route weights use the existing bound.
Untouched cache bytes stay exact. Original NPZ replay binds canonical input bytes,
four unique owner assignments, exact field inventory and rederived comparisons.
This is competitive agreement with B32 control, NOT an independent canonical full
score-row proof. Protected own8K §21 remains required.

CPU evidence: initial30PASS33.91s includes actual CPU32 layer6 execution of both
paths and all12-field equality, not just shape tests. Subsequent76PASS24.25s
(one already-passed actual CPU layer test deselected) covers final protocol,
production layer6 B128/B32 real35-leaf schema without weight allocation, original
NPZ replay/mutations, historical layer admission and reuse registry. Mutations
include score/order/selection/route drift, late health, wrong/untouched cache,
padded output and a row whose error passes the aggregate but fails row bounds.

Protected worker dispatch, exact new HLO profiles, resident-memory budget,
timing/observed route occupancy and fleet publication are still pending. This
module cannot launch a workflow; existing wrapper/worker admission is unchanged.
No metal performance or full-model result is claimed.

## Why this change

DB588 established the genuine B17/B11 full-model 2K path (102.203s request-prefill),
not useful long-context throughput. DB589 compared equal128 supplied-route real-MoE
work: eight B16 calls versus one B128 call, p50 ratio3.329 distributed and1.280
concentrated. Those phase measurements justify integrating larger MLP windows;
they do not predict a whole-model speedup or real router occupancy.

## Reused implementation

- `kernels/ws32_prefill_layer.py`: static `prefix_only=False` preserves the existing
  complete-layer API by default. The opt-in returns a distinct typed prefix containing
  actual post-attention normalized MLP input, split residual, own proposed caches,
  row selections/health and the actual input normalization. No placeholder layer output.
- `ws32_prefill_mlp_mapped`: shared existing dense/router/grouped suffix. Original
  BF16 boundaries, FP32 route sum and feature4/expert8 collectives retained. Only the
  router's row guard expands to128; no DSA/attention/repair guard expands.
- `kernels/ws32_prefill_window.py`: static <=32-row prefix tiles inside each layer,
  followed by one <=128-row MLP. Carry each layer's own KV and each producer's own
  unrepaired/repaired index slots between tiles. Concatenate original row order.
- `runtime/ws32_batched_prefill.py`: `mlp_window=False` builder/mapped opt-in selects
  that layer body; embedding supports128 rows, and all existing final-head, all-owner
  health consensus, whole-window rollback and final-only repair promotion are reused.
  Existing worker/adapter/enforcement still select the old default. No automatic launch.

Tile live count is `clip(window_count - tile_start, 0, tile_rows)`. Offset addition
is bounded before arithmetic, including malformed metadata and zero-live trailing
tiles near capacity. Main rotary rows remain corresponding absolute-position slices.
No partial externally committed frontier, host dispatch between tiles, copied model
implementation, new checkpoint or early repaired-key visibility is introduced.

## Tests and review obligations

CPU32 compares two128-row windows against explicit four32-row full-decoder blocks
on the existing eight-layer fixture, including producers0/1/2/6 and shared transitions
2→3/6→7, reordered pages and distinct populated caches. Tail31/32/33/127 tests
include invalid padded IDs, NaN padded rotary and empty tiles near capacity.
Late-row invalid tokens and single-owner proposed-layer health failure must roll back
every earlier write, retain frontiers/phase and return token−1. Invalid counts/offsets,
repaired-history isolation and final-only repaired-cache promotion are also required.
These are synthetic populated-prefix mechanics, not authenticated resume evidence.
Existing B17 layer/decoder tests protect the unchanged default's numerical API.

Completed CPU evidence:

- Initial window plus existing layer/B17 decoder/78-layer schema regressions:
  4PASS510.69s. The expanded final window test subsequently passed325.67s,
  including repaired-history isolation and malformed offset interventions.
- Full78-layer B128/B33 abstract schema: PASS78.32s with actual production
  shapes, no model weight allocation. Default schema and reuse checks also pass
  (5PASS57.62s in the initial schema batch, alongside one test-wiring failure).
- That initial schema test mistakenly called the deliberately32-row-only host
  adapter for B128. Corrected the test to call the new runtime API directly;
  no production/worker admission guard was widened to make the test pass.
- Black25.1.0 check and `git diff --check` pass. No TPU execution, new artifact
  pack, memory/headroom measurement or performance claim in this change.

Independent Astra reviewer inspected the actual source and found no P0-P2 before
tests. Shared-suffix extraction changes source metadata and may affect TPU fusion;
do not claim byte-identical compiled HLO. No repeated cleared B17 TPU trial is needed
for this refactor. New B128 execution must earn its own bounded admission.

## Next decisive evidence, not a new symbolic-proof project

1. Finish focused CPU composition/regression checks and review final changes.
2. Adapt the existing selected-layer worker for a bounded complete-layer window
   comparison: actual output/routes/cache/health and original-array replay, exact
   HLO groups, compiled/observed memory. Reuse real retained weights; no full load.
   Candidate is one B128 window; control is four B32 complete-layer calls with
   causal cache carry on identical128 inputs. Preserve existing bounded tensor
   contracts and exact routing/selection/state requirements; no scalar-reference
   archaeology. Use a separate protocol, not changed historical B17 case defaults.
   The first full-indexer+MoE layer is **layer6**, not layer3 (which is IndexShare).
   Confirm its retained manifest/owner bytes before admission. Include a page
   crossing and competitive DSA prefix beyond2048. Capture actual128-row router
   IDs/weights and derive tile occupancy from those original arrays.
3. Measure equivalent-work complete-layer wall, actual router occupancy and prefix
   phase costs. DB589 supplied-route timing cannot substitute for these.
   Separately compiled prefix timing is diagnostic, not an exact decomposition of
   the full layer: changed observables/fusion can change its cost and arithmetic.
4. Use resulting attention/DSA/MLP/communication budgets to fix quantitative128K/256K
   prefill and delivered-TTFT targets before candidate performance promotion.
5. Acquire/admit actual full-model window memory/HLO, own competitive8K numerical
   proof under unchanged §21, then efficient four-depth L7 and full L8.

At long capacity, old lifetime HBM is not a budget for this graph. Attention selected
KV stays tile-bounded, but compiler overlap, proposed/old caches and window temporaries
need actual allocation evidence. No additional model artifact or full-size backup.
