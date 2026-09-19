# Real-weight challenger validation (2026-09-19)

Status: real-weight acquisition completed; DB610 token parity failed. The goal is still open.

`tools/perf_real_validation.py` is a research worker outside the frozen source.
Its controller must hold both workload leases, authenticate eight-host idle,
deploy identical source bytes and retain the leases through final cleanup.
There is no infrastructure provisioning, DB write or serving promotion.

The input constructor authenticates DB610's original remote ledger against the
sealed receipt, then authenticates its rank0 runner against that ledger. The
original short-context oracle loader verifies token/DSA manifests and terminal
SUCCESS identities. The 2,034-token prompt and 29 reference tokens remain in an
owner-only run directory outside Git. Receipts expose only hashes, counts and
the first mismatch index, never the prompt or generated-token payload.

The worker reuses the original 32-device topology initialization, source inventory
inspection, full checkpoint verifier and final-owner loader. Every host hashes
its four local files; the loader also checks every tensor while placing it on
its physical owner. Existing retained weights are reused, not repacked.
Checkpoint placement and BF16 preparation have conservative memory projections.
WK retains completed BF16 decode and separate FP32 promotion executables.

Fresh B128/B114 and greedy decode HLOs are preserved privately with SHA256 and
compiler allocation records. All ranks must agree on both graph hashes. The
research HLO checker requires 32 partitions, expert8/feature4 axis groups with
global device IDs, no unreviewed collective kind, exchanges <=128MiB and tiny
full-pod consensus <=4KiB. This is a scoped structural admission, not inheritance
of the frozen exact-HLO proof. Every graph is re-admitted with all three resident
executables present, allowing 512 MiB reserve per chip.

Prefill uses D8/P1/P2, original canonical dense placement and expert panels,
paired sorting, sorted merge and rolled B128/B114 prefixes. Decode uses
D1/D8/D5/D10, greedy sampling and unfused reductions. The wider P4 panels,
bounded owner attention and D4 host loop are excluded from this first model
comparison to localize any numerical divergence. These changes have CPU proofs
or documented boundaries and synthetic TPU receipts; real correctness remains
unproven until this run completes. DB610's exact-DSA/strategy-dense decode
preparation differs from the challenger and is not silently inherited.

A warm first block is excluded, then the complete prompt starts fresh. Prefill
wall time includes each block's fleet health vote and receipt update; decode
reports 28 model-step times separately from host health/cache checks and delivery.
All 29 generated IDs are compared to the authenticated DB610 trail. Real KV/index
caches are checked for finiteness after prefill and decode, and each decode
residual is checked. These checks cannot establish broad trained-model quality
or long-context correctness.

CPU checks: 12 tests passed, including greedy decoder construction before weight loading; the actual retained DB610 input
authentication passed without printing private arrays. Frozen source remains
unchanged. Hardware acquisition, graph/memory admission and token result pending.


The first acquisition was stopped after finding a host setup bug: the worker
omitted the explicit greedy option although the shared challenger defaults to
sampling. It could not reach decode. The corrected worker constructs its greedy
decoder before checkpoint verification/loading. Launch rank now names receipts
separately from the captured JAX process index. Source snapshots remain immutable;
failed-attempt verification and stop/cleanup records are preserved privately.


The stopped acquisition verified all 32 file slots and admitted checkpoint-load
memory on all 32 chips (verification fleet phase 235.66 s). It did not complete
loading or compile/execute a model. SIGTERM did not terminate libtpu workers;
SIGKILL was limited to their recorded PIDs/start times and exact run paths.
All eight hosts were authenticated idle, and all eight partial receipts were
recovered using the captured launch/JAX mapping. [Stopped-attempt receipt](tpu-real-db610-stopped-20260919T163147Z.json).
The corrected run `perf_real_db610_20260919T163936Z`, source `267516cf`, has started
under both workload leases after another authenticated idle check.


At the 16:53 UTC observation the corrected acquisition had completed real-weight
loading on all ranks, original WK preparation and BF16 preparation admission.
Model graph compilation/admission and the token comparison remained pending.
`--summarize` now requires all eight complete launch-rank receipts, distinct JAX
ranks, exactly 32 verified slots and measured chips, every prompt/decode health
phase, matching source/input/checkpoint/graph identities, and passed HLO/memory
checks. It reports a 29-ID mismatch as `db610_token_check_passed=false`, not a
successful numerical result. Public fields omit prompt/token arrays. Eighteen
CPU tests passed, including truncation, owner duplication, graph drift and
private-payload exclusion checks.


## Completed real-weight comparison

All eight ranks completed source/checkpoint verification, loading, fresh graph
consensus, scoped HLO/memory admission, every prefill/decode health phase and
finite-cache checks. The controller authenticated idle on all eight hosts.
The strict fleet summarizer accepted the complete experiment and reports
`db610_token_check_passed=false`.

- Prefill: 2,034 tokens in 14.6736–14.6741 s, **138.612–138.616 prompt tok/s**.
- Decode: 28 model steps, **15.175–15.278 tok/s**, median 64.091–64.554 ms,
  p99 89.600–89.943 ms. Host checks/delivery excluded.
- Peak HBM: 28,228,678,144 bytes/chip; minimum headroom 4,785,720,832 bytes.
- All ranks generated the same trail. **Only token 0 (the prefill output) matched
  DB610; all 28 subsequent outputs were zero.** No raw token payload is published.

This rejects the challenger numerically. Finite residual/cache checks alone did
not detect this degenerate output. Investigate the first decode step, including
activation magnitudes and the prefill-to-decode handoff, before promoting or
spending hardware time on further speedups. The raw-default challenger also
lacks DB610's exact-DSA and StrategyND dense realization; that distinction is
known, but has not been established as the cause of the zero output.

[Complete eight-host receipt](tpu-real-db610-20260919T163936Z.json).
Source snapshot `267516cf`; all 16 acquisitions in the cleanup ledger now ended
with authenticated eight-host idle. Original private HLOs and rank receipts remain
in `perf_real_db610_20260919T163936Z`; no frozen body was changed or promoted.


## First-step diagnostic preparation

`--diagnose-layerwise` adds a separate first-step experiment after the fresh
real prefill. It compiles/adopts no serving body: an embedding graph, each layer
kind and the final head execute separately with immutable input caches. Every
graph has fresh SHA consensus, structural collective checks and memory admission.
The private rank report records activation nonzero counts, maximum magnitude and
RMS per layer, then compares split/complete first-token results and residuals.
No prompt, raw activation or token array is added to public evidence.

CPU32: the eight-layer fixture has the same greedy output and healthy finite
activations; input cache/frontier are unchanged. Separate layer executables change
rounding: residual relative L2 error is below 1%, maximum absolute error <=0.0625
on this fixture. This documented diagnostic boundary is not bitwise admission
or an implementation swap. Nineteen focused tests pass (23.37 s), including the
eighteen existing real-acquisition checks. The real diagnostic remains pending.


The layerwise diagnostic launched as `perf_real_diagnostic_20260919T171314Z`,
source `35d827c8`, after authenticated eight-host idle under both workload leases.
It is still pending. The local summarizer now requires all 78 layer records and
matching diagnostic graph identities on every host; partial diagnostic results
cannot be represented as complete evidence.

A subsequent `--diagnose-ablation` option is also prepared (not enabled in that
immutable run). It evaluates the same first decode input with D1/D8 alone, with
D5, and with D10. Each variant already has CPU token/cache boundary tests; this
option adds no new arithmetic. Every new full graph must pass the same structural
and memory admission before execution, and outputs remain diagnostic only.
This can distinguish D5/D10 contributions without repeating checkpoint loading
for each variant. No speed or correctness result is claimed for it yet.


## Empty routed-owner output-window hypothesis

Static review found that D1 forces one dummy grid row when an owner has no
routes, but stores output only for live rows. The Pallas kernel discards its
aliased input ref and relies on the zero-initialized HBM allocation. The saved
real HLO does contain 150 distinct zero initializers for 150 routed calls; it
does not establish that an unwritten VMEM output window stays zero on hardware.
This is a hypothesis, not an established cause of the failed real decode.

An explicit `RoutedProjectionConfig(write_empty_slot=True)` candidate stores the
zero accumulator at the last K tile even for that dummy row. Live-row arithmetic
and reduction order are unchanged; the default remains false pending hardware
evidence. All 21 routed-kernel CPU tests pass (35.17 s), including all-empty and
mixed ownership, both result dtypes and bitwise frozen-kernel comparisons.

`--which empty_routes` prepares a TPU probe at real gate geometry, alternating
large finite live projections and empty-owner calls for both one/two-table
outputs. It checks empty output values and live-output bitwise agreement for
both variants. `--write-empty-route-slot` exposes the same explicit option to
the real worker. Neither the primitive probe nor the corrected real option has
yet run on hardware; the active layerwise diagnostic still uses the original
source snapshot and original store condition.


## Layerwise TPU result: first sparse MoE explodes

The diagnostic completed on all eight hosts with identical graph hashes and
authenticated cleanup. Both whole and split executions still return the same
wrong first decode token, and the 29-token trail repeats the prior failure.
Embedding and the first three dense layers have finite, ordinary magnitudes.
At zero-based layer 3, the first sparse MoE, the output update reaches
**5.4519e35** while its normalized input peaks at only **2.3125**. At layer 4 the
normalized input is exactly zero on every host; later sparse layers still emit
huge nonzero updates despite zero input. Final residual peaks are about 5.1e37.

These residuals are finite BF16, but squaring them overflows FP32 RMSNorm and
produces a zero normalized vector, explaining the tied zero logits and token
zero. Finite-only health checks therefore pass this catastrophic failure.
Splitting at layer boundaries does not repair it; the failure is now localized
to the sparse MoE path rather than solely a complete-graph compiler boundary.
This strengthens the empty output-window hypothesis but does not yet prove it.

[Eight-host layerwise diagnostic receipt](tpu-real-diagnostic-20260919T171314Z.json).
All 17 recorded acquisitions ended with authenticated eight-host idle. The next
primitive probe, `perf_empty_routes_20260919T173208Z`, has started from source
`34c86318` after another idle check and holds both workload leases.
