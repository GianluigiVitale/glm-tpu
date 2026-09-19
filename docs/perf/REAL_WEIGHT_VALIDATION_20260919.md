# Real-weight challenger validation (2026-09-19)

Status: corrected real-weight acquisition running; no token result yet. The goal is still open.

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
