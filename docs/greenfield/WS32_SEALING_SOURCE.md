# WS32 sealing-source isolation

Candidate branch: `tooling/ws32-sealer-isolation`, based on `8bf907e9`.
Do not merge into the active worker-0 checkout until the running L7 depth-0.0 result seals.

## Problem and boundary

A protected prefill lasts hours. The wrapper previously invoked evidence processing from the same
checkout used for development on worker 0. Changing enforcement code could make the late seal refuse,
despite a completed numerical run. Merely recording a launch-time hash would not prove which code
the sealer eventually executed.

The existing sealer already checks its own source root, committed enforcement objects, import
origins, declared run/recovery pins and published ancestry. Reuse those checks from a dedicated
detached checkout instead of changing their meaning.

## Execution

After acquiring the workload lease, before rollback handlers or protected work, each wrapper attempt
creates `RUN_DIR/sealing-source.<unique suffix>` at its frozen `RECOVERY_PIN`. For an ordinary run
that equals `PIN`; for recovery it is the newly reviewed recovery commit. Earlier checkout directories
are neither moved nor overwritten. The path and commit are recorded in `orchestrator.log`.

`seal_python` checks HEAD/cleanliness and sets cwd, PYTHONPATH and `JAX_PLATFORMS=cpu` for controller
materialization, validation, DB publish/rollback, acquisition recovery and inline evidence operations.
Changing PYTHONPATH alone would leave `python -m`/stdin vulnerable to the main cwd taking precedence.
Bytecode writes are disabled. A separate checkout is stable isolation, not filesystem immutability;
it remains reserved from edits and the existing enforcement checks remain mandatory.

Only sealer adjudication/profile paths move into this root. Their hashes and lessons pin are
unchanged. Worker CLI/source, checkpoint/oracle/topology/tokenizer/run paths and results DB stay where
they are. Ordinary seals still omit `--recovery-code-hash`; recovery seals declare it. Materialization
and terminal ledgers continue to bind the original run pin and frozen producing pin. Strict remote
evidence layouts do not gain extra objects. Git history preserves the source; no large backup is made.

This does not authorize editing live worker code: lazy imports and file-loaded utilities may still
use the worker checkout. It also does not add live attach to RECOVER; original-worker monitoring,
authenticated cleanup and collection remain governed by `WS32_ORPHAN_RECOVERY.md`.

## Other pre-L8 fixes in this batch

- E0 validation rejects any planned timing window other than 256 iterations before artifact reads,
  for acquisition and numerical modes. The wrapper already supplies 256. Existing per-rank checks
  still require the actual timing count and sample vector to equal that plan. L7/short-context
  ten-step windows remain unchanged.
- The enforcement surface adds `bench/engine.py` and `bench/provenance.py`, which the pinned
  legacy passkey extractor imports. Its existing SHA-bound loader remains unchanged.
- Generic refusal messages describe protected seals, not only §21.2 adjudications; L7/L8 use §23.5.

## Verification

The actual extracted shell block is exercised against temporary Git repositories: main-checkout
edits cannot change `python -m` output; artifact rebasing preserves worker arguments; the controller
forces CPU; each attempt has a distinct source; modifying the sealing checkout refuses the next
Python call. Tests drive the E0 guard before artifact access, preserve valid E0/L7/short windows,
and prove edits to either newly covered legacy dependency refuse a clean-surface check.

No TPU run, old-result reseal or performance claim is made by these CPU tests. Before deployment,
require adversarial review, published/mirrored source and completion of the existing live run.

2026-09-07 validation: 26 new behavior tests passed (1.82 s); the existing short-sealer and
long-context suite passed 78 tests with one opt-in full-eight-rank replay skipped (321.80 s).
The default real-evidence rank-loop replay passed. Independent Astra found no P0–P2 and also
loaded both actual 8K adjudication records from this isolated checkout. Approved for isolated-branch
persistence only; no deployment before the current L7 depth-0.0 result seals.
