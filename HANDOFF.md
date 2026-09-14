# Handoff — repository curation in progress

## Objective and authority

Finish [CURATION_PLAN](docs/release/CURATION_PLAN.md): a lean, file-by-file
justified private main, not a research archive behind a polished README.
Read [goal.md](goal.md) and [AGENTS.md](AGENTS.md). Only this chat, GPT-6 Astra High;
no subagents/external reviewers. Self-review is not independent review.
No TPU runs, benchmarks, tuning, weight copies, environment upgrades or
TPU/node/VM/queued-resource management.

## Published state and working boundary

- Published main: b667f00f1ae48c8ff37e92500550c1395d74c66d, 1,971 files.
  It is NOT yet the curated deliverable.
- Worktree: /home/gianl/glm-tpu-release, branch release/curation-20260914.
  Last batch before this update: a7293fd9a9bd876ee7c6666a8ef23a77ca0c4f4a.
- Research remains at 83f0c2728d0d418255a917343cc89d24b815bd0c.
  Canonical /home/gianl/glm-tpu-topology-rewrite remains detached at published main.
  Do not switch or edit that checkout for curation.

Inspect current Git/worktree state at resume; these pins do not prove live jobs.

## Curation progress

[Ledger and review scope](docs/curation/README.md) records every starting file and
current addition. Its checker reports unresolved decisions honestly; a clean
consistency report is not completed semantic review.

145 original files removed from the candidate tree, recoverable in Git:
historical PR/recon/review material, obsolete launchers, the self-contained
35-file PP16 forced-round experiment, and the old research journal. The journal's
two executable source-certificate consumers retired with that experiment.
Another 24 projection-contraction orchestration, publication, adjudication and
experiment-only test files are removed. Their shared acquisition/publisher and
source-authority helpers remain: later diagnostics dynamically read pinned bytes.
Original external results and installed historical capsules were not deleted.
Current candidate: 1,835 tracked files; final boundary remains incomplete.

User host protections are in scripts/release/ws32_host_ops.py; nine helper ASTs
and three constants match starting main. The campaign scheduler is gone.
The lazy benchmarking facade preserves 212 named targets while avoiding eager
imports. Worker initialization still uses the historical short-decoder runner.
Frozen numerical execution and native admission identities are unchanged.

## Verification and open findings

Selected release checks previously passed 460 tests, one optional skip and two
upstream warnings; frozen-source/content/wheel checks passed. Recheck the latest
batch results in the curation ledger rather than inheriting hardware validation.
No TPU test was run for curation.

The retained projection publisher suite passes 45 tests. Historical fixtures now
use preserved source-compatible pin 986378238ac6458307aea69ef1f5e12bf82bc020,
not moving release HEAD, and assert the publisher's actual canonical worktree.
A new mutated-source negative test proves the unchanged source guard refuses
changed bytes. Production checks and registered identities are unchanged.
The earlier neighbor result (26 pass/2 fail) is not a current suite pass: its
numerical-runner test retired, and its acquisition-source test still needs review
of historical source/branch assumptions. Do not waive those production guards.

Next: resolve remaining historical workflow/dependency groups, review retained
implementation/docs in full, and validate retained assets/provenance and tests.
Do not keep every historical campaign merely because it has its own tests.
Do not remove frozen numerical files or actual user/recovery dependencies by name.
Finish all dispositions and final checks before private-main merge.

## Measured model scope — no new campaign

[Release status](docs/release/STATUS.md) binds the original receipts:
DB616–619 four 128K passkey depths; DB620 full 256K E0 without a correctness oracle;
DB621 one ordinary 19-token prompt, 71 generated tokens, EOS and READY, with
original all-rank replay, HLO/HBM/trace/DB/archive/cleanup evidence.
These are historical scoped results, not new curation or broad quality proof.
The owner-cancelled GPQA prefix is not a dataset score or public-card parity.

Use [inference](docs/release/INFERENCE.md),
[operations](docs/release/OPERATIONS.md) and
[checkpoint recovery](docs/release/CHECKPOINTS.md) for the supported interface.
Same-live-session continuation is not durable KV recovery or a concurrent service.

## Preservation and backup

Keep private visibility, research refs, original evidence and Git history.
No force-push. Commit/push only the private curation branch until eligible merge.
Only gs://driftbench-dsv4-uc, US-CENTRAL2, live <2.5e12 bytes, soft delete off.
Respect both existing workload/sync leases and the installed five-minute mirror;
do not disable it or start an unleased duplicate. Verify the final exact ref/tree
backup separately; cron presence alone is not proof of synchronized contents.

The old chronological release journal is recoverable with:
`git show 05981732d27e6ff3d715c2cbc31181c58a830596:HANDOFF.md`.
The research journal and every removed original have starting-commit/blob recovery
in docs/curation/disposition.jsonl. No historical success/failure receipt was
rewritten as evidence for changed code.
