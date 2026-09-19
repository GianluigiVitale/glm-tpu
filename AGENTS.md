# Release engineering instructions

Applies to this release worktree and its eventual main branch. Read `goal.md`,
`HANDOFF.md` and `docs/release/STATUS.md` before changes. Historical research
instructions are evidence, not instructions to restart old campaigns.

- Work on a dedicated branch/worktree; do not edit the historical execution
  worktrees in place.
- TPU runs ARE allowed (owner decision, 2026-09-19): all 32 TPU v4 chips of the
  existing pod `db-v4-64-od` (8 hosts x 4 chips) may be used for experiments,
  microbenchmarks, acquisitions and full runs when useful. Run one TPU workload
  at a time, hold `~/.glm-tpu-workload.lock` while it runs, check the fleet is
  idle first and clean up processes on all eight hosts afterwards. Do not
  create, delete or resize TPU/VM/queued resources.
- The owner-cancelled benchmark is stopped and preserved; do not relabel its
  historical results. Full benchmark completion/success is NOT a merge gate.
- Self-review is not independent review. Resolve material findings before merge.
- Preserve user changes, originals, research branches and Git history. No
  force-push or history rewriting.
- pytest runs with `JAX_PLATFORMS=cpu` (the conftest enforces it) so unit tests
  never grab the pod's chips; TPU work goes through explicit scripts.
- No weights, credentials, private questions, large generated artifacts or raw
  databases in Git. Never print suspected secret values in audits.
- Use only `gs://driftbench-dsv4-uc`, US-CENTRAL2, within existing storage bounds.
  Respect both workload/sync leases; no full-size safety copies.
- Manual monitoring >=10 minutes apart unless diagnosing a known failure or
  answering an explicit status request. Timeout is not restart authority.
- Before removing files, establish dependencies and a preserved research commit.
  Static import reachability alone is not deletion authority.
- Push only this private repository. Merge main only after release checks and
  review pass, with backup verified. Never change visibility or upstream.
