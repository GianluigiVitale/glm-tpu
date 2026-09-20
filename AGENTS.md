# Release engineering instructions

Applies to this release worktree and its eventual main branch. Read `goal.md`,
`HANDOFF.md` and `docs/release/STATUS.md` before changes. Historical research
instructions are evidence, not instructions to restart old campaigns.

- Use this chat, GPT-6 Astra High only. No subagents or external reviewers.
- Work on the release branch; do not edit the active research execution worktree.
- Do not manage TPU/node/VM/queued resources, launch duplicate work, change active
  benchmark source/enforcement or interrupt evidence sealing.
- The owner-cancelled benchmark is stopped and preserved. Do not resume it or
  optimization campaigns. The owner's later request is a usable optimized
  ordinary release on main; bounded validation of that integration is in scope.
  Full benchmark completion/success is NOT a merge gate.
- Self-review is not independent review. Resolve material findings before merge.
- Use `apply_patch`. Preserve user changes, originals, research branches and Git
  history. No force-push or history rewriting.
- Tests require `JAX_PLATFORMS=cpu`. Never initialize TPU during a source audit.
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
