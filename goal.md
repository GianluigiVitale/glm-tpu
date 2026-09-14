# Goal — Curate GLM TPU main, file by file

Authoritative full objective: docs/release/CURATION_PLAN.md. Read it, AGENTS.md,
HANDOFF head, docs/release/STATUS.md, INVENTORY.md and READINESS_AUDIT.md at resume.
This goal supersedes the completed initial release and its instruction to stop
cleanup. It does NOT reopen model benchmarking or throughput optimization.

## Scope and pins

Private GianluigiVitale/glm-tpu; never change visibility or upstream repositories.
Starting main b667f00f1ae48c8ff37e92500550c1395d74c66d:
1,971 tracked files,36,166,207 bytes. Prior release checks did NOT justify every
file. Do not present a dependency scan as full semantic review.
README branch19cd0b60c4e58fcb2d147747ecf62c96e66f5dd4 is inherited.
Work in /home/gianl/glm-tpu-release on release/curation-20260914.
Do not change canonical execution checkout or merge main before curation passes.
Preserve research refs/history, original DB616–621 evidence and exact recovery
pins. No force-push, history rewrite or deletion of unique external artifacts.

## Required work

1. A versioned starting-pin disposition inventory covers EVERY tracked file:
   path, purpose, consumers, category, action, justification and review state.
   Categories: supported code/config; required dependency/evidence; relevant
   test/docs; research-only/superseded; unresolved with precise uncertainty.
   Read retained implementation/docs IN FULL; record which bytes were reviewed.
   Generated receipts need schema/provenance/consumer validation, not a fictional
   full prose review. AST/search/name matching is not semantic justification.
2. Trace actual user inference, loader, request state, recovery and protection
   roots, including dynamic imports, subprocesses, assets and source contracts.
   Do NOT root everything in historical benchmarks and then call it all necessary.
   Separate historical coupling safely, with tests, without changing numerical
   execution or weakening validation/HLO/source identities.
3. Remove verified research-only/superseded material from main, preserving exact
   branch/commit/path recovery. Moving the archive to another main folder is not
   curation. Retain only justified compact evidence and connected documentation.
   No arbitrary file quota and no blanket just-in-case retention.
4. Check remaining links/imports/assets/package/docs commands; run applicable
   retained CPU tests and negative/recovery cases. Report skips and coverage gaps.
   Self-review the actual final diff, resolve material findings, commit/push,
   merge eligible private main and verify exact same-region backup under locks.

## Current state / resume

Curation branch started from presentation19cd0b60; published main is untouched.
User controller/recovery use scripts/release/ws32_host_ops.py, not the campaign
launcher. Nine helpers/three constants match source. Worker loading
still uses the historical runner; numerical source and checks remain unchanged.
File decisions and runtime separation are still INCOMPLETE; no curation success.
Ledger: docs/curation/README.md;214files removed, exact recovery recorded.
Initial-release CPU431/DB621 receipts remain scoped historical validation only.

## Safety

ONLY current chat GPT-6 Astra High; no Ultra/subagents/external reviewers/Claude.
Self-review is not independent review. Tests ALWAYS JAX_PLATFORMS=cpu.
No new TPU runs, model tuning, environment upgrades or weight copies.
NEVER manage TPU/node/VM/queued resources, especially db-v4-64-od-qr4.
Only gs://driftbench-dsv4-uc,US-CENTRAL2; live<2.5e12B,softdeleteoff.
Respect existing workload/sync/cron locks, source freeze and essential backups.
No broad deletion. No costly reruns to validate cosmetic changes.

Done ONLY when every remaining file is justified, unresolved dispositions closed,
research-only material off main, dependencies/tests/docs connected, before/after
counts and recovery ledger published, eligible main pushed and backup verified.
A nicer README alone is not completion.
