# Goal — Prepare the private GLM TPU production repository

Prepare and merge a clean, documented, tested release into main of
GianluigiVitale/glm-tpu. Keep PRIVATE. Never change upstream tpu-inference.
Read this, HANDOFF.md and docs/release/STATUS.md at resume. Original engineering
contracts/evidence remain on rewrite/topology-first-decode; §25/§26 froze tuning
and replaced cross-build token matching with task quality and structural checks.

## Branches and preservation

main: supported release. release/production-20260914: isolated cleanup worktree
/home/gianl/glm-tpu-release. Research continues on other branches of THIS repo,
especially rewrite/topology-first-decode. Starting research pin
83f0c2728d0d418255a917343cc89d24b815bd0c; starting main
a4a17ac4e90b15f1994bd8b26917ef62daa52660. Preserve history/branches/evidence;
no force-push, history rewrite or deletion of unique work. Resolve dependencies
and preserved Git location before removing files from the release tree.

## Protect live work

Native tag greenfield_ws32_native_benchmark_20260913T150600000000000Z runs from
the original research pin. Inspect authoritative process state before action.
Do not interrupt or modify its source/enforcement during execution AND sealing.
Release work may proceed independently here. Manual polls >=10 minutes apart
except explicit status questions/known failures; timeout is not restart.
All FOUR128K DB616–619 and full256K E0 DB620 are sealed: never repeat them.

## Deliver

1. Inventory actual supported engine, imports, assets, configuration and entry
   points. Separate obsolete experiments without breaking runtime/protection.
2. Clean release tree; preserve history on research branches. Provide README,
   architecture, tested installation, checkpoint preparation/loading, real
   inference example, operations/recovery and troubleshooting.
3. Audit quality, dependencies, licensing, secrets/private data and error paths.
   No weights/private questions/credentials/raw DB/large outputs in Git.
4. Add proportionate automated checks; smallest decisive tests first. Reuse valid
   evidence, no expensive reruns for cosmetics. Execution changes need their own
   appropriate validation; preserve original math and source identities.
5. Report actual evidence scope: measured speeds, complete/partial scores,
   scoring defects, protocol gaps and limitations. No unsupported production,
   card-parity, concurrency, network-TTFT or durable-resume claim.
6. Adversarial self-review; resolve P0–P2 or report genuine release blockers.
   Commit/push release branch; merge main only when checks pass. Verify regional
   mirror without overriding active sync lease. Never make repository public.

## Constraints and completion

ONLY current chat GPT-6 Astra High. No Ultra/subagents/external reviewers/Claude.
Self-review is not independent review. Freeze accepted speed; no tuning campaign.
Existing8hosts/32v4 only. NEVER manage TPU/node/VM/queuedresources, especially
db-v4-64-od-qr4. Only gs://driftbench-dsv4-uc (US-CENTRAL2), live<2.5e12B,
softdeleteoff. Respect both leases and source freeze. No full-size safety copies;
>100GB needs peak/retained/replacement explanation. Tests JAX_PLATFORMS=cpu.
Done when main is a coherent reproducible supported release, research remains
recoverable, validation/limits explicit, changes pushed and regional mirror
verified. Do not declare success or merge around an unresolved release blocker.
