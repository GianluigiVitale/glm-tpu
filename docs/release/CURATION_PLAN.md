# Goal — Make GLM TPU main lean, thoroughly curated and reviewable

Deliver a genuinely curated main branch in private GianluigiVitale/glm-tpu:
the supported native-JAX engine, its necessary dependencies, relevant tests,
minimal evidence and coherent documentation. Not a research archive behind a
polished README. Keep the repository PRIVATE.

## Starting point and authority

Starting main: b667f00f1ae48c8ff37e92500550c1395d74c66d.
Main currently has 1,971 tracked files (~36.2 MB), including substantial historical
code and evidence. Prior release checks did NOT establish a purpose for every
file. Do not repeat that claim.

README improvements are on release/research-presentation-20260914 at
19cd0b60c4e58fcb2d147747ecf62c96e66f5dd4. Reuse useful changes.

Read this goal, AGENTS.md, current release status, inventory and audit before
acting. Inspect actual branches/worktrees and live state. Work on a dedicated
curation branch. This goal supersedes earlier instructions to stop repository
cleanup, but does not reopen model optimization or benchmark campaigns.

## Preservation and boundaries

Preserve research branches, Git history, original results and recovery pins.
Research-only files may leave main after dependency verification; keep their
exact recovery commit/path. Do not rewrite history, force-push, delete unique
external evidence or create full-size backups. Never change upstream repositories
or repository visibility.

Use only this chat, GPT-6 Astra High. No subagents/external reviewers/Claude.
Self-review is not independent review. NEVER manage TPU/node/VM/queued resources.
No new TPU runs, quality campaigns, throughput tuning or model-weight copies.
Existing DB616–621 evidence stays historical evidence, not proof of changed code.

## 1. Account for every tracked file

Create a versioned disposition inventory bound to the starting commit. For EVERY
file record path, purpose, consumers, category, action and justification:

- supported implementation/configuration;
- required dependency or compact evidence;
- relevant test/documentation;
- research-only or superseded;
- unresolved, with the precise uncertainty.

Read retained implementation and documentation in full. Searches, AST parsing,
file names and automated summaries are not substitutes for semantic review.
For generated receipts/data, validate their structure, provenance and actual
consumers; do not mistake generated payloads for hand-written source.

Do not claim every file was read or justified unless that work was done.
No blanket “keep just in case” classifications.

## 2. Establish the real supported boundary

Start dependency analysis from the supported user-inference entry points,
loader, request runtime, recovery and required operational protections.
Do not root the closure in every historical benchmark or diagnostic launcher.

Check imports, dynamic loading, subprocesses, configuration, package assets,
source-hash registrations and file reads. Explain each retained historical
dependency. Static reachability alone is neither deletion nor retention proof.

Preserve numerical execution and protection semantics. Where historical coupling
prevents removal, isolate it safely with focused tests or document the exact
blocker. Never weaken validation or silently accept new source/HLO identities.

## 3. Curate main

Remove verified research-only/superseded material from main, preserving recovery
locations on research branches. Do not merely move the entire archive into
another folder on main. Keep compact, directly relevant evidence and its index.

Organize supported engine, interfaces, tests, tools and docs consistently.
Remove stale instructions, duplicate explanations and misleading entry points.
Explain unavoidable large modules or legacy dependencies; do not cosmetically
refactor frozen numerical code to improve appearance.

## 4. Verify and deliver

Check all retained internal links, imports, assets, packaging and documented
commands. Run applicable retained CPU tests with JAX_PLATFORMS=cpu, including
negative/recovery tests. Report skips and coverage gaps honestly. No environment
upgrades or expensive reruns for cosmetic changes.

Review the final diff adversarially. Resolve material findings. Document actual
research contribution, measured scope and limitations without unsupported
novelty, universal exactness or production-service claims.

Commit/push, merge eligible private main without force, and verify regional
backup under existing locks using only gs://driftbench-dsv4-uc (US-CENTRAL2).
Preserve storage limits and essential backups.

Done: every remaining file has a justified role; unresolved dispositions are
closed; research-only material is off main; tests/docs/assets remain connected;
before/after counts, removal/recovery ledger and audit scope are published.
Report remaining limitations plainly. A nicer README alone is not completion
