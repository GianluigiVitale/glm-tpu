# GLM-5.3 release handoff

The local chat UI and its stateless OpenAI-compatible `/v1` API share one server
process, which holds the resident producer lock. [API instructions](docs/API.md);
tool calling, streaming and `reasoning_effort` are supported, decoding stays
greedy and sequential. The API key lives at `<state>/api-key`, never in Git.

The local chat UI adapts the owner's as-pt design and attaches to the existing
resident model. [UI instructions and provenance](docs/UI.md). Current private
handoff: `/home/gianl/glm-run/glm53_ui_20260922/HANDOFF.md`; publication receipts
there establish the final UI commit/archive. UI server uses loopback port8011
and owns the resident producer lock. Never run a second producer or unload the
model to deploy source/docs. Browser smoke turns771–772 completed correctly;
next input773 belongs to the UI. The stopped GSM8K evaluation remains740/770.

On the owner's explicit instruction the 32K session was stopped and replaced by a
166,912-slot session for autonomous agent work: run
`optimized_request_20260922T120221127955Z`, controller3593921 superseded by
controller873230, executable `19ccbc46`. Its first answer completed at normal EOS
with all-host agreement, 11.20 decode tokens/s and 28.94 GB peak HBM per chip.
[Capacity admission receipt](docs/release/glm53-context-profiles-20260922.json).
A 262,144-slot attempt was refused: XLA exceeded HBM by 1.07 GB. Do not unload or
redeploy this session for documentation, source backup or Git promotion.

Stopping the previous session exposed a defect: after the documented
`inbox/stop.json` unload, the controller crashed writing its final receipts with
`FileExistsError` on `runner.rank1.json`, because the run root already held those
files from its first request. The model did unload and both workload leases were
released, but no eight-host cleanup receipt was produced. Unfixed.

The preceding 32K evidence remains historical: executable `5c3c1d6b`, verified solo
speed 13.57 tokens/s and a partial GSM8K result of 740/770 correct (96.1%).
The owner stopped that evaluation; never resume its remaining 549 questions.

Current release preparation handoff:
`/home/gianl/glm-run/glm53_resident_release_20260922/HANDOFF.md`.
Its publication/promotion.json and release.json establish final main/tag/archive.
[Current results and scope](docs/release/STATUS.md) include exact timing and
scoring boundaries. Promotion is offline and must preserve the resident owner's
workload leases; use sync locks only for source backup, without fleet staging.

The earlier migration handoff and receipts below remain preserved history.

The real-weight four-question run passed at executablec2f60efe: four correct
completed final answers, normal EOS, all-host token/graph agreement, fresh memory
admission and eight-host cleanup. [Current status](docs/release/STATUS.md) and
[receipt](docs/release/glm53-four-answers-20260921.json) define the measured scope.

The active operational handoff is
`/home/gianl/glm-run/glm53_migration_20260921/HANDOFF.md`.
Final private-main/archive promotion is established only by its publication/
promotion.json receipt. Do not launch another workload to complete documentation
or packaging; preserve the successful existing evidence. No new research campaign.

Keep the private repository, GLM-5.2 tag/history, research refs, originals,
DB616–621, licenses and source identities. Both workload/sync locks, existing
fleet and US-CENTRAL2 constraints remain. Latest owner authorization permits
necessary fixes/retries; avoid duplicate live jobs and use completion wakeups.
Leave a paused slash goal paused. Review is self-review, not independent review.
Previous handoff is exactly recoverable at `git show glm-5.2:HANDOFF.md`.
