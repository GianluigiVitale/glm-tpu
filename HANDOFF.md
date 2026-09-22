# GLM-5.3 release handoff

The resident release is prepared from executable `5c3c1d6b`, with a verified solo
speed of 13.57 tokens/s and a partial GSM8K result of 740/770 correct (96.1%).
The owner stopped the evaluation; never resume its remaining 549 questions.
The model stays loaded under controller3593921 in
`optimized_request_20260921T233911390408Z`; next private input sequence is771.
Do not unload/redeploy it for documentation, source backup or Git promotion.

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
