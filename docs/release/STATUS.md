# GLM-5.3 release status

The real-weight four-conversation acceptance passed at executable
`c2f60efe3743ab9c23c4af30990694a62b496f5c`, run
`optimized_request_20260921T223026777871Z`. All four fixed GSM8K examples produced
correct final answers and normal EOS. [Receipt](glm53-four-answers-20260921.json).
Final private-main/archive promotion is established by the external
`/home/gianl/glm-run/glm53_migration_20260921/publication/promotion.json` receipt.
An absent or failed receipt does not establish promotion.

| Measurement | Slowest-host result |
|---|---:|
| Active-chat decode | 4.89–5.12tokens/s |
| Aggregate decode | 9.269431tokens/s |
| Prompt prefill | 317tokens /3.884926s |
| Decode | 563timed tokens /60.737280s |
| Batch including final text decoding, after startup/warmup | 66.296246s |
| Cold verification/loading/compilation | 1178.091731s |
| Compiler calls included in cold time | 692.653205s |
| Worker including initialization and disposable warmup | 1269.086057s |
| Peak HBM/chip | 28,789,189,632bytes |

The first token per conversation comes from prefill;567total output tokens include
those four tokens and reasoning. Decode includes fleet votes and local writes.
Worker wall excludes controller staging,SSH and cleanup. Rank0 alone recorded
237.93s checkpoint verification and150.46s checkpoint loading; these are explicitly
single-host phases, not slowest-host measurements. Timing intervals overlap and
must not be added as independent durations.

Source141shards/755632050320bytes verified against pinned upstream SHA256s;
32owner files/786181673984bytes packed and independently rehashed on all8hosts.
Checkpoint seal metadata has generation-bound US-CENTRAL2 readback. No separate
legacy dense overlay is consumed. [Checkpoint identities](../../configs/glm53-site.json).

Fresh graphs and memory admission, all-host graph/token agreement, independent
EOS and authenticated8hostcleanup passed. Four final answers were checked against
private references, separately from numerical agreement. Full remaining32K output
allowances were32668,32706,32683,32698; actual outputs112,57,298,100tokens.
The fixed inputs retained the prior concise-explanation/boxed-answer suffix.
No new instruction shortened reasoning; thinking was enabled at maximum effort.

The full CPU release gate at e5404830 passed627tests/1skip. Geometry correction
90777a44 passed24affected tests and complete real-inventory placement checks;
site binding c2f60efe passed50affected tests. Test sets overlap. Final CLI/docs/
archive checks and self-review are recorded in the promotion directory. Unchanged
numerical implementation reuses this same hardware run; no additional campaign.

Limits: four familiar short examples, one recognized by the model, provide no
broad accuracy or clean held-out estimate. Full32K inputs and other context
profiles are untested for5.3. One fixed batch, sequential prefill, cold startup
per invocation, no HTTP service/online admission/durable KV recovery. The fleet
has32TPUv4chips; available memory and compiler temporaries determine capacity.
Review is assistant self-review. [History](GLM53_MIGRATION.md) preserves failures.
Historical5.2status is exactly recoverable with
`git show glm-5.2:docs/release/STATUS.md`; its speeds are not5.3measurements.
