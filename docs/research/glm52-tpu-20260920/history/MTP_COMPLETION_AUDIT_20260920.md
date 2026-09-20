> **Frozen historical notebook.** The owner stopped this campaign on 2026-09-20. Pending work and live-run statements below describe their original dates; do not execute them. See [final decisions](../RESULTS_AND_DECISIONS.md).

# MTP objective completion audit — 2026-09-20

The objective is complete as a measured mixed/negative outcome, including
verified evidence-only main publication. The 25% speed target is unmet.
This audit maps its requirements to evidence;
a compiled graph, a verifier-only estimate, or a short token match cannot
substitute for accepted speculative wall throughput and representative outputs.
Current execution status belongs in [goal.md](../../../../goal.md).

| Requirement | Evidence available | Remaining boundary |
|---|---|---|
| Native layer-78 weights and upstream semantics | [Source audit](../../../perf/mtp-source-audit-20260919.json), [position/IndexShare contract](MTP_STATE_CONTRACT_20260919.md), [exact owner placement](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-native-placement-20260920.json), [completed native acquisition](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-native-acquisition-20260920T013227Z.json) | Trained native loading and accepted execution completed in the [first fleet run](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/tpu-real-native-mtp-20260920T015817Z.json). No independent upstream trained-native parity is claimed. |
| Multi-row target verification with causal prefix state | [M8 CPU evidence](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-m8-rowwise-cpu-20260919.json); `test_speculative_commit.py` independently checks all prefixes across owner/page boundaries with permuted physical pages, EOS/budget limits and a single-owner refusal | Two/three-row model comparisons have an explicit floating-point boundary. Matching selected test tokens is not bitwise cache equivalence. Five-row verification remains rejected by its numerical envelope. |
| Real verifier economics and bottleneck evidence | [Initial synthetic result](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/tpu-mtp-verifier-20260919T214702Z.json), [batched trace](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/tpu-trace-mtp-verifier-batched-20260919T223024Z.json), [trained M8 result](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/tpu-real-mtp-verifier-m8-20260919T235646Z.json) | Perfect-acceptance estimates omit native drafting/refresh, rejected work, votes and delivery. They are not accepted output throughput. |
| Native drafting, acceptance and guarded delivery | [Projection](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-projection-cpu-20260920.json), [target hidden export](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-prefill-export-cpu-20260920.json), [native component checks](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-native-components-cpu-20260920.json), [complete CPU session checks](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-native-session-cpu-20260920.json) | The real worker completed bootstrap, recurrent drafting, target verification and accepted-history refresh. Long outputs diverge from ordinary at token index 6; this is not a token-exact replacement. Greedy only; no sampled acceptance/RNG claim. |
| Paired DB610 and longer prose/code/reasoning/structured runs | Ordinary [long-question baseline](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/tpu-real-long-question-20260920T005757Z.json); [representative inputs and repeats](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-representative-inputs-20260920.json) are prepared | Both real runs completed. The [representative suite](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/tpu-real-native-suite-20260920T031727Z.json) contains both fresh repeats for prose, code and structured prompts, with matched budgets/capacity/greedy policy and all-host output agreement per mode. |
| Accepted output tok/s, acceptance by position, component/wall costs, latency and HBM | Worker and strict summary implement these fields; [ordinary sustained result](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/tpu-real-long-question-20260920T005757Z.json) is 14.4141 wall tok/s | First long measured rates: ordinary 14.2902, R2 13.0927, R3 12.7682 wall tok/s. Repeated R3 gains are 4.3–4.6% on code and 10.7–10.9% on structured output; prose loses 5.9–6.6%. Complete per-mode acceptance, components, latency and HBM are in the representative receipt. The 25% criterion is unmet. |
| Answer correctness | Exact TSP oracle retained privately; representative scheduling optimum checked by exhaustive subsets and independent DP; structured totals independently cross-checked | The long question and all code responses are unfinished. All prose answers have hash-bound scoped self-reviews identifying corrections. All structured answers have exact correct values but fail standalone format due to Markdown fences. No delivered code function exists to execute; no broad or independent quality claim is made. |
| Final cleanup and publication | [Earlier main documentation checkpoint](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/perf-checkpoint-promotion-20260920.json) records verified regional backups and eight-host cleanup at `5e9ce605` | Both native comparisons completed authenticated eight-host cleanup and strict summaries. No hardware workload is active/queued. [Final MTP publication](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-evidence-promotion-20260920.json) records private main `c142d284`, passing release checks (524 passed, one skipped), self-review, pre/final regional checksum/generation backups and fresh eight-host idle checks. Main gains evidence only; research implementation remains preserved at `f097649a`. |

The working comparison is ordinary greedy decoding versus one- and two-draft
native MTP-assisted speculation. MTP is the drafter inside that protocol; an
unverified MTP output stream would change the decoding contract and is not a
separate comparable mode. Each paired request must use the same prompt, capacity,
output budget and greedy policy. Prefill/native bootstrap and cold compilation
must stay separate from sustained decode rate.

The completed representative receipt is bound to immutable execution `dc047933`,
all eight original rank hashes, all 17 program identities, authenticated native
pack and input suite, and eight original cleanup reports. The summarizer validates
token counts/hashes against NPZ and JSONL originals, including EOS/cap semantics.
The pinned tokenizer, private exact oracles and bound prose self-reviews provide
only the answer scopes stated above. Every mode repeats its own token output
exactly; speculative output differs from ordinary on all three prompts.

The [extended input receipt](https://github.com/GianluigiVitale/glm-tpu/blob/9dedce4b4e1d10845146a3ddb6ebccb655868fec/docs/perf/mtp-representative-inputs-extended-20260920.json)
authenticates the executing matched budgets. The original shorter unlaunched set
and all failed/unfinished measurements remain preserved. No favourable prompt
replaces the incomplete long or code responses.

All historical rejected trials, research branches and DB616–621 evidence remain
preserved. Work remains outside frozen `MODEL_SOURCE` `edecdd94`. No resource
provisioning, model switch, environment upgrade or sampled-speculation claim is
needed to complete this objective. A negative native result would retain the
faster qualified ordinary path, with the reason documented from measurements.
