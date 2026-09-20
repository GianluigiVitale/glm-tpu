# MTP objective completion audit — 2026-09-20

The objective is still open. This audit maps its requirements to evidence;
a compiled graph, a verifier-only estimate, or a short token match cannot
substitute for accepted speculative wall throughput and representative outputs.
Current execution status belongs in [goal.md](../../goal.md).

| Requirement | Evidence available | Remaining boundary |
|---|---|---|
| Native layer-78 weights and upstream semantics | [Source audit](mtp-source-audit-20260919.json), [position/IndexShare contract](MTP_STATE_CONTRACT_20260919.md), [exact owner placement](mtp-native-placement-20260920.json), [completed native acquisition](mtp-native-acquisition-20260920T013227Z.json) | Trained native loading and accepted execution completed in the [first fleet run](tpu-real-native-mtp-20260920T015817Z.json). No independent upstream trained-native parity is claimed. |
| Multi-row target verification with causal prefix state | [M8 CPU evidence](mtp-m8-rowwise-cpu-20260919.json); `test_speculative_commit.py` independently checks all prefixes across owner/page boundaries with permuted physical pages, EOS/budget limits and a single-owner refusal | Two/three-row model comparisons have an explicit floating-point boundary. Matching selected test tokens is not bitwise cache equivalence. Five-row verification remains rejected by its numerical envelope. |
| Real verifier economics and bottleneck evidence | [Initial synthetic result](tpu-mtp-verifier-20260919T214702Z.json), [batched trace](tpu-trace-mtp-verifier-batched-20260919T223024Z.json), [trained M8 result](tpu-real-mtp-verifier-m8-20260919T235646Z.json) | Perfect-acceptance estimates omit native drafting/refresh, rejected work, votes and delivery. They are not accepted output throughput. |
| Native drafting, acceptance and guarded delivery | [Projection](mtp-projection-cpu-20260920.json), [target hidden export](mtp-prefill-export-cpu-20260920.json), [native component checks](mtp-native-components-cpu-20260920.json), [complete CPU session checks](mtp-native-session-cpu-20260920.json) | The real worker completed bootstrap, recurrent drafting, target verification and accepted-history refresh. Long outputs diverge from ordinary at token index 6; this is not a token-exact replacement. Greedy only; no sampled acceptance/RNG claim. |
| Paired DB610 and longer prose/code/reasoning/structured runs | Ordinary [long-question baseline](tpu-real-long-question-20260920T005757Z.json); [representative inputs and repeats](mtp-representative-inputs-20260920.json) are prepared | The first run completed DB610 and the long question. The representative suite controller started after its authenticated cleanup; repeat results and completed-answer checks remain pending. |
| Accepted output tok/s, acceptance by position, component/wall costs, latency and HBM | Worker and strict summary implement these fields; [ordinary sustained result](tpu-real-long-question-20260920T005757Z.json) is 14.4141 wall tok/s | First long measured rates: ordinary 14.2902, R2 13.0927, R3 12.7682 wall tok/s. Keep ordinary; representative repeat ranges remain outstanding. The 25% criterion has not been met. |
| Answer correctness | Exact TSP oracle retained privately; representative scheduling optimum checked by exhaustive subsets and independent DP; structured totals independently cross-checked | All three modes in the first long run ended during reasoning. Final-data checks do not prove generated Python or explanatory prose correct. Prose/code review must remain explicit and scoped. |
| Final cleanup and publication | [Earlier main documentation checkpoint](perf-checkpoint-promotion-20260920.json) records verified regional backups and eight-host cleanup at `5e9ce605` | The first native comparison completed authenticated eight-host cleanup. The representative suite is active and its final cleanup remains mandatory. Main received documentation/evidence only; no experimental engine deployment occurred. Final MTP evidence must be reviewed for eligible publication after the experiment finishes. |

The working comparison is ordinary greedy decoding versus one- and two-draft
native MTP-assisted speculation. MTP is the drafter inside that protocol; an
unverified MTP output stream would change the decoding contract and is not a
separate comparable mode. Each paired request must use the same prompt, capacity,
output budget and greedy policy. Prefill/native bootstrap and cold compilation
must stay separate from sustained decode rate.

Live representative update at 05:29 UTC: both prose and code repeats completed in all
three modes. Both speculative modes are slower, and each mode's repeated
output is identical. Scoped self-review found technical corrections in each
distinct prose answer; this is not independent or model-wide quality assessment.
Code R3 is 4.3–4.6% faster in both paired repeats, below the working 25% criterion;
R2 is slower. All six code responses exhausted 7,168 tokens during reasoning
with no finished answer, and speculative outputs differ from ordinary.
Structured comparisons, strict completed-fleet
aggregation, final cleanup and eligible publication are still outstanding.
The executing suite uses the [extended matched budgets](mtp-representative-inputs-extended-20260920.json);
the earlier input receipt in the table remains preserved as preparation history.

All historical rejected trials, research branches and DB616–621 evidence remain
preserved. Work remains outside frozen `MODEL_SOURCE` `edecdd94`. No resource
provisioning, model switch, environment upgrade or sampled-speculation claim is
needed to complete this objective. A negative native result would retain the
faster qualified ordinary path, with the reason documented from measurements.
