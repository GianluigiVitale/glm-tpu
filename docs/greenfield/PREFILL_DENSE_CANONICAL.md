# Dense-only canonical row placement — corrective candidate

Authority: goal.md and specification §25. Status: unwired, CPU mechanism only.
This is a correctness intervention, not renewed performance tuning or an accepted
live32 engine. The frozen DB603 baseline is preserved.

## Evidence and decision

DB606 sealed the fixed18-call norm diagnostic on all32 chips. Receipt:
`../artifacts/prefill-dense-norm-db606-sealed-20260909.json`. Its retained-byte
checks reproduced DB605 fields and DB604 cache endpoints before attribution.
Every captured norm field is byte-equal between wide and narrow first128 cases.
Moving identical normalized32-row segments to rows0:32 of the same completed
physicalB128 dense suffix reproduces all128 own-narrow output/health comparisons.
The wide comparison differs by1200 unique BF16 output words, matching DB605.
This localizes the observed difference to dense suffix placement/co-batch
realization; it does not prove a hardware defect or cause of generated token11.

The prior independent reviewer and main analysis selected a direct corrective
candidate instead of another intermediate-projection capture. Current source
receives its own independent adversarial review before deployment.

## Implementation and unchanged boundaries

`glm_tpu/greenfield/kernels/ws32_prefill_dense_canonical.py` reuses
`ws32_prefill_mlp_mapped` with original dense weights and no MoE branch. Four
uniform device-side scan iterations place each32-row segment in a physical128
row input, zero the rest, preserve live-output/finite-health checks and assemble
the original row order. Empty segments still take the uniform path. B114 pads
to128 then crops; its numerical identity is not inherited from B128.

Only the first three dense layers may opt in. Keep B128 host stride, rolled
attention/DSA/cache schedule, MoE panels, FP8/BF16/FP32 boundaries, checkpoint,
decode and numerical contracts unchanged. No serial token-prefill fallback.
Four dense calls replace one for these layers: disclose and measure this
correctness cost; no performance promise follows from CPU equivalence.

## Shortest decisive admission sequence

1. CPU32 compare real scan candidate with four original physicalB128 calls:
   distinct owners, full/partial/empty rows, B114 tails, poisoned padding and
   nonfinite live health. Candidate+reuse tests5PASS23.60s; candidate is one
   pytest case with multiple shapes and an independent original-full-row reference.
   Isolated-owner NaN atrow33 checks later-tile health placement;
   this is synthetic CPU evidence, not actual TPU row-placement reproduction.
2. Reuse the reduced dense0/1 builder, selected checkpoint loader, WK programs,
   protected workflow, retained originals and guards. Distinct candidate profile
   must preserve actual changed HLO/memory before any refusal. Compare one wide
   candidate first128 against retained NARROW row outputs and endpoint caches
   on all32 owners. Do not demand identity to the intentionally corrected wide
   baseline or rerun historical reference/capture campaigns.
3. Only after bounded reproduction: own2K regression including B114 tail, then
   own8K exact raw tokens/cache/DSA under §21. A continued8K failure stops
   escalation and requires evidence-based diagnosis; no authority to broaden
   the intervention to MoE or sweep precision/windows.
4. After own8K: long-capacity HLO/measured HBM, four128K depths, full256K E0,
   serving/resume/actualTTFT, DB/archive/cleanup. Historical results stay historical.

No TPU candidate run,8K fix, new throughput result or complete engine is claimed.
