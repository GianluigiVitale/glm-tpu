# PLAN — greenfield gates

CURRENT2026-09-11: §26 and `docs/greenfield/DELIVERY_PLAN.md` govern. Finish current
128K/256K, official Hugging Face card benchmarks (NOT old local scores) and usable
delivery. Exact incidental continuation is diagnostic, not a gate. Keep integrity,
own-score semantics/HBM/provenance protections. Historical descriptions below do
not override the current delivery sequence. No new rounding-history campaign.

Continue autonomously to full project completion under §18 amended by §24/§25.
OWNER FREEZE2026-09-09: accept DB603 speed; no further throughput tuning.
Original depth0.05 sealed DB575; do not launch another serial128K/256K campaign.
Rolled78-layer own2K DB603 sealed:31.950s/63.661prompttok/s,2.055x DB597;20/20tokens,
own DSA/cache checks and32HBM/8trace/8cleanup. Changed score/order/cachebits, not8K proof.
Next THIS path's own8K, long-capacity HLO/HBM, fourdepth L7/L8 and serving/TTFT.
HANDOFF has current evidence. DB573–575 remain serial references.10K/500 and stronger
decode targets no longer block completion; numerical/quality/protection gates remain.

The authoritative details, stop conditions, and Definition of Done are in
`docs/glm-tpu-revolution.md`. This is only the execution index.

1. **Gate A — topology/runtime:** immutable plans; real topology; explicit PP8/PP16 local groups;
   HLO linter; dependent collective distributions; device-resident stage chains; clean provenance.
2. **Gate B — checkpoint:** plan-aware final-layout manifest and packer; byte/checksum reconciliation;
   FP8 scale ownership; direct loader; corruption refusal; measured per-chip HBM.
3. **Gate C — one layer:** exact dense, full-DSA, IndexShare, and adversarial MoE equivalence against
   captured legacy-oracle inputs, with topology-local collectives only.
4. **Gate D — full short context (CLOSED 2026-09-05 under spec §21, see §21.6):** independent 78-layer 2K/8K decoder; exact tokens; within-engine
   exact DSA tie order; cross-oracle DSA agreement exact or boundary-explained against an FP64
   reference with a systematic-bias check (spec §21); exact cache/state structure with bounded
   values; no repeated 32-chip layer collective; fresh trace, wall result, and memory report.
5. **Gates E/F — useful/strong base:** first `<=200 ms` and `>=4.5` wall tok/s; then target
   `<=125 ms`/`>=8 tok/s`, stretch `<=100 ms`/`>=10 tok/s`.
6. **Gate G — plan adjudication (CLOSED 2026-09-05, spec §22):** protected identical-condition PP8 vs
   WS32 at 2K (DB563 vs DB553, WS32 2.00× faster, both exact), WS32 8K (DB567); PP16 rejected with
   evidence (§22.3); WS32_2D promoted.
7. **Batched prefill and long-context gates (§24/§25, IN PROGRESS):** freeze accepted DB603,
   finish its own8K numerical proof, then allfour L7 depths/full L8 and serving/TTFT.
   No performance-search prerequisite. Historical serial §23.3 Step B (DB 568/569) and Step C
   capacity measurement at 131,072/262,656 (DB 571/572) CLOSED; §23.5 L7 protected four-depth 128K
   passkey — depths 1.0/0.0/0.05 CLOSED (DB573–575), depth0.95 outstanding; §23.5 L8 protected 256K E0
   outstanding on the serial reference; those old runs do not certify changed prefill.
   L7/L8 claim nothing raw-token or cross-oracle exact: no legacy capture exists at
   these lengths, so L7 stands on the extracted passkey and L8 has no correctness oracle at all.
8. **Gate H — effective throughput:** only after base stability, add exact multi-token verification/
   speculation and report base versus accepted effective tok/s separately.

Never skip a gate because a later test passes. CPU, synthetic, or HLO results prove mechanisms—not
full-model performance.
