# PLAN — greenfield gates

The authoritative details, stop conditions, and Definition of Done are in
`docs/glm-tpu-revolution.md`. This is only the execution index.

1. **Gate A — topology/runtime:** immutable plans; real topology; explicit PP8/PP16 local groups;
   HLO linter; dependent collective distributions; device-resident stage chains; clean provenance.
2. **Gate B — checkpoint:** plan-aware final-layout manifest and packer; byte/checksum reconciliation;
   FP8 scale ownership; direct loader; corruption refusal; measured per-chip HBM.
3. **Gate C — one layer:** exact dense, full-DSA, IndexShare, and adversarial MoE equivalence against
   captured legacy-oracle inputs, with topology-local collectives only.
4. **Gate D — full short context:** independent 78-layer 2K/8K decoder; exact tokens/DSA/cache/state;
   no repeated 32-chip layer collective; fresh trace, wall result, and memory report.
5. **Gates E/F — useful/strong base:** first `<=200 ms` and `>=4.5` wall tok/s; then target
   `<=125 ms`/`>=8 tok/s`, stretch `<=100 ms`/`>=10 tok/s`.
6. **Gate G — plan adjudication:** protected identical-condition PP8, PP16, and WS32 comparison or
   specified evidence-backed WS32 rejection; promote the fastest correct plan.
7. **Long-context gates:** protected four-depth 128K smoke and protected 256K E0 with DB/archive and
   authenticated zero-work cleanup.
8. **Gate H — effective throughput:** only after base stability, add exact multi-token verification/
   speculation and report base versus accepted effective tok/s separately.

Never skip a gate because a later test passes. CPU, synthetic, or HLO results prove mechanisms—not
full-model performance.
