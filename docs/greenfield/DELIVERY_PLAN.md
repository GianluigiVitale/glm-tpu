# Delivery plan — current engine, official-card quality, then stop

Owner decision2026-09-11; binding specification §26. This supersedes historical
next-step lists. Current GPT-6 Astra High chat only; adversarial self-review,
not independent review. Preserve all original failures and historical reviews.

## Finish line

Deliver the existing native-JAX WS32_2D engine, including DB610's tested dense
correction, at its current speed. Complete allfour128K passkey depths, full256K
E0, official-card benchmark evaluation, usable request/resume/first-token delivery,
DB/archive and authenticated cleanup. No bonus optimization or alternate engine.

Exact incidental continuation and cross-engine intermediate bits are diagnostics,
not delivery gates. The corrected8K run returned passkey881446 correctly, then
diverged at zero-based token11 (` There` versus ` The`). This does not establish
broad quality or justify retroactively marking its old §21 failure PASS.

Keep checkpoint/load/scale integrity, own-score DSA/ties and routing semantics,
causal/cache structure, finite healthy state, existing bounded kernel tests,
topology-local collectives, measured per-chip HBM, truthful timing/provenance,
storage limits and8/8 cleanup. Stop rounding-history reconstruction, repetitive
cleared-code reviews and redundant acquisitions. Every new test must answer a
remaining delivery question or guard a changed path.

## Benchmark authority — not an older local build

Use the GLM-5.2 column on the [official GLM-5.2-FP8 model card](https://huggingface.co/zai-org/GLM-5.2-FP8).
Immutable revision/content and retained-byte reconstruction are recorded in
[HF_GLM52_CARD_PIN.md](HF_GLM52_CARD_PIN.md); no older local model score is a target.
Verified2026-09-11: GPQA-Diamond91.2 and AIME2026 99.2. Reasoning footnotes specify
temperature1.0, top_p0.95 and maximum generation163,840tokens; math tasks have
an answer-format system prompt and GPT-5.5 medium judging. These are target
protocol facts, not evidence that the port already implements them.

Before candidate outputs, pin the card revision/content, dataset revisions/full
item sets, prompts/template, seeds/sampling/stops, generation cap, judge/scorer,
aggregation/sample count, uncertainty and material-score-deficit rule, plus
projected runtime/token budget. Resolve unspecified protocol details honestly.
A different judge, shorter cap, greedy generation or a small subset is a disclosed
deviation, not matched public-card parity. No tolerance fitting after results.

Primary tractable card suite: GPQA-Diamond and AIME2026, using existing
`bench/benchmarks.py`, `extract.py`, `run_bench.py` and `provenance.py` through a
thin native generation adapter. GSM8K/old local scores may aid diagnosis but
CANNOT replace the official-card baseline. Do not rerun legacy solely to create
a comparison baseline. Agentic/tool benchmarks require distinct harnesses and
are not silently claimed covered by these two tasks.

The local card-mode harness already discloses a deterministic extractor in place
of a model judge; this gap must be resolved or explicitly approved as a deviation
before claiming equivalent evaluation. Paid external judging is not authorized
by this plan alone. Preserve every miss/truncation; no completed-item cherry-pick.
Long generation caps can require substantial wall time at current decode speed;
publish that budget before launch, rather than silently reducing the workload.

## Execution order

1. COMPLETE: history diagnostic212325 at290ea13a and preserved originals.
   All32owners completed331calls/host and reproduced both original observations;
   DB612/8clean/regional archive verified (receipt `../artifacts/prefill-history-db612-sealed-20260911.json`). No successor capture
   merely to eliminate recorded differences. Freeze enforcement through collection.
2. Implemented explicit §26 worker/sealer task profile, preserving historical §21
   behavior. Saved8K offline assessment verifies16 exact regional generations,
   task/14DSA observations/cache/state on8/8. Receipt below; original failed
   DB/SUCCESS unchanged. This is NOT card parity or a new protected performance
   seal. No rerun solely for prose; repeat only a missing integration/safety test.
3. Admit batched long-capacity HLO/allocation and actual32-chip HBM. Combine
   capacity preflight with protected real-workload startup where equally safe.
   Run128K depths1.0/0.0/0.05/0.95, requiring exact extracted gold; then full
   262,144-token E0/256-step measurement under §23.5. Old serial passes are not
   changed-path coverage. E0 has no correctness oracle: quality comes separately.
4. Run the registered official-card suite through the native engine, reusing
   resident weights/executables across sequential isolated requests. Diagnose a
   material deficit against card targets, not a wording difference. Insufficient
   statistical/protocol evidence cannot be labelled parity.
5. Verify request/resume and actual delivery, report separate input/cache/prefill/
   warmTTFT/coldload/compile/decode/requestwall, publish reproducible commands and
   limitations, seal DB/archive and leave8/8clean. Then stop.

This document changes policy, not executed results. Long-context integration,
the native benchmark/request adapter, protocol registration and actual benchmark
scores remain open. No new128K/256K success, quality-parity claim or ETA is implied.

## Concrete long-context integration boundary (source inspected 2026-09-11)

Implemented locally: `validation/ws32_delivery_prefill.py:long_plan` binds all
five existing L7/L8 labels; `ws32_rolled_prefill_compile.prepare` accepts an
explicit `long_context_label` with `full_canonical=True`. It constructs actual
2310-leaf abstract inputs and capacity-sized page tables using the unchanged
production builders. Short preparation is unchanged. CPU tests cover both long
capacities, all five plans, full decode headroom and no payload/device placement.
This is preparation only: numerical launch remains deliberately refused until
capacity-specific HLO/memory and §26 worker/sealer integration are complete.
Both main/tail production graphs lowered successfully on CPU for each capacity;
E0's two B128 graphs have identical raw fingerprints. Reproduction method and
all hashes are in `../artifacts/prefill-delivery-long-preparation-20260911.json`.
No optimized TPU HLO, memory fit or numerical result follows from that check.

Reuse the existing `run_short_decoder_ws32.py` / `seal_short_decoder_ws32.py`
and detached `run_short_decoder_ws32.sh` workflow. The §23.5 passkey/E0 loaders,
gold extraction, observer, cache checks, tracing and cleanup already exist.
Do not create a second engine or another legacy reference run.

The current batched refusal is explicit in `validation/ws32_prefill.py`:
`require_batched_profile` rejects any long context. Merely deleting that refusal
would be wrong: `ws32_prefill_admission.py` also fixes short plans, budgets, graph
pins and original memory sizes. Its rolled/canonical HLO checks contain 16-page
cache and 8192-row RoPE shapes. Those are short-profile assumptions, not model
requirements. Add explicit capacity-aware admission while preserving old defaults.

Retain the canonical dense correction and all existing program options. Reuse
`ws32_rolled_prefill_compile.prepare` for abstract metadata-only inputs; its
currently fixed plan and `[1,16]` block-table shape must follow actual capacity.
Use existing protected compile/memory publication, preserve all graph originals
before a refusal, and inspect the whole shape-dependent inventory once. Neither
short HLO hashes nor old serial HBM establish larger batched allocations.

Long workload plans are fixed by the sealed inputs: 127,363 tokens / capacity
131,072 for all four depths; 262,144 tokens / capacity 262,656 for E0. Keep
physical B128 and bounded masked tails, with no serial fallback. Both old and
proposed cache generations, repaired-index storage, exact decode weights and
resident code must fit actual per-chip memory with the existing reserve before
dispatch. Use the same first real workload for runtime capacity checks where safe.

Read-only production-config calculation (CPU, no payload/backend execution):
128K is 995 full B128 calls plus 3 live tail rows; E0 is 2047 full B128 calls
plus 128 live tail rows. KV plus BOTH index-cache payloads are 1,811,939,328
bytes/chip at 128K and 3,630,956,544 at E0; host-main-RoPE adds 16,777,216 and
33,619,968 bytes/chip respectively. These exclude weights, output generations,
scratch and code. In particular E0's live tail cannot use physical B114; use
the existing B128 program, not new model math or a truncated prompt. A geometry
calculation is not measured HBM admission or a claim that the full state fits.

For §26 short-task reassessment, retain original failing records and use a new
explicit contract in BOTH producer and sealer. Do not rename ORACLE_MISMATCH to
SUCCESS, silently bypass correctness, or equate a correct passkey with card quality.
For long runs, retain §23.5's passkey/E0 distinctions and full E0 sample count.
The existing within-engine observer checks selected-row order/range/tails; it
does not independently recompute scores for all unselected keys. Retain source/
kernel selection evidence and describe that scope accurately.

## §26 saved8K task assessment — 2026-09-11

`../artifacts/prefill-delivery-s26-saved8k-assessment-20260911.json`
SHA256 `10ef7c050d110d5e165a9c0d9d2d79e2016076c64c9807d61bbc4f3e26f9597b`.
Reproduce read-only from the repository root:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. /home/gianl/vllm-env/bin/python scripts/greenfield/assess_ws32_delivery_saved8k.py
```

Reads16 generation-qualified JSON/NPZ originals from US-CENTRAL2 into memory;
no checkpoint/model execution, cloud mutation, retained payload copy or DB change.
All8 return881446 using the pinned tokenizer/extractor. Each has14 selected-row
DSA order/tie/count/producer checks and a valid compact cache/state witness.
Unselected score rows are NOT recomputed; frozen kernel/source tests retain that
obligation. This does not certify full cache values, long capacity, broad quality
or protected performance. Later prose mismatch remains in each diagnostic.

Distinct profile `ws32_b128_b114_8k_cap8192_delivery_s26_v1` keeps original model
options, graph/source guards and1GiB reserve. It records its contract in worker,
sealer and DB identity; never claims verified raw-token count or cross-oracle
DSA equality. Historical§21 profiles still require their original criteria.
Next: capacity-specific actual HLO/HBM and protected long workload integration.
