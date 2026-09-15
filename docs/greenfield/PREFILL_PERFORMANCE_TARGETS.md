# Prospective prefill targets — 2026-09-09

Authority: specification §24 and the owner's requested 10,000 prompt tokens/s.
Registration-basis commit `e87d7702ce27103e88b61825b9410bdebb3c26c1` (documentation).
Measured runtime pins: DB597 `2b3177d69d0ff91923c4aff22837cd70e497692e`;
DB598 `6d72e5d3e3dd7b62d77f46ef336e0c3c05ddcb6c`.
Machine-readable registration: `../../configs/prefill-performance-targets-v1.json`.
This registration precedes the next candidate timing run. Earlier trials are NOT
retrospectively preregistered. Bind this file/config's committed bytes and the
candidate's own code/config/topology/compiler hashes in subsequent experiments.

## Final objective versus forecast

Keep **10,000 prompt tokens/s as the requested final performance target**, not a
forecast of achievable performance. Neither DB598 nor theoretical compute alone
establishes feasibility. We must not replace this target with an easier measured
result, or claim completion at the intermediate milestone below. If subsequent
evidence requires changing the final objective, report it to the owner for an
explicit decision, not a post-failure threshold adjustment.

| Workload | Prompt tokens | Final request-prefill ceiling | Final warm local TTFT ceiling |
|---|---:|---:|---:|
| Each of four L7 passkey depths | 127363 | 12.7363 s | 13.7363 s |
| Full L8 E0 | 262144 | 26.2144 s | 27.2144 s |

The one-second TTFT increment is an engineering delivery allowance, NOT a measured
model-delivery result. DB598's 0.130 ms synthetic callback cannot certify it.
Final acceptance still requires actual first-token delivery and all §21/§23.5,
HLO, memory, profiler-free wall, fresh eight-host traces, archive/DB and cleanup
protections. Base decode minimum/strong/stretch targets are unchanged. A short
2K speedup or a synthetic 32-row kernel cannot meet a full-prompt criterion.

## Measurement boundaries (no hidden initialization or cache hits)

Weights and all serving executables must be resident before the warm request.
Start with the workload's tokenized input ready in host memory and NO reusable
prefix state. Request-prefill includes input placement, fresh KV/index state
initialization, all prompt blocks, repairs, health checks required for serving,
host/device synchronization and the final prompt head. End only when its state
and first-token computation are ready. Warm local TTFT ends when an in-process
consumer acknowledges the actual first token. Include any extra decode needed
for that token; never count the same head computation twice.

Record device prefill separately. Tokenization and client-network transport are
outside this local boundary and must be disclosed, not advertised as measured
client TTFT. Cold load/compile, harness-only oracle observers, profiling and
sealing are separate. Prefill XPlanes are separate from unprofiled wall samples;
decode traces do not attribute prefill. Preserve DB597's historical boundary and
disclose any difference from the new delivery instrumentation before comparing.

## Component allocations and the size of the implementation gap

These are prospective wall allocations needed to reach the requested target,
not predicted speedups or component performance claims. Each phase includes its
own transfers and synchronization; remaining overhead must not double-count them.
Integrated request wall is authoritative even if components overlap under tracing.

| Component | L7 allocation | L8 allocation | Baseline/budget rationale |
|---|---:|---:|---|
| Fresh cache initialization | 1.0 s | 1.5 s | DB598 observed maxima 8.065/8.899 s: requires a separate initialization improvement; not free |
| Input placement and host orchestration | 0.5 s | 0.5 s | Sub-ms block placements accumulate; bulk placement is unintegrated |
| DSA scoring, exact selection/merge and its communication | 2.5 s | 6.0 s | Current B32 integral ≈1772/6895 s: roughly709×/1149× improvement required, NOT promised |
| Linear projections, MoE and their communication | 6.0 s | 12.0 s | Useful linear arithmetic 9.992/20.567 PF; requires ≈1.67/1.71 useful PF/s, excluding padding/format work |
| Selected sparse attention and its communication | 2.0 s | 4.0 s | Useful attention arithmetic 2.811/5.809 PF; requires ≈1.41/1.45 useful PF/s |
| Remaining work and contingency | 0.7363 s | 2.2144 s | Repairs, heads and otherwise unaccounted work must fit the residual budget |
| **Request-prefill total** | **12.7363 s** | **26.2144 s** | Actual token counts / requested 10K tok/s |

The DSA integral is a planning estimate from six synthetic B32 cases, not actual
whole-model timing or a hard lower bound; current full-model prefill uses B17.
Scorer, selection and communication contributions are not isolated by DB598.
The useful-FLOP counts in the retired prefill cost-model notes (recoverable through the [curation ledger](../curation/README.md)) are
incomplete physical work estimates. Weight traffic, owner imbalance and exactness can dominate regardless
of useful FLOPs. These allocations expose why a small merge change alone cannot
establish 10K. Do not spend hours running a predictably failing long prompt merely
to reconfirm that gap.

## Intermediate milestone — explicitly NOT completion

Use 500 request-prefill tokens/s as a nonfinal integration checkpoint:
L7 ≤254.726 s / local TTFT ≤255.726 s; L8 ≤524.288 s / local TTFT ≤525.288 s.
Component budgets: DSA 75/200 s; projections/MoE/selected-attention 125/250 s;
remaining transfers/host work 30/45 s; initialization 11/12 s; contingency
13.726/17.288 s. The initializer budgets exceed the observed maxima; DSA still
needs ≈24×/34× reduction against the synthetic planning integral. No component
or final-model achievement is implied. Do not lower this milestone after failure
or use passing it to claim §18; it only directs whether expensive integration is
worthwhile while the final target remains open.

## Next bounded experiment and stopping rules

Candidate `sorted_local_merge=True`, default OFF, in `kernels/prefill_dsa.py`:
only replace local two-list merges with a vectorized exact half-merge. Preserve
B32, key512, top2048, F32 queries/head weights, BF16 keys, default scorer precision,
paired global position sort and the same four expert8 exchange groups. Preserve
DB598 as control. No scoring, larger query/key tiles, panel or communication-tree
change is bundled. An unchanged baseline rerun or cache-overhead remeasurement is
not needed; an in-run matched control is allowed only to answer a stated comparison
question. Reuse original case/analytic witness and HLO/memory/publication machinery.

Before timing: original output score bits/positions/counts/ties and all32 health
owners must pass, including full-prefix ties; actual candidate HLO must retain
exact candidate-only physical groups and fit the existing memory ceilings. Stop
at the first mismatch, unknown memory or transport issue. Measure complete DSA
call p50/p99 at the same six prefixes; no component-to-model tok/s conversion.
Record all cases, not just winners. A useful local candidate should reduce the
two final-prefix p50s by at least10% without >5% regression at either midpoint;
these are engineering selection thresholds, not statistical confidence bounds.
Five samples cannot establish tight tail statistics. A marginal/inconsistent
result remains unpromoted; no full-model reload to resolve that ambiguity.

Larger key tiles, wider query batching, specialized local selection, global merge
and expert panels remain separate candidate mechanisms. Rank them by measured
remaining cost after this discriminator, not by repeated correctness archaeology.
