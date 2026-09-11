# Long-prefill capacity blocker — 2026-09-11

## Outcome

Both frozen 128K graphs compiled; 256K failed during TPU compilation, before any
checkpoint payload load or model call. All eight hosts report the same HBM OOM.
No long numerical run, runtime-fit proof, performance result or protected SUCCESS.
This is a concrete capacity defect to fix under §26, not a throughput campaign.

Run `greenfield_fp8_ws32_delivery_long_prefill_compile_20260911T232546214124179Z`,
pin `e8119f6f194e061f242927e88dbe259dec53bdbb`. Wrapper finished FAILED at23:45:23Z;
process returned1. Normal and root accelerator/libtpu censuses are8/8 clean.
91 regional objects /2,394,149,857B preserve partial originals and diagnostics.
No checkpoint created. Failure occurred before benchmark DB accounting; DB id is
null, not a fabricated passing or numerical row. No historical record changed.

Verified receipt: `../artifacts/prefill-delivery-long-compile-oom-20260911.json`,
SHA256 `08205198675bce0dd828648946f128ce15b70290ef6574787ae66b6f30c9676f`.
Eight generation-bound runner/log/journal triples were downloaded and hashed;
all worker ledger generations/sizes/CRCs checked. Large HLOs were not downloaded
again. Controller rank0 retained HLOs were separately hashed against the records.
Both compiled graphs' RAW/optimized hashes and memory reports agree across8hosts;
local slot bindings and journal memory owners cover32 devices. Cleanup originals
were downloaded, byte-compared and boot-bound. This is current-chat self-review,
not independent review. An initial read-only assessment mistook the global8x4
mesh field for local IDs and refused before writing; corrected assessment uses
`local_device_slots` and the journal's actual devices.

## Actual compiler reports

All byte values below are per chip. Compiler memory is not numerical peak HBM.

| Graph | Arguments | Outputs | Scratch | Code | Alias |
|---|---:|---:|---:|---:|---:|
| 128K B128 | 26,483,416,576 | 1,811,960,832 | 2,862,960,640 | 106,096,128 | 0 |
| 128K B114 | 26,483,416,576 | 1,811,960,832 | 2,889,717,760 | 110,083,072 | 0 |

Rank0 compile seconds265.859/296.564. The sum of one row's allocations is not a
full runtime budget: both resident executables, all live arrays and1GiB reserve
must be included. Do not infer128K execution permission from successful compilation.

256K has the pinned RAW graph but NO optimized executable or completed memory
analysis. XLA's rounded binary-unit report says:

> Used 33.57G of 30.75G hbm. Exceeded hbm capacity by 2.82G.

It reports arguments26.37G, outputs3.38G sharing0B with arguments, program3.81G,
and HLO temporary3.80G including760.65M fragmentation. The largest temporary is
the entire KV cache: `bf16[78,513,64,640]`,3.05G,
`copy.30310 = copy(copy.35042)`. Do not add the separately reported reservation
twice, treat rounded report values as exact bytes, or invent an E0 optimized HLO.

The saved128K B128 analogue supplies inspectable dataflow:

- ENTRY parameter tuple index2 → `get-tuple-element.249370` → `copy.35021`.
- `copy.35021` feeds layer0's cache read and the original-state rollback tuple.
- `copy.30289 = copy(copy.35021)` feeds the proposed full-stack scatter.
- The accepted branch also contains full-stack `copy.33759`.

These are full `bf16[78,256,64,640]` copies, not weights or small DSA candidates.
Source `runtime/ws32_batched_prefill.py` uses an undonated JIT; the numerical
runner explicitly donates only the historical serial path, not this batched path.
The original array/rollback contract explains why input/output ownership matters;
it does not establish that adding donation alone will remove enough memory.

## Smallest next action — capacity, not optimization

1. Inspect the existing batched-state caller's aliases/lifetimes and memory reader.
   Reuse existing JAX builders and protected compiler route. A default-off
   state-only donation wrapper is the first bounded candidate, not weight donation,
   cache quantization, smaller prompt, reserve reduction or a new checkpoint.
   CPU tests must establish only state is consumed, caller never reads donated
   buffers again, and healthy/failed calls preserve the declared cache/rollback
   semantics. Historical no-donation profiles stay unchanged.
2. If that candidate is sound, compile ONLY the changed worst-capacity graph from
   abstract inputs and preserve its alias/allocation report. No repeated baseline,
   numerical full-model load or assumption that alias bytes are guaranteed savings.
   Before dispatch, require the actual simultaneous allocation budget plus reserve.
3. If ownership transfer cannot fit, use the existing cache-writer/commit analysis
   to design bounded pending row updates rather than another full-cache generation.
   That is a larger, separately tested change; do not bypass rollback or health.
   No blind sequence of flags or precision/arithmetic searches.
4. Adapt explicit long-capacity HLO and worker/sealer profiles using the retained
   originals and actual changed graphs. Then allfour128K passkeys, full256KE0,
   original-HF-card benchmarks and request/resume delivery. No claim is closed here.

Additional residency warning: DB610's saved `batched_prefill_memory.rank0.json`
accounts25,896,458,240B resident versus24,768,984,576B active inputs:1,127,473,664B
outside the latter. This is short-run evidence, NOT measured long residency or
proof those bytes are disposable. Identify owners before releasing anything;
defer genuinely decode-only materialization if safe, preserving required WK repair.

Storage: localspace was6,345,682,944B after failure, below the fresh6GiB launch
floor. Restore only reviewed exact-recoverable copies before another launch.
Keep these128K originals for analysis; no cloud deletion or full-size safety copy.
