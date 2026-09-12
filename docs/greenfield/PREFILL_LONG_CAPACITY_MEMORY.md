# Long-prefill capacity blocker — 2026-09-11

## Latest — DB613: compiler OOM cleared, runtime budget not cleared

The single changed graph completed on all8hosts and sealed as DB613 at00:44:37Z
2026-09-12. Run greenfield_fp8_ws32_owned_state_prefill_compile_20260912T003527251082976Z,
pin988fdc0bc1fbef2b880e9796034c71f25bed23d3. All32owners/normal-root8clean,
boot/device identities unchanged. No checkpoint payload, WK or model executable
calls. Compile274–305s/host; worker/collector486s is not inference timing.
127regionalobjects/2,396,580,852B, no new checkpoint. Exact receipt:
../artifacts/prefill-owned-state-db613-sealed-20260912.json.

All8 original compiler analyses agree, bytes per chip:

| Arguments | Outputs | Aliases | Scratch | Code |
|---:|---:|---:|---:|---:|
|28,319,278,080|3,630,979,584|3,630,978,560|4,960,770,048|111,742,976|

Optimized6462ada514db90a60f326c3492ff7b08dd11fb97a5bf0d5d3a474a0da97088b7
has entry aliases for EXACT state tuple leaves2..13. Even assuming all alias
credit, args+outputs-alias+scratch+code =33,391,792,128B. With1GiBreserve that
exceeds the observed33,014,398,976B limit by1,451,134,976B, BEFORE additional
resident arrays or companion executables. This is a conservative planning
calculation, not measured execution peak or an explanation of XLA's scheduled
allocation. Compilation success does not authorize runtime dispatch. Do not
subtract scratch/copy counts by assumption or reduce the reserve to call it fit.

Two visible fullKV copy sites remain in the actual optimized original:
copy.30310(get-tuple-element.247172) and
copy.33780(bitcast_dynamic-update-slice_fusion.1), both
bf16[78,513,64,640],3,277,946,880B logical. Two sites do NOT prove two simultaneous
allocations. They identify the remaining whole-cache proposal/commit mechanism.

Decision: no unchanged compile or numerical launch. Next bounded pending-row
transaction: preserve old cache values for failed proposals, retain only this
window's changed rows until the global commit, then update the owned cache.
Reuse actual layer writers/dual-index repair/health/commit and original state-only
ownership; no precision or throughput search. Required CPU cases: multi-window
causality, page/stripe boundaries, live tails/NaNpadding, invalid token/count/
health rollback and no consumed-handle reads. Changed production E0 RAW/optimized
allocation then decides, without repeating128K or loading fullweights for compile.
Later numerical admission must still include all-live buffers, companion code,
reserve and actual32-chip peaks. Existing no-donation guard stays unchanged.

Source frozen through seal. Localfleet replay and all77controller-ledger object
generations/sizes/CRCs checked; sub-MiB remote originals SHA-read back, larger
local graphs rehashed. Remote SUCCESS/ledger bytes match, DB613 env_json fullpin
and NULL score/correct/latency checked. Self-review only. Localfree5.514GB now
below6GiB: restore reviewed archived copies before hardware, never these needed
rank0 graph originals or primaryDB. This updates the older next-step notes below.


## 2026-09-12 — protected compiler route wired, hardware result still pending

Use the existing run_fp8_matmul_microbench.sh with
GLM_GREENFIELD_FP8_MATMUL_KERNEL=ws32_owned_state_prefill_compile.
The distinct one-graph mode preserves original RAW/optimized/alias/allocation,
eight-host journals and generation-qualified collection; SQLite quality/latency
remain NULL.116CPU tests93.07s pass, including actual production metadata/RAW
and fixture CLI→publication→32-owner collector→DB. Real source guards are tested
separately from synthetic lifecycle fixtures. No model/source arithmetic change.
Receipt ../artifacts/prefill-owned-state-compiler-route-local-20260912.json.

Budget900sworker/1080sSSH,192MiB/rank (1.5GiB originals), fresh6GiB controller
floor,8GiB wholearchive planning including controller/DB/late failure copies.
This is an allowance, not predicted retained payload or another checkpoint.
Eleven exact-generation archived local DBcopies evicted402,063,360B; cloud0,
primaryDB unchanged,6,930,128,896B free. Recovery mappings in delivery-db-copy-
review-20260912.json; application receipt alongside it.47CPU eviction tests pass.
Self-review only; fresh root8/8idle. Persist/mirror and fresh guards before ONE
changed E0 compile. It does not authorize numerical execution; original128K
graphs remain available and are not repeated.


## 2026-09-12 continuation — ownership candidate, not a TPU fix yet

`scripts/greenfield/ws32_prefill_owned_state.py` introduces an explicit
`ws32-prefill-consumed-state-argument2-v1` ABI by wrapping the unchanged original
JIT with `donate_argnums=(2,)`. A distinct program type prevents presenting this
as the old non-donating profile. No model arithmetic, precision, weight layout,
checkpoint, original source guard or worker/sealer acceptance was changed.

Eight CPU tests pass178.05s. The real canonical eight-layer B128/B114 programs
match the original before/after a populated-prefix transaction and three-live-row
masked tail; invalid token, oversized count and false incoming health preserve
old cache/frontier VALUES in the returned refused state. Input cache handles
become invalid; tokens/count/weights/WK/RoPE remain readable and byte-identical.
CPU compiler alias_size_in_bytes=2,360,350, NOT a TPU memory-saving measurement.
An actual host-loop test with small donating CPU fixture arrays completes three
calls without reading consumed state; a health failure stops after one call.
Its model/memory/fleet are fixtures, not a production admission bypass.

Caller inspection: graph_inputs retains current state until dispatch, then checks
only result.state; previous result/current aliases are overwritten without reading
them. The initial memory record stores metadata, not JAX buffers. Worker compile
placeholders are already cleared before creating the numerical state. Future
integration must keep these lifetimes, consume only the returned state after a
refusal, and treat a thrown dispatch as terminal (never retry donated inputs).
No production runner is opted in yet: its plan and memory record still correctly
describe/refuse nonzero aliases under the historical no-donation contract.

The new abstract preparation selects ONLY E0 B128 from the existing production
builders. Next route that single changed graph through the existing protected
compiler/journal/fleet with a distinct identity and original allocation capture.
Do not repeat the128K baseline acquisition or load weights to answer this.
Actual TPU HBM fit, alias allocation/rollback copies, all-live reserve, long HLO,
worker/sealer admission and long numerical/quality/serving tests remain open.

Development-test corrections: the first CPU test used nonexistent private
`Traced._args_info` (5pass/1fail60.71s); fixed to public compiled args_info.
The first E0 lowering test incorrectly required assigned `tf.aliasing_output`:
installed JAX defers partitioned ownership via `jax.buffer_donor`. The corrected
test checks the exact donated entry-argument set in either representation, not
actual alias savings. No TPU test or model change occurred in either correction.

Corrected production E0 lowering plus11 historical bad-budget/alias refusal tests
PASS12 in89.56s; reuse registry PASS4 in1.96s. Real2310-leaf metadata, all inputs
abstract, payload/placement/compilation forbidden. RAW20,859,926B, SHA256
`55c3d5775eb5630a61fd8e5cd54caa30e447cf92c3ed09dcafaae0217a63cba1`.
JAX entry donors are exactly state leaves2..13 (12 inputs); no other argument.
Receipt `../artifacts/prefill-owned-state-cpu-20260912.json` binds tests/source.
Adversarial self-review checked lifetimes, original-source guards and explicitly
unwired production admission. This is not independent review or runtime fit.

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
