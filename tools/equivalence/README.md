# Graph-equivalence harness

`tools/equivalence` proves that a refactor commit leaves the engine's behaviour byte-identical to
the recorded baseline (`181c013e`): the TPU device programs, their CPU numerics, the checkpoint
and format identities, the import structure and the serving wire formats. It never touches a
TPU: every process runs with `JAX_PLATFORMS=cpu`, TPU programs are *lowered* for the TPU platform
on the CPU host and never compiled or executed. Baseline data live in `tests/golden/data/*.json`
(digests and small summaries only); `tests/golden/test_*.py` are thin pytest wrappers.

## Commands and gates

```bash
export JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1
python -m tools.equivalence check                      # G1 G3 G4 G6 G7 G9 against tests/golden/data
python -m tools.equivalence check --tier production    # G2
python -m tools.equivalence selftest                   # G14 (run on every commit touching this directory)
python -m tools.equivalence site-check [--record] [--requests DIR]   # G5, rank 0 only, fleet idle
python -m tools.equivalence compare-run RUN --golden DIR [--golden DIR] [--out FILE]
python -m tools.equivalence diff --program decode@1536 [--against REV] [--tier fixture|production]
python -m tools.equivalence authenticity DIR [DIR ...] # v0 adapter vs TPU StableHLO originals
python -m tools.equivalence record --gates G1,...      # integrator only; see "Re-baselining"
python -m tools.equivalence budget                     # is a TPU run live on this host?
```

Pytest: `pytest tests/golden -p no:cacheprovider` (markers `golden`, `cpu32`, `slow`; the production
tier runs only with `GLM_EQUIVALENCE_PRODUCTION=1`).

| Gate | What must be identical | Module | Measured at S0 (240-core host; 4-vCPU CI roughly 3-6x) |
|---|---|---|---|
| G1 FP-FIX | normalized TPU StableHLO digest and N8 signature of the 21 fixture-tier programs | `programs.py`, `lowering.py`, `normalize.py` | 33 s lowering; `check` 65 s incl. adapter consistency |
| G2 FP-PROD | the same for the 27 production-tier programs (78 layers; 8,192 / 32,768 / 166,912; batch n=4) | same | 282-287 s |
| G3 GOLD | positional leaf digests of the CPU32 execution goldens | `golden_run.py`, `fixture.py` | 79 s |
| G4 CKPT-CI | geometry, tensor names, partition specs, placement, the 32 owner-file header SHA-256s, key sets, contracts | `identities.py` | 40 s |
| G5 SITE | the same against the real assets, request re-validation, launcher-constant digest | `identities.py` | read-only, minutes |
| G6 IMPORT | repository modules per serving stage; controller JAX-free; static layering scan may only shrink | `import_closure.py` | 28 s |
| G7 TRACE | executed repository functions of the fixture composition (S1-S3; informational from S4) | `trace_closure.py` | 102-107 s |
| G9 WIRE | request bytes/`request_sha256`, TokenEvent lines, worker/controller records, resident protocol, HTTP/SSE | `wire.py` | 7.5 s |
| G14 SELFTEST | the normalizer detects every sensitivity case and ignores every invariance case | `selftest.py` | 54 s |

Heavy gates (G2, G3, G7, `selftest`, `authenticity`) refuse while a TPU run is live on the host
(`budget.py`, D25): a controller/worker/pack-worker process, a holder of `/tmp/libtpu_lockfile`, or
a held *workload* lock. The two sync locks are not indicators (a five-minute cron backup holds
them). Light gates then run under `nice 19` with single-threaded Eigen.

## Method

### Programs: the v0 adapter (`programs.py`)

At `181c013e` production has no program builder: `OrdinaryRuntime._load` builds and compiles its
programs inline. `load_programs` replicates `_load` from the bound checkpoint on, call for call:
`build_wk_programs(..., P(None,'feature'), P(None,'feature'), contract=...)`, `bf16_resident_weights`
(its per-table decoders are captured by wrapping `bf16_resident._decode_program`, giving one
`fp8_table[bits/scale/spec/block]` program per distinct key), the host RoPE table, `cache_init`
with the sample length 2034, `prefill_128`/`prefill_114` with the production options (key tile
512, MLP window, rolled prefix, expert panels, paired sort, sorted merge, canonical dense) and
`jax.jit(fn, donate_argnums=(2,))` above 8,192 slots, and `decode` with `donate_argnums=(1,)` above
8,192. The batch programs come from calling the **real** `batched_runtime.compile_batch` with a
recording runtime. The recorded key is the program name, qualified by capacity, `+donated` and `#n4`.

* Fixture tier: frozen fixture v1 (8 layers, hidden 1024, 64 experts, 1,536 slots, segment 128),
  one run plain, one donated, one concurrent (n=4): 21 programs.
* Production tier: GLM-5.3 geometry from the pinned config; the raw checkpoint arrays are built
  exactly as `load_ws32_runtime_checkpoint` builds them (global shape, dtype and
  `NamedSharding(mesh, P(*partition_spec))` of every tensor plan), from the synthetic inventory's
  file plans (whose headers G4 proves equal to the live checkpoint's). 27 programs.

Two modes share one code path: *concrete* (fixture only; the producer programs -- WK, FP8 tables,
cache initializer -- execute on the 32-device CPU mesh exactly as `_load` executes them on TPU)
and *abstract* (`ShapeDtypeStruct` with shardings; a producer's outputs take the shardings its
CPU-compiled executable reports, never executed). **Adapter consistency** (recorded in
`fingerprints_fixture.json`): on the fixture both modes hand all 21 programs identical arguments
(shape, dtype, sharding), which licenses the abstract production tier.

**Adapter authenticity** (S0, read-only, `authenticity.py`): the production-tier 32,768 programs
were compared with the StableHLO originals the fleet compiled at `181c013e` in today's baseline
runs B1 (`batch-core`, sequential, 32K) and B2 (concurrent, n=4). Result: **9/9 programs equal**.
`cache_init`, `wk_decode`, `wk_promote`, `batch_cache_init` and `batch_insert` are **byte-identical**
to the TPU originals; `prefill_128`, `prefill_114`, `decode` and `batch_decode` are identical with
the Mosaic kernel bodies masked (kernel count 303/303/228/228, kernel names and operand/result
types still compared). The TPU bodies differ only because they embed source locations of the
staged run directory. So the CPU-hosted TPU lowering reproduces what production compiled, with no
platform-attribute differences.

### Lowering and normalization (`lowering.py`, `normalize.py`; nothing else is rewritten)

* **N1** `jit.trace(*args).lower(lowering_platforms=("tpu",))` inside `location_free()`, which patches
  `jax._src.interpreters.mlir.source_info_to_location` to return `ir.Location.unknown()` (Pallas
  looks it up through the module at call time, `mlir.py` through its globals), sets
  `jax_include_full_tracebacks_in_locations=False`, installs the TPU v4 `tpu_info` entry for the
  CPU device kind and, from S4.2b, patches kernel names back through `kernel_renames.toml`.
* **N2** `lowered.as_text(debug_info=False)`. **N3** strip residual `loc(...)`/`#loc` (defensive).
  **N4** delete `jax.result_info`/`jax.arg_info` strings. **N5** `module @jit_main`; private
  functions `@f0, @f1, ...` in definition order. **N6** each `tpu_custom_call` body becomes
  `<mosaic:sha256=...>` of its decoded bytes (bodies are location-free under N1, so the kernel code
  stays compared); `kernel_name` and all other fields verbatim. **N7** everything else byte-compared.
  **N8** the signature (flattened input shape/dtype/`str(spec)`/donated, output avals, counts) is
  compared separately, so a pytree field reorder fails even though N4 hides names.
* A structural summary (op histogram, collectives with replica groups, `(kernel, body)` table,
  parameter count, bytes) is stored for diagnosis only; `diff` prints the unified diff of the
  normalized text between a git revision and the worktree.

### Mutation self-test (G14, `selftest.py`)

| Case | Kind | Program | S0 |
|---|---|---|---|
| i+ii relocated tree copy, 10 blank lines prepended to a kernel module (`sparse_attention.py`) and a layer module (`bf16_resident.py`) | invariance | fixture `decode` | pass |
| iii result NamedTuple field renamed | invariance | synthetic | pass |
| iv `jax.named_scope` renamed (plain JAX) | invariance | synthetic | pass |
| iv-pallas `jax.named_scope` renamed inside a Pallas kernel | **sensitivity** (see findings) | synthetic | pass |
| v outer and nested jitted functions renamed | invariance | synthetic | pass |
| vi explicit Pallas `name=` equal to the implicit name | invariance | synthetic Pallas | pass |
| vii explicit `name=`, Python kernel body renamed | invariance | synthetic Pallas | pass |
| n1-control same kernel at another line: equal under N1, different without it | invariance + control | synthetic Pallas | pass |
| a RMS epsilon 1e-5 -> 1e-6 | sensitivity | fixture `decode` | pass |
| b FP32 accumulation -> BF16 | sensitivity | synthetic | pass |
| c dot precision default -> highest | sensitivity | synthetic | pass |
| d two input pytree fields swapped | sensitivity (signature) | synthetic | pass |
| e routed FP8 projection tiles 256 -> 128 | sensitivity | production Pallas kernel `fp8_routed_projection` | pass |
| f donation removed | sensitivity | synthetic | pass |
| g one constant inside a Pallas kernel body | sensitivity | synthetic Pallas | pass |
| h two independent ops reordered | sensitivity | synthetic | pass |
| i one finished-lane `jnp.where` removed from the batched decode body | sensitivity | fixture `batch_decode` (mutated module copy) | pass |
| j kernel `name=` changed without a rename entry | sensitivity | synthetic Pallas + temporary names module | pass |
| j-mapped the same with a `kernel_renames` entry (patched back) | invariance | same | pass |

### CPU32 execution goldens (G3, `golden_run.py`, `fixture.py`)

Frozen fixture v1 reproduces the RNG call order of the historical
`tests/greenfield/runtime/ws32_prefill_cpu_fixture.fixture` exactly but returns
`{checkpoint tensor name: array}` bound through the production name-based binder;
`fixture.json` records that all 235 leaves equal the historical fixture's, for panel and
non-panel geometry, and that the geometries are equal (the GLM-5.2 -> GLM-5.3 config swap is
inert). Recorded groups: fixture checkpoint; WK decode/promote; resident BF16 weights; RoPE tables
(1,536 / 8,192 / 32,768 / 166,912); `cache_init(157)`; prompt A (157 = B128 + B114 tail) state,
token and health after each block; prompt B (114, one block); prompt C (refused prefill on a
finished state: caches and frontier unchanged, health false, token -1); three packed decode steps;
an 8-token `PackedRequestSession` loop with identity votes (tokens and TokenEvent JSONL digest);
`batch_cache_init` and `batch_insert` through the real `compile_batch`; components
`decode_fp8_table` and the 60-trial `two_stage_topk` sequence (ties, skew, forced fallback).
`batch_decode` is fingerprint-only (CPU cannot execute its vmapped BF16xBF16->F32 dot). Recorded
with `--xla_cpu_multi_thread_eigen=false`; the baseline was produced twice in separate processes,
the second pinned to 4 CPUs, identical; a third run in a jax-only venv on 4 CPUs was identical too.

### Checkpoint identity (G4 / G5, `identities.py`)

A synthetic GLM-5.3-shaped inventory (117,060 base tensors: HF names, dtypes and shapes derived
from the pinned geometry) reproduces the live placement report exactly (`f498b064...`, 520,320
placements, 73,920 destinations), and `build_ws32_runtime_file_plans` with the inventory digest
string pinned (test-only subclass) reproduces **all 32 owner-file header SHA-256s** of the live
manifest and its tensor schema; geometry `c6ccb3f0...`. The live values were copied read-only into
`checkpoint_identity.json` at S0; no fallback to G5 was needed. G5 (`site-check`) re-derives the
same facts from the real assets on rank 0 (inventory `813eb5e4...`, checkpoint metadata, the local
owner-file headers byte for byte, topology/mesh from the real captures, golden request
re-validation, a digest of the launcher's site constants) and stores its site-specific
expectations outside Git in `$GLM_TPU_CONFIG_ROOT/equivalence/site_baseline.json`. The CPU worker
`--preflight-only` against a locally staged bundle is deferred to S1 (it needs the relocatable run
root that the site file introduces).

### Import closure (G6) and executed-function trace (G7)

G6 imports each stage's entry modules (and the lazy imports the stage performs) in a fresh
interpreter: controller (26 modules, no JAX), worker preflight (43), worker main (139), graph
construction (97). The static scan lists every `scripts|tools|bench|benchmarks|tests|examples`
import inside `glm_tpu` (9 at S0); the check fails if it grows. G7 records the 446 repository
functions executed by the adapter (concrete mode), the tracing of every fixture program and the
CPU golden composition, under `sys.monitoring`.

### Wire and characterization goldens (G9, `wire.py`)

Every byte comes from the real code with fakes for the fleet, tokenizer and devices: request
bodies for each profile, sequential and concurrent batches, refusals (the 181c013e profile rejects
a non-ASCII request id; the goldens record that) and both canonical-JSON contracts on non-ASCII
message content; `run_queued` over the real `OrdinaryRuntime.generate` with synthetic device
results (TokenEvent lines, `answer.txt`, report keys, phase names), `run_concurrent` (lines with
`batch_round`), `resident_loop` and `resident_controller` (ready file bytes, worker stdin command
and stop bytes, measurement keys), the worker `main` record keys, `summarize()`; and HTTP through
the real UI/API handler with a fake resident (status, headers incl. CSP, body bytes and full SSE
streams; `chatcmpl-`, `call_`, uuid ids and timestamps normalized by regex).

### TPU comparison (`compare_run.py`)

Port of the proven private comparator (including its resident `output_directory` fix), with the
golden runs as arguments and hash/id-only output. On today's baselines: B1 10/10 and B2 4/4
identical to the goldens.

## D27: TPU lowering without libtpu (CI capability)

Checked at S0: a fresh venv with only `jax==0.10.1 jaxlib==0.10.1 numpy==2.3.5 ml_dtypes==0.5.4`
(pip added `scipy` and `opt_einsum`; **no libtpu**) produced the fixture-tier fingerprints with the
identical tier digest `aacc12d0...` (all 21 programs), in 34 s. TPU-platform lowering, including
Pallas/Mosaic kernel serialization, needs only jaxlib. **G1 can run in CI.** The same venv pinned
to 4 CPUs also reproduced every G3 group.

## Findings at S0

* A `jax.named_scope` *inside a Pallas kernel body* is not a location: Mosaic lowering emits
  `tpu.trace_start(message=<scope>)` into the kernel, so renaming it changes the device program
  (case `iv-pallas`). Such scopes are frozen in pure-refactor commits (design 7.5.3 expected
  invariance). Plain-JAX scopes are invariant.
* An implicit Pallas kernel name follows the Python function name (renaming the body changes the
  kernel), an explicit `name=` pins it (cases vi/vii) -- the D5 procedure is sound.
* The production tier lowers in about 5 minutes here, not 20-40.

## Known weaknesses

* G1/G2 prove the *lowered* StableHLO; XLA's TPU compilation is not re-run. A changed compiler or
  libtpu is caught only by the TPU comparison (`compare-run`), not here.
* The production tier is abstract: its inputs are derived, not loaded. This is licensed by the
  fixture adapter-consistency check and by the byte-level authenticity result above, but the
  authenticity check depends on volatile `/dev/shm` originals and was run once, at S0.
* G3 digests are bound to jax/jaxlib/numpy versions and, in principle, to the host CPU's
  floating-point code generation (XLA:CPU targets the host ISA). They were reproduced on 4 and 240
  CPUs of this host; a CI runner with a different ISA may need G3 host-only.
* The fixture model degenerates in decode (it repeats one token), so G3's token lists are weak
  signals; the full per-step state digests carry the detection.
* G6's stage entry lists, G7's module mapping and G9's runtime-record locator name 181c013e
  modules; renames (S2f/S3/S4) require a reviewed, rename-only re-record by the integrator.
* G9's HTTP stream digests assume the fake resident completes in one observation; they were
  stable across repeated runs but depend on the server's polling structure.
* G5 is only as strong as the rank-0 assets; the other seven hosts are covered by the worker's
  own load-time verification during a TPU run.
* Kernel components `fp8_grouped_matmul`, `sparse_mla_partial` + merge, `sparse_mla_decode` and
  `fp8_panel_matmul` have no stand-alone G3 entries yet; they are covered end to end by the prefill
  and decode goldens (interpret mode) and by G1/G2 body hashes.

## Re-baselining

`tests/golden/data` is written only by the integrator (`record`). Graph goldens (G1/G2/G3) never
change in this refactor. Any data change is a dedicated commit titled
`[Equivalence] Re-baseline <gate> for <reason>` that shows the normalized diff; pure-refactor
stages forbid it; wire goldens change only with an H-numbered commit. The normalizer changes only
in a commit that re-runs G14 and re-records nothing.
