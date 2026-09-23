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
python -m tools.equivalence authenticity DIR [DIR ...] [--out FILE]  # programs vs TPU StableHLO originals
python -m tools.equivalence record --gates G1,...      # integrator only; see "Re-baselining"
python -m tools.equivalence budget                     # is a TPU run live on this host?
```

`check` exits 0 only when every requested gate reports `pass`. A gate reports `fail` for a difference
or a missing/unusable data file, `error` when its child crashed or timed out (the remaining gates are
still run and reported), and `skip` when an installed package bound to the gate's data differs from
the recorded version (jax/jaxlib for every gate, plus numpy/ml_dtypes for G3 and torch/safetensors
for G4, whose tiny pack writes its source with them). A skip is a failure unless `--allow-skip` is
given (local convenience only, never in CI).

Pytest: `pytest tests/golden -p no:cacheprovider` (markers `golden`, `cpu32`, `slow`; the production
tier runs only with `GLM_EQUIVALENCE_PRODUCTION=1`). A missing data file fails; a version mismatch
skips with the reason, or fails with `GLM_EQUIVALENCE_STRICT=1` (set it in CI).

| Gate | What must be identical | Module | Measured at S0 (240-core host; 4-vCPU CI roughly 3-6x) |
|---|---|---|---|
| G1 FP-FIX | normalized TPU StableHLO digest and N8 signature of the 98 fixture-tier programs built by the real runtime (every run's programs under their own key); load protocol; production defaults; adapter consistency and v0 cross-check | `programs.py`, `driver.py`, `lowering.py`, `normalize.py` | `check` ~2 min incl. consistency (v0 in parallel) |
| G2 FP-PROD | the same for the 66 production-tier programs (78 layers; 8,192 / 32,768 / 166,912; batch n=4) | same | ~300 s |
| G3 GOLD | positional leaf digests of the CPU32 execution goldens (load, `generate` and `generate_concurrent` by the real runtime) | `golden_run.py`, `driver.py`, `fixture.py` | ~145 s |
| G4 CKPT-CI | geometry, tensor names, partition specs, placement, the 32 owner-file header SHA-256s, key sets, contracts | `identities.py` | 40 s |
| G5 SITE | the same against the real assets, request re-validation, launcher-constant digest | `identities.py` | read-only, minutes |
| G6 IMPORT | every source-tree module and third-party package per stage (controller, worker preflight/main, real `_load` + tracing, serving exercise) may only shrink, mapped through `closure_map.toml`; controller JAX-free; static layering scan may only shrink | `import_closure.py`, `closure_map.py` | 30 s |
| G7 TRACE | executed repository functions of the real runtime's load, tracing and CPU composition, equal through `closure_map.toml` (S1-S3; informational from S4) | `trace_closure.py`, `closure_map.py` | 120 s |
| G9 WIRE | request bytes/`request_sha256`, TokenEvent lines, worker/controller records, resident protocol, HTTP/SSE | `wire.py` | 7.5 s |
| G14 SELFTEST | the normalizer detects every sensitivity case and ignores every invariance case | `selftest.py` | 54 s |

Heavy gates (G2, G3, G7, `selftest`, `authenticity`) refuse while a TPU run is live on the host
(`budget.py`, D25): a controller/worker/pack-worker process, a holder of `/tmp/libtpu_lockfile`, or
a held *workload* lock. The two sync locks are not indicators (a five-minute cron backup holds
them). Light gates then run under `nice 19` with single-threaded Eigen.

## Method

### Programs: built by the real runtime (`programs.py`, `driver.py`)

At `181c013e` production has no program builder: `OrdinaryRuntime._load` builds and compiles its
programs inline. `program_specs` therefore constructs the **real** `OrdinaryRuntime` with its real
`__init__`, which runs the real `_load` (and, for a concurrent runtime, the real
`batched_runtime.compile_batch`). `_load`'s `compile` calls run production's **real compile
path**: `OrdinaryRuntime.compile` -> `compile_program` (`fn.lower(*inputs)`, the StableHLO
original, `lowered.compile()`, the optimized-HLO original, `runner.json`), the graph-consensus
phase, the HLO admission and the memory admission. A compiled program's fingerprint is taken from
the `Lowered` that `compile_program` itself built -- donation, prefill options, the 2,034-token
cache sample, the config `__init__` builds (`host_main_rope_table` included) and anything
`compile_program` does to the function are all in it -- and the arguments production passed to
`Lowered.compile` are recorded with it (`compile`). Plus one `fp8_table[bits/scale/spec/block]`
program per distinct resident-table decoder (compiled by jit dispatch; captured by wrapping
`bf16_resident._decode_program` while the real `bf16_resident_weights` runs, and lowered by the
harness). `driver.py` fakes only what needs a fleet, private assets or the TPU compiler:
`authenticated_inventory` (returns the pinned synthetic GLM-5.3 inventory),
`verify_ws32_runtime_checkpoint` / `load_ws32_runtime_checkpoint` (plans and arrays); while a
program compiles, `jax.stages.Traced.lower` lowers for the TPU platform (what `fn.lower` does on
the fleet) and `jax.stages.Lowered.compile` returns a stand-in executable (runs the program on the
CPU mesh, or returns abstract outputs; zero compiler memory; a stand-in optimized-HLO text); the
HLO admission parser `inspect_research_hlo` (no TPU optimized module exists on a CPU host: a
recorder checks it is handed the stand-in text read back from the HLO directory);
`process_allgather` (stacks the local value; a probe makes one host differ and records that the
real graph-consensus phase refuses it); `stats` (four idle synthetic chips, so the real
`admit`/`admit_memory`/`memory_projection` run); identity votes; every file under `/dev/shm` (an
in-memory overlay: the HLO directory, the originals and `runner.json` are recorded relative to the
runtime's `hlo` attribute and never written; any other file write during the build is refused);
and the free-space probe. `model.require_site`, `model.require_inventory`, `phase`, the binder and
every builder run for real. Every run's programs are recorded under their own key: the program name
qualified by the run that built it (`<name>@<capacity>[+donated][#n<lanes>]`, e.g.
`prefill_128@1536#n2`, `wk_decode@32768+donated`), so a change confined to the donated or to a
concurrent runtime (an option, a donation, a sample shape) cannot hide behind an equal program
recorded from another run; a program equal to an earlier one of the tier keeps its own digests
and refers to that record for its diagnostic summary (`same_as`).

Besides the programs, G1 and G2 compare each run's **load protocol** (`runtime` in the data): the
config call `__init__` made and the resulting config, the ordered `phase` names, the admission
requests, the keyword arguments of the faked loader calls (e.g. `verify_file_hashes=True`,
`local_slot_layout=True`, the four local slots), the runtime record's keys and ownership mode, what
production did in the HLO directory (directories, files, free-space probe), the HLO admissions and
consensus calls, and the graph-consensus probe (`RuntimeError: optimized graph differs across
hosts`). G1 also records the **production defaults** (`defaults`): every field
default of `Ws32DecoderConfig`, `Ws32PerfOptions`, `RoutedProjectionConfig`, `SparseMlaConfig` and
the keyword defaults of the program builders, so a changed default that the fixture overrides or
never exercises (e.g. `sparse_segment_block`) fails the per-commit gate.

* Fixture tier: frozen fixture v1 (8 layers, hidden 1024, 64 experts, 1,536 slots). `model.geometry`
  returns the fixture geometry and the one config construction in `__init__` gets
  `sparse_segment_block=128` when it passes none (the fixture's DSA top-k is 128). Runs: plain;
  donated (production's own rule `capacity > CAPACITY` with `runtime.CAPACITY` lowered to 1,024);
  concurrent with n = 1, 2, 3 and 4 (the worker compiles `concurrent_size=len(pending)`, 1..4, with
  `request.CONCURRENT_CAPACITY` set to 1,536): 98 programs in 6 runs.
* Production tier: GLM-5.3 geometry from the pinned config, no override; capacities 8,192, 32,768,
  166,912 and 32,768 with n=4, donation by production's rule; the checkpoint arrays are built
  exactly as `load_ws32_runtime_checkpoint` builds them (global shape, dtype and
  `NamedSharding(mesh, P(*partition_spec))` of every tensor plan) from the synthetic inventory's
  file plans (whose headers G4 proves equal to the live checkpoint's). 66 programs in 4 runs.

Two modes share one code path: *concrete* (fixture only; the producer programs -- WK, FP8 tables,
cache initializer -- execute on the 32-device CPU mesh exactly as `_load` executes them on TPU)
and *abstract* (`ShapeDtypeStruct` with shardings; a producer's outputs take the shardings its
CPU-compiled executable reports, never executed; `ShapeDtypeStruct.addressable_shards` answers the
per-shard byte probe of the BF16 admission). **Adapter consistency** (enforced by G1): on the
fixture both modes hand all 98 programs identical arguments (shape, dtype, sharding), compile
identical programs (production's lowering digest and signature, `Lowered.compile` arguments) and
record an identical load protocol, which licenses the abstract production tier.

**v0 cross-check** (enforced by G1 and G2 while it can be built): the S0 replica of `_load` with
frozen copies of its constants (`load_programs_v0`, `--adapter v0`) runs in a parallel child and
must fingerprint every program exactly like the real runtime (98/98 and 66/66). It imports
the 181c013e helper homes, so it reports `unavailable` once S2a/S2c move them (S2c retires it).

**Adapter authenticity** (read-only, `authenticity.py`): the production-tier 32,768 programs are
compared with the StableHLO originals the fleet compiled at `181c013e` in the baseline runs B1
(`batch-core`, sequential, 32K) and B2 (concurrent, n=4). Result (re-run with the real runtime
building the programs): **9/9 programs equal**. `cache_init`, `wk_decode`, `wk_promote`,
`batch_cache_init` and `batch_insert` are **byte-identical** to the TPU originals. `prefill_128`,
`prefill_114`, `decode` and `batch_decode` (303/303/228/228 kernel calls) are identical with the
Mosaic bodies masked, and -- the stronger check -- identical with every body **decoded** (Mosaic
bytecode parsed with the TPU dialect and deserialized, exactly as jax re-reads it) and printed
without locations: **1,062/1,062 kernels equal**. The raw bodies differ for two reasons only:
the TPU bodies embed source locations of the staged run directory, and they are serialized at
Mosaic IR version 13 while the CPU host serializes at jax's forward-compatible version 11 (no TPU
backend: `tpu_custom_call.get_ir_version` returns `_FWD_COMPAT_VERSION`). The per-kernel table
goes to a small report outside Git (`authenticity --out FILE`). So the CPU-hosted TPU lowering of
the real runtime's programs reproduces what production compiled, kernel IR included, with no
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
  **N8** the signature (flattened input shape/dtype/canonical spec/donated, output avals, counts) is
  compared separately, so a pytree field reorder fails even though N4 hides names. The spec is
  canonical: trailing unsharded `None` entries are dropped, so `P(None, 'expert')` and
  `P(None, 'expert', None)` -- the same placement, identical lowering -- agree (a builder may spell
  state specs either way); the adapter-consistency and abstract-output keys use the same canonical
  form (`normalize.sharding_key`). Checkpoint partition specs are serialized and stay
  spelling-exact in G4.
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
| n8-spec input `PartitionSpec` spelled with a trailing `None` (same placement) | invariance | synthetic, sharded | pass |
| d-spec input `PartitionSpec` with the sharded axis moved | sensitivity (signature) | synthetic, sharded | pass |
| e routed FP8 projection tiles 256 -> 128 | sensitivity | production Pallas kernel `fp8_routed_projection` | pass |
| f donation removed | sensitivity | synthetic | pass |
| g one constant inside a Pallas kernel body | sensitivity | synthetic Pallas | pass |
| h two independent ops reordered | sensitivity | synthetic | pass |
| i one finished-lane `jnp.where` removed from the batched decode body | sensitivity | fixture `batch_decode` (mutated module copy) | pass |
| j kernel `name=` changed without a rename entry | sensitivity | synthetic Pallas + temporary names module | pass |
| j-mapped the same with a `kernel_renames` entry (patched back) | invariance | same | pass |

### CPU32 execution goldens (G3, `golden_run.py`, `fixture.py`)

Executed by production's own runtime code: the real `__init__`/`_load` load the fixture (the prefill
and decode builders get the two Pallas interpret flags), and prompts A and B run through the real
`OrdinaryRuntime.generate` (identity votes, `process_allgather` faked, request validation relaxed
for the fixture profile) with the compiled prefill/decode programs wrapped to record each block and
step; the batch groups come from a real concurrent (n=4) runtime. Driven this way, every group is
byte-identical to the S0 recording made with the harness's own copy of the loop.

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
`batch_cache_init` and `batch_insert` through the real `compile_batch`; `batch_generate`: the real
`generate_concurrent` -> `generate_batch` of the concurrent runtime over four lanes (prompts A, B
and two short ones) with that runtime's **own** `cache_init`, prefill and `batch_insert` programs
(block states and tokens, the bank handed to `batch_decode`, round-0 TokenEvent lines, and whether
lanes A and B equal the sequential runtime's prefill); components `decode_fp8_table` and the
60-trial `two_stage_topk` sequence (ties, skew, forced fallback). `batch_decode` is
fingerprint-only (CPU cannot execute its vmapped BF16xBF16->F32 dot), so the batched loop is
stopped at its first call, whose arguments are recorded. Recorded
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

G6 runs each stage in a fresh interpreter and records **every** module whose file lies in the
source tree, whatever its top-level package (`glm_tpu`, `scripts`, but also `bench`, `tools`,
`tests`, ...; only this harness is excluded), the third-party top-level packages and whether JAX
was imported. Stages: controller (26 modules, no JAX), worker preflight (43) and worker main (139)
import their entry modules and the lazy imports those processes perform; `graph` drives the real
`OrdinaryRuntime.__init__`/`_load` for every fixture run and traces every program (103), so a lazy
import inside `_load` or graph construction is recorded; `serving` runs the G9 exercise (the real
`run_queued`/`generate`, `run_concurrent`/`generate_batch`, worker `main`, `resident_loop`,
`resident_controller`, `summarize`, the UI/API handler). The static scan lists every
`scripts|tools|bench|benchmarks|tests|examples` import inside `glm_tpu` (9 at S0).

The G6 comparison lets every stage closure, its third-party set and the static scan **only
shrink**: a module or package that is new in a stage fails unless it is reviewed; a stage may not
start importing JAX. G7 records the repository functions executed (under `sys.monitoring`) by the
real runtime's load of every fixture run, the tracing of every fixture program and the CPU golden
composition (500 at S0); it must equal the recorded set. Both compare through the reviewed rename
table `closure_map.toml` (`closure_map.py`): `[modules]` maps a moved module or package prefix to
its recorded name (also for G7 entries and the static scan), `[functions]` a renamed function,
`[added]` declares a genuinely new module or executed function and `[removed]` a G7 function that
may stop executing -- each entry lands, reviewed, in the commit that moves or renames the code.
The parent package created by a move (e.g. `glm_tpu/optimized/routed/__init__.py`) is allowed
implicitly.

### Wire and characterization goldens (G9, `wire.py`)

Every byte comes from the real code with fakes for the fleet, tokenizer and devices: request
bodies for each profile, sequential and concurrent batches, refusals (the 181c013e profile rejects
a non-ASCII request id; the goldens record that) and both canonical-JSON contracts on non-ASCII
message content; the API's messages-size measure, found by bisection over `glm_tpu.api.convert`
itself (largest accepted one-message content per character class: ASCII, Latin, CJK, astral);
`run_queued` over the real `OrdinaryRuntime.generate` with synthetic device results (TokenEvent
lines, `answer.txt`, report keys and values, phase names, and the prefill block schedule incl. a
114-token tail and a 128+114 prompt); `run_concurrent` over the real `generate_concurrent` ->
`batched_runtime.generate_batch` -> `BatchedSession` with synthetic device results (four lanes of
different lengths and budgets, one EOS: lines with `batch_round`, answers, reports, aggregate,
prefill schedule); `resident_loop` and `resident_controller` (ready file bytes, worker stdin
command and stop bytes, measurement keys), the worker `main` record keys for a sequential request
and a concurrent batch together with the arguments `main` passes to `OrdinaryRuntime` (names and
described values: `context_capacity`, `concurrent_size`, the vote function, the file `save`
writes, ...), `summarize()`; and HTTP through the real UI/API handler with a fake resident
(status, headers incl. CSP, body bytes and full SSE streams; `chatcmpl-`, `call_`, uuid ids and
timestamps normalized by regex).

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

* G1/G2 prove the *lowered* StableHLO that production's own `compile_program` builds, and the
  arguments it passes to `Lowered.compile`; XLA's TPU compilation itself is not run. A changed
  compiler or libtpu is caught only by the TPU comparison (`compare-run`), not here, and the HLO
  admission parser (`inspect_research_hlo`) never sees a real TPU optimized module on the CPU host
  (its call is checked; its verdict is covered by its unit tests and by the TPU runs).
* The production tier is abstract: its inputs are derived, not loaded. This is licensed by the
  fixture adapter-consistency check and by the authenticity result above, but the authenticity
  check depends on volatile `/dev/shm` originals (run at S0 and again when the programs moved to
  the real runtime).
* Mosaic kernel bodies are hashed as serialized on the CPU host, i.e. at jax's forward-compatible
  Mosaic IR version (11 in jax 0.10.1), not the fleet's (13); the serialization is deterministic
  and the decoded IR equals the fleet's, but a jax upgrade that moves the forward-compatible
  version changes every body digest (G1/G2 are bound to the jax version anyway).
* G3 digests are bound to jax/jaxlib/numpy versions and, in principle, to the host CPU's
  floating-point code generation (XLA:CPU targets the host ISA). They were reproduced on 4 and 240
  CPUs of this host; a CI runner with a different ISA may need G3 host-only.
* The fixture model degenerates in decode (it repeats one token), so G3's token lists are weak
  signals; the full per-step state digests carry the detection.
* G6's stage entry lists and G9's runtime-record locator name 181c013e modules, and the driver
  patches the loader functions in their 181c013e homes (plus the S2a destinations); a later move
  fails closed (the real loader runs on placeholder arguments) until the integrator updates them.
  Renames are handled by `closure_map.toml` and the rename-only re-record below.
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

`record` enforces this. On a tree whose production paths differ from `181c013e` it refuses G1-G4
and `fixture` (graph and identity goldens come only from the baseline production tree; to add a
field, extract that tree and point `GLM_EQUIVALENCE_SOURCE_ROOT` at it), and it records G6, G7 and
G9 only with `--reason` naming an H number, a stage or a commit, written into the file as a
`rebaseline` marker (`tests/golden/test_data_contract.py` requires it). The **rename-only**
re-record for a move or rename (S2a, S2b, S2f, S3, S4):

1. in the commit that moves the code, add the reviewed entries to `closure_map.toml`; G6 and G7
   then pass through the mapping (G1, G3, G4 must stay identical anyway);
2. `python -m tools.equivalence record --gates G6,G7 --rename-only --reason S3` re-runs both,
   refuses unless the fresh records pass through the table, and writes them under the current
   names with the marker (`kind: rename-only`, table digest, previous digest);
3. clear the table in the same commit (a stale entry maps a current name to a name the new data
   no longer contain, so the check fails until it is removed).
