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
python -m tools.equivalence check                      # G1 G1-protocol G3 G4 G6 G7 G9 against tests/golden/data
python -m tools.equivalence check --tier production    # G2 G2-protocol
python -m tools.equivalence check --gates G4,G6-static,G9   # the light gates (allowed while a TPU run is live)
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
tier runs only with `GLM_EQUIVALENCE_PRODUCTION=1`; `-m "not cpu32"` is the light selection: G4,
G6-static, G9 and the data contracts). A missing data file fails; a version mismatch
skips with the reason, or fails with `GLM_EQUIVALENCE_STRICT=1` (set it in CI).

| Gate | What must be identical | Module | Measured on a quiet 240-core host at S0 round 4 (4-vCPU CI roughly 3-6x) |
|---|---|---|---|
| G1 FP-FIX | normalized TPU StableHLO digest, N8 signature, compiler options bound by `jax.jit` and `Lowered.compile` arguments of the 98 fixture-tier programs compiled by the real runtime (every run's programs under their own key); the frozen safety record (checkpoint-verification arguments, graph-consensus probe, memory and HLO admission verdicts, memory admission requests); adapter consistency and v0 cross-check | `programs.py`, `driver.py`, `lowering.py`, `normalize.py`, `verdicts.py` | 200 s incl. consistency (v0 in parallel) |
| G1-protocol | characterization (re-baselined only with a reviewed reason): the load protocol of every fixture run, the production option/builder defaults and the full admission reports | same child as G1 | shared with G1 |
| G2 FP-PROD | the same for the 66 production-tier programs (78 layers; 8,192 / 32,768 / 166,912; batch n=4) | same | 380 s |
| G2-protocol | the same characterization for the production runs | same child as G2 | shared with G2 |
| G3 GOLD | positional leaf digests of the CPU32 execution goldens (load, `generate` and `generate_concurrent` by the real runtime; `generate` of the donated 8,704-slot runtime; every fixture run's load products) | `golden_run.py`, `driver.py`, `fixture.py` | 296 s (the 8,704-slot `generate` ~70 s, the n = 1..3 builds ~18 s each) |
| G4 CKPT-CI | geometry, tensor names, partition specs, placement, the 32 owner-file header SHA-256s, key sets, contracts; the real pack, verify and load code on a tiny checkpoint (loaded arrays' digests and shardings, refusals of tampered inputs) | `identities.py` | 42 s |
| G5 SITE | the same against the real assets, request re-validation, launcher-constant digest | `identities.py` | read-only, minutes |
| G6 IMPORT | every source-tree module and third-party package per stage (controller incl. the real launcher `main`, worker preflight/main, real `_load` + tracing, serving exercise) may only shrink, mapped through `closure_map.toml`; controller JAX-free; static layering scan may only shrink | `import_closure.py`, `closure_map.py`, `controller.py` | 100 s: the `graph` stage builds and traces all six fixture runs through the real compile path, `serving` runs the G9 exercise |
| G6-static | the light part of G6: the controller (with the launcher exercise), worker-preflight and worker-main closures and the static scan, against the same record | `import_closure.py --light` | 3 s |
| G7 TRACE | executed repository functions of the real runtime's load and compile, tracing, CPU composition and the G9 serving exercise, equal through `closure_map.toml` (S1-S3; informational from S4) | `trace_closure.py`, `closure_map.py` | 464 s |
| G9 WIRE | request bytes/`request_sha256`, TokenEvent lines, worker/controller records, the worker's real `preflight` and `_initialize_runtime` (bound arguments, topology binding, mesh axes and device order, refusals, jax configuration and compile environment at runtime construction), the launcher's real `main` (locks, staged bundle, remote commands, worker environment, failure path), resident protocol, HTTP/SSE | `wire.py`, `controller.py` | 11 s |
| G14 SELFTEST | the normalizer detects every sensitivity case and ignores every invariance case | `selftest.py` | 175 s (its CPU32 child builds all six fixture runs) |

Totals from the same runs: `check` (G1 G1-protocol G3 G4 G6 G7 G9) 1,113 s; `check --tier production`
381 s; `check --gates G4,G6-static,G9` 56 s; `pytest tests/golden -m "not cpu32"` 58 s (105 tests).

Heavy gates -- every gate that builds the real runtime or runs fixture-scale programs on the
32-device CPU mesh: G1, G1-protocol, G2, G2-protocol, G3, G6 (its `graph` stage builds every fixture
run), G7, `selftest`, `authenticity` -- refuse while a TPU run is live on the host (`budget.py`, D25, DESIGN
7.5.10): a controller/worker/pack-worker process, a holder of `/tmp/libtpu_lockfile`, or a held
*workload* lock. The two sync locks are not indicators (a five-minute cron backup holds them).
Light gates (G4, whose loader exercise places a few kilobytes on the CPU mesh; G6-static; G9) then
run under `nice 19` with single-threaded Eigen; pytest skips
every `cpu32` test (`test_import_closures` included; `test_import_closures_static` is light).

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
`compile_program` does to the function are all in its text or signature -- together with the XLA
options a production `jax.jit(..., compiler_options=...)` bound to that `Lowered`
(`jit_compiler_options`: they never reach the StableHLO text, but `Lowered.compile` merges them into
what the TPU compiler receives) and the arguments production passed to `Lowered.compile`
(`compile`). Plus one `fp8_table[bits/scale/spec/block]`
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

Besides the programs, the program child records each run's **load protocol** (`runtime`): the
config call `__init__` made and the resulting config, the ordered `phase` names, the admission
requests, the keyword arguments of the faked loader calls (e.g. `verify_file_hashes=True`,
`local_slot_layout=True`, the four local slots), the runtime record's keys and ownership mode, what
production did in the HLO directory (directories, files, free-space probe), the HLO admissions and
consensus calls, and the graph-consensus probe (`RuntimeError: optimized graph differs across
hosts`), and the **production defaults** (`defaults`): every field default of
`Ws32DecoderConfig`, `Ws32PerfOptions`, `RoutedProjectionConfig`, `SparseMlaConfig` and the keyword
defaults of the program builders, so a changed default that the fixture overrides or never
exercises (e.g. `sparse_segment_block`) fails the per-commit check. Both go to the
**characterization records** `G1-protocol` / `G2-protocol` (`load_protocol_{fixture,production}.json`),
not to the frozen G1/G2 fingerprint files: they describe host behaviour and option defaults that
planned stages change on purpose (S1 moves the HLO root to the site file, S2d deletes the knob
classes, S4 renames builders), so they are re-recorded like G6/G7/G9, with a reviewed reason.
Defaults are looked up under their current names through `closure_map.toml` `[functions]`; a
removed or ambiguous class or builder is recorded (`<absent>`, `<ambiguous: n definitions>`),
never raised, so the gate reports it instead of crashing. HLO-directory paths are recorded
relative to the runtime's own `hlo` attribute, so moving the dump root under `/dev/shm` changes
nothing but the root.

What no planned stage may change is kept out of that re-baselined record: the **frozen safety
record** (`safety`, `verdicts.py`) sits in the G1/G2 fingerprint files, is compared by G1/G2 and is
recorded only from the baseline production tree. Per run: the graph-consensus probe verdict
(`refused (RuntimeError)`), the arguments of the checkpoint verification call
(`verify_file_hashes=True`, the four local slots, `local_slot_layout=True`, the site pins), the
memory admission requests (order-insensitive), the programs that pass the HLO admission and the
number of graph-consensus calls. Per tier: the verdicts of production's own `memory_projection` on
fixed synthetic chip statistics around the 512 MiB reserve (one byte below and exactly at the chip
limit, the alias credit, one fuller chip, five invalid accountings) and of `inspect_research_hlo`
on a small synthetic optimized-HLO module (physical expert-8/feature-4 axes with a 4,096-byte
full-pod all-reduce: accepted; 4,100 bytes, an all-to-all, non-physical groups, no collectives:
refused). The frozen record keeps verdicts only (accepted / refused and the exception type, fits
or not); the full reports and messages, whose wording and key names planned stages rename (H11
renames the HLO profile string, WU-R the admission functions), are in `G1-protocol` /
`G2-protocol` (`verdicts`).

* Fixture tier: frozen fixture v1 (8 layers, hidden 1024, 64 experts, 1,536 slots). The one config
  construction in `__init__` is adjusted at the class (`driver._config_injection` wraps
  `Ws32DecoderConfig.__init__`, so a from-import of the class or of `model.geometry` changes
  nothing): the pinned production geometry becomes the fixture geometry and
  `sparse_segment_block=128` is added when `__init__` passes none (the fixture's DSA top-k is 128);
  the protocol records the call as made, with the substituted geometry. Runs: plain
  (1,536); donated at 8,704 slots (production's own rule `capacity > 8,192`: no constant is
  patched, so the run keeps donating wherever S2c moves the rule; the build refuses a run whose
  ownership mode differs from the 181c013e rule); concurrent with n = 1, 2, 3 and 4 at 1,536 (the
  worker compiles `concurrent_size=len(pending)`, 1..4; only the concurrent *guard*
  `request.CONCURRENT_CAPACITY` is relaxed to 1,536, which shapes no program -- a moved guard makes
  the build fail loudly): 98 programs in 6 runs.
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
identical programs (production's lowering digest and signature, the options bound by `jax.jit`,
`Lowered.compile` arguments) and record an identical load protocol, which licenses the abstract
production tier.

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
* Compared with the text and the signature: `jit_compiler_options`, the options bound to the
  `Lowered` by `jax.jit` (`Lowered._lowering._compiler_options_kvs`, a private jax attribute read
  fail-closed: a jax without it makes the fingerprint raise).
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
| k `jax.jit(..., compiler_options=...)` added (byte-identical StableHLO) | sensitivity (bound options) | synthetic | pass |
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
byte-identical to the S0 recording made with the harness's own copy of the loop. Each program's
result is recorded when the call returns, because a donating runtime consumes every state in its
next call: prompt A also runs through the real `generate` of the **donated 8,704-slot runtime**
(`capacity > 8,192`: exclusive state ownership, the long-context branch of every host rule), and
every fixture run (1,536, 8,704 donated, concurrent n = 1..4) records what its own `_load`
produced (`load_by_run`: RoPE table, promoted WK tables, resident BF16 weights, and its
`cache_init` executed at 157), so a value confined to one run cannot hide behind another run's
goldens.

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
lanes A and B equal the sequential runtime's prefill); `donated_prompt_a` (the 8,704-slot
runtime: per-block state, token and health, three decode steps, the 8-token session); `load_by_run`
(count and digest per tree and run); components `decode_fp8_table` and the
60-trial `two_stage_topk` sequence (ties, skew, forced fallback). `batch_decode` is
fingerprint-only (CPU cannot execute its vmapped BF16xBF16->F32 dot), so the batched loop is
stopped at its first call, whose arguments are recorded. Recorded
with `--xla_cpu_multi_thread_eigen=false`; the baseline was produced twice in separate processes,
the second pinned to 4 CPUs, identical (also for the round-4 groups `donated_prompt_a` and
`load_by_run`); for the S0 groups a third run in a jax-only venv on 4 CPUs was identical too.

### Checkpoint identity (G4 / G5, `identities.py`)

A synthetic GLM-5.3-shaped inventory (117,060 base tensors: HF names, dtypes and shapes derived
from the pinned geometry) reproduces the live placement report exactly (`f498b064...`, 520,320
placements, 73,920 destinations), and `build_ws32_runtime_file_plans` with the inventory digest
string pinned (test-only subclass) reproduces **all 32 owner-file header SHA-256s** of the live
manifest and its tensor schema; geometry `c6ccb3f0...`. The live values were copied read-only into
`checkpoint_identity.json` at S0; no fallback to G5 was needed.

The checkpoint code itself runs too (`loader_record`, in the G4 child's 32-device CPU mesh,
kilobytes of scratch data): the real `pack_ws32_runtime_checkpoint` packs a tiny two-layer geometry
that has every dtype and destination family of the production name tree (FP8 projections as U8
bits + F32 `scale_inv`, BF16 norms and embeddings, a full and a shared indexer, a dense MLP, a
routed + shared MoE with its router: 58 tensors per slot, 15 dtype/spec combinations), the harness
writes the seal the pack workflow publishes, the real `verify_ws32_runtime_checkpoint` admits it
twice -- full layout with `verify_file_hashes=True`, and the per-host layout `_load` uses (the four
owned slots only, `verify_file_hash_slots`, `local_slot_layout=True`) -- and the real
`load_ws32_runtime_checkpoint` places every tensor on a mesh built from the synthetic 2x4x4 topology.
Recorded: pack and seal digests, the positional leaf digest, dtype and canonical `sharding.spec`
of every loaded array, the device-to-slot mapping, and the refusals of tampered inputs (one payload
byte flipped: full verification, local verification, and the loader itself after a successful
verification; a foreign slot in the local layout; wrong manifest and topology pins). The bytes a
loader places, the sharding it places them with and every verification step are therefore
compared on every commit (the program gates only fake the loader).

G5 (`site-check`) re-derives the
same facts from the real assets on rank 0 (inventory `813eb5e4...`, checkpoint metadata, the local
owner-file headers byte for byte, topology/mesh from the real captures, golden request
re-validation, a digest of the launcher's site constants) and stores its site-specific
expectations outside Git in `$GLM_TPU_CONFIG_ROOT/equivalence/site_baseline.json`. Since S1a every
site value comes from the untracked site file (`glm_tpu.config.site`, `SiteConfig.load()`); the
launcher-constant record is rebuilt from it under the 181c013e roles (`launcher_constants`), so an
unchanged digest proves the site file resolves to exactly the constants the 181c013e launcher,
worker and model spelled (M4). At 181c013e those site literals (coordinator address, TPU VM name,
zone) were located by their position in the launcher's argv literals (`launcher_site_literals`,
kept for the record), never by value: Git holds neither the values nor any digest derived from
them alone, since a digest of a low-entropy value (an address, a zone, a VM name) is recoverable by
enumeration even when salted with a committed salt. The worker's real `preflight` runs in G9
against a synthetic staged run; the CPU worker `--preflight-only` against a locally staged bundle
with the real site file (`tests/worker/test_local_preflight.py`, `GLM_TPU_TEST_SITE=<site file>`)
completes G5: the site file relocates the run root (`GLM_TPU_RUN_ROOT`) to a scratch directory.

The checkpoint code checks every source URI against the current site's
`storage.allowed_source_uri_prefixes` (S1a). G4's tiny packs keep the source URI they were recorded
with -- derived at S0 from the model source pinned at `181c013e` -- by reading that value from the
baseline commit at run time (`site_fixture.baseline_source_uri`) and installing a synthetic site
that admits its bucket; the harness never spells it.

### Import closure (G6) and executed-function trace (G7)

G6 runs each stage in a fresh interpreter and records **every** module whose file lies in the
source tree, whatever its top-level package (`glm_tpu`, `scripts`, but also `bench`, `tools`,
`tests`, ...; only this harness is excluded), the third-party top-level packages and whether JAX
was imported. Stages: controller (26 modules, no JAX), worker preflight (43) and worker main (139)
import their entry modules and the lazy imports those processes perform, and the controller stage
also runs the real launcher `main` (G9's `controller.launcher_record`), so a lazy import anywhere
on the launch path -- staging, preflight, dispatch, supervision, collection, cleanup -- is recorded
and must keep the controller JAX-free; `graph` drives the real
`OrdinaryRuntime.__init__`/`_load` (and the real compile path) for every fixture run and traces
every program, so a lazy import inside `_load`, `compile` or graph construction is recorded;
`serving` runs the G9 exercise (the real `run_queued`/`generate`, `run_concurrent`/`generate_batch`,
worker `main` with its real `preflight` and `_initialize_runtime`, `resident_loop`,
`resident_controller`, `summarize`, the UI/API handler), so a lazy import in the worker's
preflight or runtime initialization is recorded too. The static scan lists every
`scripts|tools|bench|benchmarks|tests|examples` import inside `glm_tpu` (9 at S0).

The G6 comparison lets every stage closure, its third-party set and the static scan **only shrink**:
a module or package that is new in a stage fails unless it is reviewed; a stage may not start
importing JAX. G7 records the repository functions executed (under `sys.monitoring`) by the real
runtime's load and compile of every fixture run, the tracing of every fixture program, the CPU
golden composition and the G9 serving exercise (launcher `main` with `remote_all`, `idle`,
`stage_bundle` and `cleanup_owned`, worker `main`, `preflight`, `_initialize_runtime`,
the host loops, the controller, the UI/API handler); it must equal the recorded set. Both compare
through the reviewed rename table `closure_map.toml` (`closure_map.py`): `[modules]` maps a moved
module or package prefix to its recorded name (also for G7 entries and the static scan),
`[functions]` a renamed function (with the nested functions, lambdas and methods under its
qualname), `[added]` declares a genuinely new module or executed function and `[removed]` a G7
function that may stop executing -- each entry lands, reviewed, in the commit that moves or renames
the code. The parent package created by a move (e.g. `glm_tpu/optimized/routed/__init__.py`) is
allowed implicitly.

### Wire and characterization goldens (G9, `wire.py`)

Every byte comes from the real code with fakes for the fleet, tokenizer and devices: request
bodies for each profile, sequential and concurrent batches, refusals (the 181c013e profile rejects
a non-ASCII request id; the goldens record that) and both canonical-JSON contracts on non-ASCII
message content; the API's messages-size measure, found by bisection over `glm_tpu.api.convert`
itself (largest accepted one-message content per character class: ASCII, Latin, CJK, astral);
`run_queued` over the real `OrdinaryRuntime.generate` with synthetic device results (TokenEvent
lines, `answer.txt`, report keys and values, phase names, and the prefill block schedule incl. a
114-token tail and a 128+114 prompt), at 8,192 and at the donated long-context capacities 32,768
and 166,912 (the capacity-dependent host rules: warm-up prompt, tail program); `run_concurrent` over the real `generate_concurrent` ->
`batched_runtime.generate_batch` -> `BatchedSession` with synthetic device results (four lanes of
different lengths and budgets, one EOS: lines with `batch_round`, answers, reports, aggregate,
prefill schedule); the refusals of the fleet votes in the same host logic (`fleet_refusals`: a
host whose output digest differs at the output consensus, and a prefill whose contract is invalid
-- the local health vote is false --, for `generate` and for `generate_batch`; `accepted` would mean
the check is gone); `resident_loop` and `resident_controller` (ready file bytes, worker stdin
command and stop bytes, measurement keys); the worker `main` for a sequential request and a
concurrent batch, run with its **real `preflight`** against a synthetic staged run directory
(owner-only request, a source manifest of real repository files, a topology rebinding with eight
synthetic 2x4x4 captures, a controller-resolved synthetic `site.json` the real preflight hashes,
validates and installs) and its **real `_initialize_runtime`** over those captures -- faked are
only the environment marker, the hostname, the checkpoint pins `site_args` binds and the template
check (both read private assets), `jax.distributed`, the device queries and `Mesh` --: the record
keys, the arguments `preflight` binds (context capacity, process id, topology capture root, ...),
the topology binding it authenticates, the `jax.distributed` arguments, the mesh axis names, shape
and device-order digest, the arguments `main` passes to `OrdinaryRuntime` (names and described
values: `context_capacity`, `concurrent_size`, the vote function, the file `save` writes, ...),
and the refusals the real `preflight` and `_initialize_runtime` must produce on inputs with exactly
one defect (deployed-source digest, existing namespace, coordinator port, owner-only modes,
request, binding and site digests, a coordinator other than the staged site's, host mapping), and
what the worker process has set when it constructs
the runtime (`runtime_construction`: every jax configuration option and `XLA_*`/`LIBTPU*`/`TPU_*`/
`JAX_*`/`PJRT_*`/`GLM_*` environment variable that differs from a reference taken at the start of
the G9 child, before any production import; a `jax.config.update` in the worker or at import time
shows up); `summarize()`; the launcher's real `main` (`controller.py`) for a successful request and
for a run whose rank-3 worker exits 1, against a synthetic host -- a tiny committed git repository
that `stage_bundle` archives, synthetic topology captures, temporary run root and lock files, all
named by a synthetic owner-only site file the launcher loads (`site_fixture`) --
with only `gcloud` discovery, `source_identity` (the site launch policy's `git ls-remote`; it must
receive the resolved checkout and the site's `LaunchPolicy`), `pinned_helpers` (the helper blobs
of the pinned commit, which the synthetic tree lacks: the package texts the real snapshot requires
equal, for the staged checkout and pin only; `tests/executor` runs the real one) and the SSH hosts
faked (an in-process emulation answers every remote command; the worker wrapper is executed with
`os.execv`/`os.chdir` captured): the lock calls (workload locks non-blocking, sync locks blocking
and released before dispatch), every remote command line (the remote helper programs by the
digest of their normalized text, with their decoded JSON argument), `helpers.json`, the staged
bundle's members and manifest keys, the preflight and worker command
lines, the worker's `execv` arguments and environment, the controller's files and stdout markers,
the failure path (authenticated cleanup, idle-after, refusal) and `controller_terminal.json`
(`divergent_records`, `collected`; the resident stop and failure paths of the final collection
are covered by `tests/executor/test_multihost_executor.py`). The run directory, interpreter,
site-packages, commit, coordinator and the staged `site.json` digest (its content names the
temporary paths, normalized to `<tmp>` in the bundle members) are recorded as placeholders; and HTTP
through the real UI/API handler with a fake resident
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
* The production tier lowers in about 6 minutes here, not 20-40.

## Known weaknesses

* G1/G2 prove the *lowered* StableHLO that production's own `compile_program` builds, the options
  bound to its `Lowered` by `jax.jit`, and the arguments it passes to `Lowered.compile`; XLA's TPU
  compilation itself is not run. Process-wide compile settings (jax configuration, `XLA_FLAGS`,
  `LIBTPU_INIT_ARGS`) are not part of a program record; G9 records them where they are set (the
  environment the launcher starts the worker with, and the jax configuration and compile
  environment the worker has when it constructs the runtime). A changed
  compiler or libtpu is caught only by the TPU comparison (`compare-run`), not here, and the HLO
  admission parser (`inspect_research_hlo`) never sees a real TPU optimized module on the CPU host
  (its call is checked; its verdicts are characterized on a small synthetic optimized module in the
  frozen safety record -- it has no unit tests at 181c013e -- and covered by the TPU runs).
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
* G4's tiny-pack source URI is read from the baseline commit (`git show 181c013e:...`), so G4 needs
  that commit in the clone (CI: fetch the history, not a depth-1 checkout); without it G4 errors
  loudly, never passes silently.
* G6's stage entry lists and G9's runtime-record locator name 181c013e modules, and the driver
  patches the loader functions in their 181c013e homes (plus the S2a destinations); a later move
  fails closed (the real loader runs on placeholder arguments) until the integrator updates them.
  Renames are handled by `closure_map.toml` and the rename-only re-record below.
* Module-attribute hooks. The fixture config and geometry are injected at the class, but these
  fakes still replace a name in a module, so an import-style refactor that binds the name
  elsewhere bypasses them. Each bypass fails loudly (never a silent pass), and the integrator moves
  the hook with the code:
  `request.CONCURRENT_CAPACITY` (fixture concurrent guard, read by `__init__` through a call-time
  import; a module-level binding makes the n = 1..4 builds refuse "requires 32K");
  `runtime.validate` and `batched_runtime.batch` (G3's relaxed fixture-request validation; a
  bypass makes production validation refuse the 1,536-slot requests); the runtime module's
  `build_ws32_prefill_challenger_program` and `build_packed_decoder_program` (G3's Pallas
  interpret flags; a bypass runs TPU kernels on CPU and crashes); the loader functions in
  `driver.HOMES` ("`_load` no longer calls the faked ..."); `inspect_research_hlo` in
  `driver.ADMISSION_HOMES` (the real parser refuses the stand-in text);
  `bf16_resident._decode_program` (the FP8-table capture; a bypass drops the `fp8_table[...]`
  programs from G1); `multihost_utils.process_allgather` (a from-import binding would see the real
  single-process gather and the graph-consensus probe would record "accepted", which fails the
  frozen G1 safety record).
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
`[Equivalence] Re-baseline <gate> for <reason>` that shows the normalized diff: `record` prints,
for every data file it replaces, the diff of the compared part against the previous file
(`diff.<file>.lines`, `path: old -> new`; added and removed names for name lists), and that diff
goes into the commit message. Pure-refactor stages forbid it; wire goldens change only with an
H-numbered commit. The normalizer changes only in a commit that re-runs G14 and re-records nothing.

`record` enforces this. On a tree whose production paths differ from `181c013e` it refuses G1-G4
and `fixture` (graph and identity goldens come only from the baseline production tree; to add a
field, extract that tree and point `GLM_EQUIVALENCE_SOURCE_ROOT` at it), and it records
G1-protocol, G2-protocol, G6, G7 and G9 only with `--reason` set to exactly one token -- an H
number (`H1`..`H16`), a stage (`S1`, `S1a`, `S2d`, `S4.2b`, ...) or an S5 work unit (`WU-E`, ...);
free text, commit hashes or a word that merely contains hex letters are refused -- written into
the file as a `rebaseline` marker (`tests/golden/test_data_contract.py` requires it). The safety
facts cannot be absorbed by such a re-baseline: they are in the frozen G1/G2 files. The **rename-only**
re-record for a move or rename (S2a, S2b, S2f, S3, S4):

1. in the commit that moves the code, add the reviewed entries to `closure_map.toml`; G6 and G7
   then pass through the mapping (G1, G3, G4 must stay identical anyway);
2. `python -m tools.equivalence record --gates G6,G7 --rename-only --reason S3` re-runs both,
   refuses unless the fresh records pass through the table, and writes them under the current
   names with the marker (`kind: rename-only`, table digest, previous digest);
3. clear the table in the same commit (a stale entry maps a current name to a name the new data
   no longer contain, so the check fails until it is removed).
