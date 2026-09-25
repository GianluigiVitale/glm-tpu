# Testing

Every test runs on CPU. `tests/conftest.py` forces `JAX_PLATFORMS=cpu` and refuses
any other platform, so no test can initialize a TPU. Fake math inside a CPU test
is never TPU validation; hardware evidence lives in the sealed receipts under
`docs/artifacts/` and `docs/release/`.

## Tiers

For a reviewer with only the source archive and an already installed Python 3.12
CPU environment, this small subset needs no checkpoint, cloud credentials,
private run directories or Git history:

```bash
JAX_PLATFORMS=cpu python -m glm_tpu info
JAX_PLATFORMS=cpu python -m pytest -q \
  tests/entrypoints/cli/test_main.py \
  tests/engine/test_request.py \
  tests/executor/test_multihost_executor.py \
  tests/executor/test_fleet.py \
  tests/worker/test_tpu_worker.py \
  tests/utils_/test_io_utils.py \
  tests/config/test_site.py \
  tests/runner/test_tpu_runner.py \
  tests/entrypoints/cli/test_ask.py
```

It checks metadata/import isolation, request integrity and capacity refusals,
controller identity/SSH/failure gates and delivery/deadline behavior with
synthetic CPU results. It does not measure model speed or answer quality. The
full release check below additionally needs the documented environment and
full private checkout; some selected historical validators read local assets.
Installing dependencies is separate from this offline path; see
[INSTALLATION](INSTALLATION.md). No package installation or network access is
performed by these commands.

| Tier | Command | Scope |
|---|---|---|
| Release check | `JAX_PLATFORMS=cpu python tools/check_release.py` | `tests/release/`, the user-request path, native host runtime, request/transport/memory tests, the DB485 compile-only validators; plus `git diff --check`, `doctor`, content audit, frozen-source check, `compileall` and the isolated wheel install |
| Whole tree | `JAX_PLATFORMS=cpu python -m pytest -q -rs tests scripts/analysis bench` | Every retained test module (kernel CPU32 subprocess tests included); several hours on one host |
| Curation ledger | `JAX_PLATFORMS=cpu python tools/curation_inventory.py` | Ledger consistency for every tracked file (not semantic review) |

Use the documented interpreter (`~/vllm-env/bin/python`, Python 3.12). Run from
the repository root. Many `tests/greenfield/kernels` and `runtime` tests spawn a
subprocess with `--xla_force_host_platform_device_count=32` and compile small
eight-layer WS32 graphs on CPU; they are slow but deterministic.

## What the suite covers

- `tests/greenfield/kernels/` — every frozen `glm_tpu/greenfield/kernels` module
  (Pallas kernels under interpret mode, references, WS32 prefill/decode pieces)
  against independent unsharded references on forced 32 CPU devices.
- `tests/greenfield/runtime/` — the WS32 decoder, batched prefill, request
  session, sampled request and PP8 decoder/prefill HLO contracts; CPU32
  composition of windows, tails and rollback.
- `tests/greenfield/hlo/` and `benchmarking/` — the HLO parser/linter, graph
  inspectors and structural proofs on synthetic HLO and, where the originals are
  present, on sealed compiler graphs.
- `tests/greenfield/validation/` — admission registries, sealer control paths,
  evidence transport, memory accounting and the Gate-D adjudication loader.
- `tests/release/` — the user controller/worker/transport/result/archive,
  host operations, import boundaries and the curation ledger checker.
- `scripts/analysis/`, `bench/` — profiler readers, the disk watchdog and the
  offline tests of the SHA-pinned legacy harness modules.

## Tests that need local sealed evidence

Some modules replay sealed compiler graphs, oracle captures or run archives that
live outside Git on this host (`/home/gianl/glm-run`, `/home/gianl/gcs-models`,
`/dev/shm/glm-ws32-runtime`, `/home/gianl/gate-d-runs`, the canonical checkout
and `/home/gianl/vllm-build-a30addc`). They validate original bytes by SHA before
use; they never fabricate evidence.

- Modules that **skip** when the local copy is absent are the batched/rolled HLO
  proofs in `tests/greenfield/benchmarking/`, the DB609/paired-location replays,
  `test_ws32_pallas_real_layer.py`, `test_ws32_decoder.py`,
  `test_callback_executable_class.py`, the `/dev/shm`-manifest preparation
  tests in `tests/greenfield/hlo/`, the long-context oracle tests and the sealed
  C=512 sealer replays. `pytest -rs` lists every skip with its reason.
- Modules that **fail** when their evidence is absent (by design, so a missing
  original is not silently passed over) are the delivery/native phase tests that
  read the DB602/DB604/L7/E0 rank-0 originals, `test_ws32_delivery_hlo.py`,
  `test_ws32_delivery_quality.py`, `test_ws32_short_context.py`,
  `test_first_divergent_event_adjudication.py` and `test_ws32_short_sealer.py`
  (sealed oracle pins), plus the `git archive` of the pinned vLLM build in
  `test_accepted_compile_only_hlo.py`. On this host all of those inputs are present.
- Three tests read history with `git show` (`cb36cb74^` in
  `tests/greenfield/kernels/test_ws32_layer.py`; `5e94239a` and `129ac4ba` in the
  delivery decode/loading tests). A shallow clone cannot run them.
- `GLM_WS32_SLOW_SEAL_TEST=1` opts into the eight-rank sealer replay (hashes
  about 2.4 GB of traces); `GLM_RELEASE_LOCAL_TOKENIZER` opts into the real
  tokenizer check.

Some local copies of sealed compiler graphs are no longer present on this host:
the DB609 rank-0 optimized/StableHLO text files and the uncompressed seven-graph
acquisition HLO (only gzip copies remain locally). The sealed originals stay in
the regional archive prefixes recorded in their receipts. The affected replays
(`test_ws32_canonical_fullmodel_hlo.py`, the DB609 worker-publication tests and
`test_ws32_paired_dsa_locations.py`) now skip with an explicit reason instead of
erroring; restoring the originals to their local paths re-enables them without
code changes. Nothing was re-downloaded for this curation.

## Admission tests bound to sealing-source pins

A family of historical admission tests compares the working tree with the
sealing-source pins of the batched-prefill campaigns rather than with the
release pin. `ws32_prefill_admission.require_acquired_model_source` diffs the
frozen `MODEL_SOURCE` against the 2K/8K numerical sealing commits (`7456bf64`,
`c672d4cd`), `rolled_registration` and the paired registration hash their
prerequisite receipts, and the capture-barrier, canonical, flat-rows and
pending-rows compile modules each pin the SHA-256 of one historical version of
`glm_tpu/greenfield/runtime/ws32_batched_prefill.py`. That file evolved after
each campaign (sampling, request session, capture barrier), so at most one of
those pins can match any checkout, and none matches the release source. The
user path no longer carries a frozen-source pin: which checkout may launch is
the site file's `[launch]` policy (`glm_tpu/executor/launch_policy.py`), and
numerics drift is caught by the equivalence gates (`tools/equivalence`). The
research guard `ws32_native_benchmark_programs.require_source` (pin `edecdd94`)
remains only for the historical research tools that still call it.

These tests therefore pass only inside the sealing-source worktrees that the
sealed runs created under `/home/gianl/glm-run/*/sealing-source.*`; on the
curated tree they refuse with "prerequisite/target evidence drifted",
"source/prerequisite differs" or "model source differs from acquisition", and
the tests that drive a worker or sealer through those refusals fail downstream
of them. A smaller group asserts sealed identities that the sealer or wrapper
later outgrew (the wrapper's second read-only preflight call, the long-context
enforcement line, frozen-8K program options, the retained WK decode original,
the acquisition census text). None of these was edited during curation: they
pin sealed identities, and weakening them was out of scope.

Measured on 2026-09-15/16 by re-running every module that failed in the
whole-tree runs both at the starting pin `b667f00f` (scratch worktree) and on the
curated code tree `cc2b36de`, with the same interpreter: 186 failures at the
starting pin, 173 on the curated tree, every curated-tree failure also present
at the starting pin, zero introduced by curation. The 13 that fail only at the
starting pin are three retired preparation tests and ten DB609/paired-location
replays that now skip when the local originals are absent. The comparison is
recorded in `curation-whole-tree-cpu-20260915.json` (`baseline_comparison`).
Modules affected: `test_ws32_batched_launch.py`, `test_ws32_batched_numerical_worker.py`,
`test_ws32_batched_acquisition_census.py`, `test_ws32_canonical_8k_integration.py`,
`test_ws32_canonical_short_integration.py`, `test_ws32_rolled_short_registration.py`,
`test_ws32_rolled_short_integration.py`, `test_ws32_frozen_8k_integration.py`,
`test_ws32_frozen_live32_diagnostic.py`, `test_ws32_prefill_admission.py`,
`test_ws32_paired_short_admission.py`, `test_ws32_paired_short_wiring.py`,
`test_ws32_prefill_frontier_entry.py`, `test_ws32_long_context_mode.py`,
`test_ws32_sealing_checkout.py`, the `test_ws32_delivery_*` entry/runtime/
companions/composed-sealer/phase-loading/quality/hlo-publication/programs/prefill
modules, `test_ws32_capture_barrier_compile.py`, `test_ws32_canonical_prefill_compile.py`,
`test_ws32_rolled_prefill_compile.py`, `test_ws32_delivery_hlo.py`, `test_ws32_native_benchmark_programs.py`,
`test_ws32_paired_prefill.py`, `test_ws32_canonical_full_prefill.py`,
`test_ws32_rolled_short_prefill.py` and `test_ws32_prefill_owned_state.py`.
Every other retained module passes or skips with a stated reason.

## Latest recorded runs

See [STATUS](STATUS.md) for the whole-tree and release-check results recorded at
the end of curation, including skip counts. Counts from earlier receipts
(`final-cpu-check-20260914.json`: 431 tests) describe the pre-curation tree.
