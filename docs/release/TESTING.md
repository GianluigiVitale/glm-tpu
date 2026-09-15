# Testing

Every test runs on CPU. `tests/conftest.py` forces `JAX_PLATFORMS=cpu` and refuses
any other platform, so no test can initialize a TPU. Fake math inside a CPU test
is never TPU validation; hardware evidence lives in the sealed receipts under
`docs/artifacts/` and `docs/release/`.

## Tiers

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
  composition of windows, tails, rollback and observers.
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
- Two tests read history with `git show` (`cb36cb74^` in
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

## Latest recorded runs

See [STATUS](STATUS.md) for the whole-tree and release-check results recorded at
the end of curation, including skip counts. Counts from earlier receipts
(`final-cpu-check-20260914.json`: 431 tests) describe the pre-curation tree.
