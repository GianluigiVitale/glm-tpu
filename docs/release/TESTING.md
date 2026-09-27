# Testing

Every test runs on the CPU. `tests/conftest.py` sets `JAX_PLATFORMS=cpu` and
refuses any other platform, so no test can open a TPU; it also points the site
configuration at a directory that does not exist, so no test reads the operator's
site file. A CPU test is never evidence about TPU speed or answers; the hardware
evidence is in the [release receipts](STATUS.md).

Run everything from the repository root with the environment of
[INSTALLATION](INSTALLATION.md) (the `runtime` and `dev` extras) and
`JAX_PLATFORMS=cpu`. `-p no:cacheprovider` keeps pytest from writing a cache into
the checkout.

## Layout

| Path | What it tests |
|---|---|
| `tests/<package>/...` | one module per `glm_tpu` module, mirroring the package (`tests/engine/test_request.py` tests `glm_tpu/engine/request.py`) |
| `tests/golden/` | the pytest wrappers of the equivalence gates; `tests/golden/data/*.json` are the records they compare with (digests and small summaries, written only by the harness) |
| `tests/reference/` | an unsharded single-device reference model of GLM-5.3 and its consistency tests; [VALIDATION](../../tests/reference/VALIDATION.md) is its acceptance record |
| `tests/models/glm_moe_dsa/test_against_reference.py` | the production composition against the reference model on the frozen fixture (with `floors.json`) |
| `tests/fixtures/` | a tiny model, a tiny checkpoint, example site files, a synthetic resident for the serving tests and the 78-layer prefill layer schema |
| `tests/test_import_boundaries.py`, `tests/test_envs.py`, `tests/test_error_messages.py`, `tests/test_documentation.py` | package-wide rules: import layering (production loads only this package's modules), the environment-variable registry, error-message wording, and that the documentation's links and documented test files exist |

Markers: `cpu32` (runs a child process on 32 forced CPU devices; slow),
`slow`, `golden`, and `site` (needs the operator's real site file; skipped
without it).

## Tiers

The offline subset of the [README](../../README.md#check-without-tpu-hardware)
covers the command line, requests, the controller's gates, the worker, the site
file and the checkpoint commands in about half a minute:

```bash
JAX_PLATFORMS=cpu python -m pytest -q -p no:cacheprovider tests/entrypoints/cli/test_main.py tests/entrypoints/cli/test_ask.py tests/entrypoints/cli/test_prepare.py tests/entrypoints/cli/test_collect_env.py tests/entrypoints/cli/test_checkpoint.py tests/engine/test_request.py tests/engine/test_llm_engine.py tests/executor/test_multihost_executor.py tests/executor/test_fleet.py tests/worker/test_tpu_worker.py tests/utils_/test_io_utils.py tests/config/test_site.py tests/runner/test_tpu_runner.py
```

The API and UI subsets are in [API](../API.md#offline-checks) and
[UI](../UI.md#offline-checks).

The whole suite outside the equivalence wrappers, and the light equivalence
wrappers:

```bash
JAX_PLATFORMS=cpu python -m pytest -q -p no:cacheprovider -rs tests --ignore=tests/golden
JAX_PLATFORMS=cpu python -m pytest -q -p no:cacheprovider -rs tests/golden -m "not cpu32"
```

The first includes the `cpu32` tests (layers, the decode and prefill programs and
the checkpoint format on 32 CPU devices, and the production composition against
the reference model) and takes about 20 minutes on a large host; one `site` test
skips. The second (the checkpoint identities, the static
import closure, the wire formats and the data contracts) takes about a minute.

The heavy equivalence gates build the real runtime and compare with the records
([harness README](../../tools/equivalence/README.md)):

```bash
JAX_PLATFORMS=cpu python -m tools.equivalence check
JAX_PLATFORMS=cpu python -m tools.equivalence check --tier production
JAX_PLATFORMS=cpu python -m tools.equivalence selftest
JAX_PLATFORMS=cpu python -m tools.equivalence check --gates G4,G6-static,G9
JAX_PLATFORMS=cpu GLM_EQUIVALENCE_PRODUCTION=1 GLM_EQUIVALENCE_STRICT=1 python -m pytest -p no:cacheprovider -rs tests/golden -m cpu32
```

Measured on a 240-core host at the last full run: `check` 19 minutes,
`check --tier production` 12 minutes, `selftest` 3 minutes, the light gates
under a minute, the `cpu32` golden wrappers 29 minutes. Expect several times more
on a small machine. `GLM_EQUIVALENCE_STRICT=1` turns a version-mismatch skip into
a failure; without `GLM_EQUIVALENCE_PRODUCTION=1` the production-tier wrappers skip.

The resident protocol, the remote helpers and their loopback run need a Python of
at least 3.10 for the helpers (the hosts' system interpreter):

```bash
JAX_PLATFORMS=cpu GLM_TPU_TEST_HELPER_PYTHON=python3 python -m pytest -q -p no:cacheprovider tests/engine/test_resident_protocol.py tests/executor/test_remote_helpers.py tests/executor/test_remote_loopback.py
```

The site-bound test runs the real preflight of one host against the operator's
site file and assets, on rank 0 with the fleet idle:

```bash
GLM_TPU_TEST_SITE=~/.config/glm-tpu/site.toml JAX_PLATFORMS=cpu python -m pytest -p no:cacheprovider tests/worker/test_local_preflight.py -m site
```

## A TPU run on the same host

`cpu32` tests and the heavy gates would compete with a TPU run for the host's CPUs
and memory. While a controller, worker or pack worker runs on the host, a process
holds libtpu, or a site workload lock is held, every `cpu32` test skips with that
reason and the heavy gates refuse (`python -m tools.equivalence budget` shows the
state). Read the skip reasons (`-rs`) before trusting a run. Two known effects of
running things side by side: a test that starts worker-named stand-in processes
(the fleet and resident-protocol tests) or runs the controller's `--help` makes a
concurrently starting `cpu32` collection or heavy gate see a live run for a moment.
Run the heavy gates and the `cpu32` tests first, or one at a time.

## What the tests do not cover

The tests establish software contracts: request integrity and capacity refusals,
controller identity, SSH command strings and failure gates, delivery and deadline
behaviour with synthetic results, the checkpoint format on a tiny model, the
kernels and layers against unsharded references on CPU, and the unchanged device
programs. They do not measure speed, memory on a TPU or answer quality, and the
failure paths of the fleet votes are exercised with CPU fakes only.
