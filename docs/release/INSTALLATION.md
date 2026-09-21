# Installation and environment

This is the site-specific installation boundary for GLM-5.3 ordinary inference.
Real-weight acceptance and final promotion are tracked in [STATUS](STATUS.md).
The four-question run used the existing environment without an upgrade. The
installation receipts below are retained environment evidence; DB621 describes
the separate historical GLM-5.2 sampled release.
Python 3.12 on Linux is the supported target; the observed
controller uses Python 3.12.13. Keep the existing running environment untouched.

## What has been checked

- The private `glm-tpu` Python wheel builds and installs into a fresh virtual
  environment without dependencies. Its console command works outside the checkout.
- The wheel includes native Python components and environment metadata, not
  weights, benchmark data, the research script tree or historical artifacts.
- All 63 installed dependency versions in `requirements/runtime-observed.txt`
  satisfy their active metadata requirements and resolve together against PyPI
  plus the explicit CPU PyTorch index. No library was installed or upgraded in
  the running environment.
- Fresh installation of all 63 dependencies plus the project passed in an
  isolated temporary venv: `uv pip check`, all doctor versions and actual CPU
  imports passed, including verified CPU-only PyTorch. 78 selected tests passed
  with one optional tokenizer test skipped. See the [installation receipt](fresh-install-20260914.json).
- TPU deployment from the reviewed release passed DB621. Version pins are not
  downloaded-wheel hashes or a security audit.

## Offline information and environment checks

From the source checkout, using the target Python:

```bash
python -m glm_tpu info
python -m glm_tpu doctor --profile core
python -m glm_tpu doctor --profile benchmark
```

After installation, `glm-tpu` is the equivalent console command. Doctor inspects
distribution metadata only: it never imports JAX/PyTorch, initializes a device,
reads model weights, contacts cloud services or grants launch authority. A mismatch
exits nonzero. It does not validate an entire dependency graph or installed bytes.

Profiles: `core` (JAX components), `runtime` (checkpoint/tokenizer/cloud tools),
`tpu` (runtime plus libtpu), `benchmark` (TPU runtime plus dataset helpers), and
`dev` (core plus pytest). PyTorch is CPU-only for checkpoint reading, not native
model execution. No CUDA or legacy vLLM engine is required by these profiles.

## Full-environment recipe

For a new inspection environment, use an isolated virtual environment with
adequate disk space. Keep the working TPU environment unchanged:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --constraint requirements/runtime-observed.txt \
  --requirement requirements/runtime-observed.txt \
  '.[runtime,tpu,benchmark,dev]'
uv pip check --python .venv/bin/python
JAX_PLATFORMS=cpu .venv/bin/glm-tpu doctor --profile benchmark
```

Use the repository's uv source configuration: only PyTorch comes from
`https://download.pytorch.org/whl/cpu`; other dependencies resolve from PyPI.
Do not add that index globally ahead of PyPI: it also contains unrelated packages
and can shadow required versions. Do not use an unsafe best-match index policy.
With another installer, explicitly obtain CPU PyTorch first; the public version
constraint `torch==2.10.0` alone does not force a CPU wheel.

The wheel alone is not a deployable server. Runtime/protection scripts currently
need the full Git checkout, historical source pins and external manifest/weight
assets. Do not shallow-clone away the source pins required by integrity checks.
The GLM-5.3 ordinary deployment passed the four-answer acceptance recorded in
STATUS.md. The legacy sampled entry and DB621 remain separate GLM-5.2 history.

The `prepare-request` command performs local pinned-tokenizer preparation,
not inference or deployment. Its default profile is `ordinary-greedy-8k`; see
[ordinary inference](OPTIMIZED_INFERENCE.md). The separate `legacy-sampled`
profile and worker have historical protected TPU admission in DB621; see
[legacy inference](INFERENCE.md) for their different limits.

## Reproduce packaging and CPU checks

With setuptools 78.1.0, wheel 0.47.0 and uv available in the test environment:

```bash
JAX_PLATFORMS=cpu python tools/check_release.py
```

The packaging check creates a temporary source copy and isolated venv, then
cleans only its own temporary directory. It never installs dependencies into
the caller's environment or downloads a model. Installing the full environment
will require substantially more disk than the approximately 1.3 MB project wheel.

The consolidated check also runs selected pytest suites (including pytest-style
tests that unittest discovery does not execute), dependency metadata checks,
tracked-content scanning and the original native model-source guard. It does
not prove full dependency installation, hardware execution or merge readiness.
See [security audit scope](../../SECURITY.md). Run the separate full Git history
scan only when its object-set evidence needs renewal, not on every CPU check.

For the independently tested full-install check, explicitly choose a scratch
directory with at least20GiB free and32GiB available host RAM:

```bash
JAX_PLATFORMS=cpu python tools/check_release_install.py --scratch-root /path/to/scratch
```

The tool creates/removes only its own temporary source/venv/cache directory. It
downloads Python packages, not weights, and never modifies the caller's Python
environment. On2026-09-14 `/dev/shm` had103GiB free and the host had264GiB available
RAM; the check's end-of-test scratch delta was2.40GB and it cleaned up afterward.
Those are dated observations, not future launch admission or peak-memory proof.
Do not rerun this on every edit; use the small offline check unless installation
requirements change. A tmpfs environment is disposable, not persistent deployment.
