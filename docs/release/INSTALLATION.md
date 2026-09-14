# Installation and environment

This is the release candidate's installation boundary, not a claim of completed
main deployment. Python 3.12 on Linux is the supported target; the observed
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
- Fresh installation of ALL those dependencies, actual wheel-content identity
  and TPU deployment from the release branch remain unverified. Version pins
  are not wheel hashes or a security audit.

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

## Candidate full-environment recipe

Do not execute this in the active benchmark environment or deploy it before
release admission. Use an isolated virtual environment with adequate disk space:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --constraint requirements/runtime-observed.txt \
  -e '.[runtime,tpu,benchmark,dev]'
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
Main-branch deployment and the user inference entry point are still release work.

The new `prepare-request` command performs local pinned-tokenizer preparation,
not inference or deployment. Its real-local-tokenizer test passes; the separate
user worker executor is CPU-tested but not TPU-admitted yet. See
[user request integration](INFERENCE.md) for the example and exact limits.

## Reproduce packaging and CPU checks

With setuptools 78.1.0, wheel 0.47.0 and uv available in the test environment:

```bash
JAX_PLATFORMS=cpu python tools/check_release_package.py
JAX_PLATFORMS=cpu python -m unittest discover -s tests/release -v
git diff --check
```

The packaging check creates a temporary source copy and isolated venv, then
cleans only its own temporary directory. It never installs dependencies into
the caller's environment or downloads a model. Installing the full environment
will require substantially more disk than the approximately 1.3 MB project wheel.
