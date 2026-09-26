# Installation and environment

Python 3.12 on Linux is the supported target (`requires-python = ">=3.12,<3.13"`).
Hardware inference additionally needs the TPU fleet, the verified checkpoint and a
site file ([OPERATIONS](OPERATIONS.md), [CHECKPOINTS](CHECKPOINTS.md)). Install into
a new virtual environment; never install or upgrade packages inside an environment
that a running model or a measurement uses.

## Dependencies

Every dependency is one exact pin in `pyproject.toml`:

| Group | Packages | Needed for |
|---|---|---|
| core | `jax`, `jaxlib` 0.10.1, `numpy` 2.3.5, `ml-dtypes` 0.5.4 | everything |
| extra `runtime` | `torch` 2.10.0 (CPU build), `safetensors`, `transformers` 5.12.0, `tokenizers`, `huggingface-hub`, `google-cloud-storage`, `google-crc32c`, `protobuf`, `grpcio`, `requests` | checkpoint reading and packing, the pinned tokenizer and chat template, request preparation; most CPU tests |
| extra `tpu` | `libtpu` 0.0.41 | running on TPU |
| extra `dev` | `pytest` 9.0.3 | the tests |

PyTorch is used only to read checkpoint tensors on the CPU; the model runs in JAX.
No CUDA and no vLLM installation is needed.

## Install

With [uv](https://docs.astral.sh/uv/), from the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python '.[runtime,dev]'
```

Add `tpu` on a TPU host (`'.[runtime,tpu,dev]'`). The repository's uv
configuration takes only PyTorch from `https://download.pytorch.org/whl/cpu`
(`[tool.uv.sources]`), so the CPU build of `torch==2.10.0` is installed and every
other package resolves from PyPI. Do not add that index globally ahead of PyPI:
it also serves unrelated packages and can shadow the pinned versions. With
another installer, obtain the CPU build of PyTorch first; the pin `torch==2.10.0`
alone does not select it.

For development and review, the ruff and codespell tools of
[CONTRIBUTING](../../CONTRIBUTING.md) live in a separate tool environment, not in
the runtime environment.

## Check an environment

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m glm_tpu info
JAX_PLATFORMS=cpu .venv/bin/python -m glm_tpu collect-env --profile runtime
```

`collect-env` (alias `doctor`) compares installed distribution metadata with the
pins, for a profile: `core`, `runtime` (core and `runtime`), `tpu` (core,
`runtime` and `tpu`) or `dev` (core and `dev`). It prints a JSON report and exits
nonzero on any mismatch or missing package. It reads metadata only: it imports
neither JAX nor PyTorch, opens no device, reads no weights and contacts no
service. The pins come from the installed `glm-tpu` distribution, else from the
`pyproject.toml` of the checkout it runs from. In a checkout that holds build
metadata of an in-tree build (a `glm_tpu.egg-info/` directory, ignored by Git),
that metadata is found first and may be stale: build wheels from a copy of the
tree (below) or remove the directory before running `collect-env` from the
checkout.

After installation, `glm-tpu` is the same command as `python -m glm_tpu`.

## The wheel

The wheel contains the `glm_tpu` package with its package data (the chat UI's
page, script and style sheet, and the pinned GLM-5.3 configuration, generation
configuration, tokenizer configuration, chat template and their license) and the
license files. It contains no weights, no tokenizer vocabulary, no tests, tools
or documentation. To build it without leaving build output in the checkout,
build from an exported copy:

```bash
out=$(mktemp -d)
mkdir "$out/src"
git archive HEAD | tar -x -C "$out/src"
uv build --wheel --out-dir "$out/dist" "$out/src"
```

The build backend is pinned (`setuptools==78.1.0`, `wheel==0.47.0`).

The wheel alone is not a deployment. The controller stages the **source
checkout** it runs from (a `git archive` of its pinned commit) to every host,
and a run needs the site file and the external assets: the packed checkpoint,
the tokenizer files and the topology binding.

## The site file

Every site-specific value (fleet, interpreters, paths, checkpoint and topology
pins, locks, launch policy) comes from one untracked TOML file.
[`examples/site.example.toml`](../../examples/site.example.toml) documents every
key; `glm_tpu/config/site.py` validates it and refuses to load it unless it is a
regular file owned by the user with mode 0600 or 0400:

```bash
mkdir -p ~/.config/glm-tpu
cp examples/site.example.toml ~/.config/glm-tpu/site.toml
chmod 600 ~/.config/glm-tpu/site.toml
```

Then fill in every `<...>` value. Commands that need it take `--site PATH`;
precedence is command-line flag, then `GLM_TPU_*` variable, then site file, then
built-in default. The fleet values and the launch policy come only from the site
file.

## Environment variables

`glm_tpu/envs.py` is the registry:

| Variable | Meaning |
|---|---|
| `GLM_TPU_CONFIG_ROOT` | directory of `site.toml`, absolute (default `$XDG_CONFIG_HOME/glm-tpu` when that is absolute, else `~/.config/glm-tpu`) |
| `GLM_TPU_SITE_CONFIG` | an explicit site file, absolute (default `$GLM_TPU_CONFIG_ROOT/site.toml`) |
| `GLM_TPU_RUN_ROOT`, `GLM_TPU_MODEL_PATH`, `GLM_TPU_HLO_DUMP_ROOT` | controller-side overrides of the site's `paths.run_root`, `paths.model_path` and `paths.hlo_dump_root` |
| `GLM_TPU_TEST_SITE` | tests: a real site file; enables the site-bound tests |
| `GLM_TPU_TEST_TIMEOUT_SCALE` | tests: a multiplier for subprocess timeouts |
| `GLM_TPU_TEST_HELPER_PYTHON` | tests: the interpreter of the remote-helper loopback test (default `python3`) |

`JAX_PLATFORMS=cpu` keeps JAX off the TPU for every command that does not run the
model; the tests force it.
