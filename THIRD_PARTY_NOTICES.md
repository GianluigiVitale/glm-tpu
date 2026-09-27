# License and third-party notices

## This repository

Copyright 2026 Gianluigi Vitale.

The project's own work in this repository (the `glm_tpu` package, its tests, the
equivalence harness, the documentation and the compact result receipts) is
licensed under the [Apache License, Version 2.0](LICENSE). The material listed
below keeps its own license; the Apache License does not relicense it. The package
metadata states the combination as the SPDX expression
`Apache-2.0 AND LicenseRef-GLM-5.3` (`pyproject.toml`).

## Material in this tree with its own license

### GLM-5.3 model configuration

`glm_tpu/models/glm_moe_dsa/hf_config/` contains the unchanged GLM-5.3
configuration, generation configuration, tokenizer metadata and chat template from
`zai-org/GLM-5.3` at revision `aca966e4e02791568aa6a4ced368624b3d897f42`.
Copyright (c) 2026 Z.AI. Its [GLM-5.3 license](glm_tpu/models/glm_moe_dsa/hf_config/LICENSE)
is retained verbatim next to the files and ships with them in the wheel. This
license is distinct from GLM-5.2's MIT license. The tokenizer vocabulary and the
model weights are external: they are never part of Git or of the wheel, and their
use is governed by the model publisher's license.

### Chat UI styling

The palette, sidebar, message layout, composer and responsive styling of
`glm_tpu/entrypoints/ui/static/style.css` are adapted from the owner's own `as-pt`
project, file `aspt_rag/web/index.html` at commit
`4ed7937910538eef2754b31d0af316c6888ad9ba` (file SHA-256 in
[docs/UI.md](docs/UI.md)). The owner contributes this adaptation under this
repository's license. This notice licenses nothing else of the `as-pt` project.

## Material in the repository history only

These files are not in this tree. They are preserved, with their original headers,
at the tag `archive/research-20260922` (the research tree this release was cut
from) and in the Git history, which a clone of this repository carries. Their
license texts stay in `licenses/` for that reason.

- **Hugging Face Transformers reference extracts**:
  `reference/configuration_glm_moe_dsa.py`, `reference/modeling_glm_moe_dsa.py`,
  `reference/modular_glm_moe_dsa.py`. Copyright 2026 the HuggingFace Team.
  License: [Apache License 2.0](licenses/Apache-2.0.txt), the LICENSE file of the
  Transformers 5.12.0 distribution copied unchanged (hence its Hugging Face
  copyright line). On 2026-09-14 each extract was byte-identical to its file in the
  installed Transformers 5.12.0 distribution (`transformers/models/glm_moe_dsa/`);
  this is a local distribution comparison, not a claim about the upstream commit.
  The receipt is `docs/release/third-party-reference-check-20260914.json` at the
  tag. The extracts were reference material, never the engine's execution path.
- **GLM-5.2-FP8 repository snapshot** (`reference/hf-repo/`): configuration,
  tokenizer metadata, chat template and a historical model-card copy from
  `zai-org/GLM-5.2-FP8` at revision `f33c6dc501ee5a2c7e35155653b1b1abbc320951`.
  Copyright (c) 2026 Zhipu AI. License: [MIT](licenses/GLM-5.2-MIT.txt), obtained
  from that exact revision. On 2026-09-14 the four configuration/template files
  matched that revision byte for byte; the local model-card copy is an earlier
  historical copy, not the benchmark protocol's pinned card. Details:
  `docs/release/hf-source-notice-check-20260914.json` at the tag.
- **A historical vLLM patch** (`patches/vllm-fused-indexer-wk-clone.patch`) is in
  neither this tree nor the wheel. It is recoverable at research commit
  `83f0c2728d0d418255a917343cc89d24b815bd0c` (Git blob
  `c1b64f0d88b768f4367e2fa1c1b4d07e7013a8fd`). No code of this tree uses it. Any
  reuse or redistribution of it needs its own upstream license reconciliation.

## Evaluation data

The GSM8K evaluation used the `openai/gsm8k` dataset, `main` configuration, test
split at revision `740312add88f781978c0658806c59bc2815b9866`. Dataset questions,
reference solutions and generated answer transcripts are not redistributed; the
aggregate receipt identifies the source and the scoring method. The questions and
solutions belong to the GSM8K dataset authors.

## Dependencies

JAX, jaxlib, libtpu, PyTorch, Transformers, tokenizers and the other packages
declared in `pyproject.toml` are distributed separately under their own licenses;
this project does not relicense them. Installing them does not mean their binary
redistribution terms were audited: do not publish a bundled runtime image as
though this notice covered its contents.
