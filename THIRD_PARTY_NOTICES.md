# Third-party material and distribution scope

This private repository does not currently grant a blanket open-source license
for the owner's original work. Third-party material retains its own licenses.
Do not infer a repository license from the model card or from these notices.
Public distribution still requires completion of the provenance/privacy audit.

## Hugging Face Transformers reference extracts

The following files retain their original copyright and Apache-2.0 headers:

- `reference/configuration_glm_moe_dsa.py`
- `reference/modeling_glm_moe_dsa.py`
- `reference/modular_glm_moe_dsa.py`

Copyright 2026 the HuggingFace Team. All rights reserved.
License text: [Apache License 2.0](licenses/Apache-2.0.txt).

On 2026-09-14 each extract was byte-identical to its corresponding file in the
installed Transformers 5.12.0 distribution under
`transformers/models/glm_moe_dsa/`. The included license text was copied from that
distribution's own LICENSE file, unchanged. This is a local distribution
comparison, not a claim to have established the original upstream commit or
revalidated downloaded wheel bytes. Hashes and limitations are in
`docs/release/third-party-reference-check-20260914.json`.

These extracts are reference material, not the native engine's execution path.
The current private wheel excludes `reference/`; source checkouts include it.

## Model repository snapshot and weights

`reference/hf-glm53/` contains the unchanged GLM-5.3 configuration, generation
configuration, tokenizer metadata and chat template from `zai-org/GLM-5.3` at
revision `aca966e4e02791568aa6a4ced368624b3d897f42`. Copyright (c) 2026 Z.AI.
Its [GLM-5.3 license](licenses/GLM-5.3.txt) is retained verbatim. This license is
distinct from GLM-5.2's MIT license. Tokenizer vocabulary and weights remain
external; the ordinary request profile pins their model/template identities.
The historical snapshot below remains unchanged.

`reference/hf-repo/` preserves GLM-5.2-FP8 configuration, tokenizer metadata, chat
template and historical model-card text. The publisher's
[MIT license](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/f33c6dc501ee5a2c7e35155653b1b1abbc320951/LICENSE)
is retained in [GLM-5.2-MIT.txt](licenses/GLM-5.2-MIT.txt):
Copyright (c) 2026 Zhipu AI.

On 2026-09-14 the four configuration/template files matched upstream revision
`f33c6dc501ee5a2c7e35155653b1b1abbc320951` byte-for-byte. The local README is an
earlier historical copy, not the benchmark protocol's pinned card; it was not
overwritten. Use the protocol's original card revision/hash for comparisons.
The copied license was obtained from that exact revision, not a different GLM
repository. Details: `docs/release/hf-source-notice-check-20260914.json`.
Model weights are external and must never be bundled into Git or the wheel.

## Legacy patch and external dependencies

The historical `patches/vllm-fused-indexer-wk-clone.patch` is excluded from this
release tree and wheel. It remains recoverable at research commit
`83f0c2728d0d418255a917343cc89d24b815bd0c` (Git blob
`c1b64f0d88b768f4367e2fa1c1b4d07e7013a8fd`). No supported runtime code consumes
it. Any future reuse or public distribution of that historical patch needs its
own upstream revision/license-notice reconciliation; removal from this tree does
not remove it from history or complete public-distribution clearance.

JAX, jaxlib, libtpu, PyTorch, Transformers and other installed dependencies are
separately distributed packages, not relicensed by this project. Exact observed
versions are in `requirements/runtime-observed.txt`; dependency installation does
not mean all binary redistribution terms have been audited. Do not publish a
bundled runtime image as though this source notice covered all of its contents.
