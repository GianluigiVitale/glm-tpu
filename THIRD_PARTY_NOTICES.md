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
distribution's `licenses/LICENSE`, unchanged. This is a local distribution
comparison, not a claim to have established the original upstream commit or
revalidated downloaded wheel bytes. Hashes and limitations are in
`docs/release/third-party-reference-check-20260914.json`.

These extracts are reference material, not the native engine's execution path.
The current private wheel excludes `reference/`; source checkouts include it.

## Model repository snapshot and weights

`reference/hf-repo/` preserves GLM-5.2-FP8 configuration, tokenizer metadata, chat
template and model-card text. The saved model card declares MIT. That declaration
is not a substitute for the upstream copyright/license notice: its exact retained
source revision and license text still need reconciliation. Do not present this
notice as completed clearance for publishing those files. Model weights are
external and must never be bundled into Git or the wheel.

## Legacy patch and external dependencies

`patches/vllm-fused-indexer-wk-clone.patch` contains vLLM source context from the
historical oracle workflow. Its upstream revision/license-notice reconciliation
remains part of the release audit. It is not native model execution and is not
included in the wheel. Preserve research provenance before removing it.

JAX, jaxlib, libtpu, PyTorch, Transformers and other installed dependencies are
separately distributed packages, not relicensed by this project. Exact observed
versions are in `requirements/runtime-observed.txt`; dependency installation does
not mean all binary redistribution terms have been audited. Do not publish a
bundled runtime image as though this source notice covered all of its contents.
