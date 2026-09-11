# Official GLM-5.2-FP8 benchmark reference

Checked 2026-09-11, before any §26 candidate benchmark outputs. This fixes the
comparison reference, not a claim of matched evaluation or benchmark success.

- Authority: https://huggingface.co/zai-org/GLM-5.2-FP8
- HF repository revision: `f33c6dc501ee5a2c7e35155653b1b1abbc320951`.
- Immutable source: https://huggingface.co/zai-org/GLM-5.2-FP8/raw/f33c6dc501ee5a2c7e35155653b1b1abbc320951/README.md
- Raw README SHA-256: `522afffe4e9bb3b3054a68df39b1796417a745aaed689511e7c0a11b449c4e9d`.
- Retained existing copy: `reference/hf-repo/README.md`, 10,909 bytes,
  SHA-256 `de23c1b7cab43a99f0fedf4edf10de0d165882a2ecb2e2bad6c5796fcabf2e46`.
- Exact difference: the immutable remote version inserts
  `new_version: zai-org/GLM-5.3` immediately after `pipeline_tag: text-generation`.
  No benchmark or evaluation footnote differs. This metadata does not change our
  target model. The existing copy plus this one-line insertion reconstructs the
  pinned remote bytes; do not overwrite the historical copy.

Use the **GLM-5.2** column: GPQA-Diamond **91.2%**, AIME 2026 **99.2%**.
Published reasoning settings are temperature 1.0, top_p 0.95, maximum generation
163,840 tokens. The math footnote supplies the Explanation / Exact Answer /
Confidence system prompt and names GPT-5.5 medium judging. See the retained full
footnote, not a paraphrase, when constructing the run protocol.

The card does not specify all GPQA prompt/extraction details or the AIME sample
count/aggregation. Existing `bench/benchmarks.py` explicitly documents those
gaps and its deterministic math-extractor substitution. That substitution is NOT
the published judge. Old local scores are never the acceptance baseline.

Before benchmark execution, separately register dataset revision/items, native
sampling and seed semantics, prompts/template/stops/cap, scoring/aggregation,
uncertainty/material-deficit rule and wall/token budget. Resolve or explicitly
disclose protocol differences. This card pin does not authorize paid judging or
claim the native benchmark adapter is already complete.
