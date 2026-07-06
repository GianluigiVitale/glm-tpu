# bench/ — GLM-5.2 benchmark reproduction + provenance DB

Pre-built (2026-07-06, CPU only — no model yet) so the port session starts with the
traceability machinery ready and the datasets staged. **Rule: nothing is scored
that is not stored.** Every item's verbatim question, timestamp, verbatim model
reply, extracted answer, and pass/fail goes into `results.db`.

## Files
- `provenance.py` — the SQLite provenance DB (`results.db`): `runs` (model+revision,
  harness+fork git, env/flags, pod), `items` (benchmark, item_id, asked_utc, prompt,
  gold, raw_output, extracted, correct, score, tokens, latency, seed), `summary`
  (n, metric, value, card_value, signed Δ). Also holds `CARD_TARGETS` (the HF-card
  published values). A number with no `items` rows behind it is not a result.
- `benchmarks.py` — the registry: GPQA-Diamond, MMLU-Pro, GSM8K, AIME-2026 — dataset
  loaders (HF), deterministic prompt builders, extractors, scorers.
- `extract.py` — answer extractors + scorers (MC letter, `\boxed{}`, final number;
  `<think>` stripping). Pure/offline.
- `run_bench.py` — the harness: load → prompt → **generate(prompt)** → extract →
  score → store every item → finalize summary. `generate` is **pluggable**; wire the
  real GLM-5.2 engine into `make_generate()` in Stage 1 (until then `--stub` runs the
  offline pipeline check).
- `download_data.py` — pre-downloads/caches the tractable datasets to `data/`
  (gitignored).
- `test_bench.py` — CPU tests (DB round-trip, extractors, scorers, item builders) — **all pass.**

## Status (2026-07-06)
- Datasets **cached** (`bench/data/`, 13 MB): **MMLU-Pro** (12032), **GSM8K** (1319),
  **AIME-2026** (30, `MathArena/aime_2026`). Pipeline validated end-to-end on real
  data with the stub.
- **GPQA-Diamond is gated** (`Idavidrein/gpqa`) — the HF token lacks access.
  **[OWNER]** request access at https://huggingface.co/datasets/Idavidrein/gpqa (or
  point `benchmarks.GPQA_DIAMOND.hf_path` at an ungated mirror), then re-run
  `download_data.py`.
- `results.db` is a clean schema-ready DB (gitignored runtime artifact; schema in
  `provenance.py`).

## Use
```bash
export HF_TOKEN=$(grep -oP 'HF_TOKEN=\K.*' ~/glm-tpu/.env)
~/vllm-env/bin/python bench/test_bench.py           # CPU tests
~/vllm-env/bin/python bench/download_data.py        # cache datasets
~/vllm-env/bin/python bench/run_bench.py --benchmark gsm8k --limit 5 --stub   # offline pipeline
# Stage 1: wire make_generate() to the GLM-5.2 engine, then drop --stub.
```

The agentic/coding card benchmarks (SWE-bench Pro, Terminal Bench, NL2Repo, …) need
full agentic harnesses and are deferred (targets are in `CARD_TARGETS`).
