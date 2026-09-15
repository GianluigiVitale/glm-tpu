# bench/ — pinned legacy harness modules

This directory holds the six legacy vLLM-era harness modules that the native
WS32 engine still depends on. They are **not** an entry point for running
benchmarks on the native engine: the runner and sealer load them from their
exact committed bytes, so their contents are frozen by SHA pins.

| File | Role today | Pinned by |
|---|---|---|
| `provenance.py` | `results.db` schema (`runs`/`items`/`summary`) and card targets | sealer enforcement surface; `scripts/release/ws32_user_database.py` |
| `extract.py` | answer extractors/scorers (MC letter, `\boxed{}`, final number, Exact Answer) | native GPQA scorer (`ws32_native_benchmark_protocol.py`); sealer surface |
| `benchmarks.py` | benchmark registry with pinned HF dataset revisions and the verbatim card protocol quote | native card registry load; `configs/greenfield-native-benchmark-protocol.json` |
| `engine.py` | legacy vLLM engine recipe, EOS ids, attention-path provenance | sealer surface (imported by the passkey extractor) |
| `glm_longctx.py` | passkey/needle construction and the L7 passkey criterion | `validation/long_context_oracle.py` pinned loader; runner and sealer |
| `dsa_throughput.py` | deterministic synthetic prompt builder behind the 256K E0 oracle | `validation/long_context_oracle.py`; `validation/ws32_delivery_quality.py` |

Because the bytes are pinned, docstrings inside these modules still mention
retired siblings (`run_bench.py`, `report_throughput.py`, `download_data.py`,
`test_bench.py`). Those files, the legacy runner and its reporters left main
during curation and are recoverable through the
[curation ledger](../docs/curation/README.md). Nothing here authorizes legacy
vLLM inference; see [inference](../docs/release/INFERENCE.md) for the supported
path.

## CPU tests

```bash
JAX_PLATFORMS=cpu ~/vllm-env/bin/python -m pytest -q bench/test_longctx.py bench/test_dsa_throughput.py
```

Both suites are offline (mock tokenizer, temporary SQLite, no vLLM import).
`bench/data/` and `bench/results.db` are runtime artifacts and stay gitignored.
