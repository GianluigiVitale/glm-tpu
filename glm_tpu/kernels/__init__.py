"""Pallas TPU kernels, one package per operation; each ``kernel.py`` holds the public entry point.

``sparse_mla``: sparse multi-head latent attention over the selected KV positions (decode, and the
owner-local partial with log-sum-exp outputs that prefill merges). ``fp8_grouped_matmul``: routed-expert
matmuls over FP8 weights with 128-block scales (decode tiles, prefill expert panels)."""
