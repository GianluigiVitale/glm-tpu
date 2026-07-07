#!/usr/bin/env python3
"""Shared GLM-5.2 vLLM engine builder for the bench harnesses.

ONE place holds the Stage-1 serving recipe so every harness (run_bench.py,
glm_longctx.py) builds the IDENTICAL engine:

  - pure TP x EP — GLM Stage 1 runs WITHOUT DP attention: no
    `additional_config={"sharding": ...}` / `enable_dp_attention` (that was
    DSV4's MLA recipe; the GLM fork branch does not require it),
  - `load_format="runai_streamer"` — stream gs:// -> HBM directly, no local
    copy, no gcsfuse (the proven DSV4 fast-load path),
  - `kv_cache_dtype="auto"`,
  - FP8 weights kept CHECKPOINT-EXACT: `DISABLE_WEIGHT_REQUANTIZATION=1`
    (+ `REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn`, `TPU_DISABLE_DSA_INDEXER=1`)
    are baked into the raylet env by scripts/launch_glm_32chip.sh — do NOT set
    REQUANTIZE_WEIGHT_DTYPE=bfloat16 (the DSV4 load-time dequant-to-bf16 path
    OOMs at GLM-5.2's 753B scale).

CPU-SAFE TO IMPORT: the vllm import lives INSIDE build_llm(), so the offline
paths (run_bench --stub, glm_longctx --stub, the CPU tests) never import
vllm or touch TPU code.

Env knobs: GLM_MODEL (checkpoint), GLM_TP (default 32),
RUNAI_STREAMER_CONCURRENCY / RUNAI_STREAMER_MEMORY_LIMIT (streaming load).
"""
from __future__ import annotations

import os
import time

# Default checkpoint: the FP8-native staged copy in the SAME-REGION (us-central2)
# bucket — streamed GCS->HBM at load (runai_streamer), never the EU bucket.
DEFAULT_MODEL = os.environ.get("GLM_MODEL",
                               "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8")

# GLM-5.2 stop tokens, from the checkpoint's generation_config.json (verified
# against the real tokenizer 2026-07-07):
#   154820 = <|endoftext|>   154827 = <|user|>   154829 = <|observation|>
EOS_IDS = [154820, 154827, 154829]


def build_llm(model: str, *, max_len: int = 8192, max_seqs: int = 8,
              max_batched_tokens: int = 4096, gmu: float = 0.94,
              num_gpu_blocks: int = 0, log_extra: str = ""):
    """Build the in-process vLLM engine on the 32-chip pod (the Stage-1 GLM
    recipe above). Returns the `vllm.LLM` instance.

    `num_gpu_blocks` > 0 caps the KV pool via `num_gpu_blocks_override` (frees
    HBM for the per-forward program — the DSV4 fragmentation-OOM lesson);
    0 = let vLLM auto-size the KV cache to the `gmu` budget ("kv auto").
    """
    # vllm import stays INSIDE build_llm: `import engine` and the --stub paths
    # must work with no vllm/TPU (a module-top import would break them).
    from vllm import LLM

    t0 = time.time()
    extra = {}
    if num_gpu_blocks and num_gpu_blocks > 0:
        extra["num_gpu_blocks_override"] = int(num_gpu_blocks)
    llm = LLM(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        max_model_len=max_len,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_batched_tokens,   # chunked-prefill chunk size
        enable_prefix_caching=False,
        gpu_memory_utilization=gmu,
        tensor_parallel_size=int(os.environ.get("GLM_TP", "32")),
        enable_expert_parallel=True,     # EP: 256 routed experts sharded chip-wise
        distributed_executor_backend="ray",
        load_format="runai_streamer",    # stream GCS -> HBM, no local copy
        model_loader_extra_config={
            "concurrency": int(os.environ.get("RUNAI_STREAMER_CONCURRENCY", "32")),
            "memory_limit": int(os.environ.get("RUNAI_STREAMER_MEMORY_LIMIT",
                                               str(32 * 1024**3))),
        },
        **extra,
    )
    print(f"[bench] engine built in {time.time() - t0:.1f}s "
          f"(model={model}, tp={os.environ.get('GLM_TP', '32')}, ep=on, "
          f"max_len={max_len}{', ' + log_extra if log_extra else ''})", flush=True)
    return llm
