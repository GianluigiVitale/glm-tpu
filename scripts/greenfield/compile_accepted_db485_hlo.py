#!/usr/bin/env python3
"""Initialize the exact accepted DB485 engine and exit before generation."""

from __future__ import annotations

import json
import os
from hashlib import sha256

ACCEPTED_CODE_PIN = "b3c25df47ac98783912dc658878181ec0a8ae16d"
ACCEPTED_MODEL = "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
ACCEPTED_VLLM_PIN = "a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c"


def accepted_engine_kwargs() -> dict[str, object]:
    """Return the exact DB485 ``LLM`` construction contract."""

    return {
        "model": ACCEPTED_MODEL,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "kv_cache_dtype": "auto",
        "max_model_len": 8704,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 2048,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": 0.90,
        "tensor_parallel_size": 32,
        "enable_expert_parallel": True,
        "distributed_executor_backend": "ray",
        "load_format": "runai_streamer",
        "model_loader_extra_config": {
            "concurrency": 32,
            "memory_limit": 32 * 1024**3,
        },
        "num_gpu_blocks_override": 24,
        "async_scheduling": False,
        "disable_log_stats": False,
        "decode_context_parallel_size": 1,
    }


def main() -> int:
    tag = os.environ.get("GLM_ACCEPTED_DB485_COMPILE_ONLY_TAG", "")
    if not tag:
        raise SystemExit("GLM_ACCEPTED_DB485_COMPILE_ONLY_TAG is required")
    print(
        "ACCEPTED_DB485_COMPILE_ONLY_START "
        f"tag={tag} code={ACCEPTED_CODE_PIN} generate_calls=0",
        flush=True,
    )
    runtime_config = {
        "engine_kwargs": accepted_engine_kwargs(),
        "tpu_inference_pin": ACCEPTED_CODE_PIN,
        "vllm_pin": ACCEPTED_VLLM_PIN,
    }
    runtime_json = json.dumps(runtime_config, separators=(",", ":"), sort_keys=True)
    print(
        "ACCEPTED_DB485_RUNTIME_CONFIG "
        f"sha256={sha256(runtime_json.encode()).hexdigest()} json={runtime_json}",
        flush=True,
    )
    from vllm import LLM

    # LLM construction performs the ordinary seven-bucket AOT precompile.
    # There is deliberately no prompt, scheduler request, or generate call.
    _llm = LLM(**accepted_engine_kwargs())
    print(
        f"ACCEPTED_DB485_COMPILE_ONLY_COMPLETE tag={tag} generate_calls=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
