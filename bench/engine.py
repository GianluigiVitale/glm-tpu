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
RUNAI_STREAMER_CONCURRENCY / RUNAI_STREAMER_MEMORY_LIMIT (streaming load),
GLM_ASYNC_SCHED=0 (sync scheduling), GLM_LOG_STATS=1 (vLLM 10s engine stats:
tok/s + running/waiting — default off, unchanged behavior), GLM_DCP=N
(decode context parallelism — shard each sequence's KV cache across N ranks
at decode; default unset/0 = the kwarg is ABSENT from the engine args,
byte-identical engine build), GLM_SPEC_K=k (Stage-3 dense-MTP speculative
decoding, runbook §7 / docs/08 §7 M2: k draft tokens per step via
speculative_config={"method": "mtp", "num_speculative_tokens": k}; default
unset/0 = the kwarg is ABSENT, byte-identical engine build).
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


def attention_path() -> str:
    """The engine's attention path, for run provenance (audit 2026-07-07:
    record it explicitly so no benchmark number can be misattributed).
    'dense-mla' = Stage-1 (DSA indexer bypassed); 'dsa-sparse:<mode>' = the
    Stage-2 kernel path. Derived from the live GLM_DSA_MODE env, not inferred.
    ONE definition shared by run_bench._run_env and glm_longctx._run_env."""
    mode = os.environ.get("GLM_DSA_MODE", "off")
    return "dsa-sparse:" + mode if mode not in ("", "off") else "dense-mla"


# Fork-side observability envs that are read inside the ray WORKER processes.
# vLLM's Ray executor forwards only VLLM_* + a fixed allow-list to the
# workers, and GLM_* is NOT on tpu_platform.additional_env_vars -- a value
# exported only in the DRIVER shell silently never reaches them. That is the
# 2026-07-09 incident: the runbook's `GLM_DCP_CACHE_DUMP=... python
# glm_longctx.py` flow completed a full pod run with ZERO dump files and zero
# errors, because every worker read "" for the gate. (Names only; keep in
# sync with the fork's dump/guard modules.)
_WORKER_SIDE_OBS_ENVS = (
    "GLM_DCP_CACHE_DUMP",
    "GLM_DCP_CACHE_DUMP_PREFWD",
    "GLM_DCP_CACHE_DUMP_LAYERS",
    "GLM_DCP_DUMP_NEWKV",
    "GLM_DCP_DUMP_NEWKV_MAXCALLS",
    "GLM_DCP_DUMP_NEWKV_ONLY_PREFILL",
    "GLM_DCP_ASSERT_SHARDING",
    "GLM_DCP_ASSERT_CACHE_SANITY",
    "GLM_MLA_DCP",
    "GLM_DUMP_STEP_HLO",
    # Review of 6f45e0944: the remaining worker trace-time reads — including
    # GLM_EXPECT_CODE_HASH, the env that series itself introduced (verified in
    # TPUWorker.__init__ on the WORKERS; set driver-only it verifies nothing —
    # the exact incident class this warning exists for).
    "GLM_EXPECT_CODE_HASH",
    "GLM_DCP_SCATTER_IMPL",
    "GLM_DSA_DCP_SCATTER_IMPL",
    "GLM_DSA_DCP_HEADSPLIT",
    "GLM_DCP_SCATTER_ONLY",
    "GLM_DCP_GATHER_POS",
    "GLM_MLA_HEAD_SHARDED",
    "GLM_DCP_NO_DONATE",
)

# Boolean-gated names: an explicit "0"/"false" means DISABLED, not armed — do
# not flag it (review of 6f45e0944: `GLM_MLA_DCP=0` used to print the full
# warning paragraph). Value-carrying envs (dump paths/prefixes, layer lists,
# call caps, hash pins, impl selectors) stay flagged for any non-empty value.
_VALUE_CARRYING_OBS_ENVS = frozenset({
    "GLM_DCP_CACHE_DUMP", "GLM_DCP_CACHE_DUMP_LAYERS",
    "GLM_DCP_DUMP_NEWKV_MAXCALLS", "GLM_EXPECT_CODE_HASH",
    "GLM_DCP_SCATTER_IMPL",
})


def warn_worker_only_envs() -> list:
    """LOUD driver-side warning for worker-consumed envs set in this process.

    Setting these in the driver environment has NO effect unless the SAME
    value is also baked into every raylet env (relaunch with
    EXTRA_ENVS="NAME=value" scripts/launch_glm_32chip.sh) -- driver-only, a
    dump produces zero files and a guard never arms, both silently. Called
    from build_llm so every harness (run_bench, glm_longctx) warns before
    burning a pod run. Warning only (never fatal): the value may legitimately
    ALSO be raylet-baked, which this process cannot see. Default-inert: no
    flagged env, no output. Returns the flagged names (tests)."""
    flagged = [
        k for k in _WORKER_SIDE_OBS_ENVS
        if os.environ.get(k) and (k in _VALUE_CARRYING_OBS_ENVS or os.environ[
            k].strip().lower() not in ("0", "false"))
    ]
    for k in flagged:
        print(
            f"[bench] WARNING: {k} is set in the DRIVER environment, but it "
            "is read inside the ray WORKER processes and does NOT propagate "
            "driver->worker (vLLM forwards only VLLM_* + an allow-list). "
            "Driver-only it has NO effect -- the 2026-07-09 incident ran a "
            "full pod dump that wrote ZERO files this way. Ensure it is "
            f'raylet-baked: EXTRA_ENVS="{k}=..." '
            "scripts/launch_glm_32chip.sh, then confirm the worker logs "
            "show the ARMED line and verify dumps post-run with "
            "`python -m tpu_inference.runner.dcp_dump_check <prefix>`.",
            flush=True)
    return flagged


def build_llm(model: str, *, max_len: int = 8192, max_seqs: int = 8,
              max_batched_tokens: int = 4096, gmu: float = 0.94,
              num_gpu_blocks: int = 0, log_extra: str = ""):
    """Build the in-process vLLM engine on the 32-chip pod (the Stage-1 GLM
    recipe above). Returns the `vllm.LLM` instance.

    `num_gpu_blocks` > 0 caps the KV pool via `num_gpu_blocks_override` (frees
    HBM for the per-forward program — the DSV4 fragmentation-OOM lesson);
    0 = let vLLM auto-size the KV cache to the `gmu` budget ("kv auto").
    """
    # Worker-consumed observability envs set driver-only are inert: warn
    # BEFORE building the engine (and before the vllm import) so the
    # operator can abort instead of burning a pod run on a dump/guard that
    # will never arm. No flagged env = no output.
    warn_worker_only_envs()
    # vllm import stays INSIDE build_llm: `import engine` and the --stub paths
    # must work with no vllm/TPU (a module-top import would break them).
    # NOTE (round-4 review finding 3): resolving vllm.platforms here imports
    # tpu_inference in the DRIVER (vllm/platforms/tpu.py does
    # `from tpu_inference.platforms import TpuPlatform`) — that is expected
    # and TPU-neutral (no jax/libtpu init, no /tmp/libtpu_lockfile). The old
    # "keep tpu_inference out of the driver" story was wrong; the real
    # driver-side lockfile causes were get_page_size()'s jax.devices() call
    # (fixed in fork 7ae390f2) and leaked EngineCore procs (launcher stop
    # phase clears them). See bench/run_bench.py's mechanism-correction note.
    from vllm import LLM  # first vllm import — resolves vllm.platforms once, up front

    t0 = time.time()
    extra = {}
    if num_gpu_blocks and num_gpu_blocks > 0:
        extra["num_gpu_blocks_override"] = int(num_gpu_blocks)
    # GLM_ASYNC_SCHED=0 disables vLLM async scheduling (docs/06 probe P2:
    # both pod core-halts hit the finish-step under async; sync mode is the
    # discriminator AND a ~1%-cost mitigation at current step times).
    # Unset/1 keeps vLLM's default (async on for the Ray TPU executor).
    if os.environ.get("GLM_ASYNC_SCHED") == "0":
        extra["async_scheduling"] = False
    # GLM_LOG_STATS=1 enables vLLM's periodic engine stats logger (~10 s
    # cadence: prompt/generation tok/s, running/waiting request counts).
    # Offline `LLM()` forcibly defaults disable_log_stats=True, so pod runs
    # fly blind on throughput except the tqdm bar — this knob is the
    # observability counterpart to GLM_FLIGHT_RECORDER (fork side). Driver-
    # side env: no raylet baking needed. Default unset = quiet (unchanged).
    if os.environ.get("GLM_LOG_STATS") == "1":
        extra["disable_log_stats"] = False
    # GLM_DCP=N sets vLLM's decode_context_parallel_size (DCP: shard each
    # sequence's KV cache across N ranks at decode — the long-context KV-
    # capacity knob; tensor_parallel_size must be divisible by N, enforced by
    # vllm/config/parallel.py). Kwarg name verified against the installed
    # vLLM: EngineArgs.decode_context_parallel_size (engine/arg_utils.py) ->
    # ParallelConfig.decode_context_parallel_size, and the engine config dump
    # echoes `decode_context_parallel_size=1` by default (gpqa198.log).
    # Default unset (or 0/empty) = the kwarg is ABSENT from the LLM(...) args
    # entirely — byte-identical engine build to the pre-DCP harness.
    dcp = int(os.environ.get("GLM_DCP") or 0)
    if dcp < 0:
        # fail at the harness boundary with a readable message, not a pydantic
        # ValidationError deep in the engine build (vLLM's tp % dcp == 0 check
        # would NOT stop a negative — 32 % -2 == 0 in Python; only
        # ParallelConfig's ge=1 field constraint does — review finding).
        raise ValueError(f"GLM_DCP must be >= 1 (got {dcp}); unset/0 = off")
    if dcp:
        extra["decode_context_parallel_size"] = dcp
    # GLM_SPEC_K=k enables Stage-3 dense-MTP speculative decoding (docs/08 §7
    # M2, runbook §7): k draft tokens per step from GLM-5.2's own MTP layer 78.
    # Kwarg + dict shape verified against the installed vLLM
    # (~/vllm-build/vllm/engine/arg_utils.py): EngineArgs.speculative_config
    # is `dict[str, Any] | None` (line 616) and create_speculative_config
    # builds SpeculativeConfig(**dict); "mtp" is a valid SpeculativeMethod
    # (config/speculative.py MTPModelTypes), the glm_moe_dsa config surgery
    # maps it to DeepSeekMTPModel with n_predict=1 (hf_config_override), and
    # k > 1 passes the MTP-module-reuse check (k % n_predict == 0,
    # speculative.py ~773 — the draft layer is re-run k times). The fork
    # routes method "mtp" to Eagle3Proposer (tpu_runner.py:711 + eagle3.py's
    # method == "mtp" branches). Provenance: GLM_SPEC_K is recorded by the
    # GLM_* os_env sweep in run_bench._run_env / glm_longctx._run_env, and the
    # engine-built line below prints spec=mtp:k=<k>. Default unset (or
    # 0/empty) = the kwarg is ABSENT from the LLM(...) args entirely —
    # byte-identical engine build (the M1 target-forward hash test's contract:
    # speculative_config is None => nothing changes).
    raw_spec = os.environ.get("GLM_SPEC_K")
    try:
        spec_k = int(raw_spec or 0)
    except ValueError:
        # readable for garbage values too (round-9 review: the bare int()
        # error names neither the knob nor the rule).
        raise ValueError(f"GLM_SPEC_K must be an integer >= 1 (got "
                         f"{raw_spec!r}); unset/0 = off") from None
    if spec_k < 0:
        # readable harness-boundary error, not a pydantic ValidationError deep
        # in the engine build (SpeculativeConfig.num_speculative_tokens is
        # Field(gt=0) — same rationale as the GLM_DCP guard above).
        raise ValueError(f"GLM_SPEC_K must be >= 1 (got {spec_k}); "
                         "unset/0 = off")
    if spec_k:
        extra["speculative_config"] = {"method": "mtp",
                                       "num_speculative_tokens": spec_k}
    # GLM_KV_CACHE_DTYPE=fp8 stores the MLA latent KV cache in fp8_e4m3
    # (dequant->bf16 per-tile in the v4 kernel via _upcast_kv_for_v4). Halves
    # the latent footprint (128K: 12.2->6.1 GiB/chip), which lets a single
    # 128K sequence + the 23 GiB FP8 weights fit one chip's 30.75 GiB WITHOUT
    # DCP (6.1+23.06=29.15 < 30.75) — an independent route to the long-ctx
    # passkey gate that sidesteps the DCP striping path entirely. The fp8 KV
    # path pre-exists + is v4-validated (flash_attn_mla quantize_kv + kernel
    # _upcast_kv_for_v4); every fp8 branch is guarded on the sub-16-bit dtype,
    # so unset/"" = "auto" (bf16) is byte-identical. Retrieval survives fp8
    # (needle 97.5-100%); CONTENT precision degrades ~layer-cumulatively
    # (single-op L2 ~3.7% vs bf16 0.23%) -> validate generation quality on the
    # pod before trusting fp8-KV for non-retrieval benchmarks (docs/RESEARCH_LOG).
    _kv_dtype = os.environ.get("GLM_KV_CACHE_DTYPE", "").strip() or "auto"
    llm = LLM(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype=_kv_dtype,
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
          f"max_len={max_len}{f', dcp={dcp}' if dcp else ''}"
          f"{f', spec=mtp:k={spec_k}' if spec_k else ''}"
          f"{', ' + log_extra if log_extra else ''})", flush=True)
    return llm
