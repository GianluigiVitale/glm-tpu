#!/usr/bin/env python3
"""GLM-5.2 benchmark harness — loads a benchmark, queries the model, extracts +
scores every item, and records EVERYTHING to the provenance DB (bench/results.db).

Model-pluggable: `generate(prompt) -> str | (str, n_gen_tokens)`. Two modes:
  --stub     offline pipeline check (no model; vllm is NEVER imported — the vllm
             import lives inside make_generate so this runs on any CPU box)
  (default)  the REAL in-process vLLM engine on the 32-chip v4 pod
             (make_generate: runai_streamer GCS->HBM load, TP=32 x EP, greedy
             thinking-mode decode through GLM-5.2's own chat template).

Nothing is scored that is not stored. Every item row carries the verbatim prompt,
the verbatim model reply, the extracted answer, correct/score, n_gen_tokens,
timing, and the run-level provenance (model+revision, harness+fork git commits,
env/flags — pv.start_run + _run_env — and the pod).

    # offline pipeline check (no model): records items with empty replies
    python run_bench.py --benchmark gsm8k --limit 5 --stub

    # real run (pod-side, Ray cluster up on all 8 hosts):
    ~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 16 \
        --max-new 2048 --note "stage1 smoke"

    # multi-benchmark convenience (ONE engine build, sequential runs):
    ~/vllm-env/bin/python -u run_bench.py --benchmarks gsm8k,gpqa_diamond \
        --limit 50 --max-new 4096
"""
from __future__ import annotations

import argparse
import os
import time

import provenance as pv
import benchmarks as B
import engine  # CPU-safe: the vllm import lives inside engine.build_llm

HERE = os.path.dirname(os.path.abspath(__file__))

# The engine recipe (checkpoint default, EOS ids, LLM(...) build) lives in
# bench/engine.py — shared with glm_longctx.py so both harnesses run the
# IDENTICAL engine. Re-exported here so `run_bench.DEFAULT_MODEL` /
# `run_bench.EOS_IDS` keep working.
DEFAULT_MODEL = engine.DEFAULT_MODEL
EOS_IDS = engine.EOS_IDS


def stub_generate(prompt: str) -> str:
    """Placeholder model: returns nothing (all items score wrong) — used only to
    validate the load→prompt→extract→score→store pipeline offline."""
    return ""


def _chat_prompt_ids(tok, prompt: str, chat_template: str | None = None) -> list[int]:
    """Render ONE user turn through GLM-5.2's own chat template and tokenize.

    The shipped chat_template.jinja produces (defaults: thinking ON, effort Max):
        [gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{prompt}<|assistant|><think>
    i.e. the model continues INSIDE the think block, reasons, closes </think>,
    then answers (the bench prompts' _THINK_HINT asks for \\boxed{}); the
    extractors strip the think block. Token ids are returned (and passed to the
    engine as prompt_token_ids) so there is no re-tokenization ambiguity.
    `chat_template` is a fallback template TEXT for a tokenizer that did not pick
    up the checkpoint's chat_template.jinja (None = use the tokenizer's own).
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=True, return_dict=False,
        chat_template=chat_template)


def make_generate(model: str, *, max_len: int = 8192, max_new: int = 2048,
                  max_seqs: int = 8, max_batched_tokens: int = 4096,
                  gmu: float = 0.94, num_gpu_blocks: int = 0,
                  temperature: float = 0.0):
    """Build the REAL GLM-5.2 generator: an in-process vLLM engine on the
    32-chip pod, modeled on the DSV4 pattern (~/moe-tpu/bench/dsv4_gen_bench.py
    build_llm + Generator) with ONE deliberate topology difference:

      GLM Stage 1 runs WITHOUT DP attention — pure TP x EP. No
      `additional_config={"sharding": ...}` / `enable_dp_attention` (that was
      DSV4's MLA recipe; the GLM fork branch does not require it).

    The LLM(...) build itself lives in bench/engine.py (shared with
    glm_longctx.py). Env knobs: GLM_MODEL (checkpoint), GLM_TP (default 32),
    RUNAI_STREAMER_CONCURRENCY / RUNAI_STREAMER_MEMORY_LIMIT (streaming load).
    Returns `generate(prompt) -> (text, n_gen_tokens)` — the VERBATIM completion
    text plus the generated-token count (for the items.n_gen_tokens column).
    """
    # vllm import stays INSIDE make_generate: `import run_bench` and --stub must
    # work with no vllm/TPU (module-top import would break the offline pipeline).
    from vllm import SamplingParams

    llm = engine.build_llm(model, max_len=max_len, max_seqs=max_seqs,
                           max_batched_tokens=max_batched_tokens, gmu=gmu,
                           num_gpu_blocks=num_gpu_blocks,
                           log_extra=f"max_new={max_new}")

    tok = llm.get_tokenizer()
    chat_template = None
    if getattr(tok, "chat_template", None) is None:
        # The checkpoint ships chat_template.jinja (transformers 5.x auto-loads
        # it); if this tokenizer instance didn't pick it up, fall back to the
        # committed reference copy (verified byte-identical to the HF repo).
        ref = os.path.join(HERE, "..", "reference", "hf-repo", "chat_template.jinja")
        with open(ref, encoding="utf-8") as f:
            chat_template = f.read()
        print("[bench] tokenizer had no chat template — using the reference "
              "chat_template.jinja copy", flush=True)

    def generate(prompt: str) -> tuple[str, int]:
        ids = _chat_prompt_ids(tok, prompt, chat_template)
        room = max_len - len(ids) - 8
        if room <= 0:
            # Prompt alone exceeds the context window: record an empty reply
            # (scored wrong, auditable) rather than crash the run.
            print(f"[bench] SKIP: prompt {len(ids)} tok > max_len {max_len}",
                  flush=True)
            return "", 0
        sp = SamplingParams(
            temperature=temperature,          # 0.0 = greedy (the bench protocol)
            max_tokens=min(max_new, room),
            stop_token_ids=EOS_IDS,
            ignore_eos=False,
        )
        outs = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
        o = outs[0].outputs[0]
        return o.text, len(o.token_ids)

    return generate


def run_benchmark(conn, run_id, spec: B.BenchSpec, generate, limit=None, seed=0):
    items = B.load_items(spec, limit=limit)
    n_correct = 0
    for i, it in enumerate(items):
        t0 = time.time()
        out = generate(it.prompt)
        # generate may return plain text (stub) or (text, n_gen_tokens) (engine).
        reply, n_gen = out if isinstance(out, tuple) else (out, None)
        latency = (time.time() - t0) * 1000.0
        # The audit trail must never be lost: store the verbatim reply even if
        # extraction/scoring raises (extracted/correct = None on failure).
        try:
            extracted = spec.extract(reply, it)
            correct = spec.score(extracted, it.gold)
        except Exception:
            extracted, correct = None, None
        n_correct += int(bool(correct))
        pv.record_item(conn, run_id, benchmark=spec.name, item_id=it.item_id,
                       prompt=it.prompt, gold=it.gold, raw_output=reply,
                       extracted=extracted, correct=correct,
                       score=(1.0 if correct else 0.0) if correct is not None else None,
                       n_gen_tokens=n_gen, latency_ms=round(latency, 1), seed=seed)
        print(f"[{spec.name}] {i + 1}/{len(items)} {it.item_id}: "
              f"correct={correct} extracted={extracted!r} gold={it.gold!r} "
              f"gen_tok={n_gen} {latency / 1000.0:.1f}s", flush=True)
    summ = pv.finalize(conn, run_id, benchmark=spec.name, metric="acc")
    print(f"[{spec.name}] n={summ['n']} acc={summ['value']}"
          f" card={summ['card_value']} Δ={summ['delta']}")
    return summ


def _run_env(args, benches) -> dict:
    """Run-level env/config provenance (stored in runs.env_json alongside the
    harness/fork git hashes that pv.start_run already records). Secrets are
    filtered — never store tokens/keys."""
    os_env = {k: v for k, v in sorted(os.environ.items())
              if k.startswith(("GLM_", "RUNAI_STREAMER_", "VLLM_", "TPU_", "JAX_",
                               "NEW_MODEL_DESIGN", "MODEL_IMPL_TYPE",
                               "OMP_NUM_THREADS"))
              and not any(s in k.upper() for s in ("TOKEN", "KEY", "SECRET"))}
    return {
        "model": args.model, "stub": bool(args.stub),
        # pinned dataset commit shas (BenchSpec.hf_revision) — reproducibility
        "dataset_revisions": {b: B.REGISTRY[b].hf_revision for b in benches},
        "tp": int(os.environ.get("GLM_TP", "32")),
        "dp_attention": False,   # GLM Stage 1: pure TP x EP (no DP attention)
        "expert_parallel": True,
        "max_len": args.max_len, "max_new": args.max_new,
        "max_seqs": args.max_seqs, "max_batched_tokens": args.max_batched_tokens,
        "gmu": args.gmu, "num_gpu_blocks": args.num_gpu_blocks,
        "temperature": 0.0, "eos_ids": EOS_IDS,
        "load_format": "runai_streamer",
        "os_env": os_env,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(B.REGISTRY), default=None)
    ap.add_argument("--benchmarks", default=None,
                    help="comma-separated multi-run, e.g. gsm8k,gpqa_diamond "
                         "(ONE engine build; overrides --benchmark)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="checkpoint path/repo (default: GLM_MODEL env or the "
                         "staged us-central2 GCS copy)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--stub", action="store_true",
                    help="offline pipeline check (no model — all items score wrong)")
    ap.add_argument("--max-new", type=int, default=2048,
                    help="max generated tokens; GLM-5.2 thinks before answering — "
                         "AIME/GPQA need long CoT, raise (e.g. 4096-16384)")
    ap.add_argument("--max-len", type=int, default=8192,
                    help="max_model_len (prompt + generation)")
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--max-batched-tokens", type=int, default=4096,
                    help="chunked-prefill chunk size")
    ap.add_argument("--num-gpu-blocks", type=int, default=0,
                    help="num_gpu_blocks_override if >0 (cap the KV pool to free "
                         "HBM for the per-forward program; 0 = auto)")
    ap.add_argument("--gmu", type=float,
                    default=float(os.environ.get("GLM_GMU", "0.94")),
                    help="gpu_memory_utilization")
    args = ap.parse_args()

    benches = [b.strip() for b in
               (args.benchmarks or args.benchmark or "").split(",") if b.strip()]
    if not benches:
        ap.error("provide --benchmark NAME or --benchmarks a,b,c")
    unknown = [b for b in benches if b not in B.REGISTRY]
    if unknown:
        ap.error(f"unknown benchmark(s) {unknown}; have {list(B.REGISTRY)}")

    conn = pv.connect()
    run_id = pv.start_run(conn, model=("STUB" if args.stub else args.model),
                          revision=args.revision, env=_run_env(args, benches),
                          note=args.note or ("offline-stub" if args.stub else ""))
    gen = stub_generate if args.stub else make_generate(
        args.model, max_len=args.max_len, max_new=args.max_new,
        max_seqs=args.max_seqs, max_batched_tokens=args.max_batched_tokens,
        gmu=args.gmu, num_gpu_blocks=args.num_gpu_blocks)
    for b in benches:
        run_benchmark(conn, run_id, B.REGISTRY[b], gen, limit=args.limit)


if __name__ == "__main__":
    main()
