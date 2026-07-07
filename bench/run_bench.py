#!/usr/bin/env python3
"""GLM-5.2 benchmark harness — loads a benchmark, queries the model, extracts +
scores every item, and records EVERYTHING to the provenance DB (bench/results.db).

Model-pluggable: `generate(prompt) -> str | (str, n_gen_tokens) |
(str, n_gen_tokens, finish_reason)`. Two modes:
  --stub     offline pipeline check (no model; vllm is NEVER imported — the vllm
             import lives inside make_generate so this runs on any CPU box)
  (default)  the REAL in-process vLLM engine on the 32-chip v4 pod
             (make_generate: runai_streamer GCS->HBM load, TP=32 x EP, greedy
             thinking-mode decode through GLM-5.2's own chat template).

Engine runs are BATCHED: make_generate's generator also exposes
`generate_batch(prompts) -> [(text, n_gen_tokens, finish_reason), ...]`, and run_benchmark
submits ALL items (or --batch-size chunks) in ONE llm.generate call so vLLM's
scheduler runs up to --max-seqs sequences concurrently (~1 tok/s single-stream
decode -> batch-factor aggregate throughput). Provenance is unchanged per item;
per-item latency_ms is NULL in batched runs (never faked) — the real batch wall
time is recorded in the run's summary note as batch_wall_ms.

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


def _sort_by_request_id(outs):
    """Defensively re-assert submission order on vLLM outputs. The LLM API
    already returns outputs in input order (its _run_engine sorts by request
    id), but the mapping back to items is correctness-critical — so re-sort by
    the integer request id when possible instead of trusting it silently."""
    try:
        return sorted(outs, key=lambda o: int(o.request_id))
    except (TypeError, ValueError):
        return list(outs)


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
    Returns `generate(prompt) -> (text, n_gen_tokens, finish_reason)` — the
    VERBATIM completion text, the generated-token count (items.n_gen_tokens),
    and vLLM's finish_reason ('length' = truncated at max_tokens; recorded in
    the items row, still scored as-is).
    The returned callable also exposes `generate.generate_batch(prompts) ->
    [(text, n_gen_tokens, finish_reason), ...]` (one llm.generate call for MANY prompts —
    vLLM's scheduler then runs up to max_seqs sequences concurrently, which is
    what turns ~1 tok/s single-stream decode into batch-factor aggregate
    throughput). run_benchmark auto-uses it when present.
    """
    # vllm import stays INSIDE make_generate: `import run_bench` and --stub must
    # work with no vllm/TPU (module-top import would break the offline pipeline).
    # Import order: `from vllm import LLM` first so vllm.platforms resolves
    # once, up front (importing a vllm submodule before platform resolution
    # can re-enter vllm.platforms: ImportError current_platform).
    # MECHANISM CORRECTION (round-4 review, docs/reviews/round4-launcher-bench.md
    # finding 3 — an earlier comment here claimed import order kept
    # tpu_inference out of the driver and that its import held the libtpu
    # lockfile; both claims were false): the driver imports tpu_inference on
    # EVERY run regardless — vllm.platforms resolution itself does
    # `from tpu_inference.platforms import TpuPlatform`. That import is
    # TPU-NEUTRAL (env-var overrides, GCE metadata, /dev/accel glob — no
    # jax/libtpu client init, no lockfile). The historical "ABORTED: lockfile"
    # hangs were caused by (1) the fork's get_page_size() calling
    # jax.devices() in the DRIVER config path — fixed in fork commit 7ae390f2
    # (see flash_attn_mla.py's NOTE) — and (2) a leaked EngineCore holding the
    # lock (cleared by the launcher stop phase, harness commit 33576f9).
    # Driver TPU-neutrality holds because no driver-side call path touches
    # jax devices TODAY — it is not guaranteed by import order.
    from vllm import LLM  # noqa: F401
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

    def _sp(room: int) -> "SamplingParams":
        # ONE sampling protocol for both paths: greedy, the GLM EOS set, and
        # max_tokens clamped to the room this prompt leaves in the window
        # (identical to what the sequential path always did).
        return SamplingParams(
            temperature=temperature,          # 0.0 = greedy (the bench protocol)
            max_tokens=min(max_new, room),
            stop_token_ids=EOS_IDS,
            ignore_eos=False,
        )

    def generate(prompt: str) -> tuple[str, int, str | None]:
        ids = _chat_prompt_ids(tok, prompt, chat_template)
        room = max_len - len(ids) - 8
        if room <= 0:
            # Prompt alone exceeds the context window: record an empty reply
            # (scored wrong, auditable) rather than crash the run.
            print(f"[bench] SKIP: prompt {len(ids)} tok > max_len {max_len}",
                  flush=True)
            return "", 0, None
        outs = llm.generate([{"prompt_token_ids": ids}], _sp(room),
                            use_tqdm=False)
        o = outs[0].outputs[0]
        # finish_reason ('stop' | 'length' | ...) rides along for truncation
        # honesty: 'length' = the reply hit max_tokens mid-CoT and is flagged
        # truncated in the items row (still scored as-is, never excluded).
        return o.text, len(o.token_ids), getattr(o, "finish_reason", None)

    def generate_batch(prompts: list[str]) -> list[tuple[str, int, str | None]]:
        """ONE llm.generate call for a list of prompts — vLLM schedules up to
        max_seqs of them concurrently. Same sampling protocol as generate()
        (the per-prompt SamplingParams differ ONLY in the room clamp on
        max_tokens, exactly as the sequential path computed it). Results are
        returned IN INPUT ORDER: over-long prompts keep their slot as
        ("", 0, None) (recorded empty + scored wrong, same as the sequential
        SKIP), and the vLLM outputs are mapped back by request order/id. Each
        result carries vLLM's finish_reason (truncation honesty — additive,
        nothing else about the batched protocol changed)."""
        results: list[tuple[str, int, str | None] | None] = [None] * len(prompts)
        submit_idx: list[int] = []
        token_prompts, sps = [], []
        for i, prompt in enumerate(prompts):
            ids = _chat_prompt_ids(tok, prompt, chat_template)
            room = max_len - len(ids) - 8
            if room <= 0:
                print(f"[bench] SKIP: prompt {len(ids)} tok > max_len "
                      f"{max_len}", flush=True)
                results[i] = ("", 0, None)
                continue
            submit_idx.append(i)
            token_prompts.append({"prompt_token_ids": ids})
            sps.append(_sp(room))
        if token_prompts:
            print(f"[bench] submitting {len(token_prompts)} prompts in ONE "
                  f"llm.generate (up to {max_seqs} concurrent)", flush=True)
            outs = _sort_by_request_id(
                llm.generate(token_prompts, sps, use_tqdm=True))
            assert len(outs) == len(token_prompts), \
                f"vLLM returned {len(outs)} outputs for {len(token_prompts)} prompts"
            for i, out in zip(submit_idx, outs):
                o = out.outputs[0]
                results[i] = (o.text, len(o.token_ids),
                              getattr(o, "finish_reason", None))
        return results  # type: ignore[return-value]  # every slot is filled

    generate.generate_batch = generate_batch
    return generate


def _score_and_record(conn, run_id, spec: B.BenchSpec, it, reply, n_gen,
                      latency_ms, seed, finish_reason=None):
    """Extract + score ONE reply and store the full item row. The audit trail
    must never be lost: the verbatim reply is stored even if extraction/scoring
    raises (extracted/correct = None on failure). finish_reason (vLLM's, when
    the generator provides it) flags truncation: 'length' means the reply hit
    max_tokens mid-CoT — the item is STILL SCORED AS-IS (scoring unchanged;
    salvage extraction from a truncated CoT can be wrong in both directions)
    but the row records finish_reason + truncated so it is auditable."""
    try:
        extracted = spec.extract(reply, it)
        correct = spec.score(extracted, it.gold)
    except Exception:
        extracted, correct = None, None
    truncated = None if finish_reason is None else (finish_reason == "length")
    pv.record_item(conn, run_id, benchmark=spec.name, item_id=it.item_id,
                   prompt=it.prompt, gold=it.gold, raw_output=reply,
                   extracted=extracted, correct=correct,
                   score=(1.0 if correct else 0.0) if correct is not None else None,
                   n_gen_tokens=n_gen, latency_ms=latency_ms, seed=seed,
                   finish_reason=finish_reason, truncated=truncated)
    return extracted, correct


def _run_items_batched(conn, run_id, spec: B.BenchSpec, generate_batch, items,
                       seed=0, batch_size=0) -> str:
    """Submit prompts in bulk through generate_batch (ONE llm.generate per
    chunk; batch_size <= 0 = ALL items in one call) so vLLM runs up to max_seqs
    sequences concurrently. Provenance per item is EXACTLY the sequential
    path's (verbatim prompt/raw output/extracted/correct/n_gen_tokens); only
    per-item latency is not individually measurable inside a batch, so
    latency_ms is stored as NULL — never faked — and the real batch wall times
    go into the summary note (returned) as batch_wall_ms."""
    chunks = ([items] if batch_size <= 0 else
              [items[i:i + batch_size] for i in range(0, len(items), batch_size)])
    total_wall_ms, total_tok, done = 0.0, 0, 0
    for chunk in chunks:
        t0 = time.time()
        outs = generate_batch([it.prompt for it in chunk])
        wall_ms = (time.time() - t0) * 1000.0
        total_wall_ms += wall_ms
        assert len(outs) == len(chunk), \
            f"generate_batch returned {len(outs)} results for {len(chunk)} prompts"
        chunk_tok = 0
        for it, out in zip(chunk, outs):
            # additive: results are (reply, n_gen) historically, now
            # (reply, n_gen, finish_reason) — accept both.
            reply, n_gen = out[0], out[1]
            finish = out[2] if len(out) > 2 else None
            extracted, correct = _score_and_record(
                conn, run_id, spec, it, reply, n_gen, latency_ms=None,
                seed=seed, finish_reason=finish)
            done += 1
            chunk_tok += n_gen or 0
            trunc_flag = " TRUNCATED" if finish == "length" else ""
            print(f"[{spec.name}] {done}/{len(items)} {it.item_id}: "
                  f"correct={correct} extracted={extracted!r} gold={it.gold!r} "
                  f"gen_tok={n_gen}{trunc_flag} (batched)", flush=True)
        total_tok += chunk_tok
        rate = chunk_tok / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
        print(f"[{spec.name}] batch of {len(chunk)} done in "
              f"{wall_ms / 1000.0:.1f}s ({chunk_tok} gen tok, "
              f"{rate:.1f} tok/s aggregate)", flush=True)
    return (f"batched:{len(items)} chunks={len(chunks)} "
            f"batch_wall_ms={total_wall_ms:.0f} gen_tok={total_tok}")


def run_benchmark(conn, run_id, spec: B.BenchSpec, generate, limit=None, seed=0,
                  batch_size=0):
    items = B.load_items(spec, limit=limit)
    note = ""
    generate_batch = getattr(generate, "generate_batch", None)
    if generate_batch is not None:
        # BATCHED: the engine generator exposes generate_batch — submit in bulk
        # (the stub does not, and keeps the sequential path below unchanged).
        note = _run_items_batched(conn, run_id, spec, generate_batch, items,
                                  seed=seed, batch_size=batch_size)
    else:
        for i, it in enumerate(items):
            t0 = time.time()
            out = generate(it.prompt)
            # generate may return plain text (stub), (text, n_gen_tokens), or
            # (text, n_gen_tokens, finish_reason) (engine).
            if isinstance(out, tuple):
                reply, n_gen = out[0], out[1]
                finish = out[2] if len(out) > 2 else None
            else:
                reply, n_gen, finish = out, None, None
            latency = (time.time() - t0) * 1000.0
            extracted, correct = _score_and_record(
                conn, run_id, spec, it, reply, n_gen,
                latency_ms=round(latency, 1), seed=seed, finish_reason=finish)
            trunc_flag = " TRUNCATED" if finish == "length" else ""
            print(f"[{spec.name}] {i + 1}/{len(items)} {it.item_id}: "
                  f"correct={correct} extracted={extracted!r} gold={it.gold!r} "
                  f"gen_tok={n_gen}{trunc_flag} {latency / 1000.0:.1f}s",
                  flush=True)
    summ = pv.finalize(conn, run_id, benchmark=spec.name, metric="acc", note=note)
    print(f"[{spec.name}] n={summ['n']} acc={summ['value']}"
          f" card={summ['card_value']} Δ={summ['delta']}"
          f" n_truncated={summ['n_truncated']}"
          f"{'  [' + note + ']' if note else ''}")
    return summ


# The env vars that decide the SERVED NUMERICS (FP8 checkpoint-exactness, the
# DSA bypass, the token-bucket floor). The workers' effective env is a MERGE
# the DB could previously not see: raylet-baked (the launcher's ENVS string)
# OVERRIDDEN by any driver-side value for vars on vLLM's TPU allow-list
# (tpu_platform.py additional_env_vars -> ray_distributed_executor copies
# driver os.environ to the workers when set). DISABLE_WEIGHT_REQUANTIZATION /
# TPU_DISABLE_DSA_INDEXER / TPU_MIN_TOKEN_BUCKET are on that allow-list —
# a stale driver-shell export silently changes what all 32 chips serve.
# REQUANTIZE_WEIGHT_DTYPE is NOT on it (a driver value stays driver-only), but
# a mismatch still signals a confused environment. Record the launcher-baked
# string verbatim + a live comparison; warn LOUDLY on mismatch.
_NUMERICS_CRITICAL_ENVS = ("DISABLE_WEIGHT_REQUANTIZATION",
                           "REQUANTIZE_WEIGHT_DTYPE",
                           "TPU_DISABLE_DSA_INDEXER",
                           "TPU_MIN_TOKEN_BUCKET")
_LAUNCHER_SH = os.path.abspath(
    os.path.join(HERE, "..", "scripts", "launch_glm_32chip.sh"))


def _launcher_env_provenance() -> dict:
    """Read the launcher's ENVS string (the env baked into every raylet — what
    the workers actually run with) at RUN time and compare the numerics-critical
    vars against the driver's live os.environ. Returns env_json fields:
    'launcher_envs_baked' (the verbatim string) and 'launcher_env_check'
    ({var: {baked, baked_nominal, driver}}). A `${VAR:-default}` token in the
    ENVS string is a launch-shell passthrough; its default is the nominal baked
    value (the value at actual launch time is not reconstructable here)."""
    import re
    out: dict = {"launcher_envs_baked": None, "launcher_env_check": {}}
    try:
        with open(_LAUNCHER_SH, encoding="utf-8") as f:
            text = f.read()
        m = re.search(r'^ENVS="export (.*)"$', text, re.MULTILINE)
        if not m:
            out["launcher_envs_baked"] = f"UNPARSED: no ENVS line in {_LAUNCHER_SH}"
            return out
        baked_str = m.group(1)
        out["launcher_envs_baked"] = baked_str
        for name in _NUMERICS_CRITICAL_ENVS:
            vm = re.search(rf"\b{name}=(\S+)", baked_str)
            baked_raw = vm.group(1) if vm else None
            baked = baked_raw
            if baked_raw:
                dm = re.fullmatch(r"\$\{" + name + r":-([^}]*)\}", baked_raw)
                if dm:
                    baked = dm.group(1)
            driver = os.environ.get(name)
            out["launcher_env_check"][name] = {
                "baked": baked_raw, "baked_nominal": baked, "driver": driver}
            if driver is not None and baked is not None and driver != baked:
                print("!" * 78, flush=True)
                print(f"[bench] WARNING: driver os.environ[{name!r}] = "
                      f"{driver!r} != launcher-baked {baked!r}.\n"
                      "  DISABLE_WEIGHT_REQUANTIZATION / TPU_DISABLE_DSA_INDEXER /"
                      " TPU_MIN_TOKEN_BUCKET\n"
                      "  FORCE-PROPAGATE driver->workers (tpu_platform"
                      " additional_env_vars), overriding\n"
                      "  the raylet-baked value on all 32 chips — served numerics"
                      " may differ from the\n"
                      "  launcher recipe. Both values are recorded in"
                      " runs.env_json['launcher_env_check'].", flush=True)
                print("!" * 78, flush=True)
    except Exception as exc:  # never fail a run over provenance introspection
        out["launcher_envs_baked"] = f"UNAVAILABLE: {exc}"
    return out


def _run_env(args, benches) -> dict:
    """Run-level env/config provenance (stored in runs.env_json alongside the
    harness/fork git hashes that pv.start_run already records). Secrets are
    filtered — never store tokens/keys."""
    os_env = {k: v for k, v in sorted(os.environ.items())
              if k.startswith(("GLM_", "RUNAI_STREAMER_", "VLLM_", "TPU_", "JAX_",
                               "NEW_MODEL_DESIGN", "MODEL_IMPL_TYPE",
                               "OMP_NUM_THREADS",
                               # numerics-critical, prefix-matched exactly —
                               # previously unrecorded (round-4 finding 2)
                               "DISABLE_WEIGHT_REQUANTIZATION",
                               "REQUANTIZE_WEIGHT_DTYPE"))
              and not any(s in k.upper() for s in ("TOKEN", "KEY", "SECRET"))}
    return {
        **_launcher_env_provenance(),
        "model": args.model, "stub": bool(args.stub),
        # pinned dataset commit shas (BenchSpec.hf_revision) — reproducibility
        "dataset_revisions": {b: B.REGISTRY[b].hf_revision for b in benches},
        "tp": int(os.environ.get("GLM_TP", "32")),
        "dp_attention": False,   # GLM Stage 1: pure TP x EP (no DP attention)
        "expert_parallel": True,
        "max_len": args.max_len, "max_new": args.max_new,
        "max_seqs": args.max_seqs, "max_batched_tokens": args.max_batched_tokens,
        "batch_size": args.batch_size,
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
    ap.add_argument("--batch-size", type=int, default=0,
                    help="engine runs submit prompts in chunks of this size "
                         "through ONE llm.generate call each (vLLM schedules "
                         "up to --max-seqs concurrently); 0 = ALL items in one "
                         "call. --stub stays sequential (no generate_batch). "
                         "Per-item latency_ms is NULL in batched runs — the "
                         "batch wall time is recorded in the summary note.")
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
        run_benchmark(conn, run_id, B.REGISTRY[b], gen, limit=args.limit,
                      batch_size=args.batch_size)


if __name__ == "__main__":
    main()
