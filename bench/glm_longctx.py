#!/usr/bin/env python3
"""Long-context passkey / needle-in-a-haystack harness for GLM-5.2 on the
32-chip TPU v4 pod — the STAGE-2 THRESHOLD INSTRUMENT (PLAN.md: passkey
retrieval >=95% out to >=128K is the gate for the DSA kernel work).

Ported from the DSV4 harness (~/moe-tpu/bench/dsv4_longctx.py) with one
protocol change: GENERATION-BASED retrieval instead of DSV4's
loglikelihood-candidate scoring. DSV4 scored candidate continuations because
its target was a BASE model on a prefill-only validated path; GLM-5.2 has a
working greedy decode path, so we do the canonical thing — hide a 6-digit
passkey at a controlled DEPTH inside filler of a target token LENGTH, ask for
it, greedy-decode <=20 tokens, and exact-match the extracted digits. The model
can only produce the key if it ATTENDED to the needle far back in the context.

PROMPT-STYLE DECISION — RAW COMPLETION, not the chat template (documented):
  * GLM-5.2's chat template ends with `<|assistant|><think>` — the model
    reasons at length before answering, so a <=20-token greedy budget would be
    consumed by thinking preamble, and a large budget would measure "reasoning
    length", not retrieval.
  * The cue "...The secret passcode is" invites the direct continuation —
    exactly the DSV4/Mohtashami-&-Jaggi passkey protocol.
  * The prompt is tokenized with add_special_tokens=True (the tokenizer's own
    [gMASK]<sop> text prefix) and decoding stops at the GLM EOS ids
    [154820 <|endoftext|>, 154827 <|user|>, 154829 <|observation|>], so a
    turn-taking continuation halts immediately.

Filler length is targeted ITERATIVELY on the CONCATENATED prompt with the real
tokenizer — BPE boundary merges make per-sentence token counts non-additive
(summing standalone counts overshoots the true concatenated length by ~17%
under the real GLM tokenizer; see build_trial) — landing within 1% of the
requested token length at every ladder length >= 1024, and never above it.
The needle's token position lands within a fraction of a percent of the
requested depth; --lengths accepts k/K (x1024) and m/M (x1024^2) suffixes.

Every trial is stored in the provenance DB (bench/results.db) via
provenance.record_item under benchmark="passkey_L{L}_d{depth}" (per-cell
summary rows via finalize; prompts above --prompt-chars-cap are stored
head+tail+sha256 with the seed for deterministic reconstruction — the raw
model output is ALWAYS stored verbatim).

RUN (cluster up via scripts/launch_glm_32chip.sh — raylet env carries
TPU_DISABLE_DSA_INDEXER=1 DISABLE_WEIGHT_REQUANTIZATION=1
REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn to all workers):
  set -a; . ~/glm-tpu/.env; set +a
  cd ~/glm-tpu/bench
  NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \
  OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \
  DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
  ~/vllm-env/bin/python -u glm_longctx.py \
    --lengths 1024,2048,4096,8192 --depths 0.25,0.5,0.75 --trials 12 \
    --note "stage2 passkey ladder"

  # offline pipeline check (no model, no vllm import; mock tokenizer):
  python glm_longctx.py --stub --lengths 256,512 --depths 0.5 --trials 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine   # CPU-safe: the vllm import lives inside engine.build_llm
import extract as ex
import provenance as pv

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# GLM-5.2 config max_position_embeddings — the hard ceiling for --lengths.
MAX_CONTEXT = 1_048_576

# Canonical passkey-retrieval filler (Mohtashami & Jaggi 2023), split into
# SENTENCE units (~4-8 tokens each) for fine-grained depth placement.
# Repetition is standard and forces genuine retrieval — the passkey is the
# only informative token span in the haystack.
FILLER_SENTENCES = (
    "The grass is green. ",
    "The sky is blue. ",
    "The sun is yellow. ",
    "Here we go. ",
    "There and back again. ",
)
NEEDLE = "Remember this. The secret passcode is {key}. Do not forget it. "
CUE = "\n\nWhat is the secret passcode? The secret passcode is"


class _MockTok:
    """Whitespace tokenizer (~1 token per word). Used by --stub and the CPU
    tests to exercise the length-targeting / depth-placement / DB pipeline
    deterministically with no model, no network, no real tokenizer."""
    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": s.split()}


def parse_lengths(spec: str) -> list[int]:
    """Parse --lengths: comma-separated ints with optional k/K (x1024) or
    m/M (x1024^2) suffix, e.g. '1024,32k,128K,1M'. Capped at 1M (=1048576,
    GLM-5.2's max_position_embeddings)."""
    out = []
    for x in spec.split(","):
        x = x.strip()
        if not x:
            continue
        m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmM]?)", x)
        if not m:
            raise ValueError(f"bad length {x!r} (want e.g. 4096, 128K, 1M)")
        v = int(float(m.group(1)) * {"": 1, "k": 1024, "m": 1024**2}[m.group(2).lower()])
        if not 0 < v <= MAX_CONTEXT:
            raise ValueError(f"length {x!r} -> {v} out of range (1..{MAX_CONTEXT} "
                             f"= GLM-5.2 max_position_embeddings)")
        out.append(v)
    return out


def _tok_len(tok, s: str) -> int:
    return len(tok(s, add_special_tokens=False)["input_ids"])


def _passkey(rng) -> str:
    # local deterministic RNG (no global seeding); ALWAYS 6 digits
    return str(int(rng.randint(100000, 1000000)))


def build_trial(tok, target_len: int, depth: float, seed: int) -> tuple[str, str]:
    """Return (context, true_key): filler targeting target_len tokens AS
    MEASURED ON THE CONCATENATED PROMPT, with the needle placed at fraction
    `depth` of the prompt tokens, ending in the retrieval CUE.

    Length targeting is ITERATIVE on the whole assembled prompt. Per-sentence
    token counts are NOT additive under a real BPE tokenizer: each filler
    sentence's trailing space tokenizes standalone as its own token but merges
    into the next sentence's first word in context, so summing standalone
    counts overshoots the real concatenated length by ~17% on the real GLM
    tokenizer (review round3-unknown finding 1 — the old code tested "128K"
    at ~108K real tokens). Newton-style loop: assemble, tokenize the WHOLE
    prompt, correct the unit count by the measured effective tokens-per-unit
    rate — O(a few whole-prompt tokenizations), never a per-sentence
    retokenize loop. Terminates within max(one filler unit, 1% of target_len)
    BELOW-or-at target_len (never above — the auto max_len headroom must
    hold), i.e. within 1% at every ladder length >= 1024.

    Depth is the fraction of prompt tokens BEFORE the needle. Placement is at
    sentence granularity; boundary merges deflate the before-needle count and
    the total uniformly, so the achieved depth stays within a fraction of a
    percent of the request (the CPU test asserts +-2%). Deterministic in
    (tok, target_len, depth, seed)."""
    rng = np.random.RandomState(seed)
    true_key = _passkey(rng)
    needle = NEEDLE.format(key=true_key)
    n_sent = len(FILLER_SENTENCES)
    unit_toks = [_tok_len(tok, u) for u in FILLER_SENTENCES]
    overhead = _tok_len(tok, needle) + _tok_len(tok, CUE)

    def assemble(n_units: int) -> str:
        # Insert the needle where the cumulative STANDALONE filler count is
        # closest to depth * (filler total + overhead): token FRACTIONS survive
        # the uniform boundary-merge deflation even though absolute counts
        # do not (clamps naturally to [0, n_units]).
        counts = [unit_toks[j % n_sent] for j in range(n_units)]
        cum = np.concatenate([[0], np.cumsum(counts)]) if n_units else np.array([0])
        ins = int(np.argmin(np.abs(cum - depth * (cum[-1] + overhead))))
        parts = [FILLER_SENTENCES[j % n_sent] for j in range(n_units)]
        context = "".join(parts[:ins]) + needle + "".join(parts[ins:])
        return context.rstrip() + CUE

    tol = max(max(unit_toks), int(0.01 * target_len))  # 1% governs at L >= ~1K
    rate = sum(unit_toks) / n_sent   # standalone rate (merge-free upper bound)
    n_units = max(0, int((target_len - overhead) // rate))
    prompt = assemble(n_units)
    best = None                      # longest measurement <= target so far
    for _ in range(8):               # converges in 2-3 passes in practice
        n = _tok_len(tok, prompt)
        if n <= target_len and (best is None or n > best[1]):
            best = (prompt, n)
        if target_len - tol <= n <= target_len:
            break
        if n_units:
            # effective tokens-per-unit measured on the CONCATENATED prompt
            rate = max((n - overhead) / n_units, 0.25)
        # aim half a tolerance under target: a tiny rate misestimate must
        # never push the prompt ABOVE target_len
        step = int(round((target_len - tol / 2 - n) / rate))
        n_units = max(0, n_units + (step or (1 if n < target_len else -1)))
        prompt = assemble(n_units)
    else:
        prompt = best[0] if best else assemble(0)
    return prompt, true_key


def extract_passkey(reply: str | None) -> str | None:
    """First digit run of >=4 digits in the reply (after stripping any think
    block and thousands-commas). Never truncates a longer run — a 7-digit
    answer fails exact-match honestly. None when no digits (scored wrong,
    raw output stored for audit)."""
    if not reply:
        return None
    body = ex.strip_think(reply).replace(",", "")
    for m in re.finditer(r"\d+", body):
        if len(m.group(0)) >= 4:
            return m.group(0)
    return None


def make_raw_generate(llm, tok, max_len: int, max_new: int = 20):
    """RAW-COMPLETION greedy generator (see the module docstring for why raw
    completion, not the chat template). Returns
    `generate(context) -> (text, n_gen_tokens, n_prompt_tokens)`."""
    from vllm import SamplingParams

    def generate(context: str):
        ids = tok(context, add_special_tokens=True)["input_ids"]
        room = max_len - len(ids)
        if room <= 0:
            # Prompt alone exceeds the context window: record an empty reply
            # (scored wrong, auditable) rather than crash the ladder.
            print(f"[longctx] SKIP: prompt {len(ids)} tok >= max_len {max_len}",
                  flush=True)
            return "", 0, len(ids)
        sp = SamplingParams(
            temperature=0.0,               # greedy — the retrieval protocol
            max_tokens=min(max_new, room),
            stop_token_ids=engine.EOS_IDS,
            ignore_eos=False,
        )
        outs = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
        o = outs[0].outputs[0]
        return o.text, len(o.token_ids), len(ids)

    return generate


def _stub_generate(tok):
    """Offline pipeline check: empty replies (all trials score wrong)."""
    def generate(context: str):
        return "", 0, len(tok(context, add_special_tokens=True)["input_ids"])
    return generate


def _prompt_for_db(ctx: str, cap: int) -> str:
    """Verbatim prompt up to `cap` chars; beyond that, head+tail+sha256 (the
    trial is deterministically reconstructible from the stored seed)."""
    if len(ctx) <= cap:
        return ctx
    h = hashlib.sha256(ctx.encode("utf-8")).hexdigest()
    return (ctx[:2048]
            + f"\n...[prompt truncated for storage: {len(ctx)} chars total, "
              f"sha256={h}; rebuild deterministically via "
              f"build_trial(tok, L, depth, seed) with the stored seed]...\n"
            + ctx[-2048:])


def run_ladder(generate, tok, lengths, depths, trials, conn=None, run_id=None,
               base_seed: int = 12345, prompt_cap: int = 65536):
    """Run the (length x depth x trial) grid; store EVERY trial in the
    provenance DB when conn is given. Returns (grid, rows, overall_acc)."""
    grid, rows = {}, []
    overall_ok = overall_n = 0
    for L in lengths:
        for d in depths:
            bench_name = f"passkey_L{L}_d{d}"
            ok, ptoks = 0, []
            for t in range(trials):
                seed = base_seed + L * 131 + int(d * 1000) * 17 + t
                ctx, true_key = build_trial(tok, L, d, seed)
                t0 = time.time()
                reply, n_gen, n_prompt = generate(ctx)
                latency = (time.time() - t0) * 1000.0
                pred = extract_passkey(reply)
                hit = int(pred == true_key)
                ok += hit
                ptoks.append(n_prompt)
                rows.append(dict(L=L, depth=d, trial=t, seed=seed,
                                 prompt_tok=n_prompt, correct=hit,
                                 true_key=true_key, pred=pred))
                if conn is not None:
                    pv.record_item(conn, run_id, benchmark=bench_name,
                                   item_id=f"t{t}",
                                   prompt=_prompt_for_db(ctx, prompt_cap),
                                   gold=true_key, raw_output=reply,
                                   extracted=pred, correct=bool(hit),
                                   score=float(hit), n_prompt_tokens=n_prompt,
                                   n_gen_tokens=n_gen,
                                   latency_ms=round(latency, 1), seed=seed)
                print(f"[longctx] L={L} d={d} t={t}: prompt_tok={n_prompt} "
                      f"pred={pred!r} gold={true_key} correct={bool(hit)} "
                      f"{latency / 1000.0:.1f}s", flush=True)
            acc = 100.0 * ok / trials
            avg_tok = float(np.mean(ptoks))
            grid[bench_name] = dict(L=L, depth=d, trials=trials, correct=ok,
                                    accuracy=round(acc, 1),
                                    avg_prompt_tok=round(avg_tok, 0))
            if conn is not None:
                pv.finalize(conn, run_id, benchmark=bench_name, metric="acc")
            overall_ok += ok
            overall_n += trials
            print(f"[longctx] L={L:7d} depth={d:.2f}  acc={acc:5.1f}%  "
                  f"({ok}/{trials})  avg_prompt_tok={avg_tok:.0f}", flush=True)
    overall = round(100.0 * overall_ok / overall_n, 1) if overall_n else 0.0
    return grid, rows, overall


def _run_env(args, lengths, depths, max_len) -> dict:
    """Run-level env/config provenance (mirrors run_bench._run_env; secrets
    are filtered — never store tokens/keys)."""
    os_env = {k: v for k, v in sorted(os.environ.items())
              if k.startswith(("GLM_", "RUNAI_STREAMER_", "VLLM_", "TPU_", "JAX_",
                               "NEW_MODEL_DESIGN", "MODEL_IMPL_TYPE",
                               "OMP_NUM_THREADS", "DISABLE_WEIGHT_REQUANTIZATION",
                               "REQUANTIZE_WEIGHT_DTYPE"))
              and not any(s in k.upper() for s in ("TOKEN", "KEY", "SECRET"))}
    return {
        "benchmark": "longctx_passkey", "stub": bool(args.stub),
        "prompt_mode": "raw_completion",   # NOT the chat template (see docstring)
        "model": args.model,
        "tp": int(os.environ.get("GLM_TP", "32")),
        "dp_attention": False, "expert_parallel": True,
        "lengths": lengths, "depths": depths, "trials": args.trials,
        "max_new": args.max_new, "max_len": max_len,
        "max_seqs": args.max_seqs, "max_batched_tokens": args.max_batched_tokens,
        "gmu": args.gmu, "num_gpu_blocks": args.num_gpu_blocks,
        "temperature": 0.0, "eos_ids": engine.EOS_IDS,
        "base_seed": args.seed, "load_format": "runai_streamer",
        "os_env": os_env,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lengths", default="1024,2048,4096,8192",
                    help="comma-separated target context lengths in TOKENS; "
                         "k/K=x1024, m/M=x1024^2 suffixes OK (e.g. 128K,1M); "
                         f"max {MAX_CONTEXT}")
    ap.add_argument("--depths", default="0.25,0.5,0.75",
                    help="comma-separated needle depths (fraction of prompt "
                         "tokens before the needle)")
    ap.add_argument("--trials", type=int, default=12,
                    help="trials per (length,depth) cell")
    ap.add_argument("--max-new", type=int, default=20,
                    help="greedy decode budget per trial (retrieval needs ~8)")
    ap.add_argument("--max-len", type=int, default=0,
                    help="engine max_model_len; 0 = auto (max length + 256)")
    ap.add_argument("--max-seqs", type=int, default=1)
    ap.add_argument("--max-batched-tokens", type=int, default=4096,
                    help="chunked-prefill chunk size")
    ap.add_argument("--gmu", type=float,
                    default=float(os.environ.get("GLM_GMU", "0.94")))
    ap.add_argument("--num-gpu-blocks", type=int, default=0,
                    help="num_gpu_blocks_override if >0; 0 = kv auto")
    ap.add_argument("--model", default=engine.DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--seed", type=int, default=12345, help="base trial seed")
    ap.add_argument("--stub", action="store_true",
                    help="offline pipeline check: mock whitespace tokenizer + "
                         "empty replies; no vllm import, no model")
    ap.add_argument("--db", default=None,
                    help="provenance DB path (default bench/results.db)")
    ap.add_argument("--out-json", default=None,
                    help="summary JSON path (default bench/results/"
                         "longctx_passkey.json)")
    ap.add_argument("--prompt-chars-cap", type=int, default=65536,
                    help="store prompts verbatim up to this many chars; above, "
                         "head+tail+sha256 (trial reconstructible from seed)")
    args = ap.parse_args(argv)

    lengths = parse_lengths(args.lengths)
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    if not lengths or not depths:
        ap.error("empty --lengths or --depths")
    max_len = args.max_len or (max(lengths) + 256)

    conn = pv.connect(args.db) if args.db else pv.connect()
    run_id = pv.start_run(conn, model=("STUB" if args.stub else args.model),
                          revision=args.revision,
                          env=_run_env(args, lengths, depths, max_len),
                          note=args.note or ("offline-stub" if args.stub
                                             else "longctx passkey ladder"))
    if args.stub:
        tok = _MockTok()
        generate = _stub_generate(tok)
    else:
        llm = engine.build_llm(args.model, max_len=max_len,
                               max_seqs=args.max_seqs,
                               max_batched_tokens=args.max_batched_tokens,
                               gmu=args.gmu, num_gpu_blocks=args.num_gpu_blocks,
                               log_extra=f"max_new={args.max_new}")
        tok = llm.get_tokenizer()
        generate = make_raw_generate(llm, tok, max_len, args.max_new)

    print(f"[longctx] run_id={run_id} lengths={lengths} depths={depths} "
          f"trials={args.trials} max_len={max_len} max_new={args.max_new} "
          f"mode={'stub' if args.stub else 'raw_completion'}", flush=True)
    t0 = time.time()
    grid, rows, overall = run_ladder(generate, tok, lengths, depths, args.trials,
                                     conn=conn, run_id=run_id,
                                     base_seed=args.seed,
                                     prompt_cap=args.prompt_chars_cap)
    dt = time.time() - t0

    # aggregate summary row (per-cell rows already finalized in run_ladder)
    pv.finalize(conn, run_id, benchmark="longctx_passkey", metric="acc",
                value=overall,
                note=f"aggregate over {len(grid)} cells x {args.trials} trials; "
                     f"per-cell rows = passkey_L*_d*")

    os.makedirs(RESULTS, exist_ok=True)
    out_json = args.out_json or os.path.join(RESULTS, "longctx_passkey.json")
    res = dict(benchmark="longctx_passkey", run_id=run_id,
               overall_accuracy=overall, lengths=lengths, depths=depths,
               trials=args.trials, max_new=args.max_new, max_len=max_len,
               prompt_mode=("stub" if args.stub else "raw_completion"),
               grid=grid, seconds=round(dt, 1))
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.splitext(out_json)[0] + ".jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n===== LONG-CONTEXT PASSKEY SUMMARY =====  overall acc={overall}%"
          f"  ({dt:.0f}s)  run_id={run_id}", flush=True)
    # per-length marginal accuracy (the long-context curve; Stage-2 gate =
    # >=95% at every length out to >=128K)
    for L in lengths:
        cells = [v for v in grid.values() if v["L"] == L]
        tot = sum(c["correct"] for c in cells)
        n = sum(c["trials"] for c in cells)
        flag = "PASS" if 100.0 * tot / n >= 95.0 else "fail"
        print(f"  L={L:7d}  acc={100.0 * tot / n:5.1f}%  (n={n})  [{flag} "
              f"@ the >=95% Stage-2 bar]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
