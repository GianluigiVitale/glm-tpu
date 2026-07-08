#!/usr/bin/env python3
"""DSA-sparse vs dense-MLA DECODE-THROUGHPUT A/B at long context — the >=256K
FLOP/throughput gate (PLAN Stage-3 threshold "measurable FLOP/throughput gain
at >=256K"; the adopted audit's second results.db instrument, next to the
passkey ladder glm_longctx.py).

METHODOLOGY — the DSV4 prefill-subtraction bench, adapted to GLM
(~/moe-tpu/bench/dsv4_decode_throughput.py; provenance discipline from
run_bench.py/glm_longctx.py). For each context length L we build --num-seqs
deterministic synthetic prompts of EXACTLY L tokens (seeded uniform draws from
the real tokenizer vocab id range — content is irrelevant for throughput;
determinism is what matters: the seed + generator params + sha256 are stored so
every prompt is reproducible WITHOUT storing 256K tokens of text) and time TWO
greedy batch generations from the SAME prompts, engine-side wall clock around
`llm.generate`:

    pass A: max_tokens=1                 (prefill wall + 1 decode step)
    pass B: max_tokens=--measure-tokens  (prefill + N decode steps)

Both passes pay the SAME prefill and the same one-time XLA compile (pass A
compiles both the prefill and the decode program), so

    decode_tok_s = (tokens_B - tokens_A) / (T_B - T_A)

is the steady-state AGGREGATE decode throughput at context ~L; per-seq =
aggregate / num_seqs. T_A is reported separately as the prefill wall (it
includes one decode step + any first-run compile — recorded honestly, never
subtracted out of itself). `ignore_eos` + greedy make the token counts exact
and the passes deterministic. `--reps` >1 takes the MIN wall per pass (the
DSV4 noise floor); default 1 — a 256K prefill is expensive, don't double it
casually.

THE A/B KNOB IS THE ENV AT LAUNCH, not a flag of this driver. `GLM_DSA_MODE`
shapes worker-side tracing AND the KV-cache spec (HANDOFF landmine: cross-host
divergence = inconsistent cache topology), so it must be RAYLET-BAKED
(`EXTRA_ENVS="GLM_DSA_MODE=pallas_decode ..."` at launch_glm_32chip.sh time)
AND exported on the driver; this harness only READS it into provenance
(env_json.attention_path — the exact run_bench.py derivation). The Δ table
comes from TWO runs of this script under different launches, joined by ctx:

    python3 bench/report_throughput.py            # dense-mla vs dsa-sparse

KV MATH GUARD (docs/05 ladder + docs/11 §5 block math; the docs/16 lesson —
a config whose KV pool cannot hold ctx x num_seqs preempts/thrashes SILENTLY,
or spins on "Should not schedule a request that does nothing"): before the
engine is built this driver prints required-vs-pool blocks per rung and
REFUSES configs that cannot fit (256K needs dcp>=8 at the 128-block pool);
after the build it re-checks against the engine's ACTUAL num_gpu_blocks and
refuses again rather than submit a doomed 256K prefill.

POD A/B (the 256K gate; DCP=8 per docs/05 — the 262144 rung does not fit any
smaller dcp; bucket 32 per round-7 F1 — the sparse branch's work scales with
the padded token bucket):

  # ---- run A: dense-MLA baseline (GLM_DSA_MODE unset) ----
  EXTRA_ENVS="GLM_MLA_DCP=1" GLM_FLIGHT_RECORDER=1 TPU_MIN_TOKEN_BUCKET=32 \\
    bash ~/glm-tpu/scripts/launch_glm_32chip.sh
  cd ~/glm-tpu/bench && set -a && . ~/glm-tpu/.env && set +a
  NEW_MODEL_DESIGN=1 MODEL_IMPL_TYPE=vllm TPU_MULTIHOST_BACKEND=ray \\
  OMP_NUM_THREADS=1 HF_HUB_DISABLE_XET=1 TPU_DISABLE_DSA_INDEXER=1 \\
  DISABLE_WEIGHT_REQUANTIZATION=1 REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \\
  TPU_MIN_TOKEN_BUCKET=32 GLM_TP=32 GLM_ASYNC_SCHED=0 GLM_LOG_STATS=1 \\
  GLM_FLIGHT_RECORDER=1 GLM_MLA_DCP=1 GLM_DCP=8 \\
  nohup ~/vllm-env/bin/python -u dsa_throughput.py \\
    --ctxs 8192,32768,131072,262144 --num-seqs 1 --measure-tokens 128 \\
    --max-batched-tokens 512 --gmu 0.90 \\
    --note "256K gate A: dense-mla dcp8" > ~/glm-run/thrpt_dense.log 2>&1 &

  # ---- run B: DSA-sparse (relaunch; GLM_DSA_MODE raylet-baked + driver) ----
  EXTRA_ENVS="GLM_MLA_DCP=1 GLM_DSA_MODE=pallas_decode" GLM_FLIGHT_RECORDER=1 \\
    TPU_MIN_TOKEN_BUCKET=32 bash ~/glm-tpu/scripts/launch_glm_32chip.sh
  # ...same driver env line + GLM_DSA_MODE=pallas_decode, then:
  #   dsa_throughput.py <same args> --note "256K gate B: dsa-sparse dcp8"

  # offline pipeline check (no vllm import, fake clock, real DB rows):
  python dsa_throughput.py --stub --ctxs 256,512 --num-seqs 2 --measure-tokens 8
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
import engine        # CPU-safe: the vllm import lives inside engine.build_llm
import glm_longctx as LC   # parse_lengths, PROMPT_PREFIX_IDS, MAX_CONTEXT
import provenance as pv

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# ---------------------------------------------------------------------------
# KV math (docs/05 §1.1 + docs/11 §5) — the numbers behind the guard.
# ---------------------------------------------------------------------------
PAGE_TOKENS = 512            # mla.v2 page size (docs/05 §1.1); with DCP the
                             # scheduler's LOGICAL block is 512 x dcp tokens
                             # (#2398 `block_size *= dcp`, docs/05 §4.3)
KV_BYTES_PER_TOKEN = 99_840  # padded mla.v2 layout, bf16: ~97.5 KiB/token/chip
                             # at dcp=1 (docs/05 §1.1); /dcp per chip under DCP
POOL_BLOCKS_ESTIMATE = 128   # the pod's auto-sized pool at gmu 0.90-0.94
                             # (Stage-1 smoke #16: 128 blocks x 512 = 65,536
                             # tokens ~ 6.1 GiB/chip) — a PRE-BUILD estimate;
                             # the post-build check reads the real number
DSA_KV_SPEC_FACTOR = 0.946   # dsa-sparse registers the indexer k-cache as a
                             # second KV spec (+5.4% bytes, runbook §3) — the
                             # auto-sized MLA pool shrinks accordingly
VALID_DCP = (1, 2, 4, 8, 16, 32)   # decode_context_parallel_size must divide
                                   # TP=32 (docs/11 §5a)

# Synthetic-prompt vocab sampling range: skip the 256 raw-byte tokens at the
# bottom and the special-token band at the top (154820 = <|endoftext|>, the
# first special id — engine.EOS_IDS live there).
SAMPLE_LOW = 256
FIRST_SPECIAL_ID = 154820
STUB_VOCAB_SIZE = 154880     # == the real GLM-5.2 vocab: stub prompts (and
                             # their sha256) are IDENTICAL to pod prompts


# The A/B axis (env_json.attention_path) is engine.attention_path() — the ONE
# definition shared with run_bench._run_env and glm_longctx._run_env (audit
# 2026-07-07): dense-mla = Stage-1 (DSA bypassed); dsa-sparse:<mode> = Stage-2.

# ---------------------------------------------------------------------------
# Deterministic synthetic prompts
# ---------------------------------------------------------------------------
def sample_range(vocab_size: int) -> tuple[int, int]:
    return SAMPLE_LOW, min(int(vocab_size), FIRST_SPECIAL_ID)


def seq_seed(base_seed: int, ctx: int, i: int) -> int:
    """Per-sequence prompt seed (recorded per item; the glm_longctx pattern)."""
    return base_seed + ctx * 131 + i * 7919


def build_prompt_ids(ctx: int, seed: int, vocab_size: int) -> list[int]:
    """EXACTLY `ctx` token ids: [gMASK]<sop> (every GLM sequence starts with
    the pair; the tokenizer never adds it — glm_longctx round3 finding 2) +
    seeded uniform draws from [SAMPLE_LOW, sample_high). np.random.RandomState
    is the legacy generator with a frozen-stream guarantee — the ids are
    reproducible forever from (ctx, seed, vocab_size)."""
    prefix = LC.PROMPT_PREFIX_IDS
    if ctx <= len(prefix):
        raise ValueError(f"ctx {ctx} <= prefix length {len(prefix)}")
    low, high = sample_range(vocab_size)
    body = np.random.RandomState(seed).randint(low, high, size=ctx - len(prefix))
    return list(prefix) + [int(x) for x in body]


def prompt_sha256(ids: list[int]) -> str:
    """Canonical digest of the exact id sequence (little-endian int32)."""
    return hashlib.sha256(np.asarray(ids, dtype="<i4").tobytes()).hexdigest()


def prompt_descriptor(ids: list[int], ctx: int, seed: int, vocab_size: int) -> str:
    """The `items.prompt` field: NOT the verbatim text (256K tokens of random
    ids is noise), but everything needed to rebuild it bit-exactly + the
    sha256 to prove a rebuild matches."""
    low, high = sample_range(vocab_size)
    return json.dumps({
        "generator": "dsa_throughput.build_prompt_ids/v1",
        "seed": seed, "ctx": ctx, "n_tokens": len(ids),
        "sample_low": low, "sample_high": high, "vocab_size": int(vocab_size),
        "prefix_ids": list(LC.PROMPT_PREFIX_IDS),
        "sha256": prompt_sha256(ids),
        "rebuild": "build_prompt_ids(ctx, seed, vocab_size)",
    }, sort_keys=True)


# ---------------------------------------------------------------------------
# tok/s math (pure; unit-tested)
# ---------------------------------------------------------------------------
def throughput_from_walls(t_a: float, t_b: float, tokens_a: int, tokens_b: int,
                          num_seqs: int) -> dict:
    """DSV4 prefill-subtraction: t_a = wall(max_tokens=1 batch), t_b =
    wall(max_tokens=N batch), tokens_* = TOTAL generated tokens per pass.
    The shared prefill + compile cancel in (t_b - t_a); the delta is
    (tokens_b - tokens_a) pure decode steps."""
    steps_total = tokens_b - tokens_a
    dt = t_b - t_a
    ok = dt > 0 and steps_total > 0 and num_seqs > 0
    agg = steps_total / dt if ok else 0.0
    steps_per_seq = steps_total / num_seqs if num_seqs else 0.0
    return {
        "t_prefill_s": round(t_a, 4),      # incl. 1 decode step + any compile
        "t_full_s": round(t_b, 4),
        "decode_tok_s": round(agg, 3),                    # aggregate
        "per_seq_tok_s": round(agg / num_seqs, 3) if ok else 0.0,
        "ms_per_step": (round(1000.0 * dt / steps_per_seq, 2)
                        if ok and steps_per_seq else None),
        "e2e_tok_s": round(tokens_b / t_b, 3) if t_b > 0 else 0.0,
        "nonpositive_dt": not ok,
    }


# ---------------------------------------------------------------------------
# KV math guard (docs/05 ladder + docs/11 §5; refuse what cannot fit)
# ---------------------------------------------------------------------------
def kv_plan(ctxs: list[int], num_seqs: int, measure_tokens: int, dcp: int,
            pool_blocks: int) -> dict:
    """Pure block math: each sequence at rung L holds L + measure_tokens KV
    tokens; the scheduler allocates LOGICAL blocks of 512 x dcp tokens
    (#2398), so blocks/seq = ceil((L+N)/(512*dcp)) and the rung needs
    num_seqs x that against a `pool_blocks` pool. min_dcp = the smallest
    valid dcp at which the rung fits THIS pool (docs/05: 256K -> dcp>=8)."""
    if dcp not in VALID_DCP:
        raise ValueError(f"dcp={dcp} must be one of {VALID_DCP} (divide TP=32)")
    if pool_blocks <= 0:
        raise ValueError(f"pool_blocks={pool_blocks} must be > 0")
    logical = PAGE_TOKENS * dcp
    pool_tokens = pool_blocks * logical
    rungs = []
    for L in sorted(ctxs):
        need = L + measure_tokens
        blocks_per_seq = -(-need // logical)          # ceil
        required = num_seqs * blocks_per_seq
        min_dcp = next(
            (d for d in VALID_DCP
             if num_seqs * (-(-need // (PAGE_TOKENS * d))) <= pool_blocks),
            None)
        rungs.append({
            "ctx": L, "need_tokens_per_seq": need,
            "blocks_per_seq": blocks_per_seq, "required_blocks": required,
            "required_tokens": num_seqs * need,
            "required_gib_per_chip": round(
                num_seqs * need * KV_BYTES_PER_TOKEN / dcp / 2**30, 2),
            "fits": required <= pool_blocks,
            "min_dcp": min_dcp,
        })
    return {
        "dcp": dcp, "num_seqs": num_seqs, "measure_tokens": measure_tokens,
        "logical_block_tokens": logical,
        "pool_blocks": pool_blocks, "pool_tokens": pool_tokens,
        "pool_gib_per_chip": round(
            pool_tokens * KV_BYTES_PER_TOKEN / dcp / 2**30, 2),
        "rungs": rungs, "fits": all(r["fits"] for r in rungs),
    }


def print_kv_plan(plan: dict, label: str) -> None:
    print(f"[thrpt] KV guard ({label}): dcp={plan['dcp']} "
          f"logical_block={plan['logical_block_tokens']} tok, pool="
          f"{plan['pool_blocks']} blocks = {plan['pool_tokens']} tok "
          f"(~{plan['pool_gib_per_chip']} GiB/chip)", flush=True)
    for r in plan["rungs"]:
        verdict = "OK" if r["fits"] else (
            f"REFUSE (needs dcp>={r['min_dcp']})" if r["min_dcp"]
            else "REFUSE (no valid dcp fits this pool)")
        print(f"[thrpt]   ctx={r['ctx']:>7d}: {plan['num_seqs']} seq x "
              f"{r['blocks_per_seq']} blocks = {r['required_blocks']} blocks "
              f"({r['required_tokens']} tok, ~{r['required_gib_per_chip']} "
              f"GiB/chip) vs pool {plan['pool_blocks']} -> {verdict}",
              flush=True)


def _actual_pool_blocks(llm) -> int | None:
    """Best-effort read of the engine's REAL post-profiling KV pool size (in
    scheduler blocks). None when the attribute path is not found — the guard
    then stands on the pre-build estimate (already printed)."""
    for get in (
        lambda: llm.llm_engine.vllm_config.cache_config.num_gpu_blocks,
        lambda: llm.llm_engine.cache_config.num_gpu_blocks,
    ):
        try:
            v = get()
            if v:
                return int(v)
        except Exception:
            pass
    return None


def _launcher_baked() -> str | None:
    """Verbatim launcher ENVS string (what every raylet exports) — recorded so
    an A/B run's worker-side env is auditable. EXTRA_ENVS is a launch-shell
    passthrough and is NOT reconstructable here (run_bench has the same
    limitation) — hence the loud GLM_DSA_MODE reminder in main()."""
    try:
        path = os.path.join(HERE, "..", "scripts", "launch_glm_32chip.sh")
        with open(path, encoding="utf-8") as f:
            m = re.search(r'^ENVS="export (.*)"$', f.read(), re.MULTILINE)
        return m.group(1) if m else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Generators (real engine + stub)
# ---------------------------------------------------------------------------
def make_batch_generate(llm):
    """gen(prompts_ids, n_tokens) -> ([(token_ids, text), ...], wall_s):
    ONE llm.generate call for the whole batch (vLLM schedules up to max_seqs
    concurrently), greedy, ignore_eos (exact token counts), engine-side wall."""
    from vllm import SamplingParams

    def gen(prompts_ids, n_tokens):
        sp = SamplingParams(temperature=0.0, max_tokens=n_tokens,
                            ignore_eos=True)
        t0 = time.time()
        outs = llm.generate([{"prompt_token_ids": list(ids)}
                             for ids in prompts_ids], sp, use_tqdm=False)
        wall = time.time() - t0
        return [(list(o.outputs[0].token_ids), o.outputs[0].text)
                for o in outs], wall

    return gen


def _stub_batch_generate():
    """Offline pipeline check: deterministic fake outputs (a function of the
    PROMPT only — identical across attention paths, so report_throughput's
    output-identity column is exercised) + a FAKE monotone clock
    (wall = n_tokens * 1e-4 s) so the tok/s math produces stable nonzero
    numbers. Fake by construction; the run is labeled model=STUB."""
    def gen(prompts_ids, n_tokens):
        res = []
        for ids in prompts_ids:
            s = ids[len(LC.PROMPT_PREFIX_IDS)] if len(ids) > 2 else 0
            toks = [(s * 31 + j) % 1000 for j in range(n_tokens)]
            res.append((toks, f"stub:{len(toks)}tok"))
        return res, n_tokens * 1e-4
    return gen


# ---------------------------------------------------------------------------
# One ladder rung
# ---------------------------------------------------------------------------
def run_rung(gen, ctx: int, num_seqs: int, measure_tokens: int,
             base_seed: int, vocab_size: int, reps: int = 1,
             warmup: int = 0) -> tuple[list, list, list, dict]:
    """Build the rung's prompts, run pass A (max_tokens=1) + pass B
    (max_tokens=N) `reps` times (min wall each — the DSV4 noise floor),
    return (prompts, seeds, pass-B outputs, stats)."""
    seeds = [seq_seed(base_seed, ctx, i) for i in range(num_seqs)]
    prompts = [build_prompt_ids(ctx, s, vocab_size) for s in seeds]
    for _ in range(warmup):
        gen(prompts, measure_tokens)
    t_as, t_bs, outs_a, outs_b = [], [], None, None
    for _ in range(reps):
        oa, ta = gen(prompts, 1)
        ob, tb = gen(prompts, measure_tokens)
        t_as.append(ta)
        t_bs.append(tb)
        if outs_b is None:          # greedy => identical across reps
            outs_a, outs_b = oa, ob
    tokens_a = sum(len(o[0]) for o in outs_a)
    tokens_b = sum(len(o[0]) for o in outs_b)
    stats = throughput_from_walls(min(t_as), min(t_bs), tokens_a, tokens_b,
                                  num_seqs)
    stats.update({
        "ctx": ctx, "num_seqs": num_seqs, "measure_tokens": measure_tokens,
        "reps": reps, "warmup": warmup,
        "tokens_pass_a": tokens_a, "tokens_pass_b": tokens_b,
        "t_prefill_all": [round(x, 4) for x in t_as],
        "t_full_all": [round(x, 4) for x in t_bs],
        # ignore_eos should make counts exact; a mismatch is flagged, never
        # hidden (the math above already uses the MEASURED counts)
        "count_mismatch": (tokens_a != num_seqs
                           or tokens_b != num_seqs * measure_tokens),
    })
    return prompts, seeds, outs_b, stats


# ---------------------------------------------------------------------------
# Run-level provenance
# ---------------------------------------------------------------------------
def _run_env(args, ctxs, max_len, dcp, plan, vocab_size) -> dict:
    """env_json for the runs row (mirrors glm_longctx._run_env + the audit's
    attention_path; secrets filtered — never store tokens/keys)."""
    os_env = {k: v for k, v in sorted(os.environ.items())
              if k.startswith(("GLM_", "RUNAI_STREAMER_", "VLLM_", "TPU_",
                               "JAX_", "NEW_MODEL_DESIGN", "MODEL_IMPL_TYPE",
                               "OMP_NUM_THREADS",
                               "DISABLE_WEIGHT_REQUANTIZATION",
                               "REQUANTIZE_WEIGHT_DTYPE"))
              and not any(s in k.upper() for s in ("TOKEN", "KEY", "SECRET"))}
    low, high = sample_range(vocab_size)
    return {
        "benchmark": "dsa_throughput", "stub": bool(args.stub),
        # THE A/B AXIS (audit 2026-07-07): derived from the live env, the
        # same derivation as run_bench.py — never inferred after the fact.
        "attention_path": engine.attention_path(),
        "glm_dsa_mode": os.environ.get("GLM_DSA_MODE", "off"),
        "dcp": dcp, "glm_mla_dcp": os.environ.get("GLM_MLA_DCP"),
        "model": args.model,
        "tp": int(os.environ.get("GLM_TP", "32")),
        "dp_attention": False, "expert_parallel": True,
        "ctxs": ctxs, "num_seqs": args.num_seqs,
        "measure_tokens": args.measure_tokens,
        "reps": args.reps, "warmup": args.warmup,
        "max_len": max_len, "max_batched_tokens": args.max_batched_tokens,
        "gmu": args.gmu, "num_gpu_blocks": args.num_gpu_blocks,
        "kv_plan": plan,
        "temperature": 0.0, "ignore_eos": True,
        "base_seed": args.seed,
        "prompt_generator": "dsa_throughput.build_prompt_ids/v1",
        "prompt_prefix_ids": list(LC.PROMPT_PREFIX_IDS),
        "sample_low": low, "sample_high": high, "vocab_size": int(vocab_size),
        "load_format": "runai_streamer",
        "launcher_envs_baked": _launcher_baked(),
        "os_env": os_env,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ctxs", default="8192,32768,131072,262144",
                    help="comma-separated context ladder in TOKENS (k/K=x1024,"
                         " m/M=x1024^2 suffixes OK — glm_longctx syntax)")
    ap.add_argument("--num-seqs", type=int, default=1,
                    help="concurrent sequences per rung (engine max_num_seqs; "
                         "the KV guard refuses what the pool cannot hold)")
    ap.add_argument("--measure-tokens", type=int, default=128,
                    help="decode tokens per sequence in the timed pass B "
                         "(pass A is max_tokens=1; decode tok/s comes from "
                         "the B-A delta)")
    ap.add_argument("--reps", type=int, default=1,
                    help="timing repetitions per rung (min taken); default 1 "
                         "— every extra rep re-pays the full prefill")
    ap.add_argument("--warmup", type=int, default=0,
                    help="extra UNTIMED full pass-B runs before timing each "
                         "rung (pass A already compiles prefill+decode; use "
                         "for small-ctx precision only — a 256K warmup costs "
                         "a full prefill)")
    ap.add_argument("--max-len", type=int, default=0,
                    help="engine max_model_len; 0 = auto "
                         "(max ctx + measure_tokens + 16)")
    ap.add_argument("--max-batched-tokens", type=int, default=4096,
                    help="chunked-prefill chunk size (512 for pallas_decode "
                         "runs per runbook §3/round-7 F1)")
    ap.add_argument("--gmu", type=float,
                    default=float(os.environ.get("GLM_GMU", "0.94")))
    ap.add_argument("--num-gpu-blocks", type=int, default=0,
                    help="num_gpu_blocks_override (LOGICAL 512*dcp-token "
                         "blocks) if >0; 0 = kv auto")
    ap.add_argument("--pool-blocks", type=int, default=0,
                    help="pre-build pool-size ESTIMATE for the KV guard when "
                         "--num-gpu-blocks is 0; default 128 (the pod's "
                         "auto-size at gmu 0.90-0.94), x0.946 under "
                         "dsa-sparse (indexer k-cache spec, +5.4%% bytes)")
    ap.add_argument("--model", default=engine.DEFAULT_MODEL)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--seed", type=int, default=12345, help="base prompt seed")
    ap.add_argument("--stub", action="store_true",
                    help="offline pipeline check: no vllm import, no model; "
                         "deterministic fake outputs + a fake clock")
    ap.add_argument("--db", default=None,
                    help="provenance DB path (default bench/results.db)")
    ap.add_argument("--out-json", default=None,
                    help="summary JSON path (default bench/results/"
                         "dsa_throughput_<path>_dcp<d>.json)")
    args = ap.parse_args(argv)

    ctxs = sorted(LC.parse_lengths(args.ctxs))
    if not ctxs:
        ap.error("empty --ctxs")
    if args.measure_tokens < 2:
        ap.error("--measure-tokens must be >= 2 (pass B must out-decode "
                 "pass A's single token)")
    if args.num_seqs < 1:
        ap.error("--num-seqs must be >= 1")

    max_len = args.max_len or (max(ctxs) + args.measure_tokens + 16)
    if max_len > LC.MAX_CONTEXT:
        ap.error(f"max_len {max_len} > max_position_embeddings "
                 f"{LC.MAX_CONTEXT}")
    if max_len < max(ctxs) + args.measure_tokens:
        ap.error(f"--max-len {max_len} < max ctx {max(ctxs)} + "
                 f"measure_tokens {args.measure_tokens}")

    attention_path = engine.attention_path()
    dcp = int(os.environ.get("GLM_DCP") or 1)

    # ---- KV guard, pre-build (docs/05 ladder; refuse, don't thrash) ----
    if args.num_gpu_blocks > 0:
        pool_est, pool_label = args.num_gpu_blocks, "explicit --num-gpu-blocks"
    elif args.pool_blocks > 0:
        pool_est, pool_label = args.pool_blocks, "explicit --pool-blocks"
    else:
        pool_est = (int(POOL_BLOCKS_ESTIMATE * DSA_KV_SPEC_FACTOR)
                    if attention_path != "dense-mla" else POOL_BLOCKS_ESTIMATE)
        pool_label = "auto-size estimate"
    plan = kv_plan(ctxs, args.num_seqs, args.measure_tokens, dcp, pool_est)
    print_kv_plan(plan, f"pre-build, {pool_label}")
    if not plan["fits"]:
        worst = max((r for r in plan["rungs"] if not r["fits"]),
                    key=lambda r: r["ctx"])
        print(f"[thrpt] REFUSED: ctx={worst['ctx']} x {args.num_seqs} seqs "
              f"needs {worst['required_blocks']} blocks > pool "
              f"{plan['pool_blocks']} at dcp={dcp} — this config would "
              f"preempt/thrash SILENTLY (docs/16). "
              + (f"Relaunch with GLM_DCP={worst['min_dcp']} "
                 f"(GLM_MLA_DCP=1 raylet-baked), " if worst["min_dcp"] else
                 "No valid dcp fits this pool — shrink --ctxs/--num-seqs, ")
              + "or override the pool estimate via --pool-blocks if the real "
                "pool is larger.", flush=True)
        return 2

    print(f"[thrpt] attention_path={attention_path} dcp={dcp} ctxs={ctxs} "
          f"num_seqs={args.num_seqs} measure_tokens={args.measure_tokens} "
          f"reps={args.reps} max_len={max_len}", flush=True)
    if attention_path != "dense-mla":
        print("[thrpt] REMINDER: GLM_DSA_MODE shapes the WORKER-side KV spec "
              "— it must be RAYLET-BAKED via EXTRA_ENVS at launch, not just "
              "exported on this driver (HANDOFF landmine).", flush=True)

    # ---- engine (built ONCE; the A/B knob is the env at launch) ----
    if args.stub:
        vocab_size = STUB_VOCAB_SIZE
        gen = _stub_batch_generate()
    else:
        llm = engine.build_llm(args.model, max_len=max_len,
                               max_seqs=args.num_seqs,
                               max_batched_tokens=args.max_batched_tokens,
                               gmu=args.gmu,
                               num_gpu_blocks=args.num_gpu_blocks,
                               log_extra=f"attention_path={attention_path}, "
                                         f"dcp={dcp}")
        tok = llm.get_tokenizer()
        vocab_size = int(getattr(tok, "vocab_size", 0) or 0) or len(tok)
        gen = make_batch_generate(llm)
        # ---- KV guard, post-build (the engine's REAL pool governs) ----
        actual = _actual_pool_blocks(llm)
        if actual is not None:
            plan = kv_plan(ctxs, args.num_seqs, args.measure_tokens, dcp,
                           actual)
            print_kv_plan(plan, "post-build, actual num_gpu_blocks")
            if not plan["fits"]:
                print("[thrpt] REFUSED post-build: the engine's actual KV "
                      "pool cannot hold the ladder — not submitting a doomed "
                      "prefill (docs/16).", flush=True)
                return 3
        else:
            print("[thrpt] WARNING: could not read the engine's actual "
                  "num_gpu_blocks — proceeding on the pre-build estimate",
                  flush=True)

    conn = pv.connect(args.db) if args.db else pv.connect()
    run_id = pv.start_run(
        conn, model=("STUB" if args.stub else args.model),
        revision=args.revision,
        env=_run_env(args, ctxs, max_len, dcp, plan, vocab_size),
        note=args.note or ("offline-stub" if args.stub
                           else f"dsa_throughput {attention_path} dcp{dcp}"))
    print(f"[thrpt] run_id={run_id}", flush=True)

    rung_stats = []
    t0 = time.time()
    for ctx in ctxs:
        prompts, seeds, outs, stats = run_rung(
            gen, ctx, args.num_seqs, args.measure_tokens, args.seed,
            vocab_size, reps=args.reps, warmup=args.warmup)
        stats["attention_path"] = attention_path
        stats["dcp"] = dcp
        bench_name = f"dsa_throughput_ctx{ctx}"
        for i, (ids, seed, (out_ids, out_text)) in enumerate(
                zip(prompts, seeds, outs)):
            pv.record_item(
                conn, run_id, benchmark=bench_name, item_id=f"seq{i}",
                prompt=prompt_descriptor(ids, ctx, seed, vocab_size),
                gold=None,
                # the OUTPUT tokens are stored VERBATIM (the A/B token-
                # identity evidence); per-item latency is NULL in batched
                # runs (run_bench discipline: never faked) — the batch
                # walls live in the rung's summary note.
                raw_output=json.dumps({"token_ids": out_ids,
                                       "text": out_text}),
                extracted=None, correct=None, score=None,
                n_prompt_tokens=len(ids), n_gen_tokens=len(out_ids),
                latency_ms=None, seed=seed)
        pv.finalize(conn, run_id, benchmark=bench_name,
                    metric="decode_tok_s", value=stats["decode_tok_s"],
                    note=json.dumps(stats, sort_keys=True))
        rung_stats.append(stats)
        print(f"[thrpt] ctx={ctx:>7d}  prefill={stats['t_prefill_s']:9.2f}s  "
              f"decode={stats['decode_tok_s']:8.2f} tok/s agg "
              f"({stats['per_seq_tok_s']:.2f}/seq, "
              f"{stats['ms_per_step'] if stats['ms_per_step'] is not None else '?'} ms/step)  "
              f"e2e={stats['e2e_tok_s']:.2f} tok/s"
              + ("  [COUNT MISMATCH]" if stats["count_mismatch"] else ""),
              flush=True)
    dt = time.time() - t0

    # aggregate run row (per-rung rows are the joinable A/B instrument)
    pv.finalize(conn, run_id, benchmark="dsa_throughput",
                metric="decode_tok_s", value=None,
                note=json.dumps({
                    "attention_path": attention_path, "dcp": dcp,
                    "rungs": {str(s["ctx"]): s["decode_tok_s"]
                              for s in rung_stats},
                    "seconds": round(dt, 1)}, sort_keys=True))

    os.makedirs(RESULTS, exist_ok=True)
    tag = attention_path.replace(":", "-").replace("/", "-")
    out_json = args.out_json or os.path.join(
        RESULTS, f"dsa_throughput_{tag}_dcp{dcp}.json")
    with open(out_json, "w") as f:
        json.dump({"benchmark": "dsa_throughput", "run_id": run_id,
                   "attention_path": attention_path, "dcp": dcp,
                   "model": ("STUB" if args.stub else args.model),
                   "ctxs": ctxs, "num_seqs": args.num_seqs,
                   "measure_tokens": args.measure_tokens,
                   "base_seed": args.seed, "kv_plan": plan,
                   "rungs": rung_stats, "seconds": round(dt, 1)}, f, indent=2)

    print(f"\n===== DSA THROUGHPUT SUMMARY ({attention_path}, dcp={dcp}) "
          f"=====  run_id={run_id}  ({dt:.0f}s)", flush=True)
    for s in rung_stats:
        print(f"  ctx={s['ctx']:>7d}: {s['decode_tok_s']:8.2f} tok/s agg  "
              f"({s['per_seq_tok_s']:.2f}/seq)  prefill {s['t_prefill_s']:.2f}s",
              flush=True)
    print(f"[thrpt] wrote {out_json}\n[thrpt] A/B table (needs a run on the "
          f"other attention_path): python3 report_throughput.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
