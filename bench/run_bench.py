#!/usr/bin/env python3
"""GLM-5.2 benchmark harness — loads a benchmark, queries the model, extracts +
scores every item, and records EVERYTHING to the provenance DB (bench/results.db).

Model-pluggable: pass a `generate(prompt: str) -> str` callable. The model is not
ported yet (Stage 1), so this module ships a stub for offline pipeline validation;
the next chat wires the real GLM-5.2 vLLM/torchax call (in-process engine or an
OpenAI-compatible client) into `make_generate()`.

Nothing is scored that is not stored. Every item row carries the verbatim prompt,
the verbatim model reply, the extracted answer, correct/score, timing, and the
run-level provenance (model+revision, harness+fork git, env, pod).

    # offline pipeline check (no model): records items with empty replies
    python run_bench.py --benchmark gsm8k --limit 5 --stub

    # real run (Stage 1+), once make_generate() is wired to the engine:
    python run_bench.py --benchmark gpqa_diamond --limit 198 --model zai-org/GLM-5.2-FP8
"""
from __future__ import annotations

import argparse
import time

import provenance as pv
import benchmarks as B


def stub_generate(prompt: str) -> str:
    """Placeholder model: returns nothing (all items score wrong) — used only to
    validate the load→prompt→extract→score→store pipeline offline."""
    return ""


def make_generate(model: str, **kw):
    """Wire the REAL GLM-5.2 model here (Stage 1). Options, in preference order:
      1. in-process vLLM engine (the DSV4 bench pattern — reuse ~/moe-tpu/bench
         build_llm + a greedy/thinking SamplingParams), or
      2. an OpenAI-compatible client against a `vllm serve` endpoint.
    Must return the model's verbatim text reply for a prompt. Until wired, raise
    so a real run never silently scores against a stub."""
    raise NotImplementedError(
        "Wire the GLM-5.2 engine into make_generate() (Stage 1). Until then use "
        "--stub for the offline pipeline check.")


def run_benchmark(conn, run_id, spec: B.BenchSpec, generate, limit=None, seed=0):
    items = B.load_items(spec, limit=limit)
    n_correct = 0
    for it in items:
        t0 = time.time()
        reply = generate(it.prompt)
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
                       n_gen_tokens=None, latency_ms=round(latency, 1), seed=seed)
    summ = pv.finalize(conn, run_id, benchmark=spec.name, metric="acc")
    print(f"[{spec.name}] n={summ['n']} acc={summ['value']}"
          f" card={summ['card_value']} Δ={summ['delta']}")
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=list(B.REGISTRY))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="zai-org/GLM-5.2-FP8")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--stub", action="store_true",
                    help="offline pipeline check (no model — all items score wrong)")
    args = ap.parse_args()

    conn = pv.connect()
    run_id = pv.start_run(conn, model=("STUB" if args.stub else args.model),
                          revision=args.revision, note=args.note or
                          ("offline-stub" if args.stub else ""))
    gen = stub_generate if args.stub else make_generate(args.model)
    run_benchmark(conn, run_id, B.REGISTRY[args.benchmark], gen, limit=args.limit)


if __name__ == "__main__":
    main()
