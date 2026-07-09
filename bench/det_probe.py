#!/usr/bin/env python3
"""Same-boot determinism probe (D0 follow-up): ONE engine, the same prompt
generated twice sequentially. Discriminates within-boot nondeterminism
(runtime race / order instability) from across-boot divergence (boot-dependent
state: uninitialized pages, stale slots) — the across-boot replicate (runs
53 vs 55) already differs 0/4 while the dense path is byte-identical across
rebuilds, so whichever way this lands localizes the defect class.

Usage (pod, engine env preset like run_bench):
  python det_probe.py --n 4 --max-new 512
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=4096)
    args = ap.parse_args()

    import benchmarks as B
    import engine

    items = B.load_items(B.REGISTRY["gsm8k"], limit=args.n)
    llm = engine.build_llm(engine.DEFAULT_MODEL, max_len=args.max_len,
                           max_seqs=1, max_batched_tokens=512, gmu=0.90,
                           num_gpu_blocks=64,
                           log_extra="det_probe")
    # the engine's own tokenizer picked up the gs:// checkpoint files
    tok = llm.get_tokenizer()
    from vllm import SamplingParams, TokensPrompt
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new,
                        stop_token_ids=engine.EOS_IDS)

    def gen_all(tag):
        outs = []
        for it in items:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": it.prompt}],
                add_generation_prompt=True, tokenize=True,
                return_dict=False)
            o = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp,
                             use_tqdm=False)
            outs.append(o[0].outputs[0].text)
            print(f"[{tag}] {it.item_id}: {len(o[0].outputs[0].token_ids)} tok",
                  flush=True)
        return outs

    a = gen_all("pass1")
    b = gen_all("pass2")
    for it, x, y in zip(items, a, b):
        n = min(len(x), len(y))
        div = next((i for i in range(n) if x[i] != y[i]), n)
        ident = x == y
        print(f"[DET] {it.item_id}: identical={ident} "
              f"div_at={div if not ident else '-'} len1={len(x)} len2={len(y)}",
              flush=True)
    n_ident = sum(x == y for x, y in zip(a, b))
    print(f"[DET] SAME-BOOT identical {n_ident}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
