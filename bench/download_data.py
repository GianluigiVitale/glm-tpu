#!/usr/bin/env python3
"""Pre-download + cache the tractable GLM-5.2 benchmark datasets, so the next
session starts with the data staged (HF datasets cache under bench/data/,
gitignored). Reports which are reachable and how many items each has.

Cheap: these eval sets are small (hundreds–thousands of rows). The EXPENSIVE
thing (the ~744 GB FP8 weights) is NOT downloaded here — that streams HF→GCS at
serve time (see CLAUDE.md COST). Needs HF_TOKEN (GPQA is gated) — export it from
~/glm-tpu/.env before running.

    export HF_TOKEN=$(grep -oP 'HF_TOKEN=\\K.*' ~/glm-tpu/.env)
    ~/vllm-env/bin/python bench/download_data.py
"""
from __future__ import annotations

import os

import benchmarks as B


def main():
    if not os.environ.get("HF_TOKEN"):
        print("⚠ HF_TOKEN not set — GPQA (gated) will fail. "
              "export HF_TOKEN from ~/glm-tpu/.env first.")
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    print(f"cache dir: {os.environ['HF_DATASETS_CACHE']}\n")
    ok, fail = [], []
    for name, spec in B.REGISTRY.items():
        try:
            items = B.load_items(spec, limit=3)   # triggers download + build
            # count full size cheaply
            from datasets import load_dataset
            ds = load_dataset(spec.hf_path, spec.hf_config, split=spec.hf_split,
                              token=os.environ.get("HF_TOKEN"),
                              cache_dir=os.environ["HF_DATASETS_CACHE"])
            n = len(ds)
            print(f"✓ {name:14s} {spec.hf_path} [{spec.hf_split}] — {n} items; "
                  f"sample gold={items[0].gold!r}")
            ok.append((name, n))
        except Exception as e:
            print(f"✗ {name:14s} {spec.hf_path} — {type(e).__name__}: "
                  f"{str(e)[:140]}")
            fail.append(name)
    print(f"\ncached: {[n for n, _ in ok]}")
    if fail:
        print(f"FAILED (fix dataset path / token): {fail}")


if __name__ == "__main__":
    main()
