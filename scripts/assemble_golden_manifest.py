#!/usr/bin/env python
"""Assemble golden.rankN.json manifests by majority-of-3 across the three
bootstrap engines, with ground-truth adjudication of disputed leaves.

Ground-truth provenance (ground_truth_sum.py run 2026-07-23 against
gs://driftbench-dsv4-uc/models/GLM-5.2-FP8):
  vllm_model.model.layers.10.self_attn.indexer.wk_weights_proj.weight
      TRUE u32 sum = 239851472
      = concat(dequant_bf16(indexer.wk) [sum 191463636],
               raw weights_proj        [sum 48387836])
      corrupt value 48387836 == the weights_proj piece alone -> streamer
      dropped/zeroed the fp8-dequant wk piece.
  vllm_model.model.layers.10.self_attn.indexer.glm_dsa_adapted_wk
      derived leaf (tool refuses). TRUE value 191463636 inferred:
      (a) equals the checkpoint-true dequant-bf16 wk byte-sum exactly;
      (b) all 19 of 24 rank-manifests with a healthy parent wk agree on it;
      (c) corrupt instances read 0, consistent with a zeroed wk input.
"""
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = ["bootstrap_1784794793", "bootstrap_1784796240", "bootstrap_1784797630"]
OUT = os.path.join(HERE, "golden_v1")

VICTIM = "['vllm_model.model.layers.10.self_attn.indexer.wk_weights_proj.weight']"
ADAPTED = "['vllm_model.model.layers.10.self_attn.indexer.glm_dsa_adapted_wk']"

# leaf-key -> (true_sum, provenance)
GROUND_TRUTH = {
    VICTIM: (239851472, "ground_truth_sum.py checkpoint adjudication"),
    ADAPTED: (191463636, "derived-leaf inference: == checkpoint-true "
              "dequant-bf16 wk piece sum; 19/24 healthy-parent consensus"),
}
GOLDEN_COMBINED = 371110325  # unanimous across all 19 healthy rank-manifests

def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for d in DIRS:
        for f in glob.glob(os.path.join(HERE, d, "*.json")):
            rank = int(re.search(r"rank(\d+)\.json$", os.path.basename(f)).group(1))
            data[(d, rank)] = json.load(open(f))

    keys = list(data[(DIRS[0], 0)]["leaves"])
    unresolved_total = 0
    for r in range(8):
        mans = [data[(d, r)] for d in DIRS]
        golden_leaves = {}
        n_unanimous = n_majority = n_overrides = n_unresolved = 0
        flags = []
        for k in keys:
            entries = [m["leaves"][k] for m in mans]
            sums = [e["sum"] for e in entries]
            shape, dtype = entries[0]["shape"], entries[0]["dtype"]
            assert all(e["shape"] == shape and e["dtype"] == dtype
                       for e in entries), (r, k)
            counter = collections.Counter(sums)
            distinct = len(counter)
            if distinct == 1:
                chosen = sums[0]
                n_unanimous += 1
            elif distinct == 2:
                maj, cnt = counter.most_common(1)[0]
                n_majority += 1
                chosen = maj
                if k in GROUND_TRUTH:
                    truth, prov = GROUND_TRUTH[k]
                    if maj != truth:
                        chosen = truth
                        n_overrides += 1
                        flags.append(
                            f"OVERRIDE rank{r} {k}: majority {maj} "
                            f"(2-of-3) CONTRADICTS ground truth {truth} "
                            f"-> ground truth wins [{prov}]")
                else:
                    # 2v1 on a leaf with no ground truth available at all:
                    # majority stands, but surface it.
                    flags.append(
                        f"MAJORITY-ONLY rank{r} {k}: 2v1 -> {maj} "
                        f"(no adjudication available)")
            else:  # 3-way
                if k in GROUND_TRUTH:
                    chosen, prov = GROUND_TRUTH[k]
                    n_overrides += 1
                    flags.append(f"3WAY-ADJUDICATED rank{r} {k} -> {chosen}")
                else:
                    chosen = None
                    n_unresolved += 1
                    flags.append(f"UNRESOLVED rank{r} {k}: 3-way split "
                                 f"{sums}, no ground truth -> BLOCKS PROMOTION")
            if chosen is None:
                golden_leaves[k] = {"shape": shape, "dtype": dtype,
                                    "sum": None, "UNRESOLVED": True,
                                    "candidates": sums}
            else:
                golden_leaves[k] = {"shape": shape, "dtype": dtype,
                                    "sum": chosen}

        # Explicit spot-adjudication of the known frequent victim, every rank.
        v = golden_leaves[VICTIM]["sum"]
        truth = GROUND_TRUTH[VICTIM][0]
        assert v == truth, f"rank{r} victim leaf {v} != ground truth {truth}"
        assert golden_leaves[VICTIM]["sum"] != 48387836
        print(f"rank{r}: victim leaf spot-check OK "
              f"(golden sum {v} == checkpoint truth {truth})")

        golden = {
            "format": mans[0]["format"],
            "host": "golden",
            "context": "majority3+groundtruth",
            "n_leaves": len(golden_leaves),
            "combined": GOLDEN_COMBINED,
            "leaves": golden_leaves,
        }
        out = os.path.join(OUT, f"golden.rank{r}.json")
        with open(out, "w") as f:
            json.dump(golden, f)
        print(f"rank{r}: unanimous={n_unanimous} majority(2v1)={n_majority} "
              f"gt_overrides={n_overrides} unresolved={n_unresolved} "
              f"-> {out}")
        for fl in flags:
            print("   ", fl)
        unresolved_total += n_unresolved
    print(f"\nTOTAL UNRESOLVED: {unresolved_total}"
          + (" -- PROMOTION BLOCKED" if unresolved_total else " (none)"))
    sys.exit(2 if unresolved_total else 0)

if __name__ == "__main__":
    main()
