#!/usr/bin/env python3
"""CPU-only replay of DB582 and its original failed MoE candidate.

No model execution or TPU initialization. This cannot promote an arithmetic
candidate; existing bounds are diagnostics against the original outputs.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys

import ml_dtypes
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.greenfield.probe_ws32_prefill_moe import PACK, ORACLE, ORACLE_SHA
from scripts.greenfield.run_real_one_layer_ws32 import _load_oracle, _bfloat16_numpy
from glm_tpu.greenfield.benchmarking import (
    compare_bounded_tensor,
    REAL_LAYER_OUTPUT_TOLERANCE,
)


def bf16(x: np.ndarray) -> np.ndarray:
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


def route_replays(parts: np.ndarray, seen: np.ndarray) -> dict:
    exact = parts.astype(np.float64).sum(axis=1).astype(np.float32)
    left = parts[:, 0].copy()
    bfleft = left.copy()
    for i in range(1, 8):
        left = left + parts[:, i]
        bfleft = bf16(bfleft + parts[:, i])
    return {
        name: int(np.count_nonzero(value.view(np.uint32) != seen.view(np.uint32)))
        for name, value in (
            ("exact_f32", exact),
            ("bf16_once", bf16(exact)),
            ("left_f32", left),
            ("bf16_left", bfleft),
        )
    }


def analyze(root: Path) -> dict:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket("driftbench-dsv4-uc")
    runner = json.loads((root / "runner.json").read_text())
    if runner["protocol"] != "ws32-prefill-real-moe-boundary-diagnostic-v1":
        raise ValueError("wrong diagnostic protocol")
    prior = json.loads(
        (
            REPO / "docs/artifacts/prefill-real-moe-arithmetic-refusal-20260907.json"
        ).read_text()
    )
    _, oracle = _load_oracle(
        ORACLE,
        expected_manifest_sha256=ORACLE_SHA,
        pack_manifest=json.loads((PACK / "manifest.json").read_text()),
    )
    legacy = _bfloat16_numpy(oracle["normal_output"])
    owners = {}
    sums = {}
    originals = []
    sources = []
    for rank in range(8):
        directory = root / "fleet" / f"rank{rank}"
        record = json.loads((directory / "runner.json").read_text())
        aggregate_records = [r for r in runner["workers"] if r["launch_rank"] == rank]
        if len(aggregate_records) != 1:
            raise ValueError("diagnostic aggregate rank inventory differs")
        bound = {
            k: v for k, v in aggregate_records[0].items() if k != "original_comparison"
        }
        if bound != record:
            raise ValueError("diagnostic worker differs from aggregate")
        npz = directory / "boundaries.npz"
        if sha256(npz.read_bytes()).hexdigest() != record["boundaries"]["npz_sha256"]:
            raise ValueError("diagnostic boundaries changed")
        with np.load(npz, allow_pickle=False) as arrays:
            for slot in record["local_device_slots"]:
                device = slot["device_id"]
                owner = slot["device_slot"]
                if owner in owners:
                    raise ValueError("duplicate diagnostic owner")
                owners[owner] = {}
                for side in ("candidate", "reference"):
                    values = {
                        name: arrays[f"{side}_{name}_{device}"]
                        for name in (
                            "weighted_routes",
                            "local_sum_operand",
                            "routed",
                            "routed_partial",
                            "routed_reduced",
                        )
                    }
                    owners[owner][side] = values
                    parts = (
                        values["weighted_routes"]
                        .view(ml_dtypes.bfloat16)
                        .astype(np.float32)
                    )
                    result = route_replays(parts, values["local_sum_operand"])
                    agg = sums.setdefault(
                        side, dict(elements=0, **{k: 0 for k in result})
                    )
                    agg["elements"] += values["local_sum_operand"].size
                    for key, value in result.items():
                        agg[key] += value
        # Exact runner generation from the committed refusal; NPZ generation
        # from its publisher ledger, and bytes bound to runner's output digests.
        receipt = next(r for r in prior["verified_runner_objects"] if r["rank"] == rank)
        prefix = f"results/{prior['tag']}/workers/rank{rank}/"
        g = int(receipt["generation"])
        data = bucket.blob(prefix + "runner.json", generation=g).download_as_bytes(
            if_generation_match=g
        )
        if sha256(data).hexdigest() != receipt["sha256"]:
            raise ValueError("old runner digest differs")
        old = json.loads(data)
        ledger = bucket.get_blob(prefix + "worker_receipts.json")
        entries = json.loads(
            ledger.download_as_bytes(if_generation_match=ledger.generation)
        )
        payload = next(r for r in entries if r["name"] == prefix + "normal.npz")
        pg = int(payload["generation"])
        raw = bucket.blob(payload["name"], generation=pg).download_as_bytes(
            if_generation_match=pg
        )
        if sha256(raw).hexdigest() != payload["original_sha256"]:
            raise ValueError("old NPZ digest differs")
        sources.append(dict(rank=rank, runner=receipt, npz=payload))
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            for s in old["cases"]["normal"]["shards"]:
                a = arrays[f"actual_{s['device_id']}"]
                b = arrays[f"reference_{s['device_id']}"]
                if (
                    sha256(a.tobytes()).hexdigest() != s["output_sha256"]
                    or sha256(b.tobytes()).hexdigest() != s["reference_sha256"]
                ):
                    raise ValueError("original final tensor binding differs")
                a, b = a.view(ml_dtypes.bfloat16), b.view(ml_dtypes.bfloat16)
                f = s["device_slot"] % 4
                originals.append(
                    dict(
                        device_slot=s["device_slot"],
                        candidate_vs_m1=compare_bounded_tensor(
                            a, b, REAL_LAYER_OUTPUT_TOLERANCE
                        ),
                        candidate_row0_vs_legacy=compare_bounded_tensor(
                            a[:1],
                            legacy[:, f * 1536 : (f + 1) * 1536],
                            REAL_LAYER_OUTPUT_TOLERANCE,
                        ),
                    )
                )
    if set(owners) != set(range(32)):
        raise ValueError("diagnostic owner coverage differs")
    reductions = {}
    for side in ("candidate", "reference"):
        expert_mismatches = feature_mismatches = 0
        feature_count = 0
        for f in range(4):
            if any(
                not np.array_equal(
                    owners[e * 4 + f][side]["routed"], owners[f][side]["routed"]
                )
                for e in range(8)
            ):
                raise ValueError("expert-reduced outputs differ across replicas")
            reduced = bf16(
                sum(
                    owners[e * 4 + f][side]["local_sum_operand"].astype(np.float64)
                    for e in range(8)
                )
            )
            seen = owners[f][side]["routed"].view(ml_dtypes.bfloat16).astype(np.float32)
            expert_mismatches += int(np.count_nonzero(reduced != seen))
        for e in range(8):
            if any(
                not np.array_equal(
                    owners[e * 4 + f][side]["routed_reduced"],
                    owners[e * 4][side]["routed_reduced"],
                )
                for f in range(4)
            ):
                raise ValueError("feature-reduced outputs differ across replicas")
            reduced = bf16(
                sum(
                    owners[e * 4 + f][side]["routed_partial"].astype(np.float64)
                    for f in range(4)
                )
            )
            seen = (
                owners[e * 4][side]["routed_reduced"]
                .view(ml_dtypes.bfloat16)
                .astype(np.float32)
            )
            feature_mismatches += int(np.count_nonzero(reduced != seen))
            feature_count += seen.size
        reductions[side] = dict(
            replica_equality_verified=True,
            expert_math_replay_mismatches=expert_mismatches,
            feature_math_replay_mismatches=feature_mismatches,
            feature_elements=feature_count,
        )
    return dict(
        classification="CPU_BOUNDARY_REPLAY_NOT_ARITHMETIC_PROMOTION",
        diagnostic_tag=root.name,
        original_failed_tag=prior["tag"],
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        diagnostic_runner_sha256=sha256(
            (root / "runner.json").read_bytes()
        ).hexdigest(),
        route_sum_replay=sums,
        collective_replays=reductions,
        original_output_bounds=originals,
        original_source_receipts=sources,
        limitation="Reference capture perturbed16/32 final witnesses; diagnostic first divergence is not original first divergence",
        numerical_promotion=False,
        performance_claim=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.run_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("original_output_bounds", "original_source_receipts")
            },
            indent=2,
        )
    )
