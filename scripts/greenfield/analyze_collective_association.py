#!/usr/bin/env python3
"""Replay a captured StrategyND fingerprint without initializing JAX/TPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking import (  # noqa: E402
    analyze_m32_strategy_nd_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bits", type=Path, required=True)
    parser.add_argument("--output-bits", type=Path, required=True)
    parser.add_argument("--host-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-width", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = json.loads(args.host_record.read_text())
    if record.get("mode") != "strategy_nd_fingerprint":
        raise ValueError("host record is not a StrategyND fingerprint capture")
    fingerprint = record.get("association_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("host record has no association_fingerprint object")
    input_bits = np.load(args.input_bits, allow_pickle=False)
    output_bits = np.load(args.output_bits, allow_pickle=False)
    coordinates = {
        int(device["device_id"]): tuple(device["coordinates"])
        for device in record["topology"]["devices"]
    }
    analysis = analyze_m32_strategy_nd_fingerprint(
        input_bits,
        output_bits,
        fingerprint["member_device_ids"],
        coordinates,
        block_width=args.block_width,
    )
    analysis.update(
        {
            "code_hash": record["code_hash"],
            "accepted_model_axis_device_ids": fingerprint[
                "accepted_model_axis_device_ids"
            ],
            "accepted_model_axis_recipe": fingerprint[
                "accepted_model_axis_recipe"
            ],
            "optimized_hlo_sha256": fingerprint["optimized_hlo_sha256"],
            "topology_hash": record["topology_hash"],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        "GREENFIELD_ASSOCIATION_ANALYSIS_OK "
        f"candidates_per_row={analysis['candidate_count_per_row']} "
        f"minimum_row_covered={analysis['minimum_row_union_exact_column_count']}/"
        f"{analysis['width']} "
        f"unique_row_outputs={analysis['unique_row_output_count']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
