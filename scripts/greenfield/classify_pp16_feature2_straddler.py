#!/usr/bin/env python3
"""Classify the sealed PP16 feature2 layer-1 BF16 straddler offline."""

from __future__ import annotations

import argparse
from pathlib import Path

from glm_tpu.greenfield.benchmarking.pp16_feature2_straddler import (
    classify_pp16_feature2_straddler,
    serialize_straddler_classification,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--optimized-hlo", type=Path, required=True)
    parser.add_argument("--accepted-layer1", type=Path, required=True)
    parser.add_argument("--db550-boundary", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = classify_pp16_feature2_straddler(
        args.capture,
        comparison_path=args.comparison,
        summary_path=args.summary,
        optimized_hlo_path=args.optimized_hlo,
        accepted_layer1_path=args.accepted_layer1,
        db550_boundary_path=args.db550_boundary,
        runner_path=args.runner,
    )
    payload = serialize_straddler_classification(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(payload.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
