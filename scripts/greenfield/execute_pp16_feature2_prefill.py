#!/usr/bin/env python3
"""Exactly-once numerical capture for the protected PP16 feature2 graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from acquire_pp16_feature2_prefill import run_feature2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--token-oracle-dir", type=Path, required=True)
    parser.add_argument("--dsa-oracle-dir", type=Path, required=True)
    parser.add_argument("--layer1-internal-reference", type=Path, required=True)
    parser.add_argument("--db529-internal-dir", type=Path, required=True)
    parser.add_argument("--expected-main-stablehlo-sha256", required=True)
    parser.add_argument("--expected-main-optimized-hlo-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    parser.add_argument("--result-npz", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    return run_feature2(parse_args(), execute_main=True)


if __name__ == "__main__":
    raise SystemExit(main())
