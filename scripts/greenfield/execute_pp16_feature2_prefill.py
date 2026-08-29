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
    parser.add_argument("--full-width-rounded-then-slice", action="store_true")
    parser.add_argument("--observe-position-113", action="store_true")
    parser.add_argument("--exact-layer0-prompt-keys", action="store_true")
    parser.add_argument("--expected-main-stablehlo-sha256", required=True)
    parser.add_argument("--expected-main-canonical-hlo-sha256", required=True)
    parser.add_argument(
        "--expected-main-canonical-hlo-byte-count", type=int, required=True
    )
    parser.add_argument(
        "--expected-main-stack-frame-reference-count", type=int, required=True
    )
    for package in ("jax", "jaxlib", "libtpu"):
        parser.add_argument(f"--expected-{package}-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    parser.add_argument("--result-npz", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    return run_feature2(_require_exact_db518_pins(parse_args()), execute_main=True)


_EXACT_DB518_PINS: dict[str, str | int] = {
    "expected_main_stablehlo_sha256": (
        "a79d4823db0fecf8b1bc980b16df283bbe7a497795fa06ff783d3cde8fa98f6b"
    ),
    "expected_main_canonical_hlo_sha256": (
        "56b9b88dc081dd5d3ea2219d46be641128cbed6e9759754be7f87578b51b5d8c"
    ),
    "expected_main_canonical_hlo_byte_count": 6_863_602,
    "expected_main_stack_frame_reference_count": 15_339,
    "expected_jax_version": "0.10.1",
    "expected_jaxlib_version": "0.10.1",
    "expected_libtpu_version": "0.0.41",
}


def _require_exact_db518_pins(args: argparse.Namespace) -> argparse.Namespace:
    if not args.exact_layer0_prompt_keys:
        return args
    if not args.full_width_rounded_then_slice or not args.observe_position_113:
        raise SystemExit(
            "exact DB518 execution requires full-width rounding and position-113 observation"
        )
    drift = {
        name: {"expected": expected, "observed": getattr(args, name)}
        for name, expected in _EXACT_DB518_PINS.items()
        if getattr(args, name) != expected
    }
    if drift:
        names = ", ".join(sorted(drift))
        raise SystemExit(f"exact DB518 acquisition pin drift: {names}")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
