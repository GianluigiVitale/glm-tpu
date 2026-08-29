#!/usr/bin/env python3
"""Build the sealed callback executable-class certificate offline."""

from __future__ import annotations

import argparse
from pathlib import Path

from glm_tpu.greenfield.benchmarking.callback_executable_class import (
    classify_callback_executable_class,
    serialize_callback_executable_class,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-root", required=True, type=Path)
    parser.add_argument("--historical-root", required=True, type=Path)
    parser.add_argument("--current-root", required=True, type=Path)
    parser.add_argument("--rejection-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = classify_callback_executable_class(
        args.accepted_root,
        historical_root=args.historical_root,
        current_root=args.current_root,
        rejection_artifact_path=args.rejection_artifact,
    )
    payload = serialize_callback_executable_class(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(payload.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
