#!/usr/bin/env python3
"""Offline causal adjudication of the acquired forced-round PP16 HLO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from glm_tpu.greenfield.validation.gate_d_forced_round_hlo import (
    EXPECTED_RUN_TAG,
    adjudicate_acquired_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/home/gianl/gate-d-runs") / EXPECTED_RUN_TAG,
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    report = adjudicate_acquired_run(arguments.run_dir)
    encoded = (
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
