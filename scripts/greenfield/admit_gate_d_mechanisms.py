#!/usr/bin/env python3
"""Classify Gate-D mechanism proposals without importing JAX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.errors import BenchmarkValidationError  # noqa: E402
from glm_tpu.greenfield.gate_d_admission import (  # noqa: E402
    admit_gate_d_mechanisms,
    write_gate_d_admission_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed offline Gate-D mechanism admission."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if "jax" in sys.modules:
        raise BenchmarkValidationError(
            "offline Gate-D admission must start without JAX"
        )
    report = admit_gate_d_mechanisms(args.contract, args.contract_sha256)
    if "jax" in sys.modules:
        raise BenchmarkValidationError(
            "offline Gate-D admission imported JAX"
        )
    write_gate_d_admission_report(args.output, report)
    print(
        json.dumps(
            {
                "admitted_candidate_ids": report["admitted_candidate_ids"],
                "classification": report["classification"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BenchmarkValidationError as error:
        raise SystemExit(f"Gate-D admission refused: {error}") from error
