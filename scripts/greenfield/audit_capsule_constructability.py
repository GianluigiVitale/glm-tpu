#!/usr/bin/env python3
"""Audit whether sealed state can form a Gate-D precompile capsule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.capsule_constructability import (  # noqa: E402
    audit_capsule_constructability,
    write_capsule_constructability_report,
)
from glm_tpu.greenfield.errors import BenchmarkValidationError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "jax" in sys.modules:
        raise BenchmarkValidationError("constructability audit must start without JAX")
    report = audit_capsule_constructability(args.contract, args.contract_sha256)
    if "jax" in sys.modules:
        raise BenchmarkValidationError("constructability audit imported JAX")
    write_capsule_constructability_report(args.output, report)
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "missing_old_artifact_watchpoints": report[
                    "missing_old_artifact_watchpoints"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BenchmarkValidationError as error:
        raise SystemExit(f"capsule constructability audit refused: {error}") from error
