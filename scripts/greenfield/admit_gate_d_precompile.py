#!/usr/bin/env python3
"""Audit Gate-D precompile admission-v2 candidates without JAX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.errors import BenchmarkValidationError  # noqa: E402
from glm_tpu.greenfield.gate_d_precompile_admission import (  # noqa: E402
    admit_gate_d_precompile_candidates,
    write_gate_d_precompile_admission_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "jax" in sys.modules:
        raise BenchmarkValidationError("precompile admission must start without JAX")
    report = admit_gate_d_precompile_candidates(
        args.contract, args.contract_sha256
    )
    if "jax" in sys.modules:
        raise BenchmarkValidationError("precompile admission imported JAX")
    write_gate_d_precompile_admission_report(args.output, report)
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
        raise SystemExit(f"Gate-D precompile admission refused: {error}") from error
