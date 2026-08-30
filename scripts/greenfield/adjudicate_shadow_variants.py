#!/usr/bin/env python3
"""Offline Gate-D shadow-variant normal-form adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.errors import BenchmarkValidationError  # noqa: E402
from glm_tpu.greenfield.shadow_variant_adjudication import (  # noqa: E402
    adjudicate_shadow_variants,
    write_shadow_variant_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "jax" in sys.modules:
        raise BenchmarkValidationError("offline adjudication must start without JAX")
    report = adjudicate_shadow_variants(args.contract, args.contract_sha256)
    if "jax" in sys.modules:
        raise BenchmarkValidationError("offline adjudication imported JAX")
    write_shadow_variant_report(args.output, report)
    print(
        json.dumps(
            {
                "classification": report["classification"],
                "output": str(args.output),
                "unresolved_variant_ids": report["unresolved_variant_ids"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BenchmarkValidationError as error:
        raise SystemExit(f"shadow variant adjudication refused: {error}") from error
