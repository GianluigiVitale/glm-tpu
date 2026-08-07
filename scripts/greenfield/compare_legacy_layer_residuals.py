#!/usr/bin/env python3
"""Reconstruct legacy GLM boundaries and compare the sealed greenfield row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation import (  # noqa: E402
    LegacyResidualComparisonConfig,
    compare_legacy_residuals,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dump-dir", type=Path, required=True)
    parser.add_argument("--greenfield-npz", type=Path, required=True)
    parser.add_argument("--greenfield-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--oracle-pin", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--process-count", type=int, default=8)
    parser.add_argument("--boundary-id", type=int, action="append", default=[])
    args = parser.parse_args()

    config_kwargs = {}
    if args.boundary_id:
        config_kwargs["expected_boundary_ids"] = tuple(args.boundary_id)
    comparison = compare_legacy_residuals(
        LegacyResidualComparisonConfig(
            source_dump_dir=args.source_dump_dir,
            greenfield_npz=args.greenfield_npz,
            greenfield_contract=args.greenfield_contract,
            output_dir=args.output,
            expected_position=args.position,
            expected_run_tag=args.run_tag,
            expected_legacy_code_hash=args.legacy_code_hash,
            expected_oracle_pin=args.oracle_pin,
            expected_model_id=args.model_id,
            expected_process_count=args.process_count,
            **config_kwargs,
        ))
    print(json.dumps({
        "divergent_boundary_count": comparison["divergent_boundary_count"],
        "first_selected_divergent_boundary": comparison[
            "first_divergent_boundary"
        ],
        "legacy_canonical_sha256": comparison["legacy"]["canonical_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
