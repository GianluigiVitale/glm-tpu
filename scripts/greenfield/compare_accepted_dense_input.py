#!/usr/bin/env python3
"""Compare the accepted normalized dense input with protected DB540."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation.dense_input import (  # noqa: E402
    DenseInputComparisonConfig,
    compare_dense_input_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-capture-dir", type=Path, required=True)
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-capture-file-sha256", required=True)
    parser.add_argument("--probe-runner-sha256", required=True)
    parser.add_argument("--probe-tensor-sha256", required=True)
    parser.add_argument("--probe-summary-sha256", required=True)
    parser.add_argument("--probe-success-sha256", required=True)
    parser.add_argument("--probe-run-id", type=int, required=True)
    parser.add_argument("--accepted-run-tag", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--oracle-pin", required=True)
    parser.add_argument("--probe-code-hash", required=True)
    parser.add_argument("--probe-tag", required=True)
    parser.add_argument("--position", type=int, default=8155)
    args = parser.parse_args()
    result = compare_dense_input_candidate(
        DenseInputComparisonConfig(
            accepted_capture_dir=args.accepted_capture_dir,
            probe_dir=args.probe_dir,
            output_dir=args.output,
            expected_accepted_capture_file_sha256=(
                args.accepted_capture_file_sha256
            ),
            expected_probe_runner_sha256=args.probe_runner_sha256,
            expected_probe_tensor_sha256=args.probe_tensor_sha256,
            expected_probe_summary_sha256=args.probe_summary_sha256,
            expected_probe_success_sha256=args.probe_success_sha256,
            expected_accepted_run_tag=args.accepted_run_tag,
            expected_legacy_code_hash=args.legacy_code_hash,
            expected_oracle_pin=args.oracle_pin,
            expected_probe_code_hash=args.probe_code_hash,
            expected_probe_tag=args.probe_tag,
            expected_position=args.position,
            expected_probe_run_id=args.probe_run_id,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
