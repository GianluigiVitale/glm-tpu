#!/usr/bin/env python3
"""Seal accepted layer-0 pre-reduction partials and compare with DB548."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation.dense_partials import (  # noqa: E402
    DensePartialsCaptureConfig,
    capture_and_compare_dense_partials,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dump-dir", type=Path, required=True)
    parser.add_argument("--db548-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--oracle-pin", required=True)
    parser.add_argument("--db548-runner-sha256", required=True)
    parser.add_argument("--db548-tensor-sha256", required=True)
    parser.add_argument("--db548-summary-sha256", required=True)
    parser.add_argument("--db548-success-sha256", required=True)
    parser.add_argument("--model-id", default="zai-org/GLM-5.2-FP8")
    parser.add_argument("--layer-name", required=True)
    parser.add_argument("--position", type=int, default=8155)
    parser.add_argument("--process-count", type=int, default=8)
    args = parser.parse_args()
    result = capture_and_compare_dense_partials(DensePartialsCaptureConfig(
        source_dump_dir=args.source_dump_dir,
        db548_dir=args.db548_dir,
        output_dir=args.output,
        expected_run_tag=args.run_tag,
        expected_legacy_code_hash=args.legacy_code_hash,
        expected_oracle_pin=args.oracle_pin,
        expected_db548_runner_sha256=args.db548_runner_sha256,
        expected_db548_tensor_sha256=args.db548_tensor_sha256,
        expected_db548_summary_sha256=args.db548_summary_sha256,
        expected_db548_success_sha256=args.db548_success_sha256,
        expected_model_id=args.model_id,
        expected_layer_name=args.layer_name,
        expected_position=args.position,
        expected_process_count=args.process_count,
    ))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
