#!/usr/bin/env python3
"""Seal the accepted layer-0 normalized dense-MLP input row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation.dense_input import (  # noqa: E402
    AcceptedDenseInputCaptureConfig,
    capture_accepted_dense_input,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--oracle-pin", required=True)
    parser.add_argument("--model-id", default="zai-org/GLM-5.2-FP8")
    parser.add_argument("--layer-name", required=True)
    parser.add_argument("--position", type=int, default=8155)
    parser.add_argument("--process-count", type=int, default=8)
    parser.add_argument("--capture-process-indices", default="0")
    args = parser.parse_args()
    result = capture_accepted_dense_input(
        AcceptedDenseInputCaptureConfig(
            source_dump_dir=args.source_dump_dir,
            output_dir=args.output,
            expected_run_tag=args.run_tag,
            expected_legacy_code_hash=args.legacy_code_hash,
            expected_oracle_pin=args.oracle_pin,
            expected_model_id=args.model_id,
            expected_layer_name=args.layer_name,
            expected_position=args.position,
            expected_process_count=args.process_count,
            expected_capture_process_indices=tuple(
                int(value.strip())
                for value in args.capture_process_indices.split(",")
                if value.strip()
            ),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
