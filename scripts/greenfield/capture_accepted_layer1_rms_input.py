#!/usr/bin/env python3
"""Seal the accepted layer-1 fused-add/RMSNorm FP32 input boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation.layer1_rms_input import (  # noqa: E402
    Layer1RmsInputCaptureConfig,
    capture_accepted_layer1_rms_input,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--db550-boundary", type=Path, required=True)
    parser.add_argument("--straddler-classification", type=Path, required=True)
    parser.add_argument("--vllm-repository", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--oracle-pin", required=True)
    parser.add_argument("--model-id", default="zai-org/GLM-5.2-FP8")
    parser.add_argument("--layer-name", default="model.layers.1.input_layernorm")
    parser.add_argument("--position", type=int, default=8155)
    parser.add_argument("--process-count", type=int, default=8)
    parser.add_argument("--capture-process-index", type=int, default=0)
    parser.add_argument("--source-row", type=int, default=0)
    args = parser.parse_args()
    result = capture_accepted_layer1_rms_input(
        Layer1RmsInputCaptureConfig(
            source_dump_dir=args.source_dump_dir,
            output_dir=args.output,
            db550_boundary_path=args.db550_boundary,
            straddler_classification_path=args.straddler_classification,
            vllm_repository=args.vllm_repository,
            expected_run_tag=args.run_tag,
            expected_legacy_code_hash=args.legacy_code_hash,
            expected_oracle_pin=args.oracle_pin,
            expected_model_id=args.model_id,
            expected_layer_name=args.layer_name,
            expected_position=args.position,
            expected_process_count=args.process_count,
            expected_capture_process_index=args.capture_process_index,
            expected_source_row=args.source_row,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
