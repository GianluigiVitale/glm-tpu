#!/usr/bin/env python3
"""Compare accepted and table-on PP8 layer-0 attention operands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation import (  # noqa: E402
    AttentionOutputComparisonConfig,
    compare_attention_output_operands,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-capture-dir", type=Path, required=True)
    parser.add_argument("--greenfield-ingredients-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-capture-file-sha256", required=True)
    parser.add_argument("--accepted-run-tag", required=True)
    parser.add_argument("--ingredients-contract-sha256", required=True)
    parser.add_argument("--ingredients-tensor-sha256", required=True)
    parser.add_argument("--greenfield-code-hash", required=True)
    parser.add_argument("--greenfield-run-tag", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--oracle-pin", required=True)
    parser.add_argument("--main-rope-table-sha256", required=True)
    parser.add_argument("--position", type=int, default=8155)
    args = parser.parse_args()
    result = compare_attention_output_operands(
        AttentionOutputComparisonConfig(
            accepted_capture_dir=args.accepted_capture_dir,
            greenfield_ingredients_dir=args.greenfield_ingredients_dir,
            output_dir=args.output,
            expected_accepted_capture_file_sha256=(
                args.accepted_capture_file_sha256
            ),
            expected_accepted_run_tag=args.accepted_run_tag,
            expected_ingredients_contract_sha256=(
                args.ingredients_contract_sha256
            ),
            expected_ingredients_tensor_sha256=args.ingredients_tensor_sha256,
            expected_greenfield_code_hash=args.greenfield_code_hash,
            expected_greenfield_run_tag=args.greenfield_run_tag,
            expected_legacy_code_hash=args.legacy_code_hash,
            expected_oracle_pin=args.oracle_pin,
            expected_main_rope_table_sha256=args.main_rope_table_sha256,
            expected_position=args.position,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
