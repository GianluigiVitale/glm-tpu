#!/usr/bin/env python3
"""Compare one sealed accepted layer-0 DSA state with reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation import (  # noqa: E402
    LegacyDsaInternalComparisonConfig,
    compare_legacy_dsa_internals,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dump-dir", type=Path, required=True)
    parser.add_argument("--layer0-input-dir", type=Path, required=True)
    parser.add_argument("--distributed-q-a-norm-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--greenfield-code-hash", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--oracle-pin", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--q-a-manifest-sha256", required=True)
    parser.add_argument("--q-a-code-hash", required=True)
    parser.add_argument("--model-id", default="zai-org/GLM-5.2-FP8")
    parser.add_argument(
        "--layer-name", default="model.layers.0.self_attn.attn"
    )
    parser.add_argument("--position", type=int, default=8155)
    parser.add_argument("--process-count", type=int, default=8)
    parser.add_argument("--capture-process-indices", default="0")
    args = parser.parse_args()
    capture_process_indices = tuple(
        int(value.strip())
        for value in args.capture_process_indices.split(",")
        if value.strip()
    )

    result = compare_legacy_dsa_internals(
        LegacyDsaInternalComparisonConfig(
            source_dump_dir=args.source_dump_dir,
            layer0_input_dir=args.layer0_input_dir,
            distributed_q_a_norm_dir=args.distributed_q_a_norm_dir,
            output_dir=args.output,
            expected_run_tag=args.run_tag,
            expected_greenfield_code_hash=args.greenfield_code_hash,
            expected_legacy_code_hash=args.legacy_code_hash,
            expected_oracle_pin=args.oracle_pin,
            expected_input_manifest_sha256=args.input_manifest_sha256,
            expected_q_a_manifest_sha256=args.q_a_manifest_sha256,
            expected_q_a_code_hash=args.q_a_code_hash,
            expected_model_id=args.model_id,
            expected_layer_name=args.layer_name,
            expected_position=args.position,
            expected_process_count=args.process_count,
            expected_capture_process_indices=capture_process_indices,
        )
    )
    print(
        json.dumps(
            {
                "all_fields_elementwise_exact": result[
                    "all_fields_elementwise_exact"
                ],
                "divergent_fields": result["divergent_fields"],
                "first_divergent_field": result["first_divergent_field"],
                "owner_actual_sha256": result["owner_actual_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
