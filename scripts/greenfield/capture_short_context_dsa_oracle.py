#!/usr/bin/env python3
"""Seal a provenance-pinned fresh 2K legacy DSA event capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation import (  # noqa: E402
    ShortContextDsaOracleConfig,
    capture_short_context_dsa_oracle,
)


PRODUCER_LAYER_IDS = (
    0,
    1,
    2,
    6,
    10,
    14,
    18,
    22,
    26,
    30,
    34,
    38,
    42,
    46,
    50,
    54,
    58,
    62,
    66,
    70,
    74,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--token-oracle-dir", type=Path, required=True)
    parser.add_argument("--source-dump-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--source-capture-code-hash")
    parser.add_argument("--legacy-repository-pin", required=True)
    parser.add_argument("--token-oracle-manifest-sha256", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--item-row-id", type=int, required=True)
    parser.add_argument("--expected-harness-git", required=True)
    parser.add_argument("--expected-fork-git", required=True)
    parser.add_argument("--expected-oob-dir", required=True)
    parser.add_argument("--expected-dump-prefix", required=True)
    args = parser.parse_args()
    manifest = capture_short_context_dsa_oracle(
        ShortContextDsaOracleConfig(
            results_db=args.results_db,
            token_oracle_dir=args.token_oracle_dir,
            source_dump_dir=args.source_dump_dir,
            output_dir=args.output,
            capture_code_hash=args.expected_code_hash,
            source_capture_code_hash=(
                args.source_capture_code_hash or args.expected_code_hash
            ),
            legacy_repository_pin=args.legacy_repository_pin,
            token_oracle_manifest_sha256=(
                args.token_oracle_manifest_sha256
            ),
            run_id=args.run_id,
            item_row_id=args.item_row_id,
            expected_harness_git=args.expected_harness_git,
            expected_fork_git=args.expected_fork_git,
            expected_benchmark="passkey_L2040_d0.25",
            expected_model_uri=(
                "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
            ),
            expected_prompt_tokens=2034,
            expected_generated_tokens=20,
            expected_seed=283835,
            expected_gold="110391",
            expected_oob_dir=args.expected_oob_dir,
            expected_dump_prefix=args.expected_dump_prefix,
            expected_process_count=8,
            first_source_step=2,
            decode_step_count=14,
            first_decode_position=2034,
            selected_width=2048,
            producer_layer_ids=PRODUCER_LAYER_IDS,
        )
    )
    print(
        json.dumps(
            {
                "decode_step_count": manifest["event_contract"][
                    "decode_step_count"
                ],
                "event_count": manifest["event_contract"]["event_count"],
                "manifest_sha256": manifest["manifest_sha256"],
                "source_dump_file_count": len(
                    manifest["source_dump_files"]
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
