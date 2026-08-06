#!/usr/bin/env python3
"""Seal a provenance-pinned accepted 2K legacy result as token IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation import (  # noqa: E402
    ShortContextOracleConfig,
    capture_short_context_oracle,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--legacy-repository-pin", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--item-row-id", type=int, required=True)
    parser.add_argument("--expected-harness-git", required=True)
    parser.add_argument("--expected-fork-git", required=True)
    parser.add_argument("--expected-benchmark", required=True)
    parser.add_argument("--expected-model-uri", required=True)
    parser.add_argument("--expected-prompt-tokens", type=int, required=True)
    parser.add_argument("--expected-generated-tokens", type=int, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-gold", required=True)
    args = parser.parse_args()
    manifest = capture_short_context_oracle(
        ShortContextOracleConfig(
            results_db=args.results_db,
            tokenizer_root=args.tokenizer_root,
            output_dir=args.output,
            capture_code_hash=args.expected_code_hash,
            legacy_repository_pin=args.legacy_repository_pin,
            run_id=args.run_id,
            item_row_id=args.item_row_id,
            expected_harness_git=args.expected_harness_git,
            expected_fork_git=args.expected_fork_git,
            expected_benchmark=args.expected_benchmark,
            expected_model_uri=args.expected_model_uri,
            expected_prompt_tokens=args.expected_prompt_tokens,
            expected_generated_tokens=args.expected_generated_tokens,
            expected_seed=args.expected_seed,
            expected_gold=args.expected_gold,
        )
    )
    print(
        json.dumps(
            {
                "generated_token_count": manifest["generated_token_count"],
                "generated_token_ids_sha256": manifest[
                    "generated_token_ids_sha256"
                ],
                "manifest_sha256": manifest["manifest_sha256"],
                "prompt_token_count": manifest["prompt_token_count"],
                "prompt_token_ids_sha256": manifest[
                    "prompt_token_ids_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
