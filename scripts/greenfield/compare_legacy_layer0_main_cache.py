#!/usr/bin/env python3
"""Compare the bounded accepted and PP8 layer-0 main-cache captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dump-dir", type=Path, required=True)
    parser.add_argument("--ingredients-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--legacy-repository-pin", required=True)
    parser.add_argument("--accepted-oracle-pin", required=True)
    parser.add_argument("--ingredients-code-hash", required=True)
    parser.add_argument("--ingredients-contract-sha256", required=True)
    parser.add_argument("--ingredients-tensor-sha256", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--source-item-row-id", type=int, required=True)
    args = parser.parse_args()

    from glm_tpu.greenfield.validation.legacy_main_cache import (
        LegacyMainCacheComparisonConfig,
        compare_legacy_layer0_main_cache,
    )

    manifest = compare_legacy_layer0_main_cache(
        LegacyMainCacheComparisonConfig(
            source_dump_dir=args.source_dump_dir,
            ingredients_dir=args.ingredients_dir,
            output_dir=args.output,
            capture_code_hash=args.expected_code_hash,
            legacy_repository_pin=args.legacy_repository_pin,
            accepted_oracle_pin=args.accepted_oracle_pin,
            ingredients_code_hash=args.ingredients_code_hash,
            ingredients_contract_sha256=args.ingredients_contract_sha256,
            ingredients_tensor_sha256=args.ingredients_tensor_sha256,
            run_tag=args.run_tag,
            source_run_id=args.source_run_id,
            source_item_row_id=args.source_item_row_id,
        ))
    print(
        json.dumps(
            {
                "classification": manifest["classification"],
                "first_divergent_primitive":
                manifest["first_divergent_primitive"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
