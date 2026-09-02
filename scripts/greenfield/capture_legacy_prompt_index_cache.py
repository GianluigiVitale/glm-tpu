#!/usr/bin/env python3
"""Seal one layer's accepted logical prompt-key cache from legacy host dumps.

``--layer-id`` selects the full-indexer layer (default 0); the legacy
``kv_caches`` slot is derived, never supplied, and a wrong slot fails closed.
"""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--legacy-repository-pin", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--source-run-id", type=int, required=True)
    parser.add_argument("--source-item-row-id", type=int, required=True)
    parser.add_argument("--layer0-input-dir", type=Path, required=True)
    parser.add_argument("--layer0-input-manifest-sha256", required=True)
    parser.add_argument("--layer-id", type=int, default=0)
    args = parser.parse_args()

    from glm_tpu.greenfield.validation.layer0_dsa_association import (
        inspect_layer0_dsa_association_input,
    )
    from glm_tpu.greenfield.validation.prompt_index_cache import (
        LegacyPromptIndexCacheConfig,
        capture_legacy_prompt_index_cache,
    )

    input_manifest, input_arrays = inspect_layer0_dsa_association_input(
        args.layer0_input_dir,
        expected_manifest_sha256=args.layer0_input_manifest_sha256,
    )
    manifest = capture_legacy_prompt_index_cache(
        LegacyPromptIndexCacheConfig(
            source_dump_dir=args.source_dump_dir,
            output_dir=args.output,
            capture_code_hash=args.expected_code_hash,
            legacy_repository_pin=args.legacy_repository_pin,
            run_tag=args.run_tag,
            source_run_id=args.source_run_id,
            source_item_row_id=args.source_item_row_id,
            layer0_input_manifest_sha256=input_manifest["manifest_sha256"],
            prompt_token_ids_sha256=input_manifest["arrays"][
                "prompt_token_ids"
            ]["sha256"],
            expected_prompt_tokens=int(input_arrays["prompt_token_ids"].size),
            layer_id=args.layer_id,
        )
    )
    print(
        json.dumps(
            {
                "artifact_kind": manifest["artifact_kind"],
                "layer_id": args.layer_id,
                "manifest_sha256": manifest["manifest_sha256"],
                "prompt_index_key_bfloat16_sha256": manifest[
                    "prompt_index_key_bfloat16_sha256"
                ],
                "source_dump_file_count": len(manifest["source_dump_files"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
