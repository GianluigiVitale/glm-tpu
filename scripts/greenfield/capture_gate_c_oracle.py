#!/usr/bin/env python3
"""Capture the standalone raw-source dense/DSA/IndexShare Gate C oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation import (  # noqa: E402
    GateCOracleConfig,
    capture_gate_c_oracle,
)


def _source_hash(value: str) -> tuple[str, str]:
    try:
        name, digest = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected NAME=SHA256") from exc
    return name, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-layer", type=int, default=2)
    parser.add_argument("--consumer-layer", type=int, default=3)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--vllm-code-hash", required=True)
    parser.add_argument(
        "--reference-source-hash",
        action="append",
        type=_source_hash,
        required=True,
    )
    parser.add_argument("--allowed-source-shard", action="append", required=True)
    args = parser.parse_args()
    manifest = capture_gate_c_oracle(
        GateCOracleConfig(
            source_root=args.source_root,
            output_dir=args.output,
            source_uri=args.source_uri,
            source_revision=args.source_revision,
            code_hash=args.expected_code_hash,
            legacy_code_hash=args.legacy_code_hash,
            vllm_code_hash=args.vllm_code_hash,
            reference_source_hashes=tuple(args.reference_source_hash),
            allowed_source_shards=tuple(args.allowed_source_shard),
            producer_layer=args.producer_layer,
            consumer_layer=args.consumer_layer,
        )
    )
    print(
        json.dumps(
            {
                "current_selected": manifest["cases"]["full_dsa"][
                    "current_selected"
                ],
                "file_sha256": manifest["file"]["sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "payload_byte_count": manifest["cases"]["index_share"][
                    "payload_byte_count"
                ],
                "selected_count": manifest["cases"]["full_dsa"][
                    "selected_count"
                ],
                "source_tensors": len(manifest["source_tensors"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
