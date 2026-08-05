#!/usr/bin/env python3
"""Capture a standalone raw-source oracle for one GLM sparse layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation.one_layer_oracle import (
    OneLayerOracleConfig,
    capture_one_layer_oracle,
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
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument(
        "--legacy-source-hash", action="append", type=_source_hash, required=True
    )
    parser.add_argument("--vllm-code-hash", required=True)
    parser.add_argument(
        "--vllm-source-hash", action="append", type=_source_hash, required=True
    )
    parser.add_argument("--allowed-source-shard", action="append", required=True)
    args = parser.parse_args()
    manifest = capture_one_layer_oracle(
        OneLayerOracleConfig(
            source_root=args.source_root,
            output_dir=args.output,
            source_uri=args.source_uri,
            source_revision=args.source_revision,
            code_hash=args.expected_code_hash,
            legacy_code_hash=args.legacy_code_hash,
            legacy_source_hashes=tuple(args.legacy_source_hash),
            vllm_code_hash=args.vllm_code_hash,
            vllm_source_hashes=tuple(args.vllm_source_hash),
            allowed_source_shards=tuple(args.allowed_source_shard),
            layer=args.layer,
        )
    )
    print(
        json.dumps(
            {
                "concentrated_routes": manifest["cases"]["concentrated"][
                    "route_indices"
                ],
                "file_sha256": manifest["file"]["sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "normal_routes": manifest["cases"]["normal"]["route_indices"],
                "source_tensors": len(manifest["source_tensors"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
