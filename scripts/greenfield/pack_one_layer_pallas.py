#!/usr/bin/env python3
"""Derive one append-only final-MXU-layout layer artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.checkpoint.one_layer_pallas import (
    PallasOneLayerPackConfig,
    pack_pallas_one_layer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--source-artifact-uri", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    args = parser.parse_args()
    manifest = pack_pallas_one_layer(
        PallasOneLayerPackConfig(
            source_artifact_dir=args.source_artifact,
            source_artifact_uri=args.source_artifact_uri,
            source_manifest_sha256=args.source_manifest_sha256,
            output_dir=args.output,
            code_hash=args.expected_code_hash,
        )
    )
    print(
        json.dumps(
            {
                "files": len(manifest["files"]),
                "layout_sha256": manifest["layout"]["layout_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "packed_payload_byte_count": manifest[
                    "packed_payload_byte_count"
                ],
                "source_manifest_sha256": manifest[
                    "source_manifest_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
