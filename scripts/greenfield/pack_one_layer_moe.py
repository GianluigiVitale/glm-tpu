#!/usr/bin/env python3
"""Pack one GLM sparse layer into final PP8 stage ownership."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.checkpoint.one_layer import (
    OneLayerPackConfig,
    pack_one_layer_moe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--topology-hash", required=True)
    parser.add_argument("--plan-group-hash", required=True)
    args = parser.parse_args()
    manifest = pack_one_layer_moe(
        OneLayerPackConfig(
            source_root=args.source_root,
            output_dir=args.output,
            source_uri=args.source_uri,
            source_revision=args.source_revision,
            code_hash=args.expected_code_hash,
            topology_hash=args.topology_hash,
            plan_group_hash=args.plan_group_hash,
            layer=args.layer,
        )
    )
    print(
        json.dumps(
            {
                "files": len(manifest["files"]),
                "layer": manifest["layer"],
                "manifest_sha256": manifest["manifest_sha256"],
                "packed_payload_byte_count": manifest[
                    "packed_payload_byte_count"
                ],
                "source_leaves": len(manifest["source_leaves"]),
                "source_payload_byte_count": manifest[
                    "source_payload_byte_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
