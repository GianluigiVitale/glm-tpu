#!/usr/bin/env python3
"""Derive one sealed PP8 MoE layer into exact WS32 final ownership."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.checkpoint.ws32_one_layer import (
    Ws32OneLayerPackConfig,
    pack_ws32_one_layer,
)


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-payload-dir", type=Path, required=True)
    parser.add_argument("--source-artifact-uri", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--mesh-hash", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observed_head = _head()
    if observed_head != args.expected_code_hash:
        raise SystemExit(
            "WS32 pack code identity drifted: "
            f"expected {args.expected_code_hash}, observed {observed_head}"
        )
    manifest = pack_ws32_one_layer(
        Ws32OneLayerPackConfig(
            source_manifest_path=args.source_manifest,
            source_payload_dir=args.source_payload_dir,
            source_artifact_uri=args.source_artifact_uri,
            source_manifest_sha256=args.source_manifest_sha256,
            output_dir=args.output,
            code_hash=observed_head,
            mesh_hash=args.mesh_hash,
        )
    )
    print(
        json.dumps(
            {
                "artifact_kind": manifest["artifact_kind"],
                "files": len(manifest["files"]),
                "layer": manifest["layer"],
                "manifest_sha256": manifest["manifest_sha256"],
                "mesh_hash": manifest["mesh_hash"],
                "packed_payload_byte_count": manifest[
                    "packed_payload_byte_count"
                ],
                "source_manifest_sha256": manifest["source"][
                    "manifest_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
