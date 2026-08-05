#!/usr/bin/env python3
"""Pack the real Gate C tensor set into exact final PP8 ownership."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    build_gate_c_layout,
    inspect_gate_c_checkpoint,
    pack_gate_c_checkpoint,
)
from glm_tpu.greenfield.partitioning import inspect_layout_manifest  # noqa: E402
from glm_tpu.greenfield.validation import inspect_gate_c_oracle  # noqa: E402


EXPECTED_WORKTREE = Path("/home/gianl/glm-tpu-topology-rewrite")
EXPECTED_BRANCH = "rewrite/topology-first-decode"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), *args], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-layout", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()
    if REPO != EXPECTED_WORKTREE:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    if _git("branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("Gate C pack must run on the isolated rewrite branch")
    code_hash = _git("rev-parse", "HEAD")
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected {args.expected_code_hash}, found {code_hash}"
        )
    if _git("status", "--porcelain"):
        raise RuntimeError("Gate C pack requires a clean worktree")
    parent = inspect_layout_manifest(args.parent_layout)
    oracle = inspect_gate_c_oracle(args.oracle_dir)
    layout = build_gate_c_layout(
        parent_layout=parent,
        oracle_manifest=oracle,
        code_hash=code_hash,
    )
    manifest = pack_gate_c_checkpoint(
        layout=layout,
        source_root=args.source_root,
        output_dir=args.output,
        chunk_bytes=args.chunk_bytes,
    )
    inspect_gate_c_checkpoint(
        args.output,
        parent_layout=parent,
        oracle_manifest=oracle,
    )
    print(
        json.dumps(
            {
                "files": manifest["file_count"],
                "layout_manifest_sha256": manifest[
                    "layout_manifest_sha256"
                ],
                "manifest_sha256": manifest["manifest_sha256"],
                "oracle_manifest_sha256": manifest[
                    "oracle_manifest_sha256"
                ],
                "packed_payload_bytes": manifest["packed_payload_bytes"],
                "source_payload_bytes": manifest["source_payload_bytes"],
                "source_tensors": manifest["source_tensor_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
