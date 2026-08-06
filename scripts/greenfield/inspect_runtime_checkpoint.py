#!/usr/bin/env python3
"""Verify a complete executable-ready runtime checkpoint without loading JAX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    RuntimeCheckpointLoadExpectation,
    verify_runtime_packed_checkpoint,
)
from scripts.greenfield.pack_runtime_checkpoint import _build_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-packed-manifest-sha256", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.runtime_root / "runtime_manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise RuntimeError("runtime manifest is not an object")
    context_args = SimpleNamespace(
        source_checkpoint_root=args.source_checkpoint_root,
        source_packed_manifest_sha256=args.source_packed_manifest_sha256,
        destination=manifest["destination"],
    )
    context = _build_context(context_args, manifest["pack_code_hash"])
    expectation = RuntimeCheckpointLoadExpectation(
        runtime_manifest_sha256=args.runtime_manifest_sha256,
        runtime_layout_manifest_sha256=manifest[
            "runtime_layout_manifest_sha256"
        ],
        runtime_layout_hash=manifest["runtime_layout_hash"],
        source_packed_manifest_sha256=manifest[
            "source_packed_manifest_sha256"
        ],
        source_layout_manifest_sha256=manifest[
            "source_layout_manifest_sha256"
        ],
        plan_hash=manifest["plan_hash"],
        schedule_hash=manifest["schedule_hash"],
        pack_code_hash=manifest["pack_code_hash"],
        destination=manifest["destination"],
        source_destination=manifest["source_checkpoint_destination"],
        plan_id=manifest["plan_id"],
        model_id=manifest["model_id"],
    )
    verified = verify_runtime_packed_checkpoint(
        args.runtime_root,
        expectation,
        context.layout,
        context.source_checkpoint,
    )
    summary = {
        "artifact_kind": "greenfield_verified_runtime_checkpoint",
        "file_count": len(verified.plans),
        "model_id": expectation.model_id,
        "padding_bytes": manifest["padding_bytes"],
        "plan_hash": expectation.plan_hash,
        "plan_id": expectation.plan_id,
        "runtime_file_bytes": manifest["runtime_file_bytes"],
        "runtime_layout_hash": expectation.runtime_layout_hash,
        "runtime_layout_manifest_sha256": (
            expectation.runtime_layout_manifest_sha256
        ),
        "runtime_manifest_sha256": expectation.runtime_manifest_sha256,
        "runtime_payload_bytes": manifest["runtime_payload_bytes"],
        "schedule_hash": expectation.schedule_hash,
        "source_leaf_count": manifest["source_leaf_count"],
        "source_packed_manifest_sha256": (
            expectation.source_packed_manifest_sha256
        ),
        "source_payload_bytes": manifest["source_payload_bytes"],
        "tensor_count": manifest["tensor_count"],
        "verified": True,
    }
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise RuntimeError(f"append-only output exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
