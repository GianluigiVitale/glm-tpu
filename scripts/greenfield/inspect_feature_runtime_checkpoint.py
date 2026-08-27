#!/usr/bin/env python3
"""Verify the complete expert-feature runtime artifact without loading JAX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    verify_feature_runtime_packed_checkpoint,
)
from glm_tpu.greenfield.model import (  # noqa: E402
    FINAL_DENSE_CONVOLUTION_RUNTIME_LAYOUT,
    FUSED_QKV_A_N82_RUNTIME_LAYOUT,
)
from scripts.greenfield.pack_feature_runtime_checkpoint import (  # noqa: E402
    _build_context,
    _mapping_hash,
    build_load_expectation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-packed-manifest-sha256", required=True)
    parser.add_argument("--source-runtime-root", type=Path, required=True)
    parser.add_argument("--source-runtime-manifest-sha256", required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--source-metadata-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads((args.runtime_root / "runtime_manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise RuntimeError("feature runtime manifest is not an object")
    if (
        _mapping_hash(manifest, hash_field="manifest_sha256")
        != args.runtime_manifest_sha256
    ):
        raise RuntimeError("feature runtime manifest identity drifted")
    if (
        manifest.get("source_runtime_manifest_sha256")
        != args.source_runtime_manifest_sha256
    ):
        raise RuntimeError("feature runtime source manifest pin drifted")
    context_args = SimpleNamespace(
        source_checkpoint_root=args.source_checkpoint_root,
        source_packed_manifest_sha256=args.source_packed_manifest_sha256,
        source_runtime_root=args.source_runtime_root,
        source_runtime_manifest_sha256=args.source_runtime_manifest_sha256,
        destination=manifest["destination"],
        fused_qkv_a=(
            manifest.get("attention_projection_layout")
            == FUSED_QKV_A_N82_RUNTIME_LAYOUT
        ),
        dense_convolution=(
            manifest.get("dense_projection_layout")
            == FINAL_DENSE_CONVOLUTION_RUNTIME_LAYOUT
        ),
        source_metadata_only=args.source_metadata_only,
    )
    context = _build_context(context_args, manifest["pack_code_hash"])
    expectation = build_load_expectation(manifest)
    verified = verify_feature_runtime_packed_checkpoint(
        args.runtime_root,
        expectation,
        context.layout,
        context.source_runtime_checkpoint,
    )
    summary = {
        "artifact_kind": "greenfield_verified_feature_runtime_checkpoint",
        "file_count": len(verified.plans),
        "model_id": expectation.model_id,
        "padding_bytes": manifest["padding_bytes"],
        "plan_hash": expectation.plan_hash,
        "plan_id": expectation.plan_id,
        "routed_expert_layout": context.layout.routed_expert_layout,
        "attention_projection_layout": (
            context.layout.attention_projection_layout
        ),
        "dense_projection_layout": context.layout.dense_projection_layout,
        "runtime_file_bytes": manifest["runtime_file_bytes"],
        "runtime_layout_hash": expectation.runtime_layout_hash,
        "runtime_layout_manifest_sha256": (expectation.runtime_layout_manifest_sha256),
        "runtime_manifest_sha256": expectation.runtime_manifest_sha256,
        "runtime_payload_bytes": manifest["runtime_payload_bytes"],
        "schedule_hash": expectation.schedule_hash,
        "source_payload_bytes": manifest["source_payload_bytes"],
        "source_runtime_layout_hash": expectation.source_runtime_layout_hash,
        "source_runtime_layout_manifest_sha256": (
            expectation.source_runtime_layout_manifest_sha256
        ),
        "source_runtime_manifest_sha256": (expectation.source_runtime_manifest_sha256),
        "source_tensor_count": manifest["source_tensor_count"],
        "source_verification_mode": (
            "metadata_lineage" if args.source_metadata_only else "complete_payload"
        ),
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
