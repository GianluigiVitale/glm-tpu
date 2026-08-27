#!/usr/bin/env python3
"""Pack a small, complete-owner probe derived from a full layout manifest."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.checkpoint import (  # noqa: E402
    build_destination_probe_plans,
    stream_pack_group,
)
from glm_tpu.greenfield.partitioning import inspect_layout_manifest  # noqa: E402


SELECTION_KIND = "greenfield_checkpoint_probe_selection"
ARTIFACT_KIND = "greenfield_checkpoint_pack_probe"


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_hashes(
    *,
    layout: Mapping[str, Any],
    source_root: Path,
    source_names: tuple[str, ...],
) -> dict[str, str]:
    placements = {
        item["source"]["name"]: item for item in layout["placements"]
    }
    files = {item["filename"]: item for item in layout["source"]["files"]}
    result = {}
    handles: dict[str, Any] = {}
    try:
        for name in source_names:
            source = placements[name]["source"]
            filename = source["filename"]
            record = files[filename]
            path = source_root / filename
            if path.stat().st_size != record["file_bytes"]:
                raise RuntimeError(f"source size drifted for {filename}")
            if filename not in handles:
                handles[filename] = path.open("rb")
            handle = handles[filename]
            offset = record["header_bytes"] + source["data_offsets"][0]
            handle.seek(offset)
            remaining = source["byte_count"]
            digest = sha256()
            while remaining:
                chunk = handle.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError(f"source tensor is truncated: {name}")
                digest.update(chunk)
                remaining -= len(chunk)
            result[name] = digest.hexdigest()
    finally:
        for handle in handles.values():
            handle.close()
    return result


def pack_probe(
    *,
    layout_path: Path,
    selection_path: Path,
    source_root: Path,
    output_dir: Path,
    code_hash: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"append-only probe output exists: {output_dir}")
    layout = inspect_layout_manifest(layout_path)
    selection = json.loads(selection_path.read_text())
    if selection.get("artifact_kind") != SELECTION_KIND:
        raise RuntimeError("wrong checkpoint probe selection kind")
    expected = {
        "layout_manifest_sha256": layout["manifest_sha256"],
        "plan_group_hash": layout["plan_group_hash"],
        "plan_id": layout["plan_id"],
        "source_inventory_sha256": layout["source"]["inventory_sha256"],
        "source_revision": layout["source"]["revision"],
        "topology_hash": layout["topology_hash"],
    }
    for key, value in expected.items():
        if selection.get(key) != value:
            raise RuntimeError(f"checkpoint probe selection drifted at {key}")
    source_names = tuple(selection.get("source_names", ()))
    plans = build_destination_probe_plans(
        layout,
        source_names=source_names,
    )
    if len(plans) != layout["plan_manifest"]["execution_plan"][
        "local_parallel_size"
    ]:
        raise RuntimeError("checkpoint probe does not cover the full local group")
    source_hashes = _source_hashes(
        layout=layout,
        source_root=source_root,
        source_names=source_names,
    )

    output_dir.mkdir(parents=True)
    selection_output = output_dir / "selection.json"
    _write_json_once(selection_output, selection)
    outputs = {}
    handles = []
    try:
        for plan in plans:
            path = output_dir / "payload" / plan.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("xb")
            handles.append(handle)
            outputs[plan.filename] = handle
        streamed = stream_pack_group(
            layout=layout,
            plans=plans,
            source_root=source_root,
            outputs=outputs,
            expected_source_sha256=source_hashes,
        )
    finally:
        for handle in handles:
            handle.close()
    by_filename = {item.filename: item for item in streamed}
    files = []
    for plan in plans:
        path = output_dir / "payload" / plan.filename
        evidence = by_filename[plan.filename]
        if path.stat().st_size != evidence.file_bytes:
            raise RuntimeError(f"checkpoint probe file size drifted: {plan.filename}")
        digest = _sha256_file(path)
        if digest != evidence.sha256:
            raise RuntimeError(f"checkpoint probe file hash drifted: {plan.filename}")
        files.append(
            {
                "device_id": plan.device_id,
                "device_slot": plan.device_slot,
                "file_bytes": evidence.file_bytes,
                "filename": f"payload/{plan.filename}",
                "header_bytes": len(plan.header),
                "payload_bytes": plan.payload_bytes,
                "sha256": digest,
                "stage_id": plan.stage_id,
                "tensor_names": [item.name for item in plan.tensors],
            }
        )
    placements = {
        item["source"]["name"]: item for item in layout["placements"]
    }
    sources = [
        {
            "byte_count": placements[name]["source"]["byte_count"],
            "destinations": [
                {
                    key: destination.get(key)
                    for key in (
                        "axis",
                        "axis_end_exclusive",
                        "axis_start",
                        "byte_count",
                        "device_slot",
                        "shape",
                    )
                }
                for destination in placements[name]["destinations"]
            ],
            "layout": placements[name]["layout"],
            "name": name,
            "sha256": source_hashes[name],
            "source_filename": placements[name]["source"]["filename"],
            "value_class": placements[name]["value_class"],
        }
        for name in source_names
    ]
    value: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "code_hash": code_hash,
        "file_count": len(files),
        "files": files,
        "layout_file_sha256": _sha256_file(layout_path),
        "layout_manifest_sha256": layout["manifest_sha256"],
        "packed_payload_bytes": sum(item["payload_bytes"] for item in files),
        "plan_group_hash": layout["plan_group_hash"],
        "plan_id": layout["plan_id"],
        "selection_file_sha256": _sha256_file(selection_output),
        "selected_source_bytes": sum(item["byte_count"] for item in sources),
        "source_inventory_sha256": layout["source"]["inventory_sha256"],
        "source_revision": layout["source"]["revision"],
        "sources": sources,
        "stage_id": plans[0].stage_id,
        "topology_hash": layout["topology_hash"],
    }
    value["manifest_sha256"] = sha256(_canonical(value)).hexdigest()
    manifest_path = output_dir / "manifest.json"
    _write_json_once(manifest_path, value)
    evidence_path = output_dir / "evidence.sha256"
    evidence_path.write_text(
        "".join(
            f"{_sha256_file(path)}  {path.relative_to(output_dir)}\n"
            for path in (
                selection_output,
                manifest_path,
                *(output_dir / item["filename"] for item in files),
            )
        )
    )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != Path("/home/gianl/glm-tpu-topology-rewrite"):
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    if subprocess.check_output(
        ["git", "-C", str(REPO), "branch", "--show-current"], text=True
    ).strip() != "rewrite/topology-first-decode":
        raise RuntimeError("checkpoint probe requires the isolated rewrite branch")
    code_hash = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if code_hash != args.expected_code_hash:
        raise RuntimeError("checkpoint probe code hash is stale")
    if subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("checkpoint probe requires a clean worktree")
    value = pack_probe(
        layout_path=args.layout_manifest,
        selection_path=args.selection,
        source_root=args.source_root,
        output_dir=args.output_dir,
        code_hash=code_hash,
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
