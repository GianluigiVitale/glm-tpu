"""Bounded append-only upload of first-window originals in the existing EXIT trap."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

LABELS = ("wide_initial", "narrow_initial", "wide_final", "narrow_32", "narrow_64", "narrow_96", "narrow_128")
FILES = {f"{label}.{ext}" for label in LABELS for ext in ("json", "npz")} | {"runner.json", "comparison.json"}
LIMIT = 128 * 1024**2


def originals(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("first-window original root must be a real directory")
    paths = sorted(root.iterdir())
    def allowed(name):
        if name in FILES or name in {f"{label}.npz.pending" for label in LABELS}:
            return True
        return any(re.fullmatch(re.escape(f".{n}.tmp.") + r"[0-9]+", name)
                   for n in FILES if n.endswith(".json"))
    if any(p.is_symlink() or not p.is_file() or not allowed(p.name) for p in paths):
        raise ValueError("first-window unknown/nonregular original")
    if sum(p.stat().st_size for p in paths) > LIMIT:
        raise ValueError("first-window originals exceed rank budget")
    return paths


def publish(root: Path, remote: str, *, client: Any) -> list[dict[str, Any]]:
    from google.api_core.exceptions import PreconditionFailed

    prefix = "gs://driftbench-dsv4-uc/results/"
    if not remote.startswith(prefix):
        raise ValueError("first-window publication requires approved result bucket")
    tag = remote[len(prefix):].rstrip("/")
    if (not re.fullmatch(r"greenfield_ws32_short_decoder_8k_numerical_[a-z0-9_]+_[0-9]{8}T[0-9]+Z", tag)
            or not re.fullmatch(r"first_window\.rank[0-7]", root.name)
            or root.parent.name != tag or root.parent.parent != Path("/home/gianl/glm-run")):
        raise ValueError("first-window publication tag/rank/root differs")
    paths = originals(root)
    bucket = client.bucket("driftbench-dsv4-uc")
    receipts = []
    observed_bytes = 0
    for path in paths:
        raw = path.read_bytes()
        observed_bytes += len(raw)
        if observed_bytes > LIMIT:
            raise ValueError("first-window original grew beyond budget")
        name = f"results/{tag}/diagnostic_local/{tag}/{root.name}/{path.name}"
        blob = bucket.blob(name)
        try:
            blob.upload_from_string(raw, if_generation_match=0, checksum="crc32c")
        except PreconditionFailed:
            blob.reload()
        generation = int(blob.generation)
        observed = bucket.blob(name, generation=generation).download_as_bytes(
            if_generation_match=generation, checksum="crc32c")
        if observed != raw:
            raise ValueError(f"first-window archived original differs: {path.name}")
        receipts.append(dict(name=name, generation=str(generation), bytes=len(raw),
                             crc32c=blob.crc32c, sha256=sha256(raw).hexdigest()))
    return receipts


def main() -> None:
    from google.cloud import storage
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.remote, client=storage.Client()), sort_keys=True))


if __name__ == "__main__":
    main()
