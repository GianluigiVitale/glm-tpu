"""Pure-NumPy safetensors reader with digest binding for sealed inputs.

The sealed Gate-D runtime carries only NumPy/JAX; ``safetensors`` and ``torch``
are not on its import path.  This module parses the safetensors container
(8-byte little-endian header length, JSON header, raw little-endian payload)
directly and exposes digest-verifying loaders for the artifacts a bounded
probe consumes: the sealed layer-0 DSA association input and a sealed legacy
prompt index cache.  FP8 E4M3FN payloads are returned as raw ``uint8`` bits
and BF16 payloads as raw ``uint16`` bits.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import struct
from typing import Any

import numpy as np

_DTYPES = {
    "F32": (np.dtype("<f4"), 4),
    "F16": (np.dtype("<f2"), 2),
    "BF16": (np.dtype("<u2"), 2),
    "I32": (np.dtype("<i4"), 4),
    "I64": (np.dtype("<i8"), 8),
    "U8": (np.dtype("u1"), 1),
    "I8": (np.dtype("i1"), 1),
    "U16": (np.dtype("<u2"), 2),
    "I16": (np.dtype("<i2"), 2),
    "U32": (np.dtype("<u4"), 4),
    "U64": (np.dtype("<u8"), 8),
    "F64": (np.dtype("<f8"), 8),
    "BOOL": (np.dtype("?"), 1),
    "F8_E4M3": (np.dtype("u1"), 1),
    "F8_E5M2": (np.dtype("u1"), 1),
}


class Snapshot:
    """A regular file opened once through a no-follow descriptor.

    All reads and the digest use the same descriptor, so a replacement of the
    path between hashing and reading cannot change what is parsed.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        metadata = os.fstat(self.fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(self.fd)
            raise ValueError(f"input file identity is unsafe: {path}")
        self.size = metadata.st_size
        self._identity = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)

    def check_identity(self) -> None:
        metadata = os.fstat(self.fd)
        if (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns) != self._identity:
            raise ValueError(f"input file changed while in use: {self.path}")

    def pread(self, length: int, offset: int) -> bytes:
        chunks = []
        remaining = length
        position = offset
        while remaining > 0:
            block = os.pread(self.fd, min(remaining, 64 << 20), position)
            if not block:
                raise ValueError(f"input file truncated: {self.path}")
            chunks.append(block)
            remaining -= len(block)
            position += len(block)
        return b"".join(chunks)

    def sha256(self) -> str:
        digest = sha256()
        position = 0
        while position < self.size:
            block = os.pread(self.fd, 64 << 20, position)
            if not block:
                break
            digest.update(block)
            position += len(block)
        self.check_identity()
        return digest.hexdigest()

    def read_all(self) -> bytes:
        raw = self.pread(self.size, 0)
        self.check_identity()
        return raw

    def close(self) -> None:
        os.close(self.fd)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _parse_header(snapshot: Snapshot) -> tuple[dict[str, Any], int]:
    raw = snapshot.pread(8, 0)
    (length,) = struct.unpack("<Q", raw)
    header = json.loads(snapshot.pread(length, 8).decode("utf-8"))
    return header, 8 + length


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    snapshot = Snapshot(path)
    try:
        return _parse_header(snapshot)
    finally:
        snapshot.close()


def read_tensors_snapshot(snapshot: Snapshot, names: list[str] | None = None) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return ``{name: array}`` (raw bits for FP8/BF16) and the metadata from one descriptor."""

    header, offset = _parse_header(snapshot)
    metadata = header.pop("__metadata__", {}) or {}
    wanted = names if names is not None else [key for key in header]
    arrays: dict[str, np.ndarray] = {}
    for name in wanted:
        entry = header[name]
        dtype, item = _DTYPES[entry["dtype"]]
        start, stop = entry["data_offsets"]
        count = int(np.prod(entry["shape"])) if entry["shape"] else 1
        if stop - start != count * item:
            raise ValueError(f"safetensors payload size drifted for {name}")
        payload = snapshot.pread(stop - start, offset + start)
        arrays[name] = np.frombuffer(payload, dtype=dtype).reshape(entry["shape"]).copy()
    snapshot.check_identity()
    return arrays, {str(k): str(v) for k, v in metadata.items()}


def read_tensors(path: Path, names: list[str] | None = None) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    snapshot = Snapshot(path)
    try:
        return read_tensors_snapshot(snapshot, names)
    finally:
        snapshot.close()


def load_layer0_dsa_association_input(artifact_dir: Path, *, expected_manifest_sha256: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Digest-bound load of the sealed layer-0 DSA association input (v2 artifact).

    Verifies the manifest's self-hash, the tensor file byte count and digest,
    and every array's shape/dtype/digest against the manifest records.
    """

    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    body = dict(manifest)
    recorded = body.pop("manifest_sha256")
    canonical = json.dumps(body, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    if sha256(canonical).hexdigest() != recorded or recorded != expected_manifest_sha256:
        raise ValueError("layer-0 DSA input manifest identity drifted")
    if manifest.get("artifact_kind") != "greenfield_layer0_dsa_association_input" or manifest.get("diagnostic_only") is not True:
        raise ValueError("layer-0 DSA input artifact kind drifted")
    snapshot = Snapshot(artifact_dir / manifest["file"]["filename"])
    try:
        if snapshot.size != manifest["file"]["byte_count"] or snapshot.sha256() != manifest["file"]["sha256"]:
            raise ValueError("layer-0 DSA input tensor file drifted")
        arrays, _ = read_tensors_snapshot(snapshot)
    finally:
        snapshot.close()
    expected = manifest["arrays"]
    if set(expected) != set(arrays):
        raise ValueError("layer-0 DSA input array set drifted")
    for name, record in expected.items():
        value = arrays[name]
        if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"] or array_sha256(value) != record["sha256"]:
            raise ValueError(f"layer-0 DSA input array drifted: {name}")
    return manifest, arrays


def load_legacy_prompt_index_cache(artifact_dir: Path, *, expected_manifest_sha256: str) -> tuple[dict[str, Any], np.ndarray]:
    """Digest-bound load of a sealed legacy prompt index cache (``prompt_index_cache.py`` format)."""

    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    body = dict(manifest)
    recorded = body.pop("manifest_sha256")
    canonical = json.dumps(body, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    if sha256(canonical).hexdigest() != recorded or recorded != expected_manifest_sha256:
        raise ValueError("prompt index-cache manifest identity drifted")
    layer_id = int(manifest.get("layer_id", 0))
    if manifest.get("artifact_kind") != f"glm52_legacy_layer{layer_id}_prompt_index_cache" or manifest.get("diagnostic_only") is not True:
        raise ValueError("prompt index-cache artifact kind drifted")
    record = manifest["tensor_file"]
    snapshot = Snapshot(artifact_dir / record["filename"])
    try:
        if snapshot.size != record["byte_count"] or snapshot.sha256() != record["sha256"]:
            raise ValueError("prompt index-cache tensor file drifted")
        arrays, metadata = read_tensors_snapshot(snapshot)
    finally:
        snapshot.close()
    if set(arrays) != {"live_block_table", "prompt_index_key_bfloat16_bits"} or metadata.get("artifact_kind") != manifest["artifact_kind"]:
        raise ValueError("prompt index-cache tensor keys/metadata drifted")
    bits = arrays["prompt_index_key_bfloat16_bits"]
    contract = manifest["numerical_contract"]
    if bits.shape != (contract["prompt_token_count"], contract["head_dim"]) or bits.dtype != np.dtype("<u2"):
        raise ValueError("prompt index-cache tensor contract drifted")
    if array_sha256(bits) != manifest["prompt_index_key_bfloat16_sha256"]:
        raise ValueError("prompt index-cache key checksum drifted")
    if arrays["live_block_table"].tolist() != manifest["source_layout"]["live_block_table"]:
        raise ValueError("prompt index-cache block table drifted")
    return manifest, bits.astype(np.uint16, copy=False)


def load_checkpoint_tensors(
    root: Path,
    names: dict[str, str],
    *,
    index_sha256: str,
    shard_digests: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Read named checkpoint tensors (FP8 as uint8 bits, BF16 as uint16 bits) via the index.

    Each shard is opened once; when ``shard_digests`` maps a shard filename to
    its expected SHA-256 the same descriptor is hashed before parsing.
    """

    index_path = Path(root) / "model.safetensors.index.json"
    if sha256_file(index_path) != index_sha256:
        raise ValueError("checkpoint index identity drifted")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for key, name in names.items():
        by_shard.setdefault(weight_map[name], []).append((key, name))
    tensors: dict[str, np.ndarray] = {}
    for shard, items in by_shard.items():
        snapshot = Snapshot(Path(root) / shard)
        try:
            if shard_digests is not None:
                expected = shard_digests.get(shard)
                if expected is None or snapshot.sha256() != expected:
                    raise ValueError(f"checkpoint shard digest drifted: {shard}")
            arrays, _ = read_tensors_snapshot(snapshot, [name for _, name in items])
        finally:
            snapshot.close()
        for key, name in items:
            tensors[key] = arrays[name]
    return tensors
