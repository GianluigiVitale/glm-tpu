"""CPU-only selected-source originals from the retained eight-host tmpfs shards.

The WS32 cloud runtime was deleted. Nothing here reads/recreates it, initializes
devices, or verifies a full checkpoint file. The protected parent calls prepare
only AFTER workers finish, supplying the existing approved single-host SSH
helper. Exactly 256 metadata-derived tiles (99,639,040 bytes) are captured once;
load_reader independently rebinds their retained capsules without any network.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from math import prod
import os
from pathlib import Path
import re
import resource
import shlex
import socket
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import ml_dtypes  # Registers NumPy's bfloat16 dtype; no backend initialization.
import numpy as np

from glm_tpu.greenfield.benchmarking.numpy_safetensors import Snapshot
from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import _validate_finite_chunk
from scripts.greenfield import ws32_history_materializer_evidence as evidence
from scripts.greenfield import ws32_history_materializers as capture
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.prefill_window_evidence import same_json
from scripts.greenfield.ws32_dense_frontier_evidence import read_npz
from scripts.greenfield.ws32_history_preflight import RUN_ROOT, _plain_path
from scripts.greenfield import ws32_history_storage as storage

SCHEMA = "ws32_history_selected_source_originals_v1"
ROOT_LIMIT = 104 << 20
MANIFEST_LIMIT = 1 << 20
WIRE_METADATA_LIMIT = 1 << 20
NOISE_LIMIT = 64 << 10
HEADER_LIMIT = 2 << 20
FETCH_TIMEOUT = 180
MARKER = "WS32_HISTORY_SOURCE_V1 "
TMP_ROOT = Path("/dev/shm")
REPO = Path("/home/gianl/glm-tpu-topology-rewrite")
PYTHON = "/home/gianl/vllm-env/bin/python"
_HEX = re.compile(r"[0-9a-f]{64}")
_PIN = re.compile(r"[0-9a-f]{40}")
_BOOT = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
_NP_DTYPES = {"U8": np.dtype("u1"), "F32": np.dtype("<f4"), "BF16": np.dtype("<u2")}


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


@dataclass(frozen=True)
class Inventory:
    metadata: Any
    bindings: Any
    captures: tuple[dict, ...]
    slots: tuple[dict[int, int], ...]
    requests: tuple[dict, ...]


def _inventory(repo: Path) -> Inventory:
    """Re-derive every request before opening any retained tensor payload."""
    from scripts.greenfield.ws32_canonical_prefill_compile import read_metadata
    from scripts.greenfield.ws32_prefill_budget_campaign import topology_bindings

    bindings = evidence.checkpoint_bindings(repo)
    metadata = read_metadata(repo)
    physical, captures = topology_bindings()
    by_device = {device: slot for slot, device in enumerate(physical.flattened_device_ids)}
    slots = tuple({device: by_device[device] for device in host["local_device_ids"]} for host in captures)
    ranks = {slot: rank for rank, owners in enumerate(slots) for slot in owners.values()}
    if (len(captures) != 8 or len(ranks) != 32 or set(ranks) != set(range(32))
            or any(len(owners) != 4 for owners in slots)):
        raise ValueError("history source captured owner inventory differs")
    plans = {plan.device_slot: plan for plan in metadata.plans}
    if set(plans) != set(range(32)):
        raise ValueError("history source metadata plan owners differ")
    tensors = {slot: {tensor.name: tensor for tensor in plan.tensors} for slot, plan in plans.items()}
    requests = []
    for layer in protocol.PRODUCERS:
        for field, shape, dtype, axes in capture.RAW:
            if field in ("wk_bits_local", "wk_scale_local"):
                continue  # WK's independent originals already preserve these inputs.
            seen = set()
            for slot in range(32):
                index = tuple(part.indices(size) for part, size in
                              zip(capture._index(shape, axes, slot), shape, strict=True))
                if index in seen:
                    continue
                seen.add(index)
                name = capture.source_names(layer)[field]
                tensor, plan = tensors[slot][name], plans[slot]
                local_shape = tuple(len(range(*part)) for part in index)
                raw_dtype = {"uint8": "U8", "float32": "F32", "bfloat16": "BF16"}[dtype]
                row = metadata.records_by_slot[slot]
                if (tensor.dtype != raw_dtype or tensor.local_shape != local_shape
                        or tensor.global_shape != shape or tensor.partition_spec != axes
                        or tensor.byte_count != prod(local_shape) * _NP_DTYPES[raw_dtype].itemsize
                        or not 8 <= len(plan.header) <= HEADER_LIMIT
                        or sha256(plan.header).hexdigest() != row["header_sha256"]
                        or plan.file_bytes != row["file_bytes"] or plan.filename != row["filename"]
                        or Path(plan.filename).name != plan.filename
                        or not 0 <= tensor.data_offset_start < tensor.data_offset_end <= plan.payload_bytes):
                    raise ValueError("history source selected layout/header differs")
                digest = bindings.owners[slot]["selected"][name]
                if not isinstance(digest, str) or not _HEX.fullmatch(digest):
                    raise ValueError("history source selected SHA differs")
                requests.append(dict(index=len(requests), key=f"tile{len(requests):03d}",
                    rank=ranks[slot], slot=slot, tensor=name, dtype=raw_dtype, shape=list(local_shape),
                    offset=len(plan.header) + tensor.data_offset_start, bytes=tensor.byte_count, sha256=digest))
    if (len(requests) != evidence.SOURCE_READ_TILES
            or sum(row["bytes"] for row in requests) != evidence.SOURCE_READ_BYTES
            or evidence.SOURCE_READ_BYTES > evidence.SOURCE_READ_LIMIT
            or len({(row["slot"], row["tensor"]) for row in requests}) != len(requests)):
        raise ValueError("history source fixed request count/byte budget differs")
    return Inventory(metadata, bindings, tuple(captures), slots, tuple(requests))


def _identity(inventory: Inventory, records: Sequence[Mapping]) -> dict:
    from scripts.greenfield.ws32_prefill_budget_campaign import FLEET_SHA

    if not isinstance(records, (list, tuple)) or len(records) != 8:
        raise ValueError("history source requires all eight completed worker records")
    tag, pin = records[0].get("tag"), records[0].get("code_hash")
    if not isinstance(tag, str) or not protocol.is_tag(tag) or not isinstance(pin, str) or not _PIN.fullmatch(pin):
        raise ValueError("history source tag/code pin differs")
    hosts = []
    for rank, (record, host, slots) in enumerate(zip(records, inventory.captures, inventory.slots, strict=True)):
        expected = dict(tag=tag, code_hash=pin, protocol=protocol.PROTOCOL, launch_rank=rank,
            status="DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION",
            hostname=host["hostname"], jax_process_index=host["jax_process_index"], topology_fleet_sha256=FLEET_SHA)
        same_json({key: record.get(key) for key in expected}, expected, "history source worker identity")
        if not isinstance(record.get("boot_id"), str) or not _BOOT.fullmatch(record["boot_id"]):
            raise ValueError("history source worker boot identity differs")
        evidence._record_owners(record, slots, inventory.bindings)
        hosts.append(dict(rank=rank, hostname=host["hostname"], process_index=host["jax_process_index"],
                          boot_id=record["boot_id"], slots=sorted(slots.values())))
    return dict(schema=SCHEMA, tag=tag, code_hash=pin, protocol=protocol.PROTOCOL,
        context=dict(inventory.bindings.context), hosts=hosts,
        requests_sha256=sha256(_json(inventory.requests)).hexdigest(),
        tile_count=len(inventory.requests), raw_bytes=sum(row["bytes"] for row in inventory.requests),
        provenance="RETAINED_HOST_TMPFS_SELECTED_BYTES_NOT_GCS_GENERATION_OR_FULL_FILE_VERIFICATION")


def _requests(inventory: Inventory, rank: int) -> tuple[dict, ...]:
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("history source launch rank differs")
    return tuple(row for row in inventory.requests if row["rank"] == rank)


def _capsule_cap(rows: Sequence[Mapping]) -> int:
    return sum(row["bytes"] for row in rows) + 1024 * (len(rows) + 1)


def _stat(snapshot: Snapshot) -> dict:
    snapshot.check_identity()
    _plain_path(snapshot.path)
    named, opened = snapshot.path.lstat(), os.fstat(snapshot.fd)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink", "st_mode")
    result = {key: int(getattr(opened, key)) for key in fields}
    if result != {key: int(getattr(named, key)) for key in fields}:
        raise ValueError("history source named file changed during selected read")
    return result


def _source_fixed(inventory: Inventory, slot: int) -> dict:
    plan = next(plan for plan in inventory.metadata.plans if plan.device_slot == slot)
    row = inventory.metadata.records_by_slot[slot]
    return dict(slot=slot, filename=plan.filename, file_bytes=plan.file_bytes,
        header_bytes=len(plan.header), header_sha256=row["header_sha256"],
        expected_full_file_sha256_not_verified=row["sha256"])


def _capture(inventory: Inventory, identity: Mapping, rank: int) -> dict:
    """Selected pread only; invoked remotely after explicit CPU CLI guards."""
    rows = _requests(inventory, rank)
    host = identity["hosts"][rank]
    if socket.gethostname() != host["hostname"] or Path("/proc/sys/kernel/random/boot_id").read_text().strip() != host["boot_id"]:
        raise ValueError("history source live hostname/boot differs")
    payloads, sources = {}, []
    for slot in sorted({row["slot"] for row in rows}):
        plan = next(plan for plan in inventory.metadata.plans if plan.device_slot == slot)
        path = inventory.metadata.root / plan.filename
        _plain_path(path)
        snapshot = Snapshot(path)
        try:
            before = _stat(snapshot)
            if snapshot.size != plan.file_bytes or snapshot.pread(len(plan.header), 0) != plan.header:
                raise ValueError("history source retained file size/header differs")
            for row in rows:
                if row["slot"] != slot:
                    continue
                raw = snapshot.pread(row["bytes"], row["offset"])
                _stat(snapshot)
                _validate_finite_chunk(raw, row["dtype"])
                if sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError("history source selected tensor SHA differs")
                payloads[row["key"]] = raw
            after = _stat(snapshot)
            if after != before:
                raise ValueError("history source retained file changed during capture")
            sources.append(dict(**_source_fixed(inventory, slot), stat_before=before, stat_after=after))
        finally:
            snapshot.close()
    raw = b"".join(payloads[row["key"]] for row in rows)
    report = dict(schema=SCHEMA, identity=identity, rank=rank, requests=list(rows), sources=sources,
                  payload_base64=base64.b64encode(raw).decode("ascii"))
    if len(_json(report)) > _wire_cap(rows) - NOISE_LIMIT - len(MARKER):
        raise ValueError("history source reply metadata exceeded budget")
    return report


def _validate_sources(inventory: Inventory, rank: int, sources: Any) -> None:
    slots = sorted({row["slot"] for row in _requests(inventory, rank)})
    if not isinstance(sources, list) or len(sources) != len(slots):
        raise ValueError("history source original file inventory differs")
    for slot, row in zip(slots, sources, strict=True):
        expected = _source_fixed(inventory, slot)
        if not isinstance(row, dict) or set(row) != {*expected, "stat_before", "stat_after"}:
            raise ValueError("history source original file schema differs")
        same_json({key: row[key] for key in expected}, expected, "history source original file identity")
        before = row["stat_before"]
        if (not isinstance(before, dict) or set(before) != {"st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink", "st_mode"}
                or any(type(value) is not int or value < 0 for value in before.values())
                or before["st_size"] != expected["file_bytes"] or before["st_nlink"] != 1
                or before["st_mode"] & 0o170000 != 0o100000):
            raise ValueError("history source original file stat differs")
        same_json(row["stat_after"], before, "history source file changed during read")


def _decode(inventory: Inventory, identity: Mapping, rank: int, report: Mapping) -> dict[str, np.ndarray]:
    rows = _requests(inventory, rank)
    if not isinstance(report, dict) or set(report) != {"schema", "identity", "rank", "requests", "sources", "payload_base64"}:
        raise ValueError("history source reply schema differs")
    same_json({key: report[key] for key in ("schema", "identity", "rank", "requests")},
              dict(schema=SCHEMA, identity=identity, rank=rank, requests=list(rows)), "history source reply identity")
    _validate_sources(inventory, rank, report["sources"])
    size = sum(row["bytes"] for row in rows)
    encoded = report["payload_base64"]
    if not isinstance(encoded, str) or len(encoded) != 4 * ((size + 2) // 3):
        raise ValueError("history source encoded payload budget differs")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != size:
        raise ValueError("history source decoded payload budget differs")
    arrays, offset = {}, 0
    for row in rows:
        tile = raw[offset:offset + row["bytes"]]
        offset += row["bytes"]
        _validate_finite_chunk(tile, row["dtype"])
        if sha256(tile).hexdigest() != row["sha256"]:
            raise ValueError("history source returned selected tensor SHA differs")
        arrays[row["key"]] = np.frombuffer(tile, dtype=_NP_DTYPES[row["dtype"]]).reshape(row["shape"])
    return arrays


def _wire_cap(rows: Sequence[Mapping]) -> int:
    return 4 * ((sum(row["bytes"] for row in rows) + 2) // 3) + WIRE_METADATA_LIMIT + NOISE_LIMIT + len(MARKER)


def _fetch(fetch: Callable, command: str, rank: int, maximum: int) -> bytes:
    """Bound the unchanged SSH helper's merged stdout before any file write.

    One ephemeral wire original lives in controller tmpfs, not the disk archive.
    The largest reply is <44 MiB; it is removed only with this owned tempdir.
    Retained validated source capsules remain even after later failures.
    """
    _plain_path(TMP_ROOT)
    with tempfile.TemporaryDirectory(prefix="ws32-history-source-", dir=TMP_ROOT) as directory:
        path = Path(directory) / "wire.txt"

        def limited():
            resource.setrlimit(resource.RLIMIT_FSIZE, (maximum, maximum))
            fetch(command, output=path, worker=str(rank), timeout=FETCH_TIMEOUT)

        try:
            # Shared WNOWAIT lifecycle reserves the child's PID until its owned
            # group is stopped, including timeout, error and interruption paths.
            storage.run_child(limited, timeout=FETCH_TIMEOUT + 15)
        except RuntimeError as error:
            raise ValueError("history bounded source SSH failed") from error
        _plain_path(path)
        snapshot = Snapshot(path)
        try:
            if not 0 < snapshot.size <= maximum:
                raise ValueError("history source SSH output exceeded budget")
            raw = snapshot.pread(snapshot.size, 0)
            _stat(snapshot)
            return raw
        finally:
            snapshot.close()


def _command(repo: Path, identity: Mapping, rank: int) -> str:
    token = base64.b64encode(_json(identity)).decode("ascii")
    return "cd " + shlex.quote(str(repo)) + " && JAX_PLATFORMS=cpu " + shlex.join([
        PYTHON, "-m", "scripts.greenfield.ws32_history_source_reader", "--serve-selected-source",
        "--rank", str(rank), "--identity-base64", token])


def _report(raw: bytes, maximum: int) -> dict:
    if len(raw) > maximum:
        raise ValueError("history source wire exceeded budget")
    lines = raw.splitlines(keepends=True)
    replies = [line[len(MARKER):] for line in lines if line.startswith(MARKER.encode())]
    noise = sum(len(line) for line in lines if not line.startswith(MARKER.encode()))
    if len(replies) != 1 or noise > NOISE_LIMIT:
        raise ValueError("history source missing/duplicate reply or excessive SSH noise")
    return json.loads(replies[0])


def _root(root: Path, *, present: bool) -> None:
    _plain_path(root)
    if (".." in root.parts or not protocol.is_tag(root.parent.name)
            or root != RUN_ROOT / root.parent.name / "source_tiles" or not root.parent.is_dir()):
        raise ValueError("history source capsule directory differs")
    if present:
        if not root.is_dir():
            raise ValueError("history source capsule directory missing")
    elif root.exists():
        raise FileExistsError(root)


class Reader:
    """Authenticated immutable in-memory tiles; never a network fallback."""

    def __init__(self, arrays: Mapping[tuple[int, str], np.ndarray], receipt: Mapping):
        self._arrays, self.receipt = dict(arrays), dict(receipt)

    def __call__(self, slot: int, tensor_name: str) -> np.ndarray:
        if type(slot) is not int or type(tensor_name) is not str or (slot, tensor_name) not in self._arrays:
            raise ValueError("history source unregistered tile request")
        return self._arrays[slot, tensor_name]


def prepare_reader(*, root: Path, repo: Path, records: Sequence[Mapping], fetch: Callable) -> Reader:
    if not callable(fetch):
        raise ValueError("history source requires the protected parent's SSH helper")
    _root(root, present=False)
    inventory = _inventory(repo)
    identity = _identity(inventory, records)
    if root.parent.name != identity["tag"] or repo != REPO:
        raise ValueError("history source capsule tag/repository differs")
    if sum(_capsule_cap(_requests(inventory, rank)) for rank in range(8)) + MANIFEST_LIMIT > ROOT_LIMIT:
        raise ValueError("history source original reservation exceeds aggregate budget")
    root.mkdir()
    capture._sync(root.parent)
    originals = []
    for rank in range(8):
        rows = _requests(inventory, rank)
        maximum = _wire_cap(rows)
        report = _report(_fetch(fetch, _command(repo, identity, rank), rank, maximum), maximum)
        arrays = _decode(inventory, identity, rank, report)
        path = root / f"rank{rank}.npz"
        stream = BytesIO()
        np.savez(capture._CappedWriter(stream, _capsule_cap(rows)), **arrays)
        capsule = stream.getvalue()
        storage.write_bytes(path, capsule)
        capture._sync(root)
        originals.append(dict(rank=rank, path=path.name, bytes=path.stat().st_size,
            npz_sha256=sha256(capsule).hexdigest(), raw_array_bytes=sum(row["bytes"] for row in rows),
            sources=report["sources"]))
    manifest = dict(identity=identity, requests=list(inventory.requests), originals=originals,
                    numerical_promotion=False, performance_claim=False)
    raw = _json(manifest)
    if len(raw) > MANIFEST_LIMIT:
        raise ValueError("history source manifest exceeded budget")
    storage.write_bytes(root / "manifest.json", raw)
    capture._sync(root)
    return _load(root, inventory, identity)


def _load(root: Path, inventory: Inventory, identity: Mapping) -> Reader:
    _root(root, present=True)
    if root.parent.name != identity["tag"]:
        raise ValueError("history source retained tag differs")
    paths = tuple(root.iterdir())
    if {path.name for path in paths} != {"manifest.json", *(f"rank{rank}.npz" for rank in range(8))}:
        raise ValueError("history source capsule inventory differs")
    for path in paths:
        _plain_path(path)
        if not path.is_file():
            raise ValueError("history source capsule must be a regular file")
    if sum(path.stat().st_size for path in paths) > ROOT_LIMIT:
        raise ValueError("history source aggregate original budget exceeded")
    path = root / "manifest.json"
    snapshot = Snapshot(path)
    try:
        if not 0 < snapshot.size <= MANIFEST_LIMIT:
            raise ValueError("history source manifest budget differs")
        raw = snapshot.pread(snapshot.size, 0)
        _stat(snapshot)
    finally:
        snapshot.close()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or set(manifest) != {"identity", "requests", "originals", "numerical_promotion", "performance_claim"}:
        raise ValueError("history source manifest schema differs")
    same_json({key: manifest[key] for key in ("identity", "requests", "numerical_promotion", "performance_claim")},
              dict(identity=identity, requests=list(inventory.requests), numerical_promotion=False,
                   performance_claim=False), "history source retained metadata binding")
    originals = manifest["originals"]
    if not isinstance(originals, list) or len(originals) != 8:
        raise ValueError("history source retained original count differs")
    result = {}
    for rank, original in enumerate(originals):
        rows = _requests(inventory, rank)
        if (not isinstance(original, dict) or set(original) != {"rank", "path", "bytes", "npz_sha256", "raw_array_bytes", "sources"}
                or type(original["rank"]) is not int or original["rank"] != rank
                or original["path"] != f"rank{rank}.npz"
                or type(original["raw_array_bytes"]) is not int
                or original["raw_array_bytes"] != sum(row["bytes"] for row in rows)
                or not isinstance(original["npz_sha256"], str) or not _HEX.fullmatch(original["npz_sha256"])):
            raise ValueError("history source retained capsule descriptor differs")
        _validate_sources(inventory, rank, original["sources"])
        path = root / original["path"]
        snapshot = Snapshot(path)
        try:
            before = _stat(snapshot)
            arrays = read_npz(path, original, limit=_capsule_cap(rows))
            if _stat(snapshot) != before:
                raise ValueError("history source capsule changed during replay")
        finally:
            snapshot.close()
        if set(arrays) != {row["key"] for row in rows}:
            raise ValueError("history source retained tile inventory differs")
        for row in rows:
            value = arrays[row["key"]]
            if value.dtype != _NP_DTYPES[row["dtype"]] or list(value.shape) != row["shape"]:
                raise ValueError("history source retained tile dtype/shape differs")
            raw_tile = value.tobytes()
            _validate_finite_chunk(raw_tile, row["dtype"])
            if sha256(raw_tile).hexdigest() != row["sha256"]:
                raise ValueError("history source retained selected tile SHA differs")
            # Immutable byte backing also prevents a caller re-enabling writes.
            dtype = np.dtype(ml_dtypes.bfloat16) if row["dtype"] == "BF16" else value.dtype
            result[row["slot"], row["tensor"]] = np.frombuffer(raw_tile, dtype=dtype).reshape(row["shape"])
    receipt = dict(identity=identity, manifest_sha256=sha256(raw).hexdigest(),
                   retained_bytes=sum(path.stat().st_size for path in paths), files=9,
                   numerical_promotion=False, performance_claim=False)
    return Reader(result, receipt)


def load_reader(*, root: Path, repo: Path, records: Sequence[Mapping]) -> Reader:
    inventory = _inventory(repo)
    return _load(root, inventory, _identity(inventory, records))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve-selected-source", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--identity-base64")
    args = parser.parse_args(argv)
    if (not args.serve_selected_source or os.environ.get("JAX_PLATFORMS") != "cpu"
            or type(args.rank) is not int or not 0 <= args.rank < 8
            or not isinstance(args.identity_base64, str) or len(args.identity_base64) > 32 << 10):
        raise ValueError("history source CLI is default-off and CPU-only")
    identity = json.loads(base64.b64decode(args.identity_base64, validate=True))
    if (not isinstance(identity, dict) or not isinstance(identity.get("code_hash"), str)
            or not _PIN.fullmatch(identity["code_hash"]) or not protocol.is_tag(identity.get("tag", ""))
            or identity.get("protocol") != protocol.PROTOCOL):
        raise ValueError("history source CLI identity differs")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != identity["code_hash"]:
        raise ValueError("history source remote code pin differs")
    inventory = _inventory(REPO)
    expected = dict(schema=SCHEMA, protocol=protocol.PROTOCOL, context=dict(inventory.bindings.context),
        requests_sha256=sha256(_json(inventory.requests)).hexdigest(), tile_count=len(inventory.requests),
        raw_bytes=sum(row["bytes"] for row in inventory.requests),
        provenance="RETAINED_HOST_TMPFS_SELECTED_BYTES_NOT_GCS_GENERATION_OR_FULL_FILE_VERIFICATION")
    if set(identity) != {*expected, "tag", "code_hash", "hosts"}:
        raise ValueError("history source CLI identity schema differs")
    same_json({key: identity[key] for key in expected}, expected, "history source remote metadata binding")
    hosts = identity["hosts"]
    if not isinstance(hosts, list) or len(hosts) != 8:
        raise ValueError("history source remote owner count differs")
    for rank, host in enumerate(hosts):
        prior = inventory.captures[rank]
        expected_host = dict(rank=rank, hostname=prior["hostname"], process_index=prior["jax_process_index"],
                             slots=sorted(inventory.slots[rank].values()))
        if (not isinstance(host, dict) or set(host) != {*expected_host, "boot_id"}
                or not isinstance(host["boot_id"], str) or not _BOOT.fullmatch(host["boot_id"])):
            raise ValueError("history source remote owner schema differs")
        same_json({key: host[key] for key in expected_host}, expected_host, "history source remote owner binding")
    print(MARKER + _json(_capture(inventory, identity, args.rank)).decode().strip(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
