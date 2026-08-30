#!/usr/bin/env python3
"""Publish compile-only evidence with generation-zero, terminal-last semantics."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import stat
from hashlib import sha256
from pathlib import Path
from typing import Any

_SUCCESS_ORDER = (
    "artifact_kind",
    "run_tag",
    "accepted_code_pin",
    "vllm_pin",
    "harness_pin",
    "callback_certificate_sha256",
    "launcher_sha256",
    "network_validator_sha256",
    "greenfield_pin",
    "manifest_sha256",
    "remote_objects_generation",
    "remote_objects_sha256",
    "db_run_id",
    "numerical_claim",
    "performance_claim",
    "gate_d_claim",
    "remote_prefix",
)
_SUCCESS_KEYS = set(_SUCCESS_ORDER)


def _split_remote(remote: str) -> tuple[str, str]:
    prefix = (
        "gs://driftbench-dsv4-uc/oracles/greenfield/glm52/accepted_compile_only_hlo/"
    )
    suffix = remote.removeprefix(prefix)
    if not remote.startswith(prefix) or not re.fullmatch(r"[A-Za-z0-9_]+", suffix):
        raise SystemExit("compile-only remote prefix is invalid")
    bucket, object_prefix = remote[5:].split("/", 1)
    return bucket, object_prefix + "/"


def _crc32c(path: Path) -> str:
    import google_crc32c

    checksum = google_crc32c.Checksum()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return base64.b64encode(checksum.digest()).decode("ascii")


def validate_regular_tree(run_dir: Path) -> list[Path]:
    """Return every regular file after refusing links and special entries."""

    try:
        root_stat = run_dir.lstat()
        root = run_dir.resolve(strict=True)
    except OSError as exc:
        raise SystemExit("compile-only run root is unavailable") from exc
    if run_dir.is_symlink() or not stat.S_ISDIR(root_stat.st_mode):
        raise SystemExit("compile-only run root is not a real directory")
    files = []
    for path in sorted(run_dir.rglob("*")):
        try:
            path_stat = path.lstat()
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"unsafe compile-only tree entry: {path}") from exc
        if stat.S_ISLNK(path_stat.st_mode) or not (
            stat.S_ISDIR(path_stat.st_mode) or stat.S_ISREG(path_stat.st_mode)
        ):
            raise SystemExit(f"unsafe compile-only tree entry: {path}")
        if stat.S_ISREG(path_stat.st_mode):
            files.append(path)
    return files


def expected_payload(manifest: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    """Validate that the sealed manifest names every nonterminal local payload."""

    tree_files = validate_regular_tree(run_dir)
    records = manifest.get("archive_files_before_manifest")
    if not isinstance(records, list):
        raise SystemExit("compile-only manifest archive inventory is absent")
    expected = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "byte_count",
            "path",
            "sha256",
        }:
            raise SystemExit("compile-only manifest archive record is malformed")
        relative = record["path"]
        path = run_dir / relative
        if (
            not isinstance(relative, str)
            or relative in seen
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or path not in tree_files
            or path.stat().st_size != record["byte_count"]
            or sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise SystemExit(f"compile-only local payload drifted: {relative}")
        seen.add(relative)
        expected.append(record)
    manifest_path = run_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    expected.append(
        {
            "byte_count": len(manifest_bytes),
            "path": "manifest.json",
            "sha256": sha256(manifest_bytes).hexdigest(),
        }
    )
    allowed = seen | {"manifest.json", "SUCCESS", "remote_objects.json"}
    local = {path.relative_to(run_dir).as_posix() for path in tree_files}
    if local - allowed:
        raise SystemExit(
            f"unsealed compile-only local files exist: {sorted(local - allowed)}"
        )
    return expected


def validate_remote_names(
    expected: set[str], observed: set[str], *, terminal_allowed: bool
) -> None:
    """Reject partial, extra, or prematurely terminal remote inventories."""

    terminal_present = "SUCCESS" in observed
    if terminal_present != terminal_allowed or observed != (
        expected | ({"SUCCESS"} if terminal_allowed else set())
    ):
        raise SystemExit(
            "compile-only remote object set drifted: "
            f"{sorted(observed ^ (expected | ({'SUCCESS'} if terminal_allowed else set())))}"
        )


def validate_ledger_records(
    records: Any, payload: list[dict[str, Any]], run_dir: Path
) -> None:
    """Replay manifest bytes against every immutable remote-ledger record."""

    if not isinstance(records, list) or len(records) != len(payload):
        raise SystemExit("compile-only ledger does not match manifest inventory")
    for record, item in zip(records, payload, strict=True):
        path = run_dir / item["path"]
        if (
            not isinstance(record, dict)
            or set(record) != {"crc32c", "generation", "path", "sha256", "size"}
            or record["path"] != item["path"]
            or record["sha256"] != item["sha256"]
            or record["size"] != item["byte_count"]
            or record["crc32c"] != _crc32c(path)
            or not isinstance(record["generation"], str)
            or not record["generation"]
        ):
            raise SystemExit(f"compile-only ledger record drifted: {item['path']}")


def _observed_names(bucket: Any, prefix: str) -> set[str]:
    return {blob.name.removeprefix(prefix) for blob in bucket.list_blobs(prefix=prefix)}


def _bucket_for(remote: str, storage_bucket: Any | None) -> tuple[Any, str]:
    bucket_name, prefix = _split_remote(remote)
    if storage_bucket is None:
        from google.cloud import storage

        storage_bucket = storage.Client().bucket(bucket_name)
    elif storage_bucket.name != bucket_name:
        raise SystemExit("compile-only publication bucket identity drifted")
    return storage_bucket, prefix


def _success_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in fields:
            raise SystemExit("compile-only SUCCESS metadata is malformed")
        fields[key] = value
    return fields


def validate_success_fields(
    fields: dict[str, str],
    manifest: dict[str, Any],
    manifest_sha256: str,
    remote: str,
    ledger_generation: str,
    ledger_sha256: str,
) -> None:
    """Require the complete no-generation/no-claim terminal contract."""

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit("compile-only SUCCESS provenance is absent")
    expected = {
        "accepted_code_pin": manifest.get("source_code_pin"),
        "artifact_kind": manifest.get("artifact_kind"),
        "callback_certificate_sha256": provenance.get("callback_certificate_sha256"),
        "db_run_id": "None",
        "gate_d_claim": "false",
        "greenfield_pin": provenance.get("greenfield_pin"),
        "harness_pin": provenance.get("harness_pin"),
        "launcher_sha256": provenance.get("launcher_sha256"),
        "manifest_sha256": manifest_sha256,
        "network_validator_sha256": provenance.get("network_validator_sha256"),
        "numerical_claim": "false",
        "performance_claim": "false",
        "remote_objects_generation": ledger_generation,
        "remote_objects_sha256": ledger_sha256,
        "remote_prefix": remote,
        "run_tag": manifest.get("run_tag"),
        "vllm_pin": provenance.get("vllm_source_pin"),
    }
    if set(fields) != _SUCCESS_KEYS or fields != expected:
        raise SystemExit("compile-only SUCCESS contract drifted")


def canonical_success_bytes(fields: dict[str, str]) -> bytes:
    """Render the only accepted terminal file ordering and newline contract."""

    return "".join(f"{key}={fields[key]}\n" for key in _SUCCESS_ORDER).encode()


def publish_nonterminal(
    run_dir: Path, remote: str, *, storage_bucket: Any | None = None
) -> dict[str, Any]:

    validate_regular_tree(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    payload = expected_payload(manifest, run_dir)
    bucket, prefix = _bucket_for(remote, storage_bucket)
    records = []
    for item in payload:
        path = run_dir / item["path"]
        crc = _crc32c(path)
        blob = bucket.blob(prefix + item["path"])
        blob.upload_from_filename(str(path), if_generation_match=0, checksum="crc32c")
        generation = str(blob.generation or "")
        blob.reload(if_generation_match=int(generation))
        if (
            not generation
            or str(blob.generation) != generation
            or int(blob.size) != item["byte_count"]
            or blob.crc32c != crc
        ):
            raise SystemExit(f"remote payload identity drifted: {item['path']}")
        records.append(
            {
                "crc32c": crc,
                "generation": generation,
                "path": item["path"],
                "sha256": item["sha256"],
                "size": item["byte_count"],
            }
        )
    ledger = {"objects": records}
    ledger_path = run_dir / "remote_objects.json"
    raw = (json.dumps(ledger, indent=2, sort_keys=True) + "\n").encode()
    temporary = ledger_path.with_name(f".{ledger_path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(ledger_path)
    ledger_crc = _crc32c(ledger_path)
    ledger_blob = bucket.blob(prefix + "remote_objects.json")
    ledger_blob.upload_from_filename(
        str(ledger_path), if_generation_match=0, checksum="crc32c"
    )
    ledger_generation = str(ledger_blob.generation or "")
    ledger_blob.reload(if_generation_match=int(ledger_generation))
    if (
        not ledger_generation
        or int(ledger_blob.size) != len(raw)
        or ledger_blob.crc32c != ledger_crc
    ):
        raise SystemExit("remote compile-only ledger identity drifted")
    expected = {item["path"] for item in payload} | {"remote_objects.json"}
    validate_remote_names(
        expected, _observed_names(bucket, prefix), terminal_allowed=False
    )
    return {
        "generation": ledger_generation,
        "sha256": sha256(raw).hexdigest(),
    }


def publish_terminal(
    run_dir: Path, remote: str, *, storage_bucket: Any | None = None
) -> dict[str, Any]:

    validate_regular_tree(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    payload = expected_payload(manifest, run_dir)
    ledger_path = run_dir / "remote_objects.json"
    ledger = json.loads(ledger_path.read_text())
    records = ledger.get("objects", [])
    validate_ledger_records(records, payload, run_dir)
    bucket, prefix = _bucket_for(remote, storage_bucket)
    for record, item in zip(records, payload, strict=True):
        blob = bucket.blob(prefix + item["path"])
        blob.reload(if_generation_match=int(record["generation"]))
        if (
            str(blob.generation) != record["generation"]
            or int(blob.size) != item["byte_count"]
            or blob.crc32c != record["crc32c"]
        ):
            raise SystemExit(f"remote payload replay failed: {item['path']}")
    ledger_blob = bucket.blob(prefix + "remote_objects.json")
    success_path = run_dir / "SUCCESS"
    success_fields = _success_fields(success_path)
    ledger_sha = sha256(ledger_path.read_bytes()).hexdigest()
    validate_success_fields(
        success_fields,
        manifest,
        sha256((run_dir / "manifest.json").read_bytes()).hexdigest(),
        remote,
        success_fields.get("remote_objects_generation", ""),
        ledger_sha,
    )
    if success_path.read_bytes() != canonical_success_bytes(success_fields):
        raise SystemExit("compile-only SUCCESS bytes drifted")
    expected_ledger_generation = success_fields["remote_objects_generation"]
    if not expected_ledger_generation.isdigit():
        raise SystemExit("compile-only SUCCESS ledger identity drifted")
    ledger_blob.reload(if_generation_match=int(expected_ledger_generation))
    if (
        str(ledger_blob.generation) != expected_ledger_generation
        or int(ledger_blob.size) != ledger_path.stat().st_size
        or ledger_blob.crc32c != _crc32c(ledger_path)
    ):
        raise SystemExit("remote compile-only ledger replay failed")
    expected = {item["path"] for item in payload} | {"remote_objects.json"}
    validate_remote_names(
        expected, _observed_names(bucket, prefix), terminal_allowed=False
    )
    success_crc = _crc32c(success_path)
    success_blob = bucket.blob(prefix + "SUCCESS")
    success_blob.upload_from_filename(
        str(success_path), if_generation_match=0, checksum="crc32c"
    )
    generation = str(success_blob.generation or "")
    success_blob.reload(if_generation_match=int(generation))
    if (
        not generation
        or int(success_blob.size) != success_path.stat().st_size
        or success_blob.crc32c != success_crc
    ):
        raise SystemExit("remote compile-only SUCCESS identity drifted")
    validate_remote_names(
        expected, _observed_names(bucket, prefix), terminal_allowed=True
    )
    return {
        "generation": generation,
        "sha256": sha256(success_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("nonterminal", "terminal"))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--remote-prefix", required=True)
    args = parser.parse_args()
    function = publish_nonterminal if args.mode == "nonterminal" else publish_terminal
    print(json.dumps(function(args.run_dir, args.remote_prefix), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
