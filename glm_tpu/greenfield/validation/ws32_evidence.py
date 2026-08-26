"""Fail-closed materialization of protected WS32 fleet evidence.

The eight workers upload one copy of every graph even when all ranks produced
identical bytes.  This module validates the remote generation/CRC against each
runner's SHA-256 record, then hard-links an already verified local copy instead
of downloading duplicate multi-gigabyte HLO text.
"""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import google_crc32c
from google.cloud import storage


BASE_GRAPHS = ("prefill", "observer", "decode", "cache_probe")
EXACT_DSA_GRAPHS = ("exact_materialize", "exact_promote") + BASE_GRAPHS
# Compatibility alias for callers inspecting the default-off graph contract.
GRAPHS = BASE_GRAPHS
HLO_FORMS = (
    ("stablehlo.mlir", "stablehlo_sha256"),
    ("optimized_hlo.txt", "optimized_hlo_sha256"),
)
RANKS = tuple(range(8))


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _crc32c_file(path: Path) -> str:
    digest = google_crc32c.Checksum()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _runner_suffixes(*, numerical: bool) -> tuple[str, ...]:
    return ("json", "npz", "log") if numerical else ("json", "log")


def _graphs(*, exact_dsa: bool) -> tuple[str, ...]:
    return EXACT_DSA_GRAPHS if exact_dsa else BASE_GRAPHS


def _expected_primary_names(*, numerical: bool, exact_dsa: bool = False) -> set[str]:
    names = {
        f"host_records/runner.rank{rank}.{suffix}"
        for rank in RANKS
        for suffix in _runner_suffixes(numerical=numerical)
    }
    names.update(
        f"hlo/{graph}.rank{rank}.{suffix}"
        for rank in RANKS
        for graph in _graphs(exact_dsa=exact_dsa)
        for suffix, _ in HLO_FORMS
    )
    if numerical:
        names.update(f"traces/trace.rank{rank}.xplane.pb" for rank in RANKS)
    return names


def _expected_runner_status(mode: str) -> str:
    if mode == "acquire":
        return "HLO_ACQUIRED"
    if mode == "numerical":
        return "SUCCESS"
    raise ValueError(f"unsupported WS32 evidence mode: {mode}")


def _split_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise SystemExit(f"not a GCS URI: {uri}")
    bucket, separator, prefix = uri[5:].partition("/")
    if not separator or not bucket or not prefix:
        raise SystemExit(f"incomplete GCS URI: {uri}")
    return bucket, prefix.rstrip("/")


def _atomic_download(blob: Any, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        partial.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(
        str(partial), if_generation_match=int(blob.generation)
    )
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, destination)


def _link_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise SystemExit(f"refusing to replace existing evidence: {destination}")
    os.link(source, destination)


def _candidate_files(root: Path) -> Iterable[Path]:
    for directory in (root / "hlo", root / "fleet_hlo", root / "traces"):
        if directory.is_dir():
            yield from (path for path in directory.rglob("*") if path.is_file())
    trace = root / "trace"
    if trace.is_dir():
        yield from trace.rglob("*.xplane.pb")


def _partial_evidence(root: Path) -> list[Path]:
    partials: list[Path] = []
    for directory in (root / "fleet", root / "fleet_hlo", root / "traces"):
        if not directory.is_dir():
            continue
        partials.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.name.endswith((".partial", "_.gstmp"))
        )
    return sorted(partials)


def _require_blob_identity(
    blob: Any,
    path: Path,
    digest: str,
    cache: dict[tuple[int, int], tuple[int, str, str]] | None = None,
) -> None:
    if not blob.generation or not blob.crc32c or blob.size is None:
        raise SystemExit(f"remote object lacks immutable metadata: {blob.name}")
    stat = path.stat()
    inode = (stat.st_dev, stat.st_ino)
    identity = None if cache is None else cache.get(inode)
    if identity is None:
        identity = (stat.st_size, _crc32c_file(path), _sha256_file(path))
        if cache is not None:
            cache[inode] = identity
    if identity != (int(blob.size), str(blob.crc32c), digest):
        raise SystemExit(f"remote/local evidence identity drifted: {blob.name}")


def _record(blob: Any, *, name: str, digest: str) -> dict[str, object]:
    return {
        "crc32c": str(blob.crc32c),
        "generation": int(blob.generation),
        "name": name,
        "sha256": digest,
        "size": int(blob.size),
    }


def materialize(
    *,
    run_dir: Path,
    remote_prefix: str,
    mode: str,
    tag: str,
    code_hash: str,
    recovery_code_hash: str,
    exact_dsa: bool,
    allow_failure_diagnostics: bool,
    output: Path,
) -> dict[str, object]:
    numerical = mode == "numerical"
    if mode not in {"acquire", "numerical"}:
        raise SystemExit(f"unsupported WS32 evidence mode: {mode}")
    partials = _partial_evidence(run_dir)
    if partials:
        raise SystemExit(
            "partial local transfers must be quarantined before materialization: "
            + ", ".join(str(path) for path in partials)
        )
    bucket_name, prefix = _split_gs(remote_prefix)
    client = storage.Client()
    blobs = {
        blob.name.removeprefix(prefix + "/"): blob
        for blob in client.list_blobs(bucket_name, prefix=prefix + "/")
    }
    graph_names = _graphs(exact_dsa=exact_dsa)
    primary = _expected_primary_names(
        numerical=numerical, exact_dsa=exact_dsa
    )
    missing = primary - set(blobs)
    if missing:
        raise SystemExit(f"remote fleet evidence is incomplete: {sorted(missing)}")
    extras = set(blobs) - primary
    forbidden = {
        name
        for name in extras
        if name == "SUCCESS"
        or name == "remote_objects.json"
        or name.startswith("orchestrator/")
        or name.startswith("recovery/")
    }
    if forbidden:
        raise SystemExit(f"remote terminal/recovery objects already exist: {sorted(forbidden)}")
    diagnostic_prefix = f"diagnostic_local/{tag}/"
    recovery_prevalidation = {
        f"recovery_prevalidation/prevalidation.rank{rank}.json"
        for rank in RANKS
    }
    observed_recovery_prevalidation = extras & recovery_prevalidation
    if observed_recovery_prevalidation and (
        not allow_failure_diagnostics
        or observed_recovery_prevalidation != recovery_prevalidation
    ):
        raise SystemExit(
            "recovered prevalidation source set is incomplete or unauthorized: "
            f"{sorted(observed_recovery_prevalidation)}"
        )
    if extras and (
        not allow_failure_diagnostics
        or any(
            not name.startswith(diagnostic_prefix)
            and name not in recovery_prevalidation
            for name in extras
        )
    ):
        raise SystemExit(f"unexpected remote preterminal objects: {sorted(extras)}")

    # Host records are small and independently downloaded before trusting their
    # graph/trace SHA declarations.
    records: list[dict[str, object]] = []
    identity_cache: dict[tuple[int, int], tuple[int, str, str]] = {}
    fleet = run_dir / "fleet"
    for rank in RANKS:
        for suffix in _runner_suffixes(numerical=numerical):
            name = f"host_records/runner.rank{rank}.{suffix}"
            blob = blobs[name]
            destination = fleet / f"runner.rank{rank}.{suffix}"
            if not blob.generation or not blob.crc32c or blob.size is None:
                raise SystemExit(f"remote object lacks immutable metadata: {name}")
            if not destination.is_file() or destination.stat().st_size != int(blob.size):
                _atomic_download(blob, destination)
            digest = _sha256_file(destination)
            _require_blob_identity(blob, destination, digest, identity_cache)
            records.append(_record(blob, name=name, digest=digest))

    runners = []
    expected_status = _expected_runner_status(mode)
    for rank in RANKS:
        path = fleet / f"runner.rank{rank}.json"
        runner = json.loads(path.read_text(encoding="utf-8"))
        if (
            runner.get("status") != expected_status
            or runner.get("compile_only") is not (not numerical)
            or runner.get("launch_process_id") != rank
            or runner.get("code_hash") != code_hash
            or runner.get("exact_dsa") is not exact_dsa
        ):
            raise SystemExit(f"runner rank {rank} identity/status drifted")
        runners.append(runner)

    candidates_by_sha: dict[str, Path] = {}
    candidate_inodes: set[tuple[int, int]] = set()
    for path in _candidate_files(run_dir):
        if path.name.endswith((".partial", "_.gstmp")):
            continue
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in candidate_inodes:
            continue
        candidate_inodes.add(inode)
        digest = _sha256_file(path)
        candidates_by_sha.setdefault(digest, path)

    fleet_hlo = run_dir / "fleet_hlo"
    for rank, runner in enumerate(runners):
        graphs = runner.get("graphs")
        if not isinstance(graphs, dict) or set(graphs) != set(graph_names):
            raise SystemExit(f"runner rank {rank} graph schema drifted")
        for graph in graph_names:
            report = graphs[graph]
            if not isinstance(report, dict):
                raise SystemExit(f"runner rank {rank} graph report drifted")
            for suffix, sha_key in HLO_FORMS:
                digest = report.get(sha_key)
                if not isinstance(digest, str) or len(digest) != 64:
                    raise SystemExit(f"runner rank {rank} graph SHA drifted")
                name = f"hlo/{graph}.rank{rank}.{suffix}"
                blob = blobs[name]
                source = candidates_by_sha.get(digest)
                destination = fleet_hlo / f"{graph}.rank{rank}.{suffix}"
                if source is None:
                    required = int(blob.size) + 1024 * 1024 * 1024
                    if shutil.disk_usage(run_dir).free < required:
                        raise SystemExit("insufficient disk for verified HLO materialization")
                    _atomic_download(blob, destination)
                    source = destination
                    candidates_by_sha[digest] = source
                elif not destination.exists():
                    _link_exact(source, destination)
                _require_blob_identity(blob, destination, digest, identity_cache)
                records.append(_record(blob, name=name, digest=digest))

    if numerical:
        traces = run_dir / "traces"
        for rank, runner in enumerate(runners):
            files = runner.get("trace", {}).get("files", [])
            if not isinstance(files, list) or len(files) != 1:
                raise SystemExit(f"runner rank {rank} trace schema drifted")
            expected = files[0]
            digest = expected.get("sha256")
            byte_count = expected.get("byte_count")
            if not isinstance(digest, str) or len(digest) != 64 or type(byte_count) is not int:
                raise SystemExit(f"runner rank {rank} trace identity drifted")
            name = f"traces/trace.rank{rank}.xplane.pb"
            blob = blobs[name]
            if int(blob.size) != byte_count:
                raise SystemExit(f"runner rank {rank} trace byte count drifted")
            source = candidates_by_sha.get(digest)
            destination = traces / f"trace.rank{rank}.xplane.pb"
            if source is None:
                required = byte_count + 1024 * 1024 * 1024
                if shutil.disk_usage(run_dir).free < required:
                    raise SystemExit("insufficient disk for verified XPlane materialization")
                _atomic_download(blob, destination)
                source = destination
                candidates_by_sha[digest] = source
            elif not destination.exists():
                _link_exact(source, destination)
            _require_blob_identity(blob, destination, digest, identity_cache)
            records.append(_record(blob, name=name, digest=digest))

    # Failure diagnostics never contribute to correctness, but preserving and
    # generation-binding them prevents a recovery from hiding the original
    # failure account.  They are small; hash their exact remote bytes directly.
    diagnostic_bytes = 0
    for name in sorted(extras):
        blob = blobs[name]
        if not blob.generation or not blob.crc32c or blob.size is None:
            raise SystemExit(f"diagnostic object lacks immutable metadata: {name}")
        diagnostic_bytes += int(blob.size)
        if diagnostic_bytes > 256 * 1024 * 1024:
            raise SystemExit("failure diagnostics exceed bounded recovery budget")
        raw = blob.download_as_bytes(if_generation_match=int(blob.generation))
        crc = google_crc32c.Checksum()
        crc.update(raw)
        crc32c = base64.b64encode(crc.digest()).decode("ascii")
        if len(raw) != int(blob.size) or crc32c != str(blob.crc32c):
            raise SystemExit(f"diagnostic bytes drifted: {name}")
        records.append(_record(blob, name=name, digest=sha256(raw).hexdigest()))

    value: dict[str, object] = {
        "artifact_kind": "greenfield_ws32_short_decoder_source_ledger",
        "code_hash": code_hash,
        "failure_diagnostics_preserved": bool(extras),
        "recovered_prevalidation": bool(observed_recovery_prevalidation),
        "exact_dsa": exact_dsa,
        "mode": mode,
        "objects": sorted(records, key=lambda item: str(item["name"])),
        "recovery_code_hash": recovery_code_hash,
        "remote_prefix": remote_prefix,
        "run_tag": tag,
    }
    value["ledger_sha256"] = sha256(_canonical(value)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        old = json.loads(output.read_text(encoding="utf-8"))
        if old != value:
            raise SystemExit("source ledger already exists with different bytes")
    else:
        output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--remote-prefix", required=True)
    parser.add_argument("--mode", choices=("acquire", "numerical"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--code-hash", required=True)
    parser.add_argument("--recovery-code-hash", required=True)
    parser.add_argument("--exact-dsa", choices=(0, 1), required=True, type=int)
    parser.add_argument("--allow-failure-diagnostics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    value = materialize(
        run_dir=args.run_dir,
        remote_prefix=args.remote_prefix,
        mode=args.mode,
        tag=args.tag,
        code_hash=args.code_hash,
        recovery_code_hash=args.recovery_code_hash,
        exact_dsa=bool(args.exact_dsa),
        allow_failure_diagnostics=args.allow_failure_diagnostics,
        output=args.output,
    )
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
