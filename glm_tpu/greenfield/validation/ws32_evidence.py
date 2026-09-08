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
from .ws32_prefill import require_fleet_prefill_mode


BASE_GRAPHS = ("prefill_chunk", "prefill_tail", "observer", "decode", "cache_probe")
# Evidence layouts (spec §23, storage plan 2026-09-05).  v1: every rank uploads
# its own HLO text (eight byte-identical copies per graph/form).  v2: each
# graph/form is uploaded once, gzip-compressed, under a rank-agnostic name;
# every rank's record still carries both SHA-256s of the inflated text and the
# sealer requires all ranks to agree and re-hashes the single inflated copy.
EVIDENCE_LAYOUT_V1 = "hlo_per_rank_v1"
EVIDENCE_LAYOUT_V2 = "hlo_single_gzip_v2"
EVIDENCE_LAYOUTS = (EVIDENCE_LAYOUT_V1, EVIDENCE_LAYOUT_V2)
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


def _hlo_object_names(*, exact_dsa: bool, layout: str) -> set[str]:
    if layout == EVIDENCE_LAYOUT_V1:
        return {
            f"hlo/{graph}.rank{rank}.{suffix}"
            for rank in RANKS
            for graph in _graphs(exact_dsa=exact_dsa)
            for suffix, _ in HLO_FORMS
        }
    if layout == EVIDENCE_LAYOUT_V2:
        return {
            f"hlo/{graph}.{suffix}.gz"
            for graph in _graphs(exact_dsa=exact_dsa)
            for suffix, _ in HLO_FORMS
        }
    raise SystemExit(f"unknown WS32 evidence layout: {layout}")


def _expected_primary_names(
    *, numerical: bool, exact_dsa: bool = False, layout: str = EVIDENCE_LAYOUT_V1
) -> set[str]:
    names = _expected_layout_free_names(numerical=numerical)
    names.update(_hlo_object_names(exact_dsa=exact_dsa, layout=layout))
    return names


def _expected_layout_free_names(*, numerical: bool) -> set[str]:
    names = {
        f"host_records/runner.rank{rank}.{suffix}"
        for rank in RANKS
        for suffix in _runner_suffixes(numerical=numerical)
    }
    if numerical:
        names.update(f"traces/trace.rank{rank}.xplane.pb" for rank in RANKS)
    return names


def _runner_layout(runners: list[dict[str, Any]]) -> str:
    try:
        require_fleet_prefill_mode(runners)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    layouts = {str(runner.get("evidence_layout", EVIDENCE_LAYOUT_V1)) for runner in runners}
    if len(layouts) != 1:
        raise SystemExit(f"runner records disagree on the evidence layout: {sorted(layouts)}")
    layout = next(iter(layouts))
    if layout not in EVIDENCE_LAYOUTS:
        raise SystemExit(f"unknown WS32 evidence layout: {layout}")
    return layout


def _inflate_gzip(source: Path, destination: Path, *, maximum_bytes: int) -> None:
    """Inflate one member, refusing to write more than ``maximum_bytes``.

    The inflated SHA is checked afterwards, so an unbounded copy would let a
    corrupt or pathological member fill the controller disk before the check
    fires.
    """
    import gzip

    if maximum_bytes <= 0:
        raise SystemExit("insufficient disk for verified HLO materialization")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SystemExit(f"refusing to replace existing evidence: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    written = 0
    try:
        with gzip.open(source, "rb") as stream, partial.open("wb") as out:
            while True:
                chunk = stream.read(8 * 1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > maximum_bytes:
                    raise SystemExit(f"inflated HLO exceeds the disk budget: {source}")
                out.write(chunk)
    except SystemExit:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)


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
    # Layout-free objects first; the HLO layout is read from the runner records
    # (sealed v1 prefixes stay re-materializable, new runs use v2).
    layout_free = _expected_layout_free_names(numerical=numerical)
    missing = layout_free - set(blobs)
    if missing:
        raise SystemExit(f"remote fleet evidence is incomplete: {sorted(missing)}")
    extras = set(blobs) - layout_free
    hlo_objects = {name for name in extras if name.startswith("hlo/")}
    extras -= hlo_objects
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
    layout = _runner_layout(runners)
    primary = _expected_primary_names(
        numerical=numerical, exact_dsa=exact_dsa, layout=layout
    )
    expected_hlo = primary - layout_free
    if hlo_objects != expected_hlo:
        missing_hlo = sorted(expected_hlo - hlo_objects)
        unexpected_hlo = sorted(hlo_objects - expected_hlo)
        raise SystemExit(
            f"remote HLO evidence does not match layout {layout}: "
            f"missing={missing_hlo} unexpected={unexpected_hlo}"
        )

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
    recorded_shas: dict[tuple[str, str], str] = {}
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
                previous = recorded_shas.setdefault((graph, suffix), digest)
                if previous != digest:
                    raise SystemExit(f"runner ranks disagree on {graph} {suffix} SHA")
                if layout == EVIDENCE_LAYOUT_V2:
                    continue
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

    if layout == EVIDENCE_LAYOUT_V2:
        for graph in graph_names:
            for suffix, _ in HLO_FORMS:
                name = f"hlo/{graph}.{suffix}.gz"
                blob = blobs[name]
                compressed = fleet_hlo / f"{graph}.{suffix}.gz"
                if not compressed.is_file() or compressed.stat().st_size != int(blob.size):
                    required = int(blob.size) + 1024 * 1024 * 1024
                    if shutil.disk_usage(run_dir).free < required:
                        raise SystemExit("insufficient disk for verified HLO materialization")
                    _atomic_download(blob, compressed)
                compressed_digest = _sha256_file(compressed)
                _require_blob_identity(blob, compressed, compressed_digest, identity_cache)
                inflated = fleet_hlo / f"{graph}.rank0.{suffix}"
                expected_digest = recorded_shas[(graph, suffix)]
                if not inflated.exists():
                    # This host is pod worker 0, so the run directory already
                    # holds the worker's own uncompressed text; hard-link it
                    # instead of writing a second copy (v1 did the same).
                    source = candidates_by_sha.get(expected_digest)
                    if source is None:
                        free = shutil.disk_usage(run_dir).free - 1024 * 1024 * 1024
                        _inflate_gzip(compressed, inflated, maximum_bytes=free)
                        candidates_by_sha[expected_digest] = inflated
                    else:
                        _link_exact(source, inflated)
                if _sha256_file(inflated) != expected_digest:
                    raise SystemExit(f"inflated HLO differs from the recorded SHA: {name}")
                for rank in RANKS[1:]:
                    _link_exact(inflated, fleet_hlo / f"{graph}.rank{rank}.{suffix}")
                record = _record(blob, name=name, digest=compressed_digest)
                record["inflated_sha256"] = expected_digest
                records.append(record)

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
        "evidence_layout": layout,
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
