#!/usr/bin/env python3
"""Recover a pre-authorization WS32 acquisition without recompiling.

Every worker writes ``hlo/prevalidation.json`` after all graphs compile and
before acquisition authorization.  A structural false positive can therefore
leave complete HLO/prevalidation bytes but no final runner JSON.  This tool
replays the preserved graphs with the corrected fail-closed validator, builds
only the deterministic ``HLO_ACQUIRED`` envelope, and conditionally publishes
the original prevalidation plus derived runner records.  It never initializes
JAX or executes TPU work.
"""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import google_crc32c
from google.api_core.exceptions import NotFound
from google.cloud import storage

from glm_tpu.greenfield.benchmarking.ws32_decoder import (
    validate_ws32_decoder_hlo,
    validate_ws32_exact_dsa_materializer_hlo,
)


GRAPHS = (
    "exact_materialize",
    "exact_promote",
    "prefill_chunk",
    "prefill_tail",
    "observer",
    "decode",
    "cache_probe",
)
RANKS = tuple(range(8))
IDENTITY_VIOLATIONS = (
    "StableHLO identity drifted",
    "optimized HLO identity drifted",
)
ORIGINAL_VIOLATIONS = {
    "cache_probe": IDENTITY_VIOLATIONS,
    "decode": (
        *IDENTITY_VIOLATIONS,
        "optimized HLO reconstructs a full-pod hidden value",
    ),
    "exact_materialize": (
        *IDENTITY_VIOLATIONS,
        "exact materializer collective geometry drifted",
        "exact materializer contains async collectives",
    ),
    "exact_promote": (
        *IDENTITY_VIOLATIONS,
        "exact materializer contains async collectives",
    ),
    "observer": (
        *IDENTITY_VIOLATIONS,
        "optimized HLO reconstructs a full-pod hidden value",
    ),
    "prefill_chunk": (
        *IDENTITY_VIOLATIONS,
        "optimized HLO reconstructs a full-pod hidden value",
    ),
    "prefill_tail": (
        *IDENTITY_VIOLATIONS,
        "optimized HLO reconstructs a full-pod hidden value",
    ),
}


def _linter_kind(graph: str) -> str:
    """Both prefill programs (chunk and tail) carry the ``prefill`` HLO contract."""
    return "prefill" if graph.startswith("prefill") else graph


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


def _atomic_json(path: Path, value: object) -> None:
    raw = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != raw:
            raise SystemExit(f"refusing to replace different recovery bytes: {path}")
        return
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise SystemExit(f"partial recovery output already exists: {partial}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(raw, encoding="utf-8")
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _split_gs(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise SystemExit(f"not a GCS URI: {uri}")
    bucket, separator, prefix = uri[5:].partition("/")
    if not separator or not bucket or not prefix:
        raise SystemExit(f"incomplete GCS URI: {uri}")
    return bucket, prefix.rstrip("/")


def _replay_graphs(
    run_dir: Path, *, exact_dsa: bool = True, strategy_nd_dense: bool = False
) -> dict[str, dict[str, Any]]:
    """Replay the preserved graphs under the run's own variant flags.

    The flags come from the workers' prevalidation records (all ranks must
    agree); replaying a dense-overlay run without ``strategy_nd_dense`` would
    report its overlay gathers as violations.
    """
    result: dict[str, dict[str, Any]] = {}
    for graph in GRAPHS:
        stable = (run_dir / "hlo" / f"{graph}.stablehlo.mlir").read_text(
            encoding="utf-8"
        )
        optimized = (
            run_dir / "hlo" / f"{graph}.optimized_hlo.txt"
        ).read_text(encoding="utf-8")
        if graph.startswith("exact_"):
            report = validate_ws32_exact_dsa_materializer_hlo(
                stable,
                optimized,
                expected_stablehlo_sha256="0" * 64,
                expected_optimized_hlo_sha256="0" * 64,
                kind=graph,
                full_indexer_count=21,
            )
        else:
            report = validate_ws32_decoder_hlo(
                stable,
                optimized,
                expected_stablehlo_sha256="0" * 64,
                expected_optimized_hlo_sha256="0" * 64,
                hidden_size=6144,
                kind=_linter_kind(graph),
                exact_dsa=exact_dsa,
                strategy_nd_dense=strategy_nd_dense,
                full_indexer_count=21,
            )
        if report.violations != IDENTITY_VIOLATIONS:
            raise SystemExit(
                f"recovered {graph} retains structural violations: "
                f"{report.violations}"
            )
        result[graph] = report.to_dict()
    return result


def synthesize(
    *, run_dir: Path, source_code_hash: str
) -> tuple[Path, ...]:
    if len(source_code_hash) != 40:
        raise SystemExit("source code hash must be one full Git SHA")
    sources: dict[int, dict[str, Any]] = {}
    for rank in RANKS:
        source_path = (
            run_dir
            / "recovery_prevalidation"
            / f"prevalidation.rank{rank}.json"
        )
        sources[rank] = json.loads(source_path.read_text(encoding="utf-8"))
    for rank, source in sources.items():
        for key in ("exact_dsa", "strategy_nd_dense"):
            if type(source.get(key)) is not bool:
                raise SystemExit(f"source prevalidation lacks a boolean {key} at rank {rank}")
    flags = {
        (bool(source["exact_dsa"]), bool(source["strategy_nd_dense"]))
        for source in sources.values()
    }
    if len(flags) != 1:
        raise SystemExit("source prevalidation variant flags disagree across ranks")
    exact_dsa, strategy_nd_dense = next(iter(flags))
    recovered_graphs = _replay_graphs(
        run_dir, exact_dsa=exact_dsa, strategy_nd_dense=strategy_nd_dense
    )
    outputs: list[Path] = []
    common_graphs: dict[str, Any] | None = None
    for rank in RANKS:
        source = sources[rank]
        original_graphs = source.get("graphs")
        if (
            source.get("artifact_kind")
            != "greenfield_ws32_short_decoder_prevalidation"
            or source.get("code_hash") != source_code_hash
            or source.get("compile_only") is not True
            or source.get("exact_dsa") is not True
            or source.get("launch_process_id") != rank
            or type(original_graphs) is not dict
            or set(original_graphs) != set(GRAPHS)
        ):
            raise SystemExit(f"source prevalidation identity drifted at rank {rank}")
        for graph in GRAPHS:
            original = original_graphs[graph]
            recovered = recovered_graphs[graph]
            # A graph either passed its original lint (identity-only, the
            # acquisition's vacant pins) or carried exactly the documented
            # structural false positive; anything else is not recoverable.
            if (
                tuple(original.get("violations", ()))
                not in (IDENTITY_VIOLATIONS, ORIGINAL_VIOLATIONS[graph])
                or original.get("passed") is not False
                or original.get("stablehlo_sha256")
                != recovered["stablehlo_sha256"]
                or original.get("optimized_hlo_sha256")
                != recovered["optimized_hlo_sha256"]
            ):
                raise SystemExit(
                    f"source prevalidation graph drifted at rank {rank}/{graph}"
                )
        if common_graphs is None:
            common_graphs = original_graphs
        elif original_graphs != common_graphs:
            raise SystemExit(f"source graph reports disagree at rank {rank}")
        runner = {
            **source,
            "graphs": recovered_graphs,
            "performance_claim": False,
            "schema_version": 1,
            "status": "HLO_ACQUIRED",
        }
        output = run_dir / "fleet" / f"runner.rank{rank}.json"
        _atomic_json(output, runner)
        outputs.append(output)
    return tuple(outputs)


def _seed_value(
    *,
    remote_prefix: str,
    source_code_hash: str,
    recovery_code_hash: str,
    objects: list[dict[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {
        "artifact_kind": "greenfield_ws32_acquisition_recovery_seed",
        "objects": sorted(objects, key=lambda item: str(item["name"])),
        "recovery_code_hash": recovery_code_hash,
        "remote_prefix": remote_prefix,
        "source_code_hash": source_code_hash,
    }
    value["seed_sha256"] = sha256(_canonical(value)).hexdigest()
    return value


def publish(
    *,
    run_dir: Path,
    remote_prefix: str,
    source_code_hash: str,
    recovery_code_hash: str,
) -> dict[str, object]:
    if len(recovery_code_hash) != 40:
        raise SystemExit("recovery code hash must be one full Git SHA")
    runners = synthesize(
        run_dir=run_dir, source_code_hash=source_code_hash
    )
    seed_path = run_dir / "recovery_seed_objects.json"
    if seed_path.exists():
        raise SystemExit("recovery seed already exists")
    sources = tuple(
        run_dir
        / "recovery_prevalidation"
        / f"prevalidation.rank{rank}.json"
        for rank in RANKS
    )
    uploads = tuple(
        (
            f"recovery_prevalidation/prevalidation.rank{rank}.json",
            sources[rank],
            "original_prevalidation",
        )
        for rank in RANKS
    ) + tuple(
        (
            f"host_records/runner.rank{rank}.json",
            runners[rank],
            "derived_runner_envelope",
        )
        for rank in RANKS
    )
    bucket_name, prefix = _split_gs(remote_prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    existing = {
        blob.name.removeprefix(prefix + "/")
        for blob in client.list_blobs(bucket_name, prefix=prefix + "/")
    }
    target_names = {name for name, _, _ in uploads}
    if existing & target_names:
        raise SystemExit(
            "recovery targets already exist without a source ledger: "
            f"{sorted(existing & target_names)}"
        )
    records: list[dict[str, object]] = []
    created: list[tuple[Any, int]] = []
    try:
        _atomic_json(
            seed_path,
            _seed_value(
                remote_prefix=remote_prefix,
                source_code_hash=source_code_hash,
                recovery_code_hash=recovery_code_hash,
                objects=records,
            ),
        )
        for name, path, provenance in uploads:
            blob = bucket.blob(f"{prefix}/{name}")
            blob.upload_from_filename(
                str(path),
                if_generation_match=0,
                checksum="crc32c",
                content_type="application/json",
            )
            blob.reload()
            if not blob.generation or blob.size is None or not blob.crc32c:
                raise RuntimeError(f"uploaded recovery object lacks identity: {name}")
            expected = (
                path.stat().st_size,
                _crc32c_file(path),
                _sha256_file(path),
            )
            observed = (int(blob.size), str(blob.crc32c), expected[2])
            if observed != expected:
                raise RuntimeError(f"uploaded recovery object drifted: {name}")
            created.append((blob, int(blob.generation)))
            records.append(
                {
                    "crc32c": str(blob.crc32c),
                    "generation": int(blob.generation),
                    "name": name,
                    "provenance": provenance,
                    "sha256": expected[2],
                    "size": int(blob.size),
                }
            )
            seed_path.unlink()
            _atomic_json(
                seed_path,
                _seed_value(
                    remote_prefix=remote_prefix,
                    source_code_hash=source_code_hash,
                    recovery_code_hash=recovery_code_hash,
                    objects=records,
                ),
            )
    except BaseException:
        cleanup_errors = []
        for blob, generation in reversed(created):
            try:
                blob.delete(if_generation_match=generation)
            except BaseException as error:  # preserve exact cleanup failure
                cleanup_errors.append(f"{blob.name}: {error}")
        if not cleanup_errors:
            seed_path.unlink(missing_ok=True)
        else:
            raise RuntimeError(
                f"recovery upload rollback failed: {cleanup_errors}"
            )
        raise
    return _seed_value(
        remote_prefix=remote_prefix,
        source_code_hash=source_code_hash,
        recovery_code_hash=recovery_code_hash,
        objects=records,
    )


def rollback(*, seed_path: Path, remote_prefix: str) -> None:
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in seed.items() if key != "seed_sha256"}
    if (
        seed.get("artifact_kind")
        != "greenfield_ws32_acquisition_recovery_seed"
        or seed.get("seed_sha256") != sha256(_canonical(unsigned)).hexdigest()
        or seed.get("remote_prefix") != remote_prefix
    ):
        raise SystemExit("recovery seed identity drifted")
    bucket_name, prefix = _split_gs(remote_prefix)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for item in reversed(seed["objects"]):
        blob = bucket.blob(f"{prefix}/{item['name']}")
        try:
            blob.reload()
        except NotFound:
            continue
        if (
            int(blob.generation) != item["generation"]
            or int(blob.size) != item["size"]
            or str(blob.crc32c) != item["crc32c"]
        ):
            raise SystemExit(
                f"refusing to delete drifted recovery object: {item['name']}"
            )
        blob.delete(if_generation_match=int(blob.generation))
    remaining = {
        blob.name.removeprefix(prefix + "/")
        for blob in client.list_blobs(bucket_name, prefix=prefix + "/")
    }
    target_names = {str(item["name"]) for item in seed["objects"]}
    if remaining & target_names:
        raise SystemExit(
            "recovery objects remain after rollback: "
            f"{sorted(remaining & target_names)}"
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--run-dir", type=Path, required=True)
    publish_parser.add_argument("--remote-prefix", required=True)
    publish_parser.add_argument("--source-code-hash", required=True)
    publish_parser.add_argument("--recovery-code-hash", required=True)
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--seed", type=Path, required=True)
    rollback_parser.add_argument("--remote-prefix", required=True)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if args.command == "publish":
        value = publish(
            run_dir=args.run_dir,
            remote_prefix=args.remote_prefix,
            source_code_hash=args.source_code_hash,
            recovery_code_hash=args.recovery_code_hash,
        )
        print(json.dumps(value, sort_keys=True))
    else:
        rollback(seed_path=args.seed, remote_prefix=args.remote_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
