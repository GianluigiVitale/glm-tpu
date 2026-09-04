#!/usr/bin/env python3
"""Create, inspect, and conditionally apply checkpoint reclamation capsules."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from google.cloud import storage

from glm_tpu.greenfield.storage.reclamation import (
    APPROVED_BUCKET,
    APPROVED_LOCATION,
    ReclamationError,
    build_reproducibility_capsule,
    capsule_sha256,
    deletion_order,
    reconcile_live_objects,
    validate_capsule,
    validate_policy,
)


REPO = Path("/home/gianl/glm-tpu-topology-rewrite")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ReclamationError(f"JSON object required: {path}")
    return value


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(REPO), *arguments), text=True
    ).strip()


def _blob_identity(blob: Any) -> dict[str, Any]:
    if blob.size is None or blob.generation is None or not blob.crc32c:
        raise ReclamationError(f"incomplete GCS identity: {blob.name}")
    return {
        "crc32c": blob.crc32c,
        "generation": int(blob.generation),
        "name": blob.name,
        "size": int(blob.size),
    }


def _list_prefix(client: storage.Client, prefix: str) -> list[dict[str, Any]]:
    return [
        _blob_identity(blob)
        for blob in client.list_blobs(APPROVED_BUCKET, prefix=prefix)
    ]


def _bucket(client: storage.Client) -> storage.Bucket:
    bucket = client.get_bucket(APPROVED_BUCKET)
    if bucket.location != APPROVED_LOCATION:
        raise ReclamationError(
            f"bucket location drifted: {bucket.location!r} != {APPROVED_LOCATION!r}"
        )
    return bucket


def _verify_recipes(policy: dict[str, Any]) -> None:
    for artifact in policy["artifacts"]:
        recipe = artifact.get("recipe")
        if recipe is None:
            continue
        code_hash = recipe["code_hash"]
        try:
            _git("cat-file", "-e", f"{code_hash}^{{commit}}")
            _git("cat-file", "-e", f"{code_hash}:{recipe['entrypoint']}")
        except subprocess.CalledProcessError as error:
            raise ReclamationError(
                f"reproduction recipe is absent at {code_hash}: "
                f"{recipe['entrypoint']}"
            ) from error


def _verify_manifest_identities(client: storage.Client, policy: dict[str, Any]) -> None:
    bucket = client.bucket(APPROVED_BUCKET)
    for artifact in policy["artifacts"]:
        expected = artifact.get("expected_manifest_sha256")
        manifest_object = artifact.get("manifest_object")
        if expected is None:
            continue
        if not isinstance(manifest_object, str):
            raise ReclamationError(f"{artifact['id']} lacks manifest_object")
        raw = bucket.blob(f"{artifact['prefix']}{manifest_object}").download_as_bytes(
            timeout=300
        )
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("manifest_sha256") != expected:
            raise ReclamationError(f"{artifact['id']} manifest identity drifted")


def _create(args: argparse.Namespace) -> None:
    if Path(_git("rev-parse", "--show-toplevel")) != REPO:
        raise ReclamationError("wrong repository")
    policy = _read_json(args.policy)
    proof = _read_json(args.proof)
    validate_policy(policy)
    _verify_recipes(policy)
    client = storage.Client()
    _bucket(client)
    _verify_manifest_identities(client, policy)
    objects = {
        artifact["id"]: _list_prefix(client, artifact["prefix"])
        for artifact in policy["artifacts"]
    }
    capsule = build_reproducibility_capsule(
        policy,
        objects,
        generator_code_hash=_git("rev-parse", "HEAD"),
        created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        proof=proof,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(capsule, indent=2, sort_keys=True) + "\n")
    candidates = deletion_order(capsule)
    print(
        json.dumps(
            {
                "capsule_sha256": capsule["capsule_sha256"],
                "delete_bytes": sum(item["size"] for item in candidates),
                "delete_objects": len(candidates),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


def _inspect(args: argparse.Namespace) -> None:
    capsule = _read_json(args.capsule)
    validate_capsule(capsule)
    client = storage.Client()
    _bucket(client)
    for artifact in capsule["artifacts"]:
        reconcile_live_objects(
            artifact["objects"], _list_prefix(client, artifact["prefix"])
        )
    candidates = deletion_order(capsule)
    print(
        json.dumps(
            {
                "capsule_sha256": capsule["capsule_sha256"],
                "delete_bytes": sum(item["size"] for item in candidates),
                "delete_objects": len(candidates),
                "live_identity": "exact",
            },
            sort_keys=True,
        )
    )


def _expected_metadata_sources(capsule: dict[str, Any]) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for artifact in capsule["artifacts"]:
        if artifact["disposition"] not in ("delete_now", "delete_payloads"):
            continue
        prefix = artifact["prefix"]
        by_name = {item["name"]: item for item in artifact["objects"]}
        for relative_name in artifact["preserve_metadata_objects"]:
            name = f"{prefix}{relative_name}"
            source = by_name.get(name)
            if source is None:
                raise ReclamationError(f"capsule metadata source is missing: {name}")
            expected[name] = dict(source)
    return expected


def _validate_identity(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "crc32c",
        "generation",
        "name",
        "size",
    }:
        raise ReclamationError(f"{field} object identity schema drifted")
    if (
        not isinstance(value["name"], str)
        or not value["name"]
        or value["name"].startswith("/")
        or ".." in value["name"].split("/")
        or not isinstance(value["crc32c"], str)
        or not value["crc32c"]
        or not isinstance(value["generation"], int)
        or isinstance(value["generation"], bool)
        or value["generation"] <= 0
        or not isinstance(value["size"], int)
        or isinstance(value["size"], bool)
        or value["size"] < 0
    ):
        raise ReclamationError(f"{field} object identity is invalid")
    return dict(value)


def _expected_metadata_archive_records(
    capsule: dict[str, Any], destination_prefix: str
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for artifact in capsule["artifacts"]:
        if artifact["disposition"] not in ("delete_now", "delete_payloads"):
            continue
        prefix = artifact["prefix"]
        by_name = {item["name"]: item for item in artifact["objects"]}
        for relative_name in artifact["preserve_metadata_objects"]:
            source_name = f"{prefix}{relative_name}"
            records[source_name] = {
                "artifact_id": artifact["id"],
                "destination_name": (
                    f"{destination_prefix}metadata/{artifact['id']}/{relative_name}"
                ),
                "source": dict(by_name[source_name]),
            }
    return records


def _archive_metadata(args: argparse.Namespace) -> None:
    capsule = _read_json(args.capsule)
    validate_capsule(capsule)
    destination = args.destination_prefix
    if (
        not destination.startswith(
            "results/greenfield_checkpoint_reproduction_capsule_"
        )
        or not destination.endswith("/")
        or ".." in destination.split("/")
    ):
        raise ReclamationError("metadata archive destination is not protected")
    client = storage.Client()
    bucket = _bucket(client)
    if _list_prefix(client, destination):
        raise ReclamationError("metadata archive destination is not vacant")

    # Bind every source generation before copying any metadata.  The archive is
    # derived from the exact capsule ledger, never from count/byte aggregates.
    for artifact in capsule["artifacts"]:
        reconcile_live_objects(
            artifact["objects"], _list_prefix(client, artifact["prefix"])
        )

    archived: list[dict[str, Any]] = []
    for artifact in capsule["artifacts"]:
        if artifact["disposition"] not in ("delete_now", "delete_payloads"):
            continue
        live = {item["name"]: item for item in artifact["objects"]}
        for relative_name in artifact["preserve_metadata_objects"]:
            source_name = f"{artifact['prefix']}{relative_name}"
            source = live.get(source_name)
            if source is None:
                raise ReclamationError(
                    f"metadata archive source is missing: {source_name}"
                )
            destination_name = f"{destination}metadata/{artifact['id']}/{relative_name}"
            copied = bucket.copy_blob(
                bucket.blob(source_name, generation=source["generation"]),
                bucket,
                destination_name,
                source_generation=source["generation"],
                if_generation_match=0,
                if_source_generation_match=source["generation"],
                timeout=300,
            )
            copied.reload(timeout=300)
            destination_identity = _blob_identity(copied)
            if (
                destination_identity["size"] != source["size"]
                or destination_identity["crc32c"] != source["crc32c"]
            ):
                raise ReclamationError(
                    f"archived metadata identity drifted: {source_name}"
                )
            archived.append(
                {
                    "artifact_id": artifact["id"],
                    "destination": destination_identity,
                    "source": source,
                }
            )
    index: dict[str, Any] = {
        "artifact_kind": "greenfield_checkpoint_reproduction_metadata_archive",
        "bucket": APPROVED_BUCKET,
        "capsule_sha256": capsule["capsule_sha256"],
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "destination_prefix": destination,
        "files": archived,
        "format_version": 1,
        "source_bytes": sum(item["source"]["size"] for item in archived),
        "source_objects": len(archived),
    }
    index["archive_sha256"] = _mapping_hash(index, "archive_sha256")
    raw = json.dumps(index, indent=2, sort_keys=True) + "\n"
    index_blob = bucket.blob(f"{destination}metadata_index.json")
    index_blob.upload_from_string(
        raw,
        content_type="application/json",
        if_generation_match=0,
        timeout=300,
    )
    index_blob.reload(timeout=300)
    index_identity = _blob_identity(index_blob)
    success = {
        "archive_sha256": index["archive_sha256"],
        "artifact_kind": "greenfield_checkpoint_reproduction_metadata_success",
        "bucket": APPROVED_BUCKET,
        "capsule_sha256": capsule["capsule_sha256"],
        "destination_prefix": destination,
        "format_version": 1,
        "metadata_index": index_identity,
        "source_bytes": index["source_bytes"],
        "source_objects": index["source_objects"],
    }
    success["success_sha256"] = _mapping_hash(success, "success_sha256")
    success_blob = bucket.blob(f"{destination}SUCCESS")
    success_blob.upload_from_string(
        json.dumps(success, indent=2, sort_keys=True) + "\n",
        content_type="application/json",
        if_generation_match=0,
        timeout=300,
    )
    success_blob.reload(timeout=300)
    receipt: dict[str, Any] = {
        "archive_sha256": index["archive_sha256"],
        "artifact_kind": "greenfield_checkpoint_reproduction_metadata_receipt",
        "bucket": APPROVED_BUCKET,
        "capsule_sha256": capsule["capsule_sha256"],
        "destination_prefix": destination,
        "format_version": 1,
        "metadata_index": index_identity,
        "source_bytes": index["source_bytes"],
        "source_objects": index["source_objects"],
        "success": _blob_identity(success_blob),
    }
    receipt["receipt_sha256"] = _mapping_hash(receipt, "receipt_sha256")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "archive_sha256": index["archive_sha256"],
                "capsule_sha256": capsule["capsule_sha256"],
                "destination": f"gs://{APPROVED_BUCKET}/{destination}",
                "receipt_sha256": receipt["receipt_sha256"],
                "source_bytes": index["source_bytes"],
                "source_objects": index["source_objects"],
            },
            sort_keys=True,
        )
    )


def _verify_remote_identity(bucket: storage.Bucket, expected: dict[str, Any]) -> None:
    expected = _validate_identity(expected, field="archived")
    blob = bucket.blob(expected["name"], generation=expected["generation"])
    blob.reload(if_generation_match=expected["generation"], timeout=300)
    if _blob_identity(blob) != expected:
        raise ReclamationError(f"archived object identity drifted: {expected['name']}")


def _download_exact_json(
    bucket: storage.Bucket, expected: dict[str, Any]
) -> dict[str, Any]:
    _verify_remote_identity(bucket, expected)
    raw = bucket.blob(
        expected["name"], generation=expected["generation"]
    ).download_as_bytes(if_generation_match=expected["generation"], timeout=300)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ReclamationError(f"archived JSON is not an object: {expected['name']}")
    return value


def _verify_archive_receipt(
    bucket: storage.Bucket,
    capsule: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    required = {
        "archive_sha256",
        "artifact_kind",
        "bucket",
        "capsule_sha256",
        "destination_prefix",
        "format_version",
        "metadata_index",
        "receipt_sha256",
        "source_bytes",
        "source_objects",
        "success",
    }
    if set(receipt) != required:
        raise ReclamationError("metadata archive receipt schema drifted")
    if (
        receipt["artifact_kind"]
        != "greenfield_checkpoint_reproduction_metadata_receipt"
    ):
        raise ReclamationError("wrong metadata archive receipt kind")
    if receipt["format_version"] != 1 or receipt["bucket"] != APPROVED_BUCKET:
        raise ReclamationError("metadata archive receipt boundary drifted")
    if receipt["capsule_sha256"] != capsule["capsule_sha256"]:
        raise ReclamationError("metadata archive belongs to another capsule")
    if receipt["receipt_sha256"] != _mapping_hash(receipt, "receipt_sha256"):
        raise ReclamationError("metadata archive receipt SHA-256 drifted")

    destination_prefix = receipt["destination_prefix"]
    if (
        not isinstance(destination_prefix, str)
        or not destination_prefix.startswith(
            "results/greenfield_checkpoint_reproduction_capsule_"
        )
        or not destination_prefix.endswith("/")
        or ".." in destination_prefix.split("/")
    ):
        raise ReclamationError("metadata archive destination boundary drifted")
    metadata_index_identity = _validate_identity(
        receipt["metadata_index"], field="metadata_index"
    )
    success_identity = _validate_identity(receipt["success"], field="success")
    if metadata_index_identity["name"] != f"{destination_prefix}metadata_index.json":
        raise ReclamationError("metadata archive index pathname drifted")
    if success_identity["name"] != f"{destination_prefix}SUCCESS":
        raise ReclamationError("metadata archive SUCCESS pathname drifted")

    index = _download_exact_json(bucket, metadata_index_identity)
    if set(index) != {
        "archive_sha256",
        "artifact_kind",
        "bucket",
        "capsule_sha256",
        "created_utc",
        "destination_prefix",
        "files",
        "format_version",
        "source_bytes",
        "source_objects",
    }:
        raise ReclamationError("remote metadata index schema drifted")
    if (
        index["artifact_kind"] != "greenfield_checkpoint_reproduction_metadata_archive"
        or index["bucket"] != APPROVED_BUCKET
        or index["format_version"] != 1
        or not isinstance(index["created_utc"], str)
        or not index["created_utc"].endswith("Z")
    ):
        raise ReclamationError("remote metadata index boundary drifted")
    if index.get("archive_sha256") != _mapping_hash(index, "archive_sha256"):
        raise ReclamationError("remote metadata index SHA-256 drifted")
    if index.get("archive_sha256") != receipt["archive_sha256"]:
        raise ReclamationError("metadata archive hash disagrees with receipt")
    if (
        index.get("capsule_sha256") != capsule["capsule_sha256"]
        or index.get("destination_prefix") != receipt["destination_prefix"]
    ):
        raise ReclamationError("remote metadata index capsule binding drifted")

    files = index.get("files")
    if not isinstance(files, list):
        raise ReclamationError("remote metadata index file ledger is invalid")
    expected_records = _expected_metadata_archive_records(capsule, destination_prefix)
    expected_sources = {
        name: record["source"] for name, record in expected_records.items()
    }
    observed_sources: dict[str, dict[str, Any]] = {}
    observed_destinations: set[tuple[str, int]] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "artifact_id",
            "destination",
            "source",
        }:
            raise ReclamationError("remote metadata archive file record drifted")
        source = _validate_identity(item["source"], field="archive source")
        if source["name"] in observed_sources:
            raise ReclamationError("remote metadata archive source ledger is invalid")
        observed_sources[source["name"]] = source
        expected_record = expected_records.get(source["name"])
        if (
            expected_record is None
            or item["artifact_id"] != expected_record["artifact_id"]
        ):
            raise ReclamationError("remote metadata archive source mapping drifted")
        destination = _validate_identity(
            item["destination"], field="archive destination"
        )
        destination_key = (destination["name"], destination["generation"])
        if (
            destination.get("size") != source.get("size")
            or destination.get("crc32c") != source.get("crc32c")
            or destination["name"] != expected_record["destination_name"]
            or destination_key in observed_destinations
        ):
            raise ReclamationError("remote metadata archive copy identity drifted")
        observed_destinations.add(destination_key)
        _verify_remote_identity(bucket, destination)
    if observed_sources != expected_sources:
        raise ReclamationError(
            "metadata archive does not cover the exact capsule sources"
        )
    if (
        index.get("source_objects") != len(expected_sources)
        or index.get("source_bytes")
        != sum(item["size"] for item in expected_sources.values())
        or receipt["source_objects"] != index["source_objects"]
        or receipt["source_bytes"] != index["source_bytes"]
    ):
        raise ReclamationError("metadata archive source totals drifted")

    success = _download_exact_json(bucket, success_identity)
    if set(success) != {
        "archive_sha256",
        "artifact_kind",
        "bucket",
        "capsule_sha256",
        "destination_prefix",
        "format_version",
        "metadata_index",
        "source_bytes",
        "source_objects",
        "success_sha256",
    }:
        raise ReclamationError("remote metadata SUCCESS schema drifted")
    if (
        success["artifact_kind"]
        != "greenfield_checkpoint_reproduction_metadata_success"
        or success["bucket"] != APPROVED_BUCKET
        or success["format_version"] != 1
        or success["destination_prefix"] != destination_prefix
    ):
        raise ReclamationError("remote metadata SUCCESS boundary drifted")
    if success.get("success_sha256") != _mapping_hash(success, "success_sha256"):
        raise ReclamationError("remote metadata SUCCESS SHA-256 drifted")
    if (
        success.get("archive_sha256") != receipt["archive_sha256"]
        or success.get("capsule_sha256") != capsule["capsule_sha256"]
        or success.get("metadata_index") != receipt["metadata_index"]
        or success.get("source_objects") != receipt["source_objects"]
        or success.get("source_bytes") != receipt["source_bytes"]
    ):
        raise ReclamationError("remote metadata SUCCESS binding drifted")


def _receipt_hash(receipt: dict[str, Any]) -> str:
    return _mapping_hash(receipt, "receipt_sha256")


def _mapping_hash(value: dict[str, Any], hash_field: str) -> str:
    unhashed = dict(value)
    unhashed.pop(hash_field, None)
    raw = json.dumps(
        unhashed,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(raw).hexdigest()


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _apply(args: argparse.Namespace) -> None:
    capsule = _read_json(args.capsule)
    validate_capsule(capsule)
    observed_hash = capsule_sha256(capsule)
    if args.expected_capsule_sha256 != observed_hash:
        raise ReclamationError("operator-pinned capsule SHA-256 disagrees")
    if args.authorization != f"DELETE_{observed_hash[:16]}":
        raise ReclamationError("exact deletion authorization token disagrees")
    if args.receipt.exists():
        raise ReclamationError("refusing to overwrite an existing deletion receipt")
    client = storage.Client()
    bucket = _bucket(client)
    archive_receipt = _read_json(args.archive_receipt)
    _verify_archive_receipt(bucket, capsule, archive_receipt)

    # Reconcile both delete candidates and protected keep records immediately
    # before the first mutation.  This catches canonical/runtime drift too.
    for artifact in capsule["artifacts"]:
        reconcile_live_objects(
            artifact["objects"], _list_prefix(client, artifact["prefix"])
        )
    candidates = deletion_order(capsule)
    receipt: dict[str, Any] = {
        "artifact_kind": "greenfield_checkpoint_deletion_receipt",
        "bucket": APPROVED_BUCKET,
        "capsule_sha256": observed_hash,
        "completed": False,
        "deleted": [],
        "format_version": 1,
        "planned_bytes": sum(item["size"] for item in candidates),
        "planned_objects": len(candidates),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    _write_receipt(args.receipt, receipt)
    for item in candidates:
        blob = bucket.blob(item["name"], generation=item["generation"])
        blob.delete(if_generation_match=item["generation"], timeout=300)
        receipt["deleted"].append(dict(item))
        _write_receipt(args.receipt, receipt)

    for artifact in capsule["artifacts"]:
        observed = _list_prefix(client, artifact["prefix"])
        if artifact["disposition"] == "delete_now":
            if observed:
                raise ReclamationError(
                    f"candidate prefix remains nonempty: {artifact['id']}"
                )
        elif artifact["disposition"] == "delete_payloads":
            prefix = artifact["prefix"]
            preserved_names = {
                f"{prefix}{name}" for name in artifact["preserve_metadata_objects"]
            }
            expected_preserved = [
                item for item in artifact["objects"] if item["name"] in preserved_names
            ]
            reconcile_live_objects(expected_preserved, observed)
        else:
            reconcile_live_objects(artifact["objects"], observed)
    receipt["completed"] = True
    receipt["completed_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt["deleted_bytes"] = sum(item["size"] for item in receipt["deleted"])
    receipt["deleted_objects"] = len(receipt["deleted"])
    _write_receipt(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--policy", type=Path, required=True)
    create.add_argument("--proof", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--capsule", type=Path, required=True)
    archive = subparsers.add_parser("archive-metadata")
    archive.add_argument("--capsule", type=Path, required=True)
    archive.add_argument("--destination-prefix", required=True)
    archive.add_argument("--output", type=Path, required=True)
    apply = subparsers.add_parser("apply")
    apply.add_argument("--capsule", type=Path, required=True)
    apply.add_argument("--archive-receipt", type=Path, required=True)
    apply.add_argument("--expected-capsule-sha256", required=True)
    apply.add_argument("--authorization", required=True)
    apply.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        _create(args)
    elif args.command == "inspect":
        _inspect(args)
    elif args.command == "archive-metadata":
        _archive_metadata(args)
    else:
        _apply(args)


if __name__ == "__main__":
    main()
