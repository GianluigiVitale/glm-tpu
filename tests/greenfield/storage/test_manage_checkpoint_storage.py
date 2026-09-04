"""Hostile tests for generation-bound metadata archival admission."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from glm_tpu.greenfield.storage.reclamation import (
    ReclamationError,
    build_reproducibility_capsule,
)


ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "scripts/greenfield/manage_checkpoint_storage.py"


def _load_module():
    specification = importlib.util.spec_from_file_location(
        "manage_checkpoint_storage_for_test", SCRIPT
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


MODULE = _load_module()


def _capsule() -> dict:
    policy = {
        "artifact_kind": "greenfield_checkpoint_retention_policy",
        "artifacts": [
            {
                "disposition": "keep",
                "expected_bytes": 10,
                "expected_object_count": 1,
                "id": "source",
                "prefix": "models/source/",
                "recipe": {
                    "code_hash": "a" * 40,
                    "dependencies": [],
                    "entrypoint": "scripts/greenfield/source.py",
                },
                "terminal_objects": ["index.json"],
            },
            {
                "disposition": "delete_payloads",
                "expected_bytes": 38,
                "expected_object_count": 3,
                "id": "derived",
                "prefix": "checkpoints/derived/",
                "preserve_metadata_objects": ["SUCCESS", "control.json"],
                "recipe": {
                    "code_hash": "a" * 40,
                    "dependencies": [
                        {
                            "artifact_id": "source",
                            "required_scope": "full_payload",
                        }
                    ],
                    "entrypoint": "scripts/greenfield/pack.py",
                },
                "terminal_objects": ["SUCCESS"],
            },
        ],
        "bucket": "driftbench-dsv4-uc",
        "bucket_location": "US-CENTRAL2",
        "format_version": 1,
        "protected_prefixes": ["models/source/", "results/"],
    }
    objects = {
        "source": [
            {
                "crc32c": "AAAAAA==",
                "generation": 1,
                "name": "models/source/index.json",
                "size": 10,
            }
        ],
        "derived": [
            {
                "crc32c": "BBBBBB==",
                "generation": 2,
                "name": "checkpoints/derived/payload.bin",
                "size": 20,
            },
            {
                "crc32c": "CCCCCC==",
                "generation": 3,
                "name": "checkpoints/derived/SUCCESS",
                "size": 9,
            },
            {
                "crc32c": "CCCCCC==",
                "generation": 7,
                "name": "checkpoints/derived/control.json",
                "size": 9,
            },
        ],
    }
    return build_reproducibility_capsule(
        policy,
        objects,
        generator_code_hash="a" * 40,
        created_utc="2026-09-04T12:00:00Z",
        proof={"proof_sha256": "b" * 64, "status": "passed"},
    )


def _archive_records(capsule: dict) -> tuple[dict, dict, dict]:
    destination_prefix = "results/greenfield_checkpoint_reproduction_capsule_test/"
    sources = [
        deepcopy(item)
        for item in capsule["artifacts"][1]["objects"]
        if not item["name"].endswith("payload.bin")
    ]
    files = []
    for offset, source in enumerate(sources, start=4):
        relative_name = source["name"].removeprefix("checkpoints/derived/")
        files.append(
            {
                "artifact_id": "derived",
                "destination": {
                    "crc32c": source["crc32c"],
                    "generation": offset,
                    "name": (f"{destination_prefix}metadata/derived/{relative_name}"),
                    "size": source["size"],
                },
                "source": source,
            }
        )
    index_identity = {
        "crc32c": "DDDDDD==",
        "generation": 5,
        "name": f"{destination_prefix}metadata_index.json",
        "size": 100,
    }
    success_identity = {
        "crc32c": "EEEEEE==",
        "generation": 6,
        "name": f"{destination_prefix}SUCCESS",
        "size": 100,
    }
    index = {
        "artifact_kind": "greenfield_checkpoint_reproduction_metadata_archive",
        "bucket": "driftbench-dsv4-uc",
        "capsule_sha256": capsule["capsule_sha256"],
        "created_utc": "2026-09-04T12:01:00Z",
        "destination_prefix": destination_prefix,
        "files": files,
        "format_version": 1,
        "source_bytes": 18,
        "source_objects": 2,
    }
    index["archive_sha256"] = MODULE._mapping_hash(index, "archive_sha256")
    success = {
        "archive_sha256": index["archive_sha256"],
        "artifact_kind": "greenfield_checkpoint_reproduction_metadata_success",
        "bucket": "driftbench-dsv4-uc",
        "capsule_sha256": capsule["capsule_sha256"],
        "destination_prefix": destination_prefix,
        "format_version": 1,
        "metadata_index": index_identity,
        "source_bytes": 18,
        "source_objects": 2,
    }
    success["success_sha256"] = MODULE._mapping_hash(success, "success_sha256")
    receipt = {
        "archive_sha256": index["archive_sha256"],
        "artifact_kind": "greenfield_checkpoint_reproduction_metadata_receipt",
        "bucket": "driftbench-dsv4-uc",
        "capsule_sha256": capsule["capsule_sha256"],
        "destination_prefix": destination_prefix,
        "format_version": 1,
        "metadata_index": index_identity,
        "source_bytes": 18,
        "source_objects": 2,
        "success": success_identity,
    }
    receipt["receipt_sha256"] = MODULE._mapping_hash(receipt, "receipt_sha256")
    return index, success, receipt


def _rebind(index: dict, success: dict, receipt: dict) -> None:
    index["archive_sha256"] = MODULE._mapping_hash(index, "archive_sha256")
    success["archive_sha256"] = index["archive_sha256"]
    success["success_sha256"] = MODULE._mapping_hash(success, "success_sha256")
    receipt["archive_sha256"] = index["archive_sha256"]
    receipt["receipt_sha256"] = MODULE._mapping_hash(receipt, "receipt_sha256")


def test_archive_receipt_is_bound_to_exact_capsule_generations(monkeypatch) -> None:
    capsule = _capsule()
    index, success, receipt = _archive_records(capsule)
    remote = {
        receipt["metadata_index"]["name"]: index,
        receipt["success"]["name"]: success,
    }
    monkeypatch.setattr(MODULE, "_verify_remote_identity", lambda *_: None)
    monkeypatch.setattr(
        MODULE,
        "_download_exact_json",
        lambda _bucket, identity: remote[identity["name"]],
    )
    MODULE._verify_archive_receipt(object(), capsule, receipt)

    index["files"][0]["source"]["generation"] += 1
    _rebind(index, success, receipt)
    with pytest.raises(ReclamationError, match="exact capsule sources"):
        MODULE._verify_archive_receipt(object(), capsule, receipt)


@pytest.mark.parametrize("attack", ["duplicate", "wrong_path", "schema"])
def test_archive_receipt_rejects_destination_and_schema_attacks(
    monkeypatch, attack: str
) -> None:
    capsule = _capsule()
    index, success, receipt = _archive_records(capsule)
    if attack == "duplicate":
        index["files"][1]["destination"] = deepcopy(index["files"][0]["destination"])
    elif attack == "wrong_path":
        index["files"][0]["destination"]["name"] += ".wrong"
    else:
        index["unexpected"] = True
    _rebind(index, success, receipt)
    remote = {
        receipt["metadata_index"]["name"]: index,
        receipt["success"]["name"]: success,
    }
    monkeypatch.setattr(MODULE, "_verify_remote_identity", lambda *_: None)
    monkeypatch.setattr(
        MODULE,
        "_download_exact_json",
        lambda _bucket, identity: remote[identity["name"]],
    )
    expected = "schema" if attack == "schema" else "copy identity"
    with pytest.raises(ReclamationError, match=expected):
        MODULE._verify_archive_receipt(object(), capsule, receipt)


def test_archive_receipt_from_another_capsule_is_rejected_before_remote_io(
    monkeypatch,
) -> None:
    capsule = _capsule()
    _index, _success, receipt = _archive_records(capsule)
    receipt["capsule_sha256"] = "f" * 64
    receipt["receipt_sha256"] = MODULE._mapping_hash(receipt, "receipt_sha256")
    monkeypatch.setattr(
        MODULE,
        "_download_exact_json",
        lambda *_: pytest.fail("remote archive must not be read"),
    )
    with pytest.raises(ReclamationError, match="another capsule"):
        MODULE._verify_archive_receipt(object(), capsule, receipt)
