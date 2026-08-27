from __future__ import annotations

from copy import deepcopy

import pytest

from glm_tpu.greenfield.storage.reclamation import (
    ReclamationError,
    build_reproducibility_capsule,
    capsule_sha256,
    deletion_order,
    reconcile_live_objects,
    validate_capsule,
    validate_policy,
)


CODE_HASH = "a" * 40
PROOF = {"proof_sha256": "b" * 64, "status": "passed"}


def _policy() -> dict:
    return {
        "artifact_kind": "greenfield_checkpoint_retention_policy",
        "artifacts": [
            {
                "disposition": "keep",
                "expected_bytes": 11,
                "expected_object_count": 1,
                "id": "canonical",
                "prefix": "models/GLM/",
                "recipe": {
                    "code_hash": CODE_HASH,
                    "dependencies": [],
                    "entrypoint": "scripts/greenfield/build.py",
                },
                "terminal_objects": ["index.json"],
            },
            {
                "disposition": "delete_now",
                "expected_bytes": 29,
                "expected_object_count": 2,
                "id": "derived",
                "prefix": "checkpoints/derived/tag/",
                "preserve_metadata_objects": ["SUCCESS"],
                "recipe": {
                    "code_hash": CODE_HASH,
                    "dependencies": ["canonical"],
                    "entrypoint": "scripts/greenfield/pack.py",
                },
                "terminal_objects": ["SUCCESS"],
            },
        ],
        "bucket": "driftbench-dsv4-uc",
        "bucket_location": "US-CENTRAL2",
        "format_version": 1,
        "protected_prefixes": ["models/", "results/", "oracles/"],
    }


def _objects() -> dict:
    return {
        "canonical": [
            {
                "crc32c": "AAAAAA==",
                "generation": 1,
                "name": "models/GLM/index.json",
                "size": 11,
            }
        ],
        "derived": [
            {
                "crc32c": "BBBBBB==",
                "generation": 2,
                "name": "checkpoints/derived/tag/payload.bin",
                "size": 20,
            },
            {
                "crc32c": "CCCCCC==",
                "generation": 3,
                "name": "checkpoints/derived/tag/SUCCESS",
                "size": 9,
            },
        ],
    }


def _capsule() -> dict:
    return build_reproducibility_capsule(
        _policy(),
        _objects(),
        generator_code_hash=CODE_HASH,
        created_utc="2026-08-27T15:00:00Z",
        proof=PROOF,
    )


def test_capsule_is_self_authenticating_and_orders_terminal_delete_first() -> None:
    capsule = _capsule()
    validate_capsule(capsule)
    assert capsule["capsule_sha256"] == capsule_sha256(capsule)
    ordered = deletion_order(capsule)
    assert [item["name"] for item in ordered] == [
        "checkpoints/derived/tag/SUCCESS",
        "checkpoints/derived/tag/payload.bin",
    ]
    assert sum(item["size"] for item in ordered) == 29


def test_capsule_refuses_generation_drift_and_tampering() -> None:
    expected = _objects()["derived"]
    observed = deepcopy(expected)
    observed[0]["generation"] += 1
    with pytest.raises(ReclamationError, match="identity drifted"):
        reconcile_live_objects(expected, observed)

    capsule = _capsule()
    capsule["artifacts"][1]["objects"][0]["generation"] += 1
    with pytest.raises(ReclamationError, match="capsule SHA-256 drifted"):
        validate_capsule(capsule)


def test_policy_refuses_any_delete_overlap_with_protected_prefix() -> None:
    policy = _policy()
    policy["artifacts"][1]["prefix"] = "models/derived/"
    with pytest.raises(ReclamationError, match="overlaps protected prefix"):
        validate_policy(policy)


def test_capsule_refuses_missing_terminal_and_failed_proof() -> None:
    objects = _objects()
    objects["derived"] = objects["derived"][:1]
    policy = _policy()
    policy["artifacts"][1]["expected_object_count"] = 1
    policy["artifacts"][1]["expected_bytes"] = 20
    with pytest.raises(ReclamationError, match="terminal objects are incomplete"):
        build_reproducibility_capsule(
            policy,
            objects,
            generator_code_hash=CODE_HASH,
            created_utc="2026-08-27T15:00:00Z",
            proof=PROOF,
        )
    with pytest.raises(ReclamationError, match="proof has not passed"):
        build_reproducibility_capsule(
            _policy(),
            _objects(),
            generator_code_hash=CODE_HASH,
            created_utc="2026-08-27T15:00:00Z",
            proof={"proof_sha256": "b" * 64, "status": "failed"},
        )


def test_policy_refuses_unknown_dependency_and_overlapping_artifacts() -> None:
    policy = _policy()
    policy["artifacts"][1]["recipe"]["dependencies"] = ["missing"]
    with pytest.raises(ReclamationError, match="unknown dependency"):
        validate_policy(policy)

    policy = _policy()
    duplicate = deepcopy(policy["artifacts"][1])
    duplicate["id"] = "nested"
    duplicate["prefix"] = "checkpoints/derived/tag/nested/"
    policy["artifacts"].append(duplicate)
    with pytest.raises(ReclamationError, match="prefixes overlap"):
        validate_policy(policy)


def test_payload_only_reclamation_preserves_named_metadata() -> None:
    policy = _policy()
    policy["artifacts"][1]["disposition"] = "delete_payloads"
    capsule = build_reproducibility_capsule(
        policy,
        _objects(),
        generator_code_hash=CODE_HASH,
        created_utc="2026-08-27T15:00:00Z",
        proof=PROOF,
    )
    ordered = deletion_order(capsule)
    assert [item["name"] for item in ordered] == [
        "checkpoints/derived/tag/payload.bin"
    ]
