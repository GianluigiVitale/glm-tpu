"""Generation-pinned checkpoint reclamation without broad prefix deletion.

The policy is reviewed source code.  A capsule is a snapshot of the live GCS
object identities admitted by that policy.  Applying a capsule is deliberately
separate: every current object must still match its name, size, generation and
CRC32C, and each delete carries ``if_generation_match``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from hashlib import sha256
import json
from typing import Any


APPROVED_BUCKET = "driftbench-dsv4-uc"
APPROVED_LOCATION = "US-CENTRAL2"
POLICY_ARTIFACT_KIND = "greenfield_checkpoint_retention_policy"
CAPSULE_ARTIFACT_KIND = "greenfield_checkpoint_reproducibility_capsule"
FORMAT_VERSION = 1


class ReclamationError(ValueError):
    """A retention, identity, or deletion-safety invariant failed."""


def _validate_retirement(value: Any, *, field: str) -> dict[str, Any]:
    """Validate an explicit decision that an artifact is not reproducible state.

    This is intentionally distinct from a reproduction recipe.  It is used for
    unrelated or failed artifacts whose payload is being permanently retired,
    while the generation ledger and compact metadata remain as audit evidence.
    """

    if not isinstance(value, Mapping) or set(value) != {
        "owner_scope",
        "reason",
        "recovery",
        "reference_audit",
    }:
        raise ReclamationError(f"{field} retirement record schema drifted")
    for key in ("owner_scope", "reason", "recovery"):
        item = value.get(key)
        if not isinstance(item, str) or not item.strip():
            raise ReclamationError(f"{field}.{key} must be nonempty")
    references = value.get("reference_audit")
    if (
        not isinstance(references, list)
        or not references
        or any(not isinstance(item, str) or not item.strip() for item in references)
    ):
        raise ReclamationError(f"{field}.reference_audit must be nonempty")
    return deepcopy(dict(value))


def _lineage_record(artifact: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    """Return exactly one validated reproducibility or retirement record."""

    recipe = artifact.get("recipe")
    retirement = artifact.get("retirement")
    if (recipe is None) == (retirement is None):
        raise ReclamationError(
            f"{field} must contain exactly one of recipe or retirement"
        )
    if retirement is not None:
        if artifact.get("disposition") != "delete_now":
            raise ReclamationError(f"{field} retirement is valid only for delete_now")
        return {"retirement": _validate_retirement(retirement, field=field)}
    if not isinstance(recipe, Mapping):
        raise ReclamationError(f"{field} lacks a reproduction recipe")
    _digest(recipe.get("code_hash"), field=f"{field}.recipe.code_hash", lengths=(40,))
    entrypoint = recipe.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.startswith(
        "scripts/greenfield/"
    ):
        raise ReclamationError(f"{field} recipe entrypoint is invalid")
    dependencies = recipe.get("dependencies")
    if not isinstance(dependencies, list):
        raise ReclamationError(f"{field} recipe dependencies are invalid")
    normalized_dependencies = []
    for index, dependency in enumerate(dependencies):
        dependency_field = f"{field}.recipe.dependencies[{index}]"
        if not isinstance(dependency, Mapping):
            raise ReclamationError(
                f"{dependency_field} must declare artifact_id and required_scope"
            )
        required_scope = dependency.get("required_scope")
        expected_keys = {"artifact_id", "required_scope"}
        if required_scope == "metadata_only":
            expected_keys.add("required_objects")
        if set(dependency) != expected_keys:
            raise ReclamationError(f"{dependency_field} schema drifted")
        artifact_id = dependency.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ReclamationError(f"{dependency_field}.artifact_id is invalid")
        if required_scope not in ("full_payload", "metadata_only"):
            raise ReclamationError(f"{dependency_field}.required_scope is invalid")
        normalized = {
            "artifact_id": artifact_id,
            "required_scope": required_scope,
        }
        if required_scope == "metadata_only":
            required_objects = dependency.get("required_objects")
            if (
                not isinstance(required_objects, list)
                or not required_objects
                or any(
                    not isinstance(name, str)
                    or not name
                    or name.startswith("/")
                    or ".." in name.split("/")
                    for name in required_objects
                )
                or len(set(required_objects)) != len(required_objects)
            ):
                raise ReclamationError(
                    f"{dependency_field}.required_objects is invalid"
                )
            normalized["required_objects"] = list(required_objects)
        normalized_dependencies.append(normalized)
    normalized_recipe = deepcopy(dict(recipe))
    normalized_recipe["dependencies"] = normalized_dependencies
    return {"recipe": normalized_recipe}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_hash(value: Mapping[str, Any], *, hash_field: str) -> str:
    unhashed = dict(value)
    unhashed.pop(hash_field, None)
    return sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest()


def _digest(value: Any, *, field: str, lengths: tuple[int, ...] = (64,)) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReclamationError(f"{field} must be a lowercase digest")
    return value


def _prefix(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or value.startswith("/")
        or not value.endswith("/")
        or "//" in value
        or value in ("", "/")
        or ".." in value.split("/")
    ):
        raise ReclamationError(f"{field} must be a normalized object prefix")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ReclamationError(f"{field} must be a positive integer")
    return value


def _object_identity(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    expected = {"crc32c", "generation", "name", "size"}
    if set(value) != expected:
        raise ReclamationError(f"{field} object identity schema drifted")
    name = value.get("name")
    crc32c = value.get("crc32c")
    if (
        not isinstance(name, str)
        or name.startswith("/")
        or ".." in name.split("/")
        or not isinstance(crc32c, str)
        or not crc32c
    ):
        raise ReclamationError(f"{field} object identity is invalid")
    return {
        "crc32c": crc32c,
        "generation": _positive_int(
            value.get("generation"), field=f"{field}.generation"
        ),
        "name": name,
        "size": _positive_int(value.get("size"), field=f"{field}.size"),
    }


def _validate_dependency_safety(
    artifact_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    """Reject lineage that would delete a retained artifact's required state."""

    for artifact_id, artifact in artifact_by_id.items():
        recipe = artifact.get("recipe")
        if recipe is None:
            continue
        normalized_recipe = _lineage_record(artifact, field=artifact_id)["recipe"]
        for dependency in normalized_recipe["dependencies"]:
            dependency_id = dependency["artifact_id"]
            if dependency_id not in artifact_by_id:
                raise ReclamationError(
                    f"{artifact_id} names unknown dependency {dependency_id}"
                )
            target = artifact_by_id[dependency_id]
            if dependency["required_scope"] == "full_payload":
                if target["disposition"] in ("delete_now", "delete_payloads"):
                    raise ReclamationError(
                        f"{artifact_id} full-payload dependency {dependency_id} "
                        "is scheduled for deletion"
                    )
                continue
            preserved = set(target.get("preserve_metadata_objects", []))
            required = set(dependency["required_objects"])
            if target["disposition"] == "delete_now" or not required <= preserved:
                raise ReclamationError(
                    f"{artifact_id} metadata-only dependency {dependency_id} is not "
                    "preserved exactly"
                )


def validate_policy(policy: Mapping[str, Any]) -> None:
    """Validate the human-reviewed keep/delete boundary."""

    if policy.get("artifact_kind") != POLICY_ARTIFACT_KIND:
        raise ReclamationError("wrong retention policy artifact kind")
    if policy.get("format_version") != FORMAT_VERSION:
        raise ReclamationError("retention policy format version drifted")
    if policy.get("bucket") != APPROVED_BUCKET:
        raise ReclamationError("retention policy names an unapproved bucket")
    if policy.get("bucket_location") != APPROVED_LOCATION:
        raise ReclamationError("retention policy location drifted")

    protected = policy.get("protected_prefixes")
    artifacts = policy.get("artifacts")
    if not isinstance(protected, list) or not protected:
        raise ReclamationError("retention policy lacks protected prefixes")
    if not isinstance(artifacts, list) or not artifacts:
        raise ReclamationError("retention policy lacks artifacts")
    protected_prefixes = tuple(
        _prefix(value, field="protected_prefix") for value in protected
    )
    if len(set(protected_prefixes)) != len(protected_prefixes):
        raise ReclamationError("protected prefixes are duplicate")

    ids: set[str] = set()
    prefixes: set[str] = set()
    delete_count = 0
    artifact_by_id: dict[str, Mapping[str, Any]] = {}
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ReclamationError("artifact policy record is not an object")
        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in ids:
            raise ReclamationError("artifact ids are invalid or duplicate")
        prefix = _prefix(artifact.get("prefix"), field=f"{artifact_id}.prefix")
        if prefix in prefixes or any(
            prefix.startswith(other) or other.startswith(prefix) for other in prefixes
        ):
            raise ReclamationError("artifact prefixes overlap")
        disposition = artifact.get("disposition")
        if disposition not in (
            "keep",
            "delete_now",
            "delete_payloads",
            "keep_until_standalone",
        ):
            raise ReclamationError(f"{artifact_id} has an invalid disposition")
        if disposition in ("delete_now", "delete_payloads"):
            delete_count += 1
            for protected_prefix in protected_prefixes:
                if prefix.startswith(protected_prefix) or protected_prefix.startswith(
                    prefix
                ):
                    raise ReclamationError(
                        f"delete artifact {artifact_id} overlaps protected prefix"
                    )
        _positive_int(
            artifact.get("expected_object_count"),
            field=f"{artifact_id}.expected_object_count",
        )
        _positive_int(
            artifact.get("expected_bytes"), field=f"{artifact_id}.expected_bytes"
        )
        _lineage_record(artifact, field=artifact_id)
        ids.add(artifact_id)
        prefixes.add(prefix)
        artifact_by_id[artifact_id] = artifact
    if delete_count == 0:
        raise ReclamationError("retention policy authorizes no deletion candidates")
    _validate_dependency_safety(artifact_by_id)


def build_reproducibility_capsule(
    policy: Mapping[str, Any],
    objects_by_artifact: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    generator_code_hash: str,
    created_utc: str,
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a reviewed policy to exact live object generations and proof."""

    validate_policy(policy)
    _digest(generator_code_hash, field="generator_code_hash", lengths=(40,))
    if not isinstance(created_utc, str) or not created_utc.endswith("Z"):
        raise ReclamationError("created_utc must be a UTC timestamp")
    if proof.get("status") != "passed":
        raise ReclamationError("bounded recreation proof has not passed")
    _digest(proof.get("proof_sha256"), field="proof.proof_sha256")

    expected_ids = {item["id"] for item in policy["artifacts"]}
    if set(objects_by_artifact) != expected_ids:
        raise ReclamationError("live artifact set disagrees with policy")
    artifact_records = []
    seen_names: set[str] = set()
    for artifact in policy["artifacts"]:
        artifact_id = artifact["id"]
        prefix = artifact["prefix"]
        objects = sorted(
            (
                _object_identity(value, field=f"{artifact_id}.object")
                for value in objects_by_artifact[artifact_id]
            ),
            key=lambda value: value["name"],
        )
        if len(objects) != artifact["expected_object_count"]:
            raise ReclamationError(f"{artifact_id} object count drifted")
        if sum(item["size"] for item in objects) != artifact["expected_bytes"]:
            raise ReclamationError(f"{artifact_id} byte total drifted")
        if any(not item["name"].startswith(prefix) for item in objects):
            raise ReclamationError(f"{artifact_id} contains an out-of-prefix object")
        names = {item["name"] for item in objects}
        if len(names) != len(objects) or names & seen_names:
            raise ReclamationError("artifact object names are duplicate or overlap")
        seen_names.update(names)
        terminal_objects = artifact.get("terminal_objects", [])
        if not isinstance(terminal_objects, list) or any(
            f"{prefix}{name}" not in names for name in terminal_objects
        ):
            raise ReclamationError(f"{artifact_id} terminal objects are incomplete")
        preserve_metadata = artifact.get("preserve_metadata_objects", [])
        if not isinstance(preserve_metadata, list) or any(
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or ".." in name.split("/")
            or f"{prefix}{name}" not in names
            for name in preserve_metadata
        ):
            raise ReclamationError(
                f"{artifact_id} preserved metadata object set is invalid"
            )
        if artifact["disposition"] == "delete_payloads" and not preserve_metadata:
            raise ReclamationError(f"{artifact_id} deletion lacks preserved metadata")
        expected_manifest = artifact.get("expected_manifest_sha256")
        if expected_manifest is not None:
            _digest(
                expected_manifest,
                field=f"{artifact_id}.expected_manifest_sha256",
            )
            manifest_object = artifact.get("manifest_object")
            if (
                not isinstance(manifest_object, str)
                or f"{prefix}{manifest_object}" not in names
            ):
                raise ReclamationError(f"{artifact_id} manifest object is incomplete")
        record = {
            "bytes": sum(item["size"] for item in objects),
            "disposition": artifact["disposition"],
            "expected_manifest_sha256": artifact.get("expected_manifest_sha256"),
            "id": artifact_id,
            "object_count": len(objects),
            "objects": objects,
            "prefix": prefix,
            "preserve_metadata_objects": list(preserve_metadata),
            "terminal_objects": list(terminal_objects),
        }
        record.update(_lineage_record(artifact, field=artifact_id))
        artifact_records.append(record)
    capsule: dict[str, Any] = {
        "artifact_kind": CAPSULE_ARTIFACT_KIND,
        "artifacts": artifact_records,
        "bucket": APPROVED_BUCKET,
        "bucket_location": APPROVED_LOCATION,
        "created_utc": created_utc,
        "format_version": FORMAT_VERSION,
        "generator_code_hash": generator_code_hash,
        "proof": deepcopy(dict(proof)),
        "protected_prefixes": list(policy["protected_prefixes"]),
    }
    capsule["capsule_sha256"] = _mapping_hash(capsule, hash_field="capsule_sha256")
    validate_capsule(capsule)
    return capsule


def capsule_sha256(capsule: Mapping[str, Any]) -> str:
    return _mapping_hash(capsule, hash_field="capsule_sha256")


def validate_capsule(capsule: Mapping[str, Any]) -> None:
    """Validate a capsule without trusting its creator or filename."""

    if capsule.get("artifact_kind") != CAPSULE_ARTIFACT_KIND:
        raise ReclamationError("wrong reproducibility capsule artifact kind")
    if capsule.get("format_version") != FORMAT_VERSION:
        raise ReclamationError("reproducibility capsule version drifted")
    if capsule.get("bucket") != APPROVED_BUCKET:
        raise ReclamationError("capsule names an unapproved bucket")
    if capsule.get("bucket_location") != APPROVED_LOCATION:
        raise ReclamationError("capsule bucket location drifted")
    observed_hash = capsule_sha256(capsule)
    if capsule.get("capsule_sha256") != observed_hash:
        raise ReclamationError("capsule SHA-256 drifted")
    _digest(
        capsule.get("generator_code_hash"), field="generator_code_hash", lengths=(40,)
    )
    proof = capsule.get("proof")
    if not isinstance(proof, Mapping) or proof.get("status") != "passed":
        raise ReclamationError("capsule lacks a passed recreation proof")
    _digest(proof.get("proof_sha256"), field="proof.proof_sha256")
    protected = capsule.get("protected_prefixes")
    artifacts = capsule.get("artifacts")
    if not isinstance(protected, list) or not isinstance(artifacts, list):
        raise ReclamationError("capsule policy boundary is missing")
    protected_prefixes = tuple(
        _prefix(value, field="protected_prefix") for value in protected
    )
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    artifact_by_id: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ReclamationError("capsule artifact record is invalid")
        artifact_id = artifact.get("id")
        if not isinstance(artifact_id, str) or artifact_id in seen_ids:
            raise ReclamationError("capsule artifact ids are invalid or duplicate")
        prefix = _prefix(artifact.get("prefix"), field=f"{artifact_id}.prefix")
        disposition = artifact.get("disposition")
        if disposition not in (
            "keep",
            "delete_now",
            "delete_payloads",
            "keep_until_standalone",
        ):
            raise ReclamationError(f"{artifact_id} disposition drifted")
        objects = artifact.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ReclamationError(f"{artifact_id} object ledger is empty")
        normalized = [
            _object_identity(item, field=f"{artifact_id}.object") for item in objects
        ]
        names = {item["name"] for item in normalized}
        if len(names) != len(normalized) or names & seen_names:
            raise ReclamationError("capsule objects are duplicate or overlap")
        if any(not name.startswith(prefix) for name in names):
            raise ReclamationError(f"{artifact_id} contains an out-of-prefix object")
        terminal_objects = artifact.get("terminal_objects")
        if not isinstance(terminal_objects, list) or any(
            not isinstance(name, str) or f"{prefix}{name}" not in names
            for name in terminal_objects
        ):
            raise ReclamationError(f"{artifact_id} terminal object set drifted")
        _lineage_record(artifact, field=artifact_id)
        if artifact.get("object_count") != len(normalized):
            raise ReclamationError(f"{artifact_id} object count drifted")
        if artifact.get("bytes") != sum(item["size"] for item in normalized):
            raise ReclamationError(f"{artifact_id} byte total drifted")
        preserve_metadata = artifact.get("preserve_metadata_objects", [])
        if not isinstance(preserve_metadata, list) or any(
            not isinstance(name, str) or f"{prefix}{name}" not in names
            for name in preserve_metadata
        ):
            raise ReclamationError(f"{artifact_id} preserved metadata set drifted")
        if disposition in ("delete_now", "delete_payloads"):
            for protected_prefix in protected_prefixes:
                if prefix.startswith(protected_prefix) or protected_prefix.startswith(
                    prefix
                ):
                    raise ReclamationError(
                        f"delete artifact {artifact_id} overlaps protected prefix"
                    )
        seen_ids.add(artifact_id)
        seen_names.update(names)
        artifact_by_id[artifact_id] = artifact
    _validate_dependency_safety(artifact_by_id)


def deletion_order(capsule: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return exact delete candidates with terminal metadata first."""

    validate_capsule(capsule)
    candidates: list[dict[str, Any]] = []
    for artifact in capsule["artifacts"]:
        if artifact["disposition"] not in ("delete_now", "delete_payloads"):
            continue
        prefix = artifact["prefix"]
        terminal = {f"{prefix}{name}" for name in artifact["terminal_objects"]}
        preserved = {
            f"{prefix}{name}" for name in artifact["preserve_metadata_objects"]
        }
        for item in artifact["objects"]:
            if (
                artifact["disposition"] == "delete_payloads"
                and item["name"] in preserved
            ):
                continue
            record = dict(item)
            record["artifact_id"] = artifact["id"]
            record["terminal"] = item["name"] in terminal
            candidates.append(record)
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                0 if item["terminal"] else 1,
                item["artifact_id"],
                item["name"],
            ),
        )
    )


def reconcile_live_objects(
    expected: Iterable[Mapping[str, Any]],
    observed: Iterable[Mapping[str, Any]],
) -> None:
    """Require exact name/size/generation/CRC identity before mutation."""

    expected_values = [_object_identity(item, field="expected") for item in expected]
    observed_values = [_object_identity(item, field="observed") for item in observed]
    expected_by_name = {item["name"]: item for item in expected_values}
    observed_by_name = {item["name"]: item for item in observed_values}
    if len(expected_by_name) != len(expected_values) or len(observed_by_name) != len(
        observed_values
    ):
        raise ReclamationError("live object identity contains duplicate names")
    if expected_by_name != observed_by_name:
        missing = sorted(set(expected_by_name) - set(observed_by_name))
        extra = sorted(set(observed_by_name) - set(expected_by_name))
        drifted = sorted(
            name
            for name in set(expected_by_name) & set(observed_by_name)
            if expected_by_name[name] != observed_by_name[name]
        )
        raise ReclamationError(
            "live object identity drifted: "
            f"missing={missing[:3]} extra={extra[:3]} drifted={drifted[:3]}"
        )
