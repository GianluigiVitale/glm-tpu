"""The source completion marker ``SOURCE_COMPLETE.json``: a local GLM-5.3 source whose every file equals upstream.

:func:`mark_source` hashes every safetensors shard the index names and the metadata files the engine reads
(:data:`METADATA_FILES`), compares each file with an upstream listing and writes the marker as a new file only when
every file, the shard count and the byte total agree. An upstream listing maps a file name to its size and its
SHA-256 or, for a file Git stores directly, its git blob id: :func:`hub_listing` asks the Hugging Face repository at
the pinned revision (the LFS SHA-256 of every shard); :func:`marker_listing` takes the digests of an earlier marker, to
check another copy of the same source against it.

Workers and the pack worker check the marker's SHA-256 (the site's ``checkpoint.source_complete_sha256``) and its
``passed``, ``repository``, ``revision``, ``verified_shards`` and ``verified_bytes`` fields
(``glm_tpu.config.site.site_args``, ``glm_tpu.model_loader.pack_worker``). Reads only; the one write is the marker,
created once, owner-only. Importing this module imports no JAX.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
from typing import Any

from glm_tpu.config.model import MODEL_ID, REVISION, SOURCE_BYTES, SOURCE_SHARDS
from glm_tpu.exceptions import CheckpointValidationError
from glm_tpu.utils.io_utils import create_private_exclusive, read_bounded

SCHEMA = "glm_tpu_source_complete_v1"
MARKER = "SOURCE_COMPLETE.json"
# The files besides the shards that the engine reads from the source directory (the tokenizer, the configuration
# and the index the inventory is built from) or that it pins in its package data.
METADATA_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
CHUNK_BYTES = 16 << 20
MARKER_CAP = 1 << 20

Listing = Mapping[str, Mapping[str, Any]]


def hash_file(path: Path) -> dict[str, Any]:
    """``size``, ``sha256`` and ``git_blob_sha1`` (Git's object id of the bytes) of one file, in one read."""
    with Path(path).open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        content, blob = sha256(), sha1(b"blob %d\0" % size)
        read = 0
        while chunk := stream.read(CHUNK_BYTES):
            content.update(chunk)
            blob.update(chunk)
            read += len(chunk)
    if read != size:
        raise CheckpointValidationError(f"source file {path} changed size while it was hashed")
    return dict(size=size, sha256=content.hexdigest(), git_blob_sha1=blob.hexdigest())


def hub_listing(repository: str = MODEL_ID, revision: str = REVISION) -> dict[str, dict[str, Any]]:
    """The top-level files of the Hugging Face repository at ``revision``: size and LFS SHA-256 of each LFS file,
    size and git blob id of every other file (network; no download)."""
    from huggingface_hub import HfApi

    listing = {}
    for item in HfApi().list_repo_tree(repository, revision=revision, recursive=False):
        size = getattr(item, "size", None)
        if size is None:  # a folder
            continue
        lfs = getattr(item, "lfs", None)
        listing[item.path] = dict(
            size=size,
            sha256=None if lfs is None else lfs.sha256,
            git_blob_sha1=item.blob_id if lfs is None else None,
        )
    if not listing:
        raise CheckpointValidationError(f"the upstream repository {repository} at {revision} lists no file")
    return listing


def marker_listing(path: Path, *, repository: str = MODEL_ID, revision: str = REVISION) -> dict[str, dict[str, Any]]:
    """The shard and metadata digests of an earlier marker of the same source (``passed`` true, same repository and
    revision)."""
    marker = json.loads(read_bounded(Path(path), MARKER_CAP))
    if (
        not isinstance(marker, dict)
        or marker.get("passed") is not True
        or marker.get("repository") != repository
        or marker.get("revision") != revision
        or not isinstance(marker.get("shards"), list)
        or not isinstance(marker.get("metadata"), list)
    ):
        raise CheckpointValidationError(f"{path} is not a passed completion marker of {repository} at {revision}")
    listing = {}
    for entry in [*marker["shards"], *marker["metadata"]]:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or entry["name"] in listing:
            raise CheckpointValidationError(f"{path} lists a file entry twice or without a name")
        listing[entry["name"]] = dict(size=entry.get("bytes"), sha256=entry.get("sha256"), git_blob_sha1=None)
    return listing


def _differences(name: str, local: Mapping[str, Any], upstream: Mapping[str, Any] | None) -> list[str]:
    if upstream is None:
        return [f"{name}: not in the upstream listing"]
    problems = []
    if local["size"] != upstream.get("size"):
        problems.append(f"{name}: {local['size']} bytes, upstream {upstream.get('size')}")
    if upstream.get("sha256") is not None:
        if local["sha256"] != upstream["sha256"]:
            problems.append(f"{name}: SHA-256 differs from upstream")
    elif upstream.get("git_blob_sha1") is not None:
        if local["git_blob_sha1"] != upstream["git_blob_sha1"]:
            problems.append(f"{name}: git blob id differs from upstream")
    else:
        problems.append(f"{name}: the upstream listing has no digest")
    return problems


def mark_source(
    source: Path,
    output: Path,
    *,
    upstream: Listing,
    upstream_kind: str,
    repository: str = MODEL_ID,
    revision: str = REVISION,
    expected_shards: int = SOURCE_SHARDS,
    expected_bytes: int = SOURCE_BYTES,
    workers: int = 8,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> dict[str, Any]:
    """Verify the source in ``source`` against ``upstream`` and write the completion marker to ``output`` (a new
    file). Refuses, writing nothing, unless the index names exactly the upstream safetensors files, their count is
    ``expected_shards``, their bytes add up to ``expected_bytes``, every metadata file exists and every file's size
    and digest equal upstream. Returns the report (with the marker's SHA-256, the site pin)."""
    from glm_tpu.model_loader.source_inventory import read_source_inventory

    source, output = Path(source), Path(output)
    if not 1 <= workers <= 64:
        raise ValueError("hash workers must be 1..64")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite the completion marker {output}")
    started = now().isoformat(timespec="seconds")
    inventory = read_source_inventory(source, model_id=repository, source_revision=revision)
    shards = {record.filename: record for record in inventory.files}
    upstream_shards = sorted(name for name in upstream if name.endswith(".safetensors"))
    problems = []
    if sorted(shards) != upstream_shards:
        missing = sorted(set(upstream_shards) - set(shards))[:3]
        extra = sorted(set(shards) - set(upstream_shards))[:3]
        problems.append(
            f"the index names {len(shards)} shards, upstream lists {len(upstream_shards)}: missing {missing}, "
            f"extra {extra}"
        )
    if len(shards) != expected_shards:
        problems.append(f"{len(shards)} shards, {expected_shards} expected")
    total = sum(record.file_bytes for record in shards.values())
    if total != expected_bytes:
        problems.append(f"{total} shard bytes, {expected_bytes} expected")
    missing = [name for name in METADATA_FILES if not (source / name).is_file()]
    problems += [f"{name}: missing from the source" for name in missing]
    if problems:
        raise CheckpointValidationError("source is not complete: " + "; ".join(problems[:8]))
    names = [*sorted(shards), *METADATA_FILES]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashed = dict(zip(names, pool.map(lambda name: hash_file(source / name), names), strict=True))
    for name in names:
        problems += _differences(name, hashed[name], upstream.get(name))
    for name, record in shards.items():
        if hashed[name]["size"] != record.file_bytes:
            problems.append(f"{name}: changed size after its header was read")
    if problems:
        raise CheckpointValidationError(
            f"source differs from upstream in {len(problems)} checks: " + "; ".join(problems[:8])
        )
    marker = dict(
        schema=SCHEMA,
        passed=True,
        repository=repository,
        revision=revision,
        expected_shards=expected_shards,
        verified_shards=len(shards),
        verified_bytes=total,
        upstream=upstream_kind,
        shards=[
            dict(
                name=name,
                bytes=hashed[name]["size"],
                sha256=hashed[name]["sha256"],
                header_sha256=shards[name].header_sha256,
                tensor_count=shards[name].tensor_count,
            )
            for name in sorted(shards)
        ],
        metadata=[
            dict(name=name, bytes=hashed[name]["size"], sha256=hashed[name]["sha256"]) for name in METADATA_FILES
        ],
        started_utc=started,
        finished_utc=now().isoformat(timespec="seconds"),
        scope=(
            "every shard the index names and every metadata file equal the upstream listing in size and digest; "
            "nothing about the packed checkpoint or inference"
        ),
    )
    raw = (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode()
    create_private_exclusive(output, raw)
    return dict(
        schema="glm_tpu_source_complete_report_v1",
        output=str(output),
        marker_sha256=sha256(raw).hexdigest(),
        repository=repository,
        revision=revision,
        upstream=upstream_kind,
        verified_shards=len(shards),
        verified_bytes=total,
        metadata_files=len(METADATA_FILES),
        passed=True,
    )
