"""The seal of a packed checkpoint: the SUCCESS record, the comparison of owner receipts with a sealed manifest, the
payload canary, and the installation of ``manifest.json`` and ``SUCCESS`` on a host.

:func:`success_record` builds the self-hashed ``SUCCESS`` of a manifest exactly as ``verify_runtime_checkpoint``
checks it, with the digests of the pack run's own preflight, terminal and post-run idle records. :func:`compare_records`
is the recovery check: every file record the eight hosts' pack receipts hold must equal the sealed manifest's record of
that slot on every key, and the receipts' mesh, geometry, placement and source-inventory digests the manifest's.
:func:`compare_plans` compares the 32 file plans re-derived from the inventory and the geometry with a manifest
(headers, sizes, coordinates, names), and :func:`rederive_tensors` builds chosen destination tensors of one slot in
memory from the source, slice by slice as ``pack_runtime_slots`` places them, for a comparison with the manifest's
tensor digests that writes nothing (:func:`canary_tensors` chooses them). :func:`install_seal` creates
``manifest.json``, then ``SUCCESS``, each once and owner-only, in a host's checkpoint root.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
from typing import Any

from glm_tpu.model_loader.sharded_state.format import (
    RUNTIME_FORMAT_VERSION,
    SUCCESS_ARTIFACT_KIND,
    SUCCESS_TAG,
    RuntimeFilePlan,
    mapping_hash,
    require_digest,
)
from glm_tpu.utils.io_utils import create_private_exclusive

RECORD_DIGESTS = ("mesh_hash", "geometry_sha256", "placement_sha256", "source_inventory_sha256")


def json_bytes(value: Mapping[str, Any]) -> bytes:
    """``manifest.json`` and ``SUCCESS`` as written: indented, sorted keys, a final newline."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def success_record(
    manifest: Mapping[str, Any],
    manifest_raw: bytes,
    *,
    tag: str,
    topology_hash: str,
    post_census_sha256: str,
    remote_preflight_sha256: str,
    remote_terminal_sha256: str,
) -> dict[str, Any]:
    """The ``SUCCESS`` seal of ``manifest`` (whose file bytes are ``manifest_raw``), self-hashed."""
    if SUCCESS_TAG.fullmatch(tag) is None:
        raise ValueError("a seal's tag must be the name of its pack run (the SUCCESS tag format)")
    if json.loads(manifest_raw) != manifest:
        raise ValueError("the manifest bytes are not this manifest")
    for name, digest in (
        ("topology_hash", topology_hash),
        ("post_census_sha256", post_census_sha256),
        ("remote_preflight_sha256", remote_preflight_sha256),
        ("remote_terminal_sha256", remote_terminal_sha256),
    ):
        require_digest(digest, field=name)
    value: dict[str, Any] = {
        "artifact_kind": SUCCESS_ARTIFACT_KIND,
        "code_hash": manifest["code_hash"],
        "file_count": len(manifest["files"]),
        "format_version": RUNTIME_FORMAT_VERSION,
        "manifest_file_sha256": sha256(manifest_raw).hexdigest(),
        "manifest_sha256": manifest["manifest_sha256"],
        "mesh_hash": manifest["mesh_hash"],
        "packed_payload_bytes": manifest["packed_payload_bytes"],
        "performance_claim": False,
        "post_census_sha256": post_census_sha256,
        "remote_preflight_sha256": remote_preflight_sha256,
        "remote_terminal_sha256": remote_terminal_sha256,
        "source_file_count": len(manifest["source"]["files"]),
        "source_inventory_sha256": manifest["source"]["inventory_sha256"],
        "tag": tag,
        "topology_hash": topology_hash,
        "tpu_initialized": False,
    }
    value["success_sha256"] = mapping_hash(value, field="success_sha256")
    return value


def compare_records(
    manifest: Mapping[str, Any], owner_records: Sequence[Mapping[str, Any]], host_to_slots: Mapping[str, Sequence[int]]
) -> list[str]:
    """The differences between the eight hosts' owner receipts (``pack_runtime_slots`` records, in rank order) and a
    sealed manifest; empty when every one of the 32 file records equals the manifest's on every key, each slot is
    covered once by the host that owns it, and every receipt's digests are the manifest's."""
    sealed = {record["device_slot"]: record for record in manifest["files"]}
    expected = dict(
        mesh_hash=manifest["mesh_hash"],
        geometry_sha256=manifest["geometry_sha256"],
        placement_sha256=manifest["placement_report"]["placement_sha256"],
        source_inventory_sha256=manifest["source"]["inventory_sha256"],
    )
    problems, seen = [], []
    if len(owner_records) != len(host_to_slots):
        problems.append(f"{len(owner_records)} owner receipts for {len(host_to_slots)} hosts")
    for rank, record in enumerate(owner_records):
        problems += [f"rank {rank}: {key} differs" for key in RECORD_DIGESTS if record.get(key) != expected[key]]
        slots = sorted(file["device_slot"] for file in record["files"])
        if slots != sorted(host_to_slots.get(str(rank), [])):
            problems.append(f"rank {rank}: slots {slots} are not the host's {sorted(host_to_slots.get(str(rank), []))}")
        for file in record["files"]:
            seen.append(file["device_slot"])
            want = sealed.get(file["device_slot"], {})
            differing = sorted(key for key in set(want) | set(file) if want.get(key) != file.get(key))
            if differing:
                problems.append(f"slot {file['device_slot']}: {differing} differ")
    if sorted(seen) != sorted(sealed):
        problems.append(f"the receipts cover slots {sorted(seen)}, the manifest {sorted(sealed)}")
    return problems


def compare_plans(plans: Sequence[RuntimeFilePlan], manifest: Mapping[str, Any]) -> list[str]:
    """The differences between the file plans (``build_runtime_file_plans``) and a manifest's file records: header
    digest and size, payload and file bytes, coordinates, file name and tensor count of every slot."""
    records = {record["device_slot"]: record for record in manifest["files"]}
    problems = []
    if sorted(records) != [plan.device_slot for plan in plans]:
        problems.append("the manifest's slots are not the plans' slots")
    for plan in plans:
        record = records.get(plan.device_slot, {})
        observed = dict(
            header_sha256=sha256(plan.header).hexdigest(),
            header_bytes=len(plan.header),
            payload_bytes=plan.payload_bytes,
            file_bytes=plan.file_bytes,
            expert_coordinate=plan.expert_coordinate,
            feature_coordinate=plan.feature_coordinate,
            filename=plan.filename,
        )
        differing = sorted(key for key, value in observed.items() if record.get(key) != value)
        if len(record.get("tensor_sha256", [])) != len(plan.tensors):
            differing.append("tensor count")
        if differing:
            problems.append(f"slot {plan.device_slot}: {differing} differ")
    return problems


def canary_tensors(plan: RuntimeFilePlan, count: int) -> list[int]:
    """Indices of ``count`` tensors of ``plan`` to re-derive (0: every tensor): the smallest FP8 (U8) weight, the
    smallest BF16 tensor, the two smallest routed-expert tensors and routed-expert U8 weights, the smallest U8 tensor
    sharded on each partition axis, then the smallest others, so that every byte class and slicing is exercised."""
    tensors = plan.tensors
    if count == 0:
        return list(range(len(tensors)))
    if count < 0:
        raise ValueError("the canary tensor count must be 0 (all) or positive")
    by_size = sorted(range(len(tensors)), key=lambda i: (tensors[i].byte_count, tensors[i].name))
    picked: list[int] = []

    def add(candidates: list[int], n: int = 1) -> None:
        for index in candidates:
            if n == 0 or len(picked) == count:
                return
            if index not in picked:
                picked.append(index)
                n -= 1

    add([i for i in by_size if tensors[i].dtype == "U8"])
    add([i for i in by_size if tensors[i].dtype == "BF16"])
    add([i for i in by_size if ".mlp.experts." in tensors[i].name], 2)
    add([i for i in by_size if ".mlp.experts." in tensors[i].name and tensors[i].dtype == "U8"], 2)
    for axis in sorted({a for i in by_size for a in tensors[i].partition_spec if a is not None}):
        add([i for i in by_size if axis in tensors[i].partition_spec and tensors[i].dtype == "U8"])
    add(by_size, count)
    return picked


def rederive_tensors(source_root: Path, inventory, geometry, plan: RuntimeFilePlan, indices: Sequence[int]) -> dict:
    """The SHA-256 of tensors ``indices`` of ``plan``, each built in memory from the source safetensors files as
    ``pack_runtime_slots`` places it (a read-only memory map of every source leaf, the placement ledger's slices at
    their flat offsets); returns ``{index: {"sha256": ..., "covered": bool}}``, covered when every byte was placed
    exactly once."""
    import numpy as np

    from glm_tpu.model_loader.placement import placements_for_source_tensor
    from glm_tpu.model_loader.sharded_state.writer import flat_contiguous_offset

    wanted = {plan.tensors[i].name: i for i in indices}
    buffers = {name: bytearray(plan.tensors[i].byte_count) for name, i in wanted.items()}
    written = dict.fromkeys(wanted, 0)
    files = {record.filename: record for record in inventory.files}
    for source in inventory.tensors:
        if source.layer_id is not None and source.layer_id >= geometry.num_layers:
            continue
        chosen = [
            placement
            for placement in placements_for_source_tensor(source, geometry)
            if placement.slot == plan.device_slot and placement.destination_name in wanted
        ]
        if not chosen:
            continue
        element_bytes = source.byte_count // prod(source.shape)
        mapped = np.memmap(
            Path(source_root) / source.filename,
            dtype=np.dtype(f"V{element_bytes}"),
            mode="r",
            offset=files[source.filename].header_bytes + source.data_offset_start,
            shape=source.shape,
            order="C",
        )
        try:
            for placement in chosen:
                slices = tuple(
                    slice(a, b) for a, b in zip(placement.source_starts, placement.source_stops, strict=True)
                )
                raw = mapped[slices].tobytes(order="C")
                start, count = flat_contiguous_offset(
                    placement.destination_shape, placement.destination_starts, placement.destination_stops
                )
                if len(raw) != placement.byte_count or count * element_bytes != len(raw):
                    raise ValueError(f"source slice size drifted for {source.name!r}")
                buffers[placement.destination_name][start * element_bytes : start * element_bytes + len(raw)] = raw
                written[placement.destination_name] += len(raw)
        finally:
            del mapped
    return {
        wanted[name]: dict(
            sha256=sha256(bytes(buffer)).hexdigest(), covered=written[name] == plan.tensors[wanted[name]].byte_count
        )
        for name, buffer in buffers.items()
    }


def install_seal(root: Path, manifest_raw: bytes, success_raw: bytes) -> None:
    """Create ``manifest.json``, then ``SUCCESS``, in the checkpoint root ``root``, each once (an existing file is
    refused) and owner-only, and sync the directory."""
    create_private_exclusive(Path(root) / "manifest.json", manifest_raw)
    create_private_exclusive(Path(root) / "SUCCESS", success_raw)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
