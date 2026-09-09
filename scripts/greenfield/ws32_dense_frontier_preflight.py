"""Selected dense01 retained preflight, invoked by the existing layer campaign.

No TPU initialization or payload read. Runtime must rebind actual owners and
load/check the selected bytes; this report alone cannot authorize execution.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import socket
from typing import Any

from glm_tpu.greenfield.checkpoint.ws32_layer_subset import (
    Ws32LayerSubsetMetadata, read_ws32_layer_subset_metadata,
)
from glm_tpu.greenfield.types import ModelGeometry
from glm_tpu.greenfield.validation import ws32_prefill_admission as admission
from scripts.greenfield.microbench_fp8_matmul import _atomic_json
from scripts.greenfield import ws32_dense_frontier_protocol as protocol

RUN_ROOT = Path("/home/gianl/glm-run")


def selected_metadata(repo: Path, slots: tuple[int, ...], *, canonical_dense: bool = False) -> tuple[dict, Ws32LayerSubsetMetadata]:
    """Original schema/header/source guards, explicit two layers + embedding."""
    from scripts.greenfield.probe_ws32_prefill_layer import authenticated_inventory

    if canonical_dense:
        from scripts.greenfield.ws32_dense_canonical import require_source
        require_source(repo)
    else:
        admission.require_acquired_model_source(repo, profile=admission.ROLLED_SHORT_PROFILE)
    pins = json.loads((repo / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json").read_text())
    inventory = authenticated_inventory(Path(pins["source_inventory"]), pins["source_inventory_sha256"])
    checkpoint = Path(pins["checkpoint_root"])
    for name, key in (("manifest.json", "manifest_file_sha256"), ("SUCCESS", "success_file_sha256")):
        if sha256((checkpoint / name).read_bytes()).hexdigest() != pins[key]:
            raise ValueError("dense retained checkpoint metadata file differs")
    geometry = ModelGeometry.from_hf_config(json.loads((repo / "configs/glm-5.2-fp8-config.json").read_text()))
    subset = read_ws32_layer_subset_metadata(
        checkpoint, layer_ids=protocol.LAYERS, include_embedding=True, local_slots=slots,
        max_payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
        expected_manifest_sha256=pins["expected_manifest_sha256"],
        expected_success_sha256=pins["expected_success_sha256"],
        expected_mesh_hash=pins["expected_mesh_sha256"],
        expected_topology_hash=pins["expected_topology_sha256"],
        inventory=inventory, geometry=geometry,
    )
    if (len(subset.tensor_indices) != protocol.SELECTED_LEAVES
            or subset.payload_bytes_per_chip != protocol.PAYLOAD_BYTES):
        raise ValueError("dense selected inventory or exact byte count differs")
    return pins, subset


def retained_preflight(*, tag: str, rank: int, pin: str, root: Path, repo: Path, client: Any) -> None:
    """Before any fleet TPU initialization, authenticate own originals/headers."""
    protocol.original_names(rank)
    from scripts.greenfield import ws32_dense_norm_protocol as norm_protocol
    from scripts.greenfield import ws32_dense_canonical as canonical
    norm_mode = norm_protocol.is_tag(tag)
    canonical_mode = canonical.is_tag(tag)
    if (not (protocol.is_tag(tag) or norm_mode or canonical_mode) or not re.fullmatch(r"[0-9a-f]{40}", pin)
            or root != RUN_ROOT / tag / f"rank{rank}" or root.is_symlink()):
        raise ValueError("dense retained tag/path differs")
    output = root / "retained_preflight.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    root.mkdir(parents=True, exist_ok=True)
    prior, witness = protocol.materialize_reference(root / "retained_reference", rank=rank, client=client)
    if prior["hostname"] != socket.gethostname():
        raise ValueError("dense retained source belongs to a different host")
    slots = tuple(sorted({key[1] for key in witness}))
    pins, subset = selected_metadata(repo, slots, **(dict(canonical_dense=True) if canonical_mode else {}))
    metadata = subset.metadata
    if (metadata.manifest["manifest_sha256"] != prior["checkpoint_manifest_sha256"]
            or metadata.manifest["source"]["inventory_sha256"] != prior["source_inventory_sha256"]
            or pins["expected_success_sha256"] != prior["checkpoint_success_sha256"]
            or pins["expected_mesh_sha256"] != prior["mesh_sha256"]
            or pins["expected_topology_sha256"] != prior["topology_sha256"]):
        raise ValueError("dense selected checkpoint/topology differs from original")
    for owner in prior["local_device_slots"]:
        if owner["file_sha256"] != metadata.records_by_slot[owner["device_slot"]]["sha256"]:
            raise ValueError("dense original owner-file ledger differs")
    runner_sha = sha256((root / "retained_reference" / protocol.original_names(rank)[0]).read_bytes()).hexdigest()
    norm_identity = {}
    if norm_mode or canonical_mode:
        from scripts.greenfield import ws32_dense_norm_originals as norm
        existing_bytes = protocol.LEDGER_PIN["size"] + sum(
            v["size"] for v in protocol.reference_pins(root / "retained_reference", rank).values())
        norm_runner, originals, identity = norm.materialize(
            root / "retained_norm_reference", repo=repo, rank=rank, client=client,
            existing_reference_bytes=existing_bytes)
        norm.bind_prior(norm_runner, prior, prior_sha256=runner_sha)
        norm_identity = dict(norm_originals=identity, combined_reference_bytes=existing_bytes+identity["bytes"])
        del originals
    _atomic_json(output, dict(
        protocol=canonical.PROTOCOL if canonical_mode else norm_protocol.PROTOCOL if norm_mode else protocol.PROTOCOL,
        tag=tag, code_hash=pin, launch_rank=rank,
        hostname=socket.gethostname(), selected_layer_ids=list(protocol.LAYERS),
        include_embedding=True, context_capacity=8192, host_main_rope_table=True,
        checkpoint_pins=pins, local_device_slots=prior["local_device_slots"],
        selected_leaf_count=protocol.SELECTED_LEAVES, payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
        headers=[{key: metadata.records_by_slot[slot][key]
                  for key in ("device_slot", "filename", "file_bytes", "header_sha256")} for slot in slots],
        original_tag=protocol.ORIGINAL_TAG, original_ledger_sha256=protocol.LEDGER_SHA,
        original_runner_sha256=runner_sha, **norm_identity,
        scope="HEADERS_AND_RETAINED_ORIGINALS_ONLY_NOT_SELECTED_PAYLOAD_OR_LIVE_TOPOLOGY",
        numerical_promotion=False, performance_claim=False,
    ))
    print(f"PREFILL_RETAINED_OK {socket.gethostname()}", flush=True)
