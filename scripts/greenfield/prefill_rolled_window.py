"""One combined layer realization against retained DB600 control originals.

No launcher or numerical admission. Reuse the existing selected-layer input/WK
builders, exact archived receipt, original assembly replay and bounded comparator.
No baseline execution, new checkpoint or independently canonical DSA claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from scripts.greenfield import prefill_completed_window_protocol as completed
from scripts.greenfield import prefill_panel_originals as panel
from scripts.greenfield import prefill_phase_originals as originals
from scripts.greenfield import prefill_window_protocol as window
from scripts.greenfield.prefill_layer_evidence import (
    INPUT_FIELDS,
    decode_arrays,
    equal_bytes,
    owner_inputs,
)
from scripts.greenfield.prefill_layer_numerical import FIELDS

PROTOCOL = "ws32-layer6-rolled128-panels-localmerge-retained-db600-v1"
KERNEL = "ws32_prefill_rolled_layer"
MAX_RANK_BYTES = 64 << 20
MAX_FLEET_BYTES = 8 * MAX_RANK_BYTES
REFERENCE_SCOPE = (
    "RETAINED_DB600_CONTROL_NEW_INDEPENDENT_LAYER_REALIZATION_NOT_FULL_MODEL"
)
PROGRAMS = ("wk_decode", "wk_promote", "candidate")
SEAL = Path(__file__).resolve().parents[2] / (
    "docs/artifacts/prefill-expert-panel-phase-db600-sealed-20260909.json"
)
SEAL_SHA = "2f2dd4e658fcba21d5025333e454482129dabec1c0d8e0c7cf20732eabb93246"
MAX_SOURCE_BYTES = 64 << 20


def is_tag(tag: str) -> bool:
    return (
        re.fullmatch(r"greenfield_fp8_" + KERNEL + r"_l6_[a-zA-Z0-9_]+", tag)
        is not None
    )


def materialize_reference(root: Path, *, rank: int) -> RetainedReference:
    """Pre-TPU, same-region exact-generation reads of four small original files.

    Existing files must match, never overwritten. No cloud writes or weight copy.
    Only this launch rank's originals are needed on each worker.
    """
    from google.cloud import storage

    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("retained reference requires launcher rank0..7")
    seal = originals._json_bound(SEAL, SEAL_SHA)
    bucket = storage.Client().bucket("driftbench-dsv4-uc")

    def fetch(relative: str, pin: Mapping[str, Any], digest_key: str) -> bytes:
        path = root / relative
        if path.exists():
            return _read_bound(path, pin[digest_key], pin["size"])
        if type(pin["size"]) is not int or not 0 < pin["size"] <= MAX_SOURCE_BYTES:
            raise ValueError("retained download size exceeds cap")
        generation = int(pin["generation"])
        blob = bucket.blob(f"results/{seal['tag']}/{relative}", generation=generation)
        blob.reload(if_generation_match=generation)
        if int(blob.size) != pin["size"] or blob.crc32c != pin["crc32c"]:
            raise ValueError("retained source generation size/CRC differs")
        raw = blob.download_as_bytes(if_generation_match=generation)
        if len(raw) != pin["size"] or sha256(raw).hexdigest() != pin[digest_key]:
            raise ValueError("retained downloaded original bytes differ")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        return raw

    for relative in ("SUCCESS", "archive_receipts.json"):
        fetch(relative, seal["source_objects"][relative], "sha256")
    ledger = json.loads((root / "archive_receipts.json").read_bytes())
    indexed = {r["name"]: r for r in ledger}
    if len(indexed) != len(ledger):
        raise ValueError("retained source ledger contains duplicate names")
    for name in ("runner.json", "phase_first.npz"):
        relative = f"fleet/rank{rank}/{name}"
        fetch(relative, indexed[f"results/{seal['tag']}/{relative}"], "original_sha256")
    return load_reference(root, rank=rank)


@dataclass(frozen=True)
class RetainedReference:
    inputs: Mapping[str, np.ndarray]
    controls: Mapping[int, Mapping[str, np.ndarray]]
    slots: Mapping[int, int]
    record: Mapping[str, Any]
    sources: tuple[Mapping[str, Any], ...]
    bounded_receipt: Mapping[str, Any]


def prepare_programs(*, mesh: Any, config: Any, weights: Any) -> tuple:
    """Reuse original20-input abstracts; return only WK/WK/new candidate.

    The legacy builder creates Python descriptors for its control; neither that
    control nor the unrolled candidate is lowered, compiled or executed here.
    Candidate-only options never reach the scalar-reference closure.
    """
    from scripts.greenfield.prefill_layer_programs import build_layer_programs
    from scripts.greenfield.prefill_window_acquisition import prepare_programs as old
    from scripts.greenfield.probe_ws32_prefill_layer import input_specs

    previous = old(mesh=mesh, config=config, weights=weights)
    values = previous[2][2]
    candidate, _ = build_layer_programs(
        mesh,
        input_specs(weights, values[14]),
        full_indexer=True,
        sparse_mlp=True,
        key_tile=window.KEY_TILE,
        candidate_window=True,
        paired_position_sort=True,
        rolled_prefix=True,
        expert_panels=True,
        sorted_local_merge=True,
        dsa_contract=config.dsa_contract,
        attention_contract=config.attention_contract,
        moe_contract=config.moe_contract,
        rms_norm_epsilon=config.rms_norm_epsilon,
    )
    return (*previous[:2], ("candidate", candidate, values))


def _read_bound(path: Path, digest: str, size: int) -> bytes:
    if (
        type(size) is not int
        or not 0 < size <= MAX_SOURCE_BYTES
        or path.is_symlink()
        or path.stat().st_size != size
    ):
        raise ValueError("retained source path/size differs")
    raw = path.read_bytes()
    if len(raw) != size or sha256(raw).hexdigest() != digest:
        raise ValueError("retained source bytes differ")
    return raw


def load_reference(root: Path, *, rank: int) -> RetainedReference:
    """Authenticate existing generation-bound files; never write/copy originals.

    `root` is the materialized DB600 archive, not a candidate-controlled receipt.
    The outer collector remains responsible for generation-qualified downloads.
    Only this host's two payloads and the compact ledger/SUCCESS are read.
    """
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("retained reference requires launcher rank0..7")
    seal = originals._json_bound(SEAL, SEAL_SHA)

    def sealed(relative: str) -> bytes:
        pin = seal["source_objects"][relative]
        return _read_bound(root / relative, pin["sha256"], pin["size"])

    sealed("SUCCESS")
    ledger = json.loads(sealed("archive_receipts.json"))
    indexed = {r["name"]: r for r in ledger}
    if len(indexed) != len(ledger):
        raise ValueError("retained ledger contains duplicate names")
    sources = []

    def archived(relative: str) -> bytes:
        name = f"results/{seal['tag']}/{relative}"
        item = indexed[name]
        raw = _read_bound(root / relative, item["original_sha256"], item["size"])
        sources.append(dict(item))
        return raw

    prefix = f"fleet/rank{rank}"
    record = json.loads(archived(prefix + "/runner.json"))
    if (
        record["status"] != "SUCCESS"
        or record["launch_rank"] != rank
        or record["code_hash"] != seal["code_hash"]
    ):
        raise ValueError("retained source run identity differs")
    slots = {r["device_id"]: r["device_slot"] for r in record["local_device_slots"]}
    capsule = originals.load_capsule()
    originals.bind_originals(record, slots, capsule)
    raw = archived(prefix + "/phase_first.npz")
    # Replay the immutable original assembly and unchanged original bounds first.
    receipt = panel.bounded_receipt(root / prefix / "phase_first.npz", slots)
    if receipt != seal["numerical_receipts"][str(rank)]:
        raise ValueError("retained original assembly receipt differs")
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        inputs = decode_arrays(arrays, "input", INPUT_FIELDS)
        if originals.manifest(inputs) != capsule["inputs"]:
            raise ValueError("retained original fixture differs")
        controls = {
            device: completed._decode(arrays, f"control_{device}", FIELDS)
            for device in slots
        }
        for device, fields in controls.items():
            originals.check_observation(
                capsule, slot=slots[device], kind="control", values=fields
            )
    return RetainedReference(inputs, controls, slots, record, tuple(sources), receipt)


def compare_observations(
    actual: Mapping[int, Mapping[str, np.ndarray]], reference: RetainedReference
) -> dict:
    """No changed tolerances: new complete result versus saved original control."""
    if set(actual) != set(reference.slots):
        raise ValueError("rolled candidate owners differ from retained reference")
    owners = {
        str(device): window.compare_case(
            actual[device],
            reference.controls[device],
            owner_inputs(reference.inputs, slot),
            slot=slot,
            case="competitive",
        )
        for device, slot in reference.slots.items()
    }
    return dict(
        protocol=PROTOCOL,
        passed=all(r["passed"] for r in owners.values()),
        owners=owners,
        source_seal_sha256=SEAL_SHA,
        source_db=600,
        reference_scope="RETAINED_ORIGINAL_CONTROL_NEW_INDEPENDENT_LAYER_REALIZATION",
        independent_canonical_dsa_claim=False,
        performance_claim=False,
    )


def replay_candidate(path: Path, reference: RetainedReference) -> dict:
    """Collector rederives from candidate originals; no worker verdict trusted."""
    with np.load(path, allow_pickle=False) as arrays:
        inputs = decode_arrays(arrays, "input", INPUT_FIELDS)
        if any(not equal_bytes(inputs[n], reference.inputs[n]) for n in INPUT_FIELDS):
            raise ValueError("rolled candidate fixture differs from original")
        expected = {f"input__{n}" for n in INPUT_FIELDS}
        actual = {}
        for device in reference.slots:
            expected.update(f"actual_{device}__{n}" for n in FIELDS)
            actual[device] = decode_arrays(arrays, f"actual_{device}", FIELDS)
        if set(arrays.files) != expected:
            raise ValueError("rolled candidate original inventory differs")
        return compare_observations(actual, reference)
