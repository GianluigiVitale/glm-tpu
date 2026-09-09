"""Bounded DB605 originals for observation-relevance checks; no cloud writes."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from scripts.greenfield.ws32_dense_frontier_evidence import CAPSULES, read_npz
from scripts.greenfield.ws32_dense_frontier_protocol import _read_bound
from glm_tpu.greenfield.validation.ws32_evidence import (
    _atomic_download,
    _require_blob_identity,
)

TAG = "greenfield_fp8_ws32_dense_frontier_d01_20260909T153537693051589Z"
PIN = "601c89d64c72c899044316ac55e0df9595b667d6"
RECEIPT = "docs/artifacts/prefill-dense01-cache-reproduction-20260909.json"
RECEIPT_SHA = "ceddbd758276f921c89fc9e9c907bdc458931bd21d0e35ed44a8fb6e1202f823"
BUCKET = "driftbench-dsv4-uc"
MAX_REFERENCE_BYTES = 128 << 20
REFERENCE_RESERVE = 1 << 30


def receipt_worker(repo: Path, rank: int) -> dict:
    """Read the one fixed reviewed receipt, never caller-supplied object names."""
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("DB605 original rank must be0..7")
    path = repo / RECEIPT
    if path.is_symlink() or not path.is_file() or path.stat().st_size != 130791:
        raise ValueError("DB605 original receipt path/size differs")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != RECEIPT_SHA:
        raise ValueError("DB605 original receipt SHA differs")
    receipt = json.loads(raw)
    record = receipt["workers"][rank]
    if record["rank"] != rank or receipt["tag"] != TAG or receipt["source_pin"] != PIN:
        raise ValueError("DB605 original identity differs")
    return record


def reference_pins(root: Path, *, repo: Path, rank: int) -> dict:
    record = receipt_worker(repo, rank)
    ledger = json.loads(
        _read_bound(root / "worker_receipts.json", record["source_ledger"])
    )
    items = {item["name"]: item for item in ledger}
    if len(items) != len(ledger):
        raise ValueError("DB605 original ledger duplicate names")
    prefix = f"results/{TAG}/workers/rank{rank}/"
    names = ("runner.json", *(label + ".npz" for label, _, _ in CAPSULES))
    selected = {
        name: {
            **items[prefix + name],
            "sha256": items[prefix + name]["original_sha256"],
        }
        for name in names
    }
    pins = {"worker_receipts.json": record["source_ledger"], **selected}
    for name, pin in pins.items():
        if (
            pin["name"] != prefix + name
            or type(pin["size"]) is not int
            or not 0 < pin["size"] <= MAX_REFERENCE_BYTES
            or not isinstance(pin["generation"], str)
            or not pin["generation"].isdecimal()
            or int(pin["generation"]) <= 0
        ):
            raise ValueError("DB605 original generation/path/size differs")
    if sum(p["size"] for p in pins.values()) > MAX_REFERENCE_BYTES:
        raise ValueError("DB605 original cumulative reference budget exceeded")
    return pins


def load_originals(
    root: Path, *, repo: Path, rank: int
) -> dict[str, dict[str, np.ndarray]]:
    """Authenticate retained originals; only endpoints have saved cache arrays."""
    record = receipt_worker(repo, rank)
    pins = reference_pins(root, repo=repo, rank=rank)

    def bound(name: str) -> bytes:
        return _read_bound(root / name, pins[name])

    runner_bytes = bound("runner.json")
    if sha256(runner_bytes).hexdigest() != record["runner_sha256"]:
        raise ValueError("DB605 original runner differs")
    runner = json.loads(runner_bytes)
    if runner["launch_rank"] != rank or runner["code_hash"] != PIN:
        raise ValueError("DB605 original runner scope differs")
    result = {}
    for label, count, endpoint in CAPSULES:
        bound(label + ".npz")
        report = runner["dense_frontier"]["originals"][label]
        if report["count"] != count or report["keep_caches"] is not endpoint:
            raise ValueError("DB605 original capsule scope differs")
        result[label] = read_npz(
            root / (label + ".npz"), report, limit=128 << 20, size_key="npz_bytes"
        )
    return result


def load_bundle(root: Path, *, repo: Path, rank: int) -> tuple[dict, dict, dict]:
    """Return authenticated runner, five capsules and compact source bindings."""
    arrays = load_originals(root, repo=repo, rank=rank)
    pins = reference_pins(root, repo=repo, rank=rank)
    runner = json.loads(_read_bound(root / "runner.json", pins["runner.json"]))
    identity = dict(
        tag=TAG,
        code_hash=PIN,
        rank=rank,
        receipt_sha256=RECEIPT_SHA,
        original_pins=pins,
        bytes=sum(v["size"] for v in pins.values()),
        raw_array_bytes=sum(v.nbytes for a in arrays.values() for v in a.values()),
    )
    return runner, arrays, identity


def materialize(
    root: Path,
    *,
    repo: Path,
    rank: int,
    client: object,
    existing_reference_bytes: int = 0,
) -> tuple[dict, dict, dict]:
    """Seven exact generation downloads, bounded, same-region and no overwrites.

    Reuses the existing atomic generation-qualified transfer and SHA/CRC checks.
    Only saved results are downloaded; never checkpoint payloads or another bucket.
    """
    record = receipt_worker(repo, rank)
    if (
        type(existing_reference_bytes) is not int
        or not 0 <= existing_reference_bytes <= MAX_REFERENCE_BYTES
    ):
        raise ValueError("DB605 existing reference budget differs")
    if root.is_symlink():
        raise ValueError("DB605 reference root cannot be a symlink")
    if (
        shutil.disk_usage(root if root.exists() else root.parent).free
        < MAX_REFERENCE_BYTES + REFERENCE_RESERVE
    ):
        raise ValueError("DB605 reference materialization needs bounded disk headroom")
    bucket = client.bucket(BUCKET)
    bucket.reload()
    if str(bucket.location).upper() != "US-CENTRAL2":
        raise ValueError("DB605 reference bucket must be US-CENTRAL2")

    def fetch(name: str, pin: dict) -> None:
        path = root / name
        partial = path.with_name(path.name + ".partial")
        if partial.exists() or partial.is_symlink():
            raise FileExistsError(partial)
        if path.exists() or path.is_symlink():
            _read_bound(path, pin)
            return
        generation = int(pin["generation"])
        blob = bucket.blob(pin["name"], generation=generation)
        blob.reload(if_generation_match=generation)
        if (
            int(blob.generation) != generation
            or blob.size != pin["size"]
            or blob.crc32c != pin["crc32c"]
        ):
            raise ValueError("DB605 remote generation/size/CRC differs")
        _atomic_download(blob, path)
        _require_blob_identity(blob, path, pin["sha256"])

    fetch("worker_receipts.json", record["source_ledger"])
    pins = reference_pins(root, repo=repo, rank=rank)
    if (
        existing_reference_bytes + sum(p["size"] for p in pins.values())
        > MAX_REFERENCE_BYTES
    ):
        raise ValueError("DB604/605 combined reference budget exceeded")
    for name, pin in pins.items():
        fetch(name, pin)
    return load_bundle(root, repo=repo, rank=rank)


def bind_prior(runner: Mapping, prior: Mapping, *, prior_sha256: str) -> None:
    """DB605 used the exact DB604 prompt, checkpoint and physical owner mapping."""
    from scripts.greenfield import ws32_dense_frontier_protocol as original
    from scripts.greenfield.prefill_window_evidence import same_json

    keys = (
        "hostname",
        "jax_process_index",
        "checkpoint_manifest_sha256",
        "checkpoint_success_sha256",
        "source_inventory_sha256",
        "mesh_sha256",
        "topology_sha256",
        "topology_fleet_sha256",
        "prompt_ids_sha256",
        "main_rope_table",
    )
    same_json(
        {k: runner[k] for k in keys},
        {k: prior[k] for k in keys},
        "DB604/605 original context",
    )
    if (
        runner["code_hash"] != PIN
        or runner["tag"] != TAG
        or runner["original_tag"] != original.ORIGINAL_TAG
        or runner["original_ledger_sha256"] != original.LEDGER_SHA
        or runner["original_runner_sha256"] != prior_sha256
    ):
        raise ValueError("DB605 original DB604 binding differs")
    actual = {v["device_slot"]: v for v in runner["local_device_slots"]}
    expected = {v["device_slot"]: v for v in prior["local_device_slots"]}
    if (
        len(actual) != 4
        or len(runner["local_device_slots"]) != 4
        or set(actual) != set(expected)
    ):
        raise ValueError("DB605 original owner inventory differs")
    for slot, owner in actual.items():
        if (
            owner["device_id"] != expected[slot]["device_id"]
            or owner["expected_full_file_sha256_not_verified"]
            != expected[slot]["file_sha256"]
            or owner["selected_payload_bytes"] != original.PAYLOAD_BYTES
            or len(owner["observed_selected_tensor_sha256"]) != original.SELECTED_LEAVES
        ):
            raise ValueError("DB605 original selected owner differs")


def require_reproduction(
    actual: Mapping[str, np.ndarray], original: Mapping[str, np.ndarray]
) -> dict:
    """Compare every retained encoded bit array, never just a cache signature.

    Inputs use the original capture's encoded names/dtypes, so BF16 NaNs and
    padding bits are compared as bytes without numerical coercion. No tolerance.
    """
    if not actual or set(actual) != set(original):
        raise ValueError("dense norm reproduction retained-field inventory differs")
    for name, expected in original.items():
        value = actual[name]
        if (
            not isinstance(value, np.ndarray)
            or not isinstance(expected, np.ndarray)
            or value.shape != expected.shape
            or value.dtype != expected.dtype
            or value.tobytes() != expected.tobytes()
        ):
            raise ValueError(f"dense norm original realization differs: {name}")
    return dict(
        passed=True,
        arrays=len(original),
        bytes=sum(v.nbytes for v in original.values()),
        scope="ALL_RETAINED_DB605_FIELDS_NOT_UNSAVED_INTERMEDIATE_CACHES",
        numerical_promotion=False,
        cause_claim=False,
        own8k_pass=False,
    )
