"""Fixed dense01 identity and pre-TPU originals; not execution authorization.

Reuses the original generation-qualified downloader and DB604 cache consumer.
Only five original payloads per host plus the small fixed ledger are needed.
No model copy, new oracle, cloud write, or infrastructure operation.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Any

from glm_tpu.greenfield.validation.ws32_evidence import (
    _atomic_download, _crc32c_file, _require_blob_identity,
)
from scripts.greenfield.ws32_dense_frontier_witness import (
    BRANCHES, LEDGER_SHA, TAG as ORIGINAL_TAG, load_witness,
)

PROTOCOL = "ws32-dense01-frozen-b128-db604-reproduction-v1"
KERNEL = "ws32_dense_frontier"
LAYERS = (0, 1)
PAYLOAD_BYTES = 102589760
SELECTED_LEAVES = 55
BUCKET = "driftbench-dsv4-uc"
MAX_REFERENCE_BYTES = 128 << 20
REFERENCE_RESERVE = 1 << 30
LEDGER_PIN = dict(
    name=f"results/{ORIGINAL_TAG}/first_window_review/sources.json",
    generation="1788958087584725", size=70389, crc32c="+BhnwQ==", sha256=LEDGER_SHA,
)


def is_tag(tag: str) -> bool:
    return isinstance(tag, str) and re.fullmatch(
        r"greenfield_fp8_ws32_dense_frontier_d01_[0-9]{8}T[0-9]+Z", tag
    ) is not None


def original_names(rank: int) -> tuple[str, ...]:
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("dense reference requires launcher rank0..7")
    return (f"host_records/runner.rank{rank}.json", *(
        f"diagnostic_local/{ORIGINAL_TAG}/first_window.rank{rank}/{branch}.{form}"
        for branch in BRANCHES for form in ("json", "npz")
    ))


def _read_bound(path: Path, pin: dict[str, Any]) -> bytes:
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != pin["size"]
            or _crc32c_file(path) != pin["crc32c"]):
        raise ValueError("dense retained original path/size/CRC differs")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != pin["sha256"]:
        raise ValueError("dense retained original SHA differs")
    return raw


def reference_pins(root: Path, rank: int) -> dict[str, dict[str, Any]]:
    names = original_names(rank)
    ledger = json.loads(_read_bound(root / "sources.json", LEDGER_PIN))
    if ledger["bucket"] != BUCKET or ledger["tag"] != ORIGINAL_TAG:
        raise ValueError("dense original ledger scope differs")
    indexed = {v["name"]: v for v in ledger["objects"]}
    if len(indexed) != len(ledger["objects"]):
        raise ValueError("dense original ledger duplicate names")
    selected = {name: indexed[f"results/{ORIGINAL_TAG}/{name}"] for name in names}
    for pin in selected.values():
        if (type(pin["size"]) is not int or not 0 < pin["size"] <= MAX_REFERENCE_BYTES
                or type(pin["generation"]) is not str or not pin["generation"].isdecimal()
                or int(pin["generation"]) <= 0):
            raise ValueError("dense original generation/size invalid")
    if sum(v["size"] for v in selected.values()) + LEDGER_PIN["size"] > MAX_REFERENCE_BYTES:
        raise ValueError("dense original rank budget exceeded")
    return selected


def load_reference(root: Path, *, rank: int) -> tuple[dict, dict]:
    """Read only this host's originals; full-fleet collector uses all eight."""
    pins = reference_pins(root, rank)
    for name, pin in pins.items():
        _read_bound(root / name, pin)
    record = json.loads((root / original_names(rank)[0]).read_bytes())
    witness = load_witness(root, rank=rank)
    if (record["launch_process_id"] != rank
            or record["code_hash"] != "aceea327b8608d73224bcb5b43671b69021baa00"
            or record["status"] != "DIAGNOSTIC_COMPLETED_NOT_NUMERICAL_PROMOTION"
            or {v["device_slot"] for v in record["local_device_slots"]}
            != {key[1] for key in witness}):
        raise ValueError("dense original worker/owner identity differs")
    return record, witness


def materialize_reference(root: Path, *, rank: int, client: Any) -> tuple[dict, dict]:
    """Exact-generation bounded preflight; existing files never overwritten."""
    original_names(rank)
    if root.is_symlink():
        raise ValueError("dense reference root cannot be a symlink")
    bucket = client.bucket(BUCKET)
    bucket.reload()
    if str(bucket.location).upper() != "US-CENTRAL2":
        raise ValueError("dense reference bucket must be US-CENTRAL2")

    def fetch(name: str, pin: dict[str, Any]) -> None:
        path = root / name
        if path.exists() or path.is_symlink():
            _read_bound(path, pin)
            return
        generation = int(pin["generation"])
        blob = bucket.blob(pin["name"], generation=generation)
        blob.reload(if_generation_match=generation)
        if (int(blob.generation) != generation or blob.size != pin["size"]
                or blob.crc32c != pin["crc32c"]):
            raise ValueError("dense remote original generation/size/CRC differs")
        _atomic_download(blob, path)
        _require_blob_identity(blob, path, pin["sha256"])

    parent = root if root.exists() else root.parent
    if shutil.disk_usage(parent).free < MAX_REFERENCE_BYTES + REFERENCE_RESERVE:
        raise ValueError("dense original materialization needs bounded headroom")
    fetch("sources.json", LEDGER_PIN)
    for name, pin in reference_pins(root, rank).items():
        fetch(name, pin)
    return load_reference(root, rank=rank)
