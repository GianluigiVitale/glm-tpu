"""Read archived DB604 diagnostic cache originals for dense0/1 reproduction.

CPU-only, read-only and fixed to one existing evidence ledger. This is not a
new correctness oracle, admission/launcher, or an explanation of token11.
The original collector already authenticated topology/graphs/journals; this
consumer binds its exact ledger and replays the cache bytes it uses, not HLO.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping
import zipfile

import numpy as np

from scripts.greenfield.ws32_prefill_frontier import replay_cache

TAG = "greenfield_ws32_short_decoder_8k_numerical_c128_hrope_bp1_ps1_rp1_ep1_lm1_first128_20260909T122536706457044Z"
LEDGER_SHA = "4c6d4cab7f26281bbdac69a85a296d1298aa5f08709d516aac1f68583242653e"
BRANCHES = ("wide_final", "narrow_128")
WIDTHS = {"kv": 640, "index": 128, "repair": 128}


def _digest(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def load_witness(root: Path, *, rank: int | None = None) -> dict[tuple[str, int, str], dict[str, Any]]:
    """Return compact two-layer witnesses for both branches and all32 owners.

    Original full-layer capsules are replayed before extracting layers0/1.
    Returned whole-cache hashes cover all pages, including untouched zeros.
    Optional launcher rank reads only its four original owners, never rank==slot.
    The default still requires all32 owners. This does not download data.
    """
    if rank is not None and (type(rank) is not int or not 0 <= rank < 8):
        raise ValueError("dense witness requires launcher rank0..7")
    raw = (root / "sources.json").read_bytes()
    if sha256(raw).hexdigest() != LEDGER_SHA:
        raise ValueError("dense witness needs the exact DB604 source ledger")
    ledger = json.loads(raw)
    if ledger["bucket"] != "driftbench-dsv4-uc" or ledger["tag"] != TAG:
        raise ValueError("dense witness source scope differs")
    by_name = {obj["name"]: obj for obj in ledger["objects"]}
    if len(by_name) != len(ledger["objects"]):
        raise ValueError("dense witness duplicate source names")

    def original(relative: str) -> bytes:
        obj = by_name[f"results/{TAG}/{relative}"]
        path = root / relative
        if path.is_symlink() or path.stat().st_size != obj["size"]:
            raise ValueError("dense witness original size differs")
        data = path.read_bytes()
        if sha256(data).hexdigest() != obj["sha256"]:
            raise ValueError("dense witness original SHA differs")
        return data

    result = {}
    all_slots = set()
    for launch_rank in range(8) if rank is None else (rank,):
        rank_slots = None
        for branch in BRANCHES:
            prefix = f"diagnostic_local/{TAG}/first_window.rank{launch_rank}/{branch}"
            report = json.loads(original(prefix + ".json"))
            data = original(prefix + ".npz")
            if (report["frontier"] != 128 or report["prompt_length"] != 8155
                    or report["context_capacity"] != 8192 or report["valid"] is not True
                    or report["npz_sha256"] != sha256(data).hexdigest()):
                raise ValueError("dense witness original frontier differs")
            slots = {int(key) for key in report["owners"]}
            if (len(slots) != 4 or not slots <= set(range(32))
                    or set(report["owners"]) != {str(slot) for slot in slots}
                    or (rank_slots is not None and slots != rank_slots)):
                raise ValueError("dense witness original rank owners differ")
            rank_slots = slots
            with zipfile.ZipFile(BytesIO(data)) as archive:
                if sum(item.file_size for item in archive.infolist()) > 128 * 1024**2:
                    raise ValueError("dense witness expanded archive exceeds budget")
            with np.load(BytesIO(data), allow_pickle=False) as arrays:
                for slot_key, owner in report["owners"].items():
                    slot = int(slot_key)
                    for family, width in WIDTHS.items():
                        key = (branch, slot, family)
                        if key in result:
                            raise ValueError("dense witness duplicates an owner")
                        capsule = owner["caches"][family]["cache"]
                        rows = arrays[f"slot{slot}_{family}_rows"]
                        replay_cache(rows, capsule)
                        if capsule["layer_ids"][:2] != [0, 1]:
                            raise ValueError("dense witness layer selection differs")
                        selected = np.ascontiguousarray(rows[:2]).copy()
                        rebuilt = np.zeros((2, 16, 64, width), np.uint16)
                        rebuilt[:, capsule["physical_pages"], capsule["local_rows"], :] = selected
                        record = {**capsule, "shape": list(rebuilt.shape), "layer_ids": [0, 1],
                                  "rows_sha256": _digest(selected), "whole_cache_sha256": _digest(rebuilt)}
                        replay_cache(selected, record)
                        result[key] = {"cache": record, "rows": selected,
                                       "source_generation": by_name[f"results/{TAG}/{prefix}.npz"]["generation"]}
        if all_slots & rank_slots:
            raise ValueError("dense witness duplicates a rank owner")
        all_slots.update(rank_slots)
    required_slots = set(range(32)) if rank is None else all_slots
    if set(result) != {(b, s, f) for b in BRANCHES for s in required_slots for f in WIDTHS}:
        raise ValueError("dense witness lacks complete branch/owner/cache coverage")
    return result


def compare_owner(
    witness: Mapping[tuple[str, int, str], Mapping[str, Any]], *,
    branch: str, slot: int, caches: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Compare full two-layer owner caches, supplied as raw uint16 BF16 bits.

    A false result disqualifies this reduced realization for causal attribution;
    it does not refute DB604. Outer integration must require all64 branch/owner
    results, real TPU execution/source/weight identity, health and memory checks.
    """
    if (branch not in BRANCHES or type(slot) is not int or not 0 <= slot < 32
            or set(caches) != set(WIDTHS)):
        raise ValueError("dense reproduction branch/owner/family scope differs")
    families = {}
    for family, width in WIDTHS.items():
        value = caches[family]
        expected = witness[(branch, slot, family)]
        record = expected["cache"]
        if (not isinstance(value, np.ndarray) or value.dtype != np.uint16
                or value.shape != (2, 16, 64, width)):
            raise ValueError("dense reproduction must supply every owner cache byte")
        replay_cache(expected["rows"], record)
        digest = _digest(value)
        families[family] = {"bytes_equal": digest == record["whole_cache_sha256"],
                            "actual_sha256": digest, "expected_sha256": record["whole_cache_sha256"]}
    return dict(branch=branch, slot=slot, families=families,
                reproduced=all(v["bytes_equal"] for v in families.values()),
                scope="DENSE01_DB604_BYTE_REPRODUCTION_ONLY",
                numerical_promotion=False, performance_claim=False)
