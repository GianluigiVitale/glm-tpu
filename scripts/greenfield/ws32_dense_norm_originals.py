"""Bounded DB605 originals for observation-relevance checks; no cloud writes."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from scripts.greenfield.ws32_dense_frontier_evidence import CAPSULES, read_npz
from scripts.greenfield.ws32_dense_frontier_protocol import _read_bound

TAG = "greenfield_fp8_ws32_dense_frontier_d01_20260909T153537693051589Z"
PIN = "601c89d64c72c899044316ac55e0df9595b667d6"
RECEIPT = "docs/artifacts/prefill-dense01-cache-reproduction-20260909.json"
RECEIPT_SHA = "ceddbd758276f921c89fc9e9c907bdc458931bd21d0e35ed44a8fb6e1202f823"


def load_originals(
    root: Path, *, repo: Path, rank: int
) -> dict[str, dict[str, np.ndarray]]:
    """Authenticate already-retained rank originals from reviewed fixed receipt.

    Every saved row field is retained for all five calls; only the two endpoints
    have cache arrays. Historical intermediate caches do not exist and cannot be
    recreated by this reader. Parent remains responsible for remote generations.
    """
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
    ledger = json.loads(
        _read_bound(root / "worker_receipts.json", record["source_ledger"])
    )
    items = {item["name"]: item for item in ledger}
    if len(items) != len(ledger):
        raise ValueError("DB605 original ledger duplicate names")
    prefix = f"results/{TAG}/workers/rank{rank}/"

    def bound(name: str) -> bytes:
        info = items[prefix + name]
        return _read_bound(root / name, {**info, "sha256": info["original_sha256"]})

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
