"""Replay actual saved DB605 outputs, not fixture labels or invented caches."""

from pathlib import Path

import numpy as np
import pytest

from scripts.greenfield.ws32_dense_norm_originals import (
    TAG,
    load_originals,
    require_reproduction,
)

REPO = Path("/home/gianl/glm-tpu-topology-rewrite")
ROOT = Path("/home/gianl/glm-run") / TAG / "fleet"


def test_all32_actual_original_fields_and_endpoint_cache_scope():
    total = 0
    for rank in range(8):
        records = load_originals(ROOT / f"rank{rank}", repo=REPO, rank=rank)
        assert len(records) == 5
        for label, arrays in records.items():
            verdict = require_reproduction(arrays, arrays)
            assert verdict["passed"] and not verdict["cause_claim"]
            caches = [n for n in arrays if n.endswith(("__kv", "__index", "__repair"))]
            assert len(caches) == (24 if label in ("wide_final", "narrow_128") else 0)
            total += verdict["bytes"]
    assert total == 8 * 81933312


@pytest.mark.parametrize("kind", ["value", "shape", "dtype", "missing", "extra"])
def test_any_retained_field_change_refuses(kind):
    reference = {
        "slot9_layer0__normalized": np.arange(12, dtype=np.uint16).reshape(3, 4)
    }
    actual = {k: v.copy() for k, v in reference.items()}
    name = next(iter(actual))
    if kind == "value":
        actual[name][1, 2] ^= 1
    elif kind == "shape":
        actual[name] = actual[name].reshape(4, 3)
    elif kind == "dtype":
        actual[name] = actual[name].astype(np.int16)
    elif kind == "missing":
        actual.clear()
    else:
        actual["unexpected"] = np.zeros(1, np.uint16)
    with pytest.raises(ValueError):
        require_reproduction(actual, reference)


@pytest.mark.parametrize("rank", [-1, 8, True, "0"])
def test_bad_rank_refuses_before_original_reads(rank):
    with pytest.raises(ValueError, match="rank"):
        load_originals(ROOT / "rank0", repo=REPO, rank=rank)


def test_wrong_rank_original_refuses():
    with pytest.raises(ValueError):
        load_originals(ROOT / "rank0", repo=REPO, rank=1)
