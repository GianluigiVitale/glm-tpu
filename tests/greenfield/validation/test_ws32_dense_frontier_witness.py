"""Local retained-original reproduction checks; not a new TPU execution."""

from pathlib import Path

import numpy as np
import pytest

from scripts.greenfield.ws32_dense_frontier_witness import (
    BRANCHES, TAG, WIDTHS, compare_owner, load_witness,
)


@pytest.fixture(scope="module")
def witness():
    root = Path("/home/gianl/glm-run") / TAG / "first_window_collected"
    if not (root / "sources.json").exists():
        pytest.skip("requires the retained DB604 original evidence")
    return load_witness(root)


def rebuild(witness, branch, slot):
    caches = {}
    for family, width in WIDTHS.items():
        entry = witness[(branch, slot, family)]
        record = entry["cache"]
        value = np.zeros((2, 16, 64, width), np.uint16)
        value[:, record["physical_pages"], record["local_rows"], :] = entry["rows"]
        caches[family] = value
    return caches


def test_all_original_branch_owner_bytes_and_changed_layers(witness):
    differences = {f: [0, 0] for f in WIDTHS}
    for branch in BRANCHES:
        for slot in range(32):
            result = compare_owner(witness, branch=branch, slot=slot,
                                   caches=rebuild(witness, branch, slot))
            assert result["reproduced"]
            assert result["numerical_promotion"] is False
    # Count unique expert owners, not four feature replicas of each cache.
    for slot in range(0, 32, 4):
        for family in WIDTHS:
            changed = witness[(BRANCHES[0], slot, family)]["rows"] != witness[(BRANCHES[1], slot, family)]["rows"]
            for layer in (0, 1):
                differences[family][layer] += np.count_nonzero(changed[layer])
    assert differences == {"kv": [0, 271], "index": [0, 12], "repair": [0, 11]}


@pytest.mark.parametrize("family", WIDTHS)
@pytest.mark.parametrize("layer", (0, 1))
@pytest.mark.parametrize("location", ((0, 44, 21), (15, 63, 127)))
def test_any_changed_byte_refuses_including_outside_frontier(witness, family, layer, location):
    caches = rebuild(witness, "wide_final", 0)
    caches[family][(layer, *location)] ^= np.uint16(1)
    result = compare_owner(witness, branch="wide_final", slot=0, caches=caches)
    assert not result["reproduced"]
    assert not result["families"][family]["bytes_equal"]


def test_branch_swap_and_incomplete_cache_refuse(witness):
    caches = rebuild(witness, "narrow_128", 0)
    assert not compare_owner(witness, branch="wide_final", slot=0, caches=caches)["reproduced"]
    caches["kv"] = caches["kv"][:1]
    with pytest.raises(ValueError, match="every owner cache byte"):
        compare_owner(witness, branch="wide_final", slot=0, caches=caches)
    with pytest.raises(ValueError, match="scope"):
        compare_owner(witness, branch="wide_final", slot=True, caches=caches)


def test_unbound_ledger_refuses(tmp_path):
    (tmp_path / "sources.json").write_text('{}')
    with pytest.raises(ValueError, match="exact DB604"):
        load_witness(tmp_path)
