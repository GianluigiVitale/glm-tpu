"""Fixed identity, schedule and retained-original pins; pure CPU, no arrays."""

from pathlib import Path

import numpy as np
import pytest

from scripts.greenfield import ws32_history_protocol as protocol

ROOT = Path(__file__).resolve().parents[3]


def test_plan_reproduces_both_production_schedules_interleaved():
    steps = protocol.plan()
    candidate = [s for s in steps if s.branch == "candidate"]
    control = [s for s in steps if s.branch == "control"]
    assert len(steps) == 319 and len(candidate) == 64 and len(control) == 255
    assert [s.count for s in candidate] == [128] * 63 + [91]
    assert [s.count for s in control] == [32] * 254 + [27]
    assert candidate[-1].program == "candidate_b114" and control[-1].program == "control_b114"
    assert all(s.program == "candidate_b128" for s in candidate[:-1])
    assert all(s.program == "control_b128" for s in control[:-1])
    assert [s.index for s in steps] == list(range(319))
    # Every control block of a group lies inside that group's candidate block.
    for group in range(64):
        own = [s for s in steps if s.group == group]
        assert own[0].branch == "candidate"
        lo, hi = own[0].offset, own[0].offset + own[0].count
        assert all(s.branch == "control" and lo <= s.offset < hi for s in own[1:])
        assert sum(s.count for s in own[1:]) == own[0].count
    assert protocol.plan(147) == protocol.plan(147)
    short = protocol.plan(147)
    assert [(s.group, s.branch, s.offset, s.count) for s in short] == [
        (0, "candidate", 0, 128), (0, "control", 0, 32), (0, "control", 32, 32),
        (0, "control", 64, 32), (0, "control", 96, 32), (1, "candidate", 128, 19),
        (1, "control", 128, 19)]
    for bad in (0, -1, 8155.0, True):
        with pytest.raises(ValueError):
            protocol.plan(bad)
    with pytest.raises(ValueError):
        protocol.plan(8155, wide=100, narrow=32)


def test_tag_shape_is_exclusive():
    assert protocol.is_tag("greenfield_fp8_ws32_history_frontier_l06_20260910T000000000000000Z")
    assert not protocol.is_tag("greenfield_fp8_ws32_dense_frontier_d01_20260910T000000000000000Z")
    assert not protocol.is_tag(None) and not protocol.is_tag("")


def test_every_rank_has_both_branches_original_pins_from_committed_receipts():
    for rank in range(8):
        pins = protocol.original_pins(ROOT, rank)
        assert set(pins) == set(protocol.BRANCHES)
        for branch, forms in pins.items():
            assert set(forms) == set(protocol.ORIGINAL_FORMS)
            for form, pin in forms.items():
                assert pin["name"] == (
                    f"results/{protocol.ORIGINAL_TAGS[branch]}/host_records/runner.rank{rank}.{form}")
                assert pin["generation"].isdecimal() and pin["size"] > 0 and len(pin["sha256"]) == 64
    for bad in (8, -1, "0", 0.0):
        with pytest.raises(ValueError):
            protocol.original_pins(ROOT, bad)


def test_receipts_are_bound_by_digest(tmp_path):
    for branch, (relative, _) in protocol.RECEIPTS.items():
        copy = tmp_path / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((ROOT / relative).read_bytes() + b"\n")
        with pytest.raises(ValueError, match="digest"):
            protocol.receipt(tmp_path, branch)


def test_reproduction_rows_take_step0_events_0_to_3_only():
    steps, producers, top_k = 3, 21, 16
    arrays = dict(
        dsa_producer_layer_ids=np.asarray([0, 1, 2, 6, 10, 14] + list(range(15, 30)), np.int32),
        dsa_selected_positions=np.arange(steps * producers * top_k, dtype=np.int32).reshape(steps, producers, 1, top_k),
        dsa_selected_valid_counts=np.full((steps, producers, 1), top_k, np.int32),
        dsa_selected_scores=np.arange(steps * producers * top_k, dtype=np.float32).reshape(steps, producers, 1, top_k),
    )
    rows = protocol.reproduction_rows(arrays)
    assert rows["positions"].shape == (4, 1, top_k) and rows["counts"].shape == (4, 1)
    np.testing.assert_array_equal(rows["positions"], arrays["dsa_selected_positions"][0, :4])
    np.testing.assert_array_equal(rows["scores"], arrays["dsa_selected_scores"][0, :4])
    with pytest.raises(ValueError, match="producers"):
        protocol.reproduction_rows({**arrays, "dsa_producer_layer_ids": np.asarray([0, 1, 2, 5], np.int32)})
    with pytest.raises(ValueError):
        protocol.reproduction_rows({**arrays, "dsa_selected_scores": arrays["dsa_selected_scores"].astype(np.float64)})
