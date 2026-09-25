"""Tests of :mod:`glm_tpu.runner.tpu_runner`, CPU only: ``TPUModelRunner``'s constructor refusals (before any
device, file or fleet access) and its fleet-voted ``phase``. Generation over a loaded runner is the engine's
(``tests/engine/test_llm_engine.py``)."""

import pytest

from glm_tpu.runner.tpu_runner import TPUModelRunner

UNUSED = dict(
    args=None, repo=None, root=None, mesh=None, physical=None, topology=None, fleet_sha=None, vote=None, save=None
)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(concurrent_size=9), "concurrent size must be zero through eight"),
        (dict(concurrent_size=True), "concurrent size must be zero through eight"),
        (dict(concurrent_size=2), "concurrent runtime requires 32K per conversation"),
    ],
)
def test_constructor_refuses_before_any_device_or_fleet_access(options, message):
    with pytest.raises(ValueError, match=message):
        TPUModelRunner(**UNUSED, **options)


def test_every_phase_votes_records_and_raises_after_the_vote():
    runner = object.__new__(TPUModelRunner)
    runner.record = dict(phases={})
    votes, saved, peers = [], [], [True]
    runner.vote = lambda valid: votes.append(valid) or peers[0]
    runner.save = lambda record: saved.append({name: row["passed"] for name, row in record["phases"].items()})

    assert runner.phase("ready", lambda: 7) == 7
    failure = OSError("local failure")

    def fail():
        raise failure

    with pytest.raises(OSError) as local:
        runner.phase("local", fail)
    assert local.value is failure
    peers[0] = False
    ran = []
    with pytest.raises(RuntimeError, match=r"^optimized peer phase failed: peer$"):
        runner.phase("peer", lambda: ran.append(True))
    # every phase votes its local outcome once, and the record is saved before anything is raised
    assert votes == [True, False, True] and ran == [True]
    assert saved == [{"ready": True}, {"ready": True, "local": False}, {"ready": True, "local": False, "peer": False}]
