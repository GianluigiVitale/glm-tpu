"""Tests of :mod:`glm_tpu.runner.tpu_runner`, CPU only: ``TPUModelRunner``'s constructor refusals (before any
device, file or fleet access), its fleet-voted ``phase`` and what ``compile`` stores for a compiled graph.
Generation over a loaded runner is the engine's (``tests/engine/test_llm_engine.py``)."""

from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from glm_tpu.runner.admission import check_hlo_collectives
from glm_tpu.runner.tpu_runner import TPUModelRunner
from glm_tpu.utils.io_utils import persist

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


FEATURE = "{" + ",".join("{" + ",".join(str(i) for i in range(r * 4, r * 4 + 4)) + "}" for r in range(8)) + "}"
# An optimized-HLO module in the TPU compiler's text form that the admission accepts: 32 partitions, one all-reduce
# over the feature axis.
OPTIMIZED_HLO = (
    "HloModule decode, entry_computation_layout={(f32[8,128]{1,0})->f32[8,128]{1,0}}, num_partitions=32\n\n"
    "%add (x: f32[], y: f32[]) -> f32[] {\n  %x = f32[] parameter(0)\n  %y = f32[] parameter(1)\n"
    "  ROOT %sum = f32[] add(f32[] %x, f32[] %y)\n}\n\n"
    "ENTRY %main (p0: f32[8,128]) -> f32[8,128] {\n  %p0 = f32[8,128]{1,0} parameter(0)\n"
    f"  ROOT %ar = f32[8,128]{{1,0}} all-reduce(f32[8,128]{{1,0}} %p0), channel_id=1, replica_groups={FEATURE}, "
    "use_global_device_ids=true, to_apply=%add\n}\n"
)
STABLEHLO = "module @decode {}"


class Compiled:
    """What ``lowered.compile()`` returns, as far as ``compile_program`` reads it."""

    def as_text(self):
        return OPTIMIZED_HLO

    def memory_analysis(self):
        sizes = ("argument", "output", "alias", "temp", "generated_code")
        return SimpleNamespace(**{f"{name}_size_in_bytes": 1024 for name in sizes})


def test_compile_persists_the_hlo_admission_of_the_compiled_graph(tmp_path):
    # The real compile path (compile_program, the graph consensus over one process, the admission of the optimized
    # HLO the compiler wrote, the memory admission) on a graph whose lowering is faked; the record goes to disk as the
    # worker saves it (native.rank<r>/runtime.json).
    compiled = Compiled()
    lowered = SimpleNamespace(compiler_ir=lambda dialect: STABLEHLO, compile=lambda: compiled)
    runtime = tmp_path / "native.rank0" / "runtime.json"
    runtime.parent.mkdir()
    runner = object.__new__(TPUModelRunner)
    runner.hlo = tmp_path / "hlo"
    runner.hlo.mkdir()
    runner.record = dict(programs={}, phases={})
    runner.vote = lambda valid: valid
    runner.save = lambda value: persist(runtime, value)
    runner.stats = lambda: [dict(device_id=d, bytes_in_use=0, bytes_limit=16 * 1024**3) for d in range(4)]

    assert runner.compile("decode", SimpleNamespace(lower=lambda *values: lowered), (None,)) is compiled
    saved = json.loads(runtime.read_text())
    row = saved["programs"]["decode"]
    assert row["hlo_admission"] == check_hlo_collectives(OPTIMIZED_HLO)
    assert row["hlo_admission"] == dict(
        passed=True,
        profile="ws32_axis_payload_v1",
        num_partitions=32,
        instructions=5,
        collectives={"all-reduce": 1},
        maximum_collective_payload_bytes=4096,
        frozen_graph_admission_inherited=False,
    )
    assert row["stablehlo_sha256"] == sha256(STABLEHLO.encode()).hexdigest()
    assert row["optimized_hlo_sha256"] == sha256(OPTIMIZED_HLO.encode()).hexdigest()
    assert (runner.hlo / "decode.optimized_hlo.txt").read_text() == OPTIMIZED_HLO
    assert runner.record["programs"]["decode"]["memory_admission"]["passed"] is True  # saved with the next phase
    assert {name: phase["passed"] for name, phase in saved["phases"].items()} == {
        "compile_decode": True,
        "graph_consensus_decode": True,
        "hlo_decode": True,
        "memory_decode": True,
    }
