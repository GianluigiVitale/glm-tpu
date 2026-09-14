"""User outer replay: real host/identity/trace parsing, fake model/cold hardware.

The cold validator and tokenizer are explicit fixture boundaries here. Separate
cold/HLO tests exercise that validator. Nothing in this test is real TPU proof.
"""

from copy import deepcopy
import gzip
from hashlib import sha256
import json
import stat

import pytest

from glm_tpu import user_request
from scripts.release import ws32_user_evidence as evidence
from scripts.analysis import parse_xplane
from tests.release.test_user_result import fleet

TAG = "greenfield_ws32_user_request_20260914T030000000000000Z"
PIN = "a" * 40


def trace_bytes(host, *, module="jit_user_observer", planes=8):
    space = parse_xplane.build_xplane_classes()["XSpace"]()
    space.hostnames.append(host)
    for index in range(planes):
        plane = space.planes.add(id=index, name=f"/device:TPU:{index}")
        plane.event_metadata[1].name = module
        plane.event_metadata[2].name = "all-reduce"
        plane.lines.add(id=1, name="XLA Modules").events.add(
            metadata_id=1, offset_ps=0, duration_ps=200_000_000_000
        )
        plane.lines.add(id=2, name="XLA Ops").events.add(
            metadata_id=2, offset_ps=10_000_000_000, duration_ps=100_000_000_000
        )
    return space.SerializeToString()


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path / TAG
    root.mkdir(mode=0o700)
    destination = root / "collected"
    destination.mkdir(mode=0o700)
    args = fleet(destination, monkeypatch)
    raw = user_request.canonical(args["request"])
    (root / "request.json").write_bytes(raw)
    (root / "launch.json").write_bytes(
        user_request.canonical(
            dict(
                tag=TAG,
                code_hash=PIN,
                benchmark=False,
                request_file_sha256=sha256(raw).hexdigest(),
                request_bytes=len(raw),
            )
        )
    )
    original = []
    for rank in range(8):
        host, boot = f"fixture-w-{rank}", f"boot{rank}"
        process = dict(pid=100 + rank, start_ticks="10", argv_sha256="b" * 64)
        original.append(
            dict(
                rank=rank,
                host=host,
                boot_id=boot,
                tag=TAG,
                pin=PIN,
                processes=[process],
                holders=[],
            )
        )
        path = destination / f"runner.rank{rank}.json"
        outer = json.loads(path.read_bytes())
        outer["owner"] = dict(**process, hostname=host, boot_id=boot)
        path.write_bytes(user_request.canonical(outer))
        ended = dict(
            tag=TAG,
            code_hash=PIN,
            rank=rank,
            host=host,
            boot_id=boot,
            request_file_sha256=sha256(raw).hexdigest(),
            worker_exit_code=0,
            supervisor_pid=500 + rank,
            worker_started=True,
            worker_error_type=None,
        )
        (destination / f"ended.rank{rank}.json").write_bytes(
            user_request.canonical(ended)
        )
        path = destination / f"sessions.rank{rank}/item000/result.json.gz"
        row = json.loads(gzip.decompress(path.read_bytes()))
        trace = trace_bytes(host)
        (destination / row["observations"]["trace"]["path"]).write_bytes(trace)
        row["observations"]["trace"].update(
            bytes=len(trace), sha256=sha256(trace).hexdigest()
        )
        path.write_bytes(gzip.compress(user_request.canonical(row)))
    idle = deepcopy(original)
    for row in idle:
        row["processes"] = []
    (root / "user_watch.jsonl").write_bytes(
        b"".join(
            user_request.canonical(dict(status="OBSERVED", tag=TAG, pin=PIN, fleet=f))
            + b"\n"
            for f in (original, idle, idle)
        )
    )
    census = "".join(
        "FP8_IDLE "
        + json.dumps(dict(host=r["host"], boot_id=r["boot_id"], devices=[0, 1, 2, 3]))
        + "\n"
        for r in original
    )
    for phase in ("pre", "post"):
        (root / f"census_{phase}.txt").write_text(census)
    hlo = destination / "native.rank0"
    hlo.mkdir()
    (hlo / "observer.optimized_hlo.txt").write_text(
        "HloModule jit_user_observer, entry_computation_layout={}\n"
    )
    monkeypatch.setattr(evidence.worker, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(evidence, "load_tokenizer", lambda: args["tokenizer"])
    calls = []

    def cold(dest, pin):
        assert dest == destination and pin == PIN
        calls.append("cold")
        return (
            {"fixture_not_cold_hardware": True},
            args["parents"],
            args["full_index_layers"],
        )

    monkeypatch.setattr(evidence, "replay_cold", cold)
    return root, calls


def test_actual_outer_join_no_db_or_seal_and_idempotent_original_watch(case):
    root, calls = case
    report = evidence.replay_collected(root, TAG, PIN)
    assert report["complete"] and report["trace_physical_coverage_verified"]
    assert report["worker_ownership_verified"] and report["cold_admission_verified"]
    assert report["trace"]["n_cores"] == 64 and report["trace"]["step_cycle_ms"] is None
    assert not report["protected_result_sealed"] and report["quality_score"] is None
    assert report["cold"]["fixture_not_cold_hardware"]
    assert calls == ["cold"]
    assert evidence.replay_collected(root, TAG, PIN) == report
    assert stat.S_IMODE((root / "final_watch.jsonl").stat().st_mode) == 0o600
    assert not (root / "SUCCESS").exists() and not (root / "results.db").exists()


@pytest.mark.parametrize(
    "fault",
    [
        "input",
        "owner",
        "boot",
        "ended",
        "prelaunch",
        "busy",
        "missing_pid",
        "census_boot",
    ],
)
def test_original_identity_or_exit_refusal_precedes_cold_replay(case, fault):
    root, calls = case
    if fault == "input":
        with (root / "request.json").open("ab") as stream:
            stream.write(b" ")
    elif fault == "census_boot":
        path = root / "census_post.txt"
        path.write_text(path.read_text().replace("boot3", "changed-boot3"))
    elif fault in ("owner", "boot"):
        path = root / "collected/runner.rank7.json"
        value = json.loads(path.read_bytes())
        value["owner"]["start_ticks" if fault == "owner" else "boot_id"] = "different"
        path.write_bytes(user_request.canonical(value))
    elif fault in ("ended", "prelaunch"):
        path = root / "collected/ended.rank7.json"
        value = json.loads(path.read_bytes())
        value["worker_exit_code" if fault == "ended" else "worker_started"] = (
            1 if fault == "ended" else False
        )
        path.write_bytes(user_request.canonical(value))
    else:
        path = root / "user_watch.jsonl"
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        if fault == "busy":
            rows[-1]["fleet"][0]["holders"] = [999]
        else:
            rows = rows[1:]
        path.write_bytes(b"".join(user_request.canonical(r) + b"\n" for r in rows))
    with pytest.raises(ValueError):
        evidence.replay_collected(root, TAG, PIN)
    assert not calls


@pytest.mark.parametrize(
    "fault", ["wrong_host", "wrong_module", "partial_planes", "extra_trace"]
)
def test_actual_physical_trace_refuses_wrong_rank_module_or_coverage(case, fault):
    root, _ = case
    dest = root / "collected"
    if fault == "extra_trace":
        (dest / "extra.xplane.pb").write_bytes(b"extra")
    else:
        path = dest / "sessions.rank7/item000/result.json.gz"
        row = json.loads(gzip.decompress(path.read_bytes()))
        raw = trace_bytes(
            "unrelated-w-7" if fault == "wrong_host" else "fixture-w-7",
            module="wrong_module" if fault == "wrong_module" else "jit_user_observer",
            planes=7 if fault == "partial_planes" else 8,
        )
        (dest / row["observations"]["trace"]["path"]).write_bytes(raw)
        row["observations"]["trace"].update(
            bytes=len(raw), sha256=sha256(raw).hexdigest()
        )
        path.write_bytes(gzip.compress(user_request.canonical(row)))
    with pytest.raises(ValueError):
        evidence.replay_collected(root, TAG, PIN)
