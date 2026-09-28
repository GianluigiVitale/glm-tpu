"""Tests of :mod:`glm_tpu.executor.topology_job` and the fleet-job machinery it runs on (:mod:`glm_tpu.executor.jobs`),
against a fake eight-host fleet (``tests.fixtures.fleet``): the lease order, idle checks before and after, the staged
commit and site, the CPU preflight, the capture processes started with their start markers, cleanup of only this
job's processes on a failure, the collection, the terminal record, and the binding derived from the run."""

from __future__ import annotations

import fcntl
from hashlib import sha256
import json
from pathlib import Path

import pytest

from glm_tpu.distributed import topology_capture as worker
from glm_tpu.distributed.topology import CAPTURE_NAMES, load_topology_binding
from glm_tpu.executor import topology_job
from tests.fixtures import fleet as fake
from tests.fixtures import topology
from tests.fixtures.site import example_site


@pytest.fixture
def site_and_repo(tmp_path: Path):
    runs, locks = tmp_path / "runs", tmp_path / "locks"
    runs.mkdir(mode=0o700)
    locks.mkdir()
    repo = tmp_path / "repo"
    pin = fake.synthetic_repo(repo)
    site = example_site(tmp_path, paths=dict(run_root=str(runs)))
    return site, repo, pin


def capture_starter(site, pin, tmp_path: Path, *, failing_rank=None):
    """Each started capture process writes the capture the real worker writes for its host (tests.fixtures.topology)."""
    (tmp_path / "records").mkdir()
    raws = topology.capture_all(tmp_path / "records", site, pin)

    def start(fleet, rank, args):
        assert args["module"] == worker.MODULE and args["pin"] == pin and args["hosts"] == fake.HOSTS
        assert args["env"] == {"JAX_PLATFORMS": "tpu", worker.ENV_FLAG: "1", "PYTHONDONTWRITEBYTECODE": "1"}
        if failing_rank is not None:  # one host fails while the others still run: cleanup must end them
            return 1 if rank == failing_rank else None
        fleet.put(rank, worker.capture_file(rank), raws[rank])
        return 0

    return start, raws


def run_capture(monkeypatch, site_and_repo, tmp_path, **options):
    site, repo, pin = site_and_repo
    start, raws = capture_starter(site, pin, tmp_path, failing_rank=options.pop("failing_rank", None))
    fleet = fake.FakeJobFleet(site.paths.run_root, cpu=options.pop("cpu", fake.preflight_facts(pin)), start=start)
    leases = fake.install(monkeypatch, fleet, repo, pin)
    try:
        outcome = topology_job.capture_topology(site, **options)
    except Exception as exc:  # the outcome is what the test asserts
        outcome = exc
    return fleet, outcome, leases, raws


def test_a_capture_run_collects_every_host_and_reports_the_fleet(monkeypatch, site_and_repo, tmp_path, capsys):
    fleet, report, leases, raws = run_capture(monkeypatch, site_and_repo, tmp_path)
    site, _, pin = site_and_repo
    root = fleet.root
    assert isinstance(report, dict), repr(report)
    assert capsys.readouterr().out.splitlines()[0] == "RUN " + str(root)
    assert report["run"] == str(root) and report["code_hash"] == pin and report["passed"] is True
    assert report["hosts"] == fake.HOSTS
    assert report["launch_to_jax_process"] == {str(r): p for r, p in enumerate(topology.PERMUTATION)}
    for rank in range(8):  # rank 0 wrote into the run directory; the others were collected once
        assert (root / CAPTURE_NAMES[rank]).read_bytes() == raws[rank]
    terminal = json.loads((root / topology_job.TERMINAL_FILE).read_text())
    assert terminal == dict(
        codes=[0] * 8,
        failed=False,
        all_hosts_idle=True,
        stalled_ssh_clients=[],
        divergent_records=[],
        uncollected_ranks=[],
        collect_error=None,
        code_hash=pin,
        hosts=fake.HOSTS,
    )
    # workload leases LOCK_NB, sync leases blocking and released after the preflight, before any start
    ex, nb, un = fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN
    assert leases == [(0, ex | nb), (1, ex | nb), (2, ex), (3, ex), (2, un), (3, un)]
    order = [helper for helper, _, _ in fleet.calls]
    assert order.index("stage_bundle") < order.index("cpu") < order.index("start_worker")
    staged = fleet.get(3, "site.json")
    assert staged == site.resolved_json() and fleet.get(3, "source/README.md") == b"synthetic checkout\n"
    assert fleet.get(3, "topology_rebinding.json") is None  # a capture stages no binding
    preflights = [words for helper, _, words in fleet.calls if helper == "cpu"]
    assert all("--preflight-only" in words and f"{worker.ENV_FLAG}=1" in words for words in preflights)
    assert all("JAX_PLATFORMS=cpu" in words and worker.MODULE in words for words in preflights)
    idle_after = [args for rank, args in fleet.helper_calls("idle_probe") if args["hosts"] == fake.HOSTS]
    assert len(idle_after) == 8
    assert fleet.helper_calls("cleanup") == []


def test_a_failed_host_ends_only_this_jobs_processes_and_refuses(monkeypatch, site_and_repo, tmp_path):
    fleet, outcome, _, _ = run_capture(monkeypatch, site_and_repo, tmp_path, failing_rank=5)
    _, _, pin = site_and_repo
    assert isinstance(outcome, ValueError) and "topology capture failed" in str(outcome)
    cleanups = fleet.helper_calls("cleanup")
    assert len(cleanups) == 8 and all(
        args == dict(root=str(fleet.root), hosts=fake.HOSTS, pin=pin, module=worker.MODULE) for _, args in cleanups
    )
    terminal = json.loads((fleet.root / topology_job.TERMINAL_FILE).read_text())
    assert terminal["failed"] is True and terminal["codes"] == [-9] * 5 + [1] + [-9] * 2
    assert len([args for _, args in fleet.helper_calls("idle_probe") if args["hosts"] == fake.HOSTS]) == 8


def test_a_preflight_that_names_another_host_starts_nothing(monkeypatch, site_and_repo, tmp_path):
    _, _, pin = site_and_repo

    def cpu(fleet, rank, words):
        facts = dict(rank=rank, hostname=fake.HOSTS[(rank + 1) % 8], code_hash=pin, tpu_initialized=False)
        return 0, json.dumps(facts).encode()

    fleet, outcome, leases, _ = run_capture(monkeypatch, site_and_repo, tmp_path, cpu=cpu)
    assert isinstance(outcome, ValueError) and str(outcome) == "preflight identity differs"
    assert fleet.helper_calls("start_worker") == []
    assert (2, fcntl.LOCK_UN) not in leases  # refused while the sync leases were held; the block released them


def test_a_held_workload_lease_refuses_before_any_host_is_contacted(monkeypatch, site_and_repo, tmp_path):
    site, _, _ = site_and_repo
    with open(site.locks.workload[1], "a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fleet, outcome, _, _ = run_capture(monkeypatch, site_and_repo, tmp_path)
    assert isinstance(outcome, BlockingIOError)
    assert fleet.calls == []


def test_the_wall_deadline_is_bounded(monkeypatch, site_and_repo, tmp_path):
    _, outcome, _, _ = run_capture(monkeypatch, site_and_repo, tmp_path, wall_seconds=10)
    assert isinstance(outcome, ValueError) and "60..3600" in str(outcome)


def test_bind_writes_a_binding_the_runtime_accepts(monkeypatch, site_and_repo, tmp_path):
    fleet, report, _, raws = run_capture(monkeypatch, site_and_repo, tmp_path)
    output = tmp_path / "binding"
    bound = topology_job.bind_topology(fleet.root, output)
    raw = (output / "topology_rebinding.json").read_bytes()
    assert bound["binding_sha256"] == sha256(raw).hexdigest()
    assert bound["topology_sha256"] == report["topology_sha256"] and bound["mesh_sha256"] == report["mesh_sha256"]
    assert bound["topology_fleet_sha256"] == report["fleet_sha256"]  # a first binding
    assert bound["capture_root"] == str(output / "captures") and bound["binding_dir"] == str(output)
    assert [(output / "captures" / n).read_bytes() for n in CAPTURE_NAMES] == raws
    binding = json.loads(raw)
    assert binding["derived_by"] == "glm-tpu topology bind of " + fleet.root.name
    staged = tmp_path / "staged"
    (staged / "topology_capture").mkdir(parents=True)
    (staged / "topology_rebinding.json").write_bytes(raw)
    for name in CAPTURE_NAMES:
        (staged / "topology_capture" / name).write_bytes((output / "captures" / name).read_bytes())
    identity = load_topology_binding(
        staged,
        bound["binding_sha256"],
        expected_topology=bound["topology_sha256"],
        expected_mesh=bound["mesh_sha256"],
        original_fleet=bound["topology_fleet_sha256"],
        slice_name=bound["slice_name"],
    )
    assert identity["host_to_slots"] == report["host_to_slots"]
    with pytest.raises(FileExistsError):
        topology_job.bind_topology(fleet.root, output)


@pytest.mark.parametrize(
    "change,message",
    [
        (dict(codes=[0] * 7 + [1]), "did not finish cleanly"),
        (dict(all_hosts_idle=False), "did not finish cleanly"),
        (dict(stalled_ssh_clients=[2]), "did not finish cleanly"),
        (dict(uncollected_ranks=[4]), "did not finish cleanly"),
        (dict(code_hash="d" * 40), "another commit"),
    ],
    ids=["exit code", "idle", "stalled", "uncollected", "commit"],
)
def test_bind_refuses_a_run_that_did_not_finish_cleanly(monkeypatch, site_and_repo, tmp_path, change, message):
    fleet, _, _, _ = run_capture(monkeypatch, site_and_repo, tmp_path)
    terminal = fleet.root / topology_job.TERMINAL_FILE
    terminal.write_text(json.dumps(dict(json.loads(terminal.read_text()), **change)))
    output = tmp_path / "binding"
    with pytest.raises(ValueError, match=message):
        topology_job.bind_topology(fleet.root, output)
    assert not output.exists()


def test_bind_checks_the_expected_identity(monkeypatch, site_and_repo, tmp_path):
    fleet, report, _, _ = run_capture(monkeypatch, site_and_repo, tmp_path)
    with pytest.raises(ValueError, match="captured topology differs"):
        topology_job.bind_topology(fleet.root, tmp_path / "b1", expected_topology_sha256="0" * 64)
    bound = topology_job.bind_topology(
        fleet.root,
        tmp_path / "b2",
        original_fleet_sha256="e" * 64,
        expected_topology_sha256=report["topology_sha256"],
        expected_mesh_sha256=report["mesh_sha256"],
        note="a rebinding",
    )
    assert bound["topology_fleet_sha256"] == "e" * 64
    assert json.loads((tmp_path / "b2" / "topology_rebinding.json").read_text())["derived_by"] == "a rebinding"
