"""Tests of :mod:`glm_tpu.executor.topology_job` and the fleet-job machinery it runs on (:mod:`glm_tpu.executor.jobs`),
against a fake eight-host fleet (``tests.fixtures.fleet``): the lease order, idle checks before and after, the staged
commit and site, the CPU preflight, the capture processes started with their start markers, cleanup of only this
job's processes on a failure, the collection, the terminal record, and the binding derived from the run."""

from __future__ import annotations

import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from glm_tpu.distributed import topology_capture as worker
from glm_tpu.distributed.topology import CAPTURE_NAMES, binding_bytes, derive_topology_binding, load_topology_binding
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
        missing=[],
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
        (dict(missing=["4:topology.rank4.json"]), "did not finish cleanly"),
        (dict(divergent_records=["topology.rank2.json"]), "did not finish cleanly"),
        (dict(collect_error=dict(type="OSError", message="x")), "did not finish cleanly"),
        (dict(code_hash="d" * 40), "another commit"),
        (dict(hosts=[*fake.HOSTS[:7], "example-host"]), "other hosts"),
    ],
    ids=["exit code", "idle", "stalled", "uncollected", "missing", "divergent", "collect error", "commit", "hosts"],
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


def test_bind_refuses_a_run_directory_that_is_not_a_capture_run(monkeypatch, site_and_repo, tmp_path):
    fleet, _, _, _ = run_capture(monkeypatch, site_and_repo, tmp_path)
    fleet.root.chmod(0o755)
    with pytest.raises(ValueError, match="owner-only"):
        topology_job.bind_topology(fleet.root, tmp_path / "b1")
    fleet.root.chmod(0o700)
    renamed = fleet.root.with_name("capture-copy")
    fleet.root.rename(renamed)
    with pytest.raises(ValueError, match="named topology_capture_<UTC>"):
        topology_job.bind_topology(renamed, tmp_path / "b2")


def test_a_preflight_naming_another_commit_starts_nothing(monkeypatch, site_and_repo, tmp_path):
    _, _, pin = site_and_repo
    cpu = fake.preflight_facts(pin, rank6=dict(code_hash="e" * 40))
    fleet, outcome, _, _ = run_capture(monkeypatch, site_and_repo, tmp_path, cpu=cpu)
    assert isinstance(outcome, ValueError) and str(outcome) == "preflight identity differs"
    assert fleet.helper_calls("start_worker") == []


@pytest.mark.parametrize(
    "change,message",
    [(dict(jax="0.9.0"), "capture environments differ"), (dict(libtpu="0.0.40"), "capture environments differ")],
    ids=["jax on one host", "libtpu on one host"],
)
def test_hosts_with_another_environment_start_nothing(monkeypatch, site_and_repo, tmp_path, change, message):
    _, _, pin = site_and_repo
    fleet, outcome, _, _ = run_capture(
        monkeypatch, site_and_repo, tmp_path, cpu=fake.preflight_facts(pin, rank2=change)
    )
    assert isinstance(outcome, ValueError) and message in str(outcome)
    assert fleet.helper_calls("start_worker") == []


def test_every_host_on_another_jax_starts_nothing(monkeypatch, site_and_repo, tmp_path):
    _, _, pin = site_and_repo
    other = {f"rank{r}": dict(jax="0.9.0", jaxlib="0.9.0") for r in range(8)}
    fleet, outcome, _, _ = run_capture(monkeypatch, site_and_repo, tmp_path, cpu=fake.preflight_facts(pin, **other))
    assert isinstance(outcome, ValueError) and "JAX version differs" in str(outcome)
    assert fleet.helper_calls("start_worker") == []


class Clock:
    """The jobs module's time: each monotonic() call advances 100 s; sleep() only counts."""

    def __init__(self):
        self.now, self.sleeps = 0.0, 0

    def monotonic(self):
        self.now += 100.0
        return self.now

    def sleep(self, seconds):
        self.sleeps += 1


def test_the_wall_deadline_ends_the_processes(monkeypatch, site_and_repo, tmp_path):
    site, repo, pin = site_and_repo
    (tmp_path / "records").mkdir()
    fleet = fake.FakeJobFleet(site.paths.run_root, cpu=fake.preflight_facts(pin), start=lambda f, r, a: None)
    fake.install(monkeypatch, fleet, repo, pin)
    from glm_tpu.executor import jobs

    monkeypatch.setattr(jobs, "time", Clock())
    with pytest.raises(ValueError, match="topology capture failed"):
        topology_job.capture_topology(site, wall_seconds=120)
    assert len(fleet.helper_calls("cleanup")) == 8  # every hung capture ended by its start marker
    terminal = json.loads((fleet.root / topology_job.TERMINAL_FILE).read_text())
    assert terminal["failed"] is True and terminal["codes"] == [-9] * 8
    assert terminal["missing"] == [f"{r}:topology.rank{r}.json" for r in range(8)]


def test_an_unresolved_cleanup_keeps_the_leases(monkeypatch, site_and_repo, tmp_path):
    site, repo, pin = site_and_repo
    start, _ = capture_starter(site, pin, tmp_path, failing_rank=1)
    fleet = fake.FakeJobFleet(site.paths.run_root, cpu=fake.preflight_facts(pin), start=start, idle_failure=True)
    leases = fake.install(monkeypatch, fleet, repo, pin)
    from glm_tpu.executor import jobs

    class Held(Exception):
        pass

    seen = []

    class Waiting:
        def monotonic(self):
            import time

            return time.monotonic()

        def sleep(self, seconds):
            if seconds == 30:  # the "Cleanup unresolved" wait: the workload leases must still be held
                for path in site.locks.workload:
                    with open(path, "a") as other, pytest.raises(BlockingIOError):
                        fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen.append(seconds)
                raise Held

    monkeypatch.setattr(jobs, "time", Waiting())
    with pytest.raises(Held):
        topology_job.capture_topology(site)
    assert seen == [30] and (0, fcntl.LOCK_UN) not in leases and (1, fcntl.LOCK_UN) not in leases


def test_the_real_capture_preflight_accepts_what_the_job_stages(monkeypatch, site_and_repo, tmp_path, capsys):
    """Rank 0's CPU step runs the capture module's own main on what the job staged: a checkout holding this
    package's files, the staged site, the job's argv."""
    import contextlib
    import io
    import shutil
    import subprocess

    from glm_tpu.engine import resident_protocol as protocol

    site, _, _ = site_and_repo
    repo = tmp_path / "package-repo"
    shutil.copytree(protocol.source_root(worker.__file__, worker.MODULE) / "glm_tpu", repo / "glm_tpu")
    for command in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", "x"]):
        subprocess.run(command, cwd=repo, check=True, capture_output=True, env=dict(os.environ, **fake.GIT_ENV))
    pin = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    synthetic = fake.preflight_facts(pin)
    ran = []

    def cpu(fleet, rank, words):
        if rank:
            return synthetic(fleet, rank, words)
        argv = words[words.index("-m") + 2 :]
        assert argv[argv.index("--output") + 1] == str(fleet.root)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = worker.main(argv)
        ran.append(json.loads(stdout.getvalue()))
        facts = dict(ran[0], **fake.ENVIRONMENT)  # the other hosts report the synthetic environment
        return code, (json.dumps(facts) + "\n").encode()

    start, _ = capture_starter(site, pin, tmp_path)
    fleet = fake.FakeJobFleet(site.paths.run_root, cpu=cpu, start=start)
    fake.install(monkeypatch, fleet, repo, pin)
    monkeypatch.setenv(worker.ENV_FLAG, "1")
    report = topology_job.capture_topology(site)
    assert report["passed"] is True
    assert ran and ran[0]["rank"] == 0 and ran[0]["code_hash"] == pin and ran[0]["tpu_initialized"] is False
    assert sorted(
        rank for helper, rank, words in fleet.calls if helper == "cpu" and "--preflight-only" in words
    ) == list(range(8))


def test_the_runtime_initializer_accepts_the_captured_fleet(site_and_repo, tmp_path, monkeypatch):
    """parallel_state.initialize_runtime, host by host on the fake JAX runtime, over the eight captures and the
    binding bind derives: its live checks (hostname, JAX process, local devices, every device record, the mesh)."""
    import argparse
    import sys
    from types import ModuleType
    from unittest import mock

    from glm_tpu.distributed import parallel_state
    from glm_tpu.distributed.topology import apply_topology_binding

    site, _, pin = site_and_repo
    (tmp_path / "records").mkdir()
    raws = topology.capture_all(tmp_path / "records", site, pin)
    binding = derive_topology_binding(raws, all_hosts_idle_after=True, note="test")
    staged = tmp_path / "staged"
    (staged / "topology_capture").mkdir(parents=True)
    raw = binding_bytes(binding)
    (staged / "topology_rebinding.json").write_bytes(raw)
    for name, data in zip(CAPTURE_NAMES, raws, strict=True):
        (staged / "topology_capture" / name).write_bytes(data)
    sharding = ModuleType("jax.sharding")
    sharding.Mesh = lambda devices, axes: ("mesh", devices.shape, axes)
    fleets = []
    for rank in range(8):
        args = argparse.Namespace(
            coordinator_address=site.fleet.coordinator_address,
            num_processes=8,
            process_id=rank,
            topology_sha256=binding["original_topology_sha256"],
            topology_fleet_sha256=binding["original_fleet_sha256"],
            mesh_sha256=binding["mesh_sha256"],
            slice_name=site.topology.slice_name,
        )
        apply_topology_binding(args, staged, sha256(raw).hexdigest())
        with topology.fake_jax(rank) as jax, mock.patch.dict(sys.modules, {"jax.sharding": sharding}):
            jax.default_backend = lambda: "tpu"
            jax.sharding = sharding
            monkeypatch.setattr(parallel_state.socket, "gethostname", lambda rank=rank: fake.HOSTS[rank])
            _, mesh, physical, _, fleet_sha = parallel_state.initialize_runtime(args)
        assert mesh == ("mesh", (8, 4), ("expert", "feature")) and physical.mesh_hash == binding["mesh_sha256"]
        fleets.append(fleet_sha)
    assert fleets == [binding["fleet_sha256"]] * 8
