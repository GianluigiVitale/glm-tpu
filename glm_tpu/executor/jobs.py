"""Model-free fleet jobs of the operator workflow: the topology capture and the checkpoint pack.

A job runs on rank 0 with the controller's own machinery (``glm_tpu.executor.multihost_executor``) and its rules: the
site's ``[launch]`` policy pins this checkout's commit, the remote helper texts are the pinned snapshot recorded in
``helpers.json``, a new owner-only run directory is created under the site's run root and printed (``RUN <dir>``),
both workload leases are held for the whole job (``LOCK_NB``: a live owner refuses) and the sync leases until
staging and the CPU preflight are done, and the eight hosts are authenticated idle before anything is staged. The
staged bundle is the ``git archive`` of the pinned commit with its source manifest and the resolved ``site.json``
(and, for packing, the site's topology binding), extracted by the ``stage_bundle`` helper on every host. A CPU
preflight of the job's module runs on every host (``--preflight-only``, the module's own handshake flag), then one
process per host is started through ``start_worker`` (its start marker is what ``cleanup`` authenticates) and waited
for with a wall deadline; any failure ends only this job's authenticated processes, the hosts are verified idle again
(an unresolved cleanup keeps the leases and waits for the operator, as the controller does), a local SSH client that
outlives the idle hosts is ended after a bounded wait, and each host's named records are collected once, never
overwriting (``io_utils.write_collected``). Nothing is retried. The job writes nothing outside its run directory on
rank 0; what the hosts write is the job module's.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
import fcntl
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import tarfile
import time
from typing import Any

from glm_tpu.config.site import SiteConfig, set_current_site
from glm_tpu.engine import resident_protocol as protocol
from glm_tpu.executor import fleet as remote
from glm_tpu.executor import launch_policy
from glm_tpu.executor import multihost_executor as launch
from glm_tpu.executor.fleet import remote_all, require, ssh_commands
from glm_tpu.utils import io_utils, json_utils

SSH_CLIENT_EXIT_SECONDS = launch.SSH_CLIENT_EXIT_SECONDS
POLL_SECONDS = 10


def source_files(repo: Path, pin: str, site: SiteConfig, *, binding: bool) -> tuple[dict[str, bytes], str]:
    """The files of a job's bundle: ``source/`` (``git archive`` of ``pin``), ``source_manifest.json`` (each source
    file's SHA-256), ``site.json`` (the resolved site) and, with ``binding``, the site's ``topology_rebinding.json``
    and ``topology_capture/`` checked against the site's pin (``glm_tpu.executor.staging.stage_bundle`` without the
    request). Returns the files and the source manifest's SHA-256."""
    archive = subprocess.check_output(["git", "archive", "--format=tar", pin], cwd=repo)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar:
            require(member.isfile() or member.isdir(), "release archive contains a non-regular entry")
            if member.isfile():
                files["source/" + member.name] = tar.extractfile(member).read()
    manifest = {name.removeprefix("source/"): sha256(data).hexdigest() for name, data in files.items()}
    manifest_raw = json_utils.canonical(manifest) + b"\n"
    files.update({"source_manifest.json": manifest_raw, "site.json": site.resolved_json()})
    if binding:
        directory = site.topology.binding_dir
        files["topology_rebinding.json"] = (directory / "topology_rebinding.json").read_bytes()
        require(
            sha256(files["topology_rebinding.json"]).hexdigest() == site.topology.binding_sha256,
            "site rebinding changed",
        )
        value = json.loads(files["topology_rebinding.json"])
        for rank in range(site.fleet.num_hosts):
            name = f"topology.rank{rank}.json"
            data = (directory / "captures" / name).read_bytes()
            require(sha256(data).hexdigest() == value["capture_sha256"][name], "topology capture differs")
            files["topology_capture/" + name] = data
    return files, sha256(manifest_raw).hexdigest()


def bundle(files: dict[str, bytes]) -> bytes:
    """A gzipped tar of ``files`` (owner-only regular files), as the ``stage_bundle`` helper extracts it."""
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(data))
    return result.getvalue()


class FleetJob:
    """One running job: its site, checkout, pinned commit and helper snapshot, run directory, SSH commands and the
    authenticated host list (rank = position)."""

    def __init__(self, site, repo, pin, helpers, root, commands, hosts, sync_locks):
        self.site, self.fleet, self.repo, self.pin, self.helpers = site, site.fleet, repo, pin, helpers
        self.root, self.commands, self.hosts = root, commands, hosts
        self._sync_locks = sync_locks

    def release_sync(self) -> None:
        """Release the sync leases (after staging and the preflight); the workload leases stay held."""
        for stream in self._sync_locks:
            fcntl.flock(stream, fcntl.LOCK_UN)
        self._sync_locks = []

    def stage(self, files: dict[str, bytes], *, into: Path | None = None, label: str = "stage") -> str:
        """Extract ``files`` into the run directory (or its new subdirectory ``into``) on every host."""
        root = self.root if into is None else into
        if into is not None:
            require(into.parent == self.root, "a staged subdirectory lies in the run directory")
            into.mkdir(mode=0o700)  # rank 0; the helper creates it on the other hosts
        payload = bundle(files)
        digest = sha256(payload).hexdigest()
        arguments = dict(root=str(root), digest=digest, hosts=self.hosts)
        command = remote.command(self.fleet, "stage_bundle", arguments, helpers=self.helpers)
        remote_all(self.commands, command, self.root, label, payload=payload)
        return digest

    def cpu_command(self, module: str, env_flag: str, argv: Sequence[str], *, extra: Sequence[str] = ()) -> str:
        """``cd <root>/source && env JAX_PLATFORMS=cpu <flag>=1 ... <worker_python> -m <module> <argv> <extra>``."""
        source = str(self.root / "source")
        return (
            "cd "
            + shlex.quote(source)
            + " && "
            + shlex.join(
                [
                    "env",
                    "JAX_PLATFORMS=cpu",
                    env_flag + "=1",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "PYTHONPATH=" + ":".join([source, *self.fleet.worker_pythonpath]),
                    self.fleet.worker_python,
                    "-m",
                    module,
                    *argv,
                    *extra,
                ]
            )
        )

    def host_reports(self, label: str) -> list[dict[str, Any]]:
        """The one JSON object each host printed for a CPU step ``label`` (its last line that is one)."""
        reports = []
        for rank in range(len(self.hosts)):
            lines = [s for s in (self.root / f"{label}.rank{rank}.log").read_text().splitlines() if s.startswith("{")]
            require(lines, f"{label}: host {rank} printed no report")
            reports.append(json.loads(lines[-1]))
        return reports

    def preflight(self, module: str, env_flag: str, argv: Sequence[str]) -> list[dict[str, Any]]:
        """The module's ``--preflight-only`` on every host: each host's facts, required to name its rank, its
        authenticated hostname and the pinned commit, with no TPU initialized."""
        remote_all(
            self.commands, self.cpu_command(module, env_flag, argv, extra=["--preflight-only"]), self.root, "preflight"
        )
        facts = self.host_reports("preflight")
        for rank, value in enumerate(facts):
            require(
                value.get("rank") == rank
                and value.get("hostname") == self.hosts[rank]
                and value.get("code_hash") == self.pin
                and value.get("tpu_initialized") is False,
                "preflight identity differs",
            )
        return facts

    def run(self, module: str, env: dict[str, str], argv: Sequence[str], *, wall_seconds: int) -> dict[str, Any]:
        """One process per host (``start_worker``: marker, then ``execv``), waited for; on any failure or after
        ``wall_seconds`` the authenticated processes are ended (``cleanup``); then idle_after and the bounded wait
        for the local SSH clients. Returns the exit codes, the stalled clients and whether the run failed."""
        wrapper = remote.command(
            self.fleet,
            "start_worker",
            dict(
                root=str(self.root),
                hosts=self.hosts,
                pin=self.pin,
                worker_python=self.fleet.worker_python,
                pythonpath=list(self.fleet.worker_pythonpath),
                module=module,
                env=env,
                argv=list(argv),
            ),
            helpers=self.helpers,
        )
        running, failed, started = [], False, time.monotonic()
        with ExitStack() as logs:
            try:
                for rank in range(len(self.hosts)):
                    log = logs.enter_context((self.root / f"run.rank{rank}.log").open("xb"))
                    running.append(
                        subprocess.Popen(
                            self.commands[rank][:-1] + [wrapper],
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                    )
                while any(p.poll() is None for p in running):
                    if any(p.poll() not in (None, 0) for p in running) or time.monotonic() - started > wall_seconds:
                        failed = True
                        break
                    time.sleep(POLL_SECONDS)
            except BaseException:
                failed = True
                raise
            finally:
                if failed:
                    launch.cleanup_owned(
                        self.commands,
                        self.root,
                        self.pin,
                        hosts=self.hosts,
                        fleet=self.fleet,
                        helpers=self.helpers,
                        module=module,
                    )
                try:
                    launch.idle(self.commands, self.root, "idle_after", self.fleet, self.hosts, helpers=self.helpers)
                except Exception:
                    print("Cleanup unresolved; workload leases retained. Inspect " + str(self.root), flush=True)
                    while True:  # an operator authenticates the cleanup; no retry, no lease timeout
                        time.sleep(30)
        deadline = time.monotonic() + SSH_CLIENT_EXIT_SECONDS
        while [p for p in running if p.poll() is None] and time.monotonic() < deadline:
            time.sleep(1)
        stalled = [rank for rank, p in enumerate(running) if p.poll() is None]
        for rank in stalled:
            running[rank].kill()
        codes = [p.wait() for p in running]
        return dict(codes=codes, stalled_ssh_clients=stalled, failed=failed or any(codes), all_hosts_idle=True)

    def collect(self, names: Sequence[str]) -> dict[str, Any]:
        """Each host's records ``names`` (``{rank}`` templates) into the run directory, once, host by host."""
        divergent, uncollected, error = [], [], None
        try:
            fetched = launch.fetch_records(
                self.commands,
                self.root,
                self.root,
                "collect",
                list(names),
                hosts=self.hosts,
                fleet=self.fleet,
                helpers=self.helpers,
                check=False,
            )
            for rank in range(1, len(self.hosts)):
                if fetched[rank] is None:
                    uncollected.append(rank)
                    continue
                for name, data in fetched[rank].items():
                    io_utils.write_collected(self.root, name, data, divergent)
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
        return dict(divergent_records=divergent, uncollected_ranks=uncollected, collect_error=error)


@contextmanager
def fleet_job(site: SiteConfig, name: str, *, repo: Path | None = None) -> Iterator[FleetJob]:
    """Start a job named ``name`` (its run directory under the site's run root): launch policy, helper snapshot,
    run directory, leases, SSH discovery and idle_before; the leases are held until the block ends."""
    repo = launch_policy.resolve_repo(site, repo, default=launch.REPO)
    pin = launch.source_identity(repo, site.launch)
    helpers = launch.pinned_helpers(repo, pin)
    with ExitStack() as stack:
        # the job's site and owner-only files for the block; the caller's site and umask after it
        stack.callback(set_current_site, set_current_site(site))
        stack.callback(os.umask, os.umask(0o077))
        root = site.paths.run_root / name
        root.mkdir(mode=0o700)
        print(protocol.STDOUT_RUN + str(root), flush=True)
        io_utils.persist(root / protocol.HELPERS_FILE, helpers.record())
        sync = []
        for path, blocking in [(p, False) for p in site.locks.workload] + [(p, True) for p in site.locks.sync]:
            stream = stack.enter_context(open(path, "a"))
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            if blocking:
                sync.append(stream)
        commands = ssh_commands(site.fleet)
        hosts = launch.idle(commands, root, "idle_before", site.fleet, helpers=helpers)
        require(hosts[0] == socket.gethostname(), "a fleet job must run on authenticated rank0")
        yield FleetJob(site, repo, pin, helpers, root, commands, hosts, sync)
