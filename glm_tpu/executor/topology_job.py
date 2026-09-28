"""The topology workflow: capture the fleet's physical topology (a model-free TPU job) and derive the binding the site
file pins.

:func:`capture_topology` is a fleet job (``glm_tpu.executor.jobs``) that runs ``glm_tpu.distributed.topology_capture``
on the eight hosts at once: the CPU preflight, then the capture on the TPUs (about a minute), then idle_after and the
collection of every host's ``topology.rank<r>.json``. It writes ``capture_terminal.json`` (exit codes, idle, stalled
clients, collection) and checks the eight captures together (``validate_topology_fleet``). :func:`bind_topology` reads
a finished capture run and writes a new binding directory (``topology_rebinding.json`` and ``captures/``), accepted by
the runtime's own ``load_topology_binding`` before it is reported; it contacts no host.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from glm_tpu.config.site import SiteConfig
from glm_tpu.distributed import topology_capture as worker
from glm_tpu.distributed.topology import CAPTURE_NAMES, binding_bytes, derive_topology_binding, load_topology_binding
from glm_tpu.executor.fleet import require
from glm_tpu.executor.jobs import fleet_job, source_files
from glm_tpu.utils import io_utils

TERMINAL_FILE = "capture_terminal.json"
DEFAULT_WALL_SECONDS = 900


def capture_topology(site: SiteConfig, *, repo: Path | None = None, wall_seconds: int = DEFAULT_WALL_SECONDS) -> dict:
    """Capture the topology of the site's fleet into a new run directory; returns the fleet's physical identity."""
    require(60 <= wall_seconds <= 3600, "a capture's wall deadline must be 60..3600 seconds")
    name = "topology_capture_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    with fleet_job(site, name, repo=repo) as job:
        files, manifest_sha = source_files(job.repo, job.pin, site, binding=False)
        job.stage(files)
        argv = [
            "--output",
            str(job.root),
            "--code-hash",
            job.pin,
            "--source-manifest-sha256",
            manifest_sha,
            "--site-sha256",
            site.resolved_sha256(),
            "--coordinator-address",
            site.fleet.coordinator_address,
        ]
        job.preflight(worker.MODULE, worker.ENV_FLAG, argv)
        job.release_sync()
        env = {"JAX_PLATFORMS": "tpu", worker.ENV_FLAG: "1", "PYTHONDONTWRITEBYTECODE": "1"}
        terminal = job.run(worker.MODULE, env, argv, wall_seconds=wall_seconds)
        terminal.update(job.collect(["topology.rank{rank}.json", "worker_started.rank{rank}.json"]))
        terminal.update(code_hash=job.pin, hosts=job.hosts)
        io_utils.persist(job.root / TERMINAL_FILE, terminal)
        require(
            not terminal["failed"] and not terminal["uncollected_ranks"] and terminal["collect_error"] is None,
            "topology capture failed; see " + str(job.root),
        )
        binding = binding_from_run(job.root)
        return dict(
            schema="glm_tpu_topology_capture_report_v1",
            run=str(job.root),
            code_hash=job.pin,
            hosts=binding["hosts"],
            launch_to_jax_process=binding["launch_to_jax_process"],
            topology_sha256=binding["original_topology_sha256"],
            mesh_sha256=binding["mesh_sha256"],
            fleet_sha256=binding["fleet_sha256"],
            host_to_slots=binding["host_to_slots"],
            passed=True,
        )


def binding_from_run(run: Path, **options: Any) -> dict:
    """The binding of a finished capture run: its terminal record must show every host exited 0, idle after the
    capture, no stalled client and complete collection; ``options`` go to ``derive_topology_binding``."""
    terminal = json.loads(io_utils.read_bounded(run / TERMINAL_FILE, 1 << 20))
    require(
        terminal.get("codes") == [0] * 8
        and terminal.get("failed") is False
        and terminal.get("all_hosts_idle") is True
        and terminal.get("stalled_ssh_clients") == []
        and terminal.get("uncollected_ranks") == []
        and terminal.get("divergent_records") == []
        and terminal.get("collect_error") is None,
        "the capture run did not finish cleanly on every host",
    )
    raws = [io_utils.read_bounded(run / name, 1 << 20) for name in CAPTURE_NAMES]
    options.setdefault("note", "glm-tpu topology bind of " + run.name)
    binding = derive_topology_binding(raws, all_hosts_idle_after=True, **options)
    require(binding["code_hash"] == terminal.get("code_hash"), "the captures name another commit than their run")
    return binding


def bind_topology(
    run: Path,
    output: Path,
    *,
    original_fleet_sha256: str | None = None,
    expected_topology_sha256: str | None = None,
    expected_mesh_sha256: str | None = None,
    slice_name: str | None = None,
    note: str | None = None,
) -> dict:
    """Write the binding of the capture run ``run`` into the new directory ``output`` and prove it with the
    runtime's ``load_topology_binding``; returns the values of the site's ``[topology]`` table."""
    options = dict(
        original_fleet_sha256=original_fleet_sha256,
        expected_topology_sha256=expected_topology_sha256,
        expected_mesh_sha256=expected_mesh_sha256,
        slice_name=slice_name,
    )
    if note is not None:
        options["note"] = note
    binding = binding_from_run(Path(run), **options)
    raw = binding_bytes(binding)
    pin = sha256(raw).hexdigest()
    # The runtime reads the binding from a run directory: topology_rebinding.json beside topology_capture/.
    with tempfile.TemporaryDirectory() as staged:
        (Path(staged) / "topology_capture").mkdir()
        (Path(staged) / "topology_rebinding.json").write_bytes(raw)
        for name in CAPTURE_NAMES:
            shutil.copyfile(Path(run) / name, Path(staged) / "topology_capture" / name)
        identity = load_topology_binding(
            Path(staged),
            pin,
            expected_topology=binding["original_topology_sha256"],
            expected_mesh=binding["mesh_sha256"],
            original_fleet=binding["original_fleet_sha256"],
            slice_name=binding["slice_name"],
        )
    output = Path(output)
    old = os.umask(0o077)
    try:
        output.mkdir()
        (output / "captures").mkdir()
        for name in CAPTURE_NAMES:
            io_utils.create_private_exclusive(output / "captures" / name, (Path(run) / name).read_bytes())
        io_utils.create_private_exclusive(output / "topology_rebinding.json", raw)
    finally:
        os.umask(old)
    return dict(
        schema="glm_tpu_topology_binding_report_v1",
        binding_dir=str(output),
        binding_sha256=pin,
        capture_root=str(output / "captures"),
        topology_sha256=identity["topology_sha256"],
        topology_fleet_sha256=binding["original_fleet_sha256"],
        mesh_sha256=identity["mesh_sha256"],
        slice_name=binding["slice_name"],
        fleet_sha256=identity["fleet_sha256"],
        host_to_slots=identity["host_to_slots"],
        hosts=identity["hosts"],
        code_hash=binding["code_hash"],
        passed=True,
    )
