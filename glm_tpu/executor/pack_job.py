"""``glm-tpu checkpoint pack``: pack, seal and install the runtime checkpoint on the eight hosts (a CPU fleet job).

A fleet job (``glm_tpu.executor.jobs``) that stages the pinned commit, the resolved site and the site's topology
binding, runs ``python -m glm_tpu.model_loader.pack_worker`` on every host and collects each host's receipt. Four
modes:

* **pack** (the default): each host packs its four slots into ``checkpoint.root`` (a new directory in the site's
  checkpoint namespace; the run directory has the same name), hashing every file and tensor it writes. The eight
  receipts are then combined into the manifest (``assemble_owner_manifest``, with the upstream SHA-256 of every
  source shard from the completion marker), the ``SUCCESS`` seal is written for it (with the digests of this run's
  preflight, terminal and post-run idle records), both are staged into the run directory's ``seal/`` and every host
  installs them into its root and verifies the root as a worker does. The report holds the new
  ``manifest_sha256`` and ``success_sha256`` for the site's ``[checkpoint]`` table.
* **recover** (``seal``): the same pack, but the eight receipts must equal the given sealed manifest on every file
  record, and only then are that seal's own ``manifest.json`` and ``SUCCESS`` installed, byte for byte; the site's
  pins must already be that seal's. For a checkpoint lost from tmpfs (a host restart) whose seal was kept.
* **compare** (``compare``): nothing is packed and nothing is written to tmpfs. Each host re-derives chosen tensors of
  its slots from the source in memory (``tensors`` per slot, 0: all) and compares them, and every file plan, with the
  given sealed manifest: a dry run that shows the pinned source, geometry and placement still produce that seal.
* **preflight** (``preflight_only``): the pack's own checks on every host (the staged source and pins, the completion
  marker, the inventory, the binding, a new target and enough tmpfs), then each host's facts; nothing is packed.

Nothing is retried; an existing target is refused on the host, and a failed pack leaves its partial files for the
operator (``OPERATIONS``). The workload leases are held for the whole job, the sync leases until staging and the
preflight are done.
"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import time
from typing import Any

from glm_tpu.config.site import SiteConfig
from glm_tpu.engine import resident_protocol as protocol
from glm_tpu.executor.fleet import remote_all, require
from glm_tpu.executor.jobs import FleetJob, fleet_job, source_files
from glm_tpu.utils import io_utils

MODULE = protocol.PACK_WORKER_MODULE
ENV = {"JAX_PLATFORMS": "cpu", protocol.PACK_WORKER_ENV_FLAG: "1", "PYTHONDONTWRITEBYTECODE": "1"}
TAG = re.compile(r"greenfield_ws32_runtime_pack_[0-9]{8}T[0-9]{15}Z")
SEAL_DIR = "seal"
DEFAULT_WALL_SECONDS = 5 * 3600


def new_tag() -> str:
    """A fresh pack run name, ``greenfield_ws32_runtime_pack_<YYYYMMDD>T<HHMMSS><nanoseconds>Z``."""
    ns = time.time_ns()
    return (
        "greenfield_ws32_runtime_pack_"
        + datetime.fromtimestamp(ns // 10**9, UTC).strftime("%Y%m%dT%H%M%S")
        + (f"{ns % 10**9:09d}Z")
    )


def read_seal(directory: Path, *, success: bool) -> dict[str, Any]:
    """A kept seal: ``manifest.json`` (self-hashed) and, with ``success``, its ``SUCCESS`` (self-hashed, naming the
    manifest file's SHA-256)."""
    from glm_tpu.model_loader.sharded_state.format import mapping_hash

    manifest_raw = io_utils.read_bounded(Path(directory) / "manifest.json", 64 << 20)
    manifest = json.loads(manifest_raw)
    require(
        manifest.get("manifest_sha256") == mapping_hash(manifest, field="manifest_sha256"),
        "the sealed manifest's self-hash differs",
    )
    seal = dict(manifest=manifest, manifest_raw=manifest_raw, success=None, success_raw=None)
    if success:
        success_raw = io_utils.read_bounded(Path(directory) / "SUCCESS", 1 << 20)
        value = json.loads(success_raw)
        require(
            value.get("success_sha256") == mapping_hash(value, field="success_sha256")
            and value.get("manifest_file_sha256") == sha256(manifest_raw).hexdigest()
            and value.get("manifest_sha256") == manifest["manifest_sha256"],
            "the SUCCESS seal does not seal this manifest",
        )
        seal.update(success=value, success_raw=success_raw)
    return seal


def source_ledger(site: SiteConfig, inventory) -> dict[str, str]:
    """The upstream SHA-256 of every source shard, from the completion marker the site pins."""
    raw = io_utils.read_bounded(site.paths.model_path / "SOURCE_COMPLETE.json", 1 << 20)
    require(sha256(raw).hexdigest() == site.checkpoint.source_complete_sha256, "the source completion marker differs")
    marker = json.loads(raw)
    ledger = {entry["name"]: entry["sha256"] for entry in marker.get("shards", [])}
    require(set(ledger) == {f.filename for f in inventory.files}, "the completion marker does not cover the inventory")
    return ledger


def worker_argv(job: FleetJob, manifest_sha: str, inventory_file_sha: str) -> list[str]:
    site = job.site
    return [
        "--output",
        str(job.root),
        "--source-inventory",
        str(site.checkpoint.source_inventory),
        "--code-hash",
        job.pin,
        "--source-manifest-sha256",
        manifest_sha,
        "--site-sha256",
        site.resolved_sha256(),
        "--source-complete-sha256",
        site.checkpoint.source_complete_sha256,
        "--source-inventory-sha256",
        site.checkpoint.source_inventory_sha256,
        "--source-inventory-file-sha256",
        inventory_file_sha,
        "--topology-rebinding-sha256",
        site.topology.binding_sha256,
    ]


def pack_checkpoint(
    site: SiteConfig,
    *,
    seal: Path | None = None,
    compare: Path | None = None,
    tensors: int = 8,
    preflight_only: bool = False,
    repo: Path | None = None,
    wall_seconds: int = DEFAULT_WALL_SECONDS,
) -> dict[str, Any]:
    """Run one pack job in the mode the arguments select (module docstring); returns the report."""
    from glm_tpu.config import model
    from glm_tpu.model_loader.sharded_state.seal import compare_records, json_bytes, success_record
    from glm_tpu.model_loader.source_inventory import authenticated_inventory

    require(sum(bool(x) for x in (seal, compare, preflight_only)) <= 1, "choose one of seal, compare, preflight only")
    require(600 <= wall_seconds <= 86400, "a pack's wall deadline must be 600..86400 seconds")
    require(tensors >= 0, "the canary tensor count must be 0 (all) or positive")
    packing = compare is None and not preflight_only
    tag = site.checkpoint.root.name if packing else new_tag()
    require(TAG.fullmatch(tag) is not None, "checkpoint.root must be named greenfield_ws32_runtime_pack_<UTC>")
    require(packing or tag != site.checkpoint.root.name, "a dry run never uses the checkpoint's own name")
    kept = read_seal(seal or compare, success=seal is not None) if (seal or compare) else None
    if seal is not None:
        require(
            kept["manifest"]["manifest_sha256"] == site.checkpoint.manifest_sha256
            and kept["success"]["success_sha256"] == site.checkpoint.success_sha256,
            "the site's checkpoint pins are not this seal's",
        )
    if packing:
        require(not site.checkpoint.root.exists(), "the checkpoint root already exists on rank 0; never repack it")
    inventory = authenticated_inventory(site.checkpoint.source_inventory, site.checkpoint.source_inventory_sha256)
    model.require_inventory(inventory)
    inventory_file_sha = sha256(io_utils.read_bounded(site.checkpoint.source_inventory, 64 << 20)).hexdigest()
    ledger = None if (compare or preflight_only or seal) else source_ledger(site, inventory)
    with fleet_job(site, tag, repo=repo) as job:
        files, manifest_sha = source_files(job.repo, job.pin, site, binding=True)
        if kept is not None:
            files[SEAL_DIR + "/manifest.json"] = kept["manifest_raw"]
            if kept["success_raw"] is not None:
                files[SEAL_DIR + "/SUCCESS"] = kept["success_raw"]
        job.stage(files)
        argv = worker_argv(job, manifest_sha, inventory_file_sha)
        if compare is not None:
            argv += ["--compare-seal", "--seal-manifest-sha256", sha256(kept["manifest_raw"]).hexdigest()]
            argv += ["--compare-tensors", str(tensors)]
        facts = job.preflight(MODULE, protocol.PACK_WORKER_ENV_FLAG, argv)
        binding = json.loads(files["topology_rebinding.json"])
        require(
            all(value["slots"] == binding["host_to_slots"][str(rank)] for rank, value in enumerate(facts)),
            "a host's slots differ from the topology binding",
        )
        io_utils.persist(job.root / "pack_preflight.json", dict(hosts=job.hosts, facts=facts, code_hash=job.pin))
        job.release_sync()
        report = dict(run=str(job.root), code_hash=job.pin, hosts=job.hosts, facts=facts)
        if preflight_only:
            return dict(report, schema="glm_tpu_pack_preflight_report_v1", passed=True)
        terminal = job.run(MODULE, ENV, argv, wall_seconds=wall_seconds)
        names = ["compare.rank{rank}.json" if compare else "packing.rank{rank}.json", "worker_started.rank{rank}.json"]
        terminal.update(job.collect(names))
        io_utils.persist(job.root / "pack_terminal.json", terminal)
        require(
            not terminal["failed"] and not terminal["uncollected_ranks"] and terminal["collect_error"] is None,
            "the pack job failed; see " + str(job.root),
        )
        if compare is not None:
            reports = [json.loads((job.root / f"compare.rank{rank}.json").read_text()) for rank in range(8)]
            return dict(
                report,
                schema="glm_tpu_pack_compare_report_v1",
                manifest_file_sha256=sha256(kept["manifest_raw"]).hexdigest(),
                manifest_sha256=kept["manifest"]["manifest_sha256"],
                slots_compared=sorted(s for r in reports for s in r["slots"]),
                plans_compared=[r["plans_compared"] for r in reports],
                tensors_compared=sum(r["tensors_compared"] for r in reports),
                tensors_equal=sum(r["tensors_equal"] for r in reports),
                problems=[p for r in reports for p in r["problems"]],
                passed=all(r["passed"] for r in reports),
            )
        receipts = [json.loads((job.root / f"packing.rank{rank}.json").read_text()) for rank in range(8)]
        require(
            all(r["rank"] == rank and r["complete"] is True for rank, r in enumerate(receipts)),
            "a pack receipt is incomplete",
        )
        owner_records = [r["owner_record"] for r in receipts]
        if seal is not None:
            problems = compare_records(kept["manifest"], owner_records, binding["host_to_slots"])
            io_utils.persist(job.root / "seal_compare.json", dict(problems=problems, passed=not problems))
            require(
                not problems, "the packed slots differ from the seal; nothing installed: " + "; ".join(problems[:4])
            )
            manifest, manifest_raw, success, success_raw = (
                kept["manifest"],
                kept["manifest_raw"],
                kept["success"],
                kept["success_raw"],
            )
        else:
            from glm_tpu.model_loader.sharded_state.manifest import assemble_owner_manifest

            manifest = assemble_owner_manifest(
                inventory=inventory,
                geometry=model.geometry(),
                code_hash=job.pin,
                mesh_hash=binding["mesh_sha256"],
                source_uri=site.storage.source_uri,
                owner_records=owner_records,
                host_to_slots=binding["host_to_slots"],
                source_file_sha256=ledger,
            )
            manifest_raw = json_bytes(manifest)
            idle = [(job.root / f"idle_after.rank{rank}.log").read_text().strip() for rank in range(8)]
            post = dict(hosts=job.hosts, idle_after=idle, all_hosts_idle=terminal["all_hosts_idle"])
            io_utils.persist(job.root / "pack_post_census.json", post)
            success = success_record(
                manifest,
                manifest_raw,
                tag=tag,
                topology_hash=site.topology.topology_sha256,
                post_census_sha256=sha256((job.root / "pack_post_census.json").read_bytes()).hexdigest(),
                remote_preflight_sha256=sha256((job.root / "pack_preflight.json").read_bytes()).hexdigest(),
                remote_terminal_sha256=sha256((job.root / "pack_terminal.json").read_bytes()).hexdigest(),
            )
            success_raw = json_bytes(success)
            job.stage({"manifest.json": manifest_raw, "SUCCESS": success_raw}, into=job.root / SEAL_DIR, label="seal")
        install = argv + ["--install-seal", "--seal-manifest-sha256", sha256(manifest_raw).hexdigest()]
        install += ["--seal-success-sha256", success["success_sha256"]]
        remote_all(job.commands, job.cpu_command(MODULE, protocol.PACK_WORKER_ENV_FLAG, install), job.root, "install")
        installed = job.host_reports("install")
        require(
            all(r.get("rank") == rank and r.get("verified") is True for rank, r in enumerate(installed)),
            "a host did not verify its sealed checkpoint",
        )
        return dict(
            report,
            schema="glm_tpu_pack_report_v1",
            mode="recover" if seal is not None else "pack",
            checkpoint_root=str(site.checkpoint.root),
            manifest_sha256=manifest["manifest_sha256"],
            success_sha256=success["success_sha256"],
            manifest_file_sha256=sha256(manifest_raw).hexdigest(),
            installed=installed,
            passed=True,
        )
