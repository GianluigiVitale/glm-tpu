"""python -m glm_tpu.distributed.topology_capture: one host's model-free topology capture, started by ``glm-tpu topology
capture`` (``glm_tpu.executor.jobs``) on all eight hosts at once.

Before any JAX import the host authenticates its staged run directory as the worker does: the handshake flag, the
run name, the staged site (``--site-sha256``) and every file of the staged source (``--source-manifest-sha256``); its
rank is the one its hostname encodes (the site's ``fleet.host_rank_regex``). Then it joins the eight-process JAX
runtime at the site's coordinator as that process id, gathers every host's ``jax.local_devices()`` ids, describes the
devices (``glm_tpu.distributed.topology.discover_topology``, ``require_v4_64``), all-gathers the digest of the
contract (the staged commit, the topology and its digest, the physical mesh digest) and requires every host to agree,
and writes ``topology.rank<r>.json`` (created once, owner-only), the record ``validate_topology_fleet`` authenticates.
No model, no checkpoint, no device program. ``--preflight-only`` runs the authentication on the CPU and prints this
host's facts.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import socket

from glm_tpu.config.site import SiteConfig, set_current_site
from glm_tpu.engine import resident_protocol as protocol
from glm_tpu.utils import io_utils

MODULE = "glm_tpu.distributed.topology_capture"
ENV_FLAG = "GLM_TPU_TOPOLOGY_CAPTURE"
RUN_NAME = re.compile(r"topology_capture_[0-9]{8}T[0-9]{12}Z")
OK_MARKER = "TOPOLOGY_CAPTURE_OK "
REPO = protocol.source_root(__file__, MODULE)
SELF = Path(__file__).resolve().relative_to(REPO).as_posix()


def capture_file(rank: int) -> str:
    return f"topology.rank{rank}.json"


def preflight(args: argparse.Namespace) -> tuple[SiteConfig, int]:
    """Authenticate the staged run directory and this host's rank before any device is opened."""
    root = args.output
    if (
        os.environ.get(ENV_FLAG) != "1"
        or RUN_NAME.fullmatch(root.name) is None
        or re.fullmatch(r"[0-9a-f]{40}", args.code_hash) is None
    ):
        raise ValueError("protected topology capture identity required")
    io_utils.private(root)
    for name in ("site.json", "source_manifest.json"):
        io_utils.private(root / name)
    site = SiteConfig.from_staged(root / "site.json", args.site_sha256)
    if root.parent != site.paths.run_root:
        raise ValueError("protected topology capture identity required")
    set_current_site(site)
    if args.coordinator_address != site.fleet.coordinator_address:
        raise ValueError("coordinator address differs from the staged site")
    raw = io_utils.read_bounded(root / "source_manifest.json", 4 << 20)
    if sha256(raw).hexdigest() != args.source_manifest_sha256:
        raise ValueError("source manifest digest differs")
    manifest = json.loads(raw)
    if not isinstance(manifest, dict) or SELF not in manifest:
        raise ValueError("source manifest lacks the topology capture")
    for name, digest in manifest.items():
        path = REPO / name
        if not path.resolve().is_relative_to(REPO) or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("deployed capture source differs")
    rank = site.fleet.host_rank(socket.gethostname())
    if rank is None or not 0 <= rank < site.fleet.num_hosts:
        raise ValueError("capture rank differs")
    if (root / capture_file(rank)).exists():
        raise ValueError("a capture of this run exists on this host; a capture is never repeated")
    return site, rank


def capture(args: argparse.Namespace, site: SiteConfig, rank: int) -> dict:
    """Join the fleet as process ``rank``, describe and cross-check the devices, write this host's capture."""
    import jax
    from jax.experimental import multihost_utils
    import numpy as np

    from glm_tpu.distributed.mesh import build_physical_mesh
    from glm_tpu.distributed.topology import discover_topology, require_v4_64

    hosts = site.fleet.num_hosts
    jax.distributed.initialize(coordinator_address=args.coordinator_address, num_processes=hosts, process_id=rank)
    try:
        # JAX may order its processes differently from the hosts' ranks: both identities are kept.
        local = np.asarray([device.id for device in jax.local_devices()], dtype=np.int32)
        fleet = np.asarray(multihost_utils.process_allgather(local)).reshape(hosts, jax.local_device_count())
        order = {int(device): index for row in fleet.tolist() for index, device in enumerate(row)}
        if len(order) != fleet.size:
            raise RuntimeError("the hosts' local device lists repeat a device")
        topology = discover_topology(jax.devices(), slice_name=site.topology.slice_name, local_device_ids=order)
        require_v4_64(topology)
        for process, row in enumerate(fleet.tolist()):
            if set(row) != {d.device_id for d in topology.devices if d.process_index == process}:
                raise RuntimeError(f"process {process}: its gathered local devices differ from the device inventory")
        contract = dict(
            code_hash=args.code_hash,
            mesh_sha256=build_physical_mesh(topology).mesh_hash,
            topology=topology.to_dict(),
            topology_hash=topology.topology_hash,
        )
        contract_hash = sha256(
            json.dumps(contract, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        digest = np.frombuffer(bytes.fromhex(contract_hash), dtype=np.uint8)
        digests = np.asarray(multihost_utils.process_allgather(digest)).reshape(hosts, len(digest))
        if not np.all(digests == digests[0]):
            raise RuntimeError("the hosts disagree on the topology contract")
        multihost_utils.sync_global_devices("glm-tpu-topology-" + contract_hash[:16])
        record = dict(
            captured_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            contract=contract,
            contract_hash=contract_hash,
            fleet_contract_hashes=[row.tobytes().hex() for row in digests],
            fleet_local_device_ids_in_runtime_order=fleet.tolist(),
            hostname=socket.gethostname(),
            jax_device_count=jax.device_count(),
            jax_local_device_count=jax.local_device_count(),
            jax_process_count=jax.process_count(),
            jax_process_index=jax.process_index(),
            jax_version=jax.__version__,
            launch_process_id=rank,
            local_device_ids=sorted(int(device.id) for device in jax.local_devices()),
            schema_version=1,
        )
        io_utils.create_private_exclusive(
            args.output / capture_file(rank), (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
        )
        return record
    finally:
        jax.distributed.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("code-hash", "source-manifest-sha256", "site-sha256", "coordinator-address"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    os.umask(0o077)
    site, rank = preflight(args)
    if args.preflight_only:
        facts = dict(rank=rank, hostname=socket.gethostname(), code_hash=args.code_hash, tpu_initialized=False)
        print(json.dumps(facts, sort_keys=True), flush=True)
        return 0
    record = capture(args, site, rank)
    print(
        OK_MARKER
        + json.dumps(
            dict(
                launch_process_id=rank,
                jax_process_index=record["jax_process_index"],
                hostname=record["hostname"],
                contract_hash=record["contract_hash"],
            ),
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
