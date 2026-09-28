"""``glm-tpu topology``: capture the fleet's physical topology and derive the topology binding the site file pins.

``topology capture`` is a model-free job on the eight hosts (``glm_tpu.executor.topology_job.capture_topology``): the
site's launch policy, both workload leases, authenticated idle hosts, the staged commit, a CPU preflight, then about a
minute on the TPUs to record each host's devices, and idle_after. It prints ``RUN <run directory>`` and then the
fleet's topology, mesh and fleet digests. ``topology bind`` turns a finished capture run into a new binding directory
(``topology_rebinding.json`` and ``captures/``) and prints the values of the site's ``[topology]`` table; it contacts no
host. Registering the parser imports neither the executor nor JAX.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from glm_tpu.entrypoints.cli.types import CLISubcommand


class TopologySubcommand(CLISubcommand):
    """``glm-tpu topology capture|bind``: the report as JSON, exit 0; a refusal (``ValueError`` or ``OSError``, a
    site-file or launch-policy refusal included) prints its type and message as JSON on standard error and exits 1."""

    name = "topology"

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        try:
            from glm_tpu.executor import topology_job

            if args.action == "capture":
                from glm_tpu.config.site import SiteConfig

                report = topology_job.capture_topology(
                    SiteConfig.load(args.site), repo=args.repo, wall_seconds=args.wall_seconds
                )
            else:
                report = topology_job.bind_topology(
                    args.run,
                    args.output,
                    original_fleet_sha256=args.original_fleet_sha256,
                    expected_topology_sha256=args.expected_topology_sha256,
                    expected_mesh_sha256=args.expected_mesh_sha256,
                    slice_name=args.slice_name,
                    note=args.note,
                )
        except (ValueError, OSError) as exc:
            refusal = dict(error=type(exc).__name__, message=str(exc), status=f"topology {args.action} refused")
            print(json.dumps(refusal), file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        topology = subparsers.add_parser(
            "topology", help="capture the fleet's topology on the TPUs, or derive the topology binding"
        )
        actions = topology.add_subparsers(dest="action", required=True)
        capture = actions.add_parser(
            "capture", help="record every host's TPU devices (a model-free fleet job, about a minute on the TPUs)"
        )
        capture.add_argument(
            "--site", type=Path, help="site file (default: $GLM_TPU_SITE_CONFIG, else $GLM_TPU_CONFIG_ROOT/site.toml)"
        )
        capture.add_argument(
            "--repo", type=Path, help="git checkout to stage; must be this one (default: the site paths.repo)"
        )
        capture.add_argument(
            "--wall-seconds", type=int, default=900, help="deadline of the capture processes, 60..3600 (default: 900)"
        )
        bind = actions.add_parser("bind", help="write the topology binding of a finished capture run")
        bind.add_argument("run", type=Path, help="the capture's run directory (the RUN line of topology capture)")
        bind.add_argument("--output", type=Path, required=True, help="the new binding directory (never overwritten)")
        bind.add_argument(
            "--original-fleet-sha256",
            help="the site's topology_fleet_sha256 that this binding reassigns (default: the captured fleet)",
        )
        bind.add_argument("--expected-topology-sha256", help="refuse unless the captured topology has this digest")
        bind.add_argument("--expected-mesh-sha256", help="refuse unless the captured physical mesh has this digest")
        bind.add_argument("--slice-name", help="the site's topology.slice_name (default: the captured one)")
        bind.add_argument("--note", help="recorded as derived_by (default: names the capture run)")
        return topology
