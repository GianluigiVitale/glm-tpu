"""``glm-tpu checkpoint``: inventory and mark a checkpoint source, verify a sealed runtime checkpoint, on local files.

``checkpoint inventory`` reads the index, the safetensors headers and the configuration of a local source checkpoint
(never a tensor payload), writes the inventory to a new file and reads the file back. ``checkpoint mark-source``
hashes every shard and metadata file of the local GLM-5.3 source, compares them with the upstream digests (the
Hugging Face repository at the pinned revision, or an earlier marker) and writes the completion marker
``SOURCE_COMPLETE.json`` (``glm_tpu.model_loader.source_marker``). ``checkpoint verify`` checks a local sealed runtime
checkpoint against the pins of the site file, as a worker does before it loads one. None of them contacts another
host, takes a lock or imports JAX (``mark-source`` reads the upstream listing over the network unless it is given a
marker). The commands add no check of their own but the read-back: they print what the checkpoint library returns and
refuse as it refuses.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from glm_tpu.entrypoints.cli.types import CLISubcommand

if TYPE_CHECKING:
    from glm_tpu.config.model import ModelGeometry
    from glm_tpu.config.site import SiteConfig


def inventory_source(source: Path, output: Path, *, model_id: str, revision: str) -> dict:
    """Inventory the safetensors checkpoint in ``source`` under the identity given, write it to ``output`` and read the
    written file back; returns the inventory's identity and summary.

    ``read_source_inventory`` reads the index, every file's header and ``config.json`` (no tensor payload),
    ``write_source_inventory`` writes a new file (an existing one is refused) and ``inspect_source_inventory`` parses
    and re-hashes it.
    """
    from glm_tpu.exceptions import CheckpointValidationError
    from glm_tpu.model_loader.source_inventory import (
        inspect_source_inventory,
        read_source_inventory,
        write_source_inventory,
    )

    inventory = read_source_inventory(source, model_id=model_id, source_revision=revision)
    write_source_inventory(inventory, output)
    written = inspect_source_inventory(output)
    if written != inventory:
        raise CheckpointValidationError(f"source inventory {output} does not read back as written")
    return dict(
        schema="glm_tpu_source_inventory_report_v1",
        output=str(output),
        model_id=written.model_id,
        source_revision=written.source_revision,
        index_sha256=written.index_sha256,
        config_sha256=written.config_sha256,
        **written.summary_dict(),
    )


def verify_checkpoint(
    site: SiteConfig,
    *,
    root: Path | None = None,
    slots: Sequence[int] | None = None,
    local_slot_layout: bool = False,
    geometry: ModelGeometry | None = None,
) -> dict:
    """Verify the sealed runtime checkpoint at ``root`` (default: the site's ``checkpoint.root``) against the pins of
    ``site``, installed as the current site for the call (its approved source buckets).

    ``authenticated_inventory`` reads the site's source inventory and checks its pinned digest;
    ``verify_runtime_checkpoint`` checks the manifest, SUCCESS, mesh and topology pins, re-derives every file record
    from the inventory and ``geometry`` (default: the pinned GLM-5.3 geometry), and hashes every file, or only those of
    ``slots``; with ``local_slot_layout`` the root holds only those slots (a worker's layout).
    """
    from glm_tpu.config import model
    from glm_tpu.config.site import set_current_site
    from glm_tpu.model_loader.sharded_state.verify import verify_runtime_checkpoint
    from glm_tpu.model_loader.source_inventory import authenticated_inventory

    checkpoint, topology = site.checkpoint, site.topology
    previous = set_current_site(site)
    try:
        inventory = authenticated_inventory(checkpoint.source_inventory, checkpoint.source_inventory_sha256)
        verified = verify_runtime_checkpoint(
            checkpoint.root if root is None else root,
            expected_manifest_sha256=checkpoint.manifest_sha256,
            expected_success_sha256=checkpoint.success_sha256,
            expected_mesh_hash=topology.mesh_sha256,
            expected_topology_hash=topology.topology_sha256,
            inventory=inventory,
            geometry=model.geometry() if geometry is None else geometry,
            verify_file_hashes=True,
            verify_file_hash_slots=None if slots is None else tuple(slots),
            local_slot_layout=local_slot_layout,
        )
    finally:
        set_current_site(previous)
    return dict(
        schema="glm_tpu_checkpoint_verification_v1",
        root=str(verified.root),
        manifest_sha256=verified.manifest["manifest_sha256"],
        success_sha256=verified.success["success_sha256"],
        source_inventory_sha256=inventory.inventory_sha256,
        mesh_sha256=topology.mesh_sha256,
        topology_sha256=topology.topology_sha256,
        files=len(verified.plans),
        hashed_slots=sorted(verified.records_by_slot if slots is None else slots),
        local_slot_layout=local_slot_layout,
        passed=True,
        scope=(
            "manifest, SUCCESS seal and every file record re-derived from the source inventory and the geometry; the "
            "size of every file present and the hashes of the hashed slots; no TPU, fleet, lock or remote host"
        ),
    )


def mark_source(source: Path, output: Path | None, *, upstream_marker: Path | None, workers: int) -> dict:
    """Verify the local source in ``source`` against the upstream digests and write its completion marker (default:
    ``source/SOURCE_COMPLETE.json``, a new file): the pinned Hugging Face repository at the pinned revision, or the
    digests of an earlier marker (``upstream_marker``, to check another copy of the same source)."""
    from glm_tpu.model_loader import source_marker

    if upstream_marker is None:
        upstream, kind = source_marker.hub_listing(), "huggingface"
    else:
        upstream, kind = source_marker.marker_listing(upstream_marker), "marker"
    target = source / source_marker.MARKER if output is None else output
    return source_marker.mark_source(source, target, upstream=upstream, upstream_kind=kind, workers=workers)


class CheckpointSubcommand(CLISubcommand):
    """``glm-tpu checkpoint inventory|mark-source|verify``: the report as JSON, exit 0; a refusal (the library's
    ``ValueError`` or ``OSError``, a site-file refusal included) prints its type and message as JSON on standard error
    and exits 1."""

    name = "checkpoint"

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        try:
            if args.action == "inventory":
                report = inventory_source(args.source, args.output, model_id=args.model_id, revision=args.revision)
            elif args.action == "mark-source":
                report = mark_source(
                    args.source, args.output, upstream_marker=args.upstream_marker, workers=args.workers
                )
            else:
                from glm_tpu.config.site import SiteConfig

                report = verify_checkpoint(
                    SiteConfig.load(args.site),
                    root=args.root,
                    slots=args.slots,
                    local_slot_layout=args.local_slot_layout,
                )
        except (ValueError, OSError) as exc:
            refusal = dict(error=type(exc).__name__, message=str(exc), status=f"checkpoint {args.action} refused")
            print(json.dumps(refusal), file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        checkpoint = subparsers.add_parser(
            "checkpoint", help="inventory or mark a checkpoint source, or verify a sealed checkpoint, on local files"
        )
        actions = checkpoint.add_subparsers(dest="action", required=True)
        inventory = actions.add_parser(
            "inventory", help="write the inventory of a local safetensors source (no payload is read)"
        )
        inventory.add_argument(
            "source", type=Path, help="directory of model.safetensors.index.json, its shards and config.json"
        )
        inventory.add_argument("--output", type=Path, required=True, help="the new inventory file (never overwritten)")
        inventory.add_argument("--model-id", required=True, help="the model id to record, e.g. zai-org/GLM-5.3")
        inventory.add_argument("--revision", required=True, help="the source revision to record")
        mark = actions.add_parser(
            "mark-source",
            help="hash a local GLM-5.3 source, compare it with upstream and write SOURCE_COMPLETE.json",
        )
        mark.add_argument("source", type=Path, help="directory of the downloaded GLM-5.3 source (shards and metadata)")
        mark.add_argument(
            "--output", type=Path, help="the new marker file (default: SOURCE/SOURCE_COMPLETE.json; never overwritten)"
        )
        mark.add_argument(
            "--upstream-marker",
            type=Path,
            help="compare with the digests of this earlier marker instead of the Hugging Face repository",
        )
        mark.add_argument("--workers", type=int, default=8, help="files hashed in parallel, 1..64 (default: 8)")
        verify = actions.add_parser(
            "verify", help="verify a local sealed runtime checkpoint against the site file's pins"
        )
        verify.add_argument(
            "--site", type=Path, help="site file (default: $GLM_TPU_SITE_CONFIG, else $GLM_TPU_CONFIG_ROOT/site.toml)"
        )
        verify.add_argument("--root", type=Path, help="checkpoint root (default: the site's checkpoint.root)")
        verify.add_argument(
            "--slots", type=int, nargs="+", metavar="SLOT", help="hash only these device slots, 0..31 (default: all)"
        )
        verify.add_argument(
            "--local-slot-layout",
            action="store_true",
            help="the root holds only the --slots files (a worker's layout); any other slot file is refused",
        )
        return checkpoint
