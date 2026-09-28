"""CPU-only GLM-5.3 owner packing on one host, started on every host by glm-tpu checkpoint pack.

Default: pack this host's slots into a new checkpoint root. --compare-seal: re-derive chosen tensors of this host's
slots in memory and compare them and every file plan with the staged sealed manifest (writes nothing to tmpfs).
--install-seal: install the staged manifest.json and SUCCESS in this host's packed root, then verify it.
--preflight-only: the mode's checks, then this host's facts."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import socket
from types import SimpleNamespace

from glm_tpu.config.site import SiteConfig, set_current_site, topology_args
from glm_tpu.config import model
from glm_tpu.distributed.topology import apply_topology_binding
from glm_tpu.engine import resident_protocol as protocol
from glm_tpu.utils.io_utils import read_bounded, private, persist

# The staged source root and this module's own manifest path, derived from this file.
REPO = protocol.source_root(__file__, protocol.PACK_WORKER_MODULE)
SELF = Path(__file__).resolve().relative_to(REPO).as_posix()
SEAL_DIR = "seal"  # the staged sealed manifest.json (and SUCCESS) in the run directory
# Site values (run root, model path, checkpoint namespaces, fleet naming) come
# from the run's staged, controller-resolved site.json (--site-sha256).


def authenticate(args):
    """The staged run directory, the source, the completion marker, the inventory and the topology binding; this
    host's rank and slots (every mode)."""
    if (
        os.environ.get("JAX_PLATFORMS") != "cpu"
        or os.environ.get(protocol.PACK_WORKER_ENV_FLAG) != "1"
        or re.fullmatch(r"greenfield_ws32_runtime_pack_[0-9]{8}T[0-9]{15}Z", args.output.name) is None
        or re.fullmatch(r"[0-9a-f]{40}", args.code_hash) is None
    ):
        raise ValueError("protected CPU owner-pack identity required")
    private(args.output)
    private(args.output / "site.json")
    site = SiteConfig.from_staged(args.output / "site.json", args.site_sha256)
    if args.output.parent != site.paths.run_root:
        raise ValueError("protected CPU owner-pack identity required")
    set_current_site(site)
    raw = read_bounded(args.output / "source_manifest.json", 4 << 20)
    if sha256(raw).hexdigest() != args.source_manifest_sha256:
        raise ValueError("source manifest digest differs")
    manifest = json.loads(raw)
    if SELF not in manifest:
        raise ValueError("source manifest lacks pack worker")
    for name, digest in manifest.items():
        path = REPO / name
        if not path.resolve().is_relative_to(REPO) or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("deployed packing source differs")
    model.verified_template(REPO, site.paths.model_path)
    raw = read_bounded(site.paths.model_path / "SOURCE_COMPLETE.json", 1 << 20)
    complete = json.loads(raw)
    if (
        sha256(raw).hexdigest() != args.source_complete_sha256
        or complete.get("passed") is not True
        or complete.get("repository") != model.MODEL_ID
        or complete.get("revision") != model.REVISION
        or complete.get("verified_shards") != 141
        or complete.get("verified_bytes") != 755632050320
    ):
        raise ValueError("verified GLM-5.3 source completion required")
    path = args.source_inventory
    if ".." in path.parts or not path.is_relative_to(site.checkpoint.inventory_namespace):
        raise ValueError("source inventory namespace differs")
    raw = read_bounded(path, 64 << 20)
    if sha256(raw).hexdigest() != args.source_inventory_file_sha256:
        raise ValueError("source inventory file differs")
    from glm_tpu.model_loader.source_inventory import SourceInventory

    inventory = SourceInventory.from_dict(json.loads(raw))
    if inventory.inventory_sha256 != args.source_inventory_sha256:
        raise ValueError("source inventory self identity differs")
    model.require_inventory(inventory)
    binding = apply_topology_binding(
        topology_args(SimpleNamespace(), site), args.output, args.topology_rebinding_sha256
    )
    rank = site.fleet.host_rank(socket.gethostname())
    if rank is None or not 0 <= rank < 8 or binding["hosts"][rank] != socket.gethostname():
        raise ValueError("packing hostname differs from authenticated binding")
    from glm_tpu.model_loader.sharded_state.format import build_runtime_file_plans

    geometry = model.geometry(REPO)
    _, plans = build_runtime_file_plans(inventory, geometry, mesh_hash=binding["mesh_sha256"])
    slots = binding["host_to_slots"][str(rank)]
    target = site.checkpoint.namespace / args.output.name
    return inventory, geometry, binding, rank, slots, target, plans, site


def preflight(args):
    """The pack's checks: :func:`authenticate`, then a new target and enough tmpfs for the owned slots."""
    inventory, geometry, binding, rank, slots, target, plans, site = authenticate(args)
    if target.exists() or target.is_symlink():
        raise ValueError("owner-pack target already exists; reconcile before recovery")
    required = sum(plans[s].file_bytes for s in slots) + (8 << 30)
    if shutil.disk_usage(site.checkpoint.namespace.parent).free < required:
        raise ValueError("insufficient tmpfs space for owned slots and working reserve")
    return inventory, geometry, binding, rank, slots, target, required, site


def staged_seal(args, *, success):
    """The staged sealed manifest (and SUCCESS), their bytes checked against the controller's digests."""
    root = args.output / SEAL_DIR
    private(root)
    manifest_raw = read_bounded(root / "manifest.json", 64 << 20)
    if sha256(manifest_raw).hexdigest() != args.seal_manifest_sha256:
        raise ValueError("staged sealed manifest differs")
    if not success:
        return json.loads(manifest_raw), manifest_raw, None, None
    success_raw = read_bounded(root / "SUCCESS", 1 << 20)
    value = json.loads(success_raw)
    if value.get("success_sha256") != args.seal_success_sha256:
        raise ValueError("staged SUCCESS differs")
    return json.loads(manifest_raw), manifest_raw, value, success_raw


def compare_seal(args, inventory, geometry, rank, slots, plans, site):
    """Every file plan and chosen tensors of this host's slots against the staged sealed manifest; nothing written
    but this host's report ``compare.rank<r>.json`` in the run directory."""
    from glm_tpu.model_loader.sharded_state import seal

    manifest, _, _, _ = staged_seal(args, success=False)
    problems = seal.compare_plans(plans, manifest)
    records = {record["device_slot"]: record for record in manifest["files"]}
    tensors = []
    for slot in slots:
        indices = seal.canary_tensors(plans[slot], args.compare_tensors)
        derived = seal.rederive_tensors(site.paths.model_path, inventory, geometry, plans[slot], indices)
        for index in indices:
            equal = derived[index]["covered"] and derived[index]["sha256"] == records[slot]["tensor_sha256"][index]
            tensor = plans[slot].tensors[index]
            tensors.append(dict(slot=slot, index=index, name=tensor.name, dtype=tensor.dtype, equal=equal))
            if not equal:
                problems.append(f"slot {slot} tensor {index} ({tensor.name}) differs from the seal")
    report = dict(
        rank=rank,
        slots=slots,
        plans_compared=len(plans),
        tensors_compared=len(tensors),
        tensors_equal=sum(t["equal"] for t in tensors),
        tensors=tensors,
        problems=problems,
        passed=not problems,
    )
    persist(args.output / f"compare.rank{rank}.json", report)
    return report


def install_seal(args, inventory, geometry, binding, rank, slots, target, site):
    """Install the staged manifest.json and SUCCESS in this host's packed root once its own slot records equal the
    manifest's, then verify the root as a worker does before it loads it."""
    from glm_tpu.model_loader.sharded_state import seal
    from glm_tpu.model_loader.sharded_state.verify import verify_runtime_checkpoint

    manifest, manifest_raw, success, success_raw = staged_seal(args, success=True)
    if not target.is_dir() or target.is_symlink():
        raise ValueError("no packed checkpoint root to seal on this host")
    packed = json.loads(read_bounded(target / "slot_records.json", 64 << 20))
    problems = seal.compare_records(
        dict(manifest, files=[r for r in manifest["files"] if r["device_slot"] in slots]), [packed], {"0": slots}
    )
    if problems:
        raise ValueError("this host's packed slots differ from the seal: " + "; ".join(problems[:4]))
    seal.install_seal(target, manifest_raw, success_raw)
    verified = verify_runtime_checkpoint(
        target,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_success_sha256=success["success_sha256"],
        expected_mesh_hash=binding["mesh_sha256"],
        expected_topology_hash=site.topology.topology_sha256,
        inventory=inventory,
        geometry=geometry,
        verify_file_hashes=True,
        verify_file_hash_slots=tuple(slots),
        local_slot_layout=True,
    )
    return dict(
        rank=rank,
        slots=slots,
        checkpoint_root=str(target),
        manifest_sha256=verified.manifest["manifest_sha256"],
        success_sha256=verified.success["success_sha256"],
        verified=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    for name in (
        "code-hash",
        "source-manifest-sha256",
        "site-sha256",
        "source-complete-sha256",
        "source-inventory-sha256",
        "source-inventory-file-sha256",
        "topology-rebinding-sha256",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare-seal", action="store_true")
    mode.add_argument("--install-seal", action="store_true")
    parser.add_argument("--seal-manifest-sha256")
    parser.add_argument("--seal-success-sha256")
    parser.add_argument("--compare-tensors", type=int, default=8)
    args = parser.parse_args(argv)
    os.umask(0o077)
    if args.compare_seal or args.install_seal:
        return seal_mode(args)
    inventory, geometry, binding, rank, slots, target, required, site = preflight(args)
    facts = dict(
        rank=rank,
        hostname=socket.gethostname(),
        code_hash=args.code_hash,
        slots=slots,
        source_inventory_sha256=inventory.inventory_sha256,
        mesh_sha256=binding["mesh_sha256"],
        checkpoint_root=str(target),
        required_tmpfs_bytes=required,
        tpu_initialized=False,
    )
    if args.preflight_only:
        print(json.dumps(facts, sort_keys=True))
        return 0
    from glm_tpu.model_loader.sharded_state.format import RuntimePackConfig
    from glm_tpu.model_loader.sharded_state.writer import pack_runtime_slots

    record = pack_runtime_slots(
        RuntimePackConfig(
            source_root=site.paths.model_path,
            source_uri=site.storage.source_uri,
            output_dir=target,
            code_hash=args.code_hash,
            mesh_hash=binding["mesh_sha256"],
        ),
        inventory,
        geometry,
        device_slots=slots,
    )
    persist(args.output / f"packing.rank{rank}.json", dict(facts, complete=True, owner_record=record))
    print(json.dumps(dict(facts, complete=True, record_sha256=record["record_sha256"]), sort_keys=True))
    return 0


def seal_mode(args):
    """--compare-seal or --install-seal (with --preflight-only: the checks, then this host's facts)."""
    if args.seal_manifest_sha256 is None or (args.install_seal and args.seal_success_sha256 is None):
        raise ValueError("the staged seal's digests are required")
    inventory, geometry, binding, rank, slots, target, plans, site = authenticate(args)
    facts = dict(
        rank=rank,
        hostname=socket.gethostname(),
        code_hash=args.code_hash,
        slots=slots,
        checkpoint_root=str(target),
        mode="compare-seal" if args.compare_seal else "install-seal",
        tpu_initialized=False,
    )
    if args.preflight_only:
        staged_seal(args, success=args.install_seal)
        print(json.dumps(facts, sort_keys=True))
        return 0
    if args.compare_seal:
        report = compare_seal(args, inventory, geometry, rank, slots, plans, site)
    else:
        report = install_seal(args, inventory, geometry, binding, rank, slots, target, site)
    print(json.dumps(dict(facts, **{k: v for k, v in report.items() if k != "tensors"}), sort_keys=True))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
