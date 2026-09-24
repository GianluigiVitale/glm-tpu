"""CPU-only GLM-5.3 owner packing under the protected fleet workflow."""
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
# Site values (run root, model path, checkpoint namespaces, fleet naming) come
# from the run's staged, controller-resolved site.json (--site-sha256).


def preflight(args):
    if (os.environ.get('JAX_PLATFORMS') != 'cpu'
            or os.environ.get(protocol.PACK_WORKER_ENV_FLAG) != '1'
            or re.fullmatch(r'greenfield_ws32_runtime_pack_[0-9]{8}T[0-9]{15}Z', args.output.name) is None
            or re.fullmatch(r'[0-9a-f]{40}', args.code_hash) is None):
        raise ValueError('protected CPU owner-pack identity required')
    private(args.output)
    private(args.output/'site.json')
    site = SiteConfig.from_staged(args.output/'site.json', args.site_sha256)
    if args.output.parent != site.paths.run_root:
        raise ValueError('protected CPU owner-pack identity required')
    set_current_site(site)
    raw = read_bounded(args.output/'source_manifest.json', 4 << 20)
    if sha256(raw).hexdigest() != args.source_manifest_sha256:
        raise ValueError('source manifest digest differs')
    manifest = json.loads(raw)
    if SELF not in manifest:
        raise ValueError('source manifest lacks pack worker')
    for name, digest in manifest.items():
        path = REPO/name
        if not path.resolve().is_relative_to(REPO) or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('deployed packing source differs')
    model.verified_template(REPO, site.paths.model_path)
    raw = read_bounded(site.paths.model_path/'SOURCE_COMPLETE.json', 1 << 20)
    complete = json.loads(raw)
    if (sha256(raw).hexdigest() != args.source_complete_sha256
            or complete.get('passed') is not True
            or complete.get('repository') != model.MODEL_ID
            or complete.get('revision') != model.REVISION
            or complete.get('verified_shards') != 141
            or complete.get('verified_bytes') != 755632050320):
        raise ValueError('verified GLM-5.3 source completion required')
    path = args.source_inventory
    if ('..' in path.parts or not path.is_relative_to(site.checkpoint.inventory_namespace)):
        raise ValueError('source inventory namespace differs')
    raw = read_bounded(path, 64 << 20)
    if sha256(raw).hexdigest() != args.source_inventory_file_sha256:
        raise ValueError('source inventory file differs')
    from glm_tpu.model_loader.source_inventory import SourceInventory
    inventory = SourceInventory.from_dict(json.loads(raw))
    if inventory.inventory_sha256 != args.source_inventory_sha256:
        raise ValueError('source inventory self identity differs')
    model.require_inventory(inventory)
    binding = apply_topology_binding(topology_args(SimpleNamespace(), site),
        args.output, args.topology_rebinding_sha256)
    rank = site.fleet.host_rank(socket.gethostname())
    if rank is None or not 0 <= rank < 8 or binding['hosts'][rank] != socket.gethostname():
        raise ValueError('packing hostname differs from authenticated binding')
    from glm_tpu.model_loader.sharded_state.format import build_runtime_file_plans
    geometry = model.geometry(REPO)
    _, plans = build_runtime_file_plans(inventory, geometry, mesh_hash=binding['mesh_sha256'])
    slots = binding['host_to_slots'][str(rank)]
    target = site.checkpoint.namespace/args.output.name
    if target.exists() or target.is_symlink():
        raise ValueError('owner-pack target already exists; reconcile before recovery')
    required = sum(plans[s].file_bytes for s in slots) + (8 << 30)
    if shutil.disk_usage(site.checkpoint.namespace.parent).free < required:
        raise ValueError('insufficient tmpfs space for owned slots and working reserve')
    return inventory, geometry, binding, rank, slots, target, required, site


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-inventory', type=Path, required=True)
    for name in ('code-hash', 'source-manifest-sha256', 'site-sha256', 'source-complete-sha256',
                 'source-inventory-sha256', 'source-inventory-file-sha256', 'topology-rebinding-sha256'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args(argv)
    os.umask(0o077)
    inventory, geometry, binding, rank, slots, target, required, site = preflight(args)
    facts = dict(rank=rank, hostname=socket.gethostname(), code_hash=args.code_hash,
        slots=slots, source_inventory_sha256=inventory.inventory_sha256,
        mesh_sha256=binding['mesh_sha256'], checkpoint_root=str(target),
        required_tmpfs_bytes=required, tpu_initialized=False)
    if args.preflight_only:
        print(json.dumps(facts, sort_keys=True))
        return 0
    from glm_tpu.model_loader.sharded_state.format import RuntimePackConfig
    from glm_tpu.model_loader.sharded_state.writer import pack_runtime_slots
    record = pack_runtime_slots(RuntimePackConfig(
        source_root=site.paths.model_path, source_uri=site.storage.source_uri,
        output_dir=target, code_hash=args.code_hash, mesh_hash=binding['mesh_sha256']),
        inventory, geometry, device_slots=slots)
    persist(args.output/f'packing.rank{rank}.json', dict(facts, complete=True, owner_record=record))
    print(json.dumps(dict(facts, complete=True, record_sha256=record['record_sha256']), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
