"""Bind the existing protected runtime to original owners and selected inputs.

No runtime initialization, CLI, deployment or launch authorization. The parent
must first use run_short_decoder_ws32._initialize_runtime, and retain its normal
leases, source checks, timeout, publication and authenticated fleet cleanup.
"""

from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import socket
from typing import Any, Callable

import numpy as np

from glm_tpu.greenfield.checkpoint.ws32_layer_subset import load_ws32_layer_subset
from glm_tpu.greenfield.runtime.ws32_decoder import _bind_weight_name_tree, ws32_decoder_weight_names
from scripts.greenfield import ws32_dense_frontier_execution as execution
from scripts.greenfield import ws32_dense_frontier_preflight as preflight_module
from scripts.greenfield import ws32_dense_frontier_prepare as preparation
from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from scripts.greenfield import ws32_dense_norm_protocol as norm_protocol
from scripts.greenfield.prefill_window_acquisition import fleet_step
from scripts.greenfield.prefill_window_evidence import same_json


def bind(*, root: Path, record: dict, runtime: tuple) -> tuple[dict, dict, dict, dict]:
    """Rebind recorded host/header identity to the actual four runtime owners."""
    jax, mesh, physical, topology, fleet = runtime
    rank = record['launch_rank']
    prior, witness = protocol.load_reference(root / 'retained_reference', rank=rank)
    raw = (root / 'retained_preflight.json').read_bytes()
    preflight = json.loads(raw)
    norm_mode = record.get('protocol') == norm_protocol.PROTOCOL
    expected_protocol = norm_protocol.PROTOCOL if norm_mode else protocol.PROTOCOL
    expected = dict(protocol=expected_protocol, code_hash=record['code_hash'],
                    launch_rank=rank, tag=record['tag'], hostname=socket.gethostname(),
                    selected_layer_ids=[0, 1], include_embedding=True, context_capacity=8192,
                    host_main_rope_table=True, selected_leaf_count=protocol.SELECTED_LEAVES,
                    payload_bytes_per_chip=protocol.PAYLOAD_BYTES,
                    original_tag=protocol.ORIGINAL_TAG, original_ledger_sha256=protocol.LEDGER_SHA)
    same_json({key: preflight[key] for key in expected}, expected, 'dense preflight identity')
    if (record.get('protocol') != expected_protocol or record.get('diagnostic_only') is not True
            or not (norm_protocol.is_tag(record['tag']) if norm_mode else protocol.is_tag(record['tag']))
            or jax.default_backend() != 'tpu'
            or jax.process_count() != 8 or jax.device_count() != 32
            or jax.process_index() != prior['jax_process_index']
            or socket.gethostname() != prior['hostname']
            or preflight['original_runner_sha256'] != sha256(
                (root / 'retained_reference' / protocol.original_names(rank)[0]).read_bytes()).hexdigest()):
        raise ValueError('dense actual runtime/original identity differs')
    for observed, key in ((physical.mesh_hash, 'mesh_sha256'),
                          (topology.topology_hash, 'topology_sha256'),
                          (fleet, 'topology_fleet_sha256')):
        if observed != prior[key]:
            raise ValueError('dense runtime physical topology differs from original')
    same_json([[int(d.id) for d in row] for row in np.asarray(mesh.devices, object)],
              physical.device_ids, 'dense runtime mesh ordering')
    if tuple(mesh.axis_names) != ('expert', 'feature'):
        raise ValueError('dense actual runtime mesh axes differ')
    slot_by_id = {int(d): slot for slot, d in enumerate(physical.flattened_device_ids)}
    devices = tuple(jax.local_devices())
    local_slots = {int(d.id): slot_by_id[int(d.id)] for d in devices}
    expected_slots = {v['device_id']: v['device_slot'] for v in prior['local_device_slots']}
    if (len(devices) != 4 or len(local_slots) != 4 or local_slots != expected_slots
            or any(d.platform != 'tpu' or d.process_index != jax.process_index() for d in devices)
            or sorted(v['device_slot'] for v in preflight['headers']) != sorted(local_slots.values())):
        raise ValueError('dense actual local owners differ from retained original')
    same_json(preflight['local_device_slots'], prior['local_device_slots'], 'dense original file owners')
    record.update(jax_process_index=jax.process_index(), physical_device_ids=physical.device_ids,
                  mesh_sha256=physical.mesh_hash, topology_sha256=topology.topology_hash,
                  topology_fleet_sha256=fleet, retained_preflight_sha256=sha256(raw).hexdigest(),
                  versions={key: version(key) for key in ('jax', 'libtpu')},
                  original_tag=protocol.ORIGINAL_TAG, original_ledger_sha256=protocol.LEDGER_SHA,
                  original_runner_sha256=preflight['original_runner_sha256'])
    return prior, witness, local_slots, preflight


def execute_bound(*, root: Path, record: dict, repo: Path, runtime: tuple,
                  consensus: Callable[[bool], bool], inspect_program: Callable[..., dict]) -> None:
    """Bind host inputs before selected load, then invoke the guarded continuation.

    The fixed inspector and independent collector must be wired by the existing
    protected parent before it authorizes launch. This helper is not a fallback
    that bypasses either. No complete-checkpoint or numerical success is emitted.
    """
    jax, mesh, physical, _, _ = runtime

    def step(name: str, action: Callable) -> Any:
        return fleet_step(name, action, root=root, record=record, consensus=consensus)

    prior, witness, slots, preflight = step('dense/bind_runtime', lambda: bind(
        root=root, record=record, runtime=runtime))
    norm_mode = record.get('protocol') == norm_protocol.PROTOCOL
    norm_originals = norm_runner = None
    if norm_mode:
        from scripts.greenfield import ws32_dense_norm_originals as norm
        def retained():
            prior_norm, originals, identity = norm.load_bundle(
                root / 'retained_norm_reference', repo=repo, rank=record['launch_rank'])
            same_json(identity, preflight['norm_originals'], 'norm retained source pins')
            norm.bind_prior(prior_norm, prior, prior_sha256=preflight['original_runner_sha256'])
            same_json(prior_norm['physical_device_ids'], physical.device_ids, 'norm actual physical owners')
            record['norm_originals'] = identity
            return prior_norm, originals
        norm_runner, norm_originals = step('norm/retained_originals', retained)

    def prepare() -> tuple:
        pins, subset = preflight_module.selected_metadata(repo, tuple(sorted(slots.values())))
        same_json(pins, preflight['checkpoint_pins'], 'dense selected checkpoint pins')
        metadata = subset.metadata
        for original in prior['local_device_slots']:
            owner = metadata.records_by_slot[original['device_slot']]
            if owner['sha256'] != original['file_sha256']:
                raise ValueError('dense selected owner file differs from original')
        same_json(preflight['headers'], [{key: metadata.records_by_slot[slot][key]
                  for key in ('device_slot', 'filename', 'file_bytes', 'header_sha256')}
                  for slot in sorted(slots.values())], 'dense retained header bytes')
        selected_prepare = preparation
        if norm_mode:
            from scripts.greenfield import ws32_dense_norm_prepare as selected_prepare
        prepared = selected_prepare.prepare(mesh, repo=repo)
        if (prepared.manifest_sha256 != prior['checkpoint_manifest_sha256']
                or metadata.manifest['source']['inventory_sha256'] != prior['source_inventory_sha256']
                or pins['expected_success_sha256'] != prior['checkpoint_success_sha256']
                or prepared.payload_bytes_per_chip != protocol.PAYLOAD_BYTES):
            raise ValueError('dense prepared checkpoint differs from original')
        tokens, rope = execution.host_inputs(prior, prepared.config)
        record.update(checkpoint_pins=pins, checkpoint_manifest_sha256=prepared.manifest_sha256,
                      checkpoint_success_sha256=prior['checkpoint_success_sha256'],
                      source_inventory_sha256=prior['source_inventory_sha256'],
                      prompt_ids_sha256=sha256(tokens.tobytes()).hexdigest(),
                      main_rope_table=prior['main_rope_table'], dsa_oracle_pins=execution.DSA_PINS,
                      dsa_pin_source_sha256=execution.DSA_PIN_SOURCE_SHA)
        return subset, prepared, tokens, rope

    subset, prepared, tokens, host_rope = step('dense/prepare_original_inputs', prepare)

    def load() -> tuple:
        loaded = load_ws32_layer_subset(subset, mesh=mesh, physical_mesh=physical)
        if (loaded.layer_ids != protocol.LAYERS or loaded.include_embedding is not True
                or loaded.payload_bytes_per_chip != protocol.PAYLOAD_BYTES
                or set(loaded.arrays) != set(prepared.tensor_names)):
            raise ValueError('dense selected loaded inventory differs')
        names = ws32_decoder_weight_names(prepared.config)
        embedding, layers = _bind_weight_name_tree((names.embedding_local, names.layers[:2]), loaded.arrays)
        record.update(local_device_slots=loaded.local_device_slots,
                      integrity_scope=loaded.integrity_scope,
                      payload_bytes_per_chip=loaded.payload_bytes_per_chip,
                      selected_layer_ids=list(loaded.layer_ids), include_embedding=True,
                      selected_load_device_memory_before=loaded.device_memory_before,
                      selected_load_device_memory_after=loaded.device_memory_after)
        if norm_mode:
            def owners(rows):
                result = {v['device_slot']: v for v in rows}
                if len(result) != 4 or len(rows) != 4:
                    raise ValueError('norm selected owner inventory differs')
                return sorted(result.values(), key=lambda value: value['device_slot'])
            same_json(owners(loaded.local_device_slots), owners(norm_runner['local_device_slots']),
                      'norm original selected weight bytes')
        return embedding, layers

    embedding, layers = step('dense/load_selected', load)

    def place_rope() -> Any:
        from jax.sharding import NamedSharding, PartitionSpec as P
        rope = jax.make_array_from_callback(host_rope.shape, NamedSharding(mesh, P()),
                                           lambda index: host_rope[index])
        jax.block_until_ready(rope)
        return rope

    rope = step('dense/place_original_rope', place_rope)
    norm_options = dict(norm_originals=norm_originals) if norm_mode else {}
    execution.execute(root=root, record=record, mesh=mesh, prepared=prepared,
                      embedding=embedding, layers=layers, tokens=tokens, rope=rope,
                      witness=witness, local_slots=slots, consensus=consensus,
                      inspect_program=inspect_program, **norm_options)
