"""Assemble distributed owner receipts using the retained checkpoint contract.

Each host hashes its actual owner files during packing. This metadata assembly
does not claim to read remote payloads; the protected workflow authenticates the
eight receipts and the runtime verifies local payloads again before loading.
"""
from pathlib import Path

from glm_tpu.model_loader.sharded_state import format as retained
from glm_tpu.model_loader.sharded_state import verify


def assemble_owner_manifest(*, inventory, geometry, code_hash, mesh_hash,
                            source_uri, owner_records, host_to_slots,
                            source_file_sha256):
    report, plans = retained.build_runtime_file_plans(
        inventory, geometry, mesh_hash=mesh_hash)
    if set(host_to_slots) != {str(i) for i in range(8)} or len(owner_records) != 8:
        raise ValueError('eight owner receipts and host bindings required')
    slots = [s for values in host_to_slots.values() for s in values]
    if (any(len(v) != 4 for v in host_to_slots.values())
            or sorted(slots) != list(range(32))):
        raise ValueError('host binding must cover 32 unique slots')
    files = []
    for rank, record in enumerate(owner_records):
        expected = dict(artifact_kind=retained.RUNTIME_SLOT_RECORD_KIND,
            code_hash=code_hash, format_version=1, geometry_sha256=geometry.geometry_hash,
            mesh_hash=mesh_hash, placement_sha256=report.placement_sha256,
            plan_id='WS32_2D', slots=sorted(host_to_slots[str(rank)]),
            source_inventory_sha256=inventory.inventory_sha256)
        if (any(record.get(k) != v for k, v in expected.items())
                or record.get('record_sha256') != retained._mapping_hash(record, field='record_sha256')
                or sorted(f['device_slot'] for f in record['files']) != expected['slots']):
            raise ValueError('owner receipt identity, checksum or slot set differs')
        files.extend(record['files'])
    if set(source_file_sha256) != {f.filename for f in inventory.files}:
        raise ValueError('source hashes must cover the complete inventory')
    for digest in source_file_sha256.values():
        retained._digest(digest, field='source SHA256')
    retained._digest(code_hash, field='code_hash', lengths=(40, 64))
    from glm_tpu.config.site import approved_source_uri
    if not approved_source_uri(source_uri):
        raise ValueError('source URI must use the approved bucket')
    manifest = dict(artifact_kind=retained.RUNTIME_ARTIFACT_KIND,
        code_hash=code_hash, files=sorted(files, key=lambda f:f['device_slot']),
        format_version=1, geometry=geometry.to_dict(), geometry_sha256=geometry.geometry_hash,
        mesh_hash=mesh_hash, packed_file_bytes=sum(p.file_bytes for p in plans),
        packed_payload_bytes=report.packed_bytes, placement_report=report.to_dict(),
        plan_id='WS32_2D', source=dict(
            files=[dict(f.to_dict(), sha256=source_file_sha256[f.filename]) for f in inventory.files],
            inventory_sha256=inventory.inventory_sha256, revision=inventory.source_revision,
            uri=source_uri.rstrip('/')),
        tensor_schema=[t.schema_dict() for t in plans[0].tensors])
    manifest['manifest_sha256'] = retained._mapping_hash(manifest, field='manifest_sha256')
    # Validate the complete original schema, owner geometry and tensor hash ledger.
    # No sparse placeholders or fabricated remote files are needed on rank zero.
    verify._verify_runtime_value(Path('.'), manifest, plans)
    return manifest
