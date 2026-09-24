"""Checkpoint and format identities (G4 CI tier; G5 site tier).

G4 needs no private assets. A synthetic GLM-5.3-shaped source inventory (HF names, dtypes and
shapes derived from the pinned geometry; placement depends only on names, shapes and slices)
reproduces the live placement report, and ``build_ws32_runtime_file_plans`` with the inventory
digest string pinned to the live value reproduces all 32 owner-file headers, whose SHA-256s are
compared with the values copied read-only from the live manifest into
``tests/golden/data/checkpoint_identity.json`` at S0. ``loader_record`` runs the real pack, verify
and load code on a tiny two-layer geometry on the 32 CPU devices of the child (kilobytes of
scratch data): every loaded array's positional leaf digest and canonical sharding spec, and the
refusals of tampered inputs.

G5 (``python -m tools.equivalence site-check``) runs read-only on rank 0 against the real assets
named by the site file (``SiteConfig.load()``: ``$GLM_TPU_SITE_CONFIG``, else
``$GLM_TPU_CONFIG_ROOT/site.toml``) and prints hashes and ``OK`` only. Its expectations that are
site specific (paths, private request digests, launcher constants) live outside Git in
``$GLM_TPU_CONFIG_ROOT/equivalence/site_baseline.json`` (default ``~/.config/glm-tpu``).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

from .common import canonical_json, digest_json, emit, sha256_hex, source_record

# Content hashes pinned in the repository at 181c013e (configs/glm53-site.json
# ``source_inventory_sha256``; the ``topology_args`` mesh pin of the model module). Frozen here so the
# CI tier survived the site values leaving Git (S1a: they are in the untracked site file).
INVENTORY_PIN = "813eb5e4d1cd96830f38458a7b36a0bb553169f9c5481451ef68b58559d6143a"
MESH_PIN = "de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88"
# The launcher's site values (coordinator address, TPU VM name, zone) are never spelled in Git (D20)
# and neither is anything derived from them alone (a digest of a low-entropy value can be
# recovered). At 181c013e ``launcher_site_literals`` located them structurally in the launcher's argv
# literals; from S1a they are site-file values, and ``launcher_constants`` rebuilds the same record
# from the site file, so G5 compares its digest with the one recorded at S0 (outside Git).
LOCK_SPLIT_LITERAL = "fcntl.LOCK_EX|(fcntl.LOCK_NB if index<2 else 0)"  # code, not a site value
# The worker module in its 181c013e role: S3 moved it (resident_protocol.WORKER_MODULE, proved by the
# G8 contract tests); the launcher record keeps the 181c013e spelling for the launcher's current
# worker module, so the G5 digest recorded at S0 (outside Git) still compares the site values.
WORKER_MODULE_181C013E = "scripts.release.ws32_optimized_worker"


def live_manifest_path() -> Path | None:
    """The live checkpoint manifest the site configuration names (G4 recording and G5 only):
    ``$GLM_EQUIVALENCE_LIVE_MANIFEST``, else ``<checkpoint.root>/manifest.json`` of the site file;
    None when no valid site file is available (CI)."""
    override = os.environ.get("GLM_EQUIVALENCE_LIVE_MANIFEST")
    if override:
        return Path(override)
    try:
        from glm_tpu.config.site import SiteConfig

        site = SiteConfig.load()
    except Exception:  # no site file on this host
        return None
    return site.checkpoint.root / "manifest.json"


def _site_pins(site: Any) -> dict[str, str]:
    """The site binding's content pins under their 181c013e names (configs/glm53-site.json keys)."""
    checkpoint = site.checkpoint
    return dict(source_inventory_sha256=checkpoint.source_inventory_sha256,
                checkpoint_manifest_sha256=checkpoint.manifest_sha256,
                checkpoint_success_sha256=checkpoint.success_sha256,
                source_complete_sha256=checkpoint.source_complete_sha256)


@contextmanager
def baseline_storage() -> Iterator[None]:
    """Install a synthetic site whose storage section admits the 181c013e model-source bucket:
    the checkpoint code checks every source URI against the current site (S1a), and G4's tiny packs
    keep the source URI they were recorded with (``site_fixture.baseline_source_uri``)."""
    from glm_tpu.config.site import set_current_site

    from .site_fixture import baseline_storage_site

    with tempfile.TemporaryDirectory(prefix="glm-equivalence-site-") as scratch:
        previous = set_current_site(baseline_storage_site(Path(scratch)))
        try:
            yield
        finally:
            set_current_site(previous)


def _fixture_source_uri() -> str:
    from .site_fixture import baseline_source_uri

    return baseline_source_uri().rsplit("/", 1)[0] + "/unit-fixture"


def inventory_pin() -> str:
    return INVENTORY_PIN


# ----------------------------------------------------------------------------- synthetic inventory
def production_geometry() -> Any:
    from glm_tpu.config import _s3_model as model

    return model.geometry()


def synthetic_tensors(geometry: Any) -> list[tuple[str, str, tuple[int, ...]]]:
    """Every base-model HF source tensor (MTP layer excluded) with dtype and shape."""
    g = geometry
    rows_block, cols_block = g.fp8_block_shape

    def fp8(name: str, shape: tuple[int, int]) -> list[tuple[str, str, tuple[int, ...]]]:
        scale = (-(-shape[0] // rows_block), -(-shape[1] // cols_block))
        return [(name + ".weight", "F8_E4M3", shape), (name + ".weight_scale_inv", "F32", scale)]

    hidden = g.hidden_size
    rows: list[tuple[str, str, tuple[int, ...]]] = [
        ("model.embed_tokens.weight", "BF16", (g.vocab_size, hidden)),
        ("lm_head.weight", "BF16", (g.vocab_size, hidden)),
        ("model.norm.weight", "BF16", (hidden,)),
    ]
    for layer in range(g.num_layers):
        p = f"model.layers.{layer}"
        a = p + ".self_attn"
        rows.append((p + ".input_layernorm.weight", "BF16", (hidden,)))
        rows.append((p + ".post_attention_layernorm.weight", "BF16", (hidden,)))
        rows += fp8(a + ".q_a_proj", (g.q_lora_rank, hidden))
        rows.append((a + ".q_a_layernorm.weight", "BF16", (g.q_lora_rank,)))
        rows += fp8(a + ".q_b_proj", (g.attention_heads * (g.qk_nope_head_dim + g.qk_rope_head_dim), g.q_lora_rank))
        rows += fp8(a + ".kv_a_proj_with_mqa", (g.kv_lora_rank + g.qk_rope_head_dim, hidden))
        rows.append((a + ".kv_a_layernorm.weight", "BF16", (g.kv_lora_rank,)))
        rows += fp8(a + ".kv_b_proj", (g.kv_heads * (g.qk_nope_head_dim + g.v_head_dim), g.kv_lora_rank))
        rows += fp8(a + ".o_proj", (hidden, g.attention_heads * g.v_head_dim))
        if g.indexer_types[layer] == "full":
            i = a + ".indexer"
            rows += fp8(i + ".wq_b", (g.dsa_indexer_heads * g.dsa_indexer_head_dim, g.q_lora_rank))
            rows += fp8(i + ".wk", (g.dsa_indexer_head_dim, hidden))
            rows.append((i + ".k_norm.weight", "BF16", (g.dsa_indexer_head_dim,)))
            rows.append((i + ".k_norm.bias", "BF16", (g.dsa_indexer_head_dim,)))
            rows.append((i + ".weights_proj.weight", "BF16", (g.dsa_indexer_heads, hidden)))
        if g.mlp_layer_types[layer] == "dense":
            for projection in ("gate_proj", "up_proj"):
                rows += fp8(f"{p}.mlp.{projection}", (g.dense_intermediate_size, hidden))
            rows += fp8(f"{p}.mlp.down_proj", (hidden, g.dense_intermediate_size))
        else:
            rows.append((p + ".mlp.gate.weight", "BF16", (g.num_routed_experts, hidden)))
            rows.append((p + ".mlp.gate.e_score_correction_bias", "F32", (g.num_routed_experts,)))
            for expert in range(g.num_routed_experts):
                for projection in ("gate_proj", "up_proj"):
                    rows += fp8(f"{p}.mlp.experts.{expert}.{projection}", (g.moe_intermediate_size, hidden))
                rows += fp8(f"{p}.mlp.experts.{expert}.down_proj", (hidden, g.moe_intermediate_size))
            for projection in ("gate_proj", "up_proj"):
                rows += fp8(f"{p}.mlp.shared_experts.{projection}", (g.moe_intermediate_size, hidden))
            rows += fp8(f"{p}.mlp.shared_experts.down_proj", (hidden, g.moe_intermediate_size))
    return rows


def synthetic_inventory(geometry: Any | None = None, *, pinned: str | None = None) -> Any:
    """A ``SourceInventory`` over one synthetic file; ``pinned`` overrides its digest string."""
    from glm_tpu.model_loader.source_inventory import SourceFile, SourceInventory, SourceTensor
    from glm_tpu.config import _s3_model as model

    geometry = geometry or production_geometry()
    width = {"F8_E4M3": 1, "F32": 4, "BF16": 2}
    offset = 0
    tensors = []
    for name, dtype, shape in synthetic_tensors(geometry):
        count = width[dtype]
        for dimension in shape:
            count *= dimension
        tensors.append(SourceTensor(name=name, filename="synthetic.safetensors", dtype=dtype, shape=shape,
                                    data_offset_start=offset, data_offset_end=offset + count))
        offset += count
    source = SourceFile(filename="synthetic.safetensors", file_bytes=offset + 8, header_bytes=8,
                        payload_bytes=offset, tensor_count=len(tensors), header_sha256="0" * 64)
    fields = dict(model_id=model.MODEL_ID, source_revision=model.REVISION,
                  index_filename="model.safetensors.index.json", index_sha256=model.INDEX_SHA,
                  config_filename="config.json", config_sha256=model.CONFIG_SHA,
                  declared_payload_bytes=offset, files=(source,), tensors=tuple(tensors))
    if pinned is None:
        return SourceInventory(**fields)

    class PinnedInventory(SourceInventory):  # test-only: the digest string of the real inventory
        @property
        def inventory_sha256(self) -> str:
            return pinned

    return PinnedInventory(**fields)


@lru_cache(maxsize=1)
def synthetic_file_plans() -> tuple[Any, Any]:
    """``(placement report, 32 file plans)`` for the pinned synthetic GLM-5.3 inventory."""
    from glm_tpu.model_loader.sharded_state.format import build_ws32_runtime_file_plans

    return build_ws32_runtime_file_plans(synthetic_inventory(pinned=inventory_pin()), production_geometry(),
                                         mesh_hash=MESH_PIN)


# ----------------------------------------------------------------------------- G4 record
def _synthetic_topology() -> Any:
    from glm_tpu.config.model import PhysicalDevice, PhysicalTopology

    devices = []
    for device_id in range(32):
        x, rest = divmod(device_id, 16)
        y, z = divmod(rest, 4)
        devices.append(PhysicalDevice(device_id=device_id, process_index=device_id // 4,
                                      local_device_id=device_id % 4, coordinates=(x, y, z), core_on_chip=0,
                                      platform="tpu", device_kind="TPU v4"))
    return PhysicalTopology(slice_name="example-v4-64", topology_shape=(2, 4, 4), devices=tuple(devices))


def _tiny_pack() -> dict[str, Any]:
    """The single-root packer on a one-tensor source (as the checkpoint unit tests do)."""
    from dataclasses import replace
    from hashlib import sha256

    import torch
    from safetensors.torch import save_file

    from glm_tpu.model_loader.sharded_state.format import Ws32RuntimePackConfig, pack_ws32_runtime_checkpoint
    from glm_tpu.model_loader.source_inventory import read_source_inventory

    from .fixture import config_json
    from glm_tpu.config.model import ModelGeometry

    geometry = replace(
        ModelGeometry.from_hf_config(config_json()), num_layers=1, first_dense_layers=1, hidden_size=8,
        dense_intermediate_size=16, num_routed_experts=8, routed_top_k=2, moe_intermediate_size=4, dsa_top_k=8,
        dsa_indexer_heads=8, dsa_indexer_head_dim=2, index_share_group_size=1, attention_heads=8, kv_heads=8,
        kv_lora_rank=4, q_lora_rank=8, qk_nope_head_dim=2, qk_rope_head_dim=2, v_head_dim=2,
        max_position_embeddings=64, vocab_size=16, fp8_block_shape=(2, 2), mlp_layer_types=("dense",),
        indexer_types=("full",))
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-pack-") as scratch:
        root = Path(scratch)
        source = root / "source"
        source.mkdir()
        embedding = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8).to(torch.bfloat16)
        save_file({"model.embed_tokens.weight": embedding}, source / "model.safetensors")
        (source / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": embedding.numel() * embedding.element_size()},
             "weight_map": {"model.embed_tokens.weight": "model.safetensors"}}))
        inventory = read_source_inventory(source, model_id=geometry.model_id, source_revision="unit-fixture",
                                          config_filename=None)
        output = root / "packed"
        # The packer only admits source URIs under the site's approved prefix; derive it from the
        # model source pinned at 181c013e instead of repeating the bucket name here.
        source_uri = _fixture_source_uri()
        config = Ws32RuntimePackConfig(source_root=source, source_uri=source_uri, output_dir=output,
                                       code_hash="a" * 40, mesh_hash="b" * 64)
        manifest = pack_ws32_runtime_checkpoint(config, inventory, geometry)
        files = {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())
                 if path.is_file()}
    return dict(inventory_sha256=inventory.inventory_sha256, manifest_sha256=manifest["manifest_sha256"],
                files_digest=digest_json(files), file_count=len(files),
                file_names_digest=digest_json(sorted(files)))


def _loader_geometry() -> Any:
    """Two layers with every destination family of the production name tree: FP8 projections
    (U8 bits + F32 scale_inv) and BF16 norms, a full indexer (layer 0) and a shared one (layer 1),
    a dense MLP (layer 0) and a routed + shared MoE with its router (layer 1)."""
    from dataclasses import replace

    from glm_tpu.config.model import ModelGeometry

    from .fixture import config_json

    return replace(
        ModelGeometry.from_hf_config(config_json()), num_layers=2, first_dense_layers=1, hidden_size=8,
        dense_intermediate_size=16, num_routed_experts=8, routed_top_k=2, moe_intermediate_size=4, dsa_top_k=8,
        dsa_indexer_heads=8, dsa_indexer_head_dim=2, index_share_group_size=2, attention_heads=8, kv_heads=8,
        kv_lora_rank=4, q_lora_rank=8, qk_nope_head_dim=2, qk_rope_head_dim=2, v_head_dim=2,
        max_position_embeddings=64, vocab_size=16, fp8_block_shape=(2, 2), mlp_layer_types=("dense", "sparse"),
        indexer_types=("full", "shared"))


def _write_loader_source(source: Path, geometry: Any) -> None:
    """Deterministic source safetensors for every base tensor of ``geometry``: FP8 bits drawn from
    every finite E4M3 pattern of both signs (0x7F and 0xFF excluded), positive F32 scales, BF16
    normals."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    rng = np.random.default_rng(5310)
    tensors = {}
    for name, dtype, shape in synthetic_tensors(geometry):
        if dtype == "F8_E4M3":
            bits = rng.integers(0, 0x7F, size=shape, dtype=np.uint8) | (rng.integers(0, 2, size=shape,
                                                                                    dtype=np.uint8) << 7)
            tensors[name] = torch.from_numpy(bits).view(torch.float8_e4m3fn)
        elif dtype == "F32":
            tensors[name] = torch.from_numpy(rng.uniform(0.5, 1.5, size=shape).astype(np.float32))
        else:
            tensors[name] = torch.from_numpy(rng.normal(0.0, 0.1, size=shape).astype(np.float32)).to(torch.bfloat16)
    save_file(tensors, source / "model.safetensors")
    total = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    (source / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": {name: "model.safetensors" for name in tensors}}))


def _seal(root: Path, manifest: dict[str, Any], topology_hash: str) -> str:
    """The SUCCESS seal the pack workflow publishes (every field ``verify`` checks)."""
    from hashlib import sha256

    source = manifest["source"]
    value: dict[str, Any] = {
        "artifact_kind": "greenfield_ws32_runtime_checkpoint_success", "code_hash": manifest["code_hash"],
        "file_count": 32, "format_version": manifest["format_version"],
        "manifest_file_sha256": sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "manifest_sha256": manifest["manifest_sha256"], "mesh_hash": manifest["mesh_hash"],
        "packed_payload_bytes": manifest["packed_payload_bytes"], "performance_claim": False,
        "post_census_sha256": "d" * 64, "remote_preflight_sha256": "e" * 64, "remote_terminal_sha256": "f" * 64,
        "source_file_count": len(source["files"]), "source_inventory_sha256": source["inventory_sha256"],
        "tag": "greenfield_ws32_runtime_pack_" + "20000101T" + "0" * 15 + "Z",
        "topology_hash": topology_hash, "tpu_initialized": False}
    value["success_sha256"] = sha256(canonical_json(value).encode()).hexdigest()
    (root / "SUCCESS").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value["success_sha256"]


def loader_record() -> dict[str, Any]:
    """The real checkpoint path on 32 CPU devices: ``pack_ws32_runtime_checkpoint`` packs a tiny
    two-layer geometry (every dtype and partition-spec family), the seal is written, the real
    ``verify_ws32_runtime_checkpoint`` admits it (full layout, and the per-host local-slot layout
    ``_load`` uses: ``verify_file_hashes=True``, the four owned slots, ``local_slot_layout=True``),
    and the real ``load_ws32_runtime_checkpoint`` places every tensor on the mesh built from the
    synthetic topology. Recorded: the pack digests, positional leaf digests and the canonical
    ``sharding.spec`` of every loaded array, the local device slots, and the refusals of tampered
    inputs (one payload byte flipped: full and local verification, and the loader itself after a
    verification; a foreign slot in the local layout; wrong manifest and topology pins)."""
    import shutil

    import jax
    import numpy as np
    from jax.sharding import Mesh

    from glm_tpu.model_loader.sharded_state.format import (
        Ws32RuntimePackConfig,
        load_ws32_runtime_checkpoint,
        pack_ws32_runtime_checkpoint,
        verify_ws32_runtime_checkpoint,
    )
    from glm_tpu.model_loader.source_inventory import read_source_inventory
    from glm_tpu.distributed.mesh import build_ws32_physical_mesh

    from .common import leaf_digest
    from .normalize import canonical_spec

    geometry = _loader_geometry()
    topology = _synthetic_topology()
    physical = build_ws32_physical_mesh(topology)
    by_id = {int(device.id): device for device in jax.devices()}
    mesh = Mesh(np.asarray([[by_id[i] for i in row] for row in physical.device_ids], dtype=object),
                ("expert", "feature"))
    owned = [slot for slot, device in enumerate(physical.flattened_device_ids)
             if device in {d.device_id for d in topology.devices if d.process_index == 0}]
    out: dict[str, Any] = dict(geometry_sha256=geometry.geometry_hash, owned_slots=owned)
    with tempfile.TemporaryDirectory(prefix="glm-equivalence-loader-") as scratch:
        base = Path(scratch)

        def attempt(call: Any) -> str:
            try:
                call()
            except Exception as exc:  # the expected refusal
                return f"{type(exc).__name__}: {exc}".replace(scratch, "<tmp>")
            return "accepted"

        source, root, local = base / "source", base / "packed", base / "local"
        source.mkdir()
        _write_loader_source(source, geometry)
        inventory = read_source_inventory(source, model_id=geometry.model_id, source_revision="unit-fixture",
                                          config_filename=None)
        config = Ws32RuntimePackConfig(source_root=source, source_uri=_fixture_source_uri(), output_dir=root,
                                       code_hash="a" * 40,
                                       mesh_hash=physical.mesh_hash)
        manifest = pack_ws32_runtime_checkpoint(config, inventory, geometry)
        success = _seal(root, manifest, topology.topology_hash)
        pins = dict(expected_manifest_sha256=manifest["manifest_sha256"], expected_success_sha256=success,
                    expected_mesh_hash=physical.mesh_hash, expected_topology_hash=topology.topology_hash,
                    inventory=inventory, geometry=geometry)
        verified = verify_ws32_runtime_checkpoint(root, verify_file_hashes=True, **pins)
        local.mkdir()
        for name in ("manifest.json", "SUCCESS", *(f"device_slot_{slot:02d}.safetensors" for slot in owned)):
            shutil.copy(root / name, local / name)
        local_kwargs = dict(verify_file_hashes=True, verify_file_hash_slots=tuple(owned), local_slot_layout=True)
        local_verified = verify_ws32_runtime_checkpoint(local, **local_kwargs, **pins)
        loaded = load_ws32_runtime_checkpoint(verified, mesh=mesh, physical_mesh=physical)
        tensors = verified.plans[0].tensors
        rows = []
        for plan in tensors:
            array = loaded.arrays[plan.name]
            rows.append([plan.name, str(array.dtype), str(canonical_spec(array.sharding.spec)),
                         leaf_digest(jax.device_get(array))])
        out.update(
            inventory_sha256=inventory.inventory_sha256, manifest_sha256=manifest["manifest_sha256"],
            success_sha256=success, files_digest=digest_json([r["sha256"] for r in manifest["files"]]),
            verified=dict(plans=len(verified.plans), slots=sorted(verified.records_by_slot),
                          local_plans=len(local_verified.plans)),
            loaded=dict(count=len(rows), names_equal_plan=sorted(loaded.arrays) == sorted(r[0] for r in rows),
                        digest=sha256_hex("\n".join(r[3] for r in rows)),
                        arrays=[[name, dtype, spec, digest[:16]] for name, dtype, spec, digest in rows]),
            local_device_slots=[[r["device_id"], r["device_slot"]] for r in loaded.local_device_slots],
            local_file_digest=digest_json([r["file_sha256"] for r in loaded.local_device_slots]))
        del loaded
        refusals: dict[str, str] = {}
        refusals["wrong_manifest_pin"] = attempt(lambda: verify_ws32_runtime_checkpoint(
            root, verify_file_hashes=True, **dict(pins, expected_manifest_sha256="0" * 64)))
        refusals["wrong_topology_pin"] = attempt(lambda: verify_ws32_runtime_checkpoint(
            root, verify_file_hashes=True, **dict(pins, expected_topology_hash="0" * 64)))
        foreign = next(slot for slot in range(32) if slot not in owned)
        shutil.copy(root / f"device_slot_{foreign:02d}.safetensors", local / f"device_slot_{foreign:02d}.safetensors")
        refusals["local_foreign_slot"] = attempt(lambda: verify_ws32_runtime_checkpoint(local, **local_kwargs,
                                                                                        **pins))
        (local / f"device_slot_{foreign:02d}.safetensors").unlink()
        for label, directory, slot in (("payload_byte_flipped", root, owned[1]),
                                       ("local_payload_byte_flipped", local, owned[1])):
            path = directory / f"device_slot_{slot:02d}.safetensors"
            with path.open("r+b") as stream:
                stream.seek(-1, 2)
                last = stream.read(1)[0]
                stream.seek(-1, 2)
                stream.write(bytes([last ^ 0x01]))
            kwargs = local_kwargs if directory == local else dict(verify_file_hashes=True)
            refusals[label] = attempt(lambda d=directory, k=kwargs: verify_ws32_runtime_checkpoint(d, **k, **pins))
        refusals["loader_after_payload_flip"] = attempt(lambda: load_ws32_runtime_checkpoint(
            verified, mesh=mesh, physical_mesh=physical))
        out["refusals"] = refusals
    return out


def ci_record() -> dict[str, Any]:
    """Every G4 identity, computed from code (no private assets)."""
    from glm_tpu.engine import _s3_user_request as legacy
    from glm_tpu.model_loader.sharded_state import format as ckpt
    from glm_tpu.models.glm_moe_dsa._s3_ws32_decoder import Ws32DecoderConfig, ws32_decoder_weight_names
    from glm_tpu.distributed.mesh import build_ws32_physical_mesh

    from .fixture import name_spec_pairs

    geometry = production_geometry()
    config = Ws32DecoderConfig(geometry, 8192, host_main_rope_table=True)
    import jax

    names = list(jax.tree.leaves(ws32_decoder_weight_names(config)))
    pairs = name_spec_pairs(config)
    report, plans = synthetic_file_plans()
    topology = _synthetic_topology()
    physical = build_ws32_physical_mesh(topology)
    fixed_mapping = {"b": [1, 2, {"c": None}], "a": "value", "unicode": "café 中文"}
    return dict(
        geometry=dict(sha256=geometry.geometry_hash, canonical_digest=sha256_hex(canonical_json(geometry.to_dict())),
                      model_id=geometry.model_id),
        tensor_names=dict(count=len(names), tree_order_digest=digest_json(names),
                          sorted_digest=digest_json(sorted(names))),
        partition_specs=dict(count=len(pairs), digest=digest_json([[n, list(s)] for n, s in pairs])),
        placement=dict(sha256=report.placement_sha256, report_digest=digest_json(report.to_dict()),
                       placement_count=report.placement_count, destination_tensors=report.destination_tensor_count,
                       source_tensors=report.source_tensor_count),
        headers=[sha256_hex(plan.header) for plan in plans],
        header_bytes=sorted({len(plan.header) for plan in plans}),
        tensor_schema_digest=digest_json([t.schema_dict() for t in plans[0].tensors]),
        tensors_per_slot=len(plans[0].tensors),
        key_sets=dict(manifest=sorted(ckpt._MANIFEST_KEYS), file=sorted(ckpt._FILE_RECORD_KEYS),
                      tensor_schema=sorted(ckpt._TENSOR_SCHEMA_KEYS), success=sorted(ckpt._SUCCESS_KEYS)),
        mapping_hash=ckpt._mapping_hash(fixed_mapping, field="fixture"),
        canonical_contracts=dict(hash=sha256_hex(ckpt._canonical_json(fixed_mapping)),
                                 wire=sha256_hex(legacy.canonical(fixed_mapping))),
        success_tag=ckpt._SUCCESS_TAG.pattern,
        artifact_kinds=[ckpt.WS32_RUNTIME_ARTIFACT_KIND, ckpt.WS32_RUNTIME_SLOT_RECORD_KIND,
                        ckpt._SUCCESS_ARTIFACT_KIND],
        plan_id=ckpt.WS32_RUNTIME_PLAN_ID, format_version=ckpt.WS32_RUNTIME_FORMAT_VERSION,
        synthetic_topology=dict(topology_sha256=topology.topology_hash, mesh_sha256=physical.mesh_hash,
                                device_order_digest=digest_json(list(physical.flattened_device_ids))),
        **_packed(),
    )


def _packed() -> dict[str, Any]:
    with baseline_storage():
        return dict(tiny_pack=_tiny_pack(), loader=loader_record())


def live_manifest_values(path: Path) -> dict[str, Any]:
    """Content hashes copied read-only from the live manifest (S0 recording only)."""
    manifest = json.loads(path.read_text())
    return dict(
        headers=[record["header_sha256"] for record in sorted(manifest["files"], key=lambda r: r["device_slot"])],
        geometry_sha256=manifest["geometry_sha256"],
        placement_sha256=manifest["placement_report"]["placement_sha256"],
        placement_report_digest=digest_json(manifest["placement_report"]),
        tensor_schema_digest=digest_json(manifest["tensor_schema"]),
        mesh_hash=manifest["mesh_hash"],
        inventory_sha256=manifest["source"]["inventory_sha256"],
        manifest_sha256=manifest["manifest_sha256"],
    )


def compare_to_live(record: dict[str, Any], live: dict[str, Any]) -> dict[str, bool]:
    return dict(
        headers=record["headers"] == live["headers"],
        geometry=record["geometry"]["sha256"] == live["geometry_sha256"],
        placement=record["placement"]["sha256"] == live["placement_sha256"],
        placement_report=record["placement"]["report_digest"] == live["placement_report_digest"],
        tensor_schema=record["tensor_schema_digest"] == live["tensor_schema_digest"],
        mesh=live["mesh_hash"] == MESH_PIN,
        inventory=inventory_pin() == live["inventory_sha256"],
    )


# ----------------------------------------------------------------------------- G5 site tier
def config_root() -> Path:
    """``$GLM_TPU_CONFIG_ROOT`` exactly as the site loader resolves it (absolute; XDG default)."""
    from glm_tpu import envs

    return envs.GLM_TPU_CONFIG_ROOT


def site_baseline_path() -> Path:
    return config_root() / "equivalence" / "site_baseline.json"


def launcher_site_literals(source: str) -> dict[str, str | None]:
    """The site values the 181c013e launcher spells in its argv literals, found by position
    (never by value): the constant after ``'--coordinator-address'``, the value of ``'--zone=...'``
    and the TPU VM name after ``'tpu-vm', 'ssh'``. None when absent."""
    import ast

    found: dict[str, str | None] = dict(coordinator=None, zone=None, tpu_name=None)
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.List, ast.Tuple)):
            continue
        items = [e.value if isinstance(e, ast.Constant) and isinstance(e.value, str) else None for e in node.elts]
        for index, item in enumerate(items):
            following = items[index + 1] if index + 1 < len(items) else None
            if item == "--coordinator-address" and following:
                found["coordinator"] = following
            elif item and item.startswith("--zone="):
                found["zone"] = item[len("--zone="):]
            elif item == "ssh" and index and items[index - 1] == "tpu-vm" and following:
                found["tpu_name"] = following
    return found


def launcher_constants(site: Any) -> dict[str, Any]:
    """The site values the 181c013e launcher, worker and model hard-coded (M4), rebuilt from the
    site file under their 181c013e roles: interpreter, site-packages, binding directory and pin, the
    four locks in acquisition order (the first two are the ``LOCK_NB`` workload leases, the last two
    the blocking sync locks: ``LOCK_SPLIT_LITERAL`` stands for that split), worker module, run root,
    model path, source URI, and the coordinator, zone and TPU name. Equal values give the digest
    recorded at S0. Only the digest of this record is ever printed or stored (outside Git)."""
    from glm_tpu.engine import resident_protocol as protocol
    from glm_tpu.executor import multihost_executor as launch

    fleet = site.fleet
    module = WORKER_MODULE_181C013E if launch.MODULE == protocol.WORKER_MODULE else launch.MODULE
    literal: dict[str, bool] = {value: True for value in (fleet.coordinator_address, fleet.zone, fleet.tpu_name)}
    literal[LOCK_SPLIT_LITERAL] = (len(site.locks.workload), len(site.locks.sync)) == (2, 2)
    return dict(python=fleet.worker_python, site=":".join(fleet.worker_pythonpath),
                binding=str(site.topology.binding_dir), binding_sha=site.topology.binding_sha256,
                locks=[str(p) for p in (*site.locks.workload, *site.locks.sync)], module=module,
                run_root=str(site.paths.run_root), tokenizer_root=str(site.paths.model_path),
                source_uri=site.storage.source_uri, literals=literal)


def site_record(requests_dir: Path | None) -> dict[str, Any]:
    """G5 facts from the real assets (read-only). Values are hashes, counts and booleans."""
    from types import SimpleNamespace

    from glm_tpu.model_loader.sharded_state.format import _read_ws32_runtime_metadata
    from glm_tpu.model_loader.source_inventory import inspect_source_inventory
    from glm_tpu.distributed.topology import validate_ws32_topology_fleet
    from glm_tpu.distributed.mesh import build_ws32_physical_mesh
    from glm_tpu.config.site import SiteConfig, set_current_site
    from glm_tpu.config import _s3_model as model
    from glm_tpu.engine import request

    facts: dict[str, Any] = {}
    site = SiteConfig.load()
    set_current_site(site)  # the checkpoint metadata check reads the approved source buckets
    args = model.site_args(SimpleNamespace(), site)
    facts["site_binding"] = dict(ok=True, pins=digest_json(_site_pins(site)))
    inventory = inspect_source_inventory(args.source_inventory)
    model.require_inventory(inventory)
    facts["inventory"] = dict(sha256=inventory.inventory_sha256, pinned=inventory.inventory_sha256 == inventory_pin())
    metadata = _read_ws32_runtime_metadata(
        args.checkpoint_root, expected_manifest_sha256=args.checkpoint_manifest_sha256,
        expected_success_sha256=args.checkpoint_success_sha256, expected_mesh_hash=args.mesh_sha256,
        expected_topology_hash=args.topology_sha256, inventory=inventory, geometry=production_geometry())
    facts["checkpoint_metadata"] = dict(ok=True, manifest_sha256=metadata.manifest["manifest_sha256"],
                                        success_sha256=metadata.success["success_sha256"])
    local = {}
    for plan in metadata.plans:
        path = Path(args.checkpoint_root) / plan.filename
        if path.exists():
            with path.open("rb") as stream:
                local[plan.device_slot] = stream.read(len(plan.header)) == plan.header
    facts["local_headers"] = dict(slots=sorted(local), equal=all(local.values()) and bool(local))
    binding = site.topology.binding_dir
    raw = (binding / "topology_rebinding.json").read_bytes()
    value = json.loads(raw)
    captures = []
    for index in range(8):
        payload = (binding / "captures" / f"topology.rank{index}.json").read_bytes()
        captures.append((sha256_hex(payload) == value["capture_sha256"][f"topology.rank{index}.json"],
                         json.loads(payload)))
    topology, _, fleet = validate_ws32_topology_fleet(
        tuple(c for _, c in captures), expected_topology_sha256=args.topology_sha256,
        expected_fleet_sha256=value["fleet_sha256"], slice_name=args.slice_name)
    physical = build_ws32_physical_mesh(topology)
    facts["topology"] = dict(binding_sha256=sha256_hex(raw) == site.topology.binding_sha256,
                             captures=all(ok for ok, _ in captures), topology_sha256=topology.topology_hash,
                             mesh_sha256=physical.mesh_hash, fleet_sha256=fleet,
                             device_order_digest=digest_json(list(physical.flattened_device_ids)))
    facts["launcher_constants_digest"] = digest_json(launcher_constants(site))
    if requests_dir is not None:
        rows = {}
        for path in sorted(Path(requests_dir).glob("*.json")):
            try:
                rows[path.name] = request.read(path)["request_sha256"]
            except (ValueError, KeyError, TypeError, OSError):
                continue  # non-request files (e.g. the stop marker)
        facts["requests"] = rows
    return facts


def site_check(*, record: bool, requests_dir: Path | None) -> dict[str, Any]:
    from .budget import REFUSAL, live_tpu_run, refusal_reason

    if live_tpu_run():
        reason = refusal_reason()
        raise SystemExit("a TPU run is live on this host; run site-check later (fleet idle)" if reason == REFUSAL
                         else reason)
    facts = site_record(requests_dir)
    path = site_baseline_path()
    if record:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(dict(facts=facts, source=source_record()), stream, sort_keys=True, indent=1)
        return dict(gate="G5", status="recorded", baseline=str(path), digest=digest_json(facts))
    baseline = json.loads(path.read_text())["facts"]
    diffs = sorted(k for k in set(baseline) | set(facts) if baseline.get(k) != facts.get(k))
    return dict(gate="G5", status="pass" if not diffs else "fail", differing=diffs, digest=digest_json(facts))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="checkpoint and format identities")
    parser.add_argument("--live", action="store_true", help="also copy the live manifest values (S0 record)")
    args = parser.parse_args(argv)
    record = ci_record()
    result: dict[str, Any] = dict(record=record)
    if args.live:
        path = live_manifest_path()
        if path is None:
            raise SystemExit("--live needs the site binding (or GLM_EQUIVALENCE_LIVE_MANIFEST)")
        live = live_manifest_values(path)
        result["live"] = live
        result["live_equal"] = compare_to_live(record, live)
    emit(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
