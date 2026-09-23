"""Checkpoint and format identities (G4 CI tier; G5 site tier).

G4 needs no private assets. A synthetic GLM-5.3-shaped source inventory (HF names, dtypes and
shapes derived from the pinned geometry; placement depends only on names, shapes and slices)
reproduces the live placement report, and ``build_ws32_runtime_file_plans`` with the inventory
digest string pinned to the live value reproduces all 32 owner-file headers, whose SHA-256s are
compared with the values copied read-only from the live manifest into
``tests/golden/data/checkpoint_identity.json`` at S0.

G5 (``python -m tools.equivalence site-check``) runs read-only on rank 0 against the real assets
and prints hashes and ``OK`` only. Its expectations that are site specific (paths, private
request digests, launcher constants) live outside Git in
``$GLM_TPU_CONFIG_ROOT/equivalence/site_baseline.json`` (default ``~/.config/glm-tpu``).
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .common import canonical_json, digest_json, emit, sha256_hex, source_record

# Content hashes already pinned in the repository at 181c013e (configs/glm53-site.json
# ``source_inventory_sha256``; glm_tpu/optimized/model.py ``topology_args`` mesh). Frozen here so the
# CI tier survives the site file leaving Git.
INVENTORY_PIN = "813eb5e4d1cd96830f38458a7b36a0bb553169f9c5481451ef68b58559d6143a"
MESH_PIN = "de5f59cbadf2116745ee1dde921656424c9555c3ddc584dcdd66cb7845050a88"
# Site values the 181c013e launcher hard-codes (coordinator address, TPU VM name, zone) are never
# spelled in Git (D20): only their salted SHA-256 is. ``launcher_constants`` finds them in the
# launcher source by digest, so its record -- and the G5 digest -- is unchanged.
LITERAL_SALT = "glm-tpu-equivalence/launcher-literals/v1"
LAUNCHER_LITERAL_DIGESTS = (
    "d90489343064832d482a2a111470281a108234cc4442b634371366a21c5827f7",
    "2a9738f6d3d8da01b2564d54fa1f39ce9a3bbfca4c1c6a515f3b9f81a21819da",
    "3131698558339e89be48eacf21259046fbe525a520f9e9518021386ebdef75c4",
)
LOCK_SPLIT_LITERAL = "fcntl.LOCK_EX|(fcntl.LOCK_NB if index<2 else 0)"  # code, not a site value
_LITERAL_CANDIDATE = re.compile(r"[0-9]{1,3}(?:\.[0-9]{1,3}){3}:[0-9]{1,5}|[a-z][a-z0-9]*(?:-[a-z0-9]+)+")


def live_manifest_path() -> Path | None:
    """The live checkpoint manifest the site configuration names (G4 recording and G5 only):
    ````, else ``<checkpoint_root>/manifest.json`` of the site binding
    ``model.site_args`` resolves; None when no site binding is available (CI)."""
    override = os.environ.get("GLM_EQUIVALENCE_LIVE_MANIFEST")
    if override:
        return Path(override)
    from types import SimpleNamespace

    from .common import REPO

    try:
        from glm_tpu.optimized import model

        args = model.site_args(SimpleNamespace(), repo=REPO)
    except Exception:  # no site binding, assets or topology captures on this host
        return None
    return Path(args.checkpoint_root) / "manifest.json"


def _site_pins() -> dict[str, str]:
    """The site binding's content pins (configs/glm53-site.json at 181c013e; G5 only)."""
    from .common import REPO

    value = json.loads((REPO / "configs" / "glm53-site.json").read_text())
    return {k: value[k] for k in ("source_inventory_sha256", "checkpoint_manifest_sha256",
                                  "checkpoint_success_sha256", "source_complete_sha256")}


def inventory_pin() -> str:
    return INVENTORY_PIN


# ----------------------------------------------------------------------------- synthetic inventory
def production_geometry() -> Any:
    from glm_tpu.optimized import model

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
    from glm_tpu.greenfield.partitioning.source_inventory import SourceFile, SourceInventory, SourceTensor
    from glm_tpu.optimized import model

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
    from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import build_ws32_runtime_file_plans

    return build_ws32_runtime_file_plans(synthetic_inventory(pinned=inventory_pin()), production_geometry(),
                                         mesh_hash=MESH_PIN)


# ----------------------------------------------------------------------------- G4 record
def _synthetic_topology() -> Any:
    from glm_tpu.greenfield.types import PhysicalDevice, PhysicalTopology

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

    from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import (
        Ws32RuntimePackConfig,
        pack_ws32_runtime_checkpoint,
    )
    from glm_tpu.greenfield.partitioning.source_inventory import read_source_inventory

    from .fixture import config_json
    from glm_tpu.greenfield.types import ModelGeometry

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
        # pinned model source instead of repeating the bucket name here.
        from glm_tpu.optimized import model

        source_uri = model.SOURCE_URI.rsplit("/", 1)[0] + "/unit-fixture"
        config = Ws32RuntimePackConfig(source_root=source, source_uri=source_uri, output_dir=output,
                                       code_hash="a" * 40, mesh_hash="b" * 64)
        manifest = pack_ws32_runtime_checkpoint(config, inventory, geometry)
        files = {path.name: sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())
                 if path.is_file()}
    return dict(inventory_sha256=inventory.inventory_sha256, manifest_sha256=manifest["manifest_sha256"],
                files_digest=digest_json(files), file_count=len(files),
                file_names_digest=digest_json(sorted(files)))


def ci_record() -> dict[str, Any]:
    """Every G4 identity, computed from code (no private assets)."""
    from glm_tpu import user_request as legacy
    from glm_tpu.greenfield.checkpoint import ws32_runtime_checkpoint as ckpt
    from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig, ws32_decoder_weight_names
    from glm_tpu.greenfield.sharding.ws32 import build_ws32_physical_mesh

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
        tiny_pack=_tiny_pack(),
    )


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
    return Path(os.environ.get("GLM_TPU_CONFIG_ROOT") or Path.home() / ".config" / "glm-tpu")


def site_baseline_path() -> Path:
    return config_root() / "equivalence" / "site_baseline.json"


def launcher_constants() -> dict[str, Any]:
    """Site values the 181c013e launcher hard-codes (M4). Only their digest is ever printed."""
    import inspect

    from scripts.release import launch_ws32_optimized_request as launch
    from scripts.release import ws32_optimized_worker as worker
    from glm_tpu.optimized import model

    text = inspect.getsource(launch)
    found = {}
    for candidate in set(_LITERAL_CANDIDATE.findall(text)):
        digest = sha256((LITERAL_SALT + "\0" + candidate).encode()).hexdigest()
        if digest in LAUNCHER_LITERAL_DIGESTS:
            found[digest] = candidate
    literal = {found.get(digest, "<absent:" + digest[:16] + ">"): digest in found
               for digest in LAUNCHER_LITERAL_DIGESTS}
    literal[LOCK_SPLIT_LITERAL] = LOCK_SPLIT_LITERAL in text
    return dict(python=launch.PYTHON, site=launch.SITE, binding=str(launch.BINDING), binding_sha=launch.BINDING_SHA,
                locks=list(launch.LOCKS), module=launch.MODULE, run_root=str(worker.RUN_ROOT),
                tokenizer_root=str(model.TOKENIZER_ROOT), source_uri=model.SOURCE_URI, literals=literal)


def site_record(requests_dir: Path | None) -> dict[str, Any]:
    """G5 facts from the real assets (read-only). Values are hashes, counts and booleans."""
    from types import SimpleNamespace

    from glm_tpu.greenfield.checkpoint.ws32_runtime_checkpoint import _read_ws32_runtime_metadata
    from glm_tpu.greenfield.partitioning.source_inventory import inspect_source_inventory
    from glm_tpu.greenfield.benchmarking.ws32_one_layer import validate_ws32_topology_fleet
    from glm_tpu.greenfield.sharding.ws32 import build_ws32_physical_mesh
    from glm_tpu.optimized import model, request
    from scripts.release import launch_ws32_optimized_request as launch

    from .common import REPO

    facts: dict[str, Any] = {}
    args = model.site_args(SimpleNamespace(), repo=REPO)
    facts["site_binding"] = dict(ok=True, pins=digest_json(_site_pins()))
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
    binding = Path(launch.BINDING)
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
    facts["topology"] = dict(binding_sha256=sha256_hex(raw) == launch.BINDING_SHA,
                             captures=all(ok for ok, _ in captures), topology_sha256=topology.topology_hash,
                             mesh_sha256=physical.mesh_hash, fleet_sha256=fleet,
                             device_order_digest=digest_json(list(physical.flattened_device_ids)))
    facts["launcher_constants_digest"] = digest_json(launcher_constants())
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
    from .budget import live_tpu_run

    if live_tpu_run():
        raise SystemExit("a TPU run is live on this host; run site-check later (fleet idle)")
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
