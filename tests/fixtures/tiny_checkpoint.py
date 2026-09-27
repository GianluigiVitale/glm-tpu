"""The tiny checkpoint of the checkpoint-format and checkpoint-command tests.

``fixture`` writes a one-tensor safetensors source (a 16 x 8 BF16 embedding, its index, no ``config.json``) under
``tmp_path`` and returns the tensor, the source inventory and the pack configuration; the current site must admit
its example source URI (``tests.fixtures.site.installed_site``). ``geometry`` is the one-layer geometry that packs
it into 32 owner files, and ``seal`` writes a synthetic ``SUCCESS`` for a packed manifest. The G4 gate packs the same
source and geometry (``tools/equivalence/identities.py``).
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from glm_tpu.config.model import ModelGeometry
from glm_tpu.model_loader.sharded_state.format import RuntimePackConfig
from glm_tpu.model_loader.source_inventory import read_source_inventory
from tests.fixtures.site import EXAMPLE_BUCKET

# The pinned GLM-5.3 config; its geometry equals the archived GLM-5.2 file's (tests/config/test_model.py).
from tools.equivalence.fixture import config_json


def geometry() -> ModelGeometry:
    full = ModelGeometry.from_hf_config(config_json())
    return replace(
        full,
        num_layers=1,
        first_dense_layers=1,
        hidden_size=8,
        dense_intermediate_size=16,
        num_routed_experts=8,
        routed_top_k=2,
        moe_intermediate_size=4,
        dsa_top_k=8,
        dsa_indexer_heads=8,
        dsa_indexer_head_dim=2,
        index_share_group_size=1,
        attention_heads=8,
        kv_heads=8,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
        max_position_embeddings=64,
        vocab_size=16,
        fp8_block_shape=(2, 2),
        mlp_layer_types=("dense",),
        indexer_types=("full",),
    )


def fixture(tmp_path: Path):
    import torch
    from safetensors.torch import save_file

    source = tmp_path / "source"
    source.mkdir()
    filename = "model.safetensors"
    embedding = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8).to(torch.bfloat16)
    save_file({"model.embed_tokens.weight": embedding}, source / filename)
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": embedding.numel() * embedding.element_size()},
                "weight_map": {"model.embed_tokens.weight": filename},
            }
        )
    )
    inventory = read_source_inventory(
        source,
        model_id="zai-org/GLM-5.2-FP8",
        source_revision="unit-fixture",
        config_filename=None,
    )
    output = tmp_path / "packed"
    config = RuntimePackConfig(
        source_root=source,
        source_uri=EXAMPLE_BUCKET + "models/unit-fixture",
        output_dir=output,
        code_hash="a" * 40,
        mesh_hash="b" * 64,
    )
    return embedding, inventory, config


def seal(
    root: Path,
    manifest: dict[str, object],
    *,
    topology_hash: str = "c" * 64,
) -> dict[str, object]:
    source = manifest["source"]
    assert isinstance(source, dict)
    source_files = source["files"]
    assert isinstance(source_files, list)
    value: dict[str, object] = {
        "artifact_kind": "greenfield_ws32_runtime_checkpoint_success",
        "code_hash": manifest["code_hash"],
        "file_count": 32,
        "format_version": 1,
        "manifest_file_sha256": sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "manifest_sha256": manifest["manifest_sha256"],
        "mesh_hash": manifest["mesh_hash"],
        "packed_payload_bytes": manifest["packed_payload_bytes"],
        "performance_claim": False,
        "post_census_sha256": "d" * 64,
        "remote_preflight_sha256": "e" * 64,
        "remote_terminal_sha256": "f" * 64,
        "source_file_count": len(source_files),
        "source_inventory_sha256": source["inventory_sha256"],
        "tag": "greenfield_ws32_runtime_pack_20260815T010203123456789Z",
        "topology_hash": topology_hash,
        "tpu_initialized": False,
    }
    value["success_sha256"] = sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    (root / "SUCCESS").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value
