"""Layer-selectable legacy prompt index cache sealing and offline row comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.validation.prompt_index_cache import (
    ARTIFACT_KIND,
    FULL_INDEXER_LAYERS,
    LegacyPromptIndexCacheConfig,
    capture_legacy_prompt_index_cache,
    expected_prompt_cache_slot,
    inspect_legacy_prompt_index_cache,
    prompt_index_cache_artifact_kind,
)

REPO = Path(__file__).resolve().parents[3]
COMPARE = REPO / "scripts/greenfield/compare_layer1_prompt_index_cache_offline.py"
WRAPPER = REPO / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
LAUNCHER = REPO / "scripts/greenfield/run_capture_legacy_prompt_index_cache.sh"
SEALER = REPO / "scripts/greenfield/capture_legacy_prompt_index_cache.py"


def _write_dump(path: Path, *, process_index: int, global_bits: np.ndarray, slot: int) -> None:
    index = (slice(None),) * 4
    prefix = f"layer{slot}"
    payload = {
        "step_index": np.asarray(2, dtype=np.int64),
        "phase": np.asarray("postfwd"),
        "num_scheduled_tokens": np.asarray(2, dtype=np.int64),
        "process_index": np.asarray(process_index, dtype=np.int64),
        "process_count": np.asarray(2, dtype=np.int64),
        "layer_indices": np.asarray([slot], dtype=np.int64),
        "mesh_shape": np.asarray(
            "{'data': 1, 'attn_dp': 1, 'attn_dp_expert': 1, 'expert': 1, 'model': 4, 'dcp': 1}"
        ),
        "meta__block_tables": np.asarray([1, 2, 0, 0], dtype=np.int32),
        "meta__seq_lens": np.asarray([66], dtype=np.int32),
        f"{prefix}__sharding": np.asarray("P(None, 'dcp')"),
        f"{prefix}__shape": np.asarray(global_bits.shape, dtype=np.int64),
        f"{prefix}__dtype": np.asarray("bfloat16"),
        f"{prefix}__nshards": np.asarray(2, dtype=np.int64),
    }
    for shard in range(2):
        payload[f"{prefix}__shard{shard}__data"] = global_bits.copy()
        payload[f"{prefix}__shard{shard}__index"] = np.asarray(str(index))
        payload[f"{prefix}__shard{shard}__device"] = np.asarray(
            f"TPU_{process_index * 2 + shard}(process={process_index},({shard},0,0,0))"
        )
    np.savez(path, **payload)


def _config(source: Path, output: Path, *, layer_id: int) -> LegacyPromptIndexCacheConfig:
    return LegacyPromptIndexCacheConfig(
        source_dump_dir=source,
        output_dir=output,
        capture_code_hash="a" * 40,
        legacy_repository_pin="b" * 40,
        run_tag="unit",
        source_run_id=1,
        source_item_row_id=2,
        layer0_input_manifest_sha256="c" * 64,
        prompt_token_ids_sha256="d" * 64,
        expected_process_count=2,
        expected_local_replication=2,
        expected_physical_replication=4,
        expected_mesh_model_size=4,
        expected_mesh_dcp_size=1,
        expected_step_index=2,
        expected_last_chunk_tokens=2,
        expected_prompt_tokens=66,
        expected_physical_pages=3,
        expected_logical_page_size=64,
        expected_head_dim=4,
        layer_id=layer_id,
    )


def _global_bits(seed: float = 0.0) -> np.ndarray:
    values = (np.arange(3 * 64 * 4, dtype=np.float32).reshape(3, 64, 4) / 100 + seed).astype(
        ml_dtypes.bfloat16
    )
    return values.view(np.uint16).reshape(3, 2, 32, 4)


def _write_source(source: Path, *, slot: int, bits: np.ndarray) -> None:
    source.mkdir(parents=True, exist_ok=True)
    for process in range(2):
        _write_dump(
            source / f"index_cache.postfwd.step0002.proc{process}.npz",
            process_index=process,
            global_bits=bits,
            slot=slot,
        )


def test_slot_derivation_follows_sealed_registration_order():
    assert expected_prompt_cache_slot(0) == 0
    assert expected_prompt_cache_slot(1) == 2
    assert expected_prompt_cache_slot(2) == 4
    assert expected_prompt_cache_slot(6) == 12
    assert expected_prompt_cache_slot(3) == 6  # skip-top-k layers register an indexer cache too
    assert expected_prompt_cache_slot(77) == 154
    assert FULL_INDEXER_LAYERS == frozenset({0, 1, 2} | set(range(6, 78, 4)))
    for layer in (78, -1, True):
        with pytest.raises(ValueError):
            expected_prompt_cache_slot(layer)
    for layer in (3, 4, 5, 7, 78, -1):
        with pytest.raises(ValueError):
            prompt_index_cache_artifact_kind(layer)
    assert prompt_index_cache_artifact_kind(0) == ARTIFACT_KIND
    assert prompt_index_cache_artifact_kind(1) == "glm52_legacy_layer1_prompt_index_cache"


def test_config_rejects_layers_without_prompt_cache(tmp_path: Path):
    for layer in (3, 77, True):
        with pytest.raises(ValueError):
            _config(tmp_path / "s", tmp_path / "o", layer_id=layer)
    config = _config(tmp_path / "s", tmp_path / "o", layer_id=1)
    assert config.cache_slot == 2
    assert config.layer_name == "model.layers.1.self_attn.attn"
    assert config.artifact_kind == "glm52_legacy_layer1_prompt_index_cache"


def test_layer1_seal_requires_slot_two_and_binds_layer(tmp_path: Path):
    bits = _global_bits()
    _write_source(tmp_path / "slot2", slot=2, bits=bits)
    manifest = capture_legacy_prompt_index_cache(
        _config(tmp_path / "slot2", tmp_path / "artifact", layer_id=1)
    )
    assert manifest["artifact_kind"] == "glm52_legacy_layer1_prompt_index_cache"
    assert manifest["layer_id"] == 1 and manifest["cache_slot"] == 2
    assert manifest["layer_name"] == "model.layers.1.self_attn.attn"
    inspected, rows = inspect_legacy_prompt_index_cache(
        tmp_path / "artifact", expected_manifest_sha256=manifest["manifest_sha256"]
    )
    assert inspected["layer_id"] == 1 and rows.shape == (66, 4)
    # Layer-0 sealing must refuse a slot-2 dump and vice versa.
    with pytest.raises(ValueError):
        capture_legacy_prompt_index_cache(
            _config(tmp_path / "slot2", tmp_path / "artifact0", layer_id=0)
        )
    _write_source(tmp_path / "slot0", slot=0, bits=bits)
    with pytest.raises(ValueError):
        capture_legacy_prompt_index_cache(
            _config(tmp_path / "slot0", tmp_path / "artifact1", layer_id=1)
        )


def test_layer0_manifest_format_is_unchanged(tmp_path: Path):
    _write_source(tmp_path / "slot0", slot=0, bits=_global_bits())
    manifest = capture_legacy_prompt_index_cache(
        _config(tmp_path / "slot0", tmp_path / "artifact", layer_id=0)
    )
    assert manifest["artifact_kind"] == ARTIFACT_KIND
    assert not {"layer_id", "cache_slot", "layer_name"} & set(manifest)
    inspect_legacy_prompt_index_cache(tmp_path / "artifact")


def test_inspector_refuses_layer_binding_drift(tmp_path: Path):
    _write_source(tmp_path / "slot2", slot=2, bits=_global_bits())
    capture_legacy_prompt_index_cache(
        _config(tmp_path / "slot2", tmp_path / "artifact", layer_id=1)
    )
    manifest_path = tmp_path / "artifact" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    from glm_tpu.greenfield.validation.prompt_index_cache import _manifest_hash

    for key, value in (("cache_slot", 3), ("layer_name", "model.layers.2.self_attn.attn"), ("layer_id", 2)):
        forged = dict(manifest)
        forged[key] = value
        forged.pop("manifest_sha256")
        forged["manifest_sha256"] = _manifest_hash(forged)
        manifest_path.write_text(json.dumps(forged, indent=2, sort_keys=True) + "\n")
        with pytest.raises(ValueError):
            inspect_legacy_prompt_index_cache(tmp_path / "artifact")


@pytest.fixture(scope="module")
def compare_module():
    spec = importlib.util.spec_from_file_location("compare_layer1_prompt_cache", COMPARE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_row_mismatch_map_locates_rows_and_chunks(compare_module):
    rows = 5000
    expected = np.zeros((rows, 4), dtype=np.uint16)
    observed = expected.copy()
    observed[7, 0] ^= 1
    observed[2048, 1] ^= 1
    observed[4999, :] ^= 1
    result = compare_module.row_mismatch_map(expected, observed)
    assert result["mismatched_row_count"] == 3
    assert result["first_mismatched_position"] == 7
    assert result["last_mismatched_position"] == 4999
    assert result["mismatched_lanes_per_row_max"] == 4
    assert [c["mismatched_rows"] for c in result["per_legacy_prefill_chunk"]] == [1, 1, 1]
    assert result["chunk_boundary_rows_only"] is False
    exact = compare_module.row_mismatch_map(expected, expected)
    assert exact["mismatched_row_count"] == 0 and exact["first_mismatched_position"] is None
    with pytest.raises(ValueError):
        compare_module.row_mismatch_map(expected, observed[:10])


def test_compare_uses_sealed_artifact_and_owner_layout(compare_module, tmp_path: Path):
    bits = _global_bits()
    _write_source(tmp_path / "slot2", slot=2, bits=bits)
    manifest = capture_legacy_prompt_index_cache(
        _config(tmp_path / "slot2", tmp_path / "artifact", layer_id=1)
    )
    _, legacy_rows = inspect_legacy_prompt_index_cache(tmp_path / "artifact")
    owners = np.zeros((2, 16, 256, 4), dtype=np.uint16)
    for position in range(66):
        local = position % 512
        owners[local // 256, position // 512, local % 256] = legacy_rows[position]
    owners[0, 0, 5, 2] ^= 1  # perturb position 5
    np.savez(tmp_path / "capture.npz", layer1_index_cache_owners_bfloat16_bits=owners)
    # Layer 1 additionally requires the byte-pinned legacy internals and oracle,
    # which unit tests do not ship; exercise the row map through layer 2 and keep
    # the layer-1 rejection path as a static contract below.
    _write_source(tmp_path / "slot4", slot=4, bits=bits)
    manifest2 = capture_legacy_prompt_index_cache(
        _config(tmp_path / "slot4", tmp_path / "artifact2", layer_id=2)
    )
    np.savez(tmp_path / "capture2.npz", layer2_index_cache_owners_bfloat16_bits=owners)
    result = compare_module.compare(
        legacy_artifact_dir=tmp_path / "artifact2",
        layer_id=2,
        greenfield_capture=tmp_path / "capture2.npz",
        expected_manifest_sha256=manifest2["manifest_sha256"],
    )
    assert result["bitwise"]["mismatch_count"] == 1
    assert result["authenticated"] is True
    assert "NO_EVENT_AUTHENTICATION_AVAILABLE_FOR_THIS_LAYER" in result["classification"]
    assert result["row_map"]["mismatched_positions_first_64"] == [5]
    assert result["classification"].startswith("LAYER2_PROMPT_INDEX_CACHE_1_OF_66_ROWS_MISMATCH")
    with pytest.raises(ValueError):
        compare_module.compare(
            legacy_artifact_dir=tmp_path / "artifact",
            layer_id=0,
            greenfield_capture=tmp_path / "capture.npz",
            expected_manifest_sha256=None,
        )


def test_compare_script_rejects_unauthenticated_layer1_capture():
    source = COMPARE.read_text()
    assert 'if internals_sha != diagnose.LEGACY_INTERNALS_SHA256[1]:' in source
    assert 'diagnose.load_oracle_events()' in source  # byte-pinned oracle
    assert 'if not result["authenticated"]:' in source
    assert "raise SystemExit(" in source
    assert "--skip-event1-check" not in source
    diagnose = (REPO / "scripts/greenfield/diagnose_event1_prompt_index_cache_offline.py").read_text()
    assert "if sha256_file(DSA_EVENTS_ORACLE) != DSA_EVENTS_ORACLE_SHA256:" in diagnose


def test_wrapper_launcher_and_sealer_thread_the_layer():
    wrapper = WRAPPER.read_text()
    assert 'readonly PROMPT_CACHE_LAYER_ID=${GLM_GREENFIELD_PROMPT_CACHE_LAYER_ID:-0}' in wrapper
    assert "expected_prompt_cache_slot(int(sys.argv[1]))" in wrapper
    assert 'GLM_DCP_CACHE_DUMP_LAYERS=$PROMPT_CACHE_SLOT"' in wrapper
    assert 'grep -qx "GLM_DCP_CACHE_DUMP_LAYERS=\'"$PROMPT_CACHE_SLOT"\'"' in wrapper
    assert '--layer-id "$PROMPT_CACHE_LAYER_ID"' in wrapper
    assert 'GLM_DCP_CACHE_DUMP_LAYERS=0"' not in wrapper
    assert "$PROMPT_CACHE_LAYER_ID == 0 && \\" in wrapper  # prompt-key capture stays layer 0
    assert 'if [[ $PROMPT_CACHE_LAYER_ID != 0 ]]; then' in wrapper  # one-host probe stays layer 0
    # Finalization branches on the layer: deeper layers bind the sealed identity
    # instead of reading the layer-0 production comparison.
    assert '"${OBSERVER_BRANCH:-none}" "$PROMPT_CACHE_LAYER_ID" \\' in wrapper
    assert 'elif sys.argv[39] != "0":' in wrapper
    assert 'raise SystemExit("prompt index-cache layer binding drifted")' in wrapper
    assert wrapper.index('elif sys.argv[39] != "0":') < wrapper.index('/ "prompt_index_cache_comparison"')
    # The sealed layer-0 DSA input is verified before any TPU work in prompt-cache mode.
    assert "sealed layer-0 DSA input required by prompt-cache sealing is unavailable" in wrapper
    preflight = wrapper.index("inspect_layer0_dsa_association_input(\n    Path(sys.argv[1])")
    assert preflight < wrapper.index("exec 9>/home/gianl/glm-run/.glm_pod_workload.lock")
    # Recreated-pod legacy runtime: the prompt-cache mode may run on the reviewed
    # layer-1 observer bundle; every bundle site keys on BUNDLE_RUNTIME.
    assert 'readonly PROMPT_CACHE_RUNTIME=${GLM_GREENFIELD_PROMPT_CACHE_RUNTIME:-oracle}' in wrapper
    assert 'elif [[ $PROMPT_CACHE_CAPTURE == 1 && $PROMPT_CACHE_RUNTIME == layer1_observer_bundle ]]; then' in wrapper
    assert 'readonly LEGACY_PIN=c7973435aa2fc948da9185ef99938f886613ce2f' in wrapper
    assert 'readonly OBSERVER_RUNTIME_REPO=/home/gianl/tpu-inference-dsa-internal-c7973435a' in wrapper
    assert 'or sys.argv[4] != "c7973435aa2fc948da9185ef99938f886613ce2f"' in wrapper
    assert 'if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 || $PROMPT_CACHE_RUNTIME == layer1_observer_bundle ]]; then' in wrapper
    assert wrapper.count("$BUNDLE_RUNTIME == 1") >= 6
    assert 'if [[ $LAYER1_RMS_INPUT_CAPTURE == 1 ]]; then\n    # The recreated pod' not in wrapper
    assert 'COMMON_ENVS="PYTHONPATH=$OBSERVER_RUNTIME_REPO:$VLLM_RUNTIME_ROOT $COMMON_ENVS"' in wrapper
    assert '\'"$prompt_cache_pythonpath_check"\'' in wrapper
    assert '"$PROMPT_CACHE_LAYER_ID" \\\n  "$PROMPT_CACHE_RUNTIME" <<\'PY\'' in wrapper
    assert 'if sys.argv[40] == "layer1_observer_bundle":' in wrapper
    assert 'raise SystemExit("prompt-cache bundle runtime identity drifted")' in wrapper
    assert 'elif sys.argv[40] != "oracle":' in wrapper
    launcher = LAUNCHER.read_text()
    assert "export GLM_GREENFIELD_PROMPT_CACHE_RUNTIME=${GLM_GREENFIELD_PROMPT_CACHE_RUNTIME:-oracle}" in launcher
    assert "greenfield_legacy_layer${PROMPT_CACHE_LAYER_ID}_prompt_index_cache_" in launcher
    assert "export GLM_GREENFIELD_PROMPT_CACHE_LAYER_ID=$PROMPT_CACHE_LAYER_ID" in launcher
    sealer = SEALER.read_text()
    assert '"--layer-id", type=int, default=0' in sealer
    assert "layer_id=args.layer_id" in sealer
