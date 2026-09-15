"""The pure-NumPy safetensors reader must agree with the reference library and bind digests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking import numpy_safetensors as ns

DSA_INPUT = Path("/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z")
DSA_INPUT_SHA = "574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141"
LEGACY_CACHE = Path(
    "/home/gianl/glm-run/greenfield_legacy_layer1_prompt_index_cache_20260903T000356727206404Z/prompt_index_cache"
)
LEGACY_CACHE_SHA = "d9058cc6584aca784212706789e72b4981754cb9880e553122b5e854bc3961ac"


def test_reader_matches_reference_library_on_synthetic_file(tmp_path: Path):
    safetensors = pytest.importorskip("safetensors.numpy")
    import ml_dtypes

    rng = np.random.default_rng(3)
    tensors = {
        "f32": rng.normal(size=(3, 5)).astype(np.float32),
        "bf16": rng.normal(size=(4,)).astype(ml_dtypes.bfloat16),
        "u8": rng.integers(0, 255, size=(2, 2), dtype=np.uint8),
        "i32": np.arange(6, dtype=np.int32).reshape(2, 3),
    }
    path = tmp_path / "t.safetensors"
    safetensors.save_file(tensors, str(path), metadata={"artifact_kind": "unit"})
    arrays, metadata = ns.read_tensors(path)
    assert metadata == {"artifact_kind": "unit"}
    np.testing.assert_array_equal(arrays["f32"], tensors["f32"])
    np.testing.assert_array_equal(arrays["bf16"], tensors["bf16"].view(np.uint16))
    np.testing.assert_array_equal(arrays["u8"], tensors["u8"])
    np.testing.assert_array_equal(arrays["i32"], tensors["i32"])
    subset, _ = ns.read_tensors(path, ["i32"])
    assert set(subset) == {"i32"}


@pytest.mark.skipif(not DSA_INPUT.exists(), reason="sealed layer-0 DSA input not present")
def test_layer0_dsa_input_loader_binds_manifest_and_arrays():
    manifest, arrays = ns.load_layer0_dsa_association_input(DSA_INPUT, expected_manifest_sha256=DSA_INPUT_SHA)
    assert manifest["manifest_sha256"] == DSA_INPUT_SHA
    assert arrays["prompt_token_ids"].shape == (8155,)
    assert arrays["self_attn__q_a_proj__weight"].shape == (2048, 6144) and arrays["self_attn__q_a_proj__weight"].dtype == np.uint8
    assert arrays["unique_embedding_bfloat16_bits"].dtype == np.dtype("<u2")
    with pytest.raises(ValueError):
        ns.load_layer0_dsa_association_input(DSA_INPUT, expected_manifest_sha256="0" * 64)


@pytest.mark.skipif(not LEGACY_CACHE.exists(), reason="sealed legacy layer-1 cache not present")
def test_legacy_cache_loader_binds_manifest_digest_and_layer():
    manifest, bits = ns.load_legacy_prompt_index_cache(LEGACY_CACHE, expected_manifest_sha256=LEGACY_CACHE_SHA)
    assert manifest["manifest_sha256"] == LEGACY_CACHE_SHA
    assert int(manifest["layer_id"]) == 1
    assert bits.dtype == np.dtype("<u2") and bits.ndim == 2
    with pytest.raises(ValueError):
        ns.load_legacy_prompt_index_cache(LEGACY_CACHE, expected_manifest_sha256="0" * 64)


def test_checkpoint_loader_reads_fp8_bits_via_index(tmp_path: Path):
    safetensors = pytest.importorskip("safetensors.numpy")
    header_weights = {"model.layers.0.x.weight": np.arange(16, dtype=np.uint8).reshape(4, 4)}
    safetensors.save_file(header_weights, str(tmp_path / "shard.safetensors"))
    index = {"weight_map": {"model.layers.0.x.weight": "shard.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    sha = ns.sha256_file(tmp_path / "model.safetensors.index.json")
    tensors = ns.load_checkpoint_tensors(tmp_path, {"x": "model.layers.0.x.weight"}, index_sha256=sha)
    np.testing.assert_array_equal(tensors["x"], header_weights["model.layers.0.x.weight"])
    with pytest.raises(ValueError):
        ns.load_checkpoint_tensors(tmp_path, {"x": "model.layers.0.x.weight"}, index_sha256="0" * 64)
