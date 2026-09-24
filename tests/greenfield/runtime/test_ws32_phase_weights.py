"""Phase ownership and original WK math; CPU mechanisms, not TPU admission."""

from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glm_tpu.greenfield.runtime.ws32_decoder import Ws32DecoderConfig, ws32_decoder_weight_names
from glm_tpu.optimized.geometry import ModelGeometry
from scripts.greenfield.ws32_phase_weights import PhaseWeights


ROOT = Path(__file__).resolve().parents[3]


def local_phase(name, action):
    return action()


def config(*, overlay=True):
    geometry = ModelGeometry.from_hf_config(json.loads((ROOT / "configs/glm-5.2-fp8-config.json").read_text()))
    return Ws32DecoderConfig(geometry, 262656, exact_dsa=True,
                             strategy_nd_dense=overlay, host_main_rope_table=True)


def base_arrays(cfg):
    raw = replace(cfg, exact_dsa=False, strategy_nd_dense=False)
    # Actual 2310-leaf NAME/tree contract; tiny data, not checkpoint contents.
    return {name: jax.device_put(np.asarray(i, np.int32))
            for i, name in enumerate(jax.tree.leaves(ws32_decoder_weight_names(raw)))}


def test_production_phase_ownership_drops_raw_dense_and_repair_but_not_shared():
    cfg = config()
    arrays = base_arrays(cfg)
    owner = PhaseWeights(arrays, cfg)
    assert len(arrays) == 2310 and len(owner._wk_sources()) == 21
    shared = owner.raw_weights.layers[3].moe.expert_gate_bits_local
    dense_names = jax.tree.leaves(ws32_decoder_weight_names(owner.raw_config).layers[:3])
    dense_only = set(jax.tree.leaves(ws32_decoder_weight_names(owner.raw_config))) - set(
        jax.tree.leaves(ws32_decoder_weight_names(cfg)))
    assert len(dense_only) == 18 and dense_only.issubset(dense_names)
    refs = [weakref.ref(arrays[name]) for name in dense_only]
    del arrays
    gc.collect()
    assert all(ref() is not None for ref in refs)

    calls = []
    def call(stage, graph, inputs):
        calls.append((stage, graph))
        layer = cfg.full_index_slots[len(calls) // 2 - (0 if graph == "wk_decode" else 1)]
        if graph == "wk_decode":
            # Each producer consumes its actual original source, in model order.
            assert inputs[0] is owner.raw_weights.layers[layer].dsa.wk_bits_local
        return jax.device_put(np.full((128, 6144), len(calls), dtype=(
            jnp.bfloat16 if graph == "wk_decode" else np.float32)))
    owner.materialize_wk(call, phase=local_phase)
    assert len(calls) == 42 and owner.phase == "prefill"
    assert calls == [(f"layer{layer}/{graph}", graph) for layer in cfg.full_index_slots
                     for graph in ("wk_decode", "wk_promote")]
    assert set(owner.resident_roots()) == {"raw_prefill_weights", "completed_repair_wk"}
    wk_refs = [weakref.ref(v) for v in owner.wk]
    overlay_names = set(jax.tree.leaves(ws32_decoder_weight_names(cfg))) - set(
        jax.tree.leaves(ws32_decoder_weight_names(owner.raw_config)))
    assert len(overlay_names) == 12
    overlay = {name: jax.device_put(np.int32(i)) for i, name in enumerate(sorted(overlay_names))}
    with pytest.raises(ValueError, match="overlay tensor set"):
        owner.begin_decode({})
    assert owner.phase == "prefill" and all(ref() is not None for ref in refs + wk_refs)
    decoded = owner.begin_decode(overlay)
    gc.collect()
    assert all(ref() is None for ref in refs + wk_refs)
    assert decoded.layers[3].moe.expert_gate_bits_local is shared
    assert all(a is b for a, b in zip(jax.tree.leaves(decoded.layers[0].dense),
        [overlay[n] for n in jax.tree.leaves(ws32_decoder_weight_names(cfg).layers[0].dense)], strict=True))
    assert owner.raw_weights is None and owner.wk == ()
    assert owner.resident_roots() == {"decode_weights": decoded}
    with pytest.raises(ValueError, match="one-shot"):
        owner.materialize_wk(call, phase=local_phase)
    with pytest.raises(ValueError, match="completed prefill"):
        owner.begin_decode(overlay)


@pytest.mark.parametrize("extra", ["overlay", "missing"])
def test_base_rejects_overlay_or_missing_source(extra):
    cfg = config()
    arrays = base_arrays(cfg)
    if extra == "overlay":
        arrays["model.layers.0.mlp.strategy_nd.extra"] = object()
    else:
        arrays.pop(next(iter(arrays)))
    with pytest.raises(ValueError, match="exactly the base"):
        PhaseWeights(arrays, cfg)


def test_failed_materialization_is_terminal():
    cfg = config()
    owner = PhaseWeights(base_arrays(cfg), cfg)
    with pytest.raises(ValueError, match="completed prefill"):
        owner.begin_decode({})
    with pytest.raises(ValueError, match="replicated bfloat16"):
        owner.materialize_wk(lambda *args: jnp.ones((2, 3)), phase=local_phase)
    assert owner.phase == "failed" and owner.raw_weights is not None
    with pytest.raises(ValueError, match="one-shot"):
        owner.materialize_wk(lambda *args: None, phase=local_phase)


@pytest.mark.parametrize("refused", ["wk_decode_ready", "wk_promote_ready"])
def test_peer_only_validation_refusal_prevents_next_collective(refused):
    owner = PhaseWeights(base_arrays(config()), config())
    calls = []
    def call(stage, graph, args):
        calls.append(graph)
        return jax.device_put(np.zeros((128, 6144), dtype=(
            jnp.bfloat16 if graph == "wk_decode" else np.float32)))
    def phase(name, action):
        value = action()
        if name.endswith(refused):
            raise RuntimeError("peer-only refusal")
        return value
    with pytest.raises(RuntimeError, match="peer-only"):
        owner.materialize_wk(call, phase=phase)
    assert calls == (["wk_decode"] if refused == "wk_decode_ready" else ["wk_decode", "wk_promote"])
    assert owner.phase == "failed"


def test_wk_only_original_programs_equal_full_exact_materializer_cpu32():
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("ws32_phase_weights_cpu.py"))],
        cwd=ROOT, env={**os.environ, "JAX_PLATFORMS": "cpu",
                       "PYTHONPATH": str(ROOT),
                       "XLA_FLAGS": "--xla_force_host_platform_device_count=32"},
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WK_PHASE_CPU32_PASS" in result.stdout
