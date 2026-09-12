"""Fixed long program choice; abstract lowering is not a TPU execution test."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from glm_tpu.greenfield.validation.long_context_oracle import WS32_LONG_CONTEXT_PROFILES
from scripts.greenfield import ws32_delivery_programs as delivery
from scripts.greenfield import ws32_rolled_prefill_compile as original

ROOT = Path(__file__).resolve().parents[3]


def test_current_source_passes_old_source_still_refuses():
    delivery.require_source(ROOT)
    with pytest.raises(ValueError, match="source/prerequisite"):
        delivery.historical.require_source(ROOT)


@pytest.mark.parametrize("label", list(WS32_LONG_CONTEXT_PROFILES))
def test_fixed_roles_and_ownership(label):
    pins = delivery.raw_registration(label)
    assert set(pins) == {"prefill_chunk", "prefill_tail"}
    assert delivery.state_ownership(label) == (delivery.CONTRACT if label == "256k_e0" else None)
    assert (pins["prefill_chunk"] == pins["prefill_tail"]) == (label == "256k_e0")


@pytest.mark.parametrize("bad", [None, 1, True, "8k", "256k", "128k_d0_5"])
def test_unknown_workload_refuses_before_source(monkeypatch, bad):
    monkeypatch.setattr(delivery, "require_source", lambda _: pytest.fail("source work"))
    for fn in (delivery.raw_registration, delivery.state_ownership):
        with pytest.raises(ValueError, match="registered long-context"):
            fn(bad)
    with pytest.raises(ValueError, match="registered long-context"):
        delivery.prepare(None, None, repo=ROOT, context_label=bad)


@pytest.mark.parametrize("bad", [None, 1, "true"])
def test_source_switch_is_strict_bool(bad):
    with pytest.raises(ValueError, match="static bool"):
        original.read_metadata(ROOT, delivery_source=bad)
    with pytest.raises(ValueError, match="static bool"):
        original.prepare(None, None, repo=ROOT, delivery_source=bad)


@pytest.mark.parametrize("option", ["canonical_dense", "pending_cache_rows", "flat_pending_rows", "capture_barrier"])
def test_source_mode_does_not_override_old_options(option):
    with pytest.raises(ValueError):
        original.read_metadata(ROOT, full_canonical=True, delivery_source=True, **{option: True})
    if option != "canonical_dense":
        with pytest.raises(ValueError, match="L7 base-only"):
            original.prepare(None, None, repo=ROOT, full_canonical=True,
                             delivery_source=True, long_context_label="128k_d1_0", **{option: True})


@pytest.mark.parametrize("label", [None, "256k_e0"])
def test_source_choice_cannot_authorize_short_or_undonated_e0(label):
    with pytest.raises(ValueError, match="L7 base-only|acquired capture-barrier"):
        original.prepare(None, None, repo=ROOT, full_canonical=True,
                         delivery_source=True, long_context_label=label)


@pytest.mark.parametrize("name", list(delivery.PREREQUISITES))
def test_original_evidence_mutation_refuses(monkeypatch, name):
    old = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: old(p) + (b"changed" if p == ROOT / name else b""))
    with pytest.raises(ValueError, match="prerequisite"):
        delivery.require_source(ROOT)


def test_original_l7_registration_mutation_refuses(monkeypatch):
    monkeypatch.setitem(delivery.historical.RAW, "prefill_128k_tail", (1, "0" * 64))
    with pytest.raises(ValueError, match="L7 RAW"):
        delivery.require_source(ROOT)


def test_production_long_roles_reproduce_originals_without_payload_or_compile():
    source = r'''
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import jax, numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import ws32_delivery_programs as delivery
assert jax.default_backend() == 'cpu'
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
mesh = Mesh(np.asarray(jax.devices(), object).reshape(8, 4), ('expert', 'feature'))
old = Path.open
def checked(path, *args, **kwargs):
    assert path.suffix not in ('.safetensors', '.bin'), path
    if '/glm-ws32-runtime/' in str(path):
        assert path.name in ('manifest.json', 'SUCCESS'), path
    return old(path, *args, **kwargs)
with patch.object(Path, 'open', checked), \
     patch('jax.device_put', side_effect=AssertionError('payload placement')), \
     patch('jax.stages.Compiled.__call__', side_effect=AssertionError('dispatch')), \
     patch('jax.stages.Lowered.compile', side_effect=AssertionError('compilation')):
    metadata = delivery.read_metadata(Path.cwd())
    assert len(metadata.manifest['tensor_schema']) == 2310
    for label in ('128k_d1_0', '256k_e0'):
        pair = delivery.prepare(mesh, metadata, repo=Path.cwd(), context_label=label)
        e0 = label == '256k_e0'
        assert (pair.programs['prefill_chunk'] is pair.programs['prefill_tail']) == e0
        if e0:
            assert pair.inputs['prefill_chunk'] is pair.inputs['prefill_tail']
        for role in (('prefill_chunk',) if e0 else ('prefill_chunk', 'prefill_tail')):
            args, program = pair.inputs[role], pair.programs[role]
            ids, count, state, weights, wk, rope = args
            base = program.original_program if e0 else program
            assert base.canonical_dense is True
            assert base.pending_cache_rows is e0
            assert base.flat_pending_rows is e0 and base.capture_barrier is e0
            assert ids.shape == ((114,) if role == 'prefill_tail' else (128,))
            capacity = 262656 if e0 else 131072
            assert state.decoder.block_tables.shape == (1, capacity // 512)
            assert rope.shape == (capacity, 64) and len(weights.layers) == 78
            assert len(wk) == 21
            assert all(isinstance(x, jax.ShapeDtypeStruct) and x.sharding is not None
                       for x in jax.tree.leaves(args))
            traced = program.execute.trace(*args)
            expected_donors = tuple(range(2, 2 + len(jax.tree.leaves(state)))) if e0 else ()
            assert traced.donate_argnums == expected_donors
            with patch('jax._src.tpu_custom_call.get_ir_version', return_value=None):
                raw = str(traced.lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
            actual = (len(raw), sha256(raw).hexdigest())
            print(label, role, actual, flush=True)
            assert actual == delivery.raw_registration(label)[role]
print('LONG_DELIVERY_ORIGINAL_RAW_PASS', flush=True)
'''
    flags = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32").strip()
    result = subprocess.run(
        [sys.executable, "-c", source], cwd=ROOT, capture_output=True, text=True,
        timeout=360, env=dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=flags),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LONG_DELIVERY_ORIGINAL_RAW_PASS" in result.stdout
