"""§26 long plans and production abstract preparation, never TPU admission."""

from pathlib import Path
import runpy

import pytest

from glm_tpu.greenfield.validation.long_context_oracle import WS32_LONG_CONTEXT_PROFILES
from glm_tpu.greenfield.validation.ws32_delivery_prefill import long_plan
from glm_tpu.greenfield.validation.ws32_prefill import PREFILL_MODE, require_batched_profile
from scripts.greenfield import ws32_rolled_prefill_compile as compiler


ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("label", tuple(WS32_LONG_CONTEXT_PROFILES))
def test_long_plan_covers_exact_prompt_and_full_decode_window(label):
    plan = long_plan(label)
    e0 = label == "256k_e0"
    assert plan.prompt_length == (262144 if e0 else 127363)
    assert plan.context_capacity == (262656 if e0 else 131072)
    assert plan.split == ((2047, 128) if e0 else (995, 3))
    assert plan.graph_rows == (("prefill_chunk", 128), ("prefill_tail", 128 if e0 else 114))
    assert plan.stride_rows == 128 and plan.mlp_window
    full, tail = plan.split
    assert full * 128 + tail == plan.prompt_length
    assert plan.prompt_length + 14 + 2 + (256 if e0 else 10) + 2 + 1 <= plan.context_capacity
    identity = plan.identity()
    assert identity["donate_argnums"] == []
    assert identity["repair_promotion"] == "final_healthy_commit_only"
    assert identity["head_execution"] == "final_live_row_only"
    assert identity["padding_semantics"] == "masked_not_prompt_tokens"


@pytest.mark.parametrize("label", (None, True, 128000, [], "8k", "128k", "256k", "128k_d0_5"))
def test_unregistered_or_nonstring_labels_refuse(label):
    with pytest.raises(ValueError, match="registered long-context"):
        long_plan(label)


def test_registry_drift_refuses(monkeypatch):
    monkeypatch.setitem(WS32_LONG_CONTEXT_PROFILES, "256k_e0", {
        **WS32_LONG_CONTEXT_PROFILES["256k_e0"], "prompt_token_count": 262143,
    })
    with pytest.raises(ValueError, match="registry differs"):
        long_plan("256k_e0")


def test_no_long_preparation_without_canonical_correction():
    with pytest.raises(ValueError, match="canonical dense correction"):
        compiler.prepare(None, None, repo=ROOT, long_context_label="256k_e0")


def test_geometry_does_not_open_historical_numerical_admission():
    with pytest.raises(ValueError, match="batched long context"):
        require_batched_profile(
            PREFILL_MODE, exact_dsa=True, host_main_rope_table=True,
            block_rows=128, long_context="e0", adjudication_record=None,
            adjudication_sha256="0" * 64, mlp_window=True,
        )


def test_production_abstract_short_and_both_long_capacities_without_payload():
    helpers = runpy.run_path(str(ROOT / "tests/greenfield/runtime/test_ws32_paired_prefill.py"))
    source = r"""
from pathlib import Path
from unittest.mock import patch
import jax,numpy as np
from jax.sharding import Mesh
from scripts.greenfield import ws32_rolled_prefill_compile as compiler
from scripts.greenfield import ws32_canonical_prefill_compile as canonical
from scripts.greenfield import ws32_batched_prefill_runner as adapter
assert jax.default_backend() == 'cpu'
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
root=Path.cwd()
old=Path.open
def checked(path,*args,**kwargs):
    if '/glm-ws32-runtime/' in str(path):
        assert path.name in ('manifest.json','SUCCESS'),path
    assert path.suffix not in ('.safetensors','.bin'),path
    return old(path,*args,**kwargs)
with patch.object(Path,'open',checked),patch('jax.device_put',side_effect=AssertionError('payload placement')):
    metadata=canonical.read_metadata(root)
    assert len(metadata.manifest['tensor_schema']) == 2310
    for label,capacity,pages,tail in ((None,8192,16,114),('128k_d1_0',131072,256,114),('256k_e0',262656,513,128)):
        with patch.object(adapter,'build_graph_pair',wraps=adapter.build_graph_pair) as build:
            prepared=compiler.prepare(mesh,metadata,repo=root,full_canonical=True,long_context_label=label)
        assert build.call_args.kwargs == canonical.program_options()
        assert set(prepared.inputs) == set(prepared.programs) == {'prefill_chunk','prefill_tail'}
        assert prepared.manifest_sha256 == metadata.manifest['manifest_sha256']
        for name,rows in (('prefill_chunk',128),('prefill_tail',tail)):
            ids,count,state,weights,wk,rope=prepared.inputs[name]
            assert ids.shape == (rows,) and count.shape == ()
            assert state.decoder.block_tables.shape == (1,pages)
            assert state.decoder.kv_cache_local.shape == (78,pages,512,640)
            assert state.decoder.index_cache_local.shape == (21,pages,512,128)
            assert state.repaired_index_local.shape == (21,pages,512,128)
            assert rope.shape == (capacity,64)
            assert len(weights.layers) == 78 and len(wk) == 21
            assert all(isinstance(x,jax.ShapeDtypeStruct) for x in jax.tree.leaves(prepared.inputs[name]))
            assert prepared.programs[name].canonical_dense
            for key,value in canonical.program_options().items():
                if key != 'key_tile':
                    assert getattr(prepared.programs[name],key) == value,(key,value)
        assert prepared.inputs['prefill_chunk'][3] is prepared.inputs['prefill_tail'][3]
print('SHORT_AND_LONG_METADATA_PREPARATION_PASS',flush=True)
"""
    assert "SHORT_AND_LONG_METADATA_PREPARATION_PASS" in helpers["_cpu"](source, timeout=180)
