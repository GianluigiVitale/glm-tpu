"""Compact capture ordering: source/abstract proof only, not actual TPU fit."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from glm_tpu.greenfield.errors import PlanValidationError
from glm_tpu.greenfield.runtime import ws32_batched_prefill as runtime
from scripts.greenfield import ws32_capture_barrier_compile as candidate
from scripts.greenfield import ws32_flat_rows_compile as previous
from scripts.greenfield import ws32_rolled_prefill_compile as original
from scripts.greenfield import ws32_batched_prefill_runner as adapter
from tests.greenfield.runtime.test_ws32_batched_prefill_runner import config
from tests.greenfield.hlo import test_ws32_pending_rows_compile as reused

ROOT = reused.ROOT


def test_exact_source_and_old_profile_refusal():
    candidate.require_source(ROOT)
    with pytest.raises(ValueError, match="source/prerequisite"):
        previous.require_source(ROOT)


@pytest.mark.parametrize("bad", [None, 0, 1, "true", True])
def test_invalid_capture_options_before_inputs_or_backend(bad):
    cfg = SimpleNamespace(exact_dsa=False, strategy_nd_dense=False,
                          host_main_rope_table=True, logical_page_size=512)
    for function, args in [(runtime.build_ws32_batched_prefill_program, (None,cfg)),
                           (runtime.ws32_batched_prefill_mapped, (None,)*6)]:
        kwargs = dict(block_rows=128) if function is runtime.build_ws32_batched_prefill_program else dict(config=cfg)
        with pytest.raises(PlanValidationError,match="capture barrier"):
            function(*args,capture_barrier=bad,**kwargs)
    with pytest.raises(ValueError):
        original.read_metadata(ROOT,full_canonical=True,capture_barrier=bad)
    with pytest.raises(ValueError):
        original.prepare(None,None,repo=ROOT,full_canonical=True,capture_barrier=bad)
    with pytest.raises(ValueError,match="capture barrier"):
        adapter.build_graph_pair(None,None,None,capture_barrier=bad)


@pytest.mark.parametrize("label",[None,"128k_d1_0"])
def test_only_e0_preparation_allowed(label):
    with pytest.raises(ValueError,match="restricted"):
        original.prepare(None,None,repo=ROOT,full_canonical=True,
                         pending_cache_rows=True,flat_pending_rows=True,
                         capture_barrier=True,long_context_label=label)


def test_both_builders_receive_capture_option_and_default_stays_off(monkeypatch):
    cfg=replace(config(),exact_dsa=False,strategy_nd_dense=False)
    plan=adapter.BatchedPrefillPlan(2034,128,8192,mlp_window=True)
    calls=[]
    monkeypatch.setattr(adapter,"build_ws32_batched_prefill_program",
                        lambda *a,**kw:calls.append(kw) or object())
    adapter.build_graph_pair(None,cfg,plan,pending_cache_rows=True,
                             flat_pending_rows=True,capture_barrier=True)
    assert len(calls)==2 and all(c['capture_barrier'] is True for c in calls)
    calls.clear()
    adapter.build_graph_pair(None,cfg,plan)
    assert len(calls)==2 and all('capture_barrier' not in c for c in calls)


def test_production_e0_lowering():
    reused.run_production_e0_lowering("ws32_capture_barrier_compile",flat=True,
                                     capture=True)


@pytest.mark.parametrize("path",list(candidate.MODEL_SOURCE_OVERRIDES)+list(candidate.PREREQUISITES))
def test_source_mutations_refuse(tmp_path,monkeypatch,path):
    monkeypatch.setattr(reused,"candidate",candidate)
    reused.test_changed_source_or_prerequisite_refuses(tmp_path,path)


def test_unregistered_raw_cannot_admit_originals(monkeypatch):
    candidate.require_source(ROOT)
    monkeypatch.setattr(candidate,"RAW",{})
    with pytest.raises(ValueError,match="RAW registration differs"):
        candidate.validate_preserved_pair(ROOT,{},repo=ROOT)
