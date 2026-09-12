"""Changed E0 source/preparation and existing protected-route refusals."""
from types import SimpleNamespace
import json

import pytest

from scripts.greenfield import ws32_flat_rows_compile as candidate
from scripts.greenfield import ws32_pending_rows_compile as pending
from scripts.greenfield import ws32_rolled_prefill_compile as original
from scripts.greenfield import ws32_prefill_budget_campaign as campaign
from scripts.greenfield import ws32_rolled_prefill_worker as worker
from scripts.greenfield import ws32_rolled_prefill_evidence as evidence
from tests.greenfield.hlo import test_ws32_pending_rows_compile as prep_checks
from tests.greenfield.hlo import test_ws32_rolled_prefill_worker as checks
from tests.greenfield.hlo import test_ws32_owned_state_compile as owned_checks
from tests.greenfield.hlo.test_ws32_rolled_prefill_worker import lifecycle

ROOT = checks.ROOT
TAG = "greenfield_fp8_" + candidate.KERNEL + "_fixture"
PIN = "c" * 40


def test_source_registration_and_historical_refusal():
    # DB614 owns the original flat-row source. A new capture-lifetime option
    # must not execute or re-lower under that historical source registration.
    for old in (candidate,pending):
        with pytest.raises(ValueError,match="source/prerequisite"):
            old.require_source(ROOT)


@pytest.mark.parametrize("path",list(candidate.MODEL_SOURCE_OVERRIDES)+list(candidate.PREREQUISITES))
def test_source_mutations_refuse(tmp_path,monkeypatch,path):
    monkeypatch.setattr(prep_checks,"candidate",candidate)
    prep_checks.test_changed_source_or_prerequisite_refuses(tmp_path,path)


def test_production_e0_lowering():
    with pytest.raises(ValueError,match="source/prerequisite"):
        candidate.read_metadata(ROOT)


def test_invalid_source_modes_and_adapter_flags():
    from dataclasses import replace
    from scripts.greenfield import ws32_batched_prefill_runner as adapter
    from tests.greenfield.runtime.test_ws32_batched_prefill_runner import config
    cfg=replace(config(),exact_dsa=False,strategy_nd_dense=False)
    plan=adapter.BatchedPrefillPlan(2034,128,8192,mlp_window=True)
    for value in (None,1,"true",True):
        # True without pending-cache opt-in is invalid too.
        with pytest.raises(ValueError):
            original.read_metadata(ROOT,full_canonical=True,flat_pending_rows=value)
        with pytest.raises(ValueError):
            original.prepare(None,None,repo=ROOT,full_canonical=True,flat_pending_rows=value)
        with pytest.raises(ValueError):
            adapter.build_graph_pair(None,cfg,plan,flat_pending_rows=value)
    for label in (None,"128k_d1_0"):
        with pytest.raises(ValueError,match="restricted"):
            original.prepare(None,None,repo=ROOT,full_canonical=True,
                             pending_cache_rows=True,flat_pending_rows=True,long_context_label=label)


@pytest.mark.parametrize("lifecycle",["flat_rows"],indirect=True)
@pytest.mark.parametrize("case",["complete","memory","identity","finalize","compile","originals"])
def test_worker_and_original_reader(lifecycle,monkeypatch,case):
    if case=="originals":
        monkeypatch.setattr(owned_checks,"candidate",candidate)
        owned_checks.test_preserved_originals_scope_and_allocation_refusals(lifecycle)
    else:
        owned_checks.test_actual_worker_refusal_and_preservation(lifecycle,monkeypatch,case)


@pytest.mark.parametrize("lifecycle",["flat_rows"],indirect=True)
@pytest.mark.parametrize("phase",range(5))
def test_each_peer_refusal(lifecycle,phase):
    checks.test_peer_refusal_never_advances_or_leaves_success(lifecycle,phase)


@pytest.mark.parametrize("lifecycle",["flat_rows"],indirect=True)
@pytest.mark.parametrize("metadata_failure",[False,True])
def test_actual_cli_before_runtime(lifecycle,monkeypatch,metadata_failure):
    checks.test_actual_probe_selects_compile_only_before_runtime(lifecycle,monkeypatch,metadata_failure)


def test_fixed_mode_and_raw_pin(monkeypatch):
    mode=worker.compile_mode(flat_rows=True)
    assert mode.programs==candidate.PROGRAMS==("prefill_256k_flat_rows",)
    assert campaign.program_names(TAG)==mode.programs
    assert len(campaign.evidence_files(TAG))==5
    assert campaign.evidence_files(TAG)==evidence.files(flat_rows=True)
    assert campaign.rank_byte_limit(TAG)==192<<20
    assert "timeout --kill-after=30s 900s" in campaign.launch_command(TAG,PIN,"10.0.0.1:8476")
    for invalid in (None,1,"true"):
        with pytest.raises(ValueError,match="static bool"):worker.compile_mode(flat_rows=invalid)
    for flag in ("history","full_canonical","canonical_dense","delivery","owned_state","pending_rows"):
        with pytest.raises(ValueError,match="exclusive"):worker.compile_mode(flat_rows=True,**{flag:True})
    with pytest.raises(ValueError,match="source/prerequisite"):
        candidate.require_source(ROOT)
    monkeypatch.setitem(candidate.RAW,candidate.PROGRAM,(1,"a"*64))
    with pytest.raises(ValueError,match="registration"):candidate.read_metadata(ROOT)


@pytest.mark.parametrize("failure",["metadata","space"])
def test_preflight_refuses_before_launch(tmp_path,monkeypatch,failure):
    events=[]
    monkeypatch.setattr(campaign,"run_root",lambda tag:tmp_path)
    monkeypatch.setattr(campaign,"deploy_existing_workers",lambda *a:events.append("deploy"))
    def metadata(*args,**kw):
        assert kw=={"flat_rows":True}
        events.append("metadata")
        if failure=="metadata":raise ValueError("metadata refusal")
    monkeypatch.setattr(campaign,"metadata_preflight",metadata)
    monkeypatch.setattr(campaign.shutil,"disk_usage",lambda _:SimpleNamespace(free=(6<<30)-1))
    monkeypatch.setattr(campaign,"ssh",lambda *a,**k:pytest.fail("launched after refusal"))
    with pytest.raises(ValueError,match="metadata refusal|insufficient space"):campaign.campaign(TAG,PIN)
    assert events==["deploy","metadata"]


@pytest.mark.parametrize("failure",[None,"missing","duplicate","manifest","inventory","pin"])
def test_all_eight_metadata_before_runtime(tmp_path,monkeypatch,failure):
    _,captures=campaign.topology_bindings()
    pins=json.loads((ROOT/"docs/artifacts/prefill-window-layer6-host-admission-20260908.json").read_text())
    rows=[["ROLLED_METADATA_OK",c["hostname"],pins["expected_manifest_sha256"],pins["source_inventory_sha256"],PIN] for c in captures]
    if failure=="missing":rows.pop()
    elif failure=="duplicate":rows[-1]=rows[0]
    elif failure:rows[0][{"manifest":2,"inventory":3,"pin":4}[failure]]="a"*64
    def ssh(command,*,output,timeout):
        assert "ws32_flat_rows_compile import read_metadata" in command
        assert "JAX_PLATFORMS=cpu" in command and "probe_ws32" not in command
        output.write_text("\n".join(" ".join(row) for row in rows)+"\n")
    monkeypatch.setattr(campaign,"ssh",ssh)
    if failure:
        with pytest.raises(ValueError,match="eight-host"):campaign.metadata_preflight(tmp_path,PIN,flat_rows=True)
    else:campaign.metadata_preflight(tmp_path,PIN,flat_rows=True)
