"""Protected capture-lifetime route; fixture compiler, no TPU claims."""
from types import SimpleNamespace
import json
import pytest
from scripts.greenfield import ws32_capture_barrier_compile as candidate
from scripts.greenfield import ws32_prefill_budget_campaign as campaign
from scripts.greenfield import ws32_rolled_prefill_worker as worker
from scripts.greenfield import ws32_rolled_prefill_evidence as evidence
from tests.greenfield.hlo import test_ws32_rolled_prefill_worker as checks
from tests.greenfield.hlo import test_ws32_owned_state_compile as owned_checks
from tests.greenfield.hlo.test_ws32_rolled_prefill_worker import lifecycle
ROOT = checks.ROOT
TAG = "greenfield_fp8_" + candidate.KERNEL + "_fixture"
PIN = "c" * 40

@pytest.mark.parametrize("lifecycle",["capture_barrier"],indirect=True)
@pytest.mark.parametrize("case",["complete","memory","identity","finalize","compile","originals"])
def test_worker_and_original_reader(lifecycle,monkeypatch,case):
    if case=="originals":
        monkeypatch.setattr(owned_checks,"candidate",candidate)
        owned_checks.test_preserved_originals_scope_and_allocation_refusals(lifecycle)
    else:
        owned_checks.test_actual_worker_refusal_and_preservation(lifecycle,monkeypatch,case)


@pytest.mark.parametrize("lifecycle",["capture_barrier"],indirect=True)
@pytest.mark.parametrize("phase",range(5))
def test_each_peer_refusal(lifecycle,phase):
    checks.test_peer_refusal_never_advances_or_leaves_success(lifecycle,phase)


@pytest.mark.parametrize("lifecycle",["capture_barrier"],indirect=True)
@pytest.mark.parametrize("metadata_failure",[False,True])
def test_actual_cli_before_runtime(lifecycle,monkeypatch,metadata_failure):
    checks.test_actual_probe_selects_compile_only_before_runtime(lifecycle,monkeypatch,metadata_failure)


def test_fixed_mode_and_raw_pin(monkeypatch):
    mode=worker.compile_mode(capture_barrier=True)
    assert mode.programs==candidate.PROGRAMS==("prefill_256k_capture_barrier",)
    assert campaign.program_names(TAG)==mode.programs
    assert len(campaign.evidence_files(TAG))==5
    assert campaign.evidence_files(TAG)==evidence.files(capture_barrier=True)
    assert campaign.rank_byte_limit(TAG)==192<<20
    assert "timeout --kill-after=30s 900s" in campaign.launch_command(TAG,PIN,"10.0.0.1:8476")
    for invalid in (None,1,"true"):
        with pytest.raises(ValueError,match="static bool"):worker.compile_mode(capture_barrier=invalid)
    for flag in ("history","full_canonical","canonical_dense","delivery","owned_state","pending_rows","flat_rows"):
        with pytest.raises(ValueError,match="exclusive"):worker.compile_mode(capture_barrier=True,**{flag:True})
    candidate.require_source(ROOT)
    monkeypatch.setitem(candidate.RAW,candidate.PROGRAM,(1,"a"*64))
    with pytest.raises(ValueError,match="registration"):candidate.read_metadata(ROOT)


@pytest.mark.parametrize("failure",["metadata","space"])
def test_preflight_refuses_before_launch(tmp_path,monkeypatch,failure):
    events=[]
    monkeypatch.setattr(campaign,"run_root",lambda tag:tmp_path)
    monkeypatch.setattr(campaign,"deploy_existing_workers",lambda *a:events.append("deploy"))
    def metadata(*args,**kw):
        assert kw=={"capture_barrier":True}
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
        assert "ws32_capture_barrier_compile import read_metadata" in command
        assert "JAX_PLATFORMS=cpu" in command and "probe_ws32" not in command
        output.write_text("\n".join(" ".join(row) for row in rows)+"\n")
    monkeypatch.setattr(campaign,"ssh",ssh)
    if failure:
        with pytest.raises(ValueError,match="eight-host"):campaign.metadata_preflight(tmp_path,PIN,capture_barrier=True)
    else:campaign.metadata_preflight(tmp_path,PIN,capture_barrier=True)
