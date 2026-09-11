"""Long compiler integration: CPU fixtures and real production abstract lowering."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.greenfield import ws32_delivery_compile as candidate
from scripts.greenfield import ws32_prefill_budget_campaign as campaign
from scripts.greenfield import ws32_rolled_prefill_worker as worker
from scripts.greenfield import ws32_rolled_prefill_evidence as evidence
from tests.greenfield.hlo import test_ws32_rolled_prefill_worker as checks
from tests.greenfield.hlo.test_ws32_rolled_prefill_worker import lifecycle

ROOT = Path(__file__).resolve().parents[3]
TAG = "greenfield_fp8_" + candidate.KERNEL + "_fixture"
PIN = "c" * 40


@pytest.mark.parametrize("lifecycle", ["delivery"], indirect=True)
@pytest.mark.parametrize("case", ["complete", "memory", "identity", "finalize", "first", "last"])
def test_actual_worker_refusal_and_preservation(lifecycle, monkeypatch, case):
    if case == "complete":
        checks.test_both_actual_writers_and_journal_complete_without_dispatch(lifecycle)
    elif case == "memory":
        checks.test_memory_refusal_happens_after_both_originals_are_preserved(lifecycle)
    elif case == "identity":
        checks.test_invalid_identity_refuses_before_metadata_or_compilation(lifecycle)
    elif case == "finalize":
        checks.test_primary_failure_survives_finalize_failure(lifecycle, monkeypatch)
    else:
        checks.test_compile_failure_retains_partial_originals_and_closes(lifecycle, 0 if case == "first" else -1)


@pytest.mark.parametrize("lifecycle", ["delivery"], indirect=True)
@pytest.mark.parametrize("phase", range(7))
def test_each_peer_failure_refuses(lifecycle, phase):
    checks.test_peer_refusal_never_advances_or_leaves_success(lifecycle, phase)


@pytest.mark.parametrize("lifecycle", ["delivery"], indirect=True)
@pytest.mark.parametrize("metadata_failure", [False, True])
def test_actual_cli_and_original_reader(lifecycle, monkeypatch, metadata_failure):
    checks.test_actual_probe_selects_compile_only_before_runtime(lifecycle, monkeypatch, metadata_failure)


def test_fixed_mode_storage_time_and_sampling():
    mode = worker.compile_mode(delivery=True)
    assert mode.programs == candidate.PROGRAMS
    assert campaign.program_names(TAG) == candidate.PROGRAMS
    assert len(campaign.evidence_files(TAG)) == 9
    assert campaign.evidence_files(TAG) == evidence.files(delivery=True)
    assert campaign.rank_byte_limit(TAG) == 384 << 20
    assert "timeout --kill-after=30s 1800s" in campaign.launch_command(TAG, PIN, "10.0.0.1:8476")
    for invalid in (None, 1, "true"):
        with pytest.raises(ValueError, match="static bool"):
            worker.compile_mode(delivery=invalid)
    for flag in ("history", "full_canonical", "canonical_dense"):
        with pytest.raises(ValueError, match="exclusive"):
            worker.compile_mode(delivery=True, **{flag: True})
    wrapper = (ROOT / "scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    assert f"[[ $KERNEL != {candidate.KERNEL} ]] || ROLLED_COMPILE=1" in wrapper


@pytest.mark.parametrize("failure", [None, "missing", "duplicate", "manifest", "inventory", "pin"])
def test_all_host_metadata_preflight(tmp_path, monkeypatch, failure):
    _, captures = campaign.topology_bindings()
    pins = json.loads((ROOT / "docs/artifacts/prefill-window-layer6-host-admission-20260908.json").read_text())
    rows = [["ROLLED_METADATA_OK", c["hostname"], pins["expected_manifest_sha256"],
             pins["source_inventory_sha256"], PIN] for c in captures]
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[-1] = rows[0]
    elif failure:
        rows[0][{"manifest": 2, "inventory": 3, "pin": 4}[failure]] = "a" * 64

    def ssh(command, *, output, timeout):
        assert "ws32_delivery_compile import read_metadata" in command
        assert "JAX_PLATFORMS=cpu" in command and "probe_ws32" not in command
        output.write_text("\n".join(" ".join(row) for row in rows) + "\n")

    monkeypatch.setattr(campaign, "ssh", ssh)
    if failure:
        with pytest.raises(ValueError, match="eight-host"):
            campaign.metadata_preflight(tmp_path, PIN, delivery=True)
    else:
        campaign.metadata_preflight(tmp_path, PIN, delivery=True)


@pytest.mark.parametrize("failure", ["metadata", "space"])
def test_preflight_refusal_blocks_launch(tmp_path, monkeypatch, failure):
    events = []
    monkeypatch.setattr(campaign, "run_root", lambda tag: tmp_path)
    monkeypatch.setattr(campaign, "deploy_existing_workers", lambda *a: events.append("deploy"))
    def metadata(*args, **kwargs):
        assert kwargs == {"delivery": True}
        events.append("metadata")
        if failure == "metadata":
            raise ValueError("metadata refusal")
    monkeypatch.setattr(campaign, "metadata_preflight", metadata)
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda _: SimpleNamespace(free=(6 << 30) - 1))
    monkeypatch.setattr(campaign, "ssh", lambda *a, **k: pytest.fail("launched after preflight refusal"))
    with pytest.raises(ValueError, match="metadata refusal|insufficient space"):
        campaign.campaign(TAG, PIN)
    assert events == ["deploy", "metadata"]


@pytest.mark.parametrize("lifecycle", ["delivery"], indirect=True)
def test_preserved_originals_and_no_dispatch_identity(lifecycle):
    case = lifecycle
    case.run()
    for field, value in (("compile_only", False), ("weights_loaded", True),
            ("model_executable_calls", False), ("model_executable_calls", 1),
            ("numerical_claim", True), ("performance_claim", True), ("profile", "old")):
        bad = deepcopy(case.record)
        bad[field] = value
        with pytest.raises(ValueError, match="identity"):
            candidate.validate_preserved_pair(case.root, bad, repo=ROOT)
    for name in candidate.PROGRAMS:
        bad = deepcopy(case.record)
        del bad["programs"][name]
        with pytest.raises(ValueError, match="inventory"):
            candidate.validate_preserved_pair(case.root, bad, repo=ROOT)
        path = case.root / f"{name}.optimized_hlo.txt"
        original = path.read_bytes()
        path.write_bytes(original + b"modified")
        with pytest.raises(ValueError, match="identity"):
            candidate.validate_preserved_pair(case.root, case.record, repo=ROOT)
        path.write_bytes(original)


def test_frozen_source_drift_refuses(monkeypatch):
    old = Path.read_bytes
    name = next(iter(candidate.canonical.MODEL_SOURCE_OVERRIDES))
    monkeypatch.setattr(Path, "read_bytes", lambda p: old(p) + (b"changed" if p == ROOT / name else b""))
    with pytest.raises(ValueError, match="source/prerequisite"):
        candidate.read_metadata(ROOT)


def test_production_three_graph_lowering_without_payload_or_compile():
    source = r'''
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
import jax, numpy as np
from jax.sharding import Mesh
from jax._src.pallas.mosaic import tpu_info
from scripts.greenfield import ws32_delivery_compile as candidate
assert jax.default_backend() == 'cpu'
tpu_info.registry['cpu'] = lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V4, 1)
tpu_info.get_tpu_info.cache_clear()
mesh=Mesh(np.asarray(jax.devices(),object).reshape(8,4),('expert','feature'))
old=Path.open
def checked(path,*args,**kwargs):
    assert path.suffix not in ('.safetensors','.bin'),path
    if '/glm-ws32-runtime/' in str(path):
        assert path.name in ('manifest.json','SUCCESS'),path
    return old(path,*args,**kwargs)
with patch.object(Path,'open',checked), \
     patch('jax.device_put',side_effect=AssertionError('payload placement')), \
     patch('jax.stages.Compiled.__call__',side_effect=AssertionError('dispatch')), \
     patch('jax.stages.Lowered.compile',side_effect=AssertionError('compilation')):
    metadata=candidate.read_metadata(Path.cwd())
    prepared=candidate.prepare(mesh,metadata,repo=Path.cwd())
    assert len(metadata.manifest['tensor_schema']) == 2310
    assert set(prepared.programs) == set(prepared.inputs) == set(candidate.PROGRAMS)
    for name in candidate.PROGRAMS:
        ids,count,state,weights,wk,rope=prepared.inputs[name]
        capacity,rows=candidate.GEOMETRY[name]
        assert ids.shape == (rows,) and state.decoder.block_tables.shape == (1,capacity//512)
        assert rope.shape == (capacity,64) and len(weights.layers) == 78
        assert all(isinstance(x,jax.ShapeDtypeStruct) and x.sharding is not None
                   for x in jax.tree.leaves(prepared.inputs[name]))
        with patch('jax._src.tpu_custom_call.get_ir_version',return_value=None):
            raw=str(prepared.programs[name].execute.trace(*prepared.inputs[name]).lower(
                lowering_platforms=('tpu',)).compiler_ir('stablehlo')).encode()
        actual=(len(raw),sha256(raw).hexdigest())
        print(name,actual,flush=True)
        assert actual == candidate.RAW[name], (name,actual,candidate.RAW[name])
print('DELIVERY_THREE_RAW_PINS_PASS',flush=True)
'''
    flags = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=32").strip()
    result = subprocess.run([sys.executable, "-c", source], cwd=ROOT, capture_output=True,
                            text=True, timeout=360, env=dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS=flags))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DELIVERY_THREE_RAW_PINS_PASS" in result.stdout
