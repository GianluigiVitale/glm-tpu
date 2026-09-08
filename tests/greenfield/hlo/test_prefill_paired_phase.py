"""Candidate plumbing and preregistration; CPU evidence, no TPU speed claim."""

import ast
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.greenfield import prefill_paired_sort_admission as paired
from scripts.greenfield import prefill_completed_window_admission as original
from scripts.greenfield import prefill_completed_window_assembly as assembly
from scripts.greenfield import prefill_phase_variant as modes
from scripts.greenfield import prefill_window_acquisition as acquisition
from tests.greenfield.hlo.test_prefill_window_admission import memory_inputs


def test_actual_production_cross_lowering_preregistration():
    # Reuse the exact production abstract fixture, not model payloads or TPU.
    source = Path("tests/greenfield/hlo/test_prefill_completed_window.py").read_text()
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef)
        and n.name == "test_completed_window_production_shapes_without_payloads"
    )
    fixture = ast.literal_eval(fn.body[0].value).split("programs=prepare_programs", 1)[
        0
    ]
    code = (
        fixture
        + r"""
from unittest.mock import patch
from hashlib import sha256
from jax._src import tpu_custom_call
from scripts.greenfield import prefill_completed_window_admission as old
from scripts.greenfield import prefill_paired_sort_admission as paired
pins=old.registered_programs()
with patch.object(tpu_custom_call,'get_ir_version',return_value=None):
 for flag in (False,True):
  programs=prepare_programs(mesh=mesh,config=config,weights=w,completed_window=True,paired_position_sort=flag)
  for name,fn,args in programs:
   text=str(fn.trace(*args).lower(lowering_platforms=('tpu',)).compiler_ir(dialect='stablehlo'))
   expected=paired.PREFIX_SHA if flag and name=='prefix' else pins[name]['stablehlo_sha256']
   assert sha256(text.encode()).hexdigest()==expected,(flag,name)
   if flag and name=='prefix':assert len(text.encode())==paired.PREFIX_BYTES
assert jax.default_backend()=='cpu'
print('RAW_PREREGISTRATION_PASS')
"""
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RAW_PREREGISTRATION_PASS" in result.stdout


@pytest.mark.parametrize("candidate", modes.variants())
def test_distinct_tag_protocol_profile(candidate):
    tag = f"greenfield_fp8_{candidate.kernel}_l6_test"
    record = dict(
        kernel=candidate.kernel,
        protocol=candidate.protocol,
        profile=candidate.admission.PROFILE,
    )
    assert modes.for_tag(tag) == modes.for_record(record) == candidate
    assert acquisition.is_phase_baseline_tag(tag) and acquisition.is_window_tag(tag)
    for field in record:
        broken = dict(record, **{field: "wrong"})
        with pytest.raises(ValueError):
            modes.for_record(broken)


def test_candidate_uses_actual_nine_allocations_without_changing_baseline():
    census, _ = memory_inputs()
    analyses = {
        n: deepcopy(p["compiled_memory"])
        for n, p in original.registered_programs().items()
    }
    analyses.update(
        {n: {k: 0 for k in assembly.ALLOCATION_CAPS} for n in assembly.PROGRAMS}
    )
    before = paired.memory_budget(census, analyses, active_graph="prefix")
    analyses["prefix"]["temp_size_in_bytes"] += 4096
    analyses["prefix"]["generated_code_size_in_bytes"] += 4096
    after = paired.memory_budget(census, analyses, active_graph="prefix")
    assert (
        after["devices"][0]["estimated_peak_bytes"]
        == before["devices"][0]["estimated_peak_bytes"] + 8192
    )
    with pytest.raises(ValueError):
        assembly.memory_budget(census, analyses, active_graph="prefix")
    analyses.pop("assemble")
    with pytest.raises(ValueError):
        paired.memory_budget(census, analyses, active_graph="prefix")


@pytest.mark.parametrize(
    "key,value",
    [
        ("temp_size_in_bytes", (128 << 20) + 1),
        ("generated_code_size_in_bytes", (32 << 20) + 1),
        ("alias_size_in_bytes", 1),
        ("alias_size_in_bytes", False),
        ("output_size_in_bytes", 0),
    ],
)
def test_candidate_memory_caps_and_signature_refuse(key, value):
    pin = original.registered_programs()["prefix"]
    memory = dict(pin["compiled_memory"], **{key: value})
    with pytest.raises(ValueError):
        paired._validate_memory("prefix", memory, pin)


@pytest.mark.parametrize("site", ["assembly", "sampler"])
@pytest.mark.parametrize("peer_only", [False, True])
def test_variant_failures_vote_before_helper_or_model_dispatch(
    tmp_path, monkeypatch, site, peer_only
):
    from scripts.greenfield import prefill_phase_baseline as phase
    from scripts.greenfield import prefill_phase_originals as originals

    value = modes.variants()[1]
    record = dict(
        protocol=value.protocol,
        profile=value.admission.PROFILE,
        compile_only=False,
        code_hash="a" * 40,
        launch_rank=0,
        performance_claim=False,
        reference_scope=phase.SCOPE,
        independent_full_layer_admission=False,
    )
    journal = phase.PhaseJournal(tmp_path / "compile_journal.jsonl", dict(record))
    record["programs"] = {n: {} for n in original.PROGRAMS}
    if not peer_only:
        record["profile"] = "wrong"
    votes = []

    def consensus(ok):
        votes.append(ok)
        return False

    monkeypatch.setattr(assembly, "prepare_programs", lambda mesh: ())
    monkeypatch.setattr(originals, "load_capsule", lambda: {})
    monkeypatch.setattr(originals, "bind_originals", lambda *a: {})
    try:
        with pytest.raises((ValueError, RuntimeError)):
            if site == "assembly":
                assembly.compile_programs(
                    mesh=None,
                    root=tmp_path,
                    record=record,
                    journal=journal,
                    consensus=consensus,
                    compiler=lambda *a, **k: pytest.fail("no compile after refusal"),
                )
            else:
                calls = phase.CompactPhaseCalls(
                    root=tmp_path,
                    record=record,
                    journal=journal,
                    consensus=consensus,
                    local_slots={},
                    budgeter=value.budgeter,
                )
                calls.programs = {
                    n: None for n in (*original.PROGRAMS, *assembly.PROGRAMS)
                }
                phase.run_competitive(calls, weights=None, wk=None, mesh=None, specs=())
        assert votes == [peer_only]
    finally:
        journal.close()
