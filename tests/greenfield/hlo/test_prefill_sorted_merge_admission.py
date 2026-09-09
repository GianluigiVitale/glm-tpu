"""Candidate registration and control comparison, CPU/host only."""

from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.greenfield import prefill_sorted_merge_admission as candidate
from scripts.greenfield import prefill_budget_probe as probe
from scripts.greenfield import prefill_budget_worker as worker
from scripts.greenfield import prefill_budget_evidence as evidence
from scripts.greenfield import probe_ws32_prefill_budget as entry
from scripts.greenfield import ws32_prefill_budget_campaign as campaign


def test_original_and_candidate_raw_tpu_target_registrations():
    code = r"""
from hashlib import sha256
import jax,numpy as np
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from scripts.greenfield import prefill_budget_probe as b
from scripts.greenfield import prefill_sorted_merge_admission as a
assert jax.default_backend()=='cpu'
mesh=Mesh(np.asarray(jax.devices()).reshape(8,4),('expert','feature'))
specs=(P(),P('expert',None),P(),P('expert'),P())
types=(np.float32,jax.numpy.bfloat16,np.float32,np.int32,np.int32)
for capacity,prompt in b.CAPACITIES:
 shapes=((32,32,128),(capacity,128),(32,32),(capacity,),(32,))
 args=tuple(jax.ShapeDtypeStruct(s,d,sharding=NamedSharding(mesh,p)) for s,d,p in zip(shapes,types,specs))
 for flag in (False,True):
  fn=b.build_dsa_program(mesh,capacity=capacity,sorted_local_merge=flag)
  text=str(fn.trace(*args).lower(lowering_platforms=('tpu',)).compiler_ir('stablehlo'))
  pins=a.CANDIDATE_SHA if flag else a.ORIGINAL_SHA
  assert sha256(text.encode()).hexdigest()==pins[f'dsa_c{capacity}'],(flag,capacity)
print('SORTED_MERGE_RAW_REGISTRATION_PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=dict(
            os.environ,
            JAX_PLATFORMS="cpu",
            XLA_FLAGS="--xla_force_host_platform_device_count=32",
        ),
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SORTED_MERGE_RAW_REGISTRATION_PASS" in result.stdout


def test_candidate_rejects_original_or_mutated_graph_before_structure():
    from tests.greenfield.hlo.test_prefill_budget_worker import memory, hlo

    for text in ("original", "changed arithmetic", ""):
        with pytest.raises(ValueError, match="preregistered StableHLO"):
            candidate.inspect_program(worker.PROGRAMS[0], text, hlo(), memory())


def test_registration_bytes_and_trusted_launch_identity(tmp_path, monkeypatch):
    receipt = candidate.registration()
    assert receipt["sorted_local_merge"] and not receipt["overhead_remeasured"]
    tag = f"greenfield_fp8_{candidate.KERNEL}_fixture"
    assert entry.kernel_for_tag(tag) == candidate.KERNEL
    protocol, profile, journal, inspect = worker.contract(True)
    assert (protocol, profile, journal, inspect) == (
        candidate.PROTOCOL,
        candidate.PROFILE,
        candidate.SortedMergeJournal,
        candidate.inspect_program,
    )
    with pytest.raises(ValueError):
        worker.contract(1)
    with pytest.raises(ValueError):
        entry.kernel_for_tag("greenfield_fp8_unregistered_fixture")
    with pytest.raises(ValueError, match="journal identity"):
        journal(
            tmp_path / "wrong.jsonl",
            dict(protocol=probe.PROTOCOL, profile=worker.PROFILE, compile_only=False),
        )
    for name in candidate.BINDINGS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((candidate.REPO / name).read_bytes())
    monkeypatch.setattr(candidate, "REPO", tmp_path)
    assert candidate.registration() == receipt
    path = tmp_path / next(iter(candidate.BINDINGS))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="registration bytes"):
        candidate.registration()


def test_baseline_aligned_samples_and_failed_speed_decision_remain_evidence():
    baseline = candidate.baseline_wall()
    for factor, passed in ((0.8, True), (1.2, False)):
        actual = deepcopy(baseline)
        for value in actual["cases"].values():
            for key in ("p50_seconds", "p99_seconds"):
                value[key] *= factor
            value["samples_seconds"] = [v * factor for v in value["samples_seconds"]]
        result = candidate.compare(actual)
        assert result["candidate_selection_passed"] is passed
        assert result["model_performance_claim"] is False
        assert len(result["cases"]) == 6
    # Endpoint wins do not excuse a midpoint regression.
    actual = deepcopy(baseline)
    for case in probe.cases():
        actual["cases"][case.name]["p50_seconds"] *= (
            0.8 if case.last_valid_length == case.prompt_length else 1.06
        )
    assert not candidate.compare(actual)["candidate_selection_passed"]


def test_candidate_overhead_and_missing_target_bindings_refuse_before_journal(tmp_path):
    record = dict(
        protocol=candidate.PROTOCOL,
        profile=candidate.PROFILE,
        compile_only=False,
        current_phase="budget/dsa_complete",
        candidate_registration=candidate.registration(),
    )
    for mutation in ("overhead", "targets", "required"):
        bad = deepcopy(record)
        if mutation == "overhead":
            bad["budget_overhead"] = {}
        if mutation == "targets":
            bad["candidate_registration"]["sorted_local_merge"] = False
        with pytest.raises(ValueError):
            evidence.validate_files(
                tmp_path,
                bad,
                slots={i: i for i in range(4)},
                sorted_local_merge=True,
                require_overhead=mutation == "required",
            )


def test_candidate_launch_uses_existing_bounded_distributed_path():
    tag = f"greenfield_fp8_{candidate.KERNEL}_fixture"
    command = campaign.launch_command(tag, "a" * 40, "10.0.0.1:8476")
    assert (
        tag in command
        and "900s" in command
        and "probe_ws32_prefill_budget.py" in command
    )
    assert "TPU_VISIBLE_DEVICES" not in command
    wrapper = Path("scripts/greenfield/run_fp8_matmul_microbench.sh").read_text()
    assert "[[ $KERNEL != ws32_prefill_sorted_merge ]] || BUDGET_WORKFLOW=1" in wrapper
    assert wrapper.index("ws32_prefill_budget_campaign campaign") < wrapper.index(
        "TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1"
    )
