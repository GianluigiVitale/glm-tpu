"""Existing numerical continuation routing/WK/failure checks, no TPU execution."""

from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.greenfield import prefill_phase_baseline as phase
from scripts.greenfield import prefill_phase_originals as original
from scripts.greenfield import prefill_window_worker as worker
from scripts.greenfield import prefill_completed_window_admission as admission
from scripts.greenfield import prefill_completed_window_assembly as assembly
from tests.greenfield.hlo.test_prefill_window_worker import fake_memory


@pytest.mark.parametrize(
    "failure", [None, "originals", "wk", "sampling", "mixed_mode", "scope", "archive", "variant", "peer_variant"]
)
def test_existing_continuation_selects_compact_calls_and_two_wk(
    tmp_path, monkeypatch, failure
):
    import jax
    from scripts.greenfield import probe_ws32_prefill_layer as layer

    names = (*admission.PROGRAMS, *assembly.PROGRAMS)
    record = dict(
        protocol=phase.PROTOCOL,
        profile=admission.PROFILE,
        compile_only=False,
        code_hash="a" * 40,
        launch_rank=0,
        jax_process_index=3,
        reference_scope=phase.SCOPE,
        performance_claim=False,
        independent_full_layer_admission=False,
        programs={n: {} for n in names},
    )
    journal = phase.PhaseJournal(
        tmp_path / "compile_journal.jsonl",
        {
            k: record[k]
            for k in ("protocol", "profile", "compile_only", "code_hash", "launch_rank")
        },
    )
    if failure == "scope":
        record["reference_scope"] = "wrong"
    if failure == "variant":
        record["profile"] = "wrong"
    slots = {9: 0, 13: 1, 25: 2, 29: 3}
    monkeypatch.setattr(jax, "block_until_ready", lambda x: x)
    monkeypatch.setattr(jax, "local_devices", lambda: ())
    monkeypatch.setattr(
        worker, "capture_resident_buffers", lambda *a, **k: fake_memory()
    )
    monkeypatch.setattr(
        worker,
        "capture_identified_device_memory",
        lambda *a: [
            dict(
                device_id=r["device_id"],
                process_index=3,
                platform="tpu",
                **r["memory_stats"]
            )
            for r in fake_memory()["devices"]
        ],
    )
    monkeypatch.setattr(
        assembly, "memory_budget", lambda *a, **k: dict(estimate_fits=True)
    )
    monkeypatch.setattr(layer, "input_specs", lambda *a: ())
    monkeypatch.setattr(original, "load_capsule", lambda: {})
    if failure == "archive":

        def refuse_archive(*args, **kwargs):
            raise ValueError("archive refused")

        monkeypatch.setattr(phase.gzip, "compress", refuse_archive)

    def bind(*args):
        if failure == "originals":
            raise ValueError("originals refused")
        return dict(test_binding=True)

    monkeypatch.setattr(original, "bind_originals", bind)
    checks = []

    def check(*args, kind, **kwargs):
        checks.append(kind)
        if failure == "wk":
            raise ValueError("wk refused")

    monkeypatch.setattr(original, "check_observation", check)
    executions = []

    class Program:
        def __init__(self, name):
            self.name = name

        def memory_analysis(self):
            return SimpleNamespace(**{k: 0 for k in worker.MEMORY_FIELDS})

        def __call__(self, *args):
            assert self.name in ("wk_decode", "wk_promote")
            executions.append(self.name)
            dtype = worker.protocol.BF16 if self.name == "wk_decode" else np.float32
            return SimpleNamespace(
                addressable_shards=[
                    SimpleNamespace(
                        device=SimpleNamespace(id=d), data=np.ones((128, 6144), dtype)
                    )
                    for d in slots
                ]
            )

    sampled = []

    def sample(calls, **kwargs):
        assert isinstance(calls, phase.CompactPhaseCalls)
        assert not calls.record["call_evidence"] and len(calls.samples) == 2
        sampled.append(True)
        if failure == "sampling":
            raise ValueError("sampling refused")

    monkeypatch.setattr(phase, "run_competitive", sample)
    votes = []

    def consensus(ok):
        votes.append(ok)
        return False if failure == "peer_variant" and len(votes) == 1 else ok

    kwargs = dict(
        args=SimpleNamespace(output_dir=tmp_path),
        record=record,
        mesh=None,
        config=None,
        weights=SimpleNamespace(
            dsa=SimpleNamespace(wk_bits_local=None, wk_scale_local=None)
        ),
        local_slots=slots,
        consensus=consensus,
        compiled=tuple(Program(n) for n in names),
        journal=journal,
        phase_baseline=True,
        completed_numerical=failure == "mixed_mode",
    )
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            worker.execute_numerical(**kwargs)
    else:
        worker.execute_numerical(**kwargs)
        assert record["current_phase"] == "phase_numerical_complete"
    assert (
        record["compile_journal_sha256"]
        == sha256((tmp_path / "compile_journal.jsonl").read_bytes()).hexdigest()
    )
    if failure in ("originals", "mixed_mode", "scope", "variant", "peer_variant"):
        assert not executions and record["wk_executable_calls"] == 0
        if failure in ("variant", "peer_variant"):
            assert votes == [failure == "peer_variant", True]
    elif failure in ("wk", "archive"):
        assert executions == ["wk_decode"] and record["wk_executable_calls"] == 1
        assert (tmp_path / "wk_decode.npz").exists()
        if failure == "archive":
            assert not record["phase_call_index"]
            assert record["call_evidence"][0]["completed"]
    else:
        assert (
            executions == ["wk_decode", "wk_promote"]
            and record["wk_executable_calls"] == 2
        )
        assert checks == ["wk_decode"] * 4 + ["wk_promote"] * 4 and sampled
        assert (
            len(
                list(
                    phase.read_call_witnesses(
                        tmp_path / "phase_calls.jsonl.gz", record["phase_call_index"]
                    )
                )
            )
            == 2
        )


def test_phase_adapter_rejects_unprotected_continuation_before_fixture(tmp_path):
    calls = SimpleNamespace(
        root=tmp_path, record={}, phase=lambda name, action: action()
    )
    with pytest.raises(ValueError, match="distinct protected"):
        phase.run_competitive(calls, weights=None, wk=None, mesh=None, specs=())
