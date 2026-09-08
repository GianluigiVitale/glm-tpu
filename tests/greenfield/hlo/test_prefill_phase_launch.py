"""Protected phase integration CPU boundaries; no hardware/performance claim."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.greenfield import prefill_phase_baseline as phase
from scripts.greenfield import prefill_phase_originals as originals
from scripts.greenfield import prefill_phase_evidence as evidence
from scripts.greenfield import prefill_window_acquisition as acquisition
from scripts.greenfield import ws32_prefill_layer_campaign as campaign


@pytest.mark.parametrize(
    "failure", [None, "missing", "duplicate", "symlink", "destination", "oversize"]
)
def test_trace_finalization_is_one_original_no_copy(tmp_path, monkeypatch, failure):
    trace = tmp_path / "phase_trace/plugins/profile/test"
    trace.mkdir(parents=True)
    path = trace / "host.xplane.pb"
    path.write_bytes(b"synthetic trace")
    inode = path.stat().st_ino
    if failure == "missing":
        path.unlink()
    elif failure == "duplicate":
        (trace / "other.xplane.pb").write_bytes(b"second")
    elif failure == "symlink":
        path.rename(trace / "other")
        path.symlink_to(trace / "other")
    elif failure == "destination":
        (tmp_path / "phase.xplane.pb").write_bytes(b"existing")
    elif failure == "oversize":
        monkeypatch.setattr(phase, "MAX_TRACE_BYTES", 1)
    if failure:
        with pytest.raises(ValueError):
            phase.finalize_trace(tmp_path)
    else:
        result = phase.finalize_trace(tmp_path)
        final = tmp_path / "phase.xplane.pb"
        assert final.stat().st_ino == inode and not path.exists()
        assert result["sha256"] == sha256(final.read_bytes()).hexdigest()
        assert list(tmp_path.rglob("*.xplane.pb")) == [final]


@pytest.mark.parametrize("paired", [False, True])
def test_actual_nine_compiler_wk_sampler_and_trace_finalization(tmp_path, monkeypatch, paired):
    import jax
    from scripts.greenfield import prefill_completed_window_protocol as completed
    from tests.greenfield.hlo.test_prefill_completed_window_worker import (
        run_cases,
        continue_from_acquisition,
    )

    # Arithmetic/byte-verifier fixtures here; original byte replay executes on
    # actual DB594 archives in test_prefill_phase_evidence. Do not conflate them.
    monkeypatch.setattr(originals, "load_capsule", lambda: {})
    monkeypatch.setattr(originals, "bind_originals", lambda *a: {})
    monkeypatch.setattr(originals, "check_observation", lambda *a, **kw: {})
    seen = []

    class Verifier:
        def __init__(self, calls, *args):
            self.calls = calls

        def component(self, kind, result, fields):
            seen.append(kind)
            assert set(completed.observe(result, fields)) == set(self.calls.local_slots)

        def assembly(self, *args):
            seen.extend(("actual", "control"))

        def finish(self, n):
            assert n == 15 and len(seen) == 165

    monkeypatch.setattr(originals, "OriginalVerifier", Verifier)

    def trace_start(path, *, profiler_options):
        defaults = jax.profiler.ProfileOptions()
        assert profiler_options.python_tracer_level == 0
        assert profiler_options.host_tracer_level == defaults.host_tracer_level
        Path(path).mkdir()
        (Path(path) / "fixture.xplane.pb").write_bytes(b"CPU fixture, not protobuf")

    monkeypatch.setattr(jax.profiler, "start_trace", trace_start)
    monkeypatch.setattr(jax.profiler, "stop_trace", lambda: None)
    with run_cases(tmp_path, create_journal=False) as (calls, sequence):
        from scripts.greenfield.prefill_phase_variant import variants
        variant = variants()[int(paired)]
        calls.record.update(protocol=variant.protocol, profile=variant.admission.PROFILE)
        if paired:
            # Fixture compiler below returns archived old HLO, not the new TPU
            # graph. Production preregistration is checked separately on CPU.
            from scripts.greenfield import prefill_paired_sort_admission as pa
            from tests.greenfield.hlo.test_prefill_completed_window_admission import ORIGINAL
            raw = (ORIGINAL / "prefix.stablehlo.mlir").read_bytes()
            monkeypatch.setattr(pa, "PREFIX_SHA", sha256(raw).hexdigest())
            monkeypatch.setattr(pa, "PREFIX_BYTES", len(raw))
        continue_from_acquisition(tmp_path, calls, sequence, phase_baseline=True)
        r = calls.record
        assert len(sequence) == 287 and r["model_executable_calls"] == 135
        assert r["assembly_executable_calls"] == 150 and r["wk_executable_calls"] == 2
        assert r["protocol"] == variant.protocol and r["reference_scope"] == phase.SCOPE
        journal = [
            json.loads(line)
            for line in (tmp_path / "compile_journal.jsonl").read_text().splitlines()
        ]
        assert [r["stage"] for r in journal] == evidence.expected_stages()
        evidence.validate_samples(
            tmp_path, json.loads(json.dumps(r)), calls.local_slots
        )
        assert r["phase_trace_file"]["bytes"] > 0


@pytest.mark.parametrize("change", [None, "oversize", "extra", "disk"])
def test_receipt_budget_precedes_any_payload_download(tmp_path, monkeypatch, change):
    tag = f"greenfield_fp8_{phase.KERNEL}_l6_fixture"
    blobs, downloads = {}, []
    for rank in range(8):
        prefix = f"results/{tag}/workers/rank{rank}/"
        receipts = [
            dict(name=prefix + n, size=1024)
            for n in campaign.evidence_files(6, phase_baseline=True)
        ]
        if rank == 7 and change == "oversize":
            receipts[-1]["size"] = phase.MAX_TRACE_BYTES + 1
        if rank == 7 and change == "extra":
            receipts.append(dict(name=prefix + "phase_failure.npz", size=1))
        raw = json.dumps(receipts).encode()
        name = prefix + "worker_receipts.json"

        def download(*, if_generation_match, raw=raw, name=name):
            assert if_generation_match == 5
            downloads.append(name)
            return raw

        blobs[name] = SimpleNamespace(
            name=name, size=len(raw), generation=5, download_as_bytes=download
        )
    bucket = SimpleNamespace(
        get_blob=lambda name: blobs[name],
        blob=lambda *a, **kw: pytest.fail("payload must not be requested"),
    )
    if change == "disk":
        import shutil

        monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=0))
    if change:
        with pytest.raises(ValueError):
            campaign.phase_receipt_preflight(bucket, tag, tmp_path)
    else:
        assert len(campaign.phase_receipt_preflight(bucket, tag, tmp_path)) == 8
    assert all(n.endswith("worker_receipts.json") for n in downloads)


@pytest.mark.parametrize("oversized_ledger", [False, True])
def test_partial_publication_caps_cumulative_bytes_and_retains_originals(
    tmp_path, monkeypatch, oversized_ledger
):
    from google.cloud import storage
    from scripts.greenfield import collect_ws32_worker_evidence as publication

    tag = f"greenfield_fp8_{phase.KERNEL}_l6_fixture"
    root = tmp_path / "rank0"
    root.mkdir()
    (root / "runner.json").write_text(json.dumps(dict(status="FAILED")))
    trace = root / "phase_trace"
    trace.mkdir()
    for i in range(3):
        (trace / f"trace{i}.xplane.pb").write_bytes(b"x" * 40)
    (root / "phase_failure.npz").write_bytes(b"failure")
    monkeypatch.setattr(phase, "MAX_RANK_BYTES", (1 << 20) + 110)
    monkeypatch.setattr(campaign, "run_root", lambda tag: tmp_path)
    monkeypatch.setattr(
        storage, "Client", lambda: SimpleNamespace(bucket=lambda n: None)
    )
    uploaded = []

    def publish(bucket, name, path, digest, *, compressed):
        uploaded.append((name, digest["size"]))
        return dict(
            name=name,
            size=digest["size"],
            fixture_metadata="x" * (600000 if oversized_ledger else 0),
        )

    monkeypatch.setattr(publication, "publish_exact", publish)
    if oversized_ledger:
        with pytest.raises(ValueError, match="ledger oversized"):
            campaign.publish_rank(tag, 0)
        assert not any(n.endswith("worker_receipts.json") for n, _ in uploaded)
        return
    campaign.publish_rank(tag, 0)
    assert sum(n for _, n in uploaded) < phase.MAX_RANK_BYTES
    assert len([n for n, _ in uploaded if "phase_failed_trace" in n]) == 2
    notice = json.loads((root / "phase_publication_omissions.json").read_text())
    assert notice["originals_retained_locally"] is True and len(notice["omitted"]) == 1
    assert len(list(trace.glob("*.pb"))) == 3
