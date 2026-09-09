"""One changed graph through the original compiler lifecycle; no TPU."""

from copy import deepcopy

import pytest

from scripts.greenfield import ws32_rolled_prefill_worker as worker
from scripts.greenfield import ws32_prefill_budget_campaign as campaign
from tests.greenfield.hlo.test_ws32_rolled_prefill_worker import (
    lifecycle,
    test_actual_probe_selects_compile_only_before_runtime as run_entry,
)


@pytest.mark.parametrize("lifecycle", [True], indirect=True)
def test_one_graph_preserved_without_wk_or_model_call(lifecycle):
    case = lifecycle
    case.run()
    assert case.events == ["dense01_canonical"]
    assert case.record["compiler_acquisition_complete"] is True
    assert case.record["model_executable_calls"] == 0
    assert case.votes == [True] * 5
    assert [r["stage"] for r in case.journal()] == [
        "identity",
        "lower_compile_started",
        "compiled",
        "raw_written",
        "inspected",
        "preserved_pair_verified",
    ]


@pytest.mark.parametrize("lifecycle", [True], indirect=True)
@pytest.mark.parametrize("failure", ["compile", "memory"])
def test_failure_preserves_available_originals(lifecycle, failure):
    case = lifecycle
    case.controls["fail_compile"] = (
        "dense01_canonical" if failure == "compile" else None
    )
    case.controls["refuse_memory"] = failure == "memory"
    with pytest.raises((RuntimeError, ValueError)):
        case.run()
    assert case.record["status"] == "FAILED"
    assert case.record["compiler_acquisition_complete"] is False
    assert case.record["model_executable_calls"] == 0
    assert (case.root / "dense01_canonical.stablehlo.mlir").exists()
    if failure == "memory":
        assert (case.root / "dense01_canonical.optimized_hlo.txt").exists()
    case.journal()


@pytest.mark.parametrize("lifecycle", [True], indirect=True)
@pytest.mark.parametrize("phase", range(5))
def test_peer_refusal_closes_without_later_compile_or_success(lifecycle, phase):
    case = lifecycle
    votes = []

    def vote(ok):
        votes.append(ok)
        return ok and len(votes) != phase + 1

    with pytest.raises(RuntimeError, match="peer failed"):
        case.run(vote)
    assert case.record["status"] == "FAILED"
    assert not case.record["compiler_acquisition_complete"]
    assert len(case.events) == min(phase, 1)
    case.journal()


@pytest.mark.parametrize("lifecycle", [True], indirect=True)
@pytest.mark.parametrize("missing", [False, True])
def test_actual_cli_source_before_runtime_and_independent_original_replay(
    lifecycle, monkeypatch, missing
):
    run_entry(lifecycle, monkeypatch, missing)


def test_exact_one_graph_small_publication_inventory():
    tag = "greenfield_fp8_ws32_dense_canonical_compile_fixture"
    assert campaign.is_compile(tag) and campaign.is_dense_compile(tag)
    assert campaign.program_names(tag) == ("dense01_canonical",)
    assert len(campaign.evidence_files(tag)) == 5
    assert campaign.rank_byte_limit(tag) == 64 << 20
    assert not any("wk_" in p for p in campaign.evidence_files(tag))
    assert "900s" in campaign.launch_command(tag, "c" * 40, "10.0.0.1:8476")
    for invalid in (None, 1, "true"):
        with pytest.raises(ValueError):
            worker.compile_mode(invalid)


@pytest.mark.parametrize("lifecycle", [True], indirect=True)
def test_mixed_mode_and_graph_mutations_refuse(lifecycle):
    case = lifecycle
    case.run()
    from scripts.greenfield.ws32_rolled_prefill_worker import journal_identity

    with pytest.raises(ValueError):
        journal_identity(case.record)
    for field in ("weights_loaded", "model_executable_calls", "numerical_claim"):
        bad = deepcopy(case.record)
        bad[field] = True if field != "model_executable_calls" else 1
        with pytest.raises(ValueError):
            journal_identity(bad, canonical_dense=True)
    bad = deepcopy(case.record)
    bad["programs"]["wk_decode"] = bad["programs"]["dense01_canonical"]
    with pytest.raises(ValueError, match="only its changed graph"):
        case.mode.preparation.validate_preserved_pair(
            case.root, bad, repo=campaign.REPO
        )
    path = case.root / "dense01_canonical.optimized_hlo.txt"
    path.write_text(path.read_text() + "tampered")
    with pytest.raises(ValueError, match="identity"):
        case.mode.preparation.validate_preserved_pair(
            case.root, case.record, repo=campaign.REPO
        )
