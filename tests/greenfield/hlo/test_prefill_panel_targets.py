"""Bind panel targets before dispatch and in actual graph-report replay."""

import pytest

from scripts.greenfield import prefill_panel_admission as panel


def test_panel_registration_original_bytes_and_fixed_decision():
    result = panel.target_registration()
    assert result["control_db_run_id"] == 596
    assert result["required_wide_suffix_p50_ratio_max"] == 0.9
    assert result["required_wide_partial_phase_sum_p50_ratio_max"] == 1.0
    assert result["model_performance_claim"] is False
    assert len(panel.registered_programs()) == 5


@pytest.mark.parametrize("mutation", ["changed", "symlink", "missing"])
def test_bad_target_refuses_existing_prejax_registration(
    tmp_path, monkeypatch, mutation
):
    for name in panel.TARGET_BINDINGS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((panel.REPO / name).read_bytes())
    path = tmp_path / next(iter(panel.TARGET_BINDINGS))
    if mutation == "changed":
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "symlink":
        path.unlink()
        path.symlink_to(panel.REPO / next(iter(panel.TARGET_BINDINGS)))
    else:
        path.unlink()
    monkeypatch.setattr(panel, "REPO", tmp_path)
    with pytest.raises((ValueError, FileNotFoundError)):
        panel.registered_programs()


def test_panel_target_binding_is_in_recomputed_report(monkeypatch):
    from tests.greenfield.hlo.test_prefill_completed_window_admission import ORIGINAL
    from hashlib import sha256

    name = "candidate"
    stable = (ORIGINAL / f"{name}.stablehlo.mlir").read_text()
    optimized = (ORIGINAL / f"{name}.optimized_hlo.txt").read_text()
    monkeypatch.setattr(panel, "CANDIDATE_SHA", sha256(stable.encode()).hexdigest())
    monkeypatch.setattr(panel, "CANDIDATE_BYTES", len(stable.encode()))
    memory = panel.registered_programs()[name]["compiled_memory"]
    report = panel.inspect_program(name, stable, optimized, memory)
    assert report["target_registration"] == panel.target_registration()
    panel.validate_program_report(report, name, stable, optimized, memory)
    report["target_registration"]["required_wide_suffix_p50_ratio_max"] = 2.0
    with pytest.raises(ValueError, match="report differs"):
        panel.validate_program_report(report, name, stable, optimized, memory)
