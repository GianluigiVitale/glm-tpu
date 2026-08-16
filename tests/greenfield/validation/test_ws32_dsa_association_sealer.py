from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace

import bench.provenance as provenance
import pytest

from glm_tpu.greenfield.validation.ws32_dsa_association import (
    _require_exact_comparison_schema,
    _require_selection_schema,
    inspect_ws32_dsa_source_summary,
)


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/greenfield/seal_ws32_dsa_association.py"
WRAPPER = ROOT / "scripts/greenfield/run_ws32_layer0_dsa_association.sh"
SOURCE_SUMMARY = Path(
    "/home/gianl/glm-run/"
    "greenfield_layer0_dsa_scorer_association_20260810T164030202890642Z/"
    "summary.json"
)
SOURCE_SUMMARY_SHA = (
    "01de9dff6595180c968a7b9a1fac7e8fe1b4773ea79a2b71fb7e0d516348d7c0"
)
SPEC = importlib.util.spec_from_file_location("ws32_dsa_sealer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SEALER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEALER)


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        association_manifest_sha256="1" * 64,
        code_hash="2" * 64,
        elapsed_seconds=7,
        internal_contract_sha256="3" * 64,
        internal_tensor_sha256="4" * 64,
        legacy_repo=Path("/home/gianl/tpu-inference"),
        prompt_cache_manifest_sha256="5" * 64,
        remote_prefix="gs://driftbench-dsv4-uc/results/unit_ws32_dsa",
        results_db=tmp_path / "results.db",
        run_dir=tmp_path / "run",
        source_summary_sha256="6" * 64,
        tag="greenfield_ws32_layer0_dsa_association_unit",
        worktree=ROOT,
    )


def _terminal() -> tuple[dict[str, object], dict[str, object]]:
    runner = {
        "claim_scope": (
            "bounded one-layer position-8155 WS32 DSA association only; "
            "no decoder, Gate-D, latency, or token-rate claim"
        ),
        "tensor_file": {"sha256": "7" * 64},
    }
    validation = {
        "artifact_kind": "greenfield_ws32_layer0_dsa_association",
        "association_restored": False,
        "candidate_mechanisms_proven": True,
        "code_hash": "2" * 64,
        "exact_arms": ["tuple4"],
        "hlo_file_count": 20,
        "runner_sha256": "8" * 64,
        "status": "VALIDATED",
        "tensor_sha256": "7" * 64,
    }
    return runner, validation


def test_ws32_dsa_db_publication_and_exact_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.run_dir.mkdir()
    provenance.connect(str(args.results_db)).close()
    terminal = _terminal()
    monkeypatch.setattr(SEALER, "_load_terminal_inputs", lambda _: terminal)
    assert SEALER._publish_db(args) == 0
    connection = sqlite3.connect(args.results_db)
    try:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone() == (1,)
        assert connection.execute("SELECT correct FROM items").fetchone() == (0,)
        assert connection.execute("SELECT value FROM summary").fetchone() == (1.0,)
    finally:
        connection.close()
    assert SEALER._rollback_db(args) == 0
    connection = sqlite3.connect(args.results_db)
    try:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM summary").fetchone() == (0,)
    finally:
        connection.close()


def test_ws32_dsa_rollback_refuses_changed_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.run_dir.mkdir()
    provenance.connect(str(args.results_db)).close()
    monkeypatch.setattr(SEALER, "_load_terminal_inputs", lambda _: _terminal())
    assert SEALER._publish_db(args) == 0
    connection = sqlite3.connect(args.results_db)
    connection.execute("UPDATE items SET raw_output='changed'")
    connection.commit()
    connection.close()
    with pytest.raises(SystemExit, match="item rollback"):
        SEALER._rollback_db(args)


@pytest.mark.skipif(not SOURCE_SUMMARY.is_file(), reason="protected DB529 absent")
def test_ws32_dsa_source_summary_is_sha_and_verdict_bound(tmp_path: Path) -> None:
    summary = inspect_ws32_dsa_source_summary(
        SOURCE_SUMMARY,
        expected_sha256=SOURCE_SUMMARY_SHA,
        association_manifest_sha256=(
            "574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141"
        ),
        prompt_cache_manifest_sha256=(
            "acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab"
        ),
        internal_tensor_sha256=(
            "c2fdeccfdcc81363fe01a566c34bf6c04f2f44b0a18e7b545e76fdf0f0d4560b"
        ),
    )
    assert summary["results_db_run_id"] == 529
    changed = json.loads(SOURCE_SUMMARY.read_text())
    changed["candidate_restored"] = False
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="verdict"):
        inspect_ws32_dsa_source_summary(
            path,
            expected_sha256=sha256(path.read_bytes()).hexdigest(),
            association_manifest_sha256=(
                "574f3553e6106a997e780b6b2a321bce86ad358b19c38989e84e2a4914b73141"
            ),
            prompt_cache_manifest_sha256=(
                "acc631e71148922448eb03c839f71544c80ca00cea47b639bdd80eb34567fdab"
            ),
            internal_tensor_sha256=(
                "c2fdeccfdcc81363fe01a566c34bf6c04f2f44b0a18e7b545e76fdf0f0d4560b"
            ),
        )


def test_ws32_dsa_wrapper_is_default_off_and_terminal_last() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text()
    assert "GLM_GREENFIELD_WS32_DSA_ASSOCIATION:-0" in source
    assert "GLM_GREENFIELD_WS32_DSA_ASSOCIATION_MODE:-off" in source
    assert ".glm_pod_workload.lock" in source
    assert source.index("strict_census post") < source.index("publish-db")
    assert source.index("publish-db") < source.index("publish-archive")
    assert source.index("publish-archive") < source.index("publish-success")
    assert "rollback-success" in source and "rollback-db" in source
    assert "remote_vacant=1" in source
    assert "&& $remote_vacant -eq 1" in source
    assert "--if-generation-match" not in source  # Python publisher owns it.
    publisher = SCRIPT.read_text()
    assert "if_generation_match=0" in publisher
    assert "download_as_bytes(if_generation_match=" in publisher
    assert "blob.delete(if_generation_match=" in publisher
    assert "remote_objects.json" in publisher


def test_ws32_dsa_census_initializes_label_before_derived_locals(
    tmp_path: Path,
) -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("strict_census() {")
    end = source.index("\n}\n\ncommon_sealer_args", start) + 3
    function = source[start:end]
    script = f"""\
set -u
RUN_DIR=$1
TAG=unit
POD=pod
ZONE=zone
gcloud() {{ :; }}
has_eight_unique_markers() {{ :; }}
{function}
strict_census probe
test -f "$RUN_DIR/census_probe.txt"
"""
    subprocess.run(
        ["bash", "-c", script, "ws32-dsa-census-test", str(tmp_path)],
        check=True,
    )


@pytest.mark.skipif(not SOURCE_SUMMARY.is_file(), reason="protected DB529 absent")
def test_ws32_dsa_wrapper_source_preflight_executes_production_heredoc() -> None:
    source = WRAPPER.read_text()
    marker = '"$SOURCE_DB_RUN_ID" <<\'PY\'\n'
    body = source.split(marker, 1)[1].split("\nPY\n", 1)[0]
    subprocess.run(
        [
            "/home/gianl/vllm-env/bin/python",
            "-c",
            body,
            str(SOURCE_SUMMARY),
            SOURCE_SUMMARY_SHA,
            "/home/gianl/glm-tpu/bench/results.db",
            "529",
        ],
        check=True,
    )


def test_ws32_dsa_terminal_types_reject_bool_as_number() -> None:
    comparison = {
        "actual_sha256": "1" * 64,
        "elementwise_exact": True,
        "expected_sha256": "1" * 64,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "mismatch_count": False,
        "shape": [1],
    }
    with pytest.raises(ValueError, match="types"):
        _require_exact_comparison_schema(comparison, field="unit")
    selection = {
        "actual_cutoff_score": 1.0,
        "actual_top_positions": [0],
        "context": 1,
        "expected_cutoff_score": 1.0,
        "passed": True,
        "position_mismatch_count": 0,
        "score_all_finite": True,
        "score_correlation": 1.0,
        "score_max_abs": True,
        "score_mean_abs": 0.0,
        "score_mean_signed": 0.0,
        "score_p99_abs": 0.0,
        "selected_order_exact": True,
        "selected_set_exact": True,
        "swapped_position_count": 0,
        "top_k": 1,
    }
    with pytest.raises(ValueError, match="finite float"):
        _require_selection_schema(selection, field="unit")


def test_terminal_payload_accepts_one_explicitly_rejected_arm(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    roots = {
        "census_post.txt",
        "census_pre.txt",
        "db_link.json",
        "orchestrator.sealed.log",
        "remote_vacancy.txt",
        "results_ckpt.db",
        "runner.json",
        "runner.log",
        "summary.json",
        "sync.txt",
        "validation.json",
        "ws32_layer0_dsa_association.npz",
    }
    run_dir.mkdir()
    for name in roots:
        (run_dir / name).write_text("{}\n" if name.endswith(".json") else "x")
    (run_dir / "validation.json").write_text(
        json.dumps({"hlo_file_count": 18}) + "\n"
    )
    hlo_dir = run_dir / "hlo"
    hlo_dir.mkdir()
    for index in range(18):
        (hlo_dir / f"graph_{index:02d}.txt").write_text("hlo")
    input_names = (
        ("association", "manifest.json"),
        ("association", "tensors.npz"),
        ("internal", "contract.json"),
        ("internal", "tensors.npz"),
        ("prompt_cache", "manifest.json"),
        ("prompt_cache", "tensors.npz"),
        ("source", "summary.json"),
    )
    for directory, name in input_names:
        path = run_dir / "inputs" / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
    payload = SEALER._expected_payload_paths(run_dir)
    assert len(payload) == 37

    (hlo_dir / "unexpected.txt").write_text("hlo")
    with pytest.raises(SystemExit, match="object set"):
        SEALER._expected_payload_paths(run_dir)
