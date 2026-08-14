from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.validation.dense_partials import (
    DensePartialsCaptureConfig,
    DensePartialsRollbackConfig,
    SHAPE,
    _rollback_environment,
    capture_and_compare_dense_partials,
    rollback_dense_partial_oracle_run,
)


RUN_TAG = "accepted_dense_partials_test"
CODE_HASH = "a" * 40
ORACLE_PIN = "b" * 40
LAYER = "model.layers.0.self_attn.attn"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bits() -> np.ndarray:
    value = (np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
             % 97 / 31).astype(ml_dtypes.bfloat16)
    return np.ascontiguousarray(value).view(np.uint16)


def _write_source(
    root: Path,
    bits: np.ndarray,
    *,
    code_hash: str = CODE_HASH,
    oracle_pin: str = ORACLE_PIN,
) -> None:
    safe = LAYER.replace("/", "_").replace(".", "_")
    for rank in range(32):
        process = rank // 4
        owner, virtual_rank = divmod(rank, 8)
        path = root / (
            f"internals.{safe}.position8155.proc{process}.rank{rank:02d}.npz")
        np.savez(
            path,
            artifact_kind=np.asarray("glm52_legacy_dense_partials"),
            format_version=np.asarray(1, np.int64),
            capture_mode=np.asarray("dense_partial"),
            process_index=np.asarray(process, np.int64),
            process_count=np.asarray(8, np.int64),
            model_rank=np.asarray(rank, np.int64),
            owner=np.asarray(owner, np.int64),
            virtual_rank=np.asarray(virtual_rank, np.int64),
            layer_name=np.asarray(LAYER),
            position=np.asarray(8155, np.int32),
            source_row=np.asarray(0, np.int32),
            run_tag=np.asarray(RUN_TAG),
            code_hash=np.asarray(code_hash),
            oracle_pin=np.asarray(oracle_pin),
            model_id=np.asarray("zai-org/GLM-5.2-FP8"),
            dense_partial=bits[owner, virtual_rank, 0],
            dense_partial__dtype=np.asarray("bfloat16"),
        )


def _config(tmp_path: Path, *, mismatch: bool = False):
    source = tmp_path / "source"
    source.mkdir()
    bits = _bits()
    _write_source(source, bits)
    db548 = tmp_path / "db548"
    db548.mkdir()
    observed = bits.copy()
    if mismatch:
        observed[3, 1, 0, 2795] += np.uint16(1)
    np.savez(
        db548 / "dense_partial_capture.npz",
        dense_virtual_partials_bfloat16_bits=observed,
    )
    (db548 / "runner.json").write_text(json.dumps({
        "artifact_kind": "glm52_layer0_dense_partial_capture",
        "capture_partials": True,
        "performance_claim": False,
        "status": "SUCCESS",
    }))
    (db548 / "summary.json").write_text("{}\n")
    (db548 / "SUCCESS").write_text("status=SUCCESS\n")
    config = DensePartialsCaptureConfig(
        source_dump_dir=source,
        db548_dir=db548,
        output_dir=tmp_path / "output",
        expected_run_tag=RUN_TAG,
        expected_legacy_code_hash=CODE_HASH,
        expected_oracle_pin=ORACLE_PIN,
        expected_db548_runner_sha256=_sha(db548 / "runner.json"),
        expected_db548_tensor_sha256=_sha(
            db548 / "dense_partial_capture.npz"),
        expected_db548_summary_sha256=_sha(db548 / "summary.json"),
        expected_db548_success_sha256=_sha(db548 / "SUCCESS"),
    )
    return config


@pytest.mark.parametrize("mismatch", [False, True])
def test_capture_and_compare_dense_partials(tmp_path, mismatch):
    result = capture_and_compare_dense_partials(
        _config(tmp_path, mismatch=mismatch))
    comparison = result["comparison"]
    assert comparison["elementwise_exact"] is (not mismatch)
    assert comparison["mismatch_count"] == int(mismatch)
    assert result["classification"] == (
        "accepted_dense_partials_nonexact_db548" if mismatch
        else "accepted_dense_partials_exact_db548")
    if mismatch:
        assert comparison["per_rank"][25] == {
            "first_mismatch_index": 2795,
            "mismatch_count": 1,
            "model_rank": 25,
            "owner": 3,
            "virtual_rank": 1,
        }
    with np.load(tmp_path / "output" / "dense_partials.npz",
                 allow_pickle=False) as payload:
        assert payload["accepted_dense_partials_bfloat16_bits"].shape == SHAPE


def test_dense_partial_capture_rejects_duplicate_rank(tmp_path):
    config = _config(tmp_path)
    paths = sorted(config.source_dump_dir.glob("*.npz"))
    with np.load(paths[-1], allow_pickle=False) as payload:
        values = {name: payload[name] for name in payload.files}
    values["model_rank"] = np.asarray(0, np.int64)
    values["owner"] = np.asarray(0, np.int64)
    values["virtual_rank"] = np.asarray(0, np.int64)
    np.savez(paths[-1], **values)
    with pytest.raises(ValueError, match="invalid/duplicate model rank"):
        capture_and_compare_dense_partials(config)


def test_dense_partial_capture_rejects_db548_hash_drift(tmp_path):
    config = _config(tmp_path)
    (config.db548_dir / "SUCCESS").write_text("status=CORRUPT\n")
    with pytest.raises(ValueError, match="DB548 success identity drifted"):
        capture_and_compare_dense_partials(config)


def test_protected_db548_replay_uses_real_tensor_key(tmp_path: Path) -> None:
    db548 = Path(
        "/home/gianl/glm-run/"
        "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z"
    )
    required = (
        "runner.json", "dense_partial_capture.npz", "summary.json", "SUCCESS")
    if not all((db548 / name).is_file() for name in required):
        pytest.skip("protected DB548 evidence is not mounted")
    with np.load(db548 / "dense_partial_capture.npz", allow_pickle=False) as payload:
        assert "dense_virtual_partials_bfloat16_bits" in payload.files
        bits = np.ascontiguousarray(
            payload["dense_virtual_partials_bfloat16_bits"])
    source = tmp_path / "source"
    source.mkdir()
    _write_source(source, bits)
    result = capture_and_compare_dense_partials(DensePartialsCaptureConfig(
        source_dump_dir=source,
        db548_dir=db548,
        output_dir=tmp_path / "output",
        expected_run_tag=RUN_TAG,
        expected_legacy_code_hash=CODE_HASH,
        expected_oracle_pin=ORACLE_PIN,
        expected_db548_runner_sha256=(
            "d22b35f664409485b697fdb579665dc4b1ecf42683a2aaff9e9a4d7481ee8343"
        ),
        expected_db548_tensor_sha256=(
            "f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298"
        ),
        expected_db548_summary_sha256=(
            "c349b5fd458f34986d8cc59c0f026af6b0a4c4e998f8691d6f6aa83a9b55e416"
        ),
        expected_db548_success_sha256=(
            "6cac897695fc1e78d0a10c0e36c993cffd281c6a88721d8955fa470bd44b0b85"
        ),
    ))
    assert result["comparison"]["elementwise_exact"] is True


def test_protected_wrapper_pins_dense_partial_sources_and_cleanup() -> None:
    repo = Path(__file__).resolve().parents[3]
    wrapper = (
        repo / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text()
    launcher = (
        repo / "scripts/greenfield/run_capture_legacy_layer0_dense_partials.sh"
    ).read_text()
    assert "GLM_GREENFIELD_DSA_INTERNALS_MODE=dense_partial" in launcher
    assert "GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k" in launcher
    for exact in (
        "LEGACY_PIN=4e3aa9666cefa38deba9c2824d5125c2e32ab2cf",
        "OBSERVER_COMMIT_DISTANCE=12",
        "DENSE_PARTIAL_PROBE_RUNNER_SHA=d22b35f664409485b697fdb579665dc4b1ecf42683a2aaff9e9a4d7481ee8343",
        "DENSE_PARTIAL_PROBE_TENSOR_SHA=f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298",
        "DENSE_PARTIAL_PROBE_SUMMARY_SHA=c349b5fd458f34986d8cc59c0f026af6b0a4c4e998f8691d6f6aa83a9b55e416",
        "DENSE_PARTIAL_PROBE_SUCCESS_SHA=6cac897695fc1e78d0a10c0e36c993cffd281c6a88721d8955fa470bd44b0b85",
    ):
        assert exact in wrapper
    assert wrapper.index(
        "protected DB548 dense-partial evidence identity drifted"
    ) < wrapper.index("strict_census pre")
    assert wrapper.index("strict_census post") < wrapper.index(
        "freezing fresh DSA-oracle evidence"
    )
    assert wrapper.index(
        "validate_exact_remote_object_set(root, prefix, listing)"
    ) < wrapper.index('(root / "SUCCESS").write_text')
    assert "rollback_dense_partial_oracle_run" in wrapper
    assert wrapper.index("remote SUCCESS checksum mismatch") < wrapper.index(
        "terminal_success_done=1")
    assert "unset GLM_GREENFIELD_DENSE_PARTIAL_CAPTURE_TAG" in launcher


def _terminal_program() -> str:
    wrapper = Path(__file__).resolve().parents[3] / (
        "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    )
    programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", wrapper.read_text(), re.DOTALL)
    return next(
        program for program in programs
        if "validate_dense_partials_artifacts" in program
    )


def _manifest(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["manifest_sha256"] = sha256(json.dumps(
        result,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()).hexdigest()
    return result


def _write_terminal_fixture(
    root: Path, *, exact: bool
) -> DensePartialsCaptureConfig:
    legacy_pin = "4e3aa9666cefa38deba9c2824d5125c2e32ab2cf"
    oracle_pin = "b3c25df47ac98783912dc658878181ec0a8ae16d"
    (root / "oracle").mkdir(parents=True)
    (root / "oracle" / "manifest.json").write_text(json.dumps({
        "artifact_kind": "oracle",
        "manifest_sha256": "5" * 64,
    }))
    (root / "evidence_sha256.json").write_text("{}\n")
    (root / "remote_objects.json").write_text("{}\n")
    (root / "dsa_exact_comparison.json").write_text('{"exact": true}\n')
    source = root / "source_dumps"
    source.mkdir()
    accepted = _bits()
    _write_source(
        source, accepted, code_hash=legacy_pin, oracle_pin=oracle_pin)
    db548 = root / "db548"
    db548.mkdir()
    observed = accepted.copy()
    if not exact:
        observed[3, 1, 0, 2795] += np.uint16(1)
    np.savez(
        db548 / "dense_partial_capture.npz",
        dense_virtual_partials_bfloat16_bits=observed,
    )
    (db548 / "runner.json").write_text(json.dumps({
        "artifact_kind": "glm52_layer0_dense_partial_capture",
        "capture_partials": True,
        "performance_claim": False,
        "status": "SUCCESS",
    }))
    (db548 / "summary.json").write_text("{}\n")
    (db548 / "SUCCESS").write_text("status=SUCCESS\n")
    config = DensePartialsCaptureConfig(
        source_dump_dir=source,
        db548_dir=db548,
        output_dir=root / "dense_partials_capture",
        expected_run_tag=RUN_TAG,
        expected_legacy_code_hash=legacy_pin,
        expected_oracle_pin=oracle_pin,
        expected_db548_runner_sha256=_sha(db548 / "runner.json"),
        expected_db548_tensor_sha256=_sha(
            db548 / "dense_partial_capture.npz"),
        expected_db548_summary_sha256=_sha(db548 / "summary.json"),
        expected_db548_success_sha256=_sha(db548 / "SUCCESS"),
    )
    capture_and_compare_dense_partials(config)
    return config


def _run_terminal(
    root: Path, config: DensePartialsCaptureConfig
) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        sys.executable,
        "-c",
        _terminal_program(),
        str(root),
        "gs://driftbench-dsv4-uc/unit",
        "7" * 40,
        "4e3aa9666cefa38deba9c2824d5125c2e32ab2cf",
        "1", "2", "294", "1", "32", "0", "0", "0",
        "dense_partial", "0", "0", "0", "0", "0", "0", "0", "0",
        RUN_TAG,
        str(config.db548_dir),
        config.expected_db548_runner_sha256,
        config.expected_db548_tensor_sha256,
        config.expected_db548_summary_sha256,
        config.expected_db548_success_sha256,
        config.expected_oracle_pin,
    ], text=True, capture_output=True, check=False)


@pytest.mark.parametrize("exact", (True, False))
def test_terminal_accepts_consistent_dense_partial_verdict(
    tmp_path: Path, exact: bool
) -> None:
    config = _write_terminal_fixture(tmp_path, exact=exact)
    completed = _run_terminal(tmp_path, config)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "mutation",
    (
        "count", "first", "classification", "db548_sha", "rank_sum",
        "accepted_sha", "rank_owner", "sealed_npz",
    ),
)
def test_terminal_refuses_dense_partial_contradiction(
    tmp_path: Path, mutation: str
) -> None:
    config = _write_terminal_fixture(tmp_path, exact=False)
    if mutation == "sealed_npz":
        path = config.output_dir / "dense_partials.npz"
        with np.load(path, allow_pickle=False) as payload:
            accepted = np.ascontiguousarray(
                payload["accepted_dense_partials_bfloat16_bits"])
            observed = np.ascontiguousarray(
                payload["db548_dense_partials_bfloat16_bits"])
        accepted[0, 0, 0, 0] += np.uint16(1)
        np.savez(
            path,
            accepted_dense_partials_bfloat16_bits=accepted,
            db548_dense_partials_bfloat16_bits=observed,
        )
    elif mutation == "accepted_sha":
        path = config.output_dir / "capture.json"
        capture = json.loads(path.read_text())
        capture["tensor"]["accepted_tensor_sha256"] = "9" * 64
        capture.pop("manifest_sha256")
        path.write_text(json.dumps(_manifest(capture)))
        comparison_path = config.output_dir / "comparison.json"
        comparison = json.loads(comparison_path.read_text())
        comparison["accepted_capture_manifest_sha256"] = _manifest(
            capture)["manifest_sha256"]
        comparison.pop("manifest_sha256")
        comparison_path.write_text(json.dumps(_manifest(comparison)))
    else:
        path = config.output_dir / "comparison.json"
        comparison = json.loads(path.read_text())
        numeric = comparison["comparison"]
        if mutation == "count":
            numeric["mismatch_count"] = 0
        elif mutation == "first":
            numeric["first_mismatch_flat_index"] = None
        elif mutation == "classification":
            comparison["classification"] = (
                "accepted_dense_partials_exact_db548")
        elif mutation == "db548_sha":
            numeric["observed_sha256"] = "8" * 64
        elif mutation == "rank_sum":
            numeric["per_rank"][25]["mismatch_count"] = 0
        elif mutation == "rank_owner":
            numeric["per_rank"][25]["owner"] = 2
        comparison.pop("manifest_sha256")
        path.write_text(json.dumps(_manifest(comparison)))
    completed = _run_terminal(tmp_path, config)
    assert completed.returncode != 0


def _rollback_config(path: Path) -> DensePartialsRollbackConfig:
    prompt = "prompt"
    return DensePartialsRollbackConfig(
        results_db=path,
        run_tag="dense_partial_rollback_test",
        expected_harness_git="harness",
        expected_fork_git="oracle",
        expected_legacy_pin="c" * 40,
        expected_legacy_short="c" * 9,
        expected_oracle_pin="d" * 40,
        expected_remote_prefix="gs://driftbench-dsv4-uc/unit/rollback",
        expected_dump_prefix="/tmp/rollback/topk.npz",
        expected_internal_dump_prefix="/tmp/rollback/internals.npz",
        expected_prompt_sha256=sha256(prompt.encode()).hexdigest(),
        expected_raw_outputs=("raw",),
    )


def _write_rollback_db(path: Path, *, prefix: int) -> DensePartialsRollbackConfig:
    config = _rollback_config(path)
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE runs (
          run_id INTEGER PRIMARY KEY AUTOINCREMENT, created_utc TEXT NOT NULL,
          model TEXT NOT NULL, model_revision TEXT, harness_git TEXT,
          fork_git TEXT, env_json TEXT, pod TEXT, note TEXT);
        CREATE TABLE items (
          id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
          benchmark TEXT NOT NULL, item_id TEXT NOT NULL,
          asked_utc TEXT NOT NULL, prompt TEXT NOT NULL, gold TEXT,
          raw_output TEXT, extracted TEXT, correct INTEGER, score REAL,
          n_prompt_tokens INTEGER, n_gen_tokens INTEGER, latency_ms REAL,
          seed INTEGER, finish_reason TEXT, truncated INTEGER);
        CREATE TABLE summary (
          id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER NOT NULL,
          benchmark TEXT NOT NULL, created_utc TEXT NOT NULL, n INTEGER,
          metric TEXT, value REAL, card_value REAL, delta REAL, note TEXT);
    """)
    timestamp = "2026-08-14T12:34:56+00:00"
    connection.execute(
        "INSERT INTO runs(created_utc,model,model_revision,harness_git,"
        "fork_git,env_json,pod,note) VALUES(?,?,?,?,?,?,?,?)",
        (
            timestamp, "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8",
            None, config.expected_harness_git, config.expected_fork_git,
            json.dumps(_rollback_environment(config)), "db-v4-64-od",
            f"fresh flat all-event DSA oracle {config.run_tag}",
        ),
    )
    if prefix >= 1:
        connection.execute(
            "INSERT INTO items(run_id,benchmark,item_id,asked_utc,prompt,gold,"
            "raw_output,extracted,correct,score,n_prompt_tokens,n_gen_tokens,"
            "latency_ms,seed,finish_reason,truncated) "
            "VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "passkey_L8192_d0.5", "t0", timestamp, "prompt", "881446",
                "raw", "881446", 1, 1.0, 8155, 20, 123.5, 1093997,
                None, None,
            ),
        )
    summaries = (
        (
            "passkey_L8192_d0.5", timestamp, 1, "acc", 100.0,
            None, None, "",
        ),
        (
            "longctx_passkey", timestamp, 0, "acc", 100.0, None, None,
            "aggregate over 1 cells x 1 trials; per-cell rows = passkey_L*_d*",
        ),
    )
    for summary in summaries[:max(0, prefix - 1)]:
        connection.execute(
            "INSERT INTO summary(run_id,benchmark,created_utc,n,metric,value,"
            "card_value,delta,note) VALUES(1,?,?,?,?,?,?,?,?)", summary,
        )
    connection.commit()
    connection.close()
    return config


@pytest.mark.parametrize("prefix", (0, 1, 2, 3))
def test_dense_partial_rollback_accepts_only_committed_prefixes(
    tmp_path: Path, prefix: int
) -> None:
    path = tmp_path / "results.db"
    config = _write_rollback_db(path, prefix=prefix)
    assert rollback_dense_partial_oracle_run(config) == (
        "ROLLED_BACK_PROVISIONAL_DB_RUN=1")
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
    assert connection.execute("SELECT COUNT(*) FROM summary").fetchone()[0] == 0
    connection.close()


@pytest.mark.parametrize("mutation", ("environment", "raw", "summary"))
def test_dense_partial_rollback_refuses_drift(
    tmp_path: Path, mutation: str
) -> None:
    path = tmp_path / "results.db"
    config = _write_rollback_db(path, prefix=3)
    connection = sqlite3.connect(path)
    if mutation == "environment":
        environment = _rollback_environment(config)
        environment["os_env"]["GLM_DSA_DUMP_INTERNALS_MODE"] = "rogue"
        connection.execute(
            "UPDATE runs SET env_json=? WHERE run_id=1",
            (json.dumps(environment),),
        )
    elif mutation == "raw":
        connection.execute("UPDATE items SET raw_output='rogue' WHERE run_id=1")
    else:
        connection.execute("UPDATE summary SET value=99 WHERE id=2")
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="refusing"):
        rollback_dense_partial_oracle_run(config)
    connection = sqlite3.connect(path)
    assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    connection.close()


def test_dense_partial_rollback_default_accepts_latest_protected_raw(
    tmp_path: Path,
) -> None:
    source = Path("/home/gianl/glm-tpu/bench/results.db")
    if not source.is_file():
        pytest.skip("protected results DB is not mounted")
    path = tmp_path / "results.db"
    shutil.copyfile(source, path)
    config = DensePartialsRollbackConfig(
        results_db=path,
        run_tag="dense_partial_default_real_schema_test",
        expected_harness_git="a4a17ac",
        expected_fork_git="b3c25df47",
        expected_legacy_pin="4e3aa9666cefa38deba9c2824d5125c2e32ab2cf",
        expected_legacy_short="4e3aa9666",
        expected_oracle_pin="b3c25df47ac98783912dc658878181ec0a8ae16d",
        expected_remote_prefix="gs://driftbench-dsv4-uc/unit/rollback-real",
        expected_dump_prefix="/tmp/rollback-real/topk.npz",
        expected_internal_dump_prefix="/tmp/rollback-real/internals.npz",
    )
    connection = sqlite3.connect(path)
    latest_raw = connection.execute(
        "SELECT raw_output FROM items WHERE run_id=543"
    ).fetchone()[0]
    assert latest_raw == (
        " 881446. Do not forget it. The grass is green. The sky is blue")
    connection.execute(
        "UPDATE runs SET harness_git=?,fork_git=?,env_json=?,note=? "
        "WHERE run_id=543",
        (
            config.expected_harness_git,
            config.expected_fork_git,
            json.dumps(_rollback_environment(config)),
            f"fresh flat all-event DSA oracle {config.run_tag}",
        ),
    )
    connection.commit()
    connection.close()
    assert rollback_dense_partial_oracle_run(config) == (
        "ROLLED_BACK_PROVISIONAL_DB_RUN=543")
    connection = sqlite3.connect(path)
    assert connection.execute(
        "SELECT COUNT(*) FROM runs WHERE run_id=543").fetchone()[0] == 0
    connection.close()
