from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from types import SimpleNamespace

import bench.provenance as provenance
from glm_tpu.greenfield.validation import ws32_evidence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/greenfield/seal_short_decoder_ws32.py"
WRAPPER = ROOT / "scripts/greenfield/run_short_decoder_ws32.sh"
RECOVERY_SCRIPT = (
    ROOT / "scripts/greenfield/recover_short_decoder_ws32_acquisition.py"
)
PROVISIONER = ROOT / "scripts/greenfield/provision_greenfield_worker.sh"
SPEC = importlib.util.spec_from_file_location("ws32_short_sealer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SEALER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEALER)
RECOVERY_SPEC = importlib.util.spec_from_file_location(
    "ws32_short_recovery", RECOVERY_SCRIPT
)
assert RECOVERY_SPEC is not None and RECOVERY_SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(RECOVERY_SPEC)
RECOVERY_SPEC.loader.exec_module(RECOVERY)


def _graph() -> dict[str, object]:
    return {
        "all_gather_count": 1,
        "all_reduce_count": 2,
        "async_collective_count": 0,
        "collective_count": 3,
        "expert_collective_count": 1,
        "feature_collective_count": 2,
        "fused_rmsnorm_collective_count": 157,
        "forbidden_full_hidden_values": [],
        "instruction_count": 100,
        "kind": "decode",
        "live_collective_count": 3,
        "live_instruction_count": 90,
        "maximum_group_size": 8,
        "optimized_hlo_sha256": "2" * 64,
        "passed": True,
        "rounded_first_rmsnorm_collective_count": 0,
        "stablehlo_sha256": "1" * 64,
        "strategy_nd_dense_expert_gather_count": 0,
        "strategy_nd_dense_hidden_gather_count": 0,
        "violations": [],
    }


def test_ws32_short_sealer_graph_contract_refuses_dead_or_global_work() -> None:
    exact = _graph()
    assert SEALER._graph_valid(exact, mode="numerical")
    assert not SEALER._graph_valid(
        {**exact, "live_collective_count": 2}, mode="numerical"
    )
    assert not SEALER._graph_valid(
        {**exact, "maximum_group_size": 32}, mode="numerical"
    )
    acquired = {
        **exact,
        "passed": False,
        "violations": [
            "StableHLO identity drifted",
            "optimized HLO identity drifted",
        ],
    }
    assert SEALER._graph_valid(acquired, mode="acquire")
    cache_probe = {
        **exact,
        "all_gather_count": 0,
        "all_reduce_count": 1,
        "collective_count": 1,
        "expert_collective_count": 1,
        "feature_collective_count": 0,
        "fused_rmsnorm_collective_count": 0,
        "kind": "cache_probe",
        "live_collective_count": 1,
    }
    assert SEALER._graph_valid(cache_probe, mode="numerical")
    exact_materialize = {
        "collective_count": 210,
        "instruction_count": 1000,
        "kind": "exact_materialize",
        "live_instruction_count": 900,
        "maximum_group_size": 8,
        "optimized_hlo_sha256": "2" * 64,
        "passed": True,
        "stablehlo_sha256": "1" * 64,
        "violations": [],
    }
    assert SEALER._graph_valid(exact_materialize, mode="numerical")
    assert not SEALER._graph_valid(
        {**exact_materialize, "maximum_group_size": 32}, mode="numerical"
    )


def test_ws32_short_db_publication_is_one_transaction(tmp_path: Path) -> None:
    results_db = tmp_path / "results.db"
    provenance.connect(str(results_db)).close()
    summary = {
        "artifact_kind": "greenfield_ws32_short_decoder_fleet",
        "checkpoint_manifest_sha256": "1" * 64,
        "checkpoint_success_sha256": "2" * 64,
        "code_hash": "3" * 40,
        "context_label": "2k",
        "dsa_oracle_manifest_sha256": "4" * 64,
        "dsa_oracle_success_sha256": "7" * 64,
        "mesh_sha256": "5" * 64,
        "mode": "numerical",
        "observed_generated_token_ids": [7, 8, 9],
        "observed_generated_token_count": 3,
        "p50_ms_per_token": 100.0,
        "run_tag": "greenfield_ws32_short_decoder_2k_20260815T000000000000000Z",
        "status": "SUCCESS",
        "steady_wall_tokens_per_second": 10.0,
        "strategy_nd_dense": False,
        "strategy_nd_dense_overlay_manifest_sha256": "0" * 64,
        "token_oracle_manifest_sha256": "6" * 64,
        "token_oracle_success_sha256": "8" * 64,
        "verified_generated_token_count": 3,
        "xla_python_client_mem_fraction": ".95",
    }
    summary["summary_sha256"] = sha256(SEALER._canonical(summary)).hexdigest()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    snapshot = tmp_path / "snapshot.db"
    output = tmp_path / "db_link.json"
    result = SEALER._publish_db(
        SimpleNamespace(
            summary=summary_path,
            results_db=results_db,
            snapshot=snapshot,
            output=output,
        )
    )
    assert result == 0
    connection = provenance.connect(str(results_db))
    try:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone() == (1,)
        assert connection.execute("SELECT correct FROM items").fetchone() == (1,)
        assert connection.execute("SELECT value FROM summary").fetchone() == (10.0,)
    finally:
        connection.close()
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["results_db_run_id"] == 1
    assert record["results_db_sha256"] == SEALER._digest_file(snapshot)
    # The wrapper arms rollback before publication.  A process failure after
    # SQLite commit but before db_link.json must still resolve and remove only
    # this exact authenticated run from the summary/run-tag identity.
    output.unlink()
    assert SEALER._rollback_db(
        SimpleNamespace(
            summary=summary_path,
            db_link=output,
            results_db=results_db,
        )
    ) == 0
    connection = provenance.connect(str(results_db))
    try:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM summary").fetchone() == (0,)
    finally:
        connection.close()


def test_ws32_short_db_rollback_resolves_the_adjudicated_row(tmp_path: Path) -> None:
    """The §21 adjudicated publication writes a different note/item_id/gold and
    extra env keys; rollback must derive the same identity or the row is orphaned."""
    results_db = tmp_path / "results.db"
    provenance.connect(str(results_db)).close()
    summary = {
        "artifact_kind": "greenfield_ws32_short_decoder_fleet",
        "checkpoint_manifest_sha256": "1" * 64,
        "checkpoint_success_sha256": "2" * 64,
        "checkpoint_transport": "shm",
        "classification": "EXACT_TOKENS;DSA_EVENT0_EXACT;LATER_EVENTS_RECORDED_NOT_ADJUDICATED",
        "code_hash": "3" * 40,
        "context_label": "8k",
        "dsa_adjudication": {"event_index": 1, "record_sha256": "9" * 64},
        "dsa_oracle_manifest_sha256": "4" * 64,
        "dsa_oracle_success_sha256": "7" * 64,
        "later_event_alarm": {"acknowledged": True, "alarmed_steps": [0, 1]},
        "mesh_sha256": "5" * 64,
        "mode": "numerical",
        "observed_generated_token_ids": [7, 8, 9],
        "observed_generated_token_count": 3,
        "p50_ms_per_token": 130.0,
        "recovery_code_hash": "a" * 40,
        "run_tag": "greenfield_ws32_short_decoder_8k_numerical_20260905T000000000000000Z",
        "status": "SUCCESS",
        "steady_wall_tokens_per_second": 7.6,
        "strategy_nd_dense": False,
        "strategy_nd_dense_overlay_manifest_sha256": "0" * 64,
        "token_oracle_manifest_sha256": "6" * 64,
        "token_oracle_success_sha256": "8" * 64,
        "verified_generated_token_count": 3,
        "xla_python_client_mem_fraction": ".95",
    }
    summary["summary_sha256"] = sha256(SEALER._canonical(summary)).hexdigest()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    output = tmp_path / "db_link.json"
    assert SEALER._publish_db(
        SimpleNamespace(
            summary=summary_path,
            results_db=results_db,
            snapshot=tmp_path / "snapshot.db",
            output=output,
        )
    ) == 0
    connection = provenance.connect(str(results_db))
    try:
        note = connection.execute("SELECT note FROM runs").fetchone()[0]
        item_id = connection.execute("SELECT item_id FROM items").fetchone()[0]
        env = json.loads(connection.execute("SELECT env_json FROM runs").fetchone()[0])
    finally:
        connection.close()
    assert "spec §21" in note and "alarm acknowledged" in note
    assert item_id == "gate_d_s21_exact_tokens_adjudicated_dsa_state_cache"
    assert env["recovery_code_hash"] == "a" * 40 and env["checkpoint_transport"] == "shm"
    # With the link present the rollback must resolve exactly this run.
    assert SEALER._rollback_db(
        SimpleNamespace(summary=summary_path, db_link=output, results_db=results_db)
    ) == 0
    connection = provenance.connect(str(results_db))
    try:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM summary").fetchone() == (0,)
    finally:
        connection.close()


def test_ws32_short_wrapper_is_default_off_and_terminal_last() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text(encoding="utf-8")
    assert "GLM_GREENFIELD_WS32_SHORT_DECODER:-0" in source
    assert "GLM_GREENFIELD_WS32_EXACT_DSA:-0" in source
    # Two local commands receive the literal shell argument; the remote runner
    # receives the same value through the quoted all-host command string.
    assert source.count('--exact-dsa "$EXACT_DSA"') >= 2
    assert source.count("--exact-dsa") >= 3
    assert "661142816aa64ec8d085553b427e99f62" in source
    assert "79aba79e24026bc4c1d17aed2ca92055" in source
    assert "greenfield_topology_20260826T194116460015528Z" in source
    assert "4a0c9a338d55b8be37dab79396569aa10fc9e85b3c7210d72a70abfafe72c301" in source
    assert "greenfield_topology_20260805T125842425591441Z" not in source
    assert "50de0729c9e5080c5ddb5ae4f5cd948317c53ce8a8f6c9f3f6064e7afc6515a0" not in source
    assert ".glm_pod_workload.lock" in source
    assert "strict_census pre" in source
    assert "strict_census post" in source
    assert "publish-db" in source and "rollback-db" in source
    assert "if_generation_match=int(blob.generation)" in source
    assert "remote_objects.json" in source
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION=.95" in source
    assert "GLM_GREENFIELD_WS32_SHORT_DECODER_RECOVER:-0" in source
    assert "recovery is only valid for a completed numerical run" not in source
    assert "glm_tpu.greenfield.validation.ws32_evidence" in source
    assert "source_remote_objects.json" in source
    assert "RESULTS_DB=/home/gianl/glm-tpu/bench/results.db" in source
    assert "less than 4 GiB" in source
    assert 'trap "upload || true" EXIT' in source
    assert 'return "$rc"' in source
    assert 'gcloud storage cp "$REMOTE_PREFIX/hlo/*"' not in source
    assert 'gcloud storage cp "$REMOTE_PREFIX/traces/*"' not in source
    assert '"$token_root/SUCCESS"' in source
    assert '"$dsa_root/SUCCESS"' in source
    assert "TOKEN_ORACLE_SUCCESS_SHA=" in source
    assert "DSA_ORACLE_SUCCESS_SHA=" in source
    assert '|| sync_rc=$?' in source
    assert '[[ $sync_rc -ne 0 ]] || ! has_eight_unique_markers' in source
    assert '|| launch_rc=$?' in source
    assert '[[ $launch_rc -ne 0 ]] || ! has_eight_unique_markers' in source
    assert "(strict_census failure_exit) || true" in source
    assert "rollback_success || true" not in source
    assert "rollback_remote_nonterminal" in source
    assert "archive_failed_publication" in source
    assert "refusing cleanup with drifted source ledger" in source
    assert "source object drifted during cleanup" in source
    assert "name.startswith('diagnostic/')" in source
    assert 'materialize.log validate.log; do' in source
    assert '>"$RUN_DIR/materialize.log" 2>&1' in source
    assert '>"$RUN_DIR/validate.log" 2>&1' in source
    assert "acquisition remote object set drifted" in source
    assert "acquisition diagnostic bytes drifted" in source
    assert 'acquisition_files+=(census_recovery_pre.txt)' in source
    assert 'RECOVER:-0} == 0 && ! -e $RUN_DIR/source_remote_objects.json' in source
    assert "a prior terminal SUCCESS verification exists" in source
    assert "recovering immutable worker prevalidation" in source
    assert "rollback_recovery_seed" in source
    assert "recovery_seed_objects.json" in source
    checkpoint_preflight = source.index(
        "checkpoint identity pins must be lowercase SHA-256 values"
    )
    recovery_publish = source.index("recovering immutable worker prevalidation")
    assert checkpoint_preflight < recovery_publish
    assert "checkpoint manifest identity pin drifted" in source
    assert "checkpoint SUCCESS identity pin drifted" in source
    assert "checkpoint SUCCESS self-hash drifted" in source
    assert "checkpoint manifest file hash drifted" in source
    assert source.index("a prior terminal SUCCESS verification exists") < source.index(
        "trap on_exit EXIT"
    )
    assert "success_absent -eq 1" in source
    success_upload = source.index(
        'gcloud storage cp --no-clobber "$RUN_DIR/SUCCESS"'
    )
    post_census = source.index("strict_census post")
    db_publish = source.index("publish-db")
    assert post_census < db_publish < success_upload
    on_exit = source[source.index("on_exit() {") : source.index("trap on_exit EXIT")]
    assert on_exit.index("rollback_success") < on_exit.index(
        "rollback_remote_nonterminal"
    ) < on_exit.index("rollback_db")
    assert "rm -rf" not in source


def test_greenfield_replacement_worker_provisioner_is_narrow_and_default_off() -> None:
    subprocess.run(["bash", "-n", str(PROVISIONER)], check=True)
    source = PROVISIONER.read_text(encoding="utf-8")
    assert "GLM_GREENFIELD_PROVISION:-0" in source
    assert "gs://driftbench-dsv4-uc" in source
    assert "rewrite/topology-first-decode" in source
    assert "gcsfuse --implicit-dirs -o ro" in source
    assert "gcsfuse-3.11.2-linux-amd64" in source
    assert "298bc02d8a6fd6948bf93aa69aee0ff74cf07339e1c018eeabb5d80db93a2225" in source
    assert "PROVISION_OK pin=$EXPECTED_PIN" in source
    assert "Worker 0 may be the orchestrator's live branch worktree" in source
    for forbidden in (
        "tpu-inference",
        "vllm-build",
        "ray start",
        "jax.devices",
        "gcloud compute tpus create",
    ):
        assert forbidden not in source


def test_ws32_evidence_primary_object_schema_is_exact() -> None:
    acquired = ws32_evidence._expected_primary_names(numerical=False)
    numerical = ws32_evidence._expected_primary_names(numerical=True)
    assert ws32_evidence._runner_suffixes(numerical=False) == ("json", "log")
    assert ws32_evidence._runner_suffixes(numerical=True) == (
        "json",
        "npz",
        "log",
    )
    # Spec §23.2: five base graphs (prefill_chunk, prefill_tail, observer,
    # decode, cache_probe) x 8 ranks x 2 HLO forms = 80, plus 16 host records
    # (layout v1); layout v2 uploads one gzip per graph/form: 10 + 16 = 26.
    assert len(acquired) == 96
    assert len(numerical) == 112
    v2 = ws32_evidence.EVIDENCE_LAYOUT_V2
    assert len(ws32_evidence._expected_primary_names(numerical=False, layout=v2)) == 26
    assert len(ws32_evidence._expected_primary_names(numerical=True, layout=v2)) == 42
    assert len(ws32_evidence._expected_primary_names(numerical=True, exact_dsa=True, layout=v2)) == 46
    assert {n for n in ws32_evidence._expected_primary_names(numerical=False, layout=v2) if n.startswith("hlo/")} == {
        f"hlo/{g}.{s}.gz" for g in ws32_evidence.BASE_GRAPHS for s, _ in ws32_evidence.HLO_FORMS
    }
    with pytest.raises(SystemExit, match="unknown WS32 evidence layout"):
        ws32_evidence._expected_primary_names(numerical=False, layout="v9")
    assert {name.split("/")[1].split(".")[0] for name in acquired if name.startswith("hlo/")} == {
        "prefill_chunk", "prefill_tail", "observer", "decode", "cache_probe"
    }
    exact_acquired = ws32_evidence._expected_primary_names(
        numerical=False, exact_dsa=True
    )
    exact_numerical = ws32_evidence._expected_primary_names(
        numerical=True, exact_dsa=True
    )
    assert len(exact_acquired) == 128
    assert len(exact_numerical) == 144
    assert exact_acquired - acquired == {
        f"hlo/{graph}.rank{rank}.{suffix}"
        for graph in ("exact_materialize", "exact_promote")
        for rank in range(8)
        for suffix, _ in ws32_evidence.HLO_FORMS
    }
    assert not any(name.endswith(".npz") for name in acquired)
    assert numerical - acquired == {
        *{f"host_records/runner.rank{rank}.npz" for rank in range(8)},
        *{f"traces/trace.rank{rank}.xplane.pb" for rank in range(8)},
    }
    assert all("_.gstmp" not in name for name in numerical)
    assert ws32_evidence._expected_runner_status("acquire") == "HLO_ACQUIRED"
    assert ws32_evidence._expected_runner_status("numerical") == "SUCCESS"


def test_ws32_acquisition_materializes_without_numerical_npz(
    tmp_path: Path, monkeypatch: object
) -> None:
    remote = tmp_path / "remote"
    run_dir = tmp_path / "run"
    prefix = "results/acquire"
    import gzip

    graph_records: dict[str, dict[str, str]] = {}
    for graph in ws32_evidence.GRAPHS:
        graph_records[graph] = {}
        for suffix, sha_key in ws32_evidence.HLO_FORMS:
            raw = f"{graph}:{suffix}\n".encode()
            graph_records[graph][sha_key] = sha256(raw).hexdigest()
            # Layout v2: one gzip object per graph/form, rank-agnostic name.
            path = remote / "hlo" / f"{graph}.{suffix}.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(raw, mtime=0))
    for rank in ws32_evidence.RANKS:
        record = {
            "code_hash": "a" * 40,
            "compile_only": True,
            "evidence_layout": ws32_evidence.EVIDENCE_LAYOUT_V2,
            "exact_dsa": False,
            "graphs": graph_records,
            "launch_process_id": rank,
            "status": "HLO_ACQUIRED",
        }
        root = remote / "host_records"
        root.mkdir(parents=True, exist_ok=True)
        (root / f"runner.rank{rank}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        (root / f"runner.rank{rank}.log").write_text(
            f"rank={rank}\n", encoding="utf-8"
        )

    class FakeBlob:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.name = f"{prefix}/{path.relative_to(remote).as_posix()}"
            self.generation = 1
            self.size = path.stat().st_size
            self.crc32c = ws32_evidence._crc32c_file(path)

        def download_to_filename(
            self, destination: str, *, if_generation_match: int
        ) -> None:
            assert if_generation_match == self.generation
            shutil.copyfile(self.path, destination)

        def download_as_bytes(self, *, if_generation_match: int) -> bytes:
            assert if_generation_match == self.generation
            return self.path.read_bytes()

    blobs = [FakeBlob(path) for path in remote.rglob("*") if path.is_file()]

    class FakeClient:
        def list_blobs(self, bucket: str, *, prefix: str) -> list[FakeBlob]:
            assert bucket == "unit"
            assert prefix == "results/acquire/"
            return blobs

    monkeypatch.setattr(ws32_evidence.storage, "Client", FakeClient)
    output = run_dir / "source_remote_objects.json"
    result = ws32_evidence.materialize(
        run_dir=run_dir,
        remote_prefix="gs://unit/results/acquire",
        mode="acquire",
        tag="greenfield_ws32_short_decoder_8k_acquire_20260816T000000000000000Z",
        code_hash="a" * 40,
        recovery_code_hash="b" * 40,
        exact_dsa=False,
        allow_failure_diagnostics=False,
        output=output,
    )
    names = {item["name"] for item in result["objects"]}
    assert names == ws32_evidence._expected_primary_names(
        numerical=False, layout=ws32_evidence.EVIDENCE_LAYOUT_V2
    )
    assert result["evidence_layout"] == ws32_evidence.EVIDENCE_LAYOUT_V2
    assert not any(name.endswith(".npz") for name in names)
    # 10 compressed objects + 5 graphs x 8 ranks x 2 forms inflated/hard-linked.
    assert len(list((run_dir / "fleet_hlo").iterdir())) == 10 + 80
    for graph in ws32_evidence.GRAPHS:
        for suffix, sha_key in ws32_evidence.HLO_FORMS:
            inflated = run_dir / "fleet_hlo" / f"{graph}.rank3.{suffix}"
            assert sha256(inflated.read_bytes()).hexdigest() == graph_records[graph][sha_key]
            gz = [o for o in result["objects"] if o["name"] == f"hlo/{graph}.{suffix}.gz"][0]
            assert gz["inflated_sha256"] == graph_records[graph][sha_key]

    for rank in ws32_evidence.RANKS:
        path = (
            remote
            / "recovery_prevalidation"
            / f"prevalidation.rank{rank}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"rank={rank}\n", encoding="utf-8")
    blobs = [FakeBlob(path) for path in remote.rglob("*") if path.is_file()]
    recovered_run = tmp_path / "recovered_run"
    recovered = ws32_evidence.materialize(
        run_dir=recovered_run,
        remote_prefix="gs://unit/results/acquire",
        mode="acquire",
        tag="greenfield_ws32_short_decoder_8k_acquire_20260816T000000000000000Z",
        code_hash="a" * 40,
        recovery_code_hash="b" * 40,
        exact_dsa=False,
        allow_failure_diagnostics=True,
        output=recovered_run / "source_remote_objects.json",
    )
    recovery_names = {
        f"recovery_prevalidation/prevalidation.rank{rank}.json"
        for rank in ws32_evidence.RANKS
    }
    assert recovered["recovered_prevalidation"] is True
    assert {item["name"] for item in recovered["objects"]} == (
        ws32_evidence._expected_primary_names(
            numerical=False, layout=ws32_evidence.EVIDENCE_LAYOUT_V2
        )
        | recovery_names
    )


def test_ws32_failed_acquisition_recovery_synthesizes_only_runner_envelope(
    tmp_path: Path, monkeypatch: object
) -> None:
    recovered_graphs = {
        graph: {
            "kind": graph,
            "optimized_hlo_sha256": f"{index + 1:064x}",
            "stablehlo_sha256": f"{index + 11:064x}",
            "passed": False,
            "violations": list(RECOVERY.IDENTITY_VIOLATIONS),
        }
        for index, graph in enumerate(RECOVERY.GRAPHS)
    }
    seen_flags: list[tuple[bool, bool]] = []

    def fake_replay(run_dir, *, exact_dsa, strategy_nd_dense):
        seen_flags.append((exact_dsa, strategy_nd_dense))
        return recovered_graphs

    monkeypatch.setattr(RECOVERY, "_replay_graphs", fake_replay)
    source_hash = "a" * 40
    for rank in RECOVERY.RANKS:
        # Mixed originals: the prefill graphs carried the documented false
        # positive, every other graph was identity-only (2026-09-05 acquisition).
        graphs = {
            graph: {
                **recovered_graphs[graph],
                "violations": list(
                    RECOVERY.ORIGINAL_VIOLATIONS[graph]
                    if graph.startswith("prefill")
                    else RECOVERY.IDENTITY_VIOLATIONS
                ),
            }
            for graph in RECOVERY.GRAPHS
        }
        source = {
            "artifact_kind": "greenfield_ws32_short_decoder_prevalidation",
            "code_hash": source_hash,
            "compile_only": True,
            "exact_dsa": True,
            "strategy_nd_dense": True,
            "graphs": graphs,
            "hostname": f"worker-{rank}",
            "launch_process_id": rank,
        }
        path = (
            tmp_path
            / "recovery_prevalidation"
            / f"prevalidation.rank{rank}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(source), encoding="utf-8")
    outputs = RECOVERY.synthesize(
        run_dir=tmp_path, source_code_hash=source_hash
    )
    assert len(outputs) == 8
    assert seen_flags == [(True, True)]
    for rank, path in enumerate(outputs):
        runner = json.loads(path.read_text(encoding="utf-8"))
        assert runner["launch_process_id"] == rank
        assert runner["graphs"] == recovered_graphs
        assert runner["performance_claim"] is False
        assert runner["schema_version"] == 1
        assert runner["status"] == "HLO_ACQUIRED"

    remote: dict[str, object] = {}
    next_generation = [100]

    class FakeBlob:
        def __init__(self, name: str) -> None:
            self.name = name
            self.generation = None
            self.size = None
            self.crc32c = None

        def upload_from_filename(self, path: str, **kwargs: object) -> None:
            assert kwargs["if_generation_match"] == 0
            assert self.name not in remote
            source = Path(path)
            self.generation = next_generation[0]
            next_generation[0] += 1
            self.size = source.stat().st_size
            self.crc32c = RECOVERY._crc32c_file(source)
            remote[self.name] = self

        def reload(self) -> None:
            assert remote.get(self.name) is self

        def delete(self, *, if_generation_match: int) -> None:
            assert if_generation_match == self.generation
            assert remote.get(self.name) is self
            del remote[self.name]

    class FakeBucket:
        def blob(self, name: str) -> FakeBlob:
            value = remote.get(name)
            return value if isinstance(value, FakeBlob) else FakeBlob(name)

    class FakeClient:
        def bucket(self, name: str) -> FakeBucket:
            assert name == "unit"
            return FakeBucket()

        def list_blobs(self, name: str, *, prefix: str) -> list[FakeBlob]:
            assert name == "unit"
            return [
                value
                for key, value in remote.items()
                if key.startswith(prefix) and isinstance(value, FakeBlob)
            ]

    monkeypatch.setattr(RECOVERY.storage, "Client", FakeClient)
    published = RECOVERY.publish(
        run_dir=tmp_path,
        remote_prefix="gs://unit/results/recover",
        source_code_hash=source_hash,
        recovery_code_hash="b" * 40,
    )
    assert len(published["objects"]) == 16
    assert len(remote) == 16
    seed = tmp_path / "recovery_seed_objects.json"
    RECOVERY.rollback(
        seed_path=seed, remote_prefix="gs://unit/results/recover"
    )
    assert remote == {}


def test_ws32_prelaunch_floor_covers_protected_unique_evidence_and_reserve() -> None:
    # Current protected fleet: worker-0 local HLO/trace/NPZ are created before
    # materialization, then seven distinct traces and peer records remain to be
    # fetched. Keep at least one GiB free throughout sealing.
    #
    # Evidence layout v2 adds only the compressed HLO objects: this host's own
    # gzip copies in ``hlo/`` and the materialized ``fleet_hlo/*.gz`` (each
    # ≈10x smaller than the text). The inflated per-rank names are hard links to
    # worker-0's own text (ws32_evidence: candidates_by_sha), so they cost no
    # bytes; a fresh inflation only happens on a host that is not worker 0.
    rank0_generated = 311_539_059 + 277_949_672 + 4_952_000
    compressed_hlo = 2 * 60_000_000
    remaining_unique = 1_946_790_017 + 35_000_000
    sealing_reserve = 1024**3
    required = rank0_generated + compressed_hlo + remaining_unique + sealing_reserve
    assert required < 4 * 1024**3
    source = WRAPPER.read_text(encoding="utf-8")
    assert "available_bytes -ge 4294967296" in source


def test_ws32_evidence_hardlink_refuses_existing_different_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"exact")
    ws32_evidence._link_exact(source, destination)
    assert destination.read_bytes() == b"exact"
    assert source.stat().st_ino == destination.stat().st_ino
    ws32_evidence._link_exact(source, destination)
    destination.unlink()
    destination.write_bytes(b"rogue")
    try:
        ws32_evidence._link_exact(source, destination)
    except SystemExit as error:
        assert "refusing to replace" in str(error)
    else:
        raise AssertionError("different existing evidence was replaced")


def test_ws32_evidence_refuses_partial_transfer_files(tmp_path: Path) -> None:
    for relative in (
        "fleet/runner.rank0.json.partial",
        "fleet_hlo/decode.rank0.optimized_hlo.txt_.gstmp",
        "traces/trace.rank0.xplane.pb.partial",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"partial")
    assert [
        path.relative_to(tmp_path).as_posix()
        for path in ws32_evidence._partial_evidence(tmp_path)
    ] == [
        "fleet/runner.rank0.json.partial",
        "fleet_hlo/decode.rank0.optimized_hlo.txt_.gstmp",
        "traces/trace.rank0.xplane.pb.partial",
    ]


def test_ws32_short_census_initializes_label_before_derived_locals(
    tmp_path: Path,
) -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("strict_census() {")
    end = source.index("\n}\n\nexec 9>", start) + 3
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
        ["bash", "-c", script, "ws32-census-test", str(tmp_path)],
        check=True,
    )


def test_ws32_run_tag_is_bound_to_context_and_mode() -> None:
    SEALER._validate_run_tag(
        "greenfield_ws32_short_decoder_2k_acquire_20260815T000000000000000Z",
        context_label="2k",
        mode="acquire",
    )
    for context, mode in (("8k", "acquire"), ("2k", "numerical")):
        try:
            SEALER._validate_run_tag(
                "greenfield_ws32_short_decoder_2k_acquire_20260815T000000000000000Z",
                context_label=context,
                mode=mode,
            )
        except SystemExit as error:
            assert "contradicts" in str(error)
        else:
            raise AssertionError("contradictory WS32 tag was accepted")


def test_ws32_failed_success_delete_retains_db_link(tmp_path: Path) -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("on_exit() {")
    end = source.index("\n}\ntrap on_exit EXIT", start) + 2
    on_exit = source[start:end]
    marker = tmp_path / "rollback-called"
    harness = f'''set -u
post_census_done=1
db_published=1
archive_upload_started=0
success_upload_started=1
terminal_success_verified=0
success_absent=0
RUN_DIR=/tmp/ws32-failure-injection
REMOTE_PREFIX=gs://driftbench-dsv4-uc/results/unit
strict_census() {{ :; }}
rollback_success() {{ return 1; }}
rollback_db() {{ touch {marker}; }}
rollback_remote_nonterminal() {{ :; }}
archive_failed_publication() {{ :; }}
say() {{ :; }}
gcloud() {{ :; }}
{on_exit}
false
on_exit
[[ ! -e {marker} ]]
'''
    subprocess.run(["bash", "-c", harness], check=True)


def test_ws32_failure_after_source_ledger_does_not_expand_remote_source_set(
    tmp_path: Path,
) -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("on_exit() {")
    end = source.index("\n}\ntrap on_exit EXIT", start) + 2
    on_exit = source[start:end]
    (tmp_path / "source_remote_objects.json").write_text(
        "sealed source set", encoding="utf-8"
    )
    marker = tmp_path / "remote-upload-called"
    harness = f'''set -u
post_census_done=1
db_published=0
archive_upload_started=0
success_upload_started=0
terminal_success_verified=0
success_absent=1
RECOVER=0
RUN_DIR=$1
REMOTE_PREFIX=gs://driftbench-dsv4-uc/results/unit
strict_census() {{ :; }}
rollback_success() {{ :; }}
rollback_db() {{ :; }}
rollback_remote_nonterminal() {{ :; }}
archive_failed_publication() {{ :; }}
say() {{ :; }}
gcloud() {{ touch {marker}; }}
{on_exit}
false
on_exit
[[ ! -e {marker} ]]
'''
    subprocess.run(
        ["bash", "-c", harness, "ws32-source-set-retry", str(tmp_path)],
        check=True,
    )


def test_ws32_recovery_quarantines_derived_outputs_only(tmp_path: Path) -> None:
    source = WRAPPER.read_text(encoding="utf-8")
    start = source.index("archive_failed_publication() {")
    end = source.index("\n}\non_exit() {", start) + 2
    function = source[start:end]
    (tmp_path / "summary.json").write_text("summary", encoding="utf-8")
    (tmp_path / "validate.log").write_text("validate", encoding="utf-8")
    (tmp_path / "source_remote_objects.json").write_text(
        "source", encoding="utf-8"
    )
    subprocess.run(
        [
            "bash",
            "-c",
            f"RUN_DIR=$1; {function}; archive_failed_publication",
            "ws32-recovery-quarantine",
            str(tmp_path),
        ],
        check=True,
    )
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "validate.log").exists()
    # on_exit archives derived outputs only: the ledger stays as rollback
    # authority for the next attempt.
    assert (tmp_path / "source_remote_objects.json").read_text() == "source"
    quarantined = list((tmp_path / "recovery_failures").glob("*/summary.json"))
    assert len(quarantined) == 1 and quarantined[0].read_text() == "summary"
    # Recovery start (after the remote set was proven equal to the stale
    # ledger) quarantines the ledger too, so it is regenerated under the new
    # recovery pin instead of refusing forever with different bytes.
    (tmp_path / "validate.log").write_text("validate2", encoding="utf-8")
    subprocess.run(
        [
            "bash",
            "-c",
            f"RUN_DIR=$1; {function}; archive_failed_publication with_ledger",
            "ws32-recovery-quarantine-ledger",
            str(tmp_path),
        ],
        check=True,
    )
    assert not (tmp_path / "source_remote_objects.json").exists()
    ledgers = list(
        (tmp_path / "recovery_failures").glob("*/source_remote_objects.json")
    )
    assert len(ledgers) == 1 and ledgers[0].read_text() == "source"
    assert len(list((tmp_path / "recovery_failures").glob("*/validate.log"))) == 2


def test_ws32_verified_success_refuses_before_cleanup_trap(tmp_path: Path) -> None:
    success = tmp_path / "success_upload.json"
    success.write_text("sealed", encoding="utf-8")
    script = f'''set -euo pipefail
RECOVER=1
RUN_DIR=$1
if [[ $RECOVER == 1 ]]; then
  [[ -d $RUN_DIR ]]
  [[ ! -e $RUN_DIR/success_upload.json ]] || {{
    echo "a prior terminal SUCCESS verification exists; refusing recovery" >&2
    exit 1
  }}
fi
touch "$RUN_DIR/trap-was-armed"
'''
    result = subprocess.run(
        ["bash", "-c", script, "ws32-verified-success", str(tmp_path)],
        check=False,
    )
    assert result.returncode == 1
    assert success.read_text(encoding="utf-8") == "sealed"
    assert not (tmp_path / "trap-was-armed").exists()


def test_ws32_wrapper_oracle_success_pins_match_all_four_trees() -> None:
    records = (
        (
            Path("/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/2k/greenfield_short_context_oracle_20260806T202544155912103Z/SUCCESS"),
            "07700db5a732f04663f0298625bbdbb68a1c73e63652a3aef27f398993e86eec",
        ),
        (
            Path("/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/2k/greenfield_short_context_dsa_oracle_recovery_20260806T231905802593249Z/SUCCESS"),
            "c091d0b56f712eb2f106ee248f69f599586ade0d5c202cbb5b46b8aa115411b2",
        ),
        (
            Path("/home/gianl/gcs-models/oracles/greenfield/glm52/short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/SUCCESS"),
            "38c0aeb6c4833a0256d4e50152b645e85d24a4f00ca7b2b1731db2d892c5b3cc",
        ),
        (
            Path("/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/SUCCESS"),
            "0b798974ae8a9f95c32d3aa2eff532213624f1e2ae7f1de809a161e18dbdf1b9",
        ),
    )
    for path, expected in records:
        assert path.is_file()
        assert sha256(path.read_bytes()).hexdigest() == expected


def test_ws32_short_sealer_selects_the_traced_decode_body_by_exact_dsa() -> None:
    """The exact-DSA path traces jit_execute_exact_body; the default path traces
    jit_execute_body.  The 8K exact-DSA seal refused with zero decode steps when
    the module regex was hard-coded to the default body."""
    import re

    decoder = (ROOT / "glm_tpu/greenfield/runtime/ws32_decoder.py").read_text(encoding="utf-8")
    for body in ("execute_body", "execute_exact_body"):
        assert f"    def {body}(" in decoder
        assert re.search(rf"execute = jax\.shard_map\(\s*{body},", decoder), body
    exact = SEALER._decode_step_module_re(exact_dsa=True)
    default = SEALER._decode_step_module_re(exact_dsa=False)
    assert re.search(exact, "jit_execute_exact_body(868403939627541584)")
    assert not re.search(exact, "jit_execute_body(1)")
    assert re.search(default, "jit_execute_body(1)")
    assert not re.search(default, "jit_execute_exact_body(1)")
    for pattern in (exact, default):
        assert not re.search(pattern, "jit_observe_exact_body(1)")
        assert not re.search(pattern, "jit_probe_cache_write(1)")
        assert not re.search(pattern, "prefix_jit_execute_body(1)")
    sealer = SCRIPT.read_text(encoding="utf-8")
    assert 'step_module_re=r"jit_execute_body"' not in sealer
    assert "step_module_re=_decode_step_module_re(exact_dsa=bool(args.exact_dsa))" in sealer


def _v2_remote_fixture(remote: Path, *, layouts: list[str] | None = None,
                       graph_sha_override: tuple[int, str, str] | None = None) -> dict:
    """Build a fake v2 acquisition prefix; returns the per-graph SHA records."""
    import gzip

    graph_records: dict[str, dict[str, str]] = {}
    for graph in ws32_evidence.GRAPHS:
        graph_records[graph] = {}
        for suffix, sha_key in ws32_evidence.HLO_FORMS:
            raw = f"{graph}:{suffix}\n".encode()
            graph_records[graph][sha_key] = sha256(raw).hexdigest()
            path = remote / "hlo" / f"{graph}.{suffix}.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(gzip.compress(raw, mtime=0))
    root = remote / "host_records"
    root.mkdir(parents=True, exist_ok=True)
    for rank in ws32_evidence.RANKS:
        graphs = json.loads(json.dumps(graph_records))
        if graph_sha_override is not None and rank == graph_sha_override[0]:
            graphs[graph_sha_override[1]][graph_sha_override[2]] = "9" * 64
        record = {
            "code_hash": "a" * 40,
            "compile_only": True,
            "evidence_layout": (
                ws32_evidence.EVIDENCE_LAYOUT_V2 if layouts is None else layouts[rank]
            ),
            "exact_dsa": False,
            "graphs": graphs,
            "launch_process_id": rank,
            "status": "HLO_ACQUIRED",
        }
        (root / f"runner.rank{rank}.json").write_text(json.dumps(record), encoding="utf-8")
        (root / f"runner.rank{rank}.log").write_text(f"rank={rank}\n", encoding="utf-8")
    return graph_records


def _materialize_fake(tmp_path: Path, remote: Path, monkeypatch: object, name: str):
    class FakeBlob:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.name = f"results/acquire/{path.relative_to(remote).as_posix()}"
            self.generation = 1
            self.size = path.stat().st_size
            self.crc32c = ws32_evidence._crc32c_file(path)

        def download_to_filename(self, destination: str, *, if_generation_match: int) -> None:
            shutil.copyfile(self.path, destination)

        def download_as_bytes(self, *, if_generation_match: int) -> bytes:
            return self.path.read_bytes()

    blobs = [FakeBlob(path) for path in remote.rglob("*") if path.is_file()]

    class FakeClient:
        def list_blobs(self, bucket: str, *, prefix: str) -> list[FakeBlob]:
            return blobs

    monkeypatch.setattr(ws32_evidence.storage, "Client", FakeClient)
    run_dir = tmp_path / name
    return lambda: ws32_evidence.materialize(
        run_dir=run_dir,
        remote_prefix="gs://unit/results/acquire",
        mode="acquire",
        tag="greenfield_ws32_short_decoder_8k_acquire_20260816T000000000000000Z",
        code_hash="a" * 40,
        recovery_code_hash="b" * 40,
        exact_dsa=False,
        allow_failure_diagnostics=False,
        output=run_dir / "source_remote_objects.json",
    )


def test_ws32_v2_materialization_refuses_every_drift(tmp_path: Path, monkeypatch: object) -> None:
    """Spec §23/storage plan: the single-copy HLO layout must fail closed on a
    missing or unexpected object, a layout disagreement, a per-rank SHA
    disagreement, and an inflated payload that differs from the recorded SHA."""
    import gzip

    # 1. missing object
    remote = tmp_path / "missing"
    _v2_remote_fixture(remote)
    (remote / "hlo" / f"{ws32_evidence.GRAPHS[0]}.stablehlo.mlir.gz").unlink()
    with pytest.raises(SystemExit, match="does not match layout"):
        _materialize_fake(tmp_path, remote, monkeypatch, "run_missing")()

    # 2. unexpected object
    remote = tmp_path / "extra"
    _v2_remote_fixture(remote)
    (remote / "hlo" / "rogue.stablehlo.mlir.gz").write_bytes(gzip.compress(b"x", mtime=0))
    with pytest.raises(SystemExit, match="does not match layout"):
        _materialize_fake(tmp_path, remote, monkeypatch, "run_extra")()

    # 3. ranks disagree on the layout
    remote = tmp_path / "layout"
    layouts = [ws32_evidence.EVIDENCE_LAYOUT_V2] * 8
    layouts[5] = ws32_evidence.EVIDENCE_LAYOUT_V1
    _v2_remote_fixture(remote, layouts=layouts)
    with pytest.raises(SystemExit, match="disagree on the evidence layout"):
        _materialize_fake(tmp_path, remote, monkeypatch, "run_layout")()

    # 4. ranks disagree on a graph SHA
    remote = tmp_path / "sha"
    _v2_remote_fixture(remote, graph_sha_override=(3, ws32_evidence.GRAPHS[1], "stablehlo_sha256"))
    with pytest.raises(SystemExit, match="disagree on"):
        _materialize_fake(tmp_path, remote, monkeypatch, "run_sha")()

    # 5. the compressed object inflates to bytes the records do not describe
    remote = tmp_path / "payload"
    _v2_remote_fixture(remote)
    target = remote / "hlo" / f"{ws32_evidence.GRAPHS[0]}.stablehlo.mlir.gz"
    target.write_bytes(gzip.compress(b"tampered\n", mtime=0))
    with pytest.raises(SystemExit, match="inflated HLO differs from the recorded SHA"):
        _materialize_fake(tmp_path, remote, monkeypatch, "run_payload")()

    # The unmutated fixture still materializes, and worker-0's own text is reused
    # by hard link rather than written a second time.
    remote = tmp_path / "good"
    records = _v2_remote_fixture(remote)
    run_dir = tmp_path / "run_good"
    (run_dir / "hlo").mkdir(parents=True)
    for graph in ws32_evidence.GRAPHS:
        for suffix, _ in ws32_evidence.HLO_FORMS:
            (run_dir / "hlo" / f"{graph}.{suffix}").write_bytes(f"{graph}:{suffix}\n".encode())
    result = _materialize_fake(tmp_path, remote, monkeypatch, "run_good")()
    assert result["evidence_layout"] == ws32_evidence.EVIDENCE_LAYOUT_V2
    for graph in ws32_evidence.GRAPHS:
        for suffix, sha_key in ws32_evidence.HLO_FORMS:
            worker = run_dir / "hlo" / f"{graph}.{suffix}"
            inflated = run_dir / "fleet_hlo" / f"{graph}.rank0.{suffix}"
            assert inflated.stat().st_ino == worker.stat().st_ino, (graph, suffix)
            assert sha256(inflated.read_bytes()).hexdigest() == records[graph][sha_key]


def test_the_sealer_states_loader_refusals_and_requires_a_committed_record() -> None:
    """P2-4 and P3: a loader ValueError is a refusal line, not a traceback."""
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[3]
    source = (root / "scripts/greenfield/seal_short_decoder_ws32.py").read_text(encoding="utf-8")
    assert "WS32 adjudication record is not loadable" in source
    assert "is not committed in the run's own pin" in source
    assert "differs from the blob committed at" in source
    assert "WS32 adjudication record is outside the repository" in source
    assert 'f"{args.code_hash}:{relative}"' in source, (
        "pre-registration is proven by the RUN's pin, not by HEAD, which moves after the run"
    )
    assert "HEAD:" not in source.split("def committed_in_run_pin")[1].split("\n\ndef ")[0]

    # Deleting the CALLS must fail this test, not just deleting the strings.
    import ast as _ast

    tree = _ast.parse(source)
    validate = next(
        node for node in tree.body
        if isinstance(node, _ast.FunctionDef) and node.name == "_validate"
    )
    calls = [
        node for node in _ast.walk(validate)
        if isinstance(node, _ast.Call)
        and isinstance(node.func, _ast.Name)
        and node.func.id in (
            "_committed_in_run_pin",
            "_committed_record_path",
            "_make_pre_registration_check",
        )
    ]
    called = {node.func.id for node in calls}
    assert called == {
        "_committed_in_run_pin",
        "_committed_record_path",
        "_make_pre_registration_check",
    }, (
        f"the pre-registration checks are not called: {sorted(called)}"
    )
    literals = {
        node.value for node in _ast.walk(validate)
        if isinstance(node, _ast.Constant) and isinstance(node.value, str)
    }
    assert {"adjudication analysis", "adjudication reference row"} <= literals, (
        "the analysis and the reference row must both be pinned to the run's commit"
    )
    attributes = {
        node.attr for node in _ast.walk(validate) if isinstance(node, _ast.Attribute)
    }
    assert {"analysis_path", "reference_row_path"} <= attributes

    import ast

    tree = ast.parse(source)
    validate = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate"
    )
    loads = [
        node for node in ast.walk(validate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_ws32_adjudicated_divergence"
    ]
    assert len(loads) == 1, "the record is loaded once"
    handlers = [
        node for node in ast.walk(validate)
        if isinstance(node, ast.Try)
        and any(loads[0] is item for item in ast.walk(node))
    ]
    assert handlers, "the load is not wrapped, so a refusal would be a traceback"
    caught = {
        name.id
        for handler in handlers
        for clause in handler.handlers
        for name in ast.walk(clause.type) if isinstance(name, ast.Name)
    }
    assert "ValueError" in caught


def test_the_sealed_gate_d_record_is_committed_in_its_own_run_pin() -> None:
    """The property the sealer now checks holds for the record that closed Gate D.

    Gate D sealed at pin 4286509; the adjudication record's blob in that commit
    is the blob on disk, so the record demonstrably existed before the run it
    judges. A record written afterwards cannot satisfy this.
    """
    import subprocess
    from pathlib import Path as _Path

    run_worktree = _Path("/home/gianl/glm-tpu-topology-rewrite")
    if not (run_worktree / ".git").exists():
        import pytest

        pytest.skip("the run worktree is unavailable")
    relative = "docs/artifacts/gate-d-ws32-8k-adjudicated-divergence-20260905.json"
    at_pin = subprocess.run(
        ["git", "-C", str(run_worktree), "rev-parse", f"4286509:{relative}"],
        capture_output=True, text=True,
    )
    on_disk = subprocess.run(
        ["git", "-C", str(run_worktree), "hash-object", "--", relative],
        capture_output=True, text=True,
    )
    assert at_pin.returncode == 0, at_pin.stderr
    assert at_pin.stdout.strip() == on_disk.stdout.strip() != ""


def _sealer_module():
    import importlib.util
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[3]
    path = root / "scripts/greenfield/seal_short_decoder_ws32.py"
    specification = importlib.util.spec_from_file_location("ws32_sealer_behaviour", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_the_rederivation_reproduces_the_sealed_gate_d_adjudication() -> None:
    """§21.2 items 3-4 are recomputed from the run being sealed.

    The numbers must be the sealed ones, on the real 8K run, or the seal-time
    re-derivation would refuse evidence that is already closed.
    """
    from pathlib import Path as _Path

    import numpy as _np
    import pytest as _pytest

    from glm_tpu.greenfield.validation.ws32_first_divergent_event import (
        adjudicate_first_divergent_event,
        event_arrays,
        oracle_event_arrays,
    )

    root = _Path(__file__).resolve().parents[3]
    oracle_dir = _Path(
        "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
        "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle"
    )
    archive = _Path(
        "/home/gianl/glm-run/greenfield_ws32_short_decoder_8k_numerical_"
        "20260905T085534575653049Z/runner.rank0.npz"
    )
    if not (archive.is_file() and (oracle_dir / "dsa_events.safetensors").is_file()):
        _pytest.skip("the sealed Gate D archive or the 8K DSA oracle is unavailable")
    from safetensors.numpy import load_file

    oracle = load_file(str(oracle_dir / "dsa_events.safetensors"))
    arrays = dict(_np.load(archive, allow_pickle=False))
    engine_positions, engine_scores = event_arrays(
        selected_positions=arrays["dsa_selected_positions"],
        selected_scores=arrays["dsa_selected_scores"],
        selected_valid_counts=arrays["dsa_selected_valid_counts"],
        step=0,
        event=1,
    )
    oracle_positions, oracle_scores = oracle_event_arrays(oracle, step=0, event=1)
    reference = _np.load(
        root / "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"
    ).astype(_np.float64)
    result = adjudicate_first_divergent_event(
        oracle_positions=oracle_positions,
        oracle_scores=oracle_scores,
        engine_positions=engine_positions,
        engine_scores=engine_scores,
        reference=reference,
        producer_layer_id=int(arrays["dsa_producer_layer_ids"][1]),
        step=0,
        event=1,
        decode_position=8155,
        expected_producer_layer_id=1,
    )
    assert result["verdict"] == "PASS"
    assert result["epsilon_oracle_vs_reference"] == 0.22697279652271618
    assert result["reference_band_size"] == 451
    assert result["shared_positions"] == 2041
    assert result["checks"]["bias"]["bound"] == 0.22563715920821653
    assert result["expected_only"] == [680, 1052, 2024, 2436, 6322, 7473, 7850]
    assert result["observed_only"] == [754, 1904, 2029, 3651, 4899, 5536, 6951]


def test_a_hand_asserted_pass_does_not_survive_the_rederivation(tmp_path) -> None:
    """A record whose analysis claims six passes is still recomputed."""
    import numpy as _np
    import pytest as _pytest

    module = _sealer_module()

    class _Adjudication:
        step = 0
        event_index = 0
        decode_position = 63
        producer_layer_id = 1
        expected_only = (31,)
        observed_only = (32,)
        reference_row_path = "docs/artifacts/gate-d-row.npy"

    reference = _np.sort(_np.arange(64, dtype=_np.float64))[::-1].copy()
    (tmp_path / "docs/artifacts").mkdir(parents=True)
    _np.save(tmp_path / "docs/artifacts/gate-d-row.npy", reference)

    top_k = 32
    oracle_positions = _np.arange(top_k, dtype=_np.int32)
    engine_positions = oracle_positions.copy()
    engine_positions[top_k - 1] = top_k

    class _Oracle:
        selected_positions = oracle_positions.reshape(1, 1, top_k)
        selected_scores = reference[:top_k].reshape(1, 1, top_k).astype(_np.float32)
        valid_counts = _np.full((1, 1), top_k, dtype=_np.int32)

    # The engine's scores are wildly biased, so the caps must fail no matter
    # what any analysis file asserts.
    engine_scores = (reference[engine_positions] + 50.0).astype(_np.float32)
    arrays = {
        "dsa_producer_layer_ids": _np.asarray([1], dtype=_np.int32),
        "dsa_selected_positions": engine_positions.reshape(1, 1, 1, top_k),
        "dsa_selected_scores": engine_scores.reshape(1, 1, 1, top_k),
        "dsa_selected_valid_counts": _np.full((1, 1, 1), top_k, dtype=_np.int32),
    }
    with _pytest.raises(SystemExit, match="items 3-4 fail on this run"):
        module._rederive_ws32_adjudication(
            arrays=arrays,
            oracle=_Oracle(),
            adjudication=_Adjudication(),
            repository_root=tmp_path,
            rank=0,
        )


def test_the_rederivation_refuses_a_divergence_the_run_did_not_produce(tmp_path) -> None:
    import numpy as _np
    import pytest as _pytest

    module = _sealer_module()
    generator = _np.random.RandomState(7)
    reference = _np.sort(generator.uniform(0.0, 1.0, size=64))[::-1].astype(_np.float64).copy()
    top_k = 32
    # Put the swapped pair inside the reference ambiguity band, so the event is
    # genuinely adjudicable and only the DECLARED sets are wrong.
    reference[top_k - 1] = reference[top_k] + 1e-6
    (tmp_path / "docs/artifacts").mkdir(parents=True)
    _np.save(tmp_path / "docs/artifacts/gate-d-row.npy", reference)
    oracle_positions = _np.arange(top_k, dtype=_np.int32)
    oracle_scores = reference[:top_k] + generator.normal(0.0, 1e-4, size=top_k)
    engine_positions = oracle_positions.copy()
    engine_positions[top_k - 1] = top_k

    class _Oracle:
        selected_positions = oracle_positions.reshape(1, 1, top_k)
        selected_scores = oracle_scores.reshape(1, 1, top_k).astype(_np.float32)
        valid_counts = _np.full((1, 1), top_k, dtype=_np.int32)

    class _Adjudication:
        step = 0
        event_index = 0
        decode_position = 63
        producer_layer_id = 1
        expected_only = (7,)          # not what the run produced
        observed_only = (99,)
        reference_row_path = "docs/artifacts/gate-d-row.npy"

    arrays = {
        "dsa_producer_layer_ids": _np.asarray([1], dtype=_np.int32),
        "dsa_selected_positions": engine_positions.reshape(1, 1, 1, top_k),
        "dsa_selected_scores": (
            reference[engine_positions] + generator.normal(0.0, 1e-4, size=top_k)
        )
        .reshape(1, 1, 1, top_k)
        .astype(_np.float32),
        "dsa_selected_valid_counts": _np.full((1, 1, 1), top_k, dtype=_np.int32),
    }
    with _pytest.raises(SystemExit, match="disagrees with the pre-registered divergence"):
        module._rederive_ws32_adjudication(
            arrays=arrays,
            oracle=_Oracle(),
            adjudication=_Adjudication(),
            repository_root=tmp_path,
            rank=0,
        )


def test_the_pre_registration_pin_refuses_an_uncommitted_or_drifted_artifact(tmp_path) -> None:
    """P1-2: the round's headline control, exercised rather than grepped."""
    import subprocess

    import pytest as _pytest

    module = _sealer_module()
    repository = tmp_path / "repo"
    (repository / "docs" / "artifacts").mkdir(parents=True)
    relative = "docs/artifacts/gate-d-record.json"
    target = repository / relative
    target.write_text('{"a": 1}', encoding="utf-8")
    for command in (
        ["init", "-q"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "record"],
    ):
        subprocess.run(["git", "-C", str(repository), *command], check=True, capture_output=True)
    pin = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    class _Args:
        code_hash = pin
        dsa_adjudication_record = target

    check = module._make_pre_registration_check(_Args(), repository)
    check(relative, "adjudication record")

    # Committed at the pin, but the working tree drifted.
    target.write_text('{"a": 2}', encoding="utf-8")
    with _pytest.raises(SystemExit, match="differs from the blob committed at"):
        check(relative, "adjudication record")
    target.write_text('{"a": 1}', encoding="utf-8")

    # Present on disk, absent from the pin.
    later = repository / "docs/artifacts/gate-d-later.json"
    later.write_text("{}", encoding="utf-8")
    with _pytest.raises(SystemExit, match="is not committed in the run's own pin"):
        check("docs/artifacts/gate-d-later.json", "adjudication record")

    # A pin that does not exist at all.
    class _Unknown:
        code_hash = "0" * 40

    unknown = module._make_pre_registration_check(_Unknown(), repository)
    with _pytest.raises(SystemExit, match="is not committed in the run's own pin"):
        unknown(relative, "adjudication record")


def test_the_pre_registration_pin_refuses_when_git_is_unavailable(tmp_path, monkeypatch) -> None:
    import subprocess

    import pytest as _pytest

    module = _sealer_module()

    class _Args:
        code_hash = "0" * 40

    check = module._make_pre_registration_check(_Args(), tmp_path)
    monkeypatch.setattr(module, "_GIT", str(tmp_path / "no-such-git"))
    with _pytest.raises(SystemExit, match="commitment cannot be checked"):
        check("docs/artifacts/gate-d-record.json", "adjudication record")


def test_the_grandfathered_record_is_re_derived_from_its_own_basis_row(tmp_path) -> None:
    """No record escapes the re-derivation, including the pre-amendment one."""
    import shutil
    from pathlib import Path as _Path

    import numpy as _np
    import pytest as _pytest

    module = _sealer_module()
    root = _Path(__file__).resolve().parents[3]
    oracle_dir = _Path(
        "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
        "greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle"
    )
    archive = _Path(
        "/home/gianl/glm-run/greenfield_ws32_short_decoder_8k_numerical_"
        "20260905T085534575653049Z/runner.rank0.npz"
    )
    if not (archive.is_file() and (oracle_dir / "dsa_events.safetensors").is_file()):
        _pytest.skip("the sealed Gate D archive or the 8K DSA oracle is unavailable")
    from safetensors.numpy import load_file

    relative = "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"
    (tmp_path / "docs/artifacts").mkdir(parents=True)
    shutil.copyfile(root / relative, tmp_path / relative)
    oracle_arrays = load_file(str(oracle_dir / "dsa_events.safetensors"))

    class _Oracle:
        selected_positions = oracle_arrays["selected_positions"]
        selected_scores = oracle_arrays["selected_scores"]
        valid_counts = oracle_arrays["valid_counts"]

    class _Grandfathered:
        step = 0
        event_index = 1
        context = "8k"
        decode_position = 8155
        producer_layer_id = 1
        expected_only = (680, 1052, 2024, 2436, 6322, 7473, 7850)
        observed_only = (754, 1904, 2029, 3651, 4899, 5536, 6951)
        # The pre-amendment record declares no `reference_row`; the loader
        # resolves it from the single `.npy` in the record's own basis.
        reference_row_path = "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"

    arrays = dict(_np.load(archive, allow_pickle=False))
    module._rederive_ws32_adjudication(
        arrays=arrays,
        oracle=_Oracle(),
        adjudication=_Grandfathered(),
        repository_root=tmp_path,
        rank=0,
    )

    class _Unregistered(_Grandfathered):
        reference_row_path = None

    with _pytest.raises(SystemExit, match="names no FP64 reference row"):
        module._rederive_ws32_adjudication(
            arrays=arrays,
            oracle=_Oracle(),
            adjudication=_Unregistered(),
            repository_root=tmp_path,
            rank=0,
        )


def _run_validate(tmp_path, **overrides):
    """Drive the sealer's `validate` over a real sealed run directory."""
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from ws32_validate_argv import SEALED_C512_RUN, available, build

    if not available():
        import pytest as _pytest

        _pytest.skip("the sealed C=512 run directory or an 8K oracle is unavailable")
    module = _sealer_module()
    # The enforcement-surface check is exercised directly by
    # test_an_adjudicated_seal_requires_a_clean_checkout; a tree being
    # edited is by definition modified while these tests are written.
    module._require_clean_worktree = lambda root: None
    repository_root = _Path(__file__).resolve().parents[3]
    argv = build(SEALED_C512_RUN, tmp_path / "summary.json", repository_root, **overrides)
    original = sys.argv
    try:
        sys.argv = ["seal"] + argv
        module.main()
    except SystemExit as error:
        return str(error)
    finally:
        sys.argv = original
    return None


def test_validate_reaches_the_run_records_with_the_genuine_adjudication(tmp_path) -> None:
    """The adjudication block must PASS on real evidence, not refuse it.

    The sealed C=512 run predates the current runner schema, so validation stops
    there; the point is that it gets that far, which it cannot do if any
    adjudication check refuses. Every refusal test below is anchored on this.
    """
    message = _run_validate(tmp_path)
    assert message is not None
    assert "runner schema drifted" in message, message


def test_validate_refuses_an_adjudication_record_outside_the_reviewed_directory(tmp_path) -> None:
    """The record-path guard, exercised rather than grepped.

    Nothing here writes into the reviewed evidence tree: a copy of the record
    outside the repository must be refused as well, and that is the branch the
    guard shares with an in-repository path outside docs/artifacts.
    """
    import shutil
    from pathlib import Path as _Path

    repository_root = _Path(__file__).resolve().parents[3]
    source = repository_root / "docs/artifacts/gate-d-ws32-8k-adjudicated-divergence-20260905.json"
    elsewhere = tmp_path / "gate-d-copied-record.json"
    shutil.copyfile(source, elsewhere)
    message = _run_validate(tmp_path, dsa_adjudication_record=str(elsewhere))
    assert message is not None
    assert "is outside the repository" in message, message

    module = _sealer_module()
    checker = module._committed_artifact_path
    assert checker("docs/artifacts/gate-d-x.json", ".json") is True
    assert checker("docs/greenfield/gate-d-x.json", ".json") is False


def test_validate_refuses_a_record_absent_from_the_runs_own_pin(tmp_path) -> None:
    """Pre-registration: the record must exist in the commit the run executed at."""
    message = _run_validate(tmp_path, code_hash="9" * 40)
    assert message is not None
    assert "is not committed in the run's own pin" in message, message


def test_validate_refuses_a_record_whose_declared_digest_is_wrong(tmp_path) -> None:
    """A record that is not the one the run was launched against is refused.

    Drift is exercised against a throwaway repository in
    test_the_pre_registration_pin_refuses_an_uncommitted_or_drifted_artifact;
    the reviewed evidence tree is never written to.
    """
    message = _run_validate(tmp_path, dsa_adjudication_sha256="0" * 64)
    assert message is not None
    assert "identity drifted" in message, message


def _stub_adjudication(**overrides):
    """A loaded record shaped like the sealed one, with fields overridden."""
    from dataclasses import dataclass

    @dataclass
    class _Stub:
        record_sha256: str = (
            "4da05468120e3c2e9b82d03931018e0d14eebc5fc28e339381658a04457cd26b"
        )
        step: int = 0
        event_index: int = 1
        expected_only: tuple = (680, 1052, 2024, 2436, 6322, 7473, 7850)
        observed_only: tuple = (754, 1904, 2029, 3651, 4899, 5536, 6951)
        context: str = "8k"
        oracle_dsa_manifest_sha256: str = (
            "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da"
        )
        decode_position: int = 8155
        producer_layer_id: int = 1
        later_event_alarm: int = 1024
        engine_source_run: str = (
            "greenfield_ws32_short_decoder_8k_numerical_20260827T011711674195301Z"
        )
        analysis_path: str | None = None
        reference_row_path: str | None = (
            "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"
        )

        def status(self, step, event):
            if (step, event) < (self.step, self.event_index):
                return "exact_required"
            if (step, event) == (self.step, self.event_index):
                return "adjudicated"
            return "recorded"

    return _Stub(**overrides)


def _run_validate_with_record(tmp_path, adjudication):
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from ws32_validate_argv import SEALED_C512_RUN, available, build

    if not available():
        import pytest as _pytest

        _pytest.skip("the sealed C=512 run directory or an 8K oracle is unavailable")
    module = _sealer_module()
    # The enforcement-surface check is exercised directly by
    # test_an_adjudicated_seal_requires_a_clean_checkout; a tree being
    # edited is by definition modified while these tests are written.
    module._require_clean_worktree = lambda root: None
    module.load_ws32_adjudicated_divergence = lambda *a, **k: adjudication
    repository_root = _Path(__file__).resolve().parents[3]
    argv = build(SEALED_C512_RUN, tmp_path / "summary.json", repository_root)
    original = sys.argv
    try:
        sys.argv = ["seal"] + argv
        module.main()
    except SystemExit as error:
        return str(error)
    finally:
        sys.argv = original
    return None


def test_validate_re_derives_and_refuses_a_divergence_the_run_did_not_produce(tmp_path) -> None:
    """The re-derivation is REACHED by the sealer, not merely defined.

    The stub keeps everything the sealed record says except the divergence,
    which is the one thing the run's own arrays decide.
    """
    message = _run_validate_with_record(
        tmp_path, _stub_adjudication(expected_only=(7,), observed_only=(9,))
    )
    assert message is not None
    assert "disagrees with the pre-registered divergence" in message, message


def test_validate_pins_the_analysis_and_the_reference_row_to_the_runs_commit(tmp_path) -> None:
    """The round-5 control, exercised through _validate rather than by grep."""
    for field in ("analysis_path", "reference_row_path"):
        message = _run_validate_with_record(
            tmp_path,
            _stub_adjudication(**{field: "docs/artifacts/gate-d-never-committed.json"}),
        )
        assert message is not None, field
        assert "is not committed in the run's own pin" in message, (field, message)


def test_the_rank0_archive_must_be_bound_to_its_own_runner_record(tmp_path) -> None:
    """The re-derivation reads rank 0's arrays; they must be the record's."""
    import numpy as _np
    import pytest as _pytest

    module = _sealer_module()
    run_dir = tmp_path / "run"
    (run_dir / "fleet").mkdir(parents=True)
    archive = run_dir / "fleet" / "runner.rank0.npz"
    _np.savez(
        archive,
        dsa_producer_layer_ids=_np.asarray([1], dtype=_np.int32),
        dsa_selected_positions=_np.zeros((1, 1, 1, 2), dtype=_np.int32),
        dsa_selected_scores=_np.zeros((1, 1, 1, 2), dtype=_np.float32),
        dsa_selected_valid_counts=_np.full((1, 1, 1), 2, dtype=_np.int32),
    )
    digest = module._digest_file(archive)
    size = archive.stat().st_size
    bound = {"byte_count": size, "filename": "runner.rank0.npz", "sha256": digest}
    arrays = module._rank0_dsa_arrays(run_dir, {"numerical_tensors": dict(bound)})
    assert set(arrays) == {
        "dsa_producer_layer_ids",
        "dsa_selected_positions",
        "dsa_selected_scores",
        "dsa_selected_valid_counts",
    }
    for record in (
        {"numerical_tensors": dict(bound, sha256="0" * 64)},
        {"numerical_tensors": dict(bound, filename="other.npz")},
        {"numerical_tensors": dict(bound, byte_count=size + 1)},
        {"numerical_tensors": None},
        {},
    ):
        with _pytest.raises(SystemExit, match="not bound to its record"):
            module._rank0_dsa_arrays(run_dir, record)

    # A missing file and a short key set are refusals, not tracebacks.
    with _pytest.raises(SystemExit, match="numerical tensor file is missing"):
        module._rank0_dsa_arrays(tmp_path / "absent", {"numerical_tensors": dict(bound)})
    short_run = tmp_path / "short"
    (short_run / "fleet").mkdir(parents=True)
    short_archive = short_run / "fleet" / "runner.rank0.npz"
    _np.savez(short_archive, dsa_producer_layer_ids=_np.zeros(1, dtype=_np.int32))
    short_record = {
        "numerical_tensors": {
            "byte_count": short_archive.stat().st_size,
            "filename": "runner.rank0.npz",
            "sha256": module._digest_file(short_archive),
        }
    }
    with _pytest.raises(SystemExit, match="numerical tensor keys missing"):
        module._rank0_dsa_arrays(short_run, short_record)


def test_an_acquisition_seal_carries_no_adjudication_record(tmp_path) -> None:
    """P3-6: an acquisition establishes graphs, never a correctness claim.

    The guard runs before the tag check so it is reachable, and so the refusal
    names the real problem rather than the tag it implies.
    """
    message = _run_validate(tmp_path, mode="acquire")
    assert message is not None
    assert "acquisition seals carry no adjudication record" in message, message


def test_validate_actually_reaches_the_pin_checks_and_the_re_derivation(tmp_path) -> None:
    """The anchor test's premise, instrumented rather than assumed.

    "It stops at the schema check" only proves the controls passed if they ran.
    This records which of them `_validate` actually invoked, on real evidence.
    """
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from ws32_validate_argv import SEALED_C512_RUN, available, build

    if not available():
        import pytest as _pytest

        _pytest.skip("the sealed C=512 run directory or an 8K oracle is unavailable")
    module = _sealer_module()
    # The enforcement-surface check is exercised directly by
    # test_an_adjudicated_seal_requires_a_clean_checkout; a tree being
    # edited is by definition modified while these tests are written.
    module._require_clean_worktree = lambda root: None
    reached = []

    original_rederive = module._rederive_ws32_adjudication

    def traced_rederive(**kwargs):
        reached.append(("rederive", kwargs["adjudication"].reference_row_path))
        return original_rederive(**kwargs)

    original_pin = module._make_pre_registration_check

    def traced_pin(args, repository_root):
        inner = original_pin(args, repository_root)

        def wrapper(relative, label):
            reached.append(("pin", label, relative))
            return inner(relative, label)

        return wrapper

    module._rederive_ws32_adjudication = traced_rederive
    module._make_pre_registration_check = traced_pin
    repository_root = _Path(__file__).resolve().parents[3]
    original_argv = sys.argv
    try:
        sys.argv = ["seal"] + build(
            SEALED_C512_RUN, tmp_path / "summary.json", repository_root
        )
        module.main()
    except SystemExit as error:
        message = str(error)
    finally:
        sys.argv = original_argv

    assert "runner schema drifted" in message, message
    row = "docs/artifacts/gate-d-event1-fp64-reference-row-20260905.npy"
    assert (
        "pin",
        "adjudication record",
        "docs/artifacts/gate-d-ws32-8k-adjudicated-divergence-20260905.json",
    ) in reached
    assert ("pin", "adjudication reference row", row) in reached
    assert ("rederive", row) in reached


def test_ranks_that_disagree_on_their_observations_are_refused() -> None:
    """P1-1: the eight-rank property rests on this, and nothing tested it."""
    import pytest as _pytest

    module = _sealer_module()
    rank0 = {"dsa_selected_scores": {"dtype": "float32", "sha256": "a" * 64, "shape": [1]}}
    records = [{"numerical_tensors": {"arrays": rank0}}]
    module._require_ranks_agree(rank0, records, rank=0)
    module._require_ranks_agree(dict(rank0), records, rank=5)
    drifted = {"dsa_selected_scores": {"dtype": "float32", "sha256": "b" * 64, "shape": [1]}}
    with _pytest.raises(SystemExit, match="numerical tensor values disagree rank 5"):
        module._require_ranks_agree(drifted, records, rank=5)
    # Rank 0 is the reference, so it is compared to nothing.
    module._require_ranks_agree(drifted, records, rank=0)


def test_every_rank_is_re_derived_not_only_rank_zero() -> None:
    """The re-derivation must be applied inside the per-rank loop as well."""
    import ast
    from pathlib import Path as _Path

    source = (
        _Path(__file__).resolve().parents[3]
        / "scripts/greenfield/seal_short_decoder_ws32.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    validate = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate"
    )
    loops = [node for node in ast.walk(validate) if isinstance(node, ast.For)]
    rank_loops = [
        node for node in loops
        if isinstance(node.target, ast.Tuple)
        and any(
            isinstance(name, ast.Name) and name.id == "rank"
            for name in ast.walk(node.target)
        )
    ]
    assert rank_loops, "the per-rank loop is gone"
    inside = [
        item
        for loop in rank_loops
        for item in ast.walk(loop)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "_rederive_ws32_adjudication"
    ]
    assert inside, (
        "the re-derivation runs only on rank 0; the other seven ranks would rest "
        "entirely on the cross-rank agreement check"
    )
    ranks = {ast.unparse(keyword.value) for call in inside for keyword in call.keywords
             if keyword.arg == "rank"}
    assert ranks == {"rank"}, f"the loop re-derivation must use each rank's own index: {ranks}"


def test_an_adjudicated_seal_requires_a_clean_checkout(tmp_path) -> None:
    """P2-4: the registry and the arithmetic are read from this working tree.

    An operator could widen `REFERENCE_ROWS` in place, seal, and revert. A
    clean-tree requirement puts any such edit into history.
    """
    import subprocess

    import pytest as _pytest

    module = _sealer_module()
    repository = tmp_path / "repo"
    surface = repository / "glm_tpu" / "greenfield" / "validation"
    surface.mkdir(parents=True)
    (surface / "ws32_short_context.py").write_text("REFERENCE_ROWS = {}\n", encoding="utf-8")
    (repository / "unrelated.txt").write_text("one", encoding="utf-8")
    for command in (
        ["init", "-q"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "one"],
    ):
        subprocess.run(["git", "-C", str(repository), *command], check=True, capture_output=True)

    module._require_clean_worktree(repository)

    # An edit to the registry is refused.
    (surface / "ws32_short_context.py").write_text(
        'REFERENCE_ROWS = {"fitted": 1}\n', encoding="utf-8"
    )
    with _pytest.raises(SystemExit, match="unmodified enforcement surface"):
        module._require_clean_worktree(repository)
    (surface / "ws32_short_context.py").write_text("REFERENCE_ROWS = {}\n", encoding="utf-8")

    # A new, untracked file inside the surface is refused too.
    (surface / "extra_registry.py").write_text("", encoding="utf-8")
    with _pytest.raises(SystemExit, match="unmodified enforcement surface"):
        module._require_clean_worktree(repository)
    (surface / "extra_registry.py").unlink()

    # An edit OUTSIDE the surface does not block a seal.
    (repository / "unrelated.txt").write_text("two", encoding="utf-8")
    module._require_clean_worktree(repository)
    with _pytest.raises(SystemExit, match="cannot verify the working tree"):
        module._require_clean_worktree(tmp_path / "not-a-repository")


def test_the_rederivation_is_given_the_records_decode_position() -> None:
    """P3-2: that argument is what stops a row built for another position."""
    import ast
    from pathlib import Path as _Path

    source = (
        _Path(__file__).resolve().parents[3]
        / "scripts/greenfield/seal_short_decoder_ws32.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_rederive_ws32_adjudication"
    )
    calls = [
        node for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "adjudicate_first_divergent_event"
    ]
    assert len(calls) == 1
    passed = {
        keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords
    }
    assert passed["decode_position"] == "adjudication.decode_position"
    assert passed["expected_producer_layer_id"] == "adjudication.producer_layer_id"


def test_the_rederivation_refuses_a_row_built_for_another_decode_position(tmp_path) -> None:
    """The behaviour that argument buys, not just its presence."""
    import numpy as _np
    import pytest as _pytest

    module = _sealer_module()
    reference = _np.linspace(1.0, 0.0, 64)
    (tmp_path / "docs/artifacts").mkdir(parents=True)
    _np.save(tmp_path / "docs/artifacts/gate-d-row.npy", reference)
    top_k = 32
    positions = _np.arange(top_k, dtype=_np.int32)

    class _Oracle:
        selected_positions = positions.reshape(1, 1, top_k)
        selected_scores = reference[:top_k].reshape(1, 1, top_k).astype(_np.float32)
        valid_counts = _np.full((1, 1), top_k, dtype=_np.int32)

    class _Adjudication:
        step = 0
        event_index = 0
        context = "8k"
        decode_position = 8155  # the row is 64 long, not 8156
        producer_layer_id = 1
        expected_only = (31,)
        observed_only = (32,)
        reference_row_path = "docs/artifacts/gate-d-row.npy"

    arrays = {
        "dsa_producer_layer_ids": _np.asarray([1], dtype=_np.int32),
        "dsa_selected_positions": positions.reshape(1, 1, 1, top_k),
        "dsa_selected_scores": reference[positions].reshape(1, 1, 1, top_k).astype(_np.float32),
        "dsa_selected_valid_counts": _np.full((1, 1, 1), top_k, dtype=_np.int32),
    }
    with _pytest.raises(SystemExit, match="does not match decode position"):
        module._rederive_ws32_adjudication(
            arrays=arrays,
            oracle=_Oracle(),
            adjudication=_Adjudication(),
            repository_root=tmp_path,
            rank=0,
        )
