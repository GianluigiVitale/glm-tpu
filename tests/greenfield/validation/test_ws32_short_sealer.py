from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import bench.provenance as provenance
from glm_tpu.greenfield.validation import ws32_evidence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/greenfield/seal_short_decoder_ws32.py"
WRAPPER = ROOT / "scripts/greenfield/run_short_decoder_ws32.sh"
PROVISIONER = ROOT / "scripts/greenfield/provision_greenfield_worker.sh"
SPEC = importlib.util.spec_from_file_location("ws32_short_sealer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SEALER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEALER)


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
    assert "PROVISION_OK pin=$EXPECTED_PIN" in source
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
    assert len(acquired) == 80
    assert len(numerical) == 96
    exact_acquired = ws32_evidence._expected_primary_names(
        numerical=False, exact_dsa=True
    )
    exact_numerical = ws32_evidence._expected_primary_names(
        numerical=True, exact_dsa=True
    )
    assert len(exact_acquired) == 112
    assert len(exact_numerical) == 128
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
    graph_records: dict[str, dict[str, str]] = {}
    for graph in ws32_evidence.GRAPHS:
        graph_records[graph] = {}
        for suffix, sha_key in ws32_evidence.HLO_FORMS:
            raw = f"{graph}:{suffix}\n".encode()
            graph_records[graph][sha_key] = sha256(raw).hexdigest()
            for rank in ws32_evidence.RANKS:
                path = remote / "hlo" / f"{graph}.rank{rank}.{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
    for rank in ws32_evidence.RANKS:
        record = {
            "code_hash": "a" * 40,
            "compile_only": True,
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
    assert names == ws32_evidence._expected_primary_names(numerical=False)
    assert not any(name.endswith(".npz") for name in names)
    assert len(list((run_dir / "fleet_hlo").iterdir())) == 64


def test_ws32_prelaunch_floor_covers_protected_unique_evidence_and_reserve() -> None:
    # Current protected fleet: worker-0 local HLO/trace/NPZ are created before
    # materialization, then seven distinct traces and peer records remain to be
    # fetched. Keep at least one GiB free throughout sealing.
    rank0_generated = 311_539_059 + 277_949_672 + 4_952_000
    remaining_unique = 1_946_790_017 + 35_000_000
    sealing_reserve = 1024**3
    required = rank0_generated + remaining_unique + sealing_reserve
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
    assert (tmp_path / "source_remote_objects.json").read_text() == "source"
    quarantined = list((tmp_path / "recovery_failures").glob("*/summary.json"))
    assert len(quarantined) == 1 and quarantined[0].read_text() == "summary"


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
