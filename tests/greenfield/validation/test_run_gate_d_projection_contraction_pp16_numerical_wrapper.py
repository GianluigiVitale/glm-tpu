from __future__ import annotations

import json
import os
import re
import subprocess
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).parents[3]
WRAPPER = (
    ROOT / "scripts/greenfield/run_gate_d_projection_contraction_pp16_numerical.sh"
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _terminal_verifier() -> str:
    source = WRAPPER.read_text(encoding="ascii")
    start_marker = (
        "read -r -d '' NUMERICAL_RESULT_VERIFIER "
        "<<'NUMERICAL_RESULT_VERIFIER_EOF' || true\n"
    )
    start = source.index(start_marker) + len(start_marker)
    end = source.index("\nNUMERICAL_RESULT_VERIFIER_EOF", start)
    return source[start:end]


TAG = "gate_d_projection_contraction_pp16_numerical_20260901T235959123456789Z"
REMOTE = (
    "gs://driftbench-dsv4-uc/results/greenfield/glm52/"
    f"gate_d_projection_contraction_pp16_numerical/{TAG}"
)


def _result_fixture(
    status: str,
    *,
    terminal_mutation: dict[str, object] | None = None,
    receipt_mutation: dict[str, object] | None = None,
    generation: str = "1788300000000001",
) -> tuple[bytes, bytes, str]:
    """Return canonical terminal bytes, receipt bytes and their authority line."""
    marker: dict[str, object] = {
        "artifact_kind": "gate_d_projection_contraction_pp16_numerical_result",
        "evidence_sha256": "1" * 64,
        "gate_d_closed": False,
        "performance_claim": False,
        "remote_ledger": {
            "crc32c": "AAAAAA==",
            "generation": "1788300000000000",
            "path": "remote_objects.json",
            "sha256": "2" * 64,
            "size": 1234,
        },
        "root_cause_fix_proven": False,
        "run_tag": TAG,
        "status": status,
        "summary_sha256": "3" * 64,
        "tpu_numerical_execution_performed": True,
    }
    marker["marker_payload_sha256"] = sha256(_canonical(marker)).hexdigest()
    if terminal_mutation:
        marker.update(terminal_mutation)
        if "marker_payload_sha256" not in terminal_mutation:
            unsigned = dict(marker)
            del unsigned["marker_payload_sha256"]
            marker["marker_payload_sha256"] = sha256(_canonical(unsigned)).hexdigest()
    terminal_raw = _canonical(marker)
    receipt: dict[str, object] = {
        "artifact_kind": (
            "gate_d_projection_contraction_pp16_numerical_terminal_receipt"
        ),
        "remote": REMOTE + "/NUMERICAL_RESULT",
        "terminal": {
            "crc32c": "AAAAAA==",
            "generation": generation,
            "path": "NUMERICAL_RESULT",
            "sha256": sha256(terminal_raw).hexdigest(),
            "size": len(terminal_raw),
        },
    }
    if receipt_mutation:
        receipt.update(receipt_mutation)
    authority = (
        f"NUMERICAL_RESULT status={status} "
        f"marker_sha256={marker['marker_payload_sha256']} "
        f"terminal_generation={generation} "
        f"terminal_sha256={sha256(terminal_raw).hexdigest()}"
    )
    return terminal_raw, _canonical(receipt), authority


def _write_result_files(
    directory: Path, terminal_raw: bytes, receipt_raw: bytes
) -> None:
    """Install (or same-UID replace) the local terminal and receipt names."""
    for name, raw in (
        ("NUMERICAL_RESULT", terminal_raw),
        ("terminal_upload_receipt.json", receipt_raw),
    ):
        staged = directory / f".{name}.substitute"
        staged.write_bytes(raw)
        staged.chmod(0o400)
        os.replace(staged, directory / name)


def _run_terminal_verifier(
    tmp_path: Path,
    status: str,
    *,
    terminal_mutation: dict[str, object] | None = None,
    receipt_mutation: dict[str, object] | None = None,
) -> tuple[subprocess.CompletedProcess[bytes], str]:
    terminal_raw, receipt_raw, authority = _result_fixture(
        status,
        terminal_mutation=terminal_mutation,
        receipt_mutation=receipt_mutation,
    )
    _write_result_files(tmp_path, terminal_raw, receipt_raw)
    run_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        completed = subprocess.run(
            [
                "/usr/bin/python3",
                "-I",
                "-S",
                "-B",
                "-c",
                _terminal_verifier(),
                TAG,
                REMOTE,
                str(run_fd),
            ],
            env={},
            pass_fds=(run_fd,),
            check=False,
            capture_output=True,
        )
    finally:
        os.close(run_fd)
    return completed, authority


def _result_dispatch_segment() -> str:
    """The wrapper lines from publisher success through local consistency check."""
    source = WRAPPER.read_text(encoding="ascii")
    start = source.index("result_authority=$(publisher success --run-dir")
    end = source.index("terminal_written=1\n", start)
    return source[start:end]


def _run_result_dispatch(
    directory: Path,
    publisher_stdout: str,
    *,
    publisher_fails: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute the wrapper's exact dispatch segment against a stubbed publisher.

    The stub stands in for the immutable publisher process whose stdout is the
    only status authority; the run-directory names are opened through fd 7
    exactly as the wrapper does, so same-UID replacement after the stub returns
    is reproduced faithfully.
    """
    harness = "\n".join(
        [
            "set -euo pipefail",
            f"readonly TAG={TAG}",
            f"readonly REMOTE_PREFIX={REMOTE}",
            "readonly RUN_DIR=/nonexistent/run-dir",
            "elapsed=7",
            "publisher() {",
            '  [[ $1 == success ]] || { echo "unexpected publisher mode: $1" >&2; exit 97; }',
            "  [[ ${PUBLISHER_STUB_FAILS:-0} == 0 ]] || exit 1",
            "  /usr/bin/printf '%s' \"$PUBLISHER_STUB_STDOUT\"",
            "}",
            (
                "read -r -d '' NUMERICAL_RESULT_VERIFIER "
                "<<'NUMERICAL_RESULT_VERIFIER_EOF' || true"
            ),
            _terminal_verifier(),
            "NUMERICAL_RESULT_VERIFIER_EOF",
            'exec 7<"$FIXTURE_DIR"',
            _result_dispatch_segment(),
            "/usr/bin/printf 'RESULT %s\\n' \"$result_status\"",
        ]
    )
    return subprocess.run(
        ["/usr/bin/bash", "--noprofile", "--norc", "-c", harness],
        env={
            "FIXTURE_DIR": str(directory),
            "PUBLISHER_STUB_FAILS": "1" if publisher_fails else "0",
            "PUBLISHER_STUB_STDOUT": publisher_stdout,
        },
        check=False,
        capture_output=True,
        text=True,
    )


def test_wrapper_has_valid_bash_syntax_and_default_off_exactly_once_boundary() -> None:
    subprocess.run(["/usr/bin/bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text(encoding="ascii")
    assert "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_NUMERICAL:-0} == 1" in source
    assert (
        "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_NUMERICAL_MODE:-off} == execute_once"
        in source
    )
    assert source.count("--execute-once 1") == 1
    assert source.count('"$DRIVER_PYTHON" -I -S -B -u "$DRIVER"') == 1
    assert "JAX_ENABLE_COMPILATION_CACHE=0" in source
    assert "JAX_PLATFORMS=tpu" in source
    assert "TPU_PROCESS_BOUNDS=1,1,1" in source
    assert "TPU_VISIBLE_DEVICES=0,1,2,3" in source


def test_wrapper_serializes_tpu_and_rsync_and_requires_eight_host_census() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    assert 'names = ("glm_pod_workload.lock", "glm_tpu_rsync.lock")' in source
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in source
    assert "exec 9>/home/gianl/glm-run/.glm_pod_workload.lock" in source
    assert "exec 8>/home/gianl/.glm-tpu-rsync.lock" in source
    assert source.count("strict_census pre") == 1
    assert source.count("strict_census post") == 1
    assert "--worker=all" in source
    assert "has_eight_unique_markers" in source
    assert "RUNTIME_BOUNDARY_VERIFIER" in source
    assert 'getattr(fcntl, "F_GET_SEALS", 1034)' in source
    assert "exec 10<&-" in source


def test_wrapper_binds_all_numerical_authorities_before_driver_invocation() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    driver = source.index('"$DRIVER_PYTHON" -I -S -B -u "$DRIVER"')
    for authority in (
        'sha256sum "$TOPOLOGY"',
        'sha256sum "$HLO_SUCCESS_AUTHORITY"',
        'sha256sum "$PROJECTION_SOURCE"',
        'sha256sum "$FRONTIER_AUTHORITY"',
        'sha256sum "$CAPSULE"',
        'sha256sum "$CAPSULE_INPUTS"',
        'sha256sum "$CAPSULE_STATE"',
        'sha256sum "$CAPSULE_EXECUTION_AUTHORITY"',
        'sha256sum "$HOST_MATERIALIZATION_AUTHORITY"',
        'sha256sum "$DRIVER"',
        'sha256sum "$PUBLISHER"',
    ):
        assert source.index(authority) < driver
    assert "--accepted-stablehlo" not in source
    assert "--accepted-optimized-hlo" not in source
    assert "--hlo-source-location-bridge" not in source
    assert "ACCEPTED_OPTIMIZED_HLO" not in source
    assert '--projection-source "$PROJECTION_SOURCE"' in source
    assert '--frontier-authority "$FRONTIER_AUTHORITY"' in source
    assert '--hlo-success-authority "$HLO_SUCCESS_AUTHORITY"' in source
    assert source.index("strict_census pre") < driver
    assert source.index('"$PUBLISHER_PYTHON" -I -S -B "$MIRROR_VERIFIER"') < driver
    publisher = (
        ROOT
        / "scripts/greenfield/publish_gate_d_projection_contraction_pp16_numerical.py"
    )
    publisher_sha = sha256(publisher.read_bytes()).hexdigest()
    assert f"readonly PUBLISHER_SHA={publisher_sha}" in source
    driver_source = (
        ROOT / "scripts/greenfield/run_gate_d_projection_contraction_pp16_numerical.py"
    )
    driver_sha = sha256(driver_source.read_bytes()).hexdigest()
    assert f"readonly DRIVER_SHA={driver_sha}" in source
    mirror_verifier = (
        ROOT / "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
    )
    mirror_verifier_sha = sha256(mirror_verifier.read_bytes()).hexdigest()
    assert f"readonly MIRROR_VERIFIER_SHA={mirror_verifier_sha}" in source
    assert (
        "readonly IMMUTABLE_CAPSULE_ROOT=/usr/local/libexec/glm-tpu/"
        "gate-d-projection-contraction-pp16-numerical-v3"
    ) in source
    assert (
        "$WORKTREE/scripts/greenfield/run_gate_d_projection_contraction_pp16_numerical.py"
        not in source
    )
    assert (
        "$WORKTREE/scripts/greenfield/publish_gate_d_projection_contraction_pp16_numerical.py"
        not in source
    )
    assert (
        "$WORKTREE/scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
        not in source
    )


def test_wrapper_set_u_local_assignments_have_no_same_command_dependencies() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("local "):
            continue
        declared = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)=", stripped)
        for name in declared:
            assert re.search(rf"\$(?:\{{{name}\}}|{name}\b)", stripped) is None

    lines = source.splitlines()
    start = lines.index("strict_census() {")
    assignments = lines[start + 1 : start + 4]
    probe = "\n".join(
        [
            "set -u",
            "TAG=exact-tag",
            "strict_census() {",
            *assignments,
            'printf \'%s %s %s\\n\' "$label" "$member" "$carrier"',
            "}",
            "strict_census pre",
        ]
    )
    completed = subprocess.run(
        ["/usr/bin/bash", "--noprofile", "--norc", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == "pre census_pre.txt exact-tag_pre\n"


def test_wrapper_has_no_mutable_path_self_reexec() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    assert "IMMUTABLE_LOCK_BROKER" not in source
    assert "WRAPPER_ABS" not in source
    assert "exec /usr/bin/python3" not in source
    assert "/proc/self/fd/10" not in source
    assert '/usr/bin/python3 -I -S -B -c "$RUNTIME_BOUNDARY_VERIFIER"' in source


def test_wrapper_archives_both_numerical_outcomes_without_gate_d_claim() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    publication = source.index("publisher success --run-dir")
    assert source.index("strict_census post") < publication
    assert source.index("NUMERICAL_ACCEPTED archive=") > publication
    assert source.index("NUMERICAL_REJECTED archive=") > publication
    assert source.count("gate_d_open=true") == 2
    assert "NUMERICAL_RESULT_VERIFIER" in source
    assert "! -e /proc/self/fd/7/NUMERICAL_RESULT" in source
    assert "$RUN_DIR/NUMERICAL_ACCEPTED" not in source
    assert "$RUN_DIR/NUMERICAL_REJECTED" not in source
    assert (
        "REMOTE_PREFIX=$BUCKET/results/greenfield/glm52/"
        "gate_d_projection_contraction_pp16_numerical/$TAG" in source
    )
    assert "gcloud storage buckets describe" in source
    assert "US-CENTRAL2" in source


def test_terminal_verifier_executes_both_result_outcomes(tmp_path: Path) -> None:
    for status in ("NUMERICAL_ACCEPTED", "NUMERICAL_REJECTED"):
        outcome = tmp_path / status
        outcome.mkdir()
        completed, authority = _run_terminal_verifier(outcome, status)
        assert completed.returncode == 0, completed.stderr
        assert authority.startswith(f"NUMERICAL_RESULT status={status} marker_sha256=")
        assert completed.stdout == f"{authority}\n".encode()
        assert completed.stderr == b""


def test_terminal_verifier_rejects_claim_status_self_hash_and_receipt_attacks(
    tmp_path: Path,
) -> None:
    attacks = (
        ({"gate_d_closed": True}, None),
        ({"status": "NUMERICAL_UNKNOWN"}, None),
        ({"marker_payload_sha256": "0" * 64}, None),
        (None, {"remote": "gs://driftbench-dsv4-uc/wrong/NUMERICAL_RESULT"}),
        (
            None,
            {
                "terminal": {
                    "crc32c": "AAAAAA==",
                    "generation": "1788300000000001",
                    "path": "NUMERICAL_RESULT",
                    "sha256": "0" * 64,
                    "size": 1,
                }
            },
        ),
    )
    for index, (terminal_mutation, receipt_mutation) in enumerate(attacks):
        outcome = tmp_path / str(index)
        outcome.mkdir()
        completed, _ = _run_terminal_verifier(
            outcome,
            "NUMERICAL_ACCEPTED",
            terminal_mutation=terminal_mutation,
            receipt_mutation=receipt_mutation,
        )
        assert completed.returncode != 0
        assert completed.stdout == b""


def test_result_status_dispatches_only_from_direct_publisher_authority() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    publication = source.index("result_authority=$(publisher success --run-dir")
    local_check = source.index(
        '/usr/bin/python3 -I -S -B -c "$NUMERICAL_RESULT_VERIFIER"', publication
    )
    mismatch = source.index("Gate-D numerical local/remote authority mismatch")
    written = source.index("terminal_written=1", publication)
    dispatch = source.index('case "$result_status" in')
    assert publication < local_check < mismatch < written < dispatch
    assert source.count("result_status=") == 1
    assert "readonly result_status=${BASH_REMATCH[1]}" in source
    assert source.count('[[ $local_result_authority == "$result_authority" ]]') == 1
    assert "Gate-D numerical publisher authority drifted" in source
    pattern = (
        "^NUMERICAL_RESULT\\ status=(NUMERICAL_ACCEPTED|NUMERICAL_REJECTED)"
        "\\ marker_sha256=([0-9a-f]{64})\\ terminal_generation=([0-9]+)"
        "\\ terminal_sha256=([0-9a-f]{64})$"
    )
    assert pattern in source
    # The verifier may deny, but no local name may feed the dispatched status.
    verifier = _terminal_verifier()
    assert "print(status)" not in verifier
    assert 'f"NUMERICAL_RESULT status={status} marker_sha256={self_hash} "' in verifier


def test_result_dispatch_executes_both_outcomes_from_publisher_authority(
    tmp_path: Path,
) -> None:
    for status in ("NUMERICAL_ACCEPTED", "NUMERICAL_REJECTED"):
        outcome = tmp_path / status
        outcome.mkdir()
        terminal_raw, receipt_raw, authority = _result_fixture(status)
        _write_result_files(outcome, terminal_raw, receipt_raw)
        completed = _run_result_dispatch(outcome, authority + "\n")
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == f"RESULT {status}\n"
        assert completed.stderr == ""


def test_local_substitution_after_publisher_return_cannot_flip_result(
    tmp_path: Path,
) -> None:
    """Sol P1 regression: same-UID replacement of both local terminal names.

    After the immutable publisher has replayed the remote terminal and returned
    its direct authority, an attacker replaces ``NUMERICAL_RESULT`` and its
    receipt with self-consistent canonical files carrying the opposite status.
    The local verifier accepts the replaced pair on its own, so the wrapper must
    deny rather than dispatch the flipped status.
    """
    for genuine_status, flipped_status in (
        ("NUMERICAL_ACCEPTED", "NUMERICAL_REJECTED"),
        ("NUMERICAL_REJECTED", "NUMERICAL_ACCEPTED"),
    ):
        outcome = tmp_path / genuine_status
        outcome.mkdir()
        genuine_terminal, genuine_receipt, authority = _result_fixture(genuine_status)
        _write_result_files(outcome, genuine_terminal, genuine_receipt)
        flipped_terminal, flipped_receipt, flipped_authority = _result_fixture(
            flipped_status
        )
        assert flipped_authority != authority
        # Same-UID substitution after the trusted publisher exited.
        _write_result_files(outcome, flipped_terminal, flipped_receipt)
        verified, _ = _run_terminal_verifier(outcome, flipped_status)
        assert verified.returncode == 0
        assert verified.stdout == f"{flipped_authority}\n".encode()
        completed = _run_result_dispatch(outcome, authority + "\n")
        assert completed.returncode == 2
        assert completed.stdout == ""
        assert "Gate-D numerical local/remote authority mismatch" in completed.stderr
        assert flipped_status not in completed.stdout


def test_receipt_generation_substitution_after_publisher_return_denies(
    tmp_path: Path,
) -> None:
    genuine_terminal, genuine_receipt, authority = _result_fixture("NUMERICAL_ACCEPTED")
    _write_result_files(tmp_path, genuine_terminal, genuine_receipt)
    replayed_terminal, replayed_receipt, replayed_authority = _result_fixture(
        "NUMERICAL_ACCEPTED", generation="1788300000000002"
    )
    assert replayed_terminal == genuine_terminal
    assert replayed_authority != authority
    _write_result_files(tmp_path, replayed_terminal, replayed_receipt)
    completed = _run_result_dispatch(tmp_path, authority + "\n")
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "Gate-D numerical local/remote authority mismatch" in completed.stderr


def test_result_dispatch_rejects_non_authority_publisher_output(
    tmp_path: Path,
) -> None:
    terminal_raw, receipt_raw, authority = _result_fixture("NUMERICAL_ACCEPTED")
    _write_result_files(tmp_path, terminal_raw, receipt_raw)
    drifted_outputs = (
        "NUMERICAL_ACCEPTED\n",
        "",
        authority.replace("NUMERICAL_ACCEPTED", "NUMERICAL_UNKNOWN") + "\n",
        authority.replace(
            "terminal_generation=1788300000000001", "terminal_generation="
        )
        + "\n",
        authority + "\nextra line\n",
        "prefix\n" + authority + "\n",
        authority.upper() + "\n",
    )
    for publisher_stdout in drifted_outputs:
        completed = _run_result_dispatch(tmp_path, publisher_stdout)
        assert completed.returncode == 2, publisher_stdout
        assert completed.stdout == ""
        assert "Gate-D numerical publisher authority drifted" in completed.stderr
    failed = _run_result_dispatch(tmp_path, authority + "\n", publisher_fails=True)
    assert failed.returncode == 1
    assert failed.stdout == ""


def test_wrapper_proves_all_remote_history_before_run_directory_creation() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    history = source.index("vacancy_live_output=")
    initialize = source.index("run_identity=$(publisher_init init --run-dir")
    execute = source.index("executing one exact-input PP16")
    assert history < initialize < execute
    assert '/snap/bin/gcloud storage ls "$REMOTE_PREFIX/**"' in source
    assert '/snap/bin/gcloud storage ls --all-versions "$REMOTE_PREFIX/**"' in source
    assert "/snap/bin/gcloud storage ls --soft-deleted --exhaustive" in source
    assert source.count('!= "$VACANCY_EXPECTED"') == 3
    assert "PYTHONWARNINGS=ignore" in source
    assert "scope=all_versions flags=--all-versions returncode=1" in source
    assert "scope=soft_deleted flags=--soft-deleted,--exhaustive returncode=1" in source


def test_wrapper_contains_no_destructive_or_host_staged_model_path() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    forbidden = (
        "rm -rf",
        "gs://driftbench-storage",
        "ray.put",
        "ray.get",
        "gcsfuse",
        "git reset",
        "git checkout",
    )
    assert not any(token in source for token in forbidden)


def test_wrapper_git_reads_ignore_configuration_replacements_and_network() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    assert source.count("git_local() {") == 1
    for binding in (
        "GIT_CONFIG_GLOBAL=/dev/null",
        "GIT_CONFIG_NOSYSTEM=1",
        "GIT_NO_LAZY_FETCH=1",
        "GIT_NO_REPLACE_OBJECTS=1",
        "GIT_OPTIONAL_LOCKS=0",
        "GIT_PROTOCOL_FROM_USER=0",
        "GIT_SSH_COMMAND=/bin/false",
        "GIT_TERMINAL_PROMPT=0",
        "core.fsmonitor=false",
        "core.untrackedCache=false",
        "core.attributesFile=/dev/null",
    ):
        assert binding in source
    assert source.count("git_local ") == 5
    assert "GIT_AUTHORITY_VERIFIER" in source
    assert 'ls-remote", "--refs", origin, expected_ref' in source
