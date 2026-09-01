from __future__ import annotations

import re
import subprocess
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).parents[3]
WRAPPER = ROOT / "scripts/greenfield/run_gate_d_forced_round_pp16_numerical.sh"


def test_wrapper_has_valid_bash_syntax_and_default_off_exactly_once_boundary() -> None:
    subprocess.run(["/usr/bin/bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text(encoding="ascii")
    assert "GLM_GATE_D_FORCED_ROUND_PP16_NUMERICAL:-0} == 1" in source
    assert "GLM_GATE_D_FORCED_ROUND_PP16_NUMERICAL_MODE:-off} == execute_once" in source
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
        'sha256sum "$ADMISSION"',
        'sha256sum "$TOPOLOGY"',
        'sha256sum "$HLO_ADJUDICATION"',
        'sha256sum "$ACCEPTED_STABLEHLO"',
        'sha256sum "$ACCEPTED_OPTIMIZED_HLO"',
        'sha256sum "$FORCED_ROUND_SOURCE"',
        'sha256sum "$CAPSULE"',
        'sha256sum "$CAPSULE_INPUTS"',
        'sha256sum "$CAPSULE_STATE"',
        'sha256sum "$CAPSULE_EXECUTION_AUTHORITY"',
        'sha256sum "$HOST_MATERIALIZATION_AUTHORITY"',
        'sha256sum "$DRIVER"',
        'sha256sum "$PUBLISHER"',
    ):
        assert source.index(authority) < driver
    assert '--accepted-stablehlo "$ACCEPTED_STABLEHLO"' in source
    assert '--accepted-optimized-hlo "$ACCEPTED_OPTIMIZED_HLO"' in source
    assert '--forced-round-source "$FORCED_ROUND_SOURCE"' in source
    assert source.index("strict_census pre") < driver
    assert source.index('"$PUBLISHER_PYTHON" -I -S -B "$MIRROR_VERIFIER"') < driver
    publisher = ROOT / "scripts/greenfield/publish_gate_d_forced_round_pp16_numerical.py"
    publisher_sha = sha256(publisher.read_bytes()).hexdigest()
    assert f"readonly PUBLISHER_SHA={publisher_sha}" in source
    driver_source = ROOT / "scripts/greenfield/run_gate_d_forced_round_pp16_numerical.py"
    driver_sha = sha256(driver_source.read_bytes()).hexdigest()
    assert f"readonly DRIVER_SHA={driver_sha}" in source
    mirror_verifier = (
        ROOT / "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
    )
    mirror_verifier_sha = sha256(mirror_verifier.read_bytes()).hexdigest()
    assert f"readonly MIRROR_VERIFIER_SHA={mirror_verifier_sha}" in source
    assert (
        "readonly IMMUTABLE_CAPSULE_ROOT=/usr/local/libexec/glm-tpu/"
        "gate-d-forced-round-pp16-numerical-v1"
    ) in source
    assert (
        "$WORKTREE/scripts/greenfield/run_gate_d_forced_round_pp16_numerical.py"
        not in source
    )
    assert (
        "$WORKTREE/scripts/greenfield/publish_gate_d_forced_round_pp16_numerical.py"
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
    assert (
        "REMOTE_PREFIX=$BUCKET/results/greenfield/glm52/"
        "gate_d_forced_round_pp16_numerical/$TAG"
        in source
    )
    assert "gcloud storage buckets describe" in source
    assert "US-CENTRAL2" in source


def test_wrapper_proves_all_remote_history_before_run_directory_creation() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    history = source.index("vacancy_live_output=")
    initialize = source.index("run_identity=$(publisher_init init --run-dir")
    execute = source.index("executing one exact-input PP16")
    assert history < initialize < execute
    assert '/snap/bin/gcloud storage ls "$REMOTE_PREFIX/**"' in source
    assert (
        '/snap/bin/gcloud storage ls --all-versions "$REMOTE_PREFIX/**"'
        in source
    )
    assert (
        "/snap/bin/gcloud storage ls --soft-deleted --exhaustive"
        in source
    )
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
