"""Hostile contracts for the protected exact-M2048 fingerprint harness."""

from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[3]
DRIVER = ROOT / "scripts/greenfield/probe_m2048_strategy_nd_association.py"
PUBLISHER = ROOT / (
    "scripts/greenfield/publish_gate_d_m2048_strategy_nd_association.py"
)
WRAPPER = ROOT / (
    "scripts/greenfield/run_gate_d_m2048_strategy_nd_association.sh"
)
LAUNCHER = ROOT / (
    "scripts/greenfield/launch_gate_d_m2048_strategy_nd_association.py"
)
INSTALLER = ROOT / (
    "scripts/greenfield/install_gate_d_m2048_strategy_nd_runtime.py"
)
FLEET_INSTALLER = ROOT / (
    "scripts/greenfield/install_gate_d_m2048_strategy_nd_fleet.sh"
)
BOOTSTRAP = ROOT / "scripts/greenfield/bootstrap_gate_d_provisioner.py"
REPO_REFRESHER = ROOT / (
    "scripts/greenfield/refresh_gate_d_m2048_worker_repository.py"
)
MIRROR = ROOT / "scripts/greenfield/verify_gate_d_rewrite_same_region_git_mirror.py"
CERTIFICATE = ROOT / "docs/artifacts/gate-d-m2048-strategy-nd-v3-source.json"


def _load_publisher():
    specification = importlib.util.spec_from_file_location(
        "gate_d_m2048_publisher_for_test", PUBLISHER
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


PUBLISHER_MODULE = _load_publisher()


def _load_bootstrap():
    specification = importlib.util.spec_from_file_location(
        "gate_d_m2048_bootstrap_for_test", BOOTSTRAP
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


BOOTSTRAP_MODULE = _load_bootstrap()


def _load_repo_refresher():
    specification = importlib.util.spec_from_file_location(
        "gate_d_m2048_repo_refresher_for_test", REPO_REFRESHER
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


REPO_REFRESHER_MODULE = _load_repo_refresher()


def _encoded(value: object) -> str:
    return base64.b64encode(PUBLISHER_MODULE._canonical(value)).decode("ascii")


def _provenance(fd: int = 41) -> dict[str, object]:
    root = f"/proc/self/fd/{fd}"
    return {
        "import_closure": [
            {"module": "glm_tpu", "path": f"{root}/glm_tpu/__init__.py"},
            {
                "module": "glm_tpu.greenfield",
                "path": f"{root}/glm_tpu/greenfield/__init__.py",
            },
        ],
        "probe_sha256": "a" * 64,
        "python": deepcopy(PUBLISHER_MODULE.EXPECTED_PYTHON_PROVENANCE),
        "sites": deepcopy(PUBLISHER_MODULE.EXPECTED_SITE_PROVENANCE),
        "source_archive": {
            "archive_sha256": "b" * 64,
            "file_manifest_count": 71,
            "file_manifest_sha256": "c" * 64,
        },
        "sys_path": [
            root,
            PUBLISHER_MODULE.JAX_SITE_ROOT,
            PUBLISHER_MODULE.LIBTPU_SITE_ROOT,
            *PUBLISHER_MODULE.EXPECTED_RUNTIME_PATH,
        ],
    }


def test_driver_preserves_graph_and_output_before_secondary_validation() -> None:
    source = DRIVER.read_text(encoding="ascii")
    assert source.index("M2048_ACQUISITION process=0") < source.index(
        "validate_m2048_strategy_nd_fingerprint_hlo("
    )
    output = source.index("M2048_OUTPUT process=0")
    assert output < source.index("fleet_input = _fleet_digest(")
    assert output < source.index("fleet_output = _fleet_digest(")
    assert output < source.index("analysis = analyze_m2048_row0_association(")
    assert source.index("gate-d-m2048-output-acquired") < source.index(
        "fleet_input = _fleet_digest("
    )


def test_driver_has_no_mutable_result_path_or_cloud_transport() -> None:
    source = DRIVER.read_text(encoding="ascii")
    for forbidden in (
        "--output",
        "--run-dir",
        "google.cloud",
        "gcloud storage",
        "gs://",
        "ray.",
    ):
        assert forbidden not in source
    assert "sys.path[:] = [" in source
    assert "sealed_runtime.verify_import_closure(" in source
    assert '"fleet_host_transfer_bytes": 25165824' not in source
    assert "M2048_RECORD process=" in source


def test_marker_parser_requires_canonical_unique_records() -> None:
    lines = []
    for rank in reversed(range(8)):
        lines.append(
            f"M2048_RECORD process={rank} "
            f"json_b64={_encoded({'jax_process_index': rank})}"
        )
    raw = ("\n".join(lines) + "\n").encode("ascii")
    assert [item["jax_process_index"] for item in PUBLISHER_MODULE._record_markers(raw)] == list(
        range(8)
    )
    with pytest.raises(RuntimeError, match="one record per process"):
        PUBLISHER_MODULE._record_markers(raw + lines[0].encode("ascii") + b"\n")
    with pytest.raises(RuntimeError, match="one record per process"):
        PUBLISHER_MODULE._record_markers(b"\n".join(raw.splitlines()[:-1]))


def test_marker_parser_rejects_duplicate_keys_and_noncanonical_json() -> None:
    duplicate = base64.b64encode(b'{"a":1,"a":1}').decode("ascii")
    with pytest.raises(RuntimeError, match="invalid hostile"):
        PUBLISHER_MODULE._decode_json_b64(duplicate, label="hostile")
    noncanonical = base64.b64encode(b'{"b":2, "a":1}').decode("ascii")
    with pytest.raises(RuntimeError, match="noncanonical hostile"):
        PUBLISHER_MODULE._decode_json_b64(noncanonical, label="hostile")


def test_single_marker_rejects_missing_duplicate_and_corruption() -> None:
    value = {"output_bits_sha256": "d" * 64}
    line = f"M2048_OUTPUT process=0 json_b64={_encoded(value)}\n".encode("ascii")
    assert PUBLISHER_MODULE._single_marker(line, "M2048_OUTPUT") == value
    for hostile in (b"", line + line, line.replace(b"json_b64=", b"json_b64=%%%")):
        with pytest.raises(RuntimeError, match="exactly one M2048_OUTPUT"):
            PUBLISHER_MODULE._single_marker(hostile, "M2048_OUTPUT")


def test_provenance_normalization_accepts_only_memfd_number_drift() -> None:
    expected_archive = _provenance()["source_archive"]
    first = PUBLISHER_MODULE._normalized_provenance(
        _provenance(41),
        expected_probe_sha256="a" * 64,
        expected_source_archive=expected_archive,
    )
    second = PUBLISHER_MODULE._normalized_provenance(
        _provenance(97),
        expected_probe_sha256="a" * 64,
        expected_source_archive=expected_archive,
    )
    assert first == second
    assert first["sys_path"][0] == "<SEALED_SOURCE>"

    mutations = []
    for path, value in (
        (("probe_sha256",), "d" * 64),
        (("python", "python_sha256"), "e" * 64),
        (("sites", "jax", "tree_sha256"), "f" * 64),
        (("source_archive", "archive_sha256"), "0" * 64),
        (("sys_path", 1), "/tmp/mutable"),
        (("import_closure", 0, "path"), "/tmp/glm_tpu/__init__.py"),
    ):
        hostile = _provenance()
        target = hostile
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        mutations.append(hostile)
    for hostile in mutations:
        with pytest.raises(RuntimeError):
            PUBLISHER_MODULE._normalized_provenance(
                hostile,
                expected_probe_sha256="a" * 64,
                expected_source_archive=expected_archive,
            )


def test_publisher_source_binds_driver_and_loads_project_only_for_success() -> None:
    source = PUBLISHER.read_text(encoding="ascii")
    assert f'"{DRIVER.relative_to(ROOT)}"' in source
    assert '_git_bytes("show", f"{code_pin}:{DRIVER_SOURCE_PATH}")' in source
    main = source[source.index("def main()") :]
    assert main.index('elif arguments.mode == "success":') < main.index(
        "api = _load_project_api("
    )
    assert main.count("api = _load_project_api(") == 1
    assert "base._require_never_used_prefix(bucket, prefix)" in source
    assert source.index("base._require_never_used_prefix(bucket, prefix)") < source.index(
        "for relative in sorted(payload)"
    )


def test_committed_source_archive_identity_is_reproducible() -> None:
    pin = PUBLISHER_MODULE._git_bytes("rev-parse", "HEAD").decode().strip()
    first_path, first_identity = PUBLISHER_MODULE._sealed_source_archive(pin)
    second_path, second_identity = PUBLISHER_MODULE._sealed_source_archive(pin)
    assert first_identity == second_identity
    assert first_identity["file_manifest_count"] > 0
    assert len(first_identity["archive_sha256"]) == 64
    assert sha256(Path(first_path).read_bytes()).hexdigest() == first_identity[
        "archive_sha256"
    ]
    assert sha256(Path(second_path).read_bytes()).hexdigest() == second_identity[
        "archive_sha256"
    ]


def test_npy_is_nonpickle_little_endian_uint16() -> None:
    import numpy as np

    value = np.arange(12, dtype=np.uint16).reshape(2, 6)
    raw = PUBLISHER_MODULE._npy(value)
    parsed = np.load(__import__("io").BytesIO(raw), allow_pickle=False)
    assert parsed.dtype.str == "<u2"
    assert np.array_equal(parsed, value)


def test_wrapper_is_full_pod_default_off_and_terminal_last() -> None:
    source = WRAPPER.read_text(encoding="ascii")
    assert "Gate-D M2048 fingerprint is default-off" in source
    assert "GLM_GATE_D_M2048_MODE=execute_once" in source
    assert source.count("--worker=all") >= 3
    assert '--num-processes 8 --process-id "$idx"' in source
    assert "idx=${HOSTNAME##*-w-}" in source
    assert "M2048_OK" in source and "has_eight_unique_markers" in source
    assert source.index("strict_census pre") < source.index(
        "executing one exact-M2048 full-pod fingerprint"
    )
    assert source.index("strict_census post") < source.index(
        "publisher success --run-dir"
    )
    assert source.count("vllm-env") == 1
    execute = source[source.index("execute_command=") : source.index(
        "probe_status=(", source.index("execute_command=")
    )]
    assert "PYTHONPATH" not in execute
    assert "gcloud storage" not in execute
    assert "--output" not in execute and "--run-dir" not in execute
    assert "JAX_ENABLE_COMPILATION_CACHE=0" in execute
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION=.20" in execute
    assert source.rstrip().endswith(
        '"$result_authority" "M2048_ASSOCIATION_UNIQUE gate_d_open=true"'
    )


def test_capsule_launcher_installer_hash_chain_is_exact() -> None:
    wrapper = WRAPPER.read_text(encoding="ascii")
    launcher = LAUNCHER.read_text(encoding="ascii")
    installer = INSTALLER.read_text(encoding="ascii")
    values = {
        DRIVER: sha256(DRIVER.read_bytes()).hexdigest(),
        PUBLISHER: sha256(PUBLISHER.read_bytes()).hexdigest(),
        WRAPPER: sha256(WRAPPER.read_bytes()).hexdigest(),
        LAUNCHER: sha256(LAUNCHER.read_bytes()).hexdigest(),
        MIRROR: sha256(MIRROR.read_bytes()).hexdigest(),
    }
    assert values[DRIVER] in wrapper and values[DRIVER] in launcher
    assert values[PUBLISHER] in wrapper and values[PUBLISHER] in launcher
    assert values[MIRROR] in wrapper and values[MIRROR] in launcher
    assert values[WRAPPER] in launcher
    assert values[DRIVER] in installer
    assert values[PUBLISHER] in installer
    assert values[MIRROR] in installer
    assert values[LAUNCHER] in installer
    for placeholder in (
        "__PROBE_SHA256__",
        "__PUBLISHER_SHA256__",
        "__MIRROR_VERIFIER_SHA256__",
        "__WRAPPER_SHA256__",
        "__LAUNCHER_SHA256__",
    ):
        assert placeholder not in "".join((wrapper, launcher, installer))


def test_launcher_and_installer_preserve_immutable_boundaries() -> None:
    launcher = LAUNCHER.read_text(encoding="ascii")
    installer = INSTALLER.read_text(encoding="ascii")
    for token in (
        "os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING",
        "F_ADD_SEALS",
        "LOCK_NAMES",
        "os.O_NOFOLLOW",
        "os.execve(",
        "_create_retained_run_fd",
        'f"/proc/self/fd/{WRAPPER_FD}"',
    ):
        assert token in launcher
    assert '"status", "--porcelain=v1", "--untracked-files=all"' in launcher
    assert '"for-each-ref", "--format=%(refname)", "refs/replace"' in launcher
    assert "_RENAME_NOREPLACE = 1" in installer
    assert "target.exists() or target.is_symlink()" in installer
    assert '"launcher_invoked": False' in installer
    assert "invoke only the exact root-owned isolated M2048 installer" in installer


def test_mirror_verifier_install_path_matches_m2048_capsule() -> None:
    def load(name: str, path: Path):
        specification = importlib.util.spec_from_file_location(name, path)
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module

    launcher = load("gate_d_m2048_launcher_path_test", LAUNCHER)
    installer = load("gate_d_m2048_installer_path_test", INSTALLER)
    mirror = load("gate_d_m2048_mirror_path_test", MIRROR)
    expected = launcher.CAPSULE_ROOT / MIRROR.name
    assert installer.CAPSULE_TARGET == launcher.CAPSULE_ROOT
    assert mirror.INSTALL_PATH == expected
    assert set(mirror.BOUND_PATHS) >= {
        "docs/artifacts/gate-d-m2048-strategy-nd-source.json",
        "docs/artifacts/gate-d-m2048-strategy-nd-v2-source.json",
        "docs/artifacts/gate-d-m2048-strategy-nd-v3-source.json",
        "docs/artifacts/gate-d-m2048-v2-install-repository-prestate-failure.json",
        "scripts/greenfield/bootstrap_gate_d_provisioner.py",
        "scripts/greenfield/install_gate_d_m2048_strategy_nd_fleet.sh",
        "scripts/greenfield/install_gate_d_m2048_strategy_nd_runtime.py",
        "scripts/greenfield/launch_gate_d_m2048_strategy_nd_association.py",
        "scripts/greenfield/probe_m2048_strategy_nd_association.py",
        "scripts/greenfield/publish_gate_d_m2048_strategy_nd_association.py",
        "scripts/greenfield/refresh_gate_d_m2048_worker_repository.py",
        "scripts/greenfield/run_gate_d_m2048_strategy_nd_association.sh",
        "tests/greenfield/validation/test_m2048_strategy_nd_protected_harness.py",
    }


def test_fleet_installer_is_install_only_and_restores_exact_runtime_trees() -> None:
    source = FLEET_INSTALLER.read_text(encoding="ascii")
    assert "fleet installation is default-off" in source
    assert "GLM_GATE_D_M2048_INSTALL_MODE=install_only" in source
    assert "launcher_invoked=false" in source
    assert "import jax" not in source and "JAX_PLATFORMS" not in source
    assert "driftbench" not in source and "gcloud storage" not in source
    assert '--worker=1-7' in source and '--worker=all' in source
    assert "All traffic remains inside the TPU pod's us-central2-b placement" in source
    for value in (
        "308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616",
        "55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df",
        "db7598c867f370756813cbf1536ad8ef7b1d9c167975e9e1724bd9b4fee78eca",
        "0451c126799ffd189d537bd4aab2fe91c3e784d91828449713589a7fddb4a133",
    ):
        assert value in source
    assert '[[ -d $path && $path == /home/gianl/gate-d-m2048-runtime-source-' in source
    assert '/usr/bin/rm -rf -- "$path"' in source
    assert '/usr/bin/install -m 0555 -o root -g root' not in source
    assert 'git_local show "$pin:scripts/greenfield/$bootstrap"' in source
    assert "os.memfd_create('gate-d-provisioner-bootstrap'" in source
    assert "F_ADD_SEALS" in source and "F_GET_SEALS" in source
    assert "--root-bootstrap" in BOOTSTRAP.read_text(encoding="ascii")
    assert "RENAME_NOREPLACE" in BOOTSTRAP.read_text(encoding="ascii")
    assert "refusing to remove an unowned staging inode" in BOOTSTRAP.read_text(
        encoding="ascii"
    )
    assert "run_root=/home/gianl/gate-d-runs" in source
    assert '"directory:gianl:gianl:700"' in source


def test_fleet_serializes_exact_worker_repository_refresh() -> None:
    source = FLEET_INSTALLER.read_text(encoding="ascii")
    assert "WORKER_REPO_PRESTATE_PIN=086d459a6acf4e3d1ec328e00e3e1b61be29ad2b" in source
    assert "REPO_REFRESHER_B64=$(git_local show" in source
    assert "os.memfd_create('gate-d-m2048-repo-refresher'" in source
    assert "for worker in 1 2 3 4 5 6 7; do" in source
    assert source.count('--worker="$worker"') == 1
    assert "has_unique_markers \"$REPORT\" REPO_REFRESH_OK 7" in source
    assert source.index("for worker in 1 2 3 4 5 6 7; do") < source.index(
        '--command="$sync_command"'
    )


def test_repository_refresher_moves_only_exact_clean_detached_prestate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def git(path: Path, *arguments: str) -> str:
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(path), *arguments],
            check=True,
            capture_output=True,
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "HOME": str(tmp_path),
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
            },
            text=True,
        )
        return result.stdout.strip()

    origin = tmp_path / "origin.git"
    seed = tmp_path / "seed"
    worker = tmp_path / "worker"
    subprocess.run(
        ["/usr/bin/git", "init", "--bare", str(origin)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["/usr/bin/git", "init", str(seed)], check=True, capture_output=True
    )
    git(seed, "config", "user.name", "Gate D test")
    git(seed, "config", "user.email", "gate-d@example.invalid")
    (seed / "payload.txt").write_text("prestate\n", encoding="ascii")
    git(seed, "add", "payload.txt")
    git(seed, "commit", "-m", "prestate")
    git(seed, "branch", "-M", REPO_REFRESHER_MODULE.BRANCH)
    git(seed, "remote", "add", "origin", str(origin))
    git(seed, "push", "-u", "origin", REPO_REFRESHER_MODULE.BRANCH)
    git(origin, "symbolic-ref", "HEAD", f"refs/heads/{REPO_REFRESHER_MODULE.BRANCH}")
    subprocess.run(
        [
            "/usr/bin/git",
            "clone",
            "--branch",
            REPO_REFRESHER_MODULE.BRANCH,
            str(origin),
            str(worker),
        ],
        check=True,
        capture_output=True,
    )
    prestate = git(worker, "rev-parse", "HEAD")
    git(worker, "switch", "--detach", prestate)
    (seed / "payload.txt").write_text("target\n", encoding="ascii")
    git(seed, "add", "payload.txt")
    git(seed, "commit", "-m", "target")
    git(seed, "push", "origin", REPO_REFRESHER_MODULE.BRANCH)
    target = git(seed, "rev-parse", "HEAD")
    config_path = worker / ".git" / "config"
    config_sha = sha256(config_path.read_bytes()).hexdigest()
    config_mode = config_path.stat().st_mode & 0o777
    tmp_path.chmod(0o755)
    safe_common = tmp_path / "safe-common"
    safe_common.mkdir()
    for name in REPO_REFRESHER_MODULE.SAFE_COMMON_SUBDIRECTORIES:
        child = safe_common / name
        child.mkdir()
        child.chmod(0o555)
    (safe_common / "config").write_bytes(REPO_REFRESHER_MODULE.SAFE_COMMON_CONFIG)
    (safe_common / "config").chmod(0o444)
    safe_common.chmod(0o555)
    compile(
        REPO_REFRESHER_MODULE._ROOT_SAFE_COMMON_PROGRAM,
        "gate-d-root-safe-common",
        "exec",
    )
    switch_environment = REPO_REFRESHER_MODULE._safe_git_environment(
        worker, safe_common, ssh_command="/bin/false"
    )
    assert switch_environment["GIT_COMMON_DIR"] == str(safe_common)
    assert switch_environment["GIT_DIR"] == str(worker / ".git")
    assert switch_environment["GIT_OBJECT_DIRECTORY"] == str(
        worker / ".git" / "objects"
    )
    assert switch_environment["GIT_WORK_TREE"] == str(worker)

    def refresh(path: Path, **kwargs: object) -> None:
        REPO_REFRESHER_MODULE.refresh_repository(
            path,
            safe_common_directory=safe_common,
            prepare_safe_common_directory=False,
            expected_safe_common_uid=os.getuid(),
            expected_safe_common_gid=os.getgid(),
            **kwargs,
        )

    refresh(
        worker,
        target_pin=target,
        prestate_pin=prestate,
        origin=str(origin),
        allow_file_transport=True,
        expected_config_sha256=config_sha,
        expected_config_mode=config_mode,
    )
    assert git(worker, "rev-parse", "HEAD") == target
    assert git(worker, "branch", "--show-current") == ""
    refresh(
        worker,
        target_pin=target,
        prestate_pin=prestate,
        origin=str(origin),
        allow_file_transport=True,
        expected_config_sha256=config_sha,
        expected_config_mode=config_mode,
    )

    (worker / "untracked.txt").write_text("refuse\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="exact clean detached contract"):
        refresh(
            worker,
            target_pin=target,
            prestate_pin=prestate,
            origin=str(origin),
            allow_file_transport=True,
            expected_config_sha256=config_sha,
            expected_config_mode=config_mode,
        )

    (worker / "untracked.txt").unlink()
    worker_include = tmp_path / "worker-include"
    worker_attributes = tmp_path / "worker-attributes"
    worker_race = tmp_path / "worker-race"
    for clone in (worker_include, worker_attributes, worker_race):
        subprocess.run(
            [
                "/usr/bin/git",
                "clone",
                "--branch",
                REPO_REFRESHER_MODULE.BRANCH,
                str(origin),
                str(clone),
            ],
            check=True,
            capture_output=True,
        )
        git(clone, "switch", "--detach", target)

    include_marker = tmp_path / "include-filter-executed"
    include = tmp_path / "hostile-filter.inc"
    include.write_text(
        f'[filter "owned"]\n\tsmudge = /usr/bin/touch {include_marker}\n',
        encoding="ascii",
    )
    include_config = worker_include / ".git" / "config"
    include_safe_sha = sha256(include_config.read_bytes()).hexdigest()
    include_mode = include_config.stat().st_mode & 0o777
    git(worker_include, "config", "include.path", str(include))
    with pytest.raises(RuntimeError, match="local config is outside the sealed schema"):
        refresh(
            worker_include,
            target_pin=target,
            prestate_pin=target,
            origin=str(origin),
            allow_file_transport=True,
            expected_config_sha256=include_safe_sha,
            expected_config_mode=include_mode,
        )
    assert not include_marker.exists()

    race_marker = tmp_path / "race-filter-executed"
    race_config = worker_race / ".git" / "config"
    race_config_raw = race_config.read_bytes()
    race_config_sha = sha256(race_config_raw).hexdigest()
    race_config_mode = race_config.stat().st_mode & 0o777
    (seed / "payload.txt").write_text("race target\n", encoding="ascii")
    git(seed, "add", "payload.txt")
    git(seed, "commit", "-m", "race target")
    git(seed, "push", "origin", REPO_REFRESHER_MODULE.BRANCH)
    race_target = git(seed, "rev-parse", "HEAD")
    real_git = REPO_REFRESHER_MODULE._git
    attack_phases: list[str] = []

    def racing_git(path: Path, *arguments: str, **kwargs: object) -> bytes:
        if arguments and arguments[0] in {"fetch", "switch"}:
            attack_phases.append(arguments[0])
            hostile = race_config_raw + (
                f'\n[filter "owned"]\n\tsmudge = /usr/bin/touch {race_marker}'
                f'\n\tprocess = /usr/bin/touch {race_marker}\n'
                f'[url "ext::/usr/bin/touch {race_marker}"]\n'
                f'\tinsteadOf = {origin}\n'
                '[protocol "ext"]\n\tallow = always\n'
            ).encode("ascii")
            race_config.write_bytes(hostile)
            info_attributes = worker_race / ".git" / "info" / "attributes"
            info_attributes.write_text("* filter=owned\n", encoding="ascii")
            (worker_race / ".gitattributes").write_text(
                "* filter=owned\n", encoding="ascii"
            )
            try:
                return real_git(path, *arguments, **kwargs)
            finally:
                race_config.write_bytes(race_config_raw)
                info_attributes.unlink()
                (worker_race / ".gitattributes").unlink()
        return real_git(path, *arguments, **kwargs)

    monkeypatch.setattr(REPO_REFRESHER_MODULE, "_git", racing_git)
    refresh(
        worker_race,
        target_pin=race_target,
        prestate_pin=target,
        origin=str(origin),
        allow_file_transport=True,
        expected_config_sha256=race_config_sha,
        expected_config_mode=race_config_mode,
    )
    monkeypatch.setattr(REPO_REFRESHER_MODULE, "_git", real_git)
    assert git(worker_race, "rev-parse", "HEAD") == race_target
    assert attack_phases == ["fetch", "switch"]
    assert not race_marker.exists()

    filter_marker = tmp_path / "attribute-filter-executed"
    git(
        worker_attributes,
        "config",
        "filter.owned.smudge",
        f"/usr/bin/touch {filter_marker}",
    )
    git(
        worker_attributes,
        "config",
        "filter.owned.process",
        f"/usr/bin/touch {filter_marker}",
    )
    hostile_config = worker_attributes / ".git" / "config"
    hostile_config_sha = sha256(hostile_config.read_bytes()).hexdigest()
    hostile_config_mode = hostile_config.stat().st_mode & 0o777
    (seed / ".gitattributes").write_text("payload.txt filter=owned\n", encoding="ascii")
    (seed / "payload.txt").write_text("attribute target\n", encoding="ascii")
    git(seed, "add", ".gitattributes", "payload.txt")
    git(seed, "commit", "-m", "attribute target")
    git(seed, "push", "origin", REPO_REFRESHER_MODULE.BRANCH)
    attribute_target = git(seed, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="target tree contains .gitattributes"):
        refresh(
            worker_attributes,
            target_pin=attribute_target,
            prestate_pin=target,
            origin=str(origin),
            allow_file_transport=True,
            expected_config_sha256=hostile_config_sha,
            expected_config_mode=hostile_config_mode,
        )
    assert git(worker_attributes, "rev-parse", "HEAD") == target
    assert not filter_marker.exists()


def test_bootstrap_sealed_payload_survives_source_path_replacement(
    tmp_path: Path,
) -> None:
    original = b"reviewed committed provisioner bytes\n"
    source = tmp_path / "provisioner.py"
    source.write_bytes(original)
    descriptor = BOOTSTRAP_MODULE._sealed_memfd("replacement-race", source.read_bytes())
    try:
        source.unlink()
        source.write_bytes(b"hostile replacement\n")
        sealed_path = Path(f"/proc/self/fd/{descriptor}")
        assert BOOTSTRAP_MODULE._read_sealed_path(
            sealed_path, sha256(original).hexdigest()
        ) == original
        with pytest.raises(OSError):
            os.write(descriptor, b"mutate")
    finally:
        os.close(descriptor)


def test_bootstrap_source_and_fleet_hash_are_exact() -> None:
    import re

    source = FLEET_INSTALLER.read_text(encoding="ascii")
    expected = re.search(r"^readonly BOOTSTRAP_SHA=([0-9a-f]{64})$", source, re.M)
    assert expected is not None
    assert sha256(BOOTSTRAP.read_bytes()).hexdigest() == expected.group(1)
    helper = BOOTSTRAP.read_text(encoding="ascii")
    assert 'PROVISIONER_REPO_PATH = "scripts/greenfield/provision_gate_d_python_runtime.py"' in helper
    assert "_read_sealed_path(payload_path, PROVISIONER_SHA256)" in helper
    assert "_publish_provisioner(payload, PROVISIONER_SHA256)" in helper


def test_bootstrap_publication_is_no_replace_and_accepts_exact_race_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    raw = b"reviewed root provisioner\n"
    digest = sha256(raw).hexdigest()
    parent = tmp_path / "exact-race"
    parent.mkdir(mode=0o755)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    original_rename = BOOTSTRAP_MODULE._rename_noreplace
    try:
        BOOTSTRAP_MODULE._publish_exact_file(
            parent_fd,
            "provisioner.py",
            raw,
            digest,
            owner_uid=os.getuid(),
            owner_gid=os.getgid(),
        )
        assert (parent / "provisioner.py").read_bytes() == raw
        BOOTSTRAP_MODULE._publish_exact_file(
            parent_fd,
            "provisioner.py",
            raw,
            digest,
            owner_uid=os.getuid(),
            owner_gid=os.getgid(),
        )
    finally:
        os.close(parent_fd)

    race_parent = tmp_path / "winner-race"
    race_parent.mkdir(mode=0o755)
    race_fd = os.open(race_parent, os.O_RDONLY | os.O_DIRECTORY)

    def exact_winner(source: str, target: str, *, parent_fd: int) -> None:
        descriptor = os.open(
            target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o555, dir_fd=parent_fd
        )
        try:
            os.write(descriptor, raw)
            os.fchmod(descriptor, 0o555)
        finally:
            os.close(descriptor)
        raise FileExistsError(errno.EEXIST, "exact concurrent winner", target)

    monkeypatch.setattr(BOOTSTRAP_MODULE, "_rename_noreplace", exact_winner)
    try:
        BOOTSTRAP_MODULE._publish_exact_file(
            race_fd,
            "provisioner.py",
            raw,
            digest,
            owner_uid=os.getuid(),
            owner_gid=os.getgid(),
        )
        assert (race_parent / "provisioner.py").read_bytes() == raw
        assert not any(".tmp-" in item.name for item in race_parent.iterdir())
    finally:
        os.close(race_fd)
        monkeypatch.setattr(
            BOOTSTRAP_MODULE, "_rename_noreplace", original_rename
        )


def test_bootstrap_cleanup_refuses_replaced_staging_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = b"reviewed root provisioner\n"
    parent = tmp_path / "hostile-cleanup"
    parent.mkdir(mode=0o755)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)

    def replace_staging(source: str, _target: str, *, parent_fd: int) -> None:
        os.unlink(source, dir_fd=parent_fd)
        descriptor = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o555,
            dir_fd=parent_fd,
        )
        try:
            os.write(descriptor, b"unowned replacement\n")
        finally:
            os.close(descriptor)
        raise RuntimeError("synthetic rename failure")

    monkeypatch.setattr(BOOTSTRAP_MODULE, "_rename_noreplace", replace_staging)
    try:
        with pytest.raises(RuntimeError, match="unowned staging inode"):
            BOOTSTRAP_MODULE._publish_exact_file(
                parent_fd,
                "provisioner.py",
                raw,
                sha256(raw).hexdigest(),
                owner_uid=os.getuid(),
                owner_gid=os.getgid(),
            )
        replacements = [item for item in parent.iterdir() if ".tmp-" in item.name]
        assert len(replacements) == 1
        assert replacements[0].read_bytes() == b"unowned replacement\n"
    finally:
        os.close(parent_fd)


def test_sealed_fleet_entry_ignores_replacement_before_bash_open(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "hostile-fleet-ran"
    source = tmp_path / "fleet-installer.sh"
    reviewed = FLEET_INSTALLER.read_bytes()
    source.write_bytes(reviewed)
    descriptor = BOOTSTRAP_MODULE._sealed_memfd("fleet-entry-race", source.read_bytes())
    try:
        source.unlink()
        source.write_text(
            f"#!/usr/bin/bash\n/usr/bin/touch {marker}\n", encoding="ascii"
        )
        result = subprocess.run(
            [
                "/usr/bin/bash",
                "--noprofile",
                "--norc",
                f"/proc/self/fd/{descriptor}",
            ],
            check=False,
            capture_output=True,
            env={
                "HOME": "/home/gianl",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/snap/bin:/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            pass_fds=(descriptor,),
            text=True,
        )
        assert result.returncode == 2
        assert "fleet installation is default-off" in result.stderr
        assert not marker.exists()
    finally:
        os.close(descriptor)


def test_certificate_entry_command_authenticates_and_seals_fleet_blob() -> None:
    record = json.loads(CERTIFICATE.read_text(encoding="ascii"))
    command = record["install_only_command_after_approved_persistence"]
    fleet_sha = sha256(FLEET_INSTALLER.read_bytes()).hexdigest()
    assert fleet_sha in command
    assert "git_tpu-topology-rewrite" not in command
    assert "git_env=" in command
    assert "GIT_NO_REPLACE_OBJECTS" in command
    assert "show" in command
    assert "scripts/greenfield/install_gate_d_m2048_strategy_nd_fleet.sh" in command
    assert "os.memfd_create" in command
    assert "F_ADD_SEALS" in command and "F_GET_SEALS" in command
    assert 'os.execve("/usr/bin/bash"' in command
    assert (
        "/usr/bin/bash --noprofile --norc "
        "/home/gianl/glm-tpu-topology-rewrite/scripts/greenfield/"
        "install_gate_d_m2048_strategy_nd_fleet.sh"
    ) not in command
    argv = shlex.split(command.replace("<FUTURE_MERGED_PIN>", "0" * 40))
    code_index = argv.index("-c") + 1
    compile(argv[code_index], "<sealed-fleet-entry>", "exec")


def test_install_source_tree_digest_matches_fleet_installer() -> None:
    import re
    import struct

    source = FLEET_INSTALLER.read_text(encoding="ascii")
    expected = re.search(r"^readonly INSTALL_SOURCE_TREE=([0-9a-f]{64})$", source, re.M)
    assert expected is not None
    payloads = (INSTALLER, LAUNCHER, DRIVER, PUBLISHER, MIRROR)
    digest = sha256()
    relative = b"."
    digest.update(b"D")
    digest.update(struct.pack(">I", len(relative)))
    digest.update(relative)
    digest.update(struct.pack(">I", 0o755))
    for path in sorted(payloads, key=lambda item: item.name):
        raw = path.read_bytes()
        relative = path.name.encode("ascii")
        digest.update(b"F")
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">I", 0o555))
        digest.update(struct.pack(">Q", len(raw)) + sha256(raw).digest())
    assert digest.hexdigest() == expected.group(1)


def _synthetic_success_fixture(monkeypatch: pytest.MonkeyPatch):
    import numpy as np

    code_pin = "1" * 40
    run_tag = "greenfield_m2048_strategy_nd_20260904T123456123456789Z"
    probe_raw = b"synthetic committed probe"
    input_bits = np.zeros((32, 32, 6144), dtype=np.uint16)
    output_bits = np.zeros((32, 6144), dtype=np.uint16)

    def array_sha256(value):
        return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()

    input_sha = array_sha256(input_bits)
    output_sha = array_sha256(output_bits)
    optimized = "HloModule synthetic_m2048\n"
    stable = "module @synthetic_m2048 {}\n"
    optimized_sha = sha256(optimized.encode()).hexdigest()
    stable_sha = sha256(stable.encode()).hexdigest()
    analysis = {
        "column_candidate_count_histogram": {"1": 6144},
        "every_lane_has_exactly_one_candidate": True,
        "uncovered_column_count": 0,
    }
    analysis_sha = sha256(PUBLISHER_MODULE._canonical(analysis)).hexdigest()
    config = {"rows": 2048, "seed": 0x4D323038, "trials": 32, "width": 6144}
    source_identity = {"kind": "synthetic-test-double"}
    report = {"status": "synthetic-hlo-valid"}
    algorithm = {"name": "StrategyND"}
    topology = {"schema": "synthetic-topology"}
    source_archive = {
        "archive_sha256": "b" * 64,
        "file_manifest_count": 71,
        "file_manifest_sha256": "c" * 64,
    }
    capture = {
        "collective_input_shape": [2048, 6144],
        "determinism_repeat_invocations": 32,
        "fleet_host_transfer_bytes": 25165824,
        "full_output_device_resident": True,
        "input_row0_bits_sha256": input_sha,
        "input_rows_1_through_2047_zero": True,
        "invocation_count": 64,
        "local_replica_row0_sha256_by_trial": [
            [array_sha256(output_bits[trial])] * 4 for trial in range(32)
        ],
        "measured_trial_invocations": 32,
        "output_row0_bits_sha256": output_sha,
        "repeated_local_replica_row0_sha256_by_trial": [
            [array_sha256(output_bits[trial])] * 4 for trial in range(32)
        ],
        "repeated_output_row0_bits_sha256": output_sha,
        "retained_output_artifact_bytes": 393216,
        "row0_slice_after_collective": True,
        "this_process_host_transfer_bytes": 3145728,
    }
    fleet_ids = [list(range(index * 4, index * 4 + 4)) for index in range(8)]
    records = []
    for rank in range(8):
        provenance = _provenance(40 + rank)
        provenance["probe_sha256"] = sha256(probe_raw).hexdigest()
        provenance["source_archive"] = deepcopy(source_archive)
        records.append(
            {
                "analysis_sha256": analysis_sha,
                "capture": capture,
                "captured_utc": "2026-09-04T12:34:56+00:00",
                "code_hash": code_pin,
                "collective_algorithm": algorithm,
                "config": config,
                "diagnostic_only": True,
                "fleet_analysis_sha256s": [analysis_sha] * 8,
                "fleet_hlo_sha256s": [optimized_sha] * 8,
                "fleet_input_sha256s": [input_sha] * 8,
                "fleet_local_device_ids_in_runtime_order": fleet_ids,
                "fleet_output_sha256s": [output_sha] * 8,
                "fleet_stablehlo_sha256s": [stable_sha] * 8,
                "gate_d_closed": False,
                "hlo": report,
                "hostname": f"db-v4-64-od-w-{rank}",
                "jax_process_index": rank,
                "jax_version": "0.10.1",
                "member_device_ids": list(range(32)),
                "optimized_hlo_sha256": optimized_sha,
                "performance_claim": False,
                "provenance": provenance,
                "run_tag": run_tag,
                "schema_version": 1,
                "source_identity": source_identity,
                "stablehlo_sha256": stable_sha,
                "topology": topology,
                "topology_hash": PUBLISHER_MODULE.EXPECTED_TOPOLOGY_HASH,
            }
        )
    acquisition = {
        "optimized_hlo_b64": base64.b64encode(optimized.encode()).decode(),
        "optimized_hlo_sha256": optimized_sha,
        "stablehlo_b64": base64.b64encode(stable.encode()).decode(),
        "stablehlo_sha256": stable_sha,
    }
    output = {
        "output_bits_b64": base64.b64encode(output_bits.tobytes()).decode(),
        "output_bits_sha256": output_sha,
    }

    def raw_log(current_records=records):
        lines = [
            f"M2048_ACQUISITION process=0 json_b64={_encoded(acquisition)}",
            f"M2048_OUTPUT process=0 json_b64={_encoded(output)}",
        ]
        lines.extend(
            f"M2048_RECORD process={rank} json_b64={_encoded(record)}"
            for rank, record in enumerate(current_records)
        )
        return ("\n".join(lines) + "\n").encode()

    members = {
        "census_post.txt": b"".join(
            f"CENSUS_OK db-v4-64-od-w-{rank}\n".encode() for rank in range(8)
        ),
        "census_pre.txt": b"".join(
            f"CENSUS_OK db-v4-64-od-w-{rank}\n".encode() for rank in range(8)
        ),
        "distributed.raw.log": raw_log(),
        "mirror.sha256": b"mirror\n",
        "orchestrator.log": b"orchestrator\n",
        "publisher_runtime.json": b"runtime\n",
        "remote_vacancy.raw.txt": b"raw vacancy\n",
        "remote_vacancy.txt": b"vacancy\n",
        "runner.log": b"runner\n",
        "sync.txt": b"sync\n",
    }

    class FakeBase:
        @staticmethod
        def snapshot_member(_run_fd, name, **_kwargs):
            return members[name]

        @staticmethod
        def write_member_exclusive(_run_fd, name, raw):
            if name in members:
                raise AssertionError(f"duplicate synthetic member: {name}")
            members[name] = raw

        @staticmethod
        def local_members(_run_fd):
            return set(members)

        @staticmethod
        def _validate_remote_vacancy_evidence(*_args, **_kwargs):
            return None

    class FakeTopology:
        topology_hash = PUBLISHER_MODULE.EXPECTED_TOPOLOGY_HASH
        devices = [
            SimpleNamespace(device_id=index, coordinates=(index % 4, index // 4, 0))
            for index in range(32)
        ]

    api = {
        "PhysicalTopology": SimpleNamespace(from_dict=lambda _value: FakeTopology()),
        "analyze": lambda *_args: deepcopy(analysis),
        "array_sha256": array_sha256,
        "config": lambda: SimpleNamespace(to_dict=lambda: deepcopy(config)),
        "generate": lambda _config: input_bits.copy(),
        "source_archive_identity": source_archive,
        "source_identity": lambda: deepcopy(source_identity),
        "validate_hlo": lambda *_args: (
            SimpleNamespace(to_dict=lambda: deepcopy(report)),
            deepcopy(algorithm),
        ),
        "validate_topology": lambda _topology: None,
    }
    monkeypatch.setattr(PUBLISHER_MODULE, "_git_bytes", lambda *_args: probe_raw)
    return FakeBase(), api, members, records, raw_log, code_pin, run_tag


def test_prepare_success_validates_and_materializes_the_complete_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, api, _members, _records, _raw_log, code_pin, run_tag = (
        _synthetic_success_fixture(monkeypatch)
    )
    payload = PUBLISHER_MODULE._prepare_success(
        base,
        7,
        code_pin=code_pin,
        run_tag=run_tag,
        remote="gs://driftbench-dsv4-uc/results/synthetic",
        elapsed=17,
        api=api,
    )
    assert set(payload) == PUBLISHER_MODULE.SUCCESS_PAYLOAD
    summary = json.loads(payload["summary.json"])
    assert summary["classification"].startswith(
        "M2048_STRATEGY_ND_ASSOCIATION_UNIQUE"
    )
    assert summary["gate_d_closed"] is False


def test_prepare_success_rejects_extra_record_and_marker_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, api, members, records, raw_log, code_pin, run_tag = (
        _synthetic_success_fixture(monkeypatch)
    )
    hostile = deepcopy(records)
    hostile[4]["unreviewed"] = True
    members["distributed.raw.log"] = raw_log(hostile)
    with pytest.raises(RuntimeError, match="record schema drifted"):
        PUBLISHER_MODULE._prepare_success(
            base,
            7,
            code_pin=code_pin,
            run_tag=run_tag,
            remote="gs://driftbench-dsv4-uc/results/synthetic",
            elapsed=17,
            api=api,
        )


def test_source_certificate_binds_every_listed_file_and_grants_no_authority() -> None:
    record = json.loads(CERTIFICATE.read_text(encoding="ascii"))
    assert record["classification"].endswith(
        "INSTALL_UNAUTHORIZED;TPU_EXECUTION_UNAUTHORIZED;NO_MODEL_OR_GATE_D_CLAIM"
    )
    assert record["future_persistence_contract"]["future_merged_pin_required"] is True
    assert record["runtime_recovery"]["missing_workers_observed_read_only"] == list(
        range(1, 8)
    )
    assert record["test_evidence"]["passed"] == 31
    assert record["proposed_fresh_tag"] in record[
        "protected_run_command_after_approved_install"
    ]
    for item in record["sources"]:
        assert sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item[
            "sha256"
        ]


def test_remote_privilege_boundary_ignores_hostile_path_and_checks_full_identity(
    tmp_path: Path,
) -> None:
    import re

    source = FLEET_INSTALLER.read_text(encoding="ascii")
    remote_lines = [
        line
        for line in source.splitlines()
        if line.startswith(("runtime_command=", "install_command="))
    ]
    assert len(remote_lines) == 2
    remote = "\n".join(remote_lines)
    hostile = tmp_path / "bin"
    hostile.mkdir()
    marker = tmp_path / "hostile-sudo-ran"
    fake_sudo = hostile / "sudo"
    fake_sudo.write_text(f"#!/usr/bin/bash\ntouch {marker}\nexit 99\n")
    fake_sudo.chmod(0o755)
    assert (hostile / "sudo").exists()
    assert re.search(r"(?<![/A-Za-z0-9_])sudo(?:\s|$)", remote) is None
    assert remote.count("/usr/bin/sudo") == 3
    assert not marker.exists()
    for identity_check in (
        '[[ -f $provisioner && ! -L $provisioner ]]',
        '/usr/bin/readlink -f -- "$provisioner"',
        '/usr/bin/stat -c "%F:%h:%U:%G:%a" -- "$provisioner"',
        '"regular file:1:root:root:555"',
        "os.listxattr(sys.argv[1],follow_symlinks=False)",
        '"$provisioner" == /opt/glm-tpu/bin/provision_gate_d_python_runtime.py',
    ):
        assert identity_check in remote
