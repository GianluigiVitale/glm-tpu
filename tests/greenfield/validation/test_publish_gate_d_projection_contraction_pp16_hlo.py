from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[3]
DRIVER = ROOT / "scripts/greenfield/acquire_gate_d_projection_contraction_pp16_hlo.py"
PUBLISHER = (
    ROOT / "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py"
)
WRAPPER = ROOT / "scripts/greenfield/run_gate_d_projection_contraction_pp16_hlo.sh"
LAUNCHER = ROOT / "scripts/greenfield/launch_gate_d_projection_contraction_pp16_hlo.py"
BASE_ACQUISITION_TEST = ROOT / (
    "tests/greenfield/validation/test_gate_d_compensated_pp16_hlo_acquisition.py"
)
MIRROR_VERIFIER = ROOT / "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
ANALYZER = ROOT / (
    "scripts/greenfield/analyze_gate_d_projection_contraction_pp16_hlo_orchestration_source.py"
)
SOURCE_CERTIFICATE = (
    ROOT / "docs/artifacts/gate-d-projection-contraction-pp16-source.json"
)
HLO_SOURCE_CERTIFICATE = ROOT / (
    "docs/artifacts/gate-d-projection-contraction-hlo-acquisition-source.json"
)
TOPOLOGY = ROOT / "docs/artifacts/gate-d-runtime-locality-authority.json"
V2_PUBLICATION_FAILURE = ROOT / (
    "docs/artifacts/gate-d-projection-contraction-pp16-hlo-v2-publication-failure.json"
)
V2_PUBLICATION_FAILURE_SHA256 = (
    "d38c43afdd6c87d671c39e8f71448d73ac71290fee2bfc8fb207647d3e9010aa"
)


def _load(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


DRIVER_MODULE = _load(DRIVER, "gate_d_projection_contraction_driver_for_publisher_test")
PUBLISHER_MODULE = _load(
    PUBLISHER, "gate_d_projection_contraction_publisher_for_publisher_test"
)
MIRROR_MODULE = _load(MIRROR_VERIFIER, "gate_d_mirror_verifier_for_publisher_test")
ANALYZER_MODULE = _load(
    ANALYZER, "gate_d_projection_contraction_orchestration_analyzer_test"
)
BASE_TEST_MODULE = _load(BASE_ACQUISITION_TEST, "gate_d_base_publisher_test_support")
TEST_PUBLICATION_RUNTIME_RAW = PUBLISHER_MODULE._canonical(
    {
        "artifact_kind": "gate_d_projection_contraction_pp16_publisher_runtime_test_double",
        "dependency_manifest_sha256": "a" * 64,
        "dependency_tree_sha256": "b" * 64,
        "python_runtime_tree_sha256": "c" * 64,
    }
)


def _git(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(ROOT), *arguments],
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )


def _mirror_report(pin: str) -> dict[str, object]:
    blobs = []
    for relative in MIRROR_MODULE.BOUND_PATHS:
        raw = _git("show", f"{pin}:{relative}")
        blobs.append(
            {
                "git_object": _git("rev-parse", f"{pin}:{relative}")
                .decode("ascii")
                .strip(),
                "path": relative,
                "sha256": sha256(raw).hexdigest(),
            }
        )
    remote_record = f"{pin}\trefs/heads/{PUBLISHER_MODULE.BRANCH}\n".encode("ascii")
    return {
        "artifact_kind": "gate_d_same_region_git_mirror_replay",
        "bound_blob_count": len(blobs),
        "bound_blobs": blobs,
        "branch": PUBLISHER_MODULE.BRANCH,
        "checkout_archive_sha256": sha256(
            _git("archive", "--format=tar", pin)
        ).hexdigest(),
        "commit": pin,
        "commit_connectivity_fsck": True,
        "mirror_uri": PUBLISHER_MODULE.MIRROR_URI,
        "origin": PUBLISHER_MODULE.ORIGIN,
        "origin_remote_exact_record": True,
        "origin_remote_ref": f"refs/heads/{PUBLISHER_MODULE.BRANCH}",
        "origin_remote_sha256": sha256(remote_record).hexdigest(),
        "schema_version": 1,
    }


def _runner(pin: str) -> dict[str, object]:
    return {
        "artifact_kind": "gate_d_projection_contraction_pp16_optimized_hlo_acquisition",
        "status": "HLO_ACQUIRED_UNADJUDICATED",
        "code_hash": pin,
        "compile_only": True,
        "compiled_executable_invocation_count": 0,
        "tpu_numerical_execution_performed": False,
        "persistent_compilation_cache_enabled": False,
        "gate_d_closed": False,
        "numerical_claim": False,
        "performance_claim": False,
        "projection_contraction_source_sha256": (
            PUBLISHER_MODULE.PROJECTION_CONTRACTION_SOURCE_SHA256
        ),
        "projection_contraction_source_authority": (
            PUBLISHER_MODULE._projection_contraction_source_authority(pin)
        ),
        "input_spec": deepcopy(PUBLISHER_MODULE._EXPECTED_INPUT_SPEC),
        "output_spec": deepcopy(PUBLISHER_MODULE._EXPECTED_OUTPUT_SPEC),
    }


def _write_preclaim_members(root: Path, *, runner: dict[str, object], pin: str) -> int:
    root.chmod(0o700)
    members = {
        "dependencies.json": b"{}\n",
        "mirror.sha256": PUBLISHER_MODULE._canonical(_mirror_report(pin)),
        "publisher_runtime.json": b"{}\n",
        "runner.json": PUBLISHER_MODULE._canonical(runner),
        "sync.txt": (
            f"SYNC_OK {os.uname().nodename} {pin} "
            "compile_host_only=1 sealed_source_archive=1\n"
        ).encode("ascii"),
    }
    for name, raw in members.items():
        path = root / name
        path.write_bytes(raw)
        path.chmod(0o600)
    return os.open(root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY)


def _complete_success_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str, str]:
    monkeypatch.setattr(
        PUBLISHER_MODULE, "_verify_dependency_record_live", lambda record: None
    )
    monkeypatch.setattr(
        PUBLISHER_MODULE,
        "_verify_accelerator_device_mapping_record_live",
        lambda record, **kwargs: None,
    )
    monkeypatch.setattr(
        PUBLISHER_MODULE, "_open_accelerator_device_directory", lambda: None
    )
    monkeypatch.setattr(
        PUBLISHER_MODULE,
        "_verify_accelerator_device_directory_named",
        lambda **kwargs: None,
    )
    run_root = tmp_path / "gate-d-runs"
    run_root.mkdir(mode=0o700)
    monkeypatch.setattr(PUBLISHER_MODULE, "RUN_ROOT", run_root)
    pin = _git("rev-parse", "HEAD").decode("ascii").strip()
    tag = "gate_d_projection_contraction_pp16_hlo_20260901T120000123456789Z"
    run = run_root / tag
    PUBLISHER_MODULE.initialize_run_dir(
        run, publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW
    )
    optimized = b"HloModule projection_contraction_test\n"
    stable = b"module @projection_contraction_test {}\n"
    accelerator_nodes = [BASE_TEST_MODULE._accelerator_device_record()]
    sealed_source = {
        "archive_sha256": "4" * 64,
        "file_manifest_count": 10,
        "file_manifest_sha256": "5" * 64,
    }
    dependencies = {
        "accelerator_device_observation_scope": (
            PUBLISHER_MODULE._ACCELERATOR_DEVICE_OBSERVATION_SCOPE
        ),
        "accelerator_device_nodes_observed_mapped": accelerator_nodes,
        "artifact_kind": "gate_d_projection_contraction_pp16_compiler_dependencies",
        "code_hash": pin,
        "dependency_sites": {
            "jax": {
                "manifest_sha256": PUBLISHER_MODULE.JAX_SITE_MANIFEST_SHA256,
                "root": str(PUBLISHER_MODULE.JAX_SITE_ROOT),
                "tree_sha256": PUBLISHER_MODULE.JAX_SITE_TREE_SHA256,
            },
            "libtpu": {
                "manifest_sha256": PUBLISHER_MODULE.LIBTPU_SITE_MANIFEST_SHA256,
                "root": str(PUBLISHER_MODULE.LIBTPU_SITE_ROOT),
                "tree_sha256": PUBLISHER_MODULE.LIBTPU_SITE_TREE_SHA256,
            },
        },
        "environment": PUBLISHER_MODULE._EXPECTED_COMPILER_ENVIRONMENT,
        "native_mappings": [
            {
                "bytes": 1,
                "device": 1,
                "inode": 1,
                "path": "/usr/lib/native.so",
                "sha256": "2" * 64,
            }
        ],
        "python_runtime": {
            "python_executable": str(PUBLISHER_MODULE.PYTHON),
            "python_runtime_root": str(PUBLISHER_MODULE.PYTHON_RUNTIME_ROOT),
            "python_runtime_tree_sha256": (PUBLISHER_MODULE.PYTHON_RUNTIME_TREE_SHA256),
            "python_sha256": PUBLISHER_MODULE.PYTHON_SHA256,
        },
        "python_modules": [
            {
                "bytes": 1,
                "device": 1,
                "inode": 2,
                "path": "/usr/lib/python.py",
                "sha256": "3" * 64,
            }
        ],
        "sealed_project_source": sealed_source,
    }
    dependencies_raw = PUBLISHER_MODULE._canonical(dependencies)
    runner = {
        "artifact_kind": "gate_d_projection_contraction_pp16_optimized_hlo_acquisition",
        "claim_scope": "compile only test double",
        "code_hash": pin,
        "compile_only": True,
        "compiler_dependency_manifest": {
            "accelerator_device_node_count": 1,
            "accelerator_device_nodes_sha256": (
                PUBLISHER_MODULE._accelerator_device_nodes_sha256(accelerator_nodes)
            ),
            "byte_count": len(dependencies_raw),
            "filename": "dependencies.json",
            "native_mapping_count": 1,
            "python_module_count": 1,
            "sha256": sha256(dependencies_raw).hexdigest(),
        },
        "compiled_executable_invocation_count": 0,
        "projection_contraction_source_authority": (
            PUBLISHER_MODULE._projection_contraction_source_authority(pin)
        ),
        "projection_contraction_source_sha256": (
            PUBLISHER_MODULE.PROJECTION_CONTRACTION_SOURCE_SHA256
        ),
        "gate_d_closed": False,
        "hlo": {
            "optimized": {
                "byte_count": len(optimized),
                "filename": "projection_contraction_pp16_stage0.optimized_hlo.txt",
                "sha256": sha256(optimized).hexdigest(),
            },
            "stablehlo": {
                "byte_count": len(stable),
                "filename": "projection_contraction_pp16_stage0.stablehlo.mlir",
                "sha256": sha256(stable).hexdigest(),
            },
        },
        "input_spec": deepcopy(PUBLISHER_MODULE._EXPECTED_INPUT_SPEC),
        "numerical_claim": False,
        "output_spec": deepcopy(PUBLISHER_MODULE._EXPECTED_OUTPUT_SPEC),
        "performance_claim": False,
        "persistent_compilation_cache_enabled": False,
        "physical_group": {
            "coordinates": [[0, 0, 0], [1, 0, 0]],
            "device_ids": [0, 1],
            "local_device_count_visible": 4,
            "mesh_device_count": 2,
            "process_index": 0,
            "stage_id": 0,
        },
        "sealed_project_source": {**sealed_source, "loaded_module_count": 3},
        "status": "HLO_ACQUIRED_UNADJUDICATED",
        "tpu_numerical_execution_performed": False,
    }
    members = {
        "dependencies.json": dependencies_raw,
        "hlo/projection_contraction_pp16_stage0.optimized_hlo.txt": optimized,
        "hlo/projection_contraction_pp16_stage0.stablehlo.mlir": stable,
        "mirror.sha256": PUBLISHER_MODULE._canonical(_mirror_report(pin)),
        "runner.json": PUBLISHER_MODULE._canonical(runner),
        "sync.txt": (
            f"SYNC_OK {os.uname().nodename} {pin} "
            "compile_host_only=1 sealed_source_archive=1\n"
        ).encode("ascii"),
    }
    census = "".join(f"CENSUS_OK host-{index}\n" for index in range(8)).encode()
    members["census_pre.txt"] = census
    members["census_post.txt"] = census
    for name in (
        "orchestrator.log",
        "runner.log",
    ):
        members[name] = f"{name}\n".encode()
    for relative, raw in members.items():
        (run / relative).write_bytes(raw)
    remote = (
        "gs://driftbench-dsv4-uc/results/greenfield/glm52/"
        f"gate_d_projection_contraction_pp16_hlo/{tag}"
    )
    no_objects = "ERROR: (gcloud.storage.ls) One or more URLs matched no objects."
    members = {
        "remote_vacancy.raw.txt": (
            "scope=live flags=none returncode=1\n"
            f"{no_objects}\n"
            "scope=all_versions flags=--all-versions returncode=1\n"
            f"{no_objects}\n"
            "scope=soft_deleted flags=--soft-deleted,--exhaustive returncode=1\n"
            f"{no_objects}\n"
        ).encode("ascii"),
        "remote_vacancy.txt": (
            f"VACANT live {remote}\n"
            f"VACANT all_versions {remote}\n"
            f"VACANT soft_deleted {remote}\n"
        ).encode("ascii"),
    }
    for relative, raw in members.items():
        (run / relative).write_bytes(raw)
    return run, remote, pin


def test_publisher_import_is_default_off_and_loads_no_jax() -> None:
    command = (
        "import importlib.util,sys;"
        f"p={str(PUBLISHER)!r};"
        "s=importlib.util.spec_from_file_location('p',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "assert not any(n=='jax' or n.startswith('jax.') for n in sys.modules)"
    )
    subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", "-c", command],
        check=True,
        env={
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )


def test_publisher_specs_and_source_authority_match_exact_driver_contract() -> None:
    expected_inputs = [
        {"dtype": dtype, "name": name, "shape": list(shape)}
        for name, shape, dtype in DRIVER_MODULE.INPUT_SPEC
    ]
    expected_outputs = [
        {"dtype": dtype, "name": name, "shape": list(shape)}
        for name, shape, dtype in DRIVER_MODULE.OUTPUT_SPEC
    ]
    source = json.loads(SOURCE_CERTIFICATE.read_text())
    pin = _git("rev-parse", "HEAD").decode("ascii").strip()
    assert PUBLISHER_MODULE._EXPECTED_INPUT_SPEC == expected_inputs
    assert PUBLISHER_MODULE._EXPECTED_OUTPUT_SPEC == expected_outputs
    assert PUBLISHER_MODULE._projection_contraction_source_authority(
        pin
    ) == DRIVER_MODULE.validate_projection_contraction_source(source)
    assert PUBLISHER_MODULE._hlo_acquisition_source_authority(pin) == {
        "certificate_sha256": sha256(HLO_SOURCE_CERTIFICATE.read_bytes()).hexdigest(),
        "source_pin": "c686e6491387ae46ebfc468f401f79c7177ca0f5",
        "source_sha256s": dict(
            sorted(
                json.loads(HLO_SOURCE_CERTIFICATE.read_bytes())[
                    "source_sha256s"
                ].items()
            )
        ),
    }


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("artifact_kind",), "wrong"),
        (("classification",), "wrong"),
        (("authorization", "tpu_compile"), True),
        (("gate_d_closed",), True),
        (("source_audit", "projection_precision"), "default"),
        (("source_audit", "pp16_stage_zero", "device_ids"), [0, 2]),
        (("source_sha256s",), {}),
        (("direct_dependency_sha256s",), {}),
    ),
)
def test_source_authority_mutations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
    value: object,
) -> None:
    source = json.loads(SOURCE_CERTIFICATE.read_text())
    current = source
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    raw = PUBLISHER_MODULE._canonical(source)

    def fake_git(*arguments: str) -> bytes:
        if arguments[0] == "for-each-ref":
            return b""
        if arguments[0] == "show":
            return raw
        if arguments[0] == "merge-base":
            return b""
        raise AssertionError(arguments)

    monkeypatch.setattr(PUBLISHER_MODULE, "_git_bytes", fake_git)
    monkeypatch.setattr(
        PUBLISHER_MODULE,
        "PROJECTION_CONTRACTION_SOURCE_SHA256",
        sha256(raw).hexdigest(),
    )
    with pytest.raises(RuntimeError, match="source certificate authority"):
        PUBLISHER_MODULE._projection_contraction_source_authority("0" * 40)


def test_same_region_mirror_replay_is_bound_to_complete_local_archive() -> None:
    pin = _git("rev-parse", "HEAD").decode("ascii").strip()
    raw = PUBLISHER_MODULE._canonical(_mirror_report(pin))
    PUBLISHER_MODULE._validate_mirror_replay(raw, pin)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("branch", "wrong"),
        ("commit", "0" * 40),
        ("commit_connectivity_fsck", False),
        ("mirror_uri", "gs://wrong/.git"),
        ("origin", "wrong"),
        ("origin_remote_exact_record", False),
        ("origin_remote_ref", "refs/heads/wrong"),
        ("origin_remote_sha256", "0" * 64),
        ("checkout_archive_sha256", "0" * 64),
    ),
)
def test_same_region_mirror_authority_mutations_fail_closed(
    field: str, value: object
) -> None:
    pin = _git("rev-parse", "HEAD").decode("ascii").strip()
    report = _mirror_report(pin)
    report[field] = value
    with pytest.raises(RuntimeError, match="mirror replay authority"):
        PUBLISHER_MODULE._validate_mirror_replay(
            PUBLISHER_MODULE._canonical(report), pin
        )


def test_same_region_mirror_bound_blob_mutation_fails_closed() -> None:
    pin = _git("rev-parse", "HEAD").decode("ascii").strip()
    report = _mirror_report(pin)
    report["bound_blobs"][0]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bound blob identity"):
        PUBLISHER_MODULE._validate_mirror_replay(
            PUBLISHER_MODULE._canonical(report), pin
        )


def test_compile_host_authority_is_exact() -> None:
    pin = _git("rev-parse", "HEAD").decode("ascii").strip()
    exact = (
        f"SYNC_OK {os.uname().nodename} {pin} "
        "compile_host_only=1 sealed_source_archive=1\n"
    ).encode("ascii")
    PUBLISHER_MODULE._validate_compile_host_authority(exact, pin)
    with pytest.raises(RuntimeError, match="compile-host code authority"):
        PUBLISHER_MODULE._validate_compile_host_authority(
            exact.replace(b"compile_host_only=1", b"compile_host_only=0"), pin
        )


def test_complete_publication_is_generation_bound_and_terminal_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote, pin = _complete_success_run(tmp_path, monkeypatch)
    bucket = BASE_TEST_MODULE._FakeBucket()
    PUBLISHER_MODULE.publish_success(
        run,
        remote,
        code_pin=pin,
        elapsed=7,
        publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
        storage_bucket=bucket,
    )
    terminal_name = remote.removeprefix("gs://driftbench-dsv4-uc/") + "/HLO_ACQUIRED"
    assert bucket.mutations[-1] == terminal_name
    terminal = json.loads(bucket.objects[terminal_name][0])
    assert (
        terminal["artifact_kind"] == "gate_d_projection_contraction_pp16_hlo_acquired"
    )
    assert terminal["adjudicated"] is False
    assert terminal["gate_d_closed"] is False
    assert terminal["tpu_numerical_execution_performed"] is False
    assert (run / "terminal_upload_receipt.json").is_file()


@pytest.mark.parametrize("scope", ("live", "all_versions", "soft_deleted"))
def test_prior_remote_history_refuses_publication_before_any_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    run, remote, pin = _complete_success_run(tmp_path, monkeypatch)
    bucket = BASE_TEST_MODULE._FakeBucket()
    prefix = remote.removeprefix("gs://driftbench-dsv4-uc/") + "/"
    target = prefix + "prior"
    record = (b"prior", "99", "AAAAAA==")
    if scope == "live":
        bucket.objects[target] = record
    elif scope == "all_versions":
        bucket.noncurrent_objects[target] = record
    else:
        bucket.soft_deleted_objects[target] = record
    with pytest.raises(RuntimeError, match="prior live/versioned/soft-deleted history"):
        PUBLISHER_MODULE.publish_success(
            run,
            remote,
            code_pin=pin,
            elapsed=7,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=bucket,
        )
    assert bucket.mutations == []


def test_remote_vacancy_evidence_is_exact_and_bound_to_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote, pin = _complete_success_run(tmp_path, monkeypatch)
    (run / "remote_vacancy.txt").write_text(
        (run / "remote_vacancy.txt").read_text().replace(remote, remote + "-other")
    )
    with pytest.raises(RuntimeError, match="canonical remote vacancy evidence"):
        PUBLISHER_MODULE.publish_success(
            run,
            remote,
            code_pin=pin,
            elapsed=7,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=BASE_TEST_MODULE._FakeBucket(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact_kind", "wrong"),
        ("status", "wrong"),
        ("code_hash", "0" * 40),
        ("compile_only", False),
        ("compiled_executable_invocation_count", 1),
        ("tpu_numerical_execution_performed", True),
        ("persistent_compilation_cache_enabled", True),
        ("gate_d_closed", True),
        ("numerical_claim", True),
        ("performance_claim", True),
        ("projection_contraction_source_sha256", "0" * 64),
        ("projection_contraction_source_authority", {}),
        ("input_spec", []),
        ("output_spec", []),
    ),
)
def test_runner_claim_boundary_mutations_fail_before_live_validation(
    tmp_path: Path, field: str, value: object
) -> None:
    pin = _git("rev-parse", "HEAD").decode("ascii").strip()
    runner = _runner(pin)
    runner[field] = value
    run_fd = _write_preclaim_members(tmp_path, runner=runner, pin=pin)
    try:
        with pytest.raises(RuntimeError, match="runner claim boundary"):
            PUBLISHER_MODULE._prepare_success(
                run_fd,
                code_pin=pin,
                run_tag="gate_d_projection_contraction_pp16_hlo_20260901T120000123456789Z",
                remote=(
                    "gs://driftbench-dsv4-uc/results/greenfield/glm52/"
                    "gate_d_projection_contraction_pp16_hlo/"
                    "gate_d_projection_contraction_pp16_hlo_20260901T120000123456789Z"
                ),
                elapsed=1,
            )
    finally:
        os.close(run_fd)


def test_publisher_names_only_the_isolated_projection_contraction_namespace() -> None:
    source = PUBLISHER.read_text()
    assert PUBLISHER_MODULE.REPO == ROOT
    assert (
        PUBLISHER_MODULE.REMOTE_ROOT
        == "results/greenfield/glm52/gate_d_projection_contraction_pp16_hlo/"
    )
    assert "gate_d_compensated_pp16_hlo/" not in source
    assert "/home/gianl/glm-tpu-topology-rewrite" not in source
    assert "compensated_pp16_stage0" not in source


def _shell_constant(source: str, name: str) -> str:
    prefix = f"readonly {name}="
    matches = [
        line.removeprefix(prefix)
        for line in source.splitlines()
        if line.startswith(prefix)
    ]
    assert len(matches) == 1
    return matches[0]


def _heredoc(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_wrapper_requires_root_owned_launcher_before_any_lock_or_cloud_action() -> None:
    completed = subprocess.run(
        ["/usr/bin/bash", "--noprofile", "--norc", str(WRAPPER)],
        check=False,
        capture_output=True,
        text=True,
        env={
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
    )
    assert completed.returncode == 2
    assert "root-owned launcher" in completed.stderr
    assert "RUN_DIR=" not in completed.stdout


def test_wrapper_requires_retained_sealed_descriptor_and_immutable_children() -> None:
    source = WRAPPER.read_text()
    start = "read -r -d '' RUNTIME_BOUNDARY_VERIFIER <<'RUNTIME_BOUNDARY_VERIFIER_EOF' || true\n"
    end = "RUNTIME_BOUNDARY_VERIFIER_EOF\n"
    verifier = _heredoc(source, start, end)
    assert "F_GET_SEALS" in verifier
    assert "wrapper_fd != 10" in verifier
    assert "GLM_GATE_D_WRAPPER_SHA256" in verifier
    assert "launch_gate_d_projection_contraction_pp16_hlo_v3.py" in verifier
    assert "IMMUTABLE_LOCK_BROKER" not in source
    assert "WRAPPER_ABS" not in source
    assert "GLM_GATE_D_IMMUTABLE_LOCKS_HELD" in source
    assert "exec 10<&-" in source
    assert (
        "readonly IMMUTABLE_CAPSULE_ROOT=/usr/local/libexec/glm-tpu/"
        "gate-d-projection-contraction-pp16-hlo-v3"
    ) in source
    assert (
        "$WORKTREE/scripts/greenfield/acquire_gate_d_projection_contraction_pp16_hlo.py"
        not in source
    )
    assert (
        "$WORKTREE/scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py"
        not in source
    )


def test_wrapper_pins_exact_projection_contraction_and_mirror_sources() -> None:
    source = WRAPPER.read_text()
    paths = {
        "TOPOLOGY_SHA": ROOT / "docs/artifacts/gate-d-runtime-locality-authority.json",
        "PROJECTION_SOURCE_SHA": SOURCE_CERTIFICATE,
        "HLO_SOURCE_SHA": HLO_SOURCE_CERTIFICATE,
        "DRIVER_SHA": DRIVER,
        "PUBLISHER_SHA": PUBLISHER,
        "MIRROR_VERIFIER_SHA": MIRROR_VERIFIER,
        "STORAGE_SITE_BUILDER_SHA": ROOT
        / "scripts/greenfield/build_gate_d_storage_site_capsule.py",
    }
    for name, path in paths.items():
        assert _shell_constant(source, name) == sha256(path.read_bytes()).hexdigest()


def test_wrapper_uses_complete_mirror_replay_and_compile_host_only_authority() -> None:
    source = WRAPPER.read_text()
    mirror = source.index("replaying the complete locked US-CENTRAL2 Git mirror")
    census = source.index("strict_census pre")
    compile_start = source.index("lowering and compiling one abstract-input")
    assert mirror < census < compile_start
    assert '"$MIRROR_VERIFIER"' in source
    assert '--expected-source-sha256 "$MIRROR_VERIFIER_SHA"' in source
    assert "compile_host_only=1 sealed_source_archive=1" in source
    assert "WORKER_REPO_VERIFY_SCRIPT" not in source
    assert "repos/glm-tpu-topology-rewrite" not in source
    assert '--projection-contraction-source "$PROJECTION_SOURCE"' in source
    assert '--projection-contraction-source-sha256 "$PROJECTION_SOURCE_SHA"' in source
    assert "GLM_GATE_D_PROJECTION_CONTRACTION_HLO=1" in source


def test_wrapper_proves_all_remote_history_scopes_before_run_directory() -> None:
    source = WRAPPER.read_text()
    history = source.index("vacancy_live_output=")
    initialize = source.index("publisher_init init --run-dir")
    compile_start = source.index("lowering and compiling one abstract-input")
    assert history < initialize < compile_start
    assert 'gcloud storage ls "$REMOTE_PREFIX/**"' in source
    assert 'gcloud storage ls --all-versions "$REMOTE_PREFIX/**"' in source
    assert 'gcloud storage ls --soft-deleted --exhaustive "$REMOTE_PREFIX/**"' in source
    assert source.count('!= "$VACANCY_EXPECTED"') == 3
    assert "PYTHONWARNINGS=ignore" in source
    assert "scope=all_versions flags=--all-versions returncode=1" in source
    assert "scope=soft_deleted flags=--soft-deleted,--exhaustive returncode=1" in source


def test_wrapper_has_one_compile_process_and_no_executable_invocation() -> None:
    source = WRAPPER.read_text()
    assert source.count('"$DRIVER_PYTHON" -I -S -B -u "$DRIVER"') == 1
    assert source.count("--compile-only 1") == 1
    assert "JAX_ENABLE_COMPILATION_CACHE=0" in source
    assert "TPU_VISIBLE_DEVICES=0,1,2,3" in source
    assert "execution_count=0" in source
    assert "strict_census pre" in source


def test_compiler_environment_contract_matches_acquirer_publisher_and_wrapper() -> None:
    assert DRIVER_MODULE._EXPECTED_ENVIRONMENT == (
        PUBLISHER_MODULE._EXPECTED_COMPILER_ENVIRONMENT
    )
    source = WRAPPER.read_text(encoding="ascii")
    compile_block = source.split(
        'say "lowering and compiling one abstract-input PP16 stage-zero graph; '
        'invocation forbidden"',
        1,
    )[1].split('"$DRIVER_PYTHON" -I -S -B -u "$DRIVER"', 1)[0]
    for name, value in DRIVER_MODULE._EXPECTED_ENVIRONMENT.items():
        assert f"{name}={value} \\" in compile_block
    assert "strict_census post" in source
    assert "--worker=all" in source
    subprocess.run(["/usr/bin/bash", "-n", str(WRAPPER)], check=True)


def test_v2_publication_failure_is_append_only_diagnostic_evidence() -> None:
    raw = V2_PUBLICATION_FAILURE.read_bytes()
    assert len(raw) == 3141
    assert sha256(raw).hexdigest() == V2_PUBLICATION_FAILURE_SHA256
    artifact = json.loads(raw)
    assert artifact["status"] == "DIAGNOSTIC_ONLY_NOT_HLO_ACQUIRED"
    assert artifact["failure"]["hlo_acquired_terminal_present"] is False
    assert artifact["compile_evidence"]["compiled_executable_invocation_count"] == 0
    assert artifact["compile_evidence"]["tpu_numerical_execution_performed"] is False
    assert artifact["fleet"] == {
        "post_census_clean_hosts": 8,
        "pre_census_clean_hosts": 8,
    }
    old_publisher = _git(
        "show",
        f"{artifact['authority']['code_hash']}:"
        "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py",
    )
    old_tree = compile(old_publisher, "old_publisher.py", "exec", ast.PyCF_ONLY_AST)
    old_environment = next(
        node.value
        for node in old_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_EXPECTED_COMPILER_ENVIRONMENT"
            for target in node.targets
        )
    )
    assert isinstance(old_environment, ast.Dict)
    assert (
        "GLM_GATE_D_PROJECTION_CONTRACTION_HLO"
        not in old_publisher.decode("ascii")
        .split("_EXPECTED_COMPILER_ENVIRONMENT = {", 1)[1]
        .split("}", 1)[0]
    )
    assert (
        PUBLISHER_MODULE._EXPECTED_COMPILER_ENVIRONMENT[
            "GLM_GATE_D_PROJECTION_CONTRACTION_HLO"
        ]
        == "1"
    )
    assert artifact["failure"]["root_cause"]["corrected_publisher_sha256"] == (
        sha256(PUBLISHER.read_bytes()).hexdigest()
    )


def test_orchestration_source_analyzer_is_default_off() -> None:
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", str(ANALYZER)],
        check=False,
        capture_output=True,
        text=True,
        env={
            "HOME": "/home/gianl",
            "JAX_PLATFORMS": "cpu",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert completed.returncode != 0
    assert "default-off" in completed.stderr


def test_orchestration_analyzer_rejects_publisher_helper_mutation() -> None:
    source = PUBLISHER.read_bytes()
    mutated = source.replace(b"compile_host_only=1", b"compile_host_only=0", 1)
    assert mutated != source
    with pytest.raises(RuntimeError, match="publisher source hash drifted"):
        ANALYZER_MODULE._audit_publisher(mutated)


def test_orchestration_analyzer_rejects_runtime_boundary_mutation() -> None:
    source = WRAPPER.read_bytes()
    mutated = source.replace(
        b'names = ("glm_pod_workload.lock", "glm_tpu_rsync.lock")',
        b'names = ("glm_pod_workload.lock",)',
        1,
    )
    assert mutated != source
    with pytest.raises(RuntimeError, match="wrapper source hash drifted"):
        ANALYZER_MODULE._audit_wrapper(mutated)


def test_orchestration_analyzer_rejects_descriptor_launcher_mutation() -> None:
    source = LAUNCHER.read_bytes()
    mutated = source.replace(b"os.MFD_ALLOW_SEALING", b"0", 1)
    assert mutated != source
    with pytest.raises(RuntimeError, match="launcher source hash drifted"):
        ANALYZER_MODULE._audit_launcher(mutated, WRAPPER.read_bytes())
