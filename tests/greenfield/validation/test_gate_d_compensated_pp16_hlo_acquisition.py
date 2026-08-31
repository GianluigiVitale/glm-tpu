from __future__ import annotations

import ast
import base64
import importlib.util
import json
import os
import re
import stat
import subprocess
import zipfile
import zipimport
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
CPU_REPLAY = ROOT / "glm_tpu/greenfield/benchmarking/gate_d_compensated_capsule.py"
TPU_REPLAY = ROOT / "glm_tpu/greenfield/benchmarking/gate_d_compensated_pp16_hlo.py"
DRIVER = ROOT / "scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py"
WRAPPER = ROOT / "scripts/greenfield/run_gate_d_compensated_pp16_hlo.sh"
PUBLISHER = ROOT / "scripts/greenfield/publish_gate_d_compensated_pp16_hlo.py"
STORAGE_SITE_BUILDER = (
    ROOT / "scripts/greenfield/build_gate_d_storage_site_capsule.py"
)
STORAGE_SITE_BUILD = (
    ROOT / "docs/artifacts/gate-d-compensated-pp16-storage-site-build.json"
)
ADMISSION = (
    ROOT / "docs/artifacts/gate-d-precompile-admission-v2-compensated-capsule.json"
)
TOPOLOGY = ROOT / "docs/artifacts/gate-d-runtime-locality-authority.json"


def _load_driver():
    spec = importlib.util.spec_from_file_location("gate_d_pp16_hlo_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER_MODULE = _load_driver()


def _load_publisher():
    spec = importlib.util.spec_from_file_location(
        "gate_d_pp16_hlo_publisher", PUBLISHER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PUBLISHER_MODULE = _load_publisher()
TEST_PUBLICATION_RUNTIME_RAW = PUBLISHER_MODULE._canonical(
    {
        "artifact_kind": "gate_d_compensated_pp16_publisher_runtime_test_double",
        "dependency_manifest_sha256": "a" * 64,
        "dependency_tree_sha256": "b" * 64,
        "python_runtime_tree_sha256": "c" * 64,
    }
)


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return deepcopy(function)


class _PlatformNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant):
        if node.value in {"cpu", "tpu"}:
            return ast.copy_location(ast.Constant("PLATFORM"), node)
        if node.value in {
            "Gate-D compensated capsule replay is forced-CPU only",
            "Gate-D compensated capsule replay is TPU compile-only",
        }:
            return ast.copy_location(
                ast.Constant("Gate-D compensated capsule replay is PLATFORM"), node
            )
        return node


def _normalized_builder(path: Path, name: str) -> str:
    function = _function(path, name)
    function.name = "build_gate_d_compensated_capsule_replay"
    if (
        function.body
        and isinstance(function.body[0], ast.Expr)
        and isinstance(function.body[0].value, ast.Constant)
        and isinstance(function.body[0].value.value, str)
    ):
        function.body.pop(0)
    function = _PlatformNormalizer().visit(function)
    ast.fix_missing_locations(function)
    return ast.dump(function, include_attributes=False)


def test_tpu_builder_changes_only_platform_admission() -> None:
    assert _normalized_builder(
        CPU_REPLAY, "build_gate_d_compensated_capsule_cpu_replay"
    ) == _normalized_builder(TPU_REPLAY, "build_gate_d_compensated_pp16_hlo_replay")


def test_exact_admission_and_stage_zero_authority_pass() -> None:
    admission = json.loads(ADMISSION.read_text())
    topology = json.loads(TOPOLOGY.read_text())
    assert DRIVER_MODULE.validate_admission_and_topology(admission, topology) == {
        "coordinates": [[0, 0, 0], [1, 0, 0]],
        "device_ids": [0, 1],
        "process_index": 0,
        "stage_id": 0,
    }
    assert DRIVER_MODULE.validate_capsule_input_spec(admission) == {
        "capsule_sha256": (
            "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4"
        ),
        "input_artifact_sha256": (
            "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b"
        ),
        "raw_input_array_count": 14,
    }


@pytest.mark.parametrize(
    ("target", "path", "value"),
    (
        ("admission", ("admitted_candidate_ids",), ["other"]),
        ("admission", ("gate_d_closed",), True),
        ("admission", ("jax_compile_or_tpu_work_performed",), True),
        ("admission", ("tpu_successor_authorized",), True),
        ("admission", ("compile_only_review_required",), False),
        ("topology", ("topology_hash",), "0" * 64),
        ("topology", ("pp16_lp2", "groups", 0, "device_ids"), [0, 2]),
        ("topology", ("tpu_successor_authorized",), True),
    ),
)
def test_admission_or_topology_drift_fails_closed(
    target: str, path: tuple[object, ...], value: object
) -> None:
    admission = json.loads(ADMISSION.read_text())
    topology = json.loads(TOPOLOGY.read_text())
    current = admission if target == "admission" else topology
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    with pytest.raises(RuntimeError):
        DRIVER_MODULE.validate_admission_and_topology(admission, topology)


def test_capsule_input_shape_drift_fails_closed() -> None:
    admission = json.loads(ADMISSION.read_text())
    candidate = next(
        item
        for item in admission["candidate_results"]
        if item["id"] == DRIVER_MODULE.ADMITTED_ID
    )
    candidate["capsule"]["producer"]["input_arrays"]["prompt_cache_bf16_bits"][
        "shape"
    ] = [32, 16, 256, 128]
    with pytest.raises(RuntimeError, match="prompt_cache"):
        DRIVER_MODULE.validate_capsule_input_spec(admission)


def test_capsule_input_artifact_rebinding_fails_closed() -> None:
    admission = json.loads(ADMISSION.read_text())
    candidate = next(
        item
        for item in admission["candidate_results"]
        if item["id"] == DRIVER_MODULE.ADMITTED_ID
    )
    candidate["capsule"]["producer"]["input_artifact_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="authority hashes"):
        DRIVER_MODULE.validate_capsule_input_spec(admission)


def test_abstract_specs_are_exactly_one_row_and_two_local_owners() -> None:
    inputs = {name: (shape, dtype) for name, shape, dtype in DRIVER_MODULE.INPUT_SPEC}
    outputs = {name: (shape, dtype) for name, shape, dtype in DRIVER_MODULE.OUTPUT_SPEC}
    assert inputs["rms_hidden_update_bf16"] == ((1, 6144), "bfloat16")
    assert inputs["prompt_cache_bf16"] == ((2, 16, 256, 128), "bfloat16")
    assert inputs["wq_b_weight_bits"] == ((2, 2048, 2048), "uint8")
    assert outputs["rms_input_fp32_owners"] == ((2, 1, 6144), "float32")
    assert outputs["selected_positions_owners"] == ((2, 1, 2048), "int32")
    assert inputs["rms_hidden_update_bf16"][0][0] == 1
    assert outputs["selected_positions_owners"][0][:2] == (2, 1)
    assert outputs["selected_scores_owners"][0][:2] == (2, 1)


def test_driver_compiles_once_and_contains_no_executable_invocation() -> None:
    source = DRIVER.read_text()
    tree = ast.parse(source)
    compile_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compile"
    ]
    compiled_invocations = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compiled"
    ]
    assert len(compile_calls) == 1
    assert not compiled_invocations
    assert "ShapeDtypeStruct" in source
    assert "jax.device_put" not in source
    assert "block_until_ready" not in source
    assert "np.load" not in source
    assert '"compiled_executable_invocation_count": 0' in source
    assert "jax_enable_compilation_cache" in source
    assert "JAX_ENABLE_COMPILATION_CACHE=0" in WRAPPER.read_text()
    assert "_sealed_git_source_archive" in source
    assert "_verify_running_source" in source
    assert "_compiler_dependency_records" in source
    assert "/usr/bin/env -i" in WRAPPER.read_text()
    assert "-I -S -B -u" in WRAPPER.read_text()
    assert 'PYTHONPATH="$WORKTREE"' not in WRAPPER.read_text()


def test_driver_environment_is_exact_and_rejects_any_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(os.environ):
        monkeypatch.delenv(name, raising=False)
    for name, value in DRIVER_MODULE._EXPECTED_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    assert DRIVER_MODULE._validate_environment() == dict(
        sorted(DRIVER_MODULE._EXPECTED_ENVIRONMENT.items())
    )
    monkeypatch.setenv("LIBTPU_INIT_ARGS", "--unexpected")
    with pytest.raises(RuntimeError, match="unexpected extras"):
        DRIVER_MODULE._validate_environment()


def test_authority_json_is_parsed_from_the_single_hashed_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = tmp_path / "authority.json"
    original = b'{"identity":"original"}\n'
    replacement = b'{"identity":"replacement"}\n'
    authority.write_bytes(original)

    def snapshot_then_replace(path: Path) -> bytes:
        assert path == authority
        authority.write_bytes(replacement)
        return original

    monkeypatch.setattr(DRIVER_MODULE, "_snapshot_regular", snapshot_then_replace)
    assert DRIVER_MODULE._load_bound_json(
        authority, sha256(original).hexdigest(), "test authority"
    ) == {"identity": "original"}
    assert authority.read_bytes() == replacement


def test_dependency_prefix_requires_prior_files_to_remain_identical() -> None:
    before = {
        "native_mappings": [{"path": "/usr/lib/a", "sha256": "1" * 64}],
        "python_modules": [{"path": "/usr/lib/b", "sha256": "2" * 64}],
    }
    after = {
        "native_mappings": [
            {"path": "/usr/lib/a", "sha256": "1" * 64},
            {"path": "/usr/lib/new", "sha256": "3" * 64},
        ],
        "python_modules": [{"path": "/usr/lib/b", "sha256": "2" * 64}],
    }
    DRIVER_MODULE._require_dependency_prefix_stable(before, after)
    after["python_modules"][0]["sha256"] = "4" * 64
    with pytest.raises(RuntimeError, match="changed during compile"):
        DRIVER_MODULE._require_dependency_prefix_stable(before, after)


def test_dependency_path_order_is_identical_for_numpy_and_numpy_libs() -> None:
    paths = [
        "/home/gianl/vllm-env/lib/python3.12/site-packages/numpy/_core/_multi.so",
        "/home/gianl/vllm-env/lib/python3.12/site-packages/numpy.libs/libopenblas.so",
    ]
    unsorted_paths = [paths[1], paths[0]]
    driver_order = [
        path.as_posix()
        for path in sorted(
            map(Path, unsorted_paths), key=DRIVER_MODULE._canonical_path_key
        )
    ]
    publisher_order = sorted(unsorted_paths, key=PUBLISHER_MODULE._canonical_path_key)
    assert driver_order == publisher_order
    assert driver_order[0].endswith("numpy.libs/libopenblas.so")
    records = [
        {
            "bytes": 1,
            "device": 1,
            "inode": index + 1,
            "path": path,
            "sha256": f"{index + 1:064x}",
        }
        for index, path in enumerate(publisher_order)
    ]
    PUBLISHER_MODULE._validate_dependency_records(records, "numpy-order")
    with pytest.raises(RuntimeError, match="order drifted"):
        PUBLISHER_MODULE._validate_dependency_records(
            list(reversed(records)), "numpy-order"
        )


def test_publisher_environment_is_exact_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in tuple(os.environ):
        monkeypatch.delenv(name, raising=False)
    for name, value in PUBLISHER_MODULE._EXPECTED_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    assert PUBLISHER_MODULE.validate_environment() == dict(
        sorted(PUBLISHER_MODULE._EXPECTED_ENVIRONMENT.items())
    )
    monkeypatch.setenv("UNDECLARED", "unexpected")
    with pytest.raises(RuntimeError, match="exact allowlist"):
        PUBLISHER_MODULE.validate_environment()


def test_publisher_and_compiler_paths_are_distinct_and_exact() -> None:
    wrapper = WRAPPER.read_text()
    assert PUBLISHER_MODULE._EXPECTED_ENVIRONMENT["PATH"] == "/usr/bin:/bin"
    assert PUBLISHER_MODULE._EXPECTED_COMPILER_ENVIRONMENT["PATH"] == (
        "/home/gianl/vllm-env/bin:/usr/bin:/bin"
    )
    assert (
        "PATH=/usr/bin:/bin \\\n"
        "    PYTHONDONTWRITEBYTECODE=1 \\\n"
        '    "$PUBLISHER_PYTHON"'
    ) in wrapper
    assert (
        "PATH=/home/gianl/vllm-env/bin:/usr/bin:/bin \\\n"
        "    PYTHONDONTWRITEBYTECODE=1 \\\n"
        "    TPU_CHIPS_PER_PROCESS_BOUNDS"
    ) in wrapper


def test_storage_publication_capsule_is_explicit_and_pinned() -> None:
    source = STORAGE_SITE_BUILDER.read_text()
    tree = ast.parse(source)
    allowlist = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ALLOWLIST"
            for target in node.targets
        )
    )
    assert len(allowlist) == len(set(allowlist))
    assert {
        "google",
        "google_crc32c",
        "google_crc32c.libs",
        "requests",
        "urllib3",
        "certifi",
        "cryptography",
        "grpc",
    }.issubset(allowlist)
    assert PUBLISHER_MODULE.STORAGE_SITE_ROOT == Path(
        "/opt/glm-tpu/gate-d-storage-site-d94fd4c3e0ff"
    )
    assert (
        PUBLISHER_MODULE.STORAGE_SITE_TREE_SHA256
        == "d94fd4c3e0ffbf024a0b53faff4565d28900b64fc5dd9944bff316f47db1b510"
    )
    assert str(PUBLISHER_MODULE.STORAGE_SITE_ROOT) in PUBLISHER.read_text()
    assert "/home/gianl/vllm-env/lib/python3.12/site-packages" not in (
        PUBLISHER.read_text()
    )
    build = json.loads(STORAGE_SITE_BUILD.read_text())
    assert build == {
        "artifact_kind": "gate_d_compensated_pp16_storage_site_reproducibility",
        "builder_path": "scripts/greenfield/build_gate_d_storage_site_capsule.py",
        "builder_sha256": sha256(STORAGE_SITE_BUILDER.read_bytes()).hexdigest(),
        "capsule_bytes": 44929742,
        "capsule_files": 1593,
        "capsule_manifest_bytes": 7378,
        "capsule_manifest_sha256": PUBLISHER_MODULE.STORAGE_SITE_MANIFEST_SHA256,
        "capsule_tree_sha256": PUBLISHER_MODULE.STORAGE_SITE_TREE_SHA256,
        "cloud_or_network_work_performed": False,
        "installed": False,
        "jax_backend_or_tpu_work_performed": False,
        "reproducible_build_count": 2,
        "schema_version": 1,
        "source_root": "/home/gianl/vllm-env/lib/python3.12/site-packages",
        "target_path": str(PUBLISHER_MODULE.STORAGE_SITE_ROOT),
    }


def test_storage_publication_capsule_replacement_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "storage-site"
    root.mkdir()
    dependency = root / "dependency.py"
    dependency.write_text("VALUE = 'sealed'\n")
    manifest = {
        "allowlist": ["dependency.py"],
        "schema_version": 1,
        "source_entries": {"dependency.py": {"sha256": "1" * 64}},
    }
    manifest_raw = PUBLISHER_MODULE._canonical(manifest)
    (root / "CAPSULE_MANIFEST.json").write_bytes(manifest_raw)
    tree_sha = PUBLISHER_MODULE._tree_sha256(root, require_sealed=False)
    manifest_sha = sha256(manifest_raw).hexdigest()
    assert PUBLISHER_MODULE._validate_storage_site(
        root,
        expected_tree_sha256=tree_sha,
        expected_manifest_sha256=manifest_sha,
        require_sealed=False,
    ) == manifest
    dependency.write_text("VALUE = 'hostile replacement'\n")
    with pytest.raises(RuntimeError, match="dependency tree drifted"):
        PUBLISHER_MODULE._validate_storage_site(
            root,
            expected_tree_sha256=tree_sha,
            expected_manifest_sha256=manifest_sha,
            require_sealed=False,
        )
    with pytest.raises(RuntimeError, match="root-owned sealed"):
        PUBLISHER_MODULE._tree_sha256(root, require_sealed=True)


def test_wrapper_is_valid_shell_and_default_off() -> None:
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    completed = subprocess.run(
        ["/usr/bin/bash", "--noprofile", "--norc", str(WRAPPER)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={
            "GLM_GATE_D_COMPENSATED_PP16_HLO_ACQUIRE": "0",
            "GLM_GATE_D_COMPENSATED_PP16_MODE": "off",
            "GLM_GATE_D_COMPENSATED_PP16_TAG": "",
            "GLM_GATE_D_WRAPPER_SANITIZED": "1",
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/snap/bin:/usr/bin:/bin:/home/gianl/vllm-env/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
    )
    assert completed.returncode == 2
    assert "default-off" in completed.stderr
    wrapper = WRAPPER.read_text()
    assert wrapper.startswith("#!/usr/bin/bash\n")
    assert "GLM_GATE_D_WRAPPER_SANITIZED" in wrapper
    assert "/usr/bin/bash --noprofile --norc" in wrapper
    assert "exec /usr/bin/env -i" not in wrapper
    assert "--run-dir-fd 7" in wrapper


def test_reviewed_entry_command_clears_hostile_bash_env(tmp_path: Path) -> None:
    marker = tmp_path / "bash-env-executed"
    bash_env = tmp_path / "hostile-bash-env"
    bash_env.write_text(f"/usr/bin/touch {marker}\n")
    completed = subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            "HOME=/home/gianl",
            "LANG=C",
            "LC_ALL=C",
            "PATH=/snap/bin:/usr/bin:/bin:/home/gianl/vllm-env/bin",
            "PYTHONDONTWRITEBYTECODE=1",
            "GLM_GATE_D_WRAPPER_SANITIZED=1",
            "GLM_GATE_D_COMPENSATED_PP16_HLO_ACQUIRE=0",
            "GLM_GATE_D_COMPENSATED_PP16_MODE=off",
            "GLM_GATE_D_COMPENSATED_PP16_TAG=",
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            str(WRAPPER),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        env={
            "BASH_ENV": str(bash_env),
            "PATH": str(tmp_path),
        },
        text=True,
    )
    assert completed.returncode == 2
    assert "default-off" in completed.stderr
    assert not marker.exists()


def test_wrapper_pins_current_builder_driver_and_publisher_bytes() -> None:
    source = WRAPPER.read_text()
    for name, path in (
        ("TPU_REPLAY_SHA", TPU_REPLAY),
        ("DRIVER_SHA", DRIVER),
        ("PUBLISHER_SHA", PUBLISHER),
        ("STORAGE_SITE_BUILDER_SHA", STORAGE_SITE_BUILDER),
    ):
        match = re.search(rf"^readonly {name}=([0-9a-f]{{64}})$", source, re.MULTILINE)
        assert match is not None
        assert match.group(1) == sha256(path.read_bytes()).hexdigest()


def test_live_private_run_root_contract() -> None:
    root = PUBLISHER_MODULE.RUN_ROOT
    metadata = root.lstat()
    assert root == Path("/home/gianl/gate-d-runs")
    assert not root.is_symlink()
    assert stat.S_ISDIR(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o700
    assert metadata.st_uid == os.geteuid()
    assert metadata.st_gid == os.getegid()
    assert not os.listxattr(root, follow_symlinks=False)
    descriptor = PUBLISHER_MODULE._open_run_root()
    os.close(descriptor)
    assert "readonly RUN_DIR=/home/gianl/gate-d-runs/$TAG" in WRAPPER.read_text()


def test_exclusive_writer_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    output.write_text("old\n")
    with pytest.raises(RuntimeError, match="append-only"):
        DRIVER_MODULE._exclusive_text(output, "new\n")


def test_exclusive_writer_refuses_final_and_intermediate_symlinks(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    final = tmp_path / "final"
    final.symlink_to(target / "captured")
    with pytest.raises(RuntimeError, match="append-only"):
        DRIVER_MODULE._exclusive_text(final, "forbidden\n")
    intermediate = tmp_path / "intermediate"
    intermediate.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        DRIVER_MODULE._exclusive_text(intermediate / "captured", "forbidden\n")
    assert not (target / "captured").exists()


def test_exclusive_writer_concurrent_final_has_one_winner(tmp_path: Path) -> None:
    output = tmp_path / "race.json"

    def write(value: str) -> str:
        DRIVER_MODULE._exclusive_text(output, value)
        return value

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write, value) for value in ("one\n", "two\n")]
    successes = [future.result() for future in futures if future.exception() is None]
    failures = [future.exception() for future in futures if future.exception()]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert output.read_text() == successes[0]
    assert not list(tmp_path.glob("*.partial.*"))


def test_wrapper_evidence_stream_refuses_existing_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "glm-run"
    run_root.mkdir(mode=0o700)
    monkeypatch.setattr(PUBLISHER_MODULE, "RUN_ROOT", run_root)
    run = run_root / "gate_d_compensated_pp16_hlo_20260831T123456123456789Z"
    PUBLISHER_MODULE.initialize_run_dir(
        run, publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW
    )
    captured = tmp_path / "captured"
    (run / "sync.txt").symlink_to(captured)
    run_fd = PUBLISHER_MODULE._run_fd(run)
    try:
        with pytest.raises(RuntimeError, match="append-only"):
            PUBLISHER_MODULE.write_member_stream_exclusive(
                run_fd, "sync.txt", BytesIO(b"forbidden\n")
            )
    finally:
        os.close(run_fd)
    assert not captured.exists()
    wrapper = WRAPPER.read_text()
    assert '>"$RUN_DIR/' not in wrapper
    assert '>>"$RUN_DIR/' not in wrapper
    assert 'say "HLO_ACQUIRED_UNADJUDICATED' not in wrapper


def test_run_fd_remains_bound_to_validated_root_during_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "gate-d-runs"
    run_root.mkdir(mode=0o700)
    monkeypatch.setattr(PUBLISHER_MODULE, "RUN_ROOT", run_root)
    run = run_root / "gate_d_compensated_pp16_hlo_20260831T123456123456789Z"
    PUBLISHER_MODULE.initialize_run_dir(run)
    moved_root = tmp_path / "validated-root"

    def replace_after_open() -> int:
        descriptor = os.open(
            run_root,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        run_root.rename(moved_root)
        run_root.mkdir(mode=0o700)
        return descriptor

    monkeypatch.setattr(PUBLISHER_MODULE, "_open_run_root", replace_after_open)
    run_fd = PUBLISHER_MODULE._run_fd(run)
    try:
        assert os.fstat(run_fd).st_ino == (moved_root / run.name).stat().st_ino
    finally:
        os.close(run_fd)


def test_inherited_run_fd_rejects_between_invocation_directory_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "gate-d-runs"
    run_root.mkdir(mode=0o700)
    monkeypatch.setattr(PUBLISHER_MODULE, "RUN_ROOT", run_root)
    monkeypatch.setattr(DRIVER_MODULE, "RUN_ROOT", run_root)
    tag = "gate_d_compensated_pp16_hlo_20260831T123456123456789Z"
    run = run_root / tag
    PUBLISHER_MODULE.initialize_run_dir(
        run, publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW
    )
    supervisor_fd = os.open(
        run, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    validated_fd = PUBLISHER_MODULE._run_fd(run, supervisor_fd)
    os.close(validated_fd)
    moved = run_root / "moved"
    run.rename(moved)
    run.mkdir(mode=0o700)
    (run / "hlo").mkdir(mode=0o700)
    try:
        with pytest.raises(RuntimeError, match="authority drifted"):
            PUBLISHER_MODULE._run_fd(run, supervisor_fd)
        with pytest.raises(RuntimeError, match="identity drifted"):
            DRIVER_MODULE._open_inherited_run_dir(run, supervisor_fd)
    finally:
        os.close(supervisor_fd)


def test_preterminal_writers_refuse_after_terminal_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "gate-d-runs"
    run_root.mkdir(mode=0o700)
    monkeypatch.setattr(PUBLISHER_MODULE, "RUN_ROOT", run_root)
    run = run_root / "gate_d_compensated_pp16_hlo_20260831T123456123456789Z"
    PUBLISHER_MODULE.initialize_run_dir(run)
    run_fd = PUBLISHER_MODULE._run_fd(run)
    try:
        PUBLISHER_MODULE.write_member_exclusive(run_fd, "HLO_ACQUIRED", b"sealed\n")
        with pytest.raises(RuntimeError, match="already begun"):
            PUBLISHER_MODULE.write_preterminal_member(
                run_fd, "sync.txt", BytesIO(b"forbidden\n")
            )
        with pytest.raises(RuntimeError, match="already begun"):
            PUBLISHER_MODULE.append_preterminal_log(run_fd, b"forbidden\n")
    finally:
        os.close(run_fd)
    assert not (run / "sync.txt").exists()
    assert not (run / "orchestrator.log").exists()


def test_sealed_git_archive_ignores_hostile_worktree_replacement(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    (repository / "glm_tpu").mkdir(parents=True)
    (repository / "glm_tpu/__init__.py").write_text("VALUE = 'sealed'\n")
    (repository / "glm_tpu/probe.py").write_text("VALUE = 'committed'\n")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Gate D Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "glm_tpu"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "sealed"], check=True
    )
    pin = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    descriptor, archive_path, record = DRIVER_MODULE._sealed_git_source_archive(
        pin, repo=repository
    )
    try:
        (repository / "glm_tpu/probe.py").write_text("VALUE = 'hostile'\n")
        with zipfile.ZipFile(archive_path) as archive:
            assert archive.read("glm_tpu/probe.py") == b"VALUE = 'committed'\n"
        code = zipimport.zipimporter(archive_path + "/glm_tpu").get_code("probe")
        assert code is not None
        assert "committed" in code.co_consts
        assert record["file_manifest_count"] == 2
        assert len(record["archive_sha256"]) == 64
    finally:
        os.close(descriptor)


class _FakeBlob:
    def __init__(self, bucket: _FakeBucket, name: str) -> None:
        self.bucket = bucket
        self.name = name
        self.generation: str | None = None
        self.size: int | None = None
        self.crc32c: str | None = None

    def _load(self) -> None:
        raw, generation, crc = self.bucket.objects[self.name]
        self.generation = generation
        self.size = len(raw)
        self.crc32c = crc

    def upload_from_string(
        self, raw: bytes, *, if_generation_match: int, checksum: str
    ) -> None:
        assert if_generation_match == 0
        assert checksum == "crc32c"
        self.bucket.operations.append(f"upload:{self.name}")
        self.bucket.mutations.append(self.name)
        if self.bucket.inject_before_suffix and self.name.endswith(
            self.bucket.inject_before_suffix
        ):
            self.bucket.inject_before_suffix = None
            self.bucket.objects[self.name] = (b"raced", "999", "AAAAAA==")
        if self.name in self.bucket.objects:
            raise RuntimeError("generation-zero refusal")
        self.bucket.next_generation += 1
        import google_crc32c

        crc = base64.b64encode(google_crc32c.value(raw).to_bytes(4, "big")).decode()
        self.bucket.objects[self.name] = (
            bytes(raw),
            str(self.bucket.next_generation),
            crc,
        )
        self._load()
        if self.bucket.clear_generation_suffix and self.name.endswith(
            self.bucket.clear_generation_suffix
        ):
            self.generation = None

    def reload(self, *, if_generation_match: int) -> None:
        self.bucket.operations.append(f"reload:{self.name}")
        if self.name not in self.bucket.objects:
            raise RuntimeError("absent object")
        self._load()
        if str(if_generation_match) != self.generation:
            raise RuntimeError("generation mismatch")

    def download_as_bytes(self, *, if_generation_match: int, checksum: str) -> bytes:
        assert checksum == "crc32c"
        self.bucket.operations.append(f"download:{self.name}")
        self._load()
        if str(if_generation_match) != self.generation:
            raise RuntimeError("generation mismatch")
        raw = self.bucket.objects[self.name][0]
        if self.bucket.corrupt_download_suffix and self.name.endswith(
            self.bucket.corrupt_download_suffix
        ):
            return raw + b"corrupt"
        return raw


class _FakeBucket:
    name = "driftbench-dsv4-uc"

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str, str]] = {}
        self.operations: list[str] = []
        self.mutations: list[str] = []
        self.next_generation = 100
        self.inject_before_suffix: str | None = None
        self.corrupt_download_suffix: str | None = None
        self.clear_generation_suffix: str | None = None

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(self, name)

    def list_blobs(self, *, prefix: str) -> list[_FakeBlob]:
        self.operations.append(f"list:{prefix}")
        return [
            self.blob(name) for name in sorted(self.objects) if name.startswith(prefix)
        ]


def _fake_success_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str]:
    run_root = tmp_path / "glm-run"
    run_root.mkdir(mode=0o700)
    monkeypatch.setattr(PUBLISHER_MODULE, "RUN_ROOT", run_root)
    tag = "gate_d_compensated_pp16_hlo_20260831T123456123456789Z"
    run = run_root / tag
    PUBLISHER_MODULE.initialize_run_dir(
        run, publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW
    )
    optimized = b"HloModule test\n"
    stable = b"module @test {}\n"
    dependencies = {
        "artifact_kind": "gate_d_compensated_pp16_compiler_dependencies",
        "code_hash": "1" * 40,
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
        "python_modules": [
            {
                "bytes": 1,
                "device": 1,
                "inode": 2,
                "path": "/usr/lib/python.py",
                "sha256": "3" * 64,
            }
        ],
        "sealed_project_source": {
            "archive_sha256": "4" * 64,
            "file_manifest_count": 10,
            "file_manifest_sha256": "5" * 64,
        },
    }
    dependencies_raw = PUBLISHER_MODULE._canonical(dependencies)
    runner = {
        "artifact_kind": "gate_d_compensated_pp16_optimized_hlo_acquisition",
        "claim_scope": "compile only",
        "code_hash": "1" * 40,
        "compile_only": True,
        "compiler_dependency_manifest": {
            "byte_count": len(dependencies_raw),
            "filename": "dependencies.json",
            "native_mapping_count": 1,
            "python_module_count": 1,
            "sha256": sha256(dependencies_raw).hexdigest(),
        },
        "compiled_executable_invocation_count": 0,
        "gate_d_closed": False,
        "hlo": {
            "optimized": {
                "byte_count": len(optimized),
                "filename": "compensated_pp16_stage0.optimized_hlo.txt",
                "sha256": sha256(optimized).hexdigest(),
            },
            "stablehlo": {
                "byte_count": len(stable),
                "filename": "compensated_pp16_stage0.stablehlo.mlir",
                "sha256": sha256(stable).hexdigest(),
            },
        },
        "numerical_claim": False,
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
        "status": "HLO_ACQUIRED_UNADJUDICATED",
        "sealed_project_source": {
            "archive_sha256": "4" * 64,
            "file_manifest_count": 10,
            "file_manifest_sha256": "5" * 64,
            "loaded_module_count": 3,
        },
        "tpu_numerical_execution_performed": False,
    }
    (run / "runner.json").write_bytes(PUBLISHER_MODULE._canonical(runner))
    (run / "dependencies.json").write_bytes(dependencies_raw)
    (run / "hlo/compensated_pp16_stage0.optimized_hlo.txt").write_bytes(optimized)
    (run / "hlo/compensated_pp16_stage0.stablehlo.mlir").write_bytes(stable)
    census = "".join(f"CENSUS_OK host-{index}\n" for index in range(8))
    for name in ("census_pre.txt", "census_post.txt"):
        (run / name).write_text(census)
    for name in (
        "mirror.sha256",
        "remote_vacancy.raw.txt",
        "remote_vacancy.txt",
        "runner.log",
        "sync.txt",
        "orchestrator.log",
    ):
        (run / name).write_text(f"{name}\n")
    remote = f"gs://driftbench-dsv4-uc/results/greenfield/glm52/gate_d_pp16_hlo/{tag}"
    return run, remote


def test_remote_publication_is_generation_zero_and_terminal_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    bucket = _FakeBucket()
    PUBLISHER_MODULE.publish_success(
        run,
        remote,
        code_pin="1" * 40,
        elapsed=7,
        publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
        storage_bucket=bucket,
    )
    assert bucket.mutations[-1].endswith("/HLO_ACQUIRED")
    assert any(name.endswith("/remote_objects.json") for name in bucket.objects)
    assert any(name.endswith("/HLO_ACQUIRED") for name in bucket.objects)
    assert (run / "terminal_upload_receipt.json").is_file()


def test_remote_concurrent_insertion_never_publishes_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    bucket = _FakeBucket()
    bucket.inject_before_suffix = "/runner.json"
    with pytest.raises(RuntimeError, match="generation-zero"):
        PUBLISHER_MODULE.publish_success(
            run,
            remote,
            code_pin="1" * 40,
            elapsed=7,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=bucket,
        )
    assert not any(name.endswith("/HLO_ACQUIRED") for name in bucket.objects)


@pytest.mark.parametrize("kind", ("file", "directory"))
def test_unexpected_local_member_refuses_before_remote_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    if kind == "file":
        (run / "unexpected.txt").write_text("hostile\n")
    else:
        (run / "unexpected").mkdir()
    bucket = _FakeBucket()
    with pytest.raises(RuntimeError, match="local preterminal|unexpected local"):
        PUBLISHER_MODULE.publish_success(
            run,
            remote,
            code_pin="1" * 40,
            elapsed=7,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=bucket,
        )
    assert not bucket.mutations


def test_generation_qualified_download_detects_remote_byte_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    bucket = _FakeBucket()
    bucket.corrupt_download_suffix = "/runner.json"
    with pytest.raises(RuntimeError, match="generation-qualified bytes"):
        PUBLISHER_MODULE.publish_success(
            run,
            remote,
            code_pin="1" * 40,
            elapsed=7,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=bucket,
        )
    assert not any(name.endswith("/HLO_ACQUIRED") for name in bucket.objects)


def test_terminal_response_without_generation_leaves_no_local_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    bucket = _FakeBucket()
    bucket.clear_generation_suffix = "/HLO_ACQUIRED"
    with pytest.raises(RuntimeError, match="returned no generation"):
        PUBLISHER_MODULE.publish_success(
            run,
            remote,
            code_pin="1" * 40,
            elapsed=7,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=bucket,
        )
    assert bucket.mutations[-1].endswith("/HLO_ACQUIRED")
    assert not (run / "terminal_upload_receipt.json").exists()


def test_diagnostic_refuses_after_terminal_publication_begins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    (run / "HLO_ACQUIRED").write_text("terminal-attempt\n")
    bucket = _FakeBucket()
    with pytest.raises(RuntimeError, match="terminal publication.*begun"):
        PUBLISHER_MODULE.publish_diagnostic(
            run,
            remote,
            code_pin="1" * 40,
            status=19,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=bucket,
        )
    assert not bucket.mutations
    assert not (run / "failure_status.json").exists()


def test_diagnostic_uses_generation_zero_ledger_last_without_success_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    bucket = _FakeBucket()
    PUBLISHER_MODULE.publish_diagnostic(
        run,
        remote,
        code_pin="1" * 40,
        status=19,
        publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
        storage_bucket=bucket,
    )
    assert bucket.mutations[-1].endswith("/diagnostic/diagnostic_objects.json")
    assert not any(name.endswith("/HLO_ACQUIRED") for name in bucket.objects)
    assert any(name.endswith("/failure_status.json") for name in bucket.objects)
    assert (run / "diagnostic_upload_receipt.json").is_file()


def test_preterminal_writers_refuse_after_diagnostic_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    bucket = _FakeBucket()
    PUBLISHER_MODULE.publish_diagnostic(
        run,
        remote,
        code_pin="1" * 40,
        status=19,
        publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
        storage_bucket=bucket,
    )
    run_fd = PUBLISHER_MODULE._run_fd(run)
    try:
        with pytest.raises(RuntimeError, match="already begun"):
            PUBLISHER_MODULE.write_preterminal_member(
                run_fd, "sync.txt", BytesIO(b"forbidden\n")
            )
        with pytest.raises(RuntimeError, match="already begun"):
            PUBLISHER_MODULE.append_preterminal_log(run_fd, b"forbidden\n")
    finally:
        os.close(run_fd)
