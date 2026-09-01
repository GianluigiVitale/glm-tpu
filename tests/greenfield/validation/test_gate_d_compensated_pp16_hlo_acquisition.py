from __future__ import annotations

import ast
import base64
import importlib.util
import json
import os
import re
import shlex
import stat
import subprocess
import zipfile
import zipimport
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[3]
CPU_REPLAY = ROOT / "glm_tpu/greenfield/benchmarking/gate_d_compensated_capsule.py"
TPU_REPLAY = ROOT / "glm_tpu/greenfield/benchmarking/gate_d_compensated_pp16_hlo.py"
DRIVER = ROOT / "scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py"
WRAPPER = ROOT / "scripts/greenfield/run_gate_d_compensated_pp16_hlo.sh"
PUBLISHER = ROOT / "scripts/greenfield/publish_gate_d_compensated_pp16_hlo.py"
STORAGE_SITE_BUILDER = ROOT / "scripts/greenfield/build_gate_d_storage_site_capsule.py"
STORAGE_SITE_BUILD = (
    ROOT / "docs/artifacts/gate-d-compensated-pp16-storage-site-build.json"
)
ADMISSION = (
    ROOT / "docs/artifacts/gate-d-precompile-admission-v2-compensated-capsule.json"
)
TOPOLOGY = ROOT / "docs/artifacts/gate-d-runtime-locality-authority.json"
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


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


def _accelerator_device_record(
    path: str = "/dev/accel2",
) -> dict[str, int | str]:
    suffix = int(path[-1])
    node_device = os.makedev(0, 5)
    return {
        "gid": 0,
        "mapped_device_major": 0,
        "mapped_device_minor": 5,
        "mapped_inode": 373,
        "mode": 0o666,
        "nlink": 1,
        "node_device": node_device,
        "node_inode": 373,
        "path": path,
        "rdev_major": 121,
        "rdev_minor": suffix,
        "uid": 0,
    }


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
    assert "_validate_python_runtime" in source
    assert source.index(
        "dependency_sites = _validate_dependency_sites()"
    ) < source.index("import jax")
    assert source.rindex("_validate_dependency_sites()") > source.index(
        "compiled = lowered.compile()"
    )
    assert source.rindex("_validate_python_runtime_storage()") > source.index(
        "compiled = lowered.compile()"
    )
    assert "/usr/bin/env -i" in WRAPPER.read_text()
    assert "-I -S -B -u" in WRAPPER.read_text()
    assert 'PYTHONPATH="$WORKTREE"' not in WRAPPER.read_text()


def test_driver_uses_exact_root_owned_python_runtime() -> None:
    wrapper = WRAPPER.read_text()
    expected = Path("/opt/glm-tpu/gate-d-python-3.12.13-021044895e95")
    assert DRIVER_MODULE.PYTHON_RUNTIME_ROOT == expected
    assert DRIVER_MODULE.PYTHON == expected / "bin/python3.12"
    assert f"readonly DRIVER_PYTHON={DRIVER_MODULE.PYTHON}\n" in wrapper
    assert "readonly DRIVER_PYTHON=/home/gianl/vllm-env/bin/python\n" not in wrapper
    assert Path(os.path.realpath(DRIVER_MODULE.PYTHON)) == DRIVER_MODULE.PYTHON

    probe = (
        "import importlib.util,json;"
        f"p={str(DRIVER)!r};"
        "s=importlib.util.spec_from_file_location('gate_d_driver_runtime_probe',p);"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        "print(json.dumps(m._validate_python_runtime(),sort_keys=True))"
    )
    completed = subprocess.run(
        [str(DRIVER_MODULE.PYTHON), "-I", "-S", "-B", "-c", probe],
        check=True,
        capture_output=True,
        env={
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "python_executable": str(DRIVER_MODULE.PYTHON),
        "python_runtime_root": str(DRIVER_MODULE.PYTHON_RUNTIME_ROOT),
        "python_runtime_tree_sha256": DRIVER_MODULE.PYTHON_RUNTIME_TREE_SHA256,
        "python_sha256": DRIVER_MODULE.PYTHON_SHA256,
    }

    mutable_interpreter = Path("/home/gianl/vllm-env/bin/python")
    rejected = subprocess.run(
        [str(mutable_interpreter), "-I", "-S", "-B", "-c", probe],
        check=False,
        capture_output=True,
        env={
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
    )
    assert rejected.returncode != 0
    assert "compiler Python runtime boundary drifted" in rejected.stderr


def test_compiler_dependency_roots_exclude_mutable_uv_runtime() -> None:
    source = DRIVER.read_text()
    function = _function(DRIVER, "_allowed_compiler_dependency_roots")
    literals = {
        node.value
        for node in ast.walk(function)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "/home/gianl/vllm-env" not in literals
    assert "/home/gianl/.local/share/uv/python" not in source
    assert "PYTHON_RUNTIME_ROOT" in ast.unparse(function)
    assert "JAX_SITE_ROOT" in ast.unparse(function)
    assert "LIBTPU_SITE_ROOT" in ast.unparse(function)
    with pytest.raises(RuntimeError, match="native mapping escaped allowed roots"):
        DRIVER_MODULE._validate_native_mapping_root(
            Path("/home/gianl/vllm-env/lib/python3.12/site-packages/libtpu.so")
        )


def _fake_accelerator_metadata(**changes: int) -> SimpleNamespace:
    values = {
        "st_dev": os.makedev(0, 5),
        "st_gid": 0,
        "st_ino": 373,
        "st_mode": stat.S_IFCHR | 0o666,
        "st_nlink": 1,
        "st_rdev": os.makedev(121, 2),
        "st_uid": 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _fake_accelerator_directory_metadata(**changes: int) -> SimpleNamespace:
    values = {
        "st_dev": os.makedev(0, 5),
        "st_gid": 0,
        "st_ino": 101,
        "st_mode": stat.S_IFDIR | 0o755,
        "st_uid": 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def mocked_accelerator_directory_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        DRIVER_MODULE,
        "_verify_accelerator_device_directory_named",
        lambda **kwargs: None,
    )


def test_accelerator_device_mapping_is_raw_exact_and_identity_bound(
    monkeypatch: pytest.MonkeyPatch,
    mocked_accelerator_directory_identity: None,
) -> None:
    metadata = _fake_accelerator_metadata()
    monkeypatch.setattr(DRIVER_MODULE.os, "stat", lambda *args, **kwargs: metadata)
    assert (
        DRIVER_MODULE._accelerator_device_mapping_record(
            "/dev/accel2",
            (0, 5, 373),
            root_directory_fd=16,
            device_directory_fd=17,
        )
        == _accelerator_device_record()
    )


@pytest.mark.parametrize(
    ("raw_path", "mapped_identity"),
    (
        ("/tmp/accel2", (0, 5, 373)),
        ("/dev/accel2 (deleted)", (0, 5, 373)),
        ("/dev/accel4", (0, 5, 373)),
        ("/dev/accel2", (0, 5, 999)),
    ),
)
def test_accelerator_device_mapping_alias_deleted_suffix_and_map_drift_reject(
    raw_path: str,
    mapped_identity: tuple[int, int, int],
    monkeypatch: pytest.MonkeyPatch,
    mocked_accelerator_directory_identity: None,
) -> None:
    metadata = _fake_accelerator_metadata()
    monkeypatch.setattr(DRIVER_MODULE.os, "stat", lambda *args, **kwargs: metadata)
    with pytest.raises(RuntimeError, match="unsupported non-regular|identity drifted"):
        DRIVER_MODULE._accelerator_device_mapping_record(
            raw_path,
            mapped_identity,
            root_directory_fd=16,
            device_directory_fd=17,
        )


def test_accelerator_device_mapping_symlink_normalization_rejects(
    monkeypatch: pytest.MonkeyPatch,
    mocked_accelerator_directory_identity: None,
) -> None:
    monkeypatch.setattr(
        DRIVER_MODULE.os.path,
        "realpath",
        lambda path: "/dev/accel1" if path == "/dev/accel2" else path,
    )
    with pytest.raises(RuntimeError, match="unsupported non-regular"):
        DRIVER_MODULE._accelerator_device_mapping_record(
            "/dev/accel2",
            (0, 5, 373),
            root_directory_fd=16,
            device_directory_fd=17,
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"st_mode": stat.S_IFREG | 0o666},
        {"st_uid": 1},
        {"st_gid": 1},
        {"st_mode": stat.S_IFCHR | 0o660},
        {"st_nlink": 2},
        {"st_rdev": os.makedev(120, 2)},
        {"st_rdev": os.makedev(121, 1)},
    ),
)
def test_accelerator_device_mapping_type_owner_mode_and_rdev_reject(
    changes: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
    mocked_accelerator_directory_identity: None,
) -> None:
    metadata = _fake_accelerator_metadata(**changes)
    monkeypatch.setattr(DRIVER_MODULE.os, "stat", lambda *args, **kwargs: metadata)
    with pytest.raises(RuntimeError, match="identity drifted"):
        DRIVER_MODULE._accelerator_device_mapping_record(
            "/dev/accel2",
            (0, 5, 373),
            root_directory_fd=16,
            device_directory_fd=17,
        )


def test_accelerator_device_mapping_named_node_replacement_rejects(
    monkeypatch: pytest.MonkeyPatch,
    mocked_accelerator_directory_identity: None,
) -> None:
    snapshots = iter(
        (_fake_accelerator_metadata(), _fake_accelerator_metadata(st_ino=374))
    )
    monkeypatch.setattr(
        DRIVER_MODULE.os, "stat", lambda *args, **kwargs: next(snapshots)
    )
    with pytest.raises(RuntimeError, match="mapping changed"):
        DRIVER_MODULE._accelerator_device_mapping_record(
            "/dev/accel2",
            (0, 5, 373),
            root_directory_fd=16,
            device_directory_fd=17,
        )


def test_arbitrary_nonregular_compiler_mapping_rejects() -> None:
    metadata = Path("/dev/null").stat()
    mapped_identity = (
        os.major(metadata.st_dev),
        os.minor(metadata.st_dev),
        metadata.st_ino,
    )
    root_directory_fd, device_directory_fd = (
        DRIVER_MODULE._open_accelerator_device_directory()
    )
    try:
        with pytest.raises(RuntimeError, match="unsupported non-regular"):
            DRIVER_MODULE._classify_compiler_mapping(
                "/dev/null",
                mapped_identity,
                root_directory_fd=root_directory_fd,
                device_directory_fd=device_directory_fd,
            )
    finally:
        os.close(device_directory_fd)
        os.close(root_directory_fd)


@pytest.mark.parametrize("module", (DRIVER_MODULE, PUBLISHER_MODULE))
def test_accelerator_device_directory_named_substitution_rejects(
    module: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = _fake_accelerator_directory_metadata()
    named_replacement = _fake_accelerator_directory_metadata(st_ino=102)
    monkeypatch.setattr(module.os, "fstat", lambda descriptor: held)
    monkeypatch.setattr(module.os, "stat", lambda *args, **kwargs: named_replacement)
    with pytest.raises(RuntimeError, match="directory identity drifted"):
        module._verify_accelerator_device_directory_named(
            root_directory_fd=16,
            device_directory_fd=17,
        )


def test_compiler_import_path_and_dependency_sites_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_archive = "/proc/self/fd/999"
    expected_path = [
        source_archive,
        str(DRIVER_MODULE.JAX_SITE_ROOT),
        str(DRIVER_MODULE.LIBTPU_SITE_ROOT),
        *DRIVER_MODULE._EXPECTED_RUNTIME_PATH,
    ]
    monkeypatch.setattr(DRIVER_MODULE.sys, "path", expected_path)
    DRIVER_MODULE._validate_compiler_import_path(source_archive)
    monkeypatch.setattr(
        DRIVER_MODULE.sys,
        "path",
        [*expected_path, "/home/gianl/vllm-env/lib/python3.12/site-packages"],
    )
    with pytest.raises(RuntimeError, match="compiler import path drifted"):
        DRIVER_MODULE._validate_compiler_import_path(source_archive)

    mutable_site = tmp_path / "jax-site"
    mutable_site.mkdir()
    (mutable_site / "CAPSULE_MANIFEST.json").write_text("{}\n")
    monkeypatch.setattr(DRIVER_MODULE, "JAX_SITE_ROOT", mutable_site)
    with pytest.raises(RuntimeError, match="root-owned sealed|dependency site drifted"):
        DRIVER_MODULE._validate_dependency_sites()


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
        "accelerator_device_nodes_observed_mapped": [_accelerator_device_record()],
        "native_mappings": [{"path": "/usr/lib/a", "sha256": "1" * 64}],
        "python_modules": [{"path": "/usr/lib/b", "sha256": "2" * 64}],
    }
    after = {
        "accelerator_device_nodes_observed_mapped": [_accelerator_device_record()],
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
    after = deepcopy(before)
    after["accelerator_device_nodes_observed_mapped"] = []
    with pytest.raises(RuntimeError, match="changed during compile"):
        DRIVER_MODULE._require_dependency_prefix_stable(before, after)
    after = deepcopy(before)
    after["accelerator_device_nodes_observed_mapped"][0]["node_inode"] = 374
    with pytest.raises(RuntimeError, match="changed during compile"):
        DRIVER_MODULE._require_dependency_prefix_stable(before, after)


def test_dependency_path_order_is_identical_for_numpy_and_numpy_libs() -> None:
    site = PUBLISHER_MODULE.JAX_SITE_ROOT
    paths = [
        str(site / "numpy/_core/_multi.so"),
        str(site / "numpy.libs/libopenblas.so"),
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


def test_accelerator_device_digest_projection_matches_producer_and_publisher() -> None:
    records = [_accelerator_device_record()]
    assert DRIVER_MODULE._accelerator_device_nodes_sha256(
        records
    ) == PUBLISHER_MODULE._accelerator_device_nodes_sha256(records)
    assert (DRIVER_MODULE._canonical(records) + "\n").encode(
        "ascii"
    ) == PUBLISHER_MODULE._canonical(records)


def test_publisher_rejects_mutable_compiler_dependency_root() -> None:
    record = {
        "bytes": 1,
        "device": 1,
        "inode": 1,
        "path": "/home/gianl/vllm-env/lib/python3.12/site-packages/libtpu/libtpu.so",
        "sha256": "1" * 64,
    }
    with pytest.raises(RuntimeError, match="dependency escaped allowed roots"):
        PUBLISHER_MODULE._validate_dependency_records([record], "native_mappings")


def test_publisher_accepts_exact_accelerator_device_node_record() -> None:
    PUBLISHER_MODULE._validate_accelerator_device_mapping_records(
        [_accelerator_device_record()]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("uid", 1),
        ("gid", 1),
        ("mode", 0o660),
        ("nlink", 2),
        ("rdev_major", 120),
        ("rdev_minor", 1),
        ("mapped_inode", True),
        ("mapped_inode", 0),
        ("node_inode", 374),
        ("mapped_device_minor", 6),
        ("path", "/dev/accel2 (deleted)"),
        ("path", "/tmp/accel2"),
    ),
)
def test_publisher_rejects_accelerator_device_node_schema_drift(
    field: str, value: object
) -> None:
    record = _accelerator_device_record()
    record[field] = value
    with pytest.raises(RuntimeError, match="mapping identity drifted"):
        PUBLISHER_MODULE._validate_accelerator_device_mapping_records([record])


def test_publisher_rejects_accelerator_extra_key_duplicates_and_order() -> None:
    extra = {**_accelerator_device_record(), "unexpected": 1}
    with pytest.raises(RuntimeError, match="mapping record drifted"):
        PUBLISHER_MODULE._validate_accelerator_device_mapping_records([extra])
    duplicate = [_accelerator_device_record(), _accelerator_device_record()]
    with pytest.raises(
        RuntimeError, match="mapping identity is duplicated|order drifted"
    ):
        PUBLISHER_MODULE._validate_accelerator_device_mapping_records(duplicate)
    first = _accelerator_device_record("/dev/accel1")
    first["mapped_inode"] = first["node_inode"] = 372
    second = _accelerator_device_record("/dev/accel2")
    with pytest.raises(RuntimeError, match="mapping order drifted"):
        PUBLISHER_MODULE._validate_accelerator_device_mapping_records([second, first])
    duplicate_identity = _accelerator_device_record("/dev/accel1")
    with pytest.raises(RuntimeError, match="identity is duplicated"):
        PUBLISHER_MODULE._validate_accelerator_device_mapping_records(
            [duplicate_identity, second]
        )


def test_publisher_rejects_live_accelerator_node_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = iter(
        (_fake_accelerator_metadata(), _fake_accelerator_metadata(st_ino=374))
    )
    monkeypatch.setattr(
        PUBLISHER_MODULE.os, "stat", lambda *args, **kwargs: next(snapshots)
    )
    monkeypatch.setattr(
        PUBLISHER_MODULE,
        "_verify_accelerator_device_directory_named",
        lambda **kwargs: None,
    )
    with pytest.raises(RuntimeError, match="mapping changed"):
        PUBLISHER_MODULE._verify_accelerator_device_mapping_record_live(
            _accelerator_device_record(),
            root_directory_fd=16,
            device_directory_fd=17,
        )


@pytest.mark.parametrize(
    "path",
    (
        "/usr/../home/gianl/evil.so",
        "/opt/glm-tpu/gate-d-jax-site-55233c63939e/../../../../home/gianl/evil.so",
        "/usr//lib/evil.so",
        "/lib",
    ),
)
def test_publisher_rejects_noncanonical_dependency_paths(path: str) -> None:
    record = {
        "bytes": 1,
        "device": 1,
        "inode": 1,
        "path": path,
        "sha256": "1" * 64,
    }
    with pytest.raises(RuntimeError, match="dependency identity drifted"):
        PUBLISHER_MODULE._validate_dependency_records([record], "native_mappings")


@pytest.mark.parametrize("bad_sha", (True, None, 1.0, ["1" * 64]))
def test_publisher_requires_exact_sha_type(bad_sha: object) -> None:
    record = {
        "bytes": 1,
        "device": 1,
        "inode": 1,
        "path": "/usr/lib/example.so",
        "sha256": bad_sha,
    }
    with pytest.raises(RuntimeError, match="dependency identity drifted"):
        PUBLISHER_MODULE._validate_dependency_records([record], "native_mappings")


def test_publisher_requires_exact_sha_string_and_live_record() -> None:
    path = Path(os.path.realpath("/usr/bin/true"))
    metadata = path.stat()
    record = {
        "bytes": metadata.st_size,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "path": str(path),
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }
    PUBLISHER_MODULE._validate_dependency_records(
        [record], "native_mappings", verify_live=True
    )
    integer_sha = {**record, "sha256": int("1" * 64)}
    with pytest.raises(RuntimeError, match="dependency identity drifted"):
        PUBLISHER_MODULE._validate_dependency_records([integer_sha], "native_mappings")
    wrong_sha = {**record, "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="live bytes drifted"):
        PUBLISHER_MODULE._validate_dependency_records(
            [wrong_sha], "native_mappings", verify_live=True
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
    assert PUBLISHER_MODULE._EXPECTED_COMPILER_ENVIRONMENT["PATH"] == "/usr/bin:/bin"
    assert (
        "PATH=/usr/bin:/bin \\\n"
        "    PYTHONDONTWRITEBYTECODE=1 \\\n"
        '    "$PUBLISHER_PYTHON"'
    ) in wrapper
    assert (
        "PATH=/usr/bin:/bin \\\n"
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
    assert (
        PUBLISHER_MODULE._validate_storage_site(
            root,
            expected_tree_sha256=tree_sha,
            expected_manifest_sha256=manifest_sha,
            require_sealed=False,
        )
        == manifest
    )
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


def test_wrapper_local_assignments_have_no_same_command_dependency() -> None:
    unsafe: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(WRAPPER.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("local "):
            continue
        declared: list[str] = []
        for token in shlex.split(stripped)[1:]:
            match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", token)
            if match is None:
                continue
            name, value = match.groups()
            for earlier in declared:
                reference = re.compile(
                    rf"\$(?:{re.escape(earlier)}\b|\{{{re.escape(earlier)}(?:\}}|[:[]))"
                )
                if reference.search(value):
                    unsafe.append((line_number, name, earlier))
            declared.append(name)
    assert unsafe == []


def test_strict_census_label_is_bound_before_member_under_nounset() -> None:
    wrapper = WRAPPER.read_text()
    fragment = (
        'strict_census() {\n  local label=$1\n  local member="census_${label}.txt"\n'
    )
    assert fragment in wrapper
    completed = subprocess.run(
        [
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            "-uc",
            (
                'f(){ local label=$1; local member="census_${label}.txt"; '
                'printf "%s\\n" "$member"; }; f pre'
            ),
        ],
        check=False,
        capture_output=True,
        env={},
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stdout == "census_pre.txt\n"
    assert completed.stderr == ""


def test_repository_verification_is_read_only_serial_and_fail_fast() -> None:
    wrapper = WRAPPER.read_text()
    block = wrapper.split(
        'say "verifying exact pushed pin read-only across all eight hosts"', 1
    )[1].split(
        'say "lowering and compiling one abstract-input PP16 stage-zero graph', 1
    )[0]
    assert "--worker=all" not in block
    assert "for worker in 0 1 2 3 4 5 6 7; do" in block
    assert '--worker="$worker"' in block
    assert "git fetch" not in block
    assert "git checkout" not in block
    assert "git clone" not in block
    assert "git reset" not in block
    assert "status --porcelain" not in block
    assert "/usr/bin/env -i" in block
    assert "GIT_CONFIG_GLOBAL=/dev/null" in block
    assert "GIT_CONFIG_NOSYSTEM=1" in block
    assert "GIT_NO_LAZY_FETCH=1" in block
    assert "GIT_NO_REPLACE_OBJECTS=1" in block
    assert "GIT_OPTIONAL_LOCKS=0" in block
    assert "GIT_PROTOCOL_FROM_USER=0" in block
    assert "GIT_SSH_COMMAND=/bin/false" in block
    assert "/usr/bin/git" in block
    assert "-c core.fsmonitor=false" in block
    assert "-c core.untrackedCache=false" in block
    assert "-c core.preloadIndex=false" in block
    assert 'rev-parse --verify "$pin^{commit}"' in block
    assert 'cat-file -t "$pin"' in block
    assert '[[ -d "$wt/.git" && ! -L "$wt/.git" ]]' in block
    assert '[[ -f "$wt/.git" && ! -L "$wt/.git" ]]' in block
    assert (
        '[[ "$expected_layout" == standalone || "$expected_layout" == standalone_promisor ]]'
        in block
    )
    assert '[[ "$expected_layout" == linked ]]' in block
    assert "[[ $worker -eq 0 ]] && expected_layout=linked" in block
    assert "linked_pointer_bytes" in block
    assert "linked_backpointer_bytes" in block
    assert "linked_commondir_bytes" in block
    assert "linked_config_boundary" in block
    assert "linked_index_boundary" in block
    assert "config_not_allowlisted" in block
    assert "rev-parse --show-toplevel" in block
    assert '[[ "$toplevel" == "$wt" ]]' in block
    assert '[[ "$git_dir" == "$expected_git_dir" ]]' in block
    assert '[[ "$common_dir" == "$expected_common_dir" ]]' in block
    assert '[[ "$index_path" == "$expected_index" ]]' in block
    assert '[[ "$sparse_checkout_path" == "$expected_sparse_checkout" ]]' in block
    assert "rev-parse --is-shallow-repository" in block
    assert '"$object_dir/info/alternates"' in block
    assert '[[ "$grafts_path" == "$common_dir/info/grafts" ]]' in block
    assert "canonical_promisor_catalogue" in block
    assert 're.compile(r"pack/pack-[0-9a-f]{40}[.]promisor")' in block
    assert "os.fwalk(" in block
    assert "follow_symlinks=False" in block
    assert "value.st_nlink != 1" in block
    assert "value.st_size != 0" in block
    assert "catalogue.append(0)" in block
    assert "expected_missing_count=$7" in block
    assert "expected_promisor_sha=${10}" in block
    assert '[[ "$missing_count" == "$expected_missing_count" ]]' in block
    assert '[[ "$promisor_sha" == "$expected_promisor_sha" ]]' in block
    assert "for-each-ref --format='%(refname)' refs/replace" in block
    assert "ls-files -v -z" in block
    assert "special_index_flag" in block
    assert 'diff-index --quiet --no-ext-diff --no-textconv "$pin" --' in block
    assert "ls-files --others --exclude-standard" in block
    assert 'fsck --connectivity-only --strict --no-dangling "$pin"' in block
    assert "if [[ $worker_status -ne 0 ]]; then" in block
    assert "break" in block


def _worker_repo_verify_script() -> str:
    source = WRAPPER.read_text()
    return source.split("<<'WORKER_REPO_VERIFY_EOF' || true\n", 1)[1].split(
        "\nWORKER_REPO_VERIFY_EOF", 1
    )[0]


def _run_worker_repo_verify(
    repo: Path,
    pin: str,
    origin: str,
    *,
    linked_common: Path | None = None,
    linked_git_dir: Path | None = None,
    expected_layout: str | None = None,
    expected_missing_count: int = 0,
    expected_missing_sha: str = EMPTY_SHA256,
    expected_promisor_count: int = 0,
    expected_promisor_sha: str = EMPTY_SHA256,
) -> subprocess.CompletedProcess[str]:
    common = linked_common if linked_common is not None else repo / ".git"
    admin = (
        linked_git_dir
        if linked_git_dir is not None
        else common / "worktrees" / repo.name
    )
    layout = (
        expected_layout
        if expected_layout is not None
        else ("linked" if linked_common is not None else "standalone")
    )
    return subprocess.run(
        [
            "/usr/bin/env",
            "-i",
            "HOME=/home/gianl",
            "LANG=C",
            "LC_ALL=C",
            "PATH=/usr/bin:/bin",
            "GIT_CONFIG_GLOBAL=/dev/null",
            "GIT_CONFIG_NOSYSTEM=1",
            "GIT_NO_LAZY_FETCH=1",
            "GIT_NO_REPLACE_OBJECTS=1",
            "GIT_OPTIONAL_LOCKS=0",
            "GIT_PROTOCOL_FROM_USER=0",
            "GIT_TERMINAL_PROMPT=0",
            "GIT_SSH_COMMAND=/bin/false",
            "/usr/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            _worker_repo_verify_script(),
            "gate-d-worker-repo-verify-test",
            pin,
            str(repo),
            origin,
            str(common),
            str(admin),
            layout,
            str(expected_missing_count),
            expected_missing_sha,
            str(expected_promisor_count),
            expected_promisor_sha,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _repository_metadata_snapshot(root: Path) -> dict[str, tuple[object, ...]]:
    records: dict[str, tuple[object, ...]] = {}
    for path in sorted((root, *root.rglob("*")), key=lambda item: str(item)):
        metadata = path.lstat()
        kind = stat.S_IFMT(metadata.st_mode)
        payload = ""
        if stat.S_ISREG(metadata.st_mode):
            payload = sha256(path.read_bytes()).hexdigest()
        elif stat.S_ISLNK(metadata.st_mode):
            payload = os.readlink(path)
        records[str(path.relative_to(root))] = (
            kind,
            stat.S_IMODE(metadata.st_mode),
            metadata.st_size,
            metadata.st_ino,
            metadata.st_nlink,
            metadata.st_uid,
            metadata.st_gid,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
            payload,
        )
    return records


def test_exact_worker_repo_verifier_is_no_write_and_fails_hostile_closure(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git_environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Gate D Test",
        "GIT_AUTHOR_EMAIL": "gate-d@example.invalid",
        "GIT_COMMITTER_NAME": "Gate D Test",
        "GIT_COMMITTER_EMAIL": "gate-d@example.invalid",
    }

    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/usr/bin/git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            env=git_environment,
            text=True,
        )

    git("init", "-q")
    origin = "git@example.invalid:owner/repo.git"
    git("remote", "add", "origin", origin)
    (repo / "tracked.txt").write_text("sealed worker repository\n")
    (repo / ".gitattributes").write_text("tracked.txt filter=hostile diff=hostile\n")
    git("add", "tracked.txt", ".gitattributes")
    git("commit", "-q", "-m", "seed")
    pin = git("rev-parse", "HEAD").stdout.strip()

    sentinel = tmp_path / "fsmonitor-invoked"
    fsmonitor = tmp_path / "hostile-fsmonitor"
    fsmonitor.write_text(f"#!/usr/bin/bash\n/usr/bin/touch {sentinel}\n")
    fsmonitor.chmod(0o755)
    git("config", "core.fsmonitor", str(fsmonitor))
    git("config", "core.untrackedCache", "true")

    before = _repository_metadata_snapshot(repo)
    completed = _run_worker_repo_verify(repo, pin, origin)
    after = _repository_metadata_snapshot(repo)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("SYNC_OK ")
    assert completed.stderr == ""
    assert not sentinel.exists()
    assert after == before

    completed = _run_worker_repo_verify(repo, pin, origin, expected_layout="linked")
    assert completed.returncode != 0
    assert "unexpected_layout" in completed.stderr

    external = tmp_path / "hostile-filter-diff"
    external.write_text(f"#!/usr/bin/bash\n/usr/bin/touch {sentinel}\n/bin/cat\n")
    external.chmod(0o755)
    git("config", "filter.hostile.clean", str(external))
    git("config", "filter.hostile.smudge", str(external))
    git("config", "diff.hostile.command", str(external))
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "config_not_allowlisted" in completed.stderr
    assert not sentinel.exists()
    git("config", "--unset-all", "filter.hostile.clean")
    git("config", "--unset-all", "filter.hostile.smudge")
    git("config", "--unset-all", "diff.hostile.command")

    redirected = tmp_path / "redirected-worktree"
    redirected.mkdir()
    git("config", "core.worktree", str(redirected))
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "config_not_allowlisted" in completed.stderr
    git("config", "--unset-all", "core.worktree")

    git("update-index", "--assume-unchanged", "tracked.txt")
    (repo / "tracked.txt").write_text("dirty but assumed unchanged\n")
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "special_index_flag" in completed.stderr
    git("update-index", "--no-assume-unchanged", "tracked.txt")
    (repo / "tracked.txt").write_text("sealed worker repository\n")

    git("update-index", "--skip-worktree", "tracked.txt")
    (repo / "tracked.txt").write_text("dirty but skipped\n")
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "special_index_flag" in completed.stderr
    git("update-index", "--no-skip-worktree", "tracked.txt")
    (repo / "tracked.txt").write_text("sealed worker repository\n")

    git("config", "remote.origin.promisor", "true")
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "config_not_allowlisted" in completed.stderr
    git("config", "--unset", "remote.origin.promisor")

    git("config", "core.repositoryFormatVersion", "1")
    git("config", "extensions.PartialClone", "origin")
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "config_not_allowlisted" in completed.stderr
    git("config", "--unset", "extensions.PartialClone")
    git("config", "core.repositoryFormatVersion", "0")

    alternates = repo / ".git/objects/info/alternates"
    alternates.write_text("/tmp/not-an-authority\n")
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "alternates" in completed.stderr
    alternates.unlink()

    promisor = repo / ".git/objects/pack/hostile.promisor"
    promisor.write_text("")
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "promisor_catalogue" in completed.stderr
    promisor.unlink()

    sparse = repo / ".git/info/sparse-checkout"
    sparse.write_text("/*\n")
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "sparse_checkout" in completed.stderr
    sparse.unlink()

    blob = git("rev-parse", "HEAD:tracked.txt").stdout.strip()
    replacement_payload = tmp_path / "replacement-payload"
    replacement_payload.write_text("replacement blob\n")
    replacement = git("hash-object", "-w", str(replacement_payload)).stdout.strip()
    git("replace", blob, replacement)
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "replace_refs" in completed.stderr
    git("replace", "-d", blob)

    git_directory = repo / ".git"
    parked_git_directory = tmp_path / "parked.git"
    git_directory.rename(parked_git_directory)
    git_directory.write_text(f"gitdir: {parked_git_directory}\n")
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    assert "unexpected_layout" in completed.stderr
    git_directory.unlink()
    parked_git_directory.rename(git_directory)

    blob_path = repo / ".git/objects" / blob[:2] / blob[2:]
    assert blob_path.is_file()
    blob_path.unlink()
    completed = _run_worker_repo_verify(repo, pin, origin)
    assert completed.returncode != 0
    # The bound missing-object inventory normally catches this first. Git may
    # instead expose it through tracked-state or fsck depending on cache state.
    assert re.fullmatch(
        r"REPO_VERIFY_BAD \S+ (missing_count|tracked_state|object_closure)\n",
        completed.stderr,
    )


def test_exact_worker_repo_verifier_accepts_only_exact_promisor_prestate(
    tmp_path: Path,
) -> None:
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Gate D Test",
        "GIT_AUTHOR_EMAIL": "gate-d@example.invalid",
        "GIT_COMMITTER_NAME": "Gate D Test",
        "GIT_COMMITTER_EMAIL": "gate-d@example.invalid",
    }

    seed = tmp_path / "seed"
    seed.mkdir()

    def seed_git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/usr/bin/git", "-C", str(seed), *arguments],
            check=True,
            capture_output=True,
            env=environment,
            text=True,
        )

    seed_git("init", "-q")
    (seed / "tracked.txt").write_text("historical blob that remains promised\n")
    seed_git("add", "tracked.txt")
    seed_git("commit", "-q", "-m", "historical")
    (seed / "tracked.txt").write_text("current checked-out blob\n")
    seed_git("commit", "-q", "-am", "current")

    bare = tmp_path / "origin.git"
    subprocess.run(
        ["/usr/bin/git", "clone", "-q", "--bare", str(seed), str(bare)],
        check=True,
        env=environment,
    )
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(bare),
            "config",
            "uploadpack.allowFilter",
            "true",
        ],
        check=True,
        env=environment,
    )
    origin = bare.as_uri()
    repo = tmp_path / "promisor-repo"
    subprocess.run(
        [
            "/usr/bin/git",
            "clone",
            "-q",
            "--filter=blob:none",
            "--no-checkout",
            origin,
            str(repo),
        ],
        check=True,
        env=environment,
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "checkout", "-q", "--detach", "HEAD"],
        check=True,
        env=environment,
    )
    pin = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    ).stdout.strip()
    passive_environment = {**environment, "GIT_NO_LAZY_FETCH": "1"}
    missing_output = subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(repo),
            "rev-list",
            "--objects",
            "--missing=print",
            pin,
        ],
        check=True,
        capture_output=True,
        env=passive_environment,
        text=True,
    ).stdout
    missing = sorted(
        line.split()[0][1:]
        for line in missing_output.splitlines()
        if line.startswith("?")
    )
    assert missing
    missing_sha = sha256(("\n".join(missing) + "\n").encode()).hexdigest()
    object_dir = repo / ".git/objects"
    promisor_paths = sorted(object_dir.rglob("*.promisor"))
    assert promisor_paths
    for path in promisor_paths:
        path.write_bytes(b"")
    promisor_records = sorted(
        (str(path.relative_to(object_dir)), path.stat().st_size)
        for path in promisor_paths
    )
    catalogue = b"".join(
        relative.encode("ascii") + b"\0" + str(size).encode("ascii") + b"\0"
        for relative, size in promisor_records
    )
    promisor_sha = sha256(catalogue).hexdigest()

    strict = _run_worker_repo_verify(repo, pin, origin)
    assert strict.returncode != 0
    assert "config_not_allowlisted" in strict.stderr

    accepted = _run_worker_repo_verify(
        repo,
        pin,
        origin,
        expected_layout="standalone_promisor",
        expected_missing_count=len(missing),
        expected_missing_sha=missing_sha,
        expected_promisor_count=len(promisor_records),
        expected_promisor_sha=promisor_sha,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.startswith("SYNC_OK ")

    promisor_paths[0].write_bytes(b"not-zero")
    nonzero = _run_worker_repo_verify(
        repo,
        pin,
        origin,
        expected_layout="standalone_promisor",
        expected_missing_count=len(missing),
        expected_missing_sha=missing_sha,
        expected_promisor_count=len(promisor_records),
        expected_promisor_sha=promisor_sha,
    )
    assert nonzero.returncode != 0
    assert "promisor_catalogue" in nonzero.stderr
    promisor_paths[0].write_bytes(b"")

    invalid_path = object_dir / "invalid.promisor"
    invalid_path.write_bytes(b"")
    invalid = _run_worker_repo_verify(
        repo,
        pin,
        origin,
        expected_layout="standalone_promisor",
        expected_missing_count=len(missing),
        expected_missing_sha=missing_sha,
        expected_promisor_count=len(promisor_records),
        expected_promisor_sha=promisor_sha,
    )
    assert invalid.returncode != 0
    assert "promisor_catalogue" in invalid.stderr
    invalid_path.unlink()

    hardlink = object_dir / "pack" / ("pack-" + "f" * 40 + ".promisor")
    os.link(promisor_paths[0], hardlink)
    linked = _run_worker_repo_verify(
        repo,
        pin,
        origin,
        expected_layout="standalone_promisor",
        expected_missing_count=len(missing),
        expected_missing_sha=missing_sha,
        expected_promisor_count=len(promisor_records),
        expected_promisor_sha=promisor_sha,
    )
    assert linked.returncode != 0
    assert "promisor_catalogue" in linked.stderr
    hardlink.unlink()

    tampered_count = _run_worker_repo_verify(
        repo,
        pin,
        origin,
        expected_layout="standalone_promisor",
        expected_missing_count=len(missing),
        expected_missing_sha=missing_sha,
        expected_promisor_count=len(promisor_records) + 1,
        expected_promisor_sha=promisor_sha,
    )
    assert tampered_count.returncode != 0
    assert "promisor_count" in tampered_count.stderr

    tampered_sha = _run_worker_repo_verify(
        repo,
        pin,
        origin,
        expected_layout="standalone_promisor",
        expected_missing_count=len(missing),
        expected_missing_sha=missing_sha,
        expected_promisor_count=len(promisor_records),
        expected_promisor_sha="0" * 64,
    )
    assert tampered_sha.returncode != 0
    assert "promisor_sha" in tampered_sha.stderr

    tampered_missing = _run_worker_repo_verify(
        repo,
        pin,
        origin,
        expected_layout="standalone_promisor",
        expected_missing_count=len(missing),
        expected_missing_sha="0" * 64,
        expected_promisor_count=len(promisor_records),
        expected_promisor_sha=promisor_sha,
    )
    assert tampered_missing.returncode != 0
    assert "missing_sha" in tampered_missing.stderr


def test_exact_worker_repo_verifier_accepts_only_bound_linked_worktree(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    linked = tmp_path / "linked"
    primary.mkdir()
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Gate D Test",
        "GIT_AUTHOR_EMAIL": "gate-d@example.invalid",
        "GIT_COMMITTER_NAME": "Gate D Test",
        "GIT_COMMITTER_EMAIL": "gate-d@example.invalid",
    }

    def git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/usr/bin/git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
            env=environment,
            text=True,
        )

    git(primary, "init", "-q")
    git(primary, "switch", "-q", "-c", "main")
    origin = "git@example.invalid:owner/repo.git"
    git(primary, "remote", "add", "origin", origin)
    (primary / "tracked.txt").write_text("linked authority\n")
    git(primary, "add", "tracked.txt")
    git(primary, "commit", "-q", "-m", "seed")
    git(
        primary,
        "worktree",
        "add",
        "-q",
        "-b",
        "rewrite/topology-first-decode",
        str(linked),
        "HEAD",
    )
    pin = git(linked, "rev-parse", "HEAD").stdout.strip()
    git(primary, "config", "branch.main.vscode-merge-base", pin)
    git(
        primary,
        "config",
        "branch.rewrite/topology-first-decode.vscode-merge-base",
        pin,
    )
    common = Path(
        git(
            linked, "rev-parse", "--path-format=absolute", "--git-common-dir"
        ).stdout.strip()
    )
    admin = Path(
        git(linked, "rev-parse", "--path-format=absolute", "--git-dir").stdout.strip()
    )
    for path in (
        linked / ".git",
        admin / "gitdir",
        admin / "commondir",
        admin / "HEAD",
        admin / "index",
        common / "config",
    ):
        path.chmod(0o664)
    admin.chmod(0o775)
    common.chmod(0o755)
    (common / "objects").chmod(0o755)

    completed = _run_worker_repo_verify(
        linked,
        pin,
        origin,
        linked_common=common,
        linked_git_dir=admin,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("SYNC_OK ")

    completed = _run_worker_repo_verify(
        linked,
        pin,
        origin,
        linked_common=common,
        linked_git_dir=admin,
        expected_layout="standalone",
    )
    assert completed.returncode != 0
    assert "unexpected_layout" in completed.stderr

    pointer = linked / ".git"
    pointer.chmod(0o600)
    completed = _run_worker_repo_verify(
        linked, pin, origin, linked_common=common, linked_git_dir=admin
    )
    assert completed.returncode != 0
    assert "linked_pointer_boundary" in completed.stderr
    pointer.chmod(0o664)

    pointer_bytes = pointer.read_bytes()
    pointer.write_text("gitdir: ../hostile-admin\n")
    completed = _run_worker_repo_verify(
        linked, pin, origin, linked_common=common, linked_git_dir=admin
    )
    assert completed.returncode != 0
    assert "linked_pointer_bytes" in completed.stderr
    pointer.write_bytes(pointer_bytes)

    backpointer = admin / "gitdir"
    backpointer_bytes = backpointer.read_bytes()
    backpointer.write_text("/tmp/hostile/.git\n")
    completed = _run_worker_repo_verify(
        linked, pin, origin, linked_common=common, linked_git_dir=admin
    )
    assert completed.returncode != 0
    assert "linked_backpointer_bytes" in completed.stderr
    backpointer.write_bytes(backpointer_bytes)

    try:
        os.setxattr(pointer, b"user.gate_d_test", b"1", follow_symlinks=False)
    except OSError:
        pass
    else:
        completed = _run_worker_repo_verify(
            linked, pin, origin, linked_common=common, linked_git_dir=admin
        )
        assert completed.returncode != 0
        assert "linked_pointer_boundary_xattr" in completed.stderr
        os.removexattr(pointer, b"user.gate_d_test", follow_symlinks=False)

    sparse = admin / "info/sparse-checkout"
    sparse.parent.mkdir(exist_ok=True)
    sparse.write_text("/*\n")
    completed = _run_worker_repo_verify(
        linked, pin, origin, linked_common=common, linked_git_dir=admin
    )
    assert completed.returncode != 0
    assert "sparse_checkout" in completed.stderr
    sparse.unlink()

    git(primary, "config", "core.hooksPath", "/tmp/hostile-hooks")
    completed = _run_worker_repo_verify(
        linked, pin, origin, linked_common=common, linked_git_dir=admin
    )
    assert completed.returncode != 0
    assert "config_not_allowlisted" in completed.stderr


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
    monkeypatch.setattr(
        PUBLISHER_MODULE, "_verify_dependency_record_live", lambda record: None
    )
    monkeypatch.setattr(
        PUBLISHER_MODULE,
        "_verify_accelerator_device_mapping_record_live",
        lambda record, **kwargs: None,
    )
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
        "accelerator_device_observation_scope": (
            PUBLISHER_MODULE._ACCELERATOR_DEVICE_OBSERVATION_SCOPE
        ),
        "accelerator_device_nodes_observed_mapped": [_accelerator_device_record()],
        "artifact_kind": "gate_d_compensated_pp16_compiler_dependencies",
        "code_hash": "1" * 40,
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
            "accelerator_device_node_count": 1,
            "accelerator_device_nodes_sha256": (
                DRIVER_MODULE._accelerator_device_nodes_sha256(
                    dependencies["accelerator_device_nodes_observed_mapped"]
                )
            ),
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


def test_compiler_runtime_rebinding_refuses_before_remote_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    dependencies_path = run / "dependencies.json"
    dependencies = json.loads(dependencies_path.read_bytes())
    dependencies["python_runtime"]["python_runtime_root"] = (
        "/home/gianl/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu"
    )
    dependencies_raw = PUBLISHER_MODULE._canonical(dependencies)
    dependencies_path.write_bytes(dependencies_raw)
    runner_path = run / "runner.json"
    runner = json.loads(runner_path.read_bytes())
    runner["compiler_dependency_manifest"]["byte_count"] = len(dependencies_raw)
    runner["compiler_dependency_manifest"]["sha256"] = sha256(
        dependencies_raw
    ).hexdigest()
    runner_path.write_bytes(PUBLISHER_MODULE._canonical(runner))
    bucket = _FakeBucket()
    with pytest.raises(RuntimeError, match="compiler dependency manifest drifted"):
        PUBLISHER_MODULE.publish_success(
            run,
            remote,
            code_pin="1" * 40,
            elapsed=7,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=bucket,
        )
    assert not bucket.mutations


def test_compiler_site_rebinding_refuses_before_remote_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    dependencies_path = run / "dependencies.json"
    dependencies = json.loads(dependencies_path.read_bytes())
    dependencies["dependency_sites"]["libtpu"]["root"] = (
        "/home/gianl/vllm-env/lib/python3.12/site-packages"
    )
    dependencies_raw = PUBLISHER_MODULE._canonical(dependencies)
    dependencies_path.write_bytes(dependencies_raw)
    runner_path = run / "runner.json"
    runner = json.loads(runner_path.read_bytes())
    runner["compiler_dependency_manifest"]["byte_count"] = len(dependencies_raw)
    runner["compiler_dependency_manifest"]["sha256"] = sha256(
        dependencies_raw
    ).hexdigest()
    runner_path.write_bytes(PUBLISHER_MODULE._canonical(runner))
    bucket = _FakeBucket()
    with pytest.raises(RuntimeError, match="compiler dependency manifest drifted"):
        PUBLISHER_MODULE.publish_success(
            run,
            remote,
            code_pin="1" * 40,
            elapsed=7,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=bucket,
        )
    assert not bucket.mutations


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("accelerator_device_node_count", 0),
        ("accelerator_device_node_count", True),
        ("accelerator_device_nodes_sha256", "0" * 64),
        ("accelerator_device_nodes_sha256", 1),
    ),
)
def test_accelerator_device_manifest_rebinding_refuses_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    runner_path = run / "runner.json"
    runner = json.loads(runner_path.read_bytes())
    runner["compiler_dependency_manifest"][field] = value
    runner_path.write_bytes(PUBLISHER_MODULE._canonical(runner))
    bucket = _FakeBucket()
    with pytest.raises(RuntimeError, match="compiler dependency manifest drifted"):
        PUBLISHER_MODULE.publish_success(
            run,
            remote,
            code_pin="1" * 40,
            elapsed=7,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=bucket,
        )
    assert not bucket.mutations


def test_accelerator_device_scope_or_extra_manifest_key_refuses_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, remote = _fake_success_run(tmp_path, monkeypatch)
    dependencies_path = run / "dependencies.json"
    dependencies = json.loads(dependencies_path.read_bytes())
    dependencies["accelerator_device_observation_scope"] = "overclaimed VMA ledger"
    dependencies["unexpected"] = True
    dependencies_raw = PUBLISHER_MODULE._canonical(dependencies)
    dependencies_path.write_bytes(dependencies_raw)
    runner_path = run / "runner.json"
    runner = json.loads(runner_path.read_bytes())
    runner["compiler_dependency_manifest"]["byte_count"] = len(dependencies_raw)
    runner["compiler_dependency_manifest"]["sha256"] = sha256(
        dependencies_raw
    ).hexdigest()
    runner_path.write_bytes(PUBLISHER_MODULE._canonical(runner))
    bucket = _FakeBucket()
    with pytest.raises(RuntimeError, match="compiler dependency manifest drifted"):
        PUBLISHER_MODULE.publish_success(
            run,
            remote,
            code_pin="1" * 40,
            elapsed=7,
            publication_runtime_raw=TEST_PUBLICATION_RUNTIME_RAW,
            storage_bucket=bucket,
        )
    assert not bucket.mutations


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
