from __future__ import annotations

import ast
from hashlib import sha256
import importlib
import importlib.util
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "scripts/greenfield/probe_original_db518_prompt_key_chunk0.py"
PUBLISHER = REPO / "scripts/greenfield/publish_gate_d_original_db518_prompt_key_chunk0.py"
WRAPPER = REPO / "scripts/greenfield/run_original_db518_prompt_key_chunk0.sh"
LAUNCHER = REPO / "scripts/greenfield/launch_gate_d_original_db518_prompt_key_chunk0.py"
INSTALLER = REPO / "scripts/greenfield/install_gate_d_original_db518_prompt_key_runtime.py"
CONTRACT = REPO / "glm_tpu/greenfield/validation/original_db518_prompt_key.py"
PARSER_CONTRACT = REPO / "glm_tpu/greenfield/validation/chunk0_embedding_hlo.py"
BOUNDARY_CONTRACT = (
    REPO / "glm_tpu/greenfield/validation/original_db518_normalized_boundary_hlo.py"
)
VERIFIER = REPO / "scripts/greenfield/verify_gate_d_original_db518_same_region_git_mirror.py"
PUBLISHER_PARENT = REPO / "scripts/greenfield/publish_gate_d_layer1_prompt_chunk0_geometry.py"
V5_MIRROR_REPLAY = REPO / "docs/artifacts/gate-d-original-db518-v5-mirror-replay.json"
V6_EXACT_SUCCESS = REPO / "docs/artifacts/gate-d-original-db518-v6-exact-success.json"
V5_PIN = "ae79a3fdcf877a8123f34b08411e4f6d0a5584a7"
V5_VERIFIER_SHA256 = (
    "288bfa0b707b041467f4b0f0686cc3e29207c0c71ad4b992139e238d29a7c084")
CERTIFICATE = REPO / "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-v10-source.json"
STAGING = Path(
    "/home/gianl/gate-d-runs/gate-d-original-db518-prompt-key-install-v10-staging"
)
V9_RUN = Path(
    "/home/gianl/gate-d-runs/"
    "greenfield_original_db518_prompt_key_chunk0_20260903T233327590939916Z"
)
V9_PIN = "33c85baf508f5aa6d7e67050326205cebe99a84a"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("test_original_db518_module",
                                                  path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _constants(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in ast.parse(path.read_text()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return values


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_exact_cross_file_hash_chain_and_paths():
    launcher = _load(LAUNCHER)
    installer = _load(INSTALLER)
    probe = _load(PROBE)
    publisher = _load(PUBLISHER)
    verifier = _load(VERIFIER)
    payloads = installer.PAYLOADS
    assert launcher.WRAPPER_SHA256 == _digest(WRAPPER)
    assert launcher.PARSER_CONTRACT_SHA256 == _digest(PARSER_CONTRACT)
    assert launcher.BOUNDARY_CONTRACT_SHA256 == _digest(BOUNDARY_CONTRACT)
    assert launcher.HLO_CONTRACT_SHA256 == _digest(CONTRACT)
    assert launcher.PROBE_SHA256 == _digest(PROBE)
    assert launcher.PUBLISHER_SHA256 == _digest(PUBLISHER)
    assert launcher.MIRROR_VERIFIER_SHA256 == _digest(VERIFIER)
    assert publisher.MIRROR_VERIFIER_SHA256 == _digest(VERIFIER)
    assert payloads == {
        "chunk0_embedding_hlo.py": _digest(PARSER_CONTRACT),
        "original_db518_normalized_boundary_hlo.py": _digest(BOUNDARY_CONTRACT),
        "launch_gate_d_original_db518_prompt_key_chunk0.py": _digest(LAUNCHER),
        "original_db518_prompt_key.py": _digest(CONTRACT),
        "probe_original_db518_prompt_key_chunk0.py": _digest(PROBE),
        "publish_gate_d_original_db518_prompt_key_chunk0.py":
        _digest(PUBLISHER),
        "verify_gate_d_original_db518_same_region_git_mirror.py": _digest(VERIFIER),
    }
    wrapper = WRAPPER.read_text()
    assert f"readonly PROBE_SHA={_digest(PROBE)}" in wrapper
    assert f"readonly PUBLISHER_SHA={_digest(PUBLISHER)}" in wrapper
    assert f"readonly MIRROR_VERIFIER_SHA={_digest(VERIFIER)}" in wrapper
    assert str(launcher.INSTALL_PATH) in wrapper
    assert launcher.INSTALL_PATH == installer.LAUNCHER_TARGET
    assert launcher.CAPSULE_ROOT == installer.CAPSULE_TARGET
    assert probe.INSTALL_PATH == installer.CAPSULE_TARGET / PROBE.name
    assert publisher.INSTALL_PATH == installer.CAPSULE_TARGET / PUBLISHER.name
    assert publisher.BOUNDARY_CONTRACT_PATH == (
        installer.CAPSULE_TARGET / BOUNDARY_CONTRACT.name
    )
    assert publisher.PARSER_CONTRACT_PATH == (
        installer.CAPSULE_TARGET / PARSER_CONTRACT.name
    )
    assert publisher.CONTRACT_PATH == installer.CAPSULE_TARGET / CONTRACT.name
    assert verifier.INSTALL_PATH == installer.CAPSULE_TARGET / VERIFIER.name


def test_mirror_verifier_exact_authority_membership():
    verifier = _load(VERIFIER)
    assert verifier.BOUND_PATHS == (
        "docs/artifacts/gate-d-chunk0-v7-original-db518-boundary-diagnosis.json",
        "docs/artifacts/gate-d-original-db518-publisher-isolated-import-failure.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-source.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-v2-source.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-v3-source.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-v4-source.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-v5-source.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-v6-source.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-v7-source.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-v8-source.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-v9-source.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-v10-source.json",
        "docs/artifacts/gate-d-original-db518-v9-publisher-contract-failure.json",
        "docs/artifacts/gate-d-original-db518-v8-key-placement-failure.json",
        "docs/artifacts/gate-d-original-db518-v7-tpu-hlo-placement-failure.json",
        "docs/artifacts/gate-d-original-db518-v5-mirror-authority-failure.json",
        "docs/artifacts/gate-d-original-db518-v5-mirror-replay.json",
        "docs/artifacts/gate-d-original-db518-v6-exact-success.json",
        "docs/artifacts/gate-d-original-db518-v2-probe-install-path-failure.json",
        "docs/artifacts/gate-d-original-db518-v3-stablehlo-scatter-count-failure.json",
        "docs/artifacts/gate-d-original-db518-v4-helper-shape-parser-failure.json",
        "glm_tpu/greenfield/benchmarking/numpy_safetensors.py",
        "glm_tpu/greenfield/benchmarking/sealed_runtime.py",
        "glm_tpu/greenfield/kernels/reference/dsa.py",
        "glm_tpu/greenfield/kernels/reference/dsa_association.py",
        "glm_tpu/greenfield/kernels/reference/prefill_index.py",
        "glm_tpu/greenfield/validation/original_db518_prompt_key.py",
        "glm_tpu/greenfield/validation/original_db518_normalized_boundary_hlo.py",
        "glm_tpu/greenfield/validation/chunk0_embedding_hlo.py",
        "glm_tpu/greenfield/validation/prompt_index_cache.py",
        "scripts/greenfield/install_gate_d_original_db518_prompt_key_runtime.py",
        "scripts/greenfield/launch_gate_d_original_db518_prompt_key_chunk0.py",
        "scripts/greenfield/probe_original_db518_prompt_key_chunk0.py",
        "scripts/greenfield/publish_gate_d_layer1_prompt_chunk0_geometry.py",
        "scripts/greenfield/publish_gate_d_original_db518_prompt_key_chunk0.py",
        "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py",
        "scripts/greenfield/run_original_db518_prompt_key_chunk0.sh",
        "scripts/greenfield/verify_gate_d_same_region_git_mirror.py",
        "scripts/greenfield/verify_gate_d_original_db518_same_region_git_mirror.py",
        "tests/greenfield/validation/test_original_db518_prompt_key.py",
        "tests/greenfield/validation/test_original_db518_prompt_key_chain.py",
        "tests/greenfield/validation/test_original_db518_prompt_key_probe.py",
        "tests/greenfield/kernels/test_original_db518_normalized_boundary.py",
    )


def test_publisher_rebinds_complete_mirror_authority_tuple():
    import pytest

    publisher = _load(PUBLISHER)
    current_pin = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    parent = publisher._load_parent(current_pin)
    raw = V5_MIRROR_REPLAY.read_bytes()

    stale = parent._load_base(current_pin)
    with pytest.raises(RuntimeError, match="authority drifted"):
        stale._validate_mirror_replay(raw, V5_PIN)

    base = publisher._load_base(parent, current_pin)
    assert base.MIRROR_VERIFIER_SHA256 == publisher.MIRROR_VERIFIER_SHA256
    with pytest.raises(RuntimeError, match="verifier binding drifted"):
        base._validate_mirror_replay(raw, V5_PIN)
    base.MIRROR_VERIFIER_SHA256 = V5_VERIFIER_SHA256
    base._validate_mirror_replay(raw, V5_PIN)
    expected = {
        "BRANCH": publisher.BRANCH,
        "ORIGIN": publisher.ORIGIN,
        "MIRROR_URI": publisher.MIRROR_URI,
        "MIRROR_VERIFIER_PATH": publisher.MIRROR_VERIFIER_PATH,
        "MIRROR_VERIFIER_SHA256": V5_VERIFIER_SHA256,
    }
    for name, value in expected.items():
        assert getattr(base, name) == value
        setattr(base, name, value + ".hostile")
        with pytest.raises(RuntimeError):
            base._validate_mirror_replay(raw, V5_PIN)
        setattr(base, name, value)
        base._validate_mirror_replay(raw, V5_PIN)


def test_wrapper_is_one_producer_only_and_binds_rehydrated_input():
    source = WRAPPER.read_text()
    assert source.count('"$SEALED_PYTHON" -I -S -B -u "$PROBE"') == 1
    assert "--accepted-cache-dir \"$ACCEPTED_CACHE_DIR\"" in source
    assert ("--accepted-cache-manifest-sha256 "
            '"$ACCEPTED_CACHE_MANIFEST_SHA"' in source)
    assert "36303f0638661b4a56d3c9a1d4023a9b39eb19dbfd6d48e0718c29a45c41c07a" in source
    assert "layer1_prompt_chunk0_geometry" not in source
    assert "decode_batch1" not in source
    assert "consumer" not in source.lower()
    assert "TPU_VISIBLE_DEVICES=0,1,2,3" in source
    assert "strict_census pre" in source and "strict_census post" in source
    assert source.count("vacancy_three_surfaces") == 2


def test_launcher_preflights_exact_publisher_before_tag_directory_creation():
    source = LAUNCHER.read_text()
    assert source.index("_preflight_publisher(pin)") < source.index(
        "run_fd = _create_retained_run_fd(tag)")
    assert source.index("_preflight_probe(pin)") < source.index(
        "run_fd = _create_retained_run_fd(tag)")
    launcher = _load(LAUNCHER)
    assert launcher.SEALED_PYTHON == Path(
        "/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12")
    assert "preflight" in source


def test_publisher_contract_loads_in_isolated_no_site_runtime():
    pin = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    prompt_contract = REPO / "glm_tpu/greenfield/validation/prompt_index_cache.py"
    program = (
        "import runpy\n"
        "from pathlib import Path\n"
        f"p=runpy.run_path({str(PUBLISHER)!r})\n"
        f"contract=Path({str(CONTRACT)!r})\n"
        f"parser=Path({str(PARSER_CONTRACT)!r})\n"
        f"boundary=Path({str(BOUNDARY_CONTRACT)!r})\n"
        f"prompt=Path({str(prompt_contract)!r})\n"
        f"pin={pin!r}\n"
        "def git_bytes(*args):\n"
        "    if args == ('show', pin + ':' + p['CONTRACT_SOURCE_PATH']):\n"
        "        return contract.read_bytes()\n"
        "    if args == ('show', pin + ':' + p['PARSER_CONTRACT_SOURCE_PATH']):\n"
        "        return parser.read_bytes()\n"
        "    if args == ('show', pin + ':' + p['PROMPT_CONTRACT_SOURCE_PATH']):\n"
        "        return prompt.read_bytes()\n"
        "    if args == ('show', pin + ':' + p['BOUNDARY_CONTRACT_SOURCE_PATH']):\n"
        "        return boundary.read_bytes()\n"
        "    raise RuntimeError('unexpected isolated test Git query')\n"
        "loader=p['_load_contract']\n"
        "loader.__globals__['CONTRACT_PATH']=contract\n"
        "loader.__globals__['PARSER_CONTRACT_PATH']=parser\n"
        "loader.__globals__['BOUNDARY_CONTRACT_PATH']=boundary\n"
        "loader.__globals__['_git_bytes']=git_bytes\n"
        "c=loader(pin)\n"
        "print(c.ORIGINAL_DB518_CODE_HASH)\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", program],
        cwd="/",
        env={
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.stdout == "86243115452920fe4244bb77a9bbf4c44110aeab\n"
    assert result.stderr == ""


def test_probe_treats_nonexact_as_completed_discriminator():
    source = PROBE.read_text()
    assert '"COMPLETED"' in source
    assert '"normalized_boundary_key_control_exact"' in source
    assert "return 0\n" in source
    assert "return 0 if exact else 1" not in source


def test_certificate_and_staging_are_exact():
    certificate = json.loads(CERTIFICATE.read_text())
    assert certificate["review_status"] == "PENDING_BATCHED_SOL_REVIEW"
    historical = json.loads(V6_EXACT_SUCCESS.read_text())
    accepted_sha = historical["numerical_result"]["accepted_chunk_bits_sha256"]
    assert accepted_sha == historical["numerical_result"][
        "candidate_chunk_bits_sha256"
    ]
    assert certificate["historical_exact_producer"]["bit_sha256"] == accepted_sha
    assert _load(PROBE).EXPECTED_CHUNK_BITS_SHA256 == accepted_sha
    assert _load(PUBLISHER).EXPECTED_CHUNK_BITS_SHA256 == accepted_sha
    for record in certificate["source_files"]:
        assert _digest(REPO / record["path"]) == record["sha256"]
    installer = _load(INSTALLER)
    assert set(item.name
               for item in STAGING.iterdir()) == set(installer.SOURCE_NAMES)
    for name, expected in installer.PAYLOADS.items():
        assert _digest(STAGING / name) == expected
    provisioner = _load(
        REPO / "scripts/greenfield/provision_gate_d_python_runtime.py")
    assert provisioner._tree_sha256(
        STAGING) == certificate["runtime_staging"]["tree_sha256"]


def test_publisher_rederives_arrays_and_rejects_inventory_drift():
    publisher = _load(PUBLISHER)

    class Parent:

        @staticmethod
        def _parse_npy(raw: bytes):
            version = raw[6]
            width = 2 if version == 1 else 4
            start = 8 + width
            size = int.from_bytes(raw[8:start], "little")
            return (*_npy_header(raw[start:start + size]), raw[start + size:])

    accepted = np.zeros((2048, 128), dtype=np.uint16)
    candidate = accepted.copy()
    rows = np.zeros((2048, ), dtype=np.int32)
    positions = np.arange(2048, dtype=np.int32)
    stream = BytesIO()
    np.savez(
        stream,
        accepted_chunk_bits=accepted,
        boundary_key_bits=candidate,
        embedding_rows=rows,
        positions=positions,
    )
    payloads = publisher._parse_npz(Parent, stream.getvalue())
    assert set(payloads) == set(publisher.EXPECTED_OUTPUT_LAYOUT)
    assert publisher._mismatch_counts(payloads["boundary_key_bits"],
                                      payloads["accepted_chunk_bits"]) == (0,
                                                                           0)

    bad = BytesIO()
    np.savez(
        bad,
        accepted_chunk_bits=accepted,
        boundary_key_bits=candidate,
        embedding_rows=rows,
    )
    import pytest

    with pytest.raises(RuntimeError, match="inventory drifted"):
        publisher._parse_npz(Parent, bad.getvalue())


def test_publisher_accepts_exact_helper_boundaries_and_rejects_drift():
    import pytest

    publisher = _load(PUBLISHER)
    parent = _load(PUBLISHER_PARENT)
    decode = "\n".join((
        "HloModule decode",
        "ENTRY %main (bits: u8[128,6144], scale: f32[1,48]) -> bf16[128,6144] {",
        "  %bits = u8[128,6144]{1,0} parameter(0)",
        "  %scale = f32[1,48]{1,0} parameter(1)",
        "  ROOT %result = bf16[128,6144]{1,0} add(%bits, %scale)",
        "}",
    ))
    promote = "\n".join((
        "HloModule promote",
        "ENTRY %main (value: bf16[128,6144]) -> f32[128,6144] {",
        "  %value = bf16[128,6144]{1,0} parameter(0)",
        "  ROOT %result = f32[128,6144]{1,0} convert(%value)",
        "}",
    ))
    publisher._validate_helper_hlo(parent, decode, "", "decode")
    publisher._validate_helper_hlo(parent, promote, "", "promote")
    with pytest.raises(RuntimeError, match="decode helper boundary drifted"):
        publisher._validate_helper_hlo(
            parent, decode.replace("f32[1,48]", "f32[1,47]"), "", "decode")
    with pytest.raises(RuntimeError, match="dead input"):
        publisher._validate_helper_hlo(
            parent, decode.replace("add(%bits, %scale)", "copy(%bits)"), "",
            "decode")
    with pytest.raises(RuntimeError, match="forbidden communication"):
        publisher._validate_helper_hlo(parent, decode, "all-reduce(",
                                       "decode")


def test_contract_comparison_is_json_semantic_strict_and_mutation_sensitive():
    import pytest

    publisher = _load(PUBLISHER)
    tuple_record = {
        "passed": True,
        "shapes": (("bf16", (2048, 6144)),),
        "sources": (("%positions",), ("%completed",)),
    }
    list_record = json.loads(json.dumps(tuple_record))
    assert publisher._canonical_contract(tuple_record) == (
        publisher._canonical_contract(list_record))

    for hostile in (
        {**list_record, "passed": False},
        {**list_record, "shapes": [["bf16", [2047, 6144]]]},
        {**list_record, "sources": list(reversed(list_record["sources"]))},
    ):
        assert publisher._canonical_contract(hostile) != (
            publisher._canonical_contract(tuple_record))

    with pytest.raises(RuntimeError, match="not strict JSON"):
        publisher._canonical_contract({"bad": {1, 2}})
    with pytest.raises(RuntimeError, match="not strict JSON"):
        publisher._canonical_contract({"bad": math.nan})


def test_actual_v9_runner_and_hlo_replay_after_json_transport(tmp_path):
    import pytest

    assert V9_RUN.is_dir()
    publisher = _load(PUBLISHER)
    parent = _load(PUBLISHER_PARENT)
    contract = importlib.import_module(
        "glm_tpu.greenfield.validation.original_db518_normalized_boundary_hlo")

    class Base:

        @staticmethod
        def snapshot_member(run_fd: int, name: str, *, limit: int) -> bytes:
            descriptor = os.open(name, os.O_RDONLY | os.O_CLOEXEC,
                                 dir_fd=run_fd)
            try:
                raw = b""
                while block := os.read(descriptor, 1 << 20):
                    raw += block
                    if len(raw) > limit:
                        raise RuntimeError("test snapshot exceeds limit")
                return raw
            finally:
                os.close(descriptor)

    replay = tmp_path / "v9"
    replay.mkdir()
    shutil.copy2(V9_RUN / "runner.json", replay / "runner.json")
    os.chmod(replay / "runner.json", 0o600)
    shutil.copy2(V9_RUN / "boundary_arrays.npz", replay / "boundary_arrays.npz")
    shutil.copytree(V9_RUN / "hlo", replay / "hlo")

    def validate() -> bool:
        descriptor = os.open(replay, os.O_RDONLY | os.O_DIRECTORY)
        try:
            _, _, exact = publisher._validate_runner_and_outputs(
                parent, contract, Base, descriptor, V9_PIN, V9_RUN.name)
            return exact
        finally:
            os.close(descriptor)

    assert validate() is True
    pristine = json.loads((replay / "runner.json").read_text())
    mutations = []
    bad_shape = json.loads(json.dumps(pristine))
    bad_shape["hlo"]["normalization_contract"][
        "optimized_parameter_shapes"][0][0][1][0] = 36
    mutations.append(bad_shape)
    bad_order = json.loads(json.dumps(pristine))
    bad_order["hlo"]["key_control_contract"][
        "optimized_initial_sources"].reverse()
    mutations.append(bad_order)
    bad_boolean = json.loads(json.dumps(pristine))
    bad_boolean["hlo"]["normalization_contract"]["passed"] = False
    mutations.append(bad_boolean)
    bad_hash = json.loads(json.dumps(pristine))
    bad_hash["hlo"]["normalization_optimized_sha256"] = "0" * 64
    mutations.append(bad_hash)
    for hostile in mutations:
        (replay / "runner.json").write_text(
            json.dumps(hostile, indent=2, sort_keys=True) + "\n")
        with pytest.raises(RuntimeError):
            validate()


def _npy_header(raw: bytes) -> tuple[tuple[int, ...], str]:
    value = ast.literal_eval(raw.decode("latin1").strip())
    assert value["fortran_order"] is False
    return value["shape"], value["descr"]
