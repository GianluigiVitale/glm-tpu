from __future__ import annotations

import ast
from hashlib import sha256
import importlib.util
from io import BytesIO
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "scripts/greenfield/probe_original_db518_prompt_key_chunk0.py"
PUBLISHER = REPO / "scripts/greenfield/publish_gate_d_original_db518_prompt_key_chunk0.py"
WRAPPER = REPO / "scripts/greenfield/run_original_db518_prompt_key_chunk0.sh"
LAUNCHER = REPO / "scripts/greenfield/launch_gate_d_original_db518_prompt_key_chunk0.py"
INSTALLER = REPO / "scripts/greenfield/install_gate_d_original_db518_prompt_key_runtime.py"
CONTRACT = REPO / "glm_tpu/greenfield/validation/original_db518_prompt_key.py"
VERIFIER = REPO / "scripts/greenfield/verify_gate_d_original_db518_same_region_git_mirror.py"
CERTIFICATE = REPO / "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-source.json"
STAGING = Path(
    "/home/gianl/gate-d-runs/gate-d-original-db518-prompt-key-install-v1-staging"
)


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
    payloads = installer.PAYLOADS
    assert launcher.WRAPPER_SHA256 == _digest(WRAPPER)
    assert launcher.HLO_CONTRACT_SHA256 == _digest(CONTRACT)
    assert launcher.PROBE_SHA256 == _digest(PROBE)
    assert launcher.PUBLISHER_SHA256 == _digest(PUBLISHER)
    assert launcher.MIRROR_VERIFIER_SHA256 == _digest(VERIFIER)
    assert payloads == {
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


def test_mirror_verifier_exact_authority_membership():
    verifier = _load(VERIFIER)
    assert verifier.BOUND_PATHS == (
        "docs/artifacts/gate-d-chunk0-v7-original-db518-boundary-diagnosis.json",
        "docs/artifacts/gate-d-original-db518-prompt-key-chunk0-source.json",
        "glm_tpu/greenfield/benchmarking/numpy_safetensors.py",
        "glm_tpu/greenfield/benchmarking/sealed_runtime.py",
        "glm_tpu/greenfield/kernels/reference/dsa.py",
        "glm_tpu/greenfield/kernels/reference/dsa_association.py",
        "glm_tpu/greenfield/kernels/reference/prefill_index.py",
        "glm_tpu/greenfield/validation/original_db518_prompt_key.py",
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
    )


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


def test_probe_treats_nonexact_as_completed_discriminator():
    source = PROBE.read_text()
    assert '"COMPLETED"' in source
    assert '"original_db518_chunk0_exact"' in source
    assert "return 0\n" in source
    assert "return 0 if exact else 1" not in source


def test_certificate_and_staging_are_exact():
    certificate = json.loads(CERTIFICATE.read_text())
    assert certificate["review_status"] == "PENDING_BATCHED_SOL_REVIEW"
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
        candidate_chunk_bits=candidate,
        embedding_rows=rows,
        positions=positions,
    )
    payloads = publisher._parse_npz(Parent, stream.getvalue())
    assert set(payloads) == set(publisher.EXPECTED_OUTPUT_LAYOUT)
    assert publisher._mismatch_counts(payloads["candidate_chunk_bits"],
                                      payloads["accepted_chunk_bits"]) == (0,
                                                                           0)

    bad = BytesIO()
    np.savez(
        bad,
        accepted_chunk_bits=accepted,
        candidate_chunk_bits=candidate,
        embedding_rows=rows,
    )
    import pytest

    with pytest.raises(RuntimeError, match="inventory drifted"):
        publisher._parse_npz(Parent, bad.getvalue())


def _npy_header(raw: bytes) -> tuple[tuple[int, ...], str]:
    value = ast.literal_eval(raw.decode("latin1").strip())
    assert value["fortran_order"] is False
    return value["shape"], value["descr"]
