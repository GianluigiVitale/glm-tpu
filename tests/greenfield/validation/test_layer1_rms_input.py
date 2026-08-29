from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from glm_tpu.greenfield.benchmarking.pp16_dense_boundary import (
    derive_expected_dense_boundary_bits,
)
from glm_tpu.greenfield.kernels.stage_local import (
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
)
from glm_tpu.greenfield.validation.layer1_rms_input import (
    CAPTURE_KIND,
    COMPARISON_KIND,
    RECONSTRUCTION_SEMANTICS,
    Layer1RmsInputCaptureConfig,
    capture_accepted_layer1_rms_input,
    validate_layer1_rms_input_artifacts,
)

LEGACY_PIN = "8dc7d20fedca5a98c27bfd1774827305973fa4c1"
ORACLE_PIN = "b3c25df47ac98783912dc658878181ec0a8ae16d"
RUN_TAG = "greenfield_legacy_layer1_rms_input_test"
WIDTH = 6144


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=ml_dtypes.bfloat16)).view(
        np.uint16
    )


def _decode(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values).view(ml_dtypes.bfloat16).astype(np.float32)


def _write_vllm_repository(root: Path) -> tuple[Path, str, str, str]:
    repository = root / "vllm"
    ir = repository / "vllm/ir/ops/layernorm.py"
    executor = repository / "vllm/model_executor/layers/layernorm.py"
    ir.parent.mkdir(parents=True)
    executor.parent.mkdir(parents=True)
    ir.write_text(
        "def fused_add_rms_norm(x, residual):\n"
        "    x = x.to(float) + residual.to(float)\n"
        "    return x\n",
        encoding="utf-8",
    )
    executor.write_text(
        "class RMSNorm:\n    pass_weight_add = True\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.invalid",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.invalid",
        }
    )
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "fixture"],
        check=True,
        env=environment,
    )
    pin = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, pin, _file_sha256(ir), _file_sha256(executor)


def _write_db550(root: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    partials = np.zeros((4, 8, 1, WIDTH), dtype=np.uint16)
    dense_values = np.linspace(-0.25, 0.25, WIDTH, dtype=np.float32)
    partials[0, 0, 0] = _bits(dense_values)
    residual = _bits(np.linspace(0.5, -0.5, WIDTH, dtype=np.float32)).reshape(1, WIDTH)
    ids = tuple(
        int(item)
        for item in np.argsort(
            np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
        )
    )
    dense, carried = derive_expected_dense_boundary_bits(partials, residual, ids)
    path = root / "db550.npz"
    np.savez(
        path,
        dense_virtual_partials_bfloat16_bits=partials,
        post_attention_residual_bfloat16_bits=residual,
    )
    return path, dense.reshape(WIDTH), residual.reshape(WIDTH), carried.reshape(WIDTH)


def _write_straddler(root: Path) -> Path:
    path = root / "straddler.json"
    path.write_text(
        json.dumps(
            {
                "artifact_kind": (
                    "greenfield_pp16_feature2_layer1_straddler_classification"
                ),
                "classification": (
                    "BF16_BOUNDARY_INSUFFICIENT_FOR_FP32_CAUSAL_ADJUDICATION"
                ),
                "localized_boundary": {
                    "accepted_normalized_bits": 48423,
                    "hidden_index": 2795,
                    "observed_normalized_bits": 48422,
                },
                "status": "CLASSIFIED",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_observer(
    root: Path,
    hidden: np.ndarray,
    residual: np.ndarray,
    *,
    fused_override: np.ndarray | None = None,
) -> Path:
    source = root / "source/w0"
    source.mkdir(parents=True)
    fused = np.ascontiguousarray(_decode(hidden) + _decode(residual), dtype=np.float32)
    if fused_override is not None:
        fused = np.ascontiguousarray(fused_override, dtype=np.float32)
    path = source / ("internals.model_layers_1_input_layernorm.position8155.proc0.npz")
    np.savez(
        path,
        artifact_kind=np.asarray("glm52_legacy_layer1_rms_input_operands"),
        format_version=np.asarray(1, dtype=np.int64),
        capture_mode=np.asarray("layer1_rms_input"),
        process_index=np.asarray(0, dtype=np.int64),
        process_count=np.asarray(8, dtype=np.int64),
        layer_name=np.asarray("model.layers.1.input_layernorm"),
        position=np.asarray(8155, dtype=np.int32),
        source_row=np.asarray(0, dtype=np.int32),
        reconstruction_semantics=np.asarray(RECONSTRUCTION_SEMANTICS),
        run_tag=np.asarray(RUN_TAG),
        code_hash=np.asarray(LEGACY_PIN),
        oracle_pin=np.asarray(ORACLE_PIN),
        model_id=np.asarray("zai-org/GLM-5.2-FP8"),
        hidden_update=hidden,
        hidden_update__dtype=np.asarray("bfloat16"),
        carried_residual=residual,
        carried_residual__dtype=np.asarray("bfloat16"),
        fused_add_float32=fused,
        fused_add_float32__dtype=np.asarray("float32"),
    )
    return source.parent


def _fixture(
    root: Path, *, hidden_mutation: bool = False
) -> Layer1RmsInputCaptureConfig:
    repository, pin, ir_sha, executor_sha = _write_vllm_repository(root)
    db550, hidden, residual, _ = _write_db550(root)
    if hidden_mutation:
        hidden = hidden.copy()
        hidden[11] ^= np.uint16(1)
    source = _write_observer(root, hidden, residual)
    straddler = _write_straddler(root)
    return Layer1RmsInputCaptureConfig(
        source_dump_dir=source,
        output_dir=root / "capture",
        db550_boundary_path=db550,
        straddler_classification_path=straddler,
        vllm_repository=repository,
        expected_run_tag=RUN_TAG,
        expected_legacy_code_hash=LEGACY_PIN,
        expected_oracle_pin=ORACLE_PIN,
        expected_db550_sha256=_file_sha256(db550),
        expected_straddler_sha256=_file_sha256(straddler),
        expected_vllm_pin=pin,
        expected_vllm_ir_layernorm_sha256=ir_sha,
        expected_vllm_executor_layernorm_sha256=executor_sha,
    )


def test_capture_binds_exact_fp32_source_to_db550(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    result = capture_accepted_layer1_rms_input(config)
    assert result["capture"]["artifact_kind"] == CAPTURE_KIND
    assert result["capture"]["capture_process_indices"] == [0]
    assert result["capture"]["reconstruction_semantics"] == (RECONSTRUCTION_SEMANTICS)
    comparison = result["comparison"]
    assert comparison["artifact_kind"] == COMPARISON_KIND
    assert comparison["classification"] == (
        "ACCEPTED_FP32_SOURCE_BOUND_TO_DB550_OPERANDS"
    )
    assert comparison["operands_match_db550"] is True
    assert all(item["bytewise_exact"] for item in comparison["comparisons"].values())
    assert validate_layer1_rms_input_artifacts(config) == result


def test_capture_classifies_direct_operand_divergence(tmp_path: Path) -> None:
    config = _fixture(tmp_path, hidden_mutation=True)
    result = capture_accepted_layer1_rms_input(config)
    comparison = result["comparison"]
    assert comparison["classification"] == (
        "ACCEPTED_FP32_SOURCE_DIVERGES_FROM_DB550_DERIVATION"
    )
    assert comparison["operands_match_db550"] is False
    assert comparison["comparisons"]["hidden_update"]["bytewise_exact"] is False
    assert comparison["comparisons"]["fused_add_float32"]["bytewise_exact"] is False


def test_capture_refuses_false_host_fp32_reconstruction(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    path = next(config.source_dump_dir.rglob("*.npz"))
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    fused = arrays["fused_add_float32"].copy()
    fused.view(np.uint32)[19] ^= np.uint32(1)
    arrays["fused_add_float32"] = fused
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="host FP32 reconstruction drifted"):
        capture_accepted_layer1_rms_input(config)


def test_capture_binds_observer_bytes_across_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _fixture(tmp_path)
    observer = next(config.source_dump_dir.rglob("*.npz"))
    original_bytes = observer.read_bytes()
    with np.load(observer, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["hidden_update"] = arrays["hidden_update"].copy()
    arrays["hidden_update"].view(np.uint16)[23] ^= np.uint16(1)
    replacement = tmp_path / "replacement.npz"
    np.savez(replacement, **arrays)
    replacement_bytes = replacement.read_bytes()
    original_read_bytes = Path.read_bytes
    replaced = False

    def replace_after_read(path: Path) -> bytes:
        nonlocal replaced
        value = original_read_bytes(path)
        if path == observer and not replaced:
            path.write_bytes(replacement_bytes)
            replaced = True
        return value

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    result = capture_accepted_layer1_rms_input(config)
    assert replaced is True
    assert result["comparison"]["operands_match_db550"] is True
    assert (
        result["capture"]["source_file"]["sha256"] == sha256(original_bytes).hexdigest()
    )
    with pytest.raises(
        ValueError, match="host FP32 reconstruction drifted|sealed tensor bytes drifted"
    ):
        validate_layer1_rms_input_artifacts(config)


def test_capture_refuses_vllm_source_drift(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    path = config.vllm_repository / "vllm/ir/ops/layernorm.py"
    path.write_text(path.read_text() + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked files are dirty"):
        capture_accepted_layer1_rms_input(config)


def test_validator_refuses_sealed_tensor_mutation(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    capture_accepted_layer1_rms_input(config)
    tensor = config.output_dir / "layer1_rms_input.npz"
    with np.load(tensor, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["fused_add_float32"] = arrays["fused_add_float32"].copy()
    arrays["fused_add_float32"].view(np.uint32)[0] ^= np.uint32(1)
    np.savez(tensor, **arrays)
    with pytest.raises(ValueError, match="sealed tensor bytes drifted"):
        validate_layer1_rms_input_artifacts(config)


def test_validator_refuses_capture_identity_crosswire(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    capture_accepted_layer1_rms_input(config)
    capture_path = config.output_dir / "capture.json"
    capture = json.loads(capture_path.read_text())
    capture["legacy_code_hash"] = "0" * 40
    capture.pop("manifest_sha256")
    capture["manifest_sha256"] = sha256(
        json.dumps(
            capture,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    capture_path.write_text(json.dumps(capture) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="capture contract drifted"):
        validate_layer1_rms_input_artifacts(config)


def test_protected_wrapper_is_default_off_and_pins_sources() -> None:
    repo = Path(__file__).resolve().parents[3]
    launcher = (
        repo / "scripts/greenfield/run_capture_legacy_layer1_rms_input.sh"
    ).read_text()
    wrapper = (
        repo / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    ).read_text()
    assert "GLM_GREENFIELD_DSA_INTERNALS_MODE=layer1_rms_input" in launcher
    assert "GLM_GREENFIELD_DSA_INTERNALS_LAYER_ID=1" in launcher
    assert "GLM_GREENFIELD_SHORT_DSA_ORACLE_PROFILE=8k" in launcher
    for exact in (
        "LEGACY_PIN=8dc7d20fedca5a98c27bfd1774827305973fa4c1",
        "OBSERVER_COMMIT_DISTANCE=12",
        "VLLM_PIN=a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c",
        "VLLM_IR_LAYERNORM_SHA=d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910",
        "VLLM_EXECUTOR_LAYERNORM_SHA=53c6abdab25dc1675f26f4c8fc5ba2094f1fb4a106334e581f630436210d0c9b",
        "GOLDEN_MANIFEST_SHA=916d421a10de9495086c0ad52645c9937746c88bae87a5ca1a69483bfe60d45d",
        "GOLDEN_MANIFEST_BYTES=321146",
        "LAYER1_OBSERVER_TRACKED_FILE_COUNT=947",
        "REFERENCE_8K_DSA_ORACLE_ROOT=/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z",
        "REFERENCE_8K_DSA_ORACLE_MANIFEST_SHA=f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da",
        "REFERENCE_8K_DSA_ORACLE_SUCCESS_SHA=0b798974ae8a9f95c32d3aa2eff532213624f1e2ae7f1de809a161e18dbdf1b9",
        "DB550_BOUNDARY_SHA=f194d757d2f9ebe27430dfec8f828ca7588e433bddb7e8d99f9b917c5aac4298",
        "STRADDLER_CLASSIFICATION_SHA=eebe1c5d5ba475a5faf000243d881657754fc47ed2345fe2691d923c1d457b36",
    ):
        assert exact in wrapper
    assert (
        "/home/gianl/glm-run/greenfield_short_context_dsa_oracle_8k_recovery_"
        not in wrapper
    )
    assert "inspect_short_context_dsa_oracle" in wrapper
    assert re.search(
        r"if \[\[ \$INTERNAL_CAPTURE == 1 \|\| "
        r"\$PREFILL_PROJECTION_CAPTURE == 1 \|\|\n\s+"
        r"\$DECODE_PROJECTION_CAPTURE == 1 \|\| "
        r"\$MAIN_CACHE_CAPTURE == 1 \]\]; then\n\s+"
        r"\[\[ -r \$REFERENCE_8K_DSA_ORACLE_ROOT/SUCCESS",
        wrapper,
    )
    assert wrapper.index("inspect_short_context_dsa_oracle") < wrapper.index(
        "strict_census pre"
    )
    assert wrapper.index("layer-1 RMS-input source contract drifted") < wrapper.index(
        "strict_census pre"
    )
    for exact in (
        'git -C "$VLLM_ARCHIVE_REPO" archive --format=tar.gz',
        'git -C "$LEGACY_SOURCE_REPO" bundle create',
        '"$POD:$OBSERVER_BUNDLE_REMOTE"',
        'actual_bundle=$(sha256sum "$bundle"',
        'git -C "$dest" bundle verify "$bundle"',
        'git -C "$temp" bundle verify "$bundle"',
        "OBSERVER_SYNC_BAD $(hostname) clone",
        "OBSERVER_SYNC_BAD $(hostname) checkout",
        "OBSERVER_SYNC_BAD $(hostname) temp_identity",
        'git clone -q --no-checkout "$bundle" "$temp"',
        "golden.rank${idx}.json",
        "OBSERVER_TRANSPORT_CLEAN_OK $(hostname)",
        '2>"$RUN_DIR/observer_transport_vacancy_ssh.txt"',
        '2>"$RUN_DIR/sync_observer_ssh.txt"',
        '2>"$RUN_DIR/golden_sync_ssh.txt"',
        '2>"$RUN_DIR/vllm_vacancy_ssh.txt"',
        '2>"$RUN_DIR/sync_vllm_ssh.txt"',
        '"$POD:$VLLM_RUNTIME_ARCHIVE_REMOTE"',
        'actual_archive=$(sha256sum "$archive"',
        "INTERNAL_PYTHONPATH=$OBSERVER_RUNTIME_REPO:$VLLM_RUNTIME_ROOT",
        'grep -Eq "^/tmp/glm_vllm_[A-Za-z0-9_]+$"',
        '--vllm-repository "$VLLM_ARCHIVE_REPO"',
        "vllm_repository=Path(sys.argv[32])",
        '"${OBSERVER_BRANCH:-none}"',
    ):
        assert exact in wrapper
    assert "VLLM_SYNC_BAD $(hostname) missing_pin" not in wrapper
    assert "VLLM_SOURCE_REPO" not in wrapper
    assert wrapper.index("cleanup_vllm_runtime post") < wrapper.index(
        "strict_census post"
    )
    assert wrapper.index("strict_census post") < wrapper.index(
        "freezing fresh DSA-oracle evidence"
    )


def test_protected_wrapper_python_heredocs_compile() -> None:
    wrapper = Path(__file__).resolve().parents[3] / (
        "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    )
    programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", wrapper.read_text(), re.DOTALL)
    assert len(programs) == 10
    for index, program in enumerate(programs):
        compile(program, f"{wrapper}:heredoc-{index}", "exec")


def test_canonical_dsa_prerequisite_passes_full_inspection() -> None:
    repo = Path(__file__).resolve().parents[3]
    wrapper = repo / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", wrapper.read_text(), re.DOTALL)
    program = next(
        item for item in programs if "inspect_short_context_dsa_oracle" in item
    )
    oracle = Path(
        "/home/gianl/gcs-models/oracles/greenfield/glm52/short_context_dsa/8k/"
        "greenfield_short_context_dsa_oracle_8k_recovery_"
        "20260807T174904381704076Z/oracle"
    )
    if not oracle.is_dir():
        pytest.skip("canonical protected 8K DSA oracle is unavailable")
    environment = dict(os.environ)
    environment.update({"JAX_PLATFORMS": "cpu", "PYTHONPATH": str(repo)})
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(oracle),
            "f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_exact_observer_bundle_reconstructs_reviewed_git_tree(tmp_path: Path) -> None:
    repository = Path("/home/gianl/tpu-inference-greenfield-layer1-rms-input-observer")
    if not repository.is_dir():
        pytest.skip("exact accepted layer-1 observer source is unavailable")
    bundle = tmp_path / "observer.bundle"
    runtime = tmp_path / "runtime"
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "bundle",
            "create",
            str(bundle),
            "greenfield/legacy-layer1-rms-input-observer",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "bundle", "verify", str(bundle)], check=True
    )
    subprocess.run(
        ["git", "clone", "-q", "--no-checkout", str(bundle), str(runtime)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(runtime), "checkout", "-q", "--detach", LEGACY_PIN],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(runtime), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    distance = subprocess.run(
        [
            "git",
            "-C",
            str(runtime),
            "rev-list",
            "--count",
            f"{ORACLE_PIN}..{LEGACY_PIN}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked = subprocess.run(
        ["git", "-C", str(runtime), "ls-tree", "-r", "--name-only", LEGACY_PIN],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dirty = subprocess.run(
        ["git", "-C", str(runtime), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert head == LEGACY_PIN
    assert distance == "12"
    assert len(tracked) == 947
    assert dirty == ""


def test_exact_vllm_archive_reconstructs_runtime_tree(tmp_path: Path) -> None:
    repository = Path("/home/gianl/vllm-build-a30addc")
    if not repository.is_dir():
        pytest.skip("exact accepted vLLM source is unavailable")
    archive = tmp_path / "vllm.tar.gz"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "archive",
            "--format=tar.gz",
            "--output",
            str(archive),
            "a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c",
        ],
        check=True,
    )
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(runtime)], check=True)
    expected = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-tree",
            "-r",
            "--name-only",
            "a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    actual = [
        path for path in runtime.rglob("*") if path.is_file() or path.is_symlink()
    ]
    assert len(actual) == len(expected) == 5493
    assert _file_sha256(runtime / "vllm/ir/ops/layernorm.py") == (
        "d8e4380ca97d2c719836a7e15fb410a73d354b15d79fc54275b573bbb1c06910"
    )
    assert _file_sha256(runtime / "vllm/model_executor/layers/layernorm.py") == (
        "53c6abdab25dc1675f26f4c8fc5ba2094f1fb4a106334e581f630436210d0c9b"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(runtime)
    imported = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib,vllm; print(pathlib.Path(vllm.__file__).resolve())",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr
    assert str(runtime.resolve()) in imported.stdout


def _terminal_program() -> str:
    wrapper = Path(__file__).resolve().parents[3] / (
        "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"
    )
    programs = re.findall(r"<<'PY'\n(.*?)\nPY\n", wrapper.read_text(), re.DOTALL)
    return next(
        program
        for program in programs
        if "layer-1 RMS-input DSA event tensors drifted" in program
    )


def test_terminal_reauthenticates_real_source_contract(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    db550 = Path(
        "/home/gianl/gcs-models/results/"
        "greenfield_layer0_dense_partial_capture_20260813T200736889447458Z/"
        "dense_partial_capture.npz"
    )
    vllm = Path("/home/gianl/vllm-build-a30addc")
    straddler = (
        repo / "docs/artifacts/pp16-feature2-layer1-straddler-classification.json"
    )
    if not (db550.is_file() and vllm.is_dir() and straddler.is_file()):
        pytest.skip("protected layer-1 RMS-input sources are unavailable")
    with np.load(db550, allow_pickle=False) as payload:
        ids = tuple(
            int(item)
            for item in np.argsort(
                np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
            )
        )
        hidden, _ = derive_expected_dense_boundary_bits(
            payload["dense_virtual_partials_bfloat16_bits"],
            payload["post_attention_residual_bfloat16_bits"],
            ids,
        )
        residual = np.ascontiguousarray(
            payload["post_attention_residual_bfloat16_bits"]
        )
    source = _write_observer(tmp_path, hidden.reshape(WIDTH), residual.reshape(WIDTH))
    source.rename(tmp_path / "source_dumps")
    source = tmp_path / "source_dumps"
    output = tmp_path / "layer1_rms_input_capture"
    capture_accepted_layer1_rms_input(
        Layer1RmsInputCaptureConfig(
            source_dump_dir=source,
            output_dir=output,
            db550_boundary_path=db550,
            straddler_classification_path=straddler,
            vllm_repository=vllm,
            expected_run_tag=RUN_TAG,
            expected_legacy_code_hash=LEGACY_PIN,
            expected_oracle_pin=ORACLE_PIN,
        )
    )
    (tmp_path / "oracle").mkdir()
    (tmp_path / "oracle/manifest.json").write_text(
        json.dumps({"artifact_kind": "oracle", "manifest_sha256": "5" * 64})
    )
    (tmp_path / "evidence_sha256.json").write_text("{}\n")
    (tmp_path / "remote_objects.json").write_text("{}\n")
    (tmp_path / "dsa_exact_comparison.json").write_text('{"exact": true}\n')
    vllm_pin = "a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c"
    archive = tmp_path / f"vllm_{vllm_pin}.tar.gz"
    archive.write_bytes(b"unit-test exact pin-derived archive")
    archive_sha = _file_sha256(archive)
    (tmp_path / "vllm_archive_identity.txt").write_text(
        f"pin={vllm_pin}\n"
        f"archive_sha256={archive_sha}\n"
        "tracked_entries=5493\n"
        "source_repository=/home/gianl/vllm-build-a30addc\n"
    )
    hosts = [f"worker-{index}" for index in range(8)]
    (tmp_path / "vllm_archive_copy.txt").write_text(
        "".join(f"VLLM_ARCHIVE_COPY_OK {index}\n" for index in range(8))
    )
    (tmp_path / "vllm_vacancy.txt").write_text(
        "".join(f"VLLM_VACANT_OK {host}\n" for host in hosts)
    )
    (tmp_path / "sync_vllm.txt").write_text(
        "".join(
            f"VLLM_SYNC_OK {host} archive_sha256={archive_sha} tracked_entries=5493\n"
            for host in hosts
        )
    )
    (tmp_path / "vllm_cleanup_post.txt").write_text(
        "".join(f"VLLM_CLEAN_OK {host}\n" for host in hosts)
    )
    observer_bundle = tmp_path / f"observer_{LEGACY_PIN}.bundle"
    observer_bundle.write_bytes(b"unit-test exact reviewed observer bundle")
    observer_bundle_sha = _file_sha256(observer_bundle)
    observer_source = "/home/gianl/tpu-inference-greenfield-layer1-rms-input-observer"
    observer_branch = "greenfield/legacy-layer1-rms-input-observer"
    (tmp_path / "observer_bundle_identity.txt").write_text(
        f"pin={LEGACY_PIN}\n"
        f"oracle_pin={ORACLE_PIN}\n"
        f"branch={observer_branch}\n"
        f"bundle_sha256={observer_bundle_sha}\n"
        "tracked_entries=947\n"
        f"source_repository={observer_source}\n"
    )
    (tmp_path / "observer_bundle_copy.txt").write_text(
        "".join(f"OBSERVER_BUNDLE_COPY_OK {index}\n" for index in range(8))
    )
    (tmp_path / "observer_transport_vacancy.txt").write_text(
        "".join(f"OBSERVER_TRANSPORT_VACANT_OK {host}\n" for host in hosts)
    )
    (tmp_path / "observer_transport_cleanup_post_sync.txt").write_text(
        "".join(f"OBSERVER_TRANSPORT_CLEAN_OK {host}\n" for host in hosts)
    )
    (tmp_path / "sync_observer.txt").write_text(
        "".join(
            f"OBSERVER_SYNC_OK {host} bundle_sha256={observer_bundle_sha} "
            "tracked_entries=947 "
            f"disposition={'existing' if index == 0 else 'installed'}\n"
            for index, host in enumerate(hosts)
        )
    )
    golden_sha = "916d421a10de9495086c0ad52645c9937746c88bae87a5ca1a69483bfe60d45d"
    (tmp_path / "golden_sync.txt").write_text(
        "".join(
            f"GOLDEN_SYNC_OK {host} sha256={golden_sha} bytes=321146 "
            "disposition=installed\n"
            for host in hosts
        )
    )
    ssh_status_names = {
        "golden_sync_ssh.txt",
        "observer_transport_cleanup_post_sync_ssh.txt",
        "observer_transport_vacancy_ssh.txt",
        "sync_observer_ssh.txt",
        "sync_vllm_ssh.txt",
        "vllm_cleanup_post_ssh.txt",
        "vllm_vacancy_ssh.txt",
    }
    real_gcloud_status = "".join(
        f"SSH: Attempting to connect to worker {index}...\n" for index in range(8)
    )
    for name in ssh_status_names:
        (tmp_path / name).write_text(real_gcloud_status)
    command = [
        sys.executable,
        "-c",
        _terminal_program(),
        str(tmp_path),
        "gs://driftbench-dsv4-uc/unit",
        "7" * 40,
        LEGACY_PIN,
        "1",
        "2",
        "294",
        "1",
        "1",
        "0",
        "0",
        "0",
        "layer1_rms_input",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        "0",
        RUN_TAG,
        str(tmp_path / "unused_dense_partial"),
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        ORACLE_PIN,
        vllm_pin,
        archive_sha,
        "5493",
        "/home/gianl/vllm-build-a30addc",
        observer_bundle_sha,
        "947",
        golden_sha,
        "321146",
        observer_source,
        observer_branch,
    ]
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        cwd=repo,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    original_observer_sync = (tmp_path / "sync_observer.txt").read_text()
    (tmp_path / "sync_observer.txt").write_text(
        real_gcloud_status + original_observer_sync
    )
    refused = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        cwd=repo,
    )
    assert refused.returncode != 0
    assert "accepted observer fleet receipt drifted" in refused.stderr
    (tmp_path / "sync_observer.txt").write_text(original_observer_sync)
    original_vllm_sync = (tmp_path / "sync_vllm.txt").read_text()
    (tmp_path / "sync_vllm.txt").write_text(
        original_vllm_sync.replace(archive_sha, "0" * 64, 1)
    )
    refused = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        cwd=repo,
    )
    assert refused.returncode != 0
    assert "accepted vLLM fleet receipt drifted" in refused.stderr
    (tmp_path / "sync_vllm.txt").write_text(original_vllm_sync)
    (tmp_path / "golden_sync.txt").write_text(
        (tmp_path / "golden_sync.txt").read_text().replace(golden_sha, "0" * 64, 1)
    )
    refused = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        cwd=repo,
    )
    assert refused.returncode != 0
    assert "accepted golden-manifest fleet receipt drifted" in refused.stderr
