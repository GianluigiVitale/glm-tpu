from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[3]
DISTRIBUTED_PROBE = (
    REPO / "scripts/greenfield/probe_layer0_distributed_q_a_norm.py"
)
STATE_PROBE = REPO / "scripts/greenfield/probe_layer0_dsa_association.py"
WRAPPER = REPO / "scripts/greenfield/run_layer0_dsa_association_probe.sh"
SCORER_PROBE = (
    REPO / "scripts/greenfield/probe_layer0_dsa_scorer_association.py"
)
SCORER_WRAPPER = (
    REPO / "scripts/greenfield/run_layer0_dsa_scorer_association_probe.sh"
)


def test_distributed_q_a_probe_is_exact_pin_and_collective_bound() -> None:
    source = DISTRIBUTED_PROBE.read_text()
    for required in (
        "jax.distributed.initialize",
        "jax.process_count() != 8",
        "jax.local_device_count() != 4",
        "jax.device_count() != 32",
        '"launch_process_id": args.process_id',
        '"jax_process_index": jax.process_index()',
        "if args.process_id == 0:",
        "jax.make_array_from_callback",
        "MODEL_CONFIG_SHA256",
        'model_config.get("rms_norm_eps") != 1e-5',
        "legacy_tp32_distributed_q_a_norm",
        "distributed_collective_violations",
        "diagnostic_only_full_pod_collectives",
        "profiler_free_timing\": False",
        "jax.distributed.shutdown",
    ):
        assert required in source
    assert "jax.process_index() != args.process_id" not in source
    assert "if jax.process_index() == 0:" not in source
    for forbidden in ("import tpu_inference", "from tpu_inference", "import vllm"):
        assert forbidden not in source
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(DISTRIBUTED_PROBE)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_state_probe_requires_checksum_bound_distributed_result() -> None:
    source = STATE_PROBE.read_text()
    for required in (
        "--distributed-q-a-norm-dir",
        "--distributed-q-a-norm-manifest-sha256",
        "--distributed-q-a-norm-code-hash",
        "_inspect_distributed_q_a_norm_artifact",
        "layer0_dsa_state_from_q_residual",
        "legacy_tp32_distributed_q_a_norm_fp32_divsqrt",
        "legacy_tp32_distributed_q_a_norm_bf16_wk_divsqrt",
        "SEALED_LEGACY_WK_WEIGHTS_PROJ_DTYPE",
        "candidate_changes_only_prompt_keys",
        "model_epsilon_bf16_wk_local_dcp_association_restored",
        "LegacyDcpXlaScoreGeometry",
        "legacy_local_dcp_score_inputs",
        "legacy_local_dcp_xla_scores",
        "model_epsilon_distributed_local_dcp_association_restored",
        'phase="legacy_local_dcp_xla_score"',
    ):
        assert required in source
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(STATE_PROBE)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_distributed_result_artifact_round_trips_and_refuses_drift(
    tmp_path: Path,
) -> None:
    from safetensors.numpy import save_file

    from scripts.greenfield.probe_layer0_dsa_association import (
        _inspect_distributed_q_a_norm_artifact,
        _manifest_hash,
        _sha256_file,
    )

    q_bits = np.arange(32 * 2048, dtype=np.uint16).reshape(32, 2048)
    tensor_path = tmp_path / "distributed_q_a_norm.safetensors"
    save_file(
        {"q_residual_bfloat16_bits": q_bits},
        tensor_path,
        metadata={
            "artifact_kind": "greenfield_distributed_q_a_norm",
            "format_version": "1",
        },
    )
    code_hash = "1" * 40
    input_hash = "2" * 64
    manifest = {
        "artifact_kind": "greenfield_distributed_q_a_norm",
        "format_version": 1,
        "diagnostic_only": True,
        "code_hash": code_hash,
        "input_manifest_sha256": input_hash,
        "model_config": {
            "path": "reference/hf-repo/config.json",
            "sha256": (
                "22e49334abf8562fecf70ca3292ba3f5b33f5602fb2bf10b52dd64a66cfe65ff"
            ),
            "rms_norm_eps": 1e-5,
        },
        "numerical_geometry": {
            "input_rms_norm_epsilon": 1e-5,
            "q_a_rms_norm_epsilon": 1e-5,
            "key_layer_norm_epsilon": 1e-6,
        },
        "hlo_sha256": "3" * 64,
        "q_residual": {
            "shape": [32, 2048],
            "dtype": "bfloat16",
            "byte_count": q_bits.nbytes,
            "sha256": sha256(q_bits.view(np.uint8)).hexdigest(),
        },
        "file": {
            "filename": tensor_path.name,
            "byte_count": tensor_path.stat().st_size,
            "sha256": _sha256_file(tensor_path),
        },
    }
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    inspected, actual = _inspect_distributed_q_a_norm_artifact(
        tmp_path,
        expected_manifest_sha256=manifest["manifest_sha256"],
        expected_code_hash=code_hash,
        expected_input_manifest_sha256=input_hash,
    )
    assert inspected == manifest
    np.testing.assert_array_equal(actual, q_bits)


def test_protected_wrapper_reuses_db491_for_bf16_wk_discriminator() -> None:
    source = WRAPPER.read_text()
    for required in (
        ".glm_pod_workload.lock",
        "strict_census pre",
        "strict_census post",
        "probe_layer0_dsa_association.py",
        "DEFAULT_DISTRIBUTED_Q_A_DIR",
        "DISTRIBUTED_Q_A_MANIFEST_SHA",
        "DISTRIBUTED_Q_A_CODE_HASH",
        "reusing sealed DB491 q-a artifact; no repeated 32-chip phase",
        "MODEL_CONFIG_SHA",
        "distributed_q_a_norm_manifest_sha256",
        "--distributed-q-a-norm-code-hash",
        "legacy_local_dcp_xla_score",
        "legacy_tp32_distributed_q_a_norm_bf16_wk_divsqrt",
        "accepted_wk_dtype_boundary",
        "model_epsilon_bf16_wk_local_dcp_association_restored",
        "bounded-real-layer0-v6-model-epsilon-bf16-origin-wk",
        "bench/results.db",
        "--no-clobber",
        "REMOTE_PREFIX/SUCCESS",
    ):
        assert required in source
    for forbidden in (
        "probe_layer0_distributed_q_a_norm.py",
        "DISTRIBUTED_Q_A_NORM_OK",
        "strict_census distributed_post",
    ):
        assert forbidden not in source
    completed = subprocess.run(
        ["bash", "-n", str(WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_scorer_association_probe_reuses_sealed_breakpoints() -> None:
    source = SCORER_PROBE.read_text()
    for required in (
        "inspect_greenfield_layer0_dsa_internal_observation",
        "inspect_greenfield_layer0_dsa_selected_observation",
        "inspect_legacy_prompt_index_cache",
        "pack_stage_local_index_keys",
        "stitch_stage_local_scores",
        "dsa_scores",
        'phase="local_wide_score"',
        'phase="local_wide_default_score"',
        'precision="default"',
        "current-wide scorer failed to reproduce",
        '"candidate_restored"',
        '"profiler_free_timing": False',
    ):
        assert required in source
    for forbidden in (
        "import tpu_inference",
        "from tpu_inference",
        "import vllm",
        "jax.distributed.initialize",
    ):
        assert forbidden not in source
    completed = subprocess.run(
        [sys.executable, "-m", "py_compile", str(SCORER_PROBE)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_scorer_association_wrapper_is_protected_and_bounded() -> None:
    source = SCORER_WRAPPER.read_text()
    for required in (
        ".glm_pod_workload.lock",
        "strict_census pre",
        "strict_census post",
        "syncing exact reviewed pin to all eight hosts",
        "probe_layer0_dsa_scorer_association.py",
        "ASSOCIATION_MANIFEST_SHA",
        "PROMPT_CACHE_MANIFEST_SHA",
        "INTERNAL_CONTRACT_SHA",
        "INTERNAL_TENSOR_SHA",
        "SELECTED_OBSERVATION_SHA",
        "bounded-wide-highest-vs-default-v1",
        "current-wide protected control did not reproduce",
        "bench/results.db",
        "--no-clobber",
        "REMOTE_PREFIX/SUCCESS",
    ):
        assert required in source
    for forbidden in (
        "compile_short_decoder.py",
        "run_short_decoder_compile_pp8",
        "jax.distributed.initialize",
    ):
        assert forbidden not in source
    completed = subprocess.run(
        ["bash", "-n", str(SCORER_WRAPPER)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
