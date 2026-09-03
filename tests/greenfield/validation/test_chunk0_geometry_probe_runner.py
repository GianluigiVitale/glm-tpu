"""Static and executed contracts of the chunk-0 legacy-geometry probe runner and probe."""

from __future__ import annotations

import json
import importlib.util
from io import BytesIO
import os
import re
import shutil
import socket
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/greenfield/run_probe_layer1_prompt_chunk0_geometry.sh"
PROBE = REPO / "scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py"
CHUNK_PIPELINE = (
    REPO / "glm_tpu/greenfield/benchmarking/legacy_prefill_chunk_probe.py"
)
DIGESTS = REPO / "docs/artifacts/gate-d-chunk0-probe-weight-digests.json"
PUBLISHER = REPO / "scripts/greenfield/publish_gate_d_layer1_prompt_chunk0_geometry.py"
LAUNCHER = REPO / "scripts/greenfield/launch_gate_d_layer1_prompt_chunk0_geometry.py"
INSTALLER = REPO / "scripts/greenfield/install_gate_d_layer1_prompt_chunk0_geometry_runtime.py"
REWRITE_MIRROR = REPO / "scripts/greenfield/verify_gate_d_rewrite_same_region_git_mirror.py"
LAYER0_INPUT = Path(
    "/home/gianl/glm-run/greenfield_layer0_dsa_input_fused_qkv_20260807T202538052784486Z"
)
LEGACY_LAYER1_CACHE = Path(
    "/home/gianl/glm-run/greenfield_legacy_layer1_prompt_index_cache_20260903T000356727206404Z/prompt_index_cache"
)


def test_runner_pins_inputs_digests_and_sealed_interpreter():
    runner = RUNNER.read_text()
    assert "readonly CHECKPOINT_INDEX_SHA=e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf" in runner
    assert f"readonly WEIGHT_DIGESTS_SHA={sha256(DIGESTS.read_bytes()).hexdigest()}" in runner
    assert "readonly SEALED_PYTHON=/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12" in runner
    assert "readonly SEALED_PYTHON_SHA=021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7" in runner
    assert 'TAG=${GLM_GATE_D_CHUNK0_GEOMETRY_TAG:-}' in runner
    assert "unsafe Gate-D chunk-0 geometry tag" in runner
    # The only executed probe pathname is the root-owned immutable capsule.
    assert "readonly CAPSULE=/usr/local/libexec/glm-tpu/gate-d-layer1-prompt-chunk0-geometry-v2" in runner
    assert '"$SEALED_PYTHON" -I -S -B -u "$PROBE"' in runner
    assert "$PROBE_LOCAL" not in runner and "/source/probe_layer1" not in runner
    assert '--code-pin "$PIN"' in runner and '--repository "$WORKTREE"' in runner
    assert "worktree add" not in runner
    assert "PYTHONPATH=" not in runner
    # the only vllm-env reference is the remote census enumerator on the pod hosts
    assert runner.count("vllm-env") == 1 and "ray_enum=" in runner


@pytest.mark.skipif(
    not LAYER0_INPUT.exists() or not LEGACY_LAYER1_CACHE.exists(),
    reason="sealed real manifest artifacts are not present",
)
def test_runner_distinguishes_manifest_self_hashes_from_raw_file_hashes():
    runner = RUNNER.read_text()

    def constant(name: str) -> str:
        match = re.search(rf"^readonly {name}=([0-9a-f]{{64}})$", runner, re.M)
        assert match is not None
        return match.group(1)

    for prefix, path in (
        ("INPUT", LAYER0_INPUT / "manifest.json"),
        ("LEGACY_LAYER1", LEGACY_LAYER1_CACHE / "manifest.json"),
    ):
        manifest = json.loads(path.read_text())
        body = dict(manifest)
        recorded_self_hash = body.pop("manifest_sha256")
        canonical = json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        self_hash = constant(f"{prefix}_MANIFEST_SHA")
        file_hash = constant(f"{prefix}_MANIFEST_FILE_SHA")
        assert self_hash == recorded_self_hash == sha256(canonical).hexdigest()
        assert file_hash == sha256(path.read_bytes()).hexdigest()
        assert self_hash != file_hash
        assert f'manifest.json:${prefix}_MANIFEST_FILE_SHA' in runner
        assert f'manifest.json:${prefix}_MANIFEST_SHA' not in runner


def test_runner_git_authentication_vacancy_grammar_and_atomic_run_dir():
    runner = RUNNER.read_text()
    for token in ("GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1", "GIT_NO_LAZY_FETCH=1", "GIT_NO_REPLACE_OBJECTS=1", "GIT_TERMINAL_PROMPT=0"):
        assert token in runner
    assert "for-each-ref --format='%(refname)' refs/replace" in runner
    assert 'ls-remote --exit-code "$ORIGIN" "refs/heads/$BRANCH"' in runner
    assert '[[ $ORIGIN_TIP == "$PIN" ]]' in runner
    assert "readonly VACANCY_EXPECTED='ERROR: (gcloud.storage.ls) One or more URLs matched no objects.'" in runner
    assert '$live_rc -eq 1 && $live == "$VACANCY_EXPECTED"' in runner
    assert "PYTHONWARNINGS=ignore /usr/bin/timeout" in runner
    assert "vacancy_three_surfaces" in runner
    assert "--soft-deleted --exhaustive" in runner
    assert 'run_identity=$(publisher init --run-dir "$RUN_DIR")' in runner
    assert 'exec 7<"$RUN_DIR"' not in runner
    assert "def _create_retained_run_fd(" in LAUNCHER.read_text()
    assert "os.O_DIRECTORY | os.O_NOFOLLOW" in LAUNCHER.read_text()
    assert "publisher success" in runner


def test_runner_publisher_census_and_descriptor_boundary():
    runner = RUNNER.read_text()
    assert '--run-dir "$RUN_DIR" --run-dir-fd 7' in runner
    assert 'publish_member runner.log' in runner
    assert 'publisher diagnostic --run-dir "$RUN_DIR"' in runner
    assert "storage cp --recursive" not in runner and "remote_objects.json" not in runner
    assert runner.index("strict_census post ||") < runner.index("publisher success --run-dir")
    # the census uses the proven carrier-marked enumerator and bracketed patterns only
    assert "GLM_CENSUS_CARRIER" in runner and "RAY_PROCESSES" in runner
    assert "[p]robe_layer1_prompt_chunk0_geometry[.]py" in runner
    assert 'pgrep -f "ray::|raylet|gcs_server|EngineCore"' not in runner


def _census_command() -> str:
    """Extract the census shell snippet exactly as the runner composes it for one label."""

    runner = RUNNER.read_text()
    ray_enum = re.search(r"ray_enum='(.*)'\n", runner).group(1)
    command = re.search(r"\n  command='(.*)'\n  set \+e", runner, re.S).group(1)
    carrier = "unit_census_carrier"
    ray_enum = ray_enum.replace("'\"$carrier\"'", carrier)
    return command.replace("'\"$ray_enum\"'", ray_enum), carrier


@pytest.mark.skipif(
    shutil.which("pgrep") is None or subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode != 0,
    reason="executed census regression needs pgrep and passwordless sudo",
)
def test_executed_census_does_not_match_its_own_shell():
    command, carrier = _census_command()
    # The census must not report itself busy: its own command line contains the
    # patterns it searches for, so a naive pgrep would self-match.
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, env={**os.environ, "GLM_CENSUS_CARRIER": carrier})
    lines = result.stdout.strip().splitlines()
    assert lines, result.stderr
    assert lines[0].split()[0] in {"CENSUS_OK", "CENSUS_BUSY"}, lines
    assert lines[0].split()[1] == socket.gethostname()
    if lines[0].startswith("CENSUS_BUSY"):
        # busy is legitimate only for real work, never for the census's own probe pattern
        assert "probe_layer1_prompt_chunk0_geometry" not in result.stdout


def test_probe_is_self_verifying_and_imports_from_a_sealed_archive():
    probe = PROBE.read_text()
    assert 'PROBE_REPOSITORY_PATH = "scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py"' in probe
    assert "def _verify_running_source(" in probe and 'raise RuntimeError("probe is not the committed blob at the approved pin")' in probe
    assert "def _sealed_git_source_archive(" in probe and "os.memfd_create(" in probe and "_F_ADD_SEALS" in probe
    assert 'sys.path[:] = [archive_path, JAX_SITE_ROOT, LIBTPU_SITE_ROOT, *EXPECTED_RUNTIME_PATH]' in probe
    assert "sealed_runtime.verify_import_closure(Path(archive_path))" in probe
    for token in ("GIT_CONFIG_GLOBAL", "GIT_NO_REPLACE_OBJECTS", "GIT_NO_LAZY_FETCH", "GIT_CONFIG_NOSYSTEM"):
        assert token in probe
    assert "refs/replace" in probe
    assert "probe is not executing from the immutable capsule" in probe
    assert "def _open_inherited_run_dir(" in probe and "def _write_run_member_exclusive(" in probe
    assert "dir_fd=run_fd" in probe and 'parser.add_argument("--run-dir-fd", type=int, choices=(7,)' in probe
    assert "args.output" not in probe and "args.hlo_dir" not in probe
    assert "import torch" not in probe and "safe_open" not in probe and "_worktree_binding" not in probe
    assert "db518_raw = _snapshot_regular(args.db518_result)" in probe and "np.load(BytesIO(db518_raw))" in probe
    assert "digests_raw = _snapshot_regular(args.weight_digests)" in probe
    assert 'shard_digests={digest_record["shard"]["filename"]: digest_record["shard"]["sha256"]}' in probe
    assert "layer1_keys_from_normalized" in probe and '"one_row_block_m64_keys_vs_legacy_lanes"' in probe
    record = json.loads(DIGESTS.read_text())
    assert record["artifact_kind"] == "gate_d_chunk0_probe_weight_digests" and len(record["tensors"]) == 19


def test_wk_materialization_is_a_completed_executable_boundary():
    probe = PROBE.read_text()
    pipeline = CHUNK_PIPELINE.read_text()
    publisher = PUBLISHER.read_text()

    assert "materialize_stage_local_prefill_index_wk" not in pipeline
    assert '"wk0", "wk1", "rope_table"' in pipeline
    assert 'w["wk0"]' in pipeline and 'w["wk1"]' in pipeline
    assert "decode_stage_local_prefill_index_wk_bf16" in probe
    assert "promote_stage_local_prefill_index_wk" in probe
    assert probe.count(".block_until_ready()") >= 4
    assert probe.index("wk1_bf16.block_until_ready()") < probe.index(
        "promote_lowered ="
    )
    assert probe.index("wk1.block_until_ready()") < probe.index(
        'weights = {'
    )
    assert '"wk0": wk0' in probe and '"wk1": wk1' in probe
    for name in (
        "hlo/wk_decode.optimized_hlo.txt",
        "hlo/wk_decode.stablehlo.mlir",
        "hlo/wk_promote.optimized_hlo.txt",
        "hlo/wk_promote.stablehlo.mlir",
    ):
        assert name in probe and name in publisher
    assert 'hlo.get("main_raw_wk_parameters") != []' in publisher
    assert '"w__wk0_bits__" in optimized_text' in publisher


def test_capsule_hash_chain_and_second_vacancy_check_are_bound():
    runner = RUNNER.read_text()
    launcher = LAUNCHER.read_text()
    installer = INSTALLER.read_text()
    publisher = PUBLISHER.read_text()
    probe_sha = sha256(PROBE.read_bytes()).hexdigest()
    publisher_sha = sha256(PUBLISHER.read_bytes()).hexdigest()
    wrapper_sha = sha256(RUNNER.read_bytes()).hexdigest()
    launcher_sha = sha256(LAUNCHER.read_bytes()).hexdigest()
    assert probe_sha in runner and probe_sha in launcher and probe_sha in installer
    assert publisher_sha in runner and publisher_sha in launcher and publisher_sha in installer
    assert wrapper_sha in launcher and launcher_sha in installer
    assert "__PROBE_SHA256__" not in "".join((runner, launcher, installer))
    assert "__PUBLISHER_SHA256__" not in "".join((runner, launcher, installer))
    assert "__WRAPPER_SHA256__" not in launcher and "__LAUNCHER_SHA256__" not in installer
    assert "base._require_never_used_prefix(bucket, prefix)" in publisher
    assert publisher.index("base._require_never_used_prefix(bucket, prefix)") < publisher.index(
        "for relative in sorted(payload)"
    )
    assert "_RENAME_NOREPLACE = 1" in installer
    assert "os.MFD_ALLOW_SEALING" in launcher and "LOCK_NAMES" in launcher
    assert "base._replay_bound(bucket, diagnostic_prefix + record[\"path\"], record)" in publisher
    assert "chunk-0 diagnostic terminal object set drifted" in publisher


def test_installer_and_launcher_have_a_strict_install_only_boundary():
    installer_module = _load_module("chunk0_installer", INSTALLER)
    launcher_module = _load_module("chunk0_launcher", LAUNCHER)
    installer = INSTALLER.read_text()
    launcher = LAUNCHER.read_text()
    expected = {
        PROBE.name: sha256(PROBE.read_bytes()).hexdigest(),
        PUBLISHER.name: sha256(PUBLISHER.read_bytes()).hexdigest(),
        LAUNCHER.name: sha256(LAUNCHER.read_bytes()).hexdigest(),
        "verify_gate_d_rewrite_same_region_git_mirror.py": sha256(
            (REPO / "scripts/greenfield/verify_gate_d_rewrite_same_region_git_mirror.py").read_bytes()
        ).hexdigest(),
    }
    assert installer_module.PAYLOADS == expected
    assert str(installer_module.CAPSULE_TARGET).startswith("/usr/local/libexec/glm-tpu/")
    assert str(installer_module.LAUNCHER_TARGET).endswith(
        "launch_gate_d_layer1_prompt_chunk0_geometry_v3.py"
    )
    assert str(installer_module.LAUNCHER_TARGET).startswith("/opt/glm-tpu/bin/")
    assert installer.startswith("#!/usr/bin/env -S /usr/bin/python3 -I -S -B\n")
    assert '"launcher_invoked": False' in installer
    assert "subprocess" not in installer and "execve" not in installer
    assert launcher.startswith("#!/usr/bin/env -S /usr/bin/python3 -I -S -B\n")
    assert str(launcher_module.INSTALL_PATH) == str(installer_module.LAUNCHER_TARGET)
    assert launcher_module.WRAPPER_SHA256 == sha256(RUNNER.read_bytes()).hexdigest()
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in launcher
    assert 'f"/proc/self/fd/{WRAPPER_FD}"' in launcher


def test_mirror_verifier_is_bound_to_the_rewrite_branch_and_reviewed_base():
    verifier = _load_module("chunk0_rewrite_mirror", REWRITE_MIRROR)
    assert verifier.WORKTREE == REPO
    assert verifier.BRANCH == "rewrite/topology-first-decode"
    assert verifier.SOURCE_PATH.endswith(
        "verify_gate_d_rewrite_same_region_git_mirror.py"
    )
    assert sha256((REPO / verifier.BASE_PATH).read_bytes()).hexdigest() == verifier.BASE_SHA256
    assert set(verifier.BOUND_PATHS) >= {
        "scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py",
        "scripts/greenfield/publish_gate_d_layer1_prompt_chunk0_geometry.py",
        "scripts/greenfield/run_probe_layer1_prompt_chunk0_geometry.sh",
        verifier.SOURCE_PATH,
    }


def test_launcher_retains_the_original_run_directory_across_path_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    launcher = _load_module("chunk0_launcher_retained_fd", LAUNCHER)
    tmp_path.chmod(0o700)
    monkeypatch.setattr(launcher, "RUN_ROOT", tmp_path)
    tag = "greenfield_layer1_prompt_chunk0_geometry_20260903T000000000000000Z"
    descriptor = launcher._create_retained_run_fd(tag)
    try:
        held = os.fstat(descriptor)
        original = tmp_path / tag
        parked = tmp_path / f"{tag}.parked"
        original.rename(parked)
        original.symlink_to(tmp_path)
        assert (held.st_dev, held.st_ino) == (
            os.fstat(descriptor).st_dev,
            os.fstat(descriptor).st_ino,
        )
        assert (held.st_dev, held.st_ino) != (
            original.stat().st_dev,
            original.stat().st_ino,
        )
    finally:
        os.close(descriptor)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_publisher():
    return _load_module("chunk0_publisher", PUBLISHER)


@pytest.mark.parametrize("terminal", ["PROBE_RESULT", "terminal_upload_receipt.json"])
def test_publisher_rejects_reentry_after_its_own_terminal(tmp_path: Path, terminal: str):
    publisher = _load_publisher()

    class Base:
        @staticmethod
        def require_preterminal(_run_fd: int) -> None:
            return None

    (tmp_path / terminal).write_bytes(b"terminal\n")
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(RuntimeError, match="terminal publication"):
            publisher._require_preterminal(Base(), descriptor)
    finally:
        os.close(descriptor)


def _synthetic_probe_archive(*, mismatch: bool, control_mismatch: bool = False):
    import numpy as np

    publisher = _load_publisher()
    arrays = {
        name: np.zeros(shape, dtype=np.uint16)
        for name, (shape, _dtype) in publisher.EXPECTED_OUTPUT_LAYOUT.items()
    }
    if mismatch:
        arrays["keys1_bits"][0, 0] = 1
    if control_mismatch:
        arrays["keys0_bits"][0, 0] = 1
    buffer = BytesIO()
    np.savez(buffer, **arrays)
    raw = buffer.getvalue()
    legacy = [1 if mismatch else 0] + [0] * 2047
    row0 = {
        "attention_row0_db533_vs_pairwise_lanes": 0,
        "legacy_geometry_vs_greenfield_db518_lanes": int(mismatch),
        "legacy_geometry_vs_legacy_lanes": int(mismatch),
        "legacy_geometry_vs_one_row_block_m64_keys_lanes": int(mismatch),
        "one_row_block_m64_keys_vs_greenfield_db518_lanes": 0,
        "one_row_block_m64_keys_vs_legacy_lanes": 0,
        "one_row_block_one_row_keys_vs_legacy_lanes": 0,
    }
    runner = {
        "arrays_sha256": sha256(raw).hexdigest(),
        "control_layer0_keys_vs_db518_mismatched_rows": int(control_mismatch),
        "row0": row0,
        "chunk0_vs_legacy": {
            "lanes_mismatched": int(mismatch),
            "per_64_row_block_mean": [int(mismatch) / 64] + [0.0] * 31,
            "per_row_first_16": legacy[:16],
            "rows": 2048,
            "rows_exact": 2048 - int(mismatch),
        },
        "chunk0_vs_greenfield_db518": {
            "lanes_mismatched": int(mismatch),
            "rows_exact": 2048 - int(mismatch),
        },
    }
    references = {
        name: sha256(arrays[name].tobytes()).hexdigest()
        for name in (
            "greenfield_layer0_bits",
            "greenfield_layer1_bits",
            "legacy_layer1_bits",
        )
    }
    return publisher, raw, runner, references


@pytest.mark.parametrize("mismatch", [False, True])
def test_publisher_rederives_row0_verdict_from_archived_arrays(mismatch: bool):
    publisher, raw, runner, references = _synthetic_probe_archive(mismatch=mismatch)
    assert publisher._validate_npz(
        raw, runner, reference_sha256s=references
    ) is (not mismatch)


def test_publisher_rejects_a_forged_exact_row0_claim():
    publisher, raw, runner, references = _synthetic_probe_archive(mismatch=True)
    runner["row0"]["legacy_geometry_vs_legacy_lanes"] = 0
    with pytest.raises(RuntimeError, match="disagree with archived arrays"):
        publisher._validate_npz(raw, runner, reference_sha256s=references)


def test_publisher_rejects_self_referential_reference_arrays():
    publisher, raw, runner, _references = _synthetic_probe_archive(mismatch=False)
    with pytest.raises(RuntimeError, match="reference array identity drifted"):
        publisher._validate_npz(raw, runner)


def test_publisher_requires_the_rederived_layer0_control_to_be_exact():
    publisher, raw, runner, references = _synthetic_probe_archive(
        mismatch=False, control_mismatch=True
    )
    with pytest.raises(RuntimeError, match="disagree with archived arrays"):
        publisher._validate_npz(raw, runner, reference_sha256s=references)


def test_publisher_mutation_checks_wk_executable_boundaries():
    publisher = _load_publisher()
    main = (
        "HloModule main\n\n"
        "ENTRY %main (w__wk0__.1: f32[128,6144], w__wk1__.1: f32[128,6144]) {\n"
        "  %w__wk0__.1 = f32[128,6144] parameter(0)\n"
        "  %w__wk1__.1 = f32[128,6144] parameter(1)\n"
        "  %other = bf16[1] constant(0)\n"
        "  %key0 = bf16[2048,128] custom-call(%w__wk0__.1)\n"
        "  %key1 = bf16[2048,128] custom-call(%w__wk1__.1)\n"
        "  ROOT %out = (bf16[1], bf16[1], bf16[1], bf16[1], bf16[1], "
        "bf16[2048,128], bf16[2048,128], bf16[1], bf16[1], bf16[1], "
        "bf16[1], bf16[1]) tuple(%other, %other, %other, %other, %other, "
        "%key0, %key1, %other, %other, %other, %other, %other)\n"
        "}\n"
    )
    decode = (
        "HloModule decode\n\n"
        "ENTRY %decode (bits: u8[128,6144], scale: f32[1,48]) {\n"
        "  %bits = u8[128,6144] parameter(0)\n"
        "  %scale = f32[1,48] parameter(1)\n"
        "  %mixed = f32[128,6144] custom-call(%bits, %scale)\n"
        "  ROOT %decoded = bf16[128,6144] convert(%mixed)\n"
        "}\n"
    )
    promote = (
        "HloModule promote\n\n"
        "ENTRY %promote (value: bf16[128,6144]) {\n"
        "  %value = bf16[128,6144] parameter(0)\n"
        "  ROOT %promoted = f32[128,6144] convert(%value)\n"
        "}\n"
    )
    main_stable = (
        "module @main {\n"
        "  func.func public @main(%arg0: tensor<128x6144xf32>, "
        "%arg1: tensor<128x6144xf32>) -> tensor<1xbf16> {\n"
        "  }\n}\n"
    )
    decode_stable = (
        "module @decode {\n"
        "  func.func public @main(%arg0: tensor<128x6144xui8>, "
        "%arg1: tensor<1x48xf32>) -> tensor<128x6144xbf16> {\n"
        "    %0 = stablehlo.convert %arg0 : tensor<128x6144xui8>\n"
        "  }\n}\n"
    )
    promote_stable = (
        "module @promote {\n"
        "  func.func public @main(%arg0: tensor<128x6144xbf16>) "
        "-> tensor<128x6144xf32> {\n"
        "    %0 = stablehlo.convert %arg0 : tensor<128x6144xbf16>\n"
        "  }\n}\n"
    )
    arguments = (
        main,
        main_stable,
        decode,
        decode_stable,
        promote,
        promote_stable,
    )
    publisher._require_wk_hlo_boundaries(*arguments)
    # A metadata decoy cannot rescue a renamed raw entry parameter.
    with pytest.raises(RuntimeError, match="boundary drifted"):
        publisher._require_wk_hlo_boundaries(
            main.replace(
                "%w__wk0__.1 = f32[128,6144] parameter(0)",
                "%renamed = u8[128,6144] parameter(0), metadata={op_name=\"w__wk0__.1: f32[128,6144]\"}",
            ).replace("%w__wk0__.1)", "%renamed)"),
            main_stable,
            decode,
            decode_stable,
            promote,
            promote_stable,
        )
    # Wk may feed only its named key result, never another returned value.
    leaking_main = main.replace(
        "%key0, %key1, %other, %other, %other, %other, %other)",
        "%key0, %key1, %w__wk0__.1, %other, %other, %other, %other)",
    )
    with pytest.raises(RuntimeError, match="boundary drifted"):
        publisher._require_wk_hlo_boundaries(
            leaking_main,
            main_stable,
            decode,
            decode_stable,
            promote,
            promote_stable,
        )
    # Decode parameters must both be live ancestors of the BF16 root.
    with pytest.raises(RuntimeError, match="boundary drifted"):
        publisher._require_wk_hlo_boundaries(
            main,
            main_stable,
            decode.replace(
                "ROOT %decoded = bf16[128,6144] convert(%mixed)",
                "ROOT %decoded = bf16[128,6144] constant(0)",
            ),
            decode_stable,
            promote,
            promote_stable,
        )
    # An unrelated live conversion cannot authenticate an alternate promotion.
    alternate_promote = promote.replace(
        "  ROOT %promoted = f32[128,6144] convert(%value)\n",
        "  %zero = bf16[] constant(0)\n"
        "  %unrelated = f32[] convert(%zero)\n"
        "  ROOT %promoted = f32[128,6144] custom-call(%value, %unrelated)\n",
    )
    with pytest.raises(RuntimeError, match="boundary drifted"):
        publisher._require_wk_hlo_boundaries(
            main,
            main_stable,
            decode,
            decode_stable,
            alternate_promote,
            promote_stable,
        )
    # Wrong helper root dtype and helper communication both fail closed.
    with pytest.raises(RuntimeError, match="boundary drifted"):
        publisher._require_wk_hlo_boundaries(
            main,
            main_stable,
            decode.replace("bf16[128,6144] convert", "f32[128,6144] convert"),
            decode_stable,
            promote,
            promote_stable,
        )
    with pytest.raises(RuntimeError, match="boundary drifted"):
        publisher._require_wk_hlo_boundaries(
            main,
            main_stable,
            decode.replace("custom-call(%bits, %scale)", "all-reduce(%bits)"),
            decode_stable,
            promote,
            promote_stable,
        )
    # A live in-main BF16 rematerialization of wk is rejected.
    rounded_main = main.replace(
        "  %key0 = bf16[2048,128] custom-call(%w__wk0__.1)\n",
        "  %rounded = bf16[128,6144] convert(%w__wk0__.1)\n"
        "  %key0 = bf16[2048,128] custom-call(%rounded)\n",
    )
    with pytest.raises(RuntimeError, match="boundary drifted"):
        publisher._require_wk_hlo_boundaries(
            rounded_main,
            main_stable,
            decode,
            decode_stable,
            promote,
            promote_stable,
        )


def test_sealed_archive_builds_from_committed_blobs_and_imports(tmp_path: Path):
    """Build the sealed archive from this repository's HEAD and import a module from it."""

    head = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    program = f'''
import importlib.util, sys
spec = importlib.util.spec_from_file_location("probe", {str(PROBE)!r})
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
from pathlib import Path
archive_path, identity = probe._sealed_git_source_archive(Path({str(REPO)!r}), {head!r})
assert identity["file_manifest_count"] > 50, identity
sys.path[:] = [archive_path, *sys.path]
import glm_tpu.greenfield.benchmarking.legacy_prefill_owner_packing as pk
assert pk.__file__.startswith(archive_path + "/"), pk.__file__
print("ARCHIVE_IMPORT_OK", identity["file_manifest_count"], identity["archive_sha256"][:12])
'''
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, env={**os.environ, "JAX_PLATFORMS": "cpu"})
    assert "ARCHIVE_IMPORT_OK" in result.stdout, result.stderr
