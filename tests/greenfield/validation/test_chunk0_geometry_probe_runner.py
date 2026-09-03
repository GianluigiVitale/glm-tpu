"""Static contract of the chunk-0 legacy-geometry probe runner and probe script."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts/greenfield/run_probe_layer1_prompt_chunk0_geometry.sh"
PROBE = REPO / "scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py"
DIGESTS = REPO / "docs/artifacts/gate-d-chunk0-probe-weight-digests.json"


def test_runner_pins_inputs_digests_and_sealed_interpreter():
    runner = RUNNER.read_text()
    assert "readonly CHECKPOINT_INDEX_SHA=e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf" in runner
    digest_sha = sha256(DIGESTS.read_bytes()).hexdigest()
    assert f"readonly WEIGHT_DIGESTS_SHA={digest_sha}" in runner
    assert "readonly WEIGHT_DIGESTS_RELATIVE=docs/artifacts/gate-d-chunk0-probe-weight-digests.json" in runner
    assert "readonly SEALED_PYTHON=/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12" in runner
    assert "readonly SEALED_PYTHON_SHA=021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7" in runner
    assert 'TAG=${GLM_GREENFIELD_CHUNK0_GEOMETRY_TAG:-}' in runner
    assert "must be an approved composed tag" in runner
    # the probe runs under the sealed interpreter in isolated mode from the detached source; no PYTHONPATH, no virtualenv
    assert '"$SEALED_PYTHON" -I -S -B "$SOURCE/scripts/greenfield/probe_layer1_prompt_chunk0_geometry.py"' in runner
    assert "vllm-env" not in runner
    assert "PYTHONPATH=" not in runner
    assert '/usr/bin/env -i HOME=/home/gianl PATH=/usr/bin:/bin LANG=C LC_ALL=C PYTHONDONTWRITEBYTECODE=1' in runner
    assert '--weight-digests "$SOURCE/$WEIGHT_DIGESTS_RELATIVE"' in runner


def test_runner_git_authentication_is_sanitized_and_atomic():
    runner = RUNNER.read_text()
    for token in ("GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1", "GIT_NO_LAZY_FETCH=1", "GIT_NO_REPLACE_OBJECTS=1", "GIT_TERMINAL_PROMPT=0"):
        assert token in runner
    assert "for-each-ref --format='%(refname)' refs/replace" in runner
    assert 'ls-remote --exit-code "$GREENFIELD_ORIGIN" "refs/heads/$BRANCH"' in runner
    assert '[[ $ORIGIN_TIP == "$PIN" ]]' in runner
    assert 'worktree add -q --detach "$SOURCE" "$PIN"' in runner
    assert "status --porcelain=v1 --ignored --untracked-files=all" in runner
    # lease before vacancy, vacancy before the atomic run-directory creation
    lease = runner.index("/usr/bin/flock -n 9")
    vacancy = runner.index("vacancy_check --soft-deleted --exhaustive")
    mkdir = runner.index('/usr/bin/mkdir "$RUN_DIR"')
    assert lease < vacancy < mkdir
    assert 'mkdir -p "$RUN_DIR"' not in runner
    assert '[[ ! -e $RUN_DIR && ! -L $RUN_DIR ]]' in runner


def test_runner_vacancy_ledger_census_and_log_freeze():
    runner = RUNNER.read_text()
    assert 'vacancy_check; vacancy_check --all-versions; vacancy_check --soft-deleted --exhaustive' in runner
    assert '$rc -eq 1 && $output == *"$VACANCY_EXPECTED_SUBSTRING"*' in runner
    assert '"objects", "describe"' in runner
    assert '"generation": str(remote["generation"])' in runner
    assert '"storage", "hash", "--skip-md5"' in runner
    assert "REMOTE_SET_EXACT" in runner and "remote object listing contains duplicates" in runner
    assert "terminal ledger checksum mismatch" in runner
    assert 'touch "$RUN_DIR/CENSUS_UNVERIFIED"' in runner
    assert "strict_census failure_exit || true" not in runner
    # orchestrator.log frozen before the upload; later messages go to the post-upload log
    assert "sealed=1" in runner and 'POST_LOG=/home/gianl/glm-run/$TAG.post_upload.log' in runner
    assert runner.index("sealed=1") < runner.index('storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/"')
    assert runner.index("strict_census post ||") < runner.index('touch "$RUN_DIR/SUCCESS"') < runner.index('storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/"')


def test_probe_binds_runtime_worktree_digests_and_m64_one_row_arm():
    probe = PROBE.read_text()
    assert 'spec_from_file_location(\n    "gate_d_sealed_runtime", REPO / "glm_tpu/greenfield/benchmarking/sealed_runtime.py"' in probe
    assert "sealed_runtime.validate_python_runtime()" in probe and "sealed_runtime.validate_dependency_sites()" in probe
    assert "sealed_runtime.install_sealed_source_path(REPO)" in probe
    assert "sealed_runtime.verify_import_closure(REPO)" in probe
    assert "import torch" not in probe and "safe_open" not in probe
    assert "def _worktree_binding(" in probe and 'if repo.resolve() != (RUN_ROOT / run_tag / "source").resolve():' in probe
    assert 'for-each-ref", "--format=%(refname)", "refs/replace"' in probe
    assert "def _verify_weight_digests(" in probe
    assert 'raise RuntimeError("checkpoint shard bytes drifted from the pinned digest")' in probe
    assert 'raise RuntimeError(f"checkpoint tensor drifted from the pinned digest: {key}")' in probe
    assert "layer1_keys_from_normalized" in probe
    assert '"one_row_block_m64_keys_vs_legacy_lanes"' in probe
    assert '"control_layer0_keys_vs_db518_mismatched_rows"' in probe
    record = json.loads(DIGESTS.read_text())
    assert record["artifact_kind"] == "gate_d_chunk0_probe_weight_digests"
    assert record["checkpoint_index_sha256"] == "e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf"
    assert record["shard"]["filename"] == "model-00001-of-00141.safetensors"
    assert len(record["tensors"]) == 19
