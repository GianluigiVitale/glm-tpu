"""Static and executed contracts of the chunk-0 legacy-geometry probe runner and probe."""

from __future__ import annotations

import json
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
DIGESTS = REPO / "docs/artifacts/gate-d-chunk0-probe-weight-digests.json"


def test_runner_pins_inputs_digests_and_sealed_interpreter():
    runner = RUNNER.read_text()
    assert "readonly CHECKPOINT_INDEX_SHA=e0fe7f28c1f853d4824e4d796374e3dacf1fe470988773952c79b063768134bf" in runner
    assert f"readonly WEIGHT_DIGESTS_SHA={sha256(DIGESTS.read_bytes()).hexdigest()}" in runner
    assert "readonly SEALED_PYTHON=/opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12" in runner
    assert "readonly SEALED_PYTHON_SHA=021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7" in runner
    assert 'TAG=${GLM_GREENFIELD_CHUNK0_GEOMETRY_TAG:-}' in runner and "must be an approved composed tag" in runner
    # committed bytes only: probe and digest record extracted from the pin's blobs
    assert 'git_local -C "$WORKTREE" show "$PIN:$PROBE_REPOSITORY_PATH" >"$PROBE_LOCAL"' in runner
    assert 'git_local -C "$WORKTREE" show "$PIN:$WEIGHT_DIGESTS_RELATIVE" >"$DIGESTS_LOCAL"' in runner
    assert '"$SEALED_PYTHON" -I -S -B "$PROBE_LOCAL"' in runner
    assert '--code-pin "$PIN"' in runner and '--repository "$WORKTREE"' in runner
    assert "worktree add" not in runner
    assert "PYTHONPATH=" not in runner
    # the only vllm-env reference is the remote census enumerator on the pod hosts
    assert runner.count("vllm-env") == 1 and "ray_enum=" in runner


def test_runner_git_authentication_vacancy_grammar_and_atomic_run_dir():
    runner = RUNNER.read_text()
    for token in ("GIT_CONFIG_GLOBAL=/dev/null", "GIT_CONFIG_NOSYSTEM=1", "GIT_NO_LAZY_FETCH=1", "GIT_NO_REPLACE_OBJECTS=1", "GIT_TERMINAL_PROMPT=0"):
        assert token in runner
    assert "for-each-ref --format='%(refname)' refs/replace" in runner
    assert 'ls-remote --exit-code "$GREENFIELD_ORIGIN" "refs/heads/$BRANCH"' in runner
    assert '[[ $ORIGIN_TIP == "$PIN" ]]' in runner
    assert "readonly VACANCY_EXPECTED='ERROR: (gcloud.storage.ls) One or more URLs matched no objects.'" in runner
    assert '[[ $rc -eq 1 && $output == "$VACANCY_EXPECTED" ]]' in runner
    assert "PYTHONWARNINGS=ignore /usr/bin/timeout" in runner
    assert 'vacancy_check; vacancy_check --all-versions; vacancy_check --soft-deleted --exhaustive' in runner
    lease = runner.index("/usr/bin/flock -n 9")
    vacancy = runner.index("vacancy_check --soft-deleted --exhaustive")
    mkdir = runner.index('/usr/bin/mkdir "$RUN_DIR"')
    assert lease < vacancy < mkdir
    assert 'mkdir -p "$RUN_DIR"' not in runner and '[[ ! -e $RUN_DIR && ! -L $RUN_DIR ]]' in runner


def test_runner_ledger_census_and_log_freeze():
    runner = RUNNER.read_text()
    assert '"objects", "describe"' in runner and '"generation": str(remote["generation"])' in runner
    assert '"storage", "hash", "--skip-md5"' in runner
    assert "REMOTE_SET_EXACT" in runner and "remote object listing contains duplicates" in runner
    assert "terminal ledger checksum mismatch" in runner
    assert 'touch "$RUN_DIR/CENSUS_UNVERIFIED"' in runner and "strict_census failure_exit || true" not in runner
    assert "sealed=1" in runner and 'POST_LOG=/home/gianl/glm-run/$TAG.post_upload.log' in runner
    assert runner.index("sealed=1") < runner.index('storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/"')
    assert runner.index("strict_census post ||") < runner.index('touch "$RUN_DIR/SUCCESS"') < runner.index('storage cp --recursive --no-clobber "$RUN_DIR" "$REMOTE_PREFIX/"')
    # the census uses the proven carrier-marked enumerator and bracketed patterns only
    assert "GLM_CENSUS_CARRIER" in runner and "RAY_PROCESSES" in runner
    assert "[p]robe_layer1_prompt_chunk0_geometry[.]py" in runner
    assert 'pgrep -f "ray::|raylet|gcs_server|EngineCore"' not in runner


def _census_command() -> str:
    """Extract the census shell snippet exactly as the runner composes it for one label."""

    runner = RUNNER.read_text()
    ray_enum = re.search(r"ray_enum='(.*)'\n", runner).group(1)
    command = re.search(r"\n  command='(.*)'\n  GLM_CENSUS_CARRIER=", runner, re.S).group(1)
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
    assert "import torch" not in probe and "safe_open" not in probe and "_worktree_binding" not in probe
    assert "db518_raw = _snapshot_regular(args.db518_result)" in probe and "np.load(BytesIO(db518_raw))" in probe
    assert "digests_raw = _snapshot_regular(args.weight_digests)" in probe
    assert 'shard_digests={digest_record["shard"]["filename"]: digest_record["shard"]["sha256"]}' in probe
    assert "layer1_keys_from_normalized" in probe and '"one_row_block_m64_keys_vs_legacy_lanes"' in probe
    record = json.loads(DIGESTS.read_text())
    assert record["artifact_kind"] == "gate_d_chunk0_probe_weight_digests" and len(record["tensors"]) == 19


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
