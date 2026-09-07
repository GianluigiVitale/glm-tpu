"""CPU-only behavior tests for the actual wrapper's pinned controller source."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "scripts/greenfield/run_short_decoder_ws32.sh"
SPEC = importlib.util.spec_from_file_location(
    "ws32_sealing_checkout_sealer", ROOT / "scripts/greenfield/seal_short_decoder_ws32.py"
)
assert SPEC and SPEC.loader
SEALER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEALER)


@pytest.mark.parametrize("mode", ["acquire", "numerical"])
@pytest.mark.parametrize("iterations", [0, 1, 10, 255, 257])
def test_e0_refuses_wrong_window_before_any_artifact_read(mode, iterations):
    # Only these fields exist: accessing an artifact/extra argument before the
    # guard would raise AttributeError instead of the intended early refusal.
    with pytest.raises(SystemExit, match="exactly 256 timed decode iterations"):
        SEALER._validate(SimpleNamespace(
            context_label="256k_e0", mode=mode, iterations=iterations,
            dsa_adjudication_record=None,
        ))


@pytest.mark.parametrize("context,iterations", [
    ("256k_e0", 256), ("128k_d0_0", 10), ("128k_d1_0", 10), ("8k", 10), ("2k", 10),
])
@pytest.mark.parametrize("mode", ["acquire", "numerical"])
def test_valid_window_reaches_existing_validation(monkeypatch, context, iterations, mode):
    def reached(*args, **kwargs):
        raise RuntimeError("existing tag validation reached")
    monkeypatch.setattr(SEALER, "_validate_run_tag", reached)
    with pytest.raises(RuntimeError, match="existing tag validation reached"):
        SEALER._validate(SimpleNamespace(
            context_label=context, iterations=iterations, mode=mode,
            dsa_adjudication_record=None, tag="test", prefill_chunk=2048,
            context_capacity=262656, host_main_rope_table=1,
        ))


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(root), *args], check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "worker"
    root.mkdir()
    (root / "probe.py").write_text(
        'import json,os\nprint(json.dumps({"value":"pinned",'
        '"cwd":os.getcwd(),"platform":os.environ["JAX_PLATFORMS"]}))\n'
    )
    (root / "bench").mkdir()
    for name in ("engine.py", "provenance.py"):
        (root / "bench" / name).write_text("VALUE = 1\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=test", "-c", "user.email=test@test",
         "commit", "-qm", "pinned")
    return root, _git(root, "rev-parse", "HEAD")


def _wrapper_block() -> str:
    source = WRAPPER.read_text()
    block = source.split("# BEGIN PINNED SEAL CHECKOUT\n", 1)[1].split(
        "# END PINNED SEAL CHECKOUT", 1
    )[0]
    # The test interpreter replaces only the deployment interpreter path; all
    # shell source-selection and invocation behavior is the production block.
    return block.replace("/home/gianl/vllm-env/bin/python", shlex.quote(sys.executable))


def _invoke(root: Path, pin: str, run: Path, body: str) -> subprocess.CompletedProcess:
    run.mkdir(exist_ok=True)
    setup = f"""set -euo pipefail
WORKTREE={shlex.quote(str(root))}
RECOVERY_PIN={pin}
RUN_DIR={shlex.quote(str(run))}
DSA_ADJUDICATION_CLI="--dsa-adjudication-record $WORKTREE/docs/artifacts/record.json --dsa-adjudication-sha256 abc"
LATER_EVENT_ALARM_CLI="--later-event-alarm-profile $WORKTREE/docs/artifacts/profile.json --later-event-alarm-lessons-pin $RECOVERY_PIN"
"""
    return subprocess.run(
        ["/bin/bash", "-c", setup + _wrapper_block() + "\n" + body],
        cwd=root, text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(root), "JAX_PLATFORMS": "tpu"},
        timeout=30,
    )


def test_pinned_checkout_ignores_development_edits_and_rebases_artifacts(tmp_path):
    root, pin = _repository(tmp_path)
    result = _invoke(root, pin, tmp_path / "run", """
seal_python -c 'from pathlib import Path; import sys; (Path(sys.argv[1])/"probe.py").write_text("raise RuntimeError(\\"mutable source imported\\")\\n")' "$WORKTREE"
seal_python -m probe
[[ $SEAL_DSA_ADJUDICATION_CLI == "--dsa-adjudication-record $SEAL_ROOT/docs/artifacts/record.json --dsa-adjudication-sha256 abc" ]]
[[ $SEAL_LATER_EVENT_ALARM_CLI == "--later-event-alarm-profile $SEAL_ROOT/docs/artifacts/profile.json --later-event-alarm-lessons-pin $RECOVERY_PIN" ]]
[[ $DSA_ADJUDICATION_CLI == *"$WORKTREE/docs/artifacts/record.json"* ]]
[[ $RECOVERY_TOOL == "$SEAL_ROOT/scripts/greenfield/recover_short_decoder_ws32_acquisition.py" ]]
""")
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["value"] == "pinned" and observed["platform"] == "cpu"
    assert Path(observed["cwd"]).parent == tmp_path / "run"
    assert _git(Path(observed["cwd"]), "rev-parse", "HEAD") == pin
    assert _git(Path(observed["cwd"]), "status", "--porcelain") == ""


def test_each_attempt_gets_a_new_checkout_without_mutating_previous(tmp_path):
    root, pin = _repository(tmp_path)
    first = _invoke(root, pin, tmp_path / "run", "seal_python -m probe")
    second = _invoke(root, pin, tmp_path / "run", "seal_python -m probe")
    assert first.returncode == second.returncode == 0
    first_root = Path(json.loads(first.stdout)["cwd"])
    second_root = Path(json.loads(second.stdout)["cwd"])
    assert first_root != second_root
    assert _git(first_root, "rev-parse", "HEAD") == pin
    assert _git(second_root, "rev-parse", "HEAD") == pin


def test_changed_sealing_checkout_refuses_before_next_python_command(tmp_path):
    root, pin = _repository(tmp_path)
    result = _invoke(root, pin, tmp_path / "run", """
seal_python -c 'from pathlib import Path; Path("probe.py").write_text("print(42)\\n")'
seal_python -m probe
""")
    assert result.returncode != 0
    assert "sealing checkout changed" in result.stderr
    assert "42" not in result.stdout


@pytest.mark.parametrize("name", ["engine.py", "provenance.py"])
def test_legacy_extractor_dependencies_are_part_of_clean_surface(tmp_path, name):
    root, _ = _repository(tmp_path)
    SEALER._require_clean_worktree(root)
    (root / "bench" / name).write_text("VALUE = 2\n")
    with pytest.raises(SystemExit, match="unmodified enforcement surface"):
        SEALER._require_clean_worktree(root)


def test_wrapper_routes_every_controller_python_call_through_pinned_root():
    source = WRAPPER.read_text()
    assert 'PYTHONPATH="$WORKTREE" /home/gianl/vllm-env/bin/python' not in source
    assert source.index("# BEGIN PINNED SEAL CHECKOUT") < source.index("trap on_exit EXIT")
    for command in ("validate", "publish-db", "rollback-db"):
        assert f'"$SEAL_ROOT/scripts/greenfield/seal_short_decoder_ws32.py" {command}' in source
    assert 'seal_python \\\n  -m glm_tpu.greenfield.validation.ws32_evidence' in source
    assert '${SEAL_DSA_ADJUDICATION_CLI:+$SEAL_DSA_ADJUDICATION_CLI}' in source
    assert '${SEAL_LATER_EVENT_ALARM_CLI:+$SEAL_LATER_EVENT_ALARM_CLI}' in source
