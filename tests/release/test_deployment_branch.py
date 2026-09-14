"""Main/release source routing; all subprocess/network actions mocked or parsed."""
import subprocess
import shlex

import pytest

from scripts.greenfield import launch_ws32_native_benchmark as launch


@pytest.mark.parametrize("branch", ["main", "release/production-20260914", "rewrite/topology-first-decode"])
def test_supported_branch_reaches_actual_sync_command(branch):
    command = launch.sync_command("a" * 40, branch=branch)
    subprocess.run(["bash", "-n", "-c", command], check=True)
    assert "git fetch -q origin " + branch in command
    assert "git reset" not in command


@pytest.mark.parametrize("branch", ["--upload-pack=bad", "main;touch x", "release/../main",
                                     "release/a//b", "release/a.lock", "release/a/", "other", "",
                                     "release/a/.hidden", "release/a.lock/b", "release/a./b"])
def test_invalid_branch_refuses_before_shell(branch):
    with pytest.raises(ValueError):
        launch.sync_command("a" * 40, branch=branch)


@pytest.mark.parametrize("problem", [None, "origin", "local_branch", "remote_pin", "remote_ref", "missing"])
def test_preflight_checks_actual_origin_branch_and_remote_head(monkeypatch, problem):
    from scripts.greenfield import run_short_decoder_ws32 as original
    from scripts.greenfield import ws32_native_benchmark_programs as programs
    pin = "a" * 40
    calls = []
    monkeypatch.setattr(original, "_require_clean_code", lambda p: calls.append(("clean", p)))
    monkeypatch.setattr(programs, "require_source", lambda p: calls.append(("model", p)))

    def read(args, **kwargs):
        if args[1:3] == ["remote", "get-url"]:
            return "git@github.com:vllm-project/tpu-inference.git" if problem == "origin" else "git@github.com:GianluigiVitale/glm-tpu.git"
        if args[1] == "branch":
            return "rewrite/topology-first-decode" if problem == "local_branch" else "main"
        assert args == ["git", "ls-remote", "origin", "refs/heads/main"]
        if problem == "missing":
            return ""
        return ("b" * 40 if problem == "remote_pin" else pin) + "\t" + (
            "refs/heads/other" if problem == "remote_ref" else "refs/heads/main") + "\n"

    monkeypatch.setattr(launch.subprocess, "check_output", read)
    if problem:
        with pytest.raises(ValueError):
            launch.source_preflight(pin)
    else:
        launch.source_preflight(pin)
        assert calls == [("clean", pin), ("model", launch.REPO)]


@pytest.mark.parametrize("problem", [None, "origin", "fetch_pin", "dirty"])
def test_worker_sync_refuses_before_checkout(tmp_path, monkeypatch, problem):
    """Execute the shell guard with a fake git function; no fetch or checkout occurs."""
    monkeypatch.setattr(launch, "REPO", tmp_path)
    pin = "a" * 40
    origin = "git@github.com:GianluigiVitale/glm-tpu.git" if problem != "origin" else "https://example.invalid/wrong.git"
    fetched = pin if problem != "fetch_pin" else "b" * 40
    dirty = " M model.py" if problem == "dirty" else ""
    stub = '''git() {
      case "$*" in
        "status --porcelain") printf '%s' DIRTY ;;
        "remote get-url origin") printf '%s' ORIGIN ;;
        "fetch -q origin main") echo MOCK_FETCH ;;
        "rev-parse FETCH_HEAD") printf '%s' FETCHED ;;
        "rev-parse HEAD") printf '%s' PIN ;;
        "checkout -q --detach "*) echo MOCK_CHECKOUT ;;
        *) return 90 ;;
      esac
    }
    HOSTNAME=test-w-1
    '''
    for key, value in (("DIRTY", dirty), ("ORIGIN", origin), ("FETCHED", fetched), ("PIN", pin)):
        stub = stub.replace(key, shlex.quote(value))
    result = subprocess.run(["bash", "-c", stub + launch.sync_command(pin)], capture_output=True, text=True)
    assert (result.returncode == 0) == (problem is None)
    assert ("MOCK_CHECKOUT" in result.stdout) == (problem is None)
    if problem in ("origin", "dirty"):
        assert "MOCK_FETCH" not in result.stdout


@pytest.mark.parametrize("branch", ["main", "release/production-20260914", "rewrite/topology-first-decode"])
def test_attach_remains_bound_to_original_branch(branch):
    row = dict(tag="tag", code_hash="a" * 40, reviewed_branch=branch)
    launch.validate_attach(row, tag="tag", pin="a" * 40, branch=branch)
    for changes in ({"tag": "other"}, {"code_hash": "b" * 40},
                    {"reviewed_branch": "release/other"}, {"reviewed_branch": None}):
        with pytest.raises(ValueError):
            launch.validate_attach(dict(row, **changes), tag="tag", pin="a" * 40, branch=branch)


def test_legacy_attach_requires_explicit_research_branch():
    row = dict(tag="tag", code_hash="a" * 40)
    launch.validate_attach(row, tag="tag", pin="a" * 40, branch="rewrite/topology-first-decode")
    with pytest.raises(ValueError):
        launch.validate_attach(row, tag="tag", pin="a" * 40, branch="main")
