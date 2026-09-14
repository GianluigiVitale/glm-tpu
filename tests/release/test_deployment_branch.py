"""Main/release source routing; all subprocess/network actions mocked or parsed."""

import subprocess
import shlex

import pytest

from scripts.greenfield import launch_ws32_native_benchmark as launch


@pytest.mark.parametrize(
    "branch", ["main", "release/production-20260914", "rewrite/topology-first-decode"]
)
def test_supported_branch_reaches_actual_sync_command(branch):
    command = launch.sync_command("a" * 40, branch=branch)
    subprocess.run(["bash", "-n", "-c", command], check=True)
    assert "git fetch -q origin " + branch in command
    assert "git reset" not in command


@pytest.mark.parametrize(
    "branch",
    [
        "--upload-pack=bad",
        "main;touch x",
        "release/../main",
        "release/a//b",
        "release/a.lock",
        "release/a/",
        "other",
        "",
        "release/a/.hidden",
        "release/a.lock/b",
        "release/a./b",
    ],
)
def test_invalid_branch_refuses_before_shell(branch):
    with pytest.raises(ValueError):
        launch.sync_command("a" * 40, branch=branch)


@pytest.mark.parametrize(
    "problem", [None, "origin", "local_branch", "remote_pin", "remote_ref", "missing"]
)
@pytest.mark.parametrize("detached", [False, True])
def test_preflight_checks_actual_origin_branch_and_remote_head(
    monkeypatch, problem, detached
):
    from scripts.greenfield import run_short_decoder_ws32 as original
    from scripts.greenfield import ws32_native_benchmark_programs as programs

    pin = "a" * 40
    calls = []
    monkeypatch.setattr(
        original, "_require_clean_code", lambda p: calls.append(("clean", p))
    )
    monkeypatch.setattr(
        programs, "require_source", lambda p: calls.append(("model", p))
    )

    def read(args, **kwargs):
        if args[1:3] == ["remote", "get-url"]:
            return (
                "git@github.com:vllm-project/tpu-inference.git"
                if problem == "origin"
                else "git@github.com:GianluigiVitale/glm-tpu.git"
            )
        if args[1] == "branch":
            return (
                "rewrite/topology-first-decode"
                if problem == "local_branch"
                else ("" if detached else "main")
            )
        assert args == ["git", "ls-remote", "origin", "refs/heads/main"]
        if problem == "missing":
            return ""
        return (
            ("b" * 40 if problem == "remote_pin" else pin)
            + "\t"
            + ("refs/heads/other" if problem == "remote_ref" else "refs/heads/main")
            + "\n"
        )

    monkeypatch.setattr(launch.subprocess, "check_output", read)
    if problem:
        with pytest.raises(ValueError):
            launch.source_preflight(pin)
    else:
        launch.source_preflight(pin)
        assert calls == [("clean", pin), ("model", launch.REPO)]


@pytest.mark.parametrize("failure", ["dirty", "wrong_pin", "model_source"])
def test_detached_deployment_keeps_original_source_refusals(monkeypatch, failure):
    from scripts.greenfield import run_short_decoder_ws32 as original
    from scripts.greenfield import ws32_native_benchmark_programs as programs

    def clean(pin):
        assert pin == "a" * 40
        if failure in ("dirty", "wrong_pin"):
            raise RuntimeError("original source guard refusal")

    def model(repo):
        assert repo == launch.REPO
        raise RuntimeError("original model guard refusal")

    monkeypatch.setattr(original, "_require_clean_code", clean)
    monkeypatch.setattr(programs, "require_source", model)
    monkeypatch.setattr(
        launch.subprocess,
        "check_output",
        lambda *a, **k: pytest.fail(
            "branch/network handling must not run after original source refusal"
        ),
    )
    with pytest.raises(RuntimeError, match="original .* guard refusal"):
        launch.source_preflight("a" * 40)


def test_detached_controller_preserves_branch_in_another_real_worktree(
    tmp_path, monkeypatch
):
    """Real Git worktree identities; model/site and remote service are fixtures."""
    from scripts.greenfield import run_short_decoder_ws32 as original
    from scripts.greenfield import ws32_native_benchmark_programs as programs

    repo, deployed = tmp_path / "release", tmp_path / "canonical"
    subprocess.run(["git", "init", "-q", "-b", "release/test", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        check=True,
    )
    read = subprocess.check_output
    pin = read(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "worktree",
            "add",
            "-q",
            "--detach",
            str(deployed),
            pin,
        ],
        check=True,
    )
    monkeypatch.setattr(launch, "REPO", deployed)
    monkeypatch.setattr(original, "REPO", deployed)
    monkeypatch.setattr(original, "EXPECTED_WORKTREE", deployed)
    monkeypatch.setattr(programs, "require_source", lambda path: None)

    def remote_fixture(args, **kwargs):
        if args[:3] == ["git", "remote", "get-url"]:
            return "git@github.com:GianluigiVitale/glm-tpu.git\n"
        if args[:2] == ["git", "ls-remote"]:
            assert args == ["git", "ls-remote", "origin", "refs/heads/release/test"]
            return f"{pin}\trefs/heads/release/test\n"
        return read(args, **kwargs)

    monkeypatch.setattr(launch.subprocess, "check_output", remote_fixture)
    launch.source_preflight(pin, branch="release/test")
    assert (
        read(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
        == "release/test"
    )
    assert (
        read(["git", "branch", "--show-current"], cwd=deployed, text=True).strip() == ""
    )
    assert read(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == pin


@pytest.mark.parametrize("problem", [None, "origin", "fetch_pin", "dirty"])
def test_worker_sync_refuses_before_checkout(tmp_path, monkeypatch, problem):
    """Execute the shell guard with a fake git function; no fetch or checkout occurs."""
    monkeypatch.setattr(launch, "REPO", tmp_path)
    pin = "a" * 40
    origin = (
        "git@github.com:GianluigiVitale/glm-tpu.git"
        if problem != "origin"
        else "https://example.invalid/wrong.git"
    )
    fetched = pin if problem != "fetch_pin" else "b" * 40
    dirty = " M model.py" if problem == "dirty" else ""
    stub = """git() {
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
    """
    for key, value in (
        ("DIRTY", dirty),
        ("ORIGIN", origin),
        ("FETCHED", fetched),
        ("PIN", pin),
    ):
        stub = stub.replace(key, shlex.quote(value))
    result = subprocess.run(
        ["bash", "-c", stub + launch.sync_command(pin)], capture_output=True, text=True
    )
    assert (result.returncode == 0) == (problem is None)
    assert ("MOCK_CHECKOUT" in result.stdout) == (problem is None)
    if problem in ("origin", "dirty"):
        assert "MOCK_FETCH" not in result.stdout


@pytest.mark.parametrize(
    "branch", ["main", "release/production-20260914", "rewrite/topology-first-decode"]
)
def test_attach_remains_bound_to_original_branch(branch):
    row = dict(tag="tag", code_hash="a" * 40, reviewed_branch=branch)
    launch.validate_attach(row, tag="tag", pin="a" * 40, branch=branch)
    for changes in (
        {"tag": "other"},
        {"code_hash": "b" * 40},
        {"reviewed_branch": "release/other"},
        {"reviewed_branch": None},
    ):
        with pytest.raises(ValueError):
            launch.validate_attach(
                dict(row, **changes), tag="tag", pin="a" * 40, branch=branch
            )


def test_legacy_attach_requires_explicit_research_branch():
    row = dict(tag="tag", code_hash="a" * 40)
    launch.validate_attach(
        row, tag="tag", pin="a" * 40, branch="rewrite/topology-first-decode"
    )
    with pytest.raises(ValueError):
        launch.validate_attach(row, tag="tag", pin="a" * 40, branch="main")
