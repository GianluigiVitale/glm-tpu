"""The site launch policy on real git checkouts (a local bare repository is the origin; no network)."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from glm_tpu.config.site import LaunchPolicy
from glm_tpu.executor import launch_policy
from glm_tpu.executor.launch_policy import LaunchPolicyError, normalize_origin, resolve_repo, source_identity
from tests.fixtures.site import example_site

GIT_ENV = dict(
    GIT_CONFIG_GLOBAL="/dev/null",
    GIT_CONFIG_NOSYSTEM="1",
    GIT_AUTHOR_NAME="fixture",
    GIT_AUTHOR_EMAIL="fixture@example.invalid",
    GIT_COMMITTER_NAME="fixture",
    GIT_COMMITTER_EMAIL="fixture@example.invalid",
)


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    for name, value in GIT_ENV.items():
        monkeypatch.setenv(name, value)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    """A clone of a local bare origin on ``main``, one pushed commit, clean."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True, capture_output=True)
    git(work, "symbolic-ref", "HEAD", "refs/heads/main")
    (work / ".gitignore").write_text("ignored.txt\n")
    (work / "README.md").write_text("fixture\n")
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", "fixture")
    git(work, "push", "-q", "origin", "main")
    return work


POLICY = LaunchPolicy()


def test_default_policy_admits_a_clean_pushed_main(checkout):
    identity = source_identity(checkout, POLICY)
    assert identity.pin == git(checkout, "rev-parse", "HEAD") and identity.branch == "main"
    assert identity.origin == normalize_origin(str(checkout.parent / "origin.git"))


def test_branch_patterns_are_fnmatch_and_a_detached_head_is_refused(checkout):
    git(checkout, "checkout", "-q", "-b", "release/2026-09")
    git(checkout, "push", "-q", "origin", "release/2026-09")
    assert source_identity(checkout, POLICY).branch == "release/2026-09"
    git(checkout, "checkout", "-q", "-b", "refactor/public")
    git(checkout, "push", "-q", "origin", "refactor/public")
    with pytest.raises(LaunchPolicyError, match=r"'refactor/public' is not allowed by launch.allowed_branches"):
        source_identity(checkout, POLICY)
    assert source_identity(checkout, LaunchPolicy(allowed_branches=("refactor/*",))).branch == "refactor/public"
    with pytest.raises(LaunchPolicyError, match="not allowed"):  # case-sensitive
        source_identity(checkout, LaunchPolicy(allowed_branches=("Refactor/*",)))
    git(checkout, "checkout", "-q", "--detach")
    with pytest.raises(LaunchPolicyError, match="detached HEAD refused"):
        source_identity(checkout, LaunchPolicy(allowed_branches=("*",)))


@pytest.mark.parametrize("change", ["modified", "untracked", "staged"])
def test_a_dirty_checkout_is_refused_unless_the_policy_allows_it(checkout, change):
    if change == "modified":
        (checkout / "README.md").write_text("changed\n")
    elif change == "untracked":
        (checkout / "new.txt").write_text("new\n")
    else:
        (checkout / "new.txt").write_text("new\n")
        git(checkout, "add", "new.txt")
    with pytest.raises(LaunchPolicyError, match="source must be clean"):
        source_identity(checkout, POLICY)
    assert source_identity(checkout, LaunchPolicy(require_clean=False)).pin == git(checkout, "rev-parse", "HEAD")


def test_ignored_files_do_not_make_a_checkout_dirty(checkout):
    (checkout / "ignored.txt").write_text("local\n")
    assert source_identity(checkout, POLICY).branch == "main"


def test_an_unpushed_commit_is_refused_unless_the_policy_allows_it(checkout):
    (checkout / "README.md").write_text("ahead\n")
    git(checkout, "commit", "-q", "-am", "ahead")
    with pytest.raises(LaunchPolicyError, match="source must be pushed before deployment"):
        source_identity(checkout, POLICY)
    assert source_identity(checkout, LaunchPolicy(require_pushed=False)).pin == git(checkout, "rev-parse", "HEAD")
    git(checkout, "checkout", "-q", "-b", "release/never-pushed")
    with pytest.raises(LaunchPolicyError, match="origin refs/heads/release/never-pushed does not name HEAD"):
        source_identity(checkout, POLICY)


def test_a_missing_origin_is_refused_only_when_the_policy_needs_it(checkout):
    git(checkout, "remote", "remove", "origin")
    with pytest.raises(LaunchPolicyError, match="no 'origin' remote"):
        source_identity(checkout, POLICY)
    identity = source_identity(checkout, LaunchPolicy(require_pushed=False))
    assert identity.origin is None


def test_expected_origin_is_compared_after_normalization_and_never_printed(checkout):
    origin = str(checkout.parent / "origin.git")
    assert source_identity(checkout, LaunchPolicy(expected_origin=origin + "/")).branch == "main"
    assert source_identity(checkout, LaunchPolicy(expected_origin="file://" + origin)).branch == "main"
    other = "https://token-value@example.invalid/owner/other.git"
    with pytest.raises(LaunchPolicyError) as refused:
        source_identity(checkout, LaunchPolicy(expected_origin=other))
    message = str(refused.value)
    assert message == "origin differs from launch.expected_origin"
    assert "token-value" not in message and origin not in message


def test_no_policy_is_a_refusal(checkout):
    with pytest.raises(LaunchPolicyError, match=r"\[launch\] table"):
        source_identity(checkout, None)


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:Example/repo.git",
        "ssh://git@github.com/Example/repo",
        "ssh://git@github.com:22/Example/repo.git",
        "https://github.com/Example/repo.git",
        "https://GitHub.com/Example/repo/",
        "https://user:secret-token@github.com:443/Example/repo.git",
        "git+ssh://git@github.com/Example/repo.git",
    ],
)
def test_https_and_ssh_spellings_of_one_origin_are_equal(url):
    assert normalize_origin(url) == "github.com/Example/repo"
    assert "secret-token" not in normalize_origin(url)


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:Example/other.git",
        "git@github.com:example/repo.git",  # path case is kept
        "https://gitlab.example.invalid/Example/repo.git",
        "ssh://git@github.com:2222/Example/repo.git",  # a non-default port is another service
        "/srv/git/Example/repo.git",
    ],
)
def test_different_origins_stay_different(url):
    assert normalize_origin(url) != "github.com/Example/repo"


def test_local_paths_normalize_as_paths():
    assert normalize_origin("/srv/git/repo.git/") == normalize_origin("file:///srv/git/repo.git") == "/srv/git/repo.git"


@pytest.mark.parametrize(
    "url", ["example.invalid/owner/repo", "./example.invalid/owner/repo", "example.invalid/owner/repo/"]
)
def test_a_relative_local_path_never_takes_the_spelling_of_a_remote_origin(tmp_path, url):
    # git reads a URL without a scheme and without the scp form as a local path, relative to the
    # checkout: it is resolved there, so it can never equal the <host>/<path> of a remote URL.
    assert normalize_origin(url, tmp_path) == str(tmp_path / "example.invalid" / "owner" / "repo")
    assert normalize_origin(url, tmp_path) != normalize_origin("git@example.invalid:owner/repo.git")
    assert normalize_origin(url, tmp_path).startswith("/")


def test_a_local_repository_inside_the_checkout_does_not_pass_for_the_expected_remote(checkout):
    # The origin is a local repository at <checkout>/example.invalid/owner/repo, hidden from
    # `git status` by .git/info/exclude, so require_clean and require_pushed (ls-remote against
    # it) both pass; its URL "example.invalid/owner/repo" must still not match the remote origin.
    local = checkout / "example.invalid" / "owner" / "repo"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(local)], check=True)
    (checkout / ".git" / "info" / "exclude").write_text("example.invalid/\n")
    git(checkout, "remote", "set-url", "origin", "example.invalid/owner/repo")
    git(checkout, "push", "-q", "origin", "main")
    identity = source_identity(checkout, POLICY)  # clean and pushed: the local repository is real
    for expected in (
        "git@example.invalid:owner/repo.git",
        "https://example.invalid/owner/repo",
        "ssh://git@example.invalid/owner/repo.git",
    ):
        with pytest.raises(LaunchPolicyError, match=r"origin differs from launch.expected_origin"):
            source_identity(checkout, LaunchPolicy(expected_origin=expected))
    assert identity.origin == str(local)
    # the same local repository, spelled as a path, is that origin
    assert source_identity(checkout, LaunchPolicy(expected_origin=str(local) + "/")).branch == "main"


def test_a_local_origin_and_a_url_without_a_host_are_different_origins(checkout):
    plain = checkout.parent / "plain"  # a local origin whose name has no .git suffix
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(plain)], check=True)
    git(checkout, "remote", "set-url", "origin", str(plain))
    git(checkout, "push", "-q", "origin", "main")
    origin = str(plain)
    assert normalize_origin("ssh://" + origin) == normalize_origin(origin)  # one spelling, two kinds
    assert source_identity(checkout, LaunchPolicy(expected_origin=origin)).branch == "main"
    for expected in ("ssh://" + origin, "https://" + origin):
        with pytest.raises(LaunchPolicyError, match=r"origin differs from launch.expected_origin"):
            source_identity(checkout, LaunchPolicy(expected_origin=expected))


def test_a_relative_origin_is_the_local_repository_it_names(checkout):
    git(checkout, "remote", "set-url", "origin", "../origin.git")
    origin = str(checkout.parent / "origin.git")
    identity = source_identity(checkout, LaunchPolicy(expected_origin=origin))  # ls-remote resolves it too
    assert identity.origin == origin


# ----------------------------------------------------------------------------- checkout resolution
def test_resolve_repo_prefers_the_flag_then_the_site_then_the_callers_checkout(tmp_path, checkout):
    site = example_site(tmp_path)
    assert resolve_repo(site, checkout) == checkout.resolve()
    assert resolve_repo(site, None, default=checkout) == checkout.resolve()
    configured = example_site(tmp_path, paths=dict(repo=str(checkout)))
    assert resolve_repo(configured, None, default=tmp_path) == checkout.resolve()
    other = tmp_path / "other"
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    assert resolve_repo(configured, other) == other.resolve()


def test_resolve_repo_refuses_anything_but_a_checkout_top_level(tmp_path, checkout):
    site = example_site(tmp_path)
    (checkout / "sub").mkdir()
    with pytest.raises(LaunchPolicyError, match="--repo must name the top level of a git checkout"):
        resolve_repo(site, checkout / "sub")
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(LaunchPolicyError, match=r"paths.repo must name the top level"):
        resolve_repo(example_site(tmp_path, paths=dict(repo=str(plain))))
    with pytest.raises(LaunchPolicyError, match=r"requires a git checkout of glm-tpu.*pass --repo"):
        resolve_repo(site, None, default=plain)


def test_the_package_checkout_is_this_repository():
    repo = Path(__file__).resolve().parents[2]
    assert launch_policy.package_checkout() == repo
    assert (repo / "glm_tpu" / "executor" / "launch_policy.py").is_file()


def test_the_launcher_refuses_a_disallowed_branch_before_any_lease_or_host(tmp_path, checkout, monkeypatch):
    import os

    from glm_tpu.engine import request
    from glm_tpu.utils.json_utils import canonical
    from glm_tpu.executor import multihost_executor as launch
    from tests.fixtures.site import example_mapping, write_example_site

    git(checkout, "checkout", "-q", "-b", "feature/x")
    runs = tmp_path / "runs"
    runs.mkdir()
    site = write_example_site(tmp_path / "site.toml", example_mapping(tmp_path, paths=dict(run_root=str(runs))))
    path = tmp_path / "input.json"
    path.write_bytes(canonical(request.from_token_ids([7], request_id="fixture", max_new_tokens=2)))
    path.chmod(0o600)
    monkeypatch.setattr(launch, "REPO", checkout)
    monkeypatch.setattr(launch_policy, "package_checkout", lambda: checkout)  # as if run from that checkout
    monkeypatch.setattr(launch, "ssh_commands", lambda fleet: pytest.fail("no host may be contacted"))
    previous = os.umask(0o077)
    try:
        with pytest.raises(LaunchPolicyError, match="'feature/x' is not allowed"):
            launch.main(["--request", str(path), "--site", str(site)])
    finally:
        os.umask(previous)
    assert list(runs.iterdir()) == []


# ----------------------------------------------------------------------------- the controller's own checkout
def test_only_the_controllers_own_checkout_may_be_staged(tmp_path, checkout):
    launch_policy.require_controller_checkout(checkout, checkout, checkout)
    launch_policy.require_controller_checkout(checkout / ".", checkout.parent / checkout.name)  # spelling-free
    other = tmp_path / "other"
    other.mkdir()
    for repo, controller in [(other, (checkout, checkout)), (checkout, (checkout, other)), (checkout, ())]:
        with pytest.raises(LaunchPolicyError, match="this controller's own checkout"):
            launch_policy.require_controller_checkout(repo, *controller)


@pytest.mark.parametrize("via", ["--repo", "paths.repo"])
def test_the_launcher_refuses_another_checkout_before_any_git_run_directory_or_host(
    tmp_path, checkout, monkeypatch, via
):
    # --repo / paths.repo may name only the controller's own checkout: the launcher and the
    # glm_tpu package (whose remote helper texts are sent) come from it, not from ``checkout``.
    import os

    from glm_tpu.engine import request
    from glm_tpu.utils.json_utils import canonical
    from glm_tpu.executor import multihost_executor as launch
    from tests.fixtures.site import example_mapping, write_example_site

    runs = tmp_path / "runs"
    runs.mkdir()
    paths = dict(run_root=str(runs), **(dict(repo=str(checkout)) if via == "paths.repo" else {}))
    site = write_example_site(tmp_path / "site.toml", example_mapping(tmp_path, paths=paths))
    path = tmp_path / "input.json"
    path.write_bytes(canonical(request.from_token_ids([7], request_id="fixture", max_new_tokens=2)))
    path.chmod(0o600)
    monkeypatch.setattr(launch_policy, "source_identity", lambda *a: pytest.fail("the policy must not run"))
    monkeypatch.setattr(launch, "ssh_commands", lambda fleet: pytest.fail("no host may be contacted"))
    argv = ["--request", str(path), "--site", str(site)] + (["--repo", str(checkout)] if via == "--repo" else [])
    previous = os.umask(0o077)
    try:
        with pytest.raises(LaunchPolicyError, match="must name this controller's own checkout"):
            launch.main(argv)
    finally:
        os.umask(previous)
    assert list(runs.iterdir()) == []
