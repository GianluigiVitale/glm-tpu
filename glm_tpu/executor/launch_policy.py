"""Which checkout may launch, and the commit it stages (DESIGN 6.6; replaces the frozen-source guard).

The policy is built only from the site file's ``[launch]`` table
(:class:`glm_tpu.config.site.LaunchPolicy`); no environment variable or flag overrides it:

* ``allowed_branches``: ``fnmatch`` patterns (case-sensitive) matched against
  ``git symbolic-ref --short HEAD``; a detached HEAD is always refused;
* ``expected_origin``: when set, ``git remote get-url origin`` must equal it after normalization
  (``git@host:owner/repo.git``, ``ssh://git@host/owner/repo`` and ``https://host/owner/repo.git``
  name one origin; credentials and default ports are dropped, the host is lower-cased). A local
  path (``file://`` or neither a URL nor the scp form, e.g. ``host/owner/repo``) is another kind
  of origin: it is resolved against the checkout, as git resolves it, and never equals a remote
  URL. Neither URL is ever printed;
* ``require_clean``: ``git status --porcelain --untracked-files=normal`` is empty (untracked files
  refuse, ignored files do not);
* ``require_pushed``: ``git ls-remote --exit-code origin refs/heads/<branch>`` names exactly HEAD.

What replaces the old guard's intent (no unreviewed numerics drift reaches the fleet): the
equivalence gates as a merge requirement and the TPU replay before merge; on the hosts the
worker re-hashes every staged file against the manifest of ``git archive <pin>`` and binds the
pin in its start marker, cleanup authentication and the fleet summary.

:func:`resolve_repo` names the checkout that is archived: ``--repo``, else the site's
``paths.repo``, else the checkout that contains this package -- in every case only if
``git rev-parse --show-toplevel`` returns exactly that directory (a wheel install is refused).
:func:`require_controller_checkout` then requires it to be the controller's own checkout: the
controller's code and the remote helper texts it sends to the hosts are read from its own
checkout, so only when that is the staged one does :func:`source_identity`'s clean-and-pinned
proof cover them (DESIGN 6.5).

Standard library only; importing this module never imports JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import os
from pathlib import Path
import re
import subprocess

from glm_tpu.config.site import LaunchPolicy, SiteConfig

__all__ = [
    "LaunchPolicy",
    "LaunchPolicyError",
    "SourceIdentity",
    "normalize_origin",
    "package_checkout",
    "require_controller_checkout",
    "resolve_repo",
    "source_identity",
]

_PIN = re.compile(r"[0-9a-f]{40}")
_SCHEME = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*)://(.*)", re.DOTALL)
_SCP = re.compile(r"(?:[^@/:]+@)?([^@/:]+):(?!//)(.*)", re.DOTALL)
_DEFAULT_PORTS = {"ssh": "22", "git+ssh": "22", "ssh+git": "22", "https": "443", "http": "80", "git": "9418"}
LS_REMOTE_TIMEOUT = 120


class LaunchPolicyError(ValueError):
    """The checkout may not launch under the site's ``[launch]`` policy (the message names the rule)."""


@dataclass(frozen=True)
class SourceIdentity:
    pin: str  # the 40-hex commit that is archived and staged
    branch: str  # the checked-out branch that matched ``allowed_branches``
    origin: str | None  # normalized origin URL (None when neither compared nor needed); never printed


def _origin(url: str, base: Path | str | None) -> tuple[bool, str]:
    """``(local, spelling)`` of an origin URL (see :func:`normalize_origin`)."""
    text = url.strip()
    scheme_match = _SCHEME.fullmatch(text)
    if scheme_match is not None:
        scheme, rest = scheme_match.group(1).lower(), scheme_match.group(2)
        if scheme == "file":
            return True, os.path.normpath("/" + rest.lstrip("/"))
        authority, _, path = rest.partition("/")
        host = authority.rpartition("@")[2]
        port = ""
        if host.startswith("["):
            end = host.find("]")
            host, port = (host[: end + 1], host[end + 2 :]) if end != -1 else (host, "")
        elif ":" in host:
            host, _, port = host.partition(":")
        if port and port != _DEFAULT_PORTS.get(scheme):
            host = f"{host}:{port}"
    else:
        scp = _SCP.fullmatch(text)
        if scp is None or "/" in scp.group(1):  # a local path: git resolves a relative one in the checkout
            return True, os.path.normpath(os.path.join(os.path.abspath(base if base is not None else "."), text))
        host, path = scp.group(1), scp.group(2)
    path = path.strip("/")
    if path.endswith(".git"):
        path = path[: -len(".git")]
    return False, f"{host.lower()}/{path.rstrip('/')}"


def normalize_origin(url: str, base: Path | str | None = None) -> str:
    """One spelling per origin: ``<host>[:<non-default port>]/<path>`` without credentials, trailing
    ``/`` or ``.git``, host lower-cased (path case kept). A local path (``file://``, or neither a URL
    nor the scp form) is normalized as an absolute path, a relative one resolved against ``base``
    (the checkout, where git resolves it; default: the current directory), so that
    ``host/owner/repo`` or ``./host/owner/repo`` never takes the spelling of a remote origin."""
    return _origin(url, base)[1]


def _git(repo: Path, *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchPolicyError(f"git {args[0]} could not run in the checkout ({type(exc).__name__})") from None


def _output(repo: Path, *args: str) -> str:
    result = _git(repo, *args)
    if result.returncode != 0:
        raise LaunchPolicyError(f"git {' '.join(args)} failed in the checkout (exit {result.returncode})")
    return result.stdout


def source_identity(repo: Path, policy: LaunchPolicy | None) -> SourceIdentity:
    """Apply the site's launch policy to ``repo`` and return the commit to stage (in this order:
    clean, HEAD, branch, origin, pushed -- the first failing rule refuses)."""
    if not isinstance(policy, LaunchPolicy):
        raise LaunchPolicyError("no launch policy: the controller needs the site file's [launch] table")
    repo = Path(repo)
    if policy.require_clean and _output(repo, "status", "--porcelain", "--untracked-files=normal").strip():
        raise LaunchPolicyError(
            "source must be clean (launch.require_clean): commit or remove every change, untracked files included"
        )
    pin = _output(repo, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if _PIN.fullmatch(pin) is None:
        raise LaunchPolicyError("HEAD is not a 40-hex commit")
    result = _git(repo, "symbolic-ref", "--quiet", "--short", "HEAD")
    branch = result.stdout.strip()
    if result.returncode != 0 or not branch:
        raise LaunchPolicyError("detached HEAD refused: check out a branch allowed by launch.allowed_branches")
    if not any(fnmatch.fnmatchcase(branch, pattern) for pattern in policy.allowed_branches):
        raise LaunchPolicyError(
            f"branch {branch!r} is not allowed by launch.allowed_branches "
            "(use a release branch or main, or add its pattern to the site file)"
        )
    origin = None
    if policy.expected_origin is not None or policy.require_pushed:
        result = _git(repo, "remote", "get-url", "origin")
        if result.returncode != 0 or not result.stdout.strip():
            raise LaunchPolicyError("the checkout has no 'origin' remote (launch.expected_origin / require_pushed)")
        local, origin = _origin(result.stdout, repo)
        # A local origin never matches a remote expected origin (nor the reverse), whatever the
        # spellings: a local repository at <checkout>/<host>/<path> must not pass for <host>/<path>.
        if policy.expected_origin is not None and (local, origin) != _origin(policy.expected_origin, repo):
            raise LaunchPolicyError("origin differs from launch.expected_origin")
    if policy.require_pushed:
        ref = "refs/heads/" + branch
        result = _git(repo, "ls-remote", "--exit-code", "origin", ref, timeout=LS_REMOTE_TIMEOUT)
        heads = [
            fields[0]
            for fields in (line.split("\t") for line in result.stdout.splitlines())
            if len(fields) == 2 and fields[1] == ref
        ]
        if result.returncode != 0 or heads != [pin]:
            raise LaunchPolicyError(
                f"source must be pushed before deployment: origin {ref} does not name HEAD (launch.require_pushed)"
            )
    return SourceIdentity(pin=pin, branch=branch, origin=origin)


def package_checkout() -> Path:
    """The directory that contains the ``glm_tpu`` package (a checkout's top level, or site-packages)."""
    return Path(__file__).resolve().parents[2]


def require_controller_checkout(repo: Path, *controller: Path) -> None:
    """Refuse to stage ``repo`` unless it is the checkout every ``controller`` path names (the
    launcher's checkout and :func:`package_checkout`, the home of the ``glm_tpu`` package whose
    remote helper texts are sent). Another checkout would be proven clean and pinned while the
    controller's own code and helper texts stayed unverified."""
    if not controller or {Path(p).resolve() for p in controller} != {Path(repo).resolve()}:
        raise LaunchPolicyError(
            "--repo / paths.repo must name this controller's own checkout: the controller's "
            "code and the remote helpers it sends are read from it, and only the staged "
            "checkout is proven clean and pinned (run the controller from the checkout to stage)"
        )


def _toplevel(path: Path) -> Path | None:
    if not path.is_dir():
        return None
    result = _git(path, "rev-parse", "--show-toplevel")
    text = result.stdout.strip()
    return Path(text).resolve() if result.returncode == 0 and text else None


def resolve_repo(site: SiteConfig, cli_repo: Path | str | None = None, *, default: Path | None = None) -> Path:
    """The checkout to archive: ``--repo`` (``cli_repo``), else ``paths.repo``, else ``default`` (the
    caller's own checkout; :func:`package_checkout` when None). Refused unless it is exactly the top
    level of a git checkout. A launcher also requires it to be its own checkout
    (:func:`require_controller_checkout`)."""
    if cli_repo not in (None, ""):
        candidate, source = Path(os.path.expanduser(str(cli_repo))), "--repo"
    elif site.paths.repo is not None:
        candidate, source = site.paths.repo, "paths.repo"
    else:
        candidate, source = Path(default) if default is not None else package_checkout(), None
    candidate = candidate.resolve()
    if _toplevel(candidate) != candidate:
        if source is None:
            raise LaunchPolicyError(
                "launching requires a git checkout of glm-tpu (the staged source is `git archive <pin>`); pass --repo"
            )
        raise LaunchPolicyError(f"{source} must name the top level of a git checkout")
    return candidate
