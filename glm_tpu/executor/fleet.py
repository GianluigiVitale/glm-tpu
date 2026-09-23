"""The exact remote command strings the controller sends over SSH (DESIGN 6.5, D6; H12).

A remote helper (``glm_tpu.executor.remote``) travels as the *text of its file* in
``<interpreter> -c <text> <one JSON argument>``; nothing is templated into the text, so a run
directory, a pin or a path with spaces, quotes or non-ASCII characters reaches the helper exactly
(JSON, ASCII-escaped, one shell word).

A controller reads the texts **once**: right after the launch policy has pinned its checkout it
takes a :class:`HelperTexts` snapshot (:meth:`HelperTexts.pinned`: the helper blobs of the pinned
commit, each required equal to this package's file), records its SHA-256s in the run's
``helpers.json`` and builds every remote command of the run from it. An edit of the checkout
during a run (a later refactor stage, a branch switch) therefore cannot change what ``cleanup``,
``fetch`` or ``idle_after`` send, and every text sent is the pinned one. Without a snapshot
(``helpers=None``) the builders read this package's files at the call (tests and tools).

Interpreters are the 181c013e ones: the stdlib helpers run on ``fleet.helper_python`` (the hosts'
system ``python3``), ``stage_bundle`` on ``fleet.worker_python`` (``tarfile`` ``data`` filter);
the CPU preflight is the worker module itself on ``fleet.worker_python``.

The same builders serve the controller and the loopback test that executes their strings
(``tests/executor/test_remote_loopback.py``). Standard library only; never imports JAX.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from importlib import resources
import json
from pathlib import Path
import shlex
import subprocess
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from glm_tpu.engine.resident_protocol import REMOTE_HELPER_PACKAGE, WORKER_ENV_FLAG, module_path
from glm_tpu.executor.remote import HELPERS

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

# The remote shell receives the whole command as one argv string: Linux caps one argument at
# MAX_ARG_STRLEN (128 KiB). Stay well below it.
MAX_COMMAND_BYTES = 96 << 10
WORKER_INTERPRETER_HELPERS = frozenset({"stage_bundle"})
GIT_TIMEOUT_SECONDS = 60


def _known(name: str) -> str:
    if name not in HELPERS:
        raise ValueError(f"unknown remote helper {name!r}")
    return name


def _package_file(name: str) -> Traversable:
    return resources.files(REMOTE_HELPER_PACKAGE).joinpath(name + ".py")


def _package_text(name: str) -> str:
    return _package_file(name).read_text(encoding="utf-8")


def helper_path(name: str) -> str:
    """A helper's path in a checkout (``glm_tpu/executor/remote/<name>.py``; the git blob path)."""
    return module_path(REMOTE_HELPER_PACKAGE + "." + _known(name))


@dataclass(frozen=True)
class HelperTexts:
    """One read-only snapshot of every remote helper's program text (``texts``: name -> text)."""

    texts: Mapping[str, str]

    def __post_init__(self) -> None:
        texts = dict(self.texts)
        if set(texts) != set(HELPERS) or not all(isinstance(text, str) for text in texts.values()):
            raise ValueError("a helper snapshot holds the text of exactly every remote helper")
        object.__setattr__(self, "texts", MappingProxyType(texts))

    @classmethod
    def from_package(cls) -> HelperTexts:
        """This package's helper files, read now."""
        return cls({name: _package_text(name) for name in HELPERS})

    @classmethod
    def pinned(cls, repo: Path | str, pin: str) -> HelperTexts:
        """The helper blobs of commit ``pin`` in checkout ``repo`` (``git cat-file blob
        <pin>:<helper_path>``), each required equal to this package's file: the controller runs from
        that checkout (``launch_policy.require_controller_checkout``), and an uncommitted helper edit
        (possible under ``launch.require_clean = false``) is refused instead of being sent."""
        texts = {}
        for name in HELPERS:
            try:
                result = subprocess.run(["git", "cat-file", "blob", f"{pin}:{helper_path(name)}"], cwd=repo,
                                        capture_output=True, timeout=GIT_TIMEOUT_SECONDS, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ValueError(f"remote helper {name}: git cat-file could not run ({type(exc).__name__})") from None
            if result.returncode != 0:
                raise ValueError(f"remote helper {name} is not in the pinned commit")
            try:
                text = result.stdout.decode("utf-8")
            except UnicodeDecodeError:
                raise ValueError(f"remote helper {name} in the pinned commit is not UTF-8") from None
            if text != _package_text(name):
                raise ValueError(f"remote helper {name} differs from the pinned commit: the controller's "
                                 "checkout must hold the committed helper files")
            texts[name] = text
        return cls(texts)

    def text(self, name: str) -> str:
        return self.texts[_known(name)]

    def sha256(self, name: str) -> str:
        return sha256(self.text(name).encode("utf-8")).hexdigest()

    def record(self) -> dict[str, Any]:
        """``helpers.json``: each helper's SHA-256 and interpreter role (no host values)."""
        return dict(schema="glm_tpu_remote_helpers_v1",
                    helpers={name: dict(sha256=self.sha256(name), interpreter=interpreter_role(name))
                             for name in HELPERS})


def _snapshot(helpers: HelperTexts | None) -> HelperTexts:
    return HelperTexts.from_package() if helpers is None else helpers


def helper_text(name: str) -> str:
    """The program text of one remote helper, read from this package now (tests and tools; a
    controller sends the texts of its :class:`HelperTexts` snapshot)."""
    return _package_text(_known(name))


def interpreter_role(name: str) -> str:
    """Which site interpreter runs a helper: ``fleet.worker_python`` or ``fleet.helper_python``."""
    return "fleet.worker_python" if _known(name) in WORKER_INTERPRETER_HELPERS else "fleet.helper_python"


def helpers_record(helpers: HelperTexts | None = None) -> dict[str, Any]:
    """``helpers.json`` of ``helpers`` (default: this package's files, read now)."""
    return _snapshot(helpers).record()


def encode_arguments(args: dict[str, Any]) -> str:
    """The helper's one JSON argument: sorted, compact, ASCII (non-ASCII paths as \\u escapes)."""
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def remote_command(helper: str, args: dict[str, Any], *, interpreter: str,
                   helpers: HelperTexts | None = None) -> str:
    """``<interpreter> -c <helper text> <json>`` as one shell string (what follows the SSH prefix);
    the text comes from ``helpers`` (a controller's run snapshot), else from this package now."""
    command = shlex.join([interpreter, "-c", _snapshot(helpers).text(helper), encode_arguments(args)])
    if len(command.encode("utf-8")) > MAX_COMMAND_BYTES:
        raise ValueError(f"remote command for {helper} exceeds {MAX_COMMAND_BYTES} bytes")
    return command


def command(fleet: Any, helper: str, args: dict[str, Any], *, helpers: HelperTexts | None = None) -> str:
    """:func:`remote_command` on the site interpreter the helper runs on (``fleet``: the site's
    :class:`~glm_tpu.config.site.FleetConfig`)."""
    role = interpreter_role(helper)
    interpreter = fleet.worker_python if role == "fleet.worker_python" else fleet.helper_python
    return remote_command(helper, args, interpreter=interpreter, helpers=helpers)


def preflight_command(fleet: Any, root: Path, worker_command: list[str]) -> str:
    """The CPU-only worker preflight: ``cd <root>/source && env JAX_PLATFORMS=cpu <flag>=1
    PYTHONPATH=<root>/source:<pythonpath> <worker_command> --preflight-only``."""
    source = str(Path(root) / "source")
    return "cd " + shlex.quote(source) + " && " + shlex.join([
        "env", "JAX_PLATFORMS=cpu", WORKER_ENV_FLAG + "=1",
        "PYTHONPATH=" + ":".join([source, *fleet.worker_pythonpath]), *worker_command, "--preflight-only"])
