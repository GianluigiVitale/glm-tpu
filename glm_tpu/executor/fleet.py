"""The exact remote command strings the controller sends over SSH (DESIGN 6.5, D6; H12).

A remote helper (``glm_tpu.executor.remote``) travels as the *text of its file* in
``<interpreter> -c <text> <one JSON argument>``; nothing is templated into the text, so a run
directory, a pin or a path with spaces, quotes or non-ASCII characters reaches the helper exactly
(JSON, ASCII-escaped, one shell word). The text is read from the controller's own package with
``importlib.resources`` and its SHA-256 is recorded in the run's ``helpers.json``.

Interpreters are the 181c013e ones: the stdlib helpers run on ``fleet.helper_python`` (the hosts'
system ``python3``), ``stage_bundle`` on ``fleet.worker_python`` (``tarfile`` ``data`` filter);
the CPU preflight is the worker module itself on ``fleet.worker_python``.

The same builders serve the controller and the loopback test that executes their strings
(``tests/executor/test_remote_loopback.py``). Standard library only; never imports JAX.
"""

from __future__ import annotations

from hashlib import sha256
from importlib import resources
import json
from pathlib import Path
import shlex
from typing import Any

from glm_tpu.engine.resident_protocol import REMOTE_HELPER_PACKAGE, WORKER_ENV_FLAG
from glm_tpu.executor.remote import HELPERS

# The remote shell receives the whole command as one argv string: Linux caps one argument at
# MAX_ARG_STRLEN (128 KiB). Stay well below it.
MAX_COMMAND_BYTES = 96 << 10
WORKER_INTERPRETER_HELPERS = frozenset({"stage_bundle"})


def helper_text(name: str) -> str:
    """The program text of one remote helper, from this package."""
    if name not in HELPERS:
        raise ValueError(f"unknown remote helper {name!r}")
    return resources.files(REMOTE_HELPER_PACKAGE).joinpath(name + ".py").read_text(encoding="utf-8")


def helper_sha256(name: str) -> str:
    return sha256(helper_text(name).encode("utf-8")).hexdigest()


def interpreter_role(name: str) -> str:
    """Which site interpreter runs a helper: ``fleet.worker_python`` or ``fleet.helper_python``."""
    if name not in HELPERS:
        raise ValueError(f"unknown remote helper {name!r}")
    return "fleet.worker_python" if name in WORKER_INTERPRETER_HELPERS else "fleet.helper_python"


def helpers_record() -> dict[str, Any]:
    """``helpers.json``: each helper's SHA-256 and interpreter role (no host values)."""
    return dict(schema="glm_tpu_remote_helpers_v1",
                helpers={name: dict(sha256=helper_sha256(name), interpreter=interpreter_role(name))
                         for name in HELPERS})


def encode_arguments(args: dict[str, Any]) -> str:
    """The helper's one JSON argument: sorted, compact, ASCII (non-ASCII paths as \\u escapes)."""
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def remote_command(helper: str, args: dict[str, Any], *, interpreter: str) -> str:
    """``<interpreter> -c <helper text> <json>`` as one shell string (what follows the SSH prefix)."""
    command = shlex.join([interpreter, "-c", helper_text(helper), encode_arguments(args)])
    if len(command.encode("utf-8")) > MAX_COMMAND_BYTES:
        raise ValueError(f"remote command for {helper} exceeds {MAX_COMMAND_BYTES} bytes")
    return command


def command(fleet: Any, helper: str, args: dict[str, Any]) -> str:
    """:func:`remote_command` on the site interpreter the helper runs on (``fleet``: the site's
    :class:`~glm_tpu.config.site.FleetConfig`)."""
    role = interpreter_role(helper)
    interpreter = fleet.worker_python if role == "fleet.worker_python" else fleet.helper_python
    return remote_command(helper, args, interpreter=interpreter)


def preflight_command(fleet: Any, root: Path, worker_command: list[str]) -> str:
    """The CPU-only worker preflight: ``cd <root>/source && env JAX_PLATFORMS=cpu <flag>=1
    PYTHONPATH=<root>/source:<pythonpath> <worker_command> --preflight-only``."""
    source = str(Path(root) / "source")
    return "cd " + shlex.quote(source) + " && " + shlex.join([
        "env", "JAX_PLATFORMS=cpu", WORKER_ENV_FLAG + "=1",
        "PYTHONPATH=" + ":".join([source, *fleet.worker_pythonpath]), *worker_command, "--preflight-only"])
