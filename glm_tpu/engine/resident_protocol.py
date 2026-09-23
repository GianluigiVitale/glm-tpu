"""The resident protocol and process contracts: one definition of every name and byte string the
controller, the worker, the CLI and serving exchange (DESIGN 4.9, 6.4, Appendix A).

Process contracts (``/proc/<pid>/cmdline`` checks, ``python -m`` targets, cleanup authentication):
``CONTROLLER_MODULE``, ``WORKER_MODULE``, ``PACK_WORKER_MODULE`` and the handshake flags. Until the
S3 moves they hold today's names (``scripts.release.*``, ``GLM_OPTIMIZED_REQUEST``,
``GLM_OWNER_PACK``); S3 changes them atomically with the moves, and
``tests/engine/test_resident_protocol.py`` proves every constant names a real module.

Run-directory protocol (frozen, G9): ``inbox/NNNN.json`` requests, ``inbox/stop.json`` =
``{"stop":true}``, ``resident-ready.json`` ``{"sequence": n}``, result directories
``resident-NNNN/`` (``root`` itself for sequence 0), ``resident-measurement.json``, the worker's
stdin command line ``{"request":...,"sequence":n}`` in canonical wire bytes, and the stdout markers
``RUN ``, ``PRIVATE_INPUT ``, ``RESIDENT_RESULT ``.

Standard library only; importing this module never imports JAX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------- process contracts
CONTROLLER_MODULE = "scripts.release.launch_ws32_optimized_request"
WORKER_MODULE = "scripts.release.ws32_optimized_worker"
PACK_WORKER_MODULE = "scripts.release.ws32_pack_worker"
REMOTE_HELPER_PACKAGE = "glm_tpu.executor.remote"
WORKER_ENV_FLAG = "GLM_OPTIMIZED_REQUEST"
PACK_WORKER_ENV_FLAG = "GLM_OWNER_PACK"

# ----------------------------------------------------------------------------- run directory
INBOX_DIR = "inbox"
STOP_FILE = "stop.json"
STOP_COMMAND = b'{"stop":true}\n'
READY_FILE = "resident-ready.json"
MEASUREMENT_FILE = "resident-measurement.json"
CONTROLLER_IDENTITY_FILE = "controller_identity.json"
CONTROLLER_TERMINAL_FILE = "controller_terminal.json"
SUMMARY_FILE = "summary.json"
HELPERS_FILE = "helpers.json"
PRODUCER_LOCK = "benchmark-producer.lock"
STDOUT_RUN = "RUN "
STDOUT_PRIVATE_INPUT = "PRIVATE_INPUT "
STDOUT_RESIDENT_RESULT = "RESIDENT_RESULT "


def inbox_name(sequence: int) -> str:
    """``inbox/<name>`` of the request with this resident sequence number (``0001.json``, ...)."""
    return f"{sequence:04d}.json"


def result_dir(root: Path, sequence: int) -> Path:
    """Where the records of a resident sequence live: the run directory itself for sequence 0."""
    return root if sequence == 0 else root / f"resident-{sequence:04d}"


def runner_file(rank: int) -> str:
    return f"runner.rank{rank}.json"


def worker_started_file(rank: int) -> str:
    return f"worker_started.rank{rank}.json"


def encode_command(sequence: int, request: Any) -> bytes:
    """The worker stdin line of one resident request (canonical wire bytes: UTF-8, sorted keys)."""
    from glm_tpu.user_request import canonical

    return canonical(dict(sequence=sequence, request=request)) + b"\n"


def module_path(module: str) -> str:
    """The source-tree path of a dotted module (``a.b.c`` -> ``a/b/c.py``; manifest membership)."""
    return module.replace(".", "/") + ".py"


def source_root(module_file: str | Path, module: str) -> Path:
    """The source root a module was loaded from (the staged ``source/`` directory on a host):
    ``module_file`` with the ``module_path(module)`` suffix removed; refuses a mismatch."""
    path = Path(module_file).resolve()
    root = path.parents[module.count(".")]
    if path.relative_to(root).as_posix() != module_path(module):
        raise ValueError(f"{path} is not the file of module {module}")
    return root
