#!/usr/bin/env python3
"""Emit the persistence-only certificate for the projection HLO acquirer."""

from __future__ import annotations

import _frozen_importlib
import _frozen_importlib_external
import json
import os
import stat
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any


def _require_isolated_import_boundary() -> None:
    flags = sys.flags
    if not (
        flags.isolated == 1
        and flags.no_site == 1
        and flags.ignore_environment == 1
        and flags.no_user_site == 1
        and flags.safe_path
    ):
        raise RuntimeError("projection HLO source analyzer requires python -I -S")
    expected = (
        _frozen_importlib.BuiltinImporter,
        _frozen_importlib.FrozenImporter,
        _frozen_importlib_external.PathFinder,
    )
    if tuple(sys.meta_path) != expected:
        raise RuntimeError(
            "projection HLO source analyzer rejects noncanonical import hooks"
        )


def _reject_preloaded_glm_tpu() -> None:
    preloaded = sorted(
        name for name in sys.modules if name == "glm_tpu" or name.startswith("glm_tpu.")
    )
    if preloaded:
        raise RuntimeError("projection HLO source analyzer rejects preloaded glm_tpu")


_require_isolated_import_boundary()
_reject_preloaded_glm_tpu()
ROOT = Path(__file__).resolve(strict=True).parents[2]
ROOT_TEXT = str(ROOT)
ISOLATED_STDLIB_PATH = tuple(sys.path)
EXPECTED_IMPORT_PATH = (ROOT_TEXT, *ISOLATED_STDLIB_PATH)
sys.path[:] = EXPECTED_IMPORT_PATH
BRANCH = "tooling/gate-d-compensated-pp16-numerical"
SOURCE_PATH = (
    "scripts/greenfield/analyze_gate_d_projection_contraction_hlo_acquisition_source.py"
)
ACQUIRER_PATH = "scripts/greenfield/acquire_gate_d_projection_contraction_pp16_hlo.py"
VALIDATOR_PATH = (
    "glm_tpu/greenfield/validation/"
    "gate_d_projection_contraction_hlo_acquisition_source.py"
)
SOURCE_AUTHORITY_PATH = "docs/artifacts/gate-d-projection-contraction-pp16-source.json"
SOURCE_AUTHORITY_SHA256 = (
    "5744eee0ef2cf35a4566cc0de165be1338160daaae3f4dd133554b2aa8280e9f"
)
TOPOLOGY_PATH = "docs/artifacts/gate-d-runtime-locality-authority.json"
TOPOLOGY_SHA256 = "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_PROTOCOL_FROM_USER": "0",
    "GIT_SSH_COMMAND": "/bin/false",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/bin:/bin",
}


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _snapshot(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe source authority file: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 8 << 20):
            raw.extend(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            len(raw) != before.st_size
            or (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"source authority changed: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _git(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(ROOT), *arguments],
        env=GIT_ENVIRONMENT,
        stderr=subprocess.PIPE,
    )


def _committed_sources() -> tuple[str, dict[str, str]]:
    if _git("for-each-ref", "--format=%(refname)", "refs/replace"):
        raise RuntimeError("projection HLO source repository has replacement refs")
    pin = _git("rev-parse", "HEAD").decode("ascii").strip()
    branch = _git("branch", "--show-current").decode("ascii").strip()
    if branch != BRANCH:
        raise RuntimeError("projection HLO source branch drifted")
    identities: dict[str, str] = {}
    for relative in (SOURCE_PATH, ACQUIRER_PATH, VALIDATOR_PATH):
        raw = _snapshot(ROOT / relative)
        if raw != _git("show", f"{pin}:{relative}"):
            raise RuntimeError(f"projection HLO source is not committed: {relative}")
        identities[relative] = sha256(raw).hexdigest()
    return pin, identities


def _load_validator_audit(validator_raw: bytes) -> Any:
    expected_validator = (ROOT / VALIDATOR_PATH).resolve(strict=True)
    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__file__": str(expected_validator),
        "__name__": "_gate_d_projection_contraction_hlo_acquisition_source",
        "__package__": None,
    }
    code = compile(
        validator_raw,
        str(expected_validator),
        "exec",
        flags=0,
        dont_inherit=True,
        optimize=0,
    )
    # The executed bytes were snapshotted and matched to the committed validator blob.
    exec(code, namespace, namespace)  # noqa: S102
    audit = namespace.get("audit_projection_contraction_hlo_acquisition_source")
    if (
        not callable(audit)
        or getattr(audit, "__module__", None) != namespace["__name__"]
        or getattr(getattr(audit, "__code__", None), "co_filename", None)
        != str(expected_validator)
    ):
        raise RuntimeError("projection HLO source validator callable drifted")
    return audit


def analyze() -> dict[str, Any]:
    if os.environ.get("GLM_GATE_D_PROJECTION_HLO_SOURCE") != "1":
        raise RuntimeError("projection HLO source analysis is default-off")
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get(
        "JAX_PLATFORM_NAME"
    ) not in (None, "cpu"):
        raise RuntimeError("projection HLO source analysis must be CPU-pinned")
    _require_isolated_import_boundary()
    _reject_preloaded_glm_tpu()
    pin, sources = _committed_sources()
    source_authority_raw = _snapshot(ROOT / SOURCE_AUTHORITY_PATH)
    topology_raw = _snapshot(ROOT / TOPOLOGY_PATH)
    if sha256(source_authority_raw).hexdigest() != SOURCE_AUTHORITY_SHA256:
        raise RuntimeError("projection source authority bytes drifted")
    if sha256(topology_raw).hexdigest() != TOPOLOGY_SHA256:
        raise RuntimeError("projection topology authority bytes drifted")
    if tuple(sys.path) != EXPECTED_IMPORT_PATH:
        raise RuntimeError("projection HLO source import root drifted")
    validator_raw = _snapshot(ROOT / VALIDATOR_PATH)
    if sha256(validator_raw).hexdigest() != sources[VALIDATOR_PATH]:
        raise RuntimeError("projection HLO source validator import drifted")
    validator_audit = _load_validator_audit(validator_raw)
    audit = validator_audit(
        _snapshot(ROOT / ACQUIRER_PATH),
        source_authority=json.loads(source_authority_raw),
        source_authority_sha256=SOURCE_AUTHORITY_SHA256,
        topology=json.loads(topology_raw),
        topology_sha256=TOPOLOGY_SHA256,
    )
    return {
        "artifact_kind": "gate_d_projection_contraction_hlo_acquisition_source",
        "authorization": {
            "compile_only_hlo_acquisition": False,
            "full_dsa_or_8k": False,
            "orchestration_or_install": False,
            "persistence_only": True,
            "tpu_execution": False,
        },
        "classification": (
            "PROJECTION_HLO_ACQUIRER_SOURCE_ACCEPTED;ORCHESTRATION_UNPROVEN;"
            "TPU_COMPILE_UNAUTHORIZED;TPU_EXECUTION_UNAUTHORIZED;GATE_D_OPEN"
        ),
        "code_hash": pin,
        "exact_next": (
            "After review and durable persistence, design a separate append-only wrapper, "
            "publisher, immutable installer and launcher; do not compile or execute."
        ),
        "gate_d_closed": False,
        "performance_claim": False,
        "schema_version": 1,
        "source_audit": audit,
        "source_sha256s": sources,
        "tpu_compile_or_execution_performed": False,
    }


def main() -> int:
    if sys.argv[1:]:
        raise RuntimeError("projection HLO source analyzer accepts no arguments")
    sys.stdout.buffer.write(_canonical(analyze()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
