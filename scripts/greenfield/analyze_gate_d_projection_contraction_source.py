#!/usr/bin/env python3
"""Emit the source-only certificate for the PP16 projection discriminator.

The CLI requires a clean interpreter with no preloaded ``glm_tpu`` modules.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any


def _reject_preloaded_glm_tpu() -> None:
    preloaded = sorted(
        name for name in sys.modules if name == "glm_tpu" or name.startswith("glm_tpu.")
    )
    if preloaded:
        raise RuntimeError(
            "projection source analyzer rejects preloaded glm_tpu modules"
        )


_reject_preloaded_glm_tpu()
ROOT = Path(__file__).resolve(strict=True).parents[2]
ROOT_TEXT = str(ROOT)
sys.path[:] = [entry for entry in sys.path if entry != ROOT_TEXT]
sys.path.insert(0, ROOT_TEXT)

SOURCE_PATH = "scripts/greenfield/analyze_gate_d_projection_contraction_source.py"
BUILDER_PATH = "glm_tpu/greenfield/benchmarking/gate_d_projection_contraction_pp16.py"
VALIDATOR_PATH = "glm_tpu/greenfield/validation/gate_d_projection_contraction_source.py"
PREDECESSOR = (
    ROOT / "docs/artifacts/gate-d-projection-arithmetic-frontier-analysis.json"
)
PREDECESSOR_SHA256 = "4a6be0f33f221df46f384cd2047da1aa664ae4c45763e68f0f1df05f8644e20c"
TOPOLOGY = ROOT / "docs/artifacts/gate-d-runtime-locality-authority.json"
TOPOLOGY_SHA256 = "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
EXPECTED_BRANCH = "tooling/gate-d-compensated-pp16-numerical"
DIRECT_DEPENDENCY_SHA256S = {
    "glm_tpu/__init__.py": (
        "902e500f466fdaf18a637710b7d1c54d15514e3dd35d838dade74c03f6a07f06"
    ),
    "glm_tpu/greenfield/__init__.py": (
        "d09e3bc884c4044669eb5791b90b97b642f5e5a3b14c266867d9996830c30ea7"
    ),
    "glm_tpu/greenfield/benchmarking/__init__.py": (
        "7af76bb93d634aced6adc3940fd3f6c7a80cc702a8e113ab9de01c4a397e5173"
    ),
    "glm_tpu/greenfield/benchmarking/gate_d_compensated_capsule.py": (
        "0f1930c079bd7244452e84dca6d0bbf9da077ea133c685f0f908760379e5313d"
    ),
    "glm_tpu/greenfield/errors.py": (
        "f7b8a079ba16f6114714124598d03ae871a2c21d365de4417b934f00cddaae64"
    ),
    "glm_tpu/greenfield/kernels/__init__.py": (
        "1345c01c8adeb22549f2b8bb42c46cc26edb760900bf80815991dfb58170f378"
    ),
    "glm_tpu/greenfield/kernels/reference/__init__.py": (
        "1f9284813df1dd38f3e8d1acfb9417da372e06c7d5336c8262348dad2f631404"
    ),
    "glm_tpu/greenfield/kernels/reference/dsa.py": (
        "c4b451ab7ca2bf79b7cc7b148a7b1b146996051891c1c82f1952d8ab9ef53fac"
    ),
    "glm_tpu/greenfield/kernels/reference/linear.py": (
        "c3d679dab63d1975f0fbfd1ceda7bd8e198621face135151f5fe05cedd07a0f6"
    ),
    "glm_tpu/greenfield/kernels/reference/rotary.py": (
        "cb17440803331de962328b409212e9647fc736590c3b525a4a2d43daa46b05b3"
    ),
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
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 << 20):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(raw) != before.st_size or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source authority changed: {path}")
        return raw
    finally:
        os.close(descriptor)


def _git(*arguments: str) -> bytes:
    return subprocess.check_output(
        ["/usr/bin/git", "-C", str(ROOT), *arguments],
        env={
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
        },
    )


def _committed_sources() -> tuple[str, dict[str, str], dict[str, str]]:
    pin = _git("rev-parse", "HEAD").decode("ascii").strip()
    branch = _git("branch", "--show-current").decode("ascii").strip()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError("projection source branch drifted")
    identities: dict[str, str] = {}
    for relative in (SOURCE_PATH, BUILDER_PATH, VALIDATOR_PATH):
        raw = _snapshot(ROOT / relative)
        if raw != _git("show", f"{pin}:{relative}"):
            raise RuntimeError(f"projection source is not committed: {relative}")
        identities[relative] = sha256(raw).hexdigest()
    dependencies: dict[str, str] = {}
    for relative, expected in DIRECT_DEPENDENCY_SHA256S.items():
        raw = _snapshot(ROOT / relative)
        digest = sha256(raw).hexdigest()
        if raw != _git("show", f"{pin}:{relative}") or digest != expected:
            raise RuntimeError(f"projection direct dependency drifted: {relative}")
        dependencies[relative] = digest
    return pin, identities, dependencies


def analyze() -> dict[str, Any]:
    if os.environ.get("GLM_GATE_D_PROJECTION_CONTRACTION_SOURCE") != "1":
        raise RuntimeError("projection source analysis is default-off")
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get(
        "JAX_PLATFORM_NAME"
    ) not in (None, "cpu"):
        raise RuntimeError("projection source analysis must be CPU-pinned")
    _reject_preloaded_glm_tpu()
    pin, sources, dependencies = _committed_sources()
    predecessor_raw = _snapshot(PREDECESSOR)
    topology_raw = _snapshot(TOPOLOGY)
    if sha256(predecessor_raw).hexdigest() != PREDECESSOR_SHA256:
        raise RuntimeError("projection predecessor bytes drifted")
    if sha256(topology_raw).hexdigest() != TOPOLOGY_SHA256:
        raise RuntimeError("projection topology bytes drifted")
    if not sys.path or sys.path[0] != ROOT_TEXT:
        raise RuntimeError("projection source import root drifted")
    from glm_tpu.greenfield.validation import (
        gate_d_projection_contraction_source as validator,
    )

    validator_path = Path(validator.__file__).resolve(strict=True)
    expected_validator_path = (ROOT / VALIDATOR_PATH).resolve(strict=True)
    if (
        validator_path != expected_validator_path
        or sha256(_snapshot(validator_path)).hexdigest() != sources[VALIDATOR_PATH]
    ):
        raise RuntimeError("projection source validator import drifted")

    audit = validator.audit_projection_contraction_pp16_source(
        _snapshot(ROOT / BUILDER_PATH),
        dependency_sha256s=dependencies,
        predecessor=json.loads(predecessor_raw),
        topology=json.loads(topology_raw),
    )
    return {
        "artifact_kind": "gate_d_projection_contraction_pp16_source",
        "authorization": {
            "full_dsa_or_8k": False,
            "hlo_acquisition": False,
            "persistence_only": True,
            "tpu_compile": False,
            "tpu_execution": False,
        },
        "classification": (
            "PROJECTION_ONLY_PP16_SOURCE_ACCEPTED;COMPILE_UNPROVEN;"
            "TPU_CAUSALITY_UNPROVEN;GATE_D_OPEN"
        ),
        "code_hash": pin,
        "direct_dependency_sha256s": dependencies,
        "exact_next": (
            "After review and durable persistence, separately authorize one compile-only "
            "HLO acquisition; do not execute the compiled program or full DSA/8K."
        ),
        "gate_d_closed": False,
        "performance_claim": False,
        "predecessor_sha256": PREDECESSOR_SHA256,
        "schema_version": 1,
        "source_audit": audit,
        "source_sha256s": sources,
        "topology_sha256": TOPOLOGY_SHA256,
        "tpu_compile_or_execution_performed": False,
    }


def main() -> int:
    if sys.argv[1:]:
        raise RuntimeError("projection source analyzer accepts no arguments")
    sys.stdout.buffer.write(_canonical(analyze()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
