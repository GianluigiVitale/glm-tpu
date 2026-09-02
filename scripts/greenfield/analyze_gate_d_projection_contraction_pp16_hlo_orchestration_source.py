#!/usr/bin/env python3
"""Certify projection-contraction HLO orchestration and install sources only."""

from __future__ import annotations

import _frozen_importlib
import _frozen_importlib_external
import ast
import json
import os
import re
import stat
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
BRANCH = "tooling/gate-d-compensated-pp16-numerical"
ANALYZER_PATH = (
    "scripts/greenfield/"
    "analyze_gate_d_projection_contraction_pp16_hlo_orchestration_source.py"
)
ACQUIRER_PATH = "scripts/greenfield/acquire_gate_d_projection_contraction_pp16_hlo.py"
PUBLISHER_PATH = "scripts/greenfield/publish_gate_d_projection_contraction_pp16_hlo.py"
WRAPPER_PATH = "scripts/greenfield/run_gate_d_projection_contraction_pp16_hlo.sh"
LAUNCHER_PATH = "scripts/greenfield/launch_gate_d_projection_contraction_pp16_hlo.py"
INSTALLER_PATH = (
    "scripts/greenfield/install_gate_d_projection_contraction_pp16_hlo_runtime.py"
)
MIRROR_VERIFIER_PATH = "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
AUDITED_SOURCE_PATHS = (
    ACQUIRER_PATH,
    ANALYZER_PATH,
    INSTALLER_PATH,
    LAUNCHER_PATH,
    MIRROR_VERIFIER_PATH,
    PUBLISHER_PATH,
    WRAPPER_PATH,
)
EXPECTED_SOURCE_SHA256S = {
    ACQUIRER_PATH: "c660d50eb60054c9b267840230fb69bbf7259104416dc96c7b2ee01a2a14934a",
    INSTALLER_PATH: "7125172b4a726ac00161212f8f96fed1a3b310b8f6c8724f903dee71d873ab03",
    LAUNCHER_PATH: "e6fceb3aa42f566f5fdece58d20928e4d26377d079a916769e530263c65c6772",
    MIRROR_VERIFIER_PATH: "091208165a149989f14c5c9b9d1cbe7ff20537e2c81b16319eea9603984e859b",
    PUBLISHER_PATH: "f3f20a01fd37bb82988cd77f69fa7b0a780d120568bab0db4162f42e7855bc97",
    WRAPPER_PATH: "4dd06dcb57fdb5a43d0eec0f7a6754724d7534637bda539eb36831359e1b1c70",
}
PROJECTION_SOURCE_PATH = "docs/artifacts/gate-d-projection-contraction-pp16-source.json"
PROJECTION_SOURCE_SHA256 = (
    "5744eee0ef2cf35a4566cc0de165be1338160daaae3f4dd133554b2aa8280e9f"
)
HLO_SOURCE_PATH = (
    "docs/artifacts/gate-d-projection-contraction-hlo-acquisition-source.json"
)
HLO_SOURCE_SHA256 = "a94395ef13f5cfa7de7bc221fdc33d4498637a02ffc55d3cba985834dd99b93a"
TOPOLOGY_PATH = "docs/artifacts/gate-d-runtime-locality-authority.json"
TOPOLOGY_SHA256 = "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
REFERENCE_PATHS = {
    HLO_SOURCE_PATH: HLO_SOURCE_SHA256,
    PROJECTION_SOURCE_PATH: PROJECTION_SOURCE_SHA256,
    TOPOLOGY_PATH: TOPOLOGY_SHA256,
}
EXPECTED_PAYLOADS = {
    "acquire_gate_d_projection_contraction_pp16_hlo.py": EXPECTED_SOURCE_SHA256S[
        ACQUIRER_PATH
    ],
    "launch_gate_d_projection_contraction_pp16_hlo.py": EXPECTED_SOURCE_SHA256S[
        LAUNCHER_PATH
    ],
    "publish_gate_d_projection_contraction_pp16_hlo.py": EXPECTED_SOURCE_SHA256S[
        PUBLISHER_PATH
    ],
    "verify_gate_d_same_region_git_mirror.py": EXPECTED_SOURCE_SHA256S[
        MIRROR_VERIFIER_PATH
    ],
}
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


def _require_isolated_import_boundary() -> None:
    flags = sys.flags
    if not (
        flags.isolated == 1
        and flags.no_site == 1
        and flags.ignore_environment == 1
        and flags.no_user_site == 1
        and getattr(flags, "safe_path", flags.isolated == 1)
    ):
        raise RuntimeError("projection orchestration analyzer requires python -I -S")
    expected = (
        _frozen_importlib.BuiltinImporter,
        _frozen_importlib.FrozenImporter,
        _frozen_importlib_external.PathFinder,
    )
    if tuple(sys.meta_path) != expected:
        raise RuntimeError("projection orchestration analyzer rejects import hooks")


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


def _loaded_jax_modules() -> list[str]:
    prefixes = ("jax", "jaxlib", "jax_plugins")
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    )


def _snapshot(path: Path) -> bytes:
    try:
        path.relative_to(WORKTREE)
    except ValueError as error:
        raise RuntimeError(f"orchestration source escapes worktree: {path}") from error
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError(f"unsafe orchestration source: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 8 << 20):
            raw.extend(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            len(raw) != before.st_size
            or identity(before) != identity(after)
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError(f"orchestration source changed while reading: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _git(*arguments: str) -> bytes:
    completed = subprocess.run(
        [
            "/usr/bin/git",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "diff.external=",
            "-c",
            "core.attributesFile=/dev/null",
            "-C",
            str(WORKTREE),
            *arguments,
        ],
        env=GIT_ENVIRONMENT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError(
            "projection orchestration Git authority failed: "
            f"returncode={completed.returncode} stderr={completed.stderr!r}"
        )
    return completed.stdout


def _reject_replace_refs() -> None:
    if _git("for-each-ref", "--format=%(refname)", "refs/replace"):
        raise RuntimeError("orchestration repository has forbidden replacement refs")


def _committed_sources() -> tuple[str, dict[str, bytes], dict[str, str]]:
    _reject_replace_refs()
    code_pin = _git("rev-parse", "HEAD").decode("ascii").strip()
    branch = _git("branch", "--show-current").decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", code_pin) or branch != BRANCH:
        raise RuntimeError("projection orchestration branch authority drifted")
    snapshots: dict[str, bytes] = {}
    source_sha256s: dict[str, str] = {}
    for relative in AUDITED_SOURCE_PATHS:
        raw = _snapshot(WORKTREE / relative)
        if raw != _git("show", f"{code_pin}:{relative}"):
            raise RuntimeError(
                f"projection orchestration source is not committed: {relative}"
            )
        observed = sha256(raw).hexdigest()
        expected = EXPECTED_SOURCE_SHA256S.get(relative)
        if expected is not None and observed != expected:
            raise RuntimeError(
                f"projection orchestration source hash drifted: {relative}"
            )
        snapshots[relative] = raw
        source_sha256s[relative] = observed
    return code_pin, snapshots, source_sha256s


def _assignment(tree: ast.Module, name: str) -> Any:
    values: list[Any] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ):
            values.append(ast.literal_eval(node.value))
    if len(values) != 1:
        raise RuntimeError(f"projection orchestration assignment drifted: {name}")
    return values[0]


def _audit_publisher(raw: bytes) -> dict[str, Any]:
    if sha256(raw).hexdigest() != EXPECTED_SOURCE_SHA256S[PUBLISHER_PATH]:
        raise RuntimeError("projection publisher source hash drifted")
    source = raw.decode("ascii")
    tree = ast.parse(source)
    required = (
        "gate_d_projection_contraction_pp16_hlo/",
        "projection_contraction_pp16_stage0.stablehlo.mlir",
        "projection_contraction_pp16_stage0.optimized_hlo.txt",
        "_projection_contraction_source_authority(code_pin)",
        "_hlo_acquisition_source_authority(code_pin)",
        'runner.get("compiled_executable_invocation_count") != 0',
        'runner.get("tpu_numerical_execution_performed") is not False',
        "_validate_mirror_replay(mirror_raw, code_pin)",
        "_validate_compile_host_authority(sync_raw, code_pin)",
    )
    forbidden = (
        "gate_d_compensated_pp16_hlo/",
        "gate_d_forced_round_pp16_hlo/",
        "decode_8k",
    )
    if any(item not in source for item in required) or any(
        item in source for item in forbidden
    ):
        raise RuntimeError("projection publisher claim boundary drifted")
    return {
        "ast_sha256": sha256(
            ast.dump(tree, include_attributes=False).encode("ascii")
        ).hexdigest(),
        "input_spec_count": 4,
        "output_spec_count": 3,
        "sha256": sha256(raw).hexdigest(),
    }


def _readonly_assignment(source: str, name: str) -> str:
    matches = re.findall(
        rf"^readonly {re.escape(name)}=([^\n]+)$", source, re.MULTILINE
    )
    if len(matches) != 1:
        raise RuntimeError(f"projection wrapper assignment drifted: {name}")
    return matches[0]


def _audit_wrapper(raw: bytes) -> dict[str, Any]:
    if sha256(raw).hexdigest() != EXPECTED_SOURCE_SHA256S[WRAPPER_PATH]:
        raise RuntimeError("projection wrapper source hash drifted")
    source = raw.decode("ascii")
    expected = {
        "DRIVER_SHA": EXPECTED_SOURCE_SHA256S[ACQUIRER_PATH],
        "HLO_SOURCE_SHA": HLO_SOURCE_SHA256,
        "MIRROR_VERIFIER_SHA": EXPECTED_SOURCE_SHA256S[MIRROR_VERIFIER_PATH],
        "PROJECTION_SOURCE_SHA": PROJECTION_SOURCE_SHA256,
        "PUBLISHER_SHA": EXPECTED_SOURCE_SHA256S[PUBLISHER_PATH],
        "TOPOLOGY_SHA": TOPOLOGY_SHA256,
    }
    if any(
        _readonly_assignment(source, name) != value for name, value in expected.items()
    ):
        raise RuntimeError("projection wrapper immutable pin drifted")
    required = (
        "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_HLO_ACQUIRE",
        "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_MODE",
        "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_TAG",
        "GLM_GATE_D_PROJECTION_CONTRACTION_HLO=1",
        "JAX_ENABLE_COMPILATION_CACHE=0",
        "strict_census pre",
        "strict_census post",
        "--compile-only 1",
        "GIT_AUTHORITY_VERIFIER",
        '"/usr/bin/env",',
        '"-i",',
        '"core.fsmonitor=false"',
        'git(Path("/"), "ls-remote", "--refs", origin, expected_ref)',
        "Git repository has forbidden replacement refs",
        '"--no-includes"',
        '/usr/bin/python3 -I -S -B -c "$GIT_AUTHORITY_VERIFIER"',
        "compile_host_only=1 sealed_source_archive=1",
        "HLO_ACQUIRED_UNADJUDICATED",
        "gate-d-projection-contraction-pp16-hlo-v3",
        "wrapper_fd != 10",
    )
    forbidden = (
        "decode_8k",
        "block_until_ready",
        "gate_d_compensated_pp16_hlo/",
        '"--includes=false"',
        '$(git -C "$WORKTREE"',
    )
    if (
        any(item not in source for item in required)
        or any(item in source for item in forbidden)
        or source.count('"$DRIVER_PYTHON" -I -S -B -u "$DRIVER"') != 1
        or source.count("--compile-only 1") != 1
        or source.index("strict_census pre")
        > source.index("lowering and compiling one abstract-input")
        or source.index("strict_census post")
        < source.index("lowering and compiling one abstract-input")
    ):
        raise RuntimeError("projection wrapper compile-only boundary drifted")
    return {"protected_compile_process_count": 1, "sha256": sha256(raw).hexdigest()}


def _audit_launcher(raw: bytes, wrapper_raw: bytes) -> dict[str, Any]:
    if sha256(raw).hexdigest() != EXPECTED_SOURCE_SHA256S[LAUNCHER_PATH]:
        raise RuntimeError("projection launcher source hash drifted")
    source = raw.decode("ascii")
    tree = ast.parse(source)
    expected = {
        "DRIVER_SHA256": EXPECTED_SOURCE_SHA256S[ACQUIRER_PATH],
        "MIRROR_VERIFIER_SHA256": EXPECTED_SOURCE_SHA256S[MIRROR_VERIFIER_PATH],
        "PUBLISHER_SHA256": EXPECTED_SOURCE_SHA256S[PUBLISHER_PATH],
        "WRAPPER_SHA256": EXPECTED_SOURCE_SHA256S[WRAPPER_PATH],
    }
    if any(_assignment(tree, name) != value for name, value in expected.items()):
        raise RuntimeError("projection launcher immutable pin drifted")
    required = (
        '"/opt/glm-tpu/bin/launch_gate_d_projection_contraction_pp16_hlo_v3.py"',
        '"/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-hlo-v3"',
        "os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING",
        "fcntl.fcntl(descriptor, F_ADD_SEALS, REQUIRED_SEALS)",
        'f"/proc/self/fd/{WRAPPER_FD}"',
        '"GLM_GATE_D_IMMUTABLE_LOCKS_HELD": "1"',
    )
    forbidden = ("OLDPWD", "shell=True", "os.chdir(")
    if (
        sha256(wrapper_raw).hexdigest() != expected["WRAPPER_SHA256"]
        or any(item not in source for item in required)
        or any(item in source for item in forbidden)
    ):
        raise RuntimeError("projection launcher descriptor boundary drifted")
    return {
        "ast_sha256": sha256(
            ast.dump(tree, include_attributes=False).encode("ascii")
        ).hexdigest(),
        "retained_lock_fds": [11, 12],
        "retained_wrapper_fd": 10,
        "sha256": sha256(raw).hexdigest(),
    }


def _audit_installer(raw: bytes) -> dict[str, Any]:
    if sha256(raw).hexdigest() != EXPECTED_SOURCE_SHA256S[INSTALLER_PATH]:
        raise RuntimeError("projection installer source hash drifted")
    source = raw.decode("ascii")
    tree = ast.parse(source)
    normalized_source = ast.unparse(tree)
    if _assignment(tree, "PAYLOADS") != EXPECTED_PAYLOADS:
        raise RuntimeError("projection installer payload map drifted")
    required = (
        "SOURCE_ROOT = Path('/opt/glm-tpu/gate-d-projection-contraction-hlo-install-v3')",
        "LAUNCHER_TARGET = LAUNCHER_PARENT / 'launch_gate_d_projection_contraction_pp16_hlo_v3.py'",
        "CAPSULE_TARGET = CAPSULE_PARENT / 'gate-d-projection-contraction-pp16-hlo-v3'",
        "os.geteuid() != 0",
        "dict(os.environ) != EXPECTED_ENVIRONMENT",
        "sys.argv != [str(INSTALLER_PATH)]",
        "_RENAME_NOREPLACE",
        "'launcher_invoked': False",
    )
    forbidden = ("subprocess", "execve", "google.cloud", "gs://")
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    main_source = ast.unparse(functions["main"])
    if (
        any(item not in normalized_source for item in required)
        or any(item in source for item in forbidden)
        or main_source.index("_publish_capsule")
        > main_source.index("_publish_launcher")
    ):
        raise RuntimeError("projection installer install-only boundary drifted")
    return {
        "capsule_published_before_launcher": True,
        "launcher_invocation_count": 0,
        "payload_count": len(EXPECTED_PAYLOADS),
        "sha256": sha256(raw).hexdigest(),
    }


def _predecessor_authority() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in sorted(REFERENCE_PATHS.items()):
        raw = _snapshot(WORKTREE / relative)
        if sha256(raw).hexdigest() != expected:
            raise RuntimeError(
                f"projection orchestration predecessor drifted: {relative}"
            )
        observed[relative] = expected
    hlo_source = json.loads(_snapshot(WORKTREE / HLO_SOURCE_PATH))
    projection_source = json.loads(_snapshot(WORKTREE / PROJECTION_SOURCE_PATH))
    topology = json.loads(_snapshot(WORKTREE / TOPOLOGY_PATH))
    if (
        hlo_source.get("gate_d_closed") is not False
        or projection_source.get("gate_d_closed") is not False
        or topology.get("tpu_successor_authorized") is not False
    ):
        raise RuntimeError("projection orchestration predecessor policy drifted")
    if hlo_source.get("authorization") != {
        "compile_only_hlo_acquisition": False,
        "full_dsa_or_8k": False,
        "orchestration_or_install": False,
        "persistence_only": True,
        "tpu_execution": False,
    }:
        raise RuntimeError("projection HLO source authorization drifted")
    return observed


def analyze() -> dict[str, Any]:
    _require_isolated_import_boundary()
    if (
        os.environ.get("GLM_GATE_D_PROJECTION_CONTRACTION_HLO_ORCHESTRATION_SOURCE")
        != "1"
    ):
        raise RuntimeError("projection HLO orchestration source audit is default-off")
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get(
        "JAX_PLATFORM_NAME"
    ) not in (None, "cpu"):
        raise RuntimeError(
            "projection HLO orchestration source audit must be CPU-pinned"
        )
    if loaded := _loaded_jax_modules():
        raise RuntimeError(f"projection orchestration audit started with JAX: {loaded}")
    code_pin, snapshots, source_sha256s = _committed_sources()
    audits = {
        "installer": _audit_installer(snapshots[INSTALLER_PATH]),
        "launcher": _audit_launcher(snapshots[LAUNCHER_PATH], snapshots[WRAPPER_PATH]),
        "publisher": _audit_publisher(snapshots[PUBLISHER_PATH]),
        "wrapper": _audit_wrapper(snapshots[WRAPPER_PATH]),
    }
    predecessors = _predecessor_authority()
    if loaded := _loaded_jax_modules():
        raise RuntimeError(f"projection orchestration audit imported JAX: {loaded}")
    return {
        "artifact_kind": (
            "gate_d_projection_contraction_pp16_hlo_orchestration_install_source"
        ),
        "authorization": {
            "cloud_write": False,
            "compile_only_hlo_acquisition": False,
            "full_dsa_or_8k": False,
            "launcher_invocation": False,
            "persistence_only": True,
            "privileged_install": False,
            "tpu_compile": False,
            "tpu_execution": False,
        },
        "classification": (
            "PROJECTION_HLO_ORCHESTRATION_INSTALL_SOURCE_ACCEPTED;"
            "INSTALL_UNAUTHORIZED;TPU_COMPILE_UNAUTHORIZED;"
            "TPU_EXECUTION_UNAUTHORIZED;GATE_D_OPEN"
        ),
        "code_hash": code_pin,
        "exact_next": (
            "After adversarial review and durable persistence, separately review the literal "
            "immutable install-only commands; do not launch, compile or execute."
        ),
        "gate_d_closed": False,
        "orchestration_install_audit": audits,
        "performance_claim": False,
        "predecessor_sha256s": predecessors,
        "process_contract": {"loaded_jax_modules": []},
        "schema_version": 1,
        "source_sha256s": source_sha256s,
        "tpu_compile_or_execution_performed": False,
    }


def main() -> int:
    if sys.argv[1:]:
        raise RuntimeError(
            "projection orchestration source analyzer accepts no arguments"
        )
    sys.stdout.buffer.write(_canonical(analyze()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
