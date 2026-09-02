#!/usr/bin/env python3
"""Certify projection-contraction numerical orchestration and install sources only."""

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
    "analyze_gate_d_projection_contraction_pp16_numerical_orchestration_source.py"
)
DRIVER_PATH = "scripts/greenfield/run_gate_d_projection_contraction_pp16_numerical.py"
PUBLISHER_PATH = (
    "scripts/greenfield/publish_gate_d_projection_contraction_pp16_numerical.py"
)
WRAPPER_PATH = "scripts/greenfield/run_gate_d_projection_contraction_pp16_numerical.sh"
LAUNCHER_PATH = (
    "scripts/greenfield/launch_gate_d_projection_contraction_pp16_numerical.py"
)
INSTALLER_PATH = (
    "scripts/greenfield/install_gate_d_projection_contraction_pp16_numerical_runtime.py"
)
MIRROR_VERIFIER_PATH = "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
AUDITED_SOURCE_PATHS = (
    DRIVER_PATH,
    ANALYZER_PATH,
    INSTALLER_PATH,
    LAUNCHER_PATH,
    MIRROR_VERIFIER_PATH,
    PUBLISHER_PATH,
    WRAPPER_PATH,
)
EXPECTED_SOURCE_SHA256S = {
    DRIVER_PATH: "a38bb8b89bb8cf70568c8562b2013f188b7c8ed17f1e41edad37a40332156404",
    INSTALLER_PATH: "8b60c832fe70c6ca917f0b8f6090d5a4f8525218f068ab3d79edfb2e7b5b3970",
    LAUNCHER_PATH: "80c9116fcf9397995bb26bb14e660d5982213b7c5d4be01afa576168b188d191",
    MIRROR_VERIFIER_PATH: "091208165a149989f14c5c9b9d1cbe7ff20537e2c81b16319eea9603984e859b",
    PUBLISHER_PATH: "5c5e58afa2c36e46377512a777df6a00669419f9f1c93e3504f71ea1334f6665",
    WRAPPER_PATH: "a5a5a5ffc1d012944c48b16e433906a6506441836f66c88f4115d8b2e70598d9",
}
PROJECTION_SOURCE_PATH = "docs/artifacts/gate-d-projection-contraction-pp16-source.json"
PROJECTION_SOURCE_SHA256 = (
    "5744eee0ef2cf35a4566cc0de165be1338160daaae3f4dd133554b2aa8280e9f"
)
HLO_SUCCESS_PATH = (
    "docs/artifacts/gate-d-projection-contraction-pp16-success-hlo-adjudication.json"
)
HLO_SUCCESS_SHA256 = "54eb6105b81d9ffdbdd3b90735059fb324701509fe8efd003033bb3cff7e0ad7"
FRONTIER_PATH = "docs/artifacts/gate-d-projection-arithmetic-frontier-analysis.json"
FRONTIER_SHA256 = "4a6be0f33f221df46f384cd2047da1aa664ae4c45763e68f0f1df05f8644e20c"
HOST_MATERIALIZATION_PATH = (
    "docs/artifacts/gate-d-pp16-numerical-host-materialization-equivalence.json"
)
HOST_MATERIALIZATION_SHA256 = (
    "a7e5b393f7c61181f6b2aab9531d788fb27594d9cf980055c165b6f278ba4f99"
)
TOPOLOGY_PATH = "docs/artifacts/gate-d-runtime-locality-authority.json"
HLO_BRIDGE_PATH = (
    "docs/artifacts/gate-d-projection-contraction-pp16-hlo-source-location-bridge.json"
)
HLO_BRIDGE_SHA256 = "c2732f7184416cce87839b4cadf552b94d983aba06fa56b687eaacc03ca26bea"
TOPOLOGY_SHA256 = "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
REFERENCE_PATHS = {
    FRONTIER_PATH: FRONTIER_SHA256,
    HLO_BRIDGE_PATH: HLO_BRIDGE_SHA256,
    HLO_SUCCESS_PATH: HLO_SUCCESS_SHA256,
    HOST_MATERIALIZATION_PATH: HOST_MATERIALIZATION_SHA256,
    PROJECTION_SOURCE_PATH: PROJECTION_SOURCE_SHA256,
    TOPOLOGY_PATH: TOPOLOGY_SHA256,
}
EXPECTED_PAYLOADS = {
    "run_gate_d_projection_contraction_pp16_numerical.py": EXPECTED_SOURCE_SHA256S[
        DRIVER_PATH
    ],
    "launch_gate_d_projection_contraction_pp16_numerical.py": EXPECTED_SOURCE_SHA256S[
        LAUNCHER_PATH
    ],
    "publish_gate_d_projection_contraction_pp16_numerical.py": EXPECTED_SOURCE_SHA256S[
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
        "gate_d_projection_contraction_pp16_numerical/",
        "NUMERICAL_ACCEPTED",
        "NUMERICAL_REJECTED",
        'runner.get("compiled_executable_invocation_count") != 1',
        'runner.get("tpu_numerical_execution_performed") is not True',
        'runner.get("root_cause_fix_proven") is not False',
        "derived_accepted = all(",
        "_validate_mirror_replay(mirror_raw, code_pin)",
    )
    forbidden = (
        "gate_d_compensated_pp16_numerical/",
        "gate_d_forced_round_pp16_numerical/",
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
        "DRIVER_SHA": EXPECTED_SOURCE_SHA256S[DRIVER_PATH],
        "FRONTIER_AUTHORITY_SHA": FRONTIER_SHA256,
        "HLO_SOURCE_LOCATION_BRIDGE_SHA": HLO_BRIDGE_SHA256,
        "HLO_SUCCESS_AUTHORITY_SHA": HLO_SUCCESS_SHA256,
        "HOST_MATERIALIZATION_AUTHORITY_SHA": HOST_MATERIALIZATION_SHA256,
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
        "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_NUMERICAL",
        "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_NUMERICAL_MODE",
        "GLM_GATE_D_PROJECTION_CONTRACTION_PP16_NUMERICAL_TAG",
        "GLM_GATE_D_PROJECTION_CONTRACTION_NUMERICAL=1",
        "JAX_ENABLE_COMPILATION_CACHE=0",
        "strict_census pre",
        "strict_census post",
        "--execute-once 1",
        "GIT_AUTHORITY_VERIFIER",
        '"/usr/bin/env",',
        '"-i",',
        '"core.fsmonitor=false"',
        'git(Path("/"), "ls-remote", "--refs", origin, expected_ref)',
        "Git repository has forbidden replacement refs",
        '"--no-includes"',
        '/usr/bin/python3 -I -S -B -c "$GIT_AUTHORITY_VERIFIER"',
        "NUMERICAL_ACCEPTED archive=",
        "NUMERICAL_REJECTED archive=",
        "NUMERICAL_RESULT_VERIFIER",
        'parse_canonical("NUMERICAL_RESULT")',
        'parse_canonical("terminal_upload_receipt.json")',
        "! -e /proc/self/fd/7/NUMERICAL_RESULT",
        'case "$result_status" in',
        "gate-d-projection-contraction-pp16-numerical-v2",
        "--hlo-source-location-bridge",
        "wrapper_fd != 10",
    )
    forbidden = (
        "decode_8k",
        "block_until_ready",
        "gate_d_compensated_pp16_hlo/",
        '"--includes=false"',
        '$(git -C "$WORKTREE"',
        "$RUN_DIR/NUMERICAL_ACCEPTED",
        "$RUN_DIR/NUMERICAL_REJECTED",
    )
    if (
        any(item not in source for item in required)
        or any(item in source for item in forbidden)
        or source.count('"$DRIVER_PYTHON" -I -S -B -u "$DRIVER"') != 1
        or source.count("--execute-once 1") != 1
        or source.index("strict_census pre")
        > source.index(
            "executing one exact-input PP16 stage-zero numerical discriminator"
        )
        or source.index("strict_census post")
        < source.index(
            "executing one exact-input PP16 stage-zero numerical discriminator"
        )
    ):
        raise RuntimeError("projection wrapper numerical exactly-once boundary drifted")
    return {"protected_numerical_process_count": 1, "sha256": sha256(raw).hexdigest()}


def _audit_launcher(raw: bytes, wrapper_raw: bytes) -> dict[str, Any]:
    if sha256(raw).hexdigest() != EXPECTED_SOURCE_SHA256S[LAUNCHER_PATH]:
        raise RuntimeError("projection launcher source hash drifted")
    source = raw.decode("ascii")
    tree = ast.parse(source)
    expected = {
        "DRIVER_SHA256": EXPECTED_SOURCE_SHA256S[DRIVER_PATH],
        "MIRROR_VERIFIER_SHA256": EXPECTED_SOURCE_SHA256S[MIRROR_VERIFIER_PATH],
        "PUBLISHER_SHA256": EXPECTED_SOURCE_SHA256S[PUBLISHER_PATH],
        "WRAPPER_SHA256": EXPECTED_SOURCE_SHA256S[WRAPPER_PATH],
    }
    if any(_assignment(tree, name) != value for name, value in expected.items()):
        raise RuntimeError("projection launcher immutable pin drifted")
    required = (
        '"/opt/glm-tpu/bin/launch_gate_d_projection_contraction_pp16_numerical_v2.py"',
        '"/usr/local/libexec/glm-tpu/gate-d-projection-contraction-pp16-numerical-v2"',
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
        "SOURCE_ROOT = Path('/opt/glm-tpu/gate-d-projection-contraction-numerical-install-v2')",
        "LAUNCHER_TARGET = LAUNCHER_PARENT / 'launch_gate_d_projection_contraction_pp16_numerical_v2.py'",
        "CAPSULE_TARGET = CAPSULE_PARENT / 'gate-d-projection-contraction-pp16-numerical-v2'",
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
    hlo_success = json.loads(_snapshot(WORKTREE / HLO_SUCCESS_PATH))
    projection_source = json.loads(_snapshot(WORKTREE / PROJECTION_SOURCE_PATH))
    frontier = json.loads(_snapshot(WORKTREE / FRONTIER_PATH))
    host_materialization = json.loads(_snapshot(WORKTREE / HOST_MATERIALIZATION_PATH))
    topology = json.loads(_snapshot(WORKTREE / TOPOLOGY_PATH))
    if (
        hlo_success.get("gate_d_closed") is not False
        or projection_source.get("gate_d_closed") is not False
        or frontier.get("gate_d_closed") is not False
        or host_materialization.get("gate_d_closed") is not False
        or topology.get("tpu_successor_authorized") is not False
    ):
        raise RuntimeError("projection numerical predecessor policy drifted")
    if hlo_success.get("authorization") != {
        "full_8k": False,
        "numerical_execution": False,
        "performance_claim": False,
        "persistence_only": True,
    }:
        raise RuntimeError("projection HLO authority numerical boundary drifted")
    return observed


def analyze() -> dict[str, Any]:
    _require_isolated_import_boundary()
    if (
        os.environ.get(
            "GLM_GATE_D_PROJECTION_CONTRACTION_NUMERICAL_ORCHESTRATION_SOURCE"
        )
        != "1"
    ):
        raise RuntimeError(
            "projection numerical orchestration source audit is default-off"
        )
    if os.environ.get("JAX_PLATFORMS") != "cpu" or os.environ.get(
        "JAX_PLATFORM_NAME"
    ) not in (None, "cpu"):
        raise RuntimeError(
            "projection numerical orchestration source audit must be CPU-pinned"
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
            "gate_d_projection_contraction_pp16_numerical_orchestration_install_source"
        ),
        "authorization": {
            "cloud_write": False,
            "numerical_execution": False,
            "full_dsa_or_8k": False,
            "launcher_invocation": False,
            "persistence_only": True,
            "privileged_install": False,
            "tpu_compile": False,
            "tpu_execution": False,
        },
        "classification": (
            "PROJECTION_NUMERICAL_ORCHESTRATION_INSTALL_SOURCE_ACCEPTED;"
            "INSTALL_UNAUTHORIZED;TPU_NUMERICAL_EXECUTION_UNAUTHORIZED;"
            "GATE_D_OPEN"
        ),
        "code_hash": code_pin,
        "exact_next": (
            "After adversarial review and durable persistence, separately review the literal "
            "immutable install-only commands; do not launch or execute."
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
