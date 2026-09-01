#!/usr/bin/env python3
"""Emit the persistence-only certificate for the forced-round HLO installer."""

from __future__ import annotations

import ast
import json
import os
import stat
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
ANALYZER = WORKTREE / (
    "scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_install_source.py"
)
INSTALLER = WORKTREE / (
    "scripts/greenfield/install_gate_d_forced_round_pp16_hlo_runtime.py"
)
ORCHESTRATION_ANALYZER = WORKTREE / (
    "scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_orchestration_source.py"
)
TEST = WORKTREE / (
    "tests/greenfield/validation/test_install_gate_d_forced_round_pp16_hlo_runtime.py"
)
ORCHESTRATION_SOURCE = WORKTREE / (
    "docs/artifacts/gate-d-forced-round-pp16-hlo-orchestration-source.json"
)
ARTIFACT_PATH = "docs/artifacts/gate-d-forced-round-pp16-hlo-install-source.json"
BASE_CODE_PIN = "a012b93fdbd7c6fe1f84db2260708ba55b38e8f6"
BASE_TREE_ID = "0cd885775ea6e200b17c50efe2e051d6f1ebbaf0"
ORCHESTRATION_SOURCE_SHA256 = (
    "f4fcba8497599dd19f94ec8ded2566ad3d27a412a04adad584fc22417e18847c"
)
INSTALLER_AST_SHA256 = (
    "0556fce5e09104aacbeed4ef222d6cefb2728280ad0d8de9b1186933b9501a90"
)
PAYLOAD_PATHS = {
    "acquire_gate_d_forced_round_pp16_hlo.py": (
        "scripts/greenfield/acquire_gate_d_forced_round_pp16_hlo.py"
    ),
    "launch_gate_d_forced_round_pp16_hlo.py": (
        "scripts/greenfield/launch_gate_d_forced_round_pp16_hlo.py"
    ),
    "publish_gate_d_forced_round_pp16_hlo.py": (
        "scripts/greenfield/publish_gate_d_forced_round_pp16_hlo.py"
    ),
    "verify_gate_d_same_region_git_mirror.py": (
        "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
    ),
}
ALLOWED_DELTA_PATHS = frozenset(
    {
        "HANDOFF.md",
        "docs/RESEARCH_LOG.md",
        ARTIFACT_PATH,
        "docs/greenfield/EVIDENCE_MAP.md",
        str(ANALYZER.relative_to(WORKTREE)),
        (
            "scripts/greenfield/"
            "analyze_gate_d_forced_round_pp16_hlo_orchestration_source.py"
        ),
        str(INSTALLER.relative_to(WORKTREE)),
        str(TEST.relative_to(WORKTREE)),
    }
)


def _loaded_jax_modules() -> list[str]:
    prefixes = ("jax", "jaxlib", "jax_plugins")
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    )


def _snapshot(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError(f"unsafe installer source input: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if len(raw) != before.st_size or identity(before) != identity(after):
            raise RuntimeError(f"installer source changed while reading: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _git(arguments: list[str]) -> bytes:
    result = subprocess.run(
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
            *arguments,
        ],
        cwd=WORKTREE,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PROTOCOL_FROM_USER": "0",
            "HOME": "/home/gianl",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or result.stderr:
        raise RuntimeError(
            "forced-round installer repository authority failed: "
            f"returncode={result.returncode} stderr={result.stderr!r}"
        )
    return result.stdout


def _reject_replace_refs() -> None:
    if _git(["for-each-ref", "--format=%(refname)", "refs/replace"]):
        raise RuntimeError("installer repository has forbidden replacement refs")


def _nul_paths(raw: bytes) -> set[str]:
    if not raw:
        return set()
    if not raw.endswith(b"\0"):
        raise RuntimeError("installer repository path stream is not terminated")
    paths = [item.decode("utf-8", errors="strict") for item in raw[:-1].split(b"\0")]
    if len(paths) != len(set(paths)) or any(not path for path in paths):
        raise RuntimeError("installer repository path stream is ambiguous")
    return set(paths)


def _verify_repository_authority() -> dict[str, Any]:
    _reject_replace_refs()
    if _git(["rev-parse", f"{BASE_CODE_PIN}^{{tree}}"]) != f"{BASE_TREE_ID}\n".encode(
        "ascii"
    ):
        raise RuntimeError("forced-round installer base tree drifted")
    _git(["merge-base", "--is-ancestor", BASE_CODE_PIN, "HEAD"])
    tracked = _nul_paths(
        _git(["diff", "--no-ext-diff", "--name-only", "-z", BASE_CODE_PIN, "--"])
    )
    untracked = _nul_paths(_git(["ls-files", "--others", "--exclude-standard", "-z"]))
    if tracked & untracked or tracked | untracked != ALLOWED_DELTA_PATHS:
        raise RuntimeError(
            "forced-round installer repository delta drifted: "
            f"tracked={sorted(tracked)!r} untracked={sorted(untracked)!r}"
        )
    return {
        "allowed_delta_paths": sorted(ALLOWED_DELTA_PATHS),
        "base_code_pin": BASE_CODE_PIN,
        "base_tree_id": BASE_TREE_ID,
        "unexpected_delta_paths": [],
    }


def _literal_assignments(tree: ast.Module) -> dict[str, Any]:
    assignments: dict[str, Any] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                assignments[node.targets[0].id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return assignments


def _audit_installer(raw: bytes, payloads: dict[str, bytes]) -> dict[str, Any]:
    source = raw.decode("ascii", errors="strict")
    tree = ast.parse(source)
    ast_sha = sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    if ast_sha != INSTALLER_AST_SHA256:
        raise RuntimeError("forced-round installer AST drifted")
    assignments = _literal_assignments(tree)
    required = (
        "#!/usr/bin/env -S /usr/bin/python3 -I -S -B\n",
        'SOURCE_ROOT = Path("/opt/glm-tpu/gate-d-forced-round-hlo-install-v1")',
        'LAUNCHER_PARENT = Path("/opt/glm-tpu/bin")',
        'CAPSULE_PARENT = Path("/usr/local/libexec/glm-tpu")',
        "os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW",
        "_RENAME_NOREPLACE",
        "staging_identity is not None and not published",
        "_remove_owned_capsule_staging(",
        "_unlink_owned_file(",
        "_require_root_chain(LAUNCHER_PARENT)",
        "_require_root_chain(CAPSULE_PARENT)",
        "_require_root_chain(SOURCE_ROOT)",
        "dict(os.environ) != EXPECTED_ENVIRONMENT",
        '"launcher_invoked": False',
    )
    forbidden = (
        "subprocess",
        "os.exec",
        "jax",
        "google.cloud",
        "gs://",
        "shutil.rmtree",
    )
    if any(item not in source for item in required) or any(
        item in source for item in forbidden
    ):
        raise RuntimeError("forced-round installer boundary drifted")
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if functions != {
        "_require_root_chain",
        "_require_directory",
        "_read_regular",
        "_rename_noreplace",
        "_write_all",
        "_create_file",
        "_unlink_owned_file",
        "_remove_owned_capsule_staging",
        "_verify_capsule",
        "_publish_capsule",
        "_publish_launcher",
        "_load_payloads",
        "main",
    }:
        raise RuntimeError("forced-round installer function surface drifted")
    main_source = ast.unparse(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
    )
    if main_source.index("_publish_capsule(") > main_source.index("_publish_launcher("):
        raise RuntimeError("launcher is not published last")
    expected_payloads = {
        name: sha256(value).hexdigest() for name, value in payloads.items()
    }
    if assignments.get("PAYLOADS") != expected_payloads:
        raise RuntimeError("forced-round installer payload pins drifted")
    if assignments.get("CAPSULE_NAMES") != tuple(
        name
        for name in expected_payloads
        if name != "launch_gate_d_forced_round_pp16_hlo.py"
    ):
        raise RuntimeError("forced-round installer capsule membership drifted")
    if assignments.get("EXPECTED_ENVIRONMENT") != {
        "HOME": "/root",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }:
        raise RuntimeError("forced-round installer environment drifted")
    return {
        "atomic_publish": "renameat2_RENAME_NOREPLACE",
        "capsule_published_before_launcher": True,
        "destination_capsule": (
            "/usr/local/libexec/glm-tpu/gate-d-forced-round-pp16-hlo"
        ),
        "destination_launcher": (
            "/opt/glm-tpu/bin/launch_gate_d_forced_round_pp16_hlo.py"
        ),
        "installer_ast_sha256": ast_sha,
        "installer_sha256": sha256(raw).hexdigest(),
        "launcher_invocation_count": 0,
        "owned_staging_cleanup_only": True,
        "payload_sha256s": expected_payloads,
    }


def main() -> int:
    if os.environ.get("GLM_GATE_D_FORCED_ROUND_HLO_INSTALL_SOURCE") != "1":
        raise RuntimeError("forced-round HLO installer source audit is default-off")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("forced-round HLO installer source audit must be CPU-pinned")
    if loaded := _loaded_jax_modules():
        raise RuntimeError(f"installer source audit started with JAX: {loaded}")
    snapshots = {
        "analyzer": _snapshot(ANALYZER),
        "installer": _snapshot(INSTALLER),
        "orchestration_analyzer": _snapshot(ORCHESTRATION_ANALYZER),
        "test": _snapshot(TEST),
        "orchestration_source": _snapshot(ORCHESTRATION_SOURCE),
    }
    if (
        sha256(snapshots["orchestration_source"]).hexdigest()
        != ORCHESTRATION_SOURCE_SHA256
    ):
        raise RuntimeError("forced-round orchestration predecessor drifted")
    predecessor = json.loads(snapshots["orchestration_source"])
    if (
        predecessor.get("authorization", {}).get("persistence_only") is not True
        or predecessor.get("authorization", {}).get("hlo_acquisition") is not False
    ):
        raise RuntimeError("forced-round orchestration predecessor policy drifted")
    payloads = {
        name: _git(["show", f"{BASE_CODE_PIN}:{path}"])
        for name, path in PAYLOAD_PATHS.items()
    }
    repository = _verify_repository_authority()
    installer = _audit_installer(snapshots["installer"], payloads)
    if loaded := _loaded_jax_modules():
        raise RuntimeError(f"installer source audit imported JAX: {loaded}")
    report = {
        "artifact_kind": "gate_d_forced_round_pp16_hlo_install_source",
        "authorization": {
            "cloud_write": False,
            "full_8k": False,
            "hlo_acquisition": False,
            "launcher_invocation": False,
            "persistence_only": True,
            "privileged_install": False,
            "tpu_compile": False,
            "tpu_execution": False,
        },
        "claim_scope": (
            "Static installer/no-replace audit only. No privileged install, launcher "
            "invocation, JAX import, HLO acquisition, TPU compile/execution, cloud "
            "write, numerical, performance or Gate-D claim."
        ),
        "install_audit": installer,
        "predecessors": {
            "base_code_pin": BASE_CODE_PIN,
            "orchestration_source_sha256": ORCHESTRATION_SOURCE_SHA256,
        },
        "process_contract": {"loaded_jax_modules": []},
        "repository_authority": repository,
        "source_authority": {
            "analyzer_path": str(ANALYZER.relative_to(WORKTREE)),
            "analyzer_sha256": sha256(snapshots["analyzer"]).hexdigest(),
            "installer_path": str(INSTALLER.relative_to(WORKTREE)),
            "installer_sha256": sha256(snapshots["installer"]).hexdigest(),
            "orchestration_analyzer_path": str(
                ORCHESTRATION_ANALYZER.relative_to(WORKTREE)
            ),
            "orchestration_analyzer_sha256": sha256(
                snapshots["orchestration_analyzer"]
            ).hexdigest(),
            "test_path": str(TEST.relative_to(WORKTREE)),
            "test_sha256": sha256(snapshots["test"]).hexdigest(),
        },
        "status": "FORCED_ROUND_HLO_INSTALL_SOURCE_PERSISTENCE_ONLY",
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
