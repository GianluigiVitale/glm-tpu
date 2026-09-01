#!/usr/bin/env python3
"""Emit the persistence-only source certificate for forced-round HLO acquisition."""

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
    "scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_acquisition_source.py"
)
DRIVER = WORKTREE / "scripts/greenfield/acquire_gate_d_forced_round_pp16_hlo.py"
BASE_DRIVER = WORKTREE / "scripts/greenfield/acquire_gate_d_compensated_pp16_hlo.py"
SOURCE_CERTIFICATE = (
    WORKTREE / "docs/artifacts/gate-d-forced-round-pp16-hlo-source.json"
)
ADMISSION = WORKTREE / (
    "docs/artifacts/gate-d-precompile-admission-v2-compensated-capsule.json"
)
TOPOLOGY = WORKTREE / "docs/artifacts/gate-d-runtime-locality-authority.json"
BASE_CODE_PIN = "2a050c1182991d93a7a4355a2820b044aa7a5861"
BASE_TREE_ID = "f3291746ff2000e64973cb330ae203ab7e484830"
BASE_DRIVER_SHA256 = "a4599e0d88a1ef8a2df897d1677196e8cd0db5009f12b866e7316ecaadb5dd15"
SOURCE_CERTIFICATE_SHA256 = (
    "518be87b650729d365dd09aba20b5f5a03d4bcddccc81978357cdbb02cbe02c6"
)
ADMISSION_SHA256 = "7cd7e569ed9ed5fd978d933efd4229d65906ae28312e264863b8336e4cc6b37d"
TOPOLOGY_SHA256 = "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
FORCED_HELPER_AST_SHA256S = {
    "validate_forced_round_source": (
        "728601cec5c656d71b38ff0aaf25816ab5842546a28a527d1c96cf60ca76157b"
    ),
    "verify_forced_round_source_git_blobs": (
        "1b53c25ff987722e19e816ac242885fae575ac875f9aaf2bbf287235e9579d37"
    ),
}
ARTIFACT_PATH = "docs/artifacts/gate-d-forced-round-pp16-hlo-acquisition-source.json"
ALLOWED_DELTA_PATHS = frozenset(
    {
        "HANDOFF.md",
        "docs/RESEARCH_LOG.md",
        ARTIFACT_PATH,
        "docs/greenfield/EVIDENCE_MAP.md",
        "scripts/greenfield/acquire_gate_d_forced_round_pp16_hlo.py",
        "scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_acquisition_source.py",
        "tests/greenfield/validation/test_gate_d_forced_round_pp16_hlo_acquisition.py",
        "tests/greenfield/validation/test_gate_d_forced_round_pp16_hlo_source.py",
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
            raise RuntimeError(f"unsafe acquisition source input: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)

        def identity(value: os.stat_result) -> tuple[int, ...]:
            return (
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

        raw = b"".join(chunks)
        if len(raw) != before.st_size or identity(before) != identity(after):
            raise RuntimeError(f"acquisition source changed while reading: {path}")
        return raw
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
            "GIT_OPTIONAL_LOCKS": "0",
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
            "forced-round acquisition repository authority failed: "
            f"returncode={result.returncode} stderr={result.stderr!r}"
        )
    return result.stdout


def _nul_paths(raw: bytes) -> set[str]:
    if not raw:
        return set()
    if not raw.endswith(b"\0"):
        raise RuntimeError("acquisition repository path stream is not NUL-terminated")
    paths = [item.decode("utf-8", errors="strict") for item in raw[:-1].split(b"\0")]
    if len(paths) != len(set(paths)) or any(not path for path in paths):
        raise RuntimeError("acquisition repository path stream is ambiguous")
    return set(paths)


def _verify_repository_authority() -> dict[str, Any]:
    if _git(["rev-parse", f"{BASE_CODE_PIN}^{{tree}}"]) != f"{BASE_TREE_ID}\n".encode(
        "ascii"
    ):
        raise RuntimeError("forced-round acquisition base tree drifted")
    _git(["merge-base", "--is-ancestor", BASE_CODE_PIN, "HEAD"])
    tracked = _nul_paths(
        _git(["diff", "--no-ext-diff", "--name-only", "-z", BASE_CODE_PIN, "--"])
    )
    untracked = _nul_paths(_git(["ls-files", "--others", "--exclude-standard", "-z"]))
    if tracked & untracked or tracked | untracked != ALLOWED_DELTA_PATHS:
        raise RuntimeError(
            "forced-round acquisition repository delta drifted: "
            f"tracked={sorted(tracked)!r} untracked={sorted(untracked)!r}"
        )
    return {
        "allowed_delta_paths": sorted(ALLOWED_DELTA_PATHS),
        "base_code_pin": BASE_CODE_PIN,
        "base_tree_id": BASE_TREE_ID,
        "unexpected_delta_paths": [],
    }


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"forced-round acquisition assignment drifted: {name}")
    value = matches[0].value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "Path"
        and len(value.args) == 1
        and not value.keywords
    ):
        return Path(ast.literal_eval(value.args[0]))
    return ast.literal_eval(value)


def _function_map(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _call_lines(function: ast.FunctionDef, name: str) -> list[int]:
    return sorted(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    )


def _audit_driver(driver_raw: bytes, base_raw: bytes) -> dict[str, Any]:
    driver_source = driver_raw.decode("utf-8", errors="strict")
    base_source = base_raw.decode("utf-8", errors="strict")
    driver_tree = ast.parse(driver_source)
    base_tree = ast.parse(base_source)
    if ast.unparse(driver_tree) != ast.unparse(ast.parse(ast.unparse(driver_tree))):
        raise RuntimeError("forced-round acquisition source is not canonical Python")
    if sha256(base_raw).hexdigest() != BASE_DRIVER_SHA256:
        raise RuntimeError("hardened compensated acquisition base drifted")
    driver_functions = _function_map(driver_tree)
    base_functions = _function_map(base_tree)
    if set(driver_functions) - set(base_functions) != {
        "validate_forced_round_source",
        "verify_forced_round_source_git_blobs",
    } or set(base_functions) - set(driver_functions):
        raise RuntimeError("forced-round acquisition function surface drifted")
    for name, expected in FORCED_HELPER_AST_SHA256S.items():
        observed = sha256(
            ast.dump(driver_functions[name], include_attributes=False).encode("utf-8")
        ).hexdigest()
        if observed != expected:
            raise RuntimeError(f"forced-round acquisition helper drifted: {name}")
    for name in sorted(set(base_functions) - {"main", "parse_args"}):
        if ast.dump(driver_functions[name], include_attributes=False) != ast.dump(
            base_functions[name], include_attributes=False
        ):
            raise RuntimeError(f"hardened acquisition primitive drifted: {name}")
    expected_output_spec = (
        ("normalized_hidden_owners", (2, 1, 6144), "bfloat16"),
        ("query_owners", (2, 1, 32, 128), "float32"),
        ("head_weights_owners", (2, 1, 32), "float32"),
        ("current_key_owners", (2, 1, 128), "float32"),
        ("index_cache_owners", (2, 16, 256, 128), "bfloat16"),
        ("selected_positions_owners", (2, 1, 2048), "int32"),
        ("valid_counts_owners", (2, 1), "int32"),
        ("selected_scores_owners", (2, 1, 2048), "float32"),
        ("contract_valid_owners", (2,), "uint8"),
    )
    if (
        _literal_assignment(driver_tree, "REPO")
        != Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
        or _literal_assignment(driver_tree, "DRIVER_REPOSITORY_PATH")
        != "scripts/greenfield/acquire_gate_d_forced_round_pp16_hlo.py"
        or _literal_assignment(driver_tree, "OUTPUT_SPEC") != expected_output_spec
        or _literal_assignment(driver_tree, "TAG_PATTERN")
        != r"gate_d_forced_round_pp16_hlo_[0-9]{8}T[0-9]{15}Z"
        or _literal_assignment(driver_tree, "FORCED_ROUND_SOURCE_SHA256")
        != SOURCE_CERTIFICATE_SHA256
        or _literal_assignment(driver_tree, "_EXPECTED_ENVIRONMENT")
        != _literal_assignment(base_tree, "_EXPECTED_ENVIRONMENT")
        or _literal_assignment(driver_tree, "INPUT_SPEC")
        != _literal_assignment(base_tree, "INPUT_SPEC")
    ):
        raise RuntimeError("forced-round acquisition profile drifted")
    main = driver_functions["main"]
    imports = [
        node for node in main.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    jax_import_lines = sorted(
        node.lineno
        for node in imports
        if (
            isinstance(node, ast.Import)
            and any(alias.name in {"jax", "jax.numpy"} for alias in node.names)
        )
    )
    if len(jax_import_lines) != 2:
        raise RuntimeError("forced-round acquisition JAX import boundary drifted")
    first_jax = jax_import_lines[0]
    for call in (
        "validate_forced_round_source",
        "verify_forced_round_source_git_blobs",
        "_open_inherited_run_dir",
        "_sealed_git_source_archive",
    ):
        lines = _call_lines(main, call)
        if len(lines) != 1 or lines[0] >= first_jax:
            raise RuntimeError(
                f"forced-round acquisition pre-JAX binding drifted: {call}"
            )
    if (
        len(_call_lines(main, "lower")) != 1
        or len(_call_lines(main, "compile")) != 1
        or _call_lines(main, "compiled")
        or "jax.device_put" in driver_source
        or "block_until_ready" in driver_source
        or "np.load" in driver_source
        or "build_gate_d_compensated_pp16_hlo_replay" in ast.unparse(main)
        or ast.unparse(main).count("build_gate_d_forced_round_pp16_hlo_replay") != 2
        or '"compiled_executable_invocation_count": 0' not in driver_source
        or '"tpu_numerical_execution_performed": False' not in driver_source
        or "gate_d_forced_round_pp16_optimized_hlo_acquisition" not in driver_source
        or "forced_round_pp16_stage0.optimized_hlo.txt" not in driver_source
    ):
        raise RuntimeError("forced-round acquisition compile-only contract drifted")
    return {
        "base_driver_sha256": BASE_DRIVER_SHA256,
        "compile_call_count": 1,
        "compiled_executable_invocation_count": 0,
        "driver_ast_sha256": sha256(
            ast.dump(driver_tree, include_attributes=False).encode("utf-8")
        ).hexdigest(),
        "driver_sha256": sha256(driver_raw).hexdigest(),
        "forced_round_source_validation_precedes_jax": True,
        "hardened_common_function_count": len(base_functions) - 2,
        "input_spec_count": 16,
        "lower_call_count": 1,
        "output_spec_count": 9,
        "tpu_execution_call_count": 0,
    }


def main() -> int:
    if os.environ.get("GLM_GATE_D_FORCED_ROUND_HLO_ACQUISITION_SOURCE") != "1":
        raise RuntimeError("forced-round HLO acquisition source audit is default-off")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError(
            "forced-round HLO acquisition source audit must be CPU-pinned"
        )
    if loaded := _loaded_jax_modules():
        raise RuntimeError(f"acquisition source audit started with JAX: {loaded}")
    snapshots = {
        "analyzer": _snapshot(ANALYZER),
        "driver": _snapshot(DRIVER),
        "base_driver": _snapshot(BASE_DRIVER),
        "source_certificate": _snapshot(SOURCE_CERTIFICATE),
        "admission": _snapshot(ADMISSION),
        "topology": _snapshot(TOPOLOGY),
    }
    for name, expected in (
        ("base_driver", BASE_DRIVER_SHA256),
        ("source_certificate", SOURCE_CERTIFICATE_SHA256),
        ("admission", ADMISSION_SHA256),
        ("topology", TOPOLOGY_SHA256),
    ):
        if sha256(snapshots[name]).hexdigest() != expected:
            raise RuntimeError(f"forced-round acquisition predecessor drifted: {name}")
    source_certificate = json.loads(snapshots["source_certificate"])
    admission = json.loads(snapshots["admission"])
    topology = json.loads(snapshots["topology"])
    if (
        source_certificate.get("status")
        != "PP16_HLO_INTEGRATION_SOURCE_PERSISTENCE_ONLY"
        or source_certificate.get("authorization", {}).get("hlo_acquisition")
        is not False
        or admission.get("compile_only_review_required") is not True
        or admission.get("tpu_successor_authorized") is not False
        or topology.get("tpu_successor_authorized") is not False
    ):
        raise RuntimeError("forced-round acquisition predecessor policy drifted")
    repository = _verify_repository_authority()
    audit = _audit_driver(snapshots["driver"], snapshots["base_driver"])
    if loaded := _loaded_jax_modules():
        raise RuntimeError(f"acquisition source audit imported JAX: {loaded}")
    report = {
        "artifact_kind": "gate_d_forced_round_pp16_hlo_acquisition_source",
        "authorization": {
            "full_8k": False,
            "hlo_acquisition": False,
            "persistence_only": True,
            "tpu_compile": False,
            "tpu_execution": False,
        },
        "claim_scope": (
            "Static acquisition-driver audit only. No JAX import, lowering, HLO, "
            "TPU compile/execution, cloud write, numerical, performance, or Gate-D claim."
        ),
        "driver_audit": audit,
        "predecessors": {
            "admission_sha256": ADMISSION_SHA256,
            "forced_round_source_sha256": SOURCE_CERTIFICATE_SHA256,
            "topology_sha256": TOPOLOGY_SHA256,
        },
        "process_contract": {"loaded_jax_modules": []},
        "repository_authority": repository,
        "source_authority": {
            "analyzer_path": str(ANALYZER.relative_to(WORKTREE)),
            "analyzer_sha256": sha256(snapshots["analyzer"]).hexdigest(),
            "driver_path": str(DRIVER.relative_to(WORKTREE)),
            "driver_sha256": sha256(snapshots["driver"]).hexdigest(),
        },
        "status": "FORCED_ROUND_HLO_ACQUISITION_SOURCE_PERSISTENCE_ONLY",
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
