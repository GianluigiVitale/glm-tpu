#!/usr/bin/env python3
"""Emit a static certificate for the forced-round PP16 HLO builder."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
ANALYZER = (
    WORKTREE / "scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_source.py"
)
BUILDER = WORKTREE / "glm_tpu/greenfield/benchmarking/gate_d_forced_round_pp16_hlo.py"
AUDITOR = WORKTREE / "glm_tpu/greenfield/validation/gate_d_forced_round_hlo_source.py"
SOURCE_DESIGN_ARTIFACT = (
    WORKTREE / "docs/artifacts/gate-d-forced-normalized-bf16-source-design.json"
)
TOPOLOGY_AUTHORITY = WORKTREE / "docs/artifacts/gate-d-runtime-locality-authority.json"
RMSNORM_MODULE = WORKTREE / "glm_tpu/greenfield/kernels/reference/rmsnorm.py"
SOURCE_DESIGN_SHA256 = (
    "54bb2750de8b4261229cd8e38b53a0df00083689eda478e94196e7b2ad7d3100"
)
TOPOLOGY_AUTHORITY_SHA256 = (
    "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
)
BASE_CODE_PIN = "fc0381688bcdb03c95021f397c6fac43443c21ba"
BASE_TREE_ID = "691f16dc9dfd6d1fb8cc13ed56a49ef9be760b4b"
ALLOWED_DELTA_PATHS = frozenset(
    {
        "HANDOFF.md",
        "docs/RESEARCH_LOG.md",
        "docs/artifacts/gate-d-forced-round-pp16-hlo-source.json",
        "docs/greenfield/EVIDENCE_MAP.md",
        "glm_tpu/greenfield/benchmarking/gate_d_forced_round_pp16_hlo.py",
        "glm_tpu/greenfield/validation/gate_d_forced_round_hlo_source.py",
        "scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_source.py",
        "tests/greenfield/validation/test_gate_d_forced_round_pp16_hlo_source.py",
    }
)
DIRECT_RUNTIME_DEPENDENCIES = (
    "glm_tpu/greenfield/benchmarking/gate_d_compensated_capsule.py",
    "glm_tpu/greenfield/errors.py",
    "glm_tpu/greenfield/kernels/reference/attention.py",
    "glm_tpu/greenfield/kernels/reference/dsa.py",
    "glm_tpu/greenfield/kernels/reference/qkv_a.py",
    "glm_tpu/greenfield/kernels/reference/rmsnorm.py",
    "glm_tpu/greenfield/kernels/stage_local.py",
)


def _loaded_jax_modules() -> list[str]:
    prefixes = ("jax", "jaxlib", "jax_plugins")
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    )


def _git(arguments: list[str], *, expected_returncode: int = 0) -> bytes:
    command = [
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
    ]
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "HOME": "/home/gianl",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }
    result = subprocess.run(
        command,
        cwd=WORKTREE,
        env=environment,
        capture_output=True,
        check=False,
    )
    if result.returncode != expected_returncode or result.stderr:
        raise RuntimeError(
            "forced-round repository authority failed: "
            f"returncode={result.returncode} stderr={result.stderr!r}"
        )
    return result.stdout


def _nul_paths(raw: bytes) -> set[str]:
    if not raw:
        return set()
    if not raw.endswith(b"\0"):
        raise RuntimeError("forced-round repository path stream is not NUL-terminated")
    paths = [item.decode("utf-8", errors="strict") for item in raw[:-1].split(b"\0")]
    if len(paths) != len(set(paths)) or any(not path for path in paths):
        raise RuntimeError("forced-round repository path stream is ambiguous")
    return set(paths)


def _verify_repository_authority() -> dict[str, object]:
    tree_raw = _git(["rev-parse", f"{BASE_CODE_PIN}^{{tree}}"])
    if tree_raw != f"{BASE_TREE_ID}\n".encode("ascii"):
        raise RuntimeError("forced-round base tree drifted")
    _git(["merge-base", "--is-ancestor", BASE_CODE_PIN, "HEAD"])
    tracked_delta = _nul_paths(
        _git(["diff", "--no-ext-diff", "--name-only", "-z", BASE_CODE_PIN, "--"])
    )
    untracked_delta = _nul_paths(
        _git(["ls-files", "--others", "--exclude-standard", "-z"])
    )
    if tracked_delta & untracked_delta:
        raise RuntimeError("forced-round repository paths overlap")
    if tracked_delta | untracked_delta != ALLOWED_DELTA_PATHS:
        raise RuntimeError(
            "forced-round repository delta drifted: "
            f"tracked={sorted(tracked_delta)!r} untracked={sorted(untracked_delta)!r}"
        )
    dependencies = {
        path: sha256(_snapshot(WORKTREE / path)).hexdigest()
        for path in DIRECT_RUNTIME_DEPENDENCIES
    }
    return {
        "allowed_delta_paths": sorted(ALLOWED_DELTA_PATHS),
        "base_code_pin": BASE_CODE_PIN,
        "base_tree_id": BASE_TREE_ID,
        "direct_runtime_dependency_sha256s": dependencies,
        "unexpected_delta_paths": [],
    }


def _function_source_sha256(source_raw: bytes, name: str) -> str:
    source = source_raw.decode("utf-8", errors="strict")
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1:
        raise RuntimeError(f"forced-round function definition drifted: {name}")
    node = functions[0]
    raw = "".join(source.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno])
    return sha256(raw.encode("utf-8")).hexdigest()


def _verify_predecessor_function_authority(
    predecessor: dict[str, Any], rmsnorm_raw: bytes
) -> dict[str, str]:
    authority = predecessor.get("source_authority", {})
    module_sha256 = sha256(rmsnorm_raw).hexdigest()
    function_sha256 = _function_source_sha256(
        rmsnorm_raw, "fused_add_rms_norm_with_forced_bf16_boundary"
    )
    if (
        authority.get("function") != "fused_add_rms_norm_with_forced_bf16_boundary"
        or authority.get("function_module_path")
        != "glm_tpu/greenfield/kernels/reference/rmsnorm.py"
        or authority.get("function_module_sha256") != module_sha256
        or authority.get("function_source_sha256") != function_sha256
    ):
        raise RuntimeError("forced-round predecessor function authority drifted")
    return {
        "forced_function_module_sha256": module_sha256,
        "forced_function_source_sha256": function_sha256,
    }


def _verify_topology_authority(topology: dict[str, Any]) -> dict[str, Any]:
    expected_stage_zero = {
        "coordinates": [[0, 0, 0], [1, 0, 0]],
        "device_ids": [0, 1],
        "process_index": 0,
        "stage_id": 0,
    }
    pp16 = topology.get("pp16_lp2")
    groups = pp16.get("groups") if isinstance(pp16, dict) else None
    if (
        topology.get("topology_hash")
        != "294e777210485f08a3b323121134296e576914eb52b42792019ceef7467dd559"
        or topology.get("pp16_lp2_hash")
        != "6383e57c81478ac0d6de4525a4675f2a0d7cbc7aa73bd67bc662dc4e05840f21"
        or not isinstance(pp16, dict)
        or pp16.get("plan") != "PP16_LP2"
        or not isinstance(groups, list)
        or len(groups) != 16
        or groups[0] != expected_stage_zero
    ):
        raise RuntimeError("forced-round PP16 stage-zero authority drifted")
    return expected_stage_zero


def _load_snapshot_auditor(
    auditor_raw: bytes,
) -> Callable[[bytes], dict[str, Any]]:
    spec = importlib.util.spec_from_file_location(
        "_gate_d_forced_round_hlo_source_auditor", AUDITOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("forced-round HLO source auditor loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if _snapshot(AUDITOR) != auditor_raw:
        raise RuntimeError("forced-round HLO source auditor changed while loading")
    audit = getattr(module, "audit_forced_round_pp16_hlo_source", None)
    if not callable(audit):
        raise TypeError("forced-round HLO source auditor entry point drifted")
    return cast(Callable[[bytes], dict[str, Any]], audit)


def _snapshot(path: Path, *, limit: int = 1 << 20) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > limit
        ):
            raise RuntimeError(f"unsafe forced-round HLO source: {path}")
        raw = bytearray()
        while block := os.read(descriptor, 1 << 20):
            raw.extend(block)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)

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

        if (
            len(raw) != before.st_size
            or identity(before) != identity(after)
            or (before.st_dev, before.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise RuntimeError(f"forced-round HLO source changed: {path}")
        return bytes(raw)
    finally:
        os.close(descriptor)


def main() -> int:
    if os.environ.get("GLM_GATE_D_FORCED_ROUND_HLO_SOURCE") != "1":
        raise RuntimeError("forced-round HLO source audit is default-off")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("forced-round HLO source audit must be CPU-pinned")
    if loaded := _loaded_jax_modules():
        raise RuntimeError(f"forced-round HLO source audit started with JAX: {loaded}")

    builder_raw = _snapshot(BUILDER)
    analyzer_raw = _snapshot(ANALYZER)
    auditor_raw = _snapshot(AUDITOR)
    predecessor_raw = _snapshot(SOURCE_DESIGN_ARTIFACT)
    topology_raw = _snapshot(TOPOLOGY_AUTHORITY)
    rmsnorm_raw = _snapshot(RMSNORM_MODULE)
    if sha256(predecessor_raw).hexdigest() != SOURCE_DESIGN_SHA256:
        raise RuntimeError("forced-round source-design predecessor drifted")
    if sha256(topology_raw).hexdigest() != TOPOLOGY_AUTHORITY_SHA256:
        raise RuntimeError("forced-round topology authority drifted")
    predecessor = json.loads(predecessor_raw)
    topology = json.loads(topology_raw)
    if (
        predecessor.get("status") != "SOURCE_DESIGN_CPU_EXACT_PERSISTENCE_ONLY"
        or predecessor.get("authorization", {}).get("tpu_compile_or_hlo_acquisition")
        is not False
        or predecessor.get("physical_cause_claim") is not False
    ):
        raise RuntimeError("forced-round source-design authority drifted")
    forced_function_authority = _verify_predecessor_function_authority(
        predecessor, rmsnorm_raw
    )
    expected_stage_zero = _verify_topology_authority(topology)
    repository_authority = _verify_repository_authority()

    audit_forced_round_pp16_hlo_source = _load_snapshot_auditor(auditor_raw)

    if loaded := _loaded_jax_modules():
        raise RuntimeError(f"forced-round HLO source audit imported JAX: {loaded}")

    report = {
        "artifact_kind": "gate_d_forced_round_pp16_hlo_source_integration",
        "authorization": {
            "full_8k": False,
            "hlo_acquisition": False,
            "persistence_only": True,
            "tpu_compile": False,
            "tpu_execution": False,
        },
        "candidate_id": "forced_normalized_bf16_boundary",
        "claim_scope": (
            "Static source and lineage audit only; certificate generation loads "
            "no JAX and performs no lowering, HLO, compile, TPU execution, cloud "
            "write, numerical, performance, or Gate-D closure claim."
        ),
        "classification": (
            "ISOLATED_PP16_SOURCE;FORCED_ROUND_PRIMARY_LINEAGE;"
            "NO_LOWER_COMPILE_EXECUTE_CALLS;PERSISTENCE_ONLY;GATE_D_OPEN"
        ),
        "predecessor": {
            "artifact": str(SOURCE_DESIGN_ARTIFACT.relative_to(WORKTREE)),
            "sha256": SOURCE_DESIGN_SHA256,
        },
        "process_contract": {"loaded_jax_modules": []},
        "repository_authority": repository_authority,
        "runtime_authority": {
            **forced_function_authority,
            "topology_artifact": str(TOPOLOGY_AUTHORITY.relative_to(WORKTREE)),
            "topology_artifact_sha256": TOPOLOGY_AUTHORITY_SHA256,
            "topology_hash": topology["topology_hash"],
            "pp16_plan_hash": topology["pp16_lp2_hash"],
            "pp16_stage_zero": expected_stage_zero,
        },
        "source_audit": audit_forced_round_pp16_hlo_source(builder_raw),
        "source_authority": {
            "analyzer_path": str(ANALYZER.relative_to(WORKTREE)),
            "analyzer_sha256": sha256(analyzer_raw).hexdigest(),
            "auditor_path": str(AUDITOR.relative_to(WORKTREE)),
            "auditor_sha256": sha256(auditor_raw).hexdigest(),
            "builder_path": str(BUILDER.relative_to(WORKTREE)),
            "builder_sha256": sha256(builder_raw).hexdigest(),
        },
        "status": "PP16_HLO_INTEGRATION_SOURCE_PERSISTENCE_ONLY",
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
