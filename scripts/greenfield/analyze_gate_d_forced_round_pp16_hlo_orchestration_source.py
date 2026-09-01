#!/usr/bin/env python3
"""Emit the persistence-only source certificate for forced-round HLO orchestration."""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
ANALYZER = WORKTREE / (
    "scripts/greenfield/analyze_gate_d_forced_round_pp16_hlo_orchestration_source.py"
)
PUBLISHER = WORKTREE / ("scripts/greenfield/publish_gate_d_forced_round_pp16_hlo.py")
WRAPPER = WORKTREE / "scripts/greenfield/run_gate_d_forced_round_pp16_hlo.sh"
LAUNCHER = WORKTREE / "scripts/greenfield/launch_gate_d_forced_round_pp16_hlo.py"
TEST = WORKTREE / (
    "tests/greenfield/validation/test_publish_gate_d_forced_round_pp16_hlo.py"
)
LAUNCHER_TEST = WORKTREE / (
    "tests/greenfield/validation/test_launch_gate_d_forced_round_pp16_hlo.py"
)
HISTORICAL_ACQUISITION_TEST = WORKTREE / (
    "tests/greenfield/validation/test_gate_d_forced_round_pp16_hlo_acquisition.py"
)
BASE_PUBLISHER = WORKTREE / (
    "scripts/greenfield/publish_gate_d_compensated_pp16_hlo.py"
)
ACQUISITION_SOURCE = WORKTREE / (
    "docs/artifacts/gate-d-forced-round-pp16-hlo-acquisition-source.json"
)
FORCED_ROUND_SOURCE = WORKTREE / (
    "docs/artifacts/gate-d-forced-round-pp16-hlo-source.json"
)
ADMISSION = WORKTREE / (
    "docs/artifacts/gate-d-precompile-admission-v2-compensated-capsule.json"
)
TOPOLOGY = WORKTREE / "docs/artifacts/gate-d-runtime-locality-authority.json"
MIRROR_VERIFIER = WORKTREE / (
    "scripts/greenfield/verify_gate_d_same_region_git_mirror.py"
)
BASE_CODE_PIN = "2f2408d7f63c747beaa8776aa6faa8b1872478ea"
BASE_TREE_ID = "3541fbf1b9e8861705c77ff7f6128c65d2ecee4f"
HISTORICAL_CODE_PIN = "a012b93fdbd7c6fe1f84db2260708ba55b38e8f6"
BASE_PUBLISHER_SHA256 = (
    "75c296a2b46aef1b878a95ee7b1dfabdaf062687bfc496416ffca102f1180ee3"
)
LAUNCHER_AST_SHA256 = "3af41e75575847bf9ed1d540f76f46a350b7078a0249aa65797064914549856d"
WRAPPER_RUNTIME_BOUNDARY_SHA256 = (
    "a25743cd922852287f8185f33954f95397ea9814651d3d24295c9f4d60d783f0"
)
ACQUISITION_SOURCE_SHA256 = (
    "efe04d98267fc5265952b28ee386a4894d028b39f2d62d0a1554f24591e73196"
)
FORCED_ROUND_SOURCE_SHA256 = (
    "518be87b650729d365dd09aba20b5f5a03d4bcddccc81978357cdbb02cbe02c6"
)
ADMISSION_SHA256 = "7cd7e569ed9ed5fd978d933efd4229d65906ae28312e264863b8336e4cc6b37d"
TOPOLOGY_SHA256 = "49cf6bb1a553985855556d1401ad85918669df52d12f8dc5150f247d18b325eb"
MIRROR_VERIFIER_SHA256 = (
    "091208165a149989f14c5c9b9d1cbe7ff20537e2c81b16319eea9603984e859b"
)
PUBLISHER_HELPER_AST_SHA256S = {
    "_observed_names": (
        "b798360b079178dc374d6b9ca3c2ddbdddb4931acbddd52b34c90b435ca1ea08"
    ),
    "_forced_round_source_authority": (
        "8a40e6cf886a5ae6724176477e2e8b2810e4799373591a1c62b983330523c663"
    ),
    "_prepare_success": (
        "d57f67ebf9f9c65875b82fef5af70dfe114e26927371b93a76b32201e473118a"
    ),
    "_require_never_used_prefix": (
        "5275699c51cd2fe7070aa46b677ef66fbc7613b7e9d93302aa23f13f7bf25d54"
    ),
    "_validate_compile_host_authority": (
        "5eeff7e4953d48dd8274e804f61f6bb157726a1d0dcf678e23f0126b12d307fe"
    ),
    "_validate_mirror_replay": (
        "16a9ea8cbda3fe7ba56713648d0c887540b97a65292bcf736f1b234c1e3c5042"
    ),
    "_validate_remote_vacancy_evidence": (
        "ff7dd46456ee84e0837acdd80fddff080396e1436d770af1fe26daa009441f7e"
    ),
    "publish_diagnostic": (
        "ff11f44796190cab2ef07542831b8cf33d5478e8613ab1b6ceda9693c42bdaa2"
    ),
    "publish_success": (
        "30cda4d650f0086e29feecb7653151e1fb2580ef17f82a580c8cb4dedb416173"
    ),
}
PUBLISHER_NEW_HELPERS = frozenset(
    {
        "_forced_round_source_authority",
        "_require_never_used_prefix",
        "_validate_compile_host_authority",
        "_validate_mirror_replay",
        "_validate_remote_vacancy_evidence",
    }
)
PUBLISHER_CHANGED_COMMON = frozenset(
    {"_observed_names", "_prepare_success", "publish_diagnostic", "publish_success"}
)
ARTIFACT_PATH = "docs/artifacts/gate-d-forced-round-pp16-hlo-orchestration-source.json"
ALLOWED_DELTA_PATHS = frozenset(
    {
        "HANDOFF.md",
        "docs/RESEARCH_LOG.md",
        ARTIFACT_PATH,
        "docs/greenfield/EVIDENCE_MAP.md",
        (
            "scripts/greenfield/"
            "analyze_gate_d_forced_round_pp16_hlo_orchestration_source.py"
        ),
        "scripts/greenfield/launch_gate_d_forced_round_pp16_hlo.py",
        "scripts/greenfield/publish_gate_d_forced_round_pp16_hlo.py",
        "scripts/greenfield/run_gate_d_forced_round_pp16_hlo.sh",
        (
            "tests/greenfield/validation/"
            "test_gate_d_forced_round_pp16_hlo_acquisition.py"
        ),
        ("tests/greenfield/validation/test_launch_gate_d_forced_round_pp16_hlo.py"),
        ("tests/greenfield/validation/test_publish_gate_d_forced_round_pp16_hlo.py"),
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
    try:
        relative = path.relative_to(WORKTREE).as_posix()
    except ValueError as error:
        raise RuntimeError(f"orchestration source escapes worktree: {path}") from error
    return _git(["show", f"{HISTORICAL_CODE_PIN}:{relative}"])


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
            "forced-round orchestration repository authority failed: "
            f"returncode={result.returncode} stderr={result.stderr!r}"
        )
    return result.stdout


def _reject_replace_refs() -> None:
    if _git(["for-each-ref", "--format=%(refname)", "refs/replace"]):
        raise RuntimeError("orchestration repository has forbidden replacement refs")


def _nul_paths(raw: bytes) -> set[str]:
    if not raw:
        return set()
    if not raw.endswith(b"\0"):
        raise RuntimeError("orchestration repository path stream is not terminated")
    paths = [item.decode("utf-8", errors="strict") for item in raw[:-1].split(b"\0")]
    if len(paths) != len(set(paths)) or any(not path for path in paths):
        raise RuntimeError("orchestration repository path stream is ambiguous")
    return set(paths)


def _verify_repository_authority() -> dict[str, Any]:
    _reject_replace_refs()
    if _git(["rev-parse", f"{BASE_CODE_PIN}^{{tree}}"]) != f"{BASE_TREE_ID}\n".encode(
        "ascii"
    ):
        raise RuntimeError("forced-round orchestration base tree drifted")
    _git(["merge-base", "--is-ancestor", BASE_CODE_PIN, HISTORICAL_CODE_PIN])
    tracked = _nul_paths(
        _git(
            [
                "diff",
                "--no-ext-diff",
                "--name-only",
                "-z",
                BASE_CODE_PIN,
                HISTORICAL_CODE_PIN,
                "--",
            ]
        )
    )
    untracked: set[str] = set()
    if tracked & untracked or tracked | untracked != ALLOWED_DELTA_PATHS:
        raise RuntimeError(
            "forced-round orchestration repository delta drifted: "
            f"tracked={sorted(tracked)!r} untracked={sorted(untracked)!r}"
        )
    return {
        "allowed_delta_paths": sorted(ALLOWED_DELTA_PATHS),
        "base_code_pin": BASE_CODE_PIN,
        "base_tree_id": BASE_TREE_ID,
        "unexpected_delta_paths": [],
    }


def _function_map(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


class _PublisherNormalizer(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if not isinstance(node.value, str):
            return node
        value = node.value
        for source, replacement in (
            ("/home/gianl/glm-tpu-gate-d-pp16-numerical", "/WORKTREE"),
            ("/home/gianl/glm-tpu-topology-rewrite", "/WORKTREE"),
            ("gate_d_forced_round_pp16", "gate_d_PROFILE_pp16"),
            ("gate_d_compensated_pp16", "gate_d_PROFILE_pp16"),
            ("forced_round_pp16_stage0", "PROFILE_pp16_stage0"),
            ("compensated_pp16_stage0", "PROFILE_pp16_stage0"),
        ):
            value = value.replace(source, replacement)
        return ast.copy_location(ast.Constant(value), node)


def _normalized_functions(raw: bytes) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(raw.decode("utf-8", errors="strict"))
    normalized = _PublisherNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return _function_map(normalized)


def _audit_publisher(raw: bytes, base_raw: bytes) -> dict[str, Any]:
    if sha256(base_raw).hexdigest() != BASE_PUBLISHER_SHA256:
        raise RuntimeError("hardened compensated HLO publisher base drifted")
    source = raw.decode("utf-8", errors="strict")
    functions = _normalized_functions(raw)
    base_functions = _normalized_functions(base_raw)
    new_helpers = set(functions) - set(base_functions)
    changed_common = PUBLISHER_CHANGED_COMMON
    if new_helpers != PUBLISHER_NEW_HELPERS or set(base_functions) - set(
        functions
    ):
        raise RuntimeError("forced-round HLO publisher function surface drifted")
    for name in sorted(set(base_functions) - changed_common):
        if ast.dump(functions[name], include_attributes=False) != ast.dump(
            base_functions[name], include_attributes=False
        ):
            raise RuntimeError(f"hardened publisher primitive drifted: {name}")
    original_tree = ast.parse(source)
    original_functions = _function_map(original_tree)
    for name, expected in PUBLISHER_HELPER_AST_SHA256S.items():
        observed = sha256(
            ast.dump(original_functions[name], include_attributes=False).encode()
        ).hexdigest()
        if observed != expected:
            raise RuntimeError(f"forced-round publisher helper drifted: {name}")
    prepare = ast.unparse(original_functions["_prepare_success"])
    required = (
        "_forced_round_source_authority(code_pin)",
        "_validate_mirror_replay(mirror_raw, code_pin)",
        "_validate_compile_host_authority(sync_raw, code_pin)",
        "runner.get('input_spec') != _EXPECTED_INPUT_SPEC",
        "runner.get('output_spec') != _EXPECTED_OUTPUT_SPEC",
        "runner.get('compiled_executable_invocation_count') != 0",
        "runner.get('tpu_numerical_execution_performed') is not False",
    )
    if any(item not in prepare for item in required):
        raise RuntimeError("forced-round publisher claim boundary drifted")
    if (
        "gate_d_forced_round_pp16_hlo/" not in source
        or "gate_d_compensated_pp16_hlo/" in source
        or "/home/gianl/glm-tpu-topology-rewrite" in source
    ):
        raise RuntimeError("forced-round publisher namespace drifted")
    return {
        "base_publisher_sha256": BASE_PUBLISHER_SHA256,
        "hardened_common_function_count": len(base_functions) - 1,
        "new_helper_count": len(new_helpers),
        "changed_common_function_count": len(changed_common),
        "publisher_ast_sha256": sha256(
            ast.dump(original_tree, include_attributes=False).encode()
        ).hexdigest(),
        "publisher_sha256": sha256(raw).hexdigest(),
        "runner_input_spec_count": 16,
        "runner_output_spec_count": 9,
    }


def _assignment(source: str, name: str) -> str:
    matches = re.findall(
        rf"^readonly {re.escape(name)}=([^\n]+)$", source, re.MULTILINE
    )
    if len(matches) != 1:
        raise RuntimeError(f"forced-round wrapper assignment drifted: {name}")
    return matches[0]


def _audit_launcher(
    raw: bytes, wrapper_raw: bytes, expected: dict[str, str]
) -> dict[str, Any]:
    source = raw.decode("utf-8", errors="strict")
    tree = ast.parse(source)
    observed_ast = sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    if observed_ast != LAUNCHER_AST_SHA256:
        raise RuntimeError("forced-round descriptor launcher AST drifted")
    required = (
        'INSTALL_PATH = Path("/opt/glm-tpu/bin/launch_gate_d_forced_round_pp16_hlo_v2.py")',
        '"/usr/local/libexec/glm-tpu/gate-d-forced-round-pp16-hlo-v2"',
        "os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING",
        "fcntl.fcntl(descriptor, F_ADD_SEALS, REQUIRED_SEALS)",
        "set(os.environ) != INPUT_ENVIRONMENT_KEYS",
        'f"/proc/self/fd/{WRAPPER_FD}"',
        '"GLM_GATE_D_IMMUTABLE_LOCKS_HELD": "1"',
    )
    forbidden = ("OLDPWD", "shell=True", "os.chdir(")
    if any(item not in source for item in required) or any(
        item in source for item in forbidden
    ):
        raise RuntimeError("forced-round descriptor launcher contract drifted")
    if expected["WRAPPER_SHA256"] != sha256(wrapper_raw).hexdigest():
        raise RuntimeError("forced-round descriptor launcher wrapper pin drifted")
    assignments: dict[str, Any] = {}
    for node in tree.body:
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
            or node.targets[0].id not in expected
        ):
            continue
        assignments[node.targets[0].id] = ast.literal_eval(node.value)
    for name, value in expected.items():
        if assignments.get(name) != value:
            raise RuntimeError(f"forced-round descriptor launcher pin drifted: {name}")
    return {
        "launcher_ast_sha256": observed_ast,
        "launcher_sha256": sha256(raw).hexdigest(),
        "retained_wrapper_fd": 10,
        "retained_lock_fds": [11, 12],
        "wrapper_sha256": expected["WRAPPER_SHA256"],
    }


def _audit_wrapper(raw: bytes, expected: dict[str, str]) -> dict[str, Any]:
    source = raw.decode("utf-8", errors="strict")
    marker = (
        "read -r -d '' RUNTIME_BOUNDARY_VERIFIER "
        "<<'RUNTIME_BOUNDARY_VERIFIER_EOF' || true\n"
    )
    terminal = "RUNTIME_BOUNDARY_VERIFIER_EOF\n"
    verifier = source.split(marker, 1)[1].split(terminal, 1)[0]
    if sha256(verifier.encode()).hexdigest() != WRAPPER_RUNTIME_BOUNDARY_SHA256:
        raise RuntimeError("forced-round wrapper runtime boundary drifted")
    for name, value in expected.items():
        if name == "WRAPPER_SHA256":
            continue
        if _assignment(source, name) != value:
            raise RuntimeError(f"forced-round wrapper pin drifted: {name}")
    required = (
        "GLM_GATE_D_FORCED_ROUND_PP16_HLO_ACQUIRE",
        "GLM_GATE_D_FORCED_ROUND_PP16_MODE",
        "GLM_GATE_D_FORCED_ROUND_PP16_TAG",
        "strict_census pre",
        "strict_census post",
        '"$MIRROR_VERIFIER"',
        "compile_host_only=1 sealed_source_archive=1",
        '--forced-round-source "$SOURCE_CERTIFICATE"',
        '--forced-round-source-sha256 "$SOURCE_CERTIFICATE_SHA"',
        "JAX_ENABLE_COMPILATION_CACHE=0",
        "TPU_VISIBLE_DEVICES=0,1,2,3",
        "HLO_ACQUIRED_UNADJUDICATED",
        "GLM_GATE_D_FORCED_ROUND_HLO_WRAPPER_SANITIZED",
        '/usr/bin/python3 -I -S -B -c "$RUNTIME_BOUNDARY_VERIFIER"',
        "exec 10<&-",
        (
            "readonly IMMUTABLE_CAPSULE_ROOT=/usr/local/libexec/glm-tpu/"
            "gate-d-forced-round-pp16-hlo-v2"
        ),
    )
    forbidden = (
        "WORKER_REPO_VERIFY_SCRIPT",
        "repos/glm-tpu-topology-rewrite",
        "gate_d_pp16_hlo/",
        "decode_8k",
        "block_until_ready",
        "IMMUTABLE_LOCK_BROKER",
        "WRAPPER_ABS",
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
        or source.index('/usr/bin/python3 -I -S -B -c "$RUNTIME_BOUNDARY_VERIFIER"')
        > source.index("cd /")
    ):
        raise RuntimeError("forced-round wrapper compile-only contract drifted")
    return {
        "runtime_boundary_sha256": sha256(verifier.encode()).hexdigest(),
        "mirror_replay_count": source.count('"$MIRROR_VERIFIER"'),
        "protected_compile_process_count": 1,
        "wrapper_sha256": sha256(raw).hexdigest(),
    }


def main() -> int:
    if os.environ.get("GLM_GATE_D_FORCED_ROUND_HLO_ORCHESTRATION_SOURCE") != "1":
        raise RuntimeError("forced-round HLO orchestration source audit is default-off")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError(
            "forced-round HLO orchestration source audit must be CPU-pinned"
        )
    if loaded := _loaded_jax_modules():
        raise RuntimeError(f"orchestration source audit started with JAX: {loaded}")
    snapshots = {
        "analyzer": _snapshot(ANALYZER),
        "launcher": _snapshot(LAUNCHER),
        "launcher_test": _snapshot(LAUNCHER_TEST),
        "publisher": _snapshot(PUBLISHER),
        "wrapper": _snapshot(WRAPPER),
        "test": _snapshot(TEST),
        "historical_acquisition_test": _snapshot(HISTORICAL_ACQUISITION_TEST),
        "base_publisher": _snapshot(BASE_PUBLISHER),
        "acquisition_source": _snapshot(ACQUISITION_SOURCE),
        "forced_round_source": _snapshot(FORCED_ROUND_SOURCE),
        "admission": _snapshot(ADMISSION),
        "topology": _snapshot(TOPOLOGY),
        "mirror_verifier": _snapshot(MIRROR_VERIFIER),
    }
    for name, expected in (
        ("base_publisher", BASE_PUBLISHER_SHA256),
        ("acquisition_source", ACQUISITION_SOURCE_SHA256),
        ("forced_round_source", FORCED_ROUND_SOURCE_SHA256),
        ("admission", ADMISSION_SHA256),
        ("topology", TOPOLOGY_SHA256),
        ("mirror_verifier", MIRROR_VERIFIER_SHA256),
    ):
        if sha256(snapshots[name]).hexdigest() != expected:
            raise RuntimeError(
                f"forced-round orchestration predecessor drifted: {name}"
            )
    acquisition = json.loads(snapshots["acquisition_source"])
    admission = json.loads(snapshots["admission"])
    topology = json.loads(snapshots["topology"])
    if (
        acquisition.get("authorization", {}).get("hlo_acquisition") is not False
        or acquisition.get("authorization", {}).get("tpu_compile") is not False
        or admission.get("compile_only_review_required") is not True
        or admission.get("tpu_successor_authorized") is not False
        or topology.get("tpu_successor_authorized") is not False
    ):
        raise RuntimeError("forced-round orchestration predecessor policy drifted")
    repository = _verify_repository_authority()
    publisher = _audit_publisher(snapshots["publisher"], snapshots["base_publisher"])
    launcher = _audit_launcher(
        snapshots["launcher"],
        snapshots["wrapper"],
        {
            "DRIVER_SHA256": "046040d382567ccef94792b97e160b0f79b673ab1f0f1d46bfd20c57e884faf7",
            "MIRROR_VERIFIER_SHA256": MIRROR_VERIFIER_SHA256,
            "PUBLISHER_SHA256": sha256(snapshots["publisher"]).hexdigest(),
            "WRAPPER_SHA256": sha256(snapshots["wrapper"]).hexdigest(),
        },
    )
    wrapper = _audit_wrapper(
        snapshots["wrapper"],
        {
            "ADMISSION_SHA": ADMISSION_SHA256,
            "TOPOLOGY_SHA": TOPOLOGY_SHA256,
            "SOURCE_CERTIFICATE_SHA": FORCED_ROUND_SOURCE_SHA256,
            "PUBLISHER_SHA": sha256(snapshots["publisher"]).hexdigest(),
            "MIRROR_VERIFIER_SHA": MIRROR_VERIFIER_SHA256,
            "WRAPPER_SHA256": sha256(snapshots["wrapper"]).hexdigest(),
        },
    )
    if loaded := _loaded_jax_modules():
        raise RuntimeError(f"orchestration source audit imported JAX: {loaded}")
    report = {
        "artifact_kind": "gate_d_forced_round_pp16_hlo_orchestration_source",
        "authorization": {
            "cloud_write": False,
            "full_8k": False,
            "hlo_acquisition": False,
            "persistence_only": True,
            "tpu_compile": False,
            "tpu_execution": False,
        },
        "claim_scope": (
            "Static launcher/publisher/wrapper audit only. No JAX import, HLO "
            "acquisition, TPU compile/execution, cloud write, numerical, performance "
            "or Gate-D claim."
        ),
        "orchestration_audit": {
            "launcher": launcher,
            "publisher": publisher,
            "wrapper": wrapper,
        },
        "predecessors": {
            "acquisition_source_sha256": ACQUISITION_SOURCE_SHA256,
            "admission_sha256": ADMISSION_SHA256,
            "forced_round_source_sha256": FORCED_ROUND_SOURCE_SHA256,
            "mirror_verifier_sha256": MIRROR_VERIFIER_SHA256,
            "topology_sha256": TOPOLOGY_SHA256,
        },
        "process_contract": {"loaded_jax_modules": []},
        "repository_authority": repository,
        "source_authority": {
            "analyzer_path": str(ANALYZER.relative_to(WORKTREE)),
            "analyzer_sha256": sha256(snapshots["analyzer"]).hexdigest(),
            "launcher_path": str(LAUNCHER.relative_to(WORKTREE)),
            "launcher_sha256": sha256(snapshots["launcher"]).hexdigest(),
            "launcher_test_path": str(LAUNCHER_TEST.relative_to(WORKTREE)),
            "launcher_test_sha256": sha256(snapshots["launcher_test"]).hexdigest(),
            "publisher_path": str(PUBLISHER.relative_to(WORKTREE)),
            "publisher_sha256": sha256(snapshots["publisher"]).hexdigest(),
            "historical_acquisition_test_path": str(
                HISTORICAL_ACQUISITION_TEST.relative_to(WORKTREE)
            ),
            "historical_acquisition_test_sha256": sha256(
                snapshots["historical_acquisition_test"]
            ).hexdigest(),
            "test_path": str(TEST.relative_to(WORKTREE)),
            "test_sha256": sha256(snapshots["test"]).hexdigest(),
            "wrapper_path": str(WRAPPER.relative_to(WORKTREE)),
            "wrapper_sha256": sha256(snapshots["wrapper"]).hexdigest(),
        },
        "status": "FORCED_ROUND_HLO_ORCHESTRATION_SOURCE_PERSISTENCE_ONLY",
    }
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
