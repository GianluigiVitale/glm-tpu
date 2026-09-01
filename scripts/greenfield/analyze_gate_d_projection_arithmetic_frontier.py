#!/usr/bin/env -S /opt/glm-tpu/gate-d-python-3.12.13-021044895e95/bin/python3.12 -I -S
"""Offline Gate-D key-projection arithmetic frontier analyzer.

The burned PP16 diagnostic proved that the normalized BF16 row is exact and
that the first available divergence is a projection output.  This read-only
program replays the smallest independently useful branch: the 128x6144 key
projection.  It separates generic reduction controls, direct 48x128 tile/lane
associations, and speculative within-lane hierarchies suggested by the
preserved TPU ``T(8,128)`` layout.  It then runs one explicitly CPU-only JAX
key-LayerNorm/RoPE suffix and compares every result with the accepted and
rejected device outputs.

It does not compile for TPU, open accelerator devices, contact cloud services,
write files, authorize a mechanism, or claim Gate-D closure.
"""

from __future__ import annotations

import importlib
import json
import os
import stat
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

WORKTREE = Path("/home/gianl/glm-tpu-gate-d-pp16-numerical")
SOURCE_PATH = "scripts/greenfield/analyze_gate_d_projection_arithmetic_frontier.py"
EXPECTED_BRANCH = "tooling/gate-d-compensated-pp16-numerical"
PYTHON_RUNTIME_ROOT = Path("/opt/glm-tpu/gate-d-python-3.12.13-021044895e95")
PYTHON = PYTHON_RUNTIME_ROOT / "bin/python3.12"
PYTHON_SHA256 = "021044895e95be79dc2f110367607e684119afbc8ce75f6f0eec94844e0acec7"
PYTHON_TREE_SHA256 = "308748a9a3c3758a6b4f233aa5c034e8cb419362dbeafe0322448be40170d616"
JAX_SITE = Path("/opt/glm-tpu/gate-d-jax-site-55233c63939e")
JAX_SITE_TREE_SHA256 = (
    "55233c63939ea28485cdf2f0fc3d9c1d2ce4d9d93aad828e94498d712a26a0df"
)
JAX_SITE_MANIFEST_SHA256 = (
    "ef454caafd2e4ba5da4bc7f7f73bef4e8f157afd6c795a91b09e319961941eff"
)
LIBTPU_SITE = Path("/opt/glm-tpu/gate-d-libtpu-site-db7598c867f3")
LIBTPU_SITE_TREE_SHA256 = (
    "db7598c867f370756813cbf1536ad8ef7b1d9c167975e9e1724bd9b4fee78eca"
)
LIBTPU_SITE_MANIFEST_SHA256 = (
    "d34064f4a0dfcdcd9ec13288ce967ae060a4650b3e86e53bc47e49756c0e97fa"
)
EXPECTED_NATIVE_BEFORE_COUNT = 11
EXPECTED_NATIVE_BEFORE_SHA256 = (
    "557c8ceccf85149652785d8c7096a7521579b29a63ca5c8a2211c4d014a41d4e"
)
EXPECTED_LOADED_SEALED_MODULE_COUNT = 532
EXPECTED_LOADED_SEALED_MODULE_SHA256 = (
    "6190c1f2c2b6e6f934a08897c927e4242b26673ef1cf80330e2824995c245b4f"
)
EXPECTED_NATIVE_AFTER_COUNT = 49
EXPECTED_NATIVE_AFTER_SHA256 = (
    "e3301c7898cefc98156ef5d0c92744f8b0aa2db678503207e3ec993053587e8c"
)
INITIAL_SYS_PATH = (
    str(PYTHON_RUNTIME_ROOT / "lib/python312.zip"),
    str(PYTHON_RUNTIME_ROOT / "lib/python3.12"),
    str(PYTHON_RUNTIME_ROOT / "lib/python3.12/lib-dynload"),
)
CAPSULE_ROOT = Path(
    "/home/gianl/gate-d-runs/greenfield_gate_d_compensated_capsule_20260831T124838Z"
)
CAPSULE_JSON = CAPSULE_ROOT / "capsule.json"
CAPSULE_INPUTS = CAPSULE_ROOT / "candidate-inputs.npz"
CAPSULE_STATE = CAPSULE_ROOT / "candidate-state.npz"
REJECTED_ROOT = Path(
    "/home/gianl/gate-d-runs/"
    "gate_d_forced_round_pp16_numerical_20260901T164835192240185Z"
)
REJECTED_OUTPUTS = REJECTED_ROOT / "outputs.npz"
OPTIMIZED_HLO = REJECTED_ROOT / "hlo/forced_round_pp16_stage0.optimized_hlo.txt"

FILE_SHA256S = {
    CAPSULE_JSON: "5b7ad71f37dbbcda0ee36a9fc0c42a7ca45d619a9e741307386755e68e87c1a4",
    CAPSULE_INPUTS: "dd5f1cbb37b722591531635a21cfa9f1d6d0157997a4f70138900d0000be8b1b",
    CAPSULE_STATE: "68ee47b1fcbf317fd41da51aa26c9ba1a8dafe0e3e95ccbbfa73285f4e46f236",
    REJECTED_OUTPUTS: "8715fb256a1266d86d278c6dae223fffa3ed8cc0ab42fb00929573dc6b35f7cc",
    OPTIMIZED_HLO: "ccd6ffb4909b1bc4dca5a36f106cde4a84304b230161afb484afb5667bb7206c",
}
EXPECTED_WK_WEIGHT_SHA256 = (
    "b2c67e0fdf4d7292494778233e9813d8256f2fb88b6f7376870379832fbab24b"
)
POSITION = 8155
TILE_ROWS = 8
TILE_LANES = 128
CONTRACTION = 6144
CONTRACTION_TILES = 48

INPUT_SCHEMA = {
    "head_weight_bf16_bits": ((2, 16, 6144), "<u2"),
    "key_norm_bias_bf16_bits": ((2, 128), "<u2"),
    "key_norm_weight_bf16_bits": ((2, 128), "<u2"),
    "prompt_cache_bf16_bits": ((2, 16, 256, 128), "<u2"),
    "q_a_norm_bf16_bits": ((2048,), "<u2"),
    "qkv_a_scale_inv": ((32, 48, 82), "<f4"),
    "qkv_a_weight_bits": ((32, 6144, 82), "|u1"),
    "rms_hidden_update_bf16_bits": ((6144,), "<u2"),
    "rms_residual_bf16_bits": ((6144,), "<u2"),
    "rms_weight_bf16_bits": ((6144,), "<u2"),
    "wk_scale_inv": ((2, 1, 48), "<f4"),
    "wk_weight_bits": ((2, 128, 6144), "|u1"),
    "wq_b_scale_inv": ((2, 16, 16), "<f4"),
    "wq_b_weight_bits": ((2, 2048, 2048), "|u1"),
}
STATE_SCHEMA = {
    "cache_history": ((2, 16, 256, 128), "<u2"),
    "current_key": ((128,), "<f4"),
    "event1_positions": ((1, 2048), "<i4"),
    "event1_scores": ((1, 2048), "<f4"),
    "event1_valid_count": ((1,), "<i4"),
    "head_weights": ((2, 1, 32), "<f4"),
    "normalized": ((2, 1, 6144), "<u2"),
    "query": ((2, 1, 32, 128), "<f4"),
    "rms_hidden_update": ((6144,), "<u2"),
    "rms_input": ((6144,), "<f4"),
    "rms_residual": ((6144,), "<u2"),
}
REJECTED_SCHEMA = {
    "contract_valid_owners": ((2,), "|u1"),
    "current_key_owners": ((2, 1, 128), "<f4"),
    "head_weights_owners": ((2, 1, 32), "<f4"),
    "index_cache_owners": ((2, 16, 256, 128), "<u2"),
    "normalized_hidden_owners": ((2, 1, 6144), "<u2"),
    "query_owners": ((2, 1, 32, 128), "<f4"),
    "selected_positions_owners": ((2, 1, 2048), "<i4"),
    "selected_scores_owners": ((2, 1, 2048), "<f4"),
    "valid_counts_owners": ((2, 1), "<i4"),
}

ReductionOrder = Literal["left", "reverse", "balanced"]
ORDERS: tuple[ReductionOrder, ...] = ("left", "reverse", "balanced")

# Third-party numerical modules are deliberately absent at module import.  The
# exact sealed runtime boundary installs them only after authenticating the
# interpreter and complete dependency trees.  Tests inject non-authoritative
# modules explicitly and cannot produce a committed artifact through ``main``.
np: Any = None
ml_dtypes: Any = None
jax: Any = None
jnp: Any = None
FORBIDDEN_PRELOADED_MODULES = (
    "jax",
    "jaxlib",
    "ml_dtypes",
    "numpy",
    "sitecustomize",
    "usercustomize",
)


class ProjectionFrontierError(RuntimeError):
    """Raised when offline authority or arithmetic constraints drift."""


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(value).tobytes(order="C"))


def _canonical_json(value: Any) -> bytes:
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


def _safe_mode(mode: int) -> int:
    return stat.S_IMODE(mode) & ~0o7022


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_capsule_manifest(root: Path, expected_sha256: str) -> frozenset[str]:
    path = root / "CAPSULE_MANIFEST.json"
    raw = _snapshot_regular(path, limit=1 << 20)
    if _sha256(raw) != expected_sha256:
        raise ProjectionFrontierError(f"sealed capsule manifest drifted: {path}")
    try:
        manifest = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectionFrontierError(
            f"sealed capsule manifest is invalid: {path}"
        ) from error
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "allowlist",
            "claim_scope",
            "schema_version",
            "source_entries",
            "source_root",
        }
        or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("claim_scope"), str)
        or not manifest["claim_scope"]
        or not isinstance(manifest.get("source_root"), str)
        or not manifest["source_root"].startswith("/")
        or not isinstance(manifest.get("allowlist"), list)
        or not manifest["allowlist"]
        or not isinstance(manifest.get("source_entries"), dict)
    ):
        raise ProjectionFrontierError(f"sealed capsule manifest schema drifted: {path}")
    allowlist = manifest["allowlist"]
    if (
        len(allowlist) != len(set(allowlist))
        or any(
            not isinstance(name, str)
            or not name
            or name in (".", "..")
            or "/" in name
            or "\\" in name
            for name in allowlist
        )
        or set(allowlist) != set(manifest["source_entries"])
    ):
        raise ProjectionFrontierError(
            f"sealed capsule manifest allowlist drifted: {path}"
        )
    actual = {entry.name for entry in root.iterdir()}
    if actual != {*allowlist, "CAPSULE_MANIFEST.json"}:
        raise ProjectionFrontierError(
            f"sealed capsule top-level catalogue drifted: {root}"
        )
    for name in allowlist:
        record = manifest["source_entries"][name]
        installed = root / name
        if not isinstance(record, dict):
            raise ProjectionFrontierError(
                f"sealed capsule source record drifted: {installed}"
            )
        if record.get("kind") == "directory":
            valid = (
                set(record) == {"kind", "tree_sha256"}
                and _is_sha256(record.get("tree_sha256"))
                and installed.is_dir()
                and not installed.is_symlink()
            )
        elif record.get("kind") == "file":
            valid = (
                set(record) == {"bytes", "kind", "sha256"}
                and isinstance(record.get("bytes"), int)
                and not isinstance(record.get("bytes"), bool)
                and record["bytes"] >= 0
                and _is_sha256(record.get("sha256"))
                and installed.is_file()
                and not installed.is_symlink()
            )
            if valid:
                installed_raw = _snapshot_regular(installed, limit=1 << 30)
                valid = (
                    len(installed_raw) == record["bytes"]
                    and _sha256(installed_raw) == record["sha256"]
                )
        else:
            valid = False
        if not valid:
            raise ProjectionFrontierError(
                f"sealed capsule source record drifted: {installed}"
            )
    return frozenset(allowlist)


def _validate_loaded_capsule_paths(
    records: Sequence[Mapping[str, Any]],
    manifests: Mapping[Path, frozenset[str]],
) -> None:
    for record in records:
        raw_path = record.get("path")
        if not isinstance(raw_path, str):
            raise ProjectionFrontierError("loaded dependency record path drifted")
        path = Path(raw_path)
        for root, allowlist in manifests.items():
            if path == root or not path.is_relative_to(root):
                continue
            relative = path.relative_to(root)
            if not relative.parts or relative.parts[0] not in allowlist:
                raise ProjectionFrontierError(
                    f"loaded dependency escaped capsule allowlist: {path}"
                )


def _validate_record_manifest(
    label: str,
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    expected_sha256: str,
) -> str:
    digest = _sha256(_canonical_json(records))
    if len(records) != expected_count or digest != expected_sha256:
        raise ProjectionFrontierError(f"{label} byte manifest drifted")
    return digest


def _verify_sealed_tree_metadata(root: Path) -> int:
    entries = [
        root,
        *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()),
    ]
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        metadata = entry.lstat()
        if (
            metadata.st_uid != 0
            or metadata.st_gid != 0
            or os.listxattr(entry, follow_symlinks=False)
            or (
                not stat.S_ISLNK(metadata.st_mode)
                and stat.S_IMODE(metadata.st_mode) != _safe_mode(metadata.st_mode)
            )
        ):
            raise ProjectionFrontierError(f"runtime tree is not sealed: {entry}")
        if stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode):
            pass
        elif stat.S_ISLNK(metadata.st_mode):
            try:
                Path(os.path.realpath(entry)).relative_to(root)
            except ValueError as error:
                raise ProjectionFrontierError(
                    f"runtime tree symlink escapes root: {entry}"
                ) from error
        else:
            raise ProjectionFrontierError(f"unsupported runtime tree entry: {entry}")
        if len(relative) > 1 << 20:
            raise ProjectionFrontierError("runtime tree relative path is unsafe")
    return len(entries)


def _native_mapping_records() -> list[dict[str, Any]]:
    paths: dict[Path, tuple[int, int, int]] = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw_path = fields[5]
        if raw_path.startswith("/memfd:"):
            continue
        if raw_path.endswith(" (deleted)"):
            raise ProjectionFrontierError(f"mapped dependency was deleted: {raw_path}")
        if raw_path.startswith(("/dev/accel", "/dev/vfio", "/dev/nvidia")):
            raise ProjectionFrontierError("CPU analyzer mapped an accelerator device")
        try:
            major_hex, minor_hex = fields[3].split(":", 1)
            mapped = (int(major_hex, 16), int(minor_hex, 16), int(fields[4]))
        except ValueError as error:
            raise ProjectionFrontierError(
                "native mapping identity is invalid"
            ) from error
        path = Path(os.path.realpath(raw_path))
        metadata = path.stat()
        named = (os.major(metadata.st_dev), os.minor(metadata.st_dev), metadata.st_ino)
        if (
            mapped != named
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_gid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ProjectionFrontierError(f"unsafe native mapping: {raw_path}")
        previous = paths.setdefault(path, mapped)
        if previous != mapped:
            raise ProjectionFrontierError(f"split native mapping identity: {raw_path}")
    records = []
    for path in sorted(paths, key=lambda item: str(item).encode("utf-8")):
        raw = _snapshot_regular(path, limit=1 << 30)
        records.append({"bytes": len(raw), "path": str(path), "sha256": _sha256(raw)})
    return records


def _reject_preloaded_modules(modules: Mapping[str, Any]) -> None:
    found = sorted(set(FORBIDDEN_PRELOADED_MODULES).intersection(modules))
    if found:
        raise ProjectionFrontierError(
            f"numerical/customization module was preloaded: {','.join(found)}"
        )


def _prepare_sealed_runtime() -> dict[str, Any]:
    global jax, jnp, ml_dtypes, np

    if (
        Path(sys.executable) != PYTHON
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
        or tuple(sys.path) != INITIAL_SYS_PATH
    ):
        raise ProjectionFrontierError(
            "invoke with the exact sealed Python interpreter using -I -S"
        )
    _reject_preloaded_modules(sys.modules)
    python_raw = _snapshot_regular(PYTHON, limit=64 << 20)
    if _sha256(python_raw) != PYTHON_SHA256:
        raise ProjectionFrontierError("sealed Python executable drifted")
    metadata_counts = {
        str(root): _verify_sealed_tree_metadata(root)
        for root in (PYTHON_RUNTIME_ROOT, JAX_SITE, LIBTPU_SITE)
    }
    manifests = {
        JAX_SITE: _parse_capsule_manifest(JAX_SITE, JAX_SITE_MANIFEST_SHA256),
        LIBTPU_SITE: _parse_capsule_manifest(LIBTPU_SITE, LIBTPU_SITE_MANIFEST_SHA256),
    }
    native_before = _native_mapping_records()
    _validate_loaded_capsule_paths(native_before, manifests)
    native_before_sha256 = _validate_record_manifest(
        "runtime prefix native mapping",
        native_before,
        expected_count=EXPECTED_NATIVE_BEFORE_COUNT,
        expected_sha256=EXPECTED_NATIVE_BEFORE_SHA256,
    )

    os.environ.clear()
    os.environ.update(
        {
            "HOME": "/nonexistent",
            "JAX_ENABLE_COMPILATION_CACHE": "0",
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(JAX_SITE), str(LIBTPU_SITE)]
    np = importlib.import_module("numpy")
    ml_dtypes = importlib.import_module("ml_dtypes")
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    if (
        Path(np.__file__).resolve().is_relative_to(JAX_SITE)
        and Path(ml_dtypes.__file__).resolve().is_relative_to(JAX_SITE)
        and Path(jax.__file__).resolve().is_relative_to(JAX_SITE)
        and Path(importlib.import_module("jaxlib").__file__)
        .resolve()
        .is_relative_to(JAX_SITE)
    ) is not True:
        raise ProjectionFrontierError("numerical modules escaped the sealed JAX site")
    return {
        "native_mapping_before_count": len(native_before),
        "native_mapping_before_sha256": native_before_sha256,
        "python_executable": str(PYTHON),
        "python_sha256": PYTHON_SHA256,
        "python_installed_tree_authority_sha256": PYTHON_TREE_SHA256,
        "jax_site_installed_tree_authority_sha256": JAX_SITE_TREE_SHA256,
        "jax_site_manifest_sha256": JAX_SITE_MANIFEST_SHA256,
        "libtpu_site_installed_tree_authority_sha256": LIBTPU_SITE_TREE_SHA256,
        "libtpu_site_manifest_sha256": LIBTPU_SITE_MANIFEST_SHA256,
        "capsule_allowlist_counts": {
            str(root): len(allowlist) for root, allowlist in manifests.items()
        },
        "sealed_tree_metadata_entry_counts": metadata_counts,
        "sys_flags": {"isolated": 1, "no_site": 1, "ignore_environment": 1},
    }


def _finalize_sealed_runtime(before: Mapping[str, Any]) -> dict[str, Any]:
    allowed_roots = (PYTHON_RUNTIME_ROOT, JAX_SITE, LIBTPU_SITE)
    manifests = {
        JAX_SITE: _parse_capsule_manifest(JAX_SITE, JAX_SITE_MANIFEST_SHA256),
        LIBTPU_SITE: _parse_capsule_manifest(LIBTPU_SITE, LIBTPU_SITE_MANIFEST_SHA256),
    }
    module_paths: set[Path] = set()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path.startswith("/"):
            continue
        path = Path(os.path.realpath(raw_path))
        if path == (WORKTREE / SOURCE_PATH).resolve():
            continue
        if not any(path == root or path.is_relative_to(root) for root in allowed_roots):
            raise ProjectionFrontierError(f"loaded module escaped sealed roots: {path}")
        module_paths.add(path)
    module_records = []
    for path in sorted(module_paths, key=lambda item: str(item).encode("utf-8")):
        raw = _snapshot_regular(path, limit=1 << 30)
        module_records.append(
            {"bytes": len(raw), "path": str(path), "sha256": _sha256(raw)}
        )
    _validate_loaded_capsule_paths(module_records, manifests)
    module_sha256 = _validate_record_manifest(
        "loaded sealed module",
        module_records,
        expected_count=EXPECTED_LOADED_SEALED_MODULE_COUNT,
        expected_sha256=EXPECTED_LOADED_SEALED_MODULE_SHA256,
    )
    native_after = _native_mapping_records()
    _validate_loaded_capsule_paths(native_after, manifests)
    native_after_sha256 = _validate_record_manifest(
        "runtime final native mapping",
        native_after,
        expected_count=EXPECTED_NATIVE_AFTER_COUNT,
        expected_sha256=EXPECTED_NATIVE_AFTER_SHA256,
    )
    before_count = before.get("native_mapping_before_count")
    before_sha = before.get("native_mapping_before_sha256")
    if (
        before_count != EXPECTED_NATIVE_BEFORE_COUNT
        or before_sha != EXPECTED_NATIVE_BEFORE_SHA256
    ):
        raise ProjectionFrontierError("runtime prefix authority drifted")
    return {
        "loaded_sealed_module_count": len(module_records),
        "loaded_sealed_module_manifest_sha256": module_sha256,
        "native_mapping_after_count": len(native_after),
        "native_mapping_after_sha256": native_after_sha256,
    }


def _snapshot_regular(path: Path, *, limit: int = 64 << 20) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > limit
        ):
            raise ProjectionFrontierError(f"unsafe authority file: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 << 20):
            chunks.append(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if (
            len(raw) != before.st_size
            or identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ProjectionFrontierError(f"authority file changed: {path}")
        return raw
    finally:
        os.close(descriptor)


def _load_npz(
    raw: bytes, schema: Mapping[str, tuple[tuple[int, ...], str]]
) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(BytesIO(raw), "r") as archive:
        names = [item.filename for item in archive.infolist()]
        expected = {f"{name}.npy" for name in schema}
        if len(names) != len(set(names)) or set(names) != expected:
            raise ProjectionFrontierError("NPZ member catalogue drifted")
        if any(
            item.is_dir() or item.file_size > 64 << 20 for item in archive.infolist()
        ):
            raise ProjectionFrontierError("unsafe NPZ member")
    result: dict[str, np.ndarray] = {}
    with np.load(BytesIO(raw), allow_pickle=False) as values:
        if set(values.files) != set(schema):
            raise ProjectionFrontierError("NPZ array catalogue drifted")
        for name, (shape, dtype) in schema.items():
            value = np.ascontiguousarray(values[name])
            if value.shape != shape or value.dtype.str != dtype:
                raise ProjectionFrontierError(f"NPZ array drifted: {name}")
            result[name] = value
    return result


def _verify_source_committed() -> tuple[str, str]:
    environment = {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }

    def git(*arguments: str) -> bytes:
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(WORKTREE), *arguments],
            check=False,
            capture_output=True,
            env=environment,
            timeout=30,
        )
        if result.returncode:
            raise ProjectionFrontierError("local Git authority check failed")
        return result.stdout

    source = _snapshot_regular(WORKTREE / SOURCE_PATH, limit=2 << 20)
    pin = git("rev-parse", "HEAD").decode("ascii").strip()
    branch = git("branch", "--show-current").decode("ascii").strip()
    if branch != EXPECTED_BRANCH or source != git("show", f"{pin}:{SOURCE_PATH}"):
        raise ProjectionFrontierError("analyzer source is not committed on its branch")
    return pin, _sha256(source)


def _verify_hlo(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8")
    required = (
        (
            "%fused_computation.21 (param_0.274: f32[1,128,6144], "
            "param_1.369: f32[6144]) -> (f32[], f32[128]) {"
        ),
        "%multiply.33 = f32[128,6144]{1,0:T(8,128)} multiply(",
        (
            "%dot_general.50.clone.1 = f32[128]{0:T(128)S(3)} "
            "reduce(%multiply.33, %constant.263.clone.3), dimensions={1}"
        ),
    )
    if any(text.count(item) != 1 for item in required):
        raise ProjectionFrontierError("optimized-HLO key projection signature drifted")
    start = text.index(required[0])
    end = text.index("\n}\n", start) + 3
    projection_slice = text[start:end].encode("utf-8")
    return {
        "contraction": CONTRACTION,
        "contraction_tiles": CONTRACTION_TILES,
        "hlo_f32_multiply_reduce": True,
        "layout": "T(8,128)",
        "projection_slice_sha256": _sha256(projection_slice),
        "tile_lanes": TILE_LANES,
        "tile_rows": TILE_ROWS,
    }


def _reduce_last(value: np.ndarray, order: ReductionOrder) -> np.ndarray:
    work = np.ascontiguousarray(value, dtype=np.float32)
    if work.shape[-1] < 1:
        raise ValueError("reduction axis must be non-empty")
    if order in ("left", "reverse"):
        indices = range(work.shape[-1])
        if order == "reverse":
            indices = range(work.shape[-1] - 1, -1, -1)
        accumulator = np.zeros(work.shape[:-1], dtype=np.float32)
        for index in indices:
            accumulator = np.add(accumulator, work[..., index])
        return np.ascontiguousarray(accumulator, dtype=np.float32)
    if order != "balanced":
        raise ValueError(f"unknown reduction order: {order}")
    while work.shape[-1] > 1:
        pairs = work.shape[-1] // 2
        reduced = np.add(work[..., 0 : 2 * pairs : 2], work[..., 1 : 2 * pairs : 2])
        if work.shape[-1] % 2:
            reduced = np.concatenate((reduced, work[..., -1:]), axis=-1)
        work = np.ascontiguousarray(reduced, dtype=np.float32)
    return np.ascontiguousarray(work[..., 0], dtype=np.float32)


def _variant_names() -> tuple[str, ...]:
    names = [f"linear_k:{order}" for order in ORDERS]
    for inner in ORDERS:
        for outer in ORDERS:
            names.append(f"lane_first:{inner}:{outer}")
            names.append(f"tile_first:{inner}:{outer}")
    for lane_shape in ("8x16", "16x8"):
        for outer in ORDERS:
            names.append(f"lane_hierarchy:{lane_shape}:balanced:balanced:{outer}")
    if len(names) != len(set(names)):
        raise AssertionError("projection variant names are not unique")
    return tuple(names)


def _variant_class(name: str) -> str:
    if name.startswith("linear_k:"):
        return "generic_linear_control"
    if name.startswith(("lane_first:", "tile_first:")):
        return "direct_48x128_tile_lane_association"
    if name.startswith("lane_hierarchy:"):
        return "speculative_within_lane_hierarchy"
    raise ValueError(f"unknown projection probe: {name}")


def _explicit_projection_variants(
    normalized: np.ndarray, weight: np.ndarray
) -> dict[str, np.ndarray]:
    if normalized.shape != (CONTRACTION,) or weight.shape != (128, CONTRACTION):
        raise ValueError("key projection geometry drifted")
    products = np.multiply(weight, normalized[None, :], dtype=np.float32)
    tiled = products.reshape(128, CONTRACTION_TILES, TILE_LANES)
    variants: dict[str, np.ndarray] = {}
    for order in ORDERS:
        variants[f"linear_k:{order}"] = _reduce_last(products, order)
    for inner in ORDERS:
        for outer in ORDERS:
            lane_partial = _reduce_last(tiled, inner)
            variants[f"lane_first:{inner}:{outer}"] = _reduce_last(lane_partial, outer)
            tile_partial = _reduce_last(np.swapaxes(tiled, 1, 2), inner)
            variants[f"tile_first:{inner}:{outer}"] = _reduce_last(tile_partial, outer)
    for major, minor in ((8, 16), (16, 8)):
        lanes = tiled.reshape(128, CONTRACTION_TILES, major, minor)
        within_minor = _reduce_last(lanes, "balanced")
        within_tile = _reduce_last(within_minor, "balanced")
        for outer in ORDERS:
            name = f"lane_hierarchy:{major}x{minor}:balanced:balanced:{outer}"
            variants[name] = _reduce_last(within_tile, outer)
    if tuple(variants) != _variant_names():
        raise AssertionError("projection variant catalogue drifted")
    return variants


def _accelerator_fds() -> tuple[str, ...]:
    observed: list[str] = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.startswith(("/dev/accel", "/dev/vfio", "/dev/nvidia")):
            observed.append(target)
    return tuple(sorted(observed))


def _cpu_suffix_and_control(
    projections: Mapping[str, np.ndarray],
    normalized: np.ndarray,
    weight: np.ndarray,
    key_norm_weight: np.ndarray,
    key_norm_bias: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    if os.environ.get("JAX_PLATFORMS") not in (None, "cpu"):
        raise ProjectionFrontierError("JAX_PLATFORMS must be unset or exactly cpu")
    if os.environ.get("JAX_PLATFORM_NAME") not in (None, "cpu"):
        raise ProjectionFrontierError("JAX_PLATFORM_NAME must be unset or exactly cpu")
    if any(module is None for module in (jax, jnp, ml_dtypes, np)):
        raise ProjectionFrontierError("sealed numerical runtime was not prepared")
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    before_fds = _accelerator_fds()
    if before_fds:
        raise ProjectionFrontierError(
            "accelerator device already open before CPU replay"
        )

    jax.config.update("jax_enable_compilation_cache", False)
    devices = tuple(jax.devices())
    if (
        jax.default_backend() != "cpu"
        or not devices
        or any(device.platform != "cpu" for device in devices)
    ):
        raise ProjectionFrontierError("projection analyzer did not force CPU-only JAX")

    names = tuple(projections)

    @jax.jit
    def suffix(projected: Any) -> Any:
        mean = jnp.mean(projected, axis=-1, keepdims=True)
        centered = projected - mean
        variance = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
        normalized_key = centered / jnp.sqrt(variance + jnp.float32(1e-6))
        keys = normalized_key * jnp.asarray(key_norm_weight) + jnp.asarray(
            key_norm_bias
        )
        frequencies = jnp.power(
            jnp.float32(8_000_000.0),
            -jnp.arange(0, 64, 2, dtype=jnp.float32) / jnp.float32(64),
        )
        angles = jnp.float32(POSITION) * frequencies
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)
        rotary = keys[..., :64]
        first = rotary[..., 0::2]
        second = rotary[..., 1::2]
        rotated = jnp.stack(
            (first * cos - second * sin, second * cos + first * sin), axis=-1
        ).reshape(rotary.shape)
        return jnp.concatenate((rotated, keys[..., 64:]), axis=-1)

    keys = np.stack(
        [
            np.asarray(suffix(jnp.asarray(projections[name][None, :])))[0]
            for name in names
        ]
    ).astype(np.float32)
    with jax.default_matmul_precision("highest"):
        control_projection = jax.lax.dot_general(
            jnp.asarray(normalized[None, :]),
            jnp.asarray(weight),
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
    control_key = np.asarray(suffix(control_projection))[0].astype(np.float32)
    after_fds = _accelerator_fds()
    if after_fds:
        raise ProjectionFrontierError("CPU replay opened an accelerator device")
    return (
        {name: np.ascontiguousarray(keys[index]) for index, name in enumerate(names)},
        np.ascontiguousarray(np.asarray(control_projection)[0], dtype=np.float32),
        np.ascontiguousarray(control_key),
        {
            "accelerator_fds_after": list(after_fds),
            "accelerator_fds_before": list(before_fds),
            "backend": jax.default_backend(),
            "device_count": len(devices),
            "device_kinds": sorted({str(device.device_kind) for device in devices}),
            "jax": str(jax.__version__),
            "jaxlib": str(jax.lib.__version__),
            "persistent_compilation_cache_enabled": bool(
                jax.config.jax_enable_compilation_cache
            ),
        },
    )


def _metrics(value: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if value.shape != reference.shape or value.dtype != np.float32:
        raise ValueError("comparison geometry or dtype drifted")
    mismatches = value.view(np.uint32) != reference.view(np.uint32)
    error = np.abs(value.astype(np.float64) - reference.astype(np.float64))
    nonrotary = mismatches[64:]
    return {
        "bit_mismatch_count": int(np.count_nonzero(mismatches)),
        "first_bit_mismatch": (
            int(np.flatnonzero(mismatches)[0]) if np.any(mismatches) else None
        ),
        "max_abs_error": float(np.max(error)),
        "mean_abs_error": float(np.mean(error)),
        "nonrotary_bit_mismatch_count": int(np.count_nonzero(nonrotary)),
        "sha256": _array_sha256(value),
    }


def _decode_wk(inputs: Mapping[str, np.ndarray]) -> np.ndarray:
    lookup_values: list[float] = []
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        if exponent == 0:
            value = mantissa * (2.0**-9)
        elif exponent == 15 and mantissa == 7:
            value = float("nan")
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        lookup_values.append(sign * value)
    lookup = np.asarray(lookup_values, dtype=np.float32)
    bits = inputs["wk_weight_bits"]
    scales = inputs["wk_scale_inv"]
    rows = np.arange(bits.shape[-2]) // 128
    columns = np.arange(bits.shape[-1]) // 128
    expanded = scales[..., rows[:, None], columns[None, :]]
    decoded = lookup[bits.astype(np.int32)] * expanded.astype(np.float32)
    bf16 = np.ascontiguousarray(decoded, dtype=ml_dtypes.bfloat16)
    weight = np.ascontiguousarray(bf16.astype(np.float32), dtype=np.float32)
    if weight.shape != (2, 128, CONTRACTION) or not np.all(np.isfinite(weight)):
        raise ProjectionFrontierError("derived key weight drifted")
    if _array_sha256(weight) != EXPECTED_WK_WEIGHT_SHA256:
        raise ProjectionFrontierError("derived key weight identity drifted")
    return weight


def analyze() -> dict[str, Any]:
    sealed_runtime = _prepare_sealed_runtime()
    code_pin, source_sha256 = _verify_source_committed()
    snapshots: dict[Path, bytes] = {}
    for path, expected in FILE_SHA256S.items():
        raw = _snapshot_regular(path)
        if _sha256(raw) != expected:
            raise ProjectionFrontierError(f"authority SHA-256 drifted: {path}")
        snapshots[path] = raw
    capsule = json.loads(snapshots[CAPSULE_JSON])
    if not isinstance(capsule, Mapping):
        raise ProjectionFrontierError("capsule JSON drifted")
    inputs = _load_npz(snapshots[CAPSULE_INPUTS], INPUT_SCHEMA)
    accepted = _load_npz(snapshots[CAPSULE_STATE], STATE_SCHEMA)
    rejected = _load_npz(snapshots[REJECTED_OUTPUTS], REJECTED_SCHEMA)
    hlo = _verify_hlo(snapshots[OPTIMIZED_HLO])

    if not np.array_equal(accepted["normalized"][0], accepted["normalized"][1]):
        raise ProjectionFrontierError("accepted normalized owners disagree")
    if not np.array_equal(accepted["normalized"], rejected["normalized_hidden_owners"]):
        raise ProjectionFrontierError("accepted and rejected normalized rows differ")
    if not np.array_equal(
        rejected["current_key_owners"][0], rejected["current_key_owners"][1]
    ):
        raise ProjectionFrontierError("rejected current-key owners disagree")

    normalized = (
        accepted["normalized"][0, 0].view(ml_dtypes.bfloat16).astype(np.float32)
    )
    weight = _decode_wk(inputs)[0]
    variants = _explicit_projection_variants(normalized, weight)
    key_norm_weight = (
        inputs["key_norm_weight_bf16_bits"][0]
        .view(ml_dtypes.bfloat16)
        .astype(np.float32)
    )
    key_norm_bias = (
        inputs["key_norm_bias_bf16_bits"][0].view(ml_dtypes.bfloat16).astype(np.float32)
    )
    variant_keys, control_projection, control_key, jax_runtime = (
        _cpu_suffix_and_control(
            variants,
            normalized,
            weight,
            key_norm_weight,
            key_norm_bias,
        )
    )
    runtime = {
        **sealed_runtime,
        **jax_runtime,
        **_finalize_sealed_runtime(sealed_runtime),
    }
    accepted_key = accepted["current_key"]
    rejected_key = rejected["current_key_owners"][0, 0]
    control = {
        "accepted": _metrics(control_key, accepted_key),
        "projection_sha256": _array_sha256(control_projection),
        "rejected": _metrics(control_key, rejected_key),
    }
    records = []
    for name in _variant_names():
        projection = variants[name]
        key = variant_keys[name]
        records.append(
            {
                "accepted": _metrics(key, accepted_key),
                "name": name,
                "probe_class": _variant_class(name),
                "projection_sha256": _array_sha256(projection),
                "rejected": _metrics(key, rejected_key),
            }
        )
    records.sort(
        key=lambda record: (
            record["accepted"]["bit_mismatch_count"],
            record["accepted"]["max_abs_error"],
            record["name"],
        )
    )
    best = records[0]
    exact = [
        record["name"]
        for record in records
        if record["accepted"]["bit_mismatch_count"] == 0
    ]
    if control["accepted"]["bit_mismatch_count"] == 0:
        classification = (
            "CPU_F32_DOT_CONTROL_EXACT_ACCEPTED_KEY_CAPTURED_INPUT;"
            "ENUMERATED_REDUCTION_PROBES_REJECTED;"
            "TPU_CAUSALITY_UNPROVEN;GATE_D_OPEN"
        )
        exact_next = (
            "Bind the exact CPU projection witness, then review the smallest "
            "default-off TPU F32 projection-contraction discriminator; do not "
            "rerun the full DSA path or 8K decoder."
        )
    elif exact:
        classification = (
            "OFFLINE_LAYOUT_ASSOCIATION_MATCHES_ACCEPTED_KEY;"
            "TPU_CAUSALITY_UNPROVEN;GATE_D_OPEN"
        )
        exact_next = (
            "Review a default-off source mechanism for the matching association, "
            "then require compile-only HLO before one bounded TPU discriminator."
        )
    elif (
        best["accepted"]["bit_mismatch_count"]
        < control["accepted"]["bit_mismatch_count"]
    ):
        classification = (
            "LAYOUT_ASSOCIATION_IMPROVES_ACCEPTED_MATCH;ROOT_CAUSE_UNPROVEN;GATE_D_OPEN"
        )
        exact_next = (
            "Adjudicate the best layout association against the preserved TPU "
            "reduction lowering before designing a source mechanism."
        )
    else:
        classification = (
            "NO_LAYOUT_ASSOCIATION_BEATS_CPU_DOT_CONTROL;ROOT_CAUSE_UNPROVEN;"
            "GATE_D_OPEN"
        )
        exact_next = (
            "Do not implement a guessed reduction tree; isolate the preserved "
            "optimized-HLO reduction/codegen or capture one bounded projection "
            "output under a separately reviewed mechanism."
        )
    return {
        "artifact_kind": "gate_d_projection_arithmetic_frontier_analysis",
        "authority": {
            "accepted_inputs_sha256": FILE_SHA256S[CAPSULE_INPUTS],
            "accepted_state_sha256": FILE_SHA256S[CAPSULE_STATE],
            "capsule_sha256": FILE_SHA256S[CAPSULE_JSON],
            "optimized_hlo_sha256": FILE_SHA256S[OPTIMIZED_HLO],
            "rejected_outputs_sha256": FILE_SHA256S[REJECTED_OUTPUTS],
        },
        "best_explicit_reduction_probe": best,
        "claim_scope": (
            "One sealed layer-1 position-8155 accepted/rejected current-key "
            "witness; CPU arithmetic discriminator only."
        ),
        "classification": classification,
        "code_pin": code_pin,
        "cpu_jax_dot_control": control,
        "exact_accepted_explicit_reduction_probes": exact,
        "exact_next": exact_next,
        "full_dsa_or_8k_authorized": False,
        "gate_d_closed": False,
        "hlo_contract": hlo,
        "performance_claim": False,
        "projection_mechanism_authorized": False,
        "projection_product_contract": {
            "accumulation_dtype": "float32",
            "input_values_are_bf16_derived_float32": True,
            "product_dtype": "float32",
            "product_rounding_controls_executed": [],
            "reason": "optimized HLO pins f32 multiply and f32 reduce",
        },
        "runtime": runtime,
        "root_cause_proven": False,
        "schema_version": 1,
        "source_sha256": source_sha256,
        "tpu_compile_or_execution_performed": False,
        "variant_count": len(records),
        "variant_classes": {
            probe_class: [
                name for name in _variant_names() if _variant_class(name) == probe_class
            ]
            for probe_class in (
                "generic_linear_control",
                "direct_48x128_tile_lane_association",
                "speculative_within_lane_hierarchy",
            )
        },
        "variant_projection_unique_sha256_count": len(
            {record["projection_sha256"] for record in records}
        ),
        "variants_ranked_by_accepted_key": records,
    }


def main(arguments: Sequence[str] | None = None) -> int:
    if list(arguments if arguments is not None else sys.argv[1:]):
        raise ProjectionFrontierError("projection analyzer accepts no arguments")
    print(_canonical_json(analyze()).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
