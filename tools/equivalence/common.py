"""Shared helpers: repository paths, canonical JSON, environment record, CPU child runner,
positional leaf digests. Importing this module never imports JAX."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

HARNESS_REPO = Path(__file__).resolve().parents[2]
# The source tree under test. ``diff`` points it at an extracted baseline tree; everything
# else measures the checkout the harness lives in.
REPO = Path(os.environ.get("GLM_EQUIVALENCE_SOURCE_ROOT") or HARNESS_REPO).resolve()
DATA = HARNESS_REPO / "tests" / "golden" / "data"
CPU_DEVICES = 32
BASELINE_COMMIT = "181c013e84ac7a2d1c2069feaa9e8aa90da51af5"
# The production paths of the baseline commit (S2f archived configs/, S3 moved scripts/ and
# reference/ into glm_tpu): what "production paths equal to 181c013e" compares.
PRODUCTION_PATHS = ("glm_tpu", "scripts", "configs", "reference")


def canonical_json(value: Any) -> str:
    """Hash contract: sorted keys, compact separators, ASCII, no NaN."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256_hex(value: bytes | str) -> str:
    return sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def digest_json(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def require_cpu() -> None:
    """Refuse to run unless JAX is pinned to the CPU platform (never initialize a TPU)."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if os.environ["JAX_PLATFORMS"] != "cpu":
        raise SystemExit("tools.equivalence requires JAX_PLATFORMS=cpu; got " + repr(os.environ["JAX_PLATFORMS"]))
    if "jax" in sys.modules:
        import jax

        if jax.default_backend() != "cpu":
            raise SystemExit("tools.equivalence refuses a non-CPU default backend")


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd or HARNESS_REPO, text=True, stderr=subprocess.DEVNULL
    ).strip()


def source_record() -> dict[str, Any]:
    """Which production tree was measured: HEAD, and whether the production paths of the tree under
    test (``REPO``; an extracted tree when ``GLM_EQUIVALENCE_SOURCE_ROOT`` points elsewhere) equal
    the baseline commit's."""
    try:
        head = git("rev-parse", "HEAD")
        if REPO == HARNESS_REPO:
            same = (
                subprocess.run(
                    ["git", "diff", "--quiet", BASELINE_COMMIT, "--", *PRODUCTION_PATHS], cwd=HARNESS_REPO, check=False
                ).returncode
                == 0
            )
        else:
            same = _tree_equals_baseline(REPO)
    except (OSError, subprocess.CalledProcessError):
        return dict(head=None, production_paths_equal_baseline=None, baseline=BASELINE_COMMIT)
    return dict(head=head, production_paths_equal_baseline=same, baseline=BASELINE_COMMIT)


def _tree_equals_baseline(root: Path) -> bool:
    """Blob-level equality of ``root``'s production paths with the baseline commit's."""
    expected = {}
    for line in git("ls-tree", "-r", BASELINE_COMMIT, "--", *PRODUCTION_PATHS).splitlines():
        meta, path = line.split("\t", 1)
        expected[path] = meta.split()[2]
    files = sorted(
        path
        for name in PRODUCTION_PATHS
        if (root / name).exists()
        for path in (root / name).rglob("*")
        if (path.is_file() or path.is_symlink()) and "__pycache__" not in path.parts
    )
    relative = [str(path.relative_to(root)) for path in files]
    if set(relative) != set(expected):
        return False
    hashes = subprocess.run(
        ["git", "hash-object", "--no-filters", "--stdin-paths"],
        cwd=HARNESS_REPO,
        check=True,
        input="\n".join(str(path) for path in files),
        capture_output=True,
        text=True,
    ).stdout.split()
    return dict(zip(relative, hashes, strict=True)) == expected


def environment() -> dict[str, Any]:
    """Versions and flags the recorded digests are bound to (never host names or paths)."""
    import jax
    import jaxlib
    import ml_dtypes
    import numpy

    return dict(
        jax=jax.__version__,
        jaxlib=jaxlib.__version__,
        numpy=numpy.__version__,
        ml_dtypes=ml_dtypes.__version__,
        python=".".join(map(str, sys.version_info[:3])),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        jax_platforms=os.environ.get("JAX_PLATFORMS", ""),
        device_count=len(jax.devices()),
    )


def static_environment(xla_flags: str = "") -> dict[str, Any]:
    """The same fields as ``environment()`` without importing JAX (for gates recorded by children
    that do not initialize a backend)."""
    from importlib.metadata import PackageNotFoundError, version

    def installed(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    # torch and safetensors write G4's tiny-pack source; recorded for every static gate, bound by G4.
    return dict(
        jax=installed("jax"),
        jaxlib=installed("jaxlib"),
        numpy=installed("numpy"),
        ml_dtypes=installed("ml_dtypes"),
        torch=installed("torch"),
        safetensors=installed("safetensors"),
        python=".".join(map(str, sys.version_info[:3])),
        xla_flags=xla_flags,
        jax_platforms="cpu",
        device_count=None,
    )


def version_mismatch(recorded: dict[str, Any], names: tuple[str, ...] = ("jax", "jaxlib")) -> str | None:
    """A skip reason when an installed package in ``names`` differs from the recorded version
    (a package the record does not name counts as a mismatch), else None."""
    from importlib.metadata import PackageNotFoundError, version

    for name in names:
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        if recorded.get(name) != installed:
            return f"golden data recorded with {name}=={recorded.get(name)}, installed {installed}"
    return None


def child_env(
    devices: int = CPU_DEVICES, extra: dict[str, str] | None = None, source_root: Path | None = None
) -> dict[str, str]:
    root = Path(source_root or REPO)
    env = dict(os.environ)
    env.update(
        JAX_PLATFORMS="cpu",
        PYTHONDONTWRITEBYTECODE="1",
        XLA_FLAGS=f"--xla_force_host_platform_device_count={devices}",
        GLM_EQUIVALENCE_SOURCE_ROOT=str(root),
    )
    path = env.get("PYTHONPATH", "")
    # The tree under test first (glm_tpu), then the harness (tools.equivalence).
    env["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys([str(root), str(HARNESS_REPO), *(p for p in path.split(os.pathsep) if p)])
    )
    if extra:
        env.update(extra)
    return env


def run_child(
    module: str,
    *args: str,
    devices: int = CPU_DEVICES,
    timeout: float = 3600.0,
    extra_env: dict[str, str] | None = None,
    source_root: Path | None = None,
    prefix: tuple[str, ...] = (),
) -> Any:
    """Run ``python -B -m module args`` on a forced CPU mesh; return the last stdout JSON line."""
    scale = float(os.environ.get("GLM_TPU_TEST_TIMEOUT_SCALE", "1"))
    result = subprocess.run(
        [*prefix, sys.executable, "-B", "-m", module, *args],
        cwd=source_root or REPO,
        env=child_env(devices, extra_env, source_root),
        capture_output=True,
        text=True,
        timeout=timeout * scale,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        tail = "\n".join(result.stderr.splitlines()[-40:])
        raise RuntimeError(f"{module} failed (rc={result.returncode}):\n{tail}")
    return json.loads(lines[-1])


def emit(value: Any) -> None:
    """Child-process result: one compact JSON line on stdout (digests and small summaries only)."""
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=1, ensure_ascii=True) + "\n")


# ----------------------------------------------------------------------------- leaf digests
def leaf_digest(value: Any) -> str:
    """``sha256(dtype | shape | raw bytes)`` of one array leaf (host copy, C order)."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    # dtype.name also names the ml_dtypes types (bfloat16, float8_e4m3fn) unambiguously.
    header = f"{array.dtype.name}|{tuple(int(d) for d in array.shape)}|"
    return sha256(header.encode() + array.tobytes()).hexdigest()


def leaf_digests(tree: Any) -> tuple[list[str], list[str]]:
    """Positional digests in pytree-flatten order and their key paths (labels only)."""
    import jax

    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    labels = [jax.tree_util.keystr(path) for path, _ in flat]
    return labels, [leaf_digest(leaf) for _, leaf in flat]


def tree_record(tree: Any, *, labels: bool = False) -> dict[str, Any]:
    """Compact positional record: full digest over the ordered leaf digests, and 16-hex
    per-leaf prefixes for diagnosis. Labels (key paths) are informational only."""
    names, digests = leaf_digests(tree)
    record: dict[str, Any] = dict(
        count=len(digests),
        digest=sha256_hex("\n".join(digests)),
        leaves=[d[:16] for d in digests],
    )
    if labels:
        record["labels"] = names
    return record
