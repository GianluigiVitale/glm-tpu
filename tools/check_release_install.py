"""Fresh full dependency installation in a disposable environment, CPU only.

Downloads Python distributions, never model weights or private benchmark data.
Requires an explicitly selected scratch filesystem with ample capacity. Neither
the active environment nor the source checkout is installed into or upgraded.
This validates dependency installation and CPU imports/tests, not TPU deployment.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

REPO = Path(__file__).resolve().parents[1]
STORAGE_FLOOR = 20 << 30
MEMORY_FLOOR = 32 << 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", type=Path, required=True)
    args = parser.parse_args()
    scratch = args.scratch_root.absolute()
    if (
        not scratch.is_dir()
        or any(p.is_symlink() for p in (scratch, *scratch.parents))
        or scratch.is_relative_to(REPO)
        or shutil.disk_usage(scratch).free < STORAGE_FLOOR
    ):
        raise ValueError(
            "scratch directory is not plain, outside source and >=20GiB free"
        )
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if available < MEMORY_FLOOR:
        raise ValueError("full environment check requires >=32GiB host memory headroom")
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("uv is required")
    initial_free = shutil.disk_usage(scratch).free
    records = []
    with TemporaryDirectory(prefix="glm-release-install-", dir=scratch) as td:
        root = Path(td)
        root.chmod(0o700)
        source, venv = root / "source", root / "venv"
        source.mkdir()
        for name in ("pyproject.toml", "README.md", "THIRD_PARTY_NOTICES.md"):
            shutil.copy2(REPO / name, source / name)
        for name in ("glm_tpu", "licenses", "requirements"):
            shutil.copytree(
                REPO / name,
                source / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        env = {
            key: value
            for key, value in os.environ.items()
            if key
            not in (
                "PYTHONPATH",
                "PYTHONHOME",
                "VIRTUAL_ENV",
                "GLM_RELEASE_LOCAL_TOKENIZER",
            )
            and not key.startswith("UV_")
        }
        env.update(
            JAX_PLATFORMS="cpu",
            UV_CACHE_DIR=str(root / "cache"),
            UV_LINK_MODE="hardlink",
            TMPDIR=str(root),
            PYTHONDONTWRITEBYTECODE="1",
        )

        def checked(command, *, cwd=source, timeout=600):
            started = time.monotonic()
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            records.append(
                dict(
                    command=[str(x).replace(str(root), "<temporary>") for x in command],
                    returncode=result.returncode,
                    seconds=round(time.monotonic() - started, 3),
                )
            )
            if result.returncode:
                raise RuntimeError(
                    "isolated environment check failed: "
                    + result.stderr[-2500:]
                    + result.stdout[-2500:]
                )
            return result.stdout

        checked([uv, "venv", "--python", sys.executable, str(venv)])
        python = str(venv / "bin/python")
        # Install every observed package, not just whichever happen to be reached
        # transitively today. The copied project retains the torch-only CPU index.
        checked(
            [
                uv,
                "pip",
                "install",
                "--python",
                python,
                "--constraint",
                "requirements/runtime-observed.txt",
                "--requirement",
                "requirements/runtime-observed.txt",
                ".[runtime,tpu,benchmark,dev]",
            ]
        )
        checked([uv, "pip", "check", "--python", python])
        doctor = json.loads(
            checked(
                [str(venv / "bin/glm-tpu"), "doctor", "--profile", "benchmark"],
                cwd=root,
            )
        )
        imported = json.loads(
            checked(
                [
                    python,
                    "-c",
                    """
import json, importlib.metadata as m
import jax, torch, transformers, safetensors, tokenizers, google.cloud.storage, pyarrow, glm_tpu
assert jax.default_backend() == 'cpu'
assert torch.version.cuda is None and not torch.cuda.is_available()
print(json.dumps(dict(backend=jax.default_backend(), torch=m.version('torch'),
    package_location=glm_tpu.__file__, installed_versions={d.metadata['Name']:d.version for d in m.distributions()})))
""",
                ],
                cwd=root,
            )
        )
        if not Path(imported["package_location"]).is_relative_to(venv):
            raise ValueError("import did not use the independently installed package")
        imported["package_location"] = imported["package_location"].replace(
            str(root), "<temporary>"
        )
        test_output = checked(
            [
                python,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/release/test_cli.py",
                "tests/release/test_user_request.py",
                "tests/release/test_user_worker.py",
                "tests/greenfield/runtime/test_ws32_request_session.py",
                "tests/greenfield/runtime/test_ws32_native_benchmark_runtime.py",
            ],
            cwd=REPO,
            timeout=180,
        )
        report = dict(
            schema="glm_release_fresh_install_v1",
            steps=records,
            source_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            source_dirty=bool(
                subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)
            ),
            constraints_sha256=sha256(
                (REPO / "requirements/runtime-observed.txt").read_bytes()
            ).hexdigest(),
            installed=imported,
            doctor=doctor,
            tests=test_output.strip().splitlines()[-1],
            scratch_used_bytes=initial_free - shutil.disk_usage(scratch).free,
            hardware_execution=False,
            model_weights_accessed=False,
            benchmark_completion=False,
            release_merge_authorized=False,
            active_environment_modified=False,
        )
    # Emit success only after cleanup of this tool's own temporary directory.
    report["temporary_environment_removed"] = not root.exists()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
