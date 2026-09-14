"""Build and install this private wheel in temporary directories, without deps.

Requires setuptools/wheel in the calling Python and uv on PATH. No model import,
network download, active-environment install or TPU access. Not a full deployment test.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import venv
import zipfile


def checked(args: list[str], *, cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stdout[-2000:] + result.stderr[-2000:])
    return result


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required for isolated offline wheel installation")
    with TemporaryDirectory(prefix="glm-release-package-") as td:
        root = Path(td)
        source, out = root / "source", root / "wheel"
        source.mkdir()
        out.mkdir()
        for name in ("pyproject.toml", "README.md", "THIRD_PARTY_NOTICES.md"):
            shutil.copy2(repo / name, source / name)
        shutil.copytree(repo / "licenses", source / "licenses")
        shutil.copytree(repo / "glm_tpu", source / "glm_tpu",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        code = "from setuptools.build_meta import build_wheel; build_wheel(" + repr(str(out)) + ")"
        checked([sys.executable, "-c", code], cwd=source)
        wheels = list(out.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("build did not produce exactly one wheel")
        wheel = wheels[0]
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            for suffix in ("licenses/THIRD_PARTY_NOTICES.md", "licenses/licenses/Apache-2.0.txt",
                           "licenses/licenses/GLM-5.2-MIT.txt"):
                if len([name for name in names if name.endswith(".dist-info/" + suffix)]) != 1:
                    raise RuntimeError("wheel is missing required third-party notices")
            if "glm_tpu/environment.json" not in names or any(
                n.endswith((".safetensors", ".db", ".pyc")) or
                n.startswith(("scripts/", "bench/", "docs/", "reference/")) for n in names
            ):
                raise RuntimeError("wheel contains forbidden payloads or lacks environment metadata")
        isolated = root / "venv"
        venv.EnvBuilder(with_pip=False).create(isolated)
        checked([uv, "pip", "install", "--no-deps", "--offline", "--python",
                 str(isolated / "bin/python"), str(wheel)], cwd=root)
        clean_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        clean_env["JAX_PLATFORMS"] = "cpu"
        command = str(isolated / "bin/glm-tpu")
        info = json.loads(checked([command, "info"], cwd=root, env=clean_env).stdout)
        if info["version"] != "0.1.0a1":
            raise RuntimeError("installed package metadata differs")
        doctor = subprocess.run([command, "doctor"], cwd=root, env=clean_env,
                                capture_output=True, text=True, timeout=20)
        report = json.loads(doctor.stdout)
        if doctor.returncode != 1 or not all(row["status"] == "missing" for row in report["packages"]):
            raise RuntimeError("dependency-free environment incorrectly passed runtime checks")
        print(json.dumps(dict(
            wheel=wheel.name, bytes=wheel.stat().st_size, members=len(names),
            sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
            isolated_install="PASS; --no-deps --offline",
            installed_console="PASS outside checkout", missing_runtime_detection="PASS",
            dependency_installation="NOT TESTED", model_import_or_TPU="none",
        ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
