"""One offline CPU-only release check command; never launches a TPU workflow.

Uses the current installed environment without upgrades. Full dependency install,
hardware validation, full benchmark completion and release admission are separate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
TESTS = (
    "tests/release",
    "scripts/analysis/test_parse_xplane.py",
    "tests/greenfield/runtime/test_ws32_request_session.py",
    "tests/greenfield/runtime/test_ws32_native_benchmark_runtime.py",
    "tests/greenfield/validation/test_ws32_native_benchmark_memory.py",
    "tests/greenfield/validation/test_ws32_native_benchmark_launch.py",
    "tests/greenfield/validation/test_ws32_native_benchmark_requests.py",
    "tests/greenfield/validation/test_ws32_native_benchmark_transport.py",
    "tests/greenfield/validation/test_ws32_native_benchmark_collect.py",
    "tests/greenfield/benchmarking/test_ws32_short_runner.py",
)


def main() -> int:
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    # Real-tokenizer checking is a separate opt-in test, never implicit GCS I/O.
    env.pop("GLM_RELEASE_LOCAL_TOKENIZER", None)
    commands = (
        ["git", "diff", "--check"],
        ["git", "diff", "--cached", "--check"],
        [sys.executable, "-m", "glm_tpu", "doctor", "--profile", "benchmark"],
        [sys.executable, "tools/audit_release_content.py"],
        [sys.executable, "-c", "from pathlib import Path; from scripts.greenfield.ws32_native_benchmark_programs import require_source; require_source(Path.cwd())"],
        [sys.executable, "-m", "compileall", "-q", "glm_tpu", "scripts/release", "tools", "tests/release"],
        [sys.executable, "-m", "pytest", "-q", *TESTS],
        [sys.executable, "tools/check_release_package.py"],
    )
    rows = []
    for command in commands:
        started = time.monotonic()
        try:
            result = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            rows.append(dict(command=command, error=type(exc).__name__,
                             seconds=round(time.monotonic()-started, 3)))
            print(json.dumps(dict(passed=False, steps=rows), indent=2))
            return 1
        row = dict(command=command, returncode=result.returncode,
                   seconds=round(time.monotonic()-started, 3))
        rows.append(row)
        if "pytest" in command and result.returncode == 0:
            row["test_summary"] = result.stdout.strip().splitlines()[-1]
        if result.returncode:
            print(json.dumps(dict(passed=False, steps=rows), indent=2))
            # The content audit emits locations only; tests contain synthetic inputs.
            print(result.stdout[-6000:], file=sys.stderr)
            print(result.stderr[-2000:], file=sys.stderr)
            return 1
    print(json.dumps(dict(passed=True, steps=rows,
        scope="selected CPU release/host-runtime checks and isolated no-deps wheel installation",
        hardware_execution=False, full_dependency_install=False,
        benchmark_completion=False, release_merge_authorized=False), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
