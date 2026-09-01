"""Authenticate the immutable PP16 HLO wrapper, then acquire its lock broker.

This prelauncher is default-off.  It prevents an untrusted first Bash pass and
executes only the exact lock broker embedded in the reviewed root-owned wrapper.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
from typing import NoReturn

WRAPPER = "/opt/glm-tpu/gate-d-pp16-hlo-wrapper-s2/run_gate_d_compensated_pp16_hlo.sh"
WRAPPER_ROOT = "/opt/glm-tpu/gate-d-pp16-hlo-wrapper-s2"
WRAPPER_NAME = "run_gate_d_compensated_pp16_hlo.sh"
WRAPPER_BYTES = 28_381
WRAPPER_SHA256 = "802d7731c70a3e7a06307754ded2e16fc4e321a7293e01d2a8c4256973bfc0b6"
BROKER_SHA256 = "a6b92472c0e0c75a56e1f1c588d10babbfbbb35e25452f3c09fdde9ff1ad6dfb"
EXPECTED_ENVIRONMENT = {
    "GLM_GATE_D_COMPENSATED_PP16_HLO_ACQUIRE",
    "GLM_GATE_D_COMPENSATED_PP16_MODE",
    "GLM_GATE_D_COMPENSATED_PP16_TAG",
    "GLM_GATE_D_WRAPPER_SANITIZED",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
}


def _fail(message: str) -> NoReturn:
    raise SystemExit(message)


def _validate_directory(path: str) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        value = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(value.st_mode)
            or value.st_uid != 0
            or value.st_gid != 0
            or stat.S_IMODE(value.st_mode) != 0o755
            or (value.st_dev, value.st_ino) != (named.st_dev, named.st_ino)
            or os.listxattr(path, follow_symlinks=False)
        ):
            _fail("unsafe prelauncher directory: " + path)
    finally:
        os.close(descriptor)


def _authenticated_broker() -> str:
    for directory in ("/opt", "/opt/glm-tpu", WRAPPER_ROOT):
        _validate_directory(directory)

    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    root_descriptor = os.open(WRAPPER_ROOT, directory_flags)
    wrapper_descriptor = os.open(
        WRAPPER_NAME,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=root_descriptor,
    )
    try:
        before = os.fstat(wrapper_descriptor)
        named_before = os.stat(
            WRAPPER_NAME, dir_fd=root_descriptor, follow_symlinks=False
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != 0
            or before.st_gid != 0
            or stat.S_IMODE(before.st_mode) != 0o555
            or before.st_nlink != 1
            or before.st_size != WRAPPER_BYTES
            or (before.st_dev, before.st_ino)
            != (named_before.st_dev, named_before.st_ino)
            or os.listxattr(WRAPPER, follow_symlinks=False)
        ):
            _fail("unsafe installed wrapper identity")

        blocks: list[bytes] = []
        while block := os.read(wrapper_descriptor, 1024 * 1024):
            blocks.append(block)
        raw = b"".join(blocks)
        after = os.fstat(wrapper_descriptor)
        named_after = os.stat(
            WRAPPER_NAME, dir_fd=root_descriptor, follow_symlinks=False
        )
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or (before.st_dev, before.st_ino)
            != (named_after.st_dev, named_after.st_ino)
            or hashlib.sha256(raw).hexdigest() != WRAPPER_SHA256
        ):
            _fail("installed wrapper changed during authentication")
    finally:
        os.close(wrapper_descriptor)
        os.close(root_descriptor)

    text = raw.decode("utf-8")
    start = (
        "read -r -d '' IMMUTABLE_LOCK_BROKER "
        "<<'IMMUTABLE_LOCK_BROKER_EOF' || true\n"
    )
    end = "\nIMMUTABLE_LOCK_BROKER_EOF"
    if text.count(start) != 1 or text.count(end) != 1:
        _fail("immutable lock broker boundary mismatch")
    broker = text.split(start, 1)[1].split(end, 1)[0]
    if hashlib.sha256(broker.encode("utf-8")).hexdigest() != BROKER_SHA256:
        _fail("immutable lock broker digest mismatch")
    return broker


def main() -> int:
    if (
        sys.executable != "/usr/bin/python3"
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or not sys.flags.ignore_environment
    ):
        _fail("unsafe prelauncher interpreter")
    if len(sys.argv) != 2 or sys.argv[1] not in {"--authenticate-only", "--acquire"}:
        _fail("unsafe prelauncher action")
    if set(os.environ) != EXPECTED_ENVIRONMENT:
        _fail("unsafe prelauncher environment")

    broker = _authenticated_broker()
    if sys.argv[1] == "--authenticate-only":
        print("PRELAUNCHER_AUTH_OK", WRAPPER_SHA256, BROKER_SHA256)
        return 0
    os.execve(
        "/usr/bin/python3",
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-B",
            "-c",
            broker,
            "acquire",
            WRAPPER,
        ],
        dict(os.environ),
    )
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
