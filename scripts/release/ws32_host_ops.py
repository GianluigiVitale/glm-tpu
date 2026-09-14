"""Shared protected host operations for the supported WS32 request controller.

No campaign scheduling or model execution. Source admission, existing-host SSH,
original-process identity and publication checks are extracted without changing
function bodies. Callers retain responsibility for both leases and run scope.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from scripts.greenfield import watch_ws32_run as watch
from scripts.greenfield import ws32_native_benchmark_transport as cold
from scripts.greenfield import ws32_native_benchmark_collect as requests
from scripts.greenfield.ws32_native_benchmark_protocol import REPO, canonical
from scripts.greenfield.ws32_native_benchmark_result import read

PYTHON = "/home/gianl/vllm-env/bin/python"
BRANCH = "main"
FILE_CAP = 64 << 20


def ssh(command: str, *, workers: str = "all", timeout: int = 55) -> str:
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            "db-v4-64-od",
            "--zone",
            "us-central2-b",
            "--worker=" + workers,
            "--command=" + command,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError("existing-host SSH failed: " + result.stderr[-1000:])
    return result.stdout


def persist(path: Path, value: Any) -> None:
    requests._write_once(path, canonical(value) + b"\n")


def reviewed_branch(value: str) -> str:
    """Explicit private release/research ref; never an option or shell fragment."""
    if (
        not isinstance(value, str)
        or re.fullmatch(
            r"(?:main|release/[A-Za-z0-9][A-Za-z0-9._/-]*|rewrite/topology-first-decode)",
            value,
        )
        is None
        or any(part in value for part in ("..", "//", "@{"))
        or any(
            not part or part.startswith(".") or part.endswith((".", ".lock"))
            for part in value.split("/")
        )
    ):
        raise ValueError("invalid reviewed deployment branch")
    return value


def sync_command(pin: str, *, branch: str = BRANCH) -> str:
    """Existing owner repository only; no clone, weights, deletion or reset."""
    cold._identity("greenfield_ws32_native_benchmark_20260912T000000000000000Z", pin, 0)
    reviewed_branch(branch)
    return (
        "set -euo pipefail; wt="
        + shlex.quote(str(REPO))
        + "; pin="
        + shlex.quote(pin)
        + '; cd "$wt"; [[ -z $(git status --porcelain) ]]; '
        "origin=$(git remote get-url origin); "
        "[[ $origin == git@github.com:GianluigiVitale/glm-tpu.git || "
        "$origin == https://github.com/GianluigiVitale/glm-tpu.git ]]; "
        "if [[ ${HOSTNAME##*-w-} != 0 ]]; then git fetch -q origin "
        + shlex.quote(branch)
        + '; [[ $(git rev-parse FETCH_HEAD) == "$pin" ]]; '
        'git checkout -q --detach "$pin"; fi; '
        '[[ $(git rev-parse HEAD) == "$pin" && -z $(git status --porcelain) ]]; '
        "echo NATIVE_SYNC_OK ${HOSTNAME##*-w-}"
    )


def source_preflight(pin: str, *, branch: str = BRANCH) -> None:
    from scripts.greenfield.run_short_decoder_ws32 import _require_clean_code
    from scripts.greenfield.ws32_native_benchmark_programs import require_source

    reviewed_branch(branch)
    _require_clean_code(pin)
    require_source(REPO)
    origin = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=REPO, text=True
    ).strip()
    if origin not in (
        "git@github.com:GianluigiVitale/glm-tpu.git",
        "https://github.com/GianluigiVitale/glm-tpu.git",
    ):
        raise ValueError(
            "native deployment origin is not the owner's glm-tpu repository"
        )
    local_branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=REPO, text=True
    ).strip()
    # The reviewed branch may already be checked out in the release worktree.
    # A detached canonical checkout is admitted ONLY at the clean exact pin
    # verified above and the owner's current published branch HEAD below.
    if local_branch not in ("", branch):
        raise ValueError("native controller is on a different named branch")
    rows = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/" + branch], cwd=REPO, text=True
    ).splitlines()
    if (
        len(rows) != 1
        or len(rows[0].split()) != 2
        or rows[0].split()[1] != "refs/heads/" + branch
    ):
        raise ValueError("native reviewed deployment ref is missing or ambiguous")
    remote = rows[0].split()[0]
    if remote != pin:
        raise ValueError("native code pin is not published at owner branch HEAD")


def validate_attach(launch: dict[str, Any], *, tag: str, pin: str, branch: str) -> None:
    """Bind recovery to the original deployment, including legacy research records."""
    reviewed_branch(branch)
    if launch.get("tag") != tag or launch.get("code_hash") != pin:
        raise ValueError("attach identity differs from original launch")
    # Historical records were research-only and omitted the field.
    original = launch.get("reviewed_branch", "rewrite/topology-first-decode")
    if reviewed_branch(original) != branch:
        raise ValueError("attach reviewed branch differs from original launch")


def markers(text: str, prefix: str) -> None:
    rows = [
        line.split()[-1] for line in text.splitlines() if line.startswith(prefix + " ")
    ]
    if len(rows) != 8 or set(rows) != set(map(str, range(8))):
        raise ValueError("native remote phase did not complete on all8 hosts")


PUBLICATION_REMOTE = """
import json, pathlib, socket, sys
tag, pin = sys.argv[1:]
host = socket.gethostname()
rank = int(host.rsplit('-w-', 1)[1])
root = pathlib.Path('/home/gianl/glm-run') / tag
row = dict(rank=rank, host=host, tag=tag, pin=pin,
    boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip())
for phase in ('ended', 'published'):
    path = root / (phase + '.rank%d.json' % rank)
    row[phase] = json.loads(path.read_bytes()) if path.exists() else None
print('NATIVE_PUBLICATION ' + json.dumps(row))
"""


def publication_state(tag: str, pin: str, fleet: list[dict]) -> list[dict]:
    """Read original supervisor receipts, not mere marker-file existence."""
    payload = base64.b64encode(PUBLICATION_REMOTE.encode()).decode()
    command = shlex.join(
        [
            "/usr/bin/python3",
            "-c",
            "import base64;exec(base64.b64decode(" + repr(payload) + "))",
            tag,
            pin,
        ]
    )
    text = ssh(command)
    prefix = "NATIVE_PUBLICATION "
    rows = [
        json.loads(line[len(prefix) :])
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(rows) != 8 or {r["rank"] for r in rows} != set(range(8)):
        raise ValueError("incomplete publication observation")
    rows.sort(key=lambda r: r["rank"])
    for row, observed in zip(rows, fleet, strict=True):
        for key in ("rank", "host", "boot_id", "tag", "pin"):
            if row[key] != observed[key]:
                raise ValueError("publication fleet identity differs: " + key)
        ended, published = row["ended"], row["published"]
        for marker in (ended, published):
            if marker is not None and (
                marker.get("tag") != tag
                or marker.get("code_hash") != pin
                or type(marker.get("rank")) is not int
                or marker["rank"] != row["rank"]
                or type(marker.get("worker_exit_code")) is not int
            ):
                raise ValueError("original supervisor marker identity differs")
        if ended is not None and (
            ended.get("host") != row["host"] or ended.get("boot_id") != row["boot_id"]
        ):
            raise ValueError("ended original host/boot differs")
        if published is not None and (
            ended is None
            or published["worker_exit_code"] != ended["worker_exit_code"]
            or type(published.get("publish_exit_code")) is not int
        ):
            raise ValueError("publication lacks consistent ended original")
    return rows


def observe_originals(original: list[dict] | None, observed: list[dict]) -> list[dict]:
    """Capture each first-seen process, including staggered worker startup."""
    if original is None:
        return observed
    watch.require_original_fleet(
        [
            old if old["processes"] else new
            for old, new in zip(original, observed, strict=True)
        ],
        observed,
    )
    # Even before a PID is observed, preserve host/boot identity from launch.
    for old, new in zip(original, observed, strict=True):
        for key in ("rank", "host", "boot_id", "tag", "pin"):
            if old[key] != new[key]:
                raise ValueError("original fleet identity changed: " + key)
    return [
        old if old["processes"] else new
        for old, new in zip(original, observed, strict=True)
    ]


def verify_ownership(root: Path, tag: str, pin: str) -> None:
    """Bind original worker PID/start/boot/argv to the authenticated SSH history."""
    original = None
    rows = read(root / "final_watch.jsonl", FILE_CAP).decode().splitlines()
    idle = 0
    for line in rows:
        record = json.loads(line)
        if record.get("tag") != tag or record.get("pin") != pin:
            raise ValueError("native archived watcher identity differs")
        if record.get("status") != "OBSERVED":
            idle = 0
            continue
        fleet = watch.parse_fleet(
            "\n".join(watch.PREFIX + json.dumps(r) for r in record["fleet"]), tag, pin
        )
        original = observe_originals(original, fleet)
        idle = (
            idle + 1
            if all(not r["processes"] and not r["holders"] for r in fleet)
            else 0
        )
    if (
        original is None
        or idle < 2
        or not all(len(r["processes"]) == 1 for r in original)
    ):
        raise ValueError(
            "native archive lacks original owners and two idle observations"
        )
    for phase in ("pre", "post"):
        census = [
            json.loads(line[len("FP8_IDLE ") :])
            for line in read(root / f"census_{phase}.txt", FILE_CAP)
            .decode()
            .splitlines()
            if line.startswith("FP8_IDLE ")
        ]
        if {(r["host"], r["boot_id"]) for r in census} != {
            (r["host"], r["boot_id"]) for r in original
        }:
            raise ValueError(
                "native root census differs from original host/boot identities"
            )
    for rank, observed in enumerate(original):
        record = json.loads(
            read(root / "collected" / f"runner.rank{rank}.json", 2 << 20)
        )
        process = observed["processes"][0]
        expected = dict(
            pid=process["pid"],
            start_ticks=process["start_ticks"],
            argv_sha256=process["argv_sha256"],
            hostname=observed["host"],
            boot_id=observed["boot_id"],
        )
        if (
            record["code_hash"] != pin
            or record["launch_process_id"] != rank
            or record["owner"] != expected
        ):
            raise ValueError(
                "native original runner differs from authenticated process owner"
            )
