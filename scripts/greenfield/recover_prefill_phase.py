"""Collect an already-finished phase run; never launch or manage TPU work.

Reuse the original protected wrapper's census, accounting and archive blocks.
An existing summary refuses: partial accounting needs separate diagnosis, not
another DB insertion. Original failure records remain unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys

from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from scripts.greenfield.prefill_phase_baseline import KERNEL

REPO = Path("/home/gianl/glm-tpu-topology-rewrite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pin", required=True)
    args = parser.parse_args()
    if not campaign.window_acquisition.is_phase_baseline_tag(args.tag):
        raise ValueError("only completed phase runs may be recovered")
    if len(args.pin) != 40 or any(c not in "0123456789abcdef" for c in args.pin):
        raise ValueError("invalid original pin")
    root = campaign.run_root(args.tag)
    if (
        not root.is_dir()
        or (root / "summary.json").exists()
        or (root / "SUCCESS").exists()
        or (root / "recovery.json").exists()
    ):
        raise ValueError("missing original or already-accounted run")
    with ExitStack() as stack:
        for path in (
            "/home/gianl/glm-run/.glm_pod_workload.lock",
            "/home/gianl/.glm-tpu-rsync.lock",
        ):
            handle = stack.enter_context(open(path, "a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(
            (root / name).exists()
            for name in ("summary.json", "SUCCESS", "recovery.json")
        ):
            raise ValueError(
                "run was accounted or recovery started before lease acquisition"
            )
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO).strip():
            raise ValueError("recovery requires clean source")
        location = subprocess.check_output(
            [
                "gcloud",
                "storage",
                "buckets",
                "describe",
                "gs://driftbench-dsv4-uc",
                "--format=value(location)",
            ],
            text=True,
        ).strip()
        if location != "US-CENTRAL2":
            raise ValueError("recovery bucket region differs")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", args.pin, "HEAD"],
            cwd=REPO,
            check=True,
        )
        source = subprocess.check_output(
            [
                "git",
                "show",
                args.pin + ":scripts/greenfield/run_fp8_matmul_microbench.sh",
            ],
            cwd=REPO,
            text=True,
        )
        definitions = source.split("has_eight_unique_markers() {", 1)[1].split(
            "\npost_census_done=0", 1
        )[0]
        environment = dict(
            POD="db-v4-64-od",
            ZONE="us-central2-b",
            WORKTREE=str(REPO),
            RUN_DIR=str(root),
            TAG=args.tag,
            BOUNDED_PREFILL="1",
        )
        header = (
            "set -euo pipefail\n"
            + "\n".join(k + "=" + shlex.quote(v) for k, v in environment.items())
            + "\n"
        )

        def census(label: str) -> None:
            subprocess.run(
                [
                    "bash",
                    "-c",
                    header
                    + "has_eight_unique_markers() {"
                    + definitions
                    + "\nstrict_census "
                    + label,
                ],
                cwd=REPO,
                check=True,
            )

        census("recovery_pre")
        record = campaign.collect(args.tag, args.pin)
        # Older wrapper DB rows lack an explicit tag. Their checksum binds these
        # exact generation-checked eight workers (including PID/start/boot), so
        # compare that identity rather than matching a timestamp-shaped string.
        with sqlite3.connect(
            "file:/home/gianl/glm-tpu/bench/results.db?mode=ro", uri=True
        ) as db:
            existing = db.execute(
                'SELECT run_id FROM items WHERE benchmark=? AND json_extract(raw_output, "$.checksum")=? LIMIT 1',
                ("greenfield_fp8_" + KERNEL, record["checksum"]),
            ).fetchone()
        if existing is not None:
            raise ValueError(f"original phase already has DB evidence: {existing}")
        campaign._atomic_json(root / "runner.json", record)
        (root / "hlo").mkdir(exist_ok=True)
        (root / "hlo/candidate.optimized_hlo.txt").write_bytes(
            (root / "fleet/rank0/candidate.optimized_hlo.txt").read_bytes()
        )
        census("post")
        campaign._atomic_json(
            root / "recovery.json",
            dict(
                original_pin=args.pin,
                recovery_pin=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
                ).strip(),
                model_rerun=False,
                original_failure_preserved=True,
            ),
        )
        blocks = [b.split("\nPY\n", 1)[0] for b in source.split("<<'PY'\n")[1:]]
        accounting = next(
            b
            for b in blocks
            if "run_dir, pin, db_path, repo, elapsed, expected_kernel" in b
        )
        sys.argv = [
            "original-wrapper-accounting",
            str(root),
            args.pin,
            "/home/gianl/glm-tpu/bench/results.db",
            str(REPO),
            "0",
            KERNEL,
        ]
        # Recovery elapsed=0 is not model wall; per-sample original wall is bound
        # inside runner, while recovery identity is explicit above.
        exec(
            compile(accounting, "<original-wrapper-accounting>", "exec"),
            {"__name__": "__main__"},
        )
        archive = next(
            b
            for b in blocks
            if "receipts = []" in b and "terminal = root / 'SUCCESS'" in b
        )
        sys.argv = ["original-wrapper-archive", str(root), args.pin]
        exec(
            compile(archive, "<original-wrapper-archive>", "exec"),
            {"__name__": "__main__"},
        )
        print(
            "PHASE_RECOVERY_SUCCESS "
            + json.loads((root / "summary.json").read_text())["runner"]["checksum"]
        )


if __name__ == "__main__":
    main()
