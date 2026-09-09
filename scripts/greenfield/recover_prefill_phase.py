"""Collect an already-finished phase/dense run; never launch or manage TPU work.

Reuse the original protected wrapper's census, accounting and archive blocks.
An existing summary refuses: partial accounting needs separate diagnosis, not
another DB insertion. Original failure records remain unchanged.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys

from scripts.greenfield import ws32_prefill_layer_campaign as campaign
from scripts.greenfield.prefill_phase_baseline import KERNEL
from scripts.greenfield import ws32_dense_frontier_protocol as dense_protocol
from scripts.greenfield import ws32_dense_norm_protocol as norm_protocol
from scripts.greenfield import ws32_dense_canonical as canonical

REPO = Path("/home/gianl/glm-tpu-topology-rewrite")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--pin", required=True)
    args = parser.parse_args()
    norm_mode = norm_protocol.is_tag(args.tag)
    canonical_mode = canonical.is_tag(args.tag)
    dense = campaign.is_dense_tag(args.tag)
    kernel = (
        canonical.KERNEL if canonical_mode else
        norm_protocol.KERNEL
        if norm_mode
        else dense_protocol.KERNEL if dense else KERNEL
    )
    if not dense and not campaign.window_acquisition.is_phase_baseline_tag(args.tag):
        raise ValueError("only completed phase/dense runs may be recovered")
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
    if dense and any(
        (root / n).exists() or (root / n).is_symlink()
        for n in (
            "runner.json",
            "hlo/candidate.optimized_hlo.txt",
            "evidence.sha256",
            "census_post.txt",
            "devices_post.txt",
            "census_recovery_pre.txt",
            "devices_recovery_pre.txt",
        )
    ):
        raise ValueError("dense recovery would overwrite prior controller evidence")
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
        recovery_pin = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        if dense:
            published = subprocess.check_output(
                [
                    "git",
                    "ls-remote",
                    "origin",
                    "refs/heads/rewrite/topology-first-decode",
                ],
                cwd=REPO,
                text=True,
            ).split()
            if not published or published[0] != recovery_pin:
                raise ValueError("dense recovery requires published source")
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
        if dense:
            from google.cloud import storage
            from scripts.greenfield.ws32_dense_frontier_transport import (
                replay_collected,
                archive_inventory,
            )

            # The failed controller already collected immutable originals.
            # Reject an oversized recovery BEFORE replay/accounting, not after.
            archive_inventory(root)
            record = replay_collected(
                tag=args.tag,
                pin=args.pin,
                root=root / "fleet",
                repo=REPO,
                **campaign.dense_reference_kwargs(args.tag),
                client=storage.Client(),
            )
        else:
            record = campaign.collect(args.tag, args.pin)
        # Older wrapper DB rows lack an explicit tag. Their checksum binds these
        # exact generation-checked eight workers (including PID/start/boot), so
        # compare that identity rather than matching a timestamp-shaped string.
        with sqlite3.connect(
            "file:/home/gianl/glm-tpu/bench/results.db?mode=ro", uri=True
        ) as db:
            existing = db.execute(
                'SELECT run_id FROM items WHERE benchmark=? AND json_extract(raw_output, "$.checksum")=? LIMIT 1',
                ("greenfield_fp8_" + kernel, record["checksum"]),
            ).fetchone()
        if existing is not None:
            raise ValueError(f"original phase already has DB evidence: {existing}")
        campaign._atomic_json(root / "runner.json", record)
        (root / "hlo").mkdir(exist_ok=True)
        graph = canonical.GRAPH if canonical_mode else "dense01_norm" if norm_mode else "dense01" if dense else "candidate"
        (root / "hlo/candidate.optimized_hlo.txt").write_bytes(
            (root / f"fleet/rank0/{graph}.optimized_hlo.txt").read_bytes()
        )
        census("post")
        campaign._atomic_json(
            root / "recovery.json",
            dict(
                original_pin=args.pin,
                recovery_pin=recovery_pin,
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
            kernel,
        ]
        # Recovery elapsed=0 is not model wall; per-sample original wall is bound
        # inside runner, while recovery identity is explicit above.
        exec(
            compile(accounting, "<original-wrapper-accounting>", "exec"),
            {"__name__": "__main__"},
        )
        if dense:
            # Original accounting includes SQLite.backup + integrity_check.
            # Reproduce the wrapper's separate evidence ledger, including the
            # new recovery provenance/censuses and immutable failure logs.
            paths = sorted(p for p in (root / "hlo").rglob("*") if p.is_file())
            paths += [
                root / n
                for n in (
                    "runner.json",
                    "runner.log",
                    "summary.json",
                    "results_ckpt.db",
                    "census_pre.txt",
                    "census_post.txt",
                    "devices_pre.txt",
                    "devices_post.txt",
                    "recovery.json",
                    "census_recovery_pre.txt",
                    "devices_recovery_pre.txt",
                    "census_failure_exit.txt",
                    "devices_failure_exit.txt",
                    "failure_archive_receipts.json",
                )
            ]
            ledger = root / "evidence.sha256"
            if ledger.exists() or ledger.is_symlink():
                raise FileExistsError(ledger)
            ledger.write_text(
                "".join(
                    f"{sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n"
                    for p in paths
                )
            )
            archive_inventory(root)
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
