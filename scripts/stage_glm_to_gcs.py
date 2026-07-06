#!/usr/bin/env python3
"""Stage zai-org/GLM-5.2-FP8 from HuggingFace -> GCS (us-central2 bucket ONLY).

Adapted from ~/moe-tpu/scripts/stage_base_to_gcs.py (DSV4 Base staging). Streams
each file HF -> `gcloud storage cp -` (no local staging; the host has limited
disk). Resumable: skips any destination object already present with the correct
size; verifies size after each upload; retries on mismatch/failure. The HF token
is passed via a 0600 curl config file (kept out of argv / `ps`).

COST RULES (CLAUDE.md #COST): destination MUST be gs://driftbench-dsv4-uc
(US-CENTRAL2, same region as the pod). NEVER gs://driftbench-storage (EU).
One FP8 copy only; no bf16 conversion.
"""
import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile
import time

from huggingface_hub import HfApi

REPO = "zai-org/GLM-5.2-FP8"
DST = "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
WORKERS = 6
RETRIES = 4
TOKEN = os.environ["HF_TOKEN"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gcs_sizes():
    """Map basename -> size for objects already in DST (resume support)."""
    out = {}
    try:
        r = subprocess.run(["gcloud", "storage", "ls", "-l", f"{DST}/**"],
                           capture_output=True, text=True, timeout=180)
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[-1].startswith(DST):
                try:
                    out[parts[-1].split("/")[-1]] = int(parts[0])
                except ValueError:
                    pass
    except Exception as e:  # noqa: BLE001
        log(f"(gcs pre-list skipped: {e})")
    return out


def dst_size(fname):
    r = subprocess.run(["gcloud", "storage", "ls", "-l", f"{DST}/{fname}"],
                       capture_output=True, text=True, timeout=120)
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-1].endswith("/" + fname):
            try:
                return int(parts[0])
            except ValueError:
                return None
    return None


def stream_one(fname, size, hdr_file):
    url = f"https://huggingface.co/{REPO}/resolve/main/{fname}"
    for attempt in range(1, RETRIES + 1):
        cur = dst_size(fname)
        if cur == size:
            return (fname, "ok", cur)
        t0 = time.time()
        p1 = subprocess.Popen(["curl", "-sfL", "-K", hdr_file, url],
                              stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["gcloud", "storage", "cp", "-", f"{DST}/{fname}"],
                              stdin=p1.stdout,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
        p1.stdout.close()
        err = p2.communicate()[1]
        p1.wait()
        dt = time.time() - t0
        cur = dst_size(fname)
        if p1.returncode == 0 and p2.returncode == 0 and cur == size:
            mbps = (size / 1e6) / dt if dt > 0 else 0
            log(f"  OK   {fname} ({size/1e9:.2f} GB, {dt:.0f}s, {mbps:.0f} MB/s)")
            return (fname, "ok", cur)
        log(f"  RETRY {fname} attempt {attempt}/{RETRIES} "
            f"(curl={p1.returncode} gcloud={p2.returncode} got={cur} want={size}) "
            f"{(err or b'').decode()[:160]}")
        time.sleep(5 * attempt)
    return (fname, "FAILED", dst_size(fname))


def main():
    api = HfApi()
    info = api.model_info(REPO, token=TOKEN, files_metadata=True)
    files = {s.rfilename: (s.size or 0) for s in info.siblings
             if "/" not in s.rfilename}  # top-level files only (verified: all 150 are)
    have = gcs_sizes()
    todo = {f: sz for f, sz in files.items() if have.get(f) != sz}
    done = len(files) - len(todo)
    total_gb = sum(files.values()) / 1e9
    todo_gb = sum(todo.values()) / 1e9
    log(f"GLM-5.2-FP8 staging -> {DST}")
    log(f"  {len(files)} files / {total_gb:.1f} GB total; "
        f"{done} already present; {len(todo)} to transfer ({todo_gb:.1f} GB)")
    if not todo:
        log("ALL FILES ALREADY STAGED (size-verified). Nothing to do.")
        return 0

    with tempfile.NamedTemporaryFile("w", suffix=".curlrc", delete=False) as hf:
        hf.write(f'header = "Authorization: Bearer {TOKEN}"\n')
        hdr_file = hf.name
    os.chmod(hdr_file, 0o600)
    try:
        results = []
        # Largest first so the big shards aren't the long tail.
        order = sorted(todo, key=lambda f: -todo[f])
        with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(stream_one, f, todo[f], hdr_file): f for f in order}
            for fut in cf.as_completed(futs):
                results.append(fut.result())
    finally:
        os.remove(hdr_file)

    failed = [r for r in results if r[1] != "ok"]
    log(f"DONE: {len(results)-len(failed)}/{len(results)} transferred this run; "
        f"{len(failed)} failed")
    for r in failed:
        log(f"  FAILED: {r[0]} (got {r[2]})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
