"""Recover only the ended 9c80ebac native trace failure; never run model code.

Original exit/publication markers stay unchanged. Both leases, fresh root idle,
published recovery source, exact generation readback and existing conditional
publisher are mandatory. No checkpoint, TPU resource or original is deleted.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import shlex

from scripts.greenfield import launch_ws32_native_benchmark as launch
from scripts.greenfield import ws32_native_benchmark_collect as collect
from scripts.greenfield import ws32_native_benchmark_transport as cold
from scripts.greenfield.fp8_baseline_guard import census_command, validate_fleet

TAG = "greenfield_ws32_native_benchmark_20260913T141500000000000Z"
RUN_PIN = "9c80ebac85b7dca0ee5423638eddbaf41089e027"


def main() -> None:
    from google.cloud import storage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-pin", required=True)
    args = parser.parse_args()
    launch.source_preflight(args.recovery_pin)
    root = launch.watch.RUN_ROOT / TAG
    with ExitStack() as stack:
        for path in launch.watch.LOCKS:
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pre = launch.ssh(census_command())
        validate_fleet(pre)
        validate_fleet((root / "census_post.txt").read_text())
        # Authenticate original exit/boot records before deploying recovery-only
        # host code. All eight exact originals must already be terminal.
        fleet = launch.watch.observe(TAG, RUN_PIN)
        rows = launch.publication_state(TAG, RUN_PIN, fleet)
        if any(r["processes"] or r["holders"] for r in fleet):
            raise ValueError("original workers are not idle")
        if any(r["ended"] is None or r["published"] is None
               or r["ended"]["worker_exit_code"] != 1
               or r["published"]["publish_exit_code"] != 1 for r in rows):
            raise ValueError("original trace-failure markers differ")
        bucket = cold._bucket(storage.Client())
        live = sum(int(b.size) for b in bucket.list_blobs())
        if live + (2 << 30) >= 2_500_000_000_000:
            raise ValueError("bounded failure recovery exceeds regional storage cap")
        sync = launch.ssh(launch.sync_command(args.recovery_pin), timeout=300)
        launch.markers(sync, "NATIVE_SYNC_OK")
        # No supervisor/model entry, marker overwrite or automatic retry.
        command = ("cd " + shlex.quote(str(launch.REPO)) +
                   " && JAX_PLATFORMS=cpu " + shlex.quote(launch.PYTHON) +
                   " -m scripts.greenfield.ws32_native_benchmark_collect --tag " + TAG +
                   " --code-hash " + RUN_PIN + " --rank ${HOSTNAME##*-w-}")
        publication = launch.ssh(command, timeout=1200)
        post = launch.ssh(census_command())
        validate_fleet(post)
        receipts = []
        for rank in range(8):
            blob = bucket.get_blob(collect.prefix(TAG, rank) + "manifest.json")
            data = cold._download(blob, cap=collect.MANIFEST_CAP)
            manifest = json.loads(data)
            if (manifest["tag"] != TAG or manifest["code_hash"] != RUN_PIN
                    or manifest["rank"] != rank or manifest["quality_proven"] is not False):
                raise ValueError("recovered original manifest identity differs")
            for item in manifest["files"]:
                actual = bucket.get_blob(item["name"])
                if (str(actual.generation), actual.size, actual.crc32c) != (
                        item["generation"], item["size"], item["crc32c"]):
                    raise ValueError("recovered original generation differs")
            receipts.append(manifest)
        result = dict(schema="native_trace_failure_recovery_v1", tag=TAG,
            execution_pin=RUN_PIN, recovery_pin=args.recovery_pin,
            live_before=live, original_publication=rows,
            pre_census=pre, post_census=post, publication_stdout=publication,
            request_manifests=receipts, model_rerun=False, quality_pass=False,
            limits="Publication/readback only; failed request has no completed answer or score")
        launch.persist(root / "trace_original_recovery.json", result)
        print(json.dumps(dict(recovered_ranks=len(receipts), quality_pass=False,
                              receipt=str(root / "trace_original_recovery.json"))))


if __name__ == "__main__":
    main()
