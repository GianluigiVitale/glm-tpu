"""Offline recovery wrapper safeguards; never execute a pack or contact workers."""

import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/greenfield/run_ws32_runtime_checkpoint_shm_pack.sh"


def test_disabled_recovery_exits_before_any_external_command():
    env = dict(os.environ, GLM_GREENFIELD_WS32_SHM_PACK="0", PATH="/nonexistent")
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT)], env=env, capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "default-off" in result.stderr


def test_recovery_parses_and_holds_both_leases_before_pack_or_publish():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text()
    lock = source.index("flock -n 8")
    pod_lock = source.index("flock -n 9")
    assert "/home/gianl/.glm-tpu-rsync.lock" in source
    assert lock < pod_lock < source.index("trap on_exit EXIT")
    assert pod_lock < source.index("gcloud storage objects list")
    assert (
        pod_lock
        < source.index("bucket = _bucket(storage.Client())")
        < source.index("trap on_exit EXIT")
    )
    assert "live + (10 << 30) >= 2_500_000_000_000" in source
    assert pod_lock < source.index("host_command='")
    assert "source_preflight(sys.argv[1], branch=sys.argv[2])" in source
    assert "GLM_GREENFIELD_WS32_SHM_PACK_BRANCH:-main" in source
    sync = source[source.index("sync_command='") : source.index("sync_rc=0")]
    assert "remote get-url origin" in sync
    assert sync.index("rev-parse FETCH_HEAD") < sync.index("checkout -q --detach")
    host = source[source.index("host_command='") : source.index("pack_rc=0")]
    assert "export JAX_PLATFORMS=cpu" in host
    assert "is NOT byte-identical to the sealed checkpoint" in host
