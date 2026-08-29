from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR = REPO_ROOT / "scripts/validate_ray_network.sh"
LAUNCHER = REPO_ROOT / "scripts/launch_glm_32chip.sh"
WRAPPER = REPO_ROOT / "scripts/greenfield/run_capture_short_context_dsa_oracle.sh"


def _run(
    *args: str, stdin: str = "", timeout: float = 5
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(VALIDATOR), *args],
        input=stdin,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )


def test_ray_firewall_contract_accepts_only_current_pod_target() -> None:
    pod_id = "3005566610109598201"
    current = f"tpu-t1v-n-ae271d05-w-{pod_id}"
    contract = (
        f"default\tINGRESS\t1000\t192.168.0.0/16\ttcp:1024-65535\tFalse\t{current}\n"
    )
    assert _run("firewall", current, pod_id, stdin=contract).returncode == 0

    stale = contract.replace("ae271d05", "6c15e171").replace(
        pod_id, "5201142156555843955"
    )
    refused = _run("firewall", current, pod_id, stdin=stale)
    assert refused.returncode == 1
    assert "stale" in refused.stderr

    for malformed in (
        contract.rstrip("\n") + "\textra\n",
        contract + contract,
        "banner\n" + contract,
        contract + "\n",
        "",
    ):
        assert _run("firewall", current, pod_id, stdin=malformed).returncode == 1


def test_ray_connectivity_receipts_accept_positive_path_and_refuse_missing_host() -> (
    None
):
    good = "".join(f"PRETAG_TCP6379_OK {index}\n" for index in range(1, 8))
    assert (
        _run(
            "receipts",
            "PRETAG_TCP6379_OK",
            "7",
            "1,2,3,4,5,6,7",
            stdin=good,
        ).returncode
        == 0
    )

    missing = good.rsplit("\n", 2)[0] + "\n"
    refused = _run(
        "receipts",
        "PRETAG_TCP6379_OK",
        "7",
        "1,2,3,4,5,6,7",
        stdin=missing,
    )
    assert refused.returncode == 1
    assert "incomplete" in refused.stderr

    wrong_owners = good.replace("PRETAG_TCP6379_OK 7", "PRETAG_TCP6379_OK 8")
    assert (
        _run(
            "receipts",
            "PRETAG_TCP6379_OK",
            "7",
            "1,2,3,4,5,6,7",
            stdin=wrong_owners,
        ).returncode
        == 1
    )

    for malformed in (
        good.replace("PRETAG_TCP6379_OK 7", "PRETAG_TCP6379_OK 7 extra"),
        good.replace("PRETAG_TCP6379_OK 7", "PRETAG_TCP6379_OK bad/owner"),
        good.replace("PRETAG_TCP6379_OK 7", "PRETAG_TCP6379_OK 6"),
    ):
        assert (
            _run(
                "receipts",
                "PRETAG_TCP6379_OK",
                "7",
                "1,2,3,4,5,6,7",
                stdin=malformed,
            ).returncode
            == 1
        )


def test_bounded_ray_join_refuses_timeout() -> None:
    refused = _run("bounded", "1", "bash", "-c", "sleep 2", timeout=3)
    assert refused.returncode == 124


def _run_mocked_launcher(tmp_path: Path, mode: str) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    ray_bin = tmp_path / "vllm-env/bin"
    fake_bin.mkdir(parents=True)
    ray_bin.mkdir(parents=True)
    call_log = tmp_path / "gcloud-calls.txt"

    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        """#!/usr/bin/env bash
set -u
printf '%s\\n' "$*" >>"$FAKE_GCLOUD_LOG"
case "$*" in
  *LAUNCH_FAILURE_CLEAN_OK*)
    if [[ $FAKE_LAUNCH_MODE == join_fail_cleanup_fail ]]; then
      echo "simulated cleanup failure" >&2
      exit 12
    fi
    for i in 0 1 2 3 4 5 6 7; do echo "LAUNCH_FAILURE_CLEAN_OK $i"; done
    ;;
  *TCP6379_OK*)
    for i in 1 2 3 4 5 6 7; do echo "TCP6379_OK $i"; done
    ;;
  *RAY_JOIN_RC*)
    if [[ $FAKE_LAUNCH_MODE == join_fail || $FAKE_LAUNCH_MODE == join_fail_cleanup_fail ]]; then
      echo "simulated join failure" >&2
      exit 9
    fi
    for i in 1 2 3 4 5 6 7; do echo "RAY_JOIN_RC_0 $i"; done
    ;;
esac
"""
    )
    gcloud.chmod(0o755)

    ray = ray_bin / "ray"
    ray.write_text(
        """#!/usr/bin/env bash
set -u
case "${1:-}" in
  stop) exit 0 ;;
  start) echo "Ray runtime started." ;;
  status)
    for i in 0 1 2 3 4 5 6 7; do echo " 1 node_$i"; done
    echo " 0.0/32.0 TPU"
    [[ $FAKE_LAUNCH_MODE != status_nonzero ]] || exit 7
    ;;
esac
"""
    )
    ray.chmod(0o755)

    sleep = fake_bin / "sleep"
    sleep.write_text("#!/usr/bin/env bash\nexit 0\n")
    sleep.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_GCLOUD_LOG": str(call_log),
            "FAKE_LAUNCH_MODE": mode,
            "RAY_JOIN_TIMEOUT_SECONDS": "2",
        }
    )
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=10,
    )


def test_launcher_cleans_all_hosts_when_worker_join_fails(tmp_path: Path) -> None:
    refused = _run_mocked_launcher(tmp_path, "join_fail")
    assert refused.returncode == 1
    assert "bounded Ray worker join failed" in refused.stderr
    calls = (tmp_path / "gcloud-calls.txt").read_text()
    assert "LAUNCH_FAILURE_CLEAN_OK" in calls
    assert "LAUNCH_FAILURE_CLEAN_OK 7" in refused.stderr


def test_launcher_success_does_not_run_failure_cleanup(tmp_path: Path) -> None:
    accepted = _run_mocked_launcher(tmp_path, "success")
    assert accepted.returncode == 0
    calls = (tmp_path / "gcloud-calls.txt").read_text()
    assert "LAUNCH_FAILURE_CLEAN_OK" not in calls
    assert "fusermount" not in calls


def test_launcher_preserves_error_when_cleanup_itself_fails(tmp_path: Path) -> None:
    refused = _run_mocked_launcher(tmp_path, "join_fail_cleanup_fail")
    assert refused.returncode == 1
    assert "bounded launcher failure cleanup failed" in refused.stderr
    calls = (tmp_path / "gcloud-calls.txt").read_text()
    assert "LAUNCH_FAILURE_CLEAN_OK" in calls


def test_launcher_refuses_nonzero_status_even_with_valid_output(tmp_path: Path) -> None:
    refused = _run_mocked_launcher(tmp_path, "status_nonzero")
    assert refused.returncode == 1
    assert "bounded Ray status failed" in refused.stderr
    assert " 0.0/32.0 TPU" in refused.stdout
    calls = (tmp_path / "gcloud-calls.txt").read_text()
    assert "LAUNCH_FAILURE_CLEAN_OK" in calls


def test_forced_dry_run_failure_never_owns_or_cleans_runtime(tmp_path: Path) -> None:
    script_dir = tmp_path / "scripts"
    fake_bin = tmp_path / "bin"
    script_dir.mkdir()
    fake_bin.mkdir()
    injected = LAUNCHER.read_text().replace(
        'if (( DRY_RUN )); then\n  dry "$RAY status',
        'if (( DRY_RUN )); then\n  exit 91\n  dry "$RAY status',
    )
    assert "exit 91" in injected
    launcher = script_dir / LAUNCHER.name
    launcher.write_text(injected)
    (script_dir / VALIDATOR.name).write_text(VALIDATOR.read_text())
    call_log = tmp_path / "gcloud-calls.txt"
    gcloud = fake_bin / "gcloud"
    gcloud.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >>"$FAKE_GCLOUD_LOG"\n')
    gcloud.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_GCLOUD_LOG": str(call_log),
        }
    )
    refused = subprocess.run(
        ["bash", str(launcher), "--dry-run"],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=5,
    )
    assert refused.returncode == 91
    assert not call_log.exists()


def test_protected_launcher_and_wrapper_are_fail_closed() -> None:
    launcher = LAUNCHER.read_text()
    wrapper = WRAPPER.read_text()
    assert 'bounded "$JOIN_TIMEOUT_SECONDS"' in launcher
    assert "JOIN_TIMEOUT_SECONDS -le 300" in launcher
    assert "RAY_JOIN_RC_\\$rc" in launcher
    assert "exit \\$rc" in launcher
    assert "TCP6379_OK" in launcher
    assert "NODE_COUNT -ne 8" in launcher
    assert "0[.]0/32[.]0 TPU" in launcher
    assert "bounded 60" in launcher
    assert '"$RAY" start --head' in launcher
    assert "RAY_JOIN_DIAGNOSTIC" in launcher
    assert "cleanup_failed_launch" in launcher
    assert "LAUNCH_FAILURE_CLEAN_OK" in launcher
    assert "fusermount -u ~/gcs-models" not in launcher
    assert "timeout --signal=TERM --kill-after=5 -- 20" in launcher
    assert "timeout --signal=TERM --kill-after=5 -- 20" in wrapper
    assert (
        'bounded 120 \\\n    gcloud compute tpus tpu-vm ssh "$POD" --zone "$ZONE" --worker=all'
        in wrapper
    )
    assert 'validate_ray_network.sh" receipts "$marker" 8' in wrapper
    assert wrapper.index("RAY_FIREWALL_RULE") < wrapper.index('mkdir -p "$RUN_DIR"')
    assert wrapper.index("pretag_probe_output") < wrapper.index('mkdir -p "$RUN_DIR"')
    assert wrapper.index("runtime_started=1") < wrapper.index(
        'bash "$HARNESS_REPO/scripts/launch_glm_32chip.sh"'
    )
    assert "firewall-rules update" not in wrapper
    assert "RAY_JOIN_TIMEOUT_SECONDS=120" in wrapper
    assert "RAY_LAUNCHER_SHA=b587a8a39262a185" in wrapper
    assert "RAY_VALIDATOR_SHA=01f675a1e701a687" in wrapper
    assert "terminal_ray_rule_contract" in wrapper
    assert "terminal pre-tag TCP/6379 receipts drifted" in wrapper
    assert 'terminal_tcp_receipts == "$ray_tcp6379_contract"' in wrapper
    assert "current_pod_state == READY" in wrapper
    assert "current_pod_health == HEALTHY" in wrapper
    assert (
        'expected_listener_receipt="PREFLIGHT_LISTENER_OK $expected_peer_ips"'
        in wrapper
    )
    assert '$(<"$RUN_DIR/ray_firewall_preflight.txt")' in wrapper
    assert wrapper.index('bash "$HARNESS_REPO/scripts/launch_glm_32chip.sh"') < (
        wrapper.index('"$RUN_DIR/prereq_post_launch.txt"')
    )
    assert wrapper.index('"$RUN_DIR/prereq_post_launch.txt"') < wrapper.index(
        "env $DRIVER_ENVS setsid --wait"
    )


def test_preflight_ip_order_is_numeric_and_terminal_receipt_is_exact() -> None:
    wrapper = WRAPPER.read_text()
    assert "key=ipaddress.ip_address" in wrapper
    assert 'terminal_tcp_receipts == "$ray_tcp6379_contract"' in wrapper
    peers = ["192.168.0.10", "192.168.0.8", "192.168.0.2"]
    assert sorted(peers, key=lambda value: tuple(map(int, value.split(".")))) == [
        "192.168.0.2",
        "192.168.0.8",
        "192.168.0.10",
    ]
