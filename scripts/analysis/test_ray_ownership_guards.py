from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    "scripts/e0_capture_arm.sh",
    "scripts/resume_health_proof.sh",
    "scripts/dcp_live_rows_exact.sh",
)


def _owned_aux_body(path: str) -> str:
    text = (ROOT / path).read_text()
    match = re.search(
        r"owned_aux_agent\(\) \{(?P<body>.*?)\n\}", text, re.DOTALL
    )
    assert match, f"missing owned_aux_agent in {path}"
    return match.group("body")


def test_all_guards_admit_only_exact_title_pairs() -> None:
    exact_pairs = (
        ("ray::DashboardA", "ray::DashboardAgent"),
        ("ray::RuntimeEnv", "ray::RuntimeEnvAgent"),
        ("ray::RayWorkerW", "ray::RayWorkerWrapper"),
        ("ray::IDLE", "ray::IDLE"),
    )
    for path in SCRIPTS:
        body = _owned_aux_body(path)
        for comm, arg0 in exact_pairs:
            assert f'[ "$name" = "{comm}" ]' in body
            assert f'[ "$arg0" = "{arg0}" ]' in body
        assert '[ "$(cat "/proc/$pp/comm" 2>/dev/null)" = "raylet" ]' in body
        assert 'owned_env "/proc/$pp/environ"' in body


def test_ray_worker_wrapper_identity_is_not_broadened() -> None:
    for path in SCRIPTS:
        body = _owned_aux_body(path)
        assert body.count('"ray::RayWorkerW"') == 1
        assert body.count('"ray::RayWorkerWrapper"') == 1
        assert "RayWorker*" not in body
        assert "*RayWorker" not in body
        assert "ray::RayWorkerWrapper*" not in body
        assert body.count('"ray::IDLE"') == 2
        assert "ray::IDLE*" not in body


def test_wrapper_discovery_does_not_self_match_controller_commands() -> None:
    for path in SCRIPTS:
        text = (ROOT / path).read_text()
        assert not re.search(
            r"^.*pgrep.*RayWorkerWrapper.*$", text, re.MULTILINE
        ), path
        assert (
            'engine_pids=$(pgrep -f "VLLM::[E]ngineCore"' in text
        ), path
        assert '"$engine_pids"' in text, path
        assert "vllm_pids" not in text, path


def test_post_driver_cleanup_retries_transient_wrapper_titles() -> None:
    for path in (
        "scripts/resume_health_proof.sh",
        "scripts/dcp_live_rows_exact.sh",
    ):
        text = (ROOT / path).read_text()
        assert 'while ! ownership_census ' in text, path
        assert '[ "$attempt" -lt 3 ] || return 1' in text, path
        assert 'sleep 10' in text, path
        assert 'census_label="prestop_${label}_retry${attempt}"' in text, path


def test_protected_watchdogs_cannot_retain_the_global_lease() -> None:
    for path in (
        "scripts/e0_capture_arm.sh",
        "scripts/resume_health_proof.sh",
        "scripts/dcp_live_rows_exact.sh",
    ):
        text = (ROOT / path).read_text()
        assert '2>&1 9>&- &' in text, path
        assert 'kill -TERM -- "-$WATCH_PID"' in text, path
        assert 'kill -KILL -- "-$WATCH_PID"' in text, path


def test_e0_reserves_trace_headroom_above_runtime_disk_floor() -> None:
    text = (ROOT / "scripts/e0_capture_arm.sh").read_text()
    assert 'E0_PREFLIGHT_MIN_FREE_GB="${E0_PREFLIGHT_MIN_FREE_GB:-17}"' in text
    assert 'if [ "$E0_PREFLIGHT_MIN_FREE_GB" -lt 17 ]' in text
    assert text.count('MIN_FREE_GB="$E0_PREFLIGHT_MIN_FREE_GB"') == 2
    assert "e0_preflight_min_free_gb=%s" in text


def test_disk_watchdog_compares_byte_exact_free_space() -> None:
    text = (ROOT / "scripts/disk_watchdog.sh").read_text()
    assert 'df -B1 --output=avail /' in text
    assert 'df -B1G --output=avail /' not in text
    assert 'min_free_bytes=$((MIN_FREE_GB * gib))' in text
    assert 'if [ "$free_bytes" -lt "$min_free_bytes" ]' in text


def test_compute_rows_exactness_selector_is_single_variable() -> None:
    shared = (ROOT / "scripts/dcp_live_rows_exact.sh").read_text()
    wrapper = (ROOT / "scripts/moe_compute_rows_exact.sh").read_text()
    assert "export EXACT_LEVER=moe_compute_rows" in wrapper
    assert 'export EXACT_PIN="${EXACT_PIN:-b3c25df47}"' in wrapper
    assert 'GLM_DSA_DCP_DECODE_LIVE_ROWS=$dcp_gate' in shared
    assert 'GLM_MOE_DECODE_ALL_GATHER=$moe_allgather_gate' in shared
    assert 'GLM_MOE_DECODE_COMPUTE_LIVE_ROWS=$moe_compute_gate' in shared
    assert 'if [ "$EXACT_LEVER" = moe_compute_rows ]; then' in shared
    assert (
        'moe_allgather_gate = side_gate if exact_lever == "moe_allgather" else "0"'
        in shared
    )
    assert (
        'moe_compute_gate = side_gate if exact_lever == "moe_compute_rows" else "0"'
        in shared
    )


def test_compute_rows_gate_reaches_health_and_e0_provenance() -> None:
    for path in (
        "scripts/resume_health_proof.sh",
        "scripts/e0_capture_arm.sh",
    ):
        text = (ROOT / path).read_text()
        assert 'MOE_DECODE_COMPUTE_LIVE_ROWS="${E0_MOE_DECODE_COMPUTE_LIVE_ROWS:-0}"' in text
        assert 'GLM_MOE_DECODE_COMPUTE_LIVE_ROWS=$MOE_DECODE_COMPUTE_LIVE_ROWS' in text
        assert 'GLM_MOE_DECODE_COMPUTE_LIVE_ROWS armed' in text
        assert '"GLM_MOE_DECODE_COMPUTE_LIVE_ROWS": moe_decode_compute_live_rows' in text
