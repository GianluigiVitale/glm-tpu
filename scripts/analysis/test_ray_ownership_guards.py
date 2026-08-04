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
