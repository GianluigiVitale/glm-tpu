"""Exercise actual wrapper census/budget code without cloud or device access."""

from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "scripts/greenfield/run_short_decoder_ws32.sh"


@pytest.mark.parametrize("normal,root", [(1, 1), (0, 1), (1, 0), (0, 0)])
def test_batched_census_requires_both_and_preserves_root(tmp_path, normal, root):
    source = WRAPPER.read_text()
    start = source.index("strict_census() {")
    function = source[start : source.index("\n}\n\nexec 9>", start) + 3]
    code = (
        r"""
set -u
RUN_DIR=$1
TAG=unit
POD=pod
ZONE=zone
PREFILL_MODE=layer_major_raw_v1
NORMAL=$2
ROOT_OK=$3
gcloud() {
  case "$*" in
    *ROOT_COMMAND*) echo 'ROOT_EVIDENCE'; test "$ROOT_OK" = 1 ;;
    *) echo 'CENSUS_OK unit'; test "$NORMAL" = 1 ;;
  esac
}
has_eight_unique_markers() { test "$NORMAL" = 1; }
seal_python() {
  case "$*" in
    *census-command*) echo ROOT_COMMAND ;;
    *validate-fleet*) test "$ROOT_OK" = 1 ;;
    *) return 1 ;;
  esac
}
"""
        + function
        + "\nstrict_census probe\n"
    )
    result = subprocess.run(
        ["bash", "-c", code, "census-test", str(tmp_path), str(normal), str(root)],
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert (result.returncode == 0) == bool(normal and root)
    assert "ROOT_EVIDENCE" in (tmp_path / "census_probe.txt").read_text()
    assert (tmp_path / "census_root_probe.txt").read_text() == "ROOT_EVIDENCE\n"


@pytest.mark.parametrize(
    "mode,context,expected",
    [
        ("layer_major_raw_v1", "2k", 2700),
        ("layer_major_raw_v1", "8k", 2700),
        ("serial_teacher_forced_v1", "8k", 14400),
        ("serial_teacher_forced_v1", "128k_d0_0", 39600),
        ("serial_teacher_forced_v1", "256k_e0", 72000),
    ],
)
def test_acquisition_has_separate_hard_ceiling(mode, context, expected):
    source = WRAPPER.read_text()
    start = source.index(
        "if [[ $PREFILL_MODE == layer_major_raw_v1 ]]; then",
        source.index("# The worker wall limit"),
    )
    stop = source.index("\nif [[ $MODE == acquire ]]; then", start)
    code = "set -eu\nPREFILL_MODE=$1\nCONTEXT=$2\n" + source[start:stop]
    code += "\n[[ $WORKER_TIMEOUT_SECONDS -eq $3 ]]\n"
    subprocess.run(
        ["bash", "-c", code, "budget-test", mode, context, str(expected)], check=True
    )


def test_pinned_guard_and_archived_census_are_used():
    source = WRAPPER.read_text()
    assert (
        "seal_python -m scripts.greenfield.fp8_baseline_guard census-command" in source
    )
    assert (
        'seal_python -m scripts.greenfield.fp8_baseline_guard validate-fleet --file "$root_out"'
        in source
    )
    assert 'cat "$root_out" >>"$out"' in source
    assert "census_pre.txt census_failure_exit.txt" in source
