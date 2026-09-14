"""Retained byte-exact disk-floor guard; never contacts workers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_disk_watchdog_compares_byte_exact_free_space() -> None:
    text = (ROOT / "scripts/disk_watchdog.sh").read_text()
    assert 'df -B1 --output=avail /' in text
    assert 'df -B1G --output=avail /' not in text
    assert 'min_free_bytes=$((MIN_FREE_GB * gib))' in text
    assert 'if [ "$free_bytes" -lt "$min_free_bytes" ]' in text
