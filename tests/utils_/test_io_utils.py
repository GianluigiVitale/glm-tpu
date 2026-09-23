"""Create-once record writes: never overwrite, never follow a link, never raise on a difference."""
from __future__ import annotations

import json
import os
import stat

import pytest

from glm_tpu.utils.io_utils import FINAL_DIR, create_private_exclusive, write_collected

RECORD = {"complete": True, "rank": 3, "request": {"emitted": 3}}


def persisted(value) -> bytes:
    """The bytes the worker's and the controller's ``persist`` write."""
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def test_create_private_exclusive_is_owner_only_and_create_once(tmp_path):
    path = tmp_path / "record.json"
    create_private_exclusive(path, b"one")
    assert path.read_bytes() == b"one" and stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        create_private_exclusive(path, b"two")
    assert path.read_bytes() == b"one"


def test_create_private_exclusive_never_follows_a_symlink(tmp_path):
    victim = tmp_path / "victim"
    victim.write_bytes(b"keep")
    (tmp_path / "link").symlink_to(victim)
    with pytest.raises(FileExistsError):
        create_private_exclusive(tmp_path / "link", b"attack")
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    with pytest.raises(FileExistsError):
        create_private_exclusive(tmp_path / "dangling", b"attack")
    assert victim.read_bytes() == b"keep" and not (tmp_path / "nowhere").exists()


def test_an_absent_record_is_created(tmp_path):
    divergent: list[str] = []
    write_collected(tmp_path, "runner.rank3.json", persisted(RECORD), divergent)
    assert (tmp_path / "runner.rank3.json").read_bytes() == persisted(RECORD)
    assert divergent == [] and not (tmp_path / FINAL_DIR).exists()


def test_equal_bytes_are_skipped(tmp_path):
    (tmp_path / "runner.rank3.json").write_bytes(persisted(RECORD))
    divergent: list[str] = []
    write_collected(tmp_path, "runner.rank3.json", persisted(RECORD), divergent)
    assert divergent == [] and not (tmp_path / FINAL_DIR).exists()


def test_a_json_equal_but_byte_different_record_is_skipped(tmp_path):
    (tmp_path / "runner.rank3.json").write_bytes(persisted(RECORD))
    compact = json.dumps(dict(reversed(list(RECORD.items())))).encode()
    assert compact != persisted(RECORD)
    divergent: list[str] = []
    write_collected(tmp_path, "runner.rank3.json", compact, divergent)
    assert divergent == [] and not (tmp_path / FINAL_DIR).exists()
    assert (tmp_path / "runner.rank3.json").read_bytes() == persisted(RECORD)


def test_a_different_record_is_kept_in_final_and_listed_never_overwritten(tmp_path):
    (tmp_path / "runner.rank3.json").write_bytes(persisted(RECORD))
    failed = persisted(dict(RECORD, failure_type="RuntimeError"))
    divergent: list[str] = []
    write_collected(tmp_path, "runner.rank3.json", failed, divergent)
    assert divergent == ["runner.rank3.json"]
    assert (tmp_path / "runner.rank3.json").read_bytes() == persisted(RECORD)
    final = tmp_path / FINAL_DIR / "runner.rank3.json"
    assert final.read_bytes() == failed and stat.S_IMODE(final.stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / FINAL_DIR).stat().st_mode) == 0o700
    # the same payload again: already preserved, listed once
    write_collected(tmp_path, "runner.rank3.json", failed, divergent)
    assert divergent == ["runner.rank3.json"] and sorted(os.listdir(tmp_path / FINAL_DIR)) == ["runner.rank3.json"]
    # yet another version: kept next to the others, nothing replaced
    other = persisted(dict(RECORD, failure_type="ValueError"))
    write_collected(tmp_path, "runner.rank3.json", other, divergent)
    assert (tmp_path / FINAL_DIR / "runner.rank3.json.1").read_bytes() == other
    assert final.read_bytes() == failed and divergent == ["runner.rank3.json"]


def test_non_json_content_counts_only_by_bytes(tmp_path):
    (tmp_path / "blob").write_bytes(b"\xff\xfe not json")
    divergent: list[str] = []
    write_collected(tmp_path, "blob", b"\xff\xfe not json", divergent)
    assert divergent == []
    write_collected(tmp_path, "blob", b"other", divergent)
    assert divergent == ["blob"] and (tmp_path / FINAL_DIR / "blob").read_bytes() == b"other"


def test_an_existing_symlink_is_divergent_and_never_followed(tmp_path):
    victim = tmp_path / "victim.json"
    victim.write_bytes(persisted(RECORD))
    (tmp_path / "runner.rank3.json").symlink_to(victim)
    divergent: list[str] = []
    write_collected(tmp_path, "runner.rank3.json", persisted(RECORD), divergent)
    assert divergent == ["runner.rank3.json"] and victim.read_bytes() == persisted(RECORD)
    assert (tmp_path / "runner.rank3.json").is_symlink()


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "../x"])
def test_a_record_name_must_be_a_plain_file_name(tmp_path, name):
    with pytest.raises(ValueError):
        write_collected(tmp_path, name, b"x", [])
