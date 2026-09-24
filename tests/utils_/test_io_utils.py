"""Create-once record writes: never overwrite, never follow a link, never raise on a difference.

Bounded input reads (``read_bounded``, in ``glm_tpu/engine/_s3_user_request.py`` until S4.1 moves it
to ``glm_tpu/utils/io_utils.py``): a regular file within its byte cap, never through a symlink.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from glm_tpu.engine._s3_user_request import PAYLOAD_CAP, read_bounded
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


def test_read_bounded_returns_a_regular_file_within_its_cap(tmp_path):
    path = tmp_path / "request.json"
    path.write_bytes(b'{"a":1}')
    assert read_bounded(path, PAYLOAD_CAP) == b'{"a":1}'
    assert read_bounded(path, 7) == b'{"a":1}'


def test_read_bounded_refuses_symlinks_oversize_and_non_regular_files(tmp_path):
    path = tmp_path / "request.json"
    path.write_bytes(b'{"a":1}')
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="symlinks"):
        read_bounded(alias, PAYLOAD_CAP)
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        read_bounded(linked / "request.json", PAYLOAD_CAP)
    with pytest.raises(ValueError, match="bounded regular file"):
        read_bounded(path, 1)
    (tmp_path / "empty").write_bytes(b"")
    with pytest.raises(ValueError, match="bounded regular file"):
        read_bounded(tmp_path / "empty", PAYLOAD_CAP)
    with pytest.raises(IsADirectoryError):  # refused, but at fdopen, before the ValueError check
        read_bounded(tmp_path, PAYLOAD_CAP)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)  # opened non-blocking: refused, never waited on
    with pytest.raises(ValueError, match="bounded regular file"):
        read_bounded(fifo, PAYLOAD_CAP)
