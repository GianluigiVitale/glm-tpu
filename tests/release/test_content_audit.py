"""Heuristic audit behavior with generated dummy credentials and isolated Git."""
import json
import subprocess

import pytest

from tools import audit_release_content as audit


def git(repo, *args, data=None):
    return subprocess.check_output(["git", *args], cwd=repo, input=data)


def repository(tmp_path):
    git(tmp_path, "init", "-q")
    return tmp_path


def test_detects_without_echoing_the_dummy_value():
    value = b"hf_" + b"x" * 35
    report = audit.matches(b"line one\n" + value)
    assert report == [{"kind": "huggingface_token", "line": 2}]
    assert value.decode() not in json.dumps(report)


@pytest.mark.parametrize("name", [".env", ".env.local", "nested/key.pem", "weights/a.safetensors",
                                 "bench/results.db", "bench/data/questions.csv", "id_ed25519",
                                 "request.json", "run/tokens.jsonl", "answer.txt.gz"])
def test_sensitive_payload_names(name):
    assert audit.forbidden_name(name)


def test_worktree_scan_refuses_symlink_and_reports_locations_only(tmp_path):
    repository(tmp_path)
    (tmp_path / "plain.txt").write_text("public fixture")
    (tmp_path / "dummy.txt").write_bytes(b"ghp_" + b"a" * 36)
    (tmp_path / "alias").symlink_to(tmp_path / "plain.txt")
    git(tmp_path, "add", "plain.txt", "dummy.txt", "alias")
    result = audit.scan_tree(tmp_path)
    assert result["files"] == 3
    assert result["findings"] == [{"path": "dummy.txt", "kind": "github_token", "line": 1}]
    assert result["errors"] == [{"path": "alias", "kind": "symlink_not_followed"}]


def test_history_includes_unreachable_blobs_and_respects_byte_budget(tmp_path, monkeypatch):
    repository(tmp_path)
    dummy = b"hf_" + b"x" * 35
    oid = git(tmp_path, "hash-object", "-w", "--stdin", data=dummy).decode().strip()
    result = audit.scan_history(tmp_path)
    assert result["scanned_blobs"] == 1
    assert result["findings"] == [dict(object=oid, path_hint=None, kind="huggingface_token", line=1)]
    assert dummy.decode() not in json.dumps(result)
    monkeypatch.setattr(audit, "HISTORY_CAP", 1)
    with pytest.raises(ValueError): audit.scan_history(tmp_path)


def test_oversized_blob_is_an_explicit_error_not_a_pass(tmp_path, monkeypatch):
    repository(tmp_path)
    git(tmp_path, "hash-object", "-w", "--stdin", data=b"too large for fixture cap")
    monkeypatch.setattr(audit, "FILE_CAP", 1)
    result = audit.scan_history(tmp_path)
    assert result["scanned_blobs"] == 0
    assert result["errors"][0]["kind"] == "oversized_blob_not_scanned"


def test_fast_prefix_filter_preserves_all_regex_results():
    data = b"\n".join((b"hf_" + b"x"*35, b"ghp_" + b"a"*36,
        b"-----BEGIN " + b"PRIVATE KEY-----", b"sk-ant-" + b"a"*40,
        b"sk-proj-" + b"a"*40, b"AIza" + b"a"*35, b"ya29." + b"a"*35,
        b"AKIA" + b"A"*16, b"ASIA" + b"A"*16, b"xoxb-" + b"a"*25))
    expected = [dict(kind=kind, line=data.count(b"\n", 0, match.start())+1)
                for kind, pattern in audit.COMPILED.items() for match in pattern.finditer(data)]
    assert audit.matches(data) == expected
