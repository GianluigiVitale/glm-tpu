"""Tests of :mod:`glm_tpu.model_loader.pack_worker`: its command line and the checks of its seal modes (the modes
themselves run on the tiny checkpoint in tests/executor/test_pack_job.py)."""

from __future__ import annotations

from argparse import Namespace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import pytest

from glm_tpu.model_loader import pack_worker

# `python -m glm_tpu.model_loader.pack_worker --help`, byte for byte: argparse wraps to the terminal width (pinned:
# COLUMNS=100) and its layout differs between Python versions (the literal is Python 3.12's). The description is the
# module docstring.
HELP = """\
usage: pack_worker.py [-h] --output OUTPUT --source-inventory SOURCE_INVENTORY --code-hash
                      CODE_HASH --source-manifest-sha256 SOURCE_MANIFEST_SHA256 --site-sha256
                      SITE_SHA256 --source-complete-sha256 SOURCE_COMPLETE_SHA256
                      --source-inventory-sha256 SOURCE_INVENTORY_SHA256
                      --source-inventory-file-sha256 SOURCE_INVENTORY_FILE_SHA256
                      --topology-rebinding-sha256 TOPOLOGY_REBINDING_SHA256 [--preflight-only]
                      [--compare-seal | --install-seal]
                      [--seal-manifest-sha256 SEAL_MANIFEST_SHA256]
                      [--seal-success-sha256 SEAL_SUCCESS_SHA256]
                      [--compare-tensors COMPARE_TENSORS]

CPU-only GLM-5.3 owner packing on one host, started on every host by glm-tpu checkpoint pack.
Default: pack this host's slots into a new checkpoint root. --compare-seal: re-derive chosen
tensors of this host's slots in memory and compare them and every file plan with the staged sealed
manifest (writes nothing to tmpfs). --install-seal: install the staged manifest.json and SUCCESS
in this host's packed root, then verify it. --preflight-only: the mode's checks, then this host's
facts.

options:
  -h, --help            show this help message and exit
  --output OUTPUT
  --source-inventory SOURCE_INVENTORY
  --code-hash CODE_HASH
  --source-manifest-sha256 SOURCE_MANIFEST_SHA256
  --site-sha256 SITE_SHA256
  --source-complete-sha256 SOURCE_COMPLETE_SHA256
  --source-inventory-sha256 SOURCE_INVENTORY_SHA256
  --source-inventory-file-sha256 SOURCE_INVENTORY_FILE_SHA256
  --topology-rebinding-sha256 TOPOLOGY_REBINDING_SHA256
  --preflight-only
  --compare-seal
  --install-seal
  --seal-manifest-sha256 SEAL_MANIFEST_SHA256
  --seal-success-sha256 SEAL_SUCCESS_SHA256
  --compare-tensors COMPARE_TENSORS
"""


def test_help_text_is_unchanged(monkeypatch, capsys):
    # In this process, as `python -m` runs it (sys.argv[0] is the module's file): a child process whose argv names
    # the pack worker would make a concurrent cpu32 test or heavy gate see a live TPU run (tools.equivalence budget).
    # `python -m glm_tpu.model_loader.pack_worker --help` itself: tests/engine/test_resident_protocol.py.
    assert sys.version_info[:2] == (3, 12), "the literal is Python 3.12 argparse output"
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setattr(sys, "argv", [pack_worker.__file__, "--help"])
    with pytest.raises(SystemExit) as exited:
        pack_worker.main()
    assert exited.value.code == 0
    assert capsys.readouterr().out == HELP


def staged(tmp_path: Path, manifest: bytes = b'{"files": []}\n', success: bytes | None = b'{"success_sha256": "5"}\n'):
    """A run directory holding a staged seal/ (manifest.json and SUCCESS), owner-only."""
    seal = tmp_path / "run" / pack_worker.SEAL_DIR
    seal.mkdir(parents=True, mode=0o700)
    (seal / "manifest.json").write_bytes(manifest)
    if success is not None:
        (seal / "SUCCESS").write_bytes(success)
    return Namespace(
        output=tmp_path / "run", seal_manifest_sha256=sha256(manifest).hexdigest(), seal_success_sha256="5"
    )


def test_staged_seal_reads_the_digests_the_controller_named(tmp_path: Path):
    args = staged(tmp_path)
    manifest, raw, success, success_raw = pack_worker.staged_seal(args, success=True)
    assert (manifest, raw, success) == ({"files": []}, b'{"files": []}\n', {"success_sha256": "5"})
    assert json.loads(success_raw) == success
    assert pack_worker.staged_seal(args, success=False)[2:] == (None, None)


@pytest.mark.parametrize(
    "change,message",
    [
        (dict(seal_manifest_sha256="0" * 64), "staged sealed manifest differs"),
        (dict(seal_success_sha256="6"), "staged SUCCESS differs"),
    ],
)
def test_staged_seal_refuses_other_bytes(tmp_path: Path, change: dict, message: str):
    args = staged(tmp_path)
    for name, value in change.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match=message):
        pack_worker.staged_seal(args, success=True)


def test_a_seal_mode_needs_the_staged_seals_digests():
    umask = os.umask(0o022)  # main sets the owner-only umask of a pack worker process: restore it after the test
    os.umask(umask)
    try:
        _seal_mode_refusals()
    finally:
        os.umask(umask)


def _seal_mode_refusals():
    base = ["--output", "/r/x", "--source-inventory", "/i"]
    base += [
        arg
        for name in (
            "code-hash",
            "source-manifest-sha256",
            "site-sha256",
            "source-complete-sha256",
            "source-inventory-sha256",
            "source-inventory-file-sha256",
            "topology-rebinding-sha256",
        )
        for arg in ("--" + name, "0")
    ]
    with pytest.raises(ValueError, match="digests are required"):
        pack_worker.main([*base, "--compare-seal"])
    with pytest.raises(ValueError, match="digests are required"):
        pack_worker.main([*base, "--install-seal", "--seal-manifest-sha256", "0" * 64])
    with pytest.raises(SystemExit):  # one mode at a time
        pack_worker.main([*base, "--compare-seal", "--install-seal", "--seal-manifest-sha256", "0" * 64])
