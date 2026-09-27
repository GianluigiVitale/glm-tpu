"""Tests of :mod:`glm_tpu.model_loader.pack_worker`: its command line."""

from __future__ import annotations

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

CPU-only GLM-5.3 owner packing on one host, started by a fleet packing driver outside this
repository.

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
