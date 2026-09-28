"""Tests of :mod:`glm_tpu.entrypoints.serve.server`: its command line."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

# `python -m glm_tpu.entrypoints.serve.server --help`, byte for byte (S5 A2): argparse wraps to the terminal
# width (pinned: COLUMNS=100) and its layout differs between Python versions (the literal is Python 3.12's).
HELP = """\
usage: server.py [-h] --run RUN [--dispatch DISPATCH] --state STATE [--port PORT]
                 [--api-key-file API_KEY_FILE] [--no-api] [--repo REPO] [--site SITE]

Loopback chat UI and OpenAI-compatible API attached to an existing resident GLM controller. This
process never initializes devices, starts workers, or sends a stop command. Private chat history
and requests live outside the checkout.

options:
  -h, --help            show this help message and exit
  --run RUN
  --dispatch DISPATCH   a dispatch receipt of the controller (default: the run's own
                        controller_identity.json and request.json)
  --state STATE
  --port PORT
  --api-key-file API_KEY_FILE
                        default: <state>/api-key, created 0600 on first start
  --no-api              serve only the browser workspace, without /v1
  --repo REPO
  --site SITE           site file for the tokenizer location (default: $GLM_TPU_SITE_CONFIG, else
                        $GLM_TPU_CONFIG_ROOT/site.toml)
"""


def test_help_text_is_unchanged():
    assert sys.version_info[:2] == (3, 12), "the literal is Python 3.12 argparse output"
    result = subprocess.run(
        [sys.executable, "-m", "glm_tpu.entrypoints.serve.server", "--help"],
        cwd=Path(__file__).resolve().parents[3],
        env=dict(os.environ, COLUMNS="100"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == HELP
