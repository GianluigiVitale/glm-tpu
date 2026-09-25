"""Release CLI must remain safe on a controller with active TPU work."""

from contextlib import redirect_stdout
from importlib import metadata
import io
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

import pytest

from glm_tpu.entrypoints.cli.collect_env import environment_manifest, environment_report
from glm_tpu.entrypoints.cli.main import main


class CliTests(unittest.TestCase):
    def test_complete_metadata(self):
        pins = {name: value for group in environment_manifest()["profiles"].values() for name, value in group.items()}
        result = environment_report("benchmark", version=pins.__getitem__, python_version=(3, 12))
        self.assertTrue(result["passed"])
        self.assertTrue(any(row["expected"] == "2.10.0+cpu" for row in result["packages"]))

    def test_missing_and_wrong_versions_refuse(self):
        def lookup(name):
            if name == "jax":
                raise metadata.PackageNotFoundError(name)
            return "wrong"

        result = environment_report("core", version=lookup, python_version=(3, 12))
        self.assertFalse(result["passed"])
        self.assertEqual({row["status"] for row in result["packages"]}, {"missing", "mismatch"})

    def test_wrong_python_refuses(self):
        pins = environment_manifest()["profiles"]["core"]
        self.assertFalse(environment_report("core", version=pins.__getitem__, python_version=(3, 13))["passed"])

    def test_unknown_profile_refuses(self):
        with self.assertRaises(ValueError):
            environment_report("unknown")

    def test_info_is_honest(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["info"]), 0)
        info = json.loads(output.getvalue())
        self.assertIn("not established", info["quality"])
        self.assertEqual(info["concurrent_requests"], 4)
        self.assertEqual(info["model"], "zai-org/GLM-5.3")
        self.assertIn("four correct completed", info["concurrent_validation"])

    def test_no_model_import_in_fresh_process(self):
        code = """
import sys
from glm_tpu.entrypoints.cli.main import main
main(["doctor", "--profile", "tpu"])
for name in ("jax", "jaxlib", "libtpu", "torch", "transformers"):
    assert name not in sys.modules, name
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_manifest_matches_declared_dependencies(self):
        repo = Path(__file__).resolve().parents[3]
        project = tomllib.loads((repo / "pyproject.toml").read_text())["project"]
        profiles = environment_manifest()["profiles"]
        for group, pins in profiles.items():
            declared = project["dependencies"] if group == "core" else project["optional-dependencies"][group]
            expected = {f"{name}=={pin.split('+')[0] if name == 'torch' else pin}" for name, pin in pins.items()}
            self.assertEqual(set(declared), expected)


# The command-line surface, byte for byte (S5 A2): the help of every subcommand and the `info` output of an
# uninstalled checkout. argparse wraps to the terminal width (pinned: COLUMNS=100) and its layout differs
# between Python versions (the literals are Python 3.12's).
HELP = {
    "": """\
usage: glm-tpu [-h] {info,ask,doctor,prepare-request} ...

Release information, local preparation and protected question submission.

positional arguments:
  {info,ask,doctor,prepare-request}
    info                show supported scope and release limitations
    ask                 answer questions on the retained TPU site; optionally batch up to four
    doctor              check installed version metadata without initializing TPU
    prepare-request     tokenize a private chat locally; does NOT launch inference

options:
  -h, --help            show this help message and exit
""",
    "info": """\
usage: glm-tpu info [-h]

options:
  -h, --help  show this help message and exit
""",
    "ask": """\
usage: glm-tpu ask [-h] [--questions QUESTIONS] [--context {8k,32k,128k,256k}] [--keep-loaded]
                   [--concurrent] [--max-new-tokens MAX_NEW_TOKENS] [--wall-seconds WALL_SECONDS]
                   [--prepare-only] [--site SITE]
                   [question]

positional arguments:
  question

options:
  -h, --help            show this help message and exit
  --questions QUESTIONS
                        private JSON array of one to ten question strings
  --context {8k,32k,128k,256k}
  --keep-loaded         retain the ordinary model and fleet leases after answering; explicit stop
                        required
  --concurrent          batch up to four conversations; requires --context 32k
  --max-new-tokens MAX_NEW_TOKENS
                        optional output cap; default: all remaining context slots (up to 163840 in
                        128k mode); thinking and answer share this space
  --wall-seconds WALL_SECONDS
  --prepare-only        prepare private inputs without launching the model
  --site SITE           site file (default: $GLM_TPU_SITE_CONFIG, else
                        $GLM_TPU_CONFIG_ROOT/site.toml)
""",
    "doctor": """\
usage: glm-tpu doctor [-h] [--profile {core,runtime,tpu,benchmark,dev}]

options:
  -h, --help            show this help message and exit
  --profile {core,runtime,tpu,benchmark,dev}
""",
    "prepare-request": """\
usage: glm-tpu prepare-request [-h] --messages MESSAGES --output OUTPUT --repo REPO
                               --tokenizer-root TOKENIZER_ROOT --request-id REQUEST_ID
                               [--profile {ordinary-greedy-8k,ordinary-greedy-128k}]
                               --max-new-tokens MAX_NEW_TOKENS

options:
  -h, --help            show this help message and exit
  --messages MESSAGES
  --output OUTPUT
  --repo REPO
  --tokenizer-root TOKENIZER_ROOT
  --request-id REQUEST_ID
  --profile {ordinary-greedy-8k,ordinary-greedy-128k}
  --max-new-tokens MAX_NEW_TOKENS
""",
}

INFO = """\
{
  "concurrent_context_capacity": 32768,
  "concurrent_requests": 4,
  "concurrent_scope": "fixed submitted group; see STATUS for hardware evidence; no online request admission",
  "concurrent_validation": "GLM-5.3: four correct completed GSM8K answers; normal EOS, eight-host agreement and cleanup; short inputs only",
  "engine": "native JAX WS32_2D",
  "hardware": "8 hosts / 32 TPU v4 chips",
  "installation": "wheel contains Python components only; full deployment requires source checkout and external assets",
  "model": "zai-org/GLM-5.3",
  "ordinary_profile": "greedy; 8K combined, concurrent 32K per conversation, or 128K prompt / 166912 combined slots; see STATUS for measured scope",
  "project": "glm-tpu",
  "quality": "full benchmark/model-card parity not established",
  "queued_questions": 10,
  "release_status": "private project; see docs/release/STATUS.md for trained admission and promotion",
  "resume": "resident ordinary inbox reuses the live model; no process-restart or durable KV recovery",
  "serving": "site-specific protected request harness; no supported HTTP endpoint",
  "version": "source checkout (not installed)"
}
"""  # noqa: E501 (the info output, byte for byte)


def run_cli(monkeypatch, capsys, argv: list[str]) -> str:
    """``glm-tpu ARGV``'s standard output (the console script's program name, 100 columns)."""
    assert sys.version_info[:2] == (3, 12), "the literals are Python 3.12 argparse output"
    monkeypatch.setenv("COLUMNS", "100")
    monkeypatch.setattr(sys, "argv", ["glm-tpu", *argv])
    capsys.readouterr()
    try:
        code = main(argv)
    except SystemExit as done:
        code = done.code
    assert code == 0
    return capsys.readouterr().out


@pytest.mark.parametrize("command", list(HELP), ids=[name or "glm-tpu" for name in HELP])
def test_help_text_is_unchanged(monkeypatch, capsys, command):
    assert run_cli(monkeypatch, capsys, [*command.split(), "--help"]) == HELP[command]


def test_info_output_is_unchanged(monkeypatch, capsys):
    def uninstalled(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", uninstalled)
    assert run_cli(monkeypatch, capsys, ["info"]) == INFO


if __name__ == "__main__":
    unittest.main()
