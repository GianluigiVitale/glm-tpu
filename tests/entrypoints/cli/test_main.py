"""Release CLI must remain safe on a controller with active TPU work."""

import argparse
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


def built_parser(monkeypatch) -> argparse.ArgumentParser:
    """The parser ``main`` builds, taken where it would parse (whichever module defines the subcommands)."""

    class Built(Exception):
        pass

    def capture(self, args=None, namespace=None):
        raise Built(self)

    monkeypatch.setattr(sys, "argv", ["glm-tpu"])
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(Built) as built:
        main([])
    return built.value.args[0]


def arguments(parser: argparse.ArgumentParser) -> list[str]:
    """Every action in order: its kind, flags (or dest) and each attribute that is set (help: the HELP snapshot)."""
    rows = []
    for action in parser._actions:
        row = [type(action).__name__, "/".join(action.option_strings) or action.dest, f"dest={action.dest}"]
        for field in ("nargs", "const", "default", "type", "choices", "metavar"):
            value = getattr(action, field)
            if field == "type" and value is not None:
                value = value.__name__
            elif field == "choices" and value is not None:
                value = list(value)
            if value is not None:
                row.append(f"{field}={value!r}")
        rows.append(" ".join(row + ["required"] * action.required))
    rows += [
        f"exclusive required={group.required}: " + " ".join(a.dest for a in group._group_actions)
        for group in parser._mutually_exclusive_groups
    ]
    return rows


# Each parser's actions (arguments(): flags, dest, nargs, const, default, type, choices, metavar, required) and its
# mutually exclusive groups, in order. The HELP snapshot above pins the help texts and the layout, not these.
ARGUMENTS = {
    "": [
        "_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='",
        "_SubParsersAction command dest=command nargs='A...' "
        "choices=['info', 'ask', 'doctor', 'prepare-request'] required",
    ],
    "info": ["_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='"],
    "ask": [
        "_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='",
        "_StoreAction question dest=question nargs='?'",
        "_StoreAction --questions dest=questions type='Path'",
        "_StoreAction --context dest=context default='32k' choices=['8k', '32k', '128k', '256k']",
        "_StoreTrueAction --keep-loaded dest=keep_loaded nargs=0 const=True default=False",
        "_StoreTrueAction --concurrent dest=concurrent nargs=0 const=True default=False",
        "_StoreAction --max-new-tokens dest=max_new_tokens type='int'",
        "_StoreAction --wall-seconds dest=wall_seconds default=86400 type='int'",
        "_StoreTrueAction --prepare-only dest=prepare_only nargs=0 const=True default=False",
        "_StoreAction --site dest=site type='Path'",
        "exclusive required=True: question questions",
    ],
    "doctor": [
        "_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='",
        "_StoreAction --profile dest=profile default='core' choices=['core', 'runtime', 'tpu', 'benchmark', 'dev']",
    ],
    "prepare-request": [
        "_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='",
        "_StoreAction --messages dest=messages type='Path' required",
        "_StoreAction --output dest=output type='Path' required",
        "_StoreAction --repo dest=repo type='Path' required",
        "_StoreAction --tokenizer-root dest=tokenizer_root type='Path' required",
        "_StoreAction --request-id dest=request_id required",
        "_StoreAction --profile dest=profile default='ordinary-greedy-8k' "
        "choices=['ordinary-greedy-8k', 'ordinary-greedy-128k']",
        "_StoreAction --max-new-tokens dest=max_new_tokens type='int' required",
    ],
}


def test_every_argument_is_unchanged(monkeypatch):
    parser = built_parser(monkeypatch)
    (commands,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    parsers = {"": parser, **commands.choices}
    assert {name: arguments(p) for name, p in parsers.items()} == ARGUMENTS
    assert [(p.prog, p.description) for p in parsers.values()] == [
        ("glm-tpu", "Release information, local preparation and protected question submission."),
        *((f"glm-tpu {name}", None) for name in list(ARGUMENTS)[1:]),
    ]
    defaults = argparse.ArgumentParser()
    settings = (
        "usage",
        "epilog",
        "formatter_class",
        "prefix_chars",
        "fromfile_prefix_chars",
        "argument_default",
        "conflict_handler",
        "add_help",
        "allow_abbrev",
        "exit_on_error",
        "_defaults",
    )
    for p in parsers.values():
        assert [getattr(p, name) for name in settings] == [getattr(defaults, name) for name in settings], p.prog


@pytest.mark.parametrize("profile", ["core", "runtime", "tpu", "benchmark", "dev", "missing jax"])
def test_doctor_prints_the_report_and_exits_by_its_verdict(monkeypatch, capsys, profile):
    """`doctor` prints the profile's report and exits 0 exactly when it passed (every pin installed, Python 3.12)."""
    pins = {name: value for group in environment_manifest()["profiles"].values() for name, value in group.items()}

    def installed(name):
        if profile == "missing jax" and name == "jax":
            raise metadata.PackageNotFoundError(name)
        return pins[name]

    # environment_report's keyword defaults: the same function object wherever the command looks it up.
    monkeypatch.setitem(environment_report.__kwdefaults__, "version", installed)
    monkeypatch.setitem(environment_report.__kwdefaults__, "python_version", (3, 12))
    name = "core" if profile == "missing jax" else profile
    report = environment_report(name)
    assert report["passed"] is (profile != "missing jax")
    capsys.readouterr()
    assert main(["doctor", "--profile", name]) == (0 if report["passed"] else 1)
    assert capsys.readouterr() == (json.dumps(report, indent=2, sort_keys=True) + "\n", "")


if __name__ == "__main__":
    unittest.main()
