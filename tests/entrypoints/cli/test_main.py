"""Release CLI must remain safe on a controller with active TPU work."""

import argparse
from contextlib import redirect_stdout
from importlib import metadata
import io
import json
import subprocess
import sys
import unittest

import pytest

from glm_tpu.entrypoints.cli.collect_env import declared_requirements, environment_report
from glm_tpu.entrypoints.cli.main import main


class CliTests(unittest.TestCase):
    def test_complete_metadata(self):
        pins = {name: value for group in declared_requirements()["groups"].values() for name, value in group.items()}
        installed = {**pins, "torch": pins["torch"] + "+cpu"}
        result = environment_report("tpu", version=installed.__getitem__, python_version=(3, 12))
        self.assertTrue(result["passed"])
        self.assertIn(
            dict(package="torch", expected="2.10.0", installed="2.10.0+cpu", status="match"), result["packages"]
        )

    def test_missing_and_wrong_versions_refuse(self):
        def lookup(name):
            if name == "jax":
                raise metadata.PackageNotFoundError(name)
            return "wrong"

        result = environment_report("core", version=lookup, python_version=(3, 12))
        self.assertFalse(result["passed"])
        self.assertEqual({row["status"] for row in result["packages"]}, {"missing", "mismatch"})

    def test_wrong_python_refuses(self):
        pins = declared_requirements()["groups"]["core"]
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


# The command-line surface, byte for byte (S5 A2): the help of every subcommand and the `info` output of an
# uninstalled checkout. argparse wraps to the terminal width (pinned: COLUMNS=100) and its layout differs
# between Python versions (the literals are Python 3.12's).
HELP = {
    "": """\
usage: glm-tpu [-h] {info,ask,collect-env,doctor,prepare-request,checkpoint} ...

Release information, environment and checkpoint checks, local request preparation and question
submission.

positional arguments:
  {info,ask,collect-env,doctor,prepare-request,checkpoint}
    info                show supported scope and release limitations
    ask                 answer questions on the configured TPU site; optionally batch up to four
    collect-env (doctor)
                        check installed version metadata without initializing TPU
    prepare-request     tokenize a private chat locally; does NOT launch inference
    checkpoint          inventory or mark a checkpoint source, or verify a sealed checkpoint, on
                        local files

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
    "collect-env": """\
usage: glm-tpu collect-env [-h] [--profile {core,runtime,tpu,dev}]

options:
  -h, --help            show this help message and exit
  --profile {core,runtime,tpu,dev}
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
    "checkpoint": """\
usage: glm-tpu checkpoint [-h] {inventory,mark-source,verify} ...

positional arguments:
  {inventory,mark-source,verify}
    inventory           write the inventory of a local safetensors source (no payload is read)
    mark-source         hash a local GLM-5.3 source, compare it with upstream and write
                        SOURCE_COMPLETE.json
    verify              verify a local sealed runtime checkpoint against the site file's pins

options:
  -h, --help            show this help message and exit
""",
    "checkpoint inventory": """\
usage: glm-tpu checkpoint inventory [-h] --output OUTPUT --model-id MODEL_ID --revision REVISION
                                    source

positional arguments:
  source               directory of model.safetensors.index.json, its shards and config.json

options:
  -h, --help           show this help message and exit
  --output OUTPUT      the new inventory file (never overwritten)
  --model-id MODEL_ID  the model id to record, e.g. zai-org/GLM-5.3
  --revision REVISION  the source revision to record
""",
    "checkpoint mark-source": """\
usage: glm-tpu checkpoint mark-source [-h] [--output OUTPUT] [--upstream-marker UPSTREAM_MARKER]
                                      [--workers WORKERS]
                                      source

positional arguments:
  source                directory of the downloaded GLM-5.3 source (shards and metadata)

options:
  -h, --help            show this help message and exit
  --output OUTPUT       the new marker file (default: SOURCE/SOURCE_COMPLETE.json; never
                        overwritten)
  --upstream-marker UPSTREAM_MARKER
                        compare with the digests of this earlier marker instead of the Hugging
                        Face repository
  --workers WORKERS     files hashed in parallel, 1..64 (default: 8)
""",
    "checkpoint verify": """\
usage: glm-tpu checkpoint verify [-h] [--site SITE] [--root ROOT] [--slots SLOT [SLOT ...]]
                                 [--local-slot-layout]

options:
  -h, --help            show this help message and exit
  --site SITE           site file (default: $GLM_TPU_SITE_CONFIG, else
                        $GLM_TPU_CONFIG_ROOT/site.toml)
  --root ROOT           checkpoint root (default: the site's checkpoint.root)
  --slots SLOT [SLOT ...]
                        hash only these device slots, 0..31 (default: all)
  --local-slot-layout   the root holds only the --slots files (a worker's layout); any other slot
                        file is refused
""",
}
HELP["doctor"] = HELP["collect-env"]  # the alias's parser is collect-env's

INFO = """\
{
  "concurrent_context_capacity": 32768,
  "concurrent_requests": 4,
  "concurrent_scope": "fixed submitted group; no online request admission",
  "concurrent_validation": "GLM-5.3: four correct completed GSM8K answers; normal EOS, eight-host agreement and cleanup; short inputs only",
  "engine": "native JAX",
  "hardware": "8 hosts / 32 TPU v4 chips",
  "installation": "the wheel holds the Python package with its UI assets and model configuration files, no weights; deployment requires the source checkout, a site file and external assets",
  "model": "zai-org/GLM-5.3",
  "ordinary_profile": "greedy; 8K combined, 32K combined (the ask default; per conversation when concurrent), 128K prompt / 166912 combined slots, or 256K combined (offered by ask --context 256k; refused by HBM admission on 32 TPU v4 chips)",
  "project": "glm-tpu",
  "quality": "full benchmark/model-card parity not established",
  "queued_questions": 10,
  "release_status": "private project",
  "resume": "resident ordinary inbox reuses the live model; no process-restart or durable KV recovery",
  "serving": "site-specific protected request harness; loopback chat UI and OpenAI-compatible /v1 API attached to a resident controller (python -m glm_tpu.entrypoints.serve.server)",
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
        "choices=['info', 'ask', 'collect-env', 'doctor', 'prepare-request', 'checkpoint'] required",
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
    "collect-env": [
        "_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='",
        "_StoreAction --profile dest=profile default='core' choices=['core', 'runtime', 'tpu', 'dev']",
    ],
    "doctor": [  # the alias: collect-env's parser
        "_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='",
        "_StoreAction --profile dest=profile default='core' choices=['core', 'runtime', 'tpu', 'dev']",
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
    "checkpoint": [
        "_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='",
        "_SubParsersAction action dest=action nargs='A...' choices=['inventory', 'mark-source', 'verify'] required",
    ],
    "checkpoint inventory": [
        "_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='",
        "_StoreAction source dest=source type='Path' required",
        "_StoreAction --output dest=output type='Path' required",
        "_StoreAction --model-id dest=model_id required",
        "_StoreAction --revision dest=revision required",
    ],
    "checkpoint mark-source": [
        "_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='",
        "_StoreAction source dest=source type='Path' required",
        "_StoreAction --output dest=output type='Path'",
        "_StoreAction --upstream-marker dest=upstream_marker type='Path'",
        "_StoreAction --workers dest=workers default=8 type='int'",
    ],
    "checkpoint verify": [
        "_HelpAction -h/--help dest=help nargs=0 default='==SUPPRESS=='",
        "_StoreAction --site dest=site type='Path'",
        "_StoreAction --root dest=root type='Path'",
        "_StoreAction --slots dest=slots nargs='+' type='int' metavar='SLOT'",
        "_StoreTrueAction --local-slot-layout dest=local_slot_layout nargs=0 const=True default=False",
    ],
}


ALIASES = {"doctor": "collect-env"}  # alias -> the subcommand whose parser answers to it


def test_every_argument_is_unchanged(monkeypatch):
    parser = built_parser(monkeypatch)
    (commands,) = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    parsers = {"": parser, **commands.choices}
    for name, subparser in commands.choices.items():  # a subcommand's own subcommands: "checkpoint inventory"
        for action in subparser._actions:
            if isinstance(action, argparse._SubParsersAction):
                parsers.update({f"{name} {inner}": p for inner, p in action.choices.items()})
    assert {name: arguments(p) for name, p in parsers.items()} == ARGUMENTS
    assert commands.choices["doctor"] is commands.choices["collect-env"]
    assert [(p.prog, p.description) for p in parsers.values()] == [
        (
            "glm-tpu",
            "Release information, environment and checkpoint checks, local request preparation and question"
            " submission.",
        ),
        *((f"glm-tpu {ALIASES.get(name, name)}", None) for name in list(ARGUMENTS)[1:]),
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


@pytest.mark.parametrize("command", ["collect-env", "doctor"])
@pytest.mark.parametrize("profile", ["core", "runtime", "tpu", "dev", "missing jax"])
def test_doctor_prints_the_report_and_exits_by_its_verdict(monkeypatch, capsys, profile, command):
    """`collect-env` and its alias `doctor` print the profile's report and exit 0 exactly when it passed (every pin
    installed, Python 3.12)."""
    pins = {name: value for group in declared_requirements()["groups"].values() for name, value in group.items()}

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
    assert main([command, "--profile", name]) == (0 if report["passed"] else 1)
    assert capsys.readouterr() == (json.dumps(report, indent=2, sort_keys=True) + "\n", "")


@pytest.mark.parametrize(
    "argv",
    [
        ["info"],
        ["collect-env"],
        ["doctor"],
        ["--help"],
        ["collect-env", "--help"],
        ["checkpoint", "--help"],
        ["checkpoint", "inventory", "--help"],
        ["checkpoint", "mark-source", "--help"],
        ["checkpoint", "verify", "--help"],
    ],
    ids=" ".join,
)
def test_info_collect_env_and_help_leave_the_command_modules_unloaded(argv):
    """The subcommands that need neither load neither the ask module nor the request module (imported by `ask` and
    `prepare-request` when they run) nor the checkpoint library (imported by `checkpoint` when it runs), nor a model
    library."""
    code = f"""
import contextlib, io, sys
from glm_tpu.entrypoints.cli.main import main
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        main({argv!r})
    except SystemExit:
        pass
unloaded = ("glm_tpu.entrypoints.cli.ask", "glm_tpu.engine.request", "glm_tpu.model_loader", "jax", "jaxlib", "libtpu",
            "torch", "transformers")
loaded = [name for name in unloaded if name in sys.modules]
assert not loaded, loaded
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


if __name__ == "__main__":
    unittest.main()
