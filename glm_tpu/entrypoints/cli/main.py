"""Release information, environment and checkpoint checks, local request preparation and question submission."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import metadata
import json
from pathlib import Path

from glm_tpu.entrypoints.cli.checkpoint import CheckpointSubcommand
from glm_tpu.entrypoints.cli.collect_env import CollectEnvSubcommand
from glm_tpu.entrypoints.cli.prepare import PrepareRequestSubcommand
from glm_tpu.entrypoints.cli.types import CLISubcommand


class InfoSubcommand(CLISubcommand):
    """``glm-tpu info``: the supported scope and the release limitations, as JSON."""

    name = "info"

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        try:
            version = metadata.version("glm-tpu")
        except metadata.PackageNotFoundError:
            version = "source checkout (not installed)"
        print(
            json.dumps(
                dict(
                    project="glm-tpu",
                    version=version,
                    release_status="private project",
                    engine="native JAX",
                    hardware="8 hosts / 32 TPU v4 chips",
                    ordinary_profile=(
                        "greedy; 8K combined, 32K combined (the ask default; per conversation when concurrent), "
                        "128K prompt / 166912 combined slots, or 256K combined (offered by ask --context 256k; refused "
                        "by HBM admission on 32 TPU v4 chips)"
                    ),
                    concurrent_requests=4,
                    model="zai-org/GLM-5.3",
                    concurrent_validation=(
                        "GLM-5.3: four correct completed GSM8K answers; normal EOS, eight-host agreement and cleanup; "
                        "short inputs only"
                    ),
                    concurrent_context_capacity=32768,
                    concurrent_scope="fixed submitted group; no online request admission",
                    queued_questions=10,
                    resume="resident ordinary inbox reuses the live model; no process-restart or durable KV recovery",
                    serving=(
                        "site-specific protected request harness; loopback chat UI and OpenAI-compatible /v1 API "
                        "attached to a resident controller (python -m glm_tpu.entrypoints.serve.server)"
                    ),
                    installation=(
                        "the wheel holds the Python package with its UI assets and model configuration files, no "
                        "weights; deployment requires the source checkout, a site file and external assets"
                    ),
                    quality="full benchmark/model-card parity not established",
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        return subparsers.add_parser("info", help="show supported scope and release limitations")


class AskSubcommand(CLISubcommand):
    """``glm-tpu ask``: answer questions on the configured TPU site (``glm_tpu.entrypoints.cli.ask``).

    Its parser is defined here and ``cmd`` imports ``ask`` only when it runs: ``ask`` imports the request module at
    load time, and registering the parser must not load it for the other subcommands.
    """

    name = "ask"

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        from glm_tpu.entrypoints.cli.ask import main as ask_main

        return ask_main(args)

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        ask = subparsers.add_parser(
            "ask", help="answer questions on the configured TPU site; optionally batch up to four"
        )
        inputs = ask.add_mutually_exclusive_group(required=True)
        inputs.add_argument("question", nargs="?")
        inputs.add_argument("--questions", type=Path, help="private JSON array of one to ten question strings")
        ask.add_argument("--context", choices=("8k", "32k", "128k", "256k"), default="32k")
        ask.add_argument(
            "--keep-loaded",
            action="store_true",
            help="retain the ordinary model and fleet leases after answering; explicit stop required",
        )
        ask.add_argument(
            "--concurrent", action="store_true", help="batch up to four conversations; requires --context 32k"
        )
        ask.add_argument(
            "--max-new-tokens",
            type=int,
            help=(
                "optional output cap; default: all remaining context slots (up to 163840 in 128k mode); thinking "
                "and answer share this space"
            ),
        )
        ask.add_argument("--wall-seconds", type=int, default=86400)
        ask.add_argument(
            "--prepare-only", action="store_true", help="prepare private inputs without launching the model"
        )
        ask.add_argument(
            "--site", type=Path, help="site file (default: $GLM_TPU_SITE_CONFIG, else $GLM_TPU_CONFIG_ROOT/site.toml)"
        )
        return ask


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    # The help lists the subcommands in this order. argparse stores the name the command was called by (an alias
    # such as ``doctor`` included), so each subcommand is keyed under every name its parser answers to.
    commands: dict[str, CLISubcommand] = {}
    for command in (
        InfoSubcommand(),
        AskSubcommand(),
        CollectEnvSubcommand(),
        PrepareRequestSubcommand(),
        CheckpointSubcommand(),
    ):
        subparser = command.subparser_init(sub)
        commands.update({name: command for name, choice in sub.choices.items() if choice is subparser})
    args = parser.parse_args(argv)
    return commands[args.command].cmd(args)
