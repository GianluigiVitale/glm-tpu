"""The base class of the ``glm-tpu`` subcommands (the vLLM CLI layout)."""

from __future__ import annotations

import argparse


class CLISubcommand:
    """One ``glm-tpu`` subcommand: its ``name``, its parser (``subparser_init``) and what it runs (``cmd``)."""

    name: str

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        """Run the subcommand on its parsed ``args``; return the process exit code."""
        raise NotImplementedError

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        """Add the subcommand's parser to ``subparsers`` and return it."""
        raise NotImplementedError
