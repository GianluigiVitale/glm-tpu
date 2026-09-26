"""``glm-tpu prepare-request``: tokenize a private chat locally, without launching inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from glm_tpu.entrypoints.cli.types import CLISubcommand


class PrepareRequestSubcommand(CLISubcommand):
    """Write a prepared request file from a private chat; a refusal names only the error type."""

    name = "prepare-request"

    @staticmethod
    def cmd(args: argparse.Namespace) -> int:
        try:
            from glm_tpu.engine.request import prepare_file

            extra = {} if args.profile == "ordinary-greedy-8k" else dict(context_capacity=166912)
            report = prepare_file(
                messages_path=args.messages,
                output=args.output,
                repo=args.repo,
                tokenizer_root=args.tokenizer_root,
                request_id=args.request_id,
                max_new_tokens=args.max_new_tokens,
                **extra,
            )
        except (ValueError, OSError, ImportError) as exc:
            # Do not print private input, tokenizer exception text or file contents.
            print(
                json.dumps(dict(error=type(exc).__name__, status="request preparation refused")),
                file=sys.stderr,
            )
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    def subparser_init(self, subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
        prepare = subparsers.add_parser(
            "prepare-request",
            help="tokenize a private chat locally; does NOT launch inference",
        )
        prepare.add_argument("--messages", type=Path, required=True)
        prepare.add_argument("--output", type=Path, required=True)
        prepare.add_argument("--repo", type=Path, required=True)
        prepare.add_argument("--tokenizer-root", type=Path, required=True)
        prepare.add_argument("--request-id", required=True)
        prepare.add_argument(
            "--profile", choices=("ordinary-greedy-8k", "ordinary-greedy-128k"), default="ordinary-greedy-8k"
        )
        prepare.add_argument("--max-new-tokens", type=int, required=True)
        return prepare
