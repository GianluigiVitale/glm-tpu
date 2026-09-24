"""Release information, local preparation and protected question submission."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import sys
from typing import Sequence
from glm_tpu.entrypoints.cli.collect_env import environment_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show supported scope and release limitations")
    ask = sub.add_parser("ask", help="answer questions on the retained TPU site; optionally batch up to four")
    inputs = ask.add_mutually_exclusive_group(required=True)
    inputs.add_argument("question", nargs="?")
    inputs.add_argument("--questions", type=Path, help="private JSON array of one to ten question strings")
    ask.add_argument("--context", choices=("8k", "32k", "128k", "256k"), default="32k")
    ask.add_argument("--keep-loaded", action="store_true",
                     help="retain the ordinary model and fleet leases after answering; explicit stop required")
    ask.add_argument("--concurrent", action="store_true",
                     help="batch up to four conversations; requires --context 32k")
    ask.add_argument("--max-new-tokens", type=int,
                     help="optional output cap; default: all remaining context slots (up to 163840 in 128k mode); thinking and answer share this space")
    ask.add_argument("--wall-seconds", type=int, default=86400)
    ask.add_argument("--prepare-only", action="store_true", help="prepare private inputs without launching the model")
    ask.add_argument("--site", type=Path,
                     help="site file (default: $GLM_TPU_SITE_CONFIG, else $GLM_TPU_CONFIG_ROOT/site.toml)")
    doctor = sub.add_parser(
        "doctor", help="check installed version metadata without initializing TPU"
    )
    doctor.add_argument(
        "--profile",
        choices=("core", "runtime", "tpu", "benchmark", "dev"),
        default="core",
    )
    prepare = sub.add_parser(
        "prepare-request",
        help="tokenize a private chat locally; does NOT launch inference",
    )
    prepare.add_argument("--messages", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--repo", type=Path, required=True)
    prepare.add_argument("--tokenizer-root", type=Path, required=True)
    prepare.add_argument("--request-id", required=True)
    prepare.add_argument("--profile", choices=("ordinary-greedy-8k", "ordinary-greedy-128k"),
                         default="ordinary-greedy-8k")
    prepare.add_argument("--max-new-tokens", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "ask":
        from glm_tpu.entrypoints.cli.ask import main as ask_main
        return ask_main(args)
    if args.command == "prepare-request":
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
                json.dumps(
                    dict(error=type(exc).__name__, status="request preparation refused")
                ),
                file=sys.stderr,
            )
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if args.command == "doctor":
        report = environment_report(args.profile)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["passed"] else 1
    try:
        version = metadata.version("glm-tpu")
    except metadata.PackageNotFoundError:
        version = "source checkout (not installed)"
    print(
        json.dumps(
            dict(
                project="glm-tpu",
                version=version,
                release_status="private project; see docs/release/STATUS.md for trained admission and promotion",
                engine="native JAX WS32_2D",
                hardware="8 hosts / 32 TPU v4 chips",
                ordinary_profile="greedy; 8K combined, concurrent 32K per conversation, or 128K prompt / 166912 combined slots; see STATUS for measured scope",
                concurrent_requests=4,
                model="zai-org/GLM-5.3",
                concurrent_validation="GLM-5.3: four correct completed GSM8K answers; normal EOS, eight-host agreement and cleanup; short inputs only",
                concurrent_context_capacity=32768,
                concurrent_scope="fixed submitted group; see STATUS for hardware evidence; no online request admission",
                queued_questions=10,
                resume="resident ordinary inbox reuses the live model; no process-restart or durable KV recovery",
                serving="site-specific protected request harness; no supported HTTP endpoint",
                installation="wheel contains Python components only; full deployment requires source checkout and external assets",
                quality="full benchmark/model-card parity not established",
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0
