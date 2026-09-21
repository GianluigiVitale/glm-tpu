"""Release information, local preparation and protected question submission."""

from __future__ import annotations

import argparse
from importlib import metadata, resources
import json
from pathlib import Path
import sys
from typing import Callable, Sequence


def environment_manifest() -> dict:
    return json.loads(
        resources.files("glm_tpu").joinpath("environment.json").read_text()
    )


def environment_report(
    profile: str,
    *,
    version: Callable[[str], str] = metadata.version,
    python_version: tuple[int, int] | None = None,
) -> dict:
    """Compare distribution metadata only; a match is NOT hardware admission."""
    manifest = environment_manifest()
    groups = {
        "core": ("core",),
        "runtime": ("core", "runtime"),
        "tpu": ("core", "runtime", "tpu"),
        "benchmark": ("core", "runtime", "tpu", "benchmark"),
        "dev": ("core", "dev"),
    }
    if profile not in groups:
        raise ValueError("unknown environment profile")
    expected = {
        name: pin
        for group in groups[profile]
        for name, pin in manifest["profiles"][group].items()
    }
    rows = []
    for name, pin in sorted(expected.items()):
        try:
            actual = version(name)
        except metadata.PackageNotFoundError:
            actual = None
        rows.append(
            dict(
                package=name,
                expected=pin,
                installed=actual,
                status=(
                    "match"
                    if actual == pin
                    else "missing" if actual is None else "mismatch"
                ),
            )
        )
    major, minor = (
        python_version if python_version is not None else sys.version_info[:2]
    )
    python = f"{major}.{minor}"
    return dict(
        schema="glm_tpu_environment_report_v1",
        profile=profile,
        python=python,
        expected_python=manifest["python"],
        packages=rows,
        passed=python == manifest["python"]
        and all(row["status"] == "match" for row in rows),
        scope="distribution metadata only; no dependency solving, payload hashes, model imports, TPU access or launch authorization",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show supported scope and release limitations")
    ask = sub.add_parser("ask", help="answer questions on the retained TPU site; optionally batch up to eight")
    inputs = ask.add_mutually_exclusive_group(required=True)
    inputs.add_argument("question", nargs="?")
    inputs.add_argument("--questions", type=Path, help="private JSON array of one to ten question strings")
    ask.add_argument("--context", choices=("8k", "32k", "128k"), default="128k")
    ask.add_argument("--concurrent", action="store_true",
                     help="batch up to eight conversations; requires --context 32k")
    ask.add_argument("--max-new-tokens", type=int,
                     help="output cap; default: available space up to 163840 in 128k mode, 2048 in 8k/32k modes")
    ask.add_argument("--wall-seconds", type=int, default=86400)
    ask.add_argument("--prepare-only", action="store_true", help="prepare private inputs without launching the model")
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
    prepare.add_argument("--profile", choices=("ordinary-greedy-8k", "ordinary-greedy-128k", "legacy-sampled"),
                         default="ordinary-greedy-8k")
    prepare.add_argument("--seed", type=int)
    prepare.add_argument("--max-new-tokens", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "ask":
        from glm_tpu.optimized.ask import main as ask_main
        return ask_main(args)
    if args.command == "prepare-request":
        try:
            if args.profile in ("ordinary-greedy-8k", "ordinary-greedy-128k"):
                from glm_tpu.optimized.request import prepare_file
                if args.seed is not None:
                    raise ValueError("greedy profile does not accept a sampling seed")
                extra = {} if args.profile == "ordinary-greedy-8k" else dict(context_capacity=166912)
            else:
                from glm_tpu.user_request import prepare_file
                extra = dict(seed=42 if args.seed is None else args.seed)
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
                concurrent_requests=1,
                experimental_batch_limit=8,
                experimental_batch_status="8x32K failed TPU compilation memory admission; not validated for serving",
                concurrent_context_capacity=32768,
                concurrent_scope="fixed submitted group; see STATUS for hardware evidence; no online request admission",
                queued_questions=10,
                resume="ordinary controller: none; legacy sampled runtime: same live session only",
                serving="site-specific protected request harness; no supported HTTP endpoint",
                installation="wheel contains Python components only; full deployment requires source checkout and external assets",
                quality="full benchmark/model-card parity not established",
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0
