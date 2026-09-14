"""Release information and local request preparation. Never opens TPU devices."""
from __future__ import annotations

import argparse
from importlib import metadata, resources
import json
from pathlib import Path
import sys
from typing import Callable, Sequence


def environment_manifest() -> dict:
    return json.loads(resources.files("glm_tpu").joinpath("environment.json").read_text())


def environment_report(
    profile: str, *, version: Callable[[str], str] = metadata.version,
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
    expected = {name: pin for group in groups[profile]
                for name, pin in manifest["profiles"][group].items()}
    rows = []
    for name, pin in sorted(expected.items()):
        try:
            actual = version(name)
        except metadata.PackageNotFoundError:
            actual = None
        rows.append(dict(package=name, expected=pin, installed=actual,
                         status="match" if actual == pin else "missing" if actual is None else "mismatch"))
    major, minor = python_version if python_version is not None else sys.version_info[:2]
    python = f"{major}.{minor}"
    return dict(
        schema="glm_tpu_environment_report_v1", profile=profile,
        python=python, expected_python=manifest["python"], packages=rows,
        passed=python == manifest["python"] and all(row["status"] == "match" for row in rows),
        scope="distribution metadata only; no dependency solving, payload hashes, model imports, TPU access or launch authorization",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show supported scope and release limitations")
    doctor = sub.add_parser("doctor", help="check installed version metadata without initializing TPU")
    doctor.add_argument("--profile", choices=("core", "runtime", "tpu", "benchmark", "dev"), default="core")
    prepare = sub.add_parser("prepare-request", help="tokenize a private chat locally; does NOT launch inference")
    prepare.add_argument("--messages", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--repo", type=Path, required=True)
    prepare.add_argument("--tokenizer-root", type=Path, required=True)
    prepare.add_argument("--request-id", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-new-tokens", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-request":
        from glm_tpu.user_request import prepare_file
        try:
            report = prepare_file(messages_path=args.messages, output=args.output, repo=args.repo,
                tokenizer_root=args.tokenizer_root, request_id=args.request_id, seed=args.seed,
                max_new_tokens=args.max_new_tokens)
        except (ValueError, OSError, ImportError) as exc:
            # Do not print private input, tokenizer exception text or file contents.
            print(json.dumps(dict(error=type(exc).__name__, status="request preparation refused")), file=sys.stderr)
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
    print(json.dumps(dict(
        project="glm-tpu", version=version, release_status="alpha; release checks incomplete",
        engine="native JAX WS32_2D", hardware="8 hosts / 32 TPU v4 chips",
        concurrent_requests=1, resume="same live session only",
        serving="site-specific protected request harness; no supported HTTP endpoint",
        installation="wheel contains Python components only; full deployment requires source checkout and external assets",
        quality="full benchmark/model-card parity not established",
    ), indent=2, sort_keys=True))
    return 0
