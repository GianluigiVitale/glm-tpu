#!/usr/bin/env python3
"""Audit sealed numerical watchpoints without importing JAX or running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.errors import BenchmarkValidationError  # noqa: E402
from glm_tpu.greenfield.observability import (  # noqa: E402
    audit_observability_contract,
    inspect_npz_artifact,
    write_observability_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed offline inspection of immutable NPZ evidence."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--contract", type=Path)
    mode.add_argument("--inspect-npz", type=Path)
    parser.add_argument("--artifact-sha256")
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.output.is_symlink() or args.output.exists():
        parser.error(f"output path is occupied: {args.output}")
    if args.inspect_npz is not None:
        if args.artifact_sha256 is None:
            parser.error("--inspect-npz requires --artifact-sha256")
        if args.contract_sha256 is not None:
            parser.error("--contract-sha256 is valid only with --contract")
        report = inspect_npz_artifact(args.inspect_npz, args.artifact_sha256)
        mode = "inspect_npz"
    else:
        if args.artifact_sha256 is not None:
            parser.error("--artifact-sha256 is valid only with --inspect-npz")
        if args.contract_sha256 is None:
            parser.error("--contract requires --contract-sha256")
        report = audit_observability_contract(
            args.contract, args.contract_sha256
        )
        mode = "audit_contract"
    write_observability_report(args.output, report)
    print(
        json.dumps(
            {
                "classification": report.get("classification"),
                "mode": mode,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except BenchmarkValidationError as error:
        raise SystemExit(f"observability audit refused: {error}") from error
