#!/usr/bin/env python3
"""Apply a serialized greenfield HLO policy to textual XLA HLO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.sharding.hlo_contract import (  # noqa: E402
    HloContractPolicy,
    lint_hlo,
    parse_hlo_module,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hlo", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = HloContractPolicy.from_dict(json.loads(args.policy.read_text()))
    report = lint_hlo(parse_hlo_module(args.hlo.read_text()), policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    print(
        f"HLO_CONTRACT_{'PASS' if report.valid else 'FAIL'} "
        f"policy={policy.name} violations={len(report.violations)} "
        f"output={args.output}"
    )
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
