#!/usr/bin/env python3
"""Compare the sealed accepted and bounded layer-0 scheduled TPU graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glm_tpu.greenfield.benchmarking.live_ssa_diff import (  # noqa: E402
    ACCEPTED_LAYER0_HLO_SHA256,
    CANDIDATE_PROFILE_ACCEPTED_SOURCE_CONTEXT,
    CANDIDATE_PROFILE_NATIVE_SOURCE_CONTEXT,
    CANDIDATE_LAYER0_HLO_SHA256,
    compare_layer0_live_ssa_files,
    write_layer0_live_ssa_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-hlo", type=Path, required=True)
    parser.add_argument("--candidate-hlo", type=Path, required=True)
    parser.add_argument(
        "--accepted-sha256", default=ACCEPTED_LAYER0_HLO_SHA256
    )
    parser.add_argument(
        "--candidate-sha256", default=CANDIDATE_LAYER0_HLO_SHA256
    )
    parser.add_argument(
        "--candidate-profile",
        choices=(
            CANDIDATE_PROFILE_ACCEPTED_SOURCE_CONTEXT,
            CANDIDATE_PROFILE_NATIVE_SOURCE_CONTEXT,
        ),
        default=CANDIDATE_PROFILE_ACCEPTED_SOURCE_CONTEXT,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = compare_layer0_live_ssa_files(
        args.accepted_hlo,
        args.candidate_hlo,
        accepted_sha256=args.accepted_sha256,
        candidate_sha256=args.candidate_sha256,
        candidate_profile=args.candidate_profile,
    )
    write_layer0_live_ssa_report(args.output, report)
    print(
        json.dumps(
            {
                "difference_count": len(report["differences"]),
                "first_divergence": report["first_divergence"],
                "output": str(args.output),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
