#!/usr/bin/env python3
"""Compare one sealed accepted DSA producer with its greenfield event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from glm_tpu.greenfield.validation import (  # noqa: E402
    AcceptedGreenfieldDsaInternalComparisonConfig,
    compare_accepted_greenfield_dsa_internal_observation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accepted-capture-dir", type=Path, required=True)
    parser.add_argument(
        "--greenfield-observation-npz", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-manifest-sha256", required=True)
    parser.add_argument("--greenfield-observation-sha256", required=True)
    parser.add_argument("--greenfield-code-hash", required=True)
    parser.add_argument("--legacy-code-hash", required=True)
    parser.add_argument("--layer-id", type=int, required=True)
    parser.add_argument("--position", type=int, default=8155)
    args = parser.parse_args()
    result = compare_accepted_greenfield_dsa_internal_observation(
        AcceptedGreenfieldDsaInternalComparisonConfig(
            accepted_capture_dir=args.accepted_capture_dir,
            greenfield_observation_path=args.greenfield_observation_npz,
            output_dir=args.output,
            expected_capture_manifest_sha256=(
                args.capture_manifest_sha256
            ),
            expected_greenfield_observation_sha256=(
                args.greenfield_observation_sha256
            ),
            expected_greenfield_code_hash=args.greenfield_code_hash,
            expected_legacy_code_hash=args.legacy_code_hash,
            expected_layer_id=args.layer_id,
            expected_position=args.position,
        )
    )
    print(
        json.dumps(
            {
                "all_fields_elementwise_exact": result[
                    "all_fields_elementwise_exact"
                ],
                "divergent_fields": result["divergent_fields"],
                "event_index": result["event_index"],
                "first_divergent_field": result["first_divergent_field"],
                "layer_id": result["layer_id"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
