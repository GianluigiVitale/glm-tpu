#!/usr/bin/env python3
"""Protected runner adapter for the distinct WS32 raw-FP8 Pallas body."""

from __future__ import annotations

import os
import sys
import run_real_one_layer_ws32 as reference_runner

from glm_tpu.greenfield.benchmarking.ws32_pallas_one_layer import (
    build_ws32_pallas_one_layer_mapped, validate_ws32_pallas_one_layer_hlo,
)


def main() -> int:
    """Reuse the sealed loader/oracle runner with only the mapped body swapped."""

    acquiring = "--compile-only" in sys.argv and sys.argv[sys.argv.index("--compile-only") + 1] == "1"
    os.environ["GLM_GREENFIELD_WS32_ARTIFACT_KIND"] = "greenfield_ws32_pallas_real_layer3_hlo_acquisition" if acquiring else "greenfield_ws32_pallas_real_layer3"
    reference_runner.build_ws32_one_layer_mapped = build_ws32_pallas_one_layer_mapped
    reference_runner.validate_ws32_one_layer_hlo = validate_ws32_pallas_one_layer_hlo
    # Preserve the acquired return source coordinate below.

    return reference_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
