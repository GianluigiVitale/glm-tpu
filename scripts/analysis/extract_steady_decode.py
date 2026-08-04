#!/usr/bin/env python3
"""Extract profiler-free steady decode cadence from an E0 driver log.

The throughput benchmark's pass subtraction includes the deliberately slow
``GLM_JAX_TRACE`` interval.  vLLM keeps logging generation cadence after the
trace has closed, so those later samples are the honest wall-rate cross-check
for the XPlane device step.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, median


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")
TRACE_RE = re.compile(r"\[GLM_JAX_TRACE\] traced (\d+) decode steps")
STATS_RE = re.compile(
    r"Avg generation throughput: ([0-9]+(?:\.[0-9]+)?) tokens/s, .*"
    r"Running: (\d+) reqs")


def extract(log_text: str, analysis: dict, expected_trace_steps: int) -> dict:
    lines = ANSI_RE.sub("", log_text).splitlines()
    trace_markers = [
        (i, int(match.group(1))) for i, line in enumerate(lines)
        if (match := TRACE_RE.search(line))
    ]
    if not trace_markers:
        raise ValueError("no completed GLM_JAX_TRACE marker")
    if any(steps != expected_trace_steps for _, steps in trace_markers):
        raise ValueError(f"unexpected trace markers: {trace_markers}")

    marker_line = trace_markers[-1][0]
    samples = []
    for line in lines[marker_line + 1:]:
        match = STATS_RE.search(line)
        if match and int(match.group(2)) == 1:
            samples.append(float(match.group(1)))
    if len(samples) < 2:
        raise ValueError(
            f"need >=2 single-request post-trace cadence samples, got {samples}")

    # vLLM's first stats window after the close marker can still contain
    # profiler time. Drop exactly that mixed window, then retain every later
    # single-request sample rather than selecting a favorable subset. E0 sets
    # a one-second interval so even a 50-100 tok/s candidate leaves samples.
    steady = samples[1:]
    wall_rate = median(steady)
    device_step_ms = float(analysis["device_step_ms"])
    device_rate = 1000.0 / device_step_ms
    ratio = wall_rate / device_rate
    if not math.isfinite(wall_rate) or wall_rate <= 0:
        raise ValueError(f"invalid wall rate: {wall_rate}")
    if not 0.70 <= ratio <= 1.10:
        raise ValueError(
            f"wall/device cadence mismatch: wall={wall_rate} "
            f"device={device_rate} ratio={ratio}")

    return {
        "method": "vllm GLM_LOG_STATS after final GLM_JAX_TRACE close",
        "expected_trace_steps": expected_trace_steps,
        "trace_marker_line_1based": marker_line + 1,
        "dropped_mixed_window_tok_s": samples[0],
        "steady_samples_tok_s": steady,
        "steady_median_tok_s": wall_rate,
        "steady_mean_tok_s": mean(steady),
        "steady_min_tok_s": min(steady),
        "steady_max_tok_s": max(steady),
        "device_step_ms": device_step_ms,
        "device_tok_s": device_rate,
        "wall_to_device_ratio": ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("driver_log", type=Path)
    parser.add_argument("analysis_json", type=Path)
    parser.add_argument("out_json", type=Path)
    parser.add_argument("--trace-steps", type=int, required=True)
    args = parser.parse_args()

    result = extract(args.driver_log.read_text(errors="replace"),
                     json.loads(args.analysis_json.read_text()),
                     args.trace_steps)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print("STEADY_DECODE_VALID", result["steady_median_tok_s"],
          result["device_tok_s"], len(result["steady_samples_tok_s"]))


if __name__ == "__main__":
    main()
