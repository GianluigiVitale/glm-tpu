"""Local fixed-profile launch recipe/preflight; never launches or accesses TPU/GCS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from glm_tpu.greenfield.validation.ws32_prefill_admission import (
    SHORT_PROFILE,
    SHORT_PLAN,
    SHORT_BUDGET_SECONDS,
    SHORT_RESERVE_BYTES,
    short_acquisition,
    require_short_numerical_request,
)

REPO = Path(__file__).resolve().parents[2]


def graph_environment_name(graph: str, form: str) -> str:
    suffix = {
        "stablehlo_sha256": "STABLEHLO_SHA",
        "optimized_hlo_sha256": "OPTIMIZED_HLO_SHA",
    }[form]
    return f"GLM_GREENFIELD_WS32_{graph.upper()}_{suffix}"


def numerical_environment(
    repo: Path = REPO, *, profile: str = SHORT_PROFILE
) -> dict[str, str]:
    """Reuse retained weights/overlay recipe; replace only mode/profile/graph pins."""
    value = json.loads(
        (repo / "configs/greenfield-ws32-batched-acquisition.json").read_text()
    )
    env = dict(value["environment"])
    env.update(
        GLM_GREENFIELD_WS32_SHORT_DECODER_MODE="numerical",
        GLM_GREENFIELD_WS32_BATCHED_PREFILL_PROFILE=profile,
    )
    for graph, pins in short_acquisition(repo, profile=profile)["graphs"].items():
        for form, digest in pins.items():
            env[graph_environment_name(graph, form)] = digest
    validate_environment(env, repo=repo)
    return env


def validate_environment(env: Mapping[str, str], *, repo: Path = REPO) -> None:
    """Refuse wrong pins/source/profile before locks/network/load/compilation."""
    prefix = "GLM_GREENFIELD_WS32_"
    if (
        env.get(prefix + "SHORT_DECODER_MODE") != "numerical"
        or env.get(prefix + "SHORT_DECODER_CONTEXT") != "2k"
    ):
        raise ValueError("batched launch is fixed2K numerical only")
    if (
        env.get(prefix + "DSA_ADJUDICATION", "0") != "0"
        or env.get(prefix + "LATER_EVENT_ALARM_ACK", "0") != "0"
    ):
        raise ValueError("first batched launch cannot inherit an adjudication/alarm")
    args = SimpleNamespace(
        prefill_mode=env.get(prefix + "PREFILL_MODE"),
        exact_dsa=int(env.get(prefix + "EXACT_DSA", "0")),
        strategy_nd_dense=int(env.get(prefix + "STRATEGY_ND_DENSE", "0")),
        host_main_rope_table=int(env.get(prefix + "HOST_MAIN_ROPE_TABLE", "0")),
        rotary_diagnostic=int(env.get(prefix + "ROTARY_DIAGNOSTIC", "0")),
        prefill_chunk=int(env.get(prefix + "PREFILL_CHUNK", "17")),
        context_capacity=int(env.get(prefix + "CONTEXT_CAPACITY", "8192")),
        batched_prefill_profile=env.get(prefix + "BATCHED_PREFILL_PROFILE", ""),
        prefill_budget_seconds=SHORT_BUDGET_SECONDS,
        prefill_memory_reserve_bytes=SHORT_RESERVE_BYTES,
        long_context=None,
        dsa_adjudication_record=None,
        dsa_adjudication_sha256="0" * 64,
        **{
            f"expected_{g}_{form}": env.get(graph_environment_name(g, form), "")
            for g, pins in short_acquisition(
                repo, profile=env.get(prefix + "BATCHED_PREFILL_PROFILE", "")
            )["graphs"].items()
            for form in pins
        },
    )
    require_short_numerical_request(
        args, prompt_length=SHORT_PLAN.prompt_length, repo=repo
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-environment", action="store_true")
    mode.add_argument("--print-environment", action="store_true")
    parser.add_argument("--profile", default=SHORT_PROFILE)
    args = parser.parse_args()
    if args.validate_environment:
        validate_environment(os.environ)
    else:
        print(json.dumps(numerical_environment(profile=args.profile), sort_keys=True))


if __name__ == "__main__":
    main()
