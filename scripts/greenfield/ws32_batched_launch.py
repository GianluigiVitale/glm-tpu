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
    short_context,
    short_budget,
    SHORT_RESERVE_BYTES,
    short_acquisition,
    short_plan,
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
    repo: Path = REPO, *, profile: str = SHORT_PROFILE, context_label: str | None = None
) -> dict[str, str]:
    """Reuse retained weights/overlay recipe; replace only mode/profile/graph pins."""
    value = json.loads(
        (repo / "configs/greenfield-ws32-batched-acquisition.json").read_text()
    )
    env = dict(value["environment"])
    from scripts.greenfield import ws32_delivery_runtime as delivery
    if profile == delivery.PROFILE:
        from scripts.greenfield import ws32_delivery_companions as companions
        plan = delivery.programs.long_plan(context_label)
        env.update(
            GLM_GREENFIELD_WS32_SHORT_DECODER_MODE="numerical",
            GLM_GREENFIELD_WS32_BATCHED_PREFILL_PROFILE=profile,
            GLM_GREENFIELD_WS32_PREFILL_CHUNK="128",
            GLM_GREENFIELD_WS32_SHORT_DECODER_CONTEXT=context_label,
            GLM_GREENFIELD_WS32_CONTEXT_CAPACITY=str(plan.context_capacity),
            GLM_GREENFIELD_WS32_DSA_ADJUDICATION="0",
        )
        raw = {g: v[1] for g, v in delivery.programs.raw_registration(context_label).items()}
        raw.update(companions.raw_registration(context_label))
        for graph, digest in raw.items():
            env[graph_environment_name(graph, "stablehlo_sha256")] = digest
            env[graph_environment_name(graph, "optimized_hlo_sha256")] = "0" * 64
        validate_environment(env, repo=repo)
        return env
    if context_label is not None:
        raise ValueError("explicit delivery context requires the long profile")
    env.update(
        GLM_GREENFIELD_WS32_SHORT_DECODER_MODE="numerical",
        GLM_GREENFIELD_WS32_BATCHED_PREFILL_PROFILE=profile,
        GLM_GREENFIELD_WS32_PREFILL_CHUNK=str(short_plan(profile).block_rows),
        GLM_GREENFIELD_WS32_SHORT_DECODER_CONTEXT=short_context(profile),
    )
    from glm_tpu.greenfield.validation.ws32_canonical_8k_admission import PROFILE

    if profile == PROFILE:
        env["GLM_GREENFIELD_WS32_DSA_ADJUDICATION"] = "1"
    for graph, pins in short_acquisition(repo, profile=profile)["graphs"].items():
        for form, digest in pins.items():
            env[graph_environment_name(graph, form)] = digest
    validate_environment(env, repo=repo)
    return env


def validate_environment(env: Mapping[str, str], *, repo: Path = REPO) -> None:
    """Refuse wrong pins/source/profile before locks/network/load/compilation."""
    prefix = "GLM_GREENFIELD_WS32_"
    from scripts.greenfield import ws32_delivery_runtime as delivery
    if env.get(prefix + "BATCHED_PREFILL_PROFILE") == delivery.PROFILE:
        from glm_tpu.greenfield.validation.long_context_oracle import WS32_LONG_CONTEXT_PROFILES
        from scripts.greenfield import ws32_delivery_companions as companions
        label = env.get(prefix + "SHORT_DECODER_CONTEXT")
        plan = delivery.programs.long_plan(label)
        entry = WS32_LONG_CONTEXT_PROFILES[label]
        if (env.get(prefix + "SHORT_DECODER_MODE") != "numerical"
                or env.get(prefix + "DSA_ADJUDICATION", "0") != "0"
                or env.get(prefix + "LATER_EVENT_ALARM_ACK", "0") != "0"):
            raise ValueError("long delivery requires numerical mode without inherited adjudication")
        raw = {g: v[1] for g, v in delivery.programs.raw_registration(label).items()}
        raw.update(companions.raw_registration(label))
        args = SimpleNamespace(
            prefill_mode=env.get(prefix + "PREFILL_MODE"),
            batched_prefill_profile=delivery.PROFILE, delivery_context_label=label,
            **{key: int(env.get(prefix + key.upper(), "0")) for key in (
                "prefill_chunk", "context_capacity", "exact_dsa", "strategy_nd_dense",
                "host_main_rope_table", "rotary_diagnostic")},
            observer_steps=14 if label == "256k_e0" else 20, warmup=2,
            iterations=256 if label == "256k_e0" else 10, trace_steps=2,
            prefill_memory_reserve_bytes=delivery.RESERVE,
            prefill_budget_seconds=delivery.budget_seconds(label),
            long_context=entry["kind"], long_context_manifest_sha256=entry["manifest_sha256"],
            long_context_success_sha256=entry["success_sha256"],
            dsa_adjudication_record=None, dsa_adjudication_sha256="0" * 64,
            **{f"expected_{g}_{form}": env.get(graph_environment_name(g, form), "")
               for g in raw for form in ("stablehlo_sha256", "optimized_hlo_sha256")},
        )
        delivery.require_request(args, context_label=label, prompt_length=plan.prompt_length, repo=repo)
        return
    if env.get(prefix + "SHORT_DECODER_MODE") != "numerical" or env.get(
        prefix + "SHORT_DECODER_CONTEXT"
    ) != short_context(env.get(prefix + "BATCHED_PREFILL_PROFILE", "")):
        raise ValueError("batched launch requires its registered short context")
    from glm_tpu.greenfield.validation import ws32_canonical_8k_admission as own8k

    canonical8k = env.get(prefix + "BATCHED_PREFILL_PROFILE") == own8k.PROFILE
    if env.get(prefix + "DSA_ADJUDICATION", "0") != ("1" if canonical8k else "0"):
        raise ValueError("first batched launch cannot inherit an adjudication/alarm")
    ack = env.get(prefix + "LATER_EVENT_ALARM_ACK", "0")
    if ack != "0" and not (
        canonical8k
        and ack == "1"
        and env.get(prefix + "SHORT_DECODER_RECOVER", "0") == "1"
    ):
        raise ValueError("batched alarm acknowledgement requires canonical8K recovery")
    args = SimpleNamespace(
        prefill_mode=env.get(prefix + "PREFILL_MODE"),
        exact_dsa=int(env.get(prefix + "EXACT_DSA", "0")),
        strategy_nd_dense=int(env.get(prefix + "STRATEGY_ND_DENSE", "0")),
        host_main_rope_table=int(env.get(prefix + "HOST_MAIN_ROPE_TABLE", "0")),
        rotary_diagnostic=int(env.get(prefix + "ROTARY_DIAGNOSTIC", "0")),
        prefill_chunk=int(env.get(prefix + "PREFILL_CHUNK", "17")),
        context_capacity=int(env.get(prefix + "CONTEXT_CAPACITY", "8192")),
        batched_prefill_profile=env.get(prefix + "BATCHED_PREFILL_PROFILE", ""),
        prefill_budget_seconds=short_budget(
            env.get(prefix + "BATCHED_PREFILL_PROFILE", "")
        ),
        prefill_memory_reserve_bytes=SHORT_RESERVE_BYTES,
        long_context=None,
        dsa_adjudication_record=repo / own8k.RECORD if canonical8k else None,
        dsa_adjudication_sha256=own8k.RECORD_SHA256 if canonical8k else "0" * 64,
        **{
            f"expected_{g}_{form}": env.get(graph_environment_name(g, form), "")
            for g, pins in short_acquisition(
                repo, profile=env.get(prefix + "BATCHED_PREFILL_PROFILE", "")
            )["graphs"].items()
            for form in pins
        },
    )
    require_short_numerical_request(
        args,
        prompt_length=short_plan(args.batched_prefill_profile).prompt_length,
        repo=repo,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-environment", action="store_true")
    mode.add_argument("--print-environment", action="store_true")
    parser.add_argument("--profile", default=SHORT_PROFILE)
    parser.add_argument("--context-label")
    args = parser.parse_args()
    if args.validate_environment:
        validate_environment(os.environ)
    else:
        print(json.dumps(numerical_environment(profile=args.profile, context_label=args.context_label), sort_keys=True))


if __name__ == "__main__":
    main()
