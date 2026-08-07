#!/usr/bin/env python3
"""Capture compact legacy top-logprob evidence for the sealed 2K prompt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "bench"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from glm_tpu.greenfield.validation import (  # noqa: E402
    ShortContextLogprobOracleConfig,
    capture_short_context_logprob_oracle,
    inspect_short_context_oracle,
    normalize_sample_logprobs,
)


MODEL_URI = "gs://driftbench-dsv4-uc/models/GLM-5.2-FP8"
DEFAULT_LEGACY_REPOSITORY = Path("/home/gianl/tpu-inference")
LAUNCHER_REPOSITORY = Path("/home/gianl/glm-tpu")


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True
    ).strip()


def _git_is_ancestor(repository: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        [
            "git", "-C", str(repository), "merge-base", "--is-ancestor",
            ancestor, descendant,
        ],
        check=False,
    ).returncode == 0


def _focus_contract(
    *,
    position: int,
    focus_token_ids: list[int],
    decode_positions: np.ndarray,
    candidate_token_ids: np.ndarray,
    candidate_logprobs: np.ndarray,
) -> dict[str, object]:
    matches = np.flatnonzero(decode_positions == position)
    if matches.size != 1:
        raise ValueError(f"focus position {position} is outside the capture")
    row = int(matches[0])
    ids = candidate_token_ids[row]
    scores = candidate_logprobs[row]
    values: dict[str, object] = {}
    for token_id in focus_token_ids:
        candidate = np.flatnonzero(ids == token_id)
        if candidate.size == 0:
            values[str(token_id)] = {"rank": None, "logprob": None}
            continue
        offset = int(candidate[0])
        values[str(token_id)] = {
            "rank": offset + 1,
            "logprob": float(scores[offset]),
            "top1_margin": float(scores[0] - scores[offset]),
        }
    return {
        "candidate_logprobs": [float(value) for value in scores],
        "candidate_token_ids": [int(value) for value in ids],
        "position": position,
        "tokens": values,
        "top1_top2_margin": float(scores[0] - scores[1]),
    }


def _raw_token_prefix_contract(
    *,
    prompt_token_count: int,
    expected_token_ids: np.ndarray,
    generated_token_ids: np.ndarray,
) -> dict[str, object]:
    """Describe an exact token-prefix comparison without losing a failed draw."""

    expected = np.asarray(expected_token_ids, dtype=np.int32).reshape(-1)
    generated = np.asarray(generated_token_ids, dtype=np.int32).reshape(-1)
    shared = min(expected.size, generated.size)
    unequal = np.flatnonzero(expected[:shared] != generated[:shared])
    first_offset: int | None
    if unequal.size:
        first_offset = int(unequal[0])
    elif expected.size != generated.size:
        first_offset = shared
    else:
        first_offset = None
    first_expected = (
        int(expected[first_offset])
        if first_offset is not None and first_offset < expected.size
        else None
    )
    first_generated = (
        int(generated[first_offset])
        if first_offset is not None and first_offset < generated.size
        else None
    )
    return {
        "expected_count": int(expected.size),
        "expected_token_ids": expected.tolist(),
        "first_mismatch_decode_position": (
            prompt_token_count - 1 + first_offset
            if first_offset is not None
            else None
        ),
        "first_mismatch_expected_token_id": first_expected,
        "first_mismatch_generated_token_id": first_generated,
        "first_mismatch_offset": first_offset,
        "generated_count": int(generated.size),
        "generated_token_ids": generated.tolist(),
        "passed": first_offset is None,
    }


def _write_append_only_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-oracle-dir", type=Path, required=True)
    parser.add_argument("--token-oracle-manifest-sha256", required=True)
    parser.add_argument("--results-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--expected-legacy-code-hash", required=True)
    parser.add_argument(
        "--legacy-repository", type=Path, default=DEFAULT_LEGACY_REPOSITORY
    )
    parser.add_argument("--expected-oracle-base-hash")
    parser.add_argument("--expected-oracle-commit-distance", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--step-count", type=int, default=15)
    parser.add_argument("--focus-position", type=int, default=2044)
    parser.add_argument("--focus-token-id", type=int, action="append", default=[])
    args = parser.parse_args()

    if args.output.exists() or args.result_json.exists():
        raise FileExistsError("append-only logprob capture destination exists")
    code_hash = _git(REPO_ROOT, "rev-parse", "HEAD")
    legacy_repository = args.legacy_repository.resolve()
    legacy_hash = _git(legacy_repository, "rev-parse", "HEAD")
    launcher_hash = _git(LAUNCHER_REPOSITORY, "rev-parse", "HEAD")
    if code_hash != args.expected_code_hash:
        raise RuntimeError("greenfield capture code hash drifted")
    if legacy_hash != args.expected_legacy_code_hash:
        raise RuntimeError("legacy oracle code hash drifted")
    if args.expected_oracle_base_hash is not None:
        if args.expected_oracle_commit_distance <= 0:
            raise ValueError("expected oracle commit distance must be positive")
        distance = int(
            _git(
                legacy_repository,
                "rev-list",
                "--count",
                f"{args.expected_oracle_base_hash}..{legacy_hash}",
            ))
        if (not _git_is_ancestor(
                legacy_repository, args.expected_oracle_base_hash, legacy_hash)
                or distance != args.expected_oracle_commit_distance):
            raise RuntimeError(
                "legacy observer ancestry from the sealed oracle pin drifted")
    if _git(REPO_ROOT, "status", "--porcelain"):
        raise RuntimeError("greenfield capture worktree is dirty")
    if _git(
        legacy_repository, "status", "--porcelain", "--untracked-files=no"
    ):
        raise RuntimeError("legacy oracle worktree is dirty")

    token_manifest = inspect_short_context_oracle(args.token_oracle_dir)
    if token_manifest["manifest_sha256"] != args.token_oracle_manifest_sha256:
        raise RuntimeError("sealed token oracle manifest hash drifted")
    from safetensors.numpy import load_file

    token_arrays = load_file(
        str(
            args.token_oracle_dir
            / token_manifest["files"]["tokens"]["filename"]
        )
    )
    prompt_token_ids = np.asarray(token_arrays["prompt_token_ids"], dtype=np.int32)
    expected_token_ids = np.asarray(
        token_arrays["generated_token_ids"], dtype=np.int32
    )
    if args.step_count <= 0 or args.step_count > expected_token_ids.size:
        raise ValueError("step-count must fit the sealed generated-token oracle")

    import engine
    import provenance as pv

    started = time.monotonic()
    llm = engine.build_llm(
        MODEL_URI,
        max_len=2560,
        max_seqs=1,
        max_batched_tokens=2048,
        gmu=0.90,
        num_gpu_blocks=8,
        log_extra=(
            f"legacy_logprob_oracle=top{args.top_k}:steps{args.step_count}"
        ),
    )
    build_seconds = time.monotonic() - started
    # Import only after build_llm has resolved the TPU platform through the
    # repository's required environment override.  Importing SamplingParams
    # first bypasses that bootstrap and leaves vllm.platforms incomplete.
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.step_count,
        stop_token_ids=[154820, 154827, 154829, 154828],
        ignore_eos=False,
        logprobs=args.top_k,
    )
    generation_started = time.monotonic()
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_token_ids.tolist()}],
        sampling,
        use_tqdm=False,
    )
    generation_seconds = time.monotonic() - generation_started
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError("legacy oracle returned an unexpected output cardinality")
    completion = outputs[0].outputs[0]
    generated_token_ids = np.asarray(completion.token_ids, dtype=np.int32)
    token_contract = _raw_token_prefix_contract(
        prompt_token_count=int(prompt_token_ids.size),
        expected_token_ids=expected_token_ids[: args.step_count],
        generated_token_ids=generated_token_ids,
    )
    if generated_token_ids.shape != (args.step_count,) or not token_contract[
        "passed"
    ]:
        failure = {
            "artifact_kind": "glm52_legacy_logprob_failed_draw",
            "build_seconds": build_seconds,
            "generation_seconds": generation_seconds,
            "greenfield_capture_code_hash": code_hash,
            "legacy_observer_code_hash": legacy_hash,
            "legacy_oracle_base_hash": args.expected_oracle_base_hash,
            "legacy_oracle_commit_distance": (
                args.expected_oracle_commit_distance
            ),
            "passed": False,
            "token_contract": token_contract,
            "token_oracle_manifest_sha256": token_manifest["manifest_sha256"],
        }
        _write_append_only_json(args.result_json, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        raise RuntimeError(
            "legacy logprob run did not reproduce the sealed raw-token prefix"
        )
    if completion.logprobs is None or len(completion.logprobs) != args.step_count:
        raise RuntimeError("legacy runtime did not return every requested logprob row")
    candidate_ids, candidate_logprobs, candidate_ranks = normalize_sample_logprobs(
        completion.logprobs,
        top_k=args.top_k,
    )

    harness_short = _git(REPO_ROOT, "rev-parse", "--short", "HEAD")
    legacy_short = _git(legacy_repository, "rev-parse", "--short", "HEAD")
    connection = pv.connect(str(args.results_db))
    run_id = pv.start_run(
        connection,
        model=MODEL_URI,
        revision=token_manifest["source"].get("model_revision"),
        env={
            "artifact_kind": "greenfield_short_context_legacy_logprobs",
            "attention_path": engine.attention_path(),
            "generated_steps": args.step_count,
            "launcher_code_hash": launcher_hash,
            "legacy_oracle_base_pin": args.expected_oracle_base_hash,
            "legacy_oracle_commit_distance": (
                args.expected_oracle_commit_distance
            ),
            "prompt_token_count": int(prompt_token_ids.size),
            "sampling": "greedy",
            "token_oracle_manifest_sha256": token_manifest["manifest_sha256"],
            "top_k": args.top_k,
        },
        note="sealed 2K legacy top-logprob arithmetic diagnostic",
        harness_repo=str(REPO_ROOT),
        fork_repo=str(legacy_repository),
    )
    prompt_path = args.token_oracle_dir / token_manifest["files"]["prompt"][
        "filename"
    ]
    pv.record_item(
        connection,
        run_id,
        benchmark="greenfield_legacy_logprob_diagnostic",
        item_id="sealed_2k_top16",
        prompt=prompt_path.read_text(encoding="utf-8"),
        gold=json.dumps(expected_token_ids[: args.step_count].tolist()),
        raw_output=completion.text,
        extracted=json.dumps(generated_token_ids.tolist()),
        correct=True,
        score=1.0,
        n_prompt_tokens=int(prompt_token_ids.size),
        n_gen_tokens=int(generated_token_ids.size),
        latency_ms=round(generation_seconds * 1000.0, 3),
        finish_reason=completion.finish_reason,
        truncated=completion.finish_reason == "length",
    )
    item_row_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
    pv.finalize(
        connection,
        run_id,
        benchmark="greenfield_legacy_logprob_diagnostic",
        metric="exact_raw_token_prefix",
        value=100.0,
        note=f"top_k={args.top_k} steps={args.step_count}",
    )
    connection.close()

    manifest = capture_short_context_logprob_oracle(
        ShortContextLogprobOracleConfig(
            token_oracle_dir=args.token_oracle_dir,
            output_dir=args.output,
            capture_code_hash=code_hash,
            legacy_repository_pin=legacy_hash,
            token_oracle_manifest_sha256=token_manifest["manifest_sha256"],
            source_run_id=run_id,
            source_item_row_id=item_row_id,
            source_harness_git=harness_short,
            source_fork_git=legacy_short,
            source_launcher_git=launcher_hash,
            top_k=args.top_k,
            step_count=args.step_count,
        ),
        generated_token_ids=generated_token_ids,
        candidate_token_ids=candidate_ids,
        candidate_logprobs=candidate_logprobs,
        candidate_ranks=candidate_ranks,
    )
    decode_positions = np.arange(
        int(prompt_token_ids.size) - 1,
        int(prompt_token_ids.size) - 1 + args.step_count,
        dtype=np.int32,
    )
    focus = _focus_contract(
        position=args.focus_position,
        focus_token_ids=args.focus_token_id,
        decode_positions=decode_positions,
        candidate_token_ids=candidate_ids,
        candidate_logprobs=candidate_logprobs,
    )
    result = {
        "build_seconds": build_seconds,
        "focus": focus,
        "generation_seconds": generation_seconds,
        "item_row_id": item_row_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "run_id": run_id,
        "step_count": args.step_count,
        "top_k": args.top_k,
    }
    _write_append_only_json(args.result_json, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
