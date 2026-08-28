#!/usr/bin/env python3
"""Compile-only protected acquisition for the PP16 feature2 discriminator."""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPECTED_REPO = Path("/home/gianl/glm-tpu-topology-rewrite")
FEATURE2_GRAPH_SHA256 = (
    "ab5be45aecf3b0b5d87ad76af8076bc9351823529a08c0eadb414b072b31cb2d"
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _memory_stats(device: Any) -> dict[str, int] | None:
    values = device.memory_stats()
    if values is None:
        return None
    return {
        str(name): int(value)
        for name, value in values.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def _memory_analysis(compiled: Any) -> dict[str, int]:
    analysis = compiled.memory_analysis()
    result = {}
    for name in (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    ):
        value = getattr(analysis, name, None)
        if value is not None:
            result[name] = int(value)
    return result


def _record_hlo(
    hlo_dir: Path,
    name: str,
    lowered: Any,
    compiled: Any,
) -> tuple[str, str, dict[str, Any]]:
    stablehlo = lowered.as_text()
    optimized_hlo = compiled.as_text()
    stable_path = hlo_dir / f"{name}.stablehlo.mlir"
    optimized_path = hlo_dir / f"{name}.optimized_hlo.txt"
    _atomic_text(stable_path, stablehlo)
    _atomic_text(optimized_path, optimized_hlo)
    return stablehlo, optimized_hlo, {
        "memory_analysis": _memory_analysis(compiled),
        "optimized_hlo": {
            "filename": optimized_path.name,
            "sha256": sha256(optimized_hlo.encode()).hexdigest(),
        },
        "stablehlo": {
            "filename": stable_path.name,
            "sha256": sha256(stablehlo.encode()).hexdigest(),
        },
    }


def _compile(
    function: Any,
    arguments: Sequence[Any],
    *,
    hlo_dir: Path,
    name: str,
) -> tuple[Any, str, str, dict[str, Any]]:
    started = time.monotonic()
    lowered = function.lower(*arguments)
    lowering_seconds = time.monotonic() - started
    stablehlo = lowered.as_text()
    _atomic_text(hlo_dir / f"{name}.stablehlo.mlir", stablehlo)
    started = time.monotonic()
    compiled = lowered.compile()
    compile_seconds = time.monotonic() - started
    stablehlo, optimized_hlo, record = _record_hlo(
        hlo_dir, name, lowered, compiled
    )
    record.update(
        {
            "compile_seconds": compile_seconds,
            "lowering_seconds": lowering_seconds,
        }
    )
    return compiled, stablehlo, optimized_hlo, record


def _delete_tree(jax: Any, value: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        try:
            leaf.delete()
        except (AttributeError, RuntimeError):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-code-hash", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--token-oracle-dir", type=Path, required=True)
    parser.add_argument("--dsa-oracle-dir", type=Path, required=True)
    parser.add_argument("--layer1-internal-reference", type=Path, required=True)
    parser.add_argument("--db529-internal-dir", type=Path, required=True)
    parser.add_argument("--compile-only", type=int, choices=(1,), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if REPO != EXPECTED_REPO:
        raise RuntimeError(f"wrong greenfield worktree: {REPO}")
    code_hash = _git_head()
    if code_hash != args.expected_code_hash:
        raise RuntimeError(
            f"stale code hash: expected={args.expected_code_hash} found={code_hash}"
        )

    from glm_tpu.greenfield.benchmarking.pp16_feature2_acquisition import (
        inspect_feature2_event1_lineage,
    )
    from glm_tpu.greenfield.benchmarking.pp16_feature2_loader import (
        inspect_feature2_selective_plan,
    )

    event1_lineage = inspect_feature2_event1_lineage(
        token_oracle_dir=args.token_oracle_dir,
        dsa_oracle_dir=args.dsa_oracle_dir,
        layer1_internal_reference=args.layer1_internal_reference,
        db529_internal_dir=args.db529_internal_dir,
    )
    selective_plan = inspect_feature2_selective_plan(args.runtime_root)

    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from glm_tpu.greenfield.benchmarking.pp16_feature2_hlo import (
        validate_feature2_main_optimized_hlo,
        validate_feature2_main_stablehlo,
        validate_feature2_materializer_optimized_hlo,
    )
    from glm_tpu.greenfield.benchmarking.pp16_feature2_loader import (
        load_feature2_selective_checkpoint,
    )
    from glm_tpu.greenfield.benchmarking.pp16_feature2_prefill import (
        build_feature2_prefill_graph,
        load_feature2_prefill_inputs,
    )
    from glm_tpu.greenfield.benchmarking.pp16_feature2_program import (
        build_feature2_prefill_program,
        load_feature2_prefill_runtime_inputs,
        validate_feature2_prefill_jaxpr,
        validate_feature2_prefill_result_abstract,
    )
    from glm_tpu.greenfield.benchmarking.pp16_feature_sharded_state import (
        derive_feature2_tensor_allowlist,
        read_feature2_owner_headers,
    )

    if jax.default_backend() != "tpu" or jax.local_device_count() != 4:
        raise RuntimeError(
            "feature2 acquisition requires one four-chip TPU host: "
            f"backend={jax.default_backend()} local={jax.local_device_count()}"
        )
    devices = tuple(jax.local_devices()[:2])
    coordinates = tuple(tuple(device.coords) for device in devices)
    if tuple(int(device.id) for device in devices) != (0, 1) or coordinates != (
        (0, 0, 0),
        (1, 0, 0),
    ):
        raise RuntimeError(
            "feature2 acquisition lost exact adjacent devices 0,1: "
            f"ids={[device.id for device in devices]} coords={coordinates}"
        )
    args.hlo_dir.mkdir(parents=True, exist_ok=True)
    memory_before_load = [_memory_stats(device) for device in devices]

    runtime_manifest = json.loads(
        (args.runtime_root / "runtime_manifest.json").read_text()
    )
    graph = build_feature2_prefill_graph(
        derive_feature2_tensor_allowlist(
            runtime_manifest,
            read_feature2_owner_headers(args.runtime_root, runtime_manifest),
        ),
        load_feature2_prefill_inputs(args.token_oracle_dir),
    )
    if graph.graph_sha256 != FEATURE2_GRAPH_SHA256:
        raise RuntimeError(
            "feature2 executable graph identity drifted: "
            f"{graph.graph_sha256}"
        )
    runtime_host = load_feature2_prefill_runtime_inputs(args.token_oracle_dir)

    loaded = None
    compiled_graphs: list[Any] = []
    disposable_arrays: list[Any] = []
    output: dict[str, Any] | None = None
    try:
        loaded = load_feature2_selective_checkpoint(
            args.runtime_root,
            devices,
            pack_dense_final_layout=True,
        )
        memory_after_load = [_memory_stats(device) for device in devices]
        program = build_feature2_prefill_program(graph, devices=devices)
        weights = loaded.weights

        query_arguments = (
            weights["indexer.slot_00.wq_b.weight_bits"],
            weights["indexer.slot_00.wq_b.scale_inv"],
            weights["indexer.slot_01.wq_b.weight_bits"],
            weights["indexer.slot_01.wq_b.scale_inv"],
        )
        query_compiled, _, query_hlo, query_record = _compile(
            jax.jit(program.materialize_query_weights_fp32),
            query_arguments,
            hlo_dir=args.hlo_dir,
            name="query_fp32",
        )
        compiled_graphs.append(query_compiled)
        query_record["contract"] = validate_feature2_materializer_optimized_hlo(
            query_hlo, phase="query_fp32"
        )
        query_weights = query_compiled(*query_arguments)
        jax.block_until_ready(query_weights)
        disposable_arrays.append(query_weights)

        wk_arguments = (
            weights["indexer.slot_00.wk.weight_bits"],
            weights["indexer.slot_00.wk.scale_inv"],
            weights["indexer.slot_01.wk.weight_bits"],
            weights["indexer.slot_01.wk.scale_inv"],
        )
        wk_decode_compiled, _, wk_decode_hlo, wk_decode_record = _compile(
            jax.jit(program.decode_index_weights_bf16),
            wk_arguments,
            hlo_dir=args.hlo_dir,
            name="wk_decode_bf16",
        )
        compiled_graphs.append(wk_decode_compiled)
        wk_decode_record["contract"] = (
            validate_feature2_materializer_optimized_hlo(
                wk_decode_hlo, phase="wk_decode_bf16"
            )
        )
        wk_bf16 = wk_decode_compiled(*wk_arguments)
        jax.block_until_ready(wk_bf16)
        disposable_arrays.append(wk_bf16)

        wk_promote_compiled, _, wk_promote_hlo, wk_promote_record = _compile(
            jax.jit(program.promote_index_weights_fp32),
            wk_bf16,
            hlo_dir=args.hlo_dir,
            name="wk_promote_fp32",
        )
        compiled_graphs.append(wk_promote_compiled)
        wk_promote_record["contract"] = (
            validate_feature2_materializer_optimized_hlo(
                wk_promote_hlo, phase="wk_promote_fp32"
            )
        )
        wk_fp32 = wk_promote_compiled(*wk_bf16)
        jax.block_until_ready(wk_fp32)
        disposable_arrays.append(wk_fp32)

        replicated = NamedSharding(program.mesh, P())
        runtime_arrays = tuple(
            jax.device_put(value, replicated) for value in runtime_host
        )
        jax.block_until_ready(runtime_arrays)
        disposable_arrays.append(runtime_arrays)
        main_arguments = (
            weights,
            query_weights[0],
            query_weights[1],
            wk_fp32[0],
            wk_fp32[1],
            *runtime_arrays,
        )
        jaxpr_contract = validate_feature2_prefill_jaxpr(
            str(jax.make_jaxpr(program.execute)(*main_arguments))
        )
        terminal_contract = validate_feature2_prefill_result_abstract(
            jax.eval_shape(program.execute, *main_arguments)
        )

        main_jit = jax.jit(program.execute)
        lowering_started = time.monotonic()
        main_lowered = main_jit.lower(*main_arguments)
        main_lowering_seconds = time.monotonic() - lowering_started
        main_stablehlo = main_lowered.as_text()
        _atomic_text(
            args.hlo_dir / "feature2_main.stablehlo.mlir", main_stablehlo
        )
        main_stable_contract = validate_feature2_main_stablehlo(main_stablehlo)
        compile_started = time.monotonic()
        main_compiled = main_lowered.compile()
        main_compile_seconds = time.monotonic() - compile_started
        compiled_graphs.append(main_compiled)
        _, main_hlo, main_record = _record_hlo(
            args.hlo_dir, "feature2_main", main_lowered, main_compiled
        )
        main_record.update(
            {
                "compile_seconds": main_compile_seconds,
                "jaxpr_contract": jaxpr_contract,
                "lowering_seconds": main_lowering_seconds,
                "optimized_contract": validate_feature2_main_optimized_hlo(
                    main_hlo
                ),
                "stablehlo_contract": main_stable_contract,
                "terminal_contract": terminal_contract,
            }
        )
        memory_after_compile = [_memory_stats(device) for device in devices]
        output = {
            "artifact_kind": "greenfield_pp16_feature2_compile_acquisition",
            "claim_scope": (
                "compile-only real selected-state/HLO/HBM acquisition; no main "
                "arithmetic, numerical, Gate-D, token-rate, or performance claim"
            ),
            "code_hash": code_hash,
            "compile_only": True,
            "device_kind": devices[0].device_kind,
            "event1_target_lineage": event1_lineage,
            "graph_sha256": graph.graph_sha256,
            "hlo": {
                "feature2_main": main_record,
                "query_fp32": query_record,
                "wk_decode_bf16": wk_decode_record,
                "wk_promote_fp32": wk_promote_record,
            },
            "main_executed": False,
            "memory": {
                "after_compile": memory_after_compile,
                "after_load": memory_after_load,
                "before_load": memory_before_load,
            },
            "numerical_claim": False,
            "performance_claim": False,
            "physical_group": {
                "coordinates": [list(value) for value in coordinates],
                "device_ids": [int(device.id) for device in devices],
                "local_device_count_visible": jax.local_device_count(),
                "mesh_device_count": 2,
            },
            "selective_load": loaded.load_record,
            "selective_plan": selective_plan,
            "state_manifest": loaded.state_manifest,
            "status": "HLO_ACQUIRED",
        }
    finally:
        for value in reversed(disposable_arrays):
            _delete_tree(jax, value)
        for compiled in reversed(compiled_graphs):
            try:
                compiled.delete()
            except (AttributeError, RuntimeError):
                pass
        if loaded is not None:
            loaded.close()
        jax.clear_caches()
        gc.collect()

    if output is None:
        raise RuntimeError("feature2 compile acquisition produced no record")
    output["memory"]["after_cleanup"] = [
        _memory_stats(device) for device in devices
    ]
    _atomic_json(args.output, output)
    print(json.dumps(output, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
