"""CPU-staged layer6 observations and same-input replay, no launch authority.

Reuse the actual layer/router, never recreate a supposedly equivalent prefix.
Captures are extra device outputs, not runtime callbacks. They can change the
compiled computation and need original-signature reproduction before attribution.
Worker publication, new graph admission and runtime memory remain separate.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

import numpy as np

from scripts.greenfield.prefill_layer_evidence import equal_bytes
from scripts.greenfield.prefill_layer_numerical import FIELDS


def compiler_output_schema(outputs: Any, *, name: str) -> dict[str, Any]:
    """Describe actual compiled outputs without dispatch or fetching any arrays.

    This is compiler-reported metadata, like memory_analysis, not an independent
    proof of output values or original failure reproduction.
    """
    import jax

    def leaf(value: Any) -> dict[str, Any]:
        return dict(shape=list(value.shape), dtype=str(value.dtype))

    if (
        not isinstance(outputs, tuple)
        or len(outputs) != 2
        or not isinstance(outputs[0], tuple)
        or not isinstance(outputs[1], dict)
    ):
        return dict(
            unexpected_structure=str(jax.tree.structure(outputs)),
            leaves=[leaf(v) for v in jax.tree.leaves(outputs)],
            source="compiled.out_info",
            numerical_claim=False,
        )
    original, captures = outputs
    shardings = {}
    for key, value in captures.items():
        spec = getattr(value.sharding, "spec", None)
        shardings[key] = None if spec is None else list(spec)
    schema = dict(
        original=[leaf(v) for v in original],
        captures={k: leaf(v) for k, v in sorted(captures.items())},
        capture_partition_specs=shardings,
        source="compiled.out_info",
        numerical_claim=False,
    )
    return schema


def validate_output_schema(schema: dict[str, Any], *, name: str) -> None:
    """Check the fixed capture envelope; raw graphs remain separately bound."""
    if name not in ("candidate", "control"):
        raise ValueError("boundary schema requires candidate/control")
    rows, count = (128, 96) if name == "candidate" else (32, 33)
    if (
        set(schema)
        != {
            "original",
            "captures",
            "capture_partition_specs",
            "source",
            "numerical_claim",
        }
        or len(schema["original"]) != 12
        or len(schema["captures"]) != count
        or set(schema["capture_partition_specs"]) != set(schema["captures"])
        or any(
            s != ["expert", "feature"]
            for s in schema["capture_partition_specs"].values()
        )
        or schema["source"] != "compiled.out_info"
        or schema["numerical_claim"] is not False
    ):
        raise ValueError("boundary compiler schema scope differs")
    for value in (*schema["original"], *schema["captures"].values()):
        if (
            set(value) != {"shape", "dtype"}
            or not isinstance(value["shape"], list)
            or any(type(d) is not int or d <= 0 for d in value["shape"])
            or value["dtype"] not in ("bfloat16", "float32", "int32", "bool")
        ):
            raise ValueError("invalid boundary compiler output leaf")
    if schema["original"][0] != dict(shape=[rows, 6144], dtype="bfloat16"):
        raise ValueError("boundary original output geometry differs")
    if any(v["shape"][:2] != [8, 4] for v in schema["captures"].values()):
        raise ValueError("boundary capture owner axes differ")
    if schema["captures"].get("router/input") != dict(
        shape=[8, 4, rows, 1536], dtype="bfloat16"
    ):
        raise ValueError("boundary router input geometry differs")


def output_schema_error(schema: dict[str, Any], *, name: str) -> str | None:
    """Acquisition records malformed schemas instead of discarding their evidence."""
    try:
        validate_output_schema(schema, name=name)
    except (ValueError, TypeError, KeyError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def validate_schema_record(program: dict[str, Any], *, name: str) -> None:
    error = output_schema_error(program["compiler_output_schema"], name=name)
    if program.get("output_schema_error") != error:
        raise ValueError("boundary schema validation outcome differs")


def build_boundary_programs(
    mesh: Any, input_specs: tuple[Any, ...], **options: Any
) -> tuple[Any, Any]:
    """Instrument actual B128 and B32 full-layer paths; no scalar reference."""
    from scripts.greenfield.prefill_layer_programs import build_layer_programs
    from scripts.greenfield.prefill_window_protocol import KEY_TILE

    fixed = dict(
        full_indexer=True,
        sparse_mlp=True,
        key_tile=KEY_TILE,
        capture_boundaries=True,
    )
    wide, _ = build_layer_programs(
        mesh, input_specs, candidate_window=True, **fixed, **options
    )
    small, _ = build_layer_programs(
        mesh, input_specs, candidate_window=False, **fixed, **options
    )
    return wide, small


def build_same_input_router_program(mesh: Any) -> Any:
    """Separately compile B128/B32 using the actual prefill router on its inputs.

    Do not reuse the old layer3 diagnostic, whose own guard accepts <=32 rows.
    It remains historical evidence. No hidden-state/weight gathering is added.
    Caller must use completed BF16 inputs from each path, retain original IDs,
    and establish checkpoint-bound weights/bias; this builder does not do that.
    """
    import jax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.ws32_prefill_layer import ws32_prefill_router_mapped

    def body(hidden: Any, weight: Any, bias: Any, live: Any) -> Any:
        captures: dict[str, Any] = {}

        def observe(name: str, arrays: dict[str, Any]) -> None:
            for field, value in arrays.items():
                key = f"{name}/{field}"
                if key in captures:
                    raise ValueError(f"duplicate router observation: {key}")
                captures[key] = value[None, None]

        ids, weights, health = ws32_prefill_router_mapped(
            hidden, weight, bias, live, _observe=observe
        )
        return (ids, weights, health[None, None]), captures

    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(P(None, "feature"), P("expert", "feature"), P("expert"), P()),
            out_specs=(
                (P(), P(), P("expert", "feature", None)),
                P("expert", "feature"),
            ),
            check_vma=False,
        )
    )


def capture_owner_arrays(
    observations: Mapping[str, Any], *, slots_by_device: Mapping[int, int]
) -> dict[int, dict[str, np.ndarray]]:
    """Read only local shards with explicit expert/feature owner axes.

    Bind each shard's global index to the independently established mesh slot;
    a device's numerical id or process launch rank is never its mesh coordinate.
    No distributed global array is fetched. The caller authenticates the mapping.
    """
    if (
        not observations
        or not slots_by_device
        or any(
            type(d) is not int or type(s) is not int or not 0 <= s < 32
            for d, s in slots_by_device.items()
        )
        or len(set(slots_by_device.values())) != len(slots_by_device)
    ):
        raise ValueError("invalid boundary physical owner mapping")
    result: dict[int, dict[str, np.ndarray]] = {d: {} for d in slots_by_device}
    for name, value in observations.items():
        if not isinstance(name, str) or value.shape[:2] != (8, 4):
            raise ValueError("boundary observation lost expert/feature axes")
        seen = set()
        for shard in value.addressable_shards:
            device = int(shard.device.id)
            if device not in slots_by_device or device in seen:
                raise ValueError("unexpected/duplicate boundary device")
            seen.add(device)
            expert, feature = divmod(slots_by_device[device], 4)
            for index, expected in zip(shard.index[:2], (expert, feature), strict=True):
                if not isinstance(index, slice) or (
                    index.start != expected
                    or index.stop != expected + 1
                    or index.step not in (None, 1)
                ):
                    raise ValueError("boundary shard index differs from physical slot")
            # Inspect only this local device's data, never np.asarray(value).
            data = np.asarray(shard.data)
            if data.shape != (1, 1, *value.shape[2:]) or data.dtype != value.dtype:
                raise ValueError("boundary local observation geometry/dtype differs")
            result[device][name] = data[0, 0].copy()
        if seen != set(slots_by_device):
            raise ValueError("boundary field is missing an authenticated owner")
    return result


def compare_original_outputs(
    observed: Mapping[str, np.ndarray], original: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Report perturbation against all12 archived outputs, not a numeric PASS.

    Caller must bind BOTH paths' originals to their exact failed-run generations
    and devices. Agreement here cannot prove compiler identity or replace the
    numerical admission comparator; disagreement forbids silently attributing
    the original failure to a newly observed boundary.
    """
    if set(observed) != set(FIELDS) or set(original) != set(FIELDS):
        raise ValueError("original boundary reproduction requires all12 fields")
    fields = {}
    for name in FIELDS:
        a, b = np.asarray(observed[name]), np.asarray(original[name])
        fields[name] = dict(
            byte_identical=equal_bytes(a, b),
            observed_shape=list(a.shape),
            original_shape=list(b.shape),
            observed_dtype=str(a.dtype),
            original_dtype=str(b.dtype),
            observed_sha256=sha256(a.tobytes()).hexdigest(),
            original_sha256=sha256(b.tobytes()).hexdigest(),
        )
    signature = ("positions", "counts", "scores", "routes", "route_weights")
    return dict(
        signature_reproduced=all(fields[n]["byte_identical"] for n in signature),
        all_outputs_reproduced=all(v["byte_identical"] for v in fields.values()),
        fields=fields,
        numerical_admission=False,
        performance_claim=False,
    )
