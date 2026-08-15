"""Model-free discriminator for the layer-1 weighted-output physical tile.

The protected native-source runs already prove that the accepted M32 and
rejected M1 programs consume the same M32 RMS inverse.  Their first remaining
difference is the final result tile.  This module holds the exact BF16 inputs
fixed, computes their shared M32 inverse once, and evaluates three live arms
in one executable:

* the accepted-shape M32 output control;
* the ordinary M1 output control;
* a true-M1 Pallas output with an internal M8-by-128 TPU scratch tile.

Only the third arm is a production candidate.  The M32 arm is diagnostic and
is retained solely to make the hardware comparison self-contained.  The
direct logical-M1/M32 result-layout override is permanently rejected by TPU
XLA and is not repeated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import ml_dtypes
import numpy as np

from ..errors import BenchmarkValidationError
from ..kernels.pallas.rmsnorm import weighted_output_m1_m8_scratch
from ..kernels.stage_local import (
    STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE,
)
from .association_fingerprint import (
    array_sha256,
    replay_db533_strategy_nd_row0_bits,
)
from .dense_rms_replay import (
    DENSE_RMS_ARRAY_RECORDS,
    DENSE_RMS_SOURCE_KEYS,
    DENSE_RMS_SOURCE_NPZ_SHA256,
)


# Acquired once by the fail-closed protected compile at ``046a1f7``.
# Complete graph pins are mandatory before the numerical replay can run.
# Keep this block line-stable: its locations are embedded in optimized HLO.
OUTPUT_GEOMETRY_STABLEHLO_SHA256 = "0fda9f03eebe0d5d68f97dcb2fabec50face70209a3de8e921d0922162449c28"
OUTPUT_GEOMETRY_OPTIMIZED_HLO_SHA256 = "0107fe68bf806de35873127be281029c618746c052047e719ed97b3f010db12b"

M32_BF16_LAYOUT = "{1,0:T(8,128)(2,1)}"
M1_AUTO_BF16_LAYOUT = "{1,0:T(2,128)(2,1)}"
M1_PALLAS_M8_SCRATCH_SHAPE = (8, 128)
OUTPUT_GEOMETRY_DENSE_ROW_RAW_SHA256 = (
    "efde853254c03dd18a5f5f22733630ce0e785dfbb4eba09c41eea9085e47b4fc"
)
OUTPUT_GEOMETRY_INVERSE_RAW_SHA256 = (
    "13eed6c367303012d09df1fc7dcaa6a4d8ccb544b112fa74ac3f24b08a9971ce"
)
_PHYSICAL_DEVICE_BY_MODEL_POSITION = tuple(
    int(value)
    for value in np.argsort(
        np.asarray(STRATEGY_ND_MODEL_POSITION_BY_PHYSICAL_DEVICE)
    )
)


@dataclass(frozen=True, slots=True)
class OutputGeometryInputs:
    """Exact sealed rows and an offline inverse used only as a source check."""

    dense_bits: np.ndarray
    carried_bits: np.ndarray
    weight_bits: np.ndarray
    accepted_bits: np.ndarray
    inverse: np.ndarray


@dataclass(frozen=True, slots=True)
class CompiledOutputGeometryReplay:
    compiled: Any
    replicated_sharding: Any
    member_device_ids: tuple[int, ...]
    stablehlo: str
    optimized_hlo: str
    stablehlo_contract: Mapping[str, Any]
    optimized_hlo_contract: Mapping[str, Any]


def _pairwise_comparison(
    observed: np.ndarray,
    expected: np.ndarray,
) -> dict[str, Any]:
    """Return the exact typed comparison used by producer and sealer."""

    observed = np.ascontiguousarray(observed, dtype=np.uint16).reshape(6144)
    expected = np.ascontiguousarray(expected, dtype=np.uint16).reshape(6144)
    mismatch_indices = np.flatnonzero(observed != expected)
    first = None if not len(mismatch_indices) else int(mismatch_indices[0])
    observed_values = (
        observed.astype(np.uint32) << np.uint32(16)
    ).view(np.float32)
    expected_values = (
        expected.astype(np.uint32) << np.uint32(16)
    ).view(np.float32)
    error = np.abs(observed_values - expected_values)
    return {
        "elementwise_exact": first is None,
        "expected_raw_sha256": _raw_sha256(expected),
        "first_mismatch_index": first,
        "max_absolute_error": float(np.max(error)),
        "mean_absolute_error": float(np.mean(error, dtype=np.float64)),
        "mismatch_count": int(len(mismatch_indices)),
        "observed_raw_sha256": _raw_sha256(observed),
    }


def compare_output_geometry_arrays(
    outputs: Mapping[str, np.ndarray],
    accepted_bits: np.ndarray,
) -> dict[str, Any]:
    """Recompute the decisive control/candidate numerical relationships."""

    if set(outputs) != {"m32_control", "m1_auto", "m1_pallas_m8"}:
        raise BenchmarkValidationError("output-geometry output set drifted")
    m32 = np.ascontiguousarray(outputs["m32_control"], dtype=np.uint16)
    auto = np.ascontiguousarray(outputs["m1_auto"], dtype=np.uint16)
    pallas = np.ascontiguousarray(outputs["m1_pallas_m8"], dtype=np.uint16)
    accepted = np.ascontiguousarray(accepted_bits, dtype=np.uint16)
    if (
        m32.shape != (32, 6144)
        or auto.shape != (1, 6144)
        or pallas.shape != (1, 6144)
        or accepted.shape != (6144,)
    ):
        raise BenchmarkValidationError("output-geometry comparison shape drifted")
    m32_row = m32[0]
    comparisons = {
        "m32_control_vs_accepted": _pairwise_comparison(m32_row, accepted),
        "m1_auto_vs_m32_control": _pairwise_comparison(auto[0], m32_row),
        "m1_pallas_m8_vs_accepted": _pairwise_comparison(pallas[0], accepted),
        "m1_pallas_m8_vs_m32_control": _pairwise_comparison(
            pallas[0], m32_row
        ),
    }
    control_exact = comparisons["m32_control_vs_accepted"]["elementwise_exact"]
    pallas_exact = comparisons["m1_pallas_m8_vs_m32_control"][
        "elementwise_exact"
    ]
    auto_exact = comparisons["m1_auto_vs_m32_control"]["elementwise_exact"]
    classification = (
        "output_geometry_invalid_m32_control"
        if not control_exact
        else "output_geometry_m1_pallas_m8_exact_control"
        if pallas_exact
        else "output_geometry_m1_pallas_m8_matches_auto_control"
        if np.array_equal(pallas, auto)
        else "output_geometry_m1_pallas_m8_nonexact_new_result"
    )
    return {
        "classification": classification,
        "m1_auto_exact_m32_control": bool(auto_exact),
        "m1_pallas_m8_exact_m32_control": bool(pallas_exact),
        "m32_control_exact_accepted": bool(control_exact),
        "pairwise": comparisons,
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_sha256(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_output_geometry_inputs(
    source_path: Path,
    model_axis_device_ids: Sequence[int],
) -> OutputGeometryInputs:
    """Reconstruct the exact DB533 dense row from the sealed DB548 bundle."""

    if (
        not source_path.is_file()
        or _file_sha256(source_path) != DENSE_RMS_SOURCE_NPZ_SHA256
    ):
        raise BenchmarkValidationError(
            "output-geometry source NPZ SHA-256 drifted"
        )
    arrays: dict[str, np.ndarray] = {}
    with np.load(source_path, allow_pickle=False) as payload:
        if tuple(payload.files) != DENSE_RMS_SOURCE_KEYS:
            raise BenchmarkValidationError(
                "output-geometry source keys drifted"
            )
        for name in payload.files:
            value = np.ascontiguousarray(payload[name])
            shape, dtype, digest = DENSE_RMS_ARRAY_RECORDS[name]
            if (
                value.shape != shape
                or value.dtype != dtype
                or _raw_sha256(value) != digest
            ):
                raise BenchmarkValidationError(
                    f"output-geometry source tensor drifted: {name}"
                )
            arrays[name] = value

    mapping = tuple(int(value) for value in model_axis_device_ids)
    if mapping != _PHYSICAL_DEVICE_BY_MODEL_POSITION:
        raise BenchmarkValidationError(
            "output-geometry model-axis physical mapping drifted"
        )

    dense = replay_db533_strategy_nd_row0_bits(
        arrays["dense_virtual_partials_bfloat16_bits"].reshape(32, 6144),
        mapping,
    ).reshape(1, 6144)
    if _raw_sha256(dense) != OUTPUT_GEOMETRY_DENSE_ROW_RAW_SHA256:
        raise BenchmarkValidationError(
            "output-geometry reconstructed dense row drifted"
        )
    carried = arrays["post_attention_residual_bfloat16_bits"]
    weight = arrays["layer1_input_norm_bfloat16_bits"]
    accepted = arrays["accepted_layer1_normalized_bfloat16_bits"]
    summed = (
        dense.view(ml_dtypes.bfloat16).astype(np.float32)
        + carried.view(ml_dtypes.bfloat16).astype(np.float32)
    )
    inverse = np.ascontiguousarray(
        np.float32(1.0)
        / np.sqrt(
            np.mean(
                np.square(summed, dtype=np.float32),
                axis=1,
                dtype=np.float32,
            )
            + np.float32(1e-5),
            dtype=np.float32,
        )
    )
    result = OutputGeometryInputs(
        dense_bits=np.ascontiguousarray(dense),
        carried_bits=np.ascontiguousarray(carried),
        weight_bits=np.ascontiguousarray(weight),
        accepted_bits=np.ascontiguousarray(accepted),
        inverse=inverse,
    )
    _validate_inputs(result)
    if _raw_sha256(result.inverse) != OUTPUT_GEOMETRY_INVERSE_RAW_SHA256:
        raise BenchmarkValidationError("output-geometry inverse drifted")
    return result


def _validate_inputs(inputs: OutputGeometryInputs) -> None:
    records = {
        "dense_bits": (inputs.dense_bits, (1, 6144), np.dtype(np.uint16)),
        "carried_bits": (
            inputs.carried_bits,
            (1, 6144),
            np.dtype(np.uint16),
        ),
        "weight_bits": (
            inputs.weight_bits,
            (6144,),
            np.dtype(np.uint16),
        ),
        "accepted_bits": (
            inputs.accepted_bits,
            (6144,),
            np.dtype(np.uint16),
        ),
        "inverse": (inputs.inverse, (1,), np.dtype(np.float32)),
    }
    for name, (value, shape, dtype) in records.items():
        if value.shape != shape or value.dtype != dtype:
            raise BenchmarkValidationError(
                f"output-geometry {name} shape/dtype drifted"
            )
    if not np.all(np.isfinite(inputs.inverse)) or not np.all(inputs.inverse > 0):
        raise BenchmarkValidationError("output-geometry inverse is invalid")


def _exact_digest(value: str, expected: str, label: str) -> str:
    observed = sha256(value.encode()).hexdigest()
    if not expected:
        raise BenchmarkValidationError(
            f"output-geometry {label} pin is deliberately empty"
        )
    if observed != expected:
        raise BenchmarkValidationError(
            f"output-geometry {label} SHA-256 drifted: {observed}"
        )
    return observed


def validate_output_geometry_stablehlo(stablehlo: str) -> Mapping[str, Any]:
    """Fail closed on the exact acquired StableHLO graph."""

    digest = _exact_digest(
        stablehlo,
        OUTPUT_GEOMETRY_STABLEHLO_SHA256,
        "StableHLO",
    )
    from .output_geometry_hlo import (
        validate_output_geometry_stable_structure,
    )
    return validate_output_geometry_stable_structure(
        stablehlo, digest)


def validate_output_geometry_hlo(optimized_hlo: str) -> Mapping[str, Any]:
    """Fail closed on the exact acquired scheduled graph."""

    digest = _exact_digest(
        optimized_hlo,
        OUTPUT_GEOMETRY_OPTIMIZED_HLO_SHA256,
        "optimized HLO",
    )
    from .output_geometry_hlo import (
        validate_output_geometry_optimized_structure,
    )
    return validate_output_geometry_optimized_structure(
        optimized_hlo, digest)


def _output_geometry_program(*, pallas_interpret: bool) -> Any:
    """Return the shared-scalar three-arm function for abstract/local tests."""

    import jax
    import jax.numpy as jnp
    from jax import lax

    if not isinstance(pallas_interpret, bool):
        raise BenchmarkValidationError(
            "output-geometry Pallas interpret flag must be boolean"
        )

    def weighted(
        dense: Any,
        carried: Any,
        inverse: Any,
        weight: Any,
    ) -> Any:
        summed = dense.astype(jnp.float32) + carried.astype(jnp.float32)
        rounded = (summed * inverse[:, None]).astype(jnp.bfloat16)
        output = (rounded * weight[None, :]).astype(jnp.bfloat16)
        return lax.bitcast_convert_type(output, jnp.uint16)

    def program(
        dense_m32: Any,
        carried_m32: Any,
        dense_auto: Any,
        carried_auto: Any,
        weight: Any,
    ) -> tuple[Any, Any, Any]:
        with jax.named_scope("output_geometry_shared_m32_inverse"):
            shared_sum = (
                lax.optimization_barrier(dense_m32).astype(jnp.float32)
                + lax.optimization_barrier(carried_m32).astype(jnp.float32)
            )
            shared_inverse = lax.rsqrt(
                jnp.mean(jnp.square(shared_sum), axis=1)
                + jnp.float32(1e-5)
            )
            shared_inverse = lax.optimization_barrier(shared_inverse)
        with jax.named_scope("output_geometry_m32_control"):
            m32 = weighted(
                dense_m32,
                carried_m32,
                shared_inverse,
                weight,
            )
        with jax.named_scope("output_geometry_m1_auto_control"):
            m1_auto = weighted(
                lax.optimization_barrier(dense_auto),
                lax.optimization_barrier(carried_auto),
                shared_inverse[:1],
                weight,
            )
        with jax.named_scope("output_geometry_m1_pallas_m8_scratch"):
            m1_pallas = weighted_output_m1_m8_scratch(
                lax.optimization_barrier(dense_auto),
                lax.optimization_barrier(carried_auto),
                shared_inverse[:1],
                weight,
                interpret=pallas_interpret,
            )
            m1_pallas = lax.bitcast_convert_type(m1_pallas, jnp.uint16)
        return m32, m1_auto, m1_pallas

    return program


def build_output_geometry_replay(
    member_device_ids: Sequence[int],
    *,
    devices: Sequence[Any] | None = None,
    validate_hlo: bool = True,
    pallas_interpret: bool = False,
) -> CompiledOutputGeometryReplay:
    """Compile the M32/M1/output-only-Pallas discriminator."""

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    runtime_devices = tuple(jax.devices() if devices is None else devices)
    members = tuple(int(value) for value in member_device_ids)
    by_id = {int(device.id): device for device in runtime_devices}
    if members != tuple(range(32)) or set(by_id) != set(members):
        raise BenchmarkValidationError(
            "output-geometry replay requires global physical ids 0..31"
        )
    mesh = Mesh(
        np.asarray([by_id[device_id] for device_id in members], dtype=object),
        ("member",),
    )
    replicated = NamedSharding(mesh, P())

    shapes = (
        (32, 6144),
        (32, 6144),
        (1, 6144),
        (1, 6144),
        (6144,),
    )
    dtypes = (
        jnp.bfloat16,
        jnp.bfloat16,
        jnp.bfloat16,
        jnp.bfloat16,
        jnp.bfloat16,
    )
    examples = tuple(
        jax.ShapeDtypeStruct(shape, dtype, sharding=replicated)
        for shape, dtype in zip(shapes, dtypes, strict=True)
    )
    mapped = jax.shard_map(
        _output_geometry_program(pallas_interpret=pallas_interpret),
        mesh=mesh,
        in_specs=(P(),) * len(examples),
        out_specs=(P(), P(), P()),
        check_vma=False,
    )
    lowered = jax.jit(mapped).lower(*examples)
    stablehlo = lowered.as_text()
    compiled = lowered.compile()
    optimized_hlo = compiled.as_text()
    if validate_hlo:
        stablehlo_contract = validate_output_geometry_stablehlo(stablehlo)
        optimized_hlo_contract = validate_output_geometry_hlo(optimized_hlo)
    else:
        stablehlo_contract = {
            "passed": False,
            "violations": ["HLO validation explicitly disabled"],
        }
        optimized_hlo_contract = {
            "passed": False,
            "violations": ["HLO validation explicitly disabled"],
        }
    return CompiledOutputGeometryReplay(
        compiled=compiled,
        replicated_sharding=replicated,
        member_device_ids=members,
        stablehlo=stablehlo,
        optimized_hlo=optimized_hlo,
        stablehlo_contract=stablehlo_contract,
        optimized_hlo_contract=optimized_hlo_contract,
    )


def execute_output_geometry_replay(
    compiled: CompiledOutputGeometryReplay,
    inputs: OutputGeometryInputs,
) -> tuple[Mapping[str, np.ndarray], Mapping[str, Any]]:
    """Execute twice and return three deterministic replicated artifacts."""

    import jax

    _validate_inputs(inputs)
    bf16 = ml_dtypes.bfloat16
    dense = inputs.dense_bits.view(bf16)
    carried = inputs.carried_bits.view(bf16)
    weight = inputs.weight_bits.view(bf16)
    dense_m32 = np.full((32, 6144), bf16(np.nan), dtype=bf16)
    carried_m32 = np.full((32, 6144), bf16(np.nan), dtype=bf16)
    dense_m32[0] = dense[0]
    carried_m32[0] = carried[0]
    arguments = (
        jax.device_put(dense_m32, compiled.replicated_sharding),
        jax.device_put(carried_m32, compiled.replicated_sharding),
        jax.device_put(dense, compiled.replicated_sharding),
        jax.device_put(carried, compiled.replicated_sharding),
        jax.device_put(weight, compiled.replicated_sharding),
    )

    def once() -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
        result = compiled.compiled(*arguments)
        jax.block_until_ready(result)
        names = ("m32_control", "m1_auto", "m1_pallas_m8")
        values: dict[str, np.ndarray] = {}
        hashes: dict[str, tuple[str, ...]] = {}
        for name, output in zip(names, result, strict=True):
            local = tuple(
                np.ascontiguousarray(
                    np.asarray(jax.device_get(shard.data), dtype=np.uint16)
                )
                for shard in sorted(
                    output.addressable_shards,
                    key=lambda shard: int(shard.device.id),
                )
            )
            local_hashes = tuple(array_sha256(value) for value in local)
            if not local or len(set(local_hashes)) != 1:
                raise BenchmarkValidationError(
                    f"output-geometry {name} differs across local replicas"
                )
            values[name] = local[0]
            hashes[name] = local_hashes
        return values, hashes

    first, first_hashes = once()
    repeated, repeated_hashes = once()
    if any(
        not np.array_equal(first[name], repeated[name]) for name in first
    ):
        raise BenchmarkValidationError("output-geometry replay is nondeterministic")
    return first, {
        "invocation_count": 2,
        "local_replica_sha256": {
            name: list(value) for name, value in first_hashes.items()
        },
        "output_sha256": {
            name: array_sha256(value) for name, value in first.items()
        },
        "repeated_local_replica_sha256": {
            name: list(value) for name, value in repeated_hashes.items()
        },
        "repeated_output_sha256": {
            name: array_sha256(value) for name, value in repeated.items()
        },
    }
