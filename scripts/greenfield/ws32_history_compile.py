"""Seven history diagnostic compiler jobs from authenticated metadata only.

This adapter supplies the existing compile-only worker and original-file
validator. It never loads checkpoint payloads or invokes a compiled program;
actual optimized-HLO admission and numerical execution remain separate gates.
"""

from __future__ import annotations

from hashlib import sha256
import json
from math import prod
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Mapping

from scripts.greenfield import ws32_canonical_prefill_compile as canonical
from scripts.greenfield import ws32_history_frontier_prepare as frontier
from scripts.greenfield import ws32_history_observer_prepare as observer
from scripts.greenfield import ws32_history_protocol as protocol
from scripts.greenfield.ws32_rolled_prefill_compile import AbstractPrefillPair

KERNEL = "ws32_history_frontier_compile"
PROTOCOL = "ws32-history-l06-metadata-compile-only-v1"
PROFILE = "ws32_history_l06_abstract_compile_v1"
PROGRAMS = (*protocol.FRONTIER_PROGRAMS, "exact_decode", "exact_promote", "observer")
NOTE = (
    "Seven layers0..6 history/observer/materializer graphs compiled from verified "
    "metadata and abstract inputs only. No weight payload loading, WK/model "
    "dispatch, numerical correctness, runtime peak HBM, original-event "
    "reproduction, token11 fix, throughput or TTFT claim. Actual optimized HLO "
    "and numerical admission remain separate."
)
ABSTRACT_IO_BYTES = {
    "candidate_b128": (1449596441, 27978625),
    "candidate_b114": (1449596385, 26151359),
    "control_b128": (1449596441, 27978625),
    "control_b114": (1449596385, 26151359),
    "exact_decode": (21156992, 106741760),
    "exact_promote": (106741760, 213696512),
    "observer": (1694113757, 130536),
}
SOURCE_SHA256 = {
    "scripts/greenfield/ws32_history_frontier.py": "a1b8899569233ff3b5543ba4f38963313ecaaa45cbebd1e19824b7a485da3bc5",
    "scripts/greenfield/ws32_history_frontier_prepare.py": "2e27e5ee592997454b304b682b33a61a93b0c961673178d2716c83daca12c646",
    "scripts/greenfield/ws32_history_observer.py": "cf13565a7fbf02e1343e9f72b4de3b2a1ad47feef887a6d08e9438a19e963256",
    "scripts/greenfield/ws32_history_observer_prepare.py": "d2dea3ea5d41736084c84280975b8f6a1e00d69a0b2b858975aa12ae3b80cf5a",
}


def require_source(repo: Path) -> None:
    """Bind the four diagnostic builders and the existing frozen model recipe."""
    canonical.require_source(repo)
    for name, digest in SOURCE_SHA256.items():
        path = repo / name
        if path.is_symlink() or not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("history compiler diagnostic source differs")


def read_metadata(repo: Path) -> Any:
    """Verify both the raw checkpoint and original overlay before runtime setup."""
    from glm_tpu.greenfield.checkpoint.ws32_strategy_nd_dense import verify_ws32_strategy_nd_dense_overlay

    _require_raw_pins()
    require_source(repo)
    metadata = canonical.read_metadata(repo)
    env = json.loads((repo / "configs/greenfield-ws32-batched-acquisition.json").read_text())["environment"]
    prefix = "GLM_GREENFIELD_WS32_STRATEGY_ND_DENSE_OVERLAY_"
    verify_ws32_strategy_nd_dense_overlay(Path(env[prefix + "ROOT"]),
        expected_manifest_sha256=env[prefix + "MANIFEST_SHA"],
        expected_manifest_file_sha256=env[prefix + "MANIFEST_FILE_SHA"],
        expected_success_file_sha256=env[prefix + "SUCCESS_FILE_SHA"])
    return metadata


def _require_raw_pins() -> None:
    if set(RAW) != set(PROGRAMS):
        raise ValueError("history compiler requires seven preregistered raw graphs")
    for pins in RAW.values():
        if (type(pins) is not tuple or len(pins) != 2 or type(pins[0]) is not int or pins[0] <= 0
                or not isinstance(pins[1], str) or re.fullmatch(r"[0-9a-f]{64}", pins[1]) is None):
            raise ValueError("history compiler raw preregistration size/digest invalid")


def _output_specs(config: Any) -> dict[str, Any]:
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.runtime.ws32_decoder import (
        ws32_decoded_exact_dsa_specs, ws32_dsa_observation_specs, ws32_exact_dsa_specs,
    )
    from scripts.greenfield.ws32_history_frontier import HistoryCaches, HistoryFrontier, LayerBoundary, ProducerRows
    from scripts.greenfield.ws32_history_observer import FirstObservation

    boundary = LayerBoundary(P(None, "feature"), P(None, "feature"), P(None, "feature"),
                             P(), P(), P("expert", "feature", None))
    caches = HistoryCaches((P("expert", None, None, None),) * 7,
                           (P("expert", None, None, None),) * 4,
                           (P("expert", None, None, None),) * 4)
    history = HistoryFrontier(caches, (boundary,) * 7, (ProducerRows(P(), P(), P()),) * 4, P())
    return {**{name: history for name in protocol.FRONTIER_PROGRAMS},
            "exact_decode": ws32_decoded_exact_dsa_specs(config),
            "exact_promote": ws32_exact_dsa_specs(config),
            "observer": FirstObservation((boundary,) * 7, ws32_dsa_observation_specs(), P())}


def prepare(mesh: Any, metadata: Any, *, repo: Path) -> AbstractPrefillPair:
    """Reuse the two preparations, restoring declared exact-operand sharding."""
    import jax
    from jax.sharding import NamedSharding

    require_source(repo)
    raw = frontier.prepare(mesh, repo=repo)
    if raw.manifest_sha256 != metadata.manifest["manifest_sha256"]:
        raise ValueError("history compiler metadata changed during preparation")
    observed = observer.prepare(mesh, raw=raw, repo=repo)
    if (len(raw.tensor_names) != protocol.SELECTED_LEAVES
            or raw.payload_bytes_per_chip != protocol.PAYLOAD_BYTES
            or len(observed.overlay_tensor_names) != protocol.OVERLAY_TENSORS
            or observed.overlay_bytes_per_chip != protocol.OVERLAY_BYTES):
        raise ValueError("history compiler selected/overlay geometry differs")
    specs = _output_specs(observed.config)

    def placed_shapes(values: Any, placements: Any) -> Any:
        # eval_shape deliberately strips sharding. In particular the query and
        # head operands stay feature-sharded; treating them as P() overcounts.
        return jax.tree.map(lambda value, spec: jax.ShapeDtypeStruct(
            value.shape, value.dtype, sharding=NamedSharding(mesh, spec)), values, placements)

    decoded = placed_shapes(observed.decoded_exact, specs["exact_decode"])
    exact = placed_shapes(observed.exact, specs["exact_promote"])
    remap = dict(candidate_b128="canonical_b128", candidate_b114="canonical_b114",
                 control_b128="live32_b128", control_b114="live32_b114")
    functions = {name: raw.programs[source] for name, source in remap.items()}
    functions.update(exact_decode=jax.jit(observed.materializer.decode),
                     exact_promote=jax.jit(observed.materializer.promote), observer=observed.program)
    inputs = {name: raw.inputs[source] for name, source in remap.items()}
    inputs.update(exact_decode=(observed.raw_exact,), exact_promote=(decoded,),
                  observer=(*observed.inputs[:8], exact, *observed.inputs[9:]))
    if any(not isinstance(value, jax.ShapeDtypeStruct) or value.sharding is None
           for value in jax.tree.leaves(inputs)):
        raise ValueError("history compiler requires explicitly sharded abstract inputs")
    return AbstractPrefillPair(
        {name: SimpleNamespace(execute=functions[name], output_specs=specs[name]) for name in PROGRAMS},
        inputs, raw.manifest_sha256, metadata.manifest["source"]["inventory_sha256"])


def abstract_io_bytes(prepared: AbstractPrefillPair, mesh: Any) -> dict[str, tuple[int, int]]:
    """Per-chip logical IO sizes, counting every declared output buffer once.

    These are metadata accounting, not compiler allocations or measured HBM.
    Explicit output specs prevent replicated dimensions or Python aliases from
    accidentally changing the accounting after eval_shape removes placements.
    """
    import jax
    from jax.sharding import NamedSharding

    sizes = {}
    for name in PROGRAMS:
        inputs, program = prepared.inputs[name], prepared.programs[name]
        argument_bytes = sum(prod(value.sharding.shard_shape(value.shape)) * value.dtype.itemsize
                             for value in jax.tree.leaves(inputs))
        outputs = jax.eval_shape(program.execute, *inputs)
        output_bytes = jax.tree.map(lambda value, spec:
            prod(NamedSharding(mesh, spec).shard_shape(value.shape)) * value.dtype.itemsize,
            outputs, program.output_specs)
        sizes[name] = (argument_bytes, sum(jax.tree.leaves(output_bytes)))
    return sizes


def memory_caps(name: str) -> dict[str, int]:
    if name not in PROGRAMS:
        raise ValueError("history compiler unregistered memory program")
    return protocol.memory_caps(name)


def validate_preserved_pair(root: Path, record: Mapping[str, Any], *, repo: Path) -> dict[str, Any]:
    """Bind ALL seven originals before testing any compiler allocation cap."""
    require_source(repo)
    _require_raw_pins()
    expected_identity = dict(kernel=KERNEL, protocol=PROTOCOL, profile=PROFILE,
        compile_only=True, weights_loaded=False, model_executable_calls=0,
        numerical_claim=False, performance_claim=False)
    if any(type(record.get(key)) is not type(value) or record.get(key) != value
           for key, value in expected_identity.items()):
        raise ValueError("history compiler-only identity or no-dispatch declaration differs")
    if set(record.get("programs", {})) != set(PROGRAMS):
        raise ValueError("history compiler requires seven preregistered/preserved graphs")
    graphs = {}
    for name in PROGRAMS:
        saved = record["programs"][name]
        stable_path, optimized_path = root / f"{name}.stablehlo.mlir", root / f"{name}.optimized_hlo.txt"
        if stable_path.is_symlink() or optimized_path.is_symlink():
            raise ValueError("history compiler original must not be a symlink")
        stable, optimized = stable_path.read_bytes(), optimized_path.read_bytes()
        size, digest = RAW[name]
        if (type(size) is not int or size <= 0 or len(stable) != size
                or sha256(stable).hexdigest() != digest or saved["stablehlo_sha256"] != digest
                or not optimized or sha256(optimized).hexdigest() != saved["optimized_hlo_sha256"]):
            raise ValueError("history compiler original graph identity differs")
        graphs[name] = dict(stablehlo_sha256=digest, optimized_hlo_sha256=sha256(optimized).hexdigest(),
                            memory=saved.get("compiled_memory"))
    # Compilation/preservation belongs to the existing worker. No cap failure
    # here can prevent later jobs' originals from having been captured already.
    for name, graph in graphs.items():
        caps, memory = memory_caps(name), graph["memory"]
        if (not isinstance(memory, Mapping) or set(memory) != set(caps)
                or any(type(value) is not int or not 0 <= value <= caps[key]
                       for key, value in memory.items())):
            raise ValueError(f"history compiler allocation inventory/caps differ: {name}")
        graph["memory"] = dict(memory)
    return dict(graphs=graphs, dispatch_evidence="REQUIRES_REVIEWED_COMPILE_ONLY_WORKER_AND_JOURNAL",
        numerical_claim=False, performance_claim=False,
        scope="ACTUAL_COMPILER_EVIDENCE_NOT_NUMERICAL_HBM_OR_ADMISSION")


# Actual CPU32 TPU-v4-target lowering, 2026-09-11: trace(*inputs).lower with
# lowering_platforms=('tpu',) and get_ir_version=None, as in the existing
# rolled-compiler raw-pin test. All seven full byte strings matched after fresh
# preparation and cache clearing from a second caller frame. Total10,330,420B;
# no TPU compile or dispatch. Pins follow builders so this table moves no trace
# source locations. Missing or malformed pins refuse before runtime setup.
RAW: dict[str, tuple[int, str]] = {
    "candidate_b128": (2044673, "9d87be894b8a927ef89aa7fc849b7d9b8587876b8edb54c7ab9243a03df7859e"),
    "candidate_b114": (2054137, "0a23a18e379aa3d10c2a72e3d5648ec18117bd5069cfae3f6b78c4aa02b96e4a"),
    "control_b128": (2024395, "2c22426e62c3ea62fd0eccb2d77f9540b18da08ed7b3650804876faf2d6eac7a"),
    "control_b114": (2035180, "a99d8182d5a0f4a6983ab17ea5c3480986e2ebd8b7732ea624ec971342937377"),
    "exact_decode": (96223, "69b40a8899edd5314a2be43fb91031dc094bdd6bd24626e4643fefefa36edf20"),
    "exact_promote": (9920, "f52d9d4dac9f79bd8fb2e4cc3b814804e38e3252b04586d913ee495850fa2251"),
    "observer": (2065892, "f92781a967ef5aafa7fb630a95b7a51bbbdc9a98e0fc96948717b28c39648033"),
}
