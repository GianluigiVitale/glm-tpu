"""Structural diff for the accepted and bounded layer-0 TPU graphs.

The accepted graph is large, so this module deliberately compares only the
live layer-0 boundary already localized by protected tensor evidence.  It
binds exact artifact hashes, follows the three physical collectives and the
two RMS reductions, and reports source, layout, fusion-ownership and stable
backend differences.  It never executes either model path.
"""

from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
import gzip
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..errors import BenchmarkValidationError
from ..sharding.hlo_contract import HloInstruction, parse_hlo_module
from .dense_rms_replay import _called_computation, _computation_base


ACCEPTED_LAYER0_HLO_SHA256 = (
    "3cd750810982608f9a3a7d557497c58f61159cc3dcdeb521f1377ba8c93fb775"
)
CANDIDATE_LAYER0_HLO_SHA256 = (
    "081d1b1f3609085a2b455357f8f6f9186c7bb3c5c218c1d10c734bea2b0163f8"
)

_STABLE_WINDOW_FIELDS = (
    "cost_model_type",
    "input_window_bounds",
    "is_mask",
    "iteration_bounds",
    "kernel_window_bounds",
    "output_window_bounds",
    "pad_input_on_minor_dim",
    "pad_output_on_minor_dim",
)
_STABLE_MEGACORE_FIELDS = (
    "megacore_allreduce_bytes",
    "megacore_split_dim",
)


def _read_hlo(path: Path, expected_sha256: str) -> str:
    if not path.is_file():
        raise BenchmarkValidationError(f"HLO artifact is absent: {path}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", errors="replace") as stream:
            value = stream.read()
    else:
        value = path.read_text()
    observed = sha256(value.encode()).hexdigest()
    if observed != expected_sha256:
        raise BenchmarkValidationError(
            f"HLO artifact SHA-256 drifted: {observed} != {expected_sha256}"
        )
    return value


def _entry_items(module: Any) -> dict[str, HloInstruction]:
    result: dict[str, HloInstruction] = {}
    for item in module.instructions:
        if item.computation.startswith("ENTRY "):
            result[item.name.lstrip("%")] = item
    return result


def _require(
    values: Mapping[str, HloInstruction], name: str
) -> HloInstruction:
    try:
        return values[name.lstrip("%")]
    except KeyError as error:
        raise BenchmarkValidationError(
            f"required live SSA value is absent: {name}"
        ) from error


def _result_prefix(value: HloInstruction) -> str:
    line = value.raw_line.lstrip()
    if line.startswith("ROOT "):
        line = line[len("ROOT ") :]
    marker = f"{value.name} = "
    if not line.startswith(marker):
        raise BenchmarkValidationError(
            f"cannot parse HLO result prefix for {value.name}"
        )
    remainder = line[len(marker) :]
    opcode_marker = f" {value.raw_opcode}("
    if opcode_marker not in remainder:
        raise BenchmarkValidationError(
            f"cannot parse HLO opcode boundary for {value.name}"
        )
    return remainder.split(opcode_marker, 1)[0]


def _backend(value: HloInstruction) -> dict[str, Any]:
    if "backend_config=" not in value.raw_line:
        return {}
    try:
        result = json.loads(value.raw_line.split("backend_config=", 1)[1])
    except json.JSONDecodeError as error:
        raise BenchmarkValidationError(
            f"backend config is not JSON for {value.name}"
        ) from error
    if not isinstance(result, dict):
        raise BenchmarkValidationError(
            f"backend config is not an object for {value.name}"
        )
    return result


def _backend_projection(value: HloInstruction) -> dict[str, Any]:
    backend = _backend(value)
    window = backend.get("window_config", {})
    megacore = backend.get("megacore_config", {})
    if not isinstance(window, dict) or not isinstance(megacore, dict):
        raise BenchmarkValidationError(
            f"backend schedule schema drifted for {value.name}"
        )
    return {
        "megacore": {
            key: megacore.get(key) for key in _STABLE_MEGACORE_FIELDS
        },
        "window": {key: window.get(key) for key in _STABLE_WINDOW_FIELDS},
    }


def _collective_projection(value: HloInstruction) -> dict[str, Any]:
    backend = _backend(value)
    return {
        "algorithm": backend.get("collective_algorithm_config"),
        "barrier": backend.get("barrier_config"),
        "channel_id": value.channel_id,
        "opcode": value.raw_opcode,
        "replica_groups": value.replica_groups,
        "result_prefix": _result_prefix(value),
        "use_global_device_ids": value.use_global_device_ids,
    }


def _classify_producer(value: HloInstruction) -> str:
    op_name = value.op_name or ""
    if "aten::embedding/jit(_take)/gather" in op_name:
        return "accepted_embedding_lookup"
    if "VllmRowParallelLinear/shard_map/dot_general" in op_name:
        return "accepted_row_parallel_projection"
    if "accepted_source_context_embedding_input" in op_name:
        return "external_m1_embedding_reconstruction"
    if "accepted_source_context_attention_input" in op_name:
        return "external_m1_attention_reconstruction"
    return "unclassified"


def _producer_record(
    values: Mapping[str, HloInstruction], collective: HloInstruction
) -> dict[str, Any]:
    if len(collective.operand_names) != 1:
        raise BenchmarkValidationError(
            f"collective operand cardinality drifted for {collective.name}"
        )
    producer = _require(values, collective.operand_names[0])
    return {
        "backend": _backend_projection(producer),
        "callee": _called_computation(producer),
        "mode": _classify_producer(producer),
        "name": producer.name,
        "op_name": producer.op_name,
        "opcode": producer.raw_opcode,
        "result_prefix": _result_prefix(producer),
    }


def _shortest_path(
    module: Any,
    values: Mapping[str, HloInstruction],
    source: str,
    target: str,
) -> tuple[str, ...]:
    source = source.lstrip("%")
    target = target.lstrip("%")
    computations = _computation_index(module)
    dependency_memo: dict[tuple[str, str, int | None], frozenset[int]] = {}
    live_memo: dict[tuple[str, str, int | None], frozenset[int]] = {}
    queue: deque[tuple[tuple[str, int | None], ...]] = deque(
        (((target, None),),)
    )
    seen = {(target, None)}
    while queue:
        path = queue.popleft()
        current_name, selected_tuple_index = path[-1]
        current = _require(values, current_name)
        if current_name == source:
            if selected_tuple_index is not None:
                raise BenchmarkValidationError(
                    f"non-tuple source {source} reached through tuple index"
                )
            return tuple(name for name, _ in reversed(path))
        if current.raw_opcode == "get-tuple-element":
            if selected_tuple_index is not None:
                raise BenchmarkValidationError(
                    f"nested tuple selection drifted for {current.name}"
                )
            tuple_index = _tuple_index(current)
            if tuple_index is None or len(current.operand_names) != 1:
                raise BenchmarkValidationError(
                    f"tuple selection is unresolved for {current.name}"
                )
            upstream = ((current.operand_names[0], tuple_index),)
        elif current.raw_opcode == "tuple":
            if selected_tuple_index is None:
                upstream = tuple(
                    (operand, None) for operand in current.operand_names
                )
            elif selected_tuple_index >= len(current.operand_names):
                raise BenchmarkValidationError(
                    f"tuple index drifted for {current.name}"
                )
            else:
                upstream = (
                    (current.operand_names[selected_tuple_index], None),
                )
        elif current.raw_opcode == "fusion":
            live_indices = _fusion_live_operand_indices(
                module,
                current,
                result_index=selected_tuple_index,
                computations=computations,
                dependency_memo=dependency_memo,
                live_memo=live_memo,
            )
            upstream = tuple(
                (current.operand_names[index], None)
                for index in sorted(live_indices)
            )
        elif selected_tuple_index is not None:
            raise BenchmarkValidationError(
                f"tuple index targets non-tuple value {current.name}"
            )
        else:
            upstream = tuple(
                (operand, None) for operand in current.operand_names
            )
        for operand, tuple_index in upstream:
            state = (operand.lstrip("%"), tuple_index)
            if state[0] in values and state not in seen:
                seen.add(state)
                queue.append(path + (state,))
    raise BenchmarkValidationError(
        f"no live SSA path from {source} to {target}"
    )


def _fusion_path_record(
    module: Any,
    values: Mapping[str, HloInstruction],
    source: str,
    target: str,
) -> dict[str, Any]:
    path = _shortest_path(module, values, source, target)
    return {
        "fusion_count": sum(
            values[name].raw_opcode == "fusion" for name in path[1:]
        ),
        "path": [
            {
                "name": values[name].name,
                "op_name": values[name].op_name,
                "opcode": values[name].raw_opcode,
                "result_prefix": _result_prefix(values[name]),
            }
            for name in path
        ],
    }


def _computation_index(module: Any) -> dict[str, dict[str, HloInstruction]]:
    result: dict[str, dict[str, HloInstruction]] = defaultdict(dict)
    for item in module.instructions:
        result[_computation_base(item.computation)][item.name.lstrip("%")] = item
    return dict(result)


def _parameter_index(value: HloInstruction) -> int | None:
    match = re.search(r"\bparameter\(([0-9]+)\)", value.raw_line)
    return None if match is None else int(match.group(1))


def _tuple_index(value: HloInstruction) -> int | None:
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', value.raw_line)
    line = re.sub(r"/\*.*?\*/", "", line)
    matches = re.findall(r"\bindex=([0-9]+)", line)
    return int(matches[0]) if len(matches) == 1 else None


def _fusion_live_operand_indices(
    module: Any,
    fusion: HloInstruction,
    *,
    result_index: int | None = None,
    computations: Mapping[str, Mapping[str, HloInstruction]] | None = None,
    dependency_memo: dict[
        tuple[str, str, int | None], frozenset[int]
    ] | None = None,
    live_memo: dict[
        tuple[str, str, int | None], frozenset[int]
    ] | None = None,
) -> frozenset[int]:
    """Return only caller operands that reach the selected fusion ROOT."""

    if computations is None:
        computations = _computation_index(module)
    if dependency_memo is None:
        dependency_memo = {}
    if live_memo is None:
        live_memo = {}
    active: set[tuple[str, str, int | None]] = set()

    def local_value(
        computation: str, name: str
    ) -> HloInstruction | None:
        base = _computation_base(computation)
        try:
            return computations[base][name.lstrip("%")]
        except KeyError as error:
            if name.lower() in {"nan", "inf", "-inf"} or re.fullmatch(
                r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
                r"(?:[eE][-+]?[0-9]+)?",
                name,
            ):
                return None
            raise BenchmarkValidationError(
                f"unresolved fusion value {base}/{name}"
            ) from error

    def live_operands(
        value: HloInstruction,
        selected_result_index: int | None = None,
    ) -> frozenset[int]:
        key = (
            _computation_base(value.computation),
            value.name.lstrip("%"),
            selected_result_index,
        )
        if key in live_memo:
            return live_memo[key]
        callee = _called_computation(value)
        if callee is None or callee not in computations:
            raise BenchmarkValidationError(
                f"fusion callee is unresolved for {value.name}"
            )
        roots = tuple(
            item
            for item in computations[callee].values()
            if item.raw_line.lstrip().startswith("ROOT ")
        )
        if len(roots) != 1:
            raise BenchmarkValidationError(
                f"fusion root cardinality drifted for {value.name}"
            )
        result = dependencies(
            callee,
            roots[0],
            selected_tuple_index=selected_result_index,
        )
        if any(index >= len(value.operand_names) for index in result):
            raise BenchmarkValidationError(
                f"fusion parameter binding drifted for {value.name}"
            )
        live_memo[key] = result
        return result

    def dependencies(
        computation: str,
        value: HloInstruction,
        *,
        selected_tuple_index: int | None = None,
    ) -> frozenset[int]:
        base = _computation_base(computation)
        key = (base, value.name.lstrip("%"), selected_tuple_index)
        if key in dependency_memo:
            return dependency_memo[key]
        if key in active:
            raise BenchmarkValidationError(
                f"cycle in fusion liveness graph at {base}/{value.name}"
            )
        active.add(key)
        try:
            if value.raw_opcode == "get-tuple-element":
                if selected_tuple_index is not None:
                    raise BenchmarkValidationError(
                        f"nested tuple selection drifted for {value.name}"
                    )
                index = _tuple_index(value)
                if index is None or len(value.operand_names) != 1:
                    raise BenchmarkValidationError(
                        f"tuple selection is unresolved for {value.name}"
                    )
                source = local_value(computation, value.operand_names[0])
                if source is None:
                    raise BenchmarkValidationError(
                        f"tuple source is unresolved for {value.name}"
                    )
                result = dependencies(
                    computation,
                    source,
                    selected_tuple_index=index,
                )
            elif value.raw_opcode == "tuple":
                if selected_tuple_index is None:
                    selected_operands = value.operand_names
                elif selected_tuple_index >= len(value.operand_names):
                    raise BenchmarkValidationError(
                        f"tuple index drifted for {value.name}"
                    )
                else:
                    selected_operands = (
                        value.operand_names[selected_tuple_index],
                    )
                indices: set[int] = set()
                for operand in selected_operands:
                    operand_value = local_value(computation, operand)
                    if operand_value is not None:
                        indices.update(
                            dependencies(computation, operand_value)
                        )
                result = frozenset(indices)
            elif selected_tuple_index is not None and value.raw_opcode != "fusion":
                raise BenchmarkValidationError(
                    f"tuple index targets non-tuple value {value.name}"
                )
            elif value.raw_opcode == "parameter":
                index = _parameter_index(value)
                if index is None:
                    raise BenchmarkValidationError(
                        f"fusion parameter index is absent for {value.name}"
                    )
                result = frozenset((index,))
            elif value.raw_opcode == "fusion":
                indices: set[int] = set()
                for operand_index in live_operands(
                    value, selected_tuple_index
                ):
                    operand_value = local_value(
                        computation,
                        value.operand_names[operand_index],
                    )
                    if operand_value is None:
                        continue
                    indices.update(
                        dependencies(
                            computation,
                            operand_value,
                        )
                    )
                result = frozenset(indices)
            else:
                indices = set()
                for operand in value.operand_names:
                    operand_value = local_value(computation, operand)
                    if operand_value is None:
                        continue
                    indices.update(
                        dependencies(
                            computation, operand_value
                        )
                    )
                result = frozenset(indices)
        finally:
            active.remove(key)
        dependency_memo[key] = result
        return result

    if fusion.raw_opcode != "fusion":
        raise BenchmarkValidationError(
            f"live fusion operand query received {fusion.raw_opcode}"
        )
    return live_operands(fusion, result_index)


def _attribute_projection(value: HloInstruction) -> dict[str, Any]:
    line = re.sub(r'"(?:\\.|[^"\\])*"', '""', value.raw_line)
    line = re.sub(r"/\*.*?\*/", "", line)
    result: dict[str, Any] = {}
    for field in ("dimensions", "padding", "slice", "direction"):
        match = re.search(rf"\b{field}=([^,}}]+(?:}})?)", line)
        if match is not None:
            result[field] = match.group(1)
    return result


def _semantic_signature(
    module: Any,
    entry_value: HloInstruction,
    semantic_roles: Mapping[str, str],
    *,
    include_fusion_boundaries: bool = False,
) -> str:
    """Hash the expanded arithmetic while ignoring SSA/callee names."""

    computations = _computation_index(module)
    memo: dict[tuple[str, str, tuple[str, ...]], Any] = {}

    def visit(
        computation: str,
        name: str,
        bindings: tuple[Any, ...] = (),
        active: frozenset[tuple[str, str]] = frozenset(),
    ) -> Any:
        name = name.lstrip("%")
        if computation.startswith("ENTRY ") and name in semantic_roles:
            return ("role", semantic_roles[name])
        base = _computation_base(computation)
        key = (base, name, tuple(repr(value) for value in bindings))
        if key in memo:
            return memo[key]
        if (base, name) in active:
            raise BenchmarkValidationError(
                f"cycle in live SSA graph at {base}/{name}"
            )
        try:
            value = computations[base][name]
        except KeyError as error:
            raise BenchmarkValidationError(
                f"unresolved live SSA value {base}/{name}"
            ) from error
        if value.raw_opcode == "parameter" and not computation.startswith("ENTRY "):
            index = _parameter_index(value)
            if index is None or index >= len(bindings):
                raise BenchmarkValidationError(
                    f"unbound fusion parameter {base}/{name}"
                )
            return bindings[index]
        next_active = active | {(base, name)}
        if value.raw_opcode == "constant":
            constant = re.search(r"\bconstant\(([^)]*)\)", value.raw_line)
            result = (
                "constant",
                _result_prefix(value),
                None if constant is None else constant.group(1),
            )
            memo[key] = result
            return result
        operands = tuple(
            visit(computation, operand, bindings, next_active)
            for operand in value.operand_names
        )
        if value.raw_opcode == "fusion":
            callee = _called_computation(value)
            if callee is None or callee not in computations:
                raise BenchmarkValidationError(
                    f"fusion callee is unresolved for {value.name}"
                )
            roots = tuple(
                item
                for item in computations[callee].values()
                if item.raw_line.lstrip().startswith("ROOT ")
            )
            if len(roots) != 1:
                raise BenchmarkValidationError(
                    f"fusion root cardinality drifted for {value.name}"
                )
            expanded = visit(callee, roots[0].name, operands, next_active)
            if include_fusion_boundaries:
                parameters = {
                    _parameter_index(item): item
                    for item in computations[callee].values()
                    if item.raw_opcode == "parameter"
                }
                if set(parameters) != set(range(len(operands))):
                    raise BenchmarkValidationError(
                        f"fusion parameter schema drifted for {value.name}"
                    )
                result = (
                    "fusion",
                    _result_prefix(value),
                    tuple(
                        (_result_prefix(parameters[index]), operands[index])
                        for index in range(len(operands))
                    ),
                    expanded,
                )
            else:
                result = expanded
        else:
            result = (
                value.raw_opcode,
                _result_prefix(value),
                _attribute_projection(value),
                operands,
            )
        memo[key] = result
        return result

    signature = visit(entry_value.computation, entry_value.name)
    payload = json.dumps(signature, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


def compare_layer0_live_ssa(
    accepted_hlo: str, candidate_hlo: str
) -> dict[str, Any]:
    accepted_module = parse_hlo_module(accepted_hlo)
    candidate_module = parse_hlo_module(candidate_hlo)
    accepted = _entry_items(accepted_module)
    candidate = _entry_items(candidate_module)

    anchors = {
        "accepted": {
            "embedding": _require(accepted, "all-reduce.3"),
            "attention": _require(accepted, "psum.1100"),
            "dense": _require(accepted, "psum.1101"),
            "predense_reduction": _require(
                accepted, "multiply_reduce_fusion.317"
            ),
            "predense_gate": _require(accepted, "fusion.7778"),
            "layer1_reduction": _require(
                accepted, "multiply_reduce_fusion.316"
            ),
            "layer1_output": _require(accepted, "fusion.6342"),
            "predense_validity": _require(
                accepted, "get-tuple-element.44327"
            ),
            "layer1_validity": _require(
                accepted, "get-tuple-element.44327"
            ),
        },
        "candidate": {
            "embedding": _require(candidate, "psum.21"),
            "attention": _require(candidate, "psum.22"),
            "dense": _require(candidate, "psum.23"),
            "predense_reduction": _require(
                candidate, "multiply_reduce_fusion.1"
            ),
            "predense_gate": _require(candidate, "fusion.21"),
            "layer1_reduction": _require(candidate, "multiply_reduce_fusion"),
            "layer1_output": _require(
                candidate, "multiply_bitcast-convert_fusion"
            ),
            "predense_validity": _require(candidate, "param.11"),
            "layer1_validity": _require(candidate, "copy-done.5"),
        },
    }

    result: dict[str, Any] = {
        "artifact_kind": "glm52_layer0_live_ssa_diff",
        "diagnostic_only": True,
        "performance_claim": False,
    }
    collective_records: dict[str, Any] = {}
    differences: list[dict[str, Any]] = []
    for label in ("embedding", "attention", "dense"):
        accepted_collective = anchors["accepted"][label]
        candidate_collective = anchors["candidate"][label]
        accepted_projection = _collective_projection(accepted_collective)
        candidate_projection = _collective_projection(candidate_collective)
        accepted_producer = _producer_record(accepted, accepted_collective)
        candidate_producer = _producer_record(candidate, candidate_collective)
        collective_records[label] = {
            "accepted": {
                "collective": accepted_projection,
                "producer": accepted_producer,
            },
            "candidate": {
                "collective": candidate_projection,
                "producer": candidate_producer,
            },
            "algorithm_result_match": (
                accepted_projection["algorithm"]
                == candidate_projection["algorithm"]
                and accepted_projection["opcode"]
                == candidate_projection["opcode"]
                and accepted_projection["replica_groups"]
                == candidate_projection["replica_groups"]
                and accepted_projection["result_prefix"]
                == candidate_projection["result_prefix"]
            ),
        }
        if accepted_producer["mode"] != candidate_producer["mode"]:
            differences.append(
                {
                    "boundary": f"{label}_collective_input",
                    "kind": "source_producer",
                    "accepted": accepted_producer,
                    "candidate": candidate_producer,
                }
            )
        if accepted_projection["barrier"] != candidate_projection["barrier"]:
            differences.append(
                {
                    "boundary": f"{label}_collective",
                    "kind": "barrier_config",
                    "accepted": accepted_projection["barrier"],
                    "candidate": candidate_projection["barrier"],
                }
            )
    result["collectives"] = collective_records

    semantic_pairs = {
        "predense_reduction": (
            "attention",
            "embedding",
            "predense_validity",
        ),
        "layer1_reduction": (
            "dense",
            "attention",
            "embedding",
            "layer1_validity",
        ),
    }
    semantic_records: dict[str, Any] = {}
    for label, roles in semantic_pairs.items():
        accepted_value = anchors["accepted"][label]
        candidate_value = anchors["candidate"][label]
        accepted_role_map = {
            anchors["accepted"][role].name.lstrip("%"): (
                "validity" if role.endswith("validity") else role
            )
            for role in roles
        }
        candidate_role_map = {
            anchors["candidate"][role].name.lstrip("%"): (
                "validity" if role.endswith("validity") else role
            )
            for role in roles
        }
        if set(name.lstrip("%") for name in accepted_value.operand_names) != set(
            accepted_role_map
        ) or set(name.lstrip("%") for name in candidate_value.operand_names) != set(
            candidate_role_map
        ):
            raise BenchmarkValidationError(
                f"{label} semantic source identity drifted"
            )
        accepted_signature = _semantic_signature(
            accepted_module,
            accepted_value,
            accepted_role_map,
        )
        candidate_signature = _semantic_signature(
            candidate_module,
            candidate_value,
            candidate_role_map,
        )
        accepted_physical_signature = _semantic_signature(
            accepted_module,
            accepted_value,
            accepted_role_map,
            include_fusion_boundaries=True,
        )
        candidate_physical_signature = _semantic_signature(
            candidate_module,
            candidate_value,
            candidate_role_map,
            include_fusion_boundaries=True,
        )
        semantic_records[label] = {
            "accepted_sha256": accepted_signature,
            "accepted_physical_sha256": accepted_physical_signature,
            "backend_match": (
                _backend_projection(accepted_value)
                == _backend_projection(candidate_value)
            ),
            "candidate_sha256": candidate_signature,
            "candidate_physical_sha256": candidate_physical_signature,
            "physical_match": (
                accepted_physical_signature == candidate_physical_signature
            ),
            "semantic_match": accepted_signature == candidate_signature,
        }
        if accepted_signature != candidate_signature:
            differences.append(
                {
                    "boundary": label,
                    "kind": "expanded_arithmetic",
                    "accepted": accepted_signature,
                    "candidate": candidate_signature,
                }
            )
        if accepted_physical_signature != candidate_physical_signature:
            differences.append(
                {
                    "boundary": label,
                    "kind": "fusion_boundary_layout",
                    "accepted": accepted_physical_signature,
                    "candidate": candidate_physical_signature,
                }
            )
    result["semantic_reductions"] = semantic_records

    schedules: dict[str, Any] = {}
    for label in ("predense_gate",):
        accepted_schedule = _backend_projection(anchors["accepted"][label])
        candidate_schedule = _backend_projection(anchors["candidate"][label])
        schedules[label] = {
            "accepted": accepted_schedule,
            "candidate": candidate_schedule,
            "match": accepted_schedule == candidate_schedule,
        }
        if accepted_schedule != candidate_schedule:
            differences.append(
                {
                    "boundary": label,
                    "kind": "backend_geometry",
                    "accepted": accepted_schedule,
                    "candidate": candidate_schedule,
                }
            )
    accepted_dense_producer = _require(
        accepted, anchors["accepted"]["dense"].operand_names[0]
    )
    candidate_dense_producer = _require(
        candidate, anchors["candidate"]["dense"].operand_names[0]
    )
    accepted_dense_schedule = _backend_projection(accepted_dense_producer)
    candidate_dense_schedule = _backend_projection(candidate_dense_producer)
    schedules["dense_projection"] = {
        "accepted": accepted_dense_schedule,
        "candidate": candidate_dense_schedule,
        "match": accepted_dense_schedule == candidate_dense_schedule,
    }
    if accepted_dense_schedule != candidate_dense_schedule:
        differences.append(
            {
                "boundary": "dense_projection",
                "kind": "backend_geometry",
                "accepted": accepted_dense_schedule,
                "candidate": candidate_dense_schedule,
            }
        )
    result["scheduled_geometry"] = schedules

    accepted_paths = {
        label: _fusion_path_record(
            accepted_module,
            accepted,
            anchors["accepted"][label].name,
            anchors["accepted"]["layer1_output"].name,
        )
        for label in ("embedding", "attention", "dense")
    }
    candidate_paths = {
        label: _fusion_path_record(
            candidate_module,
            candidate,
            anchors["candidate"][label].name,
            anchors["candidate"]["layer1_output"].name,
        )
        for label in ("embedding", "attention", "dense")
    }
    result["layer1_output_paths"] = {
        "accepted": accepted_paths,
        "candidate": candidate_paths,
    }
    for label in ("embedding", "attention", "dense"):
        if (
            accepted_paths[label]["fusion_count"]
            != candidate_paths[label]["fusion_count"]
        ):
            differences.append(
                {
                    "boundary": "layer1_output",
                    "kind": "fusion_ownership",
                    "source": label,
                    "accepted": accepted_paths[label],
                    "candidate": candidate_paths[label],
                }
            )

    result["differences"] = differences
    result["first_divergence"] = differences[0] if differences else None
    result["status"] = "DIFF_IDENTIFIED" if differences else "NO_DIFF"
    return result


def compare_layer0_live_ssa_files(
    accepted_hlo_path: Path,
    candidate_hlo_path: Path,
    *,
    accepted_sha256: str = ACCEPTED_LAYER0_HLO_SHA256,
    candidate_sha256: str = CANDIDATE_LAYER0_HLO_SHA256,
) -> dict[str, Any]:
    return compare_layer0_live_ssa(
        _read_hlo(accepted_hlo_path, accepted_sha256),
        _read_hlo(candidate_hlo_path, candidate_sha256),
    )


def write_layer0_live_ssa_report(
    output: Path, report: Mapping[str, Any]
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(payload)
    temporary.replace(output)
