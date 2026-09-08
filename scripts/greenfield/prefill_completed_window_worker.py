"""Staged numerical cases on completed prefixes; no launch/compile entry point.

The existing campaign must first bind all five DB593 graphs, completed WK,
runtime owners, journal and actual live memory. No timing promotion here.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from scripts.greenfield import prefill_completed_window_admission as admission
from scripts.greenfield import prefill_completed_window_assembly as assembly
from scripts.greenfield import prefill_completed_window_protocol as protocol
from scripts.greenfield import prefill_window_protocol as window
from scripts.greenfield.prefill_layer_evidence import encode_arrays, local_observations
from scripts.greenfield.prefill_window_worker import BudgetedCalls, save_arrays
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal


class CompletedJournal(Ws32NumericalJournal):
    artifact_kind = "greenfield_ws32_completed_window_numerical_journal_v1"

    def _check_identity(self, identity: Mapping[str, Any]) -> None:
        if (
            identity.get("protocol") != protocol.PROTOCOL
            or identity.get("profile") != admission.PROFILE
            or identity.get("compile_only") is not False
        ):
            raise ValueError("completed window requires distinct numerical journal")


def execute_window(
    calls: BudgetedCalls,
    *,
    case: str,
    values: tuple,
    tiles: tuple,
    capture: Any,
    preserve_assembly: Any,
) -> tuple:
    """Shared device traversal: four completed prefixes, both suffix paths, assembly.

    Every invocation starts from values (immutable initial caches); results never
    carry into a later traversal. Callbacks run after completion, outside timing.
    """
    import jax

    prefixes = []
    for tile in range(4):
        prepared = calls.call(
            case + f"/prepare_prefix{tile}",
            "prepare_prefix",
            assembly.prefix_arguments(values, tiles[tile]),
            preserve=lambda result: None,
        )
        inputs = calls.phase(
            case + f"/prefix{tile}_inputs",
            lambda: assembly.attach_prefix(
                values, prepared, prefixes[-1] if prefixes else None
            ),
        )
        calls.phase(
            case + f"/prefix{tile}_ready", lambda: jax.block_until_ready(inputs)
        )
        result = calls.call(
            case + f"/prefix{tile}",
            "prefix",
            inputs,
            preserve=lambda result: capture(
                f"prefix{tile}", result, protocol.PREFIX_FIELDS
            ),
        )
        prefixes.append(result)

    prepared = calls.call(
        case + "/prepare_wide",
        "prepare_wide",
        assembly.suffix_arguments(tuple(prefixes), values[9]),
        preserve=lambda result: None,
    )
    inputs = calls.phase(
        case + "/wide_inputs",
        lambda: assembly.attach_suffix(values, prepared),
    )
    calls.phase(case + "/wide_ready", lambda: jax.block_until_ready(inputs))
    wide = calls.call(
        case + "/wide",
        "candidate",
        inputs,
        preserve=lambda result: capture("wide", result, protocol.SUFFIX_FIELDS),
    )
    narrow = []
    for tile in range(4):
        prepared = calls.call(
            case + f"/prepare_narrow{tile}",
            "prepare_narrow",
            assembly.suffix_arguments((prefixes[tile],), values[9], tiles[tile]),
            preserve=lambda result: None,
        )
        inputs = calls.phase(
            case + f"/narrow{tile}_inputs",
            lambda: assembly.attach_suffix(values, prepared),
        )
        calls.phase(
            case + f"/narrow{tile}_ready", lambda: jax.block_until_ready(inputs)
        )
        result = calls.call(
            case + f"/narrow{tile}",
            "control",
            inputs,
            preserve=lambda result: capture(
                f"narrow{tile}", result, protocol.SUFFIX_FIELDS
            ),
        )
        narrow.append(result)

    return calls.call(
        case + "/assemble",
        "assemble",
        assembly.assembly_arguments(tuple(prefixes), wide, tuple(narrow)),
        preserve=lambda result: preserve_assembly(result, prefixes[-1]),
    )


def execute_cases(
    calls: BudgetedCalls, *, weights: Any, wk: Any, mesh: Any, specs: tuple[Any, ...]
) -> None:
    """Three fixed cases:4 prefixes +1 wide/4 narrow suffixes each (27 calls).

    The separate two-WK preparation is the caller's prerequisite. Original device
    outputs are retained before health/replay checks; assembly completes before
    the next per-call all-live census. No host arrays feed any model executable.
    """
    import jax
    from glm_tpu.greenfield.kernels.reference.rotary import build_rotary_table_host
    from scripts.greenfield.probe_ws32_prefill_layer import device_inputs

    def bind():
        if (
            set(calls.programs) != set((*admission.PROGRAMS, *assembly.PROGRAMS))
            or calls.budgeter is not assembly.memory_budget
            or not isinstance(calls.journal, CompletedJournal)
        ):
            raise ValueError(
                "completed cases require nine-graph memory/journal binding"
            )
        if (
            calls.record.get("protocol") != protocol.PROTOCOL
            or calls.record.get("profile") != admission.PROFILE
            or calls.record.get("compile_only") is not False
        ):
            raise ValueError("completed cases require distinct numerical profile")
        calls.record.update(
            reference_scope=protocol.REFERENCE_SCOPE,
            independent_full_layer_admission=False,
            performance_claim=False,
        )

    calls.phase("completed_case_bind", bind)
    tiles = calls.phase("assembly_tiles", lambda: assembly.place_tiles(mesh))
    calls.phase("assembly_tiles_ready", lambda: jax.block_until_ready(tiles))
    rope = calls.phase(
        "rotary_fixture",
        lambda: build_rotary_table_host(window.CAPACITY, rotary_dim=64, theta=8e6),
    )
    for case in protocol.CASES:
        host = calls.phase(case + "/host", lambda: window.host_case(case, rope))
        values = calls.phase(
            case + "/inputs", lambda: device_inputs(host, specs, weights, wk, mesh)
        )
        calls.phase(case + "/inputs_ready", lambda: jax.block_until_ready(values))
        path = calls.root / f"{case}.npz"
        arrays = encode_arrays("input", host)
        item: dict[str, Any] = dict(complete=False)
        calls.record["cases"][case] = item
        calls.phase(case + "/input_capture", lambda: save_arrays(path, arrays))

        def capture(kind, result, fields):
            observed = protocol.observe(result, fields)
            for device, parts in observed.items():
                arrays.update(protocol.encode(f"{kind}_{device}", parts))
            item["npz_sha256"] = save_arrays(path, arrays)
            if set(observed) != set(calls.local_slots):
                raise ValueError("completed output owner differs; originals preserved")
            if not all(p["health"].all() for p in observed.values()):
                raise ValueError("completed output health failed; originals preserved")
            # Scores use intentional -inf sentinels, all BF16 operands must be finite.
            if any(
                not np.isfinite(v).all()
                for p in observed.values()
                for n, v in p.items()
                if n in protocol.BF16_FIELDS or n == "route_weights"
            ):
                raise ValueError("completed nonfinite operand; originals preserved")

        def preserve_assembly(result, last_prefix):
            for kind, rows in zip(("actual", "control"), result, strict=True):
                assembled = assembly.attach_result(rows, last_prefix)
                observed = local_observations(assembled)
                for device, fields in observed.items():
                    arrays.update(protocol.encode(f"{kind}_{device}", fields))
                item["npz_sha256"] = save_arrays(path, arrays)
                if set(observed) != set(calls.local_slots):
                    raise ValueError(
                        "completed assembly owner differs; originals preserved"
                    )

        assembled = execute_window(
            calls,
            case=case,
            values=values,
            tiles=tiles,
            capture=capture,
            preserve_assembly=preserve_assembly,
        )

        def finish():
            replay = protocol.replay_case(
                path, case=case, slots_by_device=calls.local_slots
            )
            item.update(replay=replay, passed=replay["passed"], complete=True)
            if not replay["passed"]:
                raise ValueError("completed suffix numerical comparison failed")

        calls.phase(case + "/comparison", finish)
        del (
            values,
            assembled,
            arrays,
            host,
        )
