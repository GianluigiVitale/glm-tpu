"""Selected dense01 compiler/WK/model continuation, not a launcher or sealer.

The existing parent binds actual runtime owners and checked selected weights.
All three compiler originals survive an admission refusal. One BudgetedCalls
instance owns four per-layer WK calls followed by the existing five model calls.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from scripts.greenfield import ws32_dense_frontier_protocol as protocol
from scripts.greenfield import ws32_dense_frontier_worker as worker
from scripts.greenfield.prefill_window_acquisition import fleet_step
from scripts.greenfield.prefill_window_worker import BudgetedCalls, save_arrays
from scripts.greenfield.ws32_acquisition_journal import Ws32NumericalJournal

ORACLE_ROOT = Path("/home/gianl/gcs-models/oracles/greenfield/glm52")
# DB604 deliberately contains no DSA-oracle fields. These are the existing
# original8K oracle pins, verified against the SHA-bound refusal runner, not a
# new reference or a reconstruction from this diagnostic's outputs.
DSA_PIN_SOURCE_SHA = "83efb10c90c144457e1ae3ea930f416d33dd5d3c762f5e9fcd454944f9199e5e"
DSA_PINS = dict(
    expected_dsa_manifest_sha256="f8154c5f79b909efd9ebc14c8e004925482844d05ef28fcf0a4d29bb4a7b26da",
    expected_dsa_success_sha256="0b798974ae8a9f95c32d3aa2eff532213624f1e2ae7f1de809a161e18dbdf1b9",
)
WK_ORIGINALS_LIMIT = 96 << 20  # Four complete boundary/source capsules per host.


def host_inputs(prior: Mapping[str, Any], config: Any) -> tuple[np.ndarray, np.ndarray]:
    """Reuse complete oracle validation and original host rotary construction."""
    from glm_tpu.greenfield.validation.ws32_short_context import load_ws32_short_context_oracle
    from glm_tpu.greenfield.runtime.ws32_decoder import build_ws32_main_rope_table
    from scripts.greenfield.ws32_prefill_frontier_state import require_config

    require_config(config)
    oracle = load_ws32_short_context_oracle(
        ORACLE_ROOT / "short_context/8k/greenfield_short_context_oracle_8k_20260807T172307269147351Z/oracle",
        ORACLE_ROOT / "short_context_dsa/8k/greenfield_short_context_dsa_oracle_8k_recovery_20260807T174904381704076Z/oracle",
        **{f"expected_token_{suffix}": prior[f"token_oracle_{suffix}"]
           for suffix in ("manifest_sha256", "success_sha256")}, **DSA_PINS,
    )
    tokens = oracle.prompt_token_ids
    rope = np.asarray(build_ws32_main_rope_table(config))
    if (tokens.dtype != np.int32 or tokens.shape != (8155,)
            or sha256(tokens.tobytes()).hexdigest() != worker.PROMPT_SHA
            or prior["prompt_ids_sha256"] != worker.PROMPT_SHA
            or rope.shape != (8192, 64) or str(rope.dtype) != "bfloat16"
            or rope.nbytes != prior["main_rope_table"]["bytes_per_device"]
            or sha256(rope.tobytes()).hexdigest() != prior["main_rope_table"]["sha256"]):
        raise ValueError("dense original prompt/RoPE bytes differ")
    return tokens, rope


class DenseJournal(Ws32NumericalJournal):
    artifact_kind = "greenfield_ws32_dense01_numerical_journal_v1"
    protocol_id = protocol.PROTOCOL

    def _check_identity(self, identity: Mapping[str, Any]) -> None:
        if (identity.get("protocol") != self.protocol_id
                or identity.get("compile_only") is not False
                or identity.get("diagnostic_only") is not True):
            raise ValueError("dense journal requires diagnostic identity")


class NormJournal(DenseJournal):
    from scripts.greenfield.ws32_dense_norm_protocol import PROTOCOL as protocol_id
    artifact_kind = "greenfield_ws32_dense01_norm_numerical_journal_v1"


class CanonicalJournal(DenseJournal):
    from scripts.greenfield.ws32_dense_canonical import PROTOCOL as protocol_id
    artifact_kind = "greenfield_ws32_dense01_canonical_numerical_journal_v1"


def capture_wk(root: Path, record: dict, local_slots: Mapping[int, int],
               *, layer: int, name: str, value: Any, inputs: tuple = ()) -> None:
    """Retain checked source operands and each completed boundary before refusal.

    Full FP8-to-BF16 reconstruction needs feature shards from other hosts and is
    an independent fleet collector duty; finiteness/owner/dtype checks run here.
    """
    import jax.numpy as jnp
    from scripts.greenfield.ws32_prefill_frontier_state import _index_key
    expected_dtype = jnp.bfloat16 if name == "wk_decode" else jnp.float32
    arrays = {}
    invalid = False
    if (type(layer) is not int or layer not in protocol.LAYERS
            or name not in worker.PROGRAMS[:2]
            or len(inputs) != (2 if name == "wk_decode" else 1)
            or len(local_slots) != 4 or len(set(local_slots.values())) != 4
            or any(type(d) is not int or type(s) is not int or not 0 <= s < 32
                   for d, s in local_slots.items())
            or type(record.get("jax_process_index")) is not int):
        raise ValueError("dense WK role differs")
    values = (("result", value), *((f"input{i}", v) for i, v in enumerate(inputs)))
    for role, array in values:
        sharded_source = name == "wk_decode" and role != "result"
        shape = (1, 48) if sharded_source and role == "input1" else (128, 6144)
        dtype = expected_dtype if role == "result" else (
            jnp.bfloat16 if name == "wk_promote" else jnp.uint8 if role == "input0" else jnp.float32)
        if tuple(array.shape) != shape or array.dtype != dtype:
            raise ValueError("dense WK global source/result shape or dtype differs")
        seen = set()
        for shard in array.addressable_shards:
            device = int(shard.device.id)
            if (device not in local_slots or device in seen or shard.device.platform != "tpu"
                    or shard.device.process_index != record["jax_process_index"]):
                raise ValueError("dense WK addressable owner differs")
            seen.add(device)
            index = (slice(None), slice(None))
            if sharded_source:
                width = shape[1] // 4
                feature = local_slots[device] % 4
                index = (slice(None), slice(feature * width, (feature + 1) * width))
            if _index_key(shard.index, shape) != _index_key(index, shape):
                raise ValueError("dense WK source/result shard index differs")
            observed = np.asarray(shard.data).copy()
            expected_shape = (128, 6144) if role == "result" or name == "wk_promote" else (
                (128, 1536) if role == "input0" else (1, 12))
            if observed.shape != expected_shape or observed.dtype != dtype:
                raise ValueError("dense WK output/source shape or dtype differs")
            invalid |= not np.isfinite(observed).all()
            arrays[f"slot{local_slots[device]}_{role}"] = (
                observed.view(np.uint16) if observed.dtype == jnp.bfloat16 else observed)
        if seen != set(local_slots):
            raise ValueError("dense WK incomplete addressable owners")
    path = root / f"layer{layer}_{name}.npz"
    if path.exists() or path.is_symlink():
        raise ValueError("dense WK refuses to replace original capsule")
    raw_bytes = sum(v.nbytes for v in arrays.values())
    previous = record.setdefault("wk_originals", {})
    if (sum(v["raw_array_bytes"] for v in previous.values()) + raw_bytes
            + (len(previous) + 1) * (1 << 20) > WK_ORIGINALS_LIMIT):
        raise ValueError("dense WK original rank budget exceeded")
    if name == "wk_promote":
        for slot in local_slots.values():
            decoded = arrays[f"slot{slot}_input0"].view(jnp.bfloat16).astype(np.float32)
            invalid |= decoded.tobytes() != arrays[f"slot{slot}_result"].tobytes()
    digest = save_arrays(path, arrays)
    previous[f"layer{layer}/{name}"] = dict(
        npz_sha256=digest, bytes=path.stat().st_size, raw_array_bytes=raw_bytes,
        shapes={key: list(v.shape) for key, v in arrays.items()}, valid=not invalid)
    if invalid:
        raise ValueError("dense nonfinite/inexact WK boundary; originals preserved")


def execute(*, root: Path, record: dict, mesh: Any, prepared: Any,
            embedding: Any, layers: Any, tokens: np.ndarray, rope: Any,
            witness: Mapping, local_slots: Mapping[int, int],
            consensus: Callable[[bool], bool], inspect_program: Callable[..., dict],
            norm_originals: Mapping | None = None,
            canonical_originals: Mapping | None = None) -> None:
    """Parent must supply its fixed actual-HLO inspector, never worker verdicts.

    This helper alone does not authorize the diagnostic. The probe/campaign must
    retain their default-off authorization until the fleet collector is wired.
    """
    from scripts.greenfield.probe_ws32_prefill_layer import compile_program
    from scripts.greenfield import ws32_dense_frontier_prepare as preparation
    from scripts.greenfield import ws32_dense_norm_protocol as norm_protocol
    from scripts.greenfield import ws32_dense_canonical as canonical

    norm_mode = record.get("protocol") == norm_protocol.PROTOCOL
    canonical_mode = record.get("protocol") == canonical.PROTOCOL
    selected_worker, selected_prepare = worker, preparation
    selected_protocol, journal_type = protocol.PROTOCOL, DenseJournal
    if norm_mode:
        from scripts.greenfield import ws32_dense_norm_worker as selected_worker
        from scripts.greenfield import ws32_dense_norm_prepare as selected_prepare
        selected_protocol, journal_type = norm_protocol.PROTOCOL, NormJournal
    program_names = norm_protocol.PROGRAMS if norm_mode else worker.PROGRAMS
    if canonical_mode:
        selected_worker = selected_prepare = canonical
        selected_protocol, journal_type = canonical.PROTOCOL, CanonicalJournal
        program_names = canonical.PROGRAMS

    def guarded(name, action):
        return fleet_step(name, action, record=record, root=root, consensus=consensus)

    journal = None
    failure = None
    try:
        def setup():
            nonlocal journal
            if (record.get("protocol") != selected_protocol or record.get("programs") != {}
                    or record.get("diagnostic_only") is not True
                    or (norm_mode and (not isinstance(norm_originals, Mapping)
                        or set(norm_originals) != {n for n, _, _ in norm_protocol.CAPTURES}))
                    or (not norm_mode and norm_originals is not None)
                    or (canonical_mode and (not isinstance(canonical_originals, Mapping)
                        or set(canonical_originals) != {n for n, _, _ in norm_protocol.CAPTURES}))
                    or (not canonical_mode and canonical_originals is not None)):
                raise ValueError("dense continuation identity/compile state differs")
            journal = journal_type(root / "compile_journal.jsonl", dict(
                protocol=selected_protocol, compile_only=False, diagnostic_only=True,
                code_hash=record["code_hash"], launch_rank=record["launch_rank"]))
            jobs = selected_prepare.compiler_programs(prepared, mesh)
            if tuple(name for name, _, _ in jobs) != program_names or len(layers) != 2:
                raise ValueError("dense compiler/layer inventory differs")
            calls = BudgetedCalls(root=root, record=record, consensus=consensus, journal=journal,
                                 local_slots=local_slots, budgeter=selected_worker.memory_budget)
            return jobs, calls
        jobs, calls = guarded("dense/setup", setup)
        for name, fn, values in jobs:
            def compile_one():
                compiled = compile_program(fn, values, name, root, record, journal=journal)
                journal.inspect(name, (root / f"{name}.stablehlo.mlir").read_text(),
                                (root / f"{name}.optimized_hlo.txt").read_text(),
                                lambda: dict(scope="ORIGINAL_CAPTURE_ONLY_NOT_ADMISSION"))
                calls.programs[name] = compiled
            guarded(f"dense/compile/{name}", compile_one)

        def admit():
            if tuple(calls.programs) != program_names:
                raise ValueError("dense compiler inventory differs")
            for name in program_names:
                record["programs"][name]["admission"] = inspect_program(
                    name, (root / f"{name}.stablehlo.mlir").read_text(),
                    (root / f"{name}.optimized_hlo.txt").read_text(),
                    record["programs"][name]["compiled_memory"])
                if record["programs"][name]["admission"].get("passed") is not True:
                    raise ValueError("dense actual compiler admission refused")
        calls.phase("dense/admission", admit)
        wk = []
        for layer_id, layer in enumerate(layers):
            operands = (layer.dsa.wk_bits_local, layer.dsa.wk_scale_local)
            decoded = calls.call(f"layer{layer_id}/wk_decode", "wk_decode", operands,
                preserve=lambda v: capture_wk(root, record, local_slots, layer=layer_id,
                                              name="wk_decode", value=v, inputs=operands))
            promoted = calls.call(f"layer{layer_id}/wk_promote", "wk_promote", (decoded,),
                preserve=lambda v: capture_wk(root, record, local_slots, layer=layer_id,
                                              name="wk_promote", value=v, inputs=(decoded,)))
            wk.append(promoted)
            del decoded, promoted, operands
        continuation = dict(mesh=mesh, config=prepared.config, prompt_tokens=tokens,
                            embedding=embedding, layers=layers, wk=tuple(wk), rope=rope,
                            witness=witness)
        if canonical_mode:
            continuation.pop("witness")
            selected_worker.execute_after_wk(calls, **continuation, originals=canonical_originals)
        elif norm_mode:
            selected_worker.execute_after_wk(calls, **continuation, originals=norm_originals)
        else:
            worker.execute_five_calls(calls, **continuation)
    except Exception as exc:
        failure = exc
        record.update(status="DIAGNOSTIC_FAILED", error=f"{type(exc).__name__}: {exc}")
    finally:
        def finalize():
            if journal is not None:
                journal.close()
                record["compile_journal_sha256"] = sha256((root / "compile_journal.jsonl").read_bytes()).hexdigest()
        try:
            guarded("dense/finalize", finalize)
        except Exception as exc:
            record["finalization_error"] = f"{type(exc).__name__}: {exc}"
            failure = failure or exc
    if failure is not None:
        record["status"] = "DIAGNOSTIC_FAILED"
        record.setdefault("error", f"{type(failure).__name__}: {failure}")
        raise failure
