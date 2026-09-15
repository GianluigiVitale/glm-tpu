"""Opt-in weight lifetimes for long prefill; no loader, launcher or admission.

Keep the authenticated raw checkpoint and completed repair WK during prefill.
Do not create the exact decode tree or load the dense overlay until afterwards.
The protected caller owns graph authorization, voted dispatch, evidence, and the
phase transition after successful prefill. It must drop its loader/argument aliases
and clear prefill executables before decode preparation. This class cannot free
another caller's references; the all-live census remains authoritative.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from glm_tpu.greenfield.runtime.ws32_decoder import (
    Ws32DecoderConfig,
    Ws32DecoderWeights,
    bind_ws32_decoder_weights,
    ws32_decoder_weight_names,
)
from scripts.greenfield.ws32_batched_prefill_runner import bind_raw_prefill_weights


CONTRACT = "ws32-prefill-raw-plus-completed-wk-then-decode-v1"


class PhaseWeights:
    """Single owner's phase roots, with no tensor copies or arithmetic rewrite.

    WK uses the already-protected two-program implementation, including the
    completed BF16 boundary. The original full exact materializer still runs
    for decode, AFTER prefill and its WK/executables have been released.
    A failed transition is terminal; it is not permission for a partial retry.
    """

    def __init__(self, arrays: Mapping[str, Any], config: Ws32DecoderConfig):
        if (config.geometry.fp8_block_shape != (128, 128)
                or config.geometry.hidden_size % 128
                or config.geometry.dsa_indexer_head_dim % 128):
            raise ValueError("phase repair requires the original 128x128 FP8 block geometry")
        raw_config = replace(config, exact_dsa=False, strategy_nd_dense=False)
        names = set(jax.tree.leaves(ws32_decoder_weight_names(raw_config)))
        if set(arrays) != names:
            raise ValueError("phase prefill requires exactly the base checkpoint, no overlay")
        self.raw_config, self.raw_weights = bind_raw_prefill_weights(arrays, config)
        self.decode_config = config
        self.wk: tuple[Any, ...] = ()
        self.decode_weights: Ws32DecoderWeights | None = None
        self.phase = "raw"

    def resident_roots(self) -> dict[str, Any]:
        """Named roots only; caller's census MUST also include all JAX-live arrays."""
        if self.phase in ("raw", "repair", "prefill", "failed") and self.raw_weights is not None:
            return {"raw_prefill_weights": self.raw_weights, "completed_repair_wk": self.wk}
        if self.phase == "decode" and self.decode_weights is not None:
            return {"decode_weights": self.decode_weights}
        raise ValueError("phase weight roots are unavailable")

    def wk_jobs(self, mesh: Any) -> tuple[tuple[str, Any, tuple], ...]:
        """Two reusable compiler jobs, using original WK code; zero dispatch.

        The inputs can be abstract. No QKV pack, query aliases or dense overlay
        is created. Actual optimized graph/memory admission is still required.
        """
        if self.phase != "raw":
            raise ValueError("WK preparation requires the raw phase")
        from scripts.greenfield.ws32_compile_originals import build_wk_programs

        sources = self._wk_sources()
        first = sources[0]
        spec = P(None, "feature")
        expected = (((self.raw_config.geometry.dsa_indexer_head_dim,
                      self.raw_config.geometry.hidden_size), jnp.dtype("uint8")),
                    ((self.raw_config.geometry.dsa_indexer_head_dim // 128,
                      self.raw_config.geometry.hidden_size // 128), jnp.dtype("float32")))
        for operands in sources:
            for value, (shape, dtype) in zip(operands, expected, strict=True):
                if (value.shape != shape or value.dtype != dtype
                        or not isinstance(value.sharding, NamedSharding)
                        or not value.sharding.is_equivalent_to(NamedSharding(mesh, spec), 2)):
                    raise ValueError("phase WK source shape/dtype/sharding differs")
        decode, promote = build_wk_programs(mesh, spec, spec, contract=self.raw_config.dsa_contract)
        completed = jax.ShapeDtypeStruct(expected[0][0], jnp.bfloat16,
                                       sharding=NamedSharding(mesh, P()))
        return (("wk_decode", decode, first), ("wk_promote", promote, (completed,)))

    def _wk_sources(self) -> tuple[tuple[Any, Any], ...]:
        if self.raw_weights is None:
            raise ValueError("raw phase weights were released")
        result = []
        for layer in self.raw_config.full_index_slots:
            dsa = self.raw_weights.layers[layer].dsa
            if dsa is None:
                raise ValueError("phase repair producer has no original WK")
            result.append((dsa.wk_bits_local, dsa.wk_scale_local))
        return tuple(result)

    def materialize_wk(self, call: Callable[[str, str, tuple], Any], *,
                       phase: Callable[[str, Callable[[], Any]], Any]) -> None:
        """Use the caller's admitted/voted call path for each producer's own WK.

        Previously completed WK stays in resident_roots while later producers
        execute. The caller's all-live budget also sees each temporary BF16
        output. Use the caller's fleet-voted phase for EVERY local validation,
        so a peer-only refusal cannot leave other ranks in the next collective.
        No unbudgeted second dispatcher or retained executable here.
        """
        def start():
            if self.phase != "raw":
                raise ValueError("WK materialization is a one-shot raw-phase transition")
            self.phase = "repair"
            return self._wk_sources()
        try:
            sources = phase("wk/start", start)
            for layer, operands in zip(self.raw_config.full_index_slots, sources, strict=True):
                decoded = call(f"layer{layer}/wk_decode", "wk_decode", operands)
                phase(f"layer{layer}/wk_decode_ready", lambda: self._check_completed(decoded, "bfloat16"))
                promoted = call(f"layer{layer}/wk_promote", "wk_promote", (decoded,))
                def retain():
                    self._check_completed(promoted, "float32")
                    self.wk = (*self.wk, promoted)
                phase(f"layer{layer}/wk_promote_ready", retain)
                del decoded, promoted
            phase("wk/finish", lambda: setattr(self, "phase", "prefill"))
        except Exception:
            self.phase = "failed"
            raise

    def _check_completed(self, value: Any, dtype: str) -> None:
        if (not isinstance(value, jax.Array) or value.is_deleted()
                or value.shape != (self.raw_config.geometry.dsa_indexer_head_dim,
                                   self.raw_config.geometry.hidden_size)
                or str(value.dtype) != dtype or not value.sharding.is_fully_replicated):
            raise ValueError(f"phase repair requires completed replicated {dtype} WK")
        value.block_until_ready()

    def begin_decode(self, overlay_arrays: Mapping[str, Any]) -> Ws32DecoderWeights:
        """Rebind shared base leaves, replace only dense leaves, release old roots.

        Caller supplies ONLY the original loader-verified overlay, after healthy
        prefill and release of prefill state borrowers/executables. Neither this
        binder nor its CPU tests certify checkpoint bytes or a safe runtime peak.
        Missing, extra or overlapping overlay names refuse before mutation.
        """
        if self.phase != "prefill" or self.raw_weights is None:
            raise ValueError("decode transition requires completed prefill-phase weights")
        raw_names, structure = jax.tree.flatten(ws32_decoder_weight_names(self.raw_config))
        values, actual_structure = jax.tree.flatten(self.raw_weights)
        if structure != actual_structure:
            raise ValueError("raw phase weight structure changed")
        needed = set(jax.tree.leaves(ws32_decoder_weight_names(self.decode_config)))
        if set(overlay_arrays) != needed - set(raw_names):
            raise ValueError("decode overlay tensor set differs")
        arrays = {name: value for name, value in zip(raw_names, values, strict=True)
                  if name in needed}
        arrays.update(overlay_arrays)
        decoded = bind_ws32_decoder_weights(arrays, self.decode_config)
        self.decode_weights = decoded
        self.raw_weights = None
        self.wk = ()
        self.phase = "decode"
        return decoded
