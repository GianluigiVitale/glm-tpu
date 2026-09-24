"""The device programs one runtime compiles, built in one place.

``build_program_set`` constructs every program ``OrdinaryRuntime._load`` and
``batched_runtime.compile_batch`` compile, in their compile order: the indexer WK decode and
promotion, the fresh cache initializer, the B128 and B114 prefill blocks, and either the packed
decode step or, for a concurrent runtime, the batch programs (bank initializer, lane insert,
vectorized decode). Options, block shapes and the donation rule are production's: a runtime
whose context capacity exceeds ``request.CAPACITY`` (8,192) donates the prefill and decode state
(exclusive ownership of the multi-GB cache); the batch insert always donates its bank and the
batched decode its state. The runtime compiles exactly ``ProgramSpec.fn`` with the arguments its
load produces (checkpoint tensors, the promoted WK tables, the resident BF16 weights, the
initializer's state), so a spec carries no argument builder; the equivalence harness lowers the
same specs with the arguments the runtime passed and requires identical fingerprints.

The per-table FP8 decoders ``bf16_resident_weights`` compiles by jit dispatch while it prepares
the resident weights are keyed by the loaded tables' shapes and stay with that preparation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
from jax.sharding import NamedSharding, PartitionSpec as P

from glm_tpu.config import cache
from glm_tpu.models.glm_moe_dsa.state import decoder_state_specs
from glm_tpu.models.glm_moe_dsa.model import build_batched_decoder_program, build_packed_decoder_program
from glm_tpu.models.glm_moe_dsa.weights import build_wk_programs
from glm_tpu.models.glm_moe_dsa.prefill import build_prefill_program
from glm_tpu.engine.request import CAPACITY
from .kv_cache_manager import build_cache_initializer

# The admitted prefill profile is the only one build_prefill_program builds (S2d): MLP window with
# four rolled 32-row prefixes, routed-expert panels, the canonical dense MLP and the one-pass DSA
# selector.
PREFILL_BLOCK_ROWS = (128, 114)  # full blocks, and the tail program for a last block of <= 114 rows
INTERPRET = dict(sparse_attention_interpret=True, linear_interpret=True)  # Pallas interpret mode (CPU)


@dataclass(frozen=True)
class ProgramSpec:
    name: str                          # frozen: HLO original file names and runtime-record program keys
    fn: Callable[..., Any]             # the jitted program the runtime compiles (donation applied)
    donate_argnums: tuple[int, ...] = ()
    model: bool = True                 # a model graph: its optimized HLO passes the collective admission


@dataclass(frozen=True)
class BatchPrograms:
    cache_init: ProgramSpec            # batch_cache_init: one fresh decoder state per lane
    insert: ProgramSpec                # batch_insert: write one prefilled state into a lane (bank donated)
    decode: ProgramSpec                # batch_decode: one decode step for every lane
    state_shardings: Any               # the bank's shardings (lane axis unsharded)


@dataclass(frozen=True)
class ProgramSet:
    wk: tuple[ProgramSpec, ProgramSpec]    # wk_decode, wk_promote (never fused)
    cache_init: ProgramSpec
    prefill: Mapping[int, ProgramSpec]     # by block rows: 128, 114
    decode: ProgramSpec | None             # the packed decode step (sequential runtime)
    batch: BatchPrograms | None            # the batch programs (concurrent runtime)

    def specs(self) -> tuple[ProgramSpec, ...]:
        """Every program in the runtime's compile order."""
        tail = (self.decode,) if self.batch is None else (self.batch.cache_init, self.batch.insert, self.batch.decode)
        return (*self.wk, self.cache_init, *self.prefill.values(), *tail)


def donates_state(capacity: int) -> bool:
    """Whether a runtime of ``capacity`` cache slots donates its prefill and decode state: above
    ``request.CAPACITY`` (8,192) the long-context runtime transfers exclusive ownership of the
    multi-GB cache (the runtime record calls it ``exclusive_donated``). The one place the rule lives."""
    return capacity > CAPACITY


def build_program_set(mesh: Any, config: cache.CacheConfig, *, concurrent_size: int = 0,
                      interpret: bool = False) -> ProgramSet:
    """The programs of a runtime over ``mesh`` with ``config`` (its context capacity decides
    donation) and ``concurrent_size`` lanes (0: sequential). ``interpret`` runs the Pallas kernels
    of the prefill and decode programs in interpret mode (CPU tests)."""
    kernels = INTERPRET if interpret else {}
    donating = donates_state(config.context_capacity)
    decode, promote = build_wk_programs(mesh, P(None, "feature"), P(None, "feature"), contract=config.dsa_contract)
    wk = (ProgramSpec("wk_decode", decode, model=False), ProgramSpec("wk_promote", promote, model=False))
    cache_init = ProgramSpec("cache_init", build_cache_initializer(mesh, config), model=False)
    prefill = {}
    for rows in PREFILL_BLOCK_ROWS:
        fn = build_prefill_program(mesh, config, block_rows=rows, **kernels).execute
        donate = (2,) if donating else ()
        prefill[rows] = ProgramSpec(f"prefill_{rows}", jax.jit(fn, donate_argnums=donate) if donate else fn, donate)
    if concurrent_size:
        return ProgramSet(wk, cache_init, prefill, None, _batch_programs(mesh, config, concurrent_size, kernels))
    fn = build_packed_decoder_program(mesh, config, **kernels).execute
    donate = (1,) if donating else ()
    decode_step = ProgramSpec("decode", jax.jit(fn, donate_argnums=donate) if donate else fn, donate)
    return ProgramSet(wk, cache_init, prefill, decode_step, None)


def _batch_programs(mesh: Any, config: cache.CacheConfig, n: int, kernels: dict[str, bool]) -> BatchPrograms:
    specs = jax.tree.map(lambda s: P(None, *s), decoder_state_specs())
    shardings = jax.tree.map(lambda s: NamedSharding(mesh, s), specs)
    initialize = build_cache_initializer(mesh, config)
    bank = jax.jit(jax.vmap(lambda length: initialize(length).decoder), out_shardings=shardings)

    def insert(state, one, index):
        return jax.tree.map(lambda x, y: x.at[index].set(y), state, one)

    return BatchPrograms(
        ProgramSpec("batch_cache_init", bank, model=False),
        ProgramSpec("batch_insert", jax.jit(insert, donate_argnums=(0,)), (0,), model=False),
        ProgramSpec("batch_decode", build_batched_decoder_program(mesh, config, batch_size=n, **kernels), (1,)),
        shardings)
