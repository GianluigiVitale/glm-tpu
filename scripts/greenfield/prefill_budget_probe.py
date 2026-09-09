"""Fixed missing-budget building blocks, not a standalone TPU launcher.

Reuses production multirow DSA and zero-state initialization without weights.
An existing protected fleet wrapper must own deployment, votes, HLO/memory
admission, publication and cleanup before these functions run on TPU.
Synthetic arithmetic and local callback delivery do not establish model TTFT.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable

import numpy as np

PROTOCOL = "ws32-prefill-long-prefix-budget-v1"
ROWS = 32
KEY_TILE = 512
TOP_K = 2048
WARMUP = 2
ITERATIONS = 5
SAMPLING_BUDGET_SECONDS = 120
CAPACITIES = ((131072, 127363), (262656, 262144))
DELIVERY_SCOPE = "LOCAL_IN_PROCESS_CALLBACK_ACK_NO_TOKENIZATION_OR_NETWORK"


@dataclass(frozen=True)
class BudgetCase:
    capacity: int
    prompt_length: int
    last_valid_length: int

    def __post_init__(self) -> None:
        if (
            any(
                type(x) is not int
                for x in (self.capacity, self.prompt_length, self.last_valid_length)
            )
            or (self.capacity, self.prompt_length) not in CAPACITIES
            or self.last_valid_length
            not in (2048, self.prompt_length // 2, self.prompt_length)
        ):
            raise ValueError("unregistered prefill budget case")

    @property
    def name(self) -> str:
        return f"c{self.capacity}_p{self.last_valid_length}"

    @property
    def valid_lengths(self) -> np.ndarray:
        # The case prefix always names the LAST row, not the first.
        return np.arange(
            self.last_valid_length - ROWS + 1,
            self.last_valid_length + 1,
            dtype=np.int32,
        )


def cases() -> tuple[BudgetCase, ...]:
    return tuple(
        BudgetCase(capacity, prompt, prefix)
        for capacity, prompt in CAPACITIES
        for prefix in (2048, prompt // 2, prompt)
    )


def stripe_positions(capacity: int, start: int, stop: int) -> np.ndarray:
    """Global positions for owner-major storage of logical512/8=64 stripes."""
    if (
        any(type(x) is not int for x in (capacity, start, stop))
        or capacity <= 0
        or capacity % 512
        or not 0 <= start <= stop <= capacity
    ):
        raise ValueError("invalid budget stripe range")
    ordinal = np.arange(start, stop, dtype=np.int32)
    local_rows = capacity // 8
    expert, local = ordinal // local_rows, ordinal % local_rows
    return ((local // 64 * 8 + expert) * 64 + local % 64).astype(np.int32)


def query_inputs(
    *, rows: int = ROWS, tied: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Nonzero one-hot queries make every executing score independently analytic."""
    if type(rows) is not int or not 1 <= rows <= ROWS or type(tied) is not bool:
        raise ValueError("invalid analytic DSA query fixture")
    query = np.zeros((rows, 32, 128), np.float32)
    query[np.arange(rows), 0, np.arange(rows)] = 1
    heads = np.zeros((rows, 32), np.float32)
    if not tied:
        heads[:, 0] = 1
    return query, heads


def key_values(positions: np.ndarray) -> np.ndarray:
    """All integers0..250 are exactly BF16 representable; no random-state drift."""
    import ml_dtypes

    if positions.ndim != 1 or positions.dtype != np.int32:
        raise ValueError("analytic keys need an int32 position vector")
    return (
        (
            positions[:, None].astype(np.int64) * (np.arange(128, dtype=np.int64) + 1)
            + np.arange(128)
        )
        % 251
    ).astype(ml_dtypes.bfloat16)


def expected_selection(
    lengths: np.ndarray, *, top_k: int = TOP_K, tied: bool = False
) -> dict[str, np.ndarray]:
    """One CPU score vector/row; no rows*heads*context allocation or matmul."""
    if (
        lengths.ndim != 1
        or lengths.dtype != np.int32
        or not 1 <= len(lengths) <= ROWS
        or np.any(lengths < 0)
        or np.any(lengths > CAPACITIES[-1][0])
        or type(top_k) is not int
        or not 1 <= top_k <= TOP_K
        or type(tied) is not bool
    ):
        raise ValueError("invalid analytic selection geometry")
    positions = np.full((len(lengths), top_k), -1, np.int32)
    scores = np.full((len(lengths), top_k), -np.inf, np.float32)
    counts = np.minimum(lengths, top_k).astype(np.int32)
    for row, length in enumerate(lengths):
        p = np.arange(int(length), dtype=np.int32)
        integer_scores = np.zeros_like(p) if tied else (p * (row + 1) + row) % 251
        # Stable sort of increasing positions implements lowest-position ties.
        chosen = np.argsort(-integer_scores, kind="stable")[:top_k]
        count = int(counts[row])
        positions[row, :count] = chosen
        scores[row, :count] = integer_scores[chosen].astype(np.float32) * np.float32(
            128**-0.5
        )
    return dict(positions=positions, valid_counts=counts, scores=scores)


def build_dsa_program(
    mesh: Any,
    *,
    capacity: int,
    rows: int = ROWS,
    top_k: int = TOP_K,
    sorted_local_merge: bool = False,
) -> Any:
    """Production512/default/paired DSA; optional CPU-staged local merge.

    The protected baseline worker never opts in. Candidate hardware execution
    requires a distinct registered profile/collector, not this flag alone.
    """
    import jax
    from jax.sharding import PartitionSpec as P
    from glm_tpu.greenfield.kernels.prefill_dsa import (
        ws32_prefill_dsa_from_query_mapped,
    )
    from glm_tpu.greenfield.kernels.reference.dsa import ScoredSelectedPositions

    # Smaller capacities/row counts are allowed only for CPU semantic tests.
    stripe_positions(capacity, 0, 0)
    query_inputs(rows=rows)
    if dict(mesh.shape) != {"expert": 8, "feature": 4}:
        raise ValueError("budget DSA requires expert8/feature4 mesh")
    if type(top_k) is not int or not 1 <= top_k <= TOP_K:
        raise ValueError("invalid budget top_k")
    if type(sorted_local_merge) is not bool:
        raise ValueError("sorted_local_merge must be a static bool")

    def execute(query, keys, heads, positions, lengths):
        if query.shape[0] != rows or keys.shape[0] != capacity // 8:
            raise ValueError("budget program input shape differs")
        selected, healthy = ws32_prefill_dsa_from_query_mapped(
            query,
            keys,
            heads,
            positions,
            lengths,
            global_context_size=capacity,
            top_k=top_k,
            key_tile=KEY_TILE,
            precision="default",
            paired_position_sort=True,
            sorted_local_merge=sorted_local_merge,
        )
        # One local health cell per physical chip, never silently replicated.
        return selected, healthy[None, None]

    return jax.jit(
        jax.shard_map(
            execute,
            mesh=mesh,
            in_specs=(P(), P("expert", None), P(), P("expert"), P()),
            out_specs=(ScoredSelectedPositions(P(), P(), P()), P("expert", "feature")),
            check_vma=False,
        )
    )


def make_dsa_inputs(mesh: Any, case: BudgetCase, *, tied: bool = False) -> tuple:
    """Allocate only addressable key stripes; never a full replicated key cache."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    query, heads = query_inputs(tied=tied)
    replicated = NamedSharding(mesh, P())
    keys_spec = NamedSharding(mesh, P("expert", None))
    pos_spec = NamedSharding(mesh, P("expert"))

    def positions_for(index):
        section = index[0]
        start, stop, step = section.indices(case.capacity)
        if step != 1:
            raise ValueError("budget key callback has noncontiguous storage slice")
        return stripe_positions(case.capacity, start, stop)

    keys = jax.make_array_from_callback(
        (case.capacity, 128), keys_spec, lambda idx: key_values(positions_for(idx))
    )
    positions = jax.make_array_from_callback((case.capacity,), pos_spec, positions_for)
    return (
        jax.device_put(query, replicated),
        keys,
        jax.device_put(heads, replicated),
        positions,
        jax.device_put(case.valid_lengths, replicated),
    )


def check_output(result: tuple, expected: dict[str, np.ndarray]) -> dict:
    """Check every local original; all32 owners must be joined by the collector."""
    selected, health = result
    health_by_device = {
        int(s.device.id): np.asarray(s.data) for s in health.addressable_shards
    }
    if (
        len(health_by_device) != len(health.addressable_shards)
        or not health_by_device
        or any(v.dtype != np.bool_ or not v.all() for v in health_by_device.values())
    ):
        raise ValueError("DSA budget health failed")
    hashes = {d: {} for d in health_by_device}
    for name, value in zip(
        ("positions", "valid_counts", "scores"), selected, strict=True
    ):
        seen = set()
        for shard in value.addressable_shards:
            device = int(shard.device.id)
            actual = np.asarray(shard.data)
            reference = expected[name]
            if (
                device not in hashes
                or device in seen
                or actual.shape != reference.shape
                or actual.dtype != reference.dtype
                or actual.tobytes() != reference.tobytes()
            ):
                raise ValueError(
                    f"DSA budget original {name} differs from analytic reference"
                )
            seen.add(device)
            hashes[device][name] = sha256(actual.tobytes()).hexdigest()
        if seen != set(health_by_device):
            raise ValueError("DSA budget output owner coverage differs")
    return dict(
        passed=True,
        owners={str(device): fields for device, fields in hashes.items()},
        reference="ANALYTIC_ONE_HOT_FP32_SCALED_MOD251",
    )


def measure_initial_state(
    mesh: Any,
    config: Any,
    *,
    prompt_length: int,
    clock: Callable[[], float] = perf_counter,
) -> tuple[Any, dict]:
    """Production fresh-cache allocation, no weights or populated resume state.

    Caller releases this state before allocating the other capacity. The first
    synchronized allocation may include internal initialization compilation;
    report it separately from later warm allocations, never as model TTFT.
    """
    import jax
    from glm_tpu.greenfield.runtime.ws32_batched_prefill import (
        make_ws32_batched_prefill_state,
    )

    if (config.context_capacity, prompt_length) not in CAPACITIES:
        raise ValueError("unregistered initialization capacity/prompt")
    started = clock()
    state = make_ws32_batched_prefill_state(mesh, config, prompt_length=prompt_length)
    jax.block_until_ready(state)
    seconds = clock() - started
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("initialization clock is invalid")
    return state, dict(
        capacity=config.context_capacity,
        prompt_length=prompt_length,
        cache_initialization_seconds=seconds,
        weights_loaded=False,
        initialized_prefix_length=0,
        model_ttft_measured=False,
    )


def sample_dsa(
    calls: Any,
    graph: str,
    inputs: tuple,
    expected: dict[str, np.ndarray],
    *,
    case_name: str,
    preserve: Callable[[int, Any], None],
    budget_started: float,
    clock: Callable[[], float] = perf_counter,
) -> dict:
    """Fixed samples through existing BudgetedCalls memory/dispatch/vote path.

    A generic acquisition fleet_step is NOT safe around distributed dispatch:
    it can fail publishing RUNNING before only some hosts enter the executable.
    BudgetedCalls votes local preflight first and publishes after execution.
    Warmup,
    validation, preservation and votes count toward the shared120s six-case
    budget; only dispatch-through-completion enters samples. The caller saves
    first/last original results before checking and publishes failed records.
    No tracing inside samples. Precompute expected arrays before sampling.
    """

    def budget() -> None:
        elapsed = clock() - budget_started
        if not np.isfinite(elapsed) or not 0 <= elapsed < SAMPLING_BUDGET_SECONDS:
            raise ValueError("prefill budget sampling deadline exceeded")

    samples, checks = [], []
    for index in range(WARMUP + ITERATIONS):
        calls.phase(f"{case_name}/sample{index}_budget", budget)

        def capture(output):
            if index in (0, WARMUP + ITERATIONS - 1):
                # Before the phase's fleet vote: a peer may fail after this
                # host completed its output. Persistence is outside timing.
                preserve(index, output)

        result = calls.call(
            f"{case_name}/sample{index}", graph, inputs, preserve=capture
        )
        seconds = calls.record["call_evidence"][-1]["completed_call_seconds"]

        def verify():
            try:
                return check_output(result, expected)
            except Exception:
                preserve(index, result)
                raise

        def checked():
            if not np.isfinite(seconds) or seconds < 0:
                raise ValueError("DSA sampling clock is invalid")
            return verify()

        checks.append(calls.phase(f"{case_name}/sample{index}_check", checked))
        if index >= WARMUP:
            samples.append(seconds)
        del result
        calls.phase(f"{case_name}/sample{index}_post_budget", budget)
    return dict(
        warmup=WARMUP,
        iterations=ITERATIONS,
        samples_seconds=samples,
        checks=checks,
        scope="SYNTHETIC_DSA_DISPATCH_THROUGH_COMPLETION_ONLY",
        model_prefill_measured=False,
        model_ttft_measured=False,
    )


def deliver_local_token(
    token: Any,
    on_token: Callable[[int], None],
    *,
    clock: Callable[[], float] = perf_counter,
) -> dict:
    """Delivery ends after the caller's callback returns successfully.

    Conversion is inside timing, as is synchronous local consumer processing.
    A raised callback has delivered no successful acknowledgement. No claim
    about socket/client-network receipt, tokenization or model TTFT follows.
    """
    started = clock()
    value = np.asarray(token)
    if value.shape != (1,) or value.dtype != np.int32 or int(value[0]) < 0:
        raise ValueError("delivery requires one valid int32 token")
    ready = clock()
    token_id = int(value[0])
    on_token(token_id)
    delivered = clock()
    intervals = (ready - started, delivered - ready, delivered - started)
    if any(not np.isfinite(v) or v < 0 for v in intervals):
        raise ValueError("delivery clock is invalid")
    return dict(
        token_id=token_id,
        device_to_host_seconds=intervals[0],
        callback_ack_seconds=intervals[1],
        delivery_seconds=intervals[2],
        delivery_scope=DELIVERY_SCOPE,
        model_ttft_measured=False,
    )
