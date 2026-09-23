"""CPU32 child: the unsharded reference against the frozen FP8 oracle and production.

Run as ``python -B -m tests.reference.oracle_run --pair CANDIDATE:BASELINE
[--prompt a|short] [--steps N]`` with ``JAX_PLATFORMS=cpu`` and 32 forced host
devices (the frozen fixture's ``expert=8 x feature=4`` mesh); prints one JSON
report line (digests, counts and error summaries only).

Systems (all read the frozen fixture v1 checkpoint, panel geometry, 1,536 slots,
DSA ``top_k`` 128; engine programs run with Pallas in interpret mode and the
TPU-v4 chip description):

* ``reference`` -- :mod:`tests.reference.model`, prefill in 128-row blocks;
  ``reference-one-block`` -- the same with the whole prompt in one block (only
  the accumulation shapes change: the reference's own rounding-noise floor);
* ``fp8-oracle`` -- the frozen raw-FP8 programs: ``build_ws32_batched_prefill_program``
  (B128 blocks, the production window options) and ``build_ws32_decoder_program``
  ``.observe`` (every full indexer's selection);
* ``production`` -- what ``OrdinaryRuntime._load`` builds (as at 181c013e): BF16-resident
  tables and the B128/B114 prefill, packed decode and cache-initializer programs of
  ``glm_tpu.runner.programs.build_program_set`` (only the carried selection is observable).

The baseline runs free (it decodes its own greedy tokens); the candidate is
teacher-forced with the baseline's input token at every step, so every step
compares two predictions from identical inputs. Compared at every common prefill
boundary and decode step: integer/boolean state leaves exactly; float leaves with
``|candidate - baseline| <= atol + rtol * |baseline|`` (rtol 0.02, atol 0.0625,
the bounds of the frozen-prefill comparison); greedy tokens; DSA selections in
score order and as sets (with the baseline's scores of every disagreeing slot).

Prompts: ``a`` is prompt A of the G3 goldens (157 tokens: a 128-row block and a
29-row tail, so the tail rows select 128 of up to 157 positions); ``short`` is the
three-token prompt of the frozen-oracle release tests (every position selected).
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

PROMPTS = {
    "a": tuple((i * 37 + 11) % 256 for i in range(157)),  # G3 prompt A
    "short": (30, 31, 32),  # tests/release/test_optimized_{bf16,prefill_program}.py
}
BLOCK_ROWS = 128
TAIL_ROWS = 114
DECODE_STEPS = 8
RTOL, ATOL = 0.02, 0.0625
WINDOW_OPTIONS = dict(
    key_tile=512,
    mlp_window=True,
    rolled_prefix=True,
    expert_panels=True,
    paired_position_sort=True,
    sorted_local_merge=True,
    canonical_dense=True,
)
INTERPRET = dict(sparse_attention_interpret=True, linear_interpret=True)
DECODER_FIELDS = (
    "kv_cache_local",
    "index_cache_local",
    "selected_positions",
    "selected_valid_counts",
    "selected_scores",
    "position",
    "block_tables",
    "context_lengths",
    "contract_valid",
)


# ----------------------------------------------------------------------------- comparison
def host(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return array.astype(np.float32) if array.dtype.name == "bfloat16" else array


def compare_leaf(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    """Exact equality for integer/bool leaves; the tolerance summary for float leaves."""
    if candidate.shape != baseline.shape:
        return dict(
            kind="shape",
            ok=False,
            candidate=list(candidate.shape),
            baseline=list(baseline.shape),
        )
    if baseline.dtype.kind in "biu":
        mismatches = int(np.sum(candidate != baseline))
        return dict(
            kind="exact",
            ok=mismatches == 0,
            mismatches=mismatches,
            size=int(baseline.size),
        )
    cand, base = candidate.astype(np.float64), baseline.astype(np.float64)
    same_specials = np.array_equal(np.isinf(cand), np.isinf(base)) and np.array_equal(
        cand[np.isinf(cand)], base[np.isinf(base)]
    )
    finite = np.isfinite(cand) & np.isfinite(base)
    diff = np.abs(cand[finite] - base[finite])
    bound = ATOL + RTOL * np.abs(base[finite])
    nonzero = np.abs(base[finite]) > 1e-6
    return dict(
        kind="float",
        ok=bool(
            same_specials
            and not np.isnan(cand).any()
            and not np.isnan(base).any()
            and np.all(diff <= bound)
        ),
        max_abs=float(diff.max(initial=0.0)),
        max_rel=float((diff[nonzero] / np.abs(base[finite])[nonzero]).max(initial=0.0)),
        bound_ratio=float((diff / bound).max(initial=0.0)),
        outside_fraction=float(np.mean(diff > bound)) if diff.size else 0.0,
        differing_fraction=float(np.mean(cand != base)),
        size=int(base.size),
    )


def compare_selection(
    candidate: tuple[np.ndarray, np.ndarray], baseline: tuple[np.ndarray, np.ndarray]
) -> dict[str, Any]:
    """Selections ``(positions, scores)`` of one row; positions in score order, ``-1`` tail.

    ``score_noise`` is the largest absolute score difference over the positions both
    systems selected. A set disagreement is *explained* when every disagreeing
    position's score lies within twice that noise above the last selected score of the
    system that selected it: rounding alone can then swap it with a boundary position.
    """
    cand, base = candidate[0].reshape(-1), baseline[0].reshape(-1)
    cand_scores, base_scores = candidate[1].reshape(-1), baseline[1].reshape(-1)
    cand_by = {
        int(p): float(v) for p, v in zip(cand, cand_scores, strict=True) if p >= 0
    }
    base_by = {
        int(p): float(v) for p, v in zip(base, base_scores, strict=True) if p >= 0
    }
    common = set(cand_by) & set(base_by)
    noise = max((abs(cand_by[p] - base_by[p]) for p in common), default=0.0)
    only_base, only_cand = (
        sorted(set(base_by) - set(cand_by)),
        sorted(set(cand_by) - set(base_by)),
    )
    gaps = [base_by[p] - min(base_by.values()) for p in only_base]
    gaps += [cand_by[p] - min(cand_by.values()) for p in only_cand]
    return dict(
        ordered_equal=bool(np.array_equal(cand, base)),
        set_equal=not only_base and not only_cand,
        live=len(base_by),
        order_mismatches=int(np.sum(cand != base)),
        disagreeing=len(only_base) + len(only_cand),
        boundary_gaps=[float(g) for g in gaps],
        score_noise=float(noise),
        explained=all(g <= 2 * noise for g in gaps),
    )


def compare_record(
    candidate: dict[str, Any], baseline: dict[str, Any], *, top_k: int
) -> dict[str, Any]:
    leaves = {
        name: compare_leaf(candidate["leaves"][name], value)
        for name, value in baseline["leaves"].items()
        if name in candidate["leaves"]
    }
    selections = {
        layer: compare_selection(candidate["selections"][layer], value)
        for layer, value in baseline["selections"].items()
        if layer in candidate["selections"]
    }
    # The reference system's record (it carries per-row decision margins), if any.
    reference = next((r for r in (candidate, baseline) if r.get("decisions")), None)
    return dict(
        position=baseline["position"],
        token_in=baseline.get("token_in"),
        tokens=dict(candidate=candidate["next_token"], baseline=baseline["next_token"]),
        leaves=leaves,
        selections=selections,
        rows_error=rows_error(
            candidate["leaves"], baseline["leaves"], baseline["rows"]
        ),
        # Rows at positions < top_k attend to every earlier position (no DSA choice).
        decision_free_rows=(
            rows_error(
                candidate["leaves"],
                baseline["leaves"],
                (baseline["rows"][0], min(baseline["rows"][1], top_k)),
            )
            if baseline["rows"][0] < top_k
            else None
        ),
        logits=dict(candidate=candidate.get("logits"), baseline=baseline.get("logits")),
        decisions=dict(
            candidate=candidate.get("decisions"), baseline=baseline.get("decisions")
        ),
        first_excess=first_excess(
            candidate["leaves"],
            baseline["leaves"],
            baseline["rows"],
            decisions=(reference["decisions"], reference["rows"])
            if reference
            else (None, None),
        ),
    )


def written_rows(value: np.ndarray, rows: tuple[int, int]) -> np.ndarray:
    """``[layers or slots, rows written in this call, width]`` of a cache leaf (FP64)."""
    return value.reshape(value.shape[0], -1, value.shape[-1])[
        :, rows[0] : rows[1]
    ].astype(np.float64)


def rows_error(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    rows: tuple[int, int],
) -> dict[str, list[float]]:
    """Per layer / per full slot, over the cache rows written in this call: the largest
    |difference| and the largest ratio of |difference| to the float bound."""

    out: dict[str, list[float]] = {}
    for name, leaf, columns in (
        ("kv_latent", "kv_cache_local", slice(0, 512)),
        ("kv_rope", "kv_cache_local", slice(512, 576)),
        ("index", "index_cache_local", slice(None)),
    ):
        cand, base = (
            written_rows(candidate[leaf], rows)[..., columns],
            written_rows(baseline[leaf], rows)[..., columns],
        )
        diff = np.abs(cand - base)
        out[name] = [float(x) for x in diff.max(axis=(1, 2))]
        out[name + "_bound_ratio"] = [
            float(x) for x in (diff / (ATOL + RTOL * np.abs(base))).max(axis=(1, 2))
        ]
    return out


def first_excess(
    candidate: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    rows: tuple[int, int],
    *,
    decisions: tuple[dict[str, Any] | None, tuple[int, int] | None],
) -> list[dict[str, Any]]:
    """Per written row that leaves the float bound: where it first does, and what came before.

    For every cache row of this call whose KV (latent or RoPE part) exceeds the bound in
    some layer: its position, that first layer, the bound ratio there and one layer
    earlier, and -- when a reference system took part -- that row's own router margins
    (k-th minus (k+1)-th biased score) of the sparse layers and DSA margins of the full
    layers before that layer. This is the per-row root-cause evidence of VALIDATION.md.
    ``decisions`` is a reference record's ``(decisions, rows)`` (``(None, None)`` for
    two engines).
    """
    margins, margin_rows = decisions
    cand = written_rows(candidate["kv_cache_local"], rows)
    base = written_rows(baseline["kv_cache_local"], rows)
    ratio = (np.abs(cand - base) / (ATOL + RTOL * np.abs(base))).max(axis=-1)
    out = []
    for offset in np.flatnonzero((ratio > 1.0).any(axis=0)):
        layer = int(np.argmax(ratio[:, offset] > 1.0))
        entry: dict[str, Any] = dict(
            position=rows[0] + int(offset),
            layer=layer,
            bound_ratio=float(ratio[layer, offset]),
            previous_layer_ratio=float(ratio[layer - 1, offset]) if layer else None,
        )
        index = rows[0] + int(offset) - (margin_rows or (0, 0))[0]
        if margins is not None and 0 <= index < margin_rows[1] - margin_rows[0]:
            for kind in ("router", "dsa"):
                entry[kind + "_margins"] = {
                    name: values[index]
                    for name, values in margins[kind + "_margin_rows"].items()
                    if int(name) < layer
                }
        out.append(entry)
    return out


# ----------------------------------------------------------------------------- systems
class Engine:
    """Common driver of the two sharded engine compositions (global array layout)."""

    name = ""

    def __init__(
        self,
        mesh: Any,
        config: Any,
        weights: Any,
        wk: Any,
        rope: Any,
        put: Any,
        prompt: tuple[int, ...],
    ) -> None:
        from glm_tpu.greenfield.runtime import ws32_batched_prefill

        self.config, self.wk, self.rope, self.put, self.prompt = (
            config,
            wk,
            rope,
            put,
            prompt,
        )
        self.finish = ws32_batched_prefill.finish_ws32_batched_prefill

    def block_rows(self, count: int) -> int:
        raise NotImplementedError

    def prefill_record(self, result: Any, rows: tuple[int, int]) -> dict[str, Any]:
        state = result.state
        leaves = {name: host(getattr(state.decoder, name)) for name in DECODER_FIELDS}
        leaves.update(
            repaired_index_local=host(state.repaired_index_local),
            prompt_length=host(state.prompt_length),
            finished=host(state.finished),
            next_token=host(result.next_token),
        )
        selections = {
            str(self.carried_layer): (
                leaves["selected_positions"][0],
                leaves["selected_scores"][0],
            )
        }
        return dict(
            position=rows[1],
            rows=rows,
            leaves=leaves,
            selections=selections,
            next_token=int(leaves["next_token"][0]),
        )

    def prefill(self, state: Any, chunk: list[int]) -> Any:
        rows = self.block_rows(len(chunk))
        tokens = self.put(np.asarray(chunk + [-1] * (rows - len(chunk)), np.int32))
        return self.run_prefill(tokens, len(chunk), state)

    def run(self, steps: int, tokens_in: list[int] | None, on_record: Any) -> None:
        import jax

        state = self.initial
        for start in range(0, len(self.prompt), BLOCK_ROWS):
            chunk = list(self.prompt[start : start + BLOCK_ROWS])
            result = jax.block_until_ready(self.prefill(state, chunk))
            on_record(
                "prefill", self.prefill_record(result, (start, start + len(chunk)))
            )
            state = result.state
        decoder_state, token = self.finish(result)
        token = int(np.asarray(token)[0])
        for step in range(steps):
            token_in = token if tokens_in is None else tokens_in[step]
            position = len(self.prompt) + step
            out, observed = self.decode(
                self.put(np.asarray([token_in], np.int32)), decoder_state
            )
            out = jax.block_until_ready(out)
            leaves = {name: host(getattr(out.state, name)) for name in DECODER_FIELDS}
            leaves.update(
                next_token=host(out.next_token),
                final_residual_local=host(out.final_residual_local),
            )
            if observed is None:
                selections = {
                    str(self.carried_layer): (
                        leaves["selected_positions"][0],
                        leaves["selected_scores"][0],
                    )
                }
            else:
                positions, scores = (
                    host(observed.selected_positions),
                    host(observed.selected_scores),
                )
                selections = {
                    str(layer): (positions[i, 0], scores[i, 0])
                    for i, layer in enumerate(self.config.full_index_slots)
                }
            token = int(leaves["next_token"][0])
            on_record(
                "decode",
                dict(
                    position=position,
                    rows=(position, position + 1),
                    token_in=token_in,
                    leaves=leaves,
                    selections=selections,
                    next_token=token,
                ),
            )
            decoder_state = out.state

    @property
    def carried_layer(self) -> int:
        return self.config.full_index_slots[-1]


class Fp8Oracle(Engine):
    """The frozen raw-FP8 prefill and decoder programs (observed decoder)."""

    name = "fp8-oracle"

    def __init__(
        self,
        mesh: Any,
        config: Any,
        weights: Any,
        wk: Any,
        rope: Any,
        put: Any,
        prompt: tuple[int, ...],
    ) -> None:
        import jax

        from glm_tpu.greenfield.runtime import ws32_batched_prefill as prefill
        from glm_tpu.greenfield.runtime import ws32_decoder as decoder

        super().__init__(mesh, config, weights, wk, rope, put, prompt)
        self.weights = weights
        self.program = prefill.build_ws32_batched_prefill_program(
            mesh, config, block_rows=BLOCK_ROWS, **WINDOW_OPTIONS, **INTERPRET
        ).execute
        self.observe = jax.jit(
            decoder.build_ws32_decoder_program(mesh, config, **INTERPRET).observe
        )
        self.initial = prefill.make_ws32_batched_prefill_state(
            mesh, config, prompt_length=len(prompt)
        )

    def block_rows(self, count: int) -> int:
        return BLOCK_ROWS

    def run_prefill(self, tokens: Any, count: int, state: Any) -> Any:
        return self.program(
            tokens, self.put(np.int32(count)), state, self.weights, self.wk, self.rope
        )

    def decode(self, token: Any, state: Any) -> tuple[Any, Any]:
        observed = self.observe(token, state, self.weights, self.rope)
        return observed.result, observed.dsa


class Production(Engine):
    """``OrdinaryRuntime._load``'s composition (BF16-resident weights and the production program set)."""

    name = "production"

    def __init__(
        self,
        mesh: Any,
        config: Any,
        weights: Any,
        wk: Any,
        rope: Any,
        put: Any,
        prompt: tuple[int, ...],
    ) -> None:
        from glm_tpu.optimized.bf16_resident import bf16_resident_weights
        from glm_tpu.runner.programs import build_program_set

        super().__init__(mesh, config, weights, wk, rope, put, prompt)
        self.weights = bf16_resident_weights(mesh, config, weights)
        programs = build_program_set(mesh, config, interpret=True)
        self.programs = {rows: programs.prefill[rows].fn for rows in (BLOCK_ROWS, TAIL_ROWS)}
        self.decoder = programs.decode.fn
        self.initial = programs.cache_init.fn(put(np.int32(len(prompt))))

    def block_rows(self, count: int) -> int:
        return (
            TAIL_ROWS if count <= TAIL_ROWS else BLOCK_ROWS
        )  # OrdinaryRuntime.generate's rule

    def run_prefill(self, tokens: Any, count: int, state: Any) -> Any:
        return self.programs[int(tokens.shape[0])](
            tokens, self.put(np.int32(count)), state, self.weights, self.wk, self.rope
        )

    def decode(self, token: Any, state: Any) -> tuple[Any, Any]:
        return self.decoder(token, state, self.weights, self.rope).decoded, None


class Reference:
    """The unsharded reference, exported in the engine's global state layout."""

    def __init__(
        self,
        engine_config: Any,
        frozen: Any,
        prompt: tuple[int, ...],
        *,
        block_rows: int,
    ) -> None:
        from tools.equivalence import fixture

        from . import model

        self.model, self.engine_config, self.prompt, self.block_rows = (
            model,
            engine_config,
            prompt,
            block_rows,
        )
        self.name = "reference" if block_rows == BLOCK_ROWS else "reference-one-block"
        self.config = model.ReferenceConfig.from_geometry(
            engine_config.geometry,
            fixture.config_json(),
            context_capacity=engine_config.context_capacity,
        )
        self.weights = model.load_weights(frozen.arrays, self.config)
        self.rope = model.rope_table(self.config)

    def record(
        self, result: Any, rows: tuple[int, int], *, final: bool, decode: bool
    ) -> dict[str, Any]:
        import jax.numpy as jnp

        engine = self.engine_config
        state = result.state
        pages, page = engine.page_count, engine.logical_page_size
        kv = jnp.pad(
            state.kv_cache,
            ((0, 0), (0, 0), (0, engine.packed_cache_width - state.kv_cache.shape[-1])),
        )
        index = host(
            state.index_cache.reshape(state.index_cache.shape[0], pages, page, -1)
        )
        carried = result.selections[-1]
        token = int(result.next_token) if final or decode else -1
        leaves = dict(
            kv_cache_local=host(kv.reshape(kv.shape[0], pages, page, kv.shape[-1])),
            index_cache_local=index,
            selected_positions=host(carried.positions[-1:]),
            selected_valid_counts=host(carried.valid_counts[-1:]),
            selected_scores=host(carried.scores[-1:]),
            position=np.asarray([state.length], np.int32),
            block_tables=np.arange(pages, dtype=np.int32)[None],
            context_lengths=np.asarray([state.length + 1], np.int32),
            contract_valid=np.asarray([True]),
            next_token=np.asarray([token], np.int32),
        )
        if decode:
            leaves["final_residual_local"] = host(result.final_residual)
        else:
            leaves.update(
                repaired_index_local=index,
                prompt_length=np.int32(len(self.prompt)),
                finished=np.bool_(final),
            )
        selections = {
            str(layer): (host(s.positions[-1]), host(s.scores[-1]))
            for layer, s in zip(self.config.full_layers, result.selections, strict=True)
        }
        logits = host(result.logits)
        top = np.argsort(-logits, kind="stable")[:2]
        decisions = dict(
            router_margin={
                str(layer): float(np.min(host(r.margin)))
                for layer, r in zip(
                    self.config.sparse_layers, result.routes, strict=True
                )
            },
            dsa_margin={
                str(layer): float(host(s.margin)[-1])
                for layer, s in zip(
                    self.config.full_layers, result.selections, strict=True
                )
            },
            # Every row of the call, for the per-row root cause (``first_excess``).
            router_margin_rows={
                str(layer): [float(m) for m in host(r.margin)]
                for layer, r in zip(
                    self.config.sparse_layers, result.routes, strict=True
                )
            },
            dsa_margin_rows={
                str(layer): [float(m) for m in host(s.margin)]
                for layer, s in zip(
                    self.config.full_layers, result.selections, strict=True
                )
            },
        )
        return dict(
            position=rows[0] if decode else rows[1],
            rows=rows,
            leaves=leaves,
            selections=selections,
            next_token=token,
            decisions=decisions,
            logits=dict(
                top=[int(i) for i in top], values=[float(logits[i]) for i in top]
            ),
        )

    def run(self, steps: int, tokens_in: list[int] | None, on_record: Any) -> None:
        state = self.model.initial_state(self.config)
        for start in range(0, len(self.prompt), self.block_rows):
            chunk = list(self.prompt[start : start + self.block_rows])
            result = self.model.forward(
                self.config, self.weights, state, chunk, rope=self.rope
            )
            final = start + len(chunk) == len(self.prompt)
            on_record(
                "prefill",
                self.record(
                    result, (start, start + len(chunk)), final=final, decode=False
                ),
            )
            state = result.state
        token = int(result.next_token)
        for step in range(steps):
            token_in = token if tokens_in is None else tokens_in[step]
            position = state.length
            result = self.model.forward(
                self.config, self.weights, state, [token_in], rope=self.rope
            )
            record = self.record(
                result, (position, position + 1), final=False, decode=True
            )
            record["token_in"] = token_in
            on_record("decode", record)
            state, token = result.state, int(result.next_token)


def production_wk(mesh: Any, raw: Any, config: Any) -> tuple[Any, ...]:
    """The promoted FP32 indexer ``wk`` tables exactly as ``_load`` computes them (the
    production program set's ``wk_decode`` and ``wk_promote``)."""
    import jax

    from glm_tpu.runner.programs import build_program_set

    decode, promote = (spec.fn for spec in build_program_set(mesh, config).wk)
    tables = []
    for layer_id in config.full_index_slots:
        dsa = raw.layers[layer_id].dsa
        tables.append(
            jax.block_until_ready(
                promote(decode(dsa.wk_bits_local, dsa.wk_scale_local))
            )
        )
    return tuple(tables)


# ----------------------------------------------------------------------------- run
def run(pair: str, prompt_name: str, steps: int) -> dict[str, Any]:
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from glm_tpu.greenfield.runtime import ws32_decoder as decoder
    from tools.equivalence import fixture
    from tools.equivalence.common import sha256_hex, tree_record
    from tools.equivalence.lowering import tpu_v4_info

    candidate_name, baseline_name = pair.split(":")
    prompt = PROMPTS[prompt_name]
    timings: dict[str, float] = {}
    started = [time.perf_counter()]

    def lap(name: str) -> None:
        timings[name] = round(time.perf_counter() - started[0], 1)
        started[0] = time.perf_counter()

    mesh = fixture.cpu_mesh()

    def put(value: Any) -> Any:
        return jax.device_put(value, NamedSharding(mesh, P()))

    frozen = fixture.fixture_v1(panel_geometry=True)
    config = frozen.config
    report: dict[str, Any] = dict(
        pair=pair,
        prompt=dict(
            name=prompt_name,
            length=len(prompt),
            sha256=sha256_hex(np.asarray(prompt, np.int32).tobytes()),
        ),
        steps=steps,
        fixture=dict(
            version=fixture.VERSION, checkpoint=tree_record(frozen.arrays)["digest"]
        ),
        tolerances=dict(rtol=RTOL, atol=ATOL),
    )
    with tpu_v4_info():
        systems: dict[str, Any] = {}
        weights = wk = rope = None
        for name in (baseline_name, candidate_name):
            if name.startswith("reference"):
                system = Reference(
                    config,
                    frozen,
                    prompt,
                    block_rows=BLOCK_ROWS if name == "reference" else len(prompt),
                )
                report["reference_weights"] = tree_record(system.weights)["digest"]
            else:
                if weights is None:
                    weights = fixture.bind(mesh, frozen)
                    wk = production_wk(mesh, weights, config)
                    rope = put(np.asarray(decoder.build_ws32_main_rope_table(config)))
                system = {"fp8-oracle": Fp8Oracle, "production": Production}[name](
                    mesh, config, weights, wk, rope, put, prompt
                )
            systems[name] = system
        if weights is not None and any(n.startswith("reference") for n in systems):
            ref = next(s for n, s in systems.items() if n.startswith("reference"))
            report["engine_inputs_equal_reference"] = dict(
                wk=all(
                    np.array_equal(
                        host(a),
                        host(ref.weights.layers[i].indexer.wk.astype("float32")),
                    )
                    for a, i in zip(wk, ref.config.full_layers, strict=True)
                ),
                rope=bool(np.array_equal(host(rope), host(ref.rope))),
            )
        lap("build")

        baseline: dict[str, list[dict[str, Any]]] = dict(prefill=[], decode=[])
        systems[baseline_name].run(
            steps, None, lambda kind, record: baseline[kind].append(record)
        )
        lap("baseline")
        tokens_in = [record["token_in"] for record in baseline["decode"]]
        by_position = {record["position"]: record for record in baseline["prefill"]}
        compared: dict[str, list[dict[str, Any]]] = dict(prefill=[], decode=[])
        top_k = config.geometry.dsa_top_k

        def on_candidate(kind: str, record: dict[str, Any]) -> None:
            if kind == "decode":
                compared["decode"].append(
                    compare_record(
                        record, baseline["decode"][len(compared["decode"])], top_k=top_k
                    )
                )
            elif record["position"] in by_position:  # a common prefill boundary
                compared["prefill"].append(
                    compare_record(record, by_position[record["position"]], top_k=top_k)
                )

        systems[candidate_name].run(steps, tokens_in, on_candidate)
        lap("candidate")
    report.update(compared)
    report["summary"] = summarize(
        compared,
        first_sparse_layer=config.geometry.mlp_layer_types.index("sparse"),
        full_layers=tuple(config.full_index_slots),
    )
    report["timings"] = timings
    return report


def bf16_ulp(value: float) -> float:
    return 2.0 ** (np.floor(np.log2(abs(value))) - 7) if value else 2.0**-133


def summarize(
    compared: dict[str, list[dict[str, Any]]],
    *,
    first_sparse_layer: int,
    full_layers: tuple[int, ...],
) -> dict[str, Any]:
    """Every acceptance criterion of DESIGN 7.6 as measured, plus the tie-aware evidence.

    ``*_integer_state_equal``: every integer/boolean leaf except the score-ordered
    ``selected_positions`` and ``next_token`` (reported by the token and selection rows).
    ``decision_free_within``: the cache rows no discrete decision can reach -- positions
    below ``top_k`` (every earlier position is selected) in the KV of layers
    ``0..first_sparse_layer`` and the index keys of the full layers before it (no MoE
    routing upstream) -- are within the float bounds at every boundary and step. ``tokens_equal_or_tied``:
    every greedy disagreement is a BF16 tie in the reference candidate's logits (its two
    best logits at most one BF16 ulp apart). ``selections_explained``: every DSA set
    disagreement lies within the pair's own score noise of the top-k boundary.
    """

    def worst(records: list[dict[str, Any]], key: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for record in records:
            for name, leaf in record["leaves"].items():
                if leaf["kind"] == "float":
                    out[name] = max(out.get(name, 0.0), leaf[key])
        return out

    def ok(
        records: list[dict[str, Any]],
        kinds: tuple[str, ...],
        skip: tuple[str, ...] = (),
    ) -> bool:
        return all(
            leaf["ok"]
            for r in records
            for name, leaf in r["leaves"].items()
            if leaf["kind"] in kinds and name not in skip
        )

    def decision_free(record: dict[str, Any]) -> float:
        rows = record["decision_free_rows"]
        if rows is None:
            return 0.0
        slots = [i for i, layer in enumerate(full_layers) if layer < first_sparse_layer]
        return max(
            max(rows["kv_latent_bound_ratio"][: first_sparse_layer + 1]),
            max(rows["kv_rope_bound_ratio"][: first_sparse_layer + 1]),
            max(rows["index_bound_ratio"][i] for i in slots),
        )

    def tied(record: dict[str, Any]) -> bool:
        logits = record["logits"]["candidate"]
        if record["tokens"]["candidate"] == record["tokens"]["baseline"]:
            return True
        return logits is not None and logits["values"][0] - logits["values"][
            1
        ] <= bf16_ulp(logits["values"][0])

    prefill, decode = compared["prefill"], compared["decode"]
    selections = [v for r in decode for v in r["selections"].values()]
    ordered_only = ("selected_positions", "next_token")
    records = prefill + decode
    return dict(
        prefill_integer_leaves_equal=ok(prefill, ("exact", "shape")),
        prefill_float_leaves_within=ok(prefill, ("float", "shape")),
        prefill_next_token_equal=prefill[-1]["tokens"]["candidate"]
        == prefill[-1]["tokens"]["baseline"],
        decode_tokens_equal=all(
            r["tokens"]["candidate"] == r["tokens"]["baseline"] for r in decode
        ),
        decode_integer_leaves_equal=ok(decode, ("exact", "shape")),
        decode_float_leaves_within=ok(decode, ("float", "shape")),
        decode_selections_ordered_equal=all(s["ordered_equal"] for s in selections),
        decode_selections_set_equal=all(s["set_equal"] for s in selections),
        prefill_integer_state_equal=ok(prefill, ("exact", "shape"), ordered_only),
        decode_integer_state_equal=ok(decode, ("exact", "shape"), ordered_only),
        decision_free_within=all(decision_free(r) <= 1.0 for r in records),
        decision_free_bound_ratio=max(decision_free(r) for r in records),
        decision_free_rows_compared=sum(
            r["decision_free_rows"] is not None for r in records
        ),
        tokens_equal_or_tied=all(tied(r) for r in decode),
        selections_explained=all(s["explained"] for s in selections),
        decode_selections_compared=len(selections),
        decode_selections_set_unequal=sum(not s["set_equal"] for s in selections),
        decode_selection_order_mismatches=sum(
            s["order_mismatches"] for s in selections
        ),
        baseline_tokens=[prefill[-1]["tokens"]["baseline"]]
        + [r["tokens"]["baseline"] for r in decode],
        candidate_tokens=[prefill[-1]["tokens"]["candidate"]]
        + [r["tokens"]["candidate"] for r in decode],
        prefill_max_abs=worst(prefill, "max_abs"),
        prefill_bound_ratio=worst(prefill, "bound_ratio"),
        prefill_outside_fraction=worst(prefill, "outside_fraction"),
        decode_max_abs=worst(decode, "max_abs"),
        decode_bound_ratio=worst(decode, "bound_ratio"),
        decode_outside_fraction=worst(decode, "outside_fraction"),
        # Per row leaving the KV bound: the first such layer and the smallest router /
        # DSA margins of that row before it (reference pairs only; VALIDATION.md).
        first_excess=[
            dict(
                phase=phase,
                position=e["position"],
                layer=e["layer"],
                bound_ratio=round(e["bound_ratio"], 3),
                previous_layer_ratio=e["previous_layer_ratio"],
                router_margin_before=min(
                    e.get("router_margins", {}).values(), default=None
                ),
                router_margin_layer_before=e.get("router_margins", {}).get(
                    str(e["layer"] - 1)
                ),
                dsa_margin_before=min(e.get("dsa_margins", {}).values(), default=None),
            )
            for phase, records in (("prefill", prefill), ("decode", decode))
            for r in records
            for e in r["first_excess"]
        ],
    )


def main(argv: list[str] | None = None) -> int:
    from tools.equivalence.common import emit, environment, require_cpu, source_record

    names = ("reference", "reference-one-block", "fp8-oracle", "production")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pair",
        required=True,
        help="CANDIDATE:BASELINE, each one of " + ", ".join(names),
    )
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="a")
    parser.add_argument("--steps", type=int, default=DECODE_STEPS)
    args = parser.parse_args(argv)
    candidate, _, baseline = args.pair.partition(":")
    if candidate not in names or baseline not in names or candidate == baseline:
        parser.error("--pair needs two different systems")
    require_cpu()
    report = run(args.pair, args.prompt, args.steps)
    report.update(environment=environment(), source=source_record())
    emit(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
