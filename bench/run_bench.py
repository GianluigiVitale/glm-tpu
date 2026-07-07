#!/usr/bin/env python3
"""GLM-5.2 benchmark harness — loads a benchmark, queries the model, extracts +
scores every item, and records EVERYTHING to the provenance DB (bench/results.db).

Model-pluggable: `generate(prompt) -> str | (str, n_gen_tokens) |
(str, n_gen_tokens, finish_reason)`. Two modes:
  --stub     offline pipeline check (no model; vllm is NEVER imported — the vllm
             import lives inside make_generate so this runs on any CPU box)
  (default)  the REAL in-process vLLM engine on the 32-chip v4 pod
             (make_generate: runai_streamer GCS->HBM load, TP=32 x EP, greedy
             thinking-mode decode through GLM-5.2's own chat template).

Engine runs are BATCHED: make_generate's generator also exposes
`generate_batch(prompts) -> [(text, n_gen_tokens, finish_reason), ...]`, and run_benchmark
submits ALL items (or --batch-size chunks) in ONE llm.generate call so vLLM's
scheduler runs up to --max-seqs sequences concurrently (~1 tok/s single-stream
decode -> batch-factor aggregate throughput). Provenance is unchanged per item;
per-item latency_ms is NULL in batched runs (never faked) — the real batch wall
time is recorded in the run's summary note as batch_wall_ms.

Nothing is scored that is not stored. Every item row carries the verbatim prompt,
the verbatim model reply, the extracted answer, correct/score, n_gen_tokens,
timing, and the run-level provenance (model+revision, harness+fork git commits,
env/flags — pv.start_run + _run_env — and the pod).

    # offline pipeline check (no model): records items with empty replies
    python run_bench.py --benchmark gsm8k --limit 5 --stub

    # real run (pod-side, Ray cluster up on all 8 hosts):
    ~/vllm-env/bin/python -u run_bench.py --benchmark gsm8k --limit 16 \
        --max-new 2048 --note "stage1 smoke"

    # multi-benchmark convenience (ONE engine build, sequential runs):
    ~/vllm-env/bin/python -u run_bench.py --benchmarks gsm8k,gpqa_diamond \
        --limit 50 --max-new 4096

Protocols (--protocol, default greedy): `greedy` is the harness's own protocol
(temperature=0, \\boxed{} prompts) — byte-identical to the pre-protocol
harness. `card` runs the HF model card's PUBLISHED protocol (temperature=1.0,
top_p=0.95, 163,840-token generation cap; for AIME the card's Explanation/
Exact Answer/Confidence system prompt + the Exact-Answer extractor as the
labeled GPT-5.5-judge substitute; optional --samples N = avg@N with per-sample
provenance rows). What card mode can and cannot reproduce, with the card
quotes: docs/07-card-protocol-fidelity.md.
"""
from __future__ import annotations

import argparse
import os
import time

import provenance as pv
import benchmarks as B
import engine  # CPU-safe: the vllm import lives inside engine.build_llm

HERE = os.path.dirname(os.path.abspath(__file__))

# The engine recipe (checkpoint default, EOS ids, LLM(...) build) lives in
# bench/engine.py — shared with glm_longctx.py so both harnesses run the
# IDENTICAL engine. Re-exported here so `run_bench.DEFAULT_MODEL` /
# `run_bench.EOS_IDS` keep working.
DEFAULT_MODEL = engine.DEFAULT_MODEL
EOS_IDS = engine.EOS_IDS


def stub_generate(prompt: str, **kwargs) -> str:
    """Placeholder model: returns nothing (all items score wrong) — used only to
    validate the load→prompt→extract→score→store pipeline offline. Accepts (and
    ignores) the card-protocol kwargs (system_prompt/sampling/seed) so
    --stub --protocol card exercises the card pipeline offline too."""
    return ""


def _chat_prompt_ids(tok, prompt: str, chat_template: str | None = None,
                     system_prompt: str | None = None) -> list[int]:
    """Render one turn through GLM-5.2's own chat template and tokenize.

    The shipped chat_template.jinja produces (defaults: thinking ON, effort Max):
        [gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{prompt}<|assistant|><think>
    i.e. the model continues INSIDE the think block, reasons, closes </think>,
    then answers (the bench prompts' _THINK_HINT asks for \\boxed{}); the
    extractors strip the think block. Token ids are returned (and passed to the
    engine as prompt_token_ids) so there is no re-tokenization ambiguity.
    `chat_template` is a fallback template TEXT for a tokenizer that did not pick
    up the checkpoint's chat_template.jinja (None = use the tokenizer's own).

    `system_prompt` (card protocol) rides as a {"role": "system"} message; the
    shipped template renders it as a SECOND <|system|> block AFTER its own
    default reasoning-effort block (verified against chat_template.jinja):
        [gMASK]<sop><|system|>Reasoning Effort: Max<|system|>{system_prompt}<|user|>...
    — that IS the model's own system-message convention (the template gives no
    way to replace the effort block without disabling thinking), and it is
    recorded verbatim in the run's provenance (env_json card_protocols).
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return tok.apply_chat_template(
        msgs,
        add_generation_prompt=True, tokenize=True, return_dict=False,
        chat_template=chat_template)


def _sort_by_request_id(outs):
    """Defensively re-assert submission order on vLLM outputs. The LLM API
    already returns outputs in input order (its _run_engine sorts by request
    id), but the mapping back to items is correctness-critical — so re-sort by
    the integer request id when possible instead of trusting it silently."""
    try:
        return sorted(outs, key=lambda o: int(o.request_id))
    except (TypeError, ValueError):
        return list(outs)


def make_generate(model: str, *, max_len: int = 8192, max_new: int | None = 2048,
                  max_seqs: int = 8, max_batched_tokens: int = 4096,
                  gmu: float = 0.94, num_gpu_blocks: int = 0,
                  temperature: float = 0.0, top_p: float = 1.0):
    """Build the REAL GLM-5.2 generator: an in-process vLLM engine on the
    32-chip pod, modeled on the DSV4 pattern (~/moe-tpu/bench/dsv4_gen_bench.py
    build_llm + Generator) with ONE deliberate topology difference:

      GLM Stage 1 runs WITHOUT DP attention — pure TP x EP. No
      `additional_config={"sharding": ...}` / `enable_dp_attention` (that was
      DSV4's MLA recipe; the GLM fork branch does not require it).

    The LLM(...) build itself lives in bench/engine.py (shared with
    glm_longctx.py). Env knobs: GLM_MODEL (checkpoint), GLM_TP (default 32),
    RUNAI_STREAMER_CONCURRENCY / RUNAI_STREAMER_MEMORY_LIMIT (streaming load).
    Returns `generate(prompt) -> (text, n_gen_tokens, finish_reason)` — the
    VERBATIM completion text, the generated-token count (items.n_gen_tokens),
    and vLLM's finish_reason ('length' = truncated at max_tokens; recorded in
    the items row, still scored as-is).
    The returned callable also exposes `generate.generate_batch(prompts) ->
    [(text, n_gen_tokens, finish_reason), ...]` (one llm.generate call for MANY prompts —
    vLLM's scheduler then runs up to max_seqs sequences concurrently, which is
    what turns ~1 tok/s single-stream decode into batch-factor aggregate
    throughput). run_benchmark auto-uses it when present.
    """
    # vllm import stays INSIDE make_generate: `import run_bench` and --stub must
    # work with no vllm/TPU (module-top import would break the offline pipeline).
    # Import order: `from vllm import LLM` first so vllm.platforms resolves
    # once, up front (importing a vllm submodule before platform resolution
    # can re-enter vllm.platforms: ImportError current_platform).
    # MECHANISM CORRECTION (round-4 review, docs/reviews/round4-launcher-bench.md
    # finding 3 — an earlier comment here claimed import order kept
    # tpu_inference out of the driver and that its import held the libtpu
    # lockfile; both claims were false): the driver imports tpu_inference on
    # EVERY run regardless — vllm.platforms resolution itself does
    # `from tpu_inference.platforms import TpuPlatform`. That import is
    # TPU-NEUTRAL (env-var overrides, GCE metadata, /dev/accel glob — no
    # jax/libtpu client init, no lockfile). The historical "ABORTED: lockfile"
    # hangs were caused by (1) the fork's get_page_size() calling
    # jax.devices() in the DRIVER config path — fixed in fork commit 7ae390f2
    # (see flash_attn_mla.py's NOTE) — and (2) a leaked EngineCore holding the
    # lock (cleared by the launcher stop phase, harness commit 33576f9).
    # Driver TPU-neutrality holds because no driver-side call path touches
    # jax devices TODAY — it is not guaranteed by import order.
    from vllm import LLM  # noqa: F401
    from vllm import SamplingParams

    llm = engine.build_llm(model, max_len=max_len, max_seqs=max_seqs,
                           max_batched_tokens=max_batched_tokens, gmu=gmu,
                           num_gpu_blocks=num_gpu_blocks,
                           log_extra=f"max_new={max_new}")

    tok = llm.get_tokenizer()
    chat_template = None
    if getattr(tok, "chat_template", None) is None:
        # The checkpoint ships chat_template.jinja (transformers 5.x auto-loads
        # it); if this tokenizer instance didn't pick it up, fall back to the
        # committed reference copy (verified byte-identical to the HF repo).
        ref = os.path.join(HERE, "..", "reference", "hf-repo", "chat_template.jinja")
        with open(ref, encoding="utf-8") as f:
            chat_template = f.read()
        print("[bench] tokenizer had no chat template — using the reference "
              "chat_template.jinja copy", flush=True)

    def _sp(room: int, *, sampling: dict | None = None,
            seed: int | None = None) -> "SamplingParams":
        # ONE sampling constructor for both protocols and both paths. With no
        # overrides (sampling=None, seed=None — the default/greedy protocol)
        # this is byte-identical to the pre-protocol harness: temperature=0.0,
        # top_p=1.0 (the SamplingParams default), no seed, max_tokens clamped
        # to the room this prompt leaves in the window. Card mode passes the
        # per-benchmark card params (spec.card: temperature/top_p/max_new) +
        # a per-sample seed — all recorded in provenance. The effective
        # max_tokens is the min of every cap present: the engine-level
        # --max-new (None = no CLI cap), the card's max generation length,
        # and the window room.
        s = sampling or {}
        caps = [c for c in (max_new, s.get("max_new"), room) if c is not None]
        return SamplingParams(
            temperature=s.get("temperature", temperature),  # default 0.0 = greedy
            top_p=s.get("top_p", top_p),
            seed=seed,
            max_tokens=min(caps),
            stop_token_ids=EOS_IDS,
            ignore_eos=False,
        )

    def generate(prompt: str, *, system_prompt: str | None = None,
                 sampling: dict | None = None,
                 seed: int | None = None) -> tuple[str, int, str | None]:
        ids = _chat_prompt_ids(tok, prompt, chat_template, system_prompt)
        room = max_len - len(ids) - 8
        if room <= 0:
            # Prompt alone exceeds the context window: record an empty reply
            # (scored wrong, auditable) rather than crash the run.
            print(f"[bench] SKIP: prompt {len(ids)} tok > max_len {max_len}",
                  flush=True)
            return "", 0, None
        outs = llm.generate([{"prompt_token_ids": ids}],
                            _sp(room, sampling=sampling, seed=seed),
                            use_tqdm=False)
        o = outs[0].outputs[0]
        # finish_reason ('stop' | 'length' | ...) rides along for truncation
        # honesty: 'length' = the reply hit max_tokens mid-CoT and is flagged
        # truncated in the items row (still scored as-is, never excluded).
        return o.text, len(o.token_ids), getattr(o, "finish_reason", None)

    def generate_batch(prompts: list[str], *,
                       system_prompts: list[str | None] | None = None,
                       sampling: dict | None = None,
                       seed: int | None = None,
                       ) -> list[tuple[str, int, str | None]]:
        """ONE llm.generate call for a list of prompts — vLLM schedules up to
        max_seqs of them concurrently. Same sampling protocol as generate()
        (the per-prompt SamplingParams differ ONLY in the room clamp on
        max_tokens, exactly as the sequential path computed it). Results are
        returned IN INPUT ORDER: over-long prompts keep their slot as
        ("", 0, None) (recorded empty + scored wrong, same as the sequential
        SKIP), and the vLLM outputs are mapped back by request order/id. Each
        result carries vLLM's finish_reason (truncation honesty — additive,
        nothing else about the batched protocol changed).

        Card protocol (all optional, None = the greedy call, unchanged):
        `system_prompts` aligns 1:1 with `prompts`; `sampling` is the
        per-benchmark card param dict (temperature/top_p/max_new); `seed` is
        the per-SAMPLE seed (uniform across the batch — vLLM seeds its RNG
        per request, so equal seeds on different prompts stay independent,
        while re-running the same (prompt, seed) reproduces)."""
        results: list[tuple[str, int, str | None] | None] = [None] * len(prompts)
        submit_idx: list[int] = []
        token_prompts, sps = [], []
        sys_prompts = system_prompts or [None] * len(prompts)
        assert len(sys_prompts) == len(prompts), \
            f"{len(sys_prompts)} system prompts for {len(prompts)} prompts"
        for i, prompt in enumerate(prompts):
            ids = _chat_prompt_ids(tok, prompt, chat_template, sys_prompts[i])
            room = max_len - len(ids) - 8
            if room <= 0:
                print(f"[bench] SKIP: prompt {len(ids)} tok > max_len "
                      f"{max_len}", flush=True)
                results[i] = ("", 0, None)
                continue
            submit_idx.append(i)
            token_prompts.append({"prompt_token_ids": ids})
            sps.append(_sp(room, sampling=sampling, seed=seed))
        if token_prompts:
            print(f"[bench] submitting {len(token_prompts)} prompts in ONE "
                  f"llm.generate (up to {max_seqs} concurrent)", flush=True)
            outs = _sort_by_request_id(
                llm.generate(token_prompts, sps, use_tqdm=True))
            assert len(outs) == len(token_prompts), \
                f"vLLM returned {len(outs)} outputs for {len(token_prompts)} prompts"
            for i, out in zip(submit_idx, outs):
                o = out.outputs[0]
                results[i] = (o.text, len(o.token_ids),
                              getattr(o, "finish_reason", None))
        return results  # type: ignore[return-value]  # every slot is filled

    generate.generate_batch = generate_batch
    return generate


def _score_and_record(conn, run_id, spec: B.BenchSpec, it, reply, n_gen,
                      latency_ms, seed, finish_reason=None, extract_fn=None,
                      item_id=None):
    """Extract + score ONE reply and store the full item row. The audit trail
    must never be lost: the verbatim reply is stored even if extraction/scoring
    raises (extracted/correct = None on failure). finish_reason (vLLM's, when
    the generator provides it) flags truncation: 'length' means the reply hit
    max_tokens mid-CoT — the item is STILL SCORED AS-IS (scoring unchanged;
    salvage extraction from a truncated CoT can be wrong in both directions)
    but the row records finish_reason + truncated so it is auditable.
    extract_fn (card protocol) overrides spec.extract (None = spec.extract,
    unchanged); item_id overrides it.item_id (the '#sN' per-sample suffix in
    multi-sample card runs — every sample is its own auditable row)."""
    try:
        extracted = (extract_fn or spec.extract)(reply, it)
        correct = spec.score(extracted, it.gold)
    except Exception:
        extracted, correct = None, None
    truncated = None if finish_reason is None else (finish_reason == "length")
    pv.record_item(conn, run_id, benchmark=spec.name,
                   item_id=item_id or it.item_id,
                   prompt=it.prompt, gold=it.gold, raw_output=reply,
                   extracted=extracted, correct=correct,
                   score=(1.0 if correct else 0.0) if correct is not None else None,
                   n_gen_tokens=n_gen, latency_ms=latency_ms, seed=seed,
                   finish_reason=finish_reason, truncated=truncated)
    return extracted, correct


def _run_items_batched(conn, run_id, spec: B.BenchSpec, generate_batch, items,
                       seed=0, batch_size=0, protocol="greedy", sampling=None,
                       extract_fn=None, tag="") -> str:
    """Submit prompts in bulk through generate_batch (ONE llm.generate per
    chunk; batch_size <= 0 = ALL items in one call) so vLLM runs up to max_seqs
    sequences concurrently. Provenance per item is EXACTLY the sequential
    path's (verbatim prompt/raw output/extracted/correct/n_gen_tokens); only
    per-item latency is not individually measurable inside a batch, so
    latency_ms is stored as NULL — never faked — and the real batch wall times
    go into the summary note (returned) as batch_wall_ms.

    protocol='greedy' (default) calls generate_batch(prompts) EXACTLY as the
    pre-protocol harness did. protocol='card' passes the items' card system
    prompts, the card sampling dict, and this sample's seed; `tag` ('#sN' in
    multi-sample runs) suffixes the stored item_id so every sample is its own
    row, and `seed` is recorded per row."""
    chunks = ([items] if batch_size <= 0 else
              [items[i:i + batch_size] for i in range(0, len(items), batch_size)])
    total_wall_ms, total_tok, done = 0.0, 0, 0
    for chunk in chunks:
        t0 = time.time()
        if protocol == "card":
            outs = generate_batch(
                [it.prompt for it in chunk],
                system_prompts=[it.system_prompt for it in chunk],
                sampling=sampling, seed=seed)
        else:
            outs = generate_batch([it.prompt for it in chunk])
        wall_ms = (time.time() - t0) * 1000.0
        total_wall_ms += wall_ms
        assert len(outs) == len(chunk), \
            f"generate_batch returned {len(outs)} results for {len(chunk)} prompts"
        chunk_tok = 0
        for it, out in zip(chunk, outs):
            # additive: results are (reply, n_gen) historically, now
            # (reply, n_gen, finish_reason) — accept both.
            reply, n_gen = out[0], out[1]
            finish = out[2] if len(out) > 2 else None
            extracted, correct = _score_and_record(
                conn, run_id, spec, it, reply, n_gen, latency_ms=None,
                seed=seed, finish_reason=finish, extract_fn=extract_fn,
                item_id=(it.item_id + tag) if tag else None)
            done += 1
            chunk_tok += n_gen or 0
            trunc_flag = " TRUNCATED" if finish == "length" else ""
            print(f"[{spec.name}] {done}/{len(items)} {it.item_id}{tag}: "
                  f"correct={correct} extracted={extracted!r} gold={it.gold!r} "
                  f"gen_tok={n_gen}{trunc_flag} (batched)", flush=True)
        total_tok += chunk_tok
        rate = chunk_tok / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
        print(f"[{spec.name}] batch of {len(chunk)} done in "
              f"{wall_ms / 1000.0:.1f}s ({chunk_tok} gen tok, "
              f"{rate:.1f} tok/s aggregate)", flush=True)
    return (f"batched:{len(items)} chunks={len(chunks)} "
            f"batch_wall_ms={total_wall_ms:.0f} gen_tok={total_tok}")


def run_benchmark(conn, run_id, spec: B.BenchSpec, generate, limit=None, seed=0,
                  batch_size=0, protocol="greedy", samples=1):
    """Run one benchmark under `protocol`:

    'greedy' (default) — the pre-protocol harness, byte-identical: greedy
    decode, \\boxed{} prompts, spec.extract, one pass, no per-request seed.

    'card' — the HF model card's published protocol (spec.card, refused when
    None): the card's sampling params (temperature/top_p/max generation
    length), the card's system prompt where it specifies one (AIME/HMMT/
    IMOAnswerBench), the card-mode extractor (Exact-Answer field; the labeled
    judge substitute), and `samples` independent samples per item (avg@N —
    the card specifies NO k/averaging, so N>1 is a harness-side variance
    knob; the summary value is the mean over ALL sample rows). Sample s uses
    seed `seed + s`, recorded on every row; multi-sample rows get an '#sN'
    item_id suffix so each sample is individually auditable."""
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    if protocol != "card" and samples != 1:
        raise ValueError("--samples N>1 requires --protocol card "
                         "(greedy resamples are identical by construction)")
    items = B.load_items(spec, limit=limit, protocol=protocol)
    cp = spec.card if protocol == "card" else None
    extract_fn = cp.extract if (cp and cp.extract) else None
    sampling = (None if cp is None else
                {"temperature": cp.temperature, "top_p": cp.top_p,
                 "max_new": cp.max_new})
    proto_note = ""
    if cp is not None:
        proto_note = (f"protocol=card temp={cp.temperature} top_p={cp.top_p} "
                      f"card_max_new={cp.max_new} samples={samples} "
                      f"base_seed={seed}")
    notes = []
    generate_batch = getattr(generate, "generate_batch", None)
    for s in range(samples):
        tag = f"#s{s}" if samples > 1 else ""
        sample_seed = (seed + s) if protocol == "card" else seed
        if samples > 1:
            print(f"[{spec.name}] sample {s + 1}/{samples} "
                  f"(seed={sample_seed})", flush=True)
        if generate_batch is not None:
            # BATCHED: the engine generator exposes generate_batch — submit in
            # bulk (the stub does not, and keeps the sequential path unchanged).
            notes.append(_run_items_batched(
                conn, run_id, spec, generate_batch, items, seed=sample_seed,
                batch_size=batch_size, protocol=protocol, sampling=sampling,
                extract_fn=extract_fn, tag=tag))
        else:
            for i, it in enumerate(items):
                t0 = time.time()
                if protocol == "card":
                    out = generate(it.prompt, system_prompt=it.system_prompt,
                                   sampling=sampling, seed=sample_seed)
                else:
                    out = generate(it.prompt)
                # generate may return plain text (stub), (text, n_gen_tokens),
                # or (text, n_gen_tokens, finish_reason) (engine).
                if isinstance(out, tuple):
                    reply, n_gen = out[0], out[1]
                    finish = out[2] if len(out) > 2 else None
                else:
                    reply, n_gen, finish = out, None, None
                latency = (time.time() - t0) * 1000.0
                extracted, correct = _score_and_record(
                    conn, run_id, spec, it, reply, n_gen,
                    latency_ms=round(latency, 1), seed=sample_seed,
                    finish_reason=finish, extract_fn=extract_fn,
                    item_id=(it.item_id + tag) if tag else None)
                trunc_flag = " TRUNCATED" if finish == "length" else ""
                print(f"[{spec.name}] {i + 1}/{len(items)} {it.item_id}{tag}: "
                      f"correct={correct} extracted={extracted!r} "
                      f"gold={it.gold!r} gen_tok={n_gen}{trunc_flag} "
                      f"{latency / 1000.0:.1f}s", flush=True)
    note = " ".join(x for x in [proto_note] + notes if x)
    summ = pv.finalize(conn, run_id, benchmark=spec.name, metric="acc", note=note)
    print(f"[{spec.name}] n={summ['n']} acc={summ['value']}"
          f" card={summ['card_value']} Δ={summ['delta']}"
          f" n_truncated={summ['n_truncated']}"
          f"{'  [' + note + ']' if note else ''}")
    return summ


# The env vars that decide the SERVED NUMERICS (FP8 checkpoint-exactness, the
# DSA bypass, the token-bucket floor). The workers' effective env is a MERGE
# the DB could previously not see: raylet-baked (the launcher's ENVS string)
# OVERRIDDEN by any driver-side value for vars on vLLM's TPU allow-list
# (tpu_platform.py additional_env_vars -> ray_distributed_executor copies
# driver os.environ to the workers when set). DISABLE_WEIGHT_REQUANTIZATION /
# TPU_DISABLE_DSA_INDEXER / TPU_MIN_TOKEN_BUCKET are on that allow-list —
# a stale driver-shell export silently changes what all 32 chips serve.
# REQUANTIZE_WEIGHT_DTYPE is NOT on it (a driver value stays driver-only), but
# a mismatch still signals a confused environment. Record the launcher-baked
# string verbatim + a live comparison; warn LOUDLY on mismatch.
_NUMERICS_CRITICAL_ENVS = ("DISABLE_WEIGHT_REQUANTIZATION",
                           "REQUANTIZE_WEIGHT_DTYPE",
                           "TPU_DISABLE_DSA_INDEXER",
                           "TPU_MIN_TOKEN_BUCKET")
_LAUNCHER_SH = os.path.abspath(
    os.path.join(HERE, "..", "scripts", "launch_glm_32chip.sh"))


def _launcher_env_provenance() -> dict:
    """Read the launcher's ENVS string (the env baked into every raylet — what
    the workers actually run with) at RUN time and compare the numerics-critical
    vars against the driver's live os.environ. Returns env_json fields:
    'launcher_envs_baked' (the verbatim string) and 'launcher_env_check'
    ({var: {baked, baked_nominal, driver}}). A `${VAR:-default}` token in the
    ENVS string is a launch-shell passthrough; its default is the nominal baked
    value (the value at actual launch time is not reconstructable here)."""
    import re
    out: dict = {"launcher_envs_baked": None, "launcher_env_check": {}}
    try:
        with open(_LAUNCHER_SH, encoding="utf-8") as f:
            text = f.read()
        m = re.search(r'^ENVS="export (.*)"$', text, re.MULTILINE)
        if not m:
            out["launcher_envs_baked"] = f"UNPARSED: no ENVS line in {_LAUNCHER_SH}"
            return out
        baked_str = m.group(1)
        out["launcher_envs_baked"] = baked_str
        for name in _NUMERICS_CRITICAL_ENVS:
            vm = re.search(rf"\b{name}=(\S+)", baked_str)
            baked_raw = vm.group(1) if vm else None
            baked = baked_raw
            if baked_raw:
                dm = re.fullmatch(r"\$\{" + name + r":-([^}]*)\}", baked_raw)
                if dm:
                    baked = dm.group(1)
            driver = os.environ.get(name)
            out["launcher_env_check"][name] = {
                "baked": baked_raw, "baked_nominal": baked, "driver": driver}
            if driver is not None and baked is not None and driver != baked:
                print("!" * 78, flush=True)
                print(f"[bench] WARNING: driver os.environ[{name!r}] = "
                      f"{driver!r} != launcher-baked {baked!r}.\n"
                      "  DISABLE_WEIGHT_REQUANTIZATION / TPU_DISABLE_DSA_INDEXER /"
                      " TPU_MIN_TOKEN_BUCKET\n"
                      "  FORCE-PROPAGATE driver->workers (tpu_platform"
                      " additional_env_vars), overriding\n"
                      "  the raylet-baked value on all 32 chips — served numerics"
                      " may differ from the\n"
                      "  launcher recipe. Both values are recorded in"
                      " runs.env_json['launcher_env_check'].", flush=True)
                print("!" * 78, flush=True)
    except Exception as exc:  # never fail a run over provenance introspection
        out["launcher_envs_baked"] = f"UNAVAILABLE: {exc}"
    return out


def _run_env(args, benches) -> dict:
    """Run-level env/config provenance (stored in runs.env_json alongside the
    harness/fork git hashes that pv.start_run already records). Secrets are
    filtered — never store tokens/keys."""
    os_env = {k: v for k, v in sorted(os.environ.items())
              if k.startswith(("GLM_", "RUNAI_STREAMER_", "VLLM_", "TPU_", "JAX_",
                               "NEW_MODEL_DESIGN", "MODEL_IMPL_TYPE",
                               "OMP_NUM_THREADS",
                               # numerics-critical, prefix-matched exactly —
                               # previously unrecorded (round-4 finding 2)
                               "DISABLE_WEIGHT_REQUANTIZATION",
                               "REQUANTIZE_WEIGHT_DTYPE"))
              and not any(s in k.upper() for s in ("TOKEN", "KEY", "SECRET"))}
    # FULL protocol provenance (docs/07-card-protocol-fidelity.md): which
    # protocol ran, the per-benchmark card sampling params + the VERBATIM card
    # system prompt + the verbatim card quote it came from + the labeled
    # substitutes, the sample count and base seed. Greedy runs record
    # protocol="greedy" with an empty card_protocols map — the model-facing
    # greedy protocol itself is unchanged.
    card_protocols = {}
    if args.protocol == "card":
        for b in benches:
            cp = B.REGISTRY[b].card
            card_protocols[b] = {
                "temperature": cp.temperature, "top_p": cp.top_p,
                "card_max_new": cp.max_new,
                "system_prompt": cp.system_prompt,
                "extractor": getattr(cp.extract or B.REGISTRY[b].extract,
                                     "__name__", "unknown"),
                "source": cp.source,
                "substitutes": cp.substitutes,
            }
    return {
        **_launcher_env_provenance(),
        "model": args.model, "stub": bool(args.stub),
        # pinned dataset commit shas (BenchSpec.hf_revision) — reproducibility
        "dataset_revisions": {b: B.REGISTRY[b].hf_revision for b in benches},
        "tp": int(os.environ.get("GLM_TP", "32")),
        "dp_attention": False,   # GLM Stage 1: pure TP x EP (no DP attention)
        "expert_parallel": True,
        "max_len": args.max_len, "max_new": args.max_new,
        "max_seqs": args.max_seqs, "max_batched_tokens": args.max_batched_tokens,
        "batch_size": args.batch_size,
        "gmu": args.gmu, "num_gpu_blocks": args.num_gpu_blocks,
        # engine-level sampling defaults; card mode overrides PER REQUEST from
        # card_protocols (temperature/top_p/max_new) + per-sample seeds.
        "temperature": 0.0 if args.protocol == "greedy" else None,
        "top_p": 1.0 if args.protocol == "greedy" else None,
        "protocol": args.protocol,
        "samples": args.samples,
        "base_seed": args.seed,
        "card_protocols": card_protocols,
        "eos_ids": EOS_IDS,
        "load_format": "runai_streamer",
        "os_env": os_env,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(B.REGISTRY), default=None)
    ap.add_argument("--benchmarks", default=None,
                    help="comma-separated multi-run, e.g. gsm8k,gpqa_diamond "
                         "(ONE engine build; overrides --benchmark)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="checkpoint path/repo (default: GLM_MODEL env or the "
                         "staged us-central2 GCS copy)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--stub", action="store_true",
                    help="offline pipeline check (no model — all items score wrong)")
    ap.add_argument("--protocol", choices=("greedy", "card"), default="greedy",
                    help="greedy (default; byte-identical to the pre-protocol "
                         "harness: temperature=0, \\boxed{} prompts, spec "
                         "extractors) or card (the HF model card's PUBLISHED "
                         "protocol: temperature=1.0 top_p=0.95, 163,840-token "
                         "generation cap, the card's Exact-Answer system "
                         "prompt + extractor where the card specifies one — "
                         "see docs/07-card-protocol-fidelity.md for exactly "
                         "what card mode can and cannot reproduce)")
    ap.add_argument("--samples", type=int, default=1,
                    help="independent samples per item in card mode (avg@N; "
                         "per-sample provenance rows with '#sN' item ids and "
                         "seed base_seed+N). The card specifies NO k/averaging "
                         "— N>1 is a harness-side variance knob, honestly "
                         "labeled in provenance. Requires --protocol card.")
    ap.add_argument("--seed", type=int, default=0,
                    help="base sampling seed (card mode; sample s uses seed+s, "
                         "recorded per item row). The card publishes no seeds "
                         "— ours are for OUR reproducibility only.")
    ap.add_argument("--max-new", type=int, default=None,
                    help="max generated tokens; GLM-5.2 thinks before answering "
                         "— AIME/GPQA need long CoT, raise (e.g. 4096-16384). "
                         "Default: 2048 (greedy) / no CLI cap (card — the "
                         "card's 163,840 cap + the context-window room apply; "
                         "an explicit value here adds a pod-capacity cap, "
                         "recorded in provenance, truncations flagged)")
    ap.add_argument("--max-len", type=int, default=8192,
                    help="max_model_len (prompt + generation)")
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--max-batched-tokens", type=int, default=4096,
                    help="chunked-prefill chunk size")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="engine runs submit prompts in chunks of this size "
                         "through ONE llm.generate call each (vLLM schedules "
                         "up to --max-seqs concurrently); 0 = ALL items in one "
                         "call. --stub stays sequential (no generate_batch). "
                         "Per-item latency_ms is NULL in batched runs — the "
                         "batch wall time is recorded in the summary note.")
    ap.add_argument("--num-gpu-blocks", type=int, default=0,
                    help="num_gpu_blocks_override if >0 (cap the KV pool to free "
                         "HBM for the per-forward program; 0 = auto)")
    ap.add_argument("--gmu", type=float,
                    default=float(os.environ.get("GLM_GMU", "0.94")),
                    help="gpu_memory_utilization")
    args = ap.parse_args()

    benches = [b.strip() for b in
               (args.benchmarks or args.benchmark or "").split(",") if b.strip()]
    if not benches:
        ap.error("provide --benchmark NAME or --benchmarks a,b,c")
    unknown = [b for b in benches if b not in B.REGISTRY]
    if unknown:
        ap.error(f"unknown benchmark(s) {unknown}; have {list(B.REGISTRY)}")
    if args.protocol == "card":
        no_card = [b for b in benches if B.REGISTRY[b].card is None]
        if no_card:
            have = [b for b in B.REGISTRY if B.REGISTRY[b].card is not None]
            ap.error(f"--protocol card: no card protocol mapped for {no_card} "
                     f"(card protocols exist for {have}); run the rest with "
                     "--protocol greedy")
    if args.samples < 1:
        ap.error("--samples must be >= 1")
    if args.samples > 1 and args.protocol != "card":
        ap.error("--samples N>1 requires --protocol card (greedy resamples "
                 "are identical by construction)")
    if args.max_new is None and args.protocol == "greedy":
        args.max_new = 2048   # the pre-protocol default, byte-identical

    conn = pv.connect()
    run_id = pv.start_run(conn, model=("STUB" if args.stub else args.model),
                          revision=args.revision, env=_run_env(args, benches),
                          note=args.note or ("offline-stub" if args.stub else ""))
    gen = stub_generate if args.stub else make_generate(
        args.model, max_len=args.max_len, max_new=args.max_new,
        max_seqs=args.max_seqs, max_batched_tokens=args.max_batched_tokens,
        gmu=args.gmu, num_gpu_blocks=args.num_gpu_blocks)
    for b in benches:
        run_benchmark(conn, run_id, B.REGISTRY[b], gen, limit=args.limit,
                      batch_size=args.batch_size, protocol=args.protocol,
                      samples=args.samples, seed=args.seed)


if __name__ == "__main__":
    main()
